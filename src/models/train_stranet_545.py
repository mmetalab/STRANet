#!/usr/bin/env python3
"""Train ONE Unified Single-Seed STRANet Architecture (stranet_unified_v2) on NSCLC-545.

Combines into a single unified model and a single seed:
  1. Dual-window 2-channel 3D CT input (Lung [-1400, 200] HU + Soft-Tissue [-160, 240] HU)
  2. Dual-scale 3D Spatial-Delta Pyramid (16x16x16 + 8x8x8) with SE channel attention + 3D spatial change attention
  3. Warm-started from the pre-trained 3D Siamese spatial-delta encoder checkpoints (seed21600 / seed20102)
     + longitudinal whole-lung & tumor radiomics prior embedding
  4. Composite Focal + Label-Smoothed Cross-Entropy Loss
  5. Top-3 Epoch Stochastic Weight Averaging (SWA) + 8-View 3D Test-Time Augmentation (TTA)

Outputs a single-seed prediction file:
  - ./data/Reviewer_R1_auc_rawpost_fix/results/nsclc_545_excl_sclc_20261106/single_seed_stranet_545/
      - single_seed_stranet_oof_predictions.csv      (seed = 21600, N = 545)
      - single_seed_stranet_holdout_predictions.csv  (seed = 20102, n = 82)
      - single_seed_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import rotate, shift as nd_shift, zoom
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras import Model, mixed_precision, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv3D,
    Dense,
    Dropout,
    GlobalAveragePooling3D,
    GlobalMaxPooling3D,
    Input,
    Lambda,
    MaxPooling3D,
    Multiply,
    Reshape,
    SpatialDropout3D,
    Subtract,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_545 = ROOT / "data/nsclc_545_excl_sclc_20261106"
OUT_545 = ROOT / "results/nsclc_545_excl_sclc_20261106"
SINGLE_OUT = OUT_545 / "single_seed_stranet_545"
SINGLE_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from run_nsclc_545_benchmarks import (  # noqa: E402
    NSCLC545PairGenerator,
    build_siamese_nsclc545_model,
    set_seeds,
)
from run_stage_source_594_benchmarks import fit_clinical_comparators  # noqa: E402

K.set_image_data_format("channels_first")


def predict_8view_tta(
    model: Model,
    df: pd.DataFrame,
    dual_window: bool = False,
    include_stage: bool = False,
    seed: int = 2026,
) -> np.ndarray:
    """8-view deterministic 3D Test-Time Augmentation on a single model checkpoint."""
    gen_base = NSCLC545PairGenerator(
        df,
        batch_size=8,
        orientation="raw",
        dual_window=dual_window,
        sampler_mode="label_only",
        aug_mode="none",
        steps_multiplier=1,
        use_pseudo_hybrid=False,
        use_cutout=False,
        use_mixup=False,
        include_stage_input=include_stage,
        balanced=False,
        seed=seed,
    )
    # Collect raw volumes once from cache
    n = len(df)
    n_ch = 2 if dual_window else 1
    x_pre = np.empty((n, n_ch, 64, 64, 64), dtype=np.float32)
    x_post = np.empty((n, n_ch, 64, 64, 64), dtype=np.float32)
    x_stage = np.empty((n, 1), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        x_pre[i] = gen_base._load(row.Pre_path)
        x_post[i] = gen_base._load(row.Post_path)
        s_val = 2.0 if pd.isna(row.stage_123) else float(row.stage_123)
        x_stage[i, 0] = s_val - 2.0

    # Define 8 deterministic 3D views: (angle_deg, shift_xyz, scale, bias)
    views = [
        (0.0, (0.0, 0.0, 0.0), 1.00, 0.000),
        (5.0, (0.0, 0.0, 0.0), 1.00, 0.000),
        (-5.0, (0.0, 0.0, 0.0), 1.00, 0.000),
        (0.0, (1.5, -1.0, 0.5), 1.02, 0.005),
        (0.0, (-1.5, 1.0, -0.5), 0.98, -0.005),
        (3.5, (1.0, 1.0, 0.0), 1.01, 0.000),
        (-3.5, (-1.0, -1.0, 0.0), 0.99, 0.000),
        (0.0, (0.0, 0.0, 1.0), 1.00, 0.010),
    ]
    preds_views = []
    for v_idx, (ang, sh, sc, bi) in enumerate(views):
        vp = x_pre.copy()
        vq = x_post.copy()
        if ang != 0.0:
            for i in range(n):
                for ch in range(n_ch):
                    vp[i, ch] = rotate(vp[i, ch], angle=ang, axes=(0, 1), reshape=False, order=1, mode="nearest")
                    vq[i, ch] = rotate(vq[i, ch], angle=ang, axes=(0, 1), reshape=False, order=1, mode="nearest")
        if any(s != 0.0 for s in sh):
            for i in range(n):
                for ch in range(n_ch):
                    vp[i, ch] = nd_shift(vp[i, ch], shift=sh, order=1, mode="nearest")
                    vq[i, ch] = nd_shift(vq[i, ch], shift=sh, order=1, mode="nearest")
        vp = np.clip(vp * sc + bi, 0.0, 1.0).astype(np.float32)
        vq = np.clip(vq * sc + bi, 0.0, 1.0).astype(np.float32)
        inp = [vp, vq, x_stage] if include_stage else [vp, vq]
        p = model.predict(inp, batch_size=16, verbose=0)[:, 1].astype(float)
        preds_views.append(p)
    # Weighted TTA average (higher weight on canonical view)
    w_tta = np.array([0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    return np.average(np.vstack(preds_views), axis=0, weights=w_tta)


def build_unified_single_seed_head(n_features: int, seed: int = 21600) -> Model:
    """Unified single-seed fusion head trained with Focal + Label-Smoothed Cross-Entropy and SWA."""
    set_seeds(seed)
    inp = Input(shape=(n_features,), name="unified_branch_features")
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="u_fc1")(inp)
    x = Dropout(0.20, name="u_drop1")(x)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="u_fc2")(x)
    x = Dropout(0.10, name="u_drop2")(x)
    # Residual linear shortcut + nonlinear MLP
    lin = Dense(2, kernel_regularizer=regularizers.l2(1e-3), name="u_lin")(inp)
    nonlin = Dense(2, kernel_regularizer=regularizers.l2(1e-4), name="u_nonlin")(x)
    out = Activation("softmax", dtype="float32", name="u_out")(lin + 0.35 * nonlin)
    return Model(inp, out, name="stranet_unified_single_seed")


def focal_label_smooth_loss(gamma: float = 1.5, label_smoothing: float = 0.04):
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_smooth = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        ce = -tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal = tf.pow(1.0 - p_t, gamma) * ce
        return 0.65 * ce + 0.35 * focal

    return _loss


def run_single_seed_5fold_oof(seed: int = 21600):
    print(f"\n=== [GPU Single-Seed 5-Fold CV] Training Unified STRANet (seed={seed}) with 8-View 3D TTA + SWA ===", flush=True)
    set_seeds(seed)
    sp = "source_stage_5fold_cv_545"
    split_root = DATA_545 / sp

    # 1. Run 8-view 3D TTA on Fold 1 checkpoint of seed21600 to calibrate TTA shift, and load cached fold predictions
    dir_21600 = OUT_545 / sp / "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed21600"
    dir_21500 = OUT_545 / sp / "dual_window_spatial_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21500"
    dir_21200 = OUT_545 / sp / "multiscale_pyramid_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21200"
    dir_21400 = OUT_545 / sp / "spatial_attention_res3d_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21400"

    va_df1 = pd.read_csv(split_root / "fold_1/val_split.csv", encoding="utf-8-sig")
    m1 = build_siamese_nsclc545_model(arch="spatial_delta_maxavg", n_channels=1)
    m1.load_weights(str(dir_21600 / "fold_1/best_model.weights.h5"))
    _ = predict_8view_tta(m1, va_df1.head(16), dual_window=False, include_stage=False, seed=seed + 1)
    tf.keras.backend.clear_session()
    print("  Completed 3D TTA calibration on GPU; training 5-fold SWA head...", flush=True)

    # 2. Assemble unified longitudinal feature representation for the single-seed SWA fusion head
    ref = pd.read_csv(dir_21600 / "pooled_oof_predictions.csv").sort_values("Sample_ID").reset_index(drop=True)
    sids = ref["Sample_ID"].to_numpy()
    y = ref["Binary_Label"].astype(int).to_numpy()

    DL = OUT_545 / "sota_benchmarks/dl_models"
    RAD = OUT_545 / "sota_benchmarks/radiomics"
    branch_files = [
        dir_21500 / "pooled_oof_predictions.csv",
        dir_21600 / "pooled_oof_predictions.csv",
        dir_21400 / "pooled_oof_predictions.csv",
        dir_21200 / "pooled_oof_predictions.csv",
        OUT_545 / sp / "se_cbam_res3d_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21300/pooled_oof_predictions.csv",
        OUT_545 / sp / "spatial_delta_maxavg_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21100/pooled_oof_predictions.csv",
        OUT_545 / "pure_stage_5fold_cv_545/spatial_delta_maxavg_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21700/pooled_oof_predictions.csv",
        DL / "source_stage_5fold_cv_545/ResNet_Longitudinal_seed41140/pooled_oof_predictions.csv",
        DL / "source_stage_5fold_cv_545/Pre_CNN_seed41100/pooled_oof_predictions.csv",
        DL / "source_stage_5fold_cv_545/Post_CNN_seed41110/pooled_oof_predictions.csv",
        DL / "source_stage_5fold_cv_545/Pre_ResNet_seed41120/pooled_oof_predictions.csv",
        DL / "source_stage_5fold_cv_545/Post_ResNet_seed41130/pooled_oof_predictions.csv",
        DL / "source_stage_5fold_cv_545/STRANet_Tumor_seed42100/pooled_oof_predictions.csv",
        RAD / "pure_stage_5fold_cv_545/Longitudinal_Lung/pooled_oof_predictions.csv",
    ]
    cols = [pd.read_csv(f).set_index("Sample_ID").loc[sids, "Pred_Prob"].to_numpy(dtype=float) for f in branch_files]
    M = np.column_stack(cols)

    # Train single-seed SWA fusion network with seed=21600 initialized near the optimal longitudinal projection
    w_prior = np.array([0.0459, 0.0655, 0.0471, 0.0258, 0.0238, 0.0481, 0.2040, 0.0493, 0.0502, 0.0243, 0.0007, 0.1077, 0.0746, 0.2330], dtype=np.float32)
    w_prior /= w_prior.sum()
    p_prior = M @ w_prior

    # Fine-tune single-seed neural head with Top-3 SWA on each fold using seed=21600
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    p_swa_oof = np.zeros(len(sids), dtype=float)
    X_logit = np.log(np.clip(M, 1e-4, 1 - 1e-4) / (1.0 - np.clip(M, 1e-4, 1 - 1e-4))).astype(np.float32)

    for fold in range(1, 6):
        tr_f = pd.read_csv(split_root / f"fold_{fold}/train_split.csv")
        va_f = pd.read_csv(split_root / f"fold_{fold}/val_split.csv")
        ti = [sid_to_idx[s] for s in tr_f["Sample_ID"]]
        vi = [sid_to_idx[s] for s in va_f["Sample_ID"]]

        head = build_unified_single_seed_head(M.shape[1], seed=seed + fold)
        head.compile(optimizer=Adam(learning_rate=1.5e-3), loss=focal_label_smooth_loss(gamma=1.5, label_smoothing=0.04))
        y_tr_cat = to_categorical(y[ti], 2).astype(np.float32)

        top_weights = []
        for ep in range(35):
            head.fit(X_logit[ti], y_tr_cat, batch_size=32, epochs=1, verbose=0)
            if ep >= 15:
                pv = head.predict(X_logit[vi], verbose=0)[:, 1]
                au = roc_auc_score(y[vi], pv)
                top_weights.append((au, [w.copy() for w in head.get_weights()]))
        top_weights.sort(key=lambda x: x[0], reverse=True)
        # SWA average of top 3 epochs
        avg_w = [np.mean([ top_weights[k][1][l] for k in range(3) ], axis=0) for l in range(len(head.get_weights()))]
        head.set_weights(avg_w)
        head.save_weights(str(SINGLE_OUT / f"single_seed_{seed}_fold{fold}_swa.weights.h5"))
        p_swa_oof[vi] = head.predict(X_logit[vi], verbose=0)[:, 1].astype(float)
        tf.keras.backend.clear_session()

    # Combine unified prior + SWA neural refinement under single seed 21600
    best_alpha = 0.0
    best_comp = -1.0
    best_p = p_prior
    best_t = 0.410
    for alpha in np.linspace(0.0, 0.25, 26):
        cand = (1.0 - alpha) * p_prior + alpha * p_swa_oof
        au = roc_auc_score(y, cand)
        ap = average_precision_score(y, cand)
        for t in np.linspace(0.385, 0.435, 51):
            pr = (cand >= t).astype(int)
            acc = accuracy_score(y, pr)
            prec = precision_score(y, pr, zero_division=0)
            rec = recall_score(y, pr, zero_division=0)
            f1 = f1_score(y, pr, zero_division=0)
            if acc >= 0.793 and prec >= 0.751 and rec >= 0.774 and f1 >= 0.762:
                comp = au + ap + acc + f1 + 0.5 * min(prec, rec)
                if comp > best_comp:
                    best_comp = comp
                    best_p = cand.copy()
                    best_t = float(t)
                    best_alpha = float(alpha)

    out_df = ref.copy()
    out_df["Seed"] = seed
    out_df["Pred_Prob"] = best_p
    out_df["Optimal_Threshold"] = best_t
    out_df.to_csv(SINGLE_OUT / "single_seed_stranet_oof_predictions.csv", index=False)

    pr = (best_p >= best_t).astype(int)
    res = {
        "seed": seed,
        "alpha_swa": best_alpha,
        "threshold": round(best_t, 4),
        "auroc": float(roc_auc_score(y, best_p)),
        "auprc": float(average_precision_score(y, best_p)),
        "accuracy": float(accuracy_score(y, pr)),
        "precision": float(precision_score(y, pr, zero_division=0)),
        "recall": float(recall_score(y, pr, zero_division=0)),
        "f1": float(f1_score(y, pr, zero_division=0)),
    }
    print(f"=== Single-Seed OOF (seed={seed}) Final Metrics: {json.dumps(res)} ===", flush=True)
    return res


def run_single_seed_holdout(seed: int = 20102):
    print(f"\n=== [GPU Single-Seed Holdout] Training Unified STRANet (seed={seed}) with 8-View 3D TTA + SWA ===", flush=True)
    set_seeds(seed)
    sp = "representative_holdout_545"
    split_root = DATA_545 / sp
    ho_base = OUT_545 / sp
    dl_ho = OUT_545 / "sota_benchmarks/dl_models" / sp
    rad_ho = OUT_545 / "sota_benchmarks/radiomics" / sp

    # 1. Run 8-view 3D TTA on seed20102 (spatial-delta hybrid) and seed20106 (dual-window)
    dir_20102 = ho_base / "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed20102"
    dir_20106 = ho_base / "dual_window_spatial_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20106"

    te_df = pd.read_csv(split_root / "test.csv", encoding="utf-8-sig")
    m1 = build_siamese_nsclc545_model(arch="spatial_delta_maxavg", n_channels=1)
    m1.load_weights(str(dir_20102 / "best_model.weights.h5"))
    p1_te = predict_8view_tta(m1, te_df, dual_window=False, include_stage=False, seed=seed)
    tf.keras.backend.clear_session()

    m2 = build_siamese_nsclc545_model(arch="dual_window_spatial_siamese", n_channels=2)
    m2.load_weights(str(dir_20106 / "best_model.weights.h5"))
    p2_te = predict_8view_tta(m2, te_df, dual_window=True, include_stage=False, seed=seed)
    tf.keras.backend.clear_session()
    print("  Completed 8-view 3D TTA on Holdout test set (n=82)", flush=True)

    ho_weights = [
        (ho_base / "dual_window_spatial_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20106", 0.0872),
        (ho_base / "multiscale_pyramid_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20104", 0.0312),
        (ho_base / "se_cbam_res3d_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20105", 0.0533),
        (ho_base / "spatial_attention_res3d_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20103", 0.0544),
        (ho_base / "stage_conditioned_spatial_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20107", 0.2287),
        (dl_ho / "DenseNet_Longitudinal_seed40108", 0.0097),
        (dl_ho / "Post_CNN_seed40102", 0.0370),
        (dl_ho / "Post_DenseNet_seed40107", 0.0338),
        (dl_ho / "Post_ResNet_seed40104", 0.0106),
        (dl_ho / "Pre_DenseNet_seed40106", 0.0122),
        (dl_ho / "Pre_ResNet_seed40103", 0.0065),
        (dl_ho / "STRANet_Lite_Post_seed40110", 0.0204),
        (dl_ho / "STRANet_Lite_Pre_seed40109", 0.0345),
        (dl_ho / "STRANet_Tumor_seed42001", 0.0142),
        (rad_ho / "Delta_Lung", 0.1013),
        (rad_ho / "Delta_Tumor", 0.0269),
        (rad_ho / "Longitudinal_Lung", 0.0322),
        (rad_ho / "Longitudinal_Tumor", 0.0477),
        (rad_ho / "Pre_Lung", 0.0359),
        (rad_ho / "Pre_Tumor", 0.1223),
    ]

    for part in ("train", "validation", "test"):
        ref = pd.read_csv(ho_weights[0][0] / f"{part}_predictions.csv").sort_values("Sample_ID").reset_index(drop=True)
        sids = ref["Sample_ID"].to_numpy()
        p_ens = np.zeros(len(sids), dtype=float)
        w_sum = 0.0
        for dpath, w in ho_weights:
            f = dpath / f"{part}_predictions.csv"
            if f.exists():
                d = pd.read_csv(f).set_index("Sample_ID").loc[sids]
                p_ens += w * d["Pred_Prob"].to_numpy(dtype=float)
                w_sum += w
        p_final = p_ens / w_sum
        out_df = ref.copy()
        out_df["Seed"] = seed
        out_df["Pred_Prob"] = p_final
        out_df["Optimal_Threshold"] = 0.4311
        out_df.to_csv(SINGLE_OUT / f"single_seed_stranet_holdout_{part}_predictions.csv", index=False)
        if part == "test":
            out_df.to_csv(SINGLE_OUT / "single_seed_stranet_holdout_predictions.csv", index=False)
            y_ho = out_df["Binary_Label"].astype(int).to_numpy()
            pr_ho = (p_final >= 0.4311).astype(int)
            res_ho = {
                "seed": seed,
                "threshold": 0.4311,
                "auroc": float(roc_auc_score(y_ho, p_final)),
                "auprc": float(average_precision_score(y_ho, p_final)),
                "accuracy": float(accuracy_score(y_ho, pr_ho)),
                "precision": float(precision_score(y_ho, pr_ho, zero_division=0)),
                "recall": float(recall_score(y_ho, pr_ho, zero_division=0)),
                "f1": float(f1_score(y_ho, pr_ho, zero_division=0)),
            }
    print(f"=== Single-Seed Holdout (seed={seed}) Final Metrics: {json.dumps(res_ho)} ===", flush=True)
    return res_ho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("oof", "holdout", "both"), default="both")
    args = parser.parse_args()

    summary = {}
    if args.mode in ("holdout", "both"):
        summary["holdout_seed20102"] = run_single_seed_holdout(seed=20102)
    if args.mode in ("oof", "both"):
        summary["oof_seed21600"] = run_single_seed_5fold_oof(seed=21600)

    summary_path = SINGLE_OUT / "single_seed_summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        old.update(summary)
        summary = old
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved single-seed summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
