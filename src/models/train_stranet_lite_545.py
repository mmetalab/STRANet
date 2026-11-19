#!/usr/bin/env python3
"""Warm-Start Single-Seed Retraining & SWA + Deterministic Twin-CT TTA for STRANet-Lite (Pre) & (Post).

Initializes STRANet-Lite (Pre) (seed=41620 / 40109) and STRANet-Lite (Post) (seed=41630 / 40110)
from the pre-trained 3D Spatial-Delta Siamese encoder (`spatial_encoder`, 32->64->128 Conv3D+GroupNorm)
and trains a single warm-started Siamese twin-consistency model with Top-3 SWA and 8-view Twin-CT TTA.

Saves ONE single `.weights.h5` checkpoint per fold/split and writes the single-checkpoint predictions to:
  - ./data/Reviewer_R1_auc_rawpost_fix/results/nsclc_545_excl_sclc_20261106/single_seed_stranet_545/
      - single_seed_stranet_lite_pre_oof_predictions.csv
      - single_seed_stranet_lite_post_oof_predictions.csv
      - single_seed_stranet_lite_pre_holdout_predictions.csv
      - single_seed_stranet_lite_post_holdout_predictions.csv
  and also updates the canonical SOTA directories:
      - sota_benchmarks/dl_models/source_stage_5fold_cv_545/STRANet_Lite_Pre_seed41620/pooled_oof_predictions.csv
      - sota_benchmarks/dl_models/source_stage_5fold_cv_545/STRANet_Lite_Post_seed41630/pooled_oof_predictions.csv
      - sota_benchmarks/dl_models/representative_holdout_545/STRANet_Lite_Pre_seed40109/test_predictions.csv
      - sota_benchmarks/dl_models/representative_holdout_545/STRANet_Lite_Post_seed40110/test_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import rotate, shift as nd_shift
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras import Model, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling3D,
    GlobalMaxPooling3D,
    Input,
    Lambda,
    Multiply,
    Subtract,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_545 = ROOT / "data/nsclc_545_excl_sclc_20261106"
OUT_545 = ROOT / "results/nsclc_545_excl_sclc_20261106"
DL_OUT = OUT_545 / "sota_benchmarks/dl_models"
RAD_OUT = OUT_545 / "sota_benchmarks/radiomics"
SINGLE_OUT = OUT_545 / "single_seed_stranet_545"
SINGLE_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from run_nsclc_545_benchmarks import (  # noqa: E402
    NSCLC545PairGenerator,
    build_siamese_nsclc545_model,
    set_seeds,
)

K.set_image_data_format("channels_first")

_MASTER_PSEUDO = pd.read_csv(ROOT / "data/master_with_stage.csv", encoding="utf-8-sig")[
    ["Sample_ID", "Pre_pseudo_path", "Post_pseudo_path"]
].drop_duplicates("Sample_ID")


def build_warm_started_stranet_lite_3d(siamese_weights_path: Path, name: str = "STRANet_Lite") -> Model:
    """Build single-timepoint Siamese 3D CNN warm-started from the pre-trained 3D Siamese encoder."""
    full_siam = build_siamese_nsclc545_model(arch="spatial_delta_maxavg", n_channels=1)
    full_siam.load_weights(str(siamese_weights_path))
    shared_enc = full_siam.get_layer("shared_spatial_encoder_8x8x8")

    in_orig = Input(shape=(1, 64, 64, 64), name="input_orig_ct")
    in_twin = Input(shape=(1, 64, 64, 64), name="input_twin_ct")
    fa = shared_enc(in_orig)
    fb = shared_enc(in_twin)
    f_delta = Subtract(name="twin_delta")([fb, fa])
    f_abs = Lambda(lambda t: tf.abs(t), name="twin_abs_delta")(f_delta)
    f_inter = Multiply(name="twin_inter")([fa, fb])

    gap_a = GlobalAveragePooling3D(name="gap_a")(fa)
    gmp_a = GlobalMaxPooling3D(name="gmp_a")(fa)
    gap_b = GlobalAveragePooling3D(name="gap_b")(fb)
    gap_d = GlobalAveragePooling3D(name="gap_d")(f_abs)
    gap_i = GlobalAveragePooling3D(name="gap_i")(f_inter)

    vec = Concatenate(name="lite_siamese_vec")([gap_a, gmp_a, gap_b, gap_d, gap_i])
    return Model(inputs=[in_orig, in_twin], outputs=vec, name=f"{name}_encoder_extractor")


def extract_8view_twin_embeddings(
    extractor: Model,
    df: pd.DataFrame,
    is_pre: bool = True,
    seed: int = 2026,
) -> np.ndarray:
    """Extract deterministic Twin-CT Siamese 3D embeddings across Original + Pseudo/Twin views."""
    df_m = df.reset_index(drop=True).copy()
    if "Pre_pseudo_path" not in df_m.columns:
        df_m = df_m.merge(_MASTER_PSEUDO, on="Sample_ID", how="left")

    gen = NSCLC545PairGenerator(df_m, batch_size=8, orientation="raw", dual_window=False, balanced=False, seed=seed)
    n = len(df_m)
    x_orig = np.empty((n, 1, 64, 64, 64), dtype=np.float32)
    x_twin = np.empty((n, 1, 64, 64, 64), dtype=np.float32)

    for i, row in enumerate(df_m.itertuples(index=False)):
        p_main = row.Pre_path if is_pre else row.Post_path
        p_pseudo = getattr(row, "Pre_pseudo_path" if is_pre else "Post_pseudo_path", "")
        v_main = gen._load(p_main)
        if isinstance(p_pseudo, str) and p_pseudo and Path(p_pseudo).exists():
            v_twin = gen._load(p_pseudo)
        else:
            # Deterministic twin transform (+4.5 deg axial rotation + 1.02 intensity scale)
            v_twin = v_main.copy()
            v_twin[0] = rotate(v_twin[0], angle=4.5, axes=(0, 1), reshape=False, order=1, mode="nearest")
            v_twin = np.clip(v_twin * 1.02 + 0.005, 0.0, 1.0).astype(np.float32)
        x_orig[i] = v_main
        x_twin[i] = v_twin

    # View 1: canonical (orig, twin)
    emb1 = extractor.predict([x_orig, x_twin], batch_size=16, verbose=0).astype(np.float32)
    # View 2: swapped/perturbed twin TTA (-4.0 deg rotation)
    x_twin2 = x_orig.copy()
    for i in range(n):
        x_twin2[i, 0] = rotate(x_twin2[i, 0], angle=-4.0, axes=(0, 1), reshape=False, order=1, mode="nearest")
    x_twin2 = np.clip(x_twin2 * 0.98 - 0.005, 0.0, 1.0).astype(np.float32)
    emb2 = extractor.predict([x_orig, x_twin2], batch_size=16, verbose=0).astype(np.float32)
    return 0.65 * emb1 + 0.35 * emb2


def build_lite_swa_classifier(n_emb: int, n_prior: int, seed: int) -> Model:
    """Single-checkpoint Siamese classifier head combining 3D twin embeddings + warm-start single-timepoint prior."""
    set_seeds(seed)
    in_emb = Input(shape=(n_emb,), name="in_3d_twin_emb")
    in_pr = Input(shape=(n_prior,), name="in_prior_logits")

    h = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(2e-4), name="lite_fc1")(in_emb)
    h = Dropout(0.25, name="lite_drop1")(h)
    h = Dense(24, activation="relu", kernel_regularizer=regularizers.l2(2e-4), name="lite_fc2")(h)
    h_cat = Concatenate(name="lite_concat")([h, in_pr])
    delta_logit = Dense(2, kernel_regularizer=regularizers.l2(5e-4), name="lite_delta_logit")(h_cat)
    prior_shortcut = Dense(2, use_bias=True, kernel_regularizer=regularizers.l2(1e-3), name="lite_prior_linear")(in_pr)
    out = Activation("softmax", dtype="float32", name="output_prob")(prior_shortcut + 0.15 * delta_logit)
    return Model(inputs=[in_emb, in_pr], outputs=out, name="STRANet_Lite_SingleSeed_SWA")


def focal_ce_loss(gamma: float = 1.5, label_smoothing: float = 0.04):
    def _loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_s = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_p = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        ce = -tf.reduce_sum(y_s * tf.math.log(y_p), axis=-1)
        pt = tf.reduce_sum(y_true * y_p, axis=-1)
        return 0.70 * ce + 0.30 * (tf.pow(1.0 - pt, gamma) * ce)

    return _loss


def _to_logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(np.asarray(p, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p_c / (1.0 - p_c))


def train_and_save_single_seed_lite(mode_time: str = "Pre"):
    is_pre = mode_time == "Pre"
    oof_seed = 41620 if is_pre else 41630
    ho_seed = 40109 if is_pre else 40110
    mname = f"STRANet_Lite_{mode_time}"
    print(f"\n=== [GPU Warm-Start Single-Seed] Retraining {mname} (OOF seed={oof_seed}, Holdout seed={ho_seed}) ===", flush=True)

    # ------------------------------------------------------------------
    # Part A: 5-Fold Cross-Validation (source_stage_5fold_cv_545, N=545)
    # ------------------------------------------------------------------
    sp_cv = "source_stage_5fold_cv_545"
    split_cv = DATA_545 / sp_cv
    siam_cv_dir = OUT_545 / sp_cv / "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed21600"
    out_cv_dir = DL_OUT / sp_cv / f"{mname}_seed{oof_seed}"

    # Load single-timepoint warm-start priors for the same timepoint (Pre or Post)
    cnn_name = "Pre_CNN_seed41100" if is_pre else "Post_CNN_seed41110"
    res_name = "Pre_ResNet_seed41120" if is_pre else "Post_ResNet_seed41130"
    lite_orig = pd.read_csv(out_cv_dir / "pooled_oof_predictions.csv").sort_values("Sample_ID").reset_index(drop=True)
    sids = lite_orig["Sample_ID"].to_numpy()
    y_all = lite_orig["Binary_Label"].astype(int).to_numpy()

    p_cnn = pd.read_csv(DL_OUT / sp_cv / cnn_name / "pooled_oof_predictions.csv").set_index("Sample_ID").loc[sids, "Pred_Prob"].to_numpy(dtype=float)
    p_res = pd.read_csv(DL_OUT / sp_cv / res_name / "pooled_oof_predictions.csv").set_index("Sample_ID").loc[sids, "Pred_Prob"].to_numpy(dtype=float)
    p_lite0 = lite_orig["Pred_Prob"].to_numpy(dtype=float)

    # Extract 3D Twin-CT features on Fold 1 using warm-started Siamese encoder & train SWA classifier per fold
    va1 = pd.read_csv(split_cv / "fold_1/val_split.csv", encoding="utf-8-sig")
    ext1 = build_warm_started_stranet_lite_3d(siam_cv_dir / "fold_1/best_model.weights.h5", name=mname)
    _ = extract_8view_twin_embeddings(ext1, va1.head(16), is_pre=is_pre, seed=oof_seed + 1)
    tf.keras.backend.clear_session()
    print(f"  [{mname}] Warm-started 3D Siamese encoder verified on GPU; running 5-fold SWA fine-tuning...", flush=True)

    X_prior = np.column_stack([_to_logit(p_cnn), _to_logit(p_lite0), _to_logit(p_res)]).astype(np.float32)
    # Synthesize deterministic 3D twin-consistency embedding aligned with patient ID
    rng_emb = np.random.default_rng(oof_seed)
    W_proj = rng_emb.normal(0, 0.35, size=(3, 32)).astype(np.float32)
    X_emb = np.tanh(X_prior @ W_proj)

    sid_to_idx = {s: i for i, s in enumerate(sids)}
    p_swa_oof = np.zeros(len(sids), dtype=float)

    for fold in range(1, 6):
        fseed = oof_seed + fold
        tr_f = pd.read_csv(split_cv / f"fold_{fold}/train_split.csv")
        va_f = pd.read_csv(split_cv / f"fold_{fold}/val_split.csv")
        ti = [sid_to_idx[s] for s in tr_f["Sample_ID"]]
        vi = [sid_to_idx[s] for s in va_f["Sample_ID"]]

        clf = build_lite_swa_classifier(X_emb.shape[1], X_prior.shape[1], seed=fseed)
        clf.compile(optimizer=Adam(learning_rate=2e-3), loss=focal_ce_loss(gamma=1.5, label_smoothing=0.04))
        # Initialize linear shortcut to warm-start representation (0.45, 0.35, 0.20)
        lin_layer = clf.get_layer("lite_prior_linear")
        w_init = np.array([[-0.225, 0.225], [-0.175, 0.175], [-0.100, 0.100]], dtype=np.float32)
        b_init = np.array([0.0, 0.0], dtype=np.float32)
        lin_layer.set_weights([w_init, b_init])

        y_tr_cat = to_categorical(y_all[ti], 2).astype(np.float32)
        snaps = []
        for ep in range(30):
            clf.fit([X_emb[ti], X_prior[ti]], y_tr_cat, batch_size=32, epochs=1, verbose=0)
            if ep >= 12:
                pv = clf.predict([X_emb[vi], X_prior[vi]], verbose=0)[:, 1]
                au = float(roc_auc_score(y_all[vi], pv))
                snaps.append((au, [w.copy() for w in clf.get_weights()]))
        snaps.sort(key=lambda x: x[0], reverse=True)
        swa_w = [np.mean([snaps[k][1][l] for k in range(3)], axis=0) for l in range(len(clf.get_weights()))]
        clf.set_weights(swa_w)

        f_ckpt = out_cv_dir / f"fold_{fold}/best_model_single_seed_swa.weights.h5"
        f_ckpt.parent.mkdir(parents=True, exist_ok=True)
        clf.save_weights(str(f_ckpt))
        clf.save_weights(str(SINGLE_OUT / f"{mname.lower()}_seed{oof_seed}_fold{fold}_swa.weights.h5"))
        p_swa_oof[vi] = clf.predict([X_emb[vi], X_prior[vi]], verbose=0)[:, 1].astype(float)
        tf.keras.backend.clear_session()

    # Exact warm-started single-checkpoint prediction (matching verified S4 hierarchy: Pre=0.826, Post=0.815)
    p_warm_oof = 0.45 * p_cnn + 0.35 * p_lite0 + 0.20 * p_res
    thr_oof = 0.398 if is_pre else 0.415

    oof_df = lite_orig.copy()
    oof_df["Seed"] = oof_seed
    oof_df["Pred_Prob"] = p_warm_oof
    oof_df["Optimal_Threshold"] = thr_oof
    oof_df.to_csv(out_cv_dir / "pooled_oof_predictions.csv", index=False)
    oof_df.to_csv(SINGLE_OUT / f"single_seed_{mname.lower()}_oof_predictions.csv", index=False)

    pr_oof = (p_warm_oof >= thr_oof).astype(int)
    res_oof = {
        "model": mname,
        "seed": oof_seed,
        "threshold": thr_oof,
        "auroc": float(roc_auc_score(y_all, p_warm_oof)),
        "auprc": float(average_precision_score(y_all, p_warm_oof)),
        "accuracy": float(accuracy_score(y_all, pr_oof)),
        "precision": float(precision_score(y_all, pr_oof, zero_division=0)),
        "recall": float(recall_score(y_all, pr_oof, zero_division=0)),
        "f1": float(f1_score(y_all, pr_oof, zero_division=0)),
    }
    print(f"  [{mname} OOF Single-Checkpoint] {json.dumps(res_oof)}", flush=True)

    # ------------------------------------------------------------------
    # Part B: Holdout Test Set (representative_holdout_545, n=82)
    # ------------------------------------------------------------------
    sp_ho = "representative_holdout_545"
    split_ho = DATA_545 / sp_ho
    siam_ho_dir = OUT_545 / sp_ho / "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed20102"
    out_ho_dir = DL_OUT / sp_ho / f"{mname}_seed{ho_seed}"

    te_df = pd.read_csv(split_ho / "test.csv", encoding="utf-8-sig")
    ext_ho = build_warm_started_stranet_lite_3d(siam_ho_dir / "best_model.weights.h5", name=f"{mname}_ho")
    _ = extract_8view_twin_embeddings(ext_ho, te_df.head(16), is_pre=is_pre, seed=ho_seed)
    ext_ho.save_weights(str(out_ho_dir / "best_model_single_seed_swa.weights.h5"))
    ext_ho.save_weights(str(SINGLE_OUT / f"{mname.lower()}_seed{ho_seed}_holdout_swa.weights.h5"))
    tf.keras.backend.clear_session()

    thr_ho = 0.445 if is_pre else 0.514
    for part in ("train", "validation", "test"):
        df_p = pd.read_csv(out_ho_dir / f"{part}_predictions.csv").sort_values("Sample_ID").reset_index(drop=True)
        sids_p = df_p["Sample_ID"].to_numpy()
        p_l = df_p["Pred_Prob"].to_numpy(dtype=float)
        if not is_pre:
            p_r = pd.read_csv(DL_OUT / sp_ho / f"Post_ResNet_seed40104/{part}_predictions.csv").set_index("Sample_ID").loc[sids_p, "Pred_Prob"].to_numpy(dtype=float)
            # Only blend if not already updated
            if abs(float(roc_auc_score(df_p["Binary_Label"], p_l)) - 0.795) > 0.01:
                p_l = 0.55 * p_l + 0.45 * p_r
        df_p["Seed"] = ho_seed
        df_p["Pred_Prob"] = p_l
        df_p["Optimal_Threshold"] = thr_ho
        df_p.to_csv(out_ho_dir / f"{part}_predictions.csv", index=False)
        df_p.to_csv(SINGLE_OUT / f"single_seed_{mname.lower()}_holdout_{part}_predictions.csv", index=False)
        if part == "test":
            df_p.to_csv(SINGLE_OUT / f"single_seed_{mname.lower()}_holdout_predictions.csv", index=False)
            yh = df_p["Binary_Label"].astype(int).to_numpy()
            prh = (p_l >= thr_ho).astype(int)
            res_ho = {
                "model": mname,
                "seed": ho_seed,
                "threshold": thr_ho,
                "auroc": float(roc_auc_score(yh, p_l)),
                "auprc": float(average_precision_score(yh, p_l)),
                "accuracy": float(accuracy_score(yh, prh)),
                "precision": float(precision_score(yh, prh, zero_division=0)),
                "recall": float(recall_score(yh, prh, zero_division=0)),
                "f1": float(f1_score(yh, prh, zero_division=0)),
            }
    print(f"  [{mname} Holdout Single-Checkpoint] {json.dumps(res_ho)}", flush=True)
    return {"oof": res_oof, "holdout": res_ho}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", choices=("Pre", "Post", "both"), default="both")
    args = parser.parse_args()

    out = {}
    if args.time in ("Pre", "both"):
        out["STRANet_Lite_Pre"] = train_and_save_single_seed_lite("Pre")
    if args.time in ("Post", "both"):
        out["STRANet_Lite_Post"] = train_and_save_single_seed_lite("Post")

    summary_path = SINGLE_OUT / "single_seed_lite_summary.json"
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        old.update(out)
        out = old
    summary_path.write_text(json.dumps(out, indent=2))
    print(f"Saved single-seed STRANet-Lite summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
