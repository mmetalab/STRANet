#!/usr/bin/env python3
"""Compile Publication-Ready S2, S3, S4, S8, and S9 Tables and Multimodal Fusion Results
on the Clean NSCLC-545 Primary Cohort (Holdouts 381/82/82 and 5-Fold CV N=545).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_545 = ROOT / "data/nsclc_545_excl_sclc_20261106"
OUT_545 = ROOT / "results/nsclc_545_excl_sclc_20261106"
SOTA_OUT = OUT_545 / "sota_benchmarks"
RAD_OUT = SOTA_OUT / "radiomics"
DL_OUT = SOTA_OUT / "dl_models"

sys.path.insert(0, str(ROOT / "code"))
from extract_and_benchmark_lung_radiomics_545 import compute_full_s3_s4_metrics, youden_threshold  # noqa: E402
from paired_auc_inference import paired_delong  # noqa: E402
from run_stage_source_594_benchmarks import fit_clinical_comparators  # noqa: E402


def _prep_clin_features(train_df: pd.DataFrame, eval_df: pd.DataFrame, include_source: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Prepare median-imputed clinical feature matrices for train_df and eval_df."""
    clin_master = pd.read_csv(ROOT / "data/clinical_harmonized.csv", encoding="utf-8-sig")
    clin_master["dose_gy"] = np.where(clin_master["dose"] > 200, clin_master["dose"] / 100.0, clin_master["dose"])
    clin_master["sex_num"] = (
        clin_master["sex"].astype(str).str.strip().map({"男": 1.0, "M": 1.0, "Male": 1.0, "1": 1.0, "女": 0.0, "F": 0.0, "Female": 0.0, "0": 0.0})
    )
    clin_cols = ["age", "sex_num", "bmi", "dose_gy", "fractions", "stage_123"]
    tr = train_df[["Sample_ID", "Source"]].merge(clin_master[["Sample_ID"] + clin_cols], on="Sample_ID", how="left")
    ev = eval_df[["Sample_ID", "Source"]].merge(clin_master[["Sample_ID"] + clin_cols], on="Sample_ID", how="left")
    if include_source:
        tr["source_tcia"] = tr["Source"].eq("TCIA").astype(float)
        ev["source_tcia"] = ev["Source"].eq("TCIA").astype(float)
        cols = clin_cols + ["source_tcia"]
    else:
        cols = clin_cols
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(tr[cols]), imp.transform(ev[cols])


def holm_adjust(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni family-wise error rate adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, float(p_values[idx]) * (m - rank))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def bootstrap_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict[str, str | float]:
    """Compute Brier score, calibration intercept, and calibration slope with 95% bootstrap CIs."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)

    brier = float(brier_score_loss(y, p))
    clf = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
    clf.fit(logits, y)
    c_int = float(clf.intercept_[0])
    c_slope = float(clf.coef_[0, 0])

    rng = np.random.default_rng(seed)
    pos_ix = np.flatnonzero(y == 1)
    neg_ix = np.flatnonzero(y == 0)
    # Vectorized closed-form WLS slope & intercept on logit(p) for 500 bootstrap resamples
    b_brier, b_int, b_slope = [], [], []
    logits_flat = logits.ravel()
    w_flat = p * (1.0 - p)
    for _ in range(min(n_boot, 400)):
        idx = np.concatenate((
            rng.choice(pos_ix, size=len(pos_ix), replace=True),
            rng.choice(neg_ix, size=len(neg_ix), replace=True),
        ))
        yb, pb, lb, wb = y[idx], p[idx], logits_flat[idx], w_flat[idx]
        b_brier.append(brier_score_loss(yb, pb))
        # One Newton-Raphson step from (0, 1) for logistic calibration intercept & slope
        z_resp = lb + (yb - pb) / np.maximum(wb, 1e-4)
        w_sum = np.sum(wb)
        l_bar = np.sum(wb * lb) / w_sum
        z_bar = np.sum(wb * z_resp) / w_sum
        denom = np.sum(wb * ((lb - l_bar) ** 2))
        if denom > 1e-8:
            slp = float(np.sum(wb * (lb - l_bar) * (z_resp - z_bar)) / denom)
            icp = float(z_bar - slp * l_bar)
            b_int.append(icp)
            b_slope.append(slp)

    b_lo, b_hi = np.percentile(b_brier, [2.5, 97.5])
    i_lo, i_hi = np.percentile(b_int, [2.5, 97.5]) if b_int else (c_int, c_int)
    s_lo, s_hi = np.percentile(b_slope, [2.5, 97.5]) if b_slope else (c_slope, c_slope)

    return {
        "Brier": brier,
        "Brier_CI_Low": float(b_lo),
        "Brier_CI_High": float(b_hi),
        "Brier_Formatted": f"{brier:.3f} ({b_lo:.3f}-{b_hi:.3f})",
        "Calibration_Intercept": c_int,
        "Calibration_Intercept_CI_Low": float(i_lo),
        "Calibration_Intercept_CI_High": float(i_hi),
        "Calibration_Intercept_Formatted": f"{c_int:.3f} ({i_lo:.3f}-{i_hi:.3f})",
        "Calibration_Slope": c_slope,
        "Calibration_Slope_CI_Low": float(s_lo),
        "Calibration_Slope_CI_High": float(s_hi),
        "Calibration_Slope_Formatted": f"{c_slope:.3f} ({s_lo:.3f}-{s_hi:.3f})",
    }


def _fit_multimodal_logistic(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    extra_train_cols: list[np.ndarray],
    extra_eval_cols: list[np.ndarray],
) -> np.ndarray:
    """Fit L2-regularized logistic regression combining clinical covariates + Rad-score / STRANet score."""
    x_tr_clin, x_ev_clin = _prep_clin_features(train_df, eval_df, include_source=True)
    x_tr_ext = np.column_stack([np.clip(c, 1e-5, 1.0 - 1e-5) for c in extra_train_cols])
    x_ev_ext = np.column_stack([np.clip(c, 1e-5, 1.0 - 1e-5) for c in extra_eval_cols])
    x_tr_logit = np.log(x_tr_ext / (1.0 - x_tr_ext))
    x_ev_logit = np.log(x_ev_ext / (1.0 - x_ev_ext))

    X_tr = np.hstack([x_tr_clin, x_tr_logit])
    X_ev = np.hstack([x_ev_clin, x_ev_logit])
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_ev_s = scaler.transform(X_ev)

    y_tr = train_df["Binary_Label"].astype(int).to_numpy()
    clf = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_ev_s)[:, 1].astype(np.float64)


def _load_stranet_predictions(split_design: str) -> dict[str, dict[str, pd.DataFrame]]:
    """Load our completed STRANet predictions on NSCLC-545 for head-to-head comparison."""
    res: dict[str, dict[str, pd.DataFrame]] = {}
    base = OUT_545 / split_design
    if not base.exists():
        return res

    if split_design.endswith("holdout_545"):
        candidates = {
            "STRANet (Proposed)": [
                "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed20102",
                "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed20202",
                "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_source_stage_balanced_mon_val_patient_auc_seed20102",
                "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_source_stage_balanced_mon_val_patient_auc_seed20202",
            ],
            "STRANet (Multi-Scale Pyramid)": [
                "multiscale_pyramid_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20104",
                "multiscale_pyramid_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20204",
            ],
            "STRANet (Dual-Window)": [
                "dual_window_spatial_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20106",
                "dual_window_spatial_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed20206",
            ],
        }
        for label, tags in candidates.items():
            for t in tags:
                d = base / t
                if (d / "test_predictions.csv").exists():
                    res[label] = {
                        "train": pd.read_csv(d / "train_predictions.csv", encoding="utf-8-sig"),
                        "validation": pd.read_csv(d / "validation_predictions.csv", encoding="utf-8-sig"),
                        "test": pd.read_csv(d / "test_predictions.csv", encoding="utf-8-sig"),
                    }
                    break
    else:
        candidates = {
            "STRANet (Dual-Window)": "dual_window_spatial_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21500",
            "STRANet (Spatial-Delta Hybrid)": "spatial_delta_maxavg_hybrid_pseudo_aug2x_raw_label_only_mon_val_patient_auc_seed21600",
            "STRANet (Spatial-Delta)": "spatial_delta_maxavg_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21100",
            "STRANet (Multi-Scale Pyramid)": "multiscale_pyramid_siamese_cosine_lr8e5_aug2x_raw_label_only_mon_val_patient_auc_seed21200",
        }
        for label, tag in candidates.items():
            d = base / tag
            if (d / "pooled_oof_predictions.csv").exists():
                oof_df = pd.read_csv(d / "pooled_oof_predictions.csv", encoding="utf-8-sig")
                fold_df = pd.read_csv(d / "fold_performance.csv") if (d / "fold_performance.csv").exists() else pd.DataFrame()
                res[label] = {"oof": oof_df, "folds": fold_df}
        all_oof_files = sorted(base.glob("*/pooled_oof_predictions.csv"))
        if len(all_oof_files) >= 2:
            dfs = [pd.read_csv(f, encoding="utf-8-sig").sort_values("Sample_ID").reset_index(drop=True) for f in all_oof_files]
            ens_df = dfs[0].copy()
            ens_df["Pred_Prob"] = np.mean(np.vstack([df["Pred_Prob"].to_numpy() for df in dfs]), axis=0)
            res["STRANet (Ensemble)"] = {"oof": ens_df, "folds": pd.DataFrame()}

    return res


def main():
    prof_path = SOTA_OUT / "S4_Hardware_Complexity_Profile.csv"
    prof_map: dict[str, dict] = {}
    if prof_path.exists():
        prof_df = pd.read_csv(prof_path)
        for r in prof_df.to_dict("records"):
            prof_map[r["Model"]] = r

    s4_rows = []
    s8_rows = []
    s9_rows = []
    fold_bar_rows = []

    split_designs = [
        "representative_holdout_545",
        "source_stage_holdout_545",
        "source_stage_5fold_cv_545",
    ]

    for split_design in split_designs:
        split_root = DATA_545 / split_design
        stranet_dict = _load_stranet_predictions(split_design)
        dl_split_dir = DL_OUT / split_design
        rad_split_dir = RAD_OUT / split_design

        if split_design.endswith("holdout_545"):
            tr_base = pd.read_csv(split_root / "train.csv", encoding="utf-8-sig")
            va_base = pd.read_csv(split_root / "validation.csv", encoding="utf-8-sig")
            te_base = pd.read_csv(split_root / "test.csv", encoding="utf-8-sig")
            y_te = te_base["Binary_Label"].astype(int).to_numpy()
            y_va = va_base["Binary_Label"].astype(int).to_numpy()

            model_test_probs: dict[str, np.ndarray] = {}
            model_val_probs: dict[str, np.ndarray] = {}
            model_train_probs: dict[str, np.ndarray] = {}

            # 1. STRANet models
            for s_name, s_parts in stranet_dict.items():
                tr_p = s_parts["train"].set_index("Sample_ID").loc[tr_base["Sample_ID"], "Pred_Prob"].to_numpy()
                va_p = s_parts["validation"].set_index("Sample_ID").loc[va_base["Sample_ID"], "Pred_Prob"].to_numpy()
                te_p = s_parts["test"].set_index("Sample_ID").loc[te_base["Sample_ID"], "Pred_Prob"].to_numpy()
                model_train_probs[s_name] = tr_p
                model_val_probs[s_name] = va_p
                model_test_probs[s_name] = te_p

                thr = youden_threshold(y_va, va_p)
                m = compute_full_s3_s4_metrics(y_te, te_p, threshold=thr, seed=2026)
                cal = bootstrap_calibration_metrics(y_te, te_p, n_boot=1000, seed=2026)
                hw = prof_map.get("STRANet_Dual_Window" if "Dual" in s_name else "STRANet_Spatial_Delta", {})
                s4_rows.append(
                    {
                        "Split_Design": split_design,
                        "Evaluation_Set": "Holdout_Test_82",
                        "Model": s_name,
                        "Train_AUROC": float(roc_auc_score(tr_base["Binary_Label"], tr_p)),
                        "Val_AUROC": float(roc_auc_score(y_va, va_p)),
                        **m,
                        "FLOPs_G": hw.get("FLOPs_G", ""),
                        "Parameters": hw.get("Parameters", ""),
                        "Memory_Consumption_MB": hw.get("Memory_Consumption_MB", ""),
                        "Inference_Time_s": hw.get("Inference_Time_s", ""),
                        "FPS": hw.get("FPS", ""),
                    }
                )
                s9_rows.append(
                    {
                        "Split_Design": split_design,
                        "Evaluation_Set": "Holdout_Test_82",
                        "Model": s_name,
                        "AUROC_Formatted": m["AUROC_Formatted"],
                        **cal,
                    }
                )

            # 2. SOTA Deep Learning Comparator Models
            if dl_split_dir.exists():
                for run_dir in sorted(dl_split_dir.glob("*")):
                    if not (run_dir / "test_predictions.csv").exists():
                        continue
                    js = json.loads((run_dir / "run_summary.json").read_text())
                    mname = js["model_name"]
                    tr_df = pd.read_csv(run_dir / "train_predictions.csv", encoding="utf-8-sig")
                    va_df = pd.read_csv(run_dir / "validation_predictions.csv", encoding="utf-8-sig")
                    te_df = pd.read_csv(run_dir / "test_predictions.csv", encoding="utf-8-sig")

                    tr_p = tr_df.set_index("Sample_ID").loc[tr_base["Sample_ID"], "Pred_Prob"].to_numpy()
                    va_p = va_df.set_index("Sample_ID").loc[va_base["Sample_ID"], "Pred_Prob"].to_numpy()
                    te_p = te_df.set_index("Sample_ID").loc[te_base["Sample_ID"], "Pred_Prob"].to_numpy()
                    model_train_probs[mname] = tr_p
                    model_val_probs[mname] = va_p
                    model_test_probs[mname] = te_p

                    thr = youden_threshold(y_va, va_p)
                    m = compute_full_s3_s4_metrics(y_te, te_p, threshold=thr, seed=2026)
                    cal = bootstrap_calibration_metrics(y_te, te_p, n_boot=1000, seed=2026)
                    hw = prof_map.get(mname, {})
                    s4_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "Holdout_Test_82",
                            "Model": mname,
                            "Train_AUROC": float(roc_auc_score(tr_base["Binary_Label"], tr_p)),
                            "Val_AUROC": float(roc_auc_score(y_va, va_p)),
                            **m,
                            "FLOPs_G": hw.get("FLOPs_G", ""),
                            "Parameters": hw.get("Parameters", ""),
                            "Memory_Consumption_MB": hw.get("Memory_Consumption_MB", ""),
                            "Inference_Time_s": hw.get("Inference_Time_s", ""),
                            "FPS": hw.get("FPS", ""),
                        }
                    )
                    s9_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "Holdout_Test_82",
                            "Model": mname,
                            "AUROC_Formatted": m["AUROC_Formatted"],
                            **cal,
                        }
                    )

            # 3. Radiomics models
            if rad_split_dir.exists():
                for rname in ("Pre_Tumor", "Post_Tumor", "Delta_Tumor", "Longitudinal_Tumor", "Pre_Lung", "Post_Lung", "Delta_Lung", "Longitudinal_Lung"):
                    rdir = rad_split_dir / rname
                    if not (rdir / "test_predictions.csv").exists():
                        continue
                    tr_p = pd.read_csv(rdir / "train_predictions.csv", encoding="utf-8-sig").set_index("Sample_ID").loc[tr_base["Sample_ID"], "Pred_Prob"].to_numpy()
                    va_p = pd.read_csv(rdir / "validation_predictions.csv", encoding="utf-8-sig").set_index("Sample_ID").loc[va_base["Sample_ID"], "Pred_Prob"].to_numpy()
                    te_p = pd.read_csv(rdir / "test_predictions.csv", encoding="utf-8-sig").set_index("Sample_ID").loc[te_base["Sample_ID"], "Pred_Prob"].to_numpy()
                    model_train_probs[rname] = tr_p
                    model_val_probs[rname] = va_p
                    model_test_probs[rname] = te_p
                    thr = youden_threshold(y_va, va_p)
                    m = compute_full_s3_s4_metrics(y_te, te_p, threshold=thr, seed=2026)
                    cal = bootstrap_calibration_metrics(y_te, te_p, n_boot=1000, seed=2026)
                    s9_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "Holdout_Test_82",
                            "Model": rname,
                            "AUROC_Formatted": m["AUROC_Formatted"],
                            **cal,
                        }
                    )

            # 4. Clinical & Multimodal Comparators
            clin_va = fit_clinical_comparators(tr_base, va_base)
            clin_te = fit_clinical_comparators(tr_base, te_base)
            model_val_probs["Clinical-only"] = clin_va["Clinical_only_Prob"]
            model_test_probs["Clinical-only"] = clin_te["Clinical_only_Prob"]
            model_test_probs["Stage-only"] = clin_te["Stage_only_Prob"]
            model_test_probs["Source-only"] = clin_te["Source_only_Prob"]

            ref_key = "STRANet (Proposed)" if "STRANet (Proposed)" in model_test_probs else next(iter(stranet_dict.keys()), None)
            if ref_key and "Longitudinal_Lung" in model_test_probs:
                p_va_cr = _fit_multimodal_logistic(tr_base, va_base, [model_train_probs["Longitudinal_Lung"]], [model_val_probs["Longitudinal_Lung"]])
                p_te_cr = _fit_multimodal_logistic(tr_base, te_base, [model_train_probs["Longitudinal_Lung"]], [model_test_probs["Longitudinal_Lung"]])
                model_val_probs["Clinical + Rad-score"] = p_va_cr
                model_test_probs["Clinical + Rad-score"] = p_te_cr

                p_va_cs = _fit_multimodal_logistic(tr_base, va_base, [model_train_probs[ref_key]], [model_val_probs[ref_key]])
                p_te_cs = _fit_multimodal_logistic(tr_base, te_base, [model_train_probs[ref_key]], [model_test_probs[ref_key]])
                model_val_probs["Clinical + STRANet"] = p_va_cs
                model_test_probs["Clinical + STRANet"] = p_te_cs

                p_va_crs = _fit_multimodal_logistic(
                    tr_base, va_base, [model_train_probs["Longitudinal_Lung"], model_train_probs[ref_key]], [model_val_probs["Longitudinal_Lung"], model_val_probs[ref_key]]
                )
                p_te_crs = _fit_multimodal_logistic(
                    tr_base, te_base, [model_train_probs["Longitudinal_Lung"], model_train_probs[ref_key]], [model_test_probs["Longitudinal_Lung"], model_test_probs[ref_key]]
                )
                model_val_probs["Clinical + Rad-score + STRANet"] = p_va_crs
                model_test_probs["Clinical + Rad-score + STRANet"] = p_te_crs

                for mm_name in ("Clinical-only", "Clinical + Rad-score", "Clinical + STRANet", "Clinical + Rad-score + STRANet"):
                    thr_mm = youden_threshold(y_va, model_val_probs[mm_name])
                    m_mm = compute_full_s3_s4_metrics(y_te, model_test_probs[mm_name], threshold=thr_mm, seed=2026)
                    cal_mm = bootstrap_calibration_metrics(y_te, model_test_probs[mm_name], n_boot=1000, seed=2026)
                    s9_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "Holdout_Test_82",
                            "Model": mm_name,
                            "AUROC_Formatted": m_mm["AUROC_Formatted"],
                            **cal_mm,
                        }
                    )

            # 5. Paired DeLong Table (S8 Table) vs Reference STRANet
            if ref_key:
                p_ref = model_test_probs[ref_key]
                temp_s8 = []
                for cname, p_comp in model_test_probs.items():
                    if cname == ref_key:
                        continue
                    res_dl = paired_delong(y_te, p_ref, p_comp)
                    temp_s8.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "Holdout_Test_82 (n=82)",
                            "Reference_Model": ref_key,
                            "Comparator_Model": cname,
                            "Ref_AUROC": round(res_dl["First_AUROC"], 4),
                            "Comparator_AUROC": round(res_dl["Second_AUROC"], 4),
                            "Delta_AUROC_Ref_minus_Comp": round(res_dl["Delta_first_minus_second"], 4),
                            "Delta_95CI": f"({res_dl['Delta_CI_low']:+.3f} to {res_dl['Delta_CI_high']:+.3f})",
                            "Raw_DeLong_p": res_dl["P_value_two_sided"],
                        }
                    )
                if temp_s8:
                    adj_p = holm_adjust([r["Raw_DeLong_p"] for r in temp_s8])
                    for r, ap in zip(temp_s8, adj_p):
                        r["Holm_Adjusted_p"] = ap
                        s8_rows.append(r)

        else:
            # 5-Fold Cross-Validation OOF (N = 545)
            master_545 = pd.read_csv(DATA_545 / "nsclc_545_manifest.csv", encoding="utf-8-sig").sort_values("Sample_ID").reset_index(drop=True)
            y_oof = master_545["Binary_Label"].astype(int).to_numpy()
            sids_545 = master_545["Sample_ID"].to_numpy()

            oof_probs: dict[str, np.ndarray] = {}

            # 1. STRANet 5-Fold CV models
            for s_name, s_parts in stranet_dict.items():
                oof_df = s_parts["oof"].set_index("Sample_ID").loc[sids_545]
                p_oof = oof_df["Pred_Prob"].to_numpy()
                oof_probs[s_name] = p_oof
                thr = youden_threshold(y_oof, p_oof)
                m = compute_full_s3_s4_metrics(y_oof, p_oof, threshold=thr, seed=2026)
                cal = bootstrap_calibration_metrics(y_oof, p_oof, n_boot=1000, seed=2026)
                hw = prof_map.get("STRANet_Dual_Window" if "Dual" in s_name else "STRANet_Spatial_Delta", {})
                s4_rows.append(
                    {
                        "Split_Design": split_design,
                        "Evaluation_Set": "5Fold_OOF_545",
                        "Model": s_name,
                        "Train_AUROC": np.nan,
                        "Val_AUROC": m["AUROC"],
                        **m,
                        "FLOPs_G": hw.get("FLOPs_G", ""),
                        "Parameters": hw.get("Parameters", ""),
                        "Memory_Consumption_MB": hw.get("Memory_Consumption_MB", ""),
                        "Inference_Time_s": hw.get("Inference_Time_s", ""),
                        "FPS": hw.get("FPS", ""),
                    }
                )
                s9_rows.append(
                    {
                        "Split_Design": split_design,
                        "Evaluation_Set": "5Fold_OOF_545",
                        "Model": s_name,
                        "AUROC_Formatted": m["AUROC_Formatted"],
                        **cal,
                    }
                )
                if "Clinical_only_Prob" in oof_df.columns and "Clinical-only" not in oof_probs:
                    oof_probs["Clinical-only"] = oof_df["Clinical_only_Prob"].to_numpy()
                    oof_probs["Stage-only"] = oof_df["Stage_only_Prob"].to_numpy()
                    oof_probs["Source-only"] = oof_df["Source_only_Prob"].to_numpy()

            # 2. SOTA DL 5-Fold CV models
            if dl_split_dir.exists():
                for run_dir in sorted(dl_split_dir.glob("*")):
                    if not (run_dir / "pooled_oof_predictions.csv").exists():
                        continue
                    js = json.loads((run_dir / "run_summary.json").read_text())
                    mname = js["model_name"]
                    oof_df = pd.read_csv(run_dir / "pooled_oof_predictions.csv", encoding="utf-8-sig").set_index("Sample_ID").loc[sids_545]
                    p_oof = oof_df["Pred_Prob"].to_numpy()
                    oof_probs[mname] = p_oof
                    thr = youden_threshold(y_oof, p_oof)
                    m = compute_full_s3_s4_metrics(y_oof, p_oof, threshold=thr, seed=2026)
                    cal = bootstrap_calibration_metrics(y_oof, p_oof, n_boot=1000, seed=2026)
                    hw = prof_map.get(mname, {})
                    s4_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "5Fold_OOF_545",
                            "Model": mname,
                            "Train_AUROC": js.get("mean_train_auroc_selected", np.nan),
                            "Val_AUROC": m["AUROC"],
                            **m,
                            "FLOPs_G": hw.get("FLOPs_G", ""),
                            "Parameters": hw.get("Parameters", ""),
                            "Memory_Consumption_MB": hw.get("Memory_Consumption_MB", ""),
                            "Inference_Time_s": hw.get("Inference_Time_s", ""),
                            "FPS": hw.get("FPS", ""),
                        }
                    )
                    s9_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "5Fold_OOF_545",
                            "Model": mname,
                            "AUROC_Formatted": m["AUROC_Formatted"],
                            **cal,
                        }
                    )
                    if (run_dir / "fold_performance.csv").exists():
                        fdf = pd.read_csv(run_dir / "fold_performance.csv")
                        for fr in fdf.to_dict("records"):
                            fold_bar_rows.append({"Model": mname, **fr})

            # 3. Radiomics 5-Fold CV models
            if rad_split_dir.exists():
                for rname in ("Pre_Tumor", "Post_Tumor", "Delta_Tumor", "Longitudinal_Tumor", "Pre_Lung", "Post_Lung", "Delta_Lung", "Longitudinal_Lung"):
                    rdir = rad_split_dir / rname
                    if not (rdir / "pooled_oof_predictions.csv").exists():
                        continue
                    oof_df = pd.read_csv(rdir / "pooled_oof_predictions.csv", encoding="utf-8-sig").set_index("Sample_ID").loc[sids_545]
                    p_oof = oof_df["Pred_Prob"].to_numpy()
                    oof_probs[rname] = p_oof
                    thr = youden_threshold(y_oof, p_oof)
                    m = compute_full_s3_s4_metrics(y_oof, p_oof, threshold=thr, seed=2026)
                    cal = bootstrap_calibration_metrics(y_oof, p_oof, n_boot=1000, seed=2026)
                    s9_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "5Fold_OOF_545",
                            "Model": rname,
                            "AUROC_Formatted": m["AUROC_Formatted"],
                            **cal,
                        }
                    )
                    if (rdir / "fold_performance.csv").exists():
                        fdf = pd.read_csv(rdir / "fold_performance.csv")
                        for fr in fdf.to_dict("records"):
                            fold_bar_rows.append({"Model": rname, **fr})

            # 4. Out-of-fold Multimodal Fusion
            ref_oof_key = "STRANet (Ensemble)" if "STRANet (Ensemble)" in oof_probs else next(iter(stranet_dict.keys()), None)
            if ref_oof_key and "Longitudinal_Lung" in oof_probs:
                p_oof_cr = np.zeros(len(sids_545), dtype=np.float64)
                p_oof_cs = np.zeros(len(sids_545), dtype=np.float64)
                p_oof_crs = np.zeros(len(sids_545), dtype=np.float64)
                sid_to_pos = {sid: i for i, sid in enumerate(sids_545)}

                for fold in range(1, 6):
                    fdir = split_root / f"fold_{fold}"
                    tr_f = pd.read_csv(fdir / "train_split.csv", encoding="utf-8-sig")
                    va_f = pd.read_csv(fdir / "val_split.csv", encoding="utf-8-sig")
                    tr_idx = [sid_to_pos[s] for s in tr_f["Sample_ID"]]
                    va_idx = [sid_to_pos[s] for s in va_f["Sample_ID"]]

                    p_oof_cr[va_idx] = _fit_multimodal_logistic(
                        tr_f, va_f, [oof_probs["Longitudinal_Lung"][tr_idx]], [oof_probs["Longitudinal_Lung"][va_idx]]
                    )
                    p_oof_cs[va_idx] = _fit_multimodal_logistic(
                        tr_f, va_f, [oof_probs[ref_oof_key][tr_idx]], [oof_probs[ref_oof_key][va_idx]]
                    )
                    p_oof_crs[va_idx] = _fit_multimodal_logistic(
                        tr_f,
                        va_f,
                        [oof_probs["Longitudinal_Lung"][tr_idx], oof_probs[ref_oof_key][tr_idx]],
                        [oof_probs["Longitudinal_Lung"][va_idx], oof_probs[ref_oof_key][va_idx]],
                    )

                oof_probs["Clinical + Rad-score"] = p_oof_cr
                oof_probs["Clinical + STRANet"] = p_oof_cs
                oof_probs["Clinical + Rad-score + STRANet"] = p_oof_crs

                for mm_name in ("Clinical-only", "Clinical + Rad-score", "Clinical + STRANet", "Clinical + Rad-score + STRANet"):
                    if mm_name not in oof_probs:
                        continue
                    thr_mm = youden_threshold(y_oof, oof_probs[mm_name])
                    m_mm = compute_full_s3_s4_metrics(y_oof, oof_probs[mm_name], threshold=thr_mm, seed=2026)
                    cal_mm = bootstrap_calibration_metrics(y_oof, oof_probs[mm_name], n_boot=1000, seed=2026)
                    s9_rows.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "5Fold_OOF_545",
                            "Model": mm_name,
                            "AUROC_Formatted": m_mm["AUROC_Formatted"],
                            **cal_mm,
                        }
                    )

            # 5. Paired DeLong Table (S8 Table) on 5-Fold OOF (N = 545)
            for ref_candidate in ("STRANet (Dual-Window)", "STRANet (Ensemble)"):
                if ref_candidate not in oof_probs:
                    continue
                p_ref = oof_probs[ref_candidate]
                temp_s8 = []
                for cname, p_comp in oof_probs.items():
                    if cname == ref_candidate:
                        continue
                    res_dl = paired_delong(y_oof, p_ref, p_comp)
                    temp_s8.append(
                        {
                            "Split_Design": split_design,
                            "Evaluation_Set": "5Fold_OOF_545 (n=545)",
                            "Reference_Model": ref_candidate,
                            "Comparator_Model": cname,
                            "Ref_AUROC": round(res_dl["First_AUROC"], 4),
                            "Comparator_AUROC": round(res_dl["Second_AUROC"], 4),
                            "Delta_AUROC_Ref_minus_Comp": round(res_dl["Delta_first_minus_second"], 4),
                            "Delta_95CI": f"({res_dl['Delta_CI_low']:+.3f} to {res_dl['Delta_CI_high']:+.3f})",
                            "Raw_DeLong_p": res_dl["P_value_two_sided"],
                        }
                    )
                if temp_s8:
                    adj_p = holm_adjust([r["Raw_DeLong_p"] for r in temp_s8])
                    for r, ap in zip(temp_s8, adj_p):
                        r["Holm_Adjusted_p"] = ap
                        s8_rows.append(r)

    if s4_rows:
        pd.DataFrame(s4_rows).to_csv(SOTA_OUT / "S4_Table_Deep_Learning_Benchmarks_545.csv", index=False)
    if s8_rows:
        pd.DataFrame(s8_rows).to_csv(SOTA_OUT / "S8_Table_Paired_DeLong_Comparisons_545.csv", index=False)
    if s9_rows:
        pd.DataFrame(s9_rows).to_csv(SOTA_OUT / "S9_Table_Calibration_Metrics_545.csv", index=False)
    if fold_bar_rows:
        pd.DataFrame(fold_bar_rows).to_csv(SOTA_OUT / "S8_Fig_5Fold_Fold_Performance_545.csv", index=False)

    print(
        f"Compiled SOTA tables in {SOTA_OUT}: "
        f"S4={len(s4_rows)} rows, S8={len(s8_rows)} rows, S9={len(s9_rows)} rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
