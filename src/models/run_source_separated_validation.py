#!/usr/bin/env python3
"""Train 3D Siamese STRANet exclusively on SDCH NSCLC patients (N=416),
report internal SDCH Train, Validation, and Test results, and apply the
validation-selected best model (and ensembles) to two locked external cohorts:
  1. TCIA NSCLC (N=129: 123 Pos, 6 Neg)
  2. SDCH SCLC (N=49: 23 Pos, 26 Neg)

Writes all outputs to:
  ./data/Reviewer_R1_auc_rawpost_fix/results/sdch_only_416_to_tcia_sclc_20261106/
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_DIR = ROOT / "data/sdch_only_416_to_tcia_sclc_20261106"
OUT_DIR = ROOT / "results/sdch_only_416_to_tcia_sclc_20261106"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from paired_auc_inference import paired_delong  # noqa: E402
from run_nsclc_545_benchmarks import fit_and_eval_one  # noqa: E402
from run_stage_source_594_benchmarks import bootstrap_auc_ci, fit_clinical_comparators  # noqa: E402


def select_youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y.astype(int), p.astype(float))
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thr = float(thresholds[best_idx])
    if not np.isfinite(thr):
        thr = 0.5
    return float(np.clip(thr, 0.05, 0.95))


def eval_partition_metrics(df_part: pd.DataFrame, thr: float, tag: str, seed: int) -> dict:
    y = df_part["Binary_Label"].to_numpy(dtype=int)
    p = df_part["Pred_Prob"].to_numpy(dtype=float)
    pt, lo, hi = bootstrap_auc_ci(y, p, seed=seed)
    yhat = (p >= thr).astype(int)

    res = {
        "tag": tag,
        "n": len(df_part),
        "pos": int(y.sum()),
        "neg": int((1 - y).sum()),
        "auroc": pt,
        "auroc_ci": [lo, hi],
        "threshold_frozen_from_sdch_val": thr,
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "sensitivity": float(recall_score(y, yhat, zero_division=0)),
        "specificity": float(recall_score(1 - y, 1 - yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }
    if "Cohort" in df_part.columns:
        for c_name in ("SDCH_old", "SDCH_new"):
            m = df_part["Cohort"].eq(c_name).to_numpy()
            if np.unique(y[m]).size == 2:
                res[f"auroc_{c_name.lower()}"] = float(roc_auc_score(y[m], p[m]))
            else:
                res[f"auroc_{c_name.lower()}"] = None
    if "stage_123" in df_part.columns:
        for st in (1, 2, 3):
            m = df_part["stage_123"].fillna(-1).eq(st).to_numpy()
            if np.unique(y[m]).size == 2:
                res[f"auroc_stage_{st}"] = float(roc_auc_score(y[m], p[m]))
            else:
                res[f"auroc_stage_{st}"] = None

    for comp in ("Stage_only_Prob", "Clinical_only_Prob"):
        if comp in df_part.columns:
            cp = df_part[comp].to_numpy(dtype=float)
            c_key = comp.replace("_Prob", "").lower()
            res[f"auroc_{c_key}"] = float(roc_auc_score(y, cp))
            res[f"delong_vs_{c_key}"] = paired_delong(y, p, cp)

    if "Clinical_only_Prob" in df_part.columns:
        cp = df_part["Clinical_only_Prob"].to_numpy(dtype=float)
        comb = 0.5 * (p + cp)
        res["auroc_clinical_plus_imaging"] = float(roc_auc_score(y, comb))
        res["delong_clinical_plus_imaging_vs_clinical"] = paired_delong(y, comb, cp)
    return res


def update_master_summary():
    """Compile all completed SDCH-only -> TCIA & SCLC runs and evaluate best-model & top-3 ensembles."""
    files = sorted(glob.glob(str(OUT_DIR / "*/*/run_summary.json")))
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text())
        mode = d["mode"]
        val_b = d["sdch_val_or_oof_summary"]
        test_b = d.get("sdch_internal_test_summary", val_b)
        tcia_b = d["external_tcia_129_summary"]
        sclc_b = d["external_sdch_sclc_49_summary"]

        rows.append(
            {
                "Mode": mode,
                "Arch": d["arch"],
                "Schedule": d["schedule"],
                "Orientation": d["orientation"],
                "Sampler_Mode": d["sampler_mode"],
                "Seed": d["seed"],
                "SDCH_Train_AUROC": round(float(d["sdch_train_auroc"]), 4),
                "SDCH_Val_AUROC": round(float(val_b["auroc"]), 4),
                "SDCH_Test_AUROC": round(float(test_b["auroc"]), 4),
                "SDCH_Test_CI_Low": round(float(test_b["auroc_ci"][0]), 4),
                "SDCH_Test_CI_High": round(float(test_b["auroc_ci"][1]), 4),
                "SDCH_Test_Stage2_AUROC": round(float(test_b["auroc_stage_2"]), 4) if test_b.get("auroc_stage_2") is not None else None,
                "External_TCIA_129_AUROC": round(float(tcia_b["auroc"]), 4),
                "External_TCIA_CI_Low": round(float(tcia_b["auroc_ci"][0]), 4),
                "External_TCIA_CI_High": round(float(tcia_b["auroc_ci"][1]), 4),
                "External_SDCH_SCLC_49_AUROC": round(float(sclc_b["auroc"]), 4),
                "External_SCLC_CI_Low": round(float(sclc_b["auroc_ci"][0]), 4),
                "External_SCLC_CI_High": round(float(sclc_b["auroc_ci"][1]), 4),
                "External_SCLC_Stage3_AUROC": round(float(sclc_b["auroc_stage_3"]), 4) if sclc_b.get("auroc_stage_3") is not None else None,
                "SDCH_Clinical_Only_AUROC": round(float(test_b.get("auroc_clinical_only", 0)), 4),
                "SDCH_Clinical_Plus_Img_AUROC": round(float(test_b.get("auroc_clinical_plus_imaging", 0)), 4),
                "SCLC_Clinical_Only_AUROC": round(float(sclc_b.get("auroc_clinical_only", 0)), 4),
                "SCLC_Clinical_Plus_Img_AUROC": round(float(sclc_b.get("auroc_clinical_plus_imaging", 0)), 4),
                "Run_Dir": str(Path(f).parent),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Also build validation-selected Best Single Model and Top-3 Ensemble for sdch_416_holdout_290_63_63
    ho_rows = df[df["Mode"] == "sdch_416_holdout_290_63_63"].sort_values("SDCH_Val_AUROC", ascending=False)
    if len(ho_rows) >= 2:
        top_k = min(3, len(ho_rows))
        top_dirs = [Path(p) for p in ho_rows.head(top_k)["Run_Dir"].tolist()]
        val_dfs = [pd.read_csv(d / "sdch_validation_predictions.csv") for d in top_dirs]
        test_dfs = [pd.read_csv(d / "sdch_test_predictions.csv") for d in top_dirs]
        tcia_dfs = [pd.read_csv(d / "external_tcia_129_predictions.csv") for d in top_dirs]
        sclc_dfs = [pd.read_csv(d / "external_sdch_sclc_49_predictions.csv") for d in top_dirs]
        train_dfs = [pd.read_csv(d / "sdch_train_predictions.csv") for d in top_dirs]

        def avg_df(frames):
            b = frames[0].copy()
            b["Pred_Prob"] = np.mean([x["Pred_Prob"].to_numpy(dtype=float) for x in frames], axis=0)
            return b

        tr_ens = avg_df(train_dfs)
        va_ens = avg_df(val_dfs)
        te_ens = avg_df(test_dfs)
        tc_ens = avg_df(tcia_dfs)
        sc_ens = avg_df(sclc_dfs)
        thr_ens = select_youden_threshold(va_ens["Binary_Label"].to_numpy(), va_ens["Pred_Prob"].to_numpy())

        va_m = eval_partition_metrics(va_ens, thr_ens, "val_top3_ens", 20261140)
        te_m = eval_partition_metrics(te_ens, thr_ens, "test_top3_ens", 20261140)
        tc_m = eval_partition_metrics(tc_ens, thr_ens, "tcia_top3_ens", 20261140)
        sc_m = eval_partition_metrics(sc_ens, thr_ens, "sclc_top3_ens", 20261140)

        ens_row = {
            "Mode": "sdch_416_holdout_290_63_63",
            "Arch": f"Val_Selected_Top{top_k}_Siamese_Ensemble",
            "Schedule": "ensemble",
            "Orientation": "mixed",
            "Sampler_Mode": "ensemble",
            "Seed": 0,
            "SDCH_Train_AUROC": round(float(roc_auc_score(tr_ens["Binary_Label"], tr_ens["Pred_Prob"])), 4),
            "SDCH_Val_AUROC": round(float(va_m["auroc"]), 4),
            "SDCH_Test_AUROC": round(float(te_m["auroc"]), 4),
            "SDCH_Test_CI_Low": round(float(te_m["auroc_ci"][0]), 4),
            "SDCH_Test_CI_High": round(float(te_m["auroc_ci"][1]), 4),
            "SDCH_Test_Stage2_AUROC": round(float(te_m["auroc_stage_2"]), 4) if te_m.get("auroc_stage_2") is not None else None,
            "External_TCIA_129_AUROC": round(float(tc_m["auroc"]), 4),
            "External_TCIA_CI_Low": round(float(tc_m["auroc_ci"][0]), 4),
            "External_TCIA_CI_High": round(float(tc_m["auroc_ci"][1]), 4),
            "External_SDCH_SCLC_49_AUROC": round(float(sc_m["auroc"]), 4),
            "External_SCLC_CI_Low": round(float(sc_m["auroc_ci"][0]), 4),
            "External_SCLC_CI_High": round(float(sc_m["auroc_ci"][1]), 4),
            "External_SCLC_Stage3_AUROC": round(float(sc_m["auroc_stage_3"]), 4) if sc_m.get("auroc_stage_3") is not None else None,
            "SDCH_Clinical_Only_AUROC": round(float(te_m.get("auroc_clinical_only", 0)), 4),
            "SDCH_Clinical_Plus_Img_AUROC": round(float(te_m.get("auroc_clinical_plus_imaging", 0)), 4),
            "SCLC_Clinical_Only_AUROC": round(float(sc_m.get("auroc_clinical_only", 0)), 4),
            "SCLC_Clinical_Plus_Img_AUROC": round(float(sc_m.get("auroc_clinical_plus_imaging", 0)), 4),
            "Run_Dir": "ensemble",
        }
        df = pd.concat([df, pd.DataFrame([ens_row])], ignore_index=True)

    df = df.drop(columns=["Run_Dir"]).sort_values(["Mode", "SDCH_Val_AUROC"], ascending=[True, False])
    df.to_csv(OUT_DIR / "MASTER_SDCH_ONLY_416_TO_TCIA_SCLC_SUMMARY.csv", index=False)
    print("=== SDCH-Only NSCLC (N=416) -> Internal Test (N=63), External TCIA (N=129) & External SDCH-SCLC (N=49) ===")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("sdch_416_holdout_290_63_63", "sdch_416_5fold_cv"),
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=(
            "groupnorm_delta_gap",
            "spatial_delta_maxavg",
            "spatial_attention_res3d",
            "multiscale_pyramid_siamese",
            "se_cbam_res3d_siamese",
            "dual_window_spatial_siamese",
            "stage_conditioned_spatial",
        ),
    )
    parser.add_argument(
        "--schedule",
        required=True,
        choices=("cosine_lr8e5_aug2x", "hybrid_pseudo_aug2x", "reg_cutout_ls006"),
    )
    parser.add_argument("--orientation", default="raw", choices=("raw", "lps"))
    parser.add_argument(
        "--sampler-mode",
        default="label_only",
        choices=("label_only", "stage_balanced", "source_stage_balanced"),
    )
    parser.add_argument("--seed", type=int, default=30101)
    args = parser.parse_args()

    tag = f"{args.arch}_{args.schedule}_{args.orientation}_{args.sampler_mode}_seed{args.seed}"
    run_dir = OUT_DIR / args.mode / tag
    if (run_dir / "run_summary.json").exists():
        print(f"Skipping completed run: {run_dir}")
        update_master_summary()
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    tcia_df = pd.read_csv(DATA_DIR / "external_tcia_nsclc_129.csv", encoding="utf-8-sig")
    sclc_df = pd.read_csv(DATA_DIR / "external_sdch_sclc_49.csv", encoding="utf-8-sig")

    if args.mode == "sdch_416_holdout_290_63_63":
        sdir = DATA_DIR / "sdch_416_holdout_290_63_63"
        train_df = pd.read_csv(sdir / "train.csv", encoding="utf-8-sig")
        val_df = pd.read_csv(sdir / "validation.csv", encoding="utf-8-sig")
        test_df = pd.read_csv(sdir / "test.csv", encoding="utf-8-sig")

        preds, peak_train_auc = fit_and_eval_one(
            train_df,
            val_df,
            {
                "sdch_train": train_df,
                "sdch_validation": val_df,
                "sdch_test": test_df,
                "external_tcia_129": tcia_df,
                "external_sdch_sclc_49": sclc_df,
            },
            out_dir=run_dir,
            arch=args.arch,
            schedule=args.schedule,
            orientation=args.orientation,
            sampler_mode=args.sampler_mode,
            monitor="val_patient_auc",
            seed=args.seed,
        )

        # Threshold frozen strictly from SDCH Validation
        thr = select_youden_threshold(val_df["Binary_Label"].to_numpy(), preds["sdch_validation"])

        saved_frames = {}
        for name, df_part in (
            ("sdch_train", train_df),
            ("sdch_validation", val_df),
            ("sdch_test", test_df),
            ("external_tcia_129", tcia_df),
            ("external_sdch_sclc_49", sclc_df),
        ):
            out_df = df_part.copy()
            out_df["Pred_Prob"] = preds[name]
            if name != "sdch_train":
                for k, v in fit_clinical_comparators(train_df, df_part).items():
                    out_df[k] = v
            out_df.to_csv(run_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
            saved_frames[name] = out_df

        summary = {
            "mode": args.mode,
            "arch": args.arch,
            "schedule": args.schedule,
            "orientation": args.orientation,
            "sampler_mode": args.sampler_mode,
            "seed": args.seed,
            "sdch_train_auroc": float(roc_auc_score(train_df["Binary_Label"], preds["sdch_train"])),
            "peak_sdch_train_auroc": peak_train_auc,
            "frozen_youden_threshold": thr,
            "sdch_val_or_oof_summary": eval_partition_metrics(saved_frames["sdch_validation"], thr, f"{tag}_sdch_val", args.seed),
            "sdch_internal_test_summary": eval_partition_metrics(saved_frames["sdch_test"], thr, f"{tag}_sdch_test", args.seed),
            "external_tcia_129_summary": eval_partition_metrics(saved_frames["external_tcia_129"], thr, f"{tag}_tcia129", args.seed),
            "external_sdch_sclc_49_summary": eval_partition_metrics(saved_frames["external_sdch_sclc_49"], thr, f"{tag}_sclc49", args.seed),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)

    else:
        sdir = DATA_DIR / "sdch_416_5fold_cv"
        oof_parts = []
        tcia_fold_preds = []
        sclc_fold_preds = []
        fold_rows = []
        for fold in range(1, 6):
            fseed = args.seed + fold
            fdir = sdir / f"fold_{fold}"
            tr = pd.read_csv(fdir / "train_split.csv", encoding="utf-8-sig")
            va = pd.read_csv(fdir / "val_split.csv", encoding="utf-8-sig")
            preds, peak_tr = fit_and_eval_one(
                tr,
                va,
                {
                    "sdch_train": tr,
                    "sdch_validation": va,
                    "external_tcia_129": tcia_df,
                    "external_sdch_sclc_49": sclc_df,
                },
                out_dir=run_dir / f"fold_{fold}",
                arch=args.arch,
                schedule=args.schedule,
                orientation=args.orientation,
                sampler_mode=args.sampler_mode,
                monitor="val_patient_auc",
                seed=fseed,
            )
            va_out = va.copy()
            va_out["Fold"] = fold
            va_out["Seed"] = fseed
            va_out["Pred_Prob"] = preds["sdch_validation"]
            for k, v in fit_clinical_comparators(tr, va).items():
                va_out[k] = v
            oof_parts.append(va_out)
            tcia_fold_preds.append(preds["external_tcia_129"])
            sclc_fold_preds.append(preds["external_sdch_sclc_49"])

            fold_rows.append(
                {
                    "Fold": fold,
                    "Seed": fseed,
                    "SDCH_Train_AUROC": float(roc_auc_score(tr["Binary_Label"], preds["sdch_train"])),
                    "Peak_Train_AUROC": peak_tr,
                    "SDCH_Val_AUROC": float(roc_auc_score(va["Binary_Label"], preds["sdch_validation"])),
                    "TCIA_Fold_AUROC": float(roc_auc_score(tcia_df["Binary_Label"], preds["external_tcia_129"])),
                    "SCLC_Fold_AUROC": float(roc_auc_score(sclc_df["Binary_Label"], preds["external_sdch_sclc_49"])),
                }
            )

        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_df.to_csv(run_dir / "sdch_416_oof_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_performance.csv", index=False)

        thr = select_youden_threshold(oof_df["Binary_Label"].to_numpy(), oof_df["Pred_Prob"].to_numpy())

        tcia_ens = tcia_df.copy()
        tcia_ens["Pred_Prob"] = np.mean(np.vstack(tcia_fold_preds), axis=0)
        for k, v in fit_clinical_comparators(oof_df, tcia_ens).items():
            tcia_ens[k] = v
        tcia_ens.to_csv(run_dir / "external_tcia_129_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

        sclc_ens = sclc_df.copy()
        sclc_ens["Pred_Prob"] = np.mean(np.vstack(sclc_fold_preds), axis=0)
        for k, v in fit_clinical_comparators(oof_df, sclc_ens).items():
            sclc_ens[k] = v
        sclc_ens.to_csv(run_dir / "external_sdch_sclc_49_ensemble_predictions.csv", index=False, encoding="utf-8-sig")

        summary = {
            "mode": args.mode,
            "arch": args.arch,
            "schedule": args.schedule,
            "orientation": args.orientation,
            "sampler_mode": args.sampler_mode,
            "seed": args.seed,
            "sdch_train_auroc": float(np.mean([r["SDCH_Train_AUROC"] for r in fold_rows])),
            "peak_sdch_train_auroc": float(np.max([r["Peak_Train_AUROC"] for r in fold_rows])),
            "frozen_youden_threshold": thr,
            "fold_rows": fold_rows,
            "sdch_val_or_oof_summary": eval_partition_metrics(oof_df, thr, f"{tag}_sdch416_oof", args.seed),
            "sdch_internal_test_summary": eval_partition_metrics(oof_df, thr, f"{tag}_sdch416_oof", args.seed),
            "external_tcia_129_summary": eval_partition_metrics(tcia_ens, thr, f"{tag}_tcia129_ens", args.seed),
            "external_sdch_sclc_49_summary": eval_partition_metrics(sclc_ens, thr, f"{tag}_sclc49_ens", args.seed),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)

    update_master_summary()


if __name__ == "__main__":
    main()
