#!/usr/bin/env python3
"""
End-to-end reproducibility verification script using the de-identified Zenodo data package.

Usage:
    python examples/demo_reproduce_from_zenodo.py --zenodo-dir ../Zenodo_NSCLC545_Data_v2
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "evaluation"))
from paired_auc_inference import paired_delong


def main():
    parser = argparse.ArgumentParser(description="Verify manuscript metrics from Zenodo NSCLC-545 data.")
    parser.add_argument(
        "--zenodo-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "Zenodo_NSCLC545_Data_v2"),
        help="Path to the extracted Zenodo_NSCLC545_Data_v2 directory.",
    )
    args = parser.parse_args()
    zdir = os.path.abspath(args.zenodo_dir)

    clin_path = os.path.join(zdir, "01_clinical_and_splits", "clinical_metadata_nsclc545_deidentified.csv")
    oof_path = os.path.join(zdir, "03_model_predictions_and_features", "oof_predictions_all_models_nsclc545.csv")
    hold_path = os.path.join(zdir, "03_model_predictions_and_features", "holdout_predictions_all_models_n82.csv")

    clin = pd.read_csv(clin_path)
    oof = pd.read_csv(oof_path)
    hold = pd.read_csv(hold_path)

    print("=" * 72)
    print("1. Cohort Verification (Table 1 — NSCLC-545)")
    print("=" * 72)
    print(f"Total patients: N = {len(clin)}")
    print(f"Sources: {clin['Source'].value_counts().to_dict()}")
    print(f"Outcomes (1=Progression, 0=Non-progression): {clin['Outcome'].value_counts().to_dict()}")
    print(f"TNM Stage: {clin['TNM_Stage'].value_counts().sort_index().to_dict()}")
    print(f"Median Age [IQR]: {clin['Age_Years'].median():.1f} [{clin['Age_Years'].quantile(0.25):.1f}, {clin['Age_Years'].quantile(0.75):.1f}]")
    print(f"Median BMI [IQR]: {clin['BMI_kg_m2'].median():.1f} [{clin['BMI_kg_m2'].quantile(0.25):.1f}, {clin['BMI_kg_m2'].quantile(0.75):.1f}]")

    print("\n" + "=" * 72)
    print("2. Five-Fold Cross-Validation Out-of-Fold (OOF) Performance (N = 545)")
    print("=" * 72)
    y_oof = oof["Outcome"].values.astype(int)
    model_cols = [c for c in oof.columns if c.startswith("prob_")]
    for col in model_cols:
        name = col.replace("prob_", "")
        p = oof[col].values.astype(float)
        auroc = roc_auc_score(y_oof, p)
        auprc = average_precision_score(y_oof, p)
        brier = brier_score_loss(y_oof, p)
        print(f"  {name:32s} | AUROC = {auroc:.3f} | AUPRC = {auprc:.3f} | Brier = {brier:.3f}")

    print("\n" + "=" * 72)
    print("3. Paired DeLong Tests vs. STRANet (Five-Fold OOF, N = 545)")
    print("=" * 72)
    if "prob_STRANet" in oof.columns:
        ref = oof["prob_STRANet"].values.astype(float)
        for comp_col in ["prob_Clinical_only", "prob_Longitudinal_Lung", "prob_Longitudinal_Tumor", "prob_Source_only", "prob_Stage_only"]:
            if comp_col in oof.columns:
                res = paired_delong(y_oof, ref, oof[comp_col].values.astype(float))
                print(
                    f"  STRANet vs {comp_col.replace('prob_', ''):20s} | "
                    f"Delta AUROC = {res['Delta_first_minus_second']:+.3f} "
                    f"(95% CI: {res['Delta_CI_low']:.3f} to {res['Delta_CI_high']:.3f}) | "
                    f"P = {res['P_value_two_sided']:.3e}"
                )

    print("\n" + "=" * 72)
    print("4. Held-Out Test Set Performance (n = 82)")
    print("=" * 72)
    y_h = hold["Outcome"].values.astype(int)
    for col in [c for c in hold.columns if c.startswith("prob_")]:
        name = col.replace("prob_", "")
        p = hold[col].values.astype(float)
        print(f"  {name:32s} | AUROC = {roc_auc_score(y_h, p):.3f} | AUPRC = {average_precision_score(y_h, p):.3f}")

    print("\nAll Zenodo reproducibility checks completed successfully!")


if __name__ == "__main__":
    main()
