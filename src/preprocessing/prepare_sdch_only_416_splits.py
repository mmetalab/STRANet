#!/usr/bin/env python3
"""Prepare SDCH-Only NSCLC (N=416) splits and locked external TCIA (N=129) & SDCH-SCLC (N=49) manifests.

Writes to:
  ./data/Reviewer_R1_auc_rawpost_fix/data/sdch_only_416_to_tcia_sclc_20261106/

Creates:
1. sdch_416_holdout_290_63_63:
   - train.csv (N=290), validation.csv (N=63), test.csv (N=63)
   - Jointly stratified by Cohort (SDCH_old vs SDCH_new) x Stage (I, II, III) x Binary_Label (0, 1)
   - Both validation and internal test have exact matched quotas (46 Neg, 17 Pos).
2. sdch_416_5fold_cv:
   - fold_1..fold_5/train_split.csv (~333) and val_split.csv (~83)
   - Jointly stratified by Cohort x Stage x Binary_Label.
3. Locked external / cross-histology evaluation sets (saved in root of data dir):
   - external_tcia_nsclc_129.csv (N=129: 123 Pos, 6 Neg)
   - external_sdch_sclc_49.csv (N=49: 23 Pos, 26 Neg)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_545 = ROOT / "data/nsclc_545_excl_sclc_20261106"
OUT = ROOT / "data/sdch_only_416_to_tcia_sclc_20261106"
OUT.mkdir(parents=True, exist_ok=True)


def build_sdch_only_splits():
    nsclc_545 = pd.read_csv(DATA_545 / "nsclc_545_manifest.csv", encoding="utf-8-sig")
    sclc_49 = pd.read_csv(DATA_545 / "excluded_49_sclc_audit.csv", encoding="utf-8-sig").drop(
        columns=["id_norm", "is_SCLC"], errors="ignore"
    )

    sdch_416 = nsclc_545[nsclc_545["Source"] == "SDCH"].reset_index(drop=True).copy()
    tcia_129 = nsclc_545[nsclc_545["Source"] == "TCIA"].reset_index(drop=True).copy()

    if len(sdch_416) != 416 or len(tcia_129) != 129 or len(sclc_49) != 49:
        raise RuntimeError(f"Unexpected counts: SDCH={len(sdch_416)}, TCIA={len(tcia_129)}, SCLC={len(sclc_49)}")

    # Save locked external manifests
    sdch_416.to_csv(OUT / "sdch_nsclc_416_manifest.csv", index=False, encoding="utf-8-sig")
    tcia_129.to_csv(OUT / "external_tcia_nsclc_129.csv", index=False, encoding="utf-8-sig")
    sclc_49.to_csv(OUT / "external_sdch_sclc_49.csv", index=False, encoding="utf-8-sig")

    # Build joint Cohort x Stage x Outcome key for the 416 SDCH NSCLC patients
    # Merge singleton cells (<5 cases):
    # - SDCH_new S1 Y0 (n=1) -> merge with SDCH_old S1 Y0 -> SDCH_S1_Y0 (n=187)
    # - SDCH_old S3 Y0 (n=1) -> merge with SDCH_new S3 Y0 -> SDCH_S3_Y0 (n=53)
    # - SDCH_old missing stage Y1 (n=1) -> merge with SDCH_old S1 Y1 -> SDCH_old_S1_Y1 (n=44)
    keys = []
    for r in sdch_416.itertuples():
        st = 1 if pd.isna(r.stage_123) else int(r.stage_123)
        y = int(r.Binary_Label)
        c = str(r.Cohort)
        if st == 1 and y == 0:
            keys.append("SDCH_S1_Y0")
        elif st == 3 and y == 0:
            keys.append("SDCH_S3_Y0")
        else:
            keys.append(f"{c}_S{st}_Y{y}")
    sdch_416["Joint_SDCH_Key"] = keys
    counts = sdch_416["Joint_SDCH_Key"].value_counts().sort_index()
    print("SDCH-416 Joint Cohort x Stage x Outcome stratum counts:\n", counts.to_string())
    if int(counts.min()) < 5:
        raise RuntimeError(f"Stratum < 5 in Joint_SDCH_Key: {counts.to_dict()}")

    # 1. Fixed Holdout Split: 290 Train / 63 Validation / 63 Internal Test
    target_total = 63
    exact = counts.to_numpy(dtype=float) * target_total / len(sdch_416)
    allocations = np.floor(exact).astype(int)
    rem = target_total - int(allocations.sum())
    order = np.argsort(-(exact - allocations), kind="stable")
    for idx in order[:rem]:
        allocations[idx] += 1
    quota_map = dict(zip(counts.index, allocations))
    print("Holdout quotas per 63-patient partition (Val & Test):", quota_map)

    rng = np.random.default_rng(20261140)
    test_ix = []
    for k_name, n_q in quota_map.items():
        cands = sdch_416.index[sdch_416["Joint_SDCH_Key"] == k_name].to_numpy()
        test_ix.extend(rng.choice(cands, size=int(n_q), replace=False).tolist())
    test_df = sdch_416.loc[test_ix].copy()
    rem_df = sdch_416.drop(index=test_df.index)

    val_ix = []
    for k_name, n_q in quota_map.items():
        cands = rem_df.index[rem_df["Joint_SDCH_Key"] == k_name].to_numpy()
        val_ix.extend(rng.choice(cands, size=int(n_q), replace=False).tolist())
    val_df = rem_df.loc[val_ix].copy()
    train_df = rem_df.drop(index=val_df.index).copy()

    ho_dir = OUT / "sdch_416_holdout_290_63_63"
    ho_dir.mkdir(parents=True, exist_ok=True)
    for role, part in (("train", train_df), ("validation", val_df), ("test", test_df)):
        p_clean = part.drop(columns=["Joint_SDCH_Key"], errors="ignore")
        p_clean.to_csv(ho_dir / f"{role}.csv", index=False, encoding="utf-8-sig")
    tcia_129.to_csv(ho_dir / "external_tcia_nsclc_129.csv", index=False, encoding="utf-8-sig")
    sclc_49.to_csv(ho_dir / "external_sdch_sclc_49.csv", index=False, encoding="utf-8-sig")

    ids = [set(p["Patient_ID"]) for p in (train_df, val_df, test_df, tcia_129, sclc_49)]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] & ids[j]:
                raise RuntimeError(f"Patient overlap between partition {i} and {j}")

    design_ho = {
        "scenario": "sdch_416_holdout_290_63_63",
        "seed": 20261140,
        "train_n": len(train_df),
        "train_pos": int(train_df["Binary_Label"].sum()),
        "validation_n": len(val_df),
        "validation_pos": int(val_df["Binary_Label"].sum()),
        "test_n": len(test_df),
        "test_pos": int(test_df["Binary_Label"].sum()),
        "external_tcia_nsclc_n": len(tcia_129),
        "external_tcia_nsclc_pos": int(tcia_129["Binary_Label"].sum()),
        "external_sdch_sclc_n": len(sclc_49),
        "external_sdch_sclc_pos": int(sclc_49["Binary_Label"].sum()),
        "quotas_val_and_test": {k: int(v) for k, v in quota_map.items()},
        "sample_id_sha256": {
            "train": hashlib.sha256("\n".join(sorted(train_df["Sample_ID"])).encode()).hexdigest(),
            "validation": hashlib.sha256("\n".join(sorted(val_df["Sample_ID"])).encode()).hexdigest(),
            "test": hashlib.sha256("\n".join(sorted(test_df["Sample_ID"])).encode()).hexdigest(),
            "external_tcia": hashlib.sha256("\n".join(sorted(tcia_129["Sample_ID"])).encode()).hexdigest(),
            "external_sclc": hashlib.sha256("\n".join(sorted(sclc_49["Sample_ID"])).encode()).hexdigest(),
        },
    }
    (ho_dir / "design.json").write_text(json.dumps(design_ho, indent=2))
    print("Saved holdout design:", json.dumps(design_ho, indent=2))

    # 2. 5-Fold Cross-Validation Split on SDCH-416
    cv_dir = OUT / "sdch_416_5fold_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    cv_df = sdch_416.drop(columns=["Joint_SDCH_Key"]).copy()
    cv_df["Fold"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261141)
    for fold, (_, val_pos) in enumerate(skf.split(sdch_416, sdch_416["Joint_SDCH_Key"]), start=1):
        cv_df.loc[sdch_416.index[val_pos], "Fold"] = fold
    for fold in range(1, 6):
        fdir = cv_dir / f"fold_{fold}"
        fdir.mkdir(parents=True, exist_ok=True)
        cv_df[cv_df["Fold"] != fold].to_csv(fdir / "train_split.csv", index=False, encoding="utf-8-sig")
        cv_df[cv_df["Fold"] == fold].to_csv(fdir / "val_split.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(cv_dir / "manifest_with_folds.csv", index=False, encoding="utf-8-sig")
    tcia_129.to_csv(cv_dir / "external_tcia_nsclc_129.csv", index=False, encoding="utf-8-sig")
    sclc_49.to_csv(cv_dir / "external_sdch_sclc_49.csv", index=False, encoding="utf-8-sig")
    print(f"Saved 5-fold CV sdch_416_5fold_cv (n={len(cv_df)})")


if __name__ == "__main__":
    build_sdch_only_splits()
