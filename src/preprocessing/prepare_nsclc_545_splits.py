#!/usr/bin/env python3
"""Create the clean 545-patient NSCLC cohort (excluding 49 SCLC cases in SDCH-New.xlsx)
and generate five matched split designs in a separate directory:
  ./data/Reviewer_R1_auc_rawpost_fix/data/nsclc_545_excl_sclc_20261106/

Split designs created:
1. source_stage_5fold_cv_545 (N=545, 5-fold CV jointly stratified by Cohort/Source x Stage x Outcome)
2. pure_stage_5fold_cv_545 (N=545, 5-fold CV stratified strictly by Stage x Outcome)
3. source_stage_holdout_545 (381 Train / 82 Val / 82 Test, jointly balanced by Cohort/Source x Stage x Outcome)
4. representative_holdout_545 (381 Train / 82 Val / 82 Test, balanced by Source x Outcome)
5. sdch_416_to_tcia_129_cv (416 Pooled SDCH 5-Fold CV -> 129 Locked TCIA External Evaluation)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
MANIFEST_594 = ROOT / "data/pooled_all_sources_exploratory/pooled_manifest.csv"
MASTER = ROOT / "data/master_with_stage.csv"
SDCH_NEW_XLSX = Path("./data/Patient_info/SDCH-New.xlsx")
SDCH_OLD_XLSX = Path("./data/Patient_info/SDCH_clinic.xlsx")
OUT = ROOT / "data/nsclc_545_excl_sclc_20261106"
OUT.mkdir(parents=True, exist_ok=True)


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().split(".")[0].lstrip("0")
    return s or "0"


def load_and_filter_nsclc_545() -> pd.DataFrame:
    m594 = pd.read_csv(MANIFEST_594, encoding="utf-8-sig").drop(columns=["Fold"], errors="ignore")
    master = pd.read_csv(MASTER, encoding="utf-8-sig")[["Sample_ID", "Patient_name", "Pre_pseudo_path", "Post_pseudo_path"]].drop_duplicates("Sample_ID")
    df = m594.merge(master, on="Sample_ID", how="left", validate="one_to_one")
    df["id_norm"] = df["Patient_ID"].map(norm_id)

    new_xl = pd.read_excel(SDCH_NEW_XLSX)
    new_xl["id_norm"] = new_xl["门诊号"].map(norm_id)
    new_xl["name_clean"] = new_xl["姓名"].astype(str).str.strip()
    old_xl = pd.read_excel(SDCH_OLD_XLSX)
    old_xl["id_norm"] = old_xl["Patient_ID"].map(norm_id)

    new_path_by_id = dict(zip(new_xl["id_norm"], new_xl["病理类型"]))
    new_path_by_name = dict(zip(new_xl["name_clean"], new_xl["病理类型"]))
    old_path_by_id = dict(zip(old_xl["id_norm"], old_xl["病理类型"]))

    pathologies = []
    for r in df.itertuples():
        if r.Cohort == "SDCH_new":
            p = new_path_by_id.get(r.id_norm)
            if p is None and pd.notna(r.Patient_name):
                p = new_path_by_name.get(str(r.Patient_name).strip())
            pathologies.append(p)
        elif r.Cohort == "SDCH_old":
            pathologies.append(old_path_by_id.get(r.id_norm))
        else:
            pathologies.append("NSCLC (TCIA)")

    df["Pathology"] = pathologies
    df["is_SCLC"] = df["Pathology"].astype(str).str.contains("小细胞", na=False)

    sclc_df = df[df["is_SCLC"]].copy()
    nsclc_df = df[~df["is_SCLC"]].drop(columns=["id_norm", "is_SCLC"]).reset_index(drop=True)

    sclc_df.to_csv(OUT / "excluded_49_sclc_audit.csv", index=False, encoding="utf-8-sig")
    nsclc_df.to_csv(OUT / "nsclc_545_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"Total input: {len(df)} | Excluded SCLC: {len(sclc_df)} | Remaining clean NSCLC: {len(nsclc_df)}")
    if len(sclc_df) != 49 or len(nsclc_df) != 545:
        raise RuntimeError(f"Expected 49 SCLC and 545 NSCLC, got {len(sclc_df)} and {len(nsclc_df)}")
    return nsclc_df


def proportional_draw(data: pd.DataFrame, strata_col: str, target_total: int, rng: np.random.Generator) -> pd.DataFrame:
    counts = data[strata_col].value_counts().sort_index()
    exact = counts.to_numpy(dtype=float) * target_total / len(data)
    allocations = np.floor(exact).astype(int)
    remainder = target_total - int(allocations.sum())
    order = np.argsort(-(exact - allocations), kind="stable")
    for idx in order[:remainder]:
        allocations[idx] += 1

    chosen = []
    for stratum_val, k in zip(counts.index, allocations):
        if k <= 0:
            continue
        candidates = data.index[data[strata_col] == stratum_val].to_numpy()
        if len(candidates) < k:
            raise RuntimeError(f"Stratum {stratum_val} has {len(candidates)} < {k}")
        chosen.extend(rng.choice(candidates, size=int(k), replace=False).tolist())
    return data.loc[chosen].copy()


def save_holdout(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: Path, name: str, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [set(p["Patient_ID"]) for p in (train, val, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError(f"Patient overlap in {name}")
    if len(train) + len(val) + len(test) != 545:
        raise RuntimeError(f"Total != 545 in {name}")

    parts = {"train": train, "validation": val, "test": test}
    audit_rows = []
    for role, part in parts.items():
        part.to_csv(out_dir / f"{role}.csv", index=False, encoding="utf-8-sig")
        for (source, cohort, stage, label), n in (
            part.groupby(["Source", "Cohort", "stage_123", "Binary_Label"], dropna=False).size().items()
        ):
            audit_rows.append(
                {
                    "Scenario": name,
                    "Role": role,
                    "Source": source,
                    "Cohort": cohort,
                    "Stage": None if pd.isna(stage) else int(stage),
                    "Binary_Label": int(label),
                    "N": int(n),
                }
            )
    pd.DataFrame(audit_rows).to_csv(out_dir / "split_audit.csv", index=False)

    design = {
        "scenario": name,
        "cohort_n": 545,
        "excluded_sclc_n": 49,
        "seed": seed,
        "train_n": len(train),
        "validation_n": len(val),
        "test_n": len(test),
        "val_pos": int(val["Binary_Label"].sum()),
        "test_pos": int(test["Binary_Label"].sum()),
        "test_source_only_auc": float(roc_auc_score(test["Binary_Label"], test["Source"].eq("TCIA").astype(int))),
        "test_stage_only_auc": float(roc_auc_score(test["Binary_Label"], test["stage_123"].fillna(3))),
        "sample_id_sha256": {
            role: hashlib.sha256("\n".join(sorted(part["Sample_ID"])).encode()).hexdigest()
            for role, part in parts.items()
        },
    }
    (out_dir / "design.json").write_text(json.dumps(design, indent=2))
    print(f"Saved holdout {name}: train={len(train)}, val={len(val)}, test={len(test)}")


def save_cv(data_with_folds: pd.DataFrame, out_dir: Path, name: str, expected_n: int = 545):
    out_dir.mkdir(parents=True, exist_ok=True)
    if (data_with_folds["Fold"] < 1).any() or len(data_with_folds) != expected_n:
        raise RuntimeError(f"Invalid fold assignments in {name} (n={len(data_with_folds)})")

    audit_rows = []
    for fold in range(1, 6):
        fdir = out_dir / f"fold_{fold}"
        fdir.mkdir(parents=True, exist_ok=True)
        tr = data_with_folds[data_with_folds["Fold"] != fold].copy()
        va = data_with_folds[data_with_folds["Fold"] == fold].copy()
        if set(tr["Patient_ID"]) & set(va["Patient_ID"]):
            raise RuntimeError(f"Fold {fold} leakage in {name}")
        tr.to_csv(fdir / "train_split.csv", index=False, encoding="utf-8-sig")
        va.to_csv(fdir / "val_split.csv", index=False, encoding="utf-8-sig")
        for role, part in (("train", tr), ("validation", va)):
            for (source, cohort, stage, label), n in (
                part.groupby(["Source", "Cohort", "stage_123", "Binary_Label"], dropna=False).size().items()
            ):
                audit_rows.append(
                    {
                        "Scenario": name,
                        "Fold": fold,
                        "Role": role,
                        "Source": source,
                        "Cohort": cohort,
                        "Stage": None if pd.isna(stage) else int(stage),
                        "Binary_Label": int(label),
                        "N": int(n),
                    }
                )
    data_with_folds.to_csv(out_dir / "manifest_with_folds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(out_dir / "fold_distribution_audit.csv", index=False)
    print(f"Saved 5-fold CV {name} (n={len(data_with_folds)})")


def build_all_545_splits():
    nsclc = load_and_filter_nsclc_545()

    # 1. Pure Stage x Outcome (Holdout 381/82/82 and 5-Fold CV on N=545)
    known = nsclc[nsclc["stage_123"].notna()].copy()
    missing = nsclc[nsclc["stage_123"].isna()].copy()
    known["Stage_Outcome_Key"] = (
        "S" + known["stage_123"].astype(int).astype(str) + "_Y" + known["Binary_Label"].astype(int).astype(str)
    )
    rng = np.random.default_rng(20261130)
    test_pure = proportional_draw(known, "Stage_Outcome_Key", 82, rng)
    rem_pure = known.drop(index=test_pure.index)
    val_pure = proportional_draw(rem_pure, "Stage_Outcome_Key", 82, rng)
    train_pure = pd.concat([rem_pure.drop(index=val_pure.index), missing], ignore_index=True)
    for part in (train_pure, val_pure, test_pure):
        part.drop(columns=["Stage_Outcome_Key"], errors="ignore", inplace=True)
    save_holdout(train_pure, val_pure, test_pure, OUT / "pure_stage_holdout_545", "pure_stage_holdout_545", 20261130)

    cv_pure = nsclc.copy()
    cv_pure["Fold"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261131)
    for fold, (_, val_pos) in enumerate(skf.split(known, known["Stage_Outcome_Key"]), start=1):
        cv_pure.loc[known.index[val_pos], "Fold"] = fold
    for idx, row in missing.iterrows():
        y = int(row["Binary_Label"])
        choices = [
            (
                int(((cv_pure["Fold"] == f) & (cv_pure["Binary_Label"] == y)).sum()),
                int((cv_pure["Fold"] == f).sum()),
                f,
            )
            for f in range(1, 6)
        ]
        cv_pure.loc[idx, "Fold"] = min(choices)[2]
    save_cv(cv_pure, OUT / "pure_stage_5fold_cv_545", "pure_stage_5fold_cv_545", expected_n=545)

    # 2. Joint Source/Cohort x Stage x Outcome (Holdout 381/82/82 and 5-Fold CV on N=545)
    df_joint = nsclc.copy()
    keys = []
    for r in df_joint.itertuples():
        st = 1 if pd.isna(r.stage_123) else int(r.stage_123)
        y = int(r.Binary_Label)
        c = str(r.Cohort)
        if st == 1 and y == 0 and r.Source == "SDCH":
            keys.append("SDCH_S1_Y0")
        elif st == 3 and y == 0 and r.Source == "SDCH":
            keys.append("SDCH_S3_Y0")
        elif c == "TCIA" and y == 1:
            keys.append("TCIA_S23_Y1")
        elif c == "TCIA" and y == 0:
            keys.append("TCIA_S3_Y0")
        else:
            keys.append(f"{c}_S{st}_Y{y}")
    df_joint["Joint_Key"] = keys
    print("NSCLC-545 Joint_Key counts:\n", df_joint["Joint_Key"].value_counts().to_string())

    rng_j = np.random.default_rng(20261132)
    test_j = proportional_draw(df_joint, "Joint_Key", 82, rng_j)
    rem_j = df_joint.drop(index=test_j.index)
    target_counts = test_j["Joint_Key"].value_counts()
    val_j_ix = []
    for k_name, n_target in target_counts.items():
        cands = rem_j.index[rem_j["Joint_Key"] == k_name].to_numpy()
        val_j_ix.extend(rng_j.choice(cands, size=int(n_target), replace=False).tolist())
    val_j = rem_j.loc[val_j_ix].copy()
    train_j = rem_j.drop(index=val_j.index).copy()
    for part in (train_j, val_j, test_j):
        part.drop(columns=["Joint_Key"], errors="ignore", inplace=True)
    save_holdout(train_j, val_j, test_j, OUT / "source_stage_holdout_545", "source_stage_holdout_545", 20261132)

    cv_j = df_joint.drop(columns=["Joint_Key"]).copy()
    cv_j["Fold"] = -1
    skf_j = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261133)
    for fold, (_, val_pos) in enumerate(skf_j.split(df_joint, df_joint["Joint_Key"]), start=1):
        cv_j.loc[df_joint.index[val_pos], "Fold"] = fold
    save_cv(cv_j, OUT / "source_stage_5fold_cv_545", "source_stage_5fold_cv_545", expected_n=545)

    # 3. Representative Source x Outcome Holdout (381 Train / 82 Val / 82 Test)
    df_rep = nsclc.copy()
    df_rep["Source_Outcome_Key"] = df_rep["Source"].astype(str) + "_Y" + df_rep["Binary_Label"].astype(int).astype(str)
    rng_r = np.random.default_rng(20261134)
    # Quotas for 82: SDCH_Y0=46, SDCH_Y1=17, TCIA_Y0=1, TCIA_Y1=18 (total=82)
    quotas = {"SDCH_Y0": 46, "SDCH_Y1": 17, "TCIA_Y0": 1, "TCIA_Y1": 18}
    test_r_ix = []
    for k_name, n_q in quotas.items():
        cands = df_rep.index[df_rep["Source_Outcome_Key"] == k_name].to_numpy()
        test_r_ix.extend(rng_r.choice(cands, size=n_q, replace=False).tolist())
    test_r = df_rep.loc[test_r_ix].copy()
    rem_r = df_rep.drop(index=test_r.index)
    val_r_ix = []
    for k_name, n_q in quotas.items():
        cands = rem_r.index[rem_r["Source_Outcome_Key"] == k_name].to_numpy()
        val_r_ix.extend(rng_r.choice(cands, size=n_q, replace=False).tolist())
    val_r = rem_r.loc[val_r_ix].copy()
    train_r = rem_r.drop(index=val_r.index).copy()
    for part in (train_r, val_r, test_r):
        part.drop(columns=["Source_Outcome_Key"], errors="ignore", inplace=True)
    save_holdout(train_r, val_r, test_r, OUT / "representative_holdout_545", "representative_holdout_545", 20261134)

    # 4. Source-Separated 416 SDCH 5-Fold CV -> 129 Locked TCIA External Evaluation
    sdch_416 = nsclc[nsclc["Source"] == "SDCH"].reset_index(drop=True).copy()
    tcia_129 = nsclc[nsclc["Source"] == "TCIA"].reset_index(drop=True).copy()
    sdch_known = sdch_416[sdch_416["stage_123"].notna()].copy()
    sdch_miss = sdch_416[sdch_416["stage_123"].isna()].copy()
    sdch_known["SDCH_Key"] = (
        "S" + sdch_known["stage_123"].astype(int).astype(str) + "_Y" + sdch_known["Binary_Label"].astype(int).astype(str)
    )
    sdch_416["Fold"] = -1
    skf_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261135)
    for fold, (_, val_pos) in enumerate(skf_s.split(sdch_known, sdch_known["SDCH_Key"]), start=1):
        sdch_416.loc[sdch_known.index[val_pos], "Fold"] = fold
    for idx, row in sdch_miss.iterrows():
        y = int(row["Binary_Label"])
        choices = [
            (
                int(((sdch_416["Fold"] == f) & (sdch_416["Binary_Label"] == y)).sum()),
                int((sdch_416["Fold"] == f).sum()),
                f,
            )
            for f in range(1, 6)
        ]
        sdch_416.loc[idx, "Fold"] = min(choices)[2]
    save_cv(sdch_416, OUT / "sdch_416_to_tcia_129_cv", "sdch_416_to_tcia_129_cv", expected_n=416)
    tcia_129.to_csv(OUT / "sdch_416_to_tcia_129_cv/tcia_external_129.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    build_all_545_splits()
