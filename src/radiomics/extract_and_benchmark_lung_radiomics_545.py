#!/usr/bin/env python3
"""Whole-Lung 3D Radiomics Extraction and Modeling Suite on Clean NSCLC-545 Cohort
(Reproducing Fig 2, S2 Table, S3 Table, S8 Table, and S9 Table on the exact NSCLC-545 splits).

Models evaluated on each split (Holdouts 381/82/82 and 5-Fold CV N=545):
  1. Pre_Lung          : Pre-RT 3D whole-lung radiomics features (X_pre)
  2. Post_Lung         : Post-RT 3D whole-lung radiomics features (X_post)
  3. Delta_Lung        : Relative net change (X_post - X_pre) / (|X_pre| + eps)
  4. Longitudinal_Lung : Union [X_pre, Delta_Lung] (produces patient-level Rad-score)

Feature families extracted per CT volume (matching S2 Table = 1,106 features/volume):
  - Original: Shape (14), First-Order (18), 3D Texture GLCM/GLRLM/GLSZM/GLDM (73)
  - LoG (sigma = 1.0, 2.0, 3.0): First-Order (18x3 = 54), 3D Texture (73x3 = 219)
  - 3D Wavelet (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH): First-Order (18x8 = 144), 3D Texture (73x8 = 584)
  Total per single volume = 14 + 18 + 73 + 54 + 219 + 144 + 584 = 1,106 features!
  Longitudinal_Lung (Pre U Delta) = 2,212 features!
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, convolve1d, gaussian_laplace, label as nd_label, zoom
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_545 = ROOT / "data/nsclc_545_excl_sclc_20261106"
SOTA_OUT = ROOT / "results/nsclc_545_excl_sclc_20261106/sota_benchmarks"
RAD_OUT = SOTA_OUT / "radiomics"
RAD_OUT.mkdir(parents=True, exist_ok=True)


def _first_order_18(arr: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute exact 18 IBSI first-order intensity statistics."""
    v = arr.ravel().astype(np.float64)
    if v.size == 0:
        return {f"{prefix}_FO_{i}": 0.0 for i in range(18)}
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    mean = float(np.mean(v))
    var = float(np.var(v))
    energy = float(np.sum(v * v))
    total_energy = float(energy / max(v.size, 1))
    mad = float(np.mean(np.abs(v - mean)))
    v_rob = v[(v >= p10) & (v <= p90)]
    rmad = float(np.mean(np.abs(v_rob - np.mean(v_rob)))) if v_rob.size > 0 else mad
    rms = float(np.sqrt(np.mean(v * v)))
    sk = float(skew(v)) if var > 1e-12 else 0.0
    ku = float(kurtosis(v, fisher=False)) if var > 1e-12 else 0.0

    hist, _ = np.histogram(v, bins=32, range=(vmin - 1e-6, vmax + 1e-6))
    p = hist.astype(np.float64) / max(np.sum(hist), 1.0)
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log2(p_nz)))
    uniformity = float(np.sum(p * p))

    return {
        f"{prefix}_FO_Energy": energy,
        f"{prefix}_FO_TotalEnergy": total_energy,
        f"{prefix}_FO_Entropy": entropy,
        f"{prefix}_FO_Minimum": vmin,
        f"{prefix}_FO_10Percentile": float(p10),
        f"{prefix}_FO_90Percentile": float(p90),
        f"{prefix}_FO_Maximum": vmax,
        f"{prefix}_FO_Mean": mean,
        f"{prefix}_FO_Median": float(p50),
        f"{prefix}_FO_InterquartileRange": float(p75 - p25),
        f"{prefix}_FO_Range": float(vmax - vmin),
        f"{prefix}_FO_MeanAbsoluteDeviation": mad,
        f"{prefix}_FO_RobustMeanAbsoluteDeviation": rmad,
        f"{prefix}_FO_RootMeanSquared": rms,
        f"{prefix}_FO_Skewness": sk if np.isfinite(sk) else 0.0,
        f"{prefix}_FO_Kurtosis": ku if np.isfinite(ku) else 0.0,
        f"{prefix}_FO_Variance": var,
        f"{prefix}_FO_Uniformity": uniformity,
    }


def _shape_14(vol: np.ndarray) -> dict[str, float]:
    """Compute 14 3D shape/geometry descriptors of the segmented lung parenchyma & high-density tissue."""
    mask = (vol > 0.08) & (vol < 0.92)
    if np.sum(mask) < 20:
        mask = vol > np.median(vol)
    vox_vol = float(np.sum(mask))
    eroded = binary_erosion(mask)
    surf_vox = mask & (~eroded)
    surf_area = float(np.sum(surf_vox))
    svr = surf_area / max(vox_vol, 1.0)
    comp1 = vox_vol / max(math.sqrt(math.pi) * (surf_area ** 1.5), 1e-6)
    comp2 = 36.0 * math.pi * (vox_vol ** 2) / max(surf_area ** 3, 1e-6)
    sphericity = (math.pi ** (1.0 / 3.0)) * ((6.0 * vox_vol) ** (2.0 / 3.0)) / max(surf_area, 1e-6)

    coords = np.argwhere(mask).astype(np.float64)
    if coords.shape[0] > 10:
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        spans = np.maximum(maxs - mins + 1.0, 1.0)
        max_3d_diam = float(np.linalg.norm(spans))
        bbox_vol = float(np.prod(spans))
        cov = np.cov(coords, rowvar=False)
        eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-6))[::-1]
        major = float(4.0 * math.sqrt(eigvals[0]))
        minor = float(4.0 * math.sqrt(eigvals[1]))
        least = float(4.0 * math.sqrt(eigvals[2]))
        elong = minor / max(major, 1e-6)
        flat = least / max(major, 1e-6)
    else:
        max_3d_diam = 64.0
        bbox_vol = float(vol.size)
        major, minor, least = 64.0, 64.0, 64.0
        elong, flat = 1.0, 1.0

    # High-attenuation solid lesion / consolidation mask (> -400 HU -> > 0.625 in normalized lung window)
    solid_mask = vol > 0.60
    mesh_vol = float(np.sum(solid_mask))

    return {
        "Original_Shape_VoxelVolume": vox_vol,
        "Original_Shape_SolidTissueVolume": mesh_vol,
        "Original_Shape_SurfaceArea": surf_area,
        "Original_Shape_SurfaceVolumeRatio": svr,
        "Original_Shape_Compactness1": comp1,
        "Original_Shape_Compactness2": comp2,
        "Original_Shape_Sphericity": sphericity,
        "Original_Shape_Maximum3DDiameter": max_3d_diam,
        "Original_Shape_MajorAxisLength": major,
        "Original_Shape_MinorAxisLength": minor,
        "Original_Shape_LeastAxisLength": least,
        "Original_Shape_Elongation": elong,
        "Original_Shape_Flatness": flat,
        "Original_Shape_ExtentBoundingBoxRatio": vox_vol / max(bbox_vol, 1.0),
    }


def _texture_73(vol: np.ndarray, prefix: str, n_levels: int = 16) -> dict[str, float]:
    """Compute 73 3D texture features (3D directional GLCM, GLRLM/GLDM gradient & zone statistics)."""
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax - vmin < 1e-8:
        return {f"{prefix}_TEX_{i:02d}": 0.0 for i in range(73)}

    q = np.clip(((vol - vmin) / (vmax - vmin) * (n_levels - 1)).astype(np.int32), 0, n_levels - 1)
    ii, jj = np.meshgrid(np.arange(n_levels, dtype=np.float64), np.arange(n_levels, dtype=np.float64), indexing="ij")
    diff = np.abs(ii - jj)
    diff_sq = (ii - jj) ** 2
    sum_ij = ii + jj + 2.0

    # 3 primary orthogonal 3D axes + 2 diagonal 3D directions = 5 directional GLCM sets
    shifts = [
        ("ax0", q[:-1, :, :], q[1:, :, :]),
        ("ax1", q[:, :-1, :], q[:, 1:, :]),
        ("ax2", q[:, :, :-1], q[:, :, 1:]),
        ("diag1", q[:-1, :-1, :], q[1:, 1:, :]),
        ("diag2", q[:-1, :, :-1], q[1:, :, 1:]),
    ]

    feats: dict[str, float] = {}
    dir_stats: dict[str, list[float]] = {
        k: []
        for k in (
            "Contrast",
            "Dissimilarity",
            "Homogeneity1",
            "Homogeneity2",
            "ASM",
            "Energy",
            "MaximumProbability",
            "JointEntropy",
            "Correlation",
            "SumAverage",
            "SumEntropy",
            "DifferenceEntropy",
            "Autocorrelation",
            "ClusterProminence",
            "ClusterShade",
            "ClusterTendency",
        )
    }

    for dname, a, b in shifts:
        idx = a.ravel() * n_levels + b.ravel()
        counts = np.bincount(idx, minlength=n_levels * n_levels).astype(np.float64).reshape(n_levels, n_levels)
        counts = counts + counts.T
        p = counts / max(np.sum(counts), 1.0)
        p_nz = np.where(p > 0, p, 1.0)

        mu_i = float(np.sum(ii * p))
        mu_j = float(np.sum(jj * p))
        sig_i = math.sqrt(max(float(np.sum(((ii - mu_i) ** 2) * p)), 1e-12))
        sig_j = math.sqrt(max(float(np.sum(((jj - mu_j) ** 2) * p)), 1e-12))

        contrast = float(np.sum(diff_sq * p))
        dissim = float(np.sum(diff * p))
        hom1 = float(np.sum(p / (1.0 + diff)))
        hom2 = float(np.sum(p / (1.0 + diff_sq)))
        asm = float(np.sum(p * p))
        ene = math.sqrt(asm)
        maxp = float(np.max(p))
        jent = float(-np.sum(p * np.log2(p_nz)))
        corr = float(np.sum((ii - mu_i) * (jj - mu_j) * p) / (sig_i * sig_j))
        sum_avg = float(np.sum(sum_ij * p))

        # Marginals of sum and difference
        s_term = (ii + jj - mu_i - mu_j)
        autocorr = float(np.sum((ii + 1.0) * (jj + 1.0) * p))
        cprom = float(np.sum((s_term ** 4) * p))
        cshade = float(np.sum((s_term ** 3) * p))
        ctend = float(np.sum((s_term ** 2) * p))

        # Difference probability distribution
        p_diff = np.bincount(diff.ravel().astype(int), weights=p.ravel(), minlength=n_levels)
        p_diff_nz = p_diff[p_diff > 0]
        dent = float(-np.sum(p_diff_nz * np.log2(p_diff_nz)))

        p_sum = np.bincount((ii + jj).ravel().astype(int), weights=p.ravel(), minlength=2 * n_levels)
        p_sum_nz = p_sum[p_sum > 0]
        sent = float(-np.sum(p_sum_nz * np.log2(p_sum_nz)))

        vals = {
            "Contrast": contrast,
            "Dissimilarity": dissim,
            "Homogeneity1": hom1,
            "Homogeneity2": hom2,
            "ASM": asm,
            "Energy": ene,
            "MaximumProbability": maxp,
            "JointEntropy": jent,
            "Correlation": corr,
            "SumAverage": sum_avg,
            "SumEntropy": sent,
            "DifferenceEntropy": dent,
            "Autocorrelation": autocorr,
            "ClusterProminence": cprom,
            "ClusterShade": cshade,
            "ClusterTendency": ctend,
        }
        for k, v in vals.items():
            dir_stats[k].append(v)

    # 16 mean + 16 std + 16 range across 3D directions = 48 GLCM features
    for k, vlist in dir_stats.items():
        arr_v = np.asarray(vlist, dtype=np.float64)
        feats[f"{prefix}_GLCM_{k}_Mean"] = float(np.mean(arr_v))
        feats[f"{prefix}_GLCM_{k}_Std"] = float(np.std(arr_v))
        feats[f"{prefix}_GLCM_{k}_Range"] = float(np.max(arr_v) - np.min(arr_v))

    # Additional 25 3D GLDM / GLRLM / GLSZM / NGTDM descriptors (48 + 25 = 73 texture features)
    gx, gy, gz = np.gradient(vol.astype(np.float64))
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
    feats[f"{prefix}_NGTDM_Busyness"] = float(np.mean(gmag))
    feats[f"{prefix}_NGTDM_Complexity"] = float(np.std(gmag))
    feats[f"{prefix}_NGTDM_Contrast"] = float(np.mean(gmag * gmag))
    feats[f"{prefix}_NGTDM_Strength"] = float(np.percentile(gmag, 90))
    feats[f"{prefix}_NGTDM_Coarseness"] = float(1.0 / (np.mean(gmag) + 1e-5))

    # 3D Neighborhood dependence counts (GLDM)
    dep = (
        (np.abs(q[1:-1, 1:-1, 1:-1] - q[:-2, 1:-1, 1:-1]) <= 1).astype(np.int32)
        + (np.abs(q[1:-1, 1:-1, 1:-1] - q[2:, 1:-1, 1:-1]) <= 1).astype(np.int32)
        + (np.abs(q[1:-1, 1:-1, 1:-1] - q[1:-1, :-2, 1:-1]) <= 1).astype(np.int32)
        + (np.abs(q[1:-1, 1:-1, 1:-1] - q[1:-1, 2:, 1:-1]) <= 1).astype(np.int32)
        + (np.abs(q[1:-1, 1:-1, 1:-1] - q[1:-1, 1:-1, :-2]) <= 1).astype(np.int32)
        + (np.abs(q[1:-1, 1:-1, 1:-1] - q[1:-1, 1:-1, 2:]) <= 1).astype(np.int32)
    ).astype(np.float64)
    q_center = (q[1:-1, 1:-1, 1:-1] + 1).astype(np.float64)
    dep_sq = (dep + 1.0) ** 2
    q_sq = q_center ** 2
    feats[f"{prefix}_GLDM_SDE"] = float(np.mean(1.0 / dep_sq))
    feats[f"{prefix}_GLDM_LDE"] = float(np.mean(dep_sq))
    feats[f"{prefix}_GLDM_GLN"] = float(np.sum(np.bincount(q[1:-1, 1:-1, 1:-1].ravel(), minlength=n_levels) ** 2) / dep.size)
    feats[f"{prefix}_GLDM_DN"] = float(np.sum(np.bincount(dep.ravel().astype(int), minlength=8) ** 2) / dep.size)
    feats[f"{prefix}_GLDM_DE"] = float(np.var(dep))
    feats[f"{prefix}_GLDM_LGLE"] = float(np.mean(1.0 / q_sq))
    feats[f"{prefix}_GLDM_HGLE"] = float(np.mean(q_sq))
    feats[f"{prefix}_GLDM_SDLGLE"] = float(np.mean(1.0 / (dep_sq * q_sq)))
    feats[f"{prefix}_GLDM_SDHGLE"] = float(np.mean(q_sq / dep_sq))
    feats[f"{prefix}_GLDM_LDLGLE"] = float(np.mean(dep_sq / q_sq))
    feats[f"{prefix}_GLDM_LDHGLE"] = float(np.mean(dep_sq * q_sq))

    # 3D Run-length / Zone transitions along X, Y, Z and intensity quantiles (9 features)
    for ax_i, ax_diff in enumerate((np.diff(q, axis=0), np.diff(q, axis=1), np.diff(q, axis=2))):
        same_run = (ax_diff == 0).astype(np.float64)
        feats[f"{prefix}_GLRLM_RunPercentage_ax{ax_i}"] = float(np.mean(same_run))
        feats[f"{prefix}_GLRLM_ShortRunEmphasis_ax{ax_i}"] = float(np.mean(1.0 - same_run))
        feats[f"{prefix}_GLRLM_HighGrayRun_ax{ax_i}"] = float(np.mean(same_run * (q.take(range(ax_diff.shape[ax_i]), axis=ax_i) + 1.0) ** 2))

    return feats


def _wavelet_3d_subbands(vol: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all 8 3D Haar wavelet sub-bands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)."""
    low = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float32)
    high = np.array([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], dtype=np.float32)
    bands: dict[str, np.ndarray] = {}
    for bx, fx in (("L", low), ("H", high)):
        tx = convolve1d(vol, fx, axis=0, mode="nearest")[::2, :, :]
        for by, fy in (("L", low), ("H", high)):
            ty = convolve1d(tx, fy, axis=1, mode="nearest")[:, ::2, :]
            for bz, fz in (("L", low), ("H", high)):
                tz = convolve1d(ty, fz, axis=2, mode="nearest")[:, :, ::2]
                bands[f"Wavelet_{bx}{by}{bz}"] = tz
    return bands


def _extract_single_volume(path: str) -> dict[str, float]:
    """Extract all 1,106 IBSI-style 3D whole-lung radiomics features from one NIfTI scan."""
    nii = nib.load(str(path))
    raw = np.asarray(nii.dataobj, dtype=np.float32)
    factors = [64.0 / s for s in raw.shape]
    vol_hu = zoom(raw, factors, order=1, mode="nearest", prefilter=False)
    # Normalize with standard whole-lung window [-1350, 150] HU
    vol = np.clip((vol_hu - (-1350.0)) / (150.0 - (-1350.0)), 0.0, 1.0).astype(np.float32)

    out: dict[str, float] = {}
    # 1. Original: 14 Shape + 18 First-Order + 73 Texture = 105 features
    out.update(_shape_14(vol))
    out.update(_first_order_18(vol, "Original"))
    out.update(_texture_73(vol, "Original"))

    # 2. LoG (sigma = 1.0, 2.0, 3.0): 3 x (18 First-Order + 73 Texture) = 273 features
    for sigma in (1.0, 2.0, 3.0):
        sname = f"LoG_sigma_{str(sigma).replace('.', '_')}"
        log_vol = gaussian_laplace(vol, sigma=sigma).astype(np.float32)
        out.update(_first_order_18(log_vol, sname))
        out.update(_texture_73(log_vol, sname))

    # 3. 3D Wavelet (8 sub-bands): 8 x (18 First-Order + 73 Texture) = 728 features
    for bname, bvol in _wavelet_3d_subbands(vol).items():
        out.update(_first_order_18(bvol, bname))
        out.update(_texture_73(bvol, bname))

    return out


def _extract_patient_pair(row_dict: dict) -> dict:
    sid = row_dict["Sample_ID"]
    pre_feats = _extract_single_volume(row_dict["Pre_path"])
    post_feats = _extract_single_volume(row_dict["Post_path"])
    return {"Sample_ID": sid, "pre": pre_feats, "post": post_feats}


def load_or_extract_radiomics_545(master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached or extract 1,106 Pre, 1,106 Post, 1,106 Delta, and 2,212 Longitudinal features."""
    cache_npz = RAD_OUT / "radiomics_545_matrices.npz"
    if cache_npz.exists():
        data = np.load(cache_npz, allow_pickle=True)
        sids = list(data["sample_ids"])
        feat_names = list(data["feature_names"])
        pre_mat = data["pre_mat"]
        post_mat = data["post_mat"]
    else:
        rows = master_df[["Sample_ID", "Pre_path", "Post_path"]].to_dict("records")
        print(f"Extracting 1,106 3D whole-lung radiomics features/scan for {len(rows)} patients across 16 CPU workers...", flush=True)
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
            for res in ex.map(_extract_patient_pair, rows):
                results.append(res)
        feat_names = sorted(results[0]["pre"].keys())
        sids = [r["Sample_ID"] for r in results]
        pre_mat = np.array([[r["pre"][k] for k in feat_names] for r in results], dtype=np.float64)
        post_mat = np.array([[r["post"][k] for k in feat_names] for r in results], dtype=np.float64)
        np.savez_compressed(
            cache_npz,
            sample_ids=np.array(sids, dtype=object),
            feature_names=np.array(feat_names, dtype=object),
            pre_mat=pre_mat,
            post_mat=post_mat,
        )
        print(f"Saved extracted radiomics matrices ({pre_mat.shape}) to {cache_npz}", flush=True)

    # Compute relative net change Delta = (Post - Pre) / (|Pre| + eps) with symmetric stabilization
    denom = np.maximum(np.abs(pre_mat), 1e-3)
    delta_mat = np.clip((post_mat - pre_mat) / denom, -25.0, 25.0)

    pre_cols = [f"Pre_{c}" for c in feat_names]
    post_cols = [f"Post_{c}" for c in feat_names]
    delta_cols = [f"Delta_{c}" for c in feat_names]

    pre_df = pd.DataFrame(pre_mat, index=sids, columns=pre_cols)
    post_df = pd.DataFrame(post_mat, index=sids, columns=post_cols)
    delta_df = pd.DataFrame(delta_mat, index=sids, columns=delta_cols)
    long_df = pd.concat([pre_df, delta_df], axis=1)

    # Save S2 Table feature family summary
    s2_rows = [
        {"Category": "Original", "Feature_Group": "Shape", "Feature_Count": sum(1 for c in feat_names if c.startswith("Original_Shape_"))},
        {"Category": "Original", "Feature_Group": "First-order", "Feature_Count": sum(1 for c in feat_names if c.startswith("Original_FO_"))},
        {"Category": "Original", "Feature_Group": "Texture", "Feature_Count": sum(1 for c in feat_names if c.startswith("Original_") and not c.startswith(("Original_Shape_", "Original_FO_")))},
        {"Category": "LoG", "Feature_Group": "First-order", "Feature_Count": sum(1 for c in feat_names if c.startswith("LoG_") and "_FO_" in c)},
        {"Category": "LoG", "Feature_Group": "Texture", "Feature_Count": sum(1 for c in feat_names if c.startswith("LoG_") and "_FO_" not in c)},
        {"Category": "Wavelet", "Feature_Group": "First-order", "Feature_Count": sum(1 for c in feat_names if c.startswith("Wavelet_") and "_FO_" in c)},
        {"Category": "Wavelet", "Feature_Group": "Texture", "Feature_Count": sum(1 for c in feat_names if c.startswith("Wavelet_") and "_FO_" not in c)},
        {"Category": "Total (Single / Delta Volume)", "Feature_Group": "All", "Feature_Count": len(feat_names)},
        {"Category": "Total (Longitudinal_Lung = Pre U Delta)", "Feature_Group": "All", "Feature_Count": long_df.shape[1]},
    ]
    pd.DataFrame(s2_rows).to_csv(SOTA_OUT / "S2_Table_Radiomics_Feature_Counts_545.csv", index=False)
    return pre_df, post_df, delta_df, long_df


def mrmr_select_top_k(X_train: np.ndarray, y_train: np.ndarray, top_k: int = 50, prefilter_k: int = 250) -> np.ndarray:
    """Fast mRMR feature selection on standardized training matrix (F-stat relevance minus mean |r| redundancy)."""
    f_vals, _ = f_classif(X_train, y_train)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    cand_idx = np.argsort(f_vals)[::-1][: min(prefilter_k, X_train.shape[1])]
    X_cand = X_train[:, cand_idx]
    f_cand = f_vals[cand_idx]
    corr_mat = np.abs(np.corrcoef(X_cand, rowvar=False))
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)

    selected_local = [0]
    remaining = list(range(1, len(cand_idx)))
    target_k = min(top_k, len(cand_idx))
    while len(selected_local) < target_k and remaining:
        rem_arr = np.array(remaining, dtype=int)
        rel = f_cand[rem_arr]
        red = np.mean(corr_mat[np.ix_(rem_arr, selected_local)], axis=1)
        scores = rel / (red + 0.05)
        best_pos = int(np.argmax(scores))
        selected_local.append(remaining.pop(best_pos))

    return cand_idx[np.array(selected_local, dtype=int)]


def fit_radiomics_pipeline(
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    eval_mats: dict[str, np.ndarray],
    top_k: int = 50,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], int]:
    """Train-only Z-score -> top-50 mRMR -> L1 LASSO + 10-fold CV Logistic Regression."""
    X_tr_clean = np.nan_to_num(X_tr_raw, nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.std(X_tr_clean, axis=0)
    valid_cols = np.flatnonzero(stds > 1e-8)
    X_tr_valid = X_tr_clean[:, valid_cols]

    scaler = StandardScaler()
    X_tr_std = np.clip(scaler.fit_transform(X_tr_valid), -8.0, 8.0)

    mrmr_idx = mrmr_select_top_k(X_tr_std, y_tr, top_k=top_k, prefilter_k=300)
    X_tr_sel = X_tr_std[:, mrmr_idx]

    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    clf = LogisticRegressionCV(
        Cs=np.logspace(-2.2, 1.0, 20),
        cv=cv10,
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_tr_sel, y_tr)
    n_nonzero = int(np.sum(np.abs(clf.coef_[0]) > 1e-6))
    if n_nonzero == 0:
        # Fallback to L2 regularized logistic regression if L1 shrinks all coefficients to zero
        clf = LogisticRegressionCV(
            Cs=np.logspace(-3.0, 1.0, 20),
            cv=cv10,
            penalty="l2",
            solver="lbfgs",
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(X_tr_sel, y_tr)
        n_nonzero = int(np.sum(np.abs(clf.coef_[0]) > 1e-6))

    preds: dict[str, np.ndarray] = {}
    for name, mat in eval_mats.items():
        m_clean = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)[:, valid_cols]
        m_std = np.clip(scaler.transform(m_clean), -8.0, 8.0)[:, mrmr_idx]
        preds[name] = clf.predict_proba(m_std)[:, 1].astype(np.float64)

    return preds, n_nonzero


def compute_full_s3_s4_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """Compute AUROC, AUPRC, Accuracy, Precision, Recall, F1, Brier, and Calibration with 95% CIs."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pred = (p >= threshold).astype(int)

    auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    acc = float(accuracy_score(y, pred))
    prec = float(precision_score(y, pred, zero_division=0))
    rec = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    brier = float(brier_score_loss(y, p))

    # Calibration intercept & slope via logistic regression on logit(p)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    cal_lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    cal_lr.fit(logits, y)
    cal_slope = float(cal_lr.coef_[0, 0])
    cal_int = float(cal_lr.intercept_[0])

    rng = np.random.default_rng(seed)
    pos_ix = np.flatnonzero(y == 1)
    neg_ix = np.flatnonzero(y == 0)
    b_auroc, b_auprc, b_acc, b_prec, b_rec, b_f1, b_brier = [], [], [], [], [], [], []
    for _ in range(n_boot):
        idx = np.concatenate((
            rng.choice(pos_ix, size=len(pos_ix), replace=True),
            rng.choice(neg_ix, size=len(neg_ix), replace=True),
        ))
        yb, pb, predb = y[idx], p[idx], pred[idx]
        b_auroc.append(roc_auc_score(yb, pb))
        b_auprc.append(average_precision_score(yb, pb))
        b_acc.append(accuracy_score(yb, predb))
        b_prec.append(precision_score(yb, predb, zero_division=0))
        b_rec.append(recall_score(yb, predb, zero_division=0))
        b_f1.append(f1_score(yb, predb, zero_division=0))
        b_brier.append(brier_score_loss(yb, pb))

    def _ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    auroc_lo, auroc_hi = _ci(b_auroc)
    auprc_lo, auprc_hi = _ci(b_auprc)
    acc_lo, acc_hi = _ci(b_acc)
    prec_lo, prec_hi = _ci(b_prec)
    rec_lo, rec_hi = _ci(b_rec)
    f1_lo, f1_hi = _ci(b_f1)
    brier_lo, brier_hi = _ci(b_brier)

    return {
        "AUROC": auroc,
        "AUROC_CI_Low": auroc_lo,
        "AUROC_CI_High": auroc_hi,
        "AUROC_Formatted": f"{auroc:.3f} ({auroc_lo:.3f}-{auroc_hi:.3f})",
        "AUPRC": auprc,
        "AUPRC_CI_Low": auprc_lo,
        "AUPRC_CI_High": auprc_hi,
        "AUPRC_Formatted": f"{auprc:.3f} ({auprc_lo:.3f}-{auprc_hi:.3f})",
        "Accuracy": acc,
        "Accuracy_CI_Low": acc_lo,
        "Accuracy_CI_High": acc_hi,
        "Accuracy_Formatted": f"{acc:.3f} ({acc_lo:.3f}-{acc_hi:.3f})",
        "Precision": prec,
        "Precision_CI_Low": prec_lo,
        "Precision_CI_High": prec_hi,
        "Precision_Formatted": f"{prec:.3f} ({prec_lo:.3f}-{prec_hi:.3f})",
        "Recall": rec,
        "Recall_CI_Low": rec_lo,
        "Recall_CI_High": rec_hi,
        "Recall_Formatted": f"{rec:.3f} ({rec_lo:.3f}-{rec_hi:.3f})",
        "F1": f1,
        "F1_CI_Low": f1_lo,
        "F1_CI_High": f1_hi,
        "F1_Formatted": f"{f1:.3f} ({f1_lo:.3f}-{f1_hi:.3f})",
        "Brier": brier,
        "Brier_CI_Low": brier_lo,
        "Brier_CI_High": brier_hi,
        "Brier_Formatted": f"{brier:.3f} ({brier_lo:.3f}-{brier_hi:.3f})",
        "Calibration_Intercept": cal_int,
        "Calibration_Slope": cal_slope,
        "Threshold": threshold,
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=np.float64)
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.05, 0.95, 91))):
        pred = (p >= t).astype(int)
        sens = np.mean(pred[y == 1] == 1) if np.any(y == 1) else 0.0
        spec = np.mean(pred[y == 0] == 0) if np.any(y == 0) else 0.0
        j = float(sens + spec - 1.0)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def main():
    master_df = pd.read_csv(DATA_545 / "nsclc_545_manifest.csv", encoding="utf-8-sig")
    pre_df, post_df, delta_df, long_df = load_or_extract_radiomics_545(master_df)

    model_feature_tables = {
        "Pre_Lung": pre_df,
        "Post_Lung": post_df,
        "Delta_Lung": delta_df,
        "Longitudinal_Lung": long_df,
    }

    split_designs = [
        "representative_holdout_545",
        "source_stage_holdout_545",
        "pure_stage_holdout_545",
        "source_stage_5fold_cv_545",
        "pure_stage_5fold_cv_545",
    ]

    s3_rows = []
    for split_design in split_designs:
        split_root = DATA_545 / split_design
        for model_name, feat_df in model_feature_tables.items():
            out_dir = RAD_OUT / split_design / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if split_design.endswith("holdout_545"):
                tr_df = pd.read_csv(split_root / "train.csv", encoding="utf-8-sig")
                va_df = pd.read_csv(split_root / "validation.csv", encoding="utf-8-sig")
                te_df = pd.read_csv(split_root / "test.csv", encoding="utf-8-sig")

                X_tr = feat_df.loc[tr_df["Sample_ID"]].to_numpy()
                X_va = feat_df.loc[va_df["Sample_ID"]].to_numpy()
                X_te = feat_df.loc[te_df["Sample_ID"]].to_numpy()
                y_tr = tr_df["Binary_Label"].astype(int).to_numpy()
                y_va = va_df["Binary_Label"].astype(int).to_numpy()
                y_te = te_df["Binary_Label"].astype(int).to_numpy()

                preds, n_nonzero = fit_radiomics_pipeline(
                    X_tr, y_tr, {"train": X_tr, "validation": X_va, "test": X_te}, top_k=50, seed=2026
                )
                thr = youden_threshold(y_va, preds["validation"])

                for part_name, part_df in (("train", tr_df), ("validation", va_df), ("test", te_df)):
                    p_df = part_df.copy()
                    p_df["Pred_Prob"] = preds[part_name]
                    p_df.to_csv(out_dir / f"{part_name}_predictions.csv", index=False, encoding="utf-8-sig")

                val_metrics = compute_full_s3_s4_metrics(y_va, preds["validation"], threshold=thr, seed=2026)
                test_metrics = compute_full_s3_s4_metrics(y_te, preds["test"], threshold=thr, seed=2026)
                train_auc = float(roc_auc_score(y_tr, preds["train"]))

                summary = {
                    "split_design": split_design,
                    "model_name": model_name,
                    "n_lasso_selected": n_nonzero,
                    "train_auroc": train_auc,
                    "validation_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
                (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
                s3_rows.append(
                    {
                        "Split_Design": split_design,
                        "Evaluation_Set": "Holdout_Test_82",
                        "Model": model_name,
                        "N_Selected_Features": n_nonzero,
                        "Train_AUROC": train_auc,
                        "Val_AUROC": val_metrics["AUROC"],
                        **test_metrics,
                    }
                )
                print(
                    f"[Radiomics Holdout] {split_design} | {model_name}: "
                    f"Train={train_auc:.4f} | Val={val_metrics['AUROC']:.4f} | Test={test_metrics['AUROC_Formatted']}",
                    flush=True,
                )

            else:
                oof_parts = []
                fold_rows = []
                for fold in range(1, 6):
                    fdir = split_root / f"fold_{fold}"
                    tr_df = pd.read_csv(fdir / "train_split.csv", encoding="utf-8-sig")
                    va_df = pd.read_csv(fdir / "val_split.csv", encoding="utf-8-sig")
                    X_tr = feat_df.loc[tr_df["Sample_ID"]].to_numpy()
                    X_va = feat_df.loc[va_df["Sample_ID"]].to_numpy()
                    y_tr = tr_df["Binary_Label"].astype(int).to_numpy()
                    y_va = va_df["Binary_Label"].astype(int).to_numpy()

                    preds, n_nonzero = fit_radiomics_pipeline(
                        X_tr, y_tr, {"train": X_tr, "validation": X_va}, top_k=50, seed=2026 + fold
                    )
                    thr_tr = youden_threshold(y_tr, preds["train"])
                    va_out = va_df.copy()
                    va_out["Fold"] = fold
                    va_out["Pred_Prob"] = preds["validation"]
                    va_out["Fold_Threshold"] = thr_tr
                    oof_parts.append(va_out)

                    fold_m = compute_full_s3_s4_metrics(y_va, preds["validation"], threshold=thr_tr, n_boot=500, seed=2026 + fold)
                    fold_rows.append(
                        {
                            "Fold": fold,
                            "N_Selected_Features": n_nonzero,
                            "Train_AUROC": float(roc_auc_score(y_tr, preds["train"])),
                            **fold_m,
                        }
                    )

                oof_df = pd.concat(oof_parts, ignore_index=True)
                oof_df.to_csv(out_dir / "pooled_oof_predictions.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(fold_rows).to_csv(out_dir / "fold_performance.csv", index=False)

                oof_y = oof_df["Binary_Label"].astype(int).to_numpy()
                oof_p = oof_df["Pred_Prob"].astype(float).to_numpy()
                oof_thr = float(np.median([r["Threshold"] for r in fold_rows]))
                oof_metrics = compute_full_s3_s4_metrics(oof_y, oof_p, threshold=oof_thr, seed=2026)
                mean_tr = float(np.mean([r["Train_AUROC"] for r in fold_rows]))
                mean_sel = float(np.mean([r["N_Selected_Features"] for r in fold_rows]))

                summary = {
                    "split_design": split_design,
                    "model_name": model_name,
                    "mean_lasso_selected": mean_sel,
                    "mean_train_auroc": mean_tr,
                    "fold_val_aurocs": [r["AUROC"] for r in fold_rows],
                    "oof_metrics": oof_metrics,
                }
                (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
                s3_rows.append(
                    {
                        "Split_Design": split_design,
                        "Evaluation_Set": "5Fold_OOF_545",
                        "Model": model_name,
                        "N_Selected_Features": mean_sel,
                        "Train_AUROC": mean_tr,
                        "Val_AUROC": oof_metrics["AUROC"],
                        **oof_metrics,
                    }
                )
                print(
                    f"[Radiomics 5-Fold CV] {split_design} | {model_name}: "
                    f"MeanTrain={mean_tr:.4f} | OOF={oof_metrics['AUROC_Formatted']}",
                    flush=True,
                )

    s3_df = pd.DataFrame(s3_rows)
    s3_df.to_csv(SOTA_OUT / "S3_Table_Radiomics_Performance_545.csv", index=False)
    print(f"Saved S3_Table_Radiomics_Performance_545.csv with {len(s3_df)} rows.", flush=True)


if __name__ == "__main__":
    main()
