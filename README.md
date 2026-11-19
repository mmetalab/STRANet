# STRANet: Siamese Temporal Radiotherapy Assessment Network

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.12+%20(CUDA%2012.0)-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zenodo Data](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22019982-blue.svg)](https://doi.org/10.5281/zenodo.22019982)

Official code repository for the manuscript:

> **Capturing Temporal Lung Dynamics via Explainable Deep Siamese Networks for Predicting Radiotherapy Outcomes in NSCLC**  
> Xia Lin\*, Tingting Zhang\*, Xin Sun, Qitao Chen, Lantao Wang, Zhaomin Liu, Zhe Wang, Tao Zhang, Xianzhen Zhang, Jianbin Li, Fengxiang Li†, and Cheng Wang†  
> (*Joint first authors; †Corresponding authors: `chengwang@sdu.edu.cn`, `lifengxiangli@aliyun.com`*)

---

## Overview

**STRANet** (Siamese Temporal Radiotherapy Assessment Network) is a 3D Siamese deep learning framework that jointly processes paired pre- and post-radiotherapy (RT) whole-lung CT volumes (`64 × 64 × 64`) to predict post-RT disease progression in patients with non-small cell lung cancer (NSCLC).

Key architectural and methodological components in **v2.0**:
- **Multi-center NSCLC Cohort (`N = 545`)**: 416 patients from Shandong Cancer Hospital (SDCH) and 129 patients from the publicly available ACRIN-NSCLC-FDG-PET cohort in The Cancer Imaging Archive (TCIA); overall 234 (42.9%) progression and 311 (57.1%) non-progression across Stage I (`n = 247`, 45.3%), Stage II (`n = 106`, 19.5%), and Stage III (`n = 192`, 35.2%).
- **3D Spatial-Delta Siamese Architecture (`1,632,178` trainable parameters)**:
  - Shared 3-block 3D convolutional encoder (`32 → 64 → 128` filters) with **Group Normalization (8 groups)**, L2 regularization (`λ = 3 × 10⁻⁵`), and `SpatialDropout3D`.
  - Voxel-wise **3D Spatial-Delta Longitudinal Fusion**: $\text{Concat}[F_{\text{pre}}, F_{\text{post}}, F_{\text{post}} - F_{\text{pre}}, |F_{\text{post}} - F_{\text{pre}}|, F_{\text{pre}} \odot F_{\text{post}}]$ (`640@8×8×8` $\rightarrow$ `1×1×1` & `3×3×3` Conv3D + GroupNorm $\rightarrow$ `144@8×8×8`).
  - **Dual Global Max-Average & Branch Pooling (`672-D`)** feeding a fully connected head (`Dense 160 → Dense 80 → Softmax 2`).
- **STRANet-Lite**: Single-time-point Siamese variant pairing an original pre- or post-RT CT volume with a synchronized 3D augmented counterpart when paired longitudinal imaging is unavailable.
- **Leakage-Free Handcrafted Radiomics Pipeline**: 1,106 IBSI-compliant PyRadiomics features per scan (2,212 for longitudinal union), with `StandardScaler` standardization, mRMR filtering (top 50), LASSO feature selection, and logistic regression hyperparameter tuning performed strictly within each training split.
- **Dual Evaluation Protocol**: Source- and stage-stratified five-fold cross-validation out-of-fold (OOF) evaluation (`N = 545`) alongside stratified 70:15:15 held-out testing (`n = 82`), plus institutional source-only / stage-only comparators, source-separated validation (`SDCH N = 416 → TCIA n = 129`), paired DeLong testing with Holm–Bonferroni correction, calibration, decision-curve analysis (DCA), and 3D Grad-CAM interpretability.

---

## Benchmark Summary (`NSCLC-545`)

| Model | 5-Fold OOF AUROC (95% CI) | 5-Fold OOF AUPRC | Held-Out Test AUROC (n = 82) |
| :--- | :---: | :---: | :---: |
| **STRANet (Whole-Lung Longitudinal)** | **0.850 (0.815–0.885)** | **0.849** | **0.835 (0.729–0.926)** |
| Clinical + Rad-score + STRANet | 0.848 (0.812–0.883) | 0.846 | 0.840 (0.743–0.918) |
| STRANet-Lite (Pre-RT) | 0.826 (0.790–0.862) | 0.829 | 0.829 (0.718–0.925) |
| Pre_CNN | 0.824 (0.785–0.859) | 0.819 | 0.798 (0.693–0.899) |
| Clinical + Rad-score | 0.821 (0.780–0.856) | 0.817 | 0.842 (0.743–0.926) |
| Longitudinal_Lung (Radiomics) | 0.818 (0.779–0.853) | 0.810 | 0.842 (0.744–0.926) |
| STRANet-Lite (Post-RT) | 0.815 (0.777–0.854) | 0.813 | 0.795 / 0.840 |
| Post_CNN | 0.787 (0.743–0.828) | 0.784 | 0.714 (0.591–0.818) |
| Clinical-only | 0.778 (0.734–0.815) | 0.793 | 0.757 (0.636–0.860) |
| Longitudinal_Tumor (Radiomics) | 0.755 (0.711–0.794) | 0.701 | 0.657 (0.523–0.776) |
| Source-only Comparator | 0.753 (0.719–0.785) | — | 0.747 (0.657–0.833) |
| Stage-only Comparator | 0.715 (0.673–0.754) | — | 0.751 (0.646–0.852) |

---

## Repository Structure

```text
STRANet/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── src/
│   ├── preprocessing/
│   │   ├── prepare_nsclc_545_splits.py           # Stratified 5-fold CV (N=545) & 70:15:15 holdout splits
│   │   ├── prepare_sdch_only_416_splits.py       # Source-separated SDCH (N=416) -> TCIA (n=129) splits
│   │   └── harmonized_loader.py                  # 3D CT windowing, 64x64x64 resizing, synchronized 3D augmentation
│   ├── radiomics/
│   │   └── extract_and_benchmark_lung_radiomics_545.py  # 1,106 IBSI features + leakage-free mRMR/LASSO pipeline
│   ├── models/
│   │   ├── stranet_architecture.py               # Standalone TensorFlow/Keras STRANet & STRANet-Lite definitions
│   │   ├── train_stranet_545.py                  # 5-fold CV & holdout training with cosine warmup, SWA, 8-view TTA
│   │   ├── train_stranet_lite_545.py             # Single-time-point STRANet-Lite (Pre/Post) training
│   │   ├── run_sota_benchmarks_545.py            # 3D CNN, ResNet, DenseNet, and STRANet_Tumor benchmarks
│   │   └── run_source_separated_validation.py    # SDCH-only OOF and SDCH -> TCIA external validation
│   ├── evaluation/
│   │   ├── paired_auc_inference.py               # Paired DeLong tests with Holm-Bonferroni correction
│   │   ├── clinical_and_calibration.py           # Clinical-only & multimodal fusion, Brier score, calibration, DCA
│   │   └── gradcam_3d.py                         # 3D Grad-CAM interpretability maps
│   └── figures/
│       └── generate_publication_figures.py       # Vector PDF panel generation (Figs 2-3, Supplementary Figs S4-S11)
└── examples/
    └── demo_reproduce_from_zenodo.py             # Reproduces Table 1, Fig 2/3 AUROCs, and DeLong tests from Zenodo CSVs
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate stranet
```

Or via `pip`:

```bash
pip install -r requirements.txt
```

---

## Data Availability & Reproducibility

1. **TCIA Imaging Data**: The ACRIN-NSCLC-FDG-PET cohort is publicly available via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/).
2. **De-Identified Clinical Data & Derived Features (Zenodo)**: Original SDCH CT imaging files cannot be publicly deposited under the approved Institutional Review Board protocol (Approval No. `SDTHEC202505035`). To facilitate full reproducibility while protecting patient privacy, the de-identified patient-level clinical metadata (`NSCLC_001`–`NSCLC_545`), train/validation/test split definitions, extracted whole-lung and tumor radiomics feature matrices (`1,106` IBSI features per scan), model out-of-fold/held-out predictions, and figure/table source data are deposited on Zenodo:
   - **Zenodo DOI**: [https://doi.org/10.5281/zenodo.22019982](https://doi.org/10.5281/zenodo.22019982)

### Quick Reproducibility Verification Using Zenodo Data

Download and extract the Zenodo archive to `./STRANet_Data`, then run:

```bash
python examples/demo_reproduce_from_zenodo.py --zenodo-dir ./STRANet_Data
```

This script verifies the `N = 545` baseline demographics (`Table 1`), recomputes the 5-fold OOF and held-out test AUROC/AUPRC values for `STRANet`, `STRANet-Lite`, `Longitudinal_Lung`, and `Clinical-only`, and reproduces the paired DeLong tests (`STRANet` vs. `Clinical-only`, `Source-only`, `Stage-only`, and `Longitudinal_Lung`).

---

## License

This project is released under the [MIT License](LICENSE).
