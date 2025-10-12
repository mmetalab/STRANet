# STRANet: Siamese Temporal Radiotherapy Assessment Network

![Python 3.9](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Introduction

Non-small cell lung cancer (NSCLC) is the most common type of lung cancer, and radiotherapy (RT) is a key treatment for patients with advanced or inoperable disease. However, predicting the efficacy of RT remains challenging due to variability in patient response.  

Traditional models rely primarily on pre-treatment CT scans but fail to capture dynamic anatomical changes during treatment.  

**STRANet (Siamese Temporal Radiotherapy Assessment Network)** is a deep learning framework designed to predict post-RT outcomes by analyzing both **pre- and post-radiotherapy 3D CT scans**. STRANet captures spatiotemporal changes across the entire lung volume, improving predictive accuracy compared to traditional tumor-focused approaches.  

We also introduce **STRANet-Lite**, a variant capable of handling incomplete longitudinal data through **pseudo-temporal augmentation**, enhancing real-world clinical applicability.

---

## Requirements

This project was developed and tested using **Python 3.9**, with dependencies managed via **Conda**.

All required packages and version specifications are listed in `environment.yml`.  
To reproduce the environment, run the following commands:

```bash
conda env create -f environment.yml
conda activate stranet
```

---

## Project Structure

The project is organized as follows:

```
STRANET/
├── Data/
│   ├── deep_features_test.csv       # Deep learning features extracted from STRANet
│   ├── deep_features_train.csv      # Deep learning features extracted from STRANet
│   ├── Image_Path.csv               # Image path references
│   ├── test_data.csv                # Test data (example)
│   └── train_data.csv               # Training data (example)
│
├── Image/
│   ├── Progress/                    # Progressing images (pre- and post-RT, and augmented versions)
│   └── Stable/                      # Stable disease images (pre- and post-RT, and augmented versions)
│
├── Model/
│   ├── _utils_siamese.ipynb         # Utility functions for the Siamese network
│   ├── DL_features.ipynb            # Deep feature extraction notebook
│   ├── final_epoch_fold4_model.h5   # Pretrained model weights
│   ├── GradCAM.ipynb                # Grad-CAM visualization for model explainability
│   ├── Metrics.ipynb                # Evaluation metrics
│   ├── Preprocessing.ipynb          # Image preprocessing steps 
│   └── Train.ipynb                  # Training notebook for STRANet model
│
└── environment.yml                  # Conda environment configuration
```

---

## How to Use STRANet

### 1. **Model Training**

To train the model, open and run **`Model/Train.ipynb`**.  
Both STRANet (two time points: pre-RT and post-RT) and STRANet-Lite (single time point with pseudo-temporal augmentation, e.g., Pre & Pseudo_Pre or Post & Pseudo_Post) share the same training code — the difference lies in input configuration.

### 2. **Model Evaluation**

To evaluate model performance:

```bash
# Run within Jupyter
Model/Metrics.ipynb
```

Key metrics include:
- AUROC (Area Under ROC Curve)  
- AUPRC (Area Under Precision-Recall Curve)

### 3. **Visualization**

For model interpretability, visualize Grad-CAM attention heatmaps by running:

```bash
Model/GradCAM.ipynb
```

### 4. **Feature Extraction and Preprocessing**

Data preprocessing and deep feature extraction are implemented in:
- **`Model/Preprocessing.ipynb`**
- **`Model/DL_features.ipynb`**

Example training and test datasets are provided in the `Data` folder, while sample CT images are available under `Image/Progress` and `Image/Stable`.

---

## Citation

If you use STRANet or STRANet-Lite in your research, please cite:

> Q. Chen, Z. Wang, X. Lin, Y. Shi, B. Xu, J. Chai, T. Zhang, C. Wang.  
> **STRANet: Siamese Temporal Radiotherapy Assessment Network for Predicting Radiotherapy Outcomes.** (2025)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please contact:  
**Cheng Wang, PhD**  
Email: cheng.wang [at] example.com  
Institution: Shandong University / Mount Sinai  
