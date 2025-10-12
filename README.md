Introduction

Non-small cell lung cancer (NSCLC) is the most common type of lung cancer, and radiotherapy (RT) is a key treatment for patients with advanced or inoperable disease. However, predicting the efficacy of RT remains challenging due to the variability in patient response. Traditional models rely on pre-treatment CT scans, but fail to capture the dynamic changes that occur during treatment.This project introduces STRANet (Siamese Temporal Radiotherapy Assessment Network), a deep learning framework designed to predict post-RT outcomes by analyzing both pre- and post-RT 3D CT scans. STRANet captures spatiotemporal changes in the entire lung volume, improving predictive accuracy over traditional tumor-focused models. We also developed STRANet-Lite, a variant that handles incomplete longitudinal data using pseudo-temporal augmentation, making it more applicable in real-world clinical scenarios.


Requirements

This project was developed and tested using Python 3.9 with dependencies managed through Conda.
All required packages and version specifications are listed in the file environment.yml.
To reproduce the environment, run:
   conda env create -f environment.yml
   conda activate stranet


How to Use STRANet

Here we list the computational pipeline for predicting radiotherapy efficacy in NSCLC patients using pre- and post-radiotherapy CT scans.

The project is organized as follows:
STRANET/
├── Data/
│   ├── deep_features_test.csv       # Deep learning features extracted from STRANet
│   ├── deep_features_train.csv      # Deep learning features extracted from STRANet
│   ├── Image_Path.csv               # Image path references
│   ├── test_data.csv                # Test data (example)
│   └── train_data.csv               # Training data (example)
│
├── Image/
│   ├── Progress/                    # Progressing images (pre- and post-RT, and their augmentated version)
│   └── Stable/                      # Stable disease images (pre- and post-RT, and their augmentated version)
│
├── Model/
│   ├── _utils_siamese.ipynb         # Utility functions for the Siamese network
│   ├── DL_features.ipynb            # Deep feature extraction notebook
│   ├── final_epoch_fold4_model.h5   # Pretrained model weights
│   ├── GradCAM.ipynb                # Grad-CAM visualization for model explainability
│   ├── Metrics.ipynb                # Evaluation metrics
│   ├── Preprocessing.ipynb          # Preprocessing steps for images 
│   └── Train.ipynb                  # Training notebook for STRANet model
│
└── environment.yml                  # Conda environment configuration

There are several Jupyter notebooks in the Model folder.
Example data for training and testing are provided in the Data folder (train_data.csv, test_data.csv), and sample CT images are available in the Image folder (Progress and Stable).

For model training, run Train.ipynb and follow the instructions in the notebook.
Both STRANet (two time points: pre-RT and post-RT) and STRANet-Lite (single time point with pseudo-temporal augmentation, e.g., Pre & Pseudo_Pre or Post & Pseudo_Post) share the same training code — the difference lies in the input data configuration.

For model evaluation and visualization:
    Run Metrics.ipynb to evaluate performance (e.g., AUROC, AUPRC).
    Run GradCAM.ipynb to visualize model interpretability via Grad-CAM heatmaps.

Preprocessing and feature extraction are implemented in Preprocessing.ipynb and DL_features.ipynb.