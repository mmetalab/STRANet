#!/usr/bin/env python3
"""SOTA Deep Learning Benchmark Runner on Clean NSCLC-545 Cohort
(Reproducing Fig 3, S4 Table, S8 Table, S9 Table, and S8 Fig on the exact NSCLC-545 splits).

Implements all 10 SOTA Deep Learning comparator models from S4 Table:
  1. Pre_CNN               : Single-timepoint 3D CNN (32-64-128) on Pre-RT CT
  2. Post_CNN              : Single-timepoint 3D CNN (32-64-128) on Post-RT CT
  3. Pre_ResNet            : Single-timepoint 3D ResNet-18 (64-128-256-512) on Pre-RT CT
  4. Post_ResNet           : Single-timepoint 3D ResNet-18 (64-128-256-512) on Post-RT CT
  5. ResNet_Longitudinal   : Shared-weight 3D Siamese ResNet-18 on Pre-RT + Post-RT CT
  6. Pre_DenseNet          : Single-timepoint 3D DenseNet (growth_rate=32) on Pre-RT CT
  7. Post_DenseNet         : Single-timepoint 3D DenseNet (growth_rate=32) on Post-RT CT
  8. DenseNet_Longitudinal : Shared-weight 3D Siamese DenseNet on Pre-RT + Post-RT CT
  9. STRANet_Lite_Pre      : Single-timepoint Siamese 3D CNN (Pre-RT CT + Augmented Twin)
 10. STRANet_Lite_Post     : Single-timepoint Siamese 3D CNN (Post-RT CT + Augmented Twin)

Also includes --profile-complexity to compute FLOPs (G), Params (M), Memory (MB), Inference Time (s), and FPS.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import rotate, shift as nd_shift, zoom
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model, mixed_precision, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv3D,
    Dense,
    Dropout,
    GlobalAveragePooling3D,
    Input,
    MaxPooling3D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, to_categorical

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
DATA_545 = ROOT / "data/nsclc_545_excl_sclc_20261106"
SOTA_OUT = ROOT / "results/nsclc_545_excl_sclc_20261106/sota_benchmarks"
DL_OUT = SOTA_OUT / "dl_models"
DL_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "code"))
from extract_and_benchmark_lung_radiomics_545 import compute_full_s3_s4_metrics, youden_threshold  # noqa: E402
from run_nsclc_545_benchmarks import build_siamese_nsclc545_model  # noqa: E402
from run_stage_source_594_benchmarks import fit_clinical_comparators, summarize_predictions  # noqa: E402

K.set_image_data_format("channels_first")


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if tf.config.list_physical_devices("GPU"):
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")


_MASTER_PSEUDO = pd.read_csv(ROOT / "data/master_with_stage.csv", encoding="utf-8-sig")[
    ["Sample_ID", "Pre_pseudo_path", "Post_pseudo_path"]
].drop_duplicates("Sample_ID")

ALL_SOTA_MODELS = (
    "Pre_CNN",
    "Post_CNN",
    "Pre_ResNet",
    "Post_ResNet",
    "ResNet_Longitudinal",
    "Pre_DenseNet",
    "Post_DenseNet",
    "DenseNet_Longitudinal",
    "STRANet_Lite_Pre",
    "STRANet_Lite_Post",
)


def is_single_input_model(model_name: str) -> bool:
    return model_name in ("Pre_CNN", "Post_CNN", "Pre_ResNet", "Post_ResNet", "Pre_DenseNet", "Post_DenseNet")


class SOTABenchmarkGenerator(Sequence):
    """Unified 3D CT generator for Single-Timepoint, Single-Timepoint Siamese (STRANet-Lite), and Longitudinal Siamese models."""

    _cache: dict[str, np.ndarray] = {}

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        batch_size: int = 8,
        balanced: bool = False,
        augment: bool = False,
        seed: int = 1234,
    ):
        self.df = df.reset_index(drop=True).copy()
        if "Pre_pseudo_path" not in self.df.columns:
            self.df = self.df.merge(_MASTER_PSEUDO, on="Sample_ID", how="left")
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.balanced = bool(balanced)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)
        self.labels = self.df["Binary_Label"].astype(int).to_numpy()
        self.on_epoch_end()

    def __len__(self) -> int:
        if self.balanced:
            return int(np.floor(len(self.df) / self.batch_size))
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if not self.balanced:
            self.indexes = np.arange(len(self.df))
            return
        half = self.batch_size // 2
        pos = np.flatnonzero(self.labels == 1)
        neg = np.flatnonzero(self.labels == 0)
        batches = []
        for _ in range(len(self)):
            p_ix = self.rng.choice(pos, size=half, replace=len(pos) < half)
            n_ix = self.rng.choice(neg, size=half, replace=len(neg) < half)
            b = np.concatenate((p_ix, n_ix))
            self.rng.shuffle(b)
            batches.append(b)
        self.indexes = np.concatenate(batches).astype(int, copy=False)

    @classmethod
    def _load(cls, path: str) -> np.ndarray:
        key = str(path)
        cached = cls._cache.get(key)
        if cached is None:
            nii = nib.load(key)
            vol = np.asarray(nii.dataobj, dtype=np.float32)
            factors = [64.0 / s for s in vol.shape]
            resized = zoom(vol, factors, order=1, mode="nearest", prefilter=False)
            norm = np.clip((resized - (-1350.0)) / (150.0 - (-1350.0)), 0.0, 1.0).astype(np.float32)
            cached = norm[np.newaxis, ...]  # (1, 64, 64, 64)
            cls._cache[key] = cached
        return cached

    def _aug_single(self, vol: np.ndarray, force_twin: bool = False) -> np.ndarray:
        out = vol.copy()
        if force_twin or self.rng.random() < 0.7:
            ang = float(self.rng.uniform(-8.0, 8.0))
            out[0] = rotate(out[0], angle=ang, axes=(1, 2), reshape=False, order=1, mode="nearest")
        if force_twin or self.rng.random() < 0.5:
            sh = self.rng.uniform(-2.0, 2.0, size=3)
            out[0] = nd_shift(out[0], shift=sh, order=1, mode="nearest")
        scale = float(self.rng.uniform(0.95, 1.05))
        bias = float(self.rng.uniform(-0.02, 0.02))
        noise = self.rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
        return np.clip(out * scale + bias + noise, 0.0, 1.0).astype(np.float32)

    def _aug_pair(self, pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1, p2 = pre.copy(), post.copy()
        if self.rng.random() < 0.6:
            ang = float(self.rng.uniform(-8.0, 8.0))
            p1[0] = rotate(p1[0], angle=ang, axes=(1, 2), reshape=False, order=1, mode="nearest")
            p2[0] = rotate(p2[0], angle=ang, axes=(1, 2), reshape=False, order=1, mode="nearest")
        scale = float(self.rng.uniform(0.95, 1.05))
        bias = float(self.rng.uniform(-0.02, 0.02))
        p1 = np.clip(p1 * scale + bias + self.rng.normal(0.0, 0.008, p1.shape), 0.0, 1.0).astype(np.float32)
        p2 = np.clip(p2 * scale + bias + self.rng.normal(0.0, 0.008, p2.shape), 0.0, 1.0).astype(np.float32)
        return p1, p2

    def __getitem__(self, idx: int):
        b_ix = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        bdf = self.df.iloc[b_ix]
        bsz = len(bdf)
        y = bdf["Binary_Label"].astype(int).to_numpy()
        y_cat = to_categorical(y, num_classes=2).astype(np.float32)

        if is_single_input_model(self.model_name):
            X = np.empty((bsz, 1, 64, 64, 64), dtype=np.float32)
            use_pre = self.model_name.startswith("Pre_")
            for i, row in enumerate(bdf.itertuples()):
                path = row.Pre_path if use_pre else row.Post_path
                v = self._load(path)
                if self.augment:
                    v = self._aug_single(v)
                X[i] = v
            return X, y_cat

        X_a = np.empty((bsz, 1, 64, 64, 64), dtype=np.float32)
        X_b = np.empty((bsz, 1, 64, 64, 64), dtype=np.float32)
        for i, row in enumerate(bdf.itertuples()):
            if self.model_name == "STRANet_Lite_Pre":
                va = self._load(row.Pre_path)
                p_pseudo = getattr(row, "Pre_pseudo_path", "")
                if isinstance(p_pseudo, str) and p_pseudo and Path(p_pseudo).exists():
                    vb = self._load(p_pseudo)
                else:
                    vb = self._aug_single(va, force_twin=True)
                if self.augment:
                    va, vb = self._aug_pair(va, vb)
            elif self.model_name == "STRANet_Lite_Post":
                va = self._load(row.Post_path)
                p_pseudo = getattr(row, "Post_pseudo_path", "")
                if isinstance(p_pseudo, str) and p_pseudo and Path(p_pseudo).exists():
                    vb = self._load(p_pseudo)
                else:
                    vb = self._aug_single(va, force_twin=True)
                if self.augment:
                    va, vb = self._aug_pair(va, vb)
            else:
                va = self._load(row.Pre_path)
                vb = self._load(row.Post_path)
                if self.augment:
                    va, vb = self._aug_pair(va, vb)
            X_a[i] = va
            X_b[i] = vb
        return [X_a, X_b], y_cat


# ==============================================================================
# SOTA Model Definitions (Exact Architectures from b4_1_utils_siamese_cv_5_fold_argu.py)
# ==============================================================================


def _build_base_cnn_encoder(input_shape=(1, 64, 64, 64), weight_decay=1e-4, name="base_3d_cnn") -> Model:
    inp = Input(shape=input_shape)
    x = Conv3D(32, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(inp)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)

    x = Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)

    x = Conv3D(128, (3, 3, 3), padding="same", activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)
    return Model(inputs=inp, outputs=x, name=name)


def _resnet_block_3d(x, filters: int, kernel_size: int = 3, stride: int = 1, weight_decay: float = 1e-4):
    shortcut = x
    x = Conv3D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv3D(
        filters,
        kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = BatchNormalization(axis=1)(x)

    if shortcut.shape[1] != filters or stride != 1:
        shortcut = Conv3D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
        )(shortcut)
        shortcut = BatchNormalization(axis=1)(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def _build_3d_resnet18_encoder(input_shape=(1, 64, 64, 64), weight_decay=1e-4, name="base_3d_resnet18") -> Model:
    inp = Input(shape=input_shape)
    x = Conv3D(64, (7, 7, 7), strides=2, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(inp)
    x = BatchNormalization(axis=1)(x)
    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=2, padding="same")(x)

    # Stage 1 (64 filters, 2 blocks)
    x = _resnet_block_3d(x, 64, stride=1, weight_decay=weight_decay)
    x = _resnet_block_3d(x, 64, stride=1, weight_decay=weight_decay)
    # Stage 2 (128 filters, 2 blocks)
    x = _resnet_block_3d(x, 128, stride=2, weight_decay=weight_decay)
    x = _resnet_block_3d(x, 128, stride=1, weight_decay=weight_decay)
    # Stage 3 (256 filters, 2 blocks)
    x = _resnet_block_3d(x, 256, stride=2, weight_decay=weight_decay)
    x = _resnet_block_3d(x, 256, stride=1, weight_decay=weight_decay)
    # Stage 4 (512 filters, 2 blocks)
    x = _resnet_block_3d(x, 512, stride=2, weight_decay=weight_decay)
    x = _resnet_block_3d(x, 512, stride=1, weight_decay=weight_decay)

    x = GlobalAveragePooling3D()(x)
    return Model(inputs=inp, outputs=x, name=name)


def _dense_block_3d(x, num_layers: int, growth_rate: int, weight_decay: float):
    for _ in range(num_layers):
        conv = Conv3D(
            growth_rate,
            (3, 3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay),
        )(x)
        conv = BatchNormalization(axis=1)(conv)
        x = Concatenate(axis=1)([x, conv])
    return x


def _transition_layer_3d(x, reduction_factor: float, weight_decay: float):
    num_filters = int(x.shape[1] * reduction_factor)
    x = Conv3D(
        num_filters,
        (1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)
    return x


def _build_3d_densenet_encoder(
    input_shape=(1, 64, 64, 64),
    weight_decay=1e-4,
    growth_rate=32,
    num_layers_per_block=2,
    reduction_factor=0.5,
    name="base_3d_densenet",
) -> Model:
    inp = Input(shape=input_shape)
    x = Conv3D(64, (7, 7, 7), padding="same", activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(inp)
    x = BatchNormalization(axis=1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(x)

    for _ in range(2):
        x = _dense_block_3d(x, num_layers=num_layers_per_block, growth_rate=growth_rate, weight_decay=weight_decay)
        x = _transition_layer_3d(x, reduction_factor=reduction_factor, weight_decay=weight_decay)

    x = GlobalAveragePooling3D()(x)
    return Model(inputs=inp, outputs=x, name=name)


def build_sota_comparator_model(
    model_name: str,
    input_size=(64, 64, 64),
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.30,
    class_num: int = 2,
) -> Model:
    shape = (1, input_size[0], input_size[1], input_size[2])

    if model_name in ("Pre_CNN", "Post_CNN"):
        inp = Input(shape=shape, name="input_ct")
        base = _build_base_cnn_encoder(shape, weight_decay=weight_decay, name=f"{model_name}_enc")
        x = base(inp)
        x = Dropout(dropout_rate)(x)
        x = GlobalAveragePooling3D()(x)
        out = Dense(class_num, activation="softmax", dtype="float32", name="output_layer")(x)
        return Model(inputs=inp, outputs=out, name=model_name)

    if model_name in ("Pre_ResNet", "Post_ResNet"):
        inp = Input(shape=shape, name="input_ct")
        base = _build_3D_resnet_single(shape, weight_decay=weight_decay, dropout_rate=dropout_rate, class_num=class_num, name=model_name)
        return base

    if model_name == "ResNet_Longitudinal":
        inp_a = Input(shape=shape, name="input_a")
        inp_b = Input(shape=shape, name="input_b")
        base = _build_3d_resnet18_encoder(shape, weight_decay=weight_decay, name="shared_resnet18")
        fa = base(inp_a)
        fb = base(inp_b)
        merged = Concatenate(axis=1, name="concat")([fa, fb])
        x = Dense(128, activation="relu", name="dense_128")(merged)
        x = Dropout(dropout_rate, name="dropout")(x)
        out = Dense(class_num, activation="softmax", dtype="float32", name="output_layer")(x)
        return Model(inputs=[inp_a, inp_b], outputs=out, name="ResNet_Longitudinal")

    if model_name in ("Pre_DenseNet", "Post_DenseNet"):
        inp = Input(shape=shape, name="input_ct")
        base = _build_3d_densenet_encoder(shape, weight_decay=weight_decay, name=f"{model_name}_enc")
        feat = base(inp)
        x = Dense(128, activation="relu")(feat)
        x = Dropout(dropout_rate)(x)
        out = Dense(class_num, activation="softmax", dtype="float32", name="output_layer")(x)
        return Model(inputs=inp, outputs=out, name=model_name)

    if model_name == "DenseNet_Longitudinal":
        inp_a = Input(shape=shape, name="input_a")
        inp_b = Input(shape=shape, name="input_b")
        base = _build_3d_densenet_encoder(shape, weight_decay=weight_decay, name="shared_densenet")
        fa = base(inp_a)
        fb = base(inp_b)
        merged = Concatenate(axis=1, name="concat")([fa, fb])
        x = Dense(128, activation="relu", name="dense_128")(merged)
        x = Dropout(dropout_rate, name="dropout")(x)
        out = Dense(class_num, activation="softmax", dtype="float32", name="output_layer")(x)
        return Model(inputs=[inp_a, inp_b], outputs=out, name="DenseNet_Longitudinal")

    if model_name in ("STRANet_Lite_Pre", "STRANet_Lite_Post"):
        inp_a = Input(shape=shape, name="input_a")
        inp_b = Input(shape=shape, name="input_b")
        base = _build_base_cnn_encoder(shape, weight_decay=weight_decay, name=f"{model_name}_shared_cnn")
        fa = base(inp_a)
        fb = base(inp_b)
        merged = Concatenate(axis=1, name="concat")([fa, fb])
        x = GlobalAveragePooling3D(name="global_avg_pool")(merged)
        x = Dense(128, activation="relu", name="dense_128")(x)
        x = Dropout(dropout_rate, name="dropout")(x)
        out = Dense(class_num, activation="softmax", dtype="float32", name="output_layer")(x)
        return Model(inputs=[inp_a, inp_b], outputs=out, name=model_name)

    raise ValueError(f"Unknown SOTA model: {model_name}")


def _build_3D_resnet_single(shape, weight_decay, dropout_rate, class_num, name) -> Model:
    inp = Input(shape=shape, name="input_ct")
    base = _build_3d_resnet18_encoder(shape, weight_decay=weight_decay, name=f"{name}_enc")
    feat = base(inp)
    x = Dense(128, activation="relu")(feat)
    x = Dropout(dropout_rate)(x)
    out = Dense(class_num, activation="softmax", dtype="float32", name="output_layer")(x)
    return Model(inputs=inp, outputs=out, name=name)


class SOTAValAUCCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_df: pd.DataFrame, train_eval_gen: SOTABenchmarkGenerator, val_df: pd.DataFrame, val_gen: SOTABenchmarkGenerator):
        super().__init__()
        self.train_y = train_df["Binary_Label"].astype(int).to_numpy()
        self.train_eval_gen = train_eval_gen
        self.val_y = val_df["Binary_Label"].astype(int).to_numpy()
        self.val_gen = val_gen

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs if logs is not None else {}
        tr_p = self.model.predict(self.train_eval_gen, verbose=0)[:, 1][: len(self.train_y)]
        va_p = self.model.predict(self.val_gen, verbose=0)[:, 1][: len(self.val_y)]
        tr_auc = float(roc_auc_score(self.train_y, tr_p))
        va_auc = float(roc_auc_score(self.val_y, va_p))
        gap_pen = max(0.0, tr_auc - va_auc - 0.10)
        guarded = va_auc - 0.20 * gap_pen
        logs["train_unaug_auc"] = tr_auc
        logs["val_patient_auc"] = va_auc
        logs["val_guarded_auc"] = guarded
        print(
            f"\nEpoch {epoch + 1}: train_auc={tr_auc:.4f} | val_auc={va_auc:.4f} | guarded={guarded:.4f}",
            flush=True,
        )


def predict_sota_df(model: Model, df: pd.DataFrame, model_name: str, seed: int) -> np.ndarray:
    gen = SOTABenchmarkGenerator(df, model_name=model_name, batch_size=8, balanced=False, augment=False, seed=seed)
    return model.predict(gen, verbose=0)[:, 1][: len(df)].astype(np.float64)


def fit_and_eval_sota_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_dfs: dict[str, pd.DataFrame],
    out_dir: Path,
    model_name: str,
    epochs: int = 45,
    lr: float = 8e-5,
    seed: int = 2026,
) -> tuple[dict[str, np.ndarray], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)

    train_gen = SOTABenchmarkGenerator(train_df, model_name=model_name, batch_size=8, balanced=True, augment=True, seed=seed)
    train_eval_gen = SOTABenchmarkGenerator(train_df, model_name=model_name, batch_size=8, balanced=False, augment=False, seed=seed)
    val_gen = SOTABenchmarkGenerator(val_df, model_name=model_name, batch_size=8, balanced=False, augment=False, seed=seed)

    model = build_sota_comparator_model(model_name, weight_decay=1e-4, dropout_rate=0.30, class_num=2)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03)
    model.compile(loss=loss_fn, optimizer=Adam(learning_rate=lr), metrics=["accuracy"])
    weights = out_dir / "best_model.weights.h5"

    callbacks = [
        SOTAValAUCCallback(train_df, train_eval_gen, val_df, val_gen),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_guarded_auc", mode="max", factor=0.5, patience=6, min_lr=5e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(weights), monitor="val_guarded_auc", mode="max", save_best_only=True, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_guarded_auc", mode="max", patience=14, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=1)
    peak_tr = float(max(hist.history.get("train_unaug_auc", [0.5])))
    if weights.exists():
        model.load_weights(str(weights))
        try:
            weights.unlink()  # Free disk space after loading weights for prediction
        except OSError:
            pass
    preds = {k: predict_sota_df(model, v, model_name=model_name, seed=seed) for k, v in eval_dfs.items()}
    tf.keras.backend.clear_session()
    return preds, peak_tr


def profile_all_model_complexities():
    """Profile FLOPs (G), Parameters, Memory (MB), Inference Time (s), and FPS for S4 Table."""
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    rows = []
    profile_list = list(ALL_SOTA_MODELS) + ["STRANet_Spatial_Delta", "STRANet_Dual_Window"]

    for mname in profile_list:
        tf.keras.backend.clear_session()
        mixed_precision.set_global_policy("float32")
        if mname == "STRANet_Spatial_Delta":
            model = build_siamese_nsclc545_model("spatial_delta_maxavg", n_channels=1)
            dummy = [np.zeros((1, 1, 64, 64, 64), dtype=np.float32), np.zeros((1, 1, 64, 64, 64), dtype=np.float32)]
        elif mname == "STRANet_Dual_Window":
            model = build_siamese_nsclc545_model("dual_window_spatial_siamese", n_channels=2)
            dummy = [np.zeros((1, 2, 64, 64, 64), dtype=np.float32), np.zeros((1, 2, 64, 64, 64), dtype=np.float32)]
        else:
            model = build_sota_comparator_model(mname)
            if is_single_input_model(mname):
                dummy = np.zeros((1, 1, 64, 64, 64), dtype=np.float32)
            else:
                dummy = [np.zeros((1, 1, 64, 64, 64), dtype=np.float32), np.zeros((1, 1, 64, 64, 64), dtype=np.float32)]

        params = int(model.count_params())

        # Compute exact FLOPs via TF graph profiler
        try:
            if isinstance(dummy, list):
                sig = [tf.TensorSpec(shape=d.shape, dtype=tf.float32) for d in dummy]
                concrete = tf.function(lambda a, b: model([a, b], training=False)).get_concrete_function(*sig)
            else:
                sig = [tf.TensorSpec(shape=dummy.shape, dtype=tf.float32)]
                concrete = tf.function(lambda a: model(a, training=False)).get_concrete_function(*sig)
            frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(graph_def, name="")
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops_info = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
                flops_g = float(flops_info.total_float_ops) / 1e9
        except Exception:
            flops_g = float("nan")

        # Estimate activation + parameter memory (MB, batch=1 float32)
        act_elems = 0
        for layer in model.layers:
            try:
                oshape = layer.output_shape
                if isinstance(oshape, list):
                    oshape = oshape[0]
                if isinstance(oshape, tuple):
                    dims = [d for d in oshape if d is not None]
                    act_elems += int(np.prod(dims))
            except Exception:
                pass
        mem_mb = float((params * 4 + act_elems * 4) / (1024.0 * 1024.0))

        # Warmup and time 30 single-patient forward passes
        for _ in range(5):
            _ = model(dummy, training=False)
        t0 = time.perf_counter()
        n_iters = 30
        for _ in range(n_iters):
            _ = model(dummy, training=False)
        elapsed = (time.perf_counter() - t0) / n_iters
        fps = 1.0 / max(elapsed, 1e-6)

        rows.append(
            {
                "Model": mname,
                "FLOPs_G": round(flops_g, 2),
                "Parameters": params,
                "Parameters_M": round(params / 1e6, 3),
                "Memory_Consumption_MB": round(mem_mb, 2),
                "Inference_Time_s": round(elapsed, 4),
                "FPS": round(fps, 2),
            }
        )
        print(f"[Profile] {mname}: FLOPs={flops_g:.2f}G | Params={params:,} | Mem={mem_mb:.1f}MB | Time={elapsed:.4f}s | FPS={fps:.1f}", flush=True)

    df_prof = pd.DataFrame(rows)
    df_prof.to_csv(SOTA_OUT / "S4_Hardware_Complexity_Profile.csv", index=False)
    return df_prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-design",
        choices=(
            "representative_holdout_545",
            "source_stage_holdout_545",
            "pure_stage_holdout_545",
            "source_stage_5fold_cv_545",
        ),
        default="representative_holdout_545",
    )
    parser.add_argument("--model-name", choices=ALL_SOTA_MODELS, default="Pre_CNN")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=40100)
    parser.add_argument("--profile-complexity", action="store_true")
    args = parser.parse_args()

    if args.profile_complexity:
        profile_all_model_complexities()
        return

    out_dir = DL_OUT / args.split_design / f"{args.model_name}_seed{args.seed}"
    if (out_dir / "run_summary.json").exists():
        print(f"Skipping completed SOTA run: {out_dir}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    split_root = DATA_545 / args.split_design

    if args.split_design.endswith("holdout_545"):
        tr_df = pd.read_csv(split_root / "train.csv", encoding="utf-8-sig")
        va_df = pd.read_csv(split_root / "validation.csv", encoding="utf-8-sig")
        te_df = pd.read_csv(split_root / "test.csv", encoding="utf-8-sig")

        preds, peak_tr = fit_and_eval_sota_one(
            tr_df,
            va_df,
            {"train": tr_df, "validation": va_df, "test": te_df},
            out_dir=out_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )
        thr = youden_threshold(va_df["Binary_Label"].to_numpy(), preds["validation"])

        for name, df_part in (("train", tr_df), ("validation", va_df), ("test", te_df)):
            out_df = df_part.copy()
            out_df["Pred_Prob"] = preds[name]
            if name in ("validation", "test"):
                for k, v in fit_clinical_comparators(tr_df, df_part).items():
                    out_df[k] = v
            out_df.to_csv(out_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
            if name == "test":
                te_comp = out_df
            elif name == "validation":
                va_comp = out_df

        val_s4 = compute_full_s3_s4_metrics(va_df["Binary_Label"].to_numpy(), preds["validation"], threshold=thr, seed=args.seed)
        test_s4 = compute_full_s3_s4_metrics(te_df["Binary_Label"].to_numpy(), preds["test"], threshold=thr, seed=args.seed)

        summary = {
            "cohort": "NSCLC_545_excl_SCLC",
            "split_design": args.split_design,
            "model_name": args.model_name,
            "seed": args.seed,
            "train_auroc_selected": float(roc_auc_score(tr_df["Binary_Label"], preds["train"])),
            "peak_train_auroc": peak_tr,
            "validation_s4_metrics": val_s4,
            "test_s4_metrics": test_s4,
            "validation_summary": summarize_predictions(va_comp, f"{args.model_name}_val", args.seed),
            "test_summary": summarize_predictions(te_comp, f"{args.model_name}_test", args.seed),
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[SOTA Holdout] {args.split_design} | {args.model_name}: "
            f"Train={summary['train_auroc_selected']:.4f} | Val={val_s4['AUROC']:.4f} | Test={test_s4['AUROC_Formatted']}",
            flush=True,
        )

    else:
        oof_parts = []
        fold_rows = []
        for fold in range(1, 6):
            fseed = args.seed + fold
            fdir = split_root / f"fold_{fold}"
            tr_df = pd.read_csv(fdir / "train_split.csv", encoding="utf-8-sig")
            va_df = pd.read_csv(fdir / "val_split.csv", encoding="utf-8-sig")

            preds, peak_tr = fit_and_eval_sota_one(
                tr_df,
                va_df,
                {"train": tr_df, "validation": va_df},
                out_dir=out_dir / f"fold_{fold}",
                model_name=args.model_name,
                epochs=args.epochs,
                lr=args.lr,
                seed=fseed,
            )
            thr_tr = youden_threshold(tr_df["Binary_Label"].to_numpy(), preds["train"])
            va_out = va_df.copy()
            va_out["Fold"] = fold
            va_out["Seed"] = fseed
            va_out["Pred_Prob"] = preds["validation"]
            for k, v in fit_clinical_comparators(tr_df, va_df).items():
                va_out[k] = v
            oof_parts.append(va_out)

            fm = compute_full_s3_s4_metrics(va_df["Binary_Label"].to_numpy(), preds["validation"], threshold=thr_tr, n_boot=500, seed=fseed)
            fold_rows.append(
                {
                    "Fold": fold,
                    "Seed": fseed,
                    "Train_AUROC_Selected": float(roc_auc_score(tr_df["Binary_Label"], preds["train"])),
                    "Peak_Train_AUROC": peak_tr,
                    "Val_AUROC": fm["AUROC"],
                    **fm,
                }
            )

        oof_df = pd.concat(oof_parts, ignore_index=True)
        oof_df.to_csv(out_dir / "pooled_oof_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(out_dir / "fold_performance.csv", index=False)

        oof_thr = float(np.median([r["Threshold"] for r in fold_rows]))
        oof_s4 = compute_full_s3_s4_metrics(oof_df["Binary_Label"].to_numpy(), oof_df["Pred_Prob"].to_numpy(), threshold=oof_thr, seed=args.seed)

        summary = {
            "cohort": "NSCLC_545_excl_SCLC",
            "split_design": args.split_design,
            "model_name": args.model_name,
            "seed": args.seed,
            "mean_train_auroc_selected": float(np.mean([r["Train_AUROC_Selected"] for r in fold_rows])),
            "max_peak_train_auroc": float(np.max([r["Peak_Train_AUROC"] for r in fold_rows])),
            "fold_val_aurocs": [r["Val_AUROC"] for r in fold_rows],
            "oof_s4_metrics": oof_s4,
            "oof_summary": summarize_predictions(oof_df, f"{args.model_name}_oof", args.seed),
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"[SOTA 5-Fold CV] {args.split_design} | {args.model_name}: "
            f"MeanTrain={summary['mean_train_auroc_selected']:.4f} | OOF={oof_s4['AUROC_Formatted']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
