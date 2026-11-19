#!/usr/bin/env python3
"""Harmonized CT pair loader and domain-balanced generators for STRANet.

Fixes three critical bugs identified in the 594-pair audit:
1. Anterior-Posterior (Y-axis) voxel inversion between SDCH_old (stored as LAS,
   affine[1,1] > 0, spine at y=0) and SDCH_new / TCIA (stored as LPS/LPI,
   affine[1,1] < 0, spine at y=63). Flipping axis 1 whenever affine[1,1] > 0
   places all 1,188 volumes into the exact same (Left, Posterior, Superior)
   anatomical voxel coordinate system while preserving the raw Inferior->Superior
   slice order (avoiding nib.as_closest_canonical Z-flip bugs on LPI headers).
2. 3D axial rotation augmentation axis bug: rotates in axes=(0, 1) (axial X-Y
   plane) instead of axes=(1, 2) (sagittal Y-Z plane).
3. Cohort-stratified / domain-balanced minibatch sampling: ensures positive and
   negative examples within each 4:4 minibatch have matched Cohort proportions
   when requested, preventing the network from using residual scanner/cohort
   differences as a shortcut for progression.
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import rotate, zoom
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv3D,
    Dense,
    Dropout,
    GlobalAveragePooling3D,
    Input,
    MaxPooling3D,
    Multiply,
    Subtract,
)
from tensorflow.keras.utils import Sequence, to_categorical

K.set_image_data_format("channels_first")


class HarmonizedDataGenerator(Sequence):
    """Load paired original pre/post CT volumes with harmonized LPS voxel orientation."""

    _normalized_volume_cache: dict[tuple[str, tuple[int, int, int], int], np.ndarray] = {}

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 8,
        dim: tuple[int, int, int] = (64, 64, 64),
        n_channels: int = 1,
        n_classes: int = 2,
        shuffle: bool = True,
        augment: bool = False,
        interp_order: int = 1,
        seed: int = 1234,
    ):
        self.dim = tuple(dim)
        self.batch_size = int(batch_size)
        self.df = df.reset_index(drop=True).copy()
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.interp_order = int(interp_order)
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index: int):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        return self._generate_batch(batch_df)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _resize(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {volume.shape}")
        factors = [a / b for a, b in zip(self.dim, volume.shape)]
        if self.interp_order == 1:
            return zoom(volume, factors, order=1, mode="nearest", prefilter=False)
        return zoom(volume, factors, order=self.interp_order, mode="nearest")

    @staticmethod
    def _normalize_hu(volume: np.ndarray) -> np.ndarray:
        window_level = -600
        window_width = 1600
        low = window_level - window_width // 2  # -1400
        high = window_level + window_width // 2  # 200
        clipped = np.clip(volume, low, high)
        return ((clipped - low) / (high - low)).astype(np.float32)

    def load_normalized(self, path: str) -> np.ndarray:
        cache_key = (str(path), self.dim, self.interp_order)
        cached = self._normalized_volume_cache.get(cache_key)
        if cached is None:
            nii = nib.load(str(path))
            volume = np.asarray(nii.dataobj, dtype=np.float32)
            # Harmonize X and Y to LPS while preserving raw Z slice order:
            # SDCH_old has affine[1,1] > 0 ('A'), so flip axis 1 to 'P'.
            orn = nib.aff2axcodes(nii.affine)
            if orn[0] == "R":
                volume = volume[::-1, :, :]
            if orn[1] == "A":
                volume = volume[:, ::-1, :]
            resized = self._resize(volume)
            cached = self._normalize_hu(resized)
            self._normalized_volume_cache[cache_key] = cached
        return cached.copy()

    def augment_pair(
        self, pre_volume: np.ndarray, post_volume: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronized training-only augmentation in the axial (X-Y) plane."""
        if self.rng.random() < 0.5:
            angle = float(self.rng.uniform(-10.0, 10.0))
            pre_volume = rotate(
                pre_volume,
                angle=angle,
                axes=(0, 1),
                reshape=False,
                order=1,
                mode="nearest",
            )
            post_volume = rotate(
                post_volume,
                angle=angle,
                axes=(0, 1),
                reshape=False,
                order=1,
                mode="nearest",
            )

        scale = float(self.rng.uniform(0.95, 1.05))
        shift = float(self.rng.uniform(-0.02, 0.02))
        pre_volume = pre_volume * scale + shift
        post_volume = post_volume * scale + shift
        pre_volume += self.rng.normal(0.0, 0.01, pre_volume.shape)
        post_volume += self.rng.normal(0.0, 0.01, post_volume.shape)
        return (
            np.clip(pre_volume, 0.0, 1.0).astype(np.float32),
            np.clip(post_volume, 0.0, 1.0).astype(np.float32),
        )

    def _generate_batch(self, batch_df: pd.DataFrame):
        n = len(batch_df)
        x_pre = np.empty((n, self.n_channels, *self.dim), dtype=np.float32)
        x_post = np.empty((n, self.n_channels, *self.dim), dtype=np.float32)
        y = np.empty((n,), dtype=int)

        for i, row in enumerate(batch_df.itertuples(index=False)):
            pre_norm = self.load_normalized(row.Pre_path)
            post_norm = self.load_normalized(row.Post_path)
            if self.augment:
                pre_norm, post_norm = self.augment_pair(pre_norm, post_norm)
            x_pre[i, 0] = pre_norm
            x_post[i, 0] = post_norm
            y[i] = int(row.Binary_Label)

        return [x_pre, x_post], to_categorical(y, num_classes=self.n_classes)


class HarmonizedBalancedDataGenerator(HarmonizedDataGenerator):
    """Label-balanced 4:4 minibatch generator with optional cohort/stratum balancing.

    When ``balance_by_cohort=True`` and ``Cohort`` is present in ``df``, positive
    and negative patients are sampled with matched cohort weights so that the
    minibatch label is orthogonal to cohort identity (preventing shortcut learning).
    """

    def __init__(self, df: pd.DataFrame, balance_by_cohort: bool = False, **kwargs):
        batch_size = int(kwargs.get("batch_size", 8))
        if batch_size % 2 != 0:
            raise ValueError("HarmonizedBalancedDataGenerator requires an even batch_size")
        labels = df["Binary_Label"].astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("HarmonizedBalancedDataGenerator requires both binary classes")
        self._balanced_labels = labels
        self.balance_by_cohort = bool(balance_by_cohort) and ("Cohort" in df.columns)
        self._sample_probs: dict[int, np.ndarray | None] = {0: None, 1: None}
        if self.balance_by_cohort:
            cohorts = df["Cohort"].astype(str).to_numpy()
            # Target cohort distribution is the marginal cohort proportion across df
            unique_cohorts, cohort_counts = np.unique(cohorts, return_counts=True)
            cohort_target = dict(zip(unique_cohorts, cohort_counts / len(df)))
            for label in (0, 1):
                mask = labels == label
                c_sub = cohorts[mask]
                weights = np.zeros(len(c_sub), dtype=float)
                for c_name, target_p in cohort_target.items():
                    c_mask = c_sub == c_name
                    if np.any(c_mask):
                        weights[c_mask] = target_p / np.sum(c_mask)
                weights /= weights.sum()
                self._sample_probs[label] = weights
        super().__init__(df, **kwargs)

    def on_epoch_end(self):
        half_batch = self.batch_size // 2
        steps = int(np.floor(len(self.df) / self.batch_size))
        positive = np.flatnonzero(self._balanced_labels == 1)
        negative = np.flatnonzero(self._balanced_labels == 0)
        batches = []
        for _ in range(steps):
            pos = self.rng.choice(
                positive,
                size=half_batch,
                replace=len(positive) < half_batch,
                p=self._sample_probs[1],
            )
            neg = self.rng.choice(
                negative,
                size=half_batch,
                replace=len(negative) < half_batch,
                p=self._sample_probs[0],
            )
            batch = np.concatenate((pos, neg))
            self.rng.shuffle(batch)
            batches.append(batch)
        self.indexes = np.concatenate(batches).astype(int, copy=False)

    def __len__(self) -> int:
        return int(np.floor(len(self.df) / self.batch_size))

    def batch_label_counts(self) -> list[dict[str, int]]:
        counts = []
        for index in range(len(self)):
            start = index * self.batch_size
            stop = start + self.batch_size
            labels = self._balanced_labels[self.indexes[start:stop]]
            counts.append(
                {
                    "Batch": index,
                    "Negative": int(np.sum(labels == 0)),
                    "Positive": int(np.sum(labels == 1)),
                }
            )
        return counts


class PatientAUROCCallback(tf.keras.callbacks.Callback):
    """Compute exact patient-level positive-class AUROC at epoch end."""

    def __init__(self, validation_generator: Sequence, metric_name: str = "val_patient_auc"):
        super().__init__()
        self.validation_generator = validation_generator
        self.metric_name = metric_name

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs if logs is not None else {}
        probabilities = []
        labels = []
        for x_batch, y_batch in self.validation_generator:
            preds = self.model.predict(x_batch, verbose=0)[:, 1]
            probabilities.extend(preds)
            labels.extend(np.argmax(y_batch, axis=1))
        labels_arr = np.asarray(labels, dtype=int)
        probs_arr = np.asarray(probabilities, dtype=float)
        auc_value = float(roc_auc_score(labels_arr, probs_arr))
        logs[self.metric_name] = auc_value
        print(f"\nEpoch {epoch + 1}: {self.metric_name}={auc_value:.6f}", flush=True)


def build_siamese_CNN_groupnorm_delta(
    input_size=(64, 64, 64),
    weight_decay: float = 1e-5,
    dropout_rate: float = 0.25,
    class_num: int = 2,
    fusion_mode: str = "signed_delta",
) -> Model:
    """GroupNorm small-batch 3D Siamese encoder with longitudinal feature fusion."""
    input_shape = (1, input_size[0], input_size[1], input_size[2])
    encoder_input = Input(shape=input_shape, name="encoder_input")
    x = encoder_input
    for block, filters in enumerate((32, 64, 128), start=1):
        x = Conv3D(
            filters,
            (3, 3, 3),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=f"conv_{block}",
        )(x)
        x = tf.keras.layers.GroupNormalization(
            groups=8, axis=1, name=f"group_norm_{block}"
        )(x)
        x = Activation("relu", name=f"relu_{block}")(x)
        x = MaxPooling3D(
            pool_size=(2, 2, 2), padding="same", name=f"pool_{block}"
        )(x)
    x = GlobalAveragePooling3D(name="encoder_pool")(x)
    encoder = Model(encoder_input, x, name="shared_groupnorm_encoder")

    input_pre = Input(shape=input_shape, name="input_pre")
    input_post = Input(shape=input_shape, name="input_post")
    feature_pre = encoder(input_pre)
    feature_post = encoder(input_post)
    delta = Subtract(name="post_minus_pre")([feature_post, feature_pre])
    delta_squared = Multiply(name="delta_squared")([delta, delta])
    interaction = Multiply(name="pre_post_interaction")([feature_pre, feature_post])
    if fusion_mode == "abs_and_signed_delta":
        abs_delta = tf.keras.layers.Lambda(lambda t: tf.abs(t), name="abs_delta")(delta)
        fused = Concatenate(name="longitudinal_fusion")(
            [feature_pre, feature_post, delta, abs_delta, delta_squared, interaction]
        )
    else:
        fused = Concatenate(name="longitudinal_fusion")(
            [feature_pre, feature_post, delta, delta_squared, interaction]
        )
    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay),
        name="dense_128",
    )(fused)
    x = Dropout(dropout_rate, name="dropout")(x)
    predictions = Dense(
        class_num, activation="softmax", dtype="float32", name="output_layer"
    )(x)
    return Model(
        inputs=[input_pre, input_post],
        outputs=predictions,
        name=f"siamese_groupnorm_{fusion_mode}",
    )
