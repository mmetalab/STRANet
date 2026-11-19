"""
Standalone 3D Spatial-Delta Siamese Architecture for STRANet and STRANet-Lite.

Matches Supplementary Table S1 and Supplementary Figure S3 of the manuscript:
  - Shared 3-block 3D CNN encoder (32 -> 64 -> 128 filters) with GroupNorm (8 groups)
  - 3D Spatial-Delta Longitudinal Fusion:
      Concat[F_pre, F_post, F_post - F_pre, |F_post - F_pre|, F_pre * F_post]
      -> Conv3D(144, 1x1x1) + GroupNorm(8) + ReLU + SpatialDropout3D(0.15)
      -> Conv3D(144, 3x3x3) + GroupNorm(8) + ReLU
  - Dual Global Max-Average & Branch Pooling (672-D vector)
  - Classification Head: Dense(160, ReLU, Dropout 0.30) -> Dense(80, ReLU, Dropout 0.18) -> Dense(2, Softmax)
  - Total trainable parameters (1-channel input): 1,632,178
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def _group_norm(groups: int = 8, name: str = None):
    if hasattr(layers, "GroupNormalization"):
        return layers.GroupNormalization(groups=groups, name=name)
    return layers.LayerNormalization(name=name)


def build_shared_3d_encoder(
    input_shape=(64, 64, 64, 1),
    l2_reg: float = 3e-5,
    groups: int = 8,
    name: str = "shared_3d_encoder",
) -> tf.keras.Model:
    """Shared 3-block 3D convolutional encoder with Group Normalization."""
    reg = regularizers.l2(l2_reg)
    inp = layers.Input(shape=input_shape, name="encoder_input")

    # Block 1: 32@32x32x32
    x = layers.Conv3D(32, 3, padding="same", kernel_regularizer=reg, use_bias=False)(inp)
    x = _group_norm(groups)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv3D(32, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = _group_norm(groups)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 2: 64@16x16x16
    x = layers.Conv3D(64, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = _group_norm(groups)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv3D(64, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = _group_norm(groups)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 3: 128@8x8x8
    x = layers.Conv3D(128, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = _group_norm(groups)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv3D(128, 3, padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = _group_norm(groups)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.SpatialDropout3D(0.075)(x)

    return models.Model(inputs=inp, outputs=x, name=name)


def build_stranet(
    input_shape=(64, 64, 64, 1),
    l2_reg: float = 3e-5,
    groups: int = 8,
    num_classes: int = 2,
    name: str = "STRANet",
) -> tf.keras.Model:
    """
    Build STRANet (or STRANet-Lite when fed [original_ct, augmented_ct]).

    Args:
        input_shape: (64, 64, 64, 1) for single-window lung CT, or (64, 64, 64, 2)
                     for dual-window (lung [-1400, 200] HU + soft-tissue [-160, 240] HU).
        l2_reg: L2 kernel regularization strength (default 3e-5).
        groups: Number of groups for Group Normalization (default 8).
        num_classes: Number of output classes (default 2: Non-progression vs. Progression).
    """
    reg = regularizers.l2(l2_reg)
    pre_in = layers.Input(shape=input_shape, name="pre_ct")
    post_in = layers.Input(shape=input_shape, name="post_ct")

    encoder = build_shared_3d_encoder(input_shape=input_shape, l2_reg=l2_reg, groups=groups)
    f_pre = encoder(pre_in)    # 128@8x8x8
    f_post = encoder(post_in)  # 128@8x8x8

    # 3D Spatial-Delta Longitudinal Fusion (640@8x8x8 -> 144@8x8x8)
    f_diff = layers.Subtract(name="spatial_diff")([f_post, f_pre])
    f_abs = layers.Lambda(lambda t: tf.abs(t), name="spatial_abs_diff")(f_diff)
    f_prod = layers.Multiply(name="spatial_hadamard")([f_pre, f_post])
    fused = layers.Concatenate(axis=-1, name="spatial_delta_concat")(
        [f_pre, f_post, f_diff, f_abs, f_prod]
    )

    fused = layers.Conv3D(144, 1, padding="same", kernel_regularizer=reg, use_bias=False)(fused)
    fused = _group_norm(groups)(fused)
    fused = layers.Activation("relu")(fused)
    fused = layers.SpatialDropout3D(0.15)(fused)
    fused = layers.Conv3D(144, 3, padding="same", kernel_regularizer=reg, use_bias=False)(fused)
    fused = _group_norm(groups)(fused)
    fused = layers.Activation("relu")(fused)

    # Dual Global Max-Average & Branch Pooling (672-D)
    gap_fused = layers.GlobalAveragePooling3D(name="fused_gap")(fused)   # 144
    gmp_fused = layers.GlobalMaxPooling3D(name="fused_gmp")(fused)       # 144
    gap_pre = layers.GlobalAveragePooling3D(name="pre_gap")(f_pre)       # 128
    gap_post = layers.GlobalAveragePooling3D(name="post_gap")(f_post)    # 128
    gap_diff = layers.GlobalAveragePooling3D(name="diff_gap")(f_diff)    # 128
    feat_672 = layers.Concatenate(axis=-1, name="dual_pooling_672")(
        [gap_fused, gmp_fused, gap_pre, gap_post, gap_diff]
    )

    # Classification Head (160 -> 80 -> 2)
    h = layers.Dense(160, activation="relu", kernel_regularizer=reg, name="fc1_160")(feat_672)
    h = layers.Dropout(0.30, name="drop1")(h)
    h = layers.Dense(80, activation="relu", kernel_regularizer=reg, name="fc2_80")(h)
    h = layers.Dropout(0.18, name="drop2")(h)
    out = layers.Dense(num_classes, activation="softmax", name="output_softmax")(h)

    return models.Model(inputs=[pre_in, post_in], outputs=out, name=name)


if __name__ == "__main__":
    model = build_stranet(input_shape=(64, 64, 64, 1))
    model.summary()
    print(f"Total trainable parameters: {model.count_params():,}")
