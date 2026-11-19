#!/usr/bin/env python
"""Regenerate Grad-CAM examples from held-out confirmatory fold models.

Examples are selected deterministically as the median-score patient in each
confusion-matrix category.  Each patient is explained by the model for the
fold in which that patient was held out, avoiding reuse of a training image.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import zoom
from tensorflow.keras import mixed_precision

ROOT = Path("./data/Reviewer_R1_auc_rawpost_fix")
RESULTS = ROOT / "results" / "confirmatory_groupnorm_balanced_cv"
OUT = ROOT / "revision_figures" / "Figure-TIFF"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

# Rebuild the fold models under the same numerical policy used during frozen
# prediction generation. Otherwise float32 reconstruction can differ from the
# mixed-float16 inference path enough to fail the held-out score audit.
if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

from b4_1_utils_siamese_cv_5_fold_argu import (  # noqa: E402
    DataGenerator,
    build_siamese_CNN_groupnorm_delta,
)


def build_grad_model(model):
    """Expose the final shared-encoder activation for both time points."""
    encoder = model.get_layer("shared_groupnorm_encoder")
    activation_encoder = tf.keras.Model(
        encoder.input,
        [encoder.get_layer("relu_3").output, encoder.output],
        name="activation_encoder",
    )
    shape = (1, 64, 64, 64)
    input_pre = tf.keras.Input(shape=shape, name="gradcam_pre")
    input_post = tf.keras.Input(shape=shape, name="gradcam_post")
    map_pre, feature_pre = activation_encoder(input_pre)
    map_post, feature_post = activation_encoder(input_post)
    delta = model.get_layer("post_minus_pre")([feature_post, feature_pre])
    squared = model.get_layer("delta_squared")([delta, delta])
    interaction = model.get_layer("pre_post_interaction")([feature_pre, feature_post])
    fused = model.get_layer("longitudinal_fusion")(
        [feature_pre, feature_post, delta, squared, interaction]
    )
    dense = model.get_layer("dense_128")(fused)
    dropped = model.get_layer("dropout")(dense, training=False)
    prediction = model.get_layer("output_layer")(dropped)
    return tf.keras.Model(
        [input_pre, input_post], [map_pre, map_post, prediction], name="gradcam_model"
    )


def normalize_heatmap(heatmap):
    heatmap = np.maximum(np.asarray(heatmap, dtype=float), 0)
    maximum = float(np.max(heatmap))
    return heatmap / maximum if maximum > 0 else heatmap


def explain(model, inputs, target_class):
    grad_model = build_grad_model(model)
    with tf.GradientTape() as tape:
        pre_map, post_map, prediction = grad_model(inputs, training=False)
        target = prediction[:, target_class]
    grad_pre, grad_post = tape.gradient(target, [pre_map, post_map])

    def cam(feature_map, gradient):
        weights = tf.reduce_mean(gradient, axis=(2, 3, 4), keepdims=True)
        heat = tf.reduce_sum(weights * feature_map, axis=1)[0]
        return normalize_heatmap(heat.numpy())

    return cam(pre_map, grad_pre), cam(post_map, grad_post), prediction.numpy()[0]


def resize_heatmap(heatmap):
    factors = [64 / size for size in heatmap.shape]
    return normalize_heatmap(zoom(heatmap, factors, order=1))


predictions = pd.read_csv(RESULTS / "pooled_sdch_oof_predictions.csv", encoding="utf-8-sig")
# The exported pooled manifest is intentionally submission-safe and excludes
# internal image paths. Reconstruct the held-out path lookup from the five
# internal validation splits, which together contain each SDCH patient once.
manifest = pd.concat(
    [
        pd.read_csv(
            ROOT / "data" / "confirmatory_sdch_cv" / f"fold_{fold}" / "val_split.csv",
            encoding="utf-8-sig",
        )[["Sample_ID", "Pre_path", "Post_path"]]
        for fold in range(1, 6)
    ],
    ignore_index=True,
)
if len(manifest) != 465 or manifest["Sample_ID"].duplicated().any():
    raise RuntimeError("Held-out Grad-CAM path lookup must contain 465 unique SDCH patients")
predictions = predictions.merge(
    manifest[["Sample_ID", "Pre_path", "Post_path"]],
    on="Sample_ID",
    how="left",
    validate="one_to_one",
)
predictions["Category"] = np.select(
    [
        (predictions["Binary_Label"] == 1) & (predictions["Pred_Label"] == 1),
        (predictions["Binary_Label"] == 0) & (predictions["Pred_Label"] == 0),
        (predictions["Binary_Label"] == 0) & (predictions["Pred_Label"] == 1),
        (predictions["Binary_Label"] == 1) & (predictions["Pred_Label"] == 0),
    ],
    ["True positive", "True negative", "False positive", "False negative"],
    default="Unclassified",
)

selected = []
for category in ("True positive", "True negative", "False positive", "False negative"):
    candidates = predictions[predictions["Category"] == category].copy()
    if candidates.empty:
        raise RuntimeError(f"No patient available for {category}")
    median = candidates["Pred_Prob"].median()
    candidates["distance_to_category_median"] = np.abs(candidates["Pred_Prob"] - median)
    selected.append(candidates.sort_values(["distance_to_category_median", "Sample_ID"]).iloc[0])
selected = pd.DataFrame(selected)
selected.to_csv(RESULTS / "gradcam_selected_examples.csv", index=False, encoding="utf-8-sig")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, axes = plt.subplots(4, 4, figsize=(10.5, 10.3))
for row_index, row in selected.reset_index(drop=True).iterrows():
    fold = int(row["Fold"])
    model = build_siamese_CNN_groupnorm_delta((64, 64, 64), 1e-5, 0.25, 2)
    model.load_weights(str(RESULTS / f"fold_{fold}" / "best_model.weights.h5"))
    generator = DataGenerator(
        pd.DataFrame([row])[["Pre_path", "Post_path", "Binary_Label"]],
        dim=(64, 64, 64),
        batch_size=1,
        n_classes=2,
        n_channels=1,
        shuffle=False,
        augment=False,
        seed=20261103,
    )
    inputs, _ = generator[0]
    target_class = int(row["Pred_Label"])
    pre_heat, post_heat, model_probability = explain(model, inputs, target_class)
    if not np.isclose(
        float(model_probability[1]), float(row["Pred_Prob"]), atol=1e-4, rtol=1e-4
    ):
        raise RuntimeError(
            f"Fold {fold} rebuilt prediction {float(model_probability[1]):.8f} "
            f"does not match frozen OOF score {float(row['Pred_Prob']):.8f}"
        )
    pre_heat = resize_heatmap(pre_heat)
    post_heat = resize_heatmap(post_heat)
    combined = pre_heat + post_heat
    axial = int(np.argmax(np.sum(combined, axis=(0, 1))))
    pre_slice = np.rot90(inputs[0][0, 0, :, :, axial])
    post_slice = np.rot90(inputs[1][0, 0, :, :, axial])
    pre_cam_slice = np.rot90(pre_heat[:, :, axial])
    post_cam_slice = np.rot90(post_heat[:, :, axial])

    title = (
        f"{row.Category}\nlabel={int(row.Binary_Label)}, "
        f"p(progression)={float(model_probability[1]):.3f}, fold={fold}"
    )
    for column, (image, heat, timepoint) in enumerate(
        [
            (pre_slice, None, "Pre-RT CT"),
            (pre_slice, pre_cam_slice, "Pre-RT Grad-CAM"),
            (post_slice, None, "Post-RT CT"),
            (post_slice, post_cam_slice, "Post-RT Grad-CAM"),
        ]
    ):
        ax = axes[row_index, column]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        if heat is not None:
            ax.imshow(heat, cmap="jet", alpha=np.clip(heat * 0.65, 0, 0.65), vmin=0, vmax=1)
        ax.set_title(timepoint, fontsize=8.5, fontweight="bold")
        ax.axis("off")
    # ``axis("off")`` also suppresses axis labels, so draw the held-out
    # outcome/prediction audit label explicitly beside each row.
    axes[row_index, 0].text(
        -0.08,
        0.5,
        title,
        transform=axes[row_index, 0].transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=8.2,
        fontweight="bold",
        clip_on=False,
    )
    tf.keras.backend.clear_session()

fig.suptitle(
    "Held-out Grad-CAM audit of the corrected longitudinal model",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
fig.tight_layout(rect=(0.03, 0.02, 1, 0.97), h_pad=1.6, w_pad=0.4)
fig.savefig(OUT / "Fig 4_corrected_gradcam.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(
    OUT / "Fig 4_corrected_gradcam.tif",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
plt.close(fig)

