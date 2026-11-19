#!/usr/bin/env python
"""Paired AUROC inference using fast DeLong and patient-level bootstrap."""
from statistics import NormalDist

import numpy as np
from sklearn.metrics import roc_auc_score


def _midrank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    output = np.empty(len(values), dtype=float)
    output[order] = ranks + 1.0
    return output


def _as_covariance(matrix):
    covariance = np.cov(matrix)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]])
    return np.asarray(covariance, dtype=float)


def delong_auc_covariance(labels, predictions):
    """Return paired AUROCs and their DeLong covariance matrix.

    ``predictions`` must have shape (n_models, n_patients). Labels use 1 for
    progression and 0 for non-progression. The implementation follows the
    structural components and fast midrank formulation of DeLong inference.
    """
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if predictions.shape[1] != len(labels):
        raise ValueError("Prediction columns must match the number of labels")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary outcome classes are required")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite")

    order = np.argsort(-labels, kind="stable")
    sorted_predictions = predictions[:, order]
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    model_count = sorted_predictions.shape[0]
    positive = sorted_predictions[:, :positive_count]
    negative = sorted_predictions[:, positive_count:]

    positive_ranks = np.empty_like(positive, dtype=float)
    negative_ranks = np.empty_like(negative, dtype=float)
    total_ranks = np.empty_like(sorted_predictions, dtype=float)
    for model in range(model_count):
        positive_ranks[model] = _midrank(positive[model])
        negative_ranks[model] = _midrank(negative[model])
        total_ranks[model] = _midrank(sorted_predictions[model])

    aucs = (
        total_ranks[:, :positive_count].sum(axis=1)
        / (positive_count * negative_count)
        - (positive_count + 1.0) / (2.0 * negative_count)
    )
    v_positive = (
        total_ranks[:, :positive_count] - positive_ranks
    ) / negative_count
    v_negative = 1.0 - (
        total_ranks[:, positive_count:] - negative_ranks
    ) / positive_count
    covariance = (
        _as_covariance(v_positive) / positive_count
        + _as_covariance(v_negative) / negative_count
    )
    return aucs, covariance


def paired_delong(labels, first, second, confidence=0.95):
    """Compare two correlated AUROCs and return two-sided DeLong inference."""
    labels = np.asarray(labels, dtype=int)
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    aucs, covariance = delong_auc_covariance(labels, np.vstack((first, second)))
    contrast = np.asarray((1.0, -1.0))
    variance = float(contrast @ covariance @ contrast.T)
    difference = float(aucs[0] - aucs[1])
    if variance <= np.finfo(float).eps:
        z_score = 0.0 if np.isclose(difference, 0.0) else np.inf
        p_value = 1.0 if np.isclose(difference, 0.0) else 0.0
    else:
        standard_error = float(np.sqrt(variance))
        z_score = difference / standard_error
        p_value = float(2.0 * NormalDist().cdf(-abs(z_score)))
    alpha = 1.0 - confidence
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    return {
        "First_AUROC": float(aucs[0]),
        "Second_AUROC": float(aucs[1]),
        "Delta_first_minus_second": difference,
        "Delta_SE": standard_error,
        "Delta_CI_low": difference - critical * standard_error,
        "Delta_CI_high": difference + critical * standard_error,
        "Z": float(z_score),
        "P_value_two_sided": p_value,
        "Method": "paired DeLong",
    }


def paired_bootstrap(
    labels, first, second, confidence=0.95, resamples=5000, seed=20261103
):
    """Patient-level paired percentile bootstrap for an AUROC difference."""
    labels = np.asarray(labels, dtype=int)
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    observed = float(roc_auc_score(labels, first) - roc_auc_score(labels, second))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        index = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[index]).size == 2:
            differences.append(
                roc_auc_score(labels[index], first[index])
                - roc_auc_score(labels[index], second[index])
            )
    alpha = 1.0 - confidence
    low, high = np.percentile(differences, (100 * alpha / 2, 100 * (1 - alpha / 2)))
    return {
        "Delta_first_minus_second": observed,
        "Delta_CI_low": float(low),
        "Delta_CI_high": float(high),
        "Valid_resamples": len(differences),
        "Method": "paired patient-level percentile bootstrap",
    }
