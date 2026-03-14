# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Evaluation metrics for content moderation models.

Computes accuracy, F1 (macro/weighted/per-class), severity correlation,
and calibration metrics for ablation comparison.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    severity_pred: np.ndarray | None = None,
    severity_true: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute classification and severity metrics.

    Parameters
    ----------
    predictions : (N,) predicted class indices
    labels : (N,) ground truth class indices
    severity_pred : (N,) predicted severity scores (optional)
    severity_true : (N,) true severity scores (optional)
    label_names : list of class names for per-class reporting

    Returns
    -------
    dict of metric_name → value
    """
    # Only evaluate over classes that appear in labels or predictions
    active_classes = np.unique(np.concatenate([labels, predictions]))

    metrics: dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["macro_f1"] = f1_score(
        labels, predictions, average="macro", labels=active_classes, zero_division=0
    )
    metrics["weighted_f1"] = f1_score(
        labels, predictions, average="weighted", labels=active_classes, zero_division=0
    )

    # Per-class F1
    per_class_f1 = f1_score(
        labels, predictions, average=None, labels=active_classes, zero_division=0
    )
    for cls_idx, f1_val in zip(active_classes, per_class_f1):
        name = label_names[cls_idx] if label_names and cls_idx < len(label_names) else str(cls_idx)
        metrics[f"f1_{name}"] = float(f1_val)

    # Severity correlation
    if severity_pred is not None and severity_true is not None:
        # Filter out cases where severity_true has no variance
        if np.std(severity_true) > 1e-8:
            rho, _ = scipy_stats.spearmanr(severity_pred, severity_true)
            metrics["severity_spearman"] = float(rho) if not np.isnan(rho) else 0.0
        else:
            metrics["severity_spearman"] = 0.0

        metrics["severity_mse"] = float(np.mean((severity_pred - severity_true) ** 2))

    return metrics


def compute_severity_calibration(
    severity_pred: np.ndarray,
    severity_true: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray | float]:
    """Compute binned calibration curve and expected calibration error.

    Parameters
    ----------
    severity_pred : (N,) predicted severity [0, 1]
    severity_true : (N,) true severity [0, 1]
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict with:
        bin_edges : (n_bins+1,) bin boundaries
        bin_means_pred : (n_bins,) mean predicted severity per bin
        bin_means_true : (n_bins,) mean true severity per bin
        bin_counts : (n_bins,) samples per bin
        ece : float, expected calibration error
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means_pred = np.zeros(n_bins)
    bin_means_true = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (severity_pred >= lo) & (severity_pred <= hi)
        else:
            mask = (severity_pred >= lo) & (severity_pred < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_means_pred[i] = severity_pred[mask].mean()
            bin_means_true[i] = severity_true[mask].mean()

    # ECE: weighted average of |predicted - true| per bin
    total = bin_counts.sum()
    if total > 0:
        ece = float(np.sum(bin_counts * np.abs(bin_means_pred - bin_means_true)) / total)
    else:
        ece = 0.0

    return {
        "bin_edges": bin_edges,
        "bin_means_pred": bin_means_pred,
        "bin_means_true": bin_means_true,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def compute_ablation_comparison(
    results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compare metrics across ablation configurations.

    Parameters
    ----------
    results : dict mapping config name → metrics dict.
        E.g., {"flat_baseline": {...}, "hyperbolic_head": {...}, ...}

    Returns
    -------
    dict mapping config name → metrics with added "delta_*" keys
    showing improvement over the flat baseline.
    """
    baseline_key = "flat_baseline"
    baseline = results.get(baseline_key, {})

    comparison = {}
    for config_name, metrics in results.items():
        row = dict(metrics)
        if config_name != baseline_key and baseline:
            for key in ["accuracy", "macro_f1", "weighted_f1", "severity_spearman"]:
                if key in metrics and key in baseline:
                    row[f"delta_{key}"] = metrics[key] - baseline[key]
        comparison[config_name] = row

    return comparison
