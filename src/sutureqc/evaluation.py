"""Ordinal-scoring and detection metrics with grouped bootstrap CIs.

The realistic target (Section 5 of the plan) is **>= 80% within-one accuracy**,
not 80-90% exact five-class accuracy.  Exact accuracy is reported but is never
the only metric.  Human inter-rater agreement is the ceiling.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from . import config

_SCORE_LABELS = list(range(config.SCORE_MIN, config.SCORE_MAX + 1))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(y_true, y_pred)
    return float(rho)


def compute_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_cont: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """Compute every required ordinal metric for one set of predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return {"label": label, "n": 0}

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    metrics = {
        "label": label,
        "n": int(y_true.size),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors.astype(float) ** 2))),
        "exact_accuracy": float(np.mean(errors == 0)),
        "within_one_accuracy": float(np.mean(abs_errors <= 1)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "linear_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "spearman": _spearman(y_true, y_pred),
        "mean_signed_error": float(np.mean(errors)),
    }
    if y_pred_cont is not None and np.asarray(y_pred_cont).size == y_true.size:
        diff = np.asarray(y_pred_cont, dtype=float) - y_true
        metrics["mae_continuous"] = float(np.mean(np.abs(diff)))
        metrics["rmse_continuous"] = float(np.sqrt(np.mean(diff ** 2)))
    return metrics


def confusion_and_error_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(confusion_matrix, ordinal_error_matrix)`` over scores 1-5."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=_SCORE_LABELS)
    err = np.zeros((len(_SCORE_LABELS), len(_SCORE_LABELS)), dtype=int)
    for i, true_score in enumerate(_SCORE_LABELS):
        for j, pred_score in enumerate(_SCORE_LABELS):
            err[i, j] = abs(pred_score - true_score)
    # Weight the ordinal error matrix by how often each (true,pred) occurred.
    err_weighted = err * cm
    return cm, err_weighted


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=_SCORE_LABELS, zero_division=0
    )
    return {
        str(score): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, score in enumerate(_SCORE_LABELS)
    }


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    base = compute_ordinal_metrics(y_true, y_pred)
    return {
        "mae": base["mae"],
        "exact_accuracy": base["exact_accuracy"],
        "within_one_accuracy": base["within_one_accuracy"],
        "quadratic_weighted_kappa": base["quadratic_weighted_kappa"],
        "spearman": base["spearman"],
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = config.SEED,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Grouped bootstrap confidence intervals.

    Resamples whole *groups* (episodes) with replacement so that correlated
    stitches from the same episode stay together.  Returns
    ``{metric: [lower, upper]}``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    groups = np.asarray(groups, dtype=object)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return {}

    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in ("mae", "exact_accuracy", "within_one_accuracy", "quadratic_weighted_kappa", "spearman")}

    for _ in range(n_boot):
        drawn = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        idx = np.concatenate([index_by_group[g] for g in drawn])
        bundle = _metric_bundle(y_true[idx], y_pred[idx], groups[idx])
        for key, value in bundle.items():
            samples[key].append(value)

    intervals: dict[str, list[float]] = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        intervals[key] = [float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))]
    return intervals


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Binary stitch-presence precision/recall/F1 (used as a fallback)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
