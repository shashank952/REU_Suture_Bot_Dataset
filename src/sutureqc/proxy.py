"""Geometric-consistency proxy target (Section 3.4 fallback).

Because SutureBot2 has no expert 1-5 ratings, we derive a transparent,
geometry-only *geometric consistency score* so the technical pipeline can be
exercised end to end.  It is explicitly **not** clinical stitch quality and is
never mixed with expert labels in one result table.

Pipeline:

1. :func:`proxy_components` -> four interpretable sub-scores in ``[0, 1]``.
2. :func:`proxy_continuous` -> weighted mean of the available sub-scores.
3. :func:`fit_proxy_thresholds` -> quantile edges fitted on the TRAIN fold.
4. :func:`apply_proxy_thresholds` -> ordinal 1-5 label.
5. :func:`proxy_evidence` -> short natural-language explanation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_COMPONENT_COLUMNS = ("s_containment", "s_proximity", "s_spacing", "s_insertion_exit")


def proxy_components(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return the four interpretable proxy sub-scores for each stitch."""
    out = pd.DataFrame(index=features_df.index)
    overlap = features_df["bbox_wound_overlap_ratio"].to_numpy(dtype=float)
    signed = features_df["dist_center_to_wound_boundary_norm"].to_numpy(dtype=float)
    spacing = features_df["spacing_cv"].to_numpy(dtype=float)
    ie_mid = features_df["ie_mid_dist_norm"].to_numpy(dtype=float)
    ie_cross = features_df["ie_crossing_score"].to_numpy(dtype=float)

    with np.errstate(all="ignore"):
        out["s_containment"] = np.clip(overlap / 0.5, 0.0, 1.0)
        # Inside the wound -> signed<=0 -> full proximity; decays outside.
        out["s_proximity"] = np.exp(-np.clip(signed, 0.0, None) / 0.5)
        out["s_spacing"] = np.where(np.isfinite(spacing), np.exp(-spacing), np.nan)
        ie_available = np.isfinite(ie_mid) & np.isfinite(ie_cross)
        out["s_insertion_exit"] = np.where(
            ie_available,
            0.5 * np.exp(-ie_mid / 0.5) + 0.5 * ie_cross,
            np.nan,
        )
    return out


def proxy_continuous(features_df: pd.DataFrame) -> pd.Series:
    """Weighted mean of available sub-scores, renormalised over present weights."""
    components = proxy_components(features_df)
    weights = np.asarray([config.PROXY_WEIGHTS[c.removeprefix("s_")] for c in _COMPONENT_COLUMNS], dtype=float)
    values = components.loc[:, list(_COMPONENT_COLUMNS)].to_numpy(dtype=float)
    mask = np.isfinite(values)
    weight_matrix = mask * weights[None, :]
    denom = weight_matrix.sum(axis=1)
    numer = np.nansum(np.where(mask, values, 0.0) * weights[None, :], axis=1)
    with np.errstate(all="ignore"):
        score = np.where(denom > 0, numer / denom, 0.5)
    return pd.Series(score, index=features_df.index, name="proxy_score_continuous")


def fit_proxy_thresholds(train_continuous: np.ndarray | pd.Series) -> list[float]:
    """Fit quantile edges on the training fold only (no leakage)."""
    values = np.asarray(train_continuous, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float(q) for q in config.PROXY_QUANTILE_EDGES]
    edges = np.quantile(values, config.PROXY_QUANTILE_EDGES)
    # Guarantee strictly increasing edges to avoid empty bins.
    edges = np.maximum.accumulate(edges + np.arange(len(edges)) * 1e-9)
    return [float(e) for e in edges]


def apply_proxy_thresholds(continuous: np.ndarray | pd.Series, edges: list[float]) -> np.ndarray:
    """Bin a continuous proxy into ordinal labels 1..5 using *edges*."""
    values = np.asarray(continuous, dtype=float)
    labels = np.ones(values.shape, dtype=int)
    for edge in edges:
        labels += (values > edge).astype(int)
    return np.clip(labels, config.SCORE_MIN, config.SCORE_MAX)


def build_proxy_targets(
    features_df: pd.DataFrame,
    train_mask: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """Add ``proxy_score_continuous`` and ordinal ``proxy_score`` columns.

    If *train_mask* is given, thresholds are fitted on those rows only.
    Returns ``(augmented_df, thresholds)``.
    """
    out = features_df.copy()
    continuous = proxy_continuous(features_df)
    out["proxy_score_continuous"] = continuous
    components = proxy_components(features_df)
    out = pd.concat([out, components], axis=1)

    if train_mask is None:
        train_values = continuous.to_numpy(dtype=float)
    else:
        train_values = continuous.to_numpy(dtype=float)[np.asarray(train_mask, dtype=bool)]
    thresholds = fit_proxy_thresholds(train_values)
    out["proxy_score"] = apply_proxy_thresholds(continuous.to_numpy(dtype=float), thresholds)
    return out, thresholds


def proxy_evidence(features_row: pd.Series, component_row: pd.Series) -> str:
    """Return a short, human-readable justification for a proxy score."""
    phrases: list[str] = []

    overlap = features_row.get("bbox_wound_overlap_ratio", float("nan"))
    if np.isfinite(overlap):
        if overlap >= 0.5:
            phrases.append("stitch bbox largely overlaps the wound")
        elif overlap >= 0.2:
            phrases.append("stitch bbox partially overlaps the wound")
        else:
            phrases.append("stitch bbox barely overlaps the wound")

    signed = features_row.get("dist_center_to_wound_boundary_norm", float("nan"))
    if np.isfinite(signed):
        if signed <= 0:
            phrases.append("stitch centre lies inside the wound")
        elif signed <= 0.5:
            phrases.append("stitch centre sits just outside the wound")
        else:
            phrases.append("stitch centre is far from the wound")

    cv = features_row.get("spacing_cv", float("nan"))
    if np.isfinite(cv):
        if cv <= 0.25:
            phrases.append("spacing between stitches is regular")
        elif cv <= 0.6:
            phrases.append("spacing slightly irregular")
        else:
            phrases.append("spacing is inconsistent")

    crossing = features_row.get("ie_crossing_score", float("nan"))
    if np.isfinite(crossing):
        if crossing >= 0.6:
            phrases.append("insertion/exit cross the wound well")
        else:
            phrases.append("insertion/exit alignment is off")

    if not phrases:
        return "insufficient geometry to explain the score"
    return "; ".join(phrases)
