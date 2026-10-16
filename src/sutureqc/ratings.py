"""Expert-rating rubric, annotation template and inter-rater agreement.

The expert 1-5 ratings do not yet exist for SutureBot2, so this module builds
the *blank* annotation template (Milestone 0 deliverable) and the machinery to
score agreement once raters fill it in.  The rubric in
:data:`config.RUBRIC` must be approved by a clinical collaborator before use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

RATER_COLUMNS: tuple[str, ...] = ("rater_1", "rater_2", "rater_3")


def rubric_dataframe() -> pd.DataFrame:
    """Return the rubric as a tidy DataFrame (score, operational meaning)."""
    return pd.DataFrame(
        [{"score": score, "operational_meaning": text} for score, text in sorted(config.RUBRIC.items())]
    )


def build_expert_rating_template(
    stitch_df: pd.DataFrame,
    crop_paths: pd.DataFrame | None = None,
    n_raters: int = 3,
) -> pd.DataFrame:
    """Build the blank annotation template: one row per stitch instance.

    Raters fill the ``rater_1..rater_n`` columns with an integer 1-5 or the
    sentinel :data:`config.NOT_SCORABLE_TOKEN`.  ``consensus`` / ``agreement``
    are left blank for the analysis step.
    """
    columns = [
        "stitch_id", "batch", "episode", "frame", "stitch_index",
        "x1", "y1", "x2", "y2", "crop_path",
        "rater_1", "rater_2", "rater_3", "consensus", "n_scorable_raters",
        "agreement", "not_scorable", "annotation_time_s", "notes",
    ]
    template = stitch_df[
        ["stitch_id", "batch", "episode", "frame", "stitch_index", "x1", "y1", "x2", "y2"]
    ].copy()
    if crop_paths is not None and not crop_paths.empty:
        template = template.merge(crop_paths[["stitch_id", "crop_path"]], on="stitch_id", how="left")
    else:
        template["crop_path"] = None

    for col in ("rater_1", "rater_2", "rater_3")[:n_raters]:
        template[col] = None
    if n_raters < 3:
        for col in ("rater_1", "rater_2", "rater_3")[n_raters:]:
            template[col] = None
    for col in ("consensus", "n_scorable_raters", "agreement", "not_scorable", "annotation_time_s", "notes"):
        if col not in template.columns:
            template[col] = None
    return template[columns]


def _parse_rating(value) -> float:
    """Convert a raw rater cell to a float score, or NaN for not-scorable/blank."""
    if value is None:
        return float("nan")
    if isinstance(value, float) and np.isnan(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", config.NOT_SCORABLE_TOKEN, "not scorable", "occluded"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def consensus_scores(ratings: pd.DataFrame, rater_columns: Iterable[str] = RATER_COLUMNS) -> pd.DataFrame:
    """Add ``consensus`` (median) and ``n_scorable_raters`` columns.

    Mean and inter-rater agreement are reported separately; the median is the
    primary target because it is robust to a single outlier rater.
    """
    out = ratings.copy()
    matrix = np.vstack([[_parse_rating(v) for v in out[col].tolist()] for col in rater_columns]).T
    with np.errstate(all="ignore"):
        out["consensus"] = np.nanmedian(matrix, axis=1)
        out["n_scorable_raters"] = np.sum(np.isfinite(matrix), axis=1)
        out["rater_mean"] = np.nanmean(matrix, axis=1)
    return out


def inter_rater_agreement(
    ratings: pd.DataFrame,
    rater_columns: Iterable[str] = RATER_COLUMNS,
) -> dict:
    """Compute inter-rater agreement (the human ceiling for the model).

    Reports mean pairwise exact agreement, mean pairwise within-one agreement
    and, when enough data exist, an ICC(2,1)-style single-measures agreement.
    All values are computed on the *scorable* (non-NaN) ratings only.
    """
    cols = list(rater_columns)
    matrix = np.vstack([[_parse_rating(v) for v in ratings[col].tolist()] for col in cols]).T

    exact_pairs: list[float] = []
    within_one_pairs: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = matrix[:, i], matrix[:, j]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() == 0:
                continue
            diff = np.abs(a[valid] - b[valid])
            exact_pairs.append(float(np.mean(diff == 0)))
            within_one_pairs.append(float(np.mean(diff <= 1)))

    result = {
        "n_items": int(matrix.shape[0]),
        "n_raters": len(cols),
        "n_scorable_items": int(np.sum(np.isfinite(matrix).any(axis=1))),
        "mean_pairwise_exact_agreement": float(np.mean(exact_pairs)) if exact_pairs else float("nan"),
        "mean_pairwise_within_one_agreement": float(np.mean(within_one_pairs)) if within_one_pairs else float("nan"),
        "icc_2_1": _icc_2_1(matrix),
    }
    return result


def _icc_2_1(matrix: np.ndarray) -> float:
    """Single-measures ICC(2,1) for a subjects x raters matrix (NaN allowed).

    Returns NaN when there is insufficient data or zero variance.
    """
    complete = matrix[np.isfinite(matrix).all(axis=1)]
    n, k = complete.shape
    if n < 3 or k < 2:
        return float("nan")
    grand = complete.mean()
    row_means = complete.mean(axis=1)
    col_means = complete.mean(axis=0)
    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((complete - grand) ** 2))
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom <= 1e-12:
        return float("nan")
    return float((ms_rows - ms_err) / denom)
