"""Audit report and annotation-overlay figure for Notebook 1.

Produces a JSON summary (counts, distributions, leakage checks, missing files,
duplicates, invalid boxes) and the ``figures/sample_annotations.png`` overlay
used to visually confirm that masks and boxes line up with the frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

import cv2
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .masks import load_mask  # noqa: E402
from .utils import ensure_dir, write_json  # noqa: E402


def build_audit_report(
    manifest_all: pd.DataFrame,
    stitch_df: pd.DataFrame,
    features_df: pd.DataFrame,
    splits_df: pd.DataFrame | None = None,
    proxy_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Assemble the audit report as a JSON-serialisable dict."""
    total_frames = int(len(manifest_all))
    annotated = manifest_all[manifest_all["is_annotated"]] if "is_annotated" in manifest_all else manifest_all
    missing_masks = manifest_all[~manifest_all["has_mask"]]["basename"].astype(str).tolist() if "has_mask" in manifest_all else []

    report: dict[str, Any] = {
        "dataset": "SutureBot2 (ex-vivo, two operators, dVRK Si)",
        "clinical_caveat": config.CLINICAL_CAVEAT,
        "counts": {
            "frames_total": total_frames,
            "frames_with_mask": int(manifest_all["has_mask"].sum()) if "has_mask" in manifest_all else 0,
            "frames_with_bbox": int(manifest_all["has_bbox"].sum()) if "has_bbox" in manifest_all else 0,
            "frames_annotated": int(len(annotated)),
            "frames_empty_bbox_negative": int(manifest_all["is_empty_bbox"].sum()) if "is_empty_bbox" in manifest_all else 0,
            "stitch_instances": int(len(stitch_df)),
            "frames_missing_mask": len(missing_masks),
        },
        "invalid_bbox_reasons": _value_counts(manifest_all, "bbox_invalid_reasons"),
        "stitch_instances_by_batch": _value_counts(stitch_df, "batch"),
        "frames_by_batch": _value_counts(manifest_all, "batch"),
        "duplicate_stitch_ids": _duplicates(stitch_df, "stitch_id"),
        "missing_mask_examples": missing_masks[:20],
        "proxy_thresholds": proxy_thresholds,
    }

    if "proxy_score" in features_df.columns:
        report["proxy_score_distribution"] = _value_counts(features_df, "proxy_score")
        report["proxy_score_by_batch"] = _nested_counts(features_df, "batch", "proxy_score")
        report["proxy_score_continuous_summary"] = {
            "mean": _safe(float(np.nanmean(features_df["proxy_score_continuous"]))),
            "std": _safe(float(np.nanstd(features_df["proxy_score_continuous"]))),
            "min": _safe(float(np.nanmin(features_df["proxy_score_continuous"]))),
            "max": _safe(float(np.nanmax(features_df["proxy_score_continuous"]))),
        }

    if splits_df is not None:
        report["split_sizes"] = _value_counts(splits_df, "split")
        report["split_by_batch"] = _nested_counts(splits_df, "split", "batch")

    report["geometry_feature_nan_fraction"] = {
        col: _safe(float(features_df[col].isna().mean()))
        for col in features_df.columns
        if col in _geometry_columns()
    }
    return report


def _safe(value: float) -> float | None:
    if value is None:
        return None
    v = float(value)
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("missing").astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _nested_counts(df: pd.DataFrame, row_col: str, col_col: str) -> dict[str, dict[str, int]]:
    if row_col not in df.columns or col_col not in df.columns:
        return {}
    table = df.groupby([row_col, col_col]).size().unstack(fill_value=0)
    return {
        str(row): {str(col): int(val) for col, val in row_values.items()}
        for row, row_values in table.to_dict(orient="index").items()
    }


def _duplicates(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    dupes = df[df.duplicated(column, keep=False)][column].astype(str).unique().tolist()
    return dupes[:50]


def _geometry_columns() -> tuple[str, ...]:
    from .geometry import GEOMETRY_FEATURE_COLUMNS

    return GEOMETRY_FEATURE_COLUMNS


def plot_sample_annotations(
    stitch_df: pd.DataFrame,
    out_png: Path = config.SAMPLE_ANNOTATIONS_PNG,
    n: int = 9,
    cols: int = 3,
) -> Path:
    """Save a grid of frame + wound-mask contour + stitch bbox overlays."""
    ensure_dir(Path(out_png).parent)
    if stitch_df.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no annotations", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_png

    # Prefer one representative stitch per batch, then fill round-robin.
    frame_ids = stitch_df["basename"].drop_duplicates().tolist()
    by_batch: dict[str, list[str]] = {}
    for record in stitch_df.drop_duplicates("basename").to_dict("records"):
        by_batch.setdefault(record["batch"], []).append(record["basename"])
    chosen: list[str] = []
    for batch in sorted(by_batch):
        chosen.append(by_batch[batch][0])
    for fid in frame_ids:
        if len(chosen) >= n:
            break
        if fid not in chosen:
            chosen.append(fid)
    chosen = chosen[:n]

    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()

    mask_cache: dict[str, np.ndarray] = {}
    for ax, basename in zip(axes, chosen):
        subset = stitch_df[stitch_df["basename"] == basename]
        frame_path = subset.iloc[0]["frame_path"]
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            ax.axis("off")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = subset.iloc[0]["mask_path"]
        if mask_path and Path(mask_path).exists():
            if mask_path not in mask_cache:
                mask_cache[mask_path] = load_mask(mask_path)
            mask = mask_cache[mask_path]
            if mask.any():
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
                overlay = image.copy()
                overlay[mask] = (0.7 * overlay[mask] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
                image = overlay

        for record in subset.to_dict("records"):
            x1, y1, x2, y2 = int(record["x1"]), int(record["y1"]), int(record["x2"]), int(record["y2"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(image, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (255, 255, 0), -1)

        ax.imshow(image)
        ax.set_title(f"{record['group_id']} (n={len(subset)})", fontsize=8)
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("SutureBot2: frames (RGB) with wound mask (green) and stitch bboxes (blue)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(out_png)


def write_audit(report: dict[str, Any], path: Path = config.AUDIT_REPORT_JSON) -> Path:
    write_json(path, report)
    return Path(path)
