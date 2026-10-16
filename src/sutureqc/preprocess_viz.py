"""Step-by-step visual diagnostics for the Notebook-1 preprocessing pipeline.

Each function returns a live ``matplotlib`` Figure so a notebook can display the
intermediate state of one preprocessing step (frame -> mask -> bbox -> crop ->
geometry -> splits -> proxy target) inline; pass ``out=...`` to also save a PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .crops import compute_padded_box, crop_stitch  # noqa: E402
from .geometry import GEOMETRY_FEATURE_COLUMNS  # noqa: E402
from .masks import load_mask  # noqa: E402

_DEFAULT_DISTRIBUTIONS = (
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "bbox_aspect",
    "dist_center_to_wound_centroid_norm",
    "dist_center_to_wound_boundary_norm",
    "bbox_wound_overlap_ratio",
    "spacing_cv",
)


def _read_rgb(path: str | Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _no_data(ax, message: str = "no data") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=8, color="#888888")
    ax.axis("off")


def _save(fig, out: Path | None):
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def show_frame(frame_path: str | Path, out: Path | None = None, title: str | None = None):
    """Show one raw endoscope frame."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    image = _read_rgb(frame_path)
    if image is None:
        _no_data(ax, f"cannot read {Path(frame_path).name}")
    else:
        ax.imshow(image)
        ax.set_title(title or Path(frame_path).name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out)


def show_mask(frame_path: str | Path, mask_path: str | Path, out: Path | None = None):
    """Show frame, binary wound mask and the green mask overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    image = _read_rgb(frame_path)
    if image is None:
        for ax in axes:
            _no_data(ax, "cannot read frame")
        return _save(fig, out)

    mask = load_mask(mask_path) if mask_path else np.zeros(image.shape[:2], dtype=bool)
    axes[0].imshow(image)
    axes[0].set_title("frame", fontsize=9)
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"wound mask (area={int(mask.sum())}px)", fontsize=9)
    overlay = image.copy()
    if mask.any():
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("mask overlay", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(Path(frame_path).name, fontsize=9)
    fig.tight_layout()
    return _save(fig, out)


def show_bboxes(frame_path: str | Path, boxes, out: Path | None = None, title: str | None = None):
    """Show a frame with every raw bbox drawn and numbered."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    image = _read_rgb(frame_path)
    if image is None:
        _no_data(ax, f"cannot read {Path(frame_path).name}")
        return _save(fig, out)
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = (int(float(v)) for v in box[:4])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, str(index), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    ax.imshow(image)
    ax.set_title(title or f"{Path(frame_path).name}  (n={len(list(boxes))})", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, out)


def show_manifest_summary(manifest_all: pd.DataFrame, out: Path | None = None):
    """Bar summary of every frame: batches, annotation flags and bbox counts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    if manifest_all.empty:
        for ax in axes:
            _no_data(ax)
        return _save(fig, out)

    manifest_all["batch"].value_counts().plot(kind="bar", ax=axes[0], color="#3b6ea5")
    axes[0].set_title("frames per batch")
    axes[0].tick_params(axis="x", rotation=60, labelsize=6)

    annot = manifest_all[["has_mask", "has_bbox", "is_annotated"]].sum()
    annot.plot(kind="bar", ax=axes[1], color="#5a9367")
    axes[1].set_title("masks / bboxes / annotated frames")

    empty = int(manifest_all["is_empty_bbox"].sum()) if "is_empty_bbox" in manifest_all else 0
    non_empty = int(len(manifest_all) - empty)
    axes[2].bar(["empty bbox\n(negative)", "non-empty\nbbox"], [empty, non_empty], color=["#c44e52", "#3b6ea5"])
    axes[2].set_title("empty vs non-empty bbox frames")

    manifest_all["bbox_valid_count"].plot(kind="hist", bins=range(0, 7), ax=axes[3], color="#8172b3", rwidth=0.85)
    axes[3].set_title("distribution of valid bboxes per frame")

    fig.suptitle("Notebook 1 - Step 2-3: frame / mask / bbox inventory", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_stitch_instances(stitch_df: pd.DataFrame, out: Path | None = None):
    """Bar summary of the expanded stitch-instance table."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    if stitch_df.empty:
        for ax in axes:
            _no_data(ax)
        return _save(fig, out)

    stitch_df["batch"].value_counts().plot(kind="bar", ax=axes[0], color="#3b6ea5")
    axes[0].set_title(f"stitch instances per batch (n={len(stitch_df)})")
    axes[0].tick_params(axis="x", rotation=60, labelsize=6)

    per_frame = stitch_df.groupby("basename").size()
    per_frame.plot(kind="hist", bins=range(0, int(per_frame.max()) + 2), ax=axes[1], color="#5a9367", rwidth=0.85)
    axes[1].set_title("stitches per frame")

    reasons = stitch_df["bbox_valid_reason"].value_counts() if "bbox_valid_reason" in stitch_df else pd.Series(dtype=int)
    if reasons.empty:
        _no_data(axes[2], "no validation reasons")
    else:
        reasons.head(6).plot(kind="bar", ax=axes[2], color="#c44e52")
        axes[2].set_title("bbox validation outcome")

    fig.suptitle("Notebook 1 - Step 6: one row per validated stitch instance", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_crop_pipeline(frame_path: str | Path, box, out: Path | None = None, pad: float = config.CROP_PAD):
    """Show the crop rule: bbox -> padded box -> resized 224x224 crop."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    image = _read_rgb(frame_path)
    if image is None:
        for ax in axes:
            _no_data(ax, "cannot read frame")
        return _save(fig, out)

    box = np.asarray(box[:4], dtype=float)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cx1, cy1, cx2, cy2 = compute_padded_box(box, pad=pad, width=w, height=h)

    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    axes[0].imshow(annotated)
    axes[0].set_title(f"bbox (red) + {int(pad * 100)}% padded crop (green)", fontsize=9)

    padded = image[cy1:cy2, cx1:cx2]
    if padded.size == 0:
        padded = image
    axes[1].imshow(padded)
    axes[1].set_title(f"padded crop {(cx2 - cx1, cy2 - cy1)}", fontsize=9)

    resized = cv2.resize(image[max(cy1, 0):cy2, max(cx1, 0):cx2] if padded.size else image,
                         (config.CROP_SIZE, config.CROP_SIZE), interpolation=cv2.INTER_AREA)
    axes[2].imshow(resized)
    axes[2].set_title(f"resized crop {config.CROP_SIZE}x{config.CROP_SIZE}", fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.suptitle("Notebook 1 - Step 9: fixed padded stitch crop", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_crops_grid(stitch_df: pd.DataFrame, out: Path | None = None, n: int = 12):
    """Show a grid of saved stitch crops."""
    cols = 4
    rows = int(np.ceil(max(n, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.9))
    axes = np.atleast_1d(axes).ravel()
    if stitch_df.empty:
        for ax in axes:
            _no_data(ax)
        return _save(fig, out)

    sample = stitch_df.head(n)
    for ax, record in zip(axes, sample.to_dict("records")):
        crop_path = config.CROPS_DIR / f"{record['stitch_id']}.png"
        crop = _read_rgb(crop_path)
        if crop is None:
            _no_data(ax, Path(crop_path).name)
            continue
        ax.imshow(crop)
        ax.set_title(str(record["stitch_id"])[-18:], fontsize=6)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.suptitle("Notebook 1 - Step 9: generated stitch crops", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_geometry_overlay(feature_row, peers=None, out: Path | None = None):
    """Overlay the wound-relative geometry measured for one stitch instance."""
    row = feature_row if isinstance(feature_row, dict) else dict(feature_row)
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    image = _read_rgb(row.get("frame_path"))
    if image is None:
        _no_data(ax, "cannot read frame")
        return _save(fig, out)

    mask_path = row.get("mask_path")
    if mask_path and Path(str(mask_path)).exists():
        mask = load_mask(mask_path)
        if mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
    cx, cy = float(row.get("cx", (x1 + x2) / 2)), float(row.get("cy", (y1 + y2) / 2))
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(image, (int(round(cx)), int(round(cy))), 4, (255, 255, 0), -1)

    if peers is not None:
        peer_points = [(float(px), float(py)) for px, py in peers]
        if peer_points:
            nearest = min(peer_points, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            cv2.line(image, (int(cx), int(cy)), (int(nearest[0]), int(nearest[1])), (255, 0, 255), 2)
            cv2.circle(image, (int(nearest[0]), int(nearest[1])), 4, (255, 0, 255), -1)

    if np.isfinite(row.get("wound_cx", np.nan)):
        cv2.drawMarker(image, (int(round(row["wound_cx"])), int(round(row["wound_cy"]))),
                       (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)

    ax.imshow(image)
    lines = [
        f"bbox_w={row.get('bbox_w', float('nan')):.1f}",
        f"dist_boundary_norm={row.get('dist_center_to_wound_boundary_norm', float('nan')):.3f}",
        f"overlap_ratio={row.get('bbox_wound_overlap_ratio', float('nan')):.3f}",
        f"spacing_cv={row.get('spacing_cv', float('nan')):.3f}",
        f"proxy_score={row.get('proxy_score', 'n/a')}",
    ]
    ax.text(1.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7.5, family="monospace")
    ax.set_title(str(row.get("stitch_id", "stitch")), fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, out)


def show_geometry_distributions(features: pd.DataFrame, out: Path | None = None, columns=None):
    """Histogram grid of the key deterministic geometry features."""
    columns = list(columns) if columns is not None else [c for c in _DEFAULT_DISTRIBUTIONS if c in features.columns]
    columns = [c for c in columns if c in features.columns][:8]
    cols = 4
    rows = int(np.ceil(max(len(columns), 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, columns):
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        if values.empty:
            _no_data(ax, column)
            continue
        ax.hist(values, bins=24, color="#3b6ea5")
        ax.set_title(column, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle("Notebook 1 - Step 8: geometry feature distributions", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_split_summary(features: pd.DataFrame, out: Path | None = None):
    """Show split sizes, split x batch composition and groups per split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    if "split" not in features.columns or features.empty:
        for ax in axes:
            _no_data(ax)
        return _save(fig, out)

    features["split"].value_counts().reindex(["train", "val", "test"]).dropna().plot(
        kind="bar", ax=axes[0], color="#3b6ea5")
    axes[0].set_title("instances per split")

    table = pd.crosstab(features["batch"], features["split"])
    table.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("split composition by batch")
    axes[1].tick_params(axis="x", rotation=60, labelsize=6)

    groups = features.groupby("split")["group_id"].nunique().reindex(["train", "val", "test"]).dropna()
    groups.plot(kind="bar", ax=axes[2], color="#5a9367")
    axes[2].set_title("episode groups per split (leakage-free)")

    fig.suptitle("Notebook 1 - Step 11: episode-grouped splits", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_proxy_distribution(features: pd.DataFrame, thresholds=None, out: Path | None = None):
    """Show the continuous proxy and its binned 1-5 ordinal target."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    if "proxy_score" not in features.columns or features.empty:
        for ax in axes:
            _no_data(ax)
        return _save(fig, out)

    continuous = pd.to_numeric(features.get("proxy_score_continuous"), errors="coerce").dropna()
    if continuous.empty:
        _no_data(axes[0], "no continuous proxy")
    else:
        axes[0].hist(continuous, bins=30, color="#8172b3")
        for threshold in (thresholds or []):
            axes[0].axvline(float(threshold), color="#c44e52", linestyle="--", linewidth=1)
        axes[0].set_title("continuous geometric-consistency proxy (+ train quantile cuts)")

    features["proxy_score"].value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#3b6ea5")
    axes[1].set_title("ordinal proxy_score (1-5)")

    fig.suptitle("Notebook 1 - Section 3.4 fallback target", fontsize=11)
    fig.tight_layout()
    return _save(fig, out)


def show_action_labels(stitch_df: pd.DataFrame, out: Path | None = None, top: int = 10):
    """Bar chart of the frame-level semantic action labels joined to stitches."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if stitch_df.empty or "action_label" not in stitch_df.columns:
        _no_data(ax, "no action labels")
        return _save(fig, out)

    labels = stitch_df["action_label"].fillna("unknown").replace({"": "unknown"})
    labels.value_counts().head(top)[::-1].plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_title("Notebook 1 - Step 7: semantic action context (top labels)", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, out)
