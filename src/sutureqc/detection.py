"""Track A -- stitch localization baseline (YOLO dataset prep + guarded train).

Detection is explicitly NOT the novelty of this work (Section 6.7).  It exists
to (a) provide a localization baseline and (b) let the scoring model be
evaluated with ground-truth boxes vs predicted boxes.  ``ultralytics`` is an
optional dependency: when it is missing the dataset is still built and the
training step degrades to a documented "unavailable" result.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, splits
from .utils import ensure_dir, resolve_path, write_json


class DetectorUnavailable(RuntimeError):
    """Raised when the optional YOLO detector stack is not installed."""


def _yolo_line(box: np.ndarray, width: int = config.IMAGE_W, height: int = config.IMAGE_H) -> str:
    x1, y1, x2, y2 = (float(v) for v in box)
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo_dataset(
    manifest_all_df: pd.DataFrame,
    out_dir: Path = config.ARTIFACTS_DIR / "yolo",
    seed: int = config.SEED,
) -> dict:
    """Materialise a YOLO detection dataset from ``manifest_all.csv``.

    Images are hard-linked (falling back to copy) into ``images/`` and one
    ``labels/<name>.txt`` is written per annotated frame; empty-bbox frames get
    an empty label file (a valid detection negative).  Splits are grouped by
    episode so no group leaks across train/val/test.
    """
    from .bbox import load_bboxes, normalize_bbox, validate_bbox

    out_dir = ensure_dir(out_dir)
    images_dir = ensure_dir(out_dir / "images")
    labels_dir = ensure_dir(out_dir / "labels")

    frame_table = manifest_all_df[manifest_all_df["has_frame"]].copy()
    frame_table["group_id"] = frame_table["group_id"].astype(str)
    frame_table = splits.assign_grouped_splits(frame_table, group_col="group_id", seed=seed)

    counts = {"positive": 0, "negative": 0, "boxes": 0, "skipped": 0}
    lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for record in frame_table.to_dict("records"):
        frame_path = resolve_path(record["frame_path"])
        basename = record["basename"]
        image_dst = images_dir / f"{basename}.png"
        label_path = labels_dir / f"{basename}.txt"

        if not frame_path.exists():
            counts["skipped"] += 1
            continue

        _link_or_copy(frame_path, image_dst)

        lines: list[str] = []
        if record["has_bbox"]:
            for box in load_bboxes(resolve_path(record["bbox_path"])):
                ok, _ = validate_bbox(box)
                if not ok:
                    continue
                lines.append(_yolo_line(normalize_bbox(box)))
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if lines:
            counts["positive"] += 1
            counts["boxes"] += len(lines)
        elif record["is_empty_bbox"]:
            counts["negative"] += 1
        else:
            counts["skipped"] += 1

        lists[record["split"]].append(str(image_dst.resolve()))

    for split_name, paths in lists.items():
        (out_dir / f"{split_name}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": str((out_dir / "train.txt").resolve()),
        "val": str((out_dir / "val.txt").resolve()),
        "test": str((out_dir / "test.txt").resolve()),
        "names": {0: "stitch"},
    }
    write_json(out_dir / "data.yaml", data_yaml)
    return {
        "out_dir": str(out_dir),
        "data_yaml": str(out_dir / "data.yaml"),
        "counts": counts,
        "split_sizes": {k: len(v) for k, v in lists.items()},
    }


def oracle_detection_metrics() -> dict:
    """Reference metrics for perfect (ground-truth) boxes."""
    return {
        "available": False,
        "box_source": "ground_truth",
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "mAP50": 1.0,
        "mAP50-95": 1.0,
        "note": "Ground-truth boxes by definition achieve perfect localization; "
        "these values define the upper reference, not a trained detector.",
    }


def train_yolo(
    data_yaml: str | Path,
    epochs: int = 40,
    imgsz: int = 640,
    model_name: str = "yolo11n.pt",
    device: int | str = 0,
    project: Path | None = None,
) -> dict:
    """Train a small YOLO detector when ``ultralytics`` is installed."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DetectorUnavailable(
            "ultralytics is not installed. Run `pip install ultralytics` to train the detector. "
            "The scoring pipeline does not require it."
        ) from exc

    project = project or (config.ARTIFACTS_DIR / "yolo_runs")
    model = YOLO(model_name)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, device=device, project=str(project), verbose=False)
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device, verbose=False)
    return {
        "available": True,
        "box_source": "predicted_yolo",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-9)),
    }


def detection_report(manifest_all_df: pd.DataFrame) -> dict:
    """Dataset-level detection summary without requiring ultralytics."""
    positives = int((manifest_all_df["bbox_valid_count"] > 0).sum())
    negatives = int(manifest_all_df["is_empty_bbox"].sum())
    total_boxes = int(manifest_all_df["bbox_valid_count"].sum())
    baseline = positives / (positives + negatives) if (positives + negatives) else float("nan")
    return {
        "available": False,
        "box_source": "ground_truth",
        "positive_frames": positives,
        "negative_frames": negatives,
        "total_boxes": total_boxes,
        "majority_class_accuracy": baseline,
        "oracle": oracle_detection_metrics(),
        "note": "ultralytics not used; predicted-box scoring is illustrative until a detector is trained.",
    }
