"""Stitch bounding-box parsing and validation.

Bounding boxes in ``stitch_bboxes/*.json`` are a JSON list of
``[x1, y1, x2, y2]`` pixel coordinates (integers or floats) in the native
960x540 endoscope frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def load_bboxes(path: str | Path) -> list[np.ndarray]:
    """Load one stitch-bbox JSON file into a list of float32 ``[x1,y1,x2,y2]``.

    Invalid entries (wrong length, non-finite) are dropped.
    """
    import json

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    boxes: list[np.ndarray] = []
    if not isinstance(raw, list):
        return boxes
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            continue
        try:
            box = np.asarray([float(v) for v in entry], dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(box)):
            continue
        boxes.append(box)
    return boxes


def normalize_bbox(box: np.ndarray) -> np.ndarray:
    """Return ``[x1, y1, x2, y2]`` with x1<=x2 and y1<=y2."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.asarray([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)


def validate_bbox(
    box: np.ndarray,
    width: int = config.IMAGE_W,
    height: int = config.IMAGE_H,
    min_size: float = 2.0,
) -> tuple[bool, str]:
    """Return ``(is_valid, reason)`` for a single ``[x1,y1,x2,y2]`` box."""
    if box.shape != (4,):
        return False, "shape"
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(np.isfinite([x1, y1, x2, y2])):
        return False, "non_finite"
    if x2 <= x1 or y2 <= y1:
        return False, "degenerate"
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return False, "too_small"
    if x2 < 0 or y2 < 0 or x1 > width or y1 > height:
        return False, "out_of_frame"
    return True, "ok"


def clip_bbox(box: np.ndarray, width: int = config.IMAGE_W, height: int = config.IMAGE_H) -> np.ndarray:
    """Clip a normalised box to the image bounds."""
    b = normalize_bbox(box)
    b[0] = np.clip(b[0], 0, width)
    b[1] = np.clip(b[1], 0, height)
    b[2] = np.clip(b[2], 0, width)
    b[3] = np.clip(b[3], 0, height)
    return b


def bbox_width(box: np.ndarray) -> float:
    return float(box[2] - box[0])


def bbox_height(box: np.ndarray) -> float:
    return float(box[3] - box[1])


def bbox_area(box: np.ndarray) -> float:
    b = normalize_bbox(box)
    return max(0.0, float((b[2] - b[0]) * (b[3] - b[1])))


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    b = normalize_bbox(box)
    return float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)


def bbox_aspect(box: np.ndarray) -> float:
    h = bbox_height(box)
    if h <= 0:
        return float("nan")
    return bbox_width(box) / h
