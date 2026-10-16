"""Fixed padded stitch crops.

Each stitch instance becomes one crop with :data:`config.CROP_PAD` padding on
every side, clipped to the frame, then resized to ``CROP_SIZE``.  Crops are
written deterministically so the image branch is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config
from .utils import ensure_dir


def compute_padded_box(
    box: np.ndarray,
    pad: float = config.CROP_PAD,
    width: int = config.IMAGE_W,
    height: int = config.IMAGE_H,
) -> tuple[int, int, int, int]:
    """Return an integer ``(x1, y1, x2, y2)`` crop box with *pad* padding."""
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    cx1 = int(np.clip(np.floor(x1 - px), 0, width - 1))
    cy1 = int(np.clip(np.floor(y1 - py), 0, height - 1))
    cx2 = int(np.clip(np.ceil(x2 + px), 1, width))
    cy2 = int(np.clip(np.ceil(y2 + py), 1, height))
    return cx1, cy1, cx2, cy2


def crop_stitch(frame_bgr: np.ndarray, box: np.ndarray, pad: float = config.CROP_PAD) -> np.ndarray:
    """Return a resized ``CROP_SIZE`` x ``CROP_SIZE`` BGR crop for *box*."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = compute_padded_box(box, pad, w, h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
    return cv2.resize(crop, (config.CROP_SIZE, config.CROP_SIZE), interpolation=cv2.INTER_AREA)


def generate_crops(
    stitch_df: pd.DataFrame,
    crops_dir: Path = config.CROPS_DIR,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write one crop per stitch instance and return a path table.

    Columns: ``stitch_id, crop_path, crop_exists``.  Frames are read once and
    reused across their stitches.
    """
    ensure_dir(crops_dir)
    records: list[dict] = []
    frame_cache: dict[str, np.ndarray] = {}

    for record in stitch_df.to_dict("records"):
        stitch_id = record["stitch_id"]
        crop_path = crops_dir / f"{stitch_id}.png"
        exists = crop_path.exists()
        if overwrite or not exists:
            frame_path = record["frame_path"]
            if frame_path not in frame_cache:
                img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Could not read frame: {frame_path}")
                frame_cache.clear()  # keep memory bounded: only current frame
                frame_cache[frame_path] = img
            frame_bgr = frame_cache[frame_path]
            box = np.asarray([record["x1"], record["y1"], record["x2"], record["y2"]], dtype=np.float32)
            crop = crop_stitch(frame_bgr, box)
            cv2.imwrite(str(crop_path), crop)
            exists = True
        records.append({"stitch_id": stitch_id, "crop_path": str(crop_path), "crop_exists": exists})

    return pd.DataFrame(records)
