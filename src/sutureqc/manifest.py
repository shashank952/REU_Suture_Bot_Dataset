"""Manifest loading, path rewriting and frame / mask / bbox matching.

The published ``manifest.csv`` contains the original author's *absolute*
Windows paths.  We rewrite them to local paths by basename so the pipeline is
portable, then match each frame to its mask and stitch-bbox JSON.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .bbox import load_bboxes, validate_bbox
from .utils import parse_frame_id


def _rewrite(original: str | None, target_dir: Path) -> str | None:
    """Map an original absolute path to a local path by basename."""
    if not original or not isinstance(original, str):
        return None
    name = Path(original.replace("\\", "/")).name
    if not name:
        return None
    return str(target_dir / name)


def load_manifest(data_root: Path = config.DATA_ROOT) -> pd.DataFrame:
    """Load + rewrite the author's mask manifest into a tidy frame table.

    Returns a DataFrame with one row per manifest entry and the columns:
    ``basename, frame_path, mask_path, bbox_path, batch, episode, frame,
    group_id``.
    """
    manifest_csv = data_root / "annotation_images" / "masks" / "manifest.csv"
    annot_dir = data_root / "annotation_images"
    masks_dir = annot_dir / "masks"
    bbox_dir = annot_dir / "stitch_bboxes"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    raw = pd.read_csv(manifest_csv)
    expected = {"frame", "mask", "stitch_bboxes"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"Manifest missing columns {expected - set(raw.columns)}")

    rows: list[dict] = []
    for record in raw.to_dict("records"):
        frame_path = _rewrite(record.get("frame"), annot_dir)
        mask_path = _rewrite(record.get("mask"), masks_dir)
        bbox_path = _rewrite(record.get("stitch_bboxes"), bbox_dir)
        if frame_path is None:
            continue
        basename = Path(frame_path).stem
        parsed = parse_frame_id(basename)
        rows.append(
            {
                "basename": basename,
                "frame_path": frame_path,
                "mask_path": mask_path,
                "bbox_path": bbox_path,
                **parsed,
            }
        )
    return pd.DataFrame(rows)


def build_frame_inventory(
    data_root: Path = config.DATA_ROOT,
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a table covering **every** frame PNG, annotated or not.

    Columns include existence flags (``has_frame``/``has_mask``/``has_bbox``),
    bbox counts and the ``is_empty_bbox`` negative-detection flag.  This is the
    ``artifacts/manifest_all.csv`` output.
    """
    annot_dir = data_root / "annotation_images"
    frames = sorted(annot_dir.glob("*.png"))

    if manifest is None:
        manifest = load_manifest(data_root)
    by_base = manifest.set_index("basename")

    rows: list[dict] = []
    for frame_path in frames:
        basename = frame_path.stem
        parsed = parse_frame_id(basename)
        if basename in by_base.index:
            entry = by_base.loc[basename]
            if isinstance(entry, pd.DataFrame):  # duplicate basenames
                entry = entry.iloc[0]
            mask_path = entry["mask_path"]
            bbox_path = entry["bbox_path"]
        else:
            mask_path = str(annot_dir / "masks" / f"{basename}_mask.png")
            bbox_path = str(annot_dir / "stitch_bboxes" / f"{basename}_stitches.json")

        has_mask = bool(mask_path) and Path(mask_path).exists()
        has_bbox = bool(bbox_path) and Path(bbox_path).exists()

        raw_boxes: list = []
        valid = 0
        invalid = 0
        invalid_reasons: list[str] = []
        if has_bbox:
            raw_boxes = load_bboxes(bbox_path)
            for box in raw_boxes:
                ok, reason = validate_bbox(box)
                if ok:
                    valid += 1
                else:
                    invalid += 1
                    invalid_reasons.append(reason)

        rows.append(
            {
                "basename": basename,
                "frame_path": str(frame_path),
                "mask_path": mask_path if has_mask else None,
                "bbox_path": bbox_path if has_bbox else None,
                **parsed,
                "has_frame": True,
                "has_mask": has_mask,
                "has_bbox": has_bbox,
                "is_annotated": bool(has_mask and has_bbox),
                "bbox_count": len(raw_boxes),
                "bbox_valid_count": valid,
                "bbox_invalid_count": invalid,
                "bbox_invalid_reasons": ";".join(invalid_reasons),
                "is_empty_bbox": bool(has_bbox and len(raw_boxes) == 0),
            }
        )
    return pd.DataFrame(rows)


def stitch_instance_rows(manifest_all: pd.DataFrame) -> pd.DataFrame:
    """Expand validated bboxes into one row per stitch instance.

    Empty-bbox frames are valid detection negatives but produce **no** scoring
    rows (they cannot be scored).  Invalid boxes are recorded in the audit but
    excluded from scoring.
    """
    rows: list[dict] = []
    for record in manifest_all.to_dict("records"):
        if not record.get("has_bbox"):
            continue
        boxes = load_bboxes(record["bbox_path"])
        stitch_index = 0
        for box in boxes:
            ok, reason = validate_bbox(box)
            if not ok:
                continue
            stitch_index += 1
            rows.append(
                {
                    "stitch_id": f"{record['basename']}__s{stitch_index}",
                    "basename": record["basename"],
                    "batch": record["batch"],
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "group_id": record["group_id"],
                    "stitch_index": stitch_index,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "bbox_valid_reason": reason,
                    "frame_path": record["frame_path"],
                    "mask_path": record["mask_path"],
                    "bbox_path": record["bbox_path"],
                }
            )
    return pd.DataFrame(rows)
