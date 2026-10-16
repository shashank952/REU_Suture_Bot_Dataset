"""Deterministic wound-relative geometry features for each stitch instance.

Everything here is computed from the mask, the bbox and (optionally) the
episode-level insertion/exit annotations.  No learned model is involved, so
these features are reproducible and directly interpretable -- they double as
the explainability evidence and as the "image-free" baseline input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config, points_io
from .bbox import bbox_area, bbox_aspect, bbox_center, bbox_height, bbox_width
from .masks import (
    bbox_wound_overlap,
    boundary_distance_maps,
    describe_mask,
    load_mask,
    signed_distance_to_boundary,
)

# Columns produced by :func:`compute_geometry_features` (order matters for the
# notebooks and for the geometry-only baseline).
GEOMETRY_FEATURE_COLUMNS: tuple[str, ...] = (
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "bbox_aspect",
    "cx",
    "cy",
    "cx_norm",
    "cy_norm",
    "wound_area",
    "wound_cx",
    "wound_cy",
    "wound_w",
    "wound_h",
    "wound_scale",
    "wound_diag",
    "wound_axis_angle_deg",
    "dist_center_to_wound_centroid_norm",
    "dist_center_to_wound_boundary_norm",
    "center_inside_wound",
    "bbox_wound_overlap_ratio",
    "n_stitches_in_frame",
    "nn_dist",
    "nn_dist_norm",
    "spacing_cv",
    "ie_available",
    "ie_length",
    "ie_length_norm",
    "ie_mid_dist_norm",
    "ie_crossing_score",
    "ie_angle_diff_deg",
)


def _angle_diff_mod180(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    d = abs((a - b) % 180.0)
    return float(min(d, 180.0 - d))


def compute_geometry_features(
    stitch_df: pd.DataFrame,
    insertion_exit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return *stitch_df* augmented with all geometry feature columns.

    Frames are processed once (mask loaded and described a single time), and
    frame-level spacing statistics are shared across the stitches of a frame.
    """
    if stitch_df.empty:
        out = stitch_df.copy()
        for col in GEOMETRY_FEATURE_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    mask_cache: dict[str, tuple[np.ndarray, object, tuple[np.ndarray, np.ndarray]]] = {}
    rows: list[dict] = []

    for basename, frame_df in stitch_df.groupby("basename", sort=False):
        mask_path = frame_df.iloc[0]["mask_path"]
        if basename not in mask_cache:
            mask = load_mask(mask_path)
            desc = describe_mask(mask)
            dist_maps = boundary_distance_maps(mask)
            mask_cache[basename] = (mask, desc, dist_maps)
        mask, desc, (inside_dist, outside_dist) = mask_cache[basename]

        centers: list[tuple[float, float]] = []
        boxes: list[np.ndarray] = []
        for record in frame_df.to_dict("records"):
            box = np.asarray([record["x1"], record["y1"], record["x2"], record["y2"]], dtype=np.float32)
            boxes.append(box)
            centers.append(bbox_center(box))

        n_stitches = len(boxes)
        # Spacing regularity along the wound principal axis (or x-axis fallback).
        axis_angle = desc.axis_angle_deg if np.isfinite(desc.axis_angle_deg) else 0.0
        theta = np.radians(axis_angle)
        projections = [cx * np.cos(theta) + cy * np.sin(theta) for cx, cy in centers]
        projections.sort()
        if n_stitches >= 2:
            gaps = np.diff(np.asarray(projections))
            mean_gap = float(np.mean(gaps))
            spacing_cv = float(np.std(gaps) / mean_gap) if mean_gap > 1e-6 else float("nan")
        else:
            spacing_cv = float("nan")

        for i, record in enumerate(frame_df.to_dict("records")):
            box = boxes[i]
            cx, cy = centers[i]
            w, h = bbox_width(box), bbox_height(box)
            area = bbox_area(box)

            if n_stitches >= 2:
                others = [np.hypot(cx - ox, cy - oy) for j, (ox, oy) in enumerate(centers) if j != i]
                nn_dist = float(min(others)) if others else float("nan")
            else:
                nn_dist = float("nan")

            scale = desc.scale if np.isfinite(desc.scale) and desc.scale > 0 else 1.0
            dist_centroid = float(np.hypot(cx - desc.cx, cy - desc.cy)) if not desc.is_empty else float("nan")
            signed_boundary = signed_distance_to_boundary((cx, cy), inside_dist, outside_dist, mask)
            overlap = bbox_wound_overlap(box, mask) if not desc.is_empty else float("nan")

            # Insertion / exit match.
            ie = None
            if insertion_exit_df is not None and not insertion_exit_df.empty:
                ie = points_io.match_insertion_exit(record["batch"], int(record["episode"]), (cx, cy), insertion_exit_df)

            row = {
                "bbox_w": w,
                "bbox_h": h,
                "bbox_area": area,
                "bbox_aspect": bbox_aspect(box),
                "cx": cx,
                "cy": cy,
                "cx_norm": cx / config.IMAGE_W,
                "cy_norm": cy / config.IMAGE_H,
                "wound_area": desc.area,
                "wound_cx": desc.cx,
                "wound_cy": desc.cy,
                "wound_w": desc.width,
                "wound_h": desc.height,
                "wound_scale": scale,
                "wound_diag": desc.diag,
                "wound_axis_angle_deg": desc.axis_angle_deg,
                "dist_center_to_wound_centroid_norm": dist_centroid / scale if np.isfinite(dist_centroid) else float("nan"),
                "dist_center_to_wound_boundary_norm": signed_boundary / scale if np.isfinite(signed_boundary) else float("nan"),
                "center_inside_wound": float(bool(not desc.is_empty and mask[int(round(cy)) if 0 <= int(round(cy)) < mask.shape[0] else 0, int(round(cx)) if 0 <= int(round(cx)) < mask.shape[1] else 0])),
                "bbox_wound_overlap_ratio": overlap,
                "n_stitches_in_frame": float(n_stitches),
                "nn_dist": nn_dist,
                "nn_dist_norm": nn_dist / scale if np.isfinite(nn_dist) else float("nan"),
                "spacing_cv": spacing_cv,
                "ie_available": 0.0,
                "ie_length": float("nan"),
                "ie_length_norm": float("nan"),
                "ie_mid_dist_norm": float("nan"),
                "ie_crossing_score": float("nan"),
                "ie_angle_diff_deg": float("nan"),
            }

            if ie is not None:
                ins = np.asarray([ie["insertion_x"], ie["insertion_y"]], dtype=float)
                ext = np.asarray([ie["exit_x"], ie["exit_y"]], dtype=float)
                mid = (ins + ext) / 2.0
                ie_len = float(np.hypot(*(ext - ins)))
                mid_dist = float(np.hypot(cx - mid[0], cy - mid[1]))
                ie_angle = float(np.degrees(np.arctan2(ext[1] - ins[1], ext[0] - ins[0])))
                angle_diff = _angle_diff_mod180(ie_angle, desc.axis_angle_deg)
                crossing = float(np.clip(1.0 - abs(angle_diff - 90.0) / 90.0, 0.0, 1.0)) if np.isfinite(angle_diff) else float("nan")
                row.update(
                    {
                        "ie_available": 1.0,
                        "ie_length": ie_len,
                        "ie_length_norm": ie_len / scale,
                        "ie_mid_dist_norm": mid_dist / scale,
                        "ie_crossing_score": crossing,
                        "ie_angle_diff_deg": angle_diff,
                    }
                )
            rows.append(row)

    features = pd.DataFrame(rows, index=stitch_df.index)
    return pd.concat([stitch_df, features], axis=1)


def feature_matrix(df: pd.DataFrame, columns: tuple[str, ...] = GEOMETRY_FEATURE_COLUMNS) -> np.ndarray:
    """Return a float64 numpy matrix of the geometry columns (NaN preserved)."""
    return df.loc[:, list(columns)].to_numpy(dtype=np.float64)
