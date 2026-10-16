"""Wound-mask loading and geometric descriptors.

Masks are 960x540 greyscale PNGs with values ``{0, 255}``; ``255`` marks the
wound region.  We expose a small set of descriptors used by the geometry
feature builder: centroid, area, bounding box, principal axis and the
signed distance from any point to the wound boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class WoundDescriptor:
    """Pre-computed descriptors for a single wound mask."""

    area: float
    width: float
    height: float
    cx: float
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float
    axis_angle_deg: float  # principal axis orientation in degrees
    axis_length: float
    scale: float  # sqrt(area) with a small floor, used to normalise distances
    diag: float
    is_empty: bool


def load_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Load a mask as a boolean array ``(H, W)`` where True == wound."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return img > threshold


def mask_area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask))


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (float("nan"),) * 4
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _principal_axis(mask: np.ndarray) -> tuple[float, float]:
    """Return ``(angle_deg, length)`` of the mask's principal axis (PCA)."""
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        return float("nan"), float("nan")
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    projected = pts @ major
    length = float(projected.max() - projected.min())
    return angle, length


def describe_mask(mask: np.ndarray) -> WoundDescriptor:
    """Compute every wound descriptor from a boolean mask."""
    area = mask_area(mask)
    if area == 0:
        nan = float("nan")
        return WoundDescriptor(nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, True)

    cx, cy = mask_centroid(mask)
    x0, y0, x1, y1 = mask_bbox(mask)
    angle, axis_length = _principal_axis(mask)
    scale = float(np.sqrt(area))
    diag = float(np.hypot(x1 - x0, y1 - y0))
    return WoundDescriptor(
        area=area,
        width=float(x1 - x0),
        height=float(y1 - y0),
        cx=cx,
        cy=cy,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        axis_angle_deg=angle,
        axis_length=axis_length,
        scale=scale,
        diag=diag,
        is_empty=False,
    )


def boundary_distance_maps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(inside_dist, outside_dist)`` distance-to-boundary maps.

    ``inside_dist`` is the Euclidean distance from each wound pixel to the
    nearest non-wound pixel; ``outside_dist`` is the distance from each
    background pixel to the nearest wound pixel.  Both are float32 of shape
    ``(H, W)``.
    """
    wound_u8 = (mask.astype(np.uint8)) * 255
    bg_u8 = 255 - wound_u8
    inside = cv2.distanceTransform(wound_u8, cv2.DIST_L2, 5).astype(np.float32)
    outside = cv2.distanceTransform(bg_u8, cv2.DIST_L2, 5).astype(np.float32)
    return inside, outside


def signed_distance_to_boundary(
    point_xy: tuple[float, float],
    inside_dist: np.ndarray,
    outside_dist: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Distance from *point_xy* to the wound boundary.

    Negative inside the wound, positive outside.  Returns ``nan`` for points
    outside the image.
    """
    x, y = point_xy
    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return float("nan")
    if mask[yi, xi]:
        return float(-inside_dist[yi, xi])
    return float(outside_dist[yi, xi])


def bbox_wound_overlap(box: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the stitch bbox covered by the wound mask (0-1)."""
    h, w = mask.shape
    x1 = int(np.clip(np.floor(box[0]), 0, w - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, h - 1))
    x2 = int(np.clip(np.ceil(box[2]), 0, w))
    y2 = int(np.clip(np.ceil(box[3]), 0, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    box_area = float((x2 - x1) * (y2 - y1))
    if box_area <= 0:
        return 0.0
    return float(np.count_nonzero(sub) / box_area)
