"""Episode-level needle insertion / exit point annotations.

``dvrk_wound_closure/insertion_exit_points.csv`` has one row per stitch pair
per episode:

    episode_index, wound_start_x, wound_start_y, wound_end_x, wound_end_y,
    pair, insertion_x, insertion_y, exit_x, exit_y

The annotated frames are a *subset* of the episode, and the CSV is
episode-level, so a stitch bbox is matched to the insertion/exit pair whose
segment midpoint is closest to the bbox centre.  This is an approximation and
is documented as such.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def insertion_exit_path(batch: str, data_root: Path = config.DATA_ROOT) -> Path:
    return data_root / batch / "dvrk_wound_closure" / "insertion_exit_points.csv"


INSERTION_EXIT_COLUMNS: tuple[str, ...] = (
    "episode_index", "wound_start_x", "wound_start_y", "wound_end_x", "wound_end_y",
    "pair", "insertion_x", "insertion_y", "exit_x", "exit_y",
)


def load_insertion_exit(batch: str, data_root: Path = config.DATA_ROOT) -> pd.DataFrame:
    """Load a batch's insertion/exit CSV (empty DataFrame if absent).

    The upstream CSV contains at least one row where two records were written
    without a line break, so parsing is tolerant: rows that do not have exactly
    ten fields are skipped and counted in ``df.attrs['n_skipped_lines']``.
    """
    path = insertion_exit_path(batch, data_root)
    if not path.exists():
        return pd.DataFrame(columns=[c for c in INSERTION_EXIT_COLUMNS if c != "episode_index"])

    import csv

    rows: list[dict] = []
    skipped = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for parts in reader:
            if len(parts) != len(INSERTION_EXIT_COLUMNS):
                skipped += 1
                continue
            rows.append(dict(zip(INSERTION_EXIT_COLUMNS, parts)))

    df = pd.DataFrame(rows, columns=list(INSERTION_EXIT_COLUMNS))
    for col in INSERTION_EXIT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["episode_index", "insertion_x", "insertion_y", "exit_x", "exit_y"]).reset_index(drop=True)
    df["episode"] = df["episode_index"].astype(int)
    df = df.drop(columns=["episode_index"])
    df.attrs["n_skipped_lines"] = skipped
    return df


def load_all_insertion_exit(
    batches: tuple[str, ...] = config.POINT_LABEL_BATCHES,
    data_root: Path = config.DATA_ROOT,
) -> pd.DataFrame:
    frames = []
    total_skipped = 0
    for batch in batches:
        df = load_insertion_exit(batch, data_root)
        total_skipped += int(df.attrs.get("n_skipped_lines", 0))
        if not df.empty:
            df = df.copy()
            df["batch"] = batch
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.attrs["n_skipped_lines"] = total_skipped
    return combined


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from point P to segment AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return float(np.hypot(apx, apy))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    projx, projy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - projx, py - projy))


def match_insertion_exit(
    batch: str,
    episode: int,
    center_xy: tuple[float, float],
    insertion_exit: pd.DataFrame,
) -> dict | None:
    """Return the best-matching insertion/exit row for a stitch centre.

    ``None`` when the episode has no point annotations.
    """
    if insertion_exit.empty:
        return None
    subset = insertion_exit[(insertion_exit["batch"] == batch) & (insertion_exit["episode"] == int(episode))]
    if subset.empty:
        return None
    cx, cy = center_xy
    best = None
    best_dist = float("inf")
    for record in subset.to_dict("records"):
        mid_x = (record["insertion_x"] + record["exit_x"]) / 2.0
        mid_y = (record["insertion_y"] + record["exit_y"]) / 2.0
        dist = _point_segment_distance(cx, cy, record["insertion_x"], record["insertion_y"], record["exit_x"], record["exit_y"])
        # Prefer the pair whose segment midpoint is close to the stitch centre.
        score = dist + 0.25 * float(np.hypot(cx - mid_x, cy - mid_y))
        if score < best_dist:
            best_dist = score
            best = record
    return best
