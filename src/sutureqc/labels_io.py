"""Frame-level semantic action labels (``labels.jsonl``).

Each batch stores ``dvrk_wound_closure/labels.jsonl`` with entries
``{episode_index, start_frame, end_frame, start_timestamp, end_timestamp,
label}`` describing the human-annotated subtask active over a frame range.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def labels_path(batch: str, data_root: Path = config.DATA_ROOT) -> Path:
    return data_root / batch / "dvrk_wound_closure" / "labels.jsonl"


def load_action_labels(batch: str, data_root: Path = config.DATA_ROOT) -> pd.DataFrame:
    """Load a batch's action labels as a DataFrame (empty if missing)."""
    path = labels_path(batch, data_root)
    if not path.exists():
        return pd.DataFrame(
            columns=["batch", "episode", "start_frame", "end_frame", "start_timestamp", "end_timestamp", "label"]
        )
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # labels.jsonl mixes action segments with episode_done markers;
            # only the former carry frame ranges and a semantic label.
            if "start_frame" not in record or "label" not in record:
                continue
            rows.append(
                {
                    "batch": batch,
                    "episode": int(record["episode_index"]),
                    "start_frame": int(record["start_frame"]),
                    "end_frame": int(record["end_frame"]),
                    "start_timestamp": float(record.get("start_timestamp", float("nan"))),
                    "end_timestamp": float(record.get("end_timestamp", float("nan"))),
                    "label": str(record.get("label", "")),
                }
            )
    return pd.DataFrame(rows)


def load_all_action_labels(
    batches: tuple[str, ...] = config.ALL_BATCHES,
    data_root: Path = config.DATA_ROOT,
) -> pd.DataFrame:
    """Concatenate action labels across every batch."""
    frames = [load_action_labels(b, data_root) for b in batches]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=["batch", "episode", "start_frame", "end_frame", "start_timestamp", "end_timestamp", "label"]
        )
    return pd.concat(frames, ignore_index=True)


def annotate_action_context(
    stitch_df: pd.DataFrame,
    action_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join the semantic action label active at each stitch's frame.

    Adds ``action_label``, ``action_start_frame``, ``action_end_frame`` and
    ``action_progress`` (position within the subtask, 0-1) columns.
    """
    out = stitch_df.copy()
    out["action_label"] = "unknown"
    out["action_start_frame"] = -1
    out["action_end_frame"] = -1
    out["action_progress"] = float("nan")
    if action_labels.empty:
        return out

    lookup: dict[tuple[str, int], list[dict]] = {}
    for record in action_labels.to_dict("records"):
        lookup.setdefault((record["batch"], int(record["episode"])), []).append(record)

    for idx, row in out.iterrows():
        key = (row["batch"], int(row["episode"]))
        candidates = lookup.get(key)
        if not candidates:
            continue
        frame = int(row["frame"])
        chosen = None
        for rec in candidates:
            if rec["start_frame"] <= frame < rec["end_frame"]:
                chosen = rec
                break
        if chosen is None:
            continue
        span = max(1, int(chosen["end_frame"]) - int(chosen["start_frame"]))
        out.at[idx, "action_label"] = chosen["label"]
        out.at[idx, "action_start_frame"] = int(chosen["start_frame"])
        out.at[idx, "action_end_frame"] = int(chosen["end_frame"])
        out.at[idx, "action_progress"] = float((frame - int(chosen["start_frame"])) / span)
    return out
