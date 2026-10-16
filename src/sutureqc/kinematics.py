"""Robot kinematics access (LeRobot v2.1 parquet episodes).

Only ``observation.state`` / ``action`` / ``timestamp`` / ``frame_index`` are
needed, all stored per-frame in ``data/chunk-000/episode_%06d.parquet``.
``pyarrow`` is required to read parquet; callers should handle the
``KinematicsUnavailable`` case gracefully.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


class KinematicsUnavailable(RuntimeError):
    """Raised when parquet support or episode files are missing."""


def episode_parquet_path(batch: str, episode: int, data_root: Path = config.DATA_ROOT) -> Path:
    return data_root / batch / "dvrk_wound_closure" / "data" / "chunk-000" / f"episode_{int(episode):06d}.parquet"


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise KinematicsUnavailable(
            "pyarrow is required to read LeRobot parquet episodes; run `pip install pyarrow`."
        ) from exc


def load_episode_table(batch: str, episode: int, data_root: Path = config.DATA_ROOT) -> pd.DataFrame:
    """Load one episode's per-frame robot table (requires pyarrow)."""
    _require_parquet()
    path = episode_parquet_path(batch, episode, data_root)
    if not path.exists():
        raise KinematicsUnavailable(f"Episode parquet not found: {path}")
    return pd.read_parquet(path)


def pose_at_frame(
    batch: str,
    episode: int,
    frame: int,
    data_root: Path = config.DATA_ROOT,
) -> dict | None:
    """Return the robot state row closest to *frame*, or ``None`` on failure.

    Result keys: ``frame_index``, ``timestamp``, ``state`` (list[float]) and
    ``action`` (list[float]).
    """
    try:
        table = load_episode_table(batch, episode, data_root)
    except KinematicsUnavailable:
        return None
    if table.empty or "frame_index" not in table.columns:
        return None
    row = table.iloc[(table["frame_index"] - int(frame)).abs().argsort()[:1]]
    record = row.iloc[0]
    state = record["observation.state"] if "observation.state" in table.columns else None
    action = record["action"] if "action" in table.columns else None
    return {
        "frame_index": int(record["frame_index"]),
        "timestamp": float(record["timestamp"]) if "timestamp" in table.columns else float("nan"),
        "state": [float(v) for v in state] if state is not None else None,
        "action": [float(v) for v in action] if action is not None else None,
    }
