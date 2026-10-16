"""Small, dependency-light helpers shared across the package."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

# Frame filename example: lerobot_jesse_batch_1__ep0__f5331.png
_FRAME_RE = re.compile(r"^(?P<batch>.+?)__ep(?P<episode>\d+)__f(?P<frame>\d+)$")


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """Write *obj* as UTF-8 JSON, creating parent directories."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
        fh.write("\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def parse_frame_id(stem: str) -> dict[str, Any]:
    """Parse ``<batch>__ep<e>__f<f>`` into batch/episode/frame.

    Accepts an optional ``_mask`` or ``_stitches`` suffix.  Raises ValueError
    for names that do not follow the dataset convention.
    """
    stem = stem.strip()
    for suffix in ("_mask", "_stitches"):
        if suffix in stem:
            stem = stem.split(suffix)[0]
    match = _FRAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Unrecognised frame identifier: {stem!r}")
    return {
        "batch": match.group("batch"),
        "episode": int(match.group("episode")),
        "frame": int(match.group("frame")),
        "group_id": f"{match.group('batch')}__ep{match.group('episode')}",
    }


def seed_everything(seed: int = 42) -> None:
    """Best-effort deterministic seeding for python/numpy/torch/sklearn."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at import time
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


def relpath(path: str | Path, base: str | Path) -> str:
    """Return *path* relative to *base* as a POSIX string (portable CSV)."""
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def resolve_path(path: str | Path) -> Path:
    """Resolve an artifact path that may be relative to the repository root."""
    from . import config

    p = Path(path)
    return p if p.is_absolute() else (config.REPO_ROOT / p)


def absolutize_path_columns(df, columns: tuple[str, ...]):
    """Return a copy of *df* with the given path columns made absolute."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(lambda v: str(resolve_path(v)) if isinstance(v, str) and v else v)
    return out

