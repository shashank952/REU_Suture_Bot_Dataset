"""Grouped train/validation/test split construction.

Grouping key is the **episode** (``group_id`` = ``batch__ep<e>``), never an
individual frame, so no frame from one episode leaks across splits.  A stronger
leave-one-batch-out scheme is also provided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class SplitResult:
    dataframe: pd.DataFrame
    group_column: str
    split_column: str = "split"


def assign_grouped_splits(
    df: pd.DataFrame,
    group_col: str = "group_id",
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Assign ``train``/``val``/``test`` at *group* level.

    Groups are shuffled once with a fixed seed, then greedily allocated to
    reach the requested proportions.  Guarantees at least one group in val and
    test when there are enough groups.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    out = df.copy()
    groups = np.array(sorted(out[group_col].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1 if n >= 3 else 0

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    def label(group: str) -> str:
        if group in train_groups:
            return "train"
        if group in val_groups:
            return "val"
        return "test"

    out["split"] = out[group_col].astype(str).map(label)
    return out


def leave_one_batch_out(df: pd.DataFrame, batch_col: str = "batch") -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Return ``(held_out_batch, train_df, test_df)`` for each batch."""
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for batch in sorted(df[batch_col].astype(str).unique()):
        test_df = df[df[batch_col].astype(str) == batch].copy()
        train_df = df[df[batch_col].astype(str) != batch].copy()
        if test_df.empty or train_df.empty:
            continue
        folds.append((batch, train_df, test_df))
    return folds


def check_no_group_leakage(
    df: pd.DataFrame,
    group_col: str = "group_id",
    split_col: str = "split",
) -> dict:
    """Verify each group appears in exactly one split.

    Returns a report dict with ``is_leaked`` and the offending groups.
    """
    if group_col not in df.columns or split_col not in df.columns:
        raise ValueError(f"Need both {group_col!r} and {split_col!r} columns")
    counts = df.groupby(group_col)[split_col].nunique()
    leaked = counts[counts > 1]
    frames_per_split = df[split_col].value_counts().to_dict()
    return {
        "is_leaked": bool(not leaked.empty),
        "n_groups": int(counts.size),
        "n_leaked_groups": int(leaked.size),
        "leaked_groups": leaked.index.astype(str).tolist(),
        "n_instances_per_split": {str(k): int(v) for k, v in frames_per_split.items()},
    }
