"""Training loop and grouped cross-validation folds for the CORAL networks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config
from .models import coral_loss, coral_probabilities


@dataclass
class Batch:
    image: np.ndarray | None
    geometry: np.ndarray | None
    y: np.ndarray


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_state: dict | None = None
    best_val_mae: float = float("inf")


def _to_tensor(array: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if array is None:
        return None
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)


def _forward(model: nn.Module, batch: Batch, device: torch.device) -> torch.Tensor:
    return model(_to_tensor(batch.image, device), _to_tensor(batch.geometry, device))


@torch.no_grad()
def predict_batch(model: nn.Module, batch: Batch, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(probabilities, hard_predictions, expected_values)``."""
    model.eval()
    logits = _forward(model, batch, device)
    proba = coral_probabilities(logits).cpu().numpy()
    scores = np.arange(config.SCORE_MIN, config.SCORE_MAX + 1, dtype=float)
    expected = proba @ scores
    hard = np.clip(np.rint(expected), config.SCORE_MIN, config.SCORE_MAX).astype(int)
    return proba, hard, expected


def _mae_from_expected(expected: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(expected - np.asarray(y, dtype=float)))) if y.size else float("nan")


def train_torch_model(
    model: nn.Module,
    train: Batch,
    val: Batch | None = None,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    patience: int = 12,
    seed: int = config.SEED,
    device: str | None = None,
    class_balanced: bool = True,
) -> TrainResult:
    """Fit a CORAL network with early stopping on (validation) MAE."""
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    n = len(train.y)

    sampler = None
    if class_balanced and n > 0:
        counts = np.bincount(np.asarray(train.y, dtype=int), minlength=config.SCORE_MAX + 1).astype(float)
        weights_per_class = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        sample_weights = weights_per_class[np.asarray(train.y, dtype=int)]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double), num_samples=n, replacement=True
        )

    history: dict[str, list[float]] = {"train_loss": [], "val_mae": []}
    result = TrainResult(history=history)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    epochs_without_improvement = 0

    order = np.arange(n)
    rng = np.random.default_rng(seed)

    for _epoch in range(epochs):
        model.train()
        if sampler is not None:
            epoch_indices = list(iter(sampler))
        else:
            rng.shuffle(order)
            epoch_indices = list(order)

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(epoch_indices), batch_size):
            chunk = epoch_indices[start:start + batch_size]
            images = train.image[chunk] if train.image is not None else None
            geometry = train.geometry[chunk] if train.geometry is not None else None
            batch = Batch(images, geometry, np.asarray(train.y, dtype=int)[chunk])
            optimizer.zero_grad()
            logits = _forward(model, batch, dev)
            loss = coral_loss(logits, torch.as_tensor(batch.y, dtype=torch.long, device=dev))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

        monitor = val if val is not None else train
        _, _, expected = predict_batch(model, monitor, dev)
        monitor_mae = _mae_from_expected(expected, np.asarray(monitor.y, dtype=float))
        history["val_mae"].append(monitor_mae)

        if monitor_mae < best_val - 1e-4:
            best_val = monitor_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    result.best_state = best_state
    result.best_val_mae = best_val
    model.load_state_dict(best_state)
    return result


def grouped_folds(
    df: pd.DataFrame,
    group_col: str = "group_id",
    n_splits: int = 5,
    seed: int = config.SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic grouped folds: returns positional ``(train_idx, val_idx)``."""
    groups = np.asarray(df[group_col].astype(str).to_numpy())
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, min(n_splits, unique.size))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk in chunks:
        val_mask = np.isin(groups, chunk)
        folds.append((np.flatnonzero(~val_mask), np.flatnonzero(val_mask)))
    return folds


def leave_one_batch_out_folds(df: pd.DataFrame, batch_col: str = "batch") -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return ``(held_out_batch, train_idx, test_idx)`` positional folds."""
    batches = np.asarray(df[batch_col].astype(str).to_numpy())
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for batch in sorted(set(batches)):
        test_idx = np.flatnonzero(batches == batch)
        train_idx = np.flatnonzero(batches != batch)
        if test_idx.size and train_idx.size:
            folds.append((batch, train_idx, test_idx))
    return folds
