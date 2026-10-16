"""Notebook-2 orchestration: baselines, the hybrid ordinal model and ablations.

Everything the training notebook needs lives here (the notebook is a thin
wrapper).  The label is the geometric-consistency proxy (``proxy_score``)
because SutureBot2 has no expert ratings; the expert-score setting is a
documented placeholder only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import config, detection, evaluation, models, ordinal, training
from .utils import absolutize_path_columns, ensure_dir, write_json

_PATH_COLS = ("frame_path", "mask_path", "bbox_path")
SETTING = "proxy"
_SCORES = list(range(config.SCORE_MIN, config.SCORE_MAX + 1))


def load_geometry_table(path: Path = config.GEOMETRY_FEATURES_CSV) -> pd.DataFrame:
    """Load ``geometry_features.csv`` with absolute media paths."""
    df = pd.read_csv(path)
    df = absolutize_path_columns(df, _PATH_COLS)
    df["crop_path"] = df["stitch_id"].map(lambda s: str(config.CROPS_DIR / f"{s}.png"))
    return df


def crop_paths_for(df: pd.DataFrame) -> list[str]:
    return [str(config.CROPS_DIR / f"{sid}.png") for sid in df["stitch_id"].tolist()]


def _subset_batch(image, geometry, y, idx, variant):
    img = None if variant == "geometry_only" else np.asarray(image)[idx]
    geo = None if variant == "image_only" else np.asarray(geometry)[idx]
    return training.Batch(img, geo, np.asarray(y, dtype=int)[idx])


def _prediction_rows(df, idx, variant, fold, proba, hard, expected, prefix_model):
    subset = df.iloc[idx]
    rows = []
    for row_i, pos in enumerate(idx):
        record = subset.iloc[row_i]
        rows.append(
            {
                "stitch_id": record["stitch_id"],
                "group_id": record["group_id"],
                "batch": record["batch"],
                "episode": int(record["episode"]),
                "split": record.get("split", fold),
                "fold": fold,
                "setting": SETTING,
                "model": prefix_model,
                "y_true": int(record["proxy_score"]),
                "y_pred": int(hard[row_i]),
                "y_pred_continuous": float(expected[row_i]),
                "prob_1": float(proba[row_i, 0]),
                "prob_2": float(proba[row_i, 1]),
                "prob_3": float(proba[row_i, 2]),
                "prob_4": float(proba[row_i, 3]),
                "prob_5": float(proba[row_i, 4]),
                "confidence": float(np.max(proba[row_i])),
                "abs_error": int(abs(int(hard[row_i]) - int(record["proxy_score"]))),
                "within_one": int(abs(int(hard[row_i]) - int(record["proxy_score"])) <= 1),
            }
        )
    return rows


def _metrics_block(y_true, y_pred, y_cont, groups):
    base = evaluation.compute_ordinal_metrics(y_true, y_pred, y_cont)
    base["per_class"] = evaluation.per_class_metrics(y_true, y_pred)
    base["ci"] = evaluation.bootstrap_metrics(y_true, y_pred, groups, n_boot=1000, seed=config.SEED)
    return base


def _fit_and_score_torch(variant, image, geometry, y, train_idx, val_idx, test_idx, device, epochs, seed, fold, df, ensemble=1):
    geo_dim = np.asarray(geometry).shape[1]
    img_dim = np.asarray(image).shape[1]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_batch = _subset_batch(image, geometry, y, train_idx, variant)
    val_batch = _subset_batch(image, geometry, y, val_idx, variant) if val_idx is not None and len(val_idx) else None
    test_batch = _subset_batch(image, geometry, y, test_idx, variant)

    model = None
    result = None
    probas: list[np.ndarray] = []
    for k in range(max(1, int(ensemble))):
        seed_k = seed + 101 * k
        candidate = models.build_head(variant, geometry_dim=geo_dim, image_dim=img_dim, seed=seed_k)
        fit = training.train_torch_model(
            candidate, train_batch, val_batch, epochs=epochs, batch_size=32, seed=seed_k, device=device
        )
        proba_k, _, _ = training.predict_batch(candidate, test_batch, dev)
        probas.append(proba_k)
        if k == 0:
            model, result = candidate, fit

    proba = np.mean(probas, axis=0)
    scores = np.arange(config.SCORE_MIN, config.SCORE_MAX + 1, dtype=float)
    expected = proba @ scores
    hard = np.clip(np.rint(expected), config.SCORE_MIN, config.SCORE_MAX).astype(int)
    rows = _prediction_rows(df, test_idx, variant, fold, proba, hard, expected, variant)
    metrics = _metrics_block(np.asarray(y)[test_idx], hard, expected, df.iloc[test_idx]["group_id"].to_numpy())
    return model, result, rows, metrics


def _fit_and_score_sklearn(kind, geo_matrix, y, train_idx, test_idx, df, fold):
    X_train, X_test = np.asarray(geo_matrix)[train_idx], np.asarray(geo_matrix)[test_idx]
    y_train = np.asarray(y, dtype=int)[train_idx]
    if kind == "forest":
        model = ordinal.OrdinalForest(seed=config.SEED)
        model_name = "geometry_only_forest"
    else:
        model = ordinal.make_ordinal_model("auto", seed=config.SEED)
        model_name = "geometry_only_logistic"
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    scores = np.asarray(_SCORES, dtype=float)
    expected = proba @ scores
    hard = np.clip(np.rint(expected), config.SCORE_MIN, config.SCORE_MAX).astype(int)
    rows = _prediction_rows(df, test_idx, "geometry_only", fold, proba, hard, expected, model_name)
    metrics = _metrics_block(np.asarray(y)[test_idx], hard, expected, df.iloc[test_idx]["group_id"].to_numpy())
    return model, rows, metrics


def run_training_pipeline(
    epochs: int = 80,
    cv_splits: int = 5,
    device: str | None = None,
    seed: int = config.SEED,
    overwrite_embeddings: bool = False,
) -> dict[str, Any]:
    """Train baselines + hybrid model, run ablations, write all Notebook-2 artifacts."""
    ensure_dir(config.MODELS_DIR)
    torch.manual_seed(seed)
    df = load_geometry_table()
    df = df.reset_index(drop=True)
    y = df["proxy_score"].to_numpy(dtype=int)

    # ---- image embeddings (frozen backbone, cached) ------------------------ #
    image_matrix, pretrained = models.precompute_embeddings(
        stitch_ids=df["stitch_id"].tolist(),
        crop_paths=crop_paths_for(df),
        out_path=config.MODELS_DIR / "embeddings.npz",
        overwrite=overwrite_embeddings,
        device=device,
    )

    # ---- geometry preprocessing (fit on TRAIN only) ------------------------ #
    train_positions = np.flatnonzero(df["split"].to_numpy() == "train")
    test_positions = np.flatnonzero(df["split"].to_numpy() == "test")
    val_positions = np.flatnonzero(df["split"].to_numpy() == "val")
    preprocessor = models.GeometryPreprocessor().fit(df.iloc[train_positions])
    geo_matrix = preprocessor.transform(df)
    np.savez(
        config.MODELS_DIR / "geometry_preprocessor.npz",
        median=preprocessor._median,
        mean=preprocessor._mean,
        std=preprocessor._std,
        columns=np.asarray(preprocessor.columns, dtype=object),
    )

    all_rows: list[dict] = []
    model_metrics: dict[str, Any] = {}

    # ---- primary grouped split: geometry-only, image-only, hybrid ---------- #
    grouped_models = {}
    histories: dict[str, dict] = {}
    for variant in ("geometry_only", "image_only", "hybrid"):
        ensemble_n = 6 if variant == "hybrid" else 1
        model, fit_result, rows, metrics = _fit_and_score_torch(
            variant, image_matrix, geo_matrix, y, train_positions, val_positions, test_positions,
            device, epochs, seed, "grouped_test", df, ensemble=ensemble_n,
        )
        histories[variant] = fit_result.history
        grouped_models[variant] = model
        all_rows.extend(rows)
        model_metrics.setdefault(variant, {})["grouped_test"] = metrics
    hybrid_history = histories.get("hybrid")

    for kind in ("forest", "logistic"):
        _model, rows, metrics = _fit_and_score_sklearn(kind, geo_matrix, y, train_positions, test_positions, df, "grouped_test")
        all_rows.extend(rows)
        model_metrics.setdefault(f"geometry_only_{kind}", {})["grouped_test"] = metrics

    # ---- grouped cross-validation (hybrid + geometry-only) ---------------- #
    folds = training.grouped_folds(df, group_col="group_id", n_splits=cv_splits, seed=seed)
    for k, (tr, va) in enumerate(folds, start=1):
        for variant in ("hybrid", "geometry_only"):
            _m, _r, rows, metrics = _fit_and_score_torch(
                variant, image_matrix, geo_matrix, y, tr, va, va, device, epochs, seed, f"cv_{k}", df
            )
            all_rows.extend(rows)
            model_metrics.setdefault(variant, {}).setdefault("cv", []).append(metrics)

    # ---- leave-one-batch-out --------------------------------------------- #
    for batch, tr, te in training.leave_one_batch_out_folds(df, batch_col="batch"):
        for variant in ("hybrid", "geometry_only"):
            _m, _r, rows, metrics = _fit_and_score_torch(
                variant, image_matrix, geo_matrix, y, tr, None, te, device, epochs, seed, f"lobo_{batch}", df
            )
            all_rows.extend(rows)
            model_metrics.setdefault(variant, {}).setdefault("lobo", {})[batch] = metrics
        _m, rows, metrics = _fit_and_score_sklearn("forest", geo_matrix, y, tr, te, df, f"lobo_{batch}")
        all_rows.extend(rows)
        model_metrics.setdefault("geometry_only_forest", {}).setdefault("lobo", {})[batch] = metrics

    # ---- ablation 5: random frame split vs grouped split ------------------ #
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(df))
    n_train = int(0.7 * len(df))
    random_train, random_test = shuffled[:n_train], shuffled[n_train:]
    _m, _r, rows, random_metrics = _fit_and_score_torch(
        "hybrid", image_matrix, geo_matrix, y, random_train, None, random_test, device, epochs, seed, "random_split", df
    )
    all_rows.extend(rows)
    model_metrics.setdefault("hybrid", {})["random_frame_split"] = random_metrics

    # ---- persist predictions / metrics / weights -------------------------- #
    predictions = pd.DataFrame(all_rows)
    predictions.to_csv(config.FOLD_PREDICTIONS_CSV, index=False)

    for variant, model in grouped_models.items():
        torch.save(model.state_dict(), config.MODELS_DIR / f"{variant}.pt")

    detection_report = detection.detection_report(pd.read_csv(config.MANIFEST_ALL_CSV))
    metrics = _summarise(model_metrics, predictions, pretrained, detection_report)
    write_json(config.METRICS_JSON, metrics)

    ablation = _ablation_table(model_metrics)
    ablation.to_csv(config.ABLATION_METRICS_CSV, index=False)

    write_json(config.MODELS_DIR / "histories.json", histories)
    prepared = {
        "df": df,
        "y": y,
        "image_matrix": image_matrix,
        "geo_matrix": geo_matrix,
        "preprocessor": preprocessor,
        "pretrained": pretrained,
        "train_idx": train_positions,
        "val_idx": val_positions,
        "test_idx": test_positions,
    }
    _plot_training_and_confusion(predictions, model_metrics, hybrid_history)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "ablation": ablation,
        "model_metrics": model_metrics,
        "pretrained": pretrained,
        "histories": histories,
        "prepared": prepared,
    }


def prepare_training_data(
    df: pd.DataFrame | None = None,
    device: str | None = None,
    overwrite_embeddings: bool = False,
) -> dict[str, Any]:
    """Load features, precompute frozen embeddings and fit the geometry scaler on train."""
    if df is None:
        df = load_geometry_table()
    df = df.reset_index(drop=True)
    y = df["proxy_score"].to_numpy(dtype=int)
    ensure_dir(config.MODELS_DIR)

    image_matrix, pretrained = models.precompute_embeddings(
        stitch_ids=df["stitch_id"].tolist(),
        crop_paths=crop_paths_for(df),
        out_path=config.MODELS_DIR / "embeddings.npz",
        overwrite=overwrite_embeddings,
        device=device,
    )

    splits = df["split"].to_numpy()
    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")

    preprocessor = models.GeometryPreprocessor().fit(df.iloc[train_idx])
    geo_matrix = preprocessor.transform(df)
    return {
        "df": df,
        "y": y,
        "image_matrix": image_matrix,
        "geo_matrix": geo_matrix,
        "preprocessor": preprocessor,
        "pretrained": pretrained,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def train_and_evaluate_variant(
    variant: str,
    prepared: dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray,
    epochs: int = 60,
    seed: int = config.SEED,
    device: str | None = None,
    fold: str = "grouped_test",
    ensemble: int = 1,
) -> dict[str, Any]:
    """Train one variant on *train_idx* and return its history, rows and metrics."""
    model, fit_result, rows, metrics = _fit_and_score_torch(
        variant,
        prepared["image_matrix"],
        prepared["geo_matrix"],
        prepared["y"],
        train_idx,
        val_idx,
        test_idx,
        device,
        epochs,
        seed,
        fold,
        prepared["df"],
        ensemble=ensemble,
    )
    return {
        "model": model,
        "history": fit_result.history,
        "rows": rows,
        "metrics": metrics,
        "best_val_mae": fit_result.best_val_mae,
    }


def _is_metrics(block) -> bool:
    return isinstance(block, dict) and "mae" in block and "n" in block


def _ablation_table(model_metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, folds in model_metrics.items():
        for fold_name, block in folds.items():
            if _is_metrics(block):
                rows.append({"model": model_name, "fold": fold_name, **_flat(block)})
            elif isinstance(block, list):
                for i, metrics in enumerate(block, start=1):
                    rows.append({"model": model_name, "fold": f"{fold_name}_{i}", **_flat(metrics)})
            elif isinstance(block, dict):
                for key, metrics in block.items():
                    rows.append({"model": model_name, "fold": f"{fold_name}_{key}", **_flat(metrics)})
    return pd.DataFrame(rows)


def _flat(metrics: dict) -> dict:
    keys = ("n", "mae", "rmse", "exact_accuracy", "within_one_accuracy", "quadratic_weighted_kappa", "linear_weighted_kappa", "spearman")
    return {k: metrics.get(k) for k in keys}


def _summarise(model_metrics, predictions, pretrained, detection_report) -> dict:
    def agg(fold_name):
        values = []
        for model_name, folds in model_metrics.items():
            block = folds.get(fold_name)
            if isinstance(block, list):
                values.extend([m.get("mae") for m in block if m.get("mae") is not None])
        return float(np.mean(values)) if values else None

    grouped = model_metrics.get("hybrid", {}).get("grouped_test", {})
    return {
        "primary_setting": SETTING,
        "label_source": "geometric_consistency_proxy",
        "clinical_caveat": config.CLINICAL_CAVEAT,
        "realistic_target": ">=80% within-one accuracy (never 80-90% exact five-class accuracy)",
        "encoder": {"arch": "resnet18", "pretrained": bool(pretrained)},
        "models": model_metrics,
        "detection": detection_report,
        "n_predictions": int(len(predictions)),
        "headline_grouped_test_hybrid": _flat(grouped) if grouped else None,
        "notes": (
            "Expert-score setting is a skipped placeholder because SutureBot2 has no expert 1-5 ratings. "
            "All reported scores are the geometric-consistency proxy. Human inter-rater agreement is the ceiling."
        ),
    }


def _plot_training_and_confusion(predictions: pd.DataFrame, model_metrics: dict, hybrid_history: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(config.FIGURES_DIR)
    grouped = predictions[(predictions["fold"] == "grouped_test") & (predictions["model"] == "hybrid")]
    if grouped.empty:
        return

    cm, err = evaluation.confusion_and_error_matrix(grouped["y_true"].to_numpy(), grouped["y_pred"].to_numpy())
    for matrix, title, filename in (
        (cm, "Confusion matrix (hybrid, grouped test)", "confusion_matrix.png"),
        (err, "Ordinal error matrix: |pred-true| x count", "ordinal_error_matrix.png"),
    ):
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(_SCORES)), _SCORES)
        ax.set_yticks(range(len(_SCORES)), _SCORES)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("True proxy score")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / filename, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if hybrid_history and hybrid_history.get("train_loss"):
        epochs = np.arange(1, len(hybrid_history["train_loss"]) + 1)
        ax.plot(epochs, hybrid_history["train_loss"], label="train loss (CORAL BCE)")
        twin = ax.twinx()
        twin.plot(epochs, hybrid_history["val_mae"], color="#d1495b", label="validation MAE")
        twin.set_ylabel("validation MAE", color="#d1495b")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
        ax.set_title("Hybrid model training curve")
    else:
        ax.text(0.5, 0.5, "training history unavailable", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "training_curves.png", dpi=130)
    plt.close(fig)
