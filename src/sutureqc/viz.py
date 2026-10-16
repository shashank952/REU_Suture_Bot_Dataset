"""Notebook-3 orchestration: publication figures and the clinician feedback.

Every figure is produced by a standalone function so the notebook stays thin
and any figure can be regenerated on its own.  Model-dependent figures degrade
gracefully when ``artifacts/fold_predictions.csv`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import sys

import cv2
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, evaluation, feedback, masks, modeling, proxy  # noqa: E402
from .geometry import GEOMETRY_FEATURE_COLUMNS  # noqa: E402
from .utils import ensure_dir  # noqa: E402

SCORES = list(range(config.SCORE_MIN, config.SCORE_MAX + 1))
PRIMARY_MODEL = "hybrid"
PRIMARY_FOLD = "grouped_test"
CAPTION = config.CLINICAL_CAVEAT


def _figure_path(name: str) -> Path:
    return config.FIGURES_DIR / name


def load_predictions() -> pd.DataFrame | None:
    if not config.FOLD_PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(config.FOLD_PREDICTIONS_CSV)


def _primary(predictions: pd.DataFrame | None) -> pd.DataFrame | None:
    if predictions is None:
        return None
    subset = predictions[(predictions["fold"] == PRIMARY_FOLD) & (predictions["model"] == PRIMARY_MODEL)]
    return subset if not subset.empty else None


def _crop(stitch_id: str) -> np.ndarray:
    image = cv2.imread(str(config.CROPS_DIR / f"{stitch_id}.png"), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((config.CROP_SIZE, config.CROP_SIZE, 3), dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _caption(fig, text: str = CAPTION) -> None:
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=6, color="#555555", wrap=True)


# --------------------------------------------------------------------------- #
# 1. Frame with boxes
# --------------------------------------------------------------------------- #
def plot_frame_with_boxes(geo: pd.DataFrame, out: Path | None = None, n_frames: int = 6) -> Path:
    out = out or _figure_path("fig01_frame_with_boxes.png")
    counts = geo.groupby("basename").size().sort_values(ascending=False)
    chosen = list(counts.index[:n_frames])
    cols = 3
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.7))
    axes = np.atleast_1d(axes).ravel()
    for ax, basename in zip(axes, chosen):
        subset = geo[geo["basename"] == basename]
        image = cv2.imread(subset.iloc[0]["frame_path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i, record in enumerate(subset.to_dict("records"), start=1):
            x1, y1, x2, y2 = int(record["x1"]), int(record["y1"]), int(record["x2"]), int(record["y2"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, str(i), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        ax.imshow(image)
        ax.set_title(f"{subset.iloc[0]['group_id']} (n={len(subset)})", fontsize=8)
        ax.axis("off")
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("1. Frames with stitch bounding boxes (blue)", fontsize=10)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 2. Stitch crops with predicted scores
# --------------------------------------------------------------------------- #
def plot_stitch_crops_with_scores(primary: pd.DataFrame, out: Path | None = None, n: int = 12) -> Path:
    out = out or _figure_path("fig02_stitch_crops_with_scores.png")
    sample = primary.sample(min(n, len(primary)), random_state=config.SEED)
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.1))
    axes = np.atleast_1d(axes).ravel()
    for ax, record in zip(axes, sample.to_dict("records")):
        ax.imshow(_crop(record["stitch_id"]))
        ax.set_title(
            f"true {int(record['y_true'])} / pred {int(record['y_pred'])}\nconf {float(record['confidence']):.2f}",
            fontsize=7,
        )
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.suptitle("2. Stitch crops with predicted 1-5 score and confidence", fontsize=10)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 3. Wound mask overlay
# --------------------------------------------------------------------------- #
def plot_wound_mask_overlay(geo: pd.DataFrame, out: Path | None = None, n_frames: int = 4) -> Path:
    out = out or _figure_path("fig03_wound_mask_overlay.png")
    chosen = list(geo["basename"].drop_duplicates()[:n_frames])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.ravel()
    for ax, basename in zip(axes, chosen):
        subset = geo[geo["basename"] == basename]
        image = cv2.cvtColor(cv2.imread(subset.iloc[0]["frame_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = masks.load_mask(subset.iloc[0]["mask_path"])
        if mask.any():
            overlay = image.copy()
            overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
            image = overlay
        for record in subset.to_dict("records"):
            x1, y1, x2, y2 = int(record["x1"]), int(record["y1"]), int(record["x2"]), int(record["y2"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(image, (int(round(record["cx"])), int(round(record["cy"]))), 3, (255, 255, 0), -1)
        ax.imshow(image)
        ax.set_title(subset.iloc[0]["group_id"], fontsize=8)
        ax.axis("off")
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("3. Wound mask (green) and stitch locations (blue box, yellow centre)", fontsize=10)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 4. Ground truth vs predicted
# --------------------------------------------------------------------------- #
def plot_gt_vs_pred(primary: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or _figure_path("fig04_gt_vs_pred.png")
    table = pd.crosstab(primary["y_true"], primary["y_pred"]).reindex(index=SCORES, columns=SCORES, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(table.to_numpy(), cmap="Purples")
    ax.set_xticks(range(len(SCORES)), SCORES)
    ax.set_yticks(range(len(SCORES)), SCORES)
    ax.set_xlabel("Predicted proxy score")
    ax.set_ylabel("True proxy score")
    for i in range(len(SCORES)):
        for j in range(len(SCORES)):
            ax.text(j, i, int(table.to_numpy()[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("4. Ground-truth vs predicted score (hybrid, grouped test)")
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 5. Confusion + ordinal error (publication)
# --------------------------------------------------------------------------- #
def plot_confusion_and_error_matrix(primary: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or _figure_path("fig05_confusion_and_error_matrix.png")
    cm, err = evaluation.confusion_and_error_matrix(primary["y_true"].to_numpy(), primary["y_pred"].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, matrix, title in zip(axes, (cm, err), ("Confusion matrix", "Ordinal error matrix (|pred-true| x count)")):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(SCORES)), SCORES)
        ax.set_yticks(range(len(SCORES)), SCORES)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("True proxy score")
        ax.set_title(title)
        for i in range(len(SCORES)):
            for j in range(len(SCORES)):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("5. Confusion and ordinal error (hybrid, grouped test)", fontsize=10)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 6. Metrics by fold
# --------------------------------------------------------------------------- #
def plot_metrics_by_fold(ablation: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or _figure_path("fig06_metrics_by_fold.png")
    subset = ablation[ablation["model"] == PRIMARY_MODEL].copy()
    subset = subset[subset["fold"].str.startswith(("lobo_", "cv_")) | (subset["fold"] == PRIMARY_FOLD)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, metric, label in zip(axes, ("mae", "quadratic_weighted_kappa", "spearman"),
                                 ("MAE (lower better)", "Quadratic weighted kappa", "Spearman")):
        ax.bar(range(len(subset)), subset[metric].to_numpy(), color="#3b6ea5")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset["fold"].tolist(), rotation=45, ha="right", fontsize=7)
    axes[0].set_title("6. Hybrid performance by fold (grouped test, grouped CV, leave-one-batch-out)")
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 7. Score distribution by batch and action
# --------------------------------------------------------------------------- #
def plot_score_distribution_by_batch_and_action(geo: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or _figure_path("fig07_score_distribution_by_batch_action.png")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    by_batch = pd.crosstab(geo["batch"], geo["proxy_score"]).reindex(columns=SCORES, fill_value=0)
    by_batch.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis")
    axes[0].set_title("Proxy-score distribution by batch")
    axes[0].set_ylabel("stitch instances")
    axes[0].legend(title="score", fontsize=7)
    top_actions = geo["action_label"].fillna("unknown").value_counts().head(8).index
    by_action = pd.crosstab(geo["action_label"].where(geo["action_label"].isin(top_actions), "other"), geo["proxy_score"])
    by_action = by_action.reindex(columns=SCORES, fill_value=0)
    by_action.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("Proxy-score distribution by semantic subtask")
    axes[1].set_ylabel("stitch instances")
    axes[1].legend(title="score", fontsize=7)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 8. Ablation comparison
# --------------------------------------------------------------------------- #
def plot_ablation_comparison(ablation: pd.DataFrame, out: Path | None = None) -> Path:
    out = out or _figure_path("fig08_ablation_comparison.png")
    subset = ablation[ablation["fold"] == PRIMARY_FOLD].copy()
    order = ["geometry_only_logistic", "geometry_only_forest", "geometry_only", "image_only", "hybrid"]
    subset = subset[subset["model"].isin(order)]
    subset["model"] = pd.Categorical(subset["model"], categories=[m for m in order if m in set(subset["model"])], ordered=True)
    subset = subset.sort_values("model")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(subset["model"].astype(str), subset["mae"], color="#3b6ea5")
    axes[0].set_ylabel("MAE (lower better)")
    axes[0].set_title("Error")
    axes[1].bar(subset["model"].astype(str), subset["within_one_accuracy"], color="#5a9367")
    axes[1].axhline(0.8, color="red", linestyle="--", linewidth=1, label="80% target")
    axes[1].set_ylabel("within-one accuracy (higher better)")
    axes[1].set_title("Within-one accuracy")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("8. Ablation on the grouped test split (proxy-label setting)", fontsize=10)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 9. Calibration
# --------------------------------------------------------------------------- #
def plot_calibration(primary: pd.DataFrame, out: Path | None = None, n_bins: int = 5) -> Path:
    out = out or _figure_path("fig09_calibration.png")
    frame = primary.copy()
    frame["correct"] = (frame["y_true"] == frame["y_pred"]).astype(int)
    frame["bin"] = pd.cut(frame["confidence"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(mean_conf=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size"))
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.plot(grouped["mean_conf"], grouped["accuracy"], "o-", color="#3b6ea5", label="model")
    for x, y, n in zip(grouped["mean_conf"], grouped["accuracy"], grouped["n"]):
        ax.annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed exact accuracy")
    ax.set_title("9. Calibration (hybrid, grouped test)")
    ax.legend(fontsize=8)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 10. Failure gallery
# --------------------------------------------------------------------------- #
def plot_failure_gallery(primary: pd.DataFrame, out: Path | None = None, n: int = 8) -> Path:
    out = out or _figure_path("fig10_failure_gallery.png")
    failures = primary.sort_values("abs_error", ascending=False).head(n)
    cols = 4
    rows = int(np.ceil(len(failures) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, record in zip(axes, failures.to_dict("records")):
        ax.imshow(_crop(record["stitch_id"]))
        tag = "batch shift" if str(record["batch"]) == "lerobot_jesse_batch_0" else "hard case"
        ax.set_title(
            f"true {int(record['y_true'])} / pred {int(record['y_pred'])}\n"
            f"err {int(record['abs_error'])} | {tag}",
            fontsize=7,
        )
        ax.axis("off")
    for ax in axes[len(failures):]:
        ax.axis("off")
    fig.suptitle("10. Failure gallery: worst ordinal errors (occlusion / ambiguous bbox / batch shift)", fontsize=10)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 11. Occlusion attention map (optional, best-effort)
# --------------------------------------------------------------------------- #
def gradcam_or_occlusion_map(primary: pd.DataFrame, out: Path | None = None, n_examples: int = 4, grid: int = 6) -> Path | None:
    """Occlusion-sensitivity maps for the image branch.

    This is *model attention*, not causal evidence.  Returns ``None`` (and
    writes nothing) when torch / the image-only checkpoint is unavailable.
    """
    out = out or _figure_path("fig11_attention_map.png")
    try:
        import torch

        from . import models

        checkpoint = config.MODELS_DIR / "image_only.pt"
        stats = np.load(config.MODELS_DIR / "geometry_preprocessor.npz", allow_pickle=True)
        if not checkpoint.exists():
            return None
        embedder = models.ImageEmbedder()
        model = models.build_head("image_only", geometry_dim=len(stats["columns"]), image_dim=embedder.embedding_dim)
        model.load_state_dict(torch.load(checkpoint, map_location=embedder.device))
        model = model.to(embedder.device)
        model.eval()
    except Exception:
        return None

    examples = primary.sort_values("confidence", ascending=False).head(n_examples)
    fig, axes = plt.subplots(2, len(examples), figsize=(len(examples) * 3.0, 6.0))
    axes = np.atleast_2d(axes)

    for col, record in enumerate(examples.to_dict("records")):
        crop = cv2.imread(str(config.CROPS_DIR / f"{record['stitch_id']}.png"), cv2.IMREAD_COLOR)
        if crop is None:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        def probability(image_rgb: np.ndarray) -> float:
            tensor = embedder.transform(image_rgb).unsqueeze(0).to(embedder.device)
            with torch.no_grad():
                features = embedder.backbone(tensor)
                logits = model(features, None)
                proba = models.coral_probabilities(logits).cpu().numpy()[0]
            return float(proba[int(record["y_pred"]) - config.SCORE_MIN])

        baseline = probability(rgb)
        occlusion = np.zeros((grid, grid), dtype=float)
        size = crop.shape[0]
        step = size // grid
        for gi in range(grid):
            for gj in range(grid):
                occluded = rgb.copy()
                occluded[gi * step:(gi + 1) * step, gj * step:(gj + 1) * step] = 128
                occlusion[gi, gj] = baseline - probability(occluded)

        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"pred {int(record['y_pred'])} conf {float(record['confidence']):.2f}", fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(occlusion, cmap="magma")
        axes[1, col].set_title("drop in predicted prob.", fontsize=7)
        axes[1, col].axis("off")

    fig.suptitle("11. Occlusion sensitivity of the image branch (model attention, NOT causal evidence)", fontsize=9)
    _caption(fig, CAPTION + " This map shows which regions the model relies on; it is not causal evidence.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Orchestrator + feedback examples
# --------------------------------------------------------------------------- #
def write_feedback_examples(primary: pd.DataFrame, geo: pd.DataFrame, out: Path | None = None, n: int = 6) -> Path:
    out = out or _figure_path("feedback_examples.txt")
    merged = primary.merge(
        geo[["stitch_id", "bbox_wound_overlap_ratio", "dist_center_to_wound_boundary_norm", "spacing_cv", "ie_crossing_score"]],
        on="stitch_id",
        how="left",
    ).head(n)
    blocks = []
    for record in merged.to_dict("records"):
        evidence = proxy.proxy_evidence(record, None)
        review = feedback.review_decision(float(record["confidence"]), in_distribution=True)
        blocks.append(feedback.format_feedback(int(record["y_pred"]), float(record["confidence"]), evidence, review))
    header = (
        "Stitch-level feedback examples\n"
        "==============================\n"
        "Where is the stitch? -> the provided bounding box.\n"
        "How good is the stitch? -> the geometric-consistency proxy score below.\n"
        f"{CAPTION}\n\n"
    )
    out.write_text(header + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return out


def _maybe_save(fig, out: Path | None, default_name: str):
    out = out or _figure_path(default_name)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


def _emit(produced: dict, key: str, name: str, func, *args, **kwargs) -> None:
    path = _figure_path(name)
    try:
        result = func(*args, out=path, **kwargs)
    except Exception as exc:
        print(f"[viz] {key} ({name}) skipped: {exc}")
        return
    if result is None:
        return
    try:
        plt.close(result)
    except Exception:
        pass
    if path.exists():
        produced[key] = str(path)


def _load_histories():
    import json

    path = config.MODELS_DIR / "histories.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_visualization_pipeline() -> dict:
    """Produce every Notebook-3 figure, degrading gracefully without predictions."""
    ensure_dir(config.FIGURES_DIR)
    geo = modeling.load_geometry_table()
    predictions = load_predictions()
    primary = _primary(predictions)
    produced: dict[str, str] = {}

    _emit(produced, "fig03", "fig03_wound_mask_overlay.png", plot_wound_mask_overlay, geo)
    _emit(produced, "fig07", "fig07_score_distribution_by_batch_action.png", plot_score_distribution_by_batch_and_action, geo)
    _emit(produced, "fig17", "fig17_geometry_correlation_heatmap.png", plot_geometry_correlation_heatmap, geo)
    _emit(produced, "fig19", "fig19_batch_score_heatmap.png", plot_batch_score_heatmap, geo)
    _emit(produced, "fig22", "fig22_embedding_projection.png", plot_embedding_projection, features=geo)
    _emit(produced, "fig23", "fig23_feature_importance.png", plot_feature_importance)

    if primary is None:
        return {"produced": produced, "predictions_available": False}

    ablation = pd.read_csv(config.ABLATION_METRICS_CSV)
    _emit(produced, "fig01", "fig01_frame_with_boxes.png", plot_frame_with_boxes, geo)
    _emit(produced, "fig02", "fig02_stitch_crops_with_scores.png", plot_stitch_crops_with_scores, primary)
    _emit(produced, "fig04", "fig04_gt_vs_pred.png", plot_gt_vs_pred, primary)
    _emit(produced, "fig05", "fig05_confusion_and_error_matrix.png", plot_confusion_and_error_matrix, primary)
    _emit(produced, "fig06", "fig06_metrics_by_fold.png", plot_metrics_by_fold, ablation)
    _emit(produced, "fig08", "fig08_ablation_comparison.png", plot_ablation_comparison, ablation)
    _emit(produced, "fig09", "fig09_calibration.png", plot_calibration, primary)
    _emit(produced, "fig10", "fig10_failure_gallery.png", plot_failure_gallery, primary)
    _emit(produced, "fig11", "fig11_attention_map.png", gradcam_or_occlusion_map, primary)
    _emit(produced, "fig12", "fig12_roc_curves.png", plot_roc_curves, primary)
    _emit(produced, "fig13", "fig13_binary_roc.png", plot_binary_roc, primary)
    _emit(produced, "fig14", "fig14_precision_recall.png", plot_precision_recall, primary)
    _emit(produced, "fig15", "fig15_true_vs_pred_scatter.png", plot_true_vs_pred_scatter, primary)
    _emit(produced, "fig16", "fig16_confidence_vs_error.png", plot_confidence_vs_error, primary)
    _emit(produced, "fig18", "fig18_per_class_metrics.png", plot_per_class_metrics, primary)
    _emit(produced, "fig20", "fig20_probability_heatmap.png", plot_probability_heatmap, primary)
    _emit(produced, "fig21", "fig21_residual_histogram.png", plot_residual_histogram, primary)
    histories = _load_histories()
    if histories:
        _emit(produced, "fig24", "fig24_training_curves.png", plot_training_curves, histories)
    produced["feedback"] = str(write_feedback_examples(primary, geo))
    return {"produced": produced, "predictions_available": True, "n_primary": len(primary)}


def plot_roc_curves(primary: pd.DataFrame, out: Path | None = None):
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    aucs: list[float] = []
    for score in SCORES:
        y_bin = (primary["y_true"].to_numpy() == score).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, primary[f"prob_{score}"].to_numpy())
        value = auc(fpr, tpr)
        aucs.append(value)
        ax.plot(fpr, tpr, linewidth=1.4, label=f"score {score} (AUC={value:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    macro = float(np.mean(aucs)) if aucs else float("nan")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"One-vs-rest ROC (macro AUC={macro:.3f})")
    ax.legend(fontsize=7, loc="lower right")
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _maybe_save(fig, out, "fig12_roc_curves.png")


def plot_binary_roc(primary: pd.DataFrame, out: Path | None = None, threshold: int = 3):
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    y_bin = (primary["y_true"].to_numpy() >= threshold).astype(int)
    good_cols = [f"prob_{s}" for s in SCORES if s >= threshold]
    p_good = primary[good_cols].sum(axis=1).to_numpy()
    if len(np.unique(y_bin)) >= 2:
        fpr, tpr, _ = roc_curve(y_bin, p_good)
        value = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#3b6ea5", label=f"acceptable (>= {threshold}) AUC={value:.2f}")
    else:
        ax.text(0.5, 0.5, "single class only", ha="center", va="center")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Binary ROC: good (>= {threshold}) vs poor")
    ax.legend(fontsize=8, loc="lower right")
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _maybe_save(fig, out, "fig13_binary_roc.png")


def plot_precision_recall(primary: pd.DataFrame, out: Path | None = None):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    aps: list[float] = []
    for score in SCORES:
        y_bin = (primary["y_true"].to_numpy() == score).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, primary[f"prob_{score}"].to_numpy())
        ap = average_precision_score(y_bin, primary[f"prob_{score}"].to_numpy())
        aps.append(ap)
        ax.plot(recall, precision, linewidth=1.4, label=f"score {score} (AP={ap:.2f})")
    macro = float(np.mean(aps)) if aps else float("nan")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"One-vs-rest precision-recall (macro AP={macro:.3f})")
    ax.legend(fontsize=7, loc="lower left")
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _maybe_save(fig, out, "fig14_precision_recall.png")


def plot_true_vs_pred_scatter(primary: pd.DataFrame, out: Path | None = None):
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    rng = np.random.default_rng(config.SEED)
    x = primary["y_true"].to_numpy(dtype=float) + rng.uniform(-0.14, 0.14, size=len(primary))
    y = primary["y_pred_continuous"].to_numpy(dtype=float)
    scatter = ax.scatter(
        x, y,
        c=primary["abs_error"].to_numpy(),
        cmap="RdYlGn_r",
        s=16 + 70 * primary["confidence"].to_numpy(),
        edgecolor="k",
        linewidth=0.2,
        alpha=0.85,
    )
    ax.plot([1, 5], [1, 5], "k--", linewidth=1)
    rho, _ = spearmanr(primary["y_true"].to_numpy(), primary["y_pred"].to_numpy())
    within = float(primary["within_one"].mean())
    ax.set_xlabel("true proxy score (jittered)")
    ax.set_ylabel("predicted expected score")
    ax.set_title(f"True vs predicted (Spearman={rho:.3f}, within-one={within:.1%})")
    fig.colorbar(scatter, ax=ax, fraction=0.046, label="absolute error")
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _maybe_save(fig, out, "fig15_true_vs_pred_scatter.png")


def plot_confidence_vs_error(primary: pd.DataFrame, out: Path | None = None):
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    rng = np.random.default_rng(config.SEED)
    confidence = primary["confidence"].to_numpy(dtype=float)
    abs_error = primary["abs_error"].to_numpy(dtype=float)
    ax.scatter(confidence, abs_error + rng.uniform(-0.12, 0.12, size=len(primary)), s=16, alpha=0.5, color="#3b6ea5")
    edges = np.linspace(confidence.min(), confidence.max(), 6) if len(confidence) else np.array([0.0, 1.0])
    bucket = np.digitize(confidence, edges[1:-1])
    centers, means = [], []
    for b in range(len(edges) - 1):
        mask = bucket == b
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            means.append(float(abs_error[mask].mean()))
    ax.plot(centers, means, "o-", color="#d1495b", label="mean |error| per bin")
    ax.set_xlabel("prediction confidence")
    ax.set_ylabel("absolute ordinal error")
    ax.set_title("Confidence vs error")
    ax.legend(fontsize=8)
    _caption(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _maybe_save(fig, out, "fig16_confidence_vs_error.png")


def plot_geometry_correlation_heatmap(geo: pd.DataFrame, out: Path | None = None):
    columns = [c for c in list(GEOMETRY_FEATURE_COLUMNS) + ["proxy_score_continuous"] if c in geo.columns]
    frame = geo[columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[:, frame.notna().any(axis=0)]
    corr = frame.corr()
    size = len(corr)
    fig, ax = plt.subplots(figsize=(0.42 * size + 3, 0.42 * size + 2.6))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(size), corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(size), corr.columns, fontsize=6)
    values = corr.to_numpy()
    for i in range(size):
        for j in range(size):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Geometry feature correlation")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig17_geometry_correlation_heatmap.png")


def plot_per_class_metrics(primary: pd.DataFrame, out: Path | None = None):
    per_class = evaluation.per_class_metrics(primary["y_true"].to_numpy(), primary["y_pred"].to_numpy())
    rows = ["precision", "recall", "f1"]
    matrix = np.array([[per_class[str(s)][r] for s in SCORES] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(SCORES)), SCORES)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03)
    ax.set_title("Per-class precision / recall / F1")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig18_per_class_metrics.png")


def plot_batch_score_heatmap(geo: pd.DataFrame, out: Path | None = None, normalize: bool = True):
    table = pd.crosstab(geo["batch"], geo["proxy_score"]).reindex(columns=SCORES, fill_value=0)
    counts = table.to_numpy(dtype=float)
    shown = counts / counts.sum(axis=1, keepdims=True) if normalize else counts
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    image = ax.imshow(shown, cmap="viridis")
    ax.set_xticks(range(len(SCORES)), SCORES)
    ax.set_yticks(range(len(table)), table.index, fontsize=7)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            label = f"{shown[i, j]:.2f}" if normalize else f"{int(counts[i, j])}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xlabel("proxy score")
    ax.set_title("Batch x score composition")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig19_batch_score_heatmap.png")


def plot_probability_heatmap(primary: pd.DataFrame, out: Path | None = None):
    frame = primary.sort_values(["y_true", "y_pred_continuous"]).reset_index(drop=True)
    probs = frame[[f"prob_{s}" for s in SCORES]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    image = ax.imshow(probs, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(SCORES)), SCORES)
    ax.set_xlabel("score")
    ax.set_ylabel("stitch instances (sorted by true score)")
    ax.set_title("Predicted probability per instance")
    fig.colorbar(image, ax=ax, fraction=0.046)
    boundary = np.cumsum([(frame["y_true"] == s).sum() for s in SCORES])
    for b in boundary[:-1]:
        ax.axhline(b - 0.5, color="cyan", linewidth=0.8)
    fig.tight_layout()
    return _maybe_save(fig, out, "fig20_probability_heatmap.png")


def plot_residual_histogram(primary: pd.DataFrame, out: Path | None = None):
    residuals = primary["y_pred"].to_numpy(dtype=int) - primary["y_true"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    values = np.arange(int(residuals.min()), int(residuals.max()) + 1)
    counts = [int((residuals == v).sum()) for v in values]
    ax.bar(values, counts, color="#3b6ea5")
    ax.set_xticks(values)
    ax.set_xlabel("predicted - true score")
    ax.set_ylabel("count")
    ax.set_title("Residual (ordinal error) histogram")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig21_residual_histogram.png")


def plot_embedding_projection(embeddings_path: Path | None = None, features: pd.DataFrame | None = None, out: Path | None = None, method: str = "pca"):
    embeddings_path = embeddings_path or (config.MODELS_DIR / "embeddings.npz")
    if not Path(embeddings_path).exists():
        return None
    data = np.load(embeddings_path, allow_pickle=True)
    matrix = data["embeddings"].astype(np.float32)
    ids = [str(v) for v in data["stitch_ids"]]
    labels = None
    if features is not None and "stitch_id" in features.columns:
        mapping = dict(zip(features["stitch_id"].astype(str), features["proxy_score"]))
        labels = np.array([mapping.get(i, np.nan) for i in ids], dtype=float)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE

            coords = TSNE(n_components=2, random_state=config.SEED, init="pca", perplexity=min(30, max(5, len(matrix) // 5))).fit_transform(matrix)
        except Exception:
            method = "pca"
    if method != "tsne":
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=config.SEED).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels if labels is not None else "#3b6ea5", cmap="viridis", s=16, alpha=0.8)
    if labels is not None:
        fig.colorbar(scatter, ax=ax, fraction=0.046, label="proxy score")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title("Projection of frozen ResNet18 stitch embeddings")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig22_embedding_projection.png")


def plot_feature_importance(out: Path | None = None):
    if not config.GEOMETRY_FEATURES_CSV.exists():
        return None
    geo = pd.read_csv(config.GEOMETRY_FEATURES_CSV)
    if "split" not in geo.columns or "proxy_score" not in geo.columns:
        return None
    train = geo[geo["split"] == "train"]
    columns = [c for c in GEOMETRY_FEATURE_COLUMNS if c in train.columns]
    if not columns or train.empty:
        return None
    matrix = train[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median())
    target = train["proxy_score"].to_numpy(dtype=int)
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=300, random_state=config.SEED, class_weight="balanced", n_jobs=-1)
    model.fit(matrix.to_numpy(), target)
    order = np.argsort(model.feature_importances_)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    ax.barh([columns[i] for i in order][::-1], model.feature_importances_[order][::-1], color="#5a9367")
    ax.set_xlabel("RandomForest importance")
    ax.set_title("Geometry feature importance (explainability)")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig23_feature_importance.png")


def plot_training_curves(histories: dict, out: Path | None = None):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    twin = ax.twinx()
    colors = {"hybrid": "#3b6ea5", "image_only": "#5a9367", "geometry_only": "#d1495b"}
    for variant, history in (histories or {}).items():
        if not history or not history.get("train_loss"):
            continue
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        color = colors.get(variant)
        ax.plot(epochs, history["train_loss"], color=color, label=f"{variant} train loss")
        if history.get("val_mae"):
            twin.plot(epochs, history["val_mae"], color=color, linestyle="--", label=f"{variant} val MAE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (CORAL BCE)")
    twin.set_ylabel("validation MAE")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper right")
    ax.set_title("Training curves")
    fig.tight_layout()
    return _maybe_save(fig, out, "fig24_training_curves.png")
