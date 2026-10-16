"""Central configuration for the stitch-level suturing quality project.

All paths, constants, rubric text and proxy-score weights live here so the
three notebooks never hard-code them.  Nothing in this module imports heavy
libraries, so it is safe to import from anywhere.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_ROOT: Path = REPO_ROOT / "SutureBot2"
ANNOT_DIR: Path = DATA_ROOT / "annotation_images"
MASKS_DIR: Path = ANNOT_DIR / "masks"
BBOX_DIR: Path = ANNOT_DIR / "stitch_bboxes"
MANIFEST_CSV: Path = MASKS_DIR / "manifest.csv"

ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
FIGURES_DIR: Path = REPO_ROOT / "figures"
CROPS_DIR: Path = ARTIFACTS_DIR / "crops"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"

# Notebook-1 artifact paths (single source of truth).
MANIFEST_ALL_CSV: Path = ARTIFACTS_DIR / "manifest_all.csv"
STITCH_INSTANCES_CSV: Path = ARTIFACTS_DIR / "stitch_instances.csv"
GEOMETRY_FEATURES_CSV: Path = ARTIFACTS_DIR / "geometry_features.csv"
EXPERT_RATING_TEMPLATE_CSV: Path = ARTIFACTS_DIR / "expert_rating_template.csv"
SPLITS_GROUPED_CSV: Path = ARTIFACTS_DIR / "splits_grouped.csv"
AUDIT_REPORT_JSON: Path = ARTIFACTS_DIR / "audit_report.json"
SAMPLE_ANNOTATIONS_PNG: Path = FIGURES_DIR / "sample_annotations.png"

# Notebook-2 / Notebook-3 artifact paths.
FOLD_PREDICTIONS_CSV: Path = ARTIFACTS_DIR / "fold_predictions.csv"
METRICS_JSON: Path = ARTIFACTS_DIR / "metrics.json"
ABLATION_METRICS_CSV: Path = ARTIFACTS_DIR / "ablation_metrics.csv"
PROXY_THRESHOLDS_JSON: Path = ARTIFACTS_DIR / "proxy_thresholds.json"

# --------------------------------------------------------------------------- #
# Image / crop constants
# --------------------------------------------------------------------------- #
IMAGE_W: int = 960
IMAGE_H: int = 540
CROP_SIZE: int = 224
CROP_PAD: float = 0.20  # 20% padding around the stitch bounding box
EVAL_IMAGE_SIZE: int = 224

SCORE_MIN: int = 1
SCORE_MAX: int = 5
N_SCORES: int = SCORE_MAX - SCORE_MIN + 1

SEED: int = 42

# --------------------------------------------------------------------------- #
# Dataset batches
# --------------------------------------------------------------------------- #
# Every LeRobot batch folder in SutureBot2/.  Only the first five contain
# wound masks + stitch bounding boxes and therefore participate in scoring.
ALL_BATCHES: tuple[str, ...] = (
    "lerobot_fausto_batch_0",
    "lerobot_fausto_batch_1",
    "lerobot_jesse_batch_0",
    "lerobot_jesse_batch_1",
    "lerobot_jesse_batch_2",
    "lerobot_wound_closure_dagger_01",
    "lerobot_wound_closure_dagger_02",
    "lerobot_wound_closure_dagger_03",
    "lerobot_wound_closure_dagger_04",
    "lerobot_wound_closure_dagger_05",
)

# Batches that contribute annotated frames (masks + stitch bboxes).
ANNOTATED_BATCHES: tuple[str, ...] = (
    "lerobot_fausto_batch_0",
    "lerobot_fausto_batch_1",
    "lerobot_jesse_batch_0",
    "lerobot_jesse_batch_1",
    "lerobot_jesse_batch_2",
)

# Batches that carry episode-level needle insertion/exit point annotations.
POINT_LABEL_BATCHES: tuple[str, ...] = (
    "lerobot_fausto_batch_0",
    "lerobot_fausto_batch_1",
    "lerobot_jesse_batch_0",
    "lerobot_jesse_batch_1",
    "lerobot_jesse_batch_2",
)

BATCH_OPERATOR: dict[str, str] = {
    "lerobot_fausto_batch_0": "operator_fausto (intermediate)",
    "lerobot_fausto_batch_1": "operator_fausto (intermediate)",
    "lerobot_jesse_batch_0": "operator_jesse (novice)",
    "lerobot_jesse_batch_1": "operator_jesse (novice)",
    "lerobot_jesse_batch_2": "operator_jesse (novice)",
    "lerobot_wound_closure_dagger_01": "operator_mixed (DAgger)",
    "lerobot_wound_closure_dagger_02": "operator_mixed (DAgger)",
    "lerobot_wound_closure_dagger_03": "operator_mixed (DAgger)",
    "lerobot_wound_closure_dagger_04": "operator_mixed (DAgger)",
    "lerobot_wound_closure_dagger_05": "operator_mixed (DAgger)",
}

# --------------------------------------------------------------------------- #
# Expert rubric (Section 3.3 of RESEARCH_PLAN.md) -- visible final-stitch
# appearance, NOT full surgical skill.  Must be approved by a clinical
# collaborator before use; it is NOT a validated clinical scale.
# --------------------------------------------------------------------------- #
RUBRIC: dict[int, str] = {
    1: "Clearly poor: stitch is misplaced, badly misaligned, or does not meaningfully approximate the wound edges",
    2: "Below acceptable: major irregularity, spacing, or alignment problem",
    3: "Acceptable: wound approximation is usable but has visible inconsistency",
    4: "Good: consistent placement and approximation with minor imperfections",
    5: "Excellent: well-positioned, well-aligned, and highly consistent visible stitch",
}

NOT_SCORABLE_TOKEN: str = "not_scorable"

# --------------------------------------------------------------------------- #
# Geometric-proxy score (Section 3.4 fallback target).
# A transparent, deterministic 1-5 ordinal target derived from geometry only.
# It is a *geometric consistency score*, NOT clinical stitch quality.
# --------------------------------------------------------------------------- #
PROXY_WEIGHTS: dict[str, float] = {
    "containment": 0.30,  # stitch bbox overlaps the wound region
    "proximity": 0.30,    # stitch centre sits on / near the wound
    "spacing": 0.20,      # regular spacing between neighbouring stitches
    "insertion_exit": 0.20,  # insertion/exit alignment with the wound (when known)
}

# Quantile edges (fractions) used to bin the continuous proxy into 1-5.
# Fitted on the TRAINING fold only to avoid leakage.
PROXY_QUANTILE_EDGES: tuple[float, ...] = (0.20, 0.40, 0.60, 0.80)

# Clinical validity statement reused in reports and figures.
CLINICAL_CAVEAT: str = (
    "SutureBot2 is a proof-of-concept robotic, ex-vivo wound-closure dataset. "
    "Scores shown here are either expert-defined (future work) or a derived "
    "geometric-consistency proxy; they are NOT a validated clinical outcome and "
    "do NOT rate a human dermatologist's performance."
)
