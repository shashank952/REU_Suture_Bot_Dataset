"""sutureqc -- explainable ordinal stitch-quality assessment for SutureBot2.

Small, reusable modules behind three notebooks:

* ``notebooks/01_preprocess_and_audit.ipynb``      -> :mod:`sutureqc.pipeline`
* ``notebooks/02_train_and_test_models.ipynb``     -> :mod:`sutureqc.models` / :mod:`sutureqc.training`
* ``notebooks/03_visualize_results_and_feedback.ipynb`` -> :mod:`sutureqc.viz`

Nothing here trains or plots at import time.  See ``config`` for every path
and constant, and ``RESEARCH_PLAN.md`` for the research rationale.
"""

from __future__ import annotations

from . import (
    audit,
    bbox,
    config,
    crops,
    detection,
    evaluation,
    feedback,
    geometry,
    kinematics,
    labels_io,
    manifest,
    masks,
    modeling,
    models,
    ordinal,
    pipeline,
    points_io,
    preprocess_viz,
    proxy,
    ratings,
    splits,
    training,
    utils,
    viz,
)

__all__ = [
    "audit",
    "bbox",
    "config",
    "crops",
    "detection",
    "evaluation",
    "feedback",
    "geometry",
    "kinematics",
    "labels_io",
    "manifest",
    "masks",
    "modeling",
    "models",
    "ordinal",
    "pipeline",
    "points_io",
    "preprocess_viz",
    "proxy",
    "ratings",
    "splits",
    "training",
    "utils",
    "viz",
]

__version__ = "0.1.0"
