"""Clinician-facing feedback formatting (Section 5, "Feedback format").

The two questions are kept strictly separate, as the plan requires:

1. *Where is the stitch?* -- answered by the provided bounding boxes
   (a localization task).
2. *How good is the stitch?* -- answered by the ordinal 1-5 score
   (a scoring task).

A low-confidence or out-of-distribution stitch is reported as
``review required`` instead of an overconfident score.
"""

from __future__ import annotations

import numpy as np

from . import config

REVIEW_CONFIDENCE_THRESHOLD: float = 0.5


def review_decision(
    confidence: float,
    in_distribution: bool = True,
    confidence_threshold: float = REVIEW_CONFIDENCE_THRESHOLD,
) -> bool:
    """Return True when the stitch should be sent for human review."""
    if confidence is None or not np.isfinite(confidence):
        return True
    return bool(confidence < confidence_threshold or not in_distribution)


def format_feedback(score: int, confidence: float, evidence: str, review: bool) -> str:
    """Render exactly the four-line feedback block used in the paper."""
    review_flag = "yes -- review required" if review else "no"
    score = int(np.clip(score, config.SCORE_MIN, config.SCORE_MAX))
    return (
        f"Stitch score: {score}/5\n"
        f"Confidence: {confidence:.2f}\n"
        f"Evidence: {evidence}\n"
        f"Review flag: {review_flag}"
    )


def where_and_how_good(bbox_row, score_row) -> dict[str, str]:
    """Return the localization answer and the (proxy) score answer separately."""
    where = (
        f"stitch bbox [{bbox_row['x1']:.0f}, {bbox_row['y1']:.0f}, "
        f"{bbox_row['x2']:.0f}, {bbox_row['y2']:.0f}] in {bbox_row['basename']}"
    )
    how_good = (
        f"geometric-consistency proxy score {int(score_row['y_pred'])}/5 "
        f"(confidence {float(score_row['confidence']):.2f})"
    )
    return {"where": where, "how_good": how_good}
