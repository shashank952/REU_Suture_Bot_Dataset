# Research Plan: Explainable AI for Suturing Quality Control

## 1. Working Title

**An Explainable Ordinal Computer-Vision Framework for Stitch-Level Quality Assessment in Robotic Wound Closure**

The original project title, *Framework for Artificial Intelligence in Quality Control of Suturing Among Undergraduates and Dermatology Residents*, should not be used for the SutureBot2-only experiment. SutureBot2 contains robotic suturing on ex-vivo pork tissue performed by two operators; it does not contain dermatologist or undergraduate performances and it does not provide expert 1-5 quality ratings.

The SutureBot2 study should therefore be described as a **proof-of-concept robotic wound-closure quality-assessment study**. A later human/dermatology dataset can validate whether the method transfers to the intended population.

## 2. Core Research Question

Can a model use an annotated stitch region, wound segmentation, needle-point annotations, and optional robot-derived geometry to predict an interpretable **ordinal stitch-quality score from 1 to 5**?

The model should answer two different questions separately:

1. **Where is the stitch?** Use the provided stitch bounding boxes. This is a detection/localization task.
2. **How good is the stitch?** Predict a 1-5 quality score from a stitch crop and measurable wound-relative features. This is an ordinal scoring task.

Do not use bounding-box presence as a quality label. A box says that a stitch exists and where it is; it does not say whether the stitch is clinically good.

## 3. Feasibility Gate: Labels

### 3.1 What SutureBot2 already provides

Local inventory under `SutureBot2/`:

- 853 original 960x540 RGB frames.
- 777 wound masks.
- 777 stitch-bounding-box JSON files.
- 470 non-empty bounding-box files and 307 valid empty negative frames.
- Frame-level semantic action labels in `labels.jsonl`.
- Needle insertion/exit tracks in `point_labels.jsonl` and `insertion_exit_points.csv` for selected batches.
- LeRobot episode parquet files containing robot actions and state/kinematics.
- Two operators, one described as intermediate and one as novice; demonstrations include recovery behavior.

### 3.2 What is missing

There is no expert 1-5 rating for each stitch, no clinician-rated OSATS score for each image, and no validated clinical outcome such as burst pressure or wound-healing quality. The existing batch ordering is not a substitute for a stitch-level label.

### 3.3 Preferred label plan

Before model training, create a small expert-labeled quality set from the 470 positive stitch instances:

- Have at least two clinically experienced raters score each stitch on a written 1-5 rubric; three raters are preferable.
- Keep raters blinded to batch, episode, operator identity, and any existing quality ordering.
- Allow a “not scorable/occluded” category during annotation; exclude it from the primary score model rather than forcing a rating.
- Record the raw score from every rater, the consensus score, and annotation time.
- Use the median or mean consensus as the primary target and report inter-rater agreement as the human ceiling.

Suggested rubric for **visible final-stitch appearance**, not full surgical skill:

| Score | Operational meaning |
|---|---|
| 1 | Clearly poor: stitch is misplaced, badly misaligned, or does not meaningfully approximate the wound edges |
| 2 | Below acceptable: major irregularity, spacing, or alignment problem |
| 3 | Acceptable: wound approximation is usable but has visible inconsistency |
| 4 | Good: consistent placement and approximation with minor imperfections |
| 5 | Excellent: well-positioned, well-aligned, and highly consistent visible stitch |

The rubric must be adapted and approved by a clinical collaborator. It must not be presented as an established clinical scale unless it is formally validated.

### 3.4 Fallback if expert labels cannot be collected

Create a **derived geometric quality proxy** from spacing, wound-relative location, insertion/exit points, and mask geometry. Call the target *geometric consistency score*, not clinical stitch quality. This fallback is useful for a technical paper but does not support the claim that the model rates a doctor’s clinical performance.

## 4. Recommended Contribution

The recommended minimal novelty is:

> An explainable ordinal stitch-quality model that fuses a stitch image crop with wound-relative geometry and reports both a 1-5 score and the measurable reasons for that score.

This is preferable to proposing another YOLO detector because the SutureBot2 README already reports a YOLOv26/SAM2 planner benchmark with high wound IoU and stitch F1. Detection alone would be a weak novelty claim.

### 4.1 Optional novelty components, in priority order

1. **Ordinal rather than binary scoring.** Most reviewed studies reduce assessment to binary novice/expert or pass/fail. A 1-5 ordinal target preserves intermediate performance.
2. **Wound-relative explainability.** Report spacing consistency, position relative to the wound, mask overlap, and insertion/exit alignment alongside the learned score.
3. **Human-versus-model agreement.** Compare model agreement with inter-rater agreement, not only raw accuracy.
4. **Strict grouped generalization.** Use episode/batch-held-out evaluation and report how performance changes on unseen batches.
5. **Uncertainty-aware feedback.** Display a score only when confidence is adequate; otherwise flag the stitch for human review.

Do not attempt all five in the first version. Use items 1-3 as the primary paper contribution and treat 4-5 as supporting experiments.

## 5. Three-Notebook Architecture

The project should contain exactly three main notebooks. Notebooks should call small reusable Python modules where practical; they should not duplicate preprocessing logic.

### Notebook 1: `01_preprocess_and_audit.ipynb`

**Purpose:** turn the downloaded repository into a clean, auditable stitch-level table.

#### Steps

1. Define `DATA_ROOT` and verify the expected folders exist.
2. Load `annotation_images/masks/manifest.csv` and rewrite the original author's absolute paths to local relative paths.
3. Match each original frame with its mask and stitch-bbox JSON by basename.
4. Parse every bbox as `[x1, y1, x2, y2]`; reject invalid coordinates and record a validation report.
5. Preserve empty bbox frames as detection negatives, but exclude them from stitch-quality scoring.
6. Convert each non-empty bbox into one stitch-instance row. If a frame contains multiple boxes, create one row per box.
7. Join episode, batch, frame index, timestamp, and semantic action context where available.
8. Compute deterministic geometry features:
   - bbox width, height, area, aspect ratio;
   - normalized center coordinates;
   - distance from stitch center to wound-mask centroid and wound boundary;
   - mask area, wound width/height, and bbox-to-wound area ratio;
   - nearest-neighbor distance between stitch centers when multiple stitches are visible;
   - insertion/exit-point distance and alignment when point labels are available.
9. Generate stitch crops with a fixed padding rule, for example 20% around the bbox, clipped to image bounds.
10. Create the expert-rating template CSV with one row per stitch instance and blank rater columns.
11. Create grouped train/validation/test manifests. The grouping key must be episode or batch, never an individual frame.
12. Produce an audit report: counts, class/score distribution, empty versus non-empty boxes, missing masks, duplicate identifiers, and representative image overlays.

#### Outputs

- `artifacts/manifest_all.csv`
- `artifacts/stitch_instances.csv`
- `artifacts/geometry_features.csv`
- `artifacts/expert_rating_template.csv`
- `artifacts/splits_grouped.csv`
- `artifacts/crops/`
- `artifacts/audit_report.json`
- `figures/sample_annotations.png`

#### Go/no-go gate

Do not train the quality model until every scored training instance has a valid image crop, a valid bbox, a known group identifier, and a real expert or explicitly derived proxy target.

### Notebook 2: `02_train_and_test_models.ipynb`

**Purpose:** train detection baselines and the proposed lightweight ordinal scorer.

#### Model tracks

##### Track A: detection baseline

- Use the provided boxes to create YOLO-format labels.
- Train a small YOLO model such as YOLOv8n or YOLO11n only as a localization baseline.
- Use empty frames as negative images.
- Evaluate with precision, recall, F1, mAP50, and mAP50-95.
- Report both ground-truth-box scoring and predicted-box scoring. This separates scoring quality from detector errors.

The detector is not the final quality model.

##### Track B: non-deep-learning quality baseline

Train a simple model on geometry features only:

- ordinal logistic regression if available;
- otherwise a regularized tree/forest baseline plus calibrated ordinal post-processing.

This baseline answers whether the visual encoder is actually necessary.

##### Track C: proposed hybrid model

1. Crop each stitch using its bbox and padding.
2. Resize to 224x224.
3. Extract image features with a pretrained ResNet18 or EfficientNet-B0.
4. Freeze the backbone initially; only train a small MLP head.
5. Concatenate the image embedding with normalized geometry features.
6. Use an ordinal regression head with cumulative thresholds for scores 1-5, rather than treating the five values as unrelated classes.
7. Add a second auxiliary head for one or two measurable geometry targets only if the label count supports it. Do not add a large multitask network by default.

This model fits a GTX 1650 Ti more realistically than a video transformer. Use mixed precision only if stable, batch size 1-8, early stopping, and precomputed embeddings where necessary.

#### Label variants

Run the proposed model under two clearly separated settings:

1. **Expert-score setting:** primary publication result, only if human ratings are collected.
2. **Geometric-proxy setting:** secondary technical experiment, clearly labeled as proxy prediction.

Never mix proxy and expert labels in one result table.

#### Evaluation protocol

- Primary split: grouped by episode, with no frames from the same episode in multiple splits.
- Stronger validation: leave-one-batch-out or grouped cross-validation.
- If the number of expert-labeled stitches is too small for a stable holdout, use repeated grouped cross-validation and bootstrap confidence intervals; do not report a single lucky split.
- Fit image normalization, feature scaling, and any thresholds on the training fold only.

#### Metrics

Detection:

- precision, recall, F1;
- mAP@0.50 and mAP@0.50:0.95;
- performance on positive and negative frames separately.

Ordinal scoring:

- mean absolute error (MAE);
- exact accuracy, reported but not optimized as the only metric;
- within-one-score accuracy;
- weighted Cohen's kappa or quadratic-weighted kappa;
- Spearman correlation;
- calibration/error by score level;
- confidence intervals from grouped bootstrap resampling.

The realistic target is **at least 80% within-one-score accuracy**, not a guaranteed 80-90% exact five-class accuracy. With only 470 positive frames and no existing quality labels, exact accuracy cannot be promised. Human inter-rater agreement is the upper reference point.

#### Required ablations

1. Geometry only.
2. Image only.
3. Image + geometry (proposed).
4. Ground-truth boxes versus predicted YOLO boxes.
5. Random frame split versus grouped split, included only to demonstrate why random splitting is misleading—not as the primary result.

#### Outputs

- Saved model weights and configuration.
- Fold-level predictions CSV.
- Metrics JSON/CSV with confidence intervals.
- Training curves.
- Confusion matrices and ordinal error matrices.
- Per-instance prediction table with score, confidence, group, and error.

### Notebook 3: `03_visualize_results_and_feedback.ipynb`

**Purpose:** produce publication figures and the clinician-facing visual explanation.

#### Required visual outputs

1. Original frame with predicted stitch boxes.
2. Each stitch crop with predicted 1-5 score and confidence.
3. Wound mask overlay and stitch location.
4. Ground-truth versus predicted score comparison.
5. Confusion matrix and ordinal error matrix.
6. MAE, weighted kappa, and Spearman results by fold/batch.
7. Score distribution by batch and semantic subtask.
8. Geometry-only versus image-only versus hybrid ablation plot.
9. Calibration plot: predicted confidence versus observed correctness.
10. Failure gallery: occlusion, empty/ambiguous bbox, unusual orientation, and batch shift.
11. Optional Grad-CAM or occlusion map for the visual branch, clearly described as model attention rather than causal evidence.

#### Feedback format

The output should say:

```text
Stitch score: 4/5
Confidence: 0.81
Evidence: good wound-relative placement; spacing slightly irregular
Review flag: no
```

If confidence is low or the image is outside the training distribution, output “review required” instead of an overconfident score.

## 6. Literature-Based Gap Analysis

The local literature spreadsheet contains 29 papers from 2024-2026. The repeated weaknesses are:

1. **Binary targets dominate.** The systematic review reports that roughly 93% of studies use binary classification. A validated ordinal 1-5 target is a defensible improvement.
2. **Random or weakly grouped splits inflate results.** Recent markerless and whole-stream studies show large performance drops under leave-one-subject-out or external validation. This plan makes grouped evaluation primary.
3. **Feedback is often coarse.** Heatmaps or novice/expert labels do not identify a specific repairable stitch defect. Wound-relative geometry provides concrete feedback.
4. **Many models require special sensors or robot kinematics.** A crop-plus-geometry model can be run from annotated images and can later be extended to ordinary camera data.
5. **Clinical validity is rarely demonstrated.** SutureBot2 has no patient outcome or mechanical-strength label. Do not claim clinical effectiveness. Frame the result as a technical proof of concept and list expert annotation plus future clinical validation.
6. **Generalization is weak.** SutureBot2 has only two operators and a robotic platform. Batch-held-out results and an explicit domain limitation are mandatory.
7. **Detection is no longer the novelty.** SutureBot2 already reports strong YOLO/SAM2 planner performance. The paper should focus on ordinal scoring and explainable feedback.
8. **Final-product quality is underused.** The 2026 ex-vivo computer-vision work linking stitch spatial metrics to mechanical strength is the closest conceptual precedent. SutureBot2 cannot reproduce that claim without strength measurements, so the present plan uses spatial consistency as an interpretable proxy only.

## 7. Publication Claims

### Claims the study can support

- The proposed model predicts expert-defined or explicitly derived stitch-quality categories on the SutureBot2 annotation set.
- Combining image appearance and wound-relative geometry improves over image-only and geometry-only baselines, if the ablation supports it.
- The model produces interpretable per-stitch feedback and uncertainty estimates.
- Grouped evaluation exposes batch/operator generalization limits.

### Claims the study cannot support without new data

- The model rates the quality of a human dermatologist's suturing.
- The 1-5 score predicts patient healing, wound strength, infection, or clinical outcome.
- The provided bbox itself is a quality label.
- 80-90% exact accuracy is guaranteed.

## 8. Data and Ethics Requirements

- Obtain clinical approval or institutional confirmation before collecting new expert ratings or human videos.
- De-identify any human/dermatology videos and restrict access to authorized researchers.
- Cite SutureBot2 and its accompanying paper.
- Preserve the dataset license and do not redistribute the downloaded media.
- Report the robotic/ex-vivo setting, two-operator limitation, number of episodes, and annotation-selection rules.

## 9. Milestones

### Milestone 0: Label feasibility

- Inspect all 470 positive stitch instances.
- Confirm whether the visible frame is sufficient for quality judgment.
- Obtain clinical collaborator approval for the rubric.
- Collect a pilot set of 30-50 ratings from at least two raters.

**Go:** agreement is acceptable and raters can apply the rubric consistently.  
**Stop/reframe:** the frame does not contain enough information; change the target to geometric consistency or acquire temporal/video context.

### Milestone 1: Data pipeline

- Complete Notebook 1.
- Verify no duplicate frame/episode leakage.
- Export reproducible manifests and crops.

### Milestone 2: Baselines

- Train geometry-only, image-only, and YOLO detection baselines.
- Establish metrics and failure modes before claiming novelty.

### Milestone 3: Hybrid ordinal model

- Train the frozen-backbone image-plus-geometry model.
- Run grouped cross-validation and ablations.
- Generate confidence intervals.

### Milestone 4: Visual report and paper figures

- Complete Notebook 3.
- Select representative successes and failures.
- Write limitations before writing the conclusion.

### Milestone 5: External validity decision

- If the target is dermatology residents, collect a small separate validation set or explicitly position that work as future research.

## 10. Definition of Done

The project is ready for manuscript drafting only when:

- A documented quality-label source exists for every scoring example.
- Detection and quality scoring are evaluated as separate tasks.
- No episode or batch leaks across primary train/test groups.
- The proposed model is compared against geometry-only and image-only baselines.
- MAE, within-one accuracy, weighted kappa, Spearman correlation, and confidence intervals are reported.
- Human inter-rater agreement is reported when expert labels are used.
- Visual predictions include boxes, masks, score, confidence, and failure examples.
- The manuscript does not claim dermatologist or clinical performance from robotic ex-vivo data.
