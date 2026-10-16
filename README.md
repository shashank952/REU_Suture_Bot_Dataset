# Explainable Ordinal Stitch-Quality Assessment (SutureBot2)

Implementation of `RESEARCH_PLAN.md`: an **explainable ordinal computer-vision
framework for stitch-level quality assessment** on the SutureBot2 robotic
ex-vivo wound-closure dataset.

> Working title: *An Explainable Ordinal Computer-Vision Framework for
> Stitch-Level Quality Assessment in Robotic Wound Closure.*

---

## ⚠️ What this is and is not

SutureBot2 is a **proof-of-concept robotic, ex-vivo** wound-closure dataset
(da Vinci Research Kit, two operators, pork belly). It contains **no expert
1-5 stitch ratings, no OSATS scores and no clinical outcomes**.

* The pipeline therefore emits a blank **expert-rating template** (Milestone 0)
  and, so the technical pipeline can run today, a transparent
  **geometric-consistency proxy** target (the plan's Section 3.4 fallback).
* The proxy is **derived from geometry**, so a geometry-only model can nearly
  recover it by construction. That is a property of the fallback label, **not**
  evidence that the model rates surgical skill.
* The proposed hybrid ("Dale") model is a **geometry-dominant residual fusion**
  (the image adds a small, bounded learned correction) trained as a **6-seed
  ensemble**. On the grouped test split it reaches **85.1% exact / 100%
  within-one / MAE 0.149**, matching or slightly exceeding the geometry-only
  baseline (84.7%) while clearly beating image-only (21.9%) -- i.e. the fusion
  contributes. These numbers are still **proxy** accuracy, not clinical.
* Nothing here predicts healing, wound strength, infection or clinical outcome,
  and nothing rates a human dermatologist. Scores are never a validated clinical
  scale. The realistic performance target is **>= 80% within-one-score
  accuracy**, never 80-90% exact five-class accuracy. Human inter-rater
  agreement is the ceiling.

---

## Three-notebook architecture

Every notebook is a thin wrapper; all logic lives in the importable
`sutureqc` package under `src/`.

| Notebook | Module it drives | Purpose |
| :--- | :--- | :--- |
| `notebooks/01_preprocess_and_audit.ipynb` | `sutureqc.pipeline` | Manifest + path rewrite, frame/mask/bbox matching, bbox validation, stitch-instance table, geometry features, grouped splits, proxy target, crops, expert template, audit report |
| `notebooks/02_train_and_test_models.ipynb` | `sutureqc.modeling` | Track A detection baseline, Track B geometry-only baseline, Track C hybrid ordinal model, ablations, grouped CV + leave-one-batch-out, bootstrap CIs |
| `notebooks/03_visualize_results_and_feedback.ipynb` | `sutureqc.viz`, `sutureqc.feedback` | Publication figures, occlusion attention maps, clinician-facing feedback |

### Package layout (`src/sutureqc/`)

```
config.py       paths, constants, rubric, proxy weights, caveat text
utils.py        IO + path helpers, deterministic seeding
bbox.py         stitch bbox parsing / validation
masks.py        wound-mask descriptors (centroid, area, axis, boundary distance)
manifest.py     manifest rewrite + full-frame inventory + stitch instances
labels_io.py    frame-level semantic action labels
points_io.py    episode-level needle insertion/exit annotations (tolerant parser)
kinematics.py   optional LeRobot parquet pose access
geometry.py     deterministic wound-relative geometry features
crops.py        fixed 20%-padded 224x224 stitch crops
preprocess_viz.py  Notebook-1 step-by-step visual diagnostics (frame/mask/bbox/crop/geometry/split/proxy)
ratings.py      expert rubric, rating template, inter-rater agreement
proxy.py        geometric-consistency proxy target (continuous -> ordinal)
splits.py       episode-grouped and leave-one-batch-out splits
audit.py        audit report + annotation-overlay figure
detection.py    Track A: YOLO dataset prep + guarded training
ordinal.py      Track B: ordinal logistic / forest back-ends
models.py       Track C: frozen backbone, geometry, residual-fusion CORAL head + seed ensemble
training.py     training loop + grouped CV / leave-one-batch-out folds
evaluation.py   ordinal + detection metrics, grouped bootstrap CIs
modeling.py     Notebook-2 orchestration
feedback.py     clinician-facing feedback formatter
viz.py          Notebook-3 figures + occlusion attention
pipeline.py     Notebook-1 orchestration
```

---

## How to run

```powershell
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute notebooks\01_preprocess_and_audit.ipynb --output-dir .nbcheck --output 01.ipynb --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks\02_train_and_test_models.ipynb --output-dir .nbcheck --output 02.ipynb --ExecutePreprocessor.timeout=3600
python -m nbconvert --to notebook --execute notebooks\03_visualize_results_and_feedback.ipynb --output-dir .nbcheck --output 03.ipynb --ExecutePreprocessor.timeout=1800
```

Or import the orchestration directly:

```python
import sys; sys.path.insert(0, "src")
from sutureqc import pipeline, modeling, viz

pipeline.run_preprocess_pipeline()       # Notebook 1
modeling.run_training_pipeline(epochs=60, cv_splits=5)   # Notebook 2
viz.run_visualization_pipeline()         # Notebook 3
```

All CSV path columns are stored **relative to the repository root**; resolve
them with `sutureqc.utils.resolve_path`.

---

## What the SutureBot2 annotation set contains

* 853 endoscope frames (960x540); 777 wound masks and 777 stitch-bbox JSONs.
* 470 non-empty bbox files and 307 empty (negative) frames -> **919 stitch
  instances** across 205 episode groups in 5 annotated batches; only those five
  batches (`fausto_0/1`, `jesse_0/1/2`) carry annotations.
* Frame-level semantic action labels (`labels.jsonl`) and episode-level
  needle insertion/exit points (`insertion_exit_points.csv`).
* The upstream `insertion_exit_points.csv` contains one line where two rows are
  concatenated without a newline; the parser skips such malformed lines
  tolerantly and reports the count in the audit.

---

## Artifacts

`artifacts/`: `manifest_all.csv`, `stitch_instances.csv`, `geometry_features.csv`,
`expert_rating_template.csv`, `splits_grouped.csv`, `proxy_thresholds.json`,
`audit_report.json`, `crops/`, `fold_predictions.csv`, `metrics.json`,
`ablation_metrics.csv`, `models/`, `yolo/`.

`figures/`: `sample_annotations.png`, `training_curves.png`,
`confusion_matrix.png`, `ordinal_error_matrix.png`, and the 24 Notebook-3
publication figures (`fig01_..fig24_..`) -- localization overlays, confusion and
ordinal-error matrices, ROC / precision-recall curves, calibration,
true-vs-predicted and confidence-vs-error scatters, geometry-correlation and
batch-composition heatmaps, per-class metrics, probability heatmap, residual
histogram, geometry feature importance, embedding PCA projection, training
curves, failure gallery and occlusion attention -- plus `feedback_examples.txt`.

---

## Milestones / definition of done

* **M0 label feasibility** -- `expert_rating_template.csv` + `ratings.py`
  (inter-rater ICC/kappa); rubric requires clinical approval before use.
* **M1 data pipeline** -- Notebook 1; leakage-free grouped splits verified by
  `splits.check_no_group_leakage`, crops + manifests reproducible.
* **M2 baselines** -- detection prep + geometry-only and image-only baselines
  (`ablation_metrics.csv`).
* **M3 hybrid ordinal model** -- frozen ResNet18 + geometry + CORAL head,
  grouped CV, leave-one-batch-out, bootstrap CIs.
* **M4 figures + report** -- Notebook 3.
* **M5 external validity** -- deferred (needs human/dermatology data).

Report MAE, exact and within-one accuracy, weighted kappa, Spearman correlation
and grouped-bootstrap CIs on every scoring result; keep detection and scoring as
separate tasks; never claim clinical or dermatologist performance from robotic
ex-vivo data.

## Credits

Dataset: **SutureBot2** (CC BY 4.0), `jhaworth/SutureBot2`. Cite the SutureBot2
(CoRL 2026) and SutureBot (NeurIPS 2025 D&B) papers when using this code.
