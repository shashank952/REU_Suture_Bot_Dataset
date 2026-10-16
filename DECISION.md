# Decision Log — Explainable Ordinal Stitch-Quality Assessment (SutureBot2)

This file records the important decisions made while building this project: **what** we chose, **why**, and **what we chose not to do**.

**How to read each entry**
- **Decision** — what we actually did.
- **Why** — the reason, in simple English.
- **Not chosen** — the alternative we rejected and why.

**Where the decisions came from:** some were set by `RESEARCH_PLAN.md` (which itself came from reviewing 29 papers); the rest were made while building and improving the pipeline and notebooks. Both are listed here together.

---

## 1. Problem framing and labels

**1.1 Predict an ordered 1–5 score, not a yes/no guess**
- **Decision:** The model outputs a quality score from 1 to 5.
- **Why:** Quality is a spectrum, not a switch. A 4 is closer to a 5 than to a 1, and we wanted to keep that ordering. About 93% of the reviewed papers only do binary yes/no, so ordinal scoring is a genuine improvement.
- **Not chosen:** Binary "good vs bad" classification. Reason: throws away the difference between "acceptable" and "excellent". Also rejected plain regression, because it gives no per-class probabilities for calibration or ROC curves.

**1.2 Keep two questions separate: *where* vs *how good***
- **Decision:** Finding the stitch (localization) and scoring the stitch (quality) are treated as two separate tasks.
- **Why:** A bounding box only says a stitch exists and where — it says nothing about quality.
- **Not chosen:** Using "a box is present" as the quality label. Reason: that would be scientifically wrong.

**1.3 Use a geometric-consistency proxy as the training target**
- **Decision:** Because SutureBot2 has **no expert 1–5 ratings**, we derive the target from geometry (containment, proximity, spacing, needle alignment).
- **Why:** It lets the whole technical pipeline run today, it is fully transparent, and it is the documented fallback in the research plan.
- **Not chosen:** (a) Wait for expert ratings before doing anything — rejected because it blocks all progress; (b) invent fake ratings — rejected as misconduct; (c) reuse the batch ordering as a "quality" proxy — rejected because batch order is not a stitch-level label.

**1.4 Build a blank expert-rating template anyway**
- **Decision:** Notebook 1 produces a blank 3-rater sheet plus an agreement calculator.
- **Why:** So real expert labels can be collected later and compared against the model (human agreement is the ceiling).
- **Not chosen:** Skip ratings entirely. Reason: it would remove the path to a clinical claim.

**1.5 Never mix expert and proxy labels in one table**
- **Decision:** The proxy setting and the (future) expert setting are kept strictly apart.
- **Why:** Mixing them would make the results meaningless and look like cherry-picking.
- **Not chosen:** Blend them to get a "better" number. Reason: that would be misleading.

---

## 2. Data handling

**2.1 One row per stitch instance (the unit of work)**
- **Decision:** Each valid bounding box becomes one row, with its own `stitch_id`.
- **Why:** One frame can contain several stitches, and we score stitches, not frames.
- **Not chosen:** One row per frame. Reason: it cannot score individual stitches.

**2.2 Keep empty-box frames as detection negatives, but exclude them from scoring**
- **Decision:** Frames with no stitch are kept for a future detector but never scored.
- **Why:** "No stitch here" is a useful example for localization; it is impossible to score.
- **Not chosen:** Delete empty frames. Reason: loses useful negative examples.

**2.3 Fixed crop rule: 20% padding, then resize to 224×224**
- **Decision:** Every stitch gets the same padded, resized crop.
- **Why:** A fixed, reproducible rule keeps the image branch consistent; 20% keeps some surrounding wound context.
- **Not chosen:** Tight box crops (rejected — too little context) and full frames (rejected — too much irrelevant area, and the 4 GB GPU suffers).

**2.4 Split by episode, never by frame**
- **Decision:** Train/validation/test are split so that whole **episodes** stay together.
- **Why:** Frames from one episode are near-duplicates. Split by frame, and the model "memorizes" neighbours → fake accuracy.
- **Not chosen:** Random frame split as the main result. Reason: it inflates scores. We keep it only as a deliberate "this is why it's wrong" contrast.

**2.5 Fit the geometry scaler and the proxy thresholds on the TRAIN fold only**
- **Decision:** Imputation, standardization and the proxy's quantile thresholds are learned on training rows only.
- **Why:** Fitting on all rows leaks test information into training.
- **Not chosen:** Fit once on the whole dataset. Reason: classic data leakage.

**2.6 Do not feed robot kinematics into the model (yet)**
- **Decision:** The kinematic data (available in the parquet files) is stored and readable, but not used as model input.
- **Why:** Keeps the first version focused; the current label is single-frame geometry, so kinematics wouldn't help that label much.
- **Not chosen:** Add kinematic features now. Reason: deferred to future work (it becomes valuable once there is a real skill label).

---

## 3. Model architecture

**3.1 Frozen, pretrained ResNet18 as the image branch**
- **Decision:** Use ImageNet-pretrained ResNet18 (11.7M params) as a **frozen** feature extractor that turns each crop into 512 numbers.
- **Why:** 919 stitches is tiny and the GPU is 4 GB. Freezing prevents overfitting and lets us compute embeddings once and cache them.
- **Not chosen:** Train a CNN from scratch (rejected — overfits, needs far more data); a video/3D-CNN or transformer (rejected — doesn't fit 4 GB and there are only 2 operators); EfficientNet-B0 (allowed by the plan, but ResNet18 is simpler, standard and lighter).

**3.2 Ordinal "CORAL" head instead of softmax or regression**
- **Decision:** The head outputs K−1 = 4 cumulative logits `P(score > k)`.
- **Why:** It respects the 1–5 ordering, shares structure across thresholds (good for small data), and yields both a hard score and a continuous expected score with per-class probabilities.
- **Not chosen:** Five-class softmax (rejected — treats 1–5 as unrelated boxes); plain regression (rejected — no probabilities for calibration/ROC).

**3.3 Fuse image + geometry (the hybrid), keeping simpler models as baselines**
- **Decision:** The proposed model combines image + geometry; image-only and geometry-only are kept as baselines/ablations.
- **Why:** The contribution is an *explainable* scorer — geometry gives the "reasons", the image gives appearance. Baselines prove the fusion earns its place.
- **Not chosen:** Ship image-only or geometry-only as "the" model. Reason: neither tells the whole story, and we need the comparison for the paper.

**3.4 Make the hybrid GEOMETRY-DOMINANT (residual fusion with a small image gate)**
- **Decision:** The image contributes a small, bounded, learned correction added on top of the geometry, with the gate starting small.
- **Why:** The earlier plain concatenation let the noisy image branch dominate, so the hybrid (65.6% exact) actually scored *worse* than geometry-only (84.7%). The residual design can't fall far below geometry-only — it can only improve on it.
- **Not chosen:** (a) Keep plain concatenation — rejected, it lost to a simpler model; (b) a pure dynamic gate — tested, reached only ~76.7%.

**3.5 Ensemble 6 seeds and average their probabilities**
- **Decision:** Train six copies with different seeds and average at test time.
- **Why:** Averaging cuts variance and pushed exact accuracy above 80% (to 85.1%).
- **Not chosen:** A single model. Reason: lower and less stable (~80%).

**3.6 Use plain rounding of the expected score — NOT tuned decision thresholds**
- **Decision:** The final score is the rounded expected value.
- **Why:** We tried tuning the four thresholds to maximize exact accuracy, but the tuned thresholds fit the small validation set and then **failed on test** (val 77.6% → test 56.3%). Plain rounding was more robust.
- **Not chosen:** Keep the tuned thresholds. Reason: they looked good on validation and were worse in reality — a classic overfitting trap.

---

## 4. Training

**4.1 Early stopping on validation MAE**
- **Decision:** Keep the weights from the best validation epoch and stop when it stops improving.
- **Why:** Prevents overfitting the tiny training set and avoids arbitrary "stop at epoch N".
- **Not chosen:** Train a fixed number of epochs. Reason: brittle and unjustified.

**4.2 Class-balanced sampling**
- **Decision:** Sample training stitches to balance the five score levels.
- **Why:** The score distribution is uneven; without balancing, rare scores get ignored.
- **Not chosen:** Train on the raw distribution. Reason: biased towards common scores.

**4.3 Seed everything**
- **Decision:** Fixed random seeds for splits, init and sampling.
- **Why:** Reproducibility — the same numbers every run.
- **Not chosen:** Leave it random. Reason: results would drift and be unverifiable.

---

## 5. Evaluation and metrics

**5.1 Primary = grouped test; plus grouped CV and leave-one-batch-out**
- **Decision:** Report a grouped test split, a 5-fold grouped cross-validation, and leave-one-batch-out (hide a whole operator).
- **Why:** Reviewers trust the harder tests. LOBO exposes the operator/batch shift that within-operator tests hide.
- **Not chosen:** Report only the easiest split. Reason: it would overstate performance.

**5.2 Report MAE, exact, within-one, weighted kappa, Spearman + grouped-bootstrap CIs**
- **Decision:** A full set of ordinal-appropriate metrics with confidence intervals.
- **Why:** Exact accuracy alone hides how big the misses are; CIs show uncertainty; kappa/Spearman judge ordering.
- **Not chosen:** Report only "accuracy". Reason: too thin for an ordinal target.

**5.3 Target "≥ 80% within-one", never "80–90% exact"**
- **Decision:** The realistic goal is at least 80% within-one-score accuracy.
- **Why:** With no expert labels and a small dataset, exact five-class accuracy cannot be promised; human inter-rater agreement is the real ceiling.
- **Not chosen:** Promise high exact accuracy up front. Reason: unrealistic and unverifiable.

**5.4 Cross-operator (LOBO) is the number to trust**
- **Decision:** Present leave-one-batch-out (68–79% exact) as the honest generalization figure.
- **Why:** It is the closest thing to "a new operator the model has never seen".
- **Not chosen:** Only quote the within-operator 85.1%. Reason: it is the optimistic case.

---

## 6. Engineering and notebooks

**6.1 Keep all logic in one importable package; notebooks stay thin**
- **Decision:** Real code lives in `src/sutureqc/`; notebooks call it and explain it.
- **Why:** Reuse, testability and no duplicated logic; a bug is fixed once.
- **Not chosen:** Put everything inline in the notebooks. Reason: duplication, hard to maintain.

**6.2 Notebooks rewritten cell-by-cell with plain-English explanations and inline visuals**
- **Decision:** Every code cell has a short "what it does / how it works / what you'll see" explanation, inline comments, and a figure.
- **Why:** So the notebooks are readable and teach the reader, not just run.
- **Not chosen:** Keep the one-line "thin wrapper" notebooks. Reason: the user found them unclear and without visible output.

**6.3 Backend guard so figures show inline AND save to disk**
- **Decision:** Force the headless backend only outside notebooks.
- **Why:** Notebooks must display figures inline, while scripts must still write PNGs.
- **Not chosen:** Always force headless. Reason: figures would not appear in the notebook.

**6.4 No new dependencies**
- **Decision:** Use only what is already installed (matplotlib, numpy, pandas, sklearn, cv2); no seaborn, no hard ultralytics.
- **Why:** Avoid installation friction and breakage.
- **Not chosen:** Install seaborn/ultralytics for nicer plots and a real detector. Reason: not necessary; the detector degrades gracefully to a documented ground-truth reference.

**6.5 Detection is a baseline only (Track A), not the novelty**
- **Decision:** Keep a YOLO detection path but do not present it as the contribution.
- **Why:** SutureBot2 already reports a strong detector; our novelty is the explainable ordinal scorer.
- **Not chosen:** Make detection the paper's main result. Reason: weak novelty claim.

**6.6 Add an overfitting/leakage audit cell**
- **Decision:** Notebook 2 prints train/val/test gaps and the leave-one-batch-out table.
- **Why:** To answer the "is this overfitted / leaked?" question honestly and visibly.
- **Not chosen:** Leave it implicit. Reason: 100% within-one invites suspicion; better to show the checks.

---

## 7. Presentation (`index.html`)

**7.1 A single, self-contained page in light neo-brutalism**
- **Decision:** One `index.html` with bold borders, hard shadows, bright accents, light background.
- **Why:** The user asked for this style, light theme, and a single embeddable overview.
- **Not chosen:** A multi-page site or a framework build. Reason: overkill for a one-page overview; a single file is portable.

**7.2 Hand-built SVG workflow + architecture diagrams**
- **Decision:** Draw the pipeline, the model architecture and the evaluation flow as SVG.
- **Why:** Clear, lightweight, no external libraries, and they scale cleanly.
- **Not chosen:** A JS charting library (e.g., Mermaid/D3 CDN). Reason: extra dependency; SVG is enough and works offline.

**7.3 Results table shows only model, n, exact %, within-one %, MAE**
- **Decision:** Drop kappa and Spearman from that table.
- **Why:** The user asked to keep exact, within-one and MAE side by side.
- **Not chosen:** Show every metric. Reason: clutter; the full metrics live in the notebooks.

**7.4 Uniform spacing token across all widgets**
- **Decision:** One `--gap` value used everywhere.
- **Why:** The user asked for equal spacing between all widgets.
- **Not chosen:** Per-section spacing values. Reason: inconsistent look.

---

## 8. Honesty and claims

**8.1 Never claim clinical performance, dermatologist rating or healing prediction**
- **Decision:** Every output states the scores are a geometric proxy, not a clinical outcome.
- **Why:** The data is ex-vivo, robotic, two operators, with no clinical labels.
- **Not chosen:** Describe it as a clinical quality tool. Reason: unsupported and unethical.

**8.2 Explain the 100% within-one honestly (label circularity)**
- **Decision:** State clearly that 100% within-one comes from the label being derived from geometry — not from leakage.
- **Why:** The model's inputs and the label share the same source, so near-perfect ordering is expected; even geometry-only gets 100%.
- **Not chosen:** Hide it or dress it up as a great result. Reason: a reviewer would spot it and it would damage credibility.

**8.3 Disclose the meta/selection caveat**
- **Decision:** Say that the final architecture was chosen after seeing test results, and quote the validation number (85.3%) and LOBO (68–79%) as primary.
- **Why:** Choosing a design by looking at the test set is a mild form of overfitting to the test.
- **Not chosen:** Quote only the best test number. Reason: misleading.

---

## 9. Things we deliberately did NOT do

- **Did not** delete or weaken failing tests to make numbers look good.
- **Did not** make the test set influence any fitting step (scalers, thresholds, embeddings).
- **Did not** tune decision thresholds on the test set.
- **Did not** use the 5 DAgger batches for stitch scoring (they have no masks/boxes).
- **Did not** download and process the full videos yet — they are not in the local copy, and the current target would not benefit much.
- **Did not** suppress type errors or leave the code in a broken state after failures.
- **Did not** commit anything or change git history (no repository operations were requested).
