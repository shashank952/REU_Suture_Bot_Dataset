# Project Flow — Explainable Ordinal Stitch-Quality Assessment (SutureBot2)

This document describes **how the project flows from raw data to a finished result**. Think of it as the map: what comes in, what happens, what comes out, and how each part connects to the next.

> Legend: **Notebook 1** = `01_preprocess_and_audit.ipynb` · **Notebook 2** = `02_train_and_test_models.ipynb` · **Notebook 3** = `03_visualize_results_and_feedback.ipynb` · `sutureqc/` = the shared Python package that does the real work.

---

## 1. The big picture (one glance)

```mermaid
flowchart LR
    A[Raw SutureBot2<br/>frames · masks · boxes · labels · robot parquet] --> B[Notebook 1<br/>Preprocess &amp; Audit]
    B --> C[(artifacts/<br/>stitch table · features · crops · splits · proxy)]
    C --> D[Notebook 2<br/>Train &amp; Test]
    D --> E[(artifacts/<br/>predictions · metrics · models · histories)]
    E --> F[Notebook 3<br/>Figures &amp; Feedback]
    F --> G[24 publication figures<br/>+ doctor-facing feedback]
    style B fill:#9BE7FF,stroke:#111,stroke-width:2px
    style D fill:#C6F26B,stroke:#111,stroke-width:2px
    style F fill:#FFB35C,stroke:#111,stroke-width:2px
```

**In words:**
- **Notebook 1** turns the messy download into a clean, leak-free **stitch table** (one row per stitch) plus features, crops, splits and a target.
- **Notebook 2** trains and tests the models on that table and writes predictions + metrics.
- **Notebook 3** turns those results into the figures and the clinician-facing explanation.

---

## 2. End-to-end sequence

```mermaid
sequenceDiagram
    participant P as User
    participant N1 as Notebook 1
    participant FS as artifacts/
    participant N2 as Notebook 2
    participant N3 as Notebook 3

    P->>N1: run preprocessing
    N1->>FS: write stitch table, features, crops, splits, proxy, audit
    P->>N2: run training + testing
    N2->>FS: read features + crops
    N2->>FS: write predictions, metrics.json, models, histories
    P->>N3: run visualization
    N3->>FS: read predictions + metrics
    N3->>FS: write 24 figures + feedback_examples.txt
    FS->>P: figures ready for the paper
```

---

## 3. Stage 1 — Notebook 1: Preprocess & Audit

Purpose: **turn raw data into one clean row per stitch**, with everything the models need.

```mermaid
flowchart TD
    S1[1. Verify dataset folders] --> S2[2. Load manifest + rewrite paths]
    S2 --> S3[3. Inventory every frame: mask? boxes? empty?]
    S3 --> S4[4. Validate each bounding box]
    S4 --> S5[5. One row per valid stitch instance]
    S5 --> S6[6. Join semantic action label per stitch]
    S6 --> S7[7. Compute wound-relative geometry features]
    S7 --> S8[8. Generate 20%-padded 224x224 crops]
    S8 --> S9[9. Episode-grouped splits + proxy target]
    S9 --> S10[10. Audit report + overlay figure]
    S10 --> OUT[(manifest_all.csv<br/>stitch_instances.csv<br/>geometry_features.csv<br/>crops/ · splits_grouped.csv<br/>proxy_thresholds.json<br/>audit_report.json)]
```

**Key rules inside this stage**
- Split by **episode**, never by frame (frames from one episode are near-duplicates).
- Fit the geometry scaler **and** the proxy thresholds on **training rows only**.
- Keep empty-box frames as detection negatives but never score them.

**Output that the rest of the project depends on:** `geometry_features.csv` (stitches + features + proxy score + split) and `crops/`.

---

## 4. Stage 2 — Notebook 2: Train & Test

Purpose: **train the models, test them honestly, and measure accuracy.**

```mermaid
flowchart TD
    A[Load stitch table + split] --> B{Leakage check<br/>episode in one split only?}
    B -- pass --> C[Fit geometry preprocessor on TRAIN only]
    C --> D[Frozen ResNet18 -> 512-d crop embeddings]
    D --> E[Build hybrid CORAL model]
    E --> F[Train 6-seed ensemble + early stopping]
    F --> G[Predict on held-out test]
    G --> H[Compute metrics + grouped-bootstrap CIs]
    H --> I[Ablations: geometry-only, image-only, hybrid]
    H --> J[5-fold grouped CV]
    H --> K[Leave-one-batch-out cross-operator]
    H --> L[Random-frame split - contrast only]
    I --> M[(fold_predictions.csv<br/>metrics.json<br/>ablation_metrics.csv<br/>models/ · histories.json<br/>training_curves.png)]
    style B fill:#FFE9B8,stroke:#111,stroke-width:2px
```

**Key rules inside this stage**
- The scaler and any thresholds are fit on **train only** (no leakage).
- The ResNet18 is **frozen** — no fitting, so the embeddings cannot leak.
- The headline model is the **geometry-dominant residual hybrid**, averaged over 6 seeds.
- The honest generalization number is **leave-one-batch-out**, not the easy split.

---

## 5. Stage 3 — Notebook 3: Figures & Feedback

Purpose: **turn results into paper figures and a doctor-facing explanation.**

```mermaid
flowchart TD
    A[Load features + predictions + ablation] --> B{Notebook 2 ran?}
    B -- yes --> C[Localization figures: frames, crops, mask overlay]
    B -- yes --> D[Scoring figures: confusion, ROC, PR, scatter, calibration]
    B -- yes --> E[Generalization: metrics-by-fold, ablation, batch heatmap]
    B -- yes --> F[Explainability: correlation heatmap, feature importance, embeddings]
    B -- yes --> G[Failure gallery + occlusion attention]
    B -- no --> H[Only geometry-only figures]
    C --> I[(figures/fig01..fig24.png)]
    D --> I
    E --> I
    F --> I
    G --> I
    I --> J[feedback_examples.txt - doctor-facing text]
```

---

## 6. Model flow (how one stitch becomes a score)

```mermaid
flowchart LR
    subgraph Image branch
      C1[Stitch crop 224x224] --> R[Frozen ResNet18] --> E1[512-d embedding]
    end
    subgraph Geometry branch
      G1[30 geometry features] --> GS[Impute + standardize<br/>fit on train]
    end
    E1 --> FU[Residual fusion<br/>geometry + gate x image]
    GS --> FU
    FU --> H[CORAL head<br/>4 cumulative logits]
    H --> SC[Score 1-5]
    SC --> EN[Average over 6 seeds]
```

**Why this shape:** the geometry is the backbone; the image adds a **small, bounded correction** (so the model can never fall far below geometry-only). The CORAL head keeps the 1–5 score **ordered**.

---

## 7. Data → artifact flow (what feeds what)

```mermaid
flowchart LR
    RAW[SutureBot2 raw] --> M[manifest_all.csv]
    M --> SI[stitch_instances.csv]
    SI --> GF[geometry_features.csv]
    SI --> CR[crops/]
    GF --> SP[splits_grouped.csv]
    GF --> PT[proxy_thresholds.json]
    GF --> N2[Notebook 2]
    CR --> N2
    N2 --> FP[fold_predictions.csv]
    N2 --> MJ[metrics.json]
    N2 --> AB[ablation_metrics.csv]
    N2 --> MO[models/ + histories.json]
    FP --> N3[Notebook 3]
    AB --> N3
    GF --> N3
    N3 --> FIG[figures/fig01..fig24.png]
```

---

## 8. Evaluation flow (how accuracy is measured)

```mermaid
flowchart TD
    E[205 episodes] --> S1[Grouped test split<br/>561 / 143 / 215]
    E --> S2[5-fold grouped CV]
    E --> S3[Leave-one-batch-out<br/>hide a whole operator]
    E --> S4[Random frame split<br/>shown only to prove it misleads]
    S1 --> M[Metrics: exact, within-one, MAE,<br/>weighted kappa, Spearman]
    S2 --> M
    S3 --> M
    S4 --> M
    M --> CI[Grouped-bootstrap 95% confidence intervals]
```

**Reading the results**
- **Exact** = score matches; **within-one** = off by at most 1; **MAE** = average miss size.
- The **cross-operator** number (leave-one-batch-out) is the honest generalization figure.

---

## 9. How to run the flow (command line)

```text
1. Install:        python -m pip install -r requirements.txt
2. Notebook 1:     python -m nbconvert --to notebook --execute notebooks\01_preprocess_and_audit.ipynb --inplace
3. Notebook 2:     python -m nbconvert --to notebook --execute notebooks\02_train_and_test_models.ipynb --inplace
4. Notebook 3:     python -m nbconvert --to notebook --execute notebooks\03_visualize_results_and_feedback.ipynb --inplace
5. Overview page:  open index.html in a browser
```

Order matters: **Notebook 1 → Notebook 2 → Notebook 3** (each reads what the previous one wrote).

---

## 10. One-line summary

Raw SutureBot2 data → **Notebook 1** makes a clean, leak-free stitch table → **Notebook 2** trains and tests the hybrid ordinal model → **Notebook 3** produces the figures and the doctor-facing feedback, and everything is summarised on the `index.html` overview page.
