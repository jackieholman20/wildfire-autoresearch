# AutoResearch Agent Instructions — Wildfire Spread Prediction

## Objective

Maximize **validation ROC‑AUC** for predicting **next‑day wildfire spread**
using satellite‑derived environmental features.

A baseline model using **mean wind speed only** achieves a validation
ROC‑AUC of approximately **0.52**. Your goal is to autonomously discover
feature combinations and models that improve upon this baseline.

---

## Rules (Strict)

1. You may **ONLY modify `model.py`**
2. `processing/`, `prepare.py`, and `run.py` are **FROZEN** — do not edit them
3. You must **not** change the function name or signature of `compute_metric`
4. The model must return **predicted probabilities**, not class labels
5. Each experiment must complete in **under 60 seconds** on CPU
6. Do not add new files, datasets, or external downloads

Violating any rule invalidates the experiment.

---

## Data Context

Each row represents a 64×64 km spatial tile derived from satellite data.

Available columns in `df_train` and `df_eval` may include (not exhaustive):

- `vs_mean`   — mean normalized wind speed
- `erc_mean`  — energy release component
- `pdsi_mean` — Palmer drought severity index
- `ndvi_mean` — vegetation index
- other aggregated satellite features

Target variable:

- `fire_any` — binary indicator of wildfire spread within the next 24 hours

You do **not** need to load or preprocess TFRecords — this is already handled.

---

## Required Workflow

```
1. Read current model.py
2. Propose ONE concrete modification (feature combination, interaction, or model change)
3. Edit model.py
4. Run:  python run.py "description of change"
5. Compare against the current best ROC‑AUC:
6. If improved:  git add model.py && git commit -m "feat: "
7. If worse:     git checkout model.py   (revert)
8. Repeat from step 1
```


---

## Ideas to Explore

- **Feature additions**
  - Include additional predictors such as `erc_mean`, `pdsi_mean`, or `ndvi_mean`
  - Combine features using simple interactions or nonlinear transforms

- **Model choices**
  - Regularized logistic regression variants
  - Tree‑based classifiers (RandomForest, GradientBoosting)
  - Other sklearn‑compatible binary classifiers

- **Inductive bias**
  - Prefer simpler, more interpretable models when performance is comparable
  - Introduce nonlinearity only when justified by performance gains

---

## What NOT to Do

- Do NOT modify `prepare.py`, `run.py`, or anything under `processing/`
- Do NOT compute ROC‑AUC inside `model.py`
- Do NOT add new files or external dependencies
- Do NOT hard‑code labels or outcomes
- Do NOT change the name or signature of `compute_metric`
