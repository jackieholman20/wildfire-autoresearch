# AutoResearch Agent Instructions — Wildfire Spread Prediction

## Objective

Maximize **validation ROC‑AUC** for predicting **next‑day wildfire spread**
using satellite‑derived environmental features.

A baseline model using **mean wind speed only** achieves a validation
ROC‑AUC of approximately **0.52**. Your goal is to autonomously discover
feature combinations and models that improve upon this baseline.

---


## Setup
To set up a new experiment, work with the user to:

1. Agree on a run tag: propose a tag based on today's date (e.g. apr30). The branch autoresearch/<tag> must not already exist — this is a fresh run.
2. Create the branch: git checkout -b autoresearch/<tag> from current master.
3. Read the in-scope files: Read these files for full context:
- model.py — the only file you may modify. Model definition, feature engineering, training.
- run.py — frozen orchestration layer. Loads data, calls compute_metric, logs, and plots.
- prepare.py — frozen. Data loading, evaluation (ROC-AUC), logging, and plotting logic.

4. Initialize results.tsv: Create results.tsv with just the header row if it does not already exist. The baseline will be recorded after the first run.
5. Confirm and go: Confirm setup looks good, then kick off the experimentation loop.

## Experimentation Rules
What you **CAN** do:
1. Modify model.py — this is the only file you edit. Everything inside is fair game: features, feature interactions, model architecture, hyperparameters.

What you **CANNOT** do:
1. Modify prepare.py, run.py, or anything under processing/.
2. Change the name or signature of compute_metric.
3. Add new files, datasets, or external downloads.
4. Install new packages beyond what is already available.
5. Compute ROC-AUC inside model.py or hard-code labels.
6. Return class labels — compute_metric must return predicted probabilities.

**The goal is simple: maximize validation ROC-AUC.**
- A baseline model using mean wind speed only achieves approximately 0.52. Every experiment must complete in under 60 seconds on CPU.
- **Simplicity criterion:** All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a win. When evaluating whether to keep a change, weigh complexity cost against improvement magnitude. A 0.001 ROC-AUC gain from 30 extra lines of hacky code? Probably not worth it. A 0.001 gain from deleting code? Keep it. Equal performance but cleaner code? Keep it.
- **The first run:** Your very first run should always establish the baseline — run the script as-is before making any changes.

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


## Ideas to Explore

- **Include additional predictors:** erc_mean, pdsi_mean, ndvi_mean, tmmx_mean, prev_fire_mean, sph_mean, th_mean, pr_mean, population_mean, elevation_mean
- Feature interactions and nonlinear transforms (e.g. erc_mean * vs_mean, log transforms)
- Model choices: regularized logistic regression, RandomForest, GradientBoosting, HistGradientBoosting
- Hyperparameter sweeps on the current best model

---

## Output Format
- Each run produces a single line printed to stdout:
```
✅ Run completed. Validation ROC-AUC: 0.7312
```
- You can also read the last logged result directly from results.tsv to confirm it was recorded correctly.

---

## Logging results
- When an experiment finishes, it is automatically logged to results.tsv by run.py. However, you must also manually append a row to keep your own record with status and description, since run.py does not write those fields.
- The TSV is tab-separated (NOT comma-separated — commas break descriptions) with this header:
```tsv
experiment_id	val_auc	status	description
```

1. experiment_id — a short unique label for this run (e.g. exp_001, baseline)
2. val_auc — ROC-AUC achieved (e.g. 0.731200) — use 0.000000 for crashes
3. status — keep, discard, or crash
4. description — short text description of what this experiment tried

- An example output:

```tsv
experiment_id	val_auc	status	description
baseline	0.520000	keep	baseline: wind speed logistic regression
exp_001	0.651200	keep	added erc_mean, pdsi_mean, gradient boosting
exp_002	0.648900	discard	switched to random forest, worse than GBM
exp_003	0.000000	crash	added polynomial features (timeout >60s)
exp_004	0.663100	keep	erc*vs and tmmx*erc interaction terms
```

Do not commit results.tsv — leave it untracked by git.

## The Experiment Loop
The experiment runs on a dedicated branch (e.g. autoresearch/apr30).

**LOOP FOREVER:**

1. Read results.tsv to review all prior experiments and the current best ROC-AUC.
2. Read the current model.py to understand the starting point.
3. Propose ONE concrete modification — a new feature, interaction, model change, or hyperparameter tweak.
4. Edit model.py with the change.
5. git commit -m "feat: <short description>"
6. Run: python run.py "<experiment_id>" "<description>" "<keep|discard|crash>" > run.log 2>&1
7. Read the result: tail -n 5 run.log
8. If the output is missing or the run crashed, read tail -n 50 run.log for the stack trace and attempt a fix. If the idea is fundamentally broken, skip it, log crash, and move on.
9. Append a row to results.tsv with the commit hash, val_auc, status, and description.
10. If improved (val_auc higher than current best): keep the commit — you have advanced the branch.
11. If equal or worse: git reset HEAD~1 && git checkout model.py to undo the commit and revert the file.

**Crashes:** If a run crashes or exceeds 60 seconds, use your judgment. If it's a trivial bug (typo, missing import), fix and re-run. If the idea is broken, log it as crash with 0.000000 and move on.

**NEVER STOP:** Once the experiment loop has begun, do NOT pause to ask the human whether to continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be away and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files, revisit near-misses, try combining successful ideas, try more radical model changes. The loop runs until the human interrupts you, period.
