# AutoResearch — Wildfire Spread Prediction

This is an autonomous experiment to discover the best model for predicting next-day wildfire spread at the **pixel level** using a 3×3 km neighborhood window of satellite-derived features.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may30`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `model.py` — the only file you may modify. Model definition, feature engineering, training.
   - `run.py` — frozen orchestration layer. Loads data, calls `compute_metric`, logs, and plots.
   - `prepare.py` — frozen. Data loading, evaluation (ROC-AUC), logging, and plotting logic.
   - `processing/features.py` — frozen. Defines the pixel-level feature schema.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row if it does not already exist. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good, then kick off the experimentation loop.

---

## Experimentation Rules

**What you CAN do:**
- Modify `model.py` — this is the **only** file you edit. Features, feature engineering, model architecture, and hyperparameters are all fair game.

**What you CANNOT do:**
- Modify `prepare.py`, `run.py`, or anything under `processing/`
- Change the name or signature of `compute_metric`
- Add new files, datasets, or external downloads
- Install new packages beyond what is already available
- Compute ROC-AUC inside `model.py` or hard-code labels
- Return class labels — `compute_metric` must return **predicted probabilities**

**The goal is simple: maximize validation ROC-AUC.**

Every experiment must complete in **under 60 seconds** on CPU.

**Simplicity criterion:** All else being equal, simpler is better. A 0.001 ROC-AUC gain from 30 extra lines of hacky code? Not worth it. A 0.001 gain from deleting code? Keep it. Equal performance but cleaner code? Keep it.

**The first run:** Always establish the baseline first — run the script as-is before making any changes.

---

## Data Context

Each row is one **pixel**. Features follow the naming convention `{feature}_{n}` where `n = 1–9` is the position in the 3×3 neighborhood (center pixel = 5).

Available features per neighbor position:

| Feature | Description |
|---------|-------------|
| `elevation_{1-9}` | Topographic elevation |
| `th_{1-9}` | Wind direction |
| `vs_{1-9}` | Wind speed |
| `tmmn_{1-9}` | Minimum temperature |
| `tmmx_{1-9}` | Maximum temperature |
| `sph_{1-9}` | Specific humidity |
| `pr_{1-9}` | Precipitation |
| `pdsi_{1-9}` | Palmer drought severity index |
| `ndvi_{1-9}` | Vegetation index |
| `population_{1-9}` | Population density |
| `erc_{1-9}` | Energy release component |
| `prev_fire_{1-9}` | Previous fire mask |

**Target variable:** `fire_any` — did any pixel in the 3×3 neighborhood burn next day?

You do **not** need to load or preprocess TFRecords — this is already handled.

---

## Known Dead Ends — Do NOT Revisit

These have been thoroughly explored and consistently fail:

- **All 108 raw pixel features** (`ALL_PIXEL_FEATURES`) — times out or underfits
- **HistGradientBoosting with `max_iter` > 500** — exceeds 60s CPU budget
- **Center pixel features only** (e.g. `vs_5` alone) — significantly worse than neighborhood means
- **Feature interactions** (e.g. `erc×vs`, `tmmx×pdsi`) — hurt performance
- **`n_estimators` > 700** for standard GBM — overfits and times out
- **`max_depth` = 5** — consistently worse than depth 4

---

## Current Best

The winning approach as of the last session:

```python
# 12 neighborhood means (average across all 9 positions per feature)
FEATURE_NAMES = ["elevation", "th", "vs", "tmmn", "tmmx",
                 "sph", "pr", "pdsi", "ndvi", "population", "erc", "prev_fire"]

# Compute means across 9 neighbor positions
for feat in FEATURE_NAMES:
    X[f"{feat}_mean"] = df[[f"{feat}_{i}" for i in range(1, 10)]].mean(axis=1)

# GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=700,
    max_depth=4,
    learning_rate=0.05,
    max_features="sqrt",
    random_state=42,
)
model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
```

**Best validation ROC-AUC: 0.7471** (exp_25)

All new experiments should build from this foundation.

---

## Ideas to Explore

Build on the current best — 12 neighborhood means + GBM + balanced weights + `max_features=sqrt`:

- **Hyperparameter tuning**: `learning_rate` (try 0.03, 0.08), `subsample` (try 0.7, 0.9)
- **Selective neighbor features**: instead of all-9 means, use specific positions for key features like `prev_fire` and `erc` (e.g. upwind neighbors only)
- **Additional engineered features**: log transforms on skewed features (`population`, `pr`), ratio features (`tmmx/tmmn`)
- **RandomForest** with the same 12 means and balanced weights as a comparison
- **Feature selection**: drop low-signal features (check if removing `th` or `population` hurts)

---

## Output Format

Each run prints to stdout:

```
✅ Run completed. Validation ROC-AUC: 0.7471 | Duration: 14.9s
```

---

## Logging Results

Results are automatically logged to `results.tsv` by `run.py`. Do **not** manually append rows — this creates duplicates. Only read `results.tsv` to check prior results, never write to it directly.

The TSV is tab-separated with this header:

```
experiment	val_auc	duration_s	status	description
```

Example:

```
experiment	val_auc	duration_s	status	description
baseline	0.519600	0.1	keep	logistic regression, center pixel wind speed only
exp_13	0.723000	58.0	keep	GBM n=700 depth=4 lr=0.05 — 12 neighborhood means
exp_18	0.740800	52.6	keep	balanced sample weights — GBM n=700 12 neighborhood means
exp_25	0.747100	14.9	keep	max_features=sqrt GBM n=700 balanced 12 means — best
```

Do **not** commit `results.tsv` — leave it untracked by git.

---

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/may30`).

**LOOP FOREVER:**

1. Read `results.tsv` to review all prior experiments and the current best ROC-AUC.
2. Read the current `model.py` to understand the starting point.
3. Propose **ONE** concrete modification — a new feature, interaction, model change, or hyperparameter tweak.
4. Edit `model.py` with the change.
5. `git commit -m "feat: <short description>"`
6. Run: `python run.py "<experiment_id>" "<description>" "keep" > run.log 2>&1`
7. Check timing: `grep "Duration" run.log` — if over 60s, log as discard and revert immediately.
8. Read the result: `tail -n 5 run.log`
9. If the output is missing or the run crashed, read `tail -n 50 run.log` for the stack trace. Fix if trivial, log as crash and move on if not.
10. If **improved** (val_auc higher than current best): keep the commit — branch advances.
11. If **equal or worse**: `git reset HEAD~1 && git checkout model.py`

**Crashes:** If a run exceeds 60 seconds or crashes, log it as `crash` with `0.000000` and move on. Do not spend more than 2 attempts fixing the same idea.

**NEVER STOP:** Once the loop begins, do NOT pause to ask the human whether to continue. The human may be away and expects you to continue working indefinitely until manually stopped. If you run out of ideas, re-read the in-scope files, revisit near-misses, try combining successful ideas, or try more radical model changes. The loop runs until the human interrupts you, period.