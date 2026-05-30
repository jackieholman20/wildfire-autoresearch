# AutoResearch Driven Discovery of Composite Satellite Metrics for Predicting Wildfire Spread

# AutoResearch — Wildfire Spread Prediction

An autonomous ML research agent that discovers models for next-day wildfire spread prediction using satellite-derived environmental features. Instead of manually designing experiments, an AI agent (Claude Code) proposes candidate models, evaluates them on a frozen pipeline, and iterates — keeping improvements and discarding failures — without human intervention between runs.

---

## Project Overview

Wildfire spread is driven by complex interactions between wind, drought, vegetation, and temperature. No single satellite variable captures this on its own. This project delegates the search for good predictive models to an AutoResearch loop:

1. Propose a single change to `model.py` (feature, model, hyperparameter)
2. Run the frozen evaluation pipeline
3. Compare validation ROC-AUC against the current best
4. Keep if improved, discard if worse
5. Repeat autonomously until interrupted

The human defines the metric and constraints. The agent performs the search.

---

## Two Phases

### Phase 1 — Tile-Level Prediction

The original formulation treats each 64×64 km tile as a single example. The target label is `fire_any = 1` if any pixel in the tile burned the next day. Features are spatial means of 12 satellite variables per tile.

- **Training examples**: ~14,979 tiles
- **Positive rate**: ~89% (tiles containing any fire)
- **Best ROC-AUC**: 0.982 (GradientBoosting, 12 features + interactions)
- **Limitation**: High positive rate inflates ROC-AUC; task is easier than pixel-level spread prediction

### Phase 2 — Pixel-Level Prediction (Current)

Restructured to emit one row per pixel using a 3×3 km neighborhood window, matching Huot et al. (2022) more closely. The target is whether any pixel in the 3×3 neighborhood burns the next day. Features are per-neighbor-position satellite values (108 raw features) or neighborhood means (12 aggregated features).

- **Training examples**: ~28,756 pixels (150 tiles × 5% sample rate)
- **Positive rate**: ~50% eval, ~2.5% test
- **Best validation ROC-AUC**: 0.747 (GradientBoosting, 12 neighborhood means, balanced weights, `max_features=sqrt`)
- **Test PR-AUC**: 0.207 (vs. paper's Random Forest baseline of 22.5%)

---

## Dataset

This project uses the **Next Day Wildfire Spread** dataset (Huot et al., 2022).

Each sample is a 64×64 km spatial tile at 1 km resolution with 12 aligned satellite-derived features and a binary next-day fire mask.

**Download**: [Kaggle — Next Day Wildfire Spread](https://www.kaggle.com/datasets/huot25/next-day-wildfire-spread)

Place TFRecord files in the following structure:

```
data/
├── next_day_wildfire_spread_train_*.tfrecord
├── next_day_wildfire_spread_eval_*.tfrecord
└── next_day_wildfire_spread_test_*.tfrecord
```

Available features: elevation, wind speed/direction, min/max temperature, humidity, precipitation, drought index (PDSI), vegetation (NDVI), population density, energy release component (ERC), previous fire mask.

---

## Project Structure

```
wildfire-autoresearch/
├── processing/
│   ├── tfdata.py          # TFRecord parsing and preprocessing (frozen)
│   └── features.py        # Spatial → tabular feature extraction (frozen)
├── prepare.py             # Data loading, evaluation, logging, plotting (frozen)
├── model.py               # Candidate model definition (editable — agent modifies this only)
├── run.py                 # Executes one experiment and logs result
├── program.md             # AutoResearch agent instructions
├── results.tsv            # Experiment log (auto-generated, not committed)
├── performance.png        # Performance plot (auto-generated)
└── test.py                # Held-out test set evaluation (run once at end)
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Setup

### Requirements

Python 3.10+ with the following packages:

```bash
pip install scikit-learn matplotlib numpy tensorflow
```

Or with `uv`:

```bash
uv pip install scikit-learn matplotlib numpy tensorflow
```

No GPU required — all experiments run on CPU.

### Verify

```bash
python3 -c "import sklearn, matplotlib, numpy, tensorflow; print('All good')"
```

---

## Running Experiments

### Baseline

```bash
python run.py "baseline" "logistic regression, center pixel wind speed only" "keep"
```

### Agent Loop

Pass `program.md` as the system prompt to Claude Code and confirm setup looks good before starting. The agent will:

- Read `results.tsv` for prior experiment history
- Read `model.py` for the current approach
- Propose and run one experiment per iteration
- Commit improvements to the `autoresearch/<tag>` branch
- Revert failures with `git reset HEAD~1 && git checkout model.py`

### Manual Run

```bash
python run.py "<experiment_id>" "<description>" "<keep|discard|crash>"
```

Results are logged to `results.tsv` automatically.

### Test Set Evaluation

After the agent loop completes, evaluate the best model on held-out test data:

```bash
python test.py
```

---

## Results Summary

| Phase | Model | Val ROC-AUC | Test PR-AUC |
|-------|-------|------------|-------------|
| Tile-level (Phase 1) | GradientBoosting, 12 means + interactions | 0.982 | — |
| Pixel-level (Phase 2) | GradientBoosting, 12 means, balanced, sqrt | **0.747** | **0.207** |
| Paper baseline (Huot 2022) | Random Forest | — | 0.225 |
| Paper best (Huot 2022) | Neural Network | — | 0.284 |

Phase 1 ROC-AUC is inflated by the 89% tile-level positive rate and the coarseness of the label. Phase 2 is the more honest and meaningful result.

---

## Experiment Log Format

`results.tsv` is tab-separated with the following columns:

```
experiment    val_auc    duration_s    status    description
```

- `experiment` — short ID (e.g. `exp_001`, `baseline`)
- `val_auc` — validation ROC-AUC (6 decimal places)
- `duration_s` — wall-clock seconds for the run
- `status` — `keep`, `discard`, or `crash`
- `description` — brief description of what was changed

Do not commit `results.tsv` — it is untracked by git.

---

## Reference

Fantine Huot, R. Lily Hu, Nita Goyal, Tharun Sankar, Matthias Ihme, and Yi-Fan Chen. *Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data.* arXiv:2112.02447, 2022.