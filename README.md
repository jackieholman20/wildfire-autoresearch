# AutoResearch Driven Discovery of Composite Satellite Metrics for Predicting Wildfire Spread

Wildfire expansion is driven by complex interactions between environmental factors such as wind, drought, vegetation, and climate. No single satellite‑derived variable captures this complexity on its own.
This project applies an AutoResearch loop to wildfire prediction: an autonomous agent iteratively proposes candidate composite metrics derived from satellite features, evaluates their predictive power for next‑day wildfire spread, and retains or discards each modification based on validation performance.
The goal is to autonomously discover composite environmental metrics that outperform any single satellite variable when predicting wildfire spread.


---

## Problem

**Task**: Predict whether wildfire spreads within a 64×64 km region over the next 24 hours.
**Metric**: Validation ROC‑AUC (higher is better).
The agent’s objective is to iteratively improve ROC‑AUC relative to a simple, interpretable baseline.


## Data 
This project uses the **Next Day Wildfire Spread** dataset
(Huot et al., Kaggle).
Each sample represents a 64×64 km spatial tile with:

- 12 aligned satellite‑derived input features (elevation, wind speed/direction, humidity, temperature, drought indices, vegetation, population, etc.)
- A binary next‑day fire mask indicating wildfire spread

Due to size constraints, TFRecord files are not included in this repository.

**How to obtain the data**

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/huot25/next-day-wildfire-spread

2. Place the TFRecord files locally in the following structure:

```
data/
├── next_day_wildfire_spread_train_*.tfrecord
├── next_day_wildfire_spread_eval_*.tfrecord
└── next_day_wildfire_spread_test_*.tfrecord
```
##Core Idea: AutoResearch for Wildfire Prediction

Instead of manually designing features and models, this project delegates exploration to an AI‑driven retry loop:

Propose a candidate composite metric or model modification
Train and evaluate it using a frozen evaluation pipeline
Compare validation ROC‑AUC against the current best
Keep if improved, discard if worse
Repeat autonomously

The human defines what “good” means; the agent performs the search.


## Project Structure

```
wildfire-autoresearch/
├── processing/
│   ├── tfdata.py        # TFRecord parsing + preprocessing (frozen)
│   └── features.py      # Spatial → tabular feature extraction (frozen)
├── prepare.py           # Data loading, evaluation, plotting (FROZEN)
├── model.py             # Candidate model definition (EDITABLE)
├── run.py               # Executes one experiment and logs result
├── program.md           # Instructions for the AutoResearch agent
├── results.tsv          # Experiment log (auto-generated)
└── performance.png      # Performance plot (auto-generated)
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Baseline Model
**Before any autonomous exploration, we establish a simple baseline:**

Model: Logistic regression
Feature: Mean wind speed per tile (vs_mean)
Rationale: Wind is a physically meaningful driver of fire spread
Interpretability: Coefficients map directly to odds of spread

Baseline Performance

| Model | Features | Validation ROC-AUC |
|-------|----------|--------------------|
| Logistic Regression (baseline) | Wind speed only | 0.518 |

This intentionally weak baseline provides a clear benchmark that AutoResearch must exceed.


## Setup

### 1. Install an AI coding agent (CLI)

#### Option A: Claude Code CLI 

```bash
# macOS / Linux / WSL — one-line install
curl -fsSL https://claude.ai/install.sh | bash

# macOS — or via Homebrew
brew install --cask claude-code

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

# Windows — or via WinGet
winget install Anthropic.ClaudeCode
```

Then launch:

```bash
cd wildfire-autoresearch
claude
```

First launch opens a browser for login — **no API key needed**.
Works with any Claude subscription (Pro $20/mo, Max, or Team).

### 2. Install Python environment

This project requires **Python 3.10+** with `scikit-learn`, `matplotlib`, and `numpy`.

#### Check if Python is installed

```bash
python3 --version
# Should print Python 3.10.x or higher
# If not installed, see below
```

#### Install Python (if needed)

```bash
# macOS
brew install python@3.12

# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv

# Windows — download from https://www.python.org/downloads/
# During install, check "Add Python to PATH"
```

#### Install dependencies

```bash
# Option A: with pip (simplest)
pip install scikit-learn matplotlib numpy

# Option B: with uv (faster, used in the main autoresearch project)
# Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install scikit-learn matplotlib numpy

# Option C: with conda
conda install scikit-learn matplotlib numpy
```

#### What gets installed

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >= 1.3 | ML models, pipelines, evaluation |
| matplotlib | >= 3.7 | Performance plotting |
| numpy | >= 1.24 | Array operations (scikit-learn dependency) |

No GPU, no PyTorch, no heavy downloads — everything runs on CPU.

### 3. Verify setup

```bash
# Quick check: all imports work
python3 -c "import sklearn, matplotlib, numpy; print('All good')"
```

# Run the baseline experiment
```
python run.py
rm -f results.tsv
```

---

## How to Run the Agent Loop

### Quick start (copy-paste this prompt into your agent)

```
Read program.md for your instructions, then read model.py.
Run `python run.py` to confirm the baseline ROC-AUC.

Then enter the AutoResearch loop:

1. Propose a single modification to model.py
   (e.g., new feature combination, nonlinear model, interactions).
2. Edit model.py with your proposed change.
3. Run: python run.py
4. Compare ROC-AUC against the current best.
   - If improved: KEEP the change.
   - If worse: REVERT model.py.
5. Repeat for at least 10–20 iterations.

Only modify model.py.
```


## Plotting Results

After running experiments:

```bash
python prepare.py
# Generates performance.png from results.tsv
```
