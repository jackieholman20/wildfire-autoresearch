"""
EDITABLE — The AutoResearch agent may modify this file only.

This file defines the candidate model used to predict next-day
wildfire spread.

The function `compute_metric` must:
- train on df_train
- return predicted probabilities on df_eval

Evaluation (ROC-AUC), logging, and plotting are handled elsewhere
and are frozen.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_NAMES = [
    "elevation", "th", "vs", "tmmn", "tmmx",
    "sph", "pr", "pdsi", "ndvi", "population", "erc", "prev_fire",
]


def _make_features(df):
    data = {}
    for feat in FEATURE_NAMES:
        cols = [f"{feat}_{i}" for i in range(1, 10)]
        data[f"{feat}_mean"] = df[cols].values.mean(axis=1)
    return pd.DataFrame(data)


def compute_metric(df_train, df_eval):
    X_train = _make_features(df_train)
    X_eval = _make_features(df_eval)
    y_train = df_train["fire_any"]
    sample_weight = compute_sample_weight("balanced", y_train)

    model = GradientBoostingClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_eval)[:, 1]