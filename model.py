"""
EDITABLE — The AutoResearch agent may modify this file only.

This file defines the candidate model used to predict next‑day
wildfire spread.

The function `compute_metric` must:
- train on df_train
- return predicted probabilities on df_eval

Evaluation (ROC‑AUC), logging, and plotting are handled elsewhere
and are frozen.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def compute_metric(df_train, df_eval):
    """
    Train a model to predict wildfire spread and return evaluation scores.

    Parameters
    ----------
    df_train : pd.DataFrame
        Must contain:
          - 'vs_mean'   : mean normalized wind speed per tile
          - 'fire_any'  : binary target (1 = fire spread, 0 = no spread)

    df_eval : pd.DataFrame
        Must contain:
          - 'vs_mean'

    Returns
    -------
    np.ndarray
        Predicted probabilities of wildfire spread for df_eval,
        shape (n_samples,)
    """
    features = ["vs_mean", "erc_mean", "pdsi_mean", "tmmx_mean", "prev_fire_mean",
                "sph_mean", "ndvi_mean", "tmmn_mean",
                "elevation_mean", "th_mean", "pr_mean", "population_mean"]
    X_train = df_train[features]
    y_train = df_train["fire_any"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(df_eval[features])[:, 1]
    return probs
