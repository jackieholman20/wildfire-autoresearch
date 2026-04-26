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
from sklearn.ensemble import GradientBoostingClassifier


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
    base_features = ["vs_mean", "erc_mean", "pdsi_mean", "tmmx_mean", "prev_fire_mean",
                     "sph_mean", "ndvi_mean", "tmmn_mean",
                     "elevation_mean", "th_mean", "pr_mean", "population_mean"]

    def add_interactions(df):
        import pandas as pd
        X = df[base_features].copy()
        X["erc_x_vs"] = df["erc_mean"] * df["vs_mean"]   # fire danger × wind
        X["tmmx_x_erc"] = df["tmmx_mean"] * df["erc_mean"]  # heat × fire danger
        return X

    X_train = add_interactions(df_train)
    y_train = df_train["fire_any"]


    model = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(add_interactions(df_eval))[:, 1]
    return probs