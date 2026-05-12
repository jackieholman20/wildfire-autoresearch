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

import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_metric(df_train, df_eval):
    """
    Baseline model: logistic regression using wind speed only (vs_5),
    the center pixel of the 3x3 neighborhood.

    Parameters
    ----------
    df_train : pd.DataFrame
        Must contain 'vs_5' and 'fire_any'
    df_eval : pd.DataFrame
        Must contain 'vs_5'

    Returns
    -------
    np.ndarray
        Predicted probabilities of wildfire spread for df_eval,
        shape (n_samples,)
    """
    base_features = ["vs_5"]  # center pixel wind speed only

    X_train = df_train[base_features]
    y_train = df_train["fire_any"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(df_eval[base_features])[:, 1]
    return probs