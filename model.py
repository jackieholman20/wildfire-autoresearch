"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for California Housing regression.
The function build_model() must return an sklearn-compatible estimator.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def compute_metric(df_train, df_eval):
    X_train = df_train[["vs_mean"]]
    y_train = df_train["fire_any"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(df_eval[["vs_mean"]])[:, 1]
    return probs
