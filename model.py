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

from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_NAMES = [
    "elevation", "th", "vs", "tmmn", "tmmx",
    "sph", "pr", "pdsi", "ndvi", "population", "erc", "prev_fire",
]

ALL_PIXEL_FEATURES = [f"{feat}_{i}" for feat in FEATURE_NAMES for i in range(1, 10)]


def compute_metric(df_train, df_eval):
    X_train = df_train[ALL_PIXEL_FEATURES]
    X_eval = df_eval[ALL_PIXEL_FEATURES]
    y_train = df_train["fire_any"]

    model = HistGradientBoostingClassifier(
        max_iter=7000,
        max_depth=4,
        learning_rate=0.05,
        early_stopping=False,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]