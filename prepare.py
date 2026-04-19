"""
FROZEN -- Do not modify this file.
Data loading, evaluation metric, logging, and plotting.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from processing.tfdata import get_dataset
from processing.features import dataset_to_dataframe
import model


RESULTS_FILE = "results.tsv"


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_data():
    train_ds = get_dataset(
        "data/next_day_wildfire_spread_train_*.tfrecord",
        data_size=64,
        sample_size=32,
        batch_size=32,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False,
    )

    eval_ds = get_dataset(
        "data/next_day_wildfire_spread_eval_*.tfrecord",
        data_size=64,
        sample_size=32,
        batch_size=32,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=True,
    )

    df_train = dataset_to_dataframe(train_ds)
    df_eval = dataset_to_dataframe(eval_ds)

    return df_train, df_eval


# ------------------------------------------------------------
# Evaluation (frozen metric)
# ------------------------------------------------------------

def evaluate(df_train, df_eval):
    y_true = df_eval["fire_any"].values
    y_scores = model.compute_metric(df_train, df_eval)
    auc = roc_auc_score(y_true, y_scores)
    return float(auc)


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

def log_result(experiment_id, val_auc, status, description):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_auc", "status", "description"])
        writer.writerow([experiment_id, f"{val_auc:.6f}", status, description])


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_results(save_path="performance.png"):
    if not os.path.exists(RESULTS_FILE):
        return

    aucs, statuses = [], []

    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            aucs.append(float(row["val_auc"]))
            statuses.append(row["status"])

    best_so_far = np.maximum.accumulate(aucs)

    plt.figure(figsize=(8, 5))
    plt.plot(aucs, "o-", alpha=0.6, label="AUC")
    plt.plot(best_so_far, linewidth=2.5, label="Best so far")
    plt.ylabel("Validation ROC-AUC")
    plt.xlabel("Experiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
