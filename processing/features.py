"""
Feature extraction utilities.

This module converts spatial tiles produced by TFRecord datasets
into tabular, tile-level features suitable for statistical models
(e.g., logistic regression) and AutoResearch experiments.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict


# Single-tile feature extraction

def tile_to_features(
    inputs: tf.Tensor,
    labels: tf.Tensor
) -> Dict[str, float]:
    """
    Convert a single spatial tile into tabular features.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tile of shape (H, W, C), already preprocessed.
        Wind speed (vs) is assumed to be channel index 2.
    labels : tf.Tensor
        Output fire mask of shape (H, W, 1).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
          - vs_mean  : mean normalized wind speed per tile
          - fire_any : binary indicator of next-day fire presence
    """
    # Wind speed is channel index 2
    vs_tile = inputs[:, :, 2]

    # Binary label: did fire occur anywhere in tile?
    fire_any = int(tf.reduce_max(labels) > 0)

    return {
        "vs_mean": float(tf.reduce_mean(vs_tile).numpy()),
        "fire_any": fire_any
    }


# Dataset-level conversion

def dataset_to_dataframe(dataset: tf.data.Dataset) -> pd.DataFrame:
    """
    Convert a TF dataset of spatial tiles into a tabular DataFrame.

    Each row corresponds to one tile.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Dataset yielding (inputs, labels) batches.

    Returns
    -------
    pd.DataFrame
        Columns:
          - vs_mean
          - fire_any
    """
    rows = []

    for inputs, labels in dataset:
        # inputs: (B, H, W, C)
        # labels: (B, H, W, 1)
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            row = tile_to_features(inputs[i], labels[i])
            rows.append(row)

    return pd.DataFrame(rows)
