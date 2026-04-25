"""
Feature extraction utilities.

This module defines the full, fixed feature universe available to
AutoResearch. Each spatial TFRecord tile is aggregated into a single
row of tabular features.

This file should be edited ONCE and then treated as frozen.
"""

import tensorflow as tf
import pandas as pd


def tile_to_features(inputs: tf.Tensor, labels: tf.Tensor) -> dict:
    """
    Convert a single spatial tile into tabular features.

    Parameters
    ----------
    inputs : tf.Tensor
        Shape (H, W, 12), preprocessed input channels in fixed order.
    labels : tf.Tensor
        Shape (H, W, 1), next-day fire mask.

    Returns
    -------
    dict
        Dictionary of scalar features for one tile.
    """

    return {
        # Topography
        "elevation_mean": tf.reduce_mean(inputs[:, :, 0]).numpy(),

        # Wind
        "th_mean": tf.reduce_mean(inputs[:, :, 1]).numpy(),
        "vs_mean": tf.reduce_mean(inputs[:, :, 2]).numpy(),

        # Temperature
        "tmmn_mean": tf.reduce_mean(inputs[:, :, 3]).numpy(),
        "tmmx_mean": tf.reduce_mean(inputs[:, :, 4]).numpy(),

        # Atmosphere / moisture
        "sph_mean": tf.reduce_mean(inputs[:, :, 5]).numpy(),
        "pr_mean": tf.reduce_mean(inputs[:, :, 6]).numpy(),

        # Drought / vegetation
        "pdsi_mean": tf.reduce_mean(inputs[:, :, 7]).numpy(),
        "ndvi_mean": tf.reduce_mean(inputs[:, :, 8]).numpy(),

        # Human factors
        "population_mean": tf.reduce_mean(inputs[:, :, 9]).numpy(),

        # Fire danger
        "erc_mean": tf.reduce_mean(inputs[:, :, 10]).numpy(),

        # Previous fire (context)
        "prev_fire_mean": tf.reduce_mean(inputs[:, :, 11]).numpy(),

        # Target variable
        "fire_any": int(tf.reduce_max(labels) > 0),
    }


def dataset_to_dataframe(dataset: tf.data.Dataset) -> pd.DataFrame:
    """
    Convert a TF dataset of spatial tiles into a tabular DataFrame.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Yields (inputs, labels) batches.

    Returns
    -------
    pd.DataFrame
        One row per tile with fixed feature columns.
    """
    rows = []

    for inputs, labels in dataset:
        batch_size = inputs.shape[0]
        for i in range(batch_size):
            rows.append(tile_to_features(inputs[i], labels[i]))

    return pd.DataFrame(rows)