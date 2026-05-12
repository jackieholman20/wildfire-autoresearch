"""
Feature extraction utilities.
This module defines the full, fixed feature universe available to
AutoResearch. Each spatial TFRecord tile is aggregated into per-pixel
rows using a 3x3 km neighborhood window.
This file should be edited ONCE and then treated as frozen.
"""
import tensorflow as tf
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Sampling controls — adjust these to manage memory usage
MAX_TILES = 2000        # max number of tiles to process
PIXEL_SAMPLE_RATE = 0.1 # fraction of valid pixels to keep per tile
RANDOM_SEED = 42
# ---------------------------------------------------------------

def tile_to_features(inputs: tf.Tensor, labels: tf.Tensor, rng: np.random.Generator) -> list:
    """
    Convert a single spatial tile into per-pixel tabular features.
    For each pixel, includes its own features plus the 8 neighboring
    pixels' features as spatial context (3x3 km window at 1km resolution).
    Target label is fire_any: did any pixel in the 3x3 neighborhood
    burn next day?
    Excludes the 1-pixel border to ensure all pixels have 8 valid neighbors.
    Skips pixels with uncertain labels (label < 0) matching Huot et al. 2022.
    Randomly subsamples pixels at PIXEL_SAMPLE_RATE.

    Parameters
    ----------
    inputs : tf.Tensor
        Shape (H, W, 12), preprocessed input channels in fixed order.
    labels : tf.Tensor
        Shape (H, W, 1), next-day fire mask.
    rng : np.random.Generator
        Random number generator for pixel subsampling.

    Returns
    -------
    list of dict
        One dict per sampled valid pixel.
    """
    inp = inputs.numpy()
    lab = labels.numpy()

    H, W, _ = inp.shape
    rows = []

    feature_names = [
        "elevation", "th", "vs", "tmmn", "tmmx",
        "sph", "pr", "pdsi", "ndvi",
        "population", "erc", "prev_fire"
    ]

    for i in range(1, H - 1):
        for j in range(1, W - 1):

            # Randomly skip pixels based on sample rate
            if rng.random() > PIXEL_SAMPLE_RATE:
                continue

            # Extract 3x3 neighborhood labels
            neighborhood_labels = lab[i-1:i+2, j-1:j+2, 0]

            # Skip uncertain labels
            if np.any(neighborhood_labels < 0):
                continue

            # Target: did any pixel in the 3x3 neighborhood burn next day?
            fire_any = int(np.any(neighborhood_labels > 0))

            row = {"fire_any": fire_any}

            # 3x3 neighborhood features (pixels 1-9, center = pixel 5)
            neighbor_idx = 1
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    pixel = inp[i + di, j + dj, :]
                    for k, name in enumerate(feature_names):
                        row[f"{name}_{neighbor_idx}"] = float(pixel[k])
                    neighbor_idx += 1

            rows.append(row)

    return rows


def dataset_to_dataframe(dataset: tf.data.Dataset) -> pd.DataFrame:
    """
    Convert a TF dataset of spatial tiles into a per-pixel DataFrame.
    Limits to MAX_TILES tiles and samples PIXEL_SAMPLE_RATE of pixels.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Yields (inputs, labels) batches.

    Returns
    -------
    pd.DataFrame
        One row per sampled pixel with 3x3 neighborhood features.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    tiles_processed = 0

    for inputs, labels in dataset:
        if tiles_processed >= MAX_TILES:
            break
        batch_size = inputs.shape[0]
        for i in range(batch_size):
            if tiles_processed >= MAX_TILES:
                break
            rows.extend(tile_to_features(inputs[i], labels[i], rng))
            tiles_processed += 1

    print(f"Processed {tiles_processed} tiles → {len(rows)} pixels")
    return pd.DataFrame(rows)