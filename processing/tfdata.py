"""
TFRecord parsing and preprocessing utilities for wildfire dataset.
"""

import tensorflow as tf
import re
from typing import Dict, List, Text, Tuple



INPUT_FEATURES = [
    'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
    'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask'
]

OUTPUT_FEATURES = ['FireMask']



DATA_STATS = {
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    'sph': (0., 1., 0.0071658953, 0.0042835088),
    'th': (0., 360.0, 190.32976, 72.59854),
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    'population': (0., 2534.06298828125, 25.531384, 154.72331),
    'PrevFireMask': (-1., 1., 0., 1.),
    'FireMask': (-1., 1., 0., 1.)
}

#preprocessing helper functions

def _get_base_key(key: Text) -> Text:
    match = re.match(r'([a-zA-Z]+)', key)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid key format: {key}")


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan(inputs - mean, std)


def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    base_key = _get_base_key(key)
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan(inputs - min_val, max_val - min_val)



# Parsing and dataset loading

def _get_features_dict(sample_size: int, features: List[Text]) -> Dict[Text, tf.io.FixedLenFeature]:
    shape = [sample_size, sample_size]
    return {
        key: tf.io.FixedLenFeature(shape=shape, dtype=tf.float32)
        for key in features
    }


def _parse_fn(
    example_proto: tf.train.Example,
    data_size: int,
    sample_size: int,
    num_in_channels: int,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
    center_crop: bool
) -> Tuple[tf.Tensor, tf.Tensor]:

    feature_names = INPUT_FEATURES + OUTPUT_FEATURES
    features = tf.io.parse_single_example(
        example_proto,
        _get_features_dict(data_size, feature_names)
    )

    if clip_and_normalize:
        inputs = [_clip_and_normalize(features[k], k) for k in INPUT_FEATURES]
    elif clip_and_rescale:
        inputs = [_clip_and_rescale(features[k], k) for k in INPUT_FEATURES]
    else:
        inputs = [features[k] for k in INPUT_FEATURES]

    input_img = tf.transpose(tf.stack(inputs, axis=0), [1, 2, 0])
    output_img = tf.transpose(tf.stack([features['FireMask']], axis=0), [1, 2, 0])
    output_img = tf.where(output_img < 0, 0, output_img)

    return input_img, output_img


def get_dataset(
    file_pattern: Text,
    data_size: int,
    sample_size: int,
    batch_size: int,
    num_in_channels: int,
    compression_type: Text,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
    center_crop: bool
) -> tf.data.Dataset:

    if clip_and_normalize and clip_and_rescale:
        raise ValueError("Cannot normalize and rescale simultaneously.")

    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        lambda x: _parse_fn(
            x, data_size, sample_size, num_in_channels,
            clip_and_normalize, clip_and_rescale,
            random_crop, center_crop
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
