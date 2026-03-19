"""Tests for illumination-aware RM75 image augmentation."""

import numpy as np

from openpi.research.shared.illumination_augment import IlluminationAugmentationConfig
from openpi.research.shared.illumination_augment import RM75IlluminationAugmentation


def _make_data() -> dict:
    return {
        'state': np.arange(8, dtype=np.float32),
        'actions': np.ones((4, 8), dtype=np.float32),
        'prompt': 'pick the object',
        'image': {
            'base_0_rgb': np.full((32, 32, 3), 180, dtype=np.uint8),
            'left_wrist_0_rgb': np.full((32, 32, 3), 120, dtype=np.uint8),
            'right_wrist_0_rgb': np.zeros((32, 32, 3), dtype=np.uint8),
        },
        'image_mask': {
            'base_0_rgb': np.True_,
            'left_wrist_0_rgb': np.True_,
            'right_wrist_0_rgb': np.False_,
        },
    }


def test_illumination_augmentation_preserves_keys_shape_dtype_and_non_image_fields():
    data = _make_data()
    state_before = data['state'].copy()
    actions_before = data['actions'].copy()
    prompt_before = data['prompt']

    augment = RM75IlluminationAugmentation(
        IlluminationAugmentationConfig(apply_probability=1.0, seed=7)
    )
    result = augment(data)

    assert set(result['image'].keys()) == {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}
    for key in result['image']:
        assert result['image'][key].shape == (32, 32, 3)
        assert result['image'][key].dtype == np.uint8

    np.testing.assert_array_equal(result['state'], state_before)
    np.testing.assert_array_equal(result['actions'], actions_before)
    assert result['prompt'] == prompt_before


def test_illumination_augmentation_skips_masked_images():
    data = _make_data()
    padded_before = data['image']['right_wrist_0_rgb'].copy()

    augment = RM75IlluminationAugmentation(
        IlluminationAugmentationConfig(apply_probability=1.0, seed=11)
    )
    result = augment(data)

    np.testing.assert_array_equal(result['image']['right_wrist_0_rgb'], padded_before)
    assert np.any(result['image']['base_0_rgb'] != 180)


def test_illumination_augmentation_probability_zero_is_noop():
    data = _make_data()
    snapshot = {key: value.copy() for key, value in data['image'].items()}

    augment = RM75IlluminationAugmentation(
        IlluminationAugmentationConfig(apply_probability=0.0, seed=5)
    )
    result = augment(data)

    for key, image in snapshot.items():
        np.testing.assert_array_equal(result['image'][key], image)


def test_illumination_augmentation_fixed_seed_is_deterministic():
    config = IlluminationAugmentationConfig(apply_probability=1.0, seed=19)
    first = RM75IlluminationAugmentation(config)(_make_data())
    second = RM75IlluminationAugmentation(config)(_make_data())

    for key in first['image']:
        np.testing.assert_array_equal(first['image'][key], second['image'][key])
