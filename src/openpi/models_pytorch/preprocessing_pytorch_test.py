"""A small image rotation must preserve a uniform field up to its sample footprint."""

from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import preprocessing_pytorch


@pytest.mark.parametrize("seed", [0, 2, 6])
@pytest.mark.parametrize("resolution", [8, 224])
def test_training_rotation_preserves_uniform_field_at_edge_pixel_centers(seed, resolution):
    image = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32)
    observation = SimpleNamespace(
        images={"base_0_rgb": image},
        image_masks={},
        state=torch.zeros((1, 4)),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    torch.manual_seed(seed)

    actual = preprocessing_pytorch.preprocess_observation_pytorch(
        observation, train=True, image_keys=("base_0_rgb",), image_resolution=(resolution, resolution)
    ).images["base_0_rgb"][0]

    # Under the configured +/-5-degree rotation, one of the two middle left-edge
    # pixel centers remains inside the constant image for either rotation sign.
    # Color changes act uniformly on equal samples and do not change this equality.
    middle = resolution // 2
    edge_value = actual[middle - 1 : middle + 1, 0, :].max(dim=0).values
    center_value = actual[middle, middle, :]
    torch.testing.assert_close(edge_value, center_value, rtol=0, atol=1e-6)


def test_eval_preserves_uniform_field():
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    observation = SimpleNamespace(
        images={"base_0_rgb": image},
        image_masks={},
        state=torch.zeros((1, 4)),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )

    actual = preprocessing_pytorch.preprocess_observation_pytorch(
        observation, image_keys=("base_0_rgb",), image_resolution=(8, 8)
    ).images["base_0_rgb"]

    torch.testing.assert_close(actual, image, rtol=0, atol=0)
