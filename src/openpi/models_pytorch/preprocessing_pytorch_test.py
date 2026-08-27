from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import preprocessing_pytorch


@pytest.mark.parametrize("image_shape", [(1, 224, 224, 3), (1, 3, 224, 224)])
def test_preprocess_normalizes_images_to_nchw(image_shape):
    image_keys = preprocessing_pytorch.IMAGE_KEYS
    observation = SimpleNamespace(
        images={key: torch.zeros(image_shape) for key in image_keys},
        image_masks={key: torch.ones((1,), dtype=torch.bool) for key in image_keys},
        state=torch.zeros((1, 32)),
        tokenized_prompt=torch.ones((1, 8), dtype=torch.int64),
        tokenized_prompt_mask=torch.ones((1, 8), dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )

    processed = preprocessing_pytorch.preprocess_observation_pytorch(observation)

    assert all(image.shape == (1, 3, 224, 224) for image in processed.images.values())
