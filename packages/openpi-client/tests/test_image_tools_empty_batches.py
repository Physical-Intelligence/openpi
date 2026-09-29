"""Regression proposal for the NumPy/Pillow client helper, not the JAX model path."""

import numpy as np
import pytest
from PIL import Image

from openpi_client import image_tools as module


@pytest.mark.parametrize("batch_shape", [(0,), (2, 0), (0, 2)])
def test_empty_batch_resize_preserves_shape_and_dtype(batch_shape):
    images = np.empty((*batch_shape, 8, 12, 3), dtype=np.uint8)
    result = module.resize_with_pad(images, height=4, width=6)
    assert result.shape == (*batch_shape, 4, 6, 3)
    assert result.dtype == images.dtype
    assert result.size == 0


def test_empty_batch_does_not_call_pillow(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Pillow must not process a batch with no images")

    monkeypatch.setattr(module.Image, "fromarray", fail)
    assert module.resize_with_pad(np.empty((0, 8, 12, 3), dtype=np.uint8), 4, 6).shape == (0, 4, 6, 3)


def test_identity_resize_is_unchanged():
    images = np.empty((0, 4, 6, 3), dtype=np.uint8)
    assert module.resize_with_pad(images, 4, 6) is images


@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 2)])
def test_nonempty_images_match_pillow_reference(batch_shape):
    images = np.arange(int(np.prod((*batch_shape, 8, 12, 3))), dtype=np.uint8).reshape((*batch_shape, 8, 12, 3))
    expected = np.stack(
        [np.array(Image.fromarray(image).resize((6, 4), Image.BILINEAR)) for image in images.reshape(-1, 8, 12, 3)]
    )
    expected = expected.reshape((*batch_shape, 4, 6, 3))
    np.testing.assert_array_equal(module.resize_with_pad(images, 4, 6), expected)
