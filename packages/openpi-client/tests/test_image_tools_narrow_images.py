"""Thin-image letterbox regressions with a pixel-level reference."""

import numpy as np
import pytest

from openpi_client import image_tools as target


@pytest.mark.parametrize("shape, row, col", [((1, 1000, 3), 49, None), ((1000, 1, 3), None, 49)])
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_thin_images_are_centered_without_zero_dimensions(shape, row, col, batch):
    module = target
    image = np.full((*batch, *shape), 255, np.uint8)
    out = module.resize_with_pad(image, 100, 100)
    expected = np.zeros((*batch, 100, 100, 3), np.uint8)
    if row is not None:
        expected[..., row, :, :] = 255
    else:
        expected[..., :, col, :] = 255
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == image.dtype


def test_regular_letterbox_unchanged():
    image = np.full((10, 20, 3), 255, np.uint8)
    expected = np.zeros((8, 8, 3), np.uint8)
    expected[2:6] = 255
    np.testing.assert_array_equal(target.resize_with_pad(image, 8, 8), expected)


def test_already_matching_image_returns_same_object():
    image = np.zeros((8, 8, 3), np.uint8)
    assert target.resize_with_pad(image, 8, 8) is image
