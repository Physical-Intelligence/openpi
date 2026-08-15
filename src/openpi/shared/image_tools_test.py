import jax.numpy as jnp
import torch

from openpi.shared import image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)


def test_resize_with_pad_torch_preserves_batch_dim():
    # Regression test for #805: resize_with_pad_torch must preserve the input
    # rank, matching the JAX resize_with_pad. A channels-last batch of size 1
    # (4D) must stay 4D and not be squeezed down to a single 3D image.
    images = torch.zeros((1, 480, 640, 3), dtype=torch.float32)
    resized = image_tools.resize_with_pad_torch(images, 224, 224)
    assert tuple(resized.shape) == (1, 224, 224, 3)

    # A genuinely unbatched 3D image still returns unbatched.
    single = torch.zeros((480, 640, 3), dtype=torch.float32)
    resized_single = image_tools.resize_with_pad_torch(single, 224, 224)
    assert tuple(resized_single.shape) == (224, 224, 3)
