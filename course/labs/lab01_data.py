"""Lab 01 — Data & representations.

Implement quantile normalization, build a fake episode, and turn it into a valid
Observation. Verify against openpi's real transforms.

Run:  uv run python course/labs/lab01_data.py
"""

import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model


# ----------------------------------------------------------------------------
# Part 1: quantile normalization (Module 01 §3)
# ----------------------------------------------------------------------------
def quantile_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Map [q01, q99] -> [-1, 1]. Match transforms.Normalize._normalize_quantile."""
    # TODO(you): implement.  (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
    raise NotImplementedError


def quantile_unnormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Inverse of quantile_normalize. Match Unnormalize._unnormalize_quantile."""
    # TODO(you): implement.
    raise NotImplementedError


def check_part1():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 7)).astype(np.float32)
    q01 = np.full((7,), -2.0, np.float32)
    q99 = np.full((7,), 2.0, np.float32)

    xn = quantile_normalize(x, q01, q99)
    xr = quantile_unnormalize(xn, q01, q99)
    assert np.allclose(x, xr, atol=1e-4), "normalize/unnormalize must invert"

    # Cross-check against openpi's implementation:
    stats = _transforms.NormStats(mean=q01 * 0, std=q01 * 0 + 1, q01=q01, q99=q99)
    ref = _transforms.Normalize(norm_stats={"x": stats}, use_quantiles=True)
    out = ref({"x": x})["x"]
    assert np.allclose(out, xn, atol=1e-5), "must match transforms.Normalize"
    print("Part 1 OK: quantile normalize matches openpi and round-trips.")


# ----------------------------------------------------------------------------
# Part 2: build a fake episode -> Observation (Module 01 §1)
# ----------------------------------------------------------------------------
def make_fake_episode(action_dim: int = 32) -> dict:
    """Return a raw dict in the 'image'/'image_mask'/'state'/'actions' format that
    Observation.from_dict expects (see model.py docstring). Use the 3 canonical
    image keys, uint8 images, a 7-dim state padded later to action_dim, and a prompt.
    """
    # TODO(you): build and return the dict. Images uint8 [0,255], shape [b,224,224,3].
    raise NotImplementedError


def check_part2():
    data = make_fake_episode()
    # Minimal model transforms (resize already 224, tokenize, pad). We skip tokenize
    # here to avoid a network download; just verify Observation.from_dict works and
    # converts uint8 -> [-1,1] float.
    obs = _model.Observation.from_dict(data)
    for k, img in obs.images.items():
        assert img.dtype == np.float32 and img.min() >= -1.0 and img.max() <= 1.0, k
    assert set(obs.images) == set(_model.IMAGE_KEYS), "must have 3 canonical image keys"
    print("Part 2 OK: fake episode -> valid Observation with [-1,1] images.")


if __name__ == "__main__":
    check_part1()
    check_part2()
    print("\nStretch: feed a tokenized + padded Observation into a dummy model's "
          "compute_loss (see Lab 04) and confirm no shape/type errors.")
