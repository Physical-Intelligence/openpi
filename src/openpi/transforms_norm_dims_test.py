import numpy as np
import pytest

from openpi import transforms
from openpi.shared.normalize import NormStats


@pytest.mark.parametrize("use_quantiles", [False, True])
@pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3, 2)])
def test_unnormalize_inverts_normalize_for_truncated_features(use_quantiles, shape):
    stats = NormStats(
        mean=np.array([1.0, 2.0, 3.0, 4.0]),
        std=np.array([0.5, 2.0, 1.0, 3.0]),
        q01=np.array([-2.0, -1.0, 0.0, 1.0]),
        q99=np.array([2.0, 5.0, 3.0, 8.0]),
    )
    actions = np.arange(np.prod(shape), dtype=float).reshape(shape)
    normalized = transforms.Normalize({"actions": stats}, use_quantiles=use_quantiles)({"actions": actions})
    restored = transforms.Unnormalize({"actions": stats}, use_quantiles=use_quantiles)(normalized)
    assert restored["actions"].shape == actions.shape
    np.testing.assert_allclose(restored["actions"], actions, atol=1e-12)


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_unnormalize_preserves_existing_padding_behavior(use_quantiles):
    stats = NormStats(
        mean=np.array([1.0, 2.0]),
        std=np.array([0.5, 2.0]),
        q01=np.array([-2.0, -1.0]),
        q99=np.array([2.0, 5.0]),
    )
    actions = np.array([[0.25, -0.5, 1.0, 0.0]])
    result = transforms.Unnormalize({"actions": stats}, use_quantiles=use_quantiles)({"actions": actions})
    expected_tail = actions[..., 2:] if use_quantiles else actions[..., 2:] * (1.0 + 1e-6)
    np.testing.assert_allclose(result["actions"][..., 2:], expected_tail)
