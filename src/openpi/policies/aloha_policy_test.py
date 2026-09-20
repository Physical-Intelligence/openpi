import numpy as np
import pytest

from openpi.policies import aloha_policy


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (2, 3)])
@pytest.mark.parametrize("action_dim", [14, 32])
@pytest.mark.parametrize("adapt_to_pi", [False, True])
def test_aloha_outputs_batch_matches_individual_samples(batch_shape, action_dim, adapt_to_pi):
    shape = (*batch_shape, 5, action_dim)
    actions = np.linspace(-1.0, 1.0, np.prod(shape)).reshape(shape)
    transform = aloha_policy.AlohaOutputs(adapt_to_pi=adapt_to_pi)

    expected = np.stack(
        [transform({"actions": sample})["actions"] for sample in actions.reshape(-1, 5, action_dim)]
    ).reshape(*batch_shape, 5, 14)
    result = transform({"actions": actions})["actions"]

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected)
