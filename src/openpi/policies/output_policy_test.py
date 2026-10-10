import numpy as np
import pytest

from openpi.policies import droid_policy
from openpi.policies import libero_policy


@pytest.mark.parametrize(
    ("output_transform", "output_dim"),
    [
        pytest.param(droid_policy.DroidOutputs, 8, id="droid"),
        pytest.param(libero_policy.LiberoOutputs, 7, id="libero"),
    ],
)
@pytest.mark.parametrize("batch_shape", [(), (1,), (2,)], ids=["unbatched", "batch-one", "batch-two"])
@pytest.mark.parametrize("horizon", [5, 11])
@pytest.mark.parametrize("padded", [False, True], ids=["native", "padded"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("non_contiguous", [False, True], ids=["contiguous", "strided"])
def test_outputs_trim_action_dimension(
    output_transform, output_dim, batch_shape, horizon, padded, dtype, non_contiguous
):
    action_dim = 32 if padded else output_dim
    stride = 2 if non_contiguous else 1
    storage_shape = (*batch_shape, horizon, action_dim * stride)
    # Every batch, timestep, and action coordinate has a distinct value.
    actions = np.arange(np.prod(storage_shape), dtype=dtype).reshape(storage_shape)[..., ::stride]
    if non_contiguous:
        assert not actions.flags.c_contiguous
    original = actions.copy()
    expected = np.take(original, np.arange(output_dim), axis=-1)

    result = output_transform()({"actions": actions})["actions"]

    assert isinstance(result, np.ndarray)
    assert result.shape == (*batch_shape, horizon, output_dim)
    assert result.dtype == actions.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(actions, original)
