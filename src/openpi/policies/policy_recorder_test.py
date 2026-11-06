from unittest.mock import Mock

import numpy as np

from openpi.policies import policy


def test_restarting_recorders_preserves_prior_steps(tmp_path):
    inner = Mock()
    inner.infer.side_effect = [{"actions": np.array([i])} for i in range(3)]
    first = policy.PolicyRecorder(inner, tmp_path)
    first.infer({"state": np.array([0])})
    saved = (tmp_path / "step_0.npy").read_bytes()
    second = policy.PolicyRecorder(inner, tmp_path)
    second.infer({"state": np.array([1])})
    first.infer({"state": np.array([2])})

    assert (tmp_path / "step_0.npy").read_bytes() == saved
    for i in range(3):
        record = np.load(tmp_path / f"step_{i}.npy", allow_pickle=True).item()
        np.testing.assert_array_equal(record["inputs/state"], [i])
        np.testing.assert_array_equal(record["outputs/actions"], [i])
