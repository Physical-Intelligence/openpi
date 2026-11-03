from unittest import mock

import numpy as np
import pytest

from openpi_client import action_chunk_broker
from openpi_client import base_policy


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_horizon_must_be_a_positive_integer(horizon):
    policy = mock.Mock(spec=base_policy.BasePolicy)
    with pytest.raises(ValueError, match="positive integer"):
        action_chunk_broker.ActionChunkBroker(policy, horizon)
    policy.infer.assert_not_called()


@pytest.mark.parametrize("actions", [np.zeros((2, 7)), np.array(1.0), np.empty((0, 7))])
def test_invalid_chunk_is_rejected_before_any_action_is_returned(actions):
    policy = mock.Mock(spec=base_policy.BasePolicy)
    policy.infer.return_value = {"actions": actions}
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=3)
    with pytest.raises(ValueError, match="action_horizon"):
        broker.infer({})


def test_invalid_chunk_does_not_poison_the_cache():
    policy = mock.Mock(spec=base_policy.BasePolicy)
    actions = np.arange(12).reshape(3, 4)
    policy.infer.side_effect = [{"actions": actions[:1]}, {"actions": actions}]
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=3)
    with pytest.raises(ValueError, match="action_horizon"):
        broker.infer({})
    for i in range(3):
        np.testing.assert_array_equal(broker.infer({})["actions"], actions[i])
    assert policy.infer.call_count == 2


def test_valid_nested_chunks_replan_at_horizon_and_preserve_metadata():
    policy = mock.Mock(spec=base_policy.BasePolicy)
    actions = np.arange(12).reshape(3, 4)
    policy.infer.return_value = {"arm": {"actions": actions}, "server_timing": {"infer_ms": 10.0}}
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)
    for i in (0, 1, 0):
        result = broker.infer({})
        np.testing.assert_array_equal(result["arm"]["actions"], actions[i])
        assert result["server_timing"] == {"infer_ms": 10.0}
    assert policy.infer.call_count == 2
    broker.reset()
    policy.reset.assert_called_once_with()
    np.testing.assert_array_equal(broker.infer({})["arm"]["actions"], actions[0])
    assert policy.infer.call_count == 3


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("step", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonfinite_nested_action_is_rejected_before_returning_any_step(value, step, dtype):
    policy = mock.Mock(spec=base_policy.BasePolicy)
    actions = np.zeros((3, 4), dtype=dtype)
    actions[step, 2] = value
    policy.infer.return_value = {"arm": {"actions": actions}}
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)

    with pytest.raises(ValueError, match="arm.*actions.*non-finite"):
        broker.infer({})


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_in_unused_tail_do_not_reject_valid_actions(value):
    policy = mock.Mock(spec=base_policy.BasePolicy)
    actions = np.arange(12, dtype=np.float32).reshape(3, 4)
    actions[2] = value
    policy.infer.return_value = {"actions": actions, "server_timing": {"infer_ms": 10.0}}
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)

    for step in (0, 1, 0):
        result = broker.infer({})
        np.testing.assert_array_equal(result["actions"], actions[step])
        assert result["server_timing"] == {"infer_ms": 10.0}
    assert policy.infer.call_count == 2


def test_nonfinite_chunk_does_not_poison_the_cache():
    policy = mock.Mock(spec=base_policy.BasePolicy)
    actions = np.arange(12, dtype=np.float64).reshape(3, 4)
    invalid = actions.copy()
    invalid[1, 0] = np.nan
    policy.infer.side_effect = [{"actions": invalid}, {"actions": actions}]
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)

    with pytest.raises(ValueError, match="non-finite"):
        broker.infer({})
    for step in range(2):
        np.testing.assert_array_equal(broker.infer({})["actions"], actions[step])
    assert policy.infer.call_count == 2
