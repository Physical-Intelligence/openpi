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
