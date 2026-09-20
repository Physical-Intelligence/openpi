from unittest import mock

import pytest

from openpi_client.runtime import agent
from openpi_client.runtime import environment
from openpi_client.runtime import runtime
from openpi_client.runtime import subscriber


@pytest.mark.parametrize("limit", [1, 2, 10])
@pytest.mark.parametrize("num_episodes", [1, 3])
def test_episode_step_limit(limit, num_episodes):
    env = mock.Mock(spec=environment.Environment)
    env.is_episode_complete.return_value = False
    env.get_observation.return_value = {}
    policy = mock.Mock(spec=agent.Agent)
    observer = mock.Mock(spec=subscriber.Subscriber)

    runtime.Runtime(env, policy, [observer], num_episodes=num_episodes, max_episode_steps=limit).run()

    assert env.apply_action.call_count == limit * num_episodes
    assert policy.get_action.call_count == limit * num_episodes
    assert observer.on_step.call_count == limit * num_episodes
    assert observer.on_episode_start.call_count == num_episodes
    assert observer.on_episode_end.call_count == num_episodes
    assert env.reset.call_count == num_episodes + 1


@pytest.mark.parametrize("limit", [0, 10])
def test_environment_can_end_episode_before_limit(limit):
    env = mock.Mock(spec=environment.Environment)
    env.is_episode_complete.side_effect = [False, True]
    policy = mock.Mock(spec=agent.Agent)
    observer = mock.Mock(spec=subscriber.Subscriber)

    runtime.Runtime(env, policy, [observer], max_episode_steps=limit).run()

    assert env.apply_action.call_count == 2
    assert observer.on_step.call_count == 2
    observer.on_episode_end.assert_called_once()
