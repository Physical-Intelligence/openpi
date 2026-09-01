from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime.runtime import Runtime
import pytest


class _FakeEnvironment(_environment.Environment):
    def __init__(self, complete_after: int | None = None) -> None:
        self.applied_actions = 0
        self._complete_after = complete_after

    def reset(self) -> None:
        pass

    def is_episode_complete(self) -> bool:
        return self._complete_after is not None and self.applied_actions >= self._complete_after

    def get_observation(self) -> dict:
        return {"step": self.applied_actions}

    def apply_action(self, action: dict) -> None:
        self.applied_actions += 1


class _FakeAgent(_agent.Agent):
    def get_action(self, observation: dict) -> dict:
        return {"step": observation["step"]}

    def reset(self) -> None:
        pass


@pytest.mark.parametrize("max_episode_steps", [1, 2, 3])
def test_max_episode_steps_counts_completed_actions(max_episode_steps: int) -> None:
    environment = _FakeEnvironment()
    runtime = Runtime(environment, _FakeAgent(), [], max_episode_steps=max_episode_steps)

    runtime.run()

    assert environment.applied_actions == max_episode_steps
    assert runtime._episode_steps == max_episode_steps  # noqa: SLF001


def test_environment_can_complete_episode_before_step_limit() -> None:
    environment = _FakeEnvironment(complete_after=2)
    runtime = Runtime(environment, _FakeAgent(), [], max_episode_steps=10)

    runtime.run()

    assert environment.applied_actions == 2
    assert runtime._episode_steps == 2  # noqa: SLF001
