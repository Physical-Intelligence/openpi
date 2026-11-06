from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpi_client.runtime import runtime


@pytest.mark.parametrize("wall_jump", [-3600, 3600])
def test_step_period_ignores_wall_clock_adjustments(monkeypatch, wall_jump):
    sleeps = []
    # A step took 20 ms. The system clock was also adjusted during that step.
    clock = SimpleNamespace(
        time=iter([1000, 1000.02 + wall_jump, 1000.1 + wall_jump]).__next__,
        monotonic=iter([10, 10.02, 10.1]).__next__,
        sleep=sleeps.append,
    )
    monkeypatch.setattr(runtime, "time", clock)
    env = Mock()
    env.is_episode_complete.return_value = True

    runtime.Runtime(env, Mock(), [], max_hz=10).run()

    assert sleeps == pytest.approx([0.08])
