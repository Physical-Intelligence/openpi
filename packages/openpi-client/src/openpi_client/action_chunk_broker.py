from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        if isinstance(action_horizon, bool) or not isinstance(action_horizon, (int, np.integer)) or action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer.")
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            results = self._policy.infer(obs)
            for path, value in tree.flatten_with_path(results):
                if isinstance(value, np.ndarray) and (value.ndim == 0 or value.shape[0] < self._action_horizon):
                    raise ValueError(
                        f"Array at {path} must contain at least action_horizon={self._action_horizon} steps; "
                        f"got shape {value.shape}."
                    )
            self._last_results = results
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
