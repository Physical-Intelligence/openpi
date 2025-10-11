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
        """Initialize the ActionChunkBroker with a policy and action horizon.

        Args:
            policy: The underlying policy to wrap for chunked action delivery.
            action_horizon: The number of action steps in each chunk from the policy.
        """
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Return the next action from the current chunk or fetch a new chunk if needed.

        Args:
            obs: Observation dictionary to pass to the underlying policy when fetching new chunks.

        Returns:
            Dictionary containing the action for the current step, extracted from the chunk.
        """
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            """Extract the current step from array data or return non-array data unchanged."""
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
        """Reset the broker state and the underlying policy."""
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
