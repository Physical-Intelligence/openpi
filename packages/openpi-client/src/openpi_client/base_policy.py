import abc
from typing import Dict
from typing import Sequence


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_batch(self, obs_batch: Sequence[Dict]) -> Dict:
        """Infer actions from a batch of observations.

        Policies with correct batched semantics should override this method. The base implementation deliberately does
        not loop over `infer`, since stateful wrappers may need per-environment state to batch safely.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support batched inference")

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
