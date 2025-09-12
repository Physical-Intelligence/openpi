from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.runtime import agent as _agent


class PolicyAgent(_agent.Agent):
    """An agent that uses a policy to determine actions."""

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        """Initialize the policy agent with a given policy.
        
        Args:
            policy: The policy instance used to infer actions from observations.
        """
        self._policy = policy

    @override
    def get_action(self, observation: dict) -> dict:
        """Get an action by inferring from the observation using the policy.
        
        Args:
            observation: The current observation state as a dictionary.
            
        Returns:
            The action determined by the policy as a dictionary.
        """
        return self._policy.infer(observation)

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        self._policy.reset()
