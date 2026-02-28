"""Action transform layer for RM75 Delta EE + gripper action space.

Provides the authoritative transform path:
  teleop output → canonical Delta EE + gripper → training format

Follows openpi's DataTransformFn pattern (like DeltaActions).
"""


def teleop_to_canonical(teleop_action):
    """Convert raw teleop controller output to canonical Delta EE + gripper."""
    raise NotImplementedError("TODO: implement teleop → canonical transform")


def canonical_to_training(canonical_action):
    """Convert canonical Delta EE + gripper to openpi training format."""
    raise NotImplementedError("TODO: implement canonical → training transform")


def training_to_canonical(training_action):
    """Convert openpi training format back to canonical Delta EE + gripper."""
    raise NotImplementedError("TODO: implement training → canonical transform")
