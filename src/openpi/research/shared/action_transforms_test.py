"""Tests for action_transforms module."""

import pytest

from openpi.research.shared import action_transforms


def test_teleop_to_canonical_not_implemented():
    with pytest.raises(NotImplementedError):
        action_transforms.teleop_to_canonical(None)


def test_canonical_to_training_not_implemented():
    with pytest.raises(NotImplementedError):
        action_transforms.canonical_to_training(None)


def test_training_to_canonical_not_implemented():
    with pytest.raises(NotImplementedError):
        action_transforms.training_to_canonical(None)
