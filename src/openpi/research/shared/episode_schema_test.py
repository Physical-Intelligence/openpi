"""Tests for episode_schema module."""

import dataclasses

from openpi.research.shared.episode_schema import EpisodeLabels, EpisodeMetadata


def test_episode_metadata_creation():
    meta = EpisodeMetadata(task_id="payload", env_id="nominal")
    assert meta.task_id == "payload"
    assert meta.env_id == "nominal"
    assert meta.operator_id == ""


def test_episode_metadata_frozen():
    meta = EpisodeMetadata(task_id="payload", env_id="nominal")
    try:
        meta.task_id = "other"  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except dataclasses.FrozenInstanceError:
        pass


def test_episode_labels_success():
    labels = EpisodeLabels(success=True)
    assert labels.success is True
    assert labels.fail_type is None


def test_episode_labels_failure():
    labels = EpisodeLabels(success=False, fail_type="timeout")
    assert labels.success is False
    assert labels.fail_type == "timeout"
