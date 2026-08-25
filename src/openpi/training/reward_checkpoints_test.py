from __future__ import annotations

from collections import namedtuple
import json
from pathlib import Path

from openpi.training.reward_checkpoints import RewardCheckpointStore


def _writer(path: Path) -> None:
    (path / "payload.bin").write_bytes(b"checkpoint")


def test_only_strict_reward_improvements_are_saved(tmp_path: Path) -> None:
    store = RewardCheckpointStore(
        tmp_path,
        metric_name="reward",
        minimum_reward=0.0,
        minimum_delta=1.0e-6,
        minimum_free_bytes=0,
        minimum_free_fraction=0.0,
    )

    assert store.maybe_save(step=10, reward=0.2, write_checkpoint=_writer).saved
    assert not store.maybe_save(step=20, reward=0.1, write_checkpoint=_writer).saved
    assert not store.maybe_save(step=30, reward=0.2000005, write_checkpoint=_writer).saved
    assert store.maybe_save(step=40, reward=0.3, write_checkpoint=_writer).saved

    assert (tmp_path / "10").is_dir()
    assert not (tmp_path / "20").exists()
    assert not (tmp_path / "30").exists()
    assert (tmp_path / "40").is_dir()
    assert store.best_checkpoint_dir() == tmp_path / "40"


def test_disk_pressure_evicts_oldest_reward_checkpoint(tmp_path: Path) -> None:
    DiskUsage = namedtuple("DiskUsage", "total used free")

    def disk_usage(path: Path):
        checkpoint_count = sum(child.is_dir() and child.name.isdigit() for child in path.iterdir())
        free = 0 if checkpoint_count > 1 else 100
        return DiskUsage(total=1000, used=1000 - free, free=free)

    store = RewardCheckpointStore(
        tmp_path,
        metric_name="reward",
        minimum_reward=0.0,
        minimum_delta=0.0,
        minimum_free_bytes=50,
        minimum_free_fraction=0.0,
        disk_usage=disk_usage,
    )

    assert store.maybe_save(step=10, reward=0.2, write_checkpoint=_writer).saved
    second = store.maybe_save(step=20, reward=0.3, write_checkpoint=_writer)

    assert second.saved
    assert second.evicted_steps == (10,)
    assert not (tmp_path / "10").exists()
    assert (tmp_path / "20").is_dir()
    manifest = json.loads((tmp_path / "reward_checkpoints.json").read_text())
    assert [record["step"] for record in manifest["records"]] == [20]
