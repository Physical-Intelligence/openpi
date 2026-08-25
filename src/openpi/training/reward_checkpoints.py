"""Atomic, reward-gated checkpoints with disk-pressure eviction."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
import logging
from pathlib import Path
import shutil
import time


@dataclasses.dataclass(frozen=True)
class SaveResult:
    saved: bool
    checkpoint_dir: Path | None
    reward: float
    best_reward: float | None
    reason: str
    evicted_steps: tuple[int, ...] = ()


class RewardCheckpointStore:
    """Keep only strict reward improvements and evict oldest under pressure."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        root: Path,
        *,
        metric_name: str,
        minimum_reward: float,
        minimum_delta: float,
        minimum_free_bytes: int,
        minimum_free_fraction: float,
        disk_usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage,
    ) -> None:
        self.root = root
        self.metric_name = metric_name
        self.minimum_reward = minimum_reward
        self.minimum_delta = minimum_delta
        self.minimum_free_bytes = minimum_free_bytes
        self.minimum_free_fraction = minimum_free_fraction
        self._disk_usage = disk_usage
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "reward_checkpoints.json"

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {
                "schema_version": self._SCHEMA_VERSION,
                "metric_name": self.metric_name,
                "records": [],
            }
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError(f"Unsupported reward checkpoint manifest: {self.manifest_path}")
        if manifest.get("metric_name") != self.metric_name:
            raise ValueError(
                f"Checkpoint metric changed from {manifest.get('metric_name')!r} to {self.metric_name!r}"
            )
        manifest["records"] = [
            record for record in manifest.get("records", []) if (self.root / str(record["step"])).is_dir()
        ]
        return manifest

    def _write_manifest(self, manifest: dict) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def best_record(self) -> dict | None:
        records = self._load_manifest()["records"]
        return max(records, key=lambda record: float(record["reward"])) if records else None

    def best_checkpoint_dir(self) -> Path | None:
        best = self.best_record()
        return self.root / str(best["step"]) if best is not None else None

    def _required_free_bytes(self) -> int:
        usage = self._disk_usage(self.root)
        return max(self.minimum_free_bytes, int(usage.total * self.minimum_free_fraction))

    def _evict_under_pressure(self, manifest: dict, *, protected_step: int | None) -> tuple[int, ...]:
        evicted: list[int] = []
        required = self._required_free_bytes()
        while self._disk_usage(self.root).free < required:
            candidates = sorted(
                (record for record in manifest["records"] if int(record["step"]) != protected_step),
                key=lambda record: (float(record["created_at"]), int(record["step"])),
            )
            if not candidates:
                raise RuntimeError(
                    f"Checkpoint filesystem has less than {required / 1024**3:.1f} GiB free and no older "
                    "reward checkpoint can be evicted safely"
                )
            oldest = candidates[0]
            step = int(oldest["step"])
            target = self.root / str(step)
            if target.is_dir():
                shutil.rmtree(target)
            manifest["records"].remove(oldest)
            evicted.append(step)
            logging.warning("Evicted oldest reward checkpoint at step %d because disk headroom is low", step)
        return tuple(evicted)

    def maybe_save(
        self,
        *,
        step: int,
        reward: float,
        write_checkpoint: Callable[[Path], None],
    ) -> SaveResult:
        manifest = self._load_manifest()
        best = max((float(record["reward"]) for record in manifest["records"]), default=None)
        if reward < self.minimum_reward:
            return SaveResult(
                saved=False,
                checkpoint_dir=None,
                reward=reward,
                best_reward=best,
                reason="below_minimum_reward",
            )
        if best is not None and reward <= best + self.minimum_delta:
            return SaveResult(
                saved=False,
                checkpoint_dir=None,
                reward=reward,
                best_reward=best,
                reason="not_a_strict_improvement",
            )

        evicted = list(self._evict_under_pressure(manifest, protected_step=None))
        self._write_manifest(manifest)

        final_dir = self.root / str(step)
        temporary_dir = self.root / f"tmp_{step}"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)
        try:
            write_checkpoint(temporary_dir)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temporary_dir.replace(final_dir)
        except BaseException:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

        record = {
            "step": step,
            "reward": reward,
            "created_at": time.time(),
            "metric_name": self.metric_name,
        }
        manifest["records"].append(record)
        evicted.extend(self._evict_under_pressure(manifest, protected_step=step))
        self._write_manifest(manifest)
        logging.info("Saved new best reward checkpoint: step=%d reward=%.8f", step, reward)
        return SaveResult(
            saved=True,
            checkpoint_dir=final_dir,
            reward=reward,
            best_reward=reward,
            reason="strict_improvement",
            evicted_steps=tuple(evicted),
        )
