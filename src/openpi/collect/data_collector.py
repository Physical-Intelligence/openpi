from __future__ import annotations

import datetime
import logging
import pathlib
import threading
from dataclasses import dataclass

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceEmbeddings:
    """Embeddings captured for a single infer() call."""

    vision_embs: list[np.ndarray]
    prompt_emb: np.ndarray
    robot_state: np.ndarray
    noise_action_steps: list[np.ndarray]
    clean_action: np.ndarray


class EpisodeDataCollector:
    """Buffers per-step embeddings for one episode and flushes them to HDF5."""

    def __init__(self, base_dir: str) -> None:
        self._base_dir = pathlib.Path(base_dir)
        self._buffer: list[InferenceEmbeddings] = []
        self._experiment = "unknown"
        self._task = ""
        self._episode_id = -1
        self._lock = threading.Lock()

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        with self._lock:
            self._buffer = []
            self._experiment = experiment
            self._task = task
            self._episode_id = episode_id
        logger.info("EpisodeDataCollector: episode %d started (%s / %s)", episode_id, experiment, task)

    def record_inference(self, embs: InferenceEmbeddings) -> None:
        with self._lock:
            self._buffer.append(embs)

    def has_pending_data(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    def on_episode_end(self, success: bool) -> None:
        with self._lock:
            if not self._buffer:
                logger.warning("EpisodeDataCollector: episode %d has no data, skipping write.", self._episode_id)
                return

            buffer = self._buffer
            experiment = self._experiment
            task = self._task
            episode_id = self._episode_id
            self._buffer = []

        out_dir = self._base_dir / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = out_dir / f"episode_{episode_id:04d}_{ts}.h5"
        tmp_path = path.with_suffix(".h5.tmp")

        try:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["experiment_name"] = experiment
                f.attrs["task"] = task
                f.attrs["episode_id"] = episode_id
                f.attrs["num_steps"] = len(buffer)
                f.attrs["timestamp"] = datetime.datetime.now().isoformat()
                f.attrs["success"] = success

                for step_idx, embs in enumerate(buffer):
                    grp = f.create_group(f"step_{step_idx:04d}")
                    for i, vision_emb in enumerate(embs.vision_embs):
                        grp.create_dataset(f"vision_{i}", data=vision_emb, compression="lzf")
                    grp.create_dataset("prompt_emb", data=embs.prompt_emb, compression="lzf")
                    grp.create_dataset("robot_state", data=embs.robot_state)
                    for i, noise_action in enumerate(embs.noise_action_steps, start=1):
                        grp.create_dataset(f"noise_action_{i}", data=noise_action)
                    grp.create_dataset("clean_action", data=embs.clean_action)

            tmp_path.rename(path)
            logger.info(
                "EpisodeDataCollector: episode %d written -> %s (%d steps, success=%s)",
                episode_id,
                path,
                len(buffer),
                success,
            )
        except Exception:
            logger.exception("EpisodeDataCollector: failed to write episode %d", episode_id)
            tmp_path.unlink(missing_ok=True)
