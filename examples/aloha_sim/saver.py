import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(self, out_dir: pathlib.Path, subsample: int = 1) -> None:
        if subsample < 1:
            raise ValueError("subsample must be positive")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample
        self._step = 0

    @override
    def on_episode_start(self) -> None:
        self._images = []
        self._step = 0

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        keep = self._step % self._subsample == 0
        self._step += 1
        if not keep:
            return
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im.copy())

    @override
    def on_episode_end(self) -> None:
        if not self._images:
            return
        indices = (p.stem.removeprefix("out_") for p in self._out_dir.glob("out_*.mp4"))
        next_idx = max((int(i) for i in indices if i.isascii() and i.isdigit()), default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"

        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(
            out_path,
            self._images,
            fps=50 / self._subsample,
        )
