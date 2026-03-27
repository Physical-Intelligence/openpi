import logging
import pathlib
import time

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

VIDEO_FPS = 50


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data with real-time latency reflected in video duration."""

    def __init__(self, out_dir: pathlib.Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._timestamps: list[float] = []

    @override
    def on_episode_start(self) -> None:
        self._images = []
        self._timestamps = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)
        self._timestamps.append(time.monotonic())

    @override
    def on_episode_end(self) -> None:
        if not self._images:
            return

        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"

        frames = []
        for i, (im, ts) in enumerate(zip(self._images, self._timestamps)):
            # compute step latency
            if i + 1 < len(self._timestamps):
                step_duration = self._timestamps[i + 1] - ts
            else:
                step_duration = self._timestamps[-1] - self._timestamps[-2] if len(self._timestamps) > 1 else 1.0 / VIDEO_FPS

            latency_ms = step_duration * 1000

            # overlay latency text on frame
            annotated = _draw_latency(im, latency_ms)

            # repeat frame to match real wall-clock time
            repeat = max(1, round(step_duration * VIDEO_FPS))
            frames.extend([annotated] * repeat)

        logging.info(f"Saving video to {out_path} ({len(frames)} frames @ {VIDEO_FPS}fps = {len(frames)/VIDEO_FPS:.1f}s real time)")
        imageio.mimwrite(out_path, frames, fps=VIDEO_FPS)


def _draw_latency(im: np.ndarray, latency_ms: float) -> np.ndarray:
    """Draw latency overlay on frame using PIL."""
    img = Image.fromarray(im.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    text = f"{latency_ms:.0f} ms"
    x, y = 8, 8

    # shadow for readability
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 80, 80))

    return np.array(img)
