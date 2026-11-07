import numpy as np
import pytest

from examples.aloha_sim import saver


def test_subsampled_video_preserves_frames_and_timing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(saver.imageio, "mimwrite", lambda path, images, fps: calls.append((path.name, images, fps)))
    (tmp_path / "out_1_preview.mp4").touch()
    (tmp_path / "out_1.mp4").touch()
    video = saver.VideoSaver(tmp_path, 3)
    observation = {"images": {"cam_high": np.zeros((3, 2, 2), dtype=np.uint8)}}
    for step in range(5):
        observation["images"]["cam_high"].fill(step)
        video.on_step(observation, {})
    video.on_episode_end()
    (name, frames, fps), = calls
    assert name == "out_2.mp4"
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 3]
    assert fps == 50 / 3

    video.on_episode_start()
    video.on_episode_end()
    assert len(calls) == 1
    video.on_step(observation, {})
    video.on_episode_end()
    assert len(calls[-1][1]) == 1


def test_rejects_invalid_subsample(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        saver.VideoSaver(tmp_path, 0)
