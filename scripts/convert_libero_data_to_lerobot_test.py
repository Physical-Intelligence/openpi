"""Regression tests for examples/libero/convert_libero_data_to_lerobot.py.

The pinned lerobot revision accumulates every written episode in the in-memory
`hf_dataset` table: `LeRobotDataset.save_episode` -> `_save_episode_table`
concatenates the new episode into `self.hf_dataset` on every call, so the RSS of
the conversion grows by the size of each episode and long conversions eventually
OOM (issue #829). The converter never reads `hf_dataset`, so it must reset the
table after every episode, while the parquet output on disk stays complete.
"""

import importlib.util
import json
import pathlib
import sys
import types

from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_module
import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CONVERTER_PATH = _REPO_ROOT / "examples" / "libero" / "convert_libero_data_to_lerobot.py"


def _load_converter_module():
    # `tensorflow_datasets` lives in the optional `rlds` dependency group that CI does
    # not install; the converter only calls `tfds.load`, which every test patches, so a
    # stub module is enough to import it.
    if "tensorflow_datasets" not in sys.modules:
        stub = types.ModuleType("tensorflow_datasets")
        stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tfds.load must be patched by the test before calling main()")
        )
        sys.modules["tensorflow_datasets"] = stub
    spec = importlib.util.spec_from_file_location("convert_libero_data_to_lerobot", _CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEpisode:
    def __init__(self, steps):
        self._steps = steps

    def __getitem__(self, key):
        assert key == "steps"
        return self

    def as_numpy_iterator(self):
        return iter(self._steps)


def _fake_rlds_dataset(num_episodes: int, frames_per_episode: int, seed: int):
    episodes = []
    states = []
    rng = np.random.default_rng(seed)
    for _ in range(num_episodes):
        steps = []
        for _ in range(frames_per_episode):
            state = rng.standard_normal(8).astype(np.float32)
            states.append(state)
            steps.append(
                {
                    "observation": {
                        # Libero camera frames are 256x256 HWC uint8; lerobot validates the
                        # declared feature shape on every add_frame.
                        "image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
                        "wrist_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
                        "state": state,
                    },
                    "action": rng.standard_normal(7).astype(np.float32),
                    "language_instruction": b"pick up the bowl",
                }
            )
        episodes.append(_FakeEpisode(steps))
    return episodes, states


@pytest.fixture
def converter(monkeypatch, tmp_path: pathlib.Path):
    module = _load_converter_module()
    # Sandbox the output dataset under tmp_path; both bindings are consulted at runtime
    # (the converter resolves HF_LEROBOT_HOME from its own globals, LeRobotDataset.create
    # from the lerobot module globals).
    monkeypatch.setattr(module, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(lerobot_dataset_module, "HF_LEROBOT_HOME", tmp_path)

    # The script requests 10 writer threads + 5 writer processes; force fully synchronous
    # image writes (identical data path, no threads or subprocesses in tests).
    real_create = lerobot_dataset_module.LeRobotDataset.create.__func__

    def create_with_sync_writes(*args, **kwargs):
        kwargs["image_writer_threads"] = 0
        kwargs["image_writer_processes"] = 0
        return real_create(*args, **kwargs)

    monkeypatch.setattr(lerobot_dataset_module.LeRobotDataset, "create", classmethod(create_with_sync_writes))
    return module


def _run_conversion(module, monkeypatch, num_episodes_per_raw: int):
    frames_per_episode = 4
    raw_names = module.RAW_DATASET_NAMES
    expected_states = []
    patches = []
    for seed, raw_name in enumerate(raw_names):
        episodes, states = _fake_rlds_dataset(num_episodes_per_raw, frames_per_episode, seed)
        expected_states.extend(states)
        patches.append((raw_name, episodes))

    def fake_load(name, data_dir=None, split=None):
        for patched_name, episodes in patches:
            if name == patched_name:
                return episodes
        raise AssertionError(f"unexpected raw dataset {name}")

    monkeypatch.setattr(module.tfds, "load", fake_load)
    module.main(data_dir="/unused")
    total_frames = num_episodes_per_raw * frames_per_episode * len(raw_names)
    return expected_states, total_frames


def test_hf_dataset_does_not_accumulate_across_episodes(converter, monkeypatch):
    """Each save_episode must not keep prior episodes in the in-memory hf_dataset table."""
    observed = []
    real_save_episode = lerobot_dataset_module.LeRobotDataset.save_episode

    def spy_save_episode(self, *args, **kwargs):
        observed.append(len(self.hf_dataset))
        return real_save_episode(self, *args, **kwargs)

    monkeypatch.setattr(lerobot_dataset_module.LeRobotDataset, "save_episode", spy_save_episode)

    _run_conversion(converter, monkeypatch, num_episodes_per_raw=1)

    # One save per raw dataset (4 total). On unfixed code this records [0, 4, 8, 12]:
    # the table still holds every previously written episode when the next one starts.
    assert observed == [0, 0, 0, 0]


def test_conversion_output_roundtrips(converter, monkeypatch, tmp_path: pathlib.Path):
    """Resetting the in-memory table must not corrupt the dataset written to disk."""
    import datasets as hf_datasets_module

    expected_states, total_frames = _run_conversion(converter, monkeypatch, num_episodes_per_raw=1)

    output_dir = tmp_path / converter.REPO_NAME
    parquet_files = sorted((output_dir / "data").rglob("*.parquet"))
    assert len(parquet_files) == len(converter.RAW_DATASET_NAMES)

    num_rows = sum(
        hf_datasets_module.load_dataset("parquet", data_files=str(path), split="train").num_rows
        for path in parquet_files
    )
    assert num_rows == total_frames

    with (output_dir / "meta/info.json").open() as f:
        info = json.load(f)
    assert info["total_episodes"] == len(converter.RAW_DATASET_NAMES)
    assert info["total_frames"] == total_frames

    dataset = lerobot_dataset_module.LeRobotDataset(repo_id=converter.REPO_NAME)
    assert dataset.num_frames == total_frames
    # The first written state frame must survive the conversion unchanged.
    assert np.allclose(np.asarray(dataset[0]["state"]), expected_states[0], atol=1e-6)
