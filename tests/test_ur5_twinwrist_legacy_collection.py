from pathlib import Path
import threading

import h5py
import numpy as np
import pytest

from examples.ur5_twinwrist.legacy_collection_adapter import AtomicControlledSpaceMouse
from examples.ur5_twinwrist.legacy_collection_adapter import HomingSafeSpaceMouse
from examples.ur5_twinwrist.legacy_collection_adapter import RawHDF5Dataset
from examples.ur5_twinwrist.record_hdf5 import EpisodeWriter


def _schema() -> dict:
    return {
        "capture": {
            "fps": 10,
            "cameras": [
                {"role": "primary", "dataset_key": "image.primary", "enabled": True},
                {"role": "wrist", "dataset_key": "image.wrist", "enabled": True},
                {"role": "secondary", "dataset_key": "image.secondary", "enabled": True},
            ],
        },
        "synchronization": {
            "state_channels": [{"name": name} for name in ("ur5", "wrist", "gripper")],
            "command_channel": {"name": "spacemouse"},
            "max_camera_skew_ms": 50.0,
            "max_camera_age_ms": 100.0,
            "max_state_age_ms": 100.0,
            "max_command_age_ms": 100.0,
        },
    }


def _hardware() -> dict:
    return {
        "cameras": {
            "devices": [
                {"role": "primary", "serial": "front-serial"},
                {"role": "wrist", "serial": "top-serial"},
                {"role": "secondary", "serial": "side-serial"},
            ]
        },
        "gripper": {"driver": "hiwonder"},
    }


def _frame(
    *,
    timestamp_s: float = 10.0,
    sequence: int = 1,
    fit: bool = False,
    target_qd: np.ndarray | None = None,
) -> dict:
    host_timestamps = timestamp_s + np.asarray([0.0, 0.010, 0.020, 0.0, 0.0, 0.0, 0.0])
    device_timestamps = timestamp_s + np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    buttons = np.zeros(12, dtype=np.int64)
    buttons[1] = int(fit)
    return {
        "image.primary": np.full((8, 12, 3), 10, dtype=np.uint8),
        "image.secondary": np.full((8, 12, 3), 20, dtype=np.uint8),
        "image.wrist": np.full((8, 12, 3), 30, dtype=np.uint8),
        "observation.state": np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.5], dtype=np.float32),
        "observation.tcp_pose": np.asarray([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0], dtype=np.float32),
        "action": np.asarray([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.8], dtype=np.float32),
        "telemetry.ur5_target_qd": (np.zeros(6, dtype=np.float32) if target_qd is None else target_qd),
        "telemetry.camera_skew_ms": np.asarray([10.0, 20.0], dtype=np.float32),
        "telemetry.source_age_ms": np.zeros(7, dtype=np.float32),
        "telemetry.host_receive_timestamps_s": host_timestamps,
        "telemetry.device_timestamps_s": device_timestamps,
        "telemetry.source_sequence_numbers": np.arange(sequence, sequence + 7, dtype=np.int64),
        "telemetry.validity_mask": np.ones(7, dtype=np.int64),
        "telemetry.spacemouse_buttons": buttons,
    }


def test_legacy_frames_stream_to_atomic_hdf5(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=10,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    assert (tmp_path / "episode_0000.tmp.hdf5").is_file()
    dataset.add_frame(_frame(timestamp_s=10.04, sequence=8, fit=True))
    path = dataset.save_episode()
    assert path == tmp_path / "episode_0000.hdf5"
    with h5py.File(path, "r") as episode:
        assert episode["observations/images/front"][0, 0, 0, 0] == 10
        assert episode["observations/images/side"][0, 0, 0, 0] == 20
        assert episode["observations/images/top"][0, 0, 0, 0] == 30
        assert episode["action"].shape == (1, 9)
        assert episode["timestamps/front"][0] == 11_000_000_000
        assert episode["timestamps_host/top"][0] == 10_010_000_000
        assert episode.attrs["fps"] == 10.0
        assert episode.attrs["capture_fps"] == 10
    dataset.finalize()


def test_unsealed_emergency_save_is_rejected(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    with pytest.raises(RuntimeError, match="not sealed"):
        dataset.save_episode()
    rejected = tmp_path / "rejected/episode_0000.hdf5"
    assert rejected.is_file()
    with h5py.File(rejected, "r") as episode:
        assert not bool(episode.attrs["success"])
    dataset.finalize()


def test_speedj_during_episode_is_rejected(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    with pytest.raises(ValueError, match="speedJ"):
        dataset.add_frame(
            _frame(
                timestamp_s=10.2,
                sequence=8,
                target_qd=np.asarray([0, 0, 0, 0, 0, 0.2], dtype=np.float32),
            )
        )
    with pytest.raises(RuntimeError, match="speedJ"):
        dataset.save_episode()
    assert (tmp_path / "rejected/episode_0000.hdf5").is_file()
    dataset.finalize()


def test_sealed_episode_is_still_rejected_on_legacy_emergency_path(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    dataset.add_frame(_frame(timestamp_s=10.2, sequence=8, fit=True))

    def emergency_save_active_episode():
        dataset.save_episode()

    with pytest.raises(RuntimeError, match="emergency-save"):
        emergency_save_active_episode()
    with h5py.File(tmp_path / "rejected/episode_0000.hdf5", "r") as episode:
        assert "emergency-save" in episode.attrs["rejection_reason"]
    dataset.finalize()


def test_stale_source_rejects_an_active_episode(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    stale = _frame(timestamp_s=10.2, sequence=8)
    stale["telemetry.source_age_ms"][4] = 101.0
    with pytest.raises(TimeoutError, match="robot state is stale"):
        dataset.add_frame(stale)
    with pytest.raises(RuntimeError, match="robot state is stale"):
        dataset.save_episode()
    assert (tmp_path / "rejected/episode_0000.hdf5").is_file()
    dataset.finalize()


def test_one_repeated_camera_frame_rejects_episode(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    repeated = _frame(timestamp_s=10.2, sequence=8)
    repeated["telemetry.source_sequence_numbers"][1] = 2
    with pytest.raises(ValueError, match="camera frames were repeated"):
        dataset.add_frame(repeated)
    dataset.clear_episode_buffer()
    dataset.finalize()


def test_thirty_hz_input_is_sampled_at_ten_hz(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        record_hz=10,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame(timestamp_s=10.00, sequence=1))
    dataset.add_frame(_frame(timestamp_s=10.04, sequence=8))
    dataset.add_frame(_frame(timestamp_s=10.11, sequence=15))
    dataset.add_frame(_frame(timestamp_s=10.15, sequence=22, fit=True))
    path = dataset.save_episode()
    with h5py.File(path, "r") as episode:
        assert episode["action"].shape == (2, 9)
        np.testing.assert_array_equal(
            episode["timestamps/control"][:],
            np.asarray([10_000_000_000, 10_110_000_000], dtype=np.int64),
        )
    dataset.finalize()


def test_crash_left_temp_episode_is_recovered_to_rejected(tmp_path: Path):
    writer = EpisodeWriter(tmp_path, 0, (8, 12, 3), {})
    writer.append(
        {
            "qpos": np.zeros(9, dtype=np.float32),
            "images": {name: np.full((8, 12, 3), 20, dtype=np.uint8) for name in ("front", "side", "top")},
            "timestamp_ns": 1,
            "camera_timestamps_ns": dict.fromkeys(("front", "side", "top"), 1),
        },
        np.zeros(9, dtype=np.float32),
    )
    writer.file.flush()
    writer.file.close()
    writer.closed = True

    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    recovered = tmp_path / "rejected/episode_0000.hdf5"
    assert recovered.is_file()
    assert not (tmp_path / "episode_0000.tmp.hdf5").exists()
    with h5py.File(recovered, "r") as episode:
        assert not bool(episode.attrs["success"])
        assert "stale temp" in episode.attrs["rejection_reason"]
    dataset.finalize()


def test_only_one_raw_writer_can_own_an_output_root(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=30,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    with pytest.raises(RuntimeError, match="another raw HDF5 writer"):
        RawHDF5Dataset(
            tmp_path,
            task="test",
            fps=30,
            hardware=_hardware(),
            input_schema=_schema(),
            openpi_root=Path(__file__).parents[1],
        )
    dataset.finalize()


def test_discard_moves_partial_episode_to_rejected(tmp_path: Path):
    dataset = RawHDF5Dataset(
        tmp_path,
        task="test",
        fps=10,
        hardware=_hardware(),
        input_schema=_schema(),
        openpi_root=Path(__file__).parents[1],
    )
    dataset.add_frame(_frame())
    rejected = dataset.clear_episode_buffer()
    assert rejected == tmp_path / "rejected/episode_0000.hdf5"
    assert rejected.is_file()
    dataset.finalize()


def test_homing_gate_zeros_cap_for_coordinated_and_t_home():
    class Mouse:
        def __init__(self):
            self.buttons = {}

        def state(self):
            return np.ones(6, dtype=np.float32), self.buttons

    delegate = Mouse()
    coordinated = False
    mouse = HomingSafeSpaceMouse(delegate, lambda: coordinated)
    assert np.all(mouse.state()[0] == 1)
    coordinated = True
    assert np.all(mouse.state()[0] == 0)
    coordinated = False
    delegate.buttons = {2: True}
    assert np.all(mouse.state()[0] == 0)


def test_station_snapshot_cannot_mix_control_cycles():
    class Controlled:
        def __init__(self):
            self._latest_lock = threading.Lock()
            self.latest = (np.ones(6), {1: False})
            self.latest_twist = np.full(6, 2.0)
            self.latest_target_qd = np.full(6, 3.0)
            self.latest_ur5_joints = np.full(6, 4.0)

    controlled = Controlled()
    proxy = AtomicControlledSpaceMouse(controlled)
    motion, _buttons = proxy.latest
    with controlled._latest_lock:  # noqa: SLF001 - exercise the legacy lock boundary
        controlled.latest = (np.full(6, 10.0), {1: True})
        controlled.latest_twist[:] = 20.0
        controlled.latest_target_qd[:] = 30.0
    assert np.all(motion == 1.0)
    assert np.all(proxy.latest_twist == 2.0)
    assert np.all(proxy.latest_target_qd == 3.0)


def test_completed_cycle_hook_publishes_one_command_receipt():
    class Controlled:
        def __init__(self):
            self._latest_lock = threading.Lock()
            self.latest = (np.zeros(6), {})
            self.latest_twist = np.zeros(6)
            self.latest_target_qd = np.zeros(6)
            self.latest_ur5_joints = np.zeros(6)
            self.recorded = []

        def _record_cycle(self, motion, buttons, twist):
            self.recorded.append((motion.copy(), buttons.copy(), twist.copy()))

    controlled = Controlled()
    proxy = AtomicControlledSpaceMouse(controlled)
    controlled.latest = (np.full(6, 1.0), {1: True})
    controlled.latest_twist = np.full(6, 2.0)
    controlled.latest_target_qd = np.full(6, 3.0)
    controlled.latest_ur5_joints = np.full(6, 4.0)
    controlled._record_cycle(  # noqa: SLF001 - emulate the audited producer callback
        controlled.latest[0], controlled.latest[1], controlled.latest_twist
    )

    # Simulate the producer beginning the next cycle before the synchronizer
    # reads.  The proxy must continue to expose the last completed receipt.
    with controlled._latest_lock:  # noqa: SLF001 - emulate the audited publication gap
        controlled.latest = (np.full(6, 10.0), {1: False})
        controlled.latest_twist = np.full(6, 20.0)
        controlled.latest_target_qd = np.full(6, 30.0)
    motion, buttons = proxy.latest
    assert np.all(motion == 1.0)
    assert buttons == {1: True}
    assert np.all(proxy.latest_twist == 2.0)
    assert np.all(proxy.latest_target_qd == 3.0)
    assert np.all(proxy.latest_ur5_joints == 4.0)
