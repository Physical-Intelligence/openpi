"""Adapt the validated legacy control/synchronization stack to raw HDF5.

This module is safe to import: legacy hardware modules are imported only from
``make_dependencies``, which the legacy CLI calls only after its real-motion
confirmation gate has passed.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import threading
from typing import Any

import numpy as np

from examples.ur5_twinwrist.gripper_worker import SingleOwnerGripper

DEFAULT_ROLE_MAP = {"front": "primary", "side": "secondary", "top": "wrist"}
LEGACY_T_BUTTON = 2
RECORDED_FIT_INDEX = 1


class HomingSafeSpaceMouse:
    """Force a zero cap while T or coordinated HOME owns the robot."""

    def __init__(self, delegate: Any, coordinated_home) -> None:
        self._delegate = delegate
        self._coordinated_home = coordinated_home

    def state(self):
        motion, buttons = self._delegate.state()
        if self._coordinated_home() or bool(buttons.get(LEGACY_T_BUTTON, False)):
            motion = np.zeros(6, dtype=np.float32)
        return motion, buttons

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


class AtomicControlledSpaceMouse:
    """Expose one completed control-cycle receipt to ``StationSynchronizer``.

    The legacy synchronizer reads ``latest``, ``latest_target_qd`` and
    ``latest_twist`` as separate attributes.  The legacy producer also updates
    ``latest`` before sending the command and updates the command fields after
    the send.  A read-side lock alone can therefore still observe that narrow
    publication gap.  This proxy hooks the existing flight-recorder callback,
    which runs only after the command fields have been updated, and publishes
    one immutable receipt for the entire completed 125 Hz cycle.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._thread_local = threading.local()
        self._receipt_lock = threading.Lock()
        self._receipt: dict[str, Any] | None = None
        self._original_record_cycle = None
        self._install_receipt_hook()

    def _install_receipt_hook(self) -> None:
        original = getattr(self._delegate, "_record_cycle", None)
        if not callable(original):
            return
        self._original_record_cycle = original

        def record_completed_cycle(motion: np.ndarray, buttons: dict[int, bool], twist: np.ndarray) -> None:
            original(motion, buttons, twist)
            # The producer cannot start its next cycle until this callback
            # returns, so target_qd/joints and the callback arguments belong to
            # exactly the same successfully issued command.
            with self._delegate._latest_lock:  # noqa: SLF001
                snapshot = {
                    "latest": (np.asarray(motion).copy(), buttons.copy()),
                    "latest_twist": np.asarray(twist).copy(),
                    "latest_target_qd": np.asarray(self._delegate.latest_target_qd).copy(),
                    "latest_ur5_joints": np.asarray(self._delegate.latest_ur5_joints).copy(),
                }
                if hasattr(self._delegate, "latest_hand_command"):
                    snapshot["latest_hand_command"] = np.asarray(self._delegate.latest_hand_command).copy()
            with self._receipt_lock:
                self._receipt = snapshot

        self._delegate._record_cycle = record_completed_cycle  # noqa: SLF001

    def _restore_receipt_hook(self) -> None:
        if self._original_record_cycle is not None:
            self._delegate._record_cycle = self._original_record_cycle  # noqa: SLF001
            self._original_record_cycle = None

    def __enter__(self):
        entered = self._delegate.__enter__()
        if entered is not self._delegate:
            self._restore_receipt_hook()
            self._delegate = entered
            self._install_receipt_hook()
        return self

    def __exit__(self, *args: Any):
        try:
            return self._delegate.__exit__(*args)
        finally:
            self._restore_receipt_hook()

    def _take_snapshot(self) -> dict[str, Any]:
        with self._receipt_lock:
            receipt = self._receipt
            snapshot = None if receipt is None else self._copy_snapshot(receipt)
        if snapshot is None:
            # Lightweight fakes and older legacy revisions may not expose the
            # callback.  Keep a lock-bounded compatibility path for them; the
            # audited production controller always uses the receipt hook.
            with self._delegate._latest_lock:  # noqa: SLF001
                motion, buttons = self._delegate.latest
                snapshot = {
                    "latest": (motion.copy(), buttons.copy()),
                    "latest_twist": np.asarray(self._delegate.latest_twist).copy(),
                    "latest_target_qd": np.asarray(self._delegate.latest_target_qd).copy(),
                    "latest_ur5_joints": np.asarray(self._delegate.latest_ur5_joints).copy(),
                }
                if hasattr(self._delegate, "latest_hand_command"):
                    snapshot["latest_hand_command"] = np.asarray(self._delegate.latest_hand_command).copy()
        self._thread_local.snapshot = snapshot
        return snapshot

    @staticmethod
    def _copy_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        copied: dict[str, Any] = {}
        for name, value in snapshot.items():
            if name == "latest":
                copied[name] = (value[0].copy(), value[1].copy())
            elif isinstance(value, np.ndarray):
                copied[name] = value.copy()
            else:
                copied[name] = copy.deepcopy(value)
        return copied

    def _snapshot_value(self, name: str) -> Any:
        snapshot = getattr(self._thread_local, "snapshot", None)
        if snapshot is None or name not in snapshot:
            snapshot = self._take_snapshot()
        value = snapshot[name]
        if isinstance(value, np.ndarray):
            return value.copy()
        if name == "latest":
            return value[0].copy(), value[1].copy()
        return value

    @property
    def latest(self):
        motion, buttons = self._take_snapshot()["latest"]
        return motion.copy(), buttons.copy()

    @property
    def latest_twist(self):
        return self._snapshot_value("latest_twist")

    @property
    def latest_target_qd(self):
        return self._snapshot_value("latest_target_qd")

    @property
    def latest_ur5_joints(self):
        return self._snapshot_value("latest_ur5_joints")

    @property
    def latest_hand_command(self):
        return self._snapshot_value("latest_hand_command")

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def _stable_hash(*values: Any) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_identity(root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _called_from(function_name: str) -> bool:
    """Identify the legacy emergency-save call without modifying legacy core."""
    frame = inspect.currentframe()
    try:
        while frame is not None:
            if frame.f_code.co_name == function_name:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


class RawHDF5Dataset:
    """Dataset-like sink expected by ``RealCollectionWorkflow``.

    Frames are appended directly to ``episode_XXXX.tmp.hdf5``. ``save_episode``
    atomically commits it; ``clear_episode_buffer`` moves it to ``rejected``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        task: str,
        fps: int,
        record_hz: float = 10.0,
        hardware: dict[str, Any],
        input_schema: dict[str, Any],
        openpi_root: str | Path,
        role_map: dict[str, str] | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._role_map = dict(role_map or DEFAULT_ROLE_MAP)
        if set(self._role_map) != {"front", "side", "top"}:
            raise ValueError("role_map must define front, side and top")
        cameras = {str(item["role"]): str(item["serial"]) for item in hardware.get("cameras", {}).get("devices", [])}
        missing = set(self._role_map.values()) - set(cameras)
        if missing:
            raise ValueError(f"camera roles missing from hardware config: {sorted(missing)}")
        capture_cameras = {
            str(item["role"]): str(item["dataset_key"])
            for item in input_schema["capture"]["cameras"]
            if item.get("enabled", True)
        }
        if not set(self._role_map.values()) <= set(capture_cameras):
            raise ValueError("role_map does not match enabled input-schema cameras")
        self._image_keys = {output: capture_cameras[source] for output, source in self._role_map.items()}
        camera_source_names = [
            str(item["role"]) for item in input_schema["capture"]["cameras"] if item.get("enabled", True)
        ]
        source_names = [
            *camera_source_names,
            *(str(item["name"]) for item in input_schema["synchronization"]["state_channels"]),
            str(input_schema["synchronization"]["command_channel"]["name"]),
        ]
        # Camera role ``wrist`` and robot state channel ``wrist`` intentionally
        # share a legacy name.  Never collapse the ordered telemetry vector into
        # one dict: camera indices are always the leading capture-role indices.
        self._camera_source_index = {name: index for index, name in enumerate(camera_source_names)}
        self._expected_sources = len(source_names)
        if not np.isfinite(record_hz) or record_hz <= 0:
            raise ValueError("record_hz must be positive and finite")
        synchronization = input_schema["synchronization"]
        self._max_camera_skew_ms = float(synchronization["max_camera_skew_ms"])
        self._max_camera_age_ms = float(synchronization["max_camera_age_ms"])
        self._max_state_age_ms = float(synchronization["max_state_age_ms"])
        self._max_command_age_ms = float(synchronization["max_command_age_ms"])
        self._record_period_ns = round(1_000_000_000 / float(record_hz))
        ur5 = hardware.get("ur5", {})
        self._max_linear_m_s = float(ur5.get("max_linear_m_s", 0.25))
        self._max_angular_rad_s = float(ur5.get("max_angular_rad_s", 0.60))
        min_tcp_z = ur5.get("min_tcp_z_m")
        self._min_tcp_z_m = None if min_tcp_z is None else float(min_tcp_z)
        self._attrs = {
            "task": task,
            "fps": float(record_hz),
            "capture_fps": int(fps),
            "git_commit": _git_identity(Path(openpi_root).resolve()),
            "legacy_git_commit": _git_identity(Path(os.environ.get("UR5_TWINWRIST_LEGACY_ROOT", ".")).resolve()),
            "camera_serials": {output: cameras[source] for output, source in self._role_map.items()},
            "camera_role_map": self._role_map,
            "gripper_backend": str(hardware.get("gripper", {}).get("driver", "disabled")),
            "gripper_state_source": "measured",
            "action_space": "tcp_speedL_6+wrist_absolute_2+gripper_absolute_1",
            "robot_config_hash": _stable_hash(hardware, input_schema, task, record_hz, self._role_map),
            "timestamp_units": "nanoseconds",
            "control_timestamp_source": "primary camera host time.monotonic timeline",
            "camera_timestamp_source": "legacy_device_timestamp_s (raw per-camera clocks)",
            "camera_host_timestamp_source": "legacy monotonic host receive timestamps",
            "ur5_max_linear_m_s": self._max_linear_m_s,
            "ur5_max_angular_rad_s": self._max_angular_rad_s,
        }
        python = Path(openpi_root).resolve() / ".venv/bin/python"
        if not python.is_file():
            raise FileNotFoundError(f"OpenPI HDF5 writer Python not found: {python}")
        self._lock_file = (self._root / ".writer.lock").open("a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise RuntimeError(f"another raw HDF5 writer owns {self._root}") from exc
        try:
            self._process = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "examples.ur5_twinwrist.hdf5_writer_process",
                    "--root",
                    str(self._root),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            self._active = False
            self._closed = False
            self._recovered_paths = tuple(Path(path) for path in self._call("ping"))
            self._episode_id = self._next_episode_id()
            self._reset_episode_state()
        except BaseException:
            process = getattr(self, "_process", None)
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            raise

    def _reset_episode_state(self) -> None:
        self._sealed = False
        self._invalid_reason: str | None = None
        self._previous_fit = False
        self._last_written_ns: int | None = None
        self._last_input_ns: int | None = None
        self._last_camera_sequences: tuple[int, int, int] | None = None

    def _invalidate(self, reason: str, exception: type[Exception] = ValueError) -> None:
        self._invalid_reason = reason
        raise exception(reason)

    def _next_episode_id(self) -> int:
        found: list[int] = []
        for parent in (self._root, self._root / "rejected"):
            for path in parent.glob("episode_*.hdf5") if parent.exists() else ():
                try:
                    found.append(int(path.name.split("_")[1].split(".")[0]))
                except (IndexError, ValueError):
                    continue
        return max(found, default=-1) + 1

    def _start(self, frame: dict[str, Any]) -> None:
        shape = tuple(np.asarray(frame[self._image_keys["front"]]).shape)
        self._call("start", episode_id=self._episode_id, image_shape=shape, attrs=self._attrs)
        self._active = True

    def _call(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("HDF5 writer is closed")
        if self._process.poll() is not None:
            raise RuntimeError(f"HDF5 writer exited with code {self._process.returncode}")
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        message = pickle.dumps({"op": operation, **payload}, protocol=5)
        self._process.stdin.write(struct.pack("!Q", len(message)))
        self._process.stdin.write(message)
        self._process.stdin.flush()
        header = self._process.stdout.read(8)
        if len(header) != 8:
            raise RuntimeError("HDF5 writer returned a truncated response")
        size = struct.unpack("!Q", header)[0]
        response = pickle.loads(self._process.stdout.read(size))
        if not response.get("ok"):
            raise RuntimeError(f"HDF5 writer failed: {response.get('error')}")
        return response.get("result")

    def add_frame(self, frame: dict[str, Any]) -> None:
        state = np.asarray(frame["observation.state"], dtype=np.float32)
        action = np.asarray(frame["action"], dtype=np.float32)
        if state.shape != (9,) or action.shape != (9,):
            self._invalidate(f"state/action must both be 9D, got {state.shape} and {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            self._invalidate("state/action contains NaN or Inf")
        if not 0.0 <= float(state[8]) <= 1.0 or not 0.0 <= float(action[8]) <= 1.0:
            self._invalidate("gripper state/action is outside normalized [0,1]")
        if float(np.linalg.norm(action[:3])) > self._max_linear_m_s + 1e-6:
            self._invalidate("TCP linear action exceeds the UR5 worker limit")
        if float(np.linalg.norm(action[3:6])) > self._max_angular_rad_s + 1e-6:
            self._invalidate("TCP angular action exceeds the UR5 worker limit")
        images = {output: np.asarray(frame[key]) for output, key in self._image_keys.items()}
        image_shapes = {image.shape for image in images.values()}
        if len(image_shapes) != 1 or any(
            image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 for image in images.values()
        ):
            self._invalidate("camera images must be equally shaped HWC uint8 RGB arrays")
        if any(image.size == 0 or float(np.mean(image)) <= 1.0 for image in images.values()):
            self._invalidate("empty or black camera frame")
        if not self._active:
            self._start(frame)

        buttons = np.asarray(frame["telemetry.spacemouse_buttons"], dtype=np.int64)
        fit_pressed = buttons.size > RECORDED_FIT_INDEX and bool(buttons[RECORDED_FIT_INDEX])
        fit_rising = fit_pressed and not self._previous_fit
        self._previous_fit = fit_pressed

        host_ts = np.asarray(frame["telemetry.host_receive_timestamps_s"], dtype=np.float64)
        device_ts = np.asarray(frame["telemetry.device_timestamps_s"], dtype=np.float64)
        if (
            host_ts.shape != (self._expected_sources,)
            or device_ts.shape != (self._expected_sources,)
            or not np.isfinite(host_ts).all()
            or not np.isfinite(device_ts).all()
        ):
            self._invalidate("non-finite synchronized timestamp")
        source_age_ms = np.asarray(frame["telemetry.source_age_ms"], dtype=np.float64)
        expected_sources = self._expected_sources
        if source_age_ms.shape != (expected_sources,) or not np.isfinite(source_age_ms).all():
            self._invalidate("source-age telemetry is missing, non-finite, or has the wrong shape")
        camera_count = len(self._role_map)
        state_count = expected_sources - camera_count - 1
        if np.any(source_age_ms[:camera_count] > self._max_camera_age_ms):
            self._invalidate("camera observation is stale", TimeoutError)
        if np.any(source_age_ms[camera_count : camera_count + state_count] > self._max_state_age_ms):
            self._invalidate("robot state is stale", TimeoutError)
        if source_age_ms[-1] > self._max_command_age_ms:
            self._invalidate("SpaceMouse command is stale", TimeoutError)
        validity = np.asarray(frame["telemetry.validity_mask"], dtype=np.int64)
        if validity.shape != (expected_sources,) or np.any(validity != 1):
            self._invalidate("one or more synchronized sources are invalid")
        camera_skew_ms = np.asarray(frame["telemetry.camera_skew_ms"], dtype=np.float64)
        if not np.isfinite(camera_skew_ms).all() or (
            camera_skew_ms.size and float(np.max(camera_skew_ms)) > self._max_camera_skew_ms
        ):
            self._invalidate("camera skew exceeds configured limit")
        sequences = np.asarray(frame["telemetry.source_sequence_numbers"], dtype=np.int64)
        if sequences.shape != (expected_sources,):
            self._invalidate("source-sequence telemetry has the wrong shape")
        camera_sequences = tuple(int(value) for value in sequences[:camera_count])
        if self._last_camera_sequences is not None:
            if any(
                current == previous
                for current, previous in zip(camera_sequences, self._last_camera_sequences, strict=True)
            ):
                self._invalidate("one or more camera frames were repeated")
            if any(
                current < previous
                for current, previous in zip(camera_sequences, self._last_camera_sequences, strict=True)
            ):
                self._invalidate("one or more camera sequence numbers moved backwards")
        self._last_camera_sequences = camera_sequences

        primary_index = self._camera_source_index[self._role_map["front"]]
        control_timestamp_ns = round(host_ts[primary_index] * 1_000_000_000)
        if self._last_input_ns is not None and control_timestamp_ns <= self._last_input_ns:
            self._invalidate("control timestamp is not strictly monotonic")
        self._last_input_ns = control_timestamp_ns

        if fit_rising:
            # The legacy workflow keeps recording while Fit initiates HOME.
            # Seal here so the successful demonstration ends before reset motion.
            self._sealed = True
        if self._sealed:
            return

        target_qd = np.asarray(frame["telemetry.ur5_target_qd"], dtype=np.float32)
        if target_qd.shape != (6,) or not np.isfinite(target_qd).all():
            self._invalidate("UR5 target_qd must contain six finite values")
        if np.any(np.abs(target_qd) > 1e-6):
            self._invalidate("joint-space speedJ command occurred during a recording window")

        if self._min_tcp_z_m is not None and action[2] < 0.0:
            tcp_pose = np.asarray(frame["observation.tcp_pose"], dtype=np.float32)
            if tcp_pose.shape != (9,) or not np.isfinite(tcp_pose).all():
                self._invalidate("TCP pose is unavailable for the floor-command receipt gate")
            projected_z = float(tcp_pose[2]) + float(action[2]) * 0.25
            if float(tcp_pose[2]) <= self._min_tcp_z_m or projected_z < self._min_tcp_z_m:
                self._invalidate("recorded TCP command may differ from the UR5 worker's floor-clamped command")

        if self._last_written_ns is not None and control_timestamp_ns - self._last_written_ns < self._record_period_ns:
            return

        camera_timestamps = {
            output: round(device_ts[self._camera_source_index[source]] * 1_000_000_000)
            for output, source in self._role_map.items()
        }
        camera_host_timestamps = {
            output: round(host_ts[self._camera_source_index[source]] * 1_000_000_000)
            for output, source in self._role_map.items()
        }
        observation = {
            "qpos": state,
            "images": images,
            "timestamp_ns": control_timestamp_ns,
            "camera_timestamps_ns": camera_timestamps,
            "camera_host_timestamps_ns": camera_host_timestamps,
        }
        self._call(
            "append",
            observation=observation,
            action=action,
        )
        self._last_written_ns = control_timestamp_ns

    def save_episode(self) -> Path | None:
        if not self._active:
            return None
        emergency_save = _called_from("emergency_save_active_episode")
        if emergency_save or not self._sealed or self._invalid_reason is not None or self._last_written_ns is None:
            reason = self._invalid_reason
            if emergency_save:
                reason = "legacy emergency-save path is never accepted as success"
            elif self._last_written_ns is None:
                reason = "episode contains no accepted frames"
            elif reason is None:
                reason = "episode was not sealed by a Fit rising edge"
            path = Path(self._call("reject", reason=reason))
            self._active = False
            self._episode_id += 1
            self._reset_episode_state()
            raise RuntimeError(f"episode rejected instead of saved: {path}: {reason}")
        path = Path(self._call("finish"))
        self._active = False
        self._episode_id += 1
        self._reset_episode_state()
        return path

    def clear_episode_buffer(self) -> Path | None:
        if not self._active:
            return None
        reason = self._invalid_reason or "operator discard or interrupted collection"
        path = Path(self._call("reject", reason=reason))
        self._active = False
        self._episode_id += 1
        self._reset_episode_state()
        return path

    def finalize(self) -> None:
        try:
            self.clear_episode_buffer()
            if not self._closed:
                self._call("close")
                assert self._process.stdin is not None
                self._process.stdin.close()
                self._process.wait(timeout=5)
                if self._process.returncode != 0:
                    raise RuntimeError(f"HDF5 writer exited with code {self._process.returncode}")
                self._closed = True
        finally:
            if self._process.poll() is None and not self._closed:
                self._process.terminate()
                self._process.wait(timeout=5)
            if not self._lock_file.closed:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()


def make_dependencies(hardware: dict[str, Any], dataset: dict[str, Any], task: dict[str, Any]):
    """Return legacy dependencies with only the dataset sink replaced."""
    from dataclasses import replace

    from slai_mi.input_schema import load_input_schema
    from slai_mi.site_adapter import make_dependencies as make_legacy_dependencies

    requested_gripper = os.environ.get("UR5_TWINWRIST_GRIPPER_BACKEND")
    if requested_gripper:
        driver = {"hiwonder": "hiwonder", "feetech": "feetech_sts3215"}.get(requested_gripper)
        if driver is None:
            raise ValueError(f"unsupported gripper backend override: {requested_gripper}")
        hardware = copy.deepcopy(hardware)
        adapters = hardware.get("gripper", {}).get("adapters", {})
        if driver not in adapters:
            raise ValueError(f"legacy hardware config has no gripper adapter {driver!r}")
        hardware["gripper"]["driver"] = driver
    dependencies = make_legacy_dependencies(hardware, dataset, task)
    schema = load_input_schema(hardware.get("input_schema"))
    raw_root = os.environ.get("UR5_TWINWRIST_RAW_ROOT")
    openpi_root = os.environ.get("UR5_TWINWRIST_OPENPI_ROOT")
    if not raw_root or not openpi_root:
        raise RuntimeError("raw HDF5 adapter environment is incomplete")
    record_hz = float(os.environ.get("UR5_TWINWRIST_RECORD_HZ", "10"))
    legacy_preflight = dependencies.preflight

    def collection_preflight(config):
        legacy_preflight(config)
        output = Path(raw_root).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if not os.access(output, os.W_OK):
            raise PermissionError(f"raw HDF5 output is not writable: {output}")
        writer_python = Path(openpi_root).resolve() / ".venv/bin/python"
        result = subprocess.run(
            [str(writer_python), "-c", "import h5py, numpy"],
            cwd=openpi_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"OpenPI HDF5 writer import preflight failed: {result.stderr.strip()}")
        if not np.isfinite(record_hz) or record_hz <= 0:
            raise ValueError("UR5_TWINWRIST_RECORD_HZ must be positive and finite")

    instruction = str(task.get("task", {}).get("instruction") or "").strip()
    resources = dict(dependencies.resource_factories or {})
    original_spacemouse_factory = resources.get("spacemouse")
    if original_spacemouse_factory is None:
        raise RuntimeError("legacy collection dependencies have no SpaceMouse resource")

    def safe_spacemouse_factory(config):
        controlled = original_spacemouse_factory(config)
        if controlled.gripper is not None:
            gripper = SingleOwnerGripper(controlled.gripper)
            controlled.gripper = gripper
            if controlled.gripper_control is not None:
                controlled.gripper_control.gripper = gripper
        controlled.mouse = HomingSafeSpaceMouse(
            controlled.mouse,
            lambda: controlled._home_requested.is_set(),  # noqa: SLF001
        )
        return AtomicControlledSpaceMouse(controlled)

    resources["spacemouse"] = safe_spacemouse_factory
    return replace(
        dependencies,
        dataset_factory=lambda _config, _task: RawHDF5Dataset(
            raw_root,
            task=instruction,
            fps=int(schema["capture"]["fps"]),
            record_hz=record_hz,
            hardware=hardware,
            input_schema=schema,
            openpi_root=openpi_root,
        ),
        preflight=collection_preflight,
        resource_factories=resources,
    )
