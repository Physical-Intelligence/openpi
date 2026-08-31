# ruff: noqa: RUF001, RUF002
"""当前项目夹爪的 fail-closed 压力测试。

默认仅读取项目 YAML 并打印计划，不导入 pyserial、不开串口、不运动。
真实测试必须同时给出 ``--enable-motion`` 和精确确认短语。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
import copy
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

if __package__:
    from .config_loader import DEFAULT_CONFIG_DIR
    from .config_loader import config_hash
    from .config_loader import load_project_config
else:  # 支持 uv run examples/ur5_twinwrist/gripper_stress_test.py
    from config_loader import DEFAULT_CONFIG_DIR
    from config_loader import config_hash
    from config_loader import load_project_config

CONFIRMATION = "I_UNDERSTAND_REAL_ROBOT_MOTION"
OPENPI_ROOT = Path(__file__).resolve().parents[2]
_BACKENDS = ("hiwonder", "feetech")
_USB_JOURNAL_PATTERN = re.compile(
    r"(?:usb.*(?:disconnect|reset)|(?:disconnect|reset).*usb|tty(?:USB|ACM).*(?:disconnect|reset))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StressSettings:
    cycles: int
    hold_every: int
    hold_duration_s: float
    feedback_timeout_s: float
    feedback_tolerance: float
    feedback_poll_s: float
    worker_poll_s: float
    worker_stale_timeout_s: float
    operation_timeout_s: float
    open_timeout_s: float
    close_timeout_s: float
    max_reconnects: int
    reconnect_delay_s: float
    unload_on_exit: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="项目 YAML 目录，默认 examples/ur5_twinwrist/config。",
    )
    parser.add_argument(
        "--backend",
        choices=_BACKENDS,
        help="覆盖 hardware.yaml 选择的 hiwonder/feetech 后端。",
    )
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--hold-every", type=int, default=10)
    parser.add_argument("--hold-duration-s", type=float, default=10.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=3.0)
    parser.add_argument("--feedback-tolerance", type=float, default=0.05)
    parser.add_argument("--feedback-poll-s", type=float, default=0.02)
    parser.add_argument("--worker-poll-s", type=float, default=0.05)
    parser.add_argument("--worker-stale-timeout-s", type=float, default=0.5)
    parser.add_argument("--operation-timeout-s", type=float, default=5.0)
    parser.add_argument("--open-timeout-s", type=float, default=5.0)
    parser.add_argument("--close-timeout-s", type=float, default=3.0)
    parser.add_argument("--max-reconnects", type=int, default=3)
    parser.add_argument("--reconnect-delay-s", type=float, default=1.0)
    parser.add_argument(
        "--unload-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the final fail-safe open, disable torque (default: true).",
    )
    parser.add_argument(
        "--output",
        "--output-json",
        dest="output_json",
        type=Path,
        default=Path("gripper_stress_report.json"),
    )
    parser.add_argument("--output-text", type=Path)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm")
    return parser


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> None:
    valid = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not valid:
        comparator = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"{name} must be finite and {comparator}")


def _settings(args: argparse.Namespace) -> StressSettings:
    if args.cycles < 1:
        raise SystemExit("--cycles must be at least 1")
    if args.hold_every < 0:
        raise SystemExit("--hold-every must be non-negative")
    if args.max_reconnects < 0:
        raise SystemExit("--max-reconnects must be non-negative")
    _positive_finite("--hold-duration-s", args.hold_duration_s, allow_zero=True)
    _positive_finite("--feedback-timeout-s", args.feedback_timeout_s)
    _positive_finite("--feedback-poll-s", args.feedback_poll_s)
    _positive_finite("--worker-poll-s", args.worker_poll_s)
    _positive_finite("--worker-stale-timeout-s", args.worker_stale_timeout_s)
    _positive_finite("--operation-timeout-s", args.operation_timeout_s)
    _positive_finite("--open-timeout-s", args.open_timeout_s)
    _positive_finite("--close-timeout-s", args.close_timeout_s)
    _positive_finite("--reconnect-delay-s", args.reconnect_delay_s, allow_zero=True)
    if not 0.0 <= args.feedback_tolerance <= 1.0:
        raise SystemExit("--feedback-tolerance must be in [0, 1]")
    if args.worker_stale_timeout_s <= args.worker_poll_s:
        raise SystemExit("--worker-stale-timeout-s must exceed --worker-poll-s")
    return StressSettings(
        cycles=args.cycles,
        hold_every=args.hold_every,
        hold_duration_s=args.hold_duration_s,
        feedback_timeout_s=args.feedback_timeout_s,
        feedback_tolerance=args.feedback_tolerance,
        feedback_poll_s=args.feedback_poll_s,
        worker_poll_s=args.worker_poll_s,
        worker_stale_timeout_s=args.worker_stale_timeout_s,
        operation_timeout_s=args.operation_timeout_s,
        open_timeout_s=args.open_timeout_s,
        close_timeout_s=args.close_timeout_s,
        max_reconnects=args.max_reconnects,
        reconnect_delay_s=args.reconnect_delay_s,
        unload_on_exit=args.unload_on_exit,
    )


def _selected_gripper_config(project: Mapping[str, Any], backend: str | None) -> dict[str, Any]:
    hardware = project.get("hardware")
    if not isinstance(hardware, Mapping):
        raise SystemExit("project config is missing hardware.yaml data")
    value = hardware.get("gripper")
    if not isinstance(value, Mapping) or value.get("enabled") is not True:
        raise SystemExit("hardware.gripper must be a mapping with enabled: true")
    requested = backend or str(value.get("backend", "")).strip().lower()
    if requested not in _BACKENDS:
        available = ", ".join(_BACKENDS)
        raise SystemExit(f"unsupported gripper backend {requested!r}; choose {available}")
    adapters = value.get("adapters")
    if not isinstance(adapters, Mapping):
        raise SystemExit("hardware.gripper.adapters must be a mapping")
    adapter = adapters.get(requested)
    if not isinstance(adapter, Mapping):
        raise SystemExit(f"gripper.adapters has no settings for {requested!r}")
    selected = copy.deepcopy(dict(adapter))
    selected["backend"] = requested
    port = str(selected.get("port", ""))
    if not port:
        raise SystemExit(f"no serial port configured for {requested}")
    selected["port"] = port
    return selected


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "backend",
        "port",
        "servo_id",
        "baud",
        "open_position",
        "closed_position",
        "binary_threshold",
        "feedback_position_bias",
        "full_stroke_time_ms",
        "speed",
        "acceleration",
        "timeout_s",
        "max_temperature_c",
        "max_load_raw",
        "max_current_raw",
    )
    return {key: config[key] for key in keys if key in config}


def _plan(
    *,
    args: argparse.Namespace,
    settings: StressSettings,
    config_dir: Path,
    project_config: Mapping[str, Any],
    gripper_config: Mapping[str, Any],
) -> dict[str, Any]:
    output_json = args.output_json.absolute()
    output_text = (args.output_text or output_json.with_suffix(".txt")).absolute()
    port = Path(str(gripper_config["port"]))
    return {
        "app": "ur5_twinwrist_gripper_stress_test",
        "mode": "motion-enabled" if args.enable_motion else "dry-run-no-device-open",
        "config_dir": str(config_dir),
        "config_sha256": config_hash(dict(project_config)),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "gripper": _public_config(gripper_config),
        "serial_by_id": str(port).startswith("/dev/serial/by-id/"),
        "serial_device_exists": port.exists(),
        "sequence": "open, close for each cycle; hold closed at configured interval; final open",
        "settings": asdict(settings),
        "reports": {"json": str(output_json), "text": str(output_text)},
    }


def _require_motion_confirmation(args: argparse.Namespace) -> None:
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"real gripper motion requires --enable-motion --confirm {CONFIRMATION}")


def _require_official_hardware_env() -> None:
    """真实测试只允许在当前仓库安装了 hardware 组的 .venv 中运行。"""
    expected = (OPENPI_ROOT / ".venv").resolve()
    actual = Path(sys.prefix).resolve()
    if actual != expected:
        raise SystemExit(f"real gripper test must use {expected}; run: uv sync --group hardware")
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version

        pyserial_version = version("pyserial")
    except PackageNotFoundError as exc:
        raise SystemExit("hardware group is missing pyserial; run: uv sync --group hardware") from exc
    if pyserial_version != "3.5":
        raise SystemExit(f"expected locked pyserial 3.5, found {pyserial_version}")


@dataclass(frozen=True)
class _WorkerState:
    """将项目内统一状态转换成现有单-owner worker 的缓存契约。"""

    position: int
    normalized_position: float
    target_position: int
    normalized_target_position: float
    binary_state: int
    temperature_c: int
    voltage_v: float
    load_enabled: bool
    host_timestamp_s: float
    sequence: int
    speed_raw: int | None
    load_raw: int | None
    current_raw: int | None
    moving: bool | None


class _LocalGripperAdapter:
    """把 controller.gripper 的新接口接到 SingleOwnerGripper。"""

    def __init__(self, device: Any) -> None:
        self._device = device
        self.driver_name = str(device.driver_name)
        self.port = str(device.config.port)
        self.servo_id = int(device.config.servo_id)
        self.open_position = int(device.config.open_position)
        self.closed_position = int(device.config.closed_position)

    def open(self) -> None:
        self._device.connect()

    def close(self) -> None:
        self._device.close()

    def command_position(self, value: float) -> None:
        self._device.set_position(value)

    def unload(self) -> None:
        self._device.stop()

    def read_state(self, *, enforce_safety: bool = True) -> _WorkerState:
        del enforce_safety  # 项目内驱动的 get() 永远执行安全检查。
        state = self._device.get()
        return _WorkerState(
            position=int(state.raw_position),
            normalized_position=float(state.position),
            target_position=int(state.raw_target),
            normalized_target_position=float(state.target),
            binary_state=int(float(state.position) >= 0.5),
            temperature_c=int(state.temperature_c),
            voltage_v=float(state.voltage_v),
            load_enabled=bool(state.torque_enabled),
            host_timestamp_s=int(state.host_monotonic_ns) / 1e9,
            sequence=int(state.sequence),
            speed_raw=state.speed_raw,
            load_raw=state.load_raw,
            current_raw=state.current_raw,
            moving=state.moving,
        )


def _load_real_factory(gripper_config: Mapping[str, Any]) -> Callable[[], Any]:
    """延迟导入项目内驱动；调用 factory 本身仍不会打开串口。"""
    if __package__:
        from .controller.gripper import create_gripper
    else:
        from controller.gripper import create_gripper

    config = copy.deepcopy(dict(gripper_config))
    return lambda: _LocalGripperAdapter(create_gripper(config))


def _exception_flags(exc: BaseException) -> dict[str, bool]:
    classes = type(exc).mro()
    serial_exception = any(cls.__name__ == "SerialException" and cls.__module__.startswith("serial") for cls in classes)
    timeout = isinstance(exc, TimeoutError) or any("Timeout" in cls.__name__ for cls in classes)
    os_error = isinstance(exc, OSError)
    return {
        "timeout": timeout,
        "serial_exception": serial_exception,
        "os_error": os_error,
        "retryable": timeout or serial_exception or os_error,
    }


def _state_snapshot(state: Any) -> dict[str, Any]:
    fields = (
        "position",
        "normalized_position",
        "target_position",
        "normalized_target_position",
        "binary_state",
        "temperature_c",
        "voltage_v",
        "load_enabled",
        "sequence",
        "host_timestamp_s",
        "speed_raw",
        "load_raw",
        "current_raw",
        "moving",
    )
    snapshot: dict[str, Any] = {}
    for name in fields:
        if hasattr(state, name):
            value = getattr(state, name)
            if isinstance(value, str | int | float | bool) or value is None:
                snapshot[name] = value
            else:
                snapshot[name] = str(value)
    return snapshot


class _StressRunner:
    def __init__(
        self,
        factory: Callable[[], Any],
        settings: StressSettings,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if __package__:
            from .gripper_worker import SingleOwnerGripper
        else:
            from gripper_worker import SingleOwnerGripper

        self._factory = factory
        self._settings = settings
        self._monotonic = monotonic
        self._sleep = sleep
        self._worker_type = SingleOwnerGripper
        self._worker: Any | None = None
        self._generation = 0
        self._reconnect_attempts = 0
        self._successful_reconnects = 0
        self.connection_events: list[dict[str, Any]] = []
        self.operation_events: list[dict[str, Any]] = []
        self.hold_events: list[dict[str, Any]] = []

    def _new_worker(self) -> Any:
        return self._worker_type(
            self._factory(),
            poll_interval_s=self._settings.worker_poll_s,
            stale_timeout_s=self._settings.worker_stale_timeout_s,
            operation_timeout_s=self._settings.operation_timeout_s,
            open_timeout_s=self._settings.open_timeout_s,
            close_timeout_s=self._settings.close_timeout_s,
        )

    def _close(self) -> BaseException | None:
        worker, self._worker = self._worker, None
        if worker is None:
            return None
        try:
            worker.close()
        except BaseException as exc:
            self.connection_events.append(
                {
                    "event": "close",
                    "generation": self._generation,
                    "status": "error",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    **_exception_flags(exc),
                }
            )
            return exc
        return None

    def _consume_reconnect(self) -> bool:
        if self._reconnect_attempts >= self._settings.max_reconnects:
            return False
        self._reconnect_attempts += 1
        if self._settings.reconnect_delay_s:
            self._sleep(self._settings.reconnect_delay_s)
        return True

    def connect(self, reason: str = "initial") -> None:
        while True:
            started = self._monotonic()
            self._generation += 1
            worker = self._new_worker()
            try:
                worker.open()
            except BaseException as exc:
                elapsed_ms = (self._monotonic() - started) * 1000.0
                event = {
                    "event": "connect",
                    "reason": reason,
                    "generation": self._generation,
                    "status": "error",
                    "elapsed_ms": elapsed_ms,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    **_exception_flags(exc),
                }
                self.connection_events.append(event)
                with suppress(BaseException):
                    worker.close()
                if not event["retryable"] or not self._consume_reconnect():
                    raise
                reason = "connect-retry"
                continue
            self._worker = worker
            self.connection_events.append(
                {
                    "event": "connect",
                    "reason": reason,
                    "generation": self._generation,
                    "status": "ok",
                    "elapsed_ms": (self._monotonic() - started) * 1000.0,
                }
            )
            if reason != "initial":
                self._successful_reconnects += 1
            return

    def _recover(self, exc: BaseException, reason: str) -> bool:
        flags = _exception_flags(exc)
        self._close()
        if not flags["retryable"] or not self._consume_reconnect():
            return False
        self.connect(reason)
        return True

    def _wait_for_target(self, target: float) -> tuple[float, dict[str, Any], int, float]:
        started = self._monotonic()
        reads = 0
        max_read_ms = 0.0
        while True:
            if self._worker is None:
                raise RuntimeError("gripper worker is not connected")
            read_started = self._monotonic()
            state = self._worker.read_state()
            read_ms = (self._monotonic() - read_started) * 1000.0
            max_read_ms = max(max_read_ms, read_ms)
            reads += 1
            position = float(state.normalized_position)
            if abs(position - target) <= self._settings.feedback_tolerance:
                return (
                    (self._monotonic() - started) * 1000.0,
                    _state_snapshot(state),
                    reads,
                    max_read_ms,
                )
            if self._monotonic() - started >= self._settings.feedback_timeout_s:
                raise TimeoutError(f"feedback did not settle at {target:.1f}: measured {position:.3f}")
            self._sleep(self._settings.feedback_poll_s)

    def command_target(self, *, cycle: int, phase: str, target: float) -> None:
        attempt = 0
        while True:
            attempt += 1
            event: dict[str, Any] = {
                "cycle": cycle,
                "phase": phase,
                "target": target,
                "attempt": attempt,
                "generation": self._generation,
            }
            total_started = self._monotonic()
            try:
                if self._worker is None:
                    raise RuntimeError("gripper worker is not connected")
                command_started = self._monotonic()
                # This returns only after write plus an immediate enforced read.
                self._worker.command_position(target)
                event["send_and_reply_ms"] = (self._monotonic() - command_started) * 1000.0
                settle_ms, state, reads, max_read_ms = self._wait_for_target(target)
                event.update(
                    {
                        "status": "ok",
                        "feedback_settle_ms": settle_ms,
                        "feedback_cache_reads": reads,
                        "feedback_cache_read_max_ms": max_read_ms,
                        "total_ms": (self._monotonic() - total_started) * 1000.0,
                        "state": state,
                    }
                )
                self.operation_events.append(event)
                return
            except BaseException as exc:
                event.update(
                    {
                        "status": "error",
                        "total_ms": (self._monotonic() - total_started) * 1000.0,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        **_exception_flags(exc),
                    }
                )
                self.operation_events.append(event)
                if not self._recover(exc, f"retry-{phase}-cycle-{cycle}"):
                    raise

    def hold_closed(self, cycle: int) -> None:
        started = self._monotonic()
        reads = 0
        errors: list[dict[str, Any]] = []
        deadline = started + self._settings.hold_duration_s
        while self._monotonic() < deadline:
            try:
                if self._worker is None:
                    raise RuntimeError("gripper worker is not connected")
                state = self._worker.read_state()
                reads += 1
                if abs(float(state.normalized_position) - 1.0) > self._settings.feedback_tolerance:
                    raise RuntimeError(
                        f"gripper left the closed tolerance during the hold: {float(state.normalized_position):.3f}"
                    )
            except BaseException as exc:
                errors.append(
                    {
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        **_exception_flags(exc),
                    }
                )
                if not self._recover(exc, f"retry-hold-cycle-{cycle}"):
                    self.hold_events.append(
                        {
                            "cycle": cycle,
                            "status": "error",
                            "requested_s": self._settings.hold_duration_s,
                            "actual_s": self._monotonic() - started,
                            "feedback_cache_reads": reads,
                            "errors": errors,
                        }
                    )
                    raise
                self.command_target(cycle=cycle, phase="reclose_after_reconnect", target=1.0)
            remaining = deadline - self._monotonic()
            if remaining > 0.0:
                self._sleep(min(self._settings.feedback_poll_s, remaining))
        self.hold_events.append(
            {
                "cycle": cycle,
                "status": "ok",
                "requested_s": self._settings.hold_duration_s,
                "actual_s": self._monotonic() - started,
                "feedback_cache_reads": reads,
                "errors": errors,
            }
        )

    def unload(self, cycle: int) -> None:
        """Disable torque through the owner worker and record retries/timing."""
        attempt = 0
        while True:
            attempt += 1
            started = self._monotonic()
            event: dict[str, Any] = {
                "cycle": cycle,
                "phase": "unload",
                "attempt": attempt,
                "generation": self._generation,
            }
            try:
                if self._worker is None:
                    raise RuntimeError("gripper worker is not connected")
                self._worker.unload()
                event.update(
                    {
                        "status": "ok",
                        "send_and_reply_ms": (self._monotonic() - started) * 1000.0,
                    }
                )
                self.operation_events.append(event)
                return
            except BaseException as exc:
                event.update(
                    {
                        "status": "error",
                        "total_ms": (self._monotonic() - started) * 1000.0,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        **_exception_flags(exc),
                    }
                )
                self.operation_events.append(event)
                if not self._recover(exc, "retry-unload"):
                    raise

    def run(self) -> dict[str, Any]:
        self.connect()
        completed = False
        try:
            for cycle in range(1, self._settings.cycles + 1):
                self.command_target(cycle=cycle, phase="open", target=0.0)
                self.command_target(cycle=cycle, phase="close", target=1.0)
                if (
                    self._settings.hold_every
                    and cycle % self._settings.hold_every == 0
                    and self._settings.hold_duration_s > 0.0
                ):
                    self.hold_closed(cycle)
            self.command_target(
                cycle=self._settings.cycles,
                phase="final_fail_safe_open",
                target=0.0,
            )
            if self._settings.unload_on_exit:
                self.unload(self._settings.cycles)
            completed = True
        finally:
            close_error = self._close()
            if completed and close_error is not None:
                raise close_error
        return self.report_data()

    def report_data(self) -> dict[str, Any]:
        """Return JSON-serializable progress, including partial failed runs."""
        return {
            "connection_events": self.connection_events,
            "operation_events": self.operation_events,
            "hold_events": self.hold_events,
            "reconnect_attempts": self._reconnect_attempts,
            "successful_reconnections": self._successful_reconnects,
        }


def _collect_usb_journal(since: datetime) -> dict[str, Any]:
    command = [
        "journalctl",
        "--since",
        since.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "--no-pager",
        "--output=short-iso",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "clues": []}
    clues = [line for line in completed.stdout.splitlines() if _USB_JOURNAL_PATTERN.search(line)]
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
        "clues": clues,
    }


def _failure_counts(run_data: Mapping[str, Any]) -> dict[str, int]:
    events = [*run_data.get("connection_events", []), *run_data.get("operation_events", [])]
    return {
        "timeouts": sum(bool(event.get("timeout")) for event in events),
        "serial_exceptions": sum(bool(event.get("serial_exception")) for event in events),
        "os_errors": sum(bool(event.get("os_error")) for event in events),
        "failed_transactions": sum(event.get("status") == "error" for event in events),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _text_report(report: Mapping[str, Any]) -> str:
    result = report["result"]
    counts = report["counts"]
    run = report["run"]
    lines = [
        "UR5 Twin-Wrist Gripper Stress Test",
        f"status: {result['status']}",
        f"started: {report['started_at']}",
        f"finished: {report['finished_at']}",
        f"backend: {report['plan']['gripper']['backend']}",
        f"port: {report['plan']['gripper']['port']}",
        f"cycles requested: {report['plan']['settings']['cycles']}",
        f"successful command transactions: {counts['successful_transactions']}",
        f"failed transactions: {counts['failed_transactions']}",
        f"timeouts: {counts['timeouts']}",
        f"SerialException: {counts['serial_exceptions']}",
        f"OS errors: {counts['os_errors']}",
        f"reconnect attempts: {run.get('reconnect_attempts', 0)}",
        f"successful reconnects: {run.get('successful_reconnections', 0)}",
        f"USB disconnect/reset journal clues: {len(report['journal']['clues'])}",
    ]
    if result.get("error"):
        lines.append(f"error: {result['error_type']}: {result['error']}")
    if report["journal"]["clues"]:
        lines.extend(["", "journal clues:", *report["journal"]["clues"]])
    lines.extend(
        [
            "",
            "Timing fields:",
            "send_and_reply_ms = serialized write plus immediate enforced feedback transaction",
            "feedback_settle_ms = time until cached measured position enters tolerance",
        ]
    )
    return "\n".join(lines) + "\n"


def execute_stress_test(
    *,
    plan: Mapping[str, Any],
    settings: StressSettings,
    factory: Callable[[], Any],
    output_json: Path,
    output_text: Path,
) -> int:
    started = datetime.now().astimezone()
    run_data: dict[str, Any] = {
        "connection_events": [],
        "operation_events": [],
        "hold_events": [],
        "reconnect_attempts": 0,
        "successful_reconnections": 0,
    }
    result: dict[str, Any] = {"status": "failed"}
    try:
        runner = _StressRunner(factory, settings)
        run_data = runner.run()
        result = {"status": "passed"}
    except BaseException as exc:
        if "runner" in locals():
            run_data = runner.report_data()
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            **_exception_flags(exc),
        }
    finished = datetime.now().astimezone()
    failure_counts = _failure_counts(run_data)
    counts = {
        **failure_counts,
        "successful_transactions": sum(event.get("status") == "ok" for event in run_data["operation_events"]),
    }
    report = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "plan": dict(plan),
        "result": result,
        "counts": counts,
        "run": run_data,
        "journal": _collect_usb_journal(started),
    }
    _atomic_write_text(output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(output_text, _text_report(report))
    print(json.dumps({"result": result, "reports": plan["reports"]}, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    settings = _settings(args)
    config_dir = args.config_dir.expanduser().resolve()
    try:
        project_config = load_project_config(config_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    gripper_config = _selected_gripper_config(project_config, args.backend)
    plan = _plan(
        args=args,
        settings=settings,
        config_dir=config_dir,
        project_config=project_config,
        gripper_config=gripper_config,
    )
    if not args.enable_motion:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    _require_motion_confirmation(args)
    port = str(gripper_config["port"])
    if not port.startswith("/dev/serial/by-id/"):
        raise SystemExit(f"refusing non-persistent serial device path: {port}")
    _require_official_hardware_env()
    factory = _load_real_factory(gripper_config)
    output_json = args.output_json.absolute()
    output_text = (args.output_text or output_json.with_suffix(".txt")).absolute()
    return execute_stress_test(
        plan=plan,
        settings=settings,
        factory=factory,
        output_json=output_json,
        output_text=output_text,
    )


if __name__ == "__main__":
    raise SystemExit(main())
