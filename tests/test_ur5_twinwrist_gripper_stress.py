from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import threading

import pytest

import examples.ur5_twinwrist.gripper_stress_test as stress
from examples.ur5_twinwrist.gripper_stress_test import StressSettings
from examples.ur5_twinwrist.gripper_stress_test import _StressRunner


@dataclass
class _State:
    position: int
    normalized_position: float
    target_position: int
    normalized_target_position: float
    binary_state: int
    temperature_c: int = 25
    voltage_v: float = 7.4
    load_enabled: bool = True
    host_timestamp_s: float = 0.0
    sequence: int = 0


class _FakeDelegate:
    driver_name = "fake"
    port = "/dev/serial/by-id/fake"
    servo_id = 1
    open_position = 0
    closed_position = 1000

    def __init__(self, owner_thread_ids: set[int]) -> None:
        self._owner_thread_ids = owner_thread_ids
        self._normalized = 0.0
        self._sequence = 0

    def _record_owner(self) -> None:
        self._owner_thread_ids.add(threading.get_ident())

    def open(self) -> None:
        self._record_owner()

    def close(self) -> None:
        self._record_owner()

    def command_position(self, value: float) -> None:
        self._record_owner()
        self._normalized = value

    def unload(self) -> None:
        self._record_owner()

    def read_state(self, *, enforce_safety: bool = True) -> _State:
        assert enforce_safety
        self._record_owner()
        self._sequence += 1
        position = round(self._normalized * 1000)
        return _State(
            position=position,
            normalized_position=self._normalized,
            target_position=position,
            normalized_target_position=self._normalized,
            binary_state=round(self._normalized),
            sequence=self._sequence,
        )


def test_fake_stress_run_uses_one_owner_and_completes_binary_sequence() -> None:
    owner_thread_ids: set[int] = set()
    settings = StressSettings(
        cycles=2,
        hold_every=0,
        hold_duration_s=0.0,
        feedback_timeout_s=0.2,
        feedback_tolerance=0.01,
        feedback_poll_s=0.001,
        worker_poll_s=0.01,
        worker_stale_timeout_s=0.2,
        operation_timeout_s=0.5,
        open_timeout_s=0.5,
        close_timeout_s=0.5,
        max_reconnects=0,
        reconnect_delay_s=0.0,
        unload_on_exit=True,
    )

    result = _StressRunner(lambda: _FakeDelegate(owner_thread_ids), settings).run()

    successful_phases = [event["phase"] for event in result["operation_events"] if event["status"] == "ok"]
    assert successful_phases == [
        "open",
        "close",
        "open",
        "close",
        "final_fail_safe_open",
        "unload",
    ]
    assert result["reconnect_attempts"] == 0
    assert len(owner_thread_ids) == 1
    assert threading.get_ident() not in owner_thread_ids


def test_fake_stress_run_writes_json_and_text_reports(tmp_path: Path, monkeypatch) -> None:
    owner_thread_ids: set[int] = set()
    settings = StressSettings(
        cycles=1,
        hold_every=0,
        hold_duration_s=0.0,
        feedback_timeout_s=0.2,
        feedback_tolerance=0.01,
        feedback_poll_s=0.001,
        worker_poll_s=0.01,
        worker_stale_timeout_s=0.2,
        operation_timeout_s=0.5,
        open_timeout_s=0.5,
        close_timeout_s=0.5,
        max_reconnects=0,
        reconnect_delay_s=0.0,
        unload_on_exit=True,
    )
    json_path = tmp_path / "report.json"
    text_path = tmp_path / "report.txt"
    plan = {
        "gripper": {"backend": "fake", "port": "/dev/serial/by-id/fake"},
        "settings": asdict(settings),
        "reports": {"json": str(json_path), "text": str(text_path)},
    }
    monkeypatch.setattr(
        stress,
        "_collect_usb_journal",
        lambda _since: {"available": True, "clues": []},
    )

    status = stress.execute_stress_test(
        plan=plan,
        settings=settings,
        factory=lambda: _FakeDelegate(owner_thread_ids),
        output_json=json_path,
        output_text=text_path,
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert status == 0
    assert report["result"]["status"] == "passed"
    assert report["counts"]["serial_exceptions"] == 0
    assert "send_and_reply_ms" in report["run"]["operation_events"][0]
    assert "status: passed" in text_path.read_text(encoding="utf-8")


def _project_config() -> dict:
    return {
        "hardware": {
            "gripper": {
                "enabled": True,
                "backend": "hiwonder",
                "adapters": {
                    "hiwonder": {
                        "port": "/dev/serial/by-id/fake-hiwonder",
                        "servo_id": 1,
                        "baud": 115_200,
                        "open_position": 630,
                        "closed_position": 860,
                    },
                    "feetech": {
                        "port": "/dev/serial/by-id/fake-feetech",
                        "servo_id": 13,
                        "baud": 1_000_000,
                        "open_position": 100,
                        "closed_position": 3995,
                    },
                },
            }
        }
    }


def test_cli_uses_project_config_and_dry_run_never_builds_driver(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(stress, "load_project_config", lambda _path: _project_config())
    monkeypatch.setattr(
        stress,
        "_load_real_factory",
        lambda _config: pytest.fail("dry-run must not import/build a real driver"),
    )

    status = stress.main(["--config-dir", str(tmp_path), "--backend", "feetech"])

    plan = json.loads(capsys.readouterr().out)
    assert status == 0
    assert plan["mode"] == "dry-run-no-device-open"
    assert plan["config_dir"] == str(tmp_path.resolve())
    assert plan["gripper"]["backend"] == "feetech"
    assert plan["gripper"]["port"] == "/dev/serial/by-id/fake-feetech"
    assert "legacy_root" not in plan
    assert "legacy_python" not in plan


def test_removed_legacy_cli_flags_are_rejected() -> None:
    for flag in ("--legacy-root", "--legacy-python", "--hardware-config"):
        with pytest.raises(SystemExit):
            stress.build_parser().parse_args([flag, "unused"])


def test_physical_stress_defaults_match_acceptance_protocol() -> None:
    args = stress.build_parser().parse_args([])
    assert args.cycles == 100
    assert args.hold_every == 10
    assert args.hold_duration_s == 10.0
    assert not args.enable_motion
    assert args.config_dir == stress.DEFAULT_CONFIG_DIR


def test_motion_requires_exact_confirmation_before_env_or_driver(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stress, "load_project_config", lambda _path: _project_config())
    monkeypatch.setattr(
        stress,
        "_require_official_hardware_env",
        lambda: pytest.fail("environment must not be checked before confirmation"),
    )
    monkeypatch.setattr(
        stress,
        "_load_real_factory",
        lambda _config: pytest.fail("driver must not be built before confirmation"),
    )

    with pytest.raises(SystemExit, match=stress.CONFIRMATION):
        stress.main(["--config-dir", str(tmp_path), "--enable-motion"])


def test_confirmed_motion_runs_in_process_without_legacy_reexec(monkeypatch, tmp_path: Path) -> None:
    owner_thread_ids: set[int] = set()
    json_path = tmp_path / "motion.json"
    text_path = tmp_path / "motion.txt"
    env_checks: list[bool] = []
    monkeypatch.setattr(stress, "load_project_config", lambda _path: _project_config())
    monkeypatch.setattr(stress, "_require_official_hardware_env", lambda: env_checks.append(True))
    monkeypatch.setattr(stress, "_load_real_factory", lambda _config: lambda: _FakeDelegate(owner_thread_ids))
    monkeypatch.setattr(stress, "_collect_usb_journal", lambda _since: {"available": True, "clues": []})

    status = stress.main(
        [
            "--config-dir",
            str(tmp_path),
            "--cycles",
            "1",
            "--hold-every",
            "0",
            "--hold-duration-s",
            "0",
            "--output",
            str(json_path),
            "--output-text",
            str(text_path),
            "--enable-motion",
            "--confirm",
            stress.CONFIRMATION,
        ]
    )

    assert status == 0
    assert env_checks == [True]
    assert json.loads(json_path.read_text(encoding="utf-8"))["result"]["status"] == "passed"
    assert len(owner_thread_ids) == 1
