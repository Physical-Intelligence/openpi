from copy import deepcopy
from subprocess import CompletedProcess

from examples.ur5_twinwrist.config_loader import load_project_config
import examples.ur5_twinwrist.teleop_preflight as preflight


def _mock_station(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_service_active", lambda _name: (True, "active"))
    monkeypatch.setattr(
        preflight,
        "_camera_serials",
        lambda: (["254522076307", "254522075176", "260322279394"], None),
    )
    monkeypatch.setattr(preflight, "_tcp_reachable", lambda _host: (True, "reachable"))
    monkeypatch.setattr(preflight, "_active_collectors", list)
    monkeypatch.setattr(preflight.Path, "exists", lambda _path: True)


def test_preflight_uses_only_project_config_and_blocks_unknown_joint_limits(monkeypatch) -> None:
    _mock_station(monkeypatch)
    report = preflight.inspect_station()
    assert not report["ready"]
    assert "legacy_root" not in report
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failed == {"calibrated_safety_limits"}


def test_preflight_can_be_ready_after_explicit_joint_limits(monkeypatch) -> None:
    _mock_station(monkeypatch)
    config = deepcopy(load_project_config())
    config["safety"]["ur5"]["joint_min_rad"] = [-6.0] * 6
    config["safety"]["ur5"]["joint_max_rad"] = [6.0] * 6
    monkeypatch.setattr(preflight, "load_project_config", lambda _root: config)
    report = preflight.inspect_station()
    assert report["ready"]


def test_active_collectors_detects_old_frontend_and_spacemouse_owner(monkeypatch) -> None:
    output = "\n".join(
        (
            "100 python -m slai_mi.ui.collection_frontend --spacemouse-collection-launch",
            "101 python -m slai_mi.devices.spacemouse.workers.events --backend spnav",
            "102 python -m unrelated.service",
        )
    )
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: CompletedProcess([], 0, stdout=output, stderr=""),
    )
    monkeypatch.setattr(preflight.os, "getpid", lambda: 999)
    conflicts = preflight._active_collectors()  # noqa: SLF001
    assert len(conflicts) == 2
    assert "collection_frontend" in conflicts[0]
    assert "spacemouse.workers" in conflicts[1]
