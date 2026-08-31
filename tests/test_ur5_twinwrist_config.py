from copy import deepcopy

import pytest

from examples.ur5_twinwrist.config_loader import config_hash
from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.config_loader import validate_project_config


def test_project_yaml_is_consistent_and_hash_is_stable() -> None:
    first = load_project_config()
    second = load_project_config()
    assert config_hash(first) == config_hash(second)
    assert [item["role"] for item in first["hardware"]["cameras"]["devices"]] == ["front", "side", "top"]
    assert first["collection"]["action"]["dimension"] == 9

    wrist = first["hardware"]["wrist"]
    assert wrist["stream_period_ms"] == 10
    assert wrist["command_hz"] == 50.0
    assert wrist["state_hz"] == 10.0
    assert wrist["control_mode"] == "servo_position_open_loop"
    assert wrist["coordinate"] == "yaml_servo_zero_relative_raw"
    assert wrist["master_mapping"] == {
        "j1_source": "enc0",
        "j1_sign": 1,
        "j1_raw_per_deg": 10.0,
        "j2_source": "enc1",
        "j2_sign": 1,
        "j2_raw_per_deg": 14.0,
        "input_deadband_deg": 0.5,
    }
    assert first["poses"]["wrist"]["servo_zero_raw"] == [3278, 2547]
    assert wrist["override_max_velocity_deg_s"] == first["teleop"]["modes"]["wrist"]["max_speed_deg_s"]


def test_real_ready_requires_polyscope_joint_limits() -> None:
    with pytest.raises(ValueError, match="PolyScope"):
        load_project_config(require_real_ready=True)


def test_duplicate_button_code_is_rejected() -> None:
    config = deepcopy(load_project_config())
    config["teleop"]["buttons"]["fit"]["code"] = config["teleop"]["buttons"]["menu"]["code"]
    with pytest.raises(ValueError, match="button code"):
        validate_project_config(config)


def test_teleop_safety_flags_and_canonical_buttons_are_enforced() -> None:
    config = deepcopy(load_project_config())
    del config["teleop"]["buttons"]["rotation_lock"]
    with pytest.raises(ValueError, match="canonical"):
        validate_project_config(config)

    config = deepcopy(load_project_config())
    config["teleop"]["episode"]["reject_joint_jog_during_recording"] = False
    with pytest.raises(ValueError, match="reject_joint_jog"):
        validate_project_config(config)


def test_rotation_and_wrist_modifiers_must_be_distinct_named_buttons() -> None:
    config = deepcopy(load_project_config())
    config["teleop"]["modes"]["rotation"]["modifier"] = "ctrl"
    with pytest.raises(ValueError, match="同一个键"):
        validate_project_config(config)


def test_missing_explicit_master_wrist_parameter_is_rejected() -> None:
    config = deepcopy(load_project_config())
    del config["hardware"]["wrist"]["stream_period_ms"]
    with pytest.raises(ValueError, match="stream_period_ms"):
        validate_project_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("j1_source", "enc1", "Enc0→J1"),
        ("j2_source", "enc0", "Enc0→J1"),
        ("j1_sign", 0, "sign"),
        ("j1_sign", 1.2, "sign"),
        ("j2_sign", 2, "sign"),
        ("input_deadband_deg", -0.1, "input_deadband_deg"),
    ],
)
def test_invalid_master_wrist_mapping_is_rejected(field: str, value: object, message: str) -> None:
    config = deepcopy(load_project_config())
    config["hardware"]["wrist"]["master_mapping"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_project_config(config)


def test_master_wrist_rates_must_be_coherent() -> None:
    config = deepcopy(load_project_config())
    config["hardware"]["wrist"]["command_hz"] = 101.0
    with pytest.raises(ValueError, match="TELE"):
        validate_project_config(config)

    config = deepcopy(load_project_config())
    config["hardware"]["wrist"]["state_hz"] = 51.0
    with pytest.raises(ValueError, match="state_hz"):
        validate_project_config(config)


def test_spacemouse_wrist_speed_duplicate_must_match() -> None:
    config = deepcopy(load_project_config())
    config["hardware"]["wrist"]["override_max_velocity_deg_s"] = 44.0
    with pytest.raises(ValueError, match="max_velocity_deg_s"):
        validate_project_config(config)


def test_wrist_servo_zero_and_cap_axis_contract_are_enforced() -> None:
    config = deepcopy(load_project_config())
    config["poses"]["wrist"]["servo_zero_raw"] = [5000, 2547]
    with pytest.raises(ValueError, match="servo_zero_raw"):
        validate_project_config(config)

    config = deepcopy(load_project_config())
    config["teleop"]["modes"]["wrist"]["cap_y_to_axis"] = "j2"
    with pytest.raises(ValueError, match="j1/j2"):
        validate_project_config(config)
