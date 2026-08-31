import numpy as np
import pytest

from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.robot_runtime import FakeHardware
from examples.ur5_twinwrist.robot_runtime import build_policy_observation
from examples.ur5_twinwrist.robot_runtime import safety_filter
from examples.ur5_twinwrist.robot_runtime import validate_action_chunk
from openpi import transforms
from openpi.models import model as _model
from openpi.policies.ur5_twinwrist_policy import UR5TwinWristInputs
from openpi.policies.ur5_twinwrist_policy import UR5TwinWristOutputs


def test_inputs_and_output_crop():
    state = np.asarray([0, 1, 2, 3, 4, 5, 22, -47, 0.25], np.float32)
    action = np.tile(np.asarray([0, 0, 0, 0, 0, 0, 20, -10, 1], np.float32), (10, 1))
    data = {"observation.state": state, "action": action, "prompt": "test"}
    for name in ("front", "side", "top"):
        data[f"observation.images.{name}"] = np.zeros((3, 32, 48), np.float32)
    result = UR5TwinWristInputs(_model.ModelType.PI05)(data)
    assert result["state"].shape == (9,)
    assert result["actions"].shape == (10, 9)
    # Policy transform 保持 J1/J2 相对 raw, 不做角度或零位换算。
    np.testing.assert_array_equal(result["state"][6:8], [22, -47])
    np.testing.assert_array_equal(result["actions"][:, 6:8], action[:, 6:8])
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (32, 48, 3) and image.dtype == np.uint8 for image in result["image"].values())
    padded = transforms.PadStatesAndActions(32)(result)
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (10, 32)
    assert np.all(padded["state"][9:] == 0)
    assert np.all(padded["actions"][:, 9:] == 0)
    padded_output = np.zeros((10, 32), np.float32)
    padded_output[:, :9] = action
    cropped = UR5TwinWristOutputs()({"actions": padded_output})["actions"]
    assert cropped.shape == (10, 9)
    np.testing.assert_array_equal(cropped, action)


def test_nan_rejected():
    data = {"observation.state": np.full(9, np.nan)}
    for name in ("front", "side", "top"):
        data[f"observation.images.{name}"] = np.zeros((8, 8, 3), np.uint8)
    try:
        UR5TwinWristInputs(_model.ModelType.PI05)(data)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN accepted")


def test_inference_safety_uses_relative_raw_and_rejects_zero_mismatch():
    project = load_project_config()
    servo_zero = tuple(project["poses"]["wrist"]["servo_zero_raw"])
    hardware = FakeHardware(shape=(8, 8, 3), servo_zero_raw=servo_zero)
    hardware.connect()
    observation = hardware.get_observation()
    candidate = np.asarray([0, 0, 0, 0, 0, 0, 100, -100, 0.5], np.float32)
    filtered = safety_filter(candidate, observation, project, now_ns=observation["timestamp_ns"])
    np.testing.assert_array_equal(filtered[6:8], [18, -18])

    observation["hardware_health"]["wrist"]["servo_zero_raw"] = (0, 0)
    with pytest.raises(RuntimeError, match="servo zero"):
        safety_filter(candidate, observation, project, now_ns=observation["timestamp_ns"])


def test_policy_client_payload_and_chunk_keep_wrist_relative_raw():
    hardware = FakeHardware(shape=(8, 8, 3), servo_zero_raw=(3278, 2547))
    hardware.connect()
    observation = hardware.get_observation()
    observation["qpos"][6:8] = [22, -47]
    payload = build_policy_observation(observation, "拿起测试物体")
    np.testing.assert_array_equal(payload["observation.state"][6:8], [22, -47])
    assert payload["observation.images.front"].dtype == np.uint8

    chunk = np.zeros((10, 9), np.float32)
    chunk[:, 6:8] = [20, -10]
    parsed = validate_action_chunk({"actions": chunk})
    np.testing.assert_array_equal(parsed[:, 6:8], chunk[:, 6:8])
    with pytest.raises(ValueError, match=r"\(H,9\)"):
        validate_action_chunk({"actions": np.zeros((10, 32), np.float32)})
