import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.ur5_twinwrist_policy import UR5TwinWristInputs
from openpi.policies.ur5_twinwrist_policy import UR5TwinWristOutputs


def test_inputs_and_output_crop():
    data = {"observation.state": np.zeros(9, np.float32), "action": np.zeros((10, 9), np.float32), "prompt": "test"}
    for name in ("front", "side", "top"):
        data[f"observation.images.{name}"] = np.zeros((3, 32, 48), np.float32)
    result = UR5TwinWristInputs(_model.ModelType.PI05)(data)
    assert result["state"].shape == (9,)
    assert result["actions"].shape == (10, 9)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (32, 48, 3) and image.dtype == np.uint8 for image in result["image"].values())
    padded = transforms.PadStatesAndActions(32)(result)
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (10, 32)
    assert np.all(padded["state"][9:] == 0)
    assert np.all(padded["actions"][:, 9:] == 0)
    assert UR5TwinWristOutputs()({"actions": np.zeros((10, 32))})["actions"].shape == (10, 9)


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
