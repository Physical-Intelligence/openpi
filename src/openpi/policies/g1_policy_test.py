import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import g1_policy


@pytest.mark.parametrize("image_shape", [(224, 224, 3), (3, 224, 224)])
def test_g1_inputs_use_head_camera_and_mask_padding(image_shape):
    transform = g1_policy.G1Inputs(model_type=_model.ModelType.PI05)

    result = transform(
        {
            "head_image": np.zeros(image_shape, dtype=np.uint8),
            "state": np.zeros(29, dtype=np.float32),
            "actions": np.zeros((10, 21), dtype=np.float32),
            "prompt": "slice the fruit",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["state"].shape == (29,)
    assert result["actions"].shape == (10, 21)


def test_g1_outputs_drop_pi_padding():
    outputs = g1_policy.G1Outputs(task_action_dim=21)

    result = outputs({"actions": np.zeros((10, 32), dtype=np.float32)})

    assert result["actions"].shape == (10, 21)
