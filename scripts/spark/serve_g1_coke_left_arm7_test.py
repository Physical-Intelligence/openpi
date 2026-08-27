import importlib.util
from pathlib import Path
import tempfile

import pytest


def _module():
    path = Path(__file__).with_name("serve_g1_coke_left_arm7.py")
    spec = importlib.util.spec_from_file_location("serve_g1_coke_left_arm7", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_identity_and_metadata_are_exact() -> None:
    module = _module()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        model = checkpoint / module.MODEL_FILENAME
        model.write_bytes(b"left-arm7")
        expected = module.sha256(model)
        resolved, actual = module.checkpoint_identity(checkpoint, expected)
        metadata = module.build_metadata({}, resolved, actual)
        assert metadata["schema"] == "wendy.g1.pi05.left-arm7-policy-server.v1"
        assert metadata["config"] == "pi05_spark_g1_coke_rgbd_left_arm7"
        assert metadata["state_dim"] == 17
        assert metadata["action_dim"] == 7
        assert metadata["action_horizon"] == 10
        assert metadata["semantic_hand"] == "physical_left"
        assert metadata["right_arm_included"] is False
        assert metadata["measured_hand_as_action"] is False
        assert metadata["checkpoint_model_sha256"] == expected
        assert all(not name.startswith("right_") for name in metadata["state_names"])


def test_checkpoint_hash_mismatch_fails_closed() -> None:
    module = _module()
    with tempfile.TemporaryDirectory() as directory:
        model = Path(directory) / module.MODEL_FILENAME
        model.write_bytes(b"left-arm7")
        with pytest.raises(ValueError, match="model hash mismatch"):
            module.checkpoint_identity(Path(directory), "0" * 64)
