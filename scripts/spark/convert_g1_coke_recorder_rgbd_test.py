from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPT = Path(__file__).with_name("convert_g1_coke_recorder_rgbd.py")
SPEC = importlib.util.spec_from_file_location("convert_g1_coke_recorder_rgbd", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _trial() -> dict:
    def frame(t_s: float, offset: float) -> dict:
        return {
            "t_s": t_s,
            "measured_q_rad": (np.arange(17, dtype=np.float32) + offset).tolist(),
            "hand": {
                "measured_q_rad": (np.arange(7, dtype=np.float32) + 20.0 + offset).tolist()
            },
            "teach_command": {
                "q_rad": (np.arange(17, dtype=np.float32) + 200.0 + offset).tolist()
            },
        }

    return {"frames": [frame(0.0, 0.0), frame(1.0, 100.0)]}


def test_left_arm7_contract_has_no_right_arm_names() -> None:
    contract = converter.LEFT_ARM7_CONTRACT

    assert contract.state_dim == 17
    assert contract.action_dim == 7
    assert all(not name.startswith("right_") for name in contract.state_names)
    assert all(name.startswith("left_") for name in contract.action_names)


def test_left_arm7_slices_only_waist_left_arm_and_left_hand() -> None:
    state, action = converter._interpolate_robot(  # noqa: SLF001
        _trial(), 0.25, converter.LEFT_ARM7_CONTRACT
    )

    np.testing.assert_allclose(state[:10], np.arange(10, dtype=np.float32) + 25.0)
    np.testing.assert_allclose(state[10:], np.arange(7, dtype=np.float32) + 45.0)
    np.testing.assert_allclose(action, np.arange(3, 10, dtype=np.float32) + 225.0)
    assert state.shape == (17,)
    assert action.shape == (7,)


def test_arm14_contract_remains_backward_compatible() -> None:
    state, action = converter._interpolate_robot(_trial(), 0.25)  # noqa: SLF001

    np.testing.assert_allclose(state[:17], np.arange(17, dtype=np.float32) + 25.0)
    np.testing.assert_allclose(state[17:], np.arange(7, dtype=np.float32) + 45.0)
    np.testing.assert_allclose(action, np.arange(3, 17, dtype=np.float32) + 225.0)
    assert state.shape == (24,)
    assert action.shape == (14,)
