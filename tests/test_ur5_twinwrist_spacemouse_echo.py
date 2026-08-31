import json

from examples.ur5_twinwrist.spacemouse_echo import main


def test_default_echo_is_plan_only(capsys) -> None:
    assert main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["robot_connected"]
    assert not result["motion_possible"]
    assert result["output_order"] == ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]
    assert result["buttons"]["three"]["code"] == 14
    assert result["buttons"]["three"]["action"] == "binary_close_while_held"
