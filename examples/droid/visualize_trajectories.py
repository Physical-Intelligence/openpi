"""Viser replay of tamp-vla trajectories on an FR3 + Robotiq 2F-85 robot.

Renders the Franka arm (Panda kinematics, identical 7-DOF to the FR3) with a Robotiq 2F-85 gripper,
replaying each successful trajectory from its dense `tiptop_plan.json` (the correct source -- see
`TAMP_VLA_EXPORT_ISSUES.md`). The viser sidebar shows the two recorded camera streams (exterior +
wrist) synced to playback, plus a dropdown to switch between trajectories and play/scrub controls.

The robot model is assembled on the fly from `robot_descriptions` (Panda arm + Robotiq 2F-85),
downloaded/cached on first run. The gripper signal (0 = open, 1 = closed) drives the Robotiq
`finger_joint`.

Run (deps are pulled in ad-hoc so they don't touch the project env):
  uv run --with viser --with yourdfpy --with robot_descriptions --group rlds \
      examples/droid/visualize_trajectories.py \
      --data-dir /home/samrat/tamp-vla/tamp-vla-data/toys-no-collision/success

Then open the printed http://localhost:8080 URL.
"""

import importlib.util
import pathlib
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import tyro

# Reuse the (already validated) plan->15Hz logic from the converter so replay matches the dataset.
_CONV = pathlib.Path(__file__).with_name("convert_combined_droid_toys_to_lerobot.py")
_spec = importlib.util.spec_from_file_location("_droid_conv", _CONV)
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)

# Robotiq 2F-85 driver joint range: 0 rad = open, ~0.725 rad = fully closed.
_ROBOTIQ_CLOSED_RAD = 0.725
# Fixed mount of the gripper on the Panda flange (panda_link8): same -45deg clocking as the Franka hand,
# small coupler offset along the approach (+z) axis.
_MOUNT_XYZ = "0 0 0.011"
_MOUNT_RPY = "0 0 -0.7853981633974483"


def _build_fr3_robotiq_urdf():
    """Assemble a Panda(FR3) arm + Robotiq 2F-85 gripper into one yourdfpy.URDF."""
    import yourdfpy
    from robot_descriptions import panda_description as pd
    from robot_descriptions import robotiq_2f85_description as rq

    cache_root = pathlib.Path(pd.REPOSITORY_PATH).parent  # ~/.cache/robot_descriptions

    def resolve(fname: str) -> str:
        # package://<pkg>/<rest>  ->  <cache_root>/<pkg>/<rest>
        if fname.startswith("package://"):
            return str(cache_root / fname[len("package://") :])
        return fname

    panda = ET.parse(pd.URDF_PATH).getroot()
    # Drop the Franka hand so we can mount the Robotiq gripper instead.
    drop_links = {"panda_hand", "panda_hand_tcp", "panda_leftfinger", "panda_rightfinger"}
    drop_joints = {"panda_hand_joint", "panda_hand_tcp_joint", "panda_finger_joint1", "panda_finger_joint2"}
    for el in list(panda):
        if el.tag == "link" and el.get("name") in drop_links:
            panda.remove(el)
        elif el.tag == "joint" and el.get("name") in drop_joints:
            panda.remove(el)

    # Append the Robotiq links/joints (and any materials it defines).
    existing_materials = {el.get("name") for el in panda.findall("material")}
    robotiq = ET.parse(rq.URDF_PATH).getroot()
    for el in list(robotiq):
        if el.tag in ("link", "joint"):
            panda.append(el)
        elif el.tag == "material" and el.get("name") not in existing_materials:
            panda.append(el)

    # Fixed joint: Panda flange -> Robotiq base.
    mount = ET.SubElement(panda, "joint", {"name": "panda_to_robotiq", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "panda_link8"})
    ET.SubElement(mount, "child", {"link": "robotiq_85_base_link"})
    ET.SubElement(mount, "origin", {"xyz": _MOUNT_XYZ, "rpy": _MOUNT_RPY})
    panda.set("name", "fr3_robotiq")

    import io

    urdf_bytes = ET.tostring(panda, encoding="utf-8")
    return yourdfpy.URDF.load(
        io.BytesIO(urdf_bytes), load_meshes=True, build_scene_graph=True, filename_handler=lambda fname: resolve(fname)
    )


def _load_trajectory(traj: pathlib.Path, img_hw=(180, 320)):
    """Return (positions[N,7], gripper[N], exterior[N,H,W,3], wrist[N,H,W,3], task) at 15 Hz."""
    import json

    plan = json.loads((traj / "tiptop_plan.json").read_text())
    task = json.loads((traj / "_lerobot_raw.json").read_text()).get("instruction", "do something")
    pos50, _vel50, grip50 = _conv._dense_plan(plan)
    n = int(round(len(pos50) * _conv._PLAN_DT * _conv._DROID_FPS))
    sel = _conv._resample_indices(len(pos50), n)
    pos, grip = pos50[sel], grip50[sel]

    exterior = _conv._decode_resized(str(traj / "external_cam.mp4"), img_hw)
    wrist = _conv._decode_resized(str(traj / "hand_cam.mp4"), img_hw)
    ext = np.asarray(exterior)[_conv._resample_indices(len(exterior), n)]
    wr = np.asarray(wrist)[_conv._resample_indices(len(wrist), n)]
    return pos, grip, ext, wr, task


def main(
    data_dir: str = "/home/samrat/tamp-vla/tamp-vla-data/toys-no-collision/success",
    port: int = 8080,
):
    import viser
    from viser.extras import ViserUrdf

    trajs = sorted(p for p in pathlib.Path(data_dir).iterdir() if (p / "tiptop_plan.json").exists())
    if not trajs:
        raise SystemExit(f"No trajectories (with tiptop_plan.json) found under {data_dir}")
    names = [p.name for p in trajs]
    print(f"Found {len(trajs)} trajectories under {data_dir}")

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))

    print("Building FR3 + Robotiq robot model (first run downloads meshes)...")
    urdf = _build_fr3_robotiq_urdf()
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot")
    joint_names = viser_urdf.get_actuated_joint_names()  # panda_joint1..7 + finger_joint
    arm_idx = [joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    grip_idx = joint_names.index("finger_joint")

    # ---- GUI ----
    server.gui.add_markdown("## Trajectory replay\nFR3 + Robotiq 2F-85")
    traj_dd = server.gui.add_dropdown("Trajectory", names, initial_value=names[0])
    task_text = server.gui.add_text("Task", initial_value="", disabled=True)
    with server.gui.add_folder("Playback"):
        play_cb = server.gui.add_checkbox("Play", initial_value=True)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=True)
        speed_sl = server.gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
        frame_sl = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        grip_text = server.gui.add_text("Gripper", initial_value="", disabled=True)
    with server.gui.add_folder("Cameras"):
        ext_img = server.gui.add_image(np.zeros((180, 320, 3), np.uint8), label="exterior", format="jpeg")
        wr_img = server.gui.add_image(np.zeros((180, 320, 3), np.uint8), label="wrist", format="jpeg")

    state = {"data": None, "n": 0}
    lock = threading.Lock()

    def set_frame(i: int):
        data = state["data"]
        if data is None:
            return
        pos, grip, ext, wr, _ = data
        i = int(np.clip(i, 0, len(pos) - 1))
        cfg = np.zeros(len(joint_names))
        for k, j in enumerate(arm_idx):
            cfg[j] = pos[i][k]
        cfg[grip_idx] = float(grip[i]) * _ROBOTIQ_CLOSED_RAD
        viser_urdf.update_cfg(cfg)
        ext_img.image = ext[i]
        wr_img.image = wr[i]
        grip_text.value = "closed" if grip[i] > 0.5 else "open"

    def load(name: str):
        traj = next(p for p in trajs if p.name == name)
        task_text.value = "loading..."
        with lock:
            data = _load_trajectory(traj)
            state["data"] = data
            state["n"] = len(data[0])
        frame_sl.max = state["n"] - 1
        frame_sl.value = 0
        task_text.value = f"{data[4]}  ({state['n']} frames @15Hz)"
        set_frame(0)

    @traj_dd.on_update
    def _(_=None):
        load(traj_dd.value)

    @frame_sl.on_update
    def _(_=None):
        # Only react to manual scrubs (playback updates .value too, but re-rendering is cheap/idempotent).
        set_frame(frame_sl.value)

    load(names[0])
    print(f"\n  Open  http://localhost:{port}  in your browser.\n")

    # Playback loop.
    while True:
        t0 = time.time()
        if play_cb.value and state["n"] > 1:
            nxt = frame_sl.value + 1
            if nxt > state["n"] - 1:
                nxt = 0 if loop_cb.value else state["n"] - 1
            frame_sl.value = nxt  # triggers on_update -> set_frame
        # 15 Hz nominal, scaled by speed.
        dt = 1.0 / (_conv._DROID_FPS * speed_sl.value)
        time.sleep(max(0.0, dt - (time.time() - t0)))


if __name__ == "__main__":
    tyro.cli(main)
