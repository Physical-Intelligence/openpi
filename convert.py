import json, shutil
from pathlib import Path
import h5py, numpy as np
try:
  from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError:
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  from lerobot.utils.constants import HF_LEROBOT_HOME

h5_path = Path("/home/ziyang10/openpi/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5")
repo_id = "physical-intelligence/libero"
out = HF_LEROBOT_HOME / repo_id
if out.exists():
  shutil.rmtree(out)

with h5py.File(h5_path, "r") as f:
  data = f["data"]
  instr = json.loads(data.attrs["problem_info"])["language_instruction"]
  ds = LeRobotDataset.create(
      repo_id=repo_id, robot_type="panda", fps=10,
      features={
          "image": {"dtype":"image","shape":(128,128,3),"names":["height","width","channel"]},
          "wrist_image": {"dtype":"image","shape":(128,128,3),"names":["height","width","channel"]},
          "state": {"dtype":"float32","shape":(8,), "names":["state"]},
          "actions": {"dtype":"float32","shape":(7,), "names":["actions"]},
      },
      image_writer_threads=4, image_writer_processes=2,
  )
  demos = sorted([k for k in data.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))
  for d in demos:
      g = data[d]
      for t in range(g["actions"].shape[0]):
          state = np.concatenate([g["obs"]["ee_states"][t], g["obs"]["gripper_states"][t]]).astype(np.float32)
          ds.add_frame({
              "image": g["obs"]["agentview_rgb"][t],
              "wrist_image": g["obs"]["eye_in_hand_rgb"][t],
              "state": state,
              "actions": g["actions"][t].astype(np.float32),
              "task": instr,
          })
      ds.save_episode()
print("saved_to", out)
