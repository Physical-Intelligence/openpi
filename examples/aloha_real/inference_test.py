from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from PIL import Image, ImageDraw
import random
import torch
import numpy as np

load_start = torch.cuda.Event(enable_timing=True)
load_end = torch.cuda.Event(enable_timing=True)
infer_start = torch.cuda.Event(enable_timing=True)
infer_end = torch.cuda.Event(enable_timing=True)


config = config.get_config("pi0_aloha_towel")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_towel")

load_start.record()
# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
load_end.record()
torch.cuda.synchronize()
load_time = load_start.elapsed_time(load_end)

print(f"\nload time: {load_time / 1000} [sec]\n")

# Run inference on a dummy example.
# example = {
#     "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#     "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#     "observation/joint_position": np.random.rand(6),
#     "observation/gripper_position": np.random.rand(1),
#     "prompt": "fold the towel",
# }
example = {
    "state": np.random.randn(14),  # [left 6 joints, left gripper, right 6 joints, right gripper]
    "images": {
        "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
    },
    "prompt": "fold the towel",
}

infer_start.record()


action_chunk = policy.infer(example)["actions"]

infer_end.record()
torch.cuda.synchronize()
infer_time = infer_start.elapsed_time(infer_end)

print(f"\noutput: {action_chunk}")
print(f"\ninfer time: {infer_time / 1000} [sec]\n")