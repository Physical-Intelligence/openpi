# Run Aloha (Real Robot)

This example demonstrates how to run with a real robot using an [ALOHA setup](https://github.com/tonyzhaozh/aloha). See [here](../../docs/remote_inference.md) for instructions on how to load checkpoints and run inference. We list the relevant checkpoint paths for each provided fine-tuned model below.

## Prerequisites

This repo uses a fork of the ALOHA repo, with very minor modifications to use Realsense cameras.

1. Follow the [hardware installation instructions](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation) in the ALOHA repo.
1. Modify the `third_party/aloha/aloha_scripts/realsense_publisher.py` file to use serial numbers for your cameras.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.aloha_real.main
```

Terminal window 2:

```bash
roslaunch aloha ros_nodes.launch
```

Terminal window 3:

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## **ALOHA Checkpoint Guide**


The `pi0_base` model can be used in zero shot for a simple task on the ALOHA platform, and we additionally provide two example fine-tuned checkpoints, “fold the towel” and “open the tupperware and put the food on the plate,” which can perform more advanced tasks on the ALOHA.

While we’ve found the policies to work in unseen conditions across multiple ALOHA stations, we provide some pointers here on how best to set up scenes to maximize the chance of policy success. We cover the prompts to use for the policies, objects we’ve seen it work well on, and well-represented initial state distributions. Running these policies in zero shot is still a very experimental feature, and there is no guarantee that they will work on your robot. The recommended way to use `pi0_base` is by finetuning with data from the target robot.


---

### **Toast Task**

This task involves the robot taking two pieces of toast out of a toaster and placing them on a plate.

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_base`
- **Prompt**: "take the toast out of the toaster"
- **Objects needed**: Two pieces of toast, a plate, and a standard toaster.
- **Object Distribution**:
  - Works on both real toast and rubber fake toast
  - Compatible with standard 2-slice toasters
  - Works with plates of varying colors

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- The toaster should be positioned in the top-left quadrant of the workspace.
- Both pieces of toast should start inside the toaster, with at least 1 cm of bread sticking out from the top.
- The plate should be placed roughly in the lower-center of the workspace.
- Works with both natural and synthetic lighting, but avoid making the scene too dark (e.g., don't place the setup inside an enclosed space or under a curtain).


### **Towel Task**

This task involves folding a small towel (e.g., roughly the size of a hand towel) into eighths.

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **Prompt**: "fold the towel"
- **Object Distribution**:
  - Works on towels of varying solid colors
  - Performance is worse on heavily textured or striped towels

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- The towel should be flattened and roughly centered on the table.
- Choose a towel that does not blend in with the table surface.


### **Tupperware Task**

This task involves opening a tupperware filled with food and pouring the contents onto a plate.

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **Prompt**: "open the tupperware and put the food on the plate"
- **Objects needed**: Tupperware, food (or food-like items), and a plate.
- **Object Distribution**:
  - Works on various types of fake food (e.g., fake chicken nuggets, fries, and fried chicken).
  - Compatible with tupperware of different lid colors and shapes, with best performance on square tupperware with a corner flap (see images below).
  - The policy has seen plates of varying solid colors.

### **Scene Setup Guidelines**
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- Best performance observed when both the tupperware and plate are roughly centered in the workspace.
- Positioning:
  - Tupperware should be on the left.
  - Plate should be on the right or bottom.
  - The tupperware flap should point toward the plate.

## Training on your own Aloha dataset

1. Convert the dataset to the LeRobot dataset v2.0 format.

    We provide a script [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py) that converts the dataset to the LeRobot dataset v2.0 format. As an example we have converted the `aloha_pen_uncap_diverse_raw` dataset from the [BiPlay repo](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw) and uploaded it to the HuggingFace Hub as [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse).


2. Define a training config that uses the custom dataset.

    We provide the [pi0_aloha_pen_uncap config](../../src/openpi/training/config.py) as an example. You should refer to the root [README](../../README.md) for how to run training with the new config.

IMPORTANT: Our base checkpoint includes normalization stats from various common robot configurations. When fine-tuning a base checkpoint with a custom dataset from one of these configurations, we recommend using the corresponding normalization stats provided in the base checkpoint. In the example, this is done by specifying the trossen asset_id and a path to the pretrained checkpoint’s asset directory within the AssetsConfig.
