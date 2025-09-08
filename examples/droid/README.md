# DROID Policies in openpi

We offer instructions for:
- [Running inference for our best $pi_{0.5}$-DROID policy](./README.md#running-droid-inference)
- [Running inference for other pre-trained DROID policies ($\pi_0$, $\pi_0$-FAST, ...)](./README.md#running-roboarena-baseline-policies)
- [Pre-training *generalist* policies on the *full* DROID dataset](./README_train.md#training-on-droid)
- [Fine-tuning expert $\pi_{0.5}$ on your custom DROID dataset](./README_train.md#fine-tuning-on-custom-droid-datasets)

## Running DROID Inference

This example shows how to run the fine-tuned $\pi_{0.5}$-DROID model on the [DROID robot platform](https://github.com/droid-dataset/droid). Based on the [public RoboArena benchmark](https://robo-arena.github.io/leaderboard), this is currently our strongest generalist DROID policy. 


### Step 1: Start a policy server

Since the DROID control laptop does not have a powerful GPU, we will start a remote policy server on a different machine with a more powerful GPU and then query it from the DROID control laptop during inference.

1. On a machine with a powerful GPU (~NVIDIA 4090), clone and install the `openpi` repository following the instructions in the [README](https://github.com/Physical-Intelligence/openpi).
2. Start the OpenPI server via the following command:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

You can also run the equivalent command below:

```bash
uv run scripts/serve_policy.py --env=DROID
```

### Step 2: Run the DROID robot

1. Make sure you have the most recent version of the DROID package installed on both the DROID control laptop and the NUC.
2. On the control laptop, activate your DROID conda environment.
3. Clone the openpi repo and install the openpi client, which we will use to connect to the policy server (this has very few dependencies and should be very fast to install): with the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`.
4. Install `tyro`, which we will use for command line parsing: `pip install tyro`.
5. Copy the `main.py` file from this directory to the `$DROID_ROOT/scripts` directory.
6. Replace the camera IDs in the `main.py` file with the IDs of your cameras (you can find the camera IDs by running `ZED_Explorer` in the command line, which will open a tool that shows you all connected cameras and their IDs -- you can also use it to make sure that the cameras are well-positioned to see the scene you want the robot to interact with).
7. Run the `main.py` file. Make sure to point the IP and host address to the policy server. (To make sure the server machine is reachable from the DROID laptop, you can run `ping <server_ip>` from the DROID laptop.) Also make sure to specify the external camera to use for the policy (we only input one external camera), choose from ["left", "right"].

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

The script will ask you to enter a free-form language instruction for the robot to follow. Make sure to point the cameras at the scene you want the robot to interact with. You _do not_ need to carefully control camera angle, object positions, etc. The policy is fairly robust in our experience. Happy prompting!

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot reach policy server | Make sure the server is running and the IP and port are correct. You can check that the server machine is reachable by running `ping <server_ip>` from the DROID laptop. |
| Cannot find cameras | Make sure the camera IDs are correct and that the cameras are connected to the DROID laptop. Sometimes replugging the cameras can help. You can check all connected cameras by running `ZED_Explore` in the command line. |
| Policy inference is slow / inconsistent | Try using a wired internet connection for the DROID laptop to reduce latency (0.5 - 1 sec latency per chunk is normal). |
| Policy does not perform the task well | In our experiments, the policy could perform simple table top manipulation tasks (pick-and-place) across a wide range of environments, camera positions, and lighting conditions. If the policy does not perform the task well, you can try modifying the scene or object placement to make the task easier. Also make sure that the camera view you are passing to the policy can see all relevant objects in the scene (the policy is only conditioned on a single external camera + wrist camera, make sure you are feeding the desired camera to the policy). Use `ZED_Explore` to check that the camera view you are passing to the policy can see all relevant objects in the scene. Finally, the policy is far from perfect and will fail on more complex manipulation tasks, but it usually makes a decent effort. :) |


## Running Other Policies

We provide configs for running the baseline DROID policies from the [RoboArena](https://robo-arena.github.io/) paper. Simply run the commands below to start inference servers for the respective policies. Then follow the instructions above to run evaluation on the DROID robot.

```
# Train from pi0-FAST, using FAST tokenizer
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# Train from pi0, using flow matching
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid

# Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid

# Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid

# Trained from PaliGemma, using FSQ tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid

# pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

You can find the inference configs in [roboarena_config.py](../../src/openpi/training/misc/roboarena_config.py).
