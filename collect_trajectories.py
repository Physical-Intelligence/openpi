import pickle
import fire
import math
import collections
import json

import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from pathlib import Path
from ribs.visualize import grid_archive_heatmap
from functools import partial
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy

# helpers
def load_scheduler(pkl_path):
    with open(pkl_path, mode="rb") as f:
        scheduler = pickle.load(f)
    return scheduler

def visualize_archive(scheduler):
    archive = scheduler.result_archive
    fig = plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()
    plt.savefig("1000_heatmap.png")

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# json helper
def numpy_to_native(obj):
    """Recursively convert numpy arrays/scalars in obj to Python lists/numbers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)): 
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_native(v) for v in obj]
    return obj

def rollout_single_trajectory(params, elite_idx, ntrials, seed):
    """
    Rollout a single solution for ntrials and collect successful trajectory 
    if exist in ntrials.

    Args:
        params (np.ndarray): Array of shape (solution_dim,) containing a single 
            solution to be evaluated.
        elite_idx (int): Index of elite in original batch.
        ntrials (int): Number of rollouts for each solution.
        seed (int): Seed.
    
    Return:
        success_trajectory (dict): Dictionary containing all images, observations, 
            action plans, and language for trajectory. None if no success.
    """
    np.random.seed(seed)
    
    # openpi config
    host = "0.0.0.0"
    port = 8000
    openpi_client = _websocket_client_policy.WebsocketClientPolicy(host, port)
    max_steps = 220
    num_steps_wait = 10
    replan_steps = 5

    # loading libero
    task_5_bddl = (
        Path(get_libero_path("bddl_files"))
        / "custom"
        / "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl"
    )
    TASK_ENV = partial(
        OffScreenRenderEnv,
        bddl_file_name=task_5_bddl,
        camera_heights=256,
        camera_widths=256,
    )

    env = TASK_ENV(
        params=params,
        repair_env=True,
        repair_config={
            'time_limit':1500,
            'seed':seed
        }
    )
    env.seed(seed)
    obs = env.reset()
    if obs is None:
        return None

    success_trajectory = None # TODO: add more success trajectories?

    for trial_id in trange(ntrials):
        print(f"Running trajectory {trial_id}")

        obs = env.reset()
        action_plan = collections.deque()
        success = False

        new_trajectory = {
            "prompt": env.language_instruction,
            "image": [],
            "wrist_image": [],
            "state": [],
            "action_plan": []
        }
        for t in range(max_steps + num_steps_wait):
            try:
                if t < num_steps_wait:
                    obs, rewards, done, info = env.step([0.0] * 6 + [-1.0])
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": env.language_instruction,
                    }

                    action_chunk = openpi_client.infer(element)["actions"]
                    assert (
                        len(action_chunk) >= replan_steps
                    ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: replan_steps])

                    # store in trajectory list
                    new_trajectory["image"].append(img)
                    new_trajectory["wrist_image"].append(wrist_img)
                    new_trajectory["state"].append(element["observation/state"]) 
                    new_trajectory["action_plan"].append(list(action_plan)) # (?) is this correct
                
                action = action_plan.popleft()

                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
            except Exception as e:
                print(e)
                return None
        
        if success:
            print("Successful trajectory found!")
            success_trajectory = new_trajectory
            break
    
    if success_trajectory is None:
        print(f"No successful trajectories in {ntrials} for elite idx: {elite_idx}")
        return None

    return success_trajectory

def rollout_trajectories(scheduler, 
                         sample_size,
                         seed):
    """
    Filter elites with high entropy and collect successful trajectories for 
    all elites.

    Args:
        scheduler (ribs.scheduler): Pyribs scheduler.
        samples_size (int): Number of elites to sample from archive.
        seed (int): Seed.
    
    Returns:
        successful_trajectories (list): List of all successful trajectories.
    """
    archive = scheduler.result_archive
    elites = archive.sample_elites(sample_size)
    
    # only keep elites with high entropy
    entropy_threshold = 0.9
    obj_high_idx = np.where(elites["objective"] > entropy_threshold)[0]
    obj_high_elites = {
        "solution": elites["solution"][obj_high_idx],
        "objective": elites["objective"][obj_high_idx],
        "measures": elites["measures"][obj_high_idx],
        "index": elites["index"][obj_high_idx]
    }

    successful_trajectories = []

    ntrials = 5

    # rollout obj high elites 
    # TODO: parallelize
    for i, solution in enumerate(obj_high_elites["solution"]):
        success_trajectory = rollout_single_trajectory(solution, i, ntrials, seed)
        if success_trajectory is not None:
            successful_trajectories.append(success_trajectory)

            # TODO: make this better, quick storage for now
            native_traj = numpy_to_native(success_trajectory)
            with open("trajectories.json", "a") as f:
                print(native_traj)
                json.dump(native_traj, f, indent=4)
                f.write("\n")

    return successful_trajectories

def main(pkl_path):
    scheduler = load_scheduler(pkl_path)
    visualize_archive(scheduler)

    sample_size = 10 

    rollout_trajectories(scheduler, sample_size, 42)

if __name__ == "__main__":
    fire.Fire(main)

# python collect_trajectories.py --pkl_path="scheduler_00001000.pkl"