import collections
import csv
import datetime
import math
import pickle as pkl
import re
from functools import partial
from pathlib import Path

import fire
import imageio
import numpy as np

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import websocket_client_policy as _websocket_client_policy

from tqdm import tqdm, trange

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

def extract_env_solutions(scheduler_pkl):
    with open(scheduler_pkl, "rb") as f:
        scheduler = pkl.load(f)
    
    archive = scheduler.archive
    params = archive.data(fields="solution")
    return params

def evaluate_libero_base(host,
                         port,
                         ntrials,
                         max_steps,
                         num_steps_wait,
                         replan_steps,
                         seed):
    # setting up libero
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()

    task_id = -1
    for i in range(10):
        task = task_suite.get_task(i)
        if task.language == "pick up the black bowl next to the ramekin and place it on the plate":
            task_id = i
            break

    initial_states = task_suite.get_task_init_states(task_id)
    task = task_suite.get_task(task_id)
    task_description = task.language

    print(task_description)

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed) 

    # setting up openpi
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    success_rate = 0
    for episode_idx in trange(ntrials):
        # Reset environment
        env.reset()
        action_plan = collections.deque()

        # Set initial states
        if initial_states is None:
            obs = env.env._get_observations()
        else:
            obs = env.set_init_state(initial_states[episode_idx])

        success = False
        t = 0
        while t < max_steps + num_steps_wait:
            try:
                if t < num_steps_wait:
                    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
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

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    assert (
                        len(action_chunk) >= replan_steps
                    ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: replan_steps])

                action = action_plan.popleft()

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success_rate += 1 / ntrials
                    success = True
                    break
                t += 1

            except Exception as e:
                break

    return success_rate


def evaluate(params,
             host,
             port, 
             ntrials, 
             max_steps,
             num_steps_wait,
             replan_steps,
             seed, 
             video_logdir=None):
    np.random.seed(seed)
    openpi_client = _websocket_client_policy.WebsocketClientPolicy(host, port)

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
        return 1e-6, 0, 0, None

    if video_logdir is not None:
        sol_logdir = Path(video_logdir) / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sol_logdir.mkdir(parents=True)
    
    success_rate = 0
    for trial_id in trange(ntrials):
        obs = env.reset()
        action_plan = collections.deque()

        success = False
        images = []
        for t in trange(max_steps + num_steps_wait):
            try:
                if t < num_steps_wait:
                    # Do nothing at the start to wait for env to settle
                    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                images.append(img)
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

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success_rate += 1 / ntrials
                    success = True
                    break
            except Exception as e:
                print(e)
                # TODO: How to handle solutions that fail to evaluate
                return 1e-6, 0, 0, None
        
            if video_logdir is not None:
                imageio.mimwrite(
                    sol_logdir / f"trial{trial_id}_{'success' if success else 'fail'}.mp4",
                    images,
                    fps=10,
                )
        
    return success_rate

def main(   
    experiment_cfg_name
):
    experiment_configs = [
        {
            "name": "pi0_libero_base",
            "outdir": "eval_base_libero_logs",
            "scheduler_pkl": "./test_logs/scheduler_00000105.pkl",
            "finetune_true": False,
            "ntrials": 5,
            "seed": 42,
            "max_steps": 220,
            "num_steps_wait": 10,
            "host": "0.0.0.0",
            "port": 8000,
            "replan_steps": 5
        },
        {
            "name": "pi0_libero_finetuned",
            "outdir": "eval_finetuned_libero_logs",
            "scheduler_pkl": "./test_logs/scheduler_00000105.pkl",
            "finetune_true": False,
            "ntrials": 5,
            "seed": 42,
            "max_steps": 220,
            "num_steps_wait": 10,
            "host": "0.0.0.0",
            "port": 8001,
            "replan_steps": 5
        },
        {
            "name": "pi0_qd_random_base",
            "outdir": "eval_random_logs",
            
            "scheduler_pkl": "./test_logs_random/scheduler_00000080.pkl",
            "finetune_true": False,
            "ntrials": 5,
            "seed": 42,
            "max_steps": 220,
            "num_steps_wait": 10,
            "host": "0.0.0.0",
            "port": 8002,
            "replan_steps": 5
        },
        {
            "name": "pi0_qd_random_finetuned",
            "outdir": "eval_finetuned_random_logs",
            "scheduler_pkl": "./test_logs_random/scheduler_00000080.pkl",
            "finetune_true": False,
            "ntrials": 5,
            "seed": 42,
            "max_steps": 220,
            "num_steps_wait": 10,
            "host": "0.0.0.0",
            "port": 8003,
            "replan_steps": 5
        }
    ]

    for cfg in experiment_configs:
        if cfg["name"] == experiment_cfg_name:
            experiment_cfg = cfg
            break

    logdir = Path(experiment_cfg["outdir"])
    
    logdir.mkdir(exist_ok=True)
    summary_filename = logdir / "summary.csv"

    with open(summary_filename, "w") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["env_num", "success_rate"])
    
    params = None
    if "qd" in experiment_cfg["name"]:
        params = extract_env_solutions(experiment_cfg["scheduler_pkl"])

    if params is not None:
        for sol_id, sol in enumerate(params):
            success_rate = evaluate(params=sol, 
                                    host=experiment_cfg["host"],
                                    port=experiment_cfg["port"],
                                    ntrials=experiment_cfg["ntrials"], 
                                    max_steps=experiment_cfg["max_steps"],
                                    num_steps_wait=experiment_cfg["num_steps_wait"],
                                    replan_steps=experiment_cfg["replan_steps"],
                                    seed=experiment_cfg["seed"]+sol_id,
                                    video_logdir="./vids")

            with open(summary_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([sol_id, success_rate])
    else:
        for sol_id in np.arange(50):
            success_rate = evaluate_libero_base(host=experiment_cfg["host"],
                                                port=experiment_cfg["port"],
                                                ntrials=experiment_cfg["ntrials"], 
                                                max_steps=experiment_cfg["max_steps"],
                                                num_steps_wait=experiment_cfg["num_steps_wait"],
                                                replan_steps=experiment_cfg["replan_steps"],
                                                seed=experiment_cfg["seed"]+sol_id)

            with open(summary_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([sol_id, success_rate])

if __name__ == "__main__":
    # python evaluate_libero_env.py --experiment_cfg_name="pi0_libero_finetuned"

    fire.Fire(main)