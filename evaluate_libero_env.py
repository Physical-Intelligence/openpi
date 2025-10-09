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
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client, LocalCluster
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

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

max_steps = 220
num_steps_wait = 10
host = "0.0.0.0"
port = 8000
replan_steps = 5

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

def evaluate(params, ntrials, seed, video_logdir=None, sol_id=0, summary_file=None):
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
        # TODO: How to handle solutions that fail to evaluate
        return 1e-6, 0, 0, None

    if video_logdir is not None:
        # ID each sol with datetime to prevent overwriting
        sol_logdir = Path(video_logdir) / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sol_logdir.mkdir(parents=True)
    
    success_rate = 0
    for trial_id in trange(ntrials):
        obs = env.reset()
        action_plan = collections.deque()

        success = False
        for t in trange(max_steps + num_steps_wait):
            try:
                if t < num_steps_wait:
                    # Do nothing at the start to wait for env to settle
                    obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
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
            
            # print(f"\t trial{trial_id}: {'success' if success else 'fail'}")
        
            if video_logdir is not None:
                imageio.mimwrite(
                    sol_logdir / f"trial{trial_id}_{'success' if success else 'fail'}.mp4",
                    fps=10,
                )

    with open(summary_file, "w") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow([sol_id, success_rate])
        
        return success_rate
    
def evaluate_parallel(client, params, ntrials, seed, video_logdir=None, summary_file=None):
    batch_size = 16
    nworkers = len(client.scheduler_info()['workers'])
    assert nworkers >= batch_size, (
        f"batch_size={batch_size} exceeds the number of workers "
        f"{nworkers}"
    )

    futures = [
        client.submit(
            evaluate,
            params=sol,
            ntrials=ntrials,
            seed=seed+sol_id,
            video_logdir=video_logdir,
            sol_id=sol_id,
            summary_file=summary_file,
            pure=False,
        )
        for sol_id, sol in enumerate(params)
    ]
    results = client.gather(futures)

    success_rates = []

    for success_rate in results:
        success_rates.append(success_rate)
    
    return success_rates

def main(    
    experiment_cfg
):
    logdir = Path(experiment_cfg["outdir"])
    logdir.mkdir(exist_ok=True)
    summary_filename = logdir / "summary.csv"

    with open(summary_filename, "w") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["env_num", "success_rate"])
    
    params = extract_env_solutions(experiment_cfg["scheduler_pkl"])

    cluster = LocalCluster(
        processes=True, 
        n_workers=16, 
        threads_per_worker=1, 
    )
    client = Client(cluster)

    success_rates = evaluate_parallel(client=client,
                                        params=params,
                                        ntrials=experiment_cfg["ntrials"],
                                        seed=experiment_cfg["seed"],
                                        summary_file=summary_filename)

experiment_cfg = {
    "outdir": "eval_logs",
    "scheduler_pkl": "./test_logs/scheduler_00000105.pkl",
    "ntrials": 5,
    "seed": 42
}

if __name__ == "__main__":
    main(experiment_cfg)