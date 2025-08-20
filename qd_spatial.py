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
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap

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
client = _websocket_client_policy.WebsocketClientPolicy(host, port)
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

def _extract_scheduler_itr(filename):
    """Tries extracting the iteration number from a scheduler filename following 
    the ``*/scheduler_[0-9]{8}.pkl`` format, where ``[0-9]{8}`` is the iteration 
    number. If match fails, returns None.
    
    Args:
        filename (str): A scheduler filename following the 
            ``*/scheduler_[0-9]{8}.pkl`` format.
    
    Returns:
        itr (int or None): Iteration number if match succeeds, else None.
    """
    pattern = r"scheduler_(\d{8})\.pkl"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def evaluate(params, ntrials, seed, video_logdir=None):
    """Evaluates params by creating LIBERO environments and computing
    objective and measure values from the environments' features and VLA
    rollout.

    Args:
        params (array-like): Array of shape (batch_size, solution_dim)
            containing params to be evaluated.
        ntrials (int): Number of rollouts for each solution.
        seed (int): Seed.
        video_logdir (str): Folder for saving rollout videos. If None no video 
            is saved.

    Return:
        objective (np.ndarray): Array of shape (batch_size,) containing VLA's
            success rates on environments created from each row of ``params``.
            Invalid rows get np.nan.
        measures (np.ndarray): Array of shape (batch_size, measure_dim)
            containing measure values computed from the environments' features.
            Invalid rows get [np.nan] * measure_dim.
    """
    # NOTE: Ideally this should be parallelized, but we may have to evaluate
    # solutions one at a time due to VLA's high GPU memory requirement.
    np.random.seed(seed)

    objectives = []
    measures = []
    for sol_id, sol in enumerate(params):
        env = TASK_ENV(params=sol, repair_env=True, repair_config={'time_limit':1500, 'seed':seed})

        env.seed(seed)
        obs = env.reset()
        if obs is None:
            # TODO: How to handle solutions that fail to generate
            objectives.append(1e-6)
            measures.append([0, 0])
            continue

        if video_logdir is not None:
            # ID each sol with datetime to prevent overwriting
            sol_logdir = Path(video_logdir) / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sol_logdir.mkdir(parents=True)
        
        # compute_spread_similarity must be called at the start before any 
        # action since actions might change objects' locations
        spread, similarity = env.env.compute_spread_similarity()
        measures.append([spread, similarity])

        # Get success rates by running openpi on env
        print(f"------------------ Rolling out solution{sol_id}")
        success_rate = 0
        for trial_id in range(ntrials):
            obs = env.reset()
            action_plan = collections.deque()

            replay_images = []

            success = False
            for t in range(max_steps + num_steps_wait):
                try:
                    if t < num_steps_wait:
                        # Do nothing at the start to wait for env to settle
                        obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    replay_images.append(img)

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

                        action_chunk = client.infer(element)["actions"]
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
                    break

            print(f"\t trial{trial_id}: {'success' if success else 'fail'}")
            
            if video_logdir is not None:
                imageio.mimwrite(
                    sol_logdir / f"trial{trial_id}_{'success' if success else 'fail'}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            
        # Maximizes entropy as objective, i.e. we want more uncertain 
        # success rates
        success_rate = np.clip(success_rate, 1e-6, 1 - 1e-6)
        objectives.append(-success_rate*math.log2(success_rate) - (1-success_rate)*math.log2(1-success_rate))

    return np.asarray(objectives), np.asarray(measures)


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=1, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def main(
    iterations=1000,
    num_trials_per_sol=5,
    batch_size=8,
    num_emitters=1,
    archive_resolution=[100,100],
    seed=42,
    outdir="test_logs",
    reload_from=None,
    log_every=10,
):
    logdir = Path(outdir)
    logdir.mkdir(exist_ok=True)
    summary_filename = logdir / "summary.csv"

    if reload_from is None:
        # For now ``params`` should be an array listing object
        # coordinates in the following order:
        #   [
        #       akita_black_bowl_1_x, akita_black_bowl_1_y,
        #       akita_black_bowl_2_x, akita_black_bowl_2_y,
        #       cookies_1_x, cookies_1_y,
        #       glazed_rim_porcelain_ramekin_1_x,
        #       glazed_rim_porcelain_ramekin_1_y,
        #       plate_1_x, plate_1_y
        #   ]
        main_archive = GridArchive(
            solution_dim=10,
            dims=archive_resolution,
            ranges=[(0, 1)] * 2,
            # learning_rate=0.1,
            # threshold_min=0,
            seed=seed,
        )
        passive_archive = GridArchive(
            solution_dim=10,
            dims=archive_resolution,
            ranges=[(0, 1)] * 2,
            seed=seed,
        )

        emitters = [
            EvolutionStrategyEmitter(
                archive=main_archive,
                # Range centers copied from BDDL file
                x0=[-0.18, 0.32, 0.13, -0.07, 0.07, 0.03, -0.20, 0.20, 0.06, 0.20],
                sigma0=0.02,
                # TODO: Define bounds if we want to stay close to the original BDDL
                bounds=None,
                batch_size=batch_size,
                seed=seed + i,
            )
            for i in range(num_emitters)
        ]

        scheduler = Scheduler(main_archive, emitters, result_archive=passive_archive)

        with open(summary_filename, "w") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow(["Iteration", "QD-Score", "Coverage", "Maximum", "Average"])
    else:
        reload_itr = _extract_scheduler_itr(reload_from)
        assert reload_itr is not None, (
            f'Received invalid reload_from parameter {reload_from}; '
            'expected */scheduler_[0-9]{8}.pkl'
        )
        with open(file=reload_from, mode="rb") as f:
            scheduler = pkl.load(f)

    start = 1 if reload_from is None else reload_itr + 1
    end = start + iterations
    for i in range(start, end):
        solutions = scheduler.ask()
        objectives, measures = evaluate(params=solutions, ntrials=num_trials_per_sol, seed=seed, video_logdir=None)
        scheduler.tell(objectives, measures)

        print(
            f"\n------------------ Iteration{i} ------------------\n"
            f"\t QD-Score: {scheduler.result_archive.stats.qd_score}\n"
            f"\t Coverage: {scheduler.result_archive.stats.coverage}\n"
            f"\t Maximum : {scheduler.result_archive.stats.obj_max}\n"
            f"\t Average : {scheduler.result_archive.stats.obj_mean}\n"
        )

        final_itr = i == end
        if i % log_every == 0 or final_itr:
            pkl.dump(
                scheduler,
                open(logdir / f"scheduler_{i:08d}.pkl", "wb"),
            )

            with open(summary_filename, "a") as summary_file:
                writer = csv.writer(summary_file)
                data = [
                    i,
                    scheduler.result_archive.stats.qd_score,
                    scheduler.result_archive.stats.coverage,
                    scheduler.result_archive.stats.obj_max,
                    scheduler.result_archive.stats.obj_mean,
                ]
                writer.writerow(data)

            save_heatmap(
                scheduler.result_archive,
                logdir / f"heatmap_{i:08d}.png",
            )


if __name__ == "__main__":
    fire.Fire(main)
