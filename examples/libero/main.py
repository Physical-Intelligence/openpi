import collections
import dataclasses
import json
import logging
import math
import pathlib
import re

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_start: int = 0  # First task id to evaluate.
    num_tasks: int = 0  # Number of tasks to evaluate. 0 means all tasks from task_start.
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = ""  # JSONL per-episode results. Defaults to video_out_path/results.jsonl
    summary_out_path: str = ""  # JSON summary. Defaults to video_out_path/summary.json
    save_per_episode_videos: bool = True  # Save unique videos instead of overwriting one file per task/status.

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    task_end = num_tasks_in_suite if args.num_tasks <= 0 else min(num_tasks_in_suite, args.task_start + args.num_tasks)
    task_ids = list(range(args.task_start, task_end))
    if not task_ids:
        raise ValueError(
            f"Empty task range: task_start={args.task_start}, num_tasks={args.num_tasks}, "
            f"num_tasks_in_suite={num_tasks_in_suite}"
        )
    logging.info(f"Task ids: {task_ids}")

    video_out_path = pathlib.Path(args.video_out_path)
    video_out_path.mkdir(parents=True, exist_ok=True)
    results_out_path = pathlib.Path(args.results_out_path) if args.results_out_path else video_out_path / "results.jsonl"
    summary_out_path = pathlib.Path(args.summary_out_path) if args.summary_out_path else video_out_path / "summary.json"
    results_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Episode results: {results_out_path}")
    logging.info(f"Summary output: {summary_out_path}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    task_summaries = []
    with results_out_path.open("w", buffering=1) as results_file:
        for task_id in tqdm.tqdm(task_ids):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            task_episodes, task_successes = 0, 0
            try:
                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                    logging.info(f"\nTask: {task_description}")
                    init_state_id = episode_idx % len(initial_states)
                    done = False
                    reward = 0.0
                    info = {}
                    exception = None
                    t = 0
                    replay_images = []
                    executed_actions = []
                    policy_queries = 0

                    try:
                        env.reset()
                        action_plan = collections.deque()
                        obs = env.set_init_state(initial_states[init_state_id])

                        logging.info(f"Starting episode {task_episodes + 1}...")
                        while t < max_steps + args.num_steps_wait:
                            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                            # and we need to wait for them to fall.
                            if t < args.num_steps_wait:
                                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                                t += 1
                                continue

                            # IMPORTANT: rotate 180 degrees to match train preprocessing.
                            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                            )
                            wrist_img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                            )
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
                                    "prompt": str(task_description),
                                }

                                action_chunk = client.infer(element)["actions"]
                                policy_queries += 1
                                assert (
                                    len(action_chunk) >= args.replan_steps
                                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                                action_plan.extend(action_chunk[: args.replan_steps])

                            action = np.asarray(action_plan.popleft(), dtype=np.float32)
                            executed_actions.append(action)
                            obs, reward, done, info = env.step(action.tolist())
                            if done:
                                break
                            t += 1
                    except Exception as e:
                        exception = repr(e)
                        logging.exception("Caught exception during rollout")

                    success = bool(done)
                    if success:
                        task_successes += 1
                        total_successes += 1
                    task_episodes += 1
                    total_episodes += 1

                    suffix = "success" if success else "failure"
                    video_path = None
                    if args.save_per_episode_videos and replay_images:
                        video_name = (
                            f"task{task_id:02d}_ep{episode_idx:03d}_"
                            f"{_slugify(task_description)}_{suffix}.mp4"
                        )
                        video_path = video_out_path / video_name
                        imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

                    row = {
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "task_name": task.name,
                        "instruction": task_description,
                        "episode_index": episode_idx,
                        "init_state_id": init_state_id,
                        "success": success,
                        "steps": t,
                        "max_steps": max_steps,
                        "num_steps_wait": args.num_steps_wait,
                        "replan_steps": args.replan_steps,
                        "policy_queries": policy_queries,
                        "reward": reward,
                        "info": info,
                        "exception": exception,
                        "video_path": str(video_path) if video_path is not None else None,
                        "action_stats": _action_stats(executed_actions),
                        "seed": args.seed,
                    }
                    results_file.write(json.dumps(row, default=_json_default) + "\n")

                    logging.info(f"Success: {success}")
                    logging.info(f"# episodes completed so far: {total_episodes}")
                    logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

                task_summary = {
                    "suite": args.task_suite_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "instruction": task_description,
                    "episodes": task_episodes,
                    "successes": task_successes,
                    "failures": task_episodes - task_successes,
                    "success_rate": float(task_successes) / float(task_episodes) if task_episodes else 0.0,
                }
                task_summaries.append(task_summary)
                logging.info(
                    "Task successes: %d/%d",
                    task_summary["successes"],
                    task_summary["episodes"],
                )
                logging.info(f"Current task success rate: {task_summary['success_rate']}")
                logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
                _write_summary(
                    summary_out_path,
                    args=args,
                    total_episodes=total_episodes,
                    total_successes=total_successes,
                    task_summaries=task_summaries,
                    results_out_path=results_out_path,
                    video_out_path=video_out_path,
                )
            finally:
                _close_env(env)

    _write_summary(
        summary_out_path,
        args=args,
        total_episodes=total_episodes,
        total_successes=total_successes,
        task_summaries=task_summaries,
        results_out_path=results_out_path,
        video_out_path=video_out_path,
    )
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Episode results written to: {results_out_path}")
    logging.info(f"Summary written to: {summary_out_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


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


def _slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return str(value)


def _action_stats(actions):
    if not actions:
        return None
    arr = np.asarray(actions, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "gripper_min": float(arr[:, -1].min()),
        "gripper_max": float(arr[:, -1].max()),
    }


def _close_env(env):
    try:
        env.close()
    except Exception:
        logging.exception("Failed to close LIBERO environment cleanly")


def _write_summary(
    path,
    *,
    args,
    total_episodes,
    total_successes,
    task_summaries,
    results_out_path,
    video_out_path,
):
    payload = {
        "suite": args.task_suite_name,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_failures": total_episodes - total_successes,
        "total_success_rate": float(total_successes) / float(total_episodes) if total_episodes else 0.0,
        "num_trials_per_task": args.num_trials_per_task,
        "task_start": args.task_start,
        "num_tasks": args.num_tasks,
        "host": args.host,
        "port": args.port,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "num_steps_wait": args.num_steps_wait,
        "seed": args.seed,
        "results_out_path": str(results_out_path),
        "video_out_path": str(video_out_path),
        "tasks": task_summaries,
    }
    tmp_path = pathlib.Path(str(path) + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    tmp_path.replace(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
