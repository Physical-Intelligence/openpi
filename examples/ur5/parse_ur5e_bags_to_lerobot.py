# Description: Script to parse UR5e robot rosbag2 files and convert them into LeRobotDataset format.
# The rosbags are created by manually teleoperating the UR5e robot in simulation for specific tasks.

import os
import datetime
import bisect
import math
import numpy as np
#import cv2
import pandas as pd
import logging
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, JointState
#from cv_bridge import CvBridge
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#_bridge = CvBridge()

ACTION_RATE = 15  # Hz
STATE_LENGTH = 7  # 6 joint positions + 1 gripper position + 1 reserved
WRIST_CAMERA_TOPIC = "/wrist_camera/zed_node/rgb/color/rect/image"
BASE_CAMERA_TOPIC = "/base_camera/zed_node/rgb/color/rect/image"
JOINT_STATES_TOPIC = "/joint_states"


def frame_is_valid(frame):
    return (
        frame["image"] is not None
        and frame["wrist_image"] is not None
        and frame["state"] is not None
        and len(frame["actions"]) >= 1
    )


def frame_passes_velocity_threshold(frame, joint_reordering, velocity_threshold, metrics=None):
    """Return True when any joint's absolute velocity meets or exceeds the threshold."""
    _, state_velocities = _parse_joint_components(frame["state"], joint_reordering)
    max_velocity = float(np.max(np.abs(state_velocities)))

    # Inspect queued actions as well; the first element becomes the stored action.
    for joint_state_msg, _ in frame["actions"]:
        _, action_velocities = _parse_joint_components(joint_state_msg, joint_reordering)
        max_velocity = max(max_velocity, float(np.max(np.abs(action_velocities))))

    if metrics is not None:
        metrics.setdefault("max_velocities", []).append(max_velocity)

    if velocity_threshold is None or velocity_threshold <= 0.0:
        return True, max_velocity

    return max_velocity >= velocity_threshold, max_velocity


def parse_image(msg, target_size=(224, 224)):
    """
    Deserialize sensor_msgs/Image and return a HxWx3 uint8 RGB numpy array resized to target_size.
    Supports common encodings: rgb8, bgr8, rgba8, bgra8, mono8.
    """
    msg = deserialize_message(msg, Image)

    # create 1D view of the raw bytes and reshape to (H, W, C) using step
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    # step is bytes per row; channels = step / width
    channels = int(msg.step // msg.width) if msg.width > 0 else 1
    try:
        arr = arr.reshape((msg.height, msg.width, channels))
    except Exception:
        # fallback: try (H, W) grayscale
        arr = arr.reshape((msg.height, msg.width))

    # normalize channel layout to RGB 3-channel uint8
    if msg.encoding == "bgr8":
        arr = arr[..., ::-1]
    elif msg.encoding == "bgra8":
        arr = arr[..., [2, 1, 0, 3]][..., :3]
    elif msg.encoding == "rgba8":
        arr = arr[..., :3]
    elif msg.encoding == "rgb8":
        arr = arr[..., :3]
    elif msg.encoding == "mono8":
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        # Unknown encoding: try to convert to RGB via PIL after treating buffer as bytes
        try:
            pil_try = PILImage.frombuffer("RGB", (msg.width, msg.height), msg.data, "raw", msg.encoding, msg.step, 1)
            pil = pil_try.convert("RGB").resize(target_size, resample=PILImage.BILINEAR)
            return np.asarray(pil)
        except Exception:
            raise RuntimeError(f"Unsupported/unknown image encoding: {msg.encoding}")

    # Use PIL for resizing (bilinear)
    pil = PILImage.fromarray(arr)
    pil = pil.convert("RGB").resize(target_size, resample=PILImage.BILINEAR)
    return np.asarray(pil)


def _parse_joint_components(msg, joint_reordering):
    joint_state = deserialize_message(msg, JointState)

    positions = np.asarray([joint_state.position[idx] for idx in joint_reordering], dtype=np.float32)

    velocities = np.zeros_like(positions)
    if joint_state.velocity:
        for i, idx in enumerate(joint_reordering):
            if idx < len(joint_state.velocity):
                value = joint_state.velocity[idx]
                if value is not None and not math.isnan(value):
                    velocities[i] = float(value)

    return positions, velocities

def iterate_bag(bag: SequentialReader):
    while bag.has_next():
        topic, msg, t = bag.read_next()
        yield topic, msg, t

def reorder_bag_timestamps(bag: SequentialReader, *, time_source: str = "record"):
    """Yield messages sorted by a chosen time source.

    Args:
        bag: Open rosbag2 SequentialReader
        time_source: 'record' to use bag record time (t), or 'header' to use
                     message header stamps when available.
    """
    if time_source not in ("record", "header"):
        raise ValueError("time_source must be 'record' or 'header'")

    if time_source == "record":
        # Use the bag record time directly; this avoids cross-topic clock skew.
        messages = list(iterate_bag(bag))
        messages.sort(key=lambda x: x[2])
        for topic, msg, t in messages:
            yield topic, msg, t
        return

    # Header-based ordering (legacy). This can fail if topics use different clocks.
    topic_types = {topic.name: topic.type for topic in bag.get_all_topics_and_types()}

    def _get_header_timestamp(topic, msg, t):
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(msg, msg_type)
        if hasattr(msg, "header"):
            return int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        return t

    unsorted_messages = [
        (topic, msg, _get_header_timestamp(topic, msg, t))
        for topic, msg, t in iterate_bag(bag)
    ]
    sorted_messages = sorted(unsorted_messages, key=lambda x: x[2])
    for topic, msg, t in sorted_messages:
        yield topic, msg, t


def process_frame(frame: dict, joint_reordering) -> dict:
    if joint_reordering is None:
        raise ValueError("Joint reordering must be computed before processing frames.")

    state_positions, _ = _parse_joint_components(frame["state"], joint_reordering)
    frame["state"] = state_positions

    actions = frame.pop("actions")
    next_joint_state, _ = actions[0]
    next_positions, _ = _parse_joint_components(next_joint_state, joint_reordering)

    # Store the absolute joint positions of the next state as the action.
    frame["action"] = next_positions

    return frame


def load_bag(
    bag_file_path: str,
    dataset: LeRobotDataset,
    task_description: str,
    *,
    time_source: str = "header",
    joint_velocity_threshold: float = 0.0,
    global_velocity_metrics=None,
):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_file_path, storage_id="mcap")
    converter_options = ConverterOptions()
    reader.open(storage_options, converter_options)

    frame = {
        "image": None,
        "wrist_image": None,
        "state": None,
        "task": task_description,
        "actions": [],
    }

    prev_state = None
    joint_reordering = None

    frame_count = 0

    first_timestamp = None
    last_timestamp = None

    # debug counters
    total_messages = 0
    image_msgs = 0
    wrist_msgs = 0
    joint_msgs = 0
    actions_appended = 0
    skipped_velocity_frames = 0
    velocity_metrics = {"max_velocities": []}

    def explain_frame_state(f):
        reasons = []
        if f["image"] is None:
            reasons.append("missing image")
        if f["wrist_image"] is None:
            reasons.append("missing wrist_image")
        if f["state"] is None:
            reasons.append("missing state")
        if len(f["actions"]) < 1:
            reasons.append("no actions")
        return ", ".join(reasons) if reasons else "valid"

    first_frame_timestamp = None
    last_frame_timestamp = None
    state_sum = None
    state_sq_sum = None
    action_sum = None
    action_sq_sum = None
    total_actions_buffered = 0

    for topic, msg, t in reorder_bag_timestamps(reader, time_source=time_source):
        total_messages += 1
        if first_timestamp is None:
            first_timestamp = t
        # keep track of the most recent timestamp
        last_timestamp = t

        # If the arrival of an image should trigger adding the previous frame, check validity first
        if frame_is_valid(frame) and (
            topic == WRIST_CAMERA_TOPIC
            or topic == BASE_CAMERA_TOPIC
        ):
            if joint_reordering is None:
                logger.warning(
                    "Skipping frame for bag %s because joint reordering is not yet available.",
                    bag_file_path,
                )
            else:
                velocity_ok, max_velocity = frame_passes_velocity_threshold(
                    frame,
                    joint_reordering,
                    joint_velocity_threshold,
                    metrics=velocity_metrics,
                )
                # If the frame passes the velocity gate, process and add it to the dataset
                if velocity_ok:
                    actions_in_frame = len(frame["actions"])
                    processed_frame = process_frame(frame, joint_reordering)
                    dataset.add_frame(processed_frame)
                    frame_count += 1
                    total_actions_buffered += actions_in_frame

                    if first_frame_timestamp is None:
                        first_frame_timestamp = t
                    last_frame_timestamp = t

                    # Compute statistics
                    state_vec = np.asarray(processed_frame["state"], dtype=np.float64)
                    action_vec = np.asarray(processed_frame["action"], dtype=np.float64)

                    if state_sum is None:
                        state_sum = np.zeros_like(state_vec, dtype=np.float64)
                        state_sq_sum = np.zeros_like(state_vec, dtype=np.float64)
                        action_sum = np.zeros_like(action_vec, dtype=np.float64)
                        action_sq_sum = np.zeros_like(action_vec, dtype=np.float64)
                    
                    # compute sums for mean/std
                    state_sum += state_vec
                    state_sq_sum += state_vec ** 2
                    action_sum += action_vec
                    action_sq_sum += action_vec ** 2
                else:
                    skipped_velocity_frames += 1
                    logger.debug(
                        "Skipping frame for bag %s due to low joint velocity (max=%.6f < threshold=%.6f)",
                        bag_file_path,
                        0.0 if max_velocity is None else max_velocity,
                        joint_velocity_threshold,
                    )

                # Start a new frame, even if velocity gating rejected the current one.
                frame = {
                    "image": None,
                    "wrist_image": None,
                    "state": None,
                    "task": task_description,
                    "actions": [],
                }

        if topic == WRIST_CAMERA_TOPIC:
            wrist_msgs += 1
            if frame["wrist_image"] is None:
                frame["wrist_image"] = parse_image(msg)
                logger.debug("Set wrist_image for current frame (bag=%s time=%s)", bag_file_path, t)
                if frame["state"] is None:
                    frame["state"] = prev_state
                    logger.debug("Initialized frame state from prev_state after wrist image (bag=%s)", bag_file_path)

        elif topic == BASE_CAMERA_TOPIC:
            image_msgs += 1
            if frame["image"] is None:
                frame["image"] = parse_image(msg)
                logger.debug("Set image for current frame (bag=%s time=%s)", bag_file_path, t)
                if frame["state"] is None:
                    frame["state"] = prev_state
                    logger.debug("Initialized frame state from prev_state after image (bag=%s)", bag_file_path)
        elif topic == JOINT_STATES_TOPIC:
            joint_msgs += 1
            if joint_reordering is None:
                prev_state = msg
                _msg = deserialize_message(msg, JointState)
                joint_reordering = [
                    _msg.name.index("ur5e_shoulder_pan_joint"),
                    _msg.name.index("ur5e_shoulder_lift_joint"),
                    _msg.name.index("ur5e_elbow_joint"),
                    _msg.name.index("ur5e_wrist_1_joint"),
                    _msg.name.index("ur5e_wrist_2_joint"),
                    _msg.name.index("ur5e_wrist_3_joint"),
                    _msg.name.index("robotiq_85_left_knuckle_joint"),
                ]
                logger.info("Computed joint_reordering=%s names=%s (bag=%s)", joint_reordering, _msg.name, bag_file_path)

            if frame["state"] is not None:
                frame["actions"].append((msg, t))
                actions_appended += 1
                logger.debug("Appended action (bag=%s time=%s). total actions for frame=%d", bag_file_path, t, len(frame["actions"]))
                prev_state = msg
            elif frame["image"] is not None and frame["wrist_image"] is not None:
                frame["state"] = msg
                logger.debug("Set frame state from joint_states (bag=%s time=%s)", bag_file_path, t)

        else:
            # print(f"Topic: {topic}, Time: {t}")
            pass

        # If this message is an image-trigger and frame wasn't valid, explain why we didn't add it
        if topic in (WRIST_CAMERA_TOPIC, BASE_CAMERA_TOPIC) and not frame_is_valid(frame):
            reason = explain_frame_state(frame)
            # logger.info(
            #     "Skipping add_frame at trigger topic=%s time=%s for bag=%s — reasons: %s",
            #     topic,
            #     t,
            #     bag_file_path,
            #     reason,
            # )

    # Only save an episode if we actually added frames
    if frame_count == 0:
        logger.warning(
            "No frames extracted from bag %s — summary: total_messages=%d image_msgs=%d wrist_msgs=%d joint_msgs=%d actions_appended=%d",
            bag_file_path,
            total_messages,
            image_msgs,
            wrist_msgs,
            joint_msgs,
            actions_appended,
        )
        return
    # compute all the statistics for the episode
    episode_duration_seconds = 0.0
    if first_frame_timestamp is not None and last_frame_timestamp is not None:
        episode_duration_seconds = (last_frame_timestamp - first_frame_timestamp) / 1_000_000_000
        episode_duration_seconds = max(episode_duration_seconds, 0.0)

    avg_fps = frame_count / episode_duration_seconds if episode_duration_seconds > 0 else None
    avg_fps_str = f"{avg_fps:.2f}" if avg_fps is not None else "N/A"
    avg_actions_per_frame = total_actions_buffered / frame_count if frame_count > 0 else 0.0

    state_mean = state_sum / frame_count if state_sum is not None else None
    state_std = None
    if state_mean is not None and state_sq_sum is not None:
        state_var = np.maximum(state_sq_sum / frame_count - state_mean ** 2, 0.0)
        state_std = np.sqrt(state_var)

    action_mean = action_sum / frame_count if action_sum is not None else None
    action_std = None
    if action_mean is not None and action_sq_sum is not None:
        action_var = np.maximum(action_sq_sum / frame_count - action_mean ** 2, 0.0)
        action_std = np.sqrt(action_var)

    state_mean_str = np.array2string(state_mean, precision=4, separator=", ") if state_mean is not None else "N/A"
    state_std_str = np.array2string(state_std, precision=4, separator=", ") if state_std is not None else "N/A"
    action_mean_str = np.array2string(action_mean, precision=4, separator=", ") if action_mean is not None else "N/A"
    action_std_str = np.array2string(action_std, precision=4, separator=", ") if action_std is not None else "N/A"

    start_frame_str = (
        datetime.datetime.fromtimestamp(first_frame_timestamp / 1_000_000_000).isoformat()
        if first_frame_timestamp is not None
        else "N/A"
    )
    end_frame_str = (
        datetime.datetime.fromtimestamp(last_frame_timestamp / 1_000_000_000).isoformat()
        if last_frame_timestamp is not None
        else "N/A"
    )

    logger.info(
        "\n==== Episode Summary ====\n"
        "Bag: %s\n"
        "Task: %s\n"
        "Frame window: %s → %s\n"
        "Frames: %d | Duration: %.2fs | Avg FPS: %s\n"
        "Actions appended: %d | Avg buffered actions/frame: %.2f\n"
        "Frames skipped by velocity gate: %d (threshold=%.6f)\n"
        "State mean/std: %s / %s\n"
        "Action mean/std: %s / %s\n"
        "Image msgs: %d | Wrist msgs: %d | Joint msgs: %d\n"
        "==========================",
        bag_file_path,
        task_description,
        start_frame_str,
        end_frame_str,
        frame_count,
        episode_duration_seconds,
        avg_fps_str,
        actions_appended,
        avg_actions_per_frame,
        skipped_velocity_frames,
        joint_velocity_threshold,
        state_mean_str,
        state_std_str,
        action_mean_str,
        action_std_str,
        image_msgs,
        wrist_msgs,
        joint_msgs,
    )

    dataset.save_episode()

    if velocity_metrics["max_velocities"]:
        velocities_arr = np.asarray(velocity_metrics["max_velocities"], dtype=np.float64)
        hist_counts, hist_edges = np.histogram(velocities_arr, bins=20)
        # output only 2 decimal places for readability
        logger.info(
            "Joint velocity histogram counts=%s edges=%s",
            hist_counts.tolist(),
            np.round(hist_edges, decimals=2).tolist(),
        )
        # percentiles = np.percentile(velocities_arr, [5, 50, 95])
        # logger.info(
        #     "Joint velocity percentiles (5/50/95) rad/s: %s",
        #     np.array2string(percentiles, precision=6, separator=", "),
        # )

        if global_velocity_metrics is not None:
            global_velocity_metrics.extend(velocity_metrics["max_velocities"])

    if first_timestamp is not None and last_timestamp is not None:
        # Convert ns->s using integer division to avoid float rounding on huge stamps.
        start_dt = datetime.datetime.fromtimestamp(int(first_timestamp) // 1_000_000_000)
        end_dt = datetime.datetime.fromtimestamp(int(last_timestamp) // 1_000_000_000)
        episode_time = end_dt - start_dt
        print(f"Time range: {episode_time}, frames: {frame_count}")
        # Warn if time range is suspiciously large (likely mixed clocks when using header time).
        if episode_time.days > 1_000 and time_source == "header":
            logger.warning(
                "Very large episode time range detected (%s). Consider --time-source=record to avoid mixed topic clocks.",
                episode_time,
            )
    else:
        print(f"Frames: {frame_count} (timestamps unavailable)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_bag_parser = subparsers.add_parser("single", help="Process a single bag")
    single_bag_parser.add_argument(
        "--bag_path",
        type=str,
        required=True,
        help="Path to the single bag file to process",
    )
    single_bag_parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="Task description for the single bag",
    )
    single_bag_parser.add_argument("--dataset_name", type=str, required=True)


    csv_parser = subparsers.add_parser("csv", help="Process all bags in a CSV")
    csv_parser.add_argument(
        "--tasks_csv",
        type=str,
        help="Path to the CSV file containing the list of bags to process",
    )
    csv_parser.add_argument(
        "--bags_dir", type=str, required=True, help="Directory containing the bag files"
    )
    csv_parser.add_argument("--dataset_name", type=str, required=True)
    csv_parser.add_argument(
        "--time_source",
        type=str,
        choices=["record", "header"],
        default="record",
        help="Use 'record' (bag record time) or 'header' (message header stamp) to order messages",
    )
    csv_parser.add_argument(
        "--joint_velocity_threshold",
        type=float,
        default=0.0,
        help="Minimum absolute joint velocity (rad/s) for any joint, required to keep a frame.",
    )

    # parse args after subparsers and args have been defined
    args = parser.parse_args()

    if args.mode == "single":
        bags = [args.bag_path]
        task_descriptions = [args.task_description]
    else:
        tasks = pd.read_csv(args.tasks_csv)
        bags = [
            os.path.join(args.bags_dir, row["filename"]) for _, row in tasks.iterrows()
        ]
        task_descriptions = [row["task_description"] for _, row in tasks.iterrows()]

    dataset = LeRobotDataset.create(
        args.dataset_name,
        fps=15,
        features={
            "image": {
                "dtype": "image",
                # "dtype": "video",
                "shape": (3, 224, 224),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            },
            "wrist_image": {
                "dtype": "image",
                # "dtype": "video",
                "shape": (3, 224, 224),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_LENGTH,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (STATE_LENGTH,),
                "names": ["action"],
            },
        },
        image_writer_threads=4,
    )

    all_velocity_samples = []

    for bag_file_path, task_description in zip(bags, task_descriptions):
        load_bag(
            bag_file_path,
            dataset,
            task_description,
            time_source=getattr(args, "time_source", "record"),
            joint_velocity_threshold=args.joint_velocity_threshold,
            global_velocity_metrics=all_velocity_samples,
        )

    if all_velocity_samples:
        velocities_arr = np.asarray(all_velocity_samples, dtype=np.float64)
        hist_counts, hist_edges = np.histogram(velocities_arr, bins=100)
        logger.info(
            "=============================================================================",
            "Aggregate joint velocity histogram counts=%s edges=%s (across %d frames)",
            hist_counts.tolist(),
            np.round(hist_edges, decimals=5).tolist(),
            len(all_velocity_samples),
        )
        
