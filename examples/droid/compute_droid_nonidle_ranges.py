"""
Iterates through the DROID dataset and a json mapping from episode unique IDs to ranges of time steps
that should not be filtered out (all others are).

Specifically, we look for ranges of consecutive steps that contain at most min_idle_len consecutive idle frames
(default to 7 -- as most DROID action-chunking policies run the first 8 actions generated in each chunk, filtering
this way means the policy will not get stuck outputting stationary actions). Additionally, we also only keep non-idle
ranges of length at least min_non_idle_len (default to 16 frames = ~1 second), while also removing the last
filter_last_n_in_ranges frames from the end of each range (as those all correspond to action chunks with many idle actions).

This leaves us with trajectory segments consisting of contiguous, significant movement. Training on this filtered set
yields policies that output fewer stationary actions (i.e., get "stuck" in states less).
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set to the GPU you want to use, or leave empty for CPU

builder = tfds.builder_from_directory(
    # path to the `droid` directory (not its parent)
    builder_dir="<path_to_droid_dataset_tfds_files>",
)
ds = builder.as_dataset(split="train", shuffle_files=False)
tf.data.experimental.ignore_errors(ds)

keep_ranges_path = "<path_to_where_to_save_the_json>"

min_idle_len = 7  # If more than this number of consecutive idle frames, filter all of them out
min_non_idle_len = 16  # If fewer than this number of consecutive non-idle frames, filter all of them out
filter_last_n_in_ranges = 10  # When using a filter dict, remove this many frames from the end of each range

keep_ranges_map = {}
if Path(keep_ranges_path).exists():
    with Path(keep_ranges_path).open("r") as f:
        keep_ranges_map = json.load(f)
    print(f"Resuming from {len(keep_ranges_map)} episodes already processed")

for ep_idx, ep in enumerate(tqdm(ds)):
    recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
    file_path = ep["episode_metadata"]["file_path"].numpy().decode()

    key = f"{recording_folderpath}--{file_path}"
    if key in keep_ranges_map:
        continue

    joint_velocities = [step["action_dict"]["joint_velocity"].numpy() for step in ep["steps"]]
    joint_velocities = np.array(joint_velocities)

    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
    )

    # Find what steps go from idle to non-idle and vice-versa
    is_idle_padded = np.concatenate(
        [[False], is_idle_array, [False]]
    )  # Start and end with False, so idle at first step is a start of motion

    is_idle_diff = np.diff(is_idle_padded.astype(int))
    is_idle_true_starts = np.where(is_idle_diff == 1)[0]  # +1 transitions --> going from idle to non-idle
    is_idle_true_ends = np.where(is_idle_diff == -1)[0]  # -1 transitions --> going from non-idle to idle

    # Find which steps correspond to idle segments of length at least min_idle_len
    true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= min_idle_len
    is_idle_true_starts = is_idle_true_starts[true_segment_masks]
    is_idle_true_ends = is_idle_true_ends[true_segment_masks]

    keep_mask = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
        keep_mask[start:end] = False

    # Get all non-idle ranges of at least 16
    # Same logic as above, but for keep_mask, allowing us to filter out contiguous ranges of length < min_non_idle_len
    keep_padded = np.concatenate([[False], keep_mask, [False]])

    keep_diff = np.diff(keep_padded.astype(int))
    keep_true_starts = np.where(keep_diff == 1)[0]  # +1 transitions --> going from filter out to keep
    keep_true_ends = np.where(keep_diff == -1)[0]  # -1 transitions --> going from keep to filter out

    # Find which steps correspond to non-idle segments of length at least min_non_idle_len
    true_segment_masks = (keep_true_ends - keep_true_starts) >= min_non_idle_len
    keep_true_starts = keep_true_starts[true_segment_masks]
    keep_true_ends = keep_true_ends[true_segment_masks]

    # Add mapping from episode unique ID key to list of non-idle ranges to keep
    keep_ranges_map[key] = []
    for start, end in zip(keep_true_starts, keep_true_ends, strict=True):
        keep_ranges_map[key].append((int(start), int(end) - filter_last_n_in_ranges))

    if ep_idx % 1000 == 0:
        with Path(keep_ranges_path).open("w") as f:
            json.dump(keep_ranges_map, f)

print("Done!")
with Path(keep_ranges_path).open("w") as f:
    json.dump(keep_ranges_map, f)
