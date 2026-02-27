"""
Step 1 (Single Threaded): Scatter Phase for Global Shuffling DROID Dataset.
High correctness version: Uses dlimp.from_rlds strictly following original logic.
"""

import os
import json
import logging
import argparse
import shutil
import random
from collections import defaultdict
from enum import Enum
from enum import auto
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Import dlimp (Assumed to be installed in the environment)
import dlimp as dl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# [COPIED] Enum from droid_rlds_dataset.py
# =============================================================================
class DroidActionSpace(Enum):
    """Action space for DROID dataset."""
    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()

# =============================================================================
# Helper: Buffered Sharder (For writing buckets)
# =============================================================================
class BufferedSharder:
    def __init__(self, output_root, num_buckets=2048, buffer_size_mb=256):
        self.output_root = output_root
        self.buffers = defaultdict(list)
        self.buffer_sizes = defaultdict(int)
        self.total_buffer_size = 0
        self.buffer_threshold = buffer_size_mb * 1024 * 1024 
        self.flush_counts = defaultdict(int)

    def add(self, bucket_id, serialized_example):
        length = len(serialized_example)
        self.buffers[bucket_id].append(serialized_example)
        self.buffer_sizes[bucket_id] += length
        self.total_buffer_size += length
        if self.total_buffer_size >= self.buffer_threshold:
            self._flush_largest()

    def _flush_largest(self):
        if not self.buffer_sizes: return
        bucket_id = max(self.buffer_sizes, key=self.buffer_sizes.get)
        self._flush(bucket_id)

    def _flush(self, bucket_id):
        examples = self.buffers.pop(bucket_id)
        size = self.buffer_sizes.pop(bucket_id)
        self.total_buffer_size -= size
        if not examples: return
        
        bucket_dir = os.path.join(self.output_root, f"bucket_{bucket_id:05d}")
        os.makedirs(bucket_dir, exist_ok=True)
        # Worker ID is fixed to 0 for single thread
        filename = f"part_0_{self.flush_counts[bucket_id]}.tfrecord"
        filepath = os.path.join(bucket_dir, filename)
        
        with tf.io.TFRecordWriter(filepath) as writer:
            for ex in examples: writer.write(ex)
        self.flush_counts[bucket_id] += 1

    def flush_all(self):
        for bid in list(self.buffers.keys()): self._flush(bid)

# =============================================================================
# Helper: Serialization (FIXED)
# =============================================================================
def _bytes_feature(value):
    """Returns a bytes_list from a list of strings/bytes."""
    # Value is already a list of bytes/strings from _serialize_flat_frame
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a list of floats/doubles."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a list of bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _serialize_flat_frame(frame_data):
    """
    Serializes a flattened dictionary to TFRecord.
    Handles recursive flattening of nested dicts (like 'observation').
    """
    features = {}
    
    # Flatten logic: {'observation': {'image': ...}} -> 'observation/image'
    def flatten_dict(d, prefix=''):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            else:
                # Convert to numpy/list for serialization
                if isinstance(v, tf.Tensor): v = v.numpy()
                
                # Flatten arrays to 1D lists
                v = np.array(v).flatten()
                
                # Check dtype kind
                # f: float, i: int, u: uint, b: bool, S: string(bytes), U: unicode, O: object
                if v.dtype.kind in {'f'}:
                    features[key] = _float_feature(v)
                elif v.dtype.kind in {'i', 'u', 'b'}:
                    features[key] = _int64_feature(v)
                elif v.dtype.kind in {'S', 'U', 'O'}:
                    # Ensure all elements are bytes
                    bytes_list = []
                    for x in v:
                        if isinstance(x, bytes):
                            bytes_list.append(x)
                        elif isinstance(x, str):
                            bytes_list.append(x.encode('utf-8'))
                        elif isinstance(x, (np.bytes_, np.object_)): # Handle numpy bytes
                            bytes_list.append(bytes(x))
                        else:
                            # Fallback
                            bytes_list.append(str(x).encode('utf-8'))
                            
                    features[key] = _bytes_feature(bytes_list)

    flatten_dict(frame_data)
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()

# =============================================================================
# MAIN PIPELINE BUILDER (Adapted from DroidRldsDataset)
# =============================================================================
def build_dataset_pipeline(
    data_dir: str,
    filter_dict_path: str,
    action_space: DroidActionSpace,
    action_chunk_size: int,
):
    """
    Reconstructs the exact TF dataset pipeline from DroidRldsDataset,
    STOPPING before shuffle/batch/decode to allow for scattering.
    """
    # [COPIED] Configure Tensorflow
    tf.config.set_visible_devices([], "GPU")

    # [COPIED] Builder & From RLDS
    # Note: using split="train" as in original
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    
    # We turn off shuffle here to ensure deterministic sequential reading for stability
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False)

    # [COPIED] Filter Success
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
        )
    )

    # [REMOVED] dataset.repeat() -> We want to iterate exactly once.

    # [COPIED] Load Filter Dict
    if filter_dict_path is not None and os.path.exists(filter_dict_path):
        # Modification: Using standard open() instead of openpi.shared.download
        with open(filter_dict_path, "r") as f:
            filter_dict = json.load(f)

        logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

        keys_tensor = []
        values_tensor = []

        for episode_key, ranges in filter_dict.items():
            for start, end in ranges:
                for t in range(start, end):
                    frame_key = f"{episode_key}--{t}"
                    keys_tensor.append(frame_key)
                    values_tensor.append(True)
        
        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
        )
        logging.info("Filter hash table initialized")
    else:
        logging.warning("No filter dict provided or file not found! Using default True.")
        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
        )

    # [COPIED] restructure (traj_map)
    def restructure(traj):
        """Reformat observation and action keys, sample language instruction."""
        actions = tf.concat(
            (
                (
                    traj["action_dict"]["joint_position"]
                    if action_space == DroidActionSpace.JOINT_POSITION
                    else traj["action_dict"]["joint_velocity"]
                ),
                traj["action_dict"]["gripper_position"],
            ),
            axis=-1,
        )
        # Randomly samples one of the two exterior images
        exterior_img = tf.cond(
            tf.random.uniform(shape=[]) > 0.5,
            lambda: traj["observation"]["exterior_image_1_left"],
            lambda: traj["observation"]["exterior_image_2_left"],
        )
        wrist_img = traj["observation"]["wrist_image_left"]
        # Randomly sample one of the three language instructions
        instruction = tf.random.shuffle(
            [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
        )[0]

        # Use action len to determine trajectory length
        # Note: Original used traj["action"] but restructure creates "actions". 
        # Using the source key to be safe:
        traj_len = tf.shape(traj["action_dict"]["joint_position"])[0]
        indices = tf.as_string(tf.range(traj_len))

        step_id = (
            traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
            + "--"
            + traj["traj_metadata"]["episode_metadata"]["file_path"]
            + "--"
            + indices
        )
        passes_filter = filter_table.lookup(step_id)

        return {
            "actions": actions,
            "observation": {
                "image": exterior_img,
                "wrist_image": wrist_img,
                "joint_position": traj["observation"]["joint_position"],
                "gripper_position": traj["observation"]["gripper_position"],
            },
            "prompt": instruction,
            "step_id": step_id,
            "passes_filter": passes_filter,
        }

    dataset = dataset.traj_map(restructure)

    # [COPIED] chunk_actions (traj_map)
    def chunk_actions(traj):
        """Splits episode into action chunks."""
        traj_len = tf.shape(traj["actions"])[0]

        action_chunk_indices = tf.broadcast_to(
            tf.range(action_chunk_size)[None],
            [traj_len, action_chunk_size],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, action_chunk_size],
        )

        action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
        traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
        return traj

    dataset = dataset.traj_map(chunk_actions)

    # [COPIED] flatten
    dataset = dataset.flatten()

    # [COPIED] filter (from dict)
    def filter_from_dict(frame):
        return frame["passes_filter"]

    dataset = dataset.filter(filter_from_dict)

    # [COPIED] remove key
    def remove_passes_filter(frame):
        frame.pop("passes_filter")
        return frame

    dataset = dataset.map(remove_passes_filter)

    # [OMITTED] decode_images 
    # Reason: We want to store bytes in TFRecords to save space.
    
    # [OMITTED] shuffle / batch
    # Reason: We are doing manual scattering.

    return dataset

# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to RLDS Droid root")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save buckets")
    parser.add_argument("--filter_dict_path", type=str, required=True)
    parser.add_argument("--num_buckets", type=int, default=2048)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--action_space", type=str, default="JOINT_POSITION")
    args = parser.parse_args()

    # Clean Output
    if os.path.exists(args.output_dir):
        logging.info(f"Cleaning output dir: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse Action Space
    action_space_enum = getattr(DroidActionSpace, args.action_space)

    logging.info("Building Dataset Pipeline (Single Threaded)...")
    dataset = build_dataset_pipeline(
        data_dir=args.data_dir,
        filter_dict_path=args.filter_dict_path,
        action_space=action_space_enum,
        action_chunk_size=args.chunk_size
    )

    # Initialize Sharder
    sharder = BufferedSharder(args.output_dir, num_buckets=args.num_buckets)

    logging.info("Starting execution... this may take a while.")
    cnt = 0
    
    # Iterate using as_numpy_iterator for performance
    iterator = dataset.as_numpy_iterator()
    
    try:
        for frame in iterator:
            # Serialize
            serialized = _serialize_flat_frame(frame)
            
            # Scatter
            bucket_id = random.randint(0, args.num_buckets - 1)
            sharder.add(bucket_id, serialized)
            
            cnt += 1
            if cnt % 10000 == 0:
                # Optional: Force sync if needed, but buffering handles it
                logging.info(f"Processed {cnt} frames...")
                
    except KeyboardInterrupt:
        logging.warning("\nInterrupted! Flushing buffers...")
    except Exception as e:
        logging.error(f"\nError occurred: {e}")
        raise
    finally:
        sharder.flush_all()
        logging.info(f"Done. Total frames processed: {cnt}")

if __name__ == "__main__":
    main()
