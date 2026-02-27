"""
Data loader for the pre-processed, globally shuffled, and flattened DROID dataset.
This replaces `DroidRldsDataset` for significantly faster training throughput.
"""

import glob
import logging
from typing import Optional
import tensorflow as tf

class FlatDroidDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # Local shuffle buffer: Since data is globally shuffled, a small buffer 
        # may be enough for inter-epoch randomness.
        shuffle_buffer_size: int = 10_000, 
        num_parallel_reads: int = tf.data.AUTOTUNE,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        **kwargs, # Accept other kwargs (like filter_dict_path) to maintain signature compatibility
    ):
        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.action_chunk_size = action_chunk_size

        # 1. 查找所有处理好的 TFRecord 文件
        tfrecord_files = sorted(glob.glob(f"{data_dir}/*.tfrecord"))
        if not tfrecord_files:
            raise ValueError(f"No .tfrecord files found in {data_dir}")
        logging.info(f"Found {len(tfrecord_files)} flat tfrecord files.")

        # 2. 构建基础 Dataset
        # 如果需要每次 epoch 读取顺序不同，可以 shuffle 文件列表
        file_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        if shuffle:
            file_dataset = file_dataset.shuffle(len(tfrecord_files))
            
        dataset = file_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, buffer_size=16 * 1024 * 1024),
            cycle_length=num_parallel_reads,
            num_parallel_calls=num_parallel_reads,
            deterministic=not shuffle
        )

        # 3. 定义解析结构 (Feature Spec)
        # 这必须与 Step 1 序列化时的结构严格对应
        self.feature_description = {
            "observation/image": tf.io.FixedLenFeature([], tf.string),
            "observation/wrist_image": tf.io.FixedLenFeature([], tf.string),
            "observation/joint_position": tf.io.FixedLenFeature([7], tf.float32),
            "observation/gripper_position": tf.io.FixedLenFeature([1], tf.float32),
            # Actions 在 step 1 中被 flatten 成了一维，长度为 chunk_size * 8
            "actions": tf.io.FixedLenFeature([self.action_chunk_size * 8], tf.float32),
            "prompt": tf.io.FixedLenFeature([], tf.string),
            "step_id": tf.io.FixedLenFeature([], tf.string),
        }

        # 4. 解析与解码函数
        def parse_and_decode(example_proto):
            # 解析二进制 Protobuf
            parsed = tf.io.parse_single_example(example_proto, self.feature_description)
            
            # 解码图片 (延迟到此处解码以优化存储和 IO)
            image = tf.io.decode_image(
                parsed["observation/image"], expand_animations=False, dtype=tf.uint8
            )
            wrist_image = tf.io.decode_image(
                parsed["observation/wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            
            # 将打平的 actions 恢复为 (chunk_size, 8) 的二维矩阵
            actions = tf.reshape(parsed["actions"], [self.action_chunk_size, 8])

            # 重组为 OpenPI 训练框架期望的字典结构
            return {
                "actions": actions,
                "observation": {
                    "image": image,
                    "wrist_image": wrist_image,
                    "joint_position": parsed["observation/joint_position"],
                    "gripper_position": parsed["observation/gripper_position"],
                },
                "prompt": parsed["prompt"],
                "step_id": parsed["step_id"],
            }

        dataset = dataset.map(parse_and_decode, num_parallel_calls=num_parallel_calls)

        # 5. 缓存、混洗与批处理
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        
        # 始终 repeat 保证数据流不断 (OpenPI 习惯逻辑)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        
        # 预取以隐藏 IO 延迟
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.dataset = dataset

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
