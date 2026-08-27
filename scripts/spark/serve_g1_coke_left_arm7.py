#!/usr/bin/env python3
"""Serve one hash-pinned physical-left G1 arm7 checkpoint for simulation."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import socket
from typing import Any

POLICY_SERVER_SCHEMA = "wendy.g1.pi05.left-arm7-policy-server.v1"
POLICY_CONFIG_NAME = "pi05_spark_g1_coke_rgbd_left_arm7"
POLICY_PROMPT = "Grasp the Coke can, lift it, and present it in front of the robot."
MODEL_FILENAME = "model.safetensors"
STATE_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
ACTION_NAMES = STATE_NAMES[3:10]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint_dir: Path, expected_model_sha256: str) -> tuple[Path, str]:
    resolved = checkpoint_dir.expanduser().resolve()
    model_path = resolved / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    actual = sha256(model_path)
    if actual != expected_model_sha256:
        raise ValueError(
            f"checkpoint model hash mismatch: expected {expected_model_sha256}, got {actual}"
        )
    return resolved, actual


def build_metadata(base: dict[str, Any], checkpoint_dir: Path, model_sha256: str) -> dict[str, Any]:
    metadata = dict(base)
    metadata.update(
        {
            "schema": POLICY_SERVER_SCHEMA,
            "config": POLICY_CONFIG_NAME,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_model_sha256": model_sha256,
            "state_dim": len(STATE_NAMES),
            "action_dim": len(ACTION_NAMES),
            "action_horizon": 10,
            "state_names": list(STATE_NAMES),
            "action_names": list(ACTION_NAMES),
            "semantic_hand": "physical_left",
            "right_arm_included": False,
            "measured_hand_as_action": False,
            "depth_representation": "three-channel uint8 inverse depth",
            "prompt": POLICY_PROMPT,
            "execution_scope": "software_only_simulation",
        }
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir, model_sha256 = checkpoint_identity(
        args.checkpoint_dir,
        args.expected_model_sha256,
    )

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    policy = policy_config.create_trained_policy(
        training_config.get_config(POLICY_CONFIG_NAME),
        str(checkpoint_dir),
        default_prompt=POLICY_PROMPT,
    )
    metadata = build_metadata(policy.metadata, checkpoint_dir, model_sha256)
    logging.info(
        "Serving %s on %s:%d with model sha256 %s",
        POLICY_CONFIG_NAME,
        socket.gethostname(),
        args.port,
        model_sha256,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

