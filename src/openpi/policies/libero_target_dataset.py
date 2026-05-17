"""Dataset loader for LIBERO with skill segments + semantic target annotations.

The trace-free counterpart of ``LiberoTraceDataset``:

  - Keeps everything needed to condition the action expert on the per-skill
    semantic target (skill segments, atomic-token, ``semantic_target_xy``) and
    supervise the completion head (``progress``, ``has_skill``, skill-end action
    zeroing via ``pad_skill_horizon_actions``).

  - **Drops** every trace-related output: no ``current_ee_xy``,
    ``future_trace_xy``, overlay image, overlay-aware augmentations.

  - **Drops** every trace-related data augmentation: no anchor-age sampling, no
    scene dropout, no overlay dropout, no trace perturbation.

The skill plan / annotation join logic is identical to ``LiberoTraceDataset`` so
the resulting per-frame dict has matching ``atomic_token``, ``skill_name``,
``skill_text``, ``plan_text``, ``skill_step_num``, and ``semantic_target_xy``
fields. This means the same ``TraceTokenizePrompt`` family of tokenizers can be
reused with no modification (although we also provide a ``TargetTokenizePrompt``
mirror in ``libero_target_policy.py`` for symmetry with the trace pipeline).
"""
from __future__ import annotations

import json
import logging
import os
import re

import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.models import trace_utils
from openpi.policies.libero_reason_dataset import (
    _resolve_dataset_root,
    pad_skill_horizon_actions,
)


_SKILL_NAME_RE = re.compile(r"^\s*([A-Za-z_]+)\s*(?:\(.*\))?\s*$")


def _strip_skill_parameters(skill: str) -> str:
    match = _SKILL_NAME_RE.match(skill)
    return match.group(1).upper() if match else skill.strip().upper()


def _normalize_segment_end(end_step: int | float) -> int:
    return int(1e9) if end_step == -1 else int(end_step)


def _segment_index_for_step(segments: list[dict], step: int) -> int:
    for i, seg in enumerate(segments):
        if int(seg["start_step"]) <= step < _normalize_segment_end(seg["end_step"]):
            return i
    raise ValueError(f"No segment contains step {step}")


def _safe_load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _index_episodes(loaded: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for k, v in loaded.items():
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = v
        elif isinstance(k, int):
            out[k] = v
    return out


def _ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce image to (H, W, 3) uint8 layout (shared helper from ``libero_trace_dataset``)."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


class LiberoTargetDataset(LeRobotDataset):
    """LIBERO dataset with skill + semantic-target annotations (trace-free variant).

    Per-frame fetch returns:

        observation/image      : (H, W, 3) uint8 — clean RGB scene
        observation/wrist_image: (H, W, 3) uint8 — clean wrist scene
        observation/state      : (8,) float32   — 3 EE pos + 3 EE rot + 2 gripper
        actions                : (ah, 7) float32 — skill-end-zero-padded action chunk
        action_is_pad          : (ah,) bool
        atomic_token           : float scalar (skill id in {0..4})
        skill_name / skill_text/ plan_text / skill_step_num
        semantic_target_xy     : (2,) float32 in [0, 1]^2
        has_skill              : bool — True iff a valid skill + semantic target exists
        progress               : float in [0, 1]
        diffusion_loss_mask    : True
        prompt                 : str (raw LIBERO task instruction)
    """

    def __init__(self, data_config, action_horizon: int):
        # Resolve dataset root the same way the trace dataset does.
        root = _resolve_dataset_root(
            data_config.repo_id, data_config.skill_annotations_path
        )
        super().__init__(
            data_config.repo_id,
            root=root,
            revision="main",
        )
        print("Using LiberoTargetDataset (trace-free TargetVLA)")
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.action_down_sample_steps = int(getattr(data_config, "action_down_sample_steps", 1))
        self.use_wrist_image = bool(getattr(data_config, "use_wrist_image", True))
        self.is_computing_norm_stats = bool(getattr(data_config, "is_computing_norm_stats", False))

        # State and actions arrays.
        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_control"]
        self.low_dim_features: dict[str, np.ndarray] = {}
        states = torch.stack(self.hf_dataset["state"]).numpy().astype(np.float32)
        self.low_dim_features["eef_pos"] = states[:, :3]
        self.low_dim_features["eef_rot_axis_angle"] = states[:, 3:6]
        self.low_dim_features["gripper_control"] = states[:, 6:]
        self.actions = torch.stack(self.hf_dataset["actions"]).numpy().astype(np.float32)

        episode_indices = np.array(self.hf_dataset["episode_index"])
        unique_episodes = np.unique(episode_indices)
        episode_masks = episode_indices[:, None] == unique_episodes[None, :]
        episode_ends = np.where(episode_masks)[0][np.cumsum(episode_masks.sum(0)) - 1] + 1
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.episode_starts = episode_starts
        self.episode_ends = episode_ends

        # Load annotation files. The "trace_annotations" file is still loaded
        # because it carries the per-segment semantic-target points alongside
        # the EE traces — we just ignore the EE-trace polyline here.
        skill_path = os.path.expanduser(str(data_config.skill_annotations_path))
        trace_path = os.path.expanduser(str(data_config.trace_annotations_path))
        if not os.path.isfile(skill_path):
            raise FileNotFoundError(f"skill_annotations_path not found: {skill_path}")
        if not os.path.isfile(trace_path):
            raise FileNotFoundError(f"trace_annotations_path not found: {trace_path}")

        self.skills_by_episode = _index_episodes(_safe_load_json(skill_path))
        self.traces_top = _safe_load_json(trace_path)
        self.traces_by_episode = _index_episodes(self.traces_top)

        # Read the (constant) image w/h that the semantic-target pixel coords live in.
        first_ep = next(iter(self.traces_by_episode.values()))
        self.trace_image_w = int(first_ep.get("image_width", 256))
        self.trace_image_h = int(first_ep.get("image_height", 256))

        self.indices = list(range(len(self.hf_dataset)))
        # Kept for parity with ``LiberoTraceDataset.__init__`` (the trace dataset
        # consumes ``self.rdm`` for its anchor-age / dropout / perturbation draws;
        # we have none of those here, but holding an RNG keeps reproducible
        # behavior on incidental future additions).
        self.rdm = np.random.RandomState(int(getattr(data_config, "seed", 42)))

        logging.info(
            "[LiberoTargetDataset] %d frames across %d episodes; sem-target coord space %dx%d",
            len(self.indices),
            len(unique_episodes),
            self.trace_image_w,
            self.trace_image_h,
        )

    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Per-sample fetch
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        idx = self.indices[idx]
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        start_idx = int(self.episode_starts[ep_idx])
        end_idx = int(self.episode_ends[ep_idx])
        episode_step = idx - start_idx

        # Lookup skill segments + traces for this episode.
        ep_skill = self.skills_by_episode.get(ep_idx)
        ep_trace = self.traces_by_episode.get(ep_idx)
        segments = ep_skill["segments"] if ep_skill is not None else []

        # Default placeholders (in case annotations are missing for this frame).
        has_skill = False
        skill_name = ""
        skill_text = ""
        skill_id = 0
        plan_text = ""
        skill_step_num = 0
        sem_target_xy_norm = np.zeros((2,), dtype=np.float32)
        progress = 0.0

        # Identify the current skill segment.
        seg_idx = -1
        if segments:
            try:
                seg_idx = _segment_index_for_step(segments, episode_step)
            except ValueError:
                seg_idx = -1

        seg = None
        trace_seg = None
        if seg_idx >= 0:
            seg = segments[seg_idx]
            skill_raw = str(seg.get("skill", "")).strip()
            skill_text = skill_raw
            skill_name = _strip_skill_parameters(skill_raw)
            skill_id = trace_utils.skill_to_expert_id(skill_raw)
            if ep_skill is not None:
                _plan = ep_skill.get("plan", "")
                if isinstance(_plan, str):
                    plan_text = _plan.strip()
            skill_step_num = seg_idx + 1
            if ep_trace is not None:
                for ts in ep_trace.get("target_traces", []):
                    if int(ts.get("skill_index", -1)) == seg_idx:
                        trace_seg = ts
                        break

        # Segment bounds (used for progress + skill-end action zeroing).
        seg_start = int(seg["start_step"]) if seg is not None else 0
        seg_end_raw = int(seg["end_step"]) if seg is not None else episode_step + 1
        seg_end = seg_end_raw if seg_end_raw != -1 else end_idx - start_idx
        seg_end = min(seg_end, end_idx - start_idx)

        # Pull the semantic-target point + progress out of the matching trace segment.
        # We do NOT use the EE-trace polyline.
        if trace_seg is not None:
            sem = trace_seg.get("semantic_target", {})
            if sem.get("status") == "OK":
                sem_pt_pixel = np.asarray(sem.get("point", [0, 0]), dtype=np.float32)
                sem_target_xy_norm = np.array(
                    [
                        sem_pt_pixel[0] / max(self.trace_image_w - 1, 1),
                        sem_pt_pixel[1] / max(self.trace_image_h - 1, 1),
                    ],
                    dtype=np.float32,
                )
                # We require both a valid skill segment (seg_idx >= 0) AND a
                # valid semantic target to call this frame "has_skill". This
                # matches the role of ``has_trace`` in the trace dataset for
                # masking the completion loss.
                if seg_idx >= 0 and seg_end > seg_start:
                    seg_len = max(1, seg_end - seg_start)
                    progress = float(np.clip((episode_step - seg_start) / seg_len, 0.0, 1.0))
                    has_skill = True

        # ----- Image preparation (no overlay, no scene dropout) -----
        base_image = item["image"]
        if isinstance(base_image, torch.Tensor):
            base_image_np = base_image.numpy()
        else:
            base_image_np = np.asarray(base_image)
        base_image_np = _ensure_hwc_uint8(base_image_np)

        wrist_image_np = None
        if self.use_wrist_image:
            w = item.get("wrist_image", base_image)
            wrist_image_np = w.numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
            wrist_image_np = _ensure_hwc_uint8(wrist_image_np)

        # ----- Build state vector (8-dim, same as LiberoTraceDataset / LiberoSkillReasonDataset) -----
        state_vec = np.concatenate(
            [
                self.low_dim_features["eef_pos"][idx].flatten(),
                self.low_dim_features["eef_rot_axis_angle"][idx].flatten(),
                self.low_dim_features["gripper_control"][idx].flatten(),
            ],
            axis=-1,
        ).astype(np.float32)

        # ----- Action chunk with skill-end zero-padding (reused from libero_reason_dataset) -----
        seg_end_idx_global = start_idx + seg_end
        slice_end = min(
            seg_end_idx_global, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1
        )
        slice_end = max(slice_end, idx + 1)
        actions_chunk = self.actions[idx:slice_end:self.action_down_sample_steps]
        if actions_chunk.shape[0] == 0:
            actions_chunk = self.actions[idx:idx + 1]
        action_is_pad_count = self.action_horizon - actions_chunk.shape[0]
        action_is_pad = torch.tensor(
            [False] * actions_chunk.shape[0] + [True] * action_is_pad_count, dtype=torch.bool
        )
        final_actions = pad_skill_horizon_actions(actions_chunk, self.action_horizon)

        # ----- Build the return dict -----
        return_dict = {
            "observation/image": base_image_np,
            "observation/wrist_image": wrist_image_np if wrist_image_np is not None else base_image_np,
            "observation/state": torch.from_numpy(state_vec),
            "actions": torch.from_numpy(final_actions.astype(np.float32)),
            "action_is_pad": action_is_pad,
            "atomic_token": float(skill_id),
            "skill_name": skill_name,
            "skill_text": skill_text,
            "plan_text": plan_text,
            "skill_step_num": int(skill_step_num),
            "semantic_target_xy": sem_target_xy_norm.astype(np.float32),
            "has_skill": bool(has_skill),
            "progress": float(progress),
            "diffusion_loss_mask": True,
        }
        # Use the dataset task as the prompt (instruction). Default-empty if absent.
        if "task_index" in item:
            try:
                task_text = self.meta.tasks[int(item["task_index"].item())]
            except Exception:
                task_text = ""
            return_dict["prompt"] = task_text
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            return_dict[key] = item[key]
        return return_dict
