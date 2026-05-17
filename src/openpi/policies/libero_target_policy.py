"""Input/output transforms for the TargetVLA family (trace-free ablation).

This mirrors ``libero_trace_policy`` but drops every trace-specific field. In
particular, the input transform does **not** forward the overlay image, the
``current_ee_xy`` keypoint, the ``future_trace_xy`` polyline, or the
``has_trace`` / ``has_overlay`` flags — because the model never consumes any
of those. The output transform is identical to ``LiberoTraceOutputs`` (trim
back to 7 action dims) so we just reuse it.
"""
from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoTargetInputs(transforms.DataTransformFn):
    """Pack a ``LiberoTargetDataset`` sample into the dict expected by the model.

    Mirrors :class:`openpi.policies.libero_trace_policy.LiberoTraceInputs` but
    drops every trace-related field. Forwarded fields:

      - ``state`` (8-dim float32)
      - ``image`` / ``image_mask`` dicts (base + wrist + zeros right_wrist)
      - ``actions`` (training only)
      - ``atomic_token`` (skill id)
      - ``semantic_target_xy`` (the conditioning signal)
      - ``has_skill`` / ``progress`` / ``diffusion_loss_mask``
      - ``prompt`` + ``skill_name`` / ``skill_text`` / ``plan_text`` /
        ``skill_step_num`` (consumed by ``TargetTokenizePrompt`` downstream)
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Actions (training only).
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Skill / target / completion fields.
        if "atomic_token" in data:
            inputs["atomic_token"] = np.asarray(data["atomic_token"], dtype=np.float32)
        if "semantic_target_xy" in data:
            inputs["semantic_target_xy"] = np.asarray(data["semantic_target_xy"], dtype=np.float32)
        if "has_skill" in data:
            inputs["has_skill"] = np.asarray(bool(data["has_skill"]))
        if "progress" in data:
            inputs["progress"] = np.asarray(data["progress"], dtype=np.float32)
        if "diffusion_loss_mask" in data:
            inputs["diffusion_loss_mask"] = np.asarray(bool(data["diffusion_loss_mask"]))

        # Prompt / skill text fields — consumed by the tokenizer below.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "skill_name" in data:
            inputs["skill_name"] = data["skill_name"]
        if "skill_text" in data:
            inputs["skill_text"] = data["skill_text"]
        if "plan_text" in data:
            inputs["plan_text"] = data["plan_text"]
        if "skill_step_num" in data:
            inputs["skill_step_num"] = np.asarray(data["skill_step_num"], dtype=np.int32)

        return inputs


@dataclasses.dataclass(frozen=True)
class TargetResizeImages(transforms.DataTransformFn):
    """Resize the ``image`` dict to the model resolution.

    This is the trace-free equivalent of
    :class:`openpi.policies.libero_trace_policy.TraceResizeImages`. There is
    no overlay-image dict to resize. We deliberately keep this as a separate
    transform (rather than reusing the stock ``transforms.ResizeImages``) so
    the import surface mirrors ``libero_trace_policy`` and so we can extend
    this later if the target_vla family grows additional image-shaped fields.
    """

    height: int
    width: int

    def __call__(self, data: dict) -> dict:
        # Lazy import to mirror ``TraceResizeImages`` and avoid pulling
        # openpi_client into module-level imports in the test setup.
        from openpi_client import image_tools  # noqa: PLC0415

        if "image" in data:
            data["image"] = {
                k: image_tools.resize_with_pad(v, self.height, self.width)
                for k, v in data["image"].items()
            }
        return data


@dataclasses.dataclass(frozen=True)
class LiberoTargetOutputs(transforms.DataTransformFn):
    """Trim padded action dimensions back to the dataset's 7-dim action.

    Identical math to :class:`openpi.policies.libero_trace_policy.LiberoTraceOutputs`.
    Kept as a separate class so we can change the output trim independently if
    the target_vla family ever needs to emit additional inference outputs.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            data["actions"] = np.asarray(data["actions"][..., :7])
        return data


@dataclasses.dataclass(frozen=True)
class TargetTokenizePrompt(transforms.DataTransformFn):
    """Tokenize the prompt for the TargetVLA family.

    Identical prompt-composition logic to ``libero_trace_policy.TraceTokenizePrompt``:

        Plan: <plan_text> Current step: <K>. <skill_text>

    with graceful fallback to ``skill_text`` -> ``skill_name`` -> raw prompt
    when pieces are missing. The skill is also routed to the action MoE via
    the ``atomic_token`` field set upstream by :class:`LiberoTargetInputs`.

    We **pop** the string-valued helper fields (``prompt``, ``skill_name``,
    ``skill_text``, ``plan_text``, ``skill_step_num``) before returning so
    the torch default collator does not try to stack variable-length strings.
    """

    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    @staticmethod
    def _to_str(x) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, np.ndarray):
            try:
                return str(x.item())
            except Exception:
                return str(x)
        try:
            return x.item() if hasattr(x, "item") else str(x)
        except Exception:
            return str(x)

    def __call__(self, data: dict) -> dict:
        prompt = self._to_str(data.pop("prompt", ""))
        skill_name = self._to_str(data.pop("skill_name", ""))
        skill_text = self._to_str(data.pop("skill_text", ""))
        plan_text = self._to_str(data.pop("plan_text", ""))
        _raw_step = data.pop("skill_step_num", 0)
        try:
            if isinstance(_raw_step, np.ndarray):
                skill_step_num = int(_raw_step.item())
            else:
                skill_step_num = int(_raw_step)
        except (TypeError, ValueError):
            skill_step_num = 0

        if skill_text and plan_text and skill_step_num > 0:
            full_prompt = f"Plan: {plan_text} Current step: {skill_step_num}. {skill_text}"
        elif skill_text and plan_text:
            full_prompt = f"Plan: {plan_text} Current step: {skill_text}"
        elif skill_text:
            full_prompt = skill_text
        elif skill_name:
            full_prompt = skill_name
        else:
            full_prompt = prompt

        state = None
        if self.discrete_state_input:
            state = data.get("state")

        tokens, token_mask = self.tokenizer.tokenize(full_prompt, state=state)
        # Trivial AR mask + loss mask (no text loss in the TargetVLA family).
        token_ar_mask = np.zeros_like(tokens, dtype=np.int32)
        token_loss_mask = np.zeros_like(tokens, dtype=np.bool_)

        return {
            **data,
            "tokenized_prompt": np.asarray(tokens, dtype=np.int32),
            "tokenized_prompt_mask": np.asarray(token_mask, dtype=np.bool_),
            "token_ar_mask": token_ar_mask,
            "token_loss_mask": token_loss_mask,
        }
