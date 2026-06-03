"""Pi0TraceVLAActionMoe: trace-augmented VLA with MoE action head + single trace head.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: action expert (HardMoE FFN, K skill experts, no shared expert; full FT).
  - Stream 2: trace expert (dense FFN, ``gemma_300m``-shape; full FT).

This is the "actionmoe" mirror of ``Pi0TraceVLA``: same dataflow, same conditioning,
same training tricks, same losses (action, trace, completion) — the only architectural
change is *which* of the two non-VLM streams carries the MoE FFN. The hard-routing
key is unchanged (skill id -> one-hot over K experts), and so is the per-skill
completion head.

Three losses per step:

  1. Trace flow-matching (planning forward pass: clean image, no trace overlay).
     - Targets ``future_trace_xy`` (normalized [0,1] pixel coords, (B, N, 2)).
     - Conditioning:
         · time embedding via adaRMS (existing pi05 pathway, on the trace stream).
         · Fourier-encoded semantic target point added to the same adaRMS cond.
         · Hard inpainting clamp on ``x_t[:, 0, :]`` to the current EE pixel.
         · When ``append_target_anchor`` is True, an extra row is appended whose
           value is inpainting-clamped to the semantic target point.
     - Stream 2 here is a *single dense FFN* (no MoE routing on the trace stream).

  2. Action flow-matching (execution forward pass: image already overlaid with the
     GT trace).
     - Targets ``actions`` (B, action_horizon, action_dim).
     - The action stream now runs through a hard-routed MoE: ``skill_id -> one-hot``
       broadcast over ``action_horizon`` tokens; every block routes every token to
       the chosen expert FFN. No conditioning beyond time (adaRMS).

  3. Completion progress regression (execution forward pass).
     - Same per-skill MLP head as TraceVLA, mean-pooled VLM prefix hidden state.
"""
from __future__ import annotations

import logging

import einops
import flax.linen as nn  # noqa: F401  (kept for parity with the original module)
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_trace_vla_actionmoe_config as _config
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
from openpi.shared import array_typing as at
from openpi.models.pi0_trace_vla_base import TraceVLABase, make_attn_mask

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Pi0TraceVLAActionMoe model
# ---------------------------------------------------------------------------

class Pi0TraceVLAActionMoe(TraceVLABase):
    """Trace-augmented VLA with an MoE action head and a single (dense) trace head.

    Inherits the trunk/head construction and embed/completion methods from ``TraceVLABase``;
    overrides ``_build_expert_configs`` (action MoE, dense trace) and keeps its action-MoE-routing
    forward/sample methods below.
    """

    @override
    def _build_expert_configs(self, config):
        """Action expert is the MoE (``gemmoe_trace`` config); the trace expert is a single dense
        FFN (``gemma`` config, ``num_local_experts=1``). Only the action stream is skill-routed, so
        ``num_skills = num_action_experts``; the base builds the (3-stream) trunk + heads.
        """
        self.num_skills = int(config.num_action_experts)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Action expert is the MoE; pulled from gemmoe_trace's variants.
        action_expert_config = _gemma_trace.get_trace_config(config.action_expert_variant)
        # Trace expert is a single dense FFN; pulled from gemmoe's variants.
        trace_expert_config = _gemma.get_config(config.trace_expert_variant)
        if int(action_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"action_expert_variant has {action_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        if int(getattr(trace_expert_config, "num_local_experts", 1)) > 1:
            raise ValueError(
                f"trace_expert_variant must be a single dense FFN (num_local_experts=1) for the "
                f"actionmoe variant; got num_local_experts={trace_expert_config.num_local_experts}."
            )
        return paligemma_config, action_expert_config, trace_expert_config

    # _forward_planning / _forward_execution are inherited from TraceVLABase.
    # For this variant trace_is_moe=False (dense trace stream -> dummy combine weights
    # during planning) and action_is_moe=True (real one-hot routing during execution);
    # the base selects each path from those flags.

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _trace_obs.TraceObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions from the action MoE (execution mode).

        The caller must set ``observation.overlay_images`` with the overlay-rendered
        base image (typically produced by a recent ``sample_trace`` call).
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build execution-mode images (overlay base + clean wrist).
        exec_images = dict(observation.images)
        exec_image_masks = dict(observation.image_masks)
        if getattr(observation, "overlay_images", None) is not None:
            for k, v in observation.overlay_images.items():
                exec_images[k] = v
                if observation.overlay_image_masks is not None and k in observation.overlay_image_masks:
                    exec_image_masks[k] = observation.overlay_image_masks[k]

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, exec_images, exec_image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Prefix prefill: action stream is None here (no tokens => no MoE call), so a
        # placeholder combine_weights is enough.
        skill_id = observation.atomic_token.astype(jnp.int32)
        dummy_weights = self._dummy_combine_weights(batch_size)
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )

        # The action stream's MoE is active during the denoising loop -> need the real
        # one-hot combine_weights of length ``action_horizon``.
        action_combine_weights = self._combine_weights(
            batch_size, skill_id, self.action_horizon
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_action_suffix(
                x_t, jnp.broadcast_to(time, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, action_out, _), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None],
                mask=full_attn_mask,
                positions=positions_s,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond, None],
                hard_combine_weights=action_combine_weights,
            )
            del prefix_out
            v_t = self.action_out_proj(action_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def sample_actions_and_completion(
        self,
        rng: at.KeyArrayLike,
        observation: _trace_obs.TraceObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Float[at.Array, " b"]]:
        """Sample actions and predict completion progress in one shared forward.

        Both heads share the execution-mode prefix (overlay image + clean wrist +
        skill prompt + state). Running them together avoids redoing the SigLIP image
        encode and Gemma prefill — the most expensive parts of the forward pass.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        exec_images = dict(observation.images)
        exec_image_masks = dict(observation.image_masks)
        if getattr(observation, "overlay_images", None) is not None:
            for k, v in observation.overlay_images.items():
                exec_images[k] = v
                if observation.overlay_image_masks is not None and k in observation.overlay_image_masks:
                    exec_image_masks[k] = observation.overlay_image_masks[k]

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, exec_images, exec_image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        skill_id = observation.atomic_token.astype(jnp.int32)
        dummy_weights = self._dummy_combine_weights(batch_size)
        (prefix_out, _action_out, _trace_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out, _trace_out

        progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)

        action_combine_weights = self._combine_weights(
            batch_size, skill_id, self.action_horizon
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_action_suffix(
                x_t, jnp.broadcast_to(time, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_p_out, action_out, _t_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None],
                mask=full_attn_mask,
                positions=positions_s,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond, None],
                hard_combine_weights=action_combine_weights,
            )
            del _p_out, _t_out
            v_t = self.action_out_proj(action_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, progress_pred

    @at.typecheck
    def predict_completion(
        self,
        rng: at.KeyArrayLike | None,
        observation: _trace_obs.TraceObservation,
    ) -> at.Float[at.Array, " b"]:
        """Predict skill-completion progress only (no action sampling)."""
        del rng
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        batch_size = observation.state.shape[0]

        exec_images = dict(observation.images)
        exec_image_masks = dict(observation.image_masks)
        if getattr(observation, "overlay_images", None) is not None:
            for k, v in observation.overlay_images.items():
                exec_images[k] = v
                if observation.overlay_image_masks is not None and k in observation.overlay_image_masks:
                    exec_image_masks[k] = observation.overlay_image_masks[k]

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, exec_images, exec_image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        dummy_weights = self._dummy_combine_weights(batch_size)

        (prefix_out, _action_out, _trace_out), _kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out, _trace_out, _kv_cache

        skill_id = observation.atomic_token.astype(jnp.int32)
        return self._completion_predict(prefix_out, prefix_mask, skill_id)

    def sample_trace(
        self,
        rng: at.KeyArrayLike,
        observation: _trace_obs.TraceObservation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b n 2"] | None = None,
    ) -> at.Float[at.Array, "b n 2"]:
        """Sample a trace from the (single dense) trace head (planning mode).

        Observation should carry the *clean* base image (no overlay) plus
        ``semantic_target_xy`` and ``current_ee_xy``.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        L = self.trace_seq_len
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, L, self.trace_dim))

        target_xy = observation.semantic_target_xy
        ee = observation.current_ee_xy

        # No MoE on the trace stream — pass a placeholder combine_weights.
        dummy_weights = self._dummy_combine_weights(batch_size)

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, observation.images, observation.image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )

        # Fix inpainting noise samples for both anchor rows (matches the training-time
        # conditional distribution at every t).
        fixed_eps_row0 = noise[:, 0, :]
        fixed_eps_row_last = noise[:, -1, :]

        append_anchor = self.append_target_anchor

        def step(carry):
            x_t, time = carry
            x_t_row0 = (1.0 - time) * ee + time * fixed_eps_row0
            x_t = x_t.at[:, 0, :].set(x_t_row0)
            if append_anchor:
                x_t_row_last = (1.0 - time) * target_xy + time * fixed_eps_row_last
                x_t = x_t.at[:, -1, :].set(x_t_row_last)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_trace_suffix(
                x_t, jnp.broadcast_to(time, (batch_size,)), target_xy
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, _action_out, trace_out), _ = self.PaliGemma.llm(
                [None, None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_s,
                kv_cache=kv_cache,
                adarms_cond=[None, None, adarms_cond],
                hard_combine_weights=dummy_weights,
            )
            del prefix_out, _action_out
            v_t = self.trace_out_proj(trace_out[:, -L :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # Final clamps to absorb any float drift.
        x_0 = x_0.at[:, 0, :].set(ee)
        if append_anchor:
            x_0 = x_0.at[:, -1, :].set(target_xy)
            x_0 = x_0[:, :self.trace_horizon, :]
        return x_0
