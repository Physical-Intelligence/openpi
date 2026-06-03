"""Pi0TraceVLAMoe: trace-augmented VLA with MoE on BOTH the action head and the trace head.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: action expert (HardMoE FFN, K skill experts, no shared expert; full FT).
              Same shape as ``trace_vla_actionmoe``'s action MoE (default
              ``trace_moe_gemma_300m``: width=1024, mlp_dim=4096, depth=18).
  - Stream 2: trace  expert (HardMoE FFN, K skill experts, no shared expert; full FT).
              Shrunk along ``width``/``mlp_dim`` (default ``trace_moe_small``:
              width=512, mlp_dim=2048, depth=18) — Recipe C from
              ``traceVLA_moe_design.md``. Randomly initialized (cannot be
              warm-started from ``pi05_base`` because of the shape mismatch).

This is the combined mirror of ``Pi0TraceVLA`` and ``Pi0TraceVLAActionMoe``: same
dataflow, same conditioning (semantic-target Fourier+AdaRMS, EE inpainting clamp,
optional appended target-anchor row), same dataset, same training tricks
(anchor-age augmentation, scene/overlay dropout, trace perturbation, image
augmentation), and the same three losses (action, trace, completion). The only
architectural change is that both non-VLM streams now carry hard-routed MoE
FFNs and route the same skill-id one-hot.

Per-pass MoE activity (only one MoE stream is ever active at a time):

  - ``_forward_planning``  : trace stream is active   → trace MoE consumes
                              ``combine_weights`` of shape ``(B, trace_seq_len, K)``.
  - ``_forward_execution`` : action stream is active  → action MoE consumes
                              ``combine_weights`` of shape ``(B, action_horizon, K)``.
  - prefix prefill (sampling endpoints): neither expert stream has tokens, so
    a small placeholder ``combine_weights`` of shape ``(B, 1, K)`` is enough.
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
from openpi.models import pi0_trace_vla_moe_config as _config
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
from openpi.shared import array_typing as at
from openpi.models.pi0_trace_vla_base import TraceVLABase, make_attn_mask

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Pi0TraceVLAMoe model
# ---------------------------------------------------------------------------

class Pi0TraceVLAMoe(TraceVLABase):
    """Trace-augmented VLA with MoE on both the action and the trace heads.

    Inherits the trunk/head construction and the embed/completion methods from ``TraceVLABase``;
    overrides ``_build_expert_configs`` (both expert streams are MoE) and keeps the MoE-routing
    forward/sample methods below.
    """

    @override
    def _build_expert_configs(self, config):
        """Both the action and the trace stream are hard-routed MoE (``gemmoe_trace`` configs),
        routed by the *same* skill-id one-hot. (The base builds the identical trunk/head set; only
        the action-expert config source differs from TraceVLA.)

        Supported configuration: ``num_action_experts == num_trace_experts``. Because both streams
        share the same skill one-hot, we collapse them into a single ``num_skills``; the config's
        separate ``num_action_experts`` / ``num_trace_experts`` fields are expected to be equal. We
        don't assert that directly — instead the per-stream ``num_local_experts`` checks below
        require each expert config to carry exactly ``num_skills`` experts, so an unequal
        ``num_trace_experts`` is rejected there.
        """
        # Take the action-expert count as the shared skill count. The trace check below then
        # requires the trace MoE to carry the same number, so unequal action/trace counts are not
        # silently accepted.
        self.num_skills = int(config.num_action_experts)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Both expert streams are MoE; both come from gemmoe_trace's variants.
        action_expert_config = _gemma_trace.get_trace_config(config.action_expert_variant)
        trace_expert_config = _gemma_trace.get_trace_config(config.trace_expert_variant)

        if int(action_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"action_expert_variant has {action_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        if int(trace_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"trace_expert_variant has {trace_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        return paligemma_config, action_expert_config, trace_expert_config

    # -----------------------------------------------------------------------
    # Helpers: hard combine weights for the MoE streams.
    # -----------------------------------------------------------------------
    def _combine_weights(self, batch_size: int, skill_id: at.Int[at.Array, " b"], length: int) -> at.Float[at.Array, "b _t _k"]:
        """Build ``(B, length, K)`` one-hot combine weights for either MoE stream.

        Both MoE streams (action and trace) share the same K=5 skill atoms, so a
        single one-hot tensor (broadcast over the active stream's token axis) is
        all either MoE block needs.
        """
        skill_one_hot = jax.nn.one_hot(skill_id, self.num_skills)
        return jnp.broadcast_to(
            skill_one_hot[:, None, :], (batch_size, length, self.num_skills)
        )

    def _dummy_combine_weights(self, batch_size: int) -> at.Float[at.Array, "b _t _k"]:
        """Tiny placeholder combine_weights for calls where no MoE stream has tokens.

        Used when neither the action nor the trace expert stream is being denoised
        (e.g. prefix-only prefill).
        """
        return jnp.zeros((batch_size, 1, self.num_skills), dtype=jnp.dtype(self.config.dtype))

    # -----------------------------------------------------------------------
    # Forward passes
    # -----------------------------------------------------------------------
    @at.typecheck
    def _forward_planning(
        self,
        rng: at.KeyArrayLike,
        obs: _trace_obs.TraceObservation,
    ) -> tuple[at.Float[at.Array, "b n 2"], at.Float[at.Array, "b n 2"], at.Bool[at.Array, "b n"]]:
        """Trace planning forward. Stream 2 (trace MoE) is active.

        Returns ``(v_pred, u_target, loss_mask)`` over the *extended* trace sequence
        (length ``trace_seq_len``), with row 0 (and, when ``append_target_anchor``,
        row -1) masked out as they're inpainted clamps.
        """
        noise_rng, time_rng = jax.random.split(rng, 2)

        future_trace = obs.future_trace_xy
        if future_trace is None:
            raise ValueError("future_trace_xy is required for trace flow-matching loss.")
        batch_shape = future_trace.shape[:-2]

        target_xy = obs.semantic_target_xy
        if target_xy is None:
            raise ValueError("semantic_target_xy is required for trace conditioning.")

        if self.append_target_anchor:
            future_trace_ext = jnp.concatenate(
                [future_trace, target_xy[:, None, :]], axis=1
            )
        else:
            future_trace_ext = future_trace

        noise = jax.random.normal(noise_rng, future_trace_ext.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_e = time[..., None, None]
        x_t = time_e * noise + (1.0 - time_e) * future_trace_ext
        u_t = noise - future_trace_ext

        # Inpainting clamp on row 0 (current EE).
        ee = obs.current_ee_xy
        x_t_row0 = (1.0 - time[:, None]) * ee + time[:, None] * noise[:, 0, :]
        x_t = x_t.at[:, 0, :].set(x_t_row0)

        # Inpainting clamp on the appended last row (semantic-target anchor).
        if self.append_target_anchor:
            x_t_row_last = (
                (1.0 - time[:, None]) * target_xy + time[:, None] * noise[:, -1, :]
            )
            x_t = x_t.at[:, -1, :].set(x_t_row_last)

        # Build prefix with the clean image (no overlay) for planning.
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            obs, obs.images, obs.image_masks
        )

        # Trace suffix (target -> AdaRMS, time -> AdaRMS).
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_trace = self._embed_trace_suffix(
            x_t, time, target_xy
        )

        # Hard-routed combine weights for the trace MoE: skill -> one-hot,
        # broadcast over the trace token axis (including the anchor row if any).
        # The action stream is None during planning, so no separate weights for it.
        if obs.atomic_token is None:
            raise ValueError("atomic_token is required for hard MoE routing.")
        skill_id = obs.atomic_token.astype(jnp.int32)
        batch_size = prefix_mask.shape[0]
        combine_weights = self._combine_weights(batch_size, skill_id, self.trace_seq_len)

        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1

        (prefix_out, action_out_unused, trace_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None, adarms_cond_trace],
            hard_combine_weights=combine_weights,
        )
        del prefix_out, action_out_unused

        v_t = self.trace_out_proj(trace_out[:, -self.trace_seq_len :])

        # Loss mask: drop row 0 (inpainted EE) and, if present, the appended last row.
        L = self.trace_seq_len
        mask_first = jnp.zeros((1, 1), dtype=jnp.bool_)
        if self.append_target_anchor:
            mask_middle = jnp.ones((1, L - 2), dtype=jnp.bool_)
            mask_last = jnp.zeros((1, 1), dtype=jnp.bool_)
            loss_mask = jnp.concatenate([mask_first, mask_middle, mask_last], axis=1)
        else:
            mask_middle = jnp.ones((1, L - 1), dtype=jnp.bool_)
            loss_mask = jnp.concatenate([mask_first, mask_middle], axis=1)
        loss_mask = jnp.broadcast_to(loss_mask, v_t.shape[:2])
        return v_t, u_t, loss_mask

    @at.typecheck
    def _forward_execution(
        self,
        rng: at.KeyArrayLike,
        obs: _trace_obs.TraceObservation,
        actions: _model.Actions,
    ) -> tuple[
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, " b"],
    ]:
        """Action execution forward. Stream 1 (action MoE) is active."""
        noise_rng, time_rng = jax.random.split(rng, 2)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_e = time[..., None, None]
        x_a_t = time_e * noise + (1.0 - time_e) * actions
        u_a_t = noise - actions

        # Execution-mode images: overlay base image + clean wrist images.
        exec_images = dict(obs.images)
        exec_image_masks = dict(obs.image_masks)
        if getattr(obs, "overlay_images", None) is not None:
            for k, v in obs.overlay_images.items():
                exec_images[k] = v
                if obs.overlay_image_masks is not None and k in obs.overlay_image_masks:
                    exec_image_masks[k] = obs.overlay_image_masks[k]

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            obs, exec_images, exec_image_masks
        )

        # Action suffix (only time -> AdaRMS; no extra conditioning).
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_action = self._embed_action_suffix(
            x_a_t, time
        )

        # Hard MoE routing for the action stream. Trace stream is None, so its MoE
        # block is never invoked this pass; the same one-hot works for either
        # active stream (only one is active per pass).
        if obs.atomic_token is None:
            raise ValueError("atomic_token is required for hard MoE routing of the action expert.")
        skill_id = obs.atomic_token.astype(jnp.int32)
        batch_size = prefix_mask.shape[0]
        combine_weights = self._combine_weights(batch_size, skill_id, self.action_horizon)

        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1

        (prefix_out, action_out, trace_out_unused), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond_action, None],
            hard_combine_weights=combine_weights,
        )
        del trace_out_unused

        v_a_t = self.action_out_proj(action_out[:, -self.action_horizon :])

        # Completion head (pooled prefix hidden states).
        progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)

        return v_a_t, u_a_t, progress_pred

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _trace_obs.TraceObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        preprocess_rng, plan_rng, exec_rng = jax.random.split(rng, 3)
        # ``image_source_hw`` keeps train-time geometric augmentation of the image-space
        # trace/keypoint targets aligned with the letterboxed model image for non-square
        # cameras (e.g. table-tasks 480x640). It is None for square sources (LIBERO), in
        # which case preprocessing is unchanged. Only the train=True path uses it; the
        # inference (sample_*) paths do not augment keypoints, so they need not pass it.
        observation = _trace_obs.preprocess_trace_observation(
            preprocess_rng,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
            image_source_hw=self.config.image_source_hw,
        )

        # Trace planning (trace MoE stream).
        v_t, u_t, trace_loss_mask = self._forward_planning(plan_rng, observation)
        per_pt_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        has_trace = (observation.has_trace.astype(per_pt_loss.dtype)
                     if observation.has_trace is not None
                     else jnp.ones((per_pt_loss.shape[0],), dtype=per_pt_loss.dtype))
        denom = jnp.maximum(jnp.sum(trace_loss_mask.astype(per_pt_loss.dtype), axis=-1), 1.0)
        trace_loss_per_sample = jnp.sum(
            per_pt_loss * trace_loss_mask.astype(per_pt_loss.dtype), axis=-1
        ) / denom
        trace_loss_per_sample = trace_loss_per_sample * has_trace

        # Action execution (action MoE stream) + completion head.
        v_a, u_a, progress_pred = self._forward_execution(exec_rng, observation, actions)
        action_loss_per_sample = jnp.mean(jnp.mean(jnp.square(v_a - u_a), axis=-1), axis=1)

        progress_target = (observation.progress
                           if observation.progress is not None
                           else jnp.zeros_like(progress_pred))
        completion_loss_per_sample = jnp.square(progress_pred - progress_target) * has_trace

        total_loss = (
            self.config.action_loss_coeff * action_loss_per_sample
            + self.config.trace_loss_coeff * trace_loss_per_sample
            + self.config.completion_loss_coeff * completion_loss_per_sample
        )
        info = {
            "action_loss": action_loss_per_sample,
            "trace_loss": trace_loss_per_sample,
            "completion_loss": completion_loss_per_sample,
            "progress_pred_mean": jnp.mean(progress_pred),
        }
        return total_loss, info

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

        # Prefix prefill: neither expert stream has tokens, so the MoE blocks are
        # not invoked — a tiny placeholder combine_weights is enough.
        skill_id = observation.atomic_token.astype(jnp.int32)
        dummy_weights = self._dummy_combine_weights(batch_size)
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )

        # During the denoising loop the action MoE is active — need real one-hot
        # combine_weights of length ``action_horizon``.
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
        """Sample a trace from the trace MoE (planning mode).

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
        skill_id = observation.atomic_token.astype(jnp.int32)

        # Real combine_weights for the trace MoE (active during the denoising loop).
        # For the prefix prefill below the trace stream is None so a dummy works,
        # but we just reuse this real one — both shapes are valid for prefill
        # (no MoE block is invoked when its stream is None).
        trace_combine_weights = self._combine_weights(batch_size, skill_id, L)

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
            hard_combine_weights=trace_combine_weights,
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
                hard_combine_weights=trace_combine_weights,
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
