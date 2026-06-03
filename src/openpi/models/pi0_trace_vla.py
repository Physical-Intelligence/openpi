"""Pi0TraceVLA: trace-augmented Vision-Language-Action model.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: gemma_300m action expert (dense FFN, optional LoRA, adaRMS time emb).
  - Stream 2: trace expert (HardMoE FFN, K skill experts, no shared expert; full FT).

The model trains three losses jointly per step:

  1. Trace flow-matching (planning forward pass: clean image, no trace overlay).
     - Targets `future_trace_xy` (normalized [0,1] pixel coords, (B, N, 2)).
     - Conditioning:
         · time embedding via adaRMS (existing pi05 pathway, on the trace stream).
         · Fourier-encoded semantic target point added to the same adaRMS cond.
         · Hard inpainting clamp on `x_t[:, 0, :]` to the current EE pixel.
     - Skill -> hard one-hot MoE routing (no learned router).

  2. Action flow-matching (execution forward pass: image already overlaid with the GT trace).
     - Targets `actions` (B, action_horizon, action_dim), like pi05/AtomicVLA.

  3. Completion progress regression (execution forward pass).
     - Mean-pool VLM prefix hidden states -> shared compression -> per-skill MLP -> sigmoid.
     - Target is `obs.progress` in [0, 1].

Both forward passes share the *same* prefix structure (images + language + state) but
differ in (a) which suffix stream is active and (b) whether the base image carries
the trace overlay.
"""
from __future__ import annotations

import logging

import einops
import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import trace_observation as _trace_obs
from openpi.shared import array_typing as at
from openpi.models.pi0_trace_vla_base import TraceVLABase, make_attn_mask

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Pi0TraceVLA model
# ---------------------------------------------------------------------------

class Pi0TraceVLA(TraceVLABase):
    """Trace-augmented Vision-Language-Action model.

    See module docstring for an architectural summary. Trunk + head construction lives in
    ``TraceVLABase.__init__``; this subclass adds the embed/forward/loss/sample logic.
    """

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
        observation = _trace_obs.preprocess_trace_observation(
            preprocess_rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # ---------------- Trace planning forward pass ----------------
        v_t, u_t, trace_loss_mask = self._forward_planning(plan_rng, observation)
        per_pt_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (B, N)
        # Ignore inpainted row 0; also mask by has_trace at the sample level.
        has_trace = (observation.has_trace.astype(per_pt_loss.dtype)
                     if observation.has_trace is not None
                     else jnp.ones((per_pt_loss.shape[0],), dtype=per_pt_loss.dtype))
        denom = jnp.maximum(jnp.sum(trace_loss_mask.astype(per_pt_loss.dtype), axis=-1), 1.0)
        trace_loss_per_sample = jnp.sum(
            per_pt_loss * trace_loss_mask.astype(per_pt_loss.dtype), axis=-1
        ) / denom
        trace_loss_per_sample = trace_loss_per_sample * has_trace

        # ---------------- Action + completion forward pass ----------------
        v_a, u_a, progress_pred = self._forward_execution(exec_rng, observation, actions)
        action_loss_per_sample = jnp.mean(jnp.mean(jnp.square(v_a - u_a), axis=-1), axis=1)  # (B,)

        progress_target = (observation.progress
                           if observation.progress is not None
                           else jnp.zeros_like(progress_pred))
        completion_loss_per_sample = jnp.square(progress_pred - progress_target) * has_trace

        # Combine.
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
        """Sample actions from the action expert (execution mode).

        Note: at inference time the user is responsible for setting `obs.overlay_images`
        with the overlay-rendered base image. We do NOT generate the trace here; that's
        done separately via `sample_trace`.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build images for execution mode (overlay base + clean wrist).
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
        dummy_weights = jnp.zeros((batch_size, 1, self.num_skills), dtype=jnp.dtype(self.config.dtype))
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
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
                hard_combine_weights=dummy_weights,
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
        """Sample actions and predict skill-completion progress in one shared forward.

        This is the recommended deployment path for the execution loop. Both action
        sampling and completion prediction consume the *same* execution-mode prefix
        (overlay base image + clean wrist images + skill prompt + state); running them
        together avoids redoing the SigLIP image encode and the Gemma prefill prefill
        — by far the most expensive parts of the forward pass — that would be
        duplicated if you called ``sample_actions`` and ``predict_completion``
        sequentially.

        At inference time the caller is responsible for setting ``obs.overlay_images``
        with the overlay-rendered base image (typically produced by a recent
        ``sample_trace`` call). The trace generator is *not* invoked here.

        Returns:
          ``actions``: ``(B, action_horizon, action_dim)`` float — the same chunk that
                        ``sample_actions`` would return.
          ``progress``: ``(B,)`` float in ``[0, 1]`` — predicted skill-completion progress.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Build images for execution mode (overlay base + clean wrist).
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
        dummy_weights = jnp.zeros(
            (batch_size, 1, self.num_skills), dtype=jnp.dtype(self.config.dtype)
        )

        # Note: sample_actions discards prefix_out; here we capture it for the
        # completion head. The kv_cache is reused inside the action denoising loop.
        (prefix_out, _action_out, _trace_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out, _trace_out

        # Completion head: pooled prefix hidden states -> per-skill MLP -> sigmoid.
        skill_id = (observation.atomic_token.astype(jnp.int32)
                    if observation.atomic_token is not None
                    else jnp.zeros((batch_size,), dtype=jnp.int32))
        progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)

        # Action denoising loop (identical to sample_actions, factored out for clarity).
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
                hard_combine_weights=dummy_weights,
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
        """Predict skill-completion progress only (no action sampling).

        Cheaper than ``sample_actions_and_completion`` when actions are not needed at
        this step — useful if the caller wants to query completion at a *lower*
        cadence than the action loop (e.g. once per K execution steps, between action
        chunks). Skips the action denoising loop entirely; only runs the SigLIP
        encode + Gemma prefill + completion head.

        ``rng`` is unused (kept for API symmetry with the other sampling methods).
        """
        del rng
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        batch_size = observation.state.shape[0]

        # Build the execution-mode prefix (overlay base + clean wrist).
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
        dummy_weights = jnp.zeros(
            (batch_size, 1, self.num_skills), dtype=jnp.dtype(self.config.dtype)
        )

        (prefix_out, _action_out, _trace_out), _kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out, _trace_out, _kv_cache

        skill_id = (observation.atomic_token.astype(jnp.int32)
                    if observation.atomic_token is not None
                    else jnp.zeros((batch_size,), dtype=jnp.int32))
        return self._completion_predict(prefix_out, prefix_mask, skill_id)

    def sample_trace(
        self,
        rng: at.KeyArrayLike,
        observation: _trace_obs.TraceObservation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b n 2"] | None = None,
    ) -> at.Float[at.Array, "b n 2"]:
        """Sample a trace from the trace expert (planning mode).

        At inference, ``observation.images`` should be the *clean* base image
        (no overlay), and ``semantic_target_xy``/``current_ee_xy`` must be provided.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # Noise lives on the *internal* trace sequence (includes the appended
        # semantic-target anchor row when enabled). The returned trace strips
        # the anchor before handing back to the caller.
        L = self.trace_seq_len
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, L, self.trace_dim))

        target_xy = observation.semantic_target_xy
        ee = observation.current_ee_xy
        skill_id = observation.atomic_token.astype(jnp.int32)
        skill_one_hot = jax.nn.one_hot(skill_id, self.num_skills)
        combine_weights = jnp.broadcast_to(
            skill_one_hot[:, None, :], (batch_size, L, self.num_skills)
        )

        # Prefill VLM with the clean (no-overlay) prefix.
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
            hard_combine_weights=combine_weights,
        )

        # Fix the row-0 noise sample (and, when present, the appended last-row
        # noise) to the corresponding entries of the initial noise tensor.
        # Training inpaints row 0 with ``(1-t)*ee + t*noise[:, 0, :]`` for a
        # single ``noise`` sample per pass. Reusing those eps values here puts
        # the inpainted rows on a continuous interpolation in t that matches
        # the training-time conditional distribution at every t. (Resampling
        # fresh noise per Euler step would jump the anchors around in noise
        # space and perturb the other waypoints via trace self-attention.)
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
                hard_combine_weights=combine_weights,
            )
            del prefix_out, _action_out
            v_t = self.trace_out_proj(trace_out[:, -L :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # Defensive final clamps: at t=0, the inpainting math collapses the
        # anchor rows exactly to ``ee`` and ``target_xy``; we re-assert that
        # explicitly to absorb any accumulated float drift.
        x_0 = x_0.at[:, 0, :].set(ee)
        if append_anchor:
            x_0 = x_0.at[:, -1, :].set(target_xy)
            # Strip the anchor row before returning: callers want the actual
            # (N, 2) trace polyline, not the (N+1, 2) extended sequence.
            x_0 = x_0[:, :self.trace_horizon, :]
        return x_0
