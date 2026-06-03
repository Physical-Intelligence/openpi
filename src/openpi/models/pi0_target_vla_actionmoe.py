"""Pi0TargetVLAActionMoe: trace-free ablation variant with MoE action head.

Two-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: action expert (HardMoE FFN, K skill experts, no shared expert; full FT).

The full design lives in ``target_vla_moe_implementation.md``. In short:

  1. There is **no trace stream** and **no trace generation**. The 3-stream
     ``TraceModule`` from ``gemmoe_trace`` is reused with only two ``Config``
     entries — its joint-attention / per-stream MoE plumbing supports an
     arbitrary number of streams.

  2. The semantic-target keypoint (the conditioning input that the trace
     family fed into the trace stream's AdaRMS) is instead injected into the
     **action stream's** AdaRMS:

        adarms_cond_action = time_emb + tgt_emb

     ``time_emb`` is the standard pi05 time pathway. ``tgt_emb`` reuses the
     Fourier(2-D, 8 freqs) -> 2-layer MLP encoder used by the trace family.

  3. There is **no image overlay**. Both forward passes of the trace family
     consumed an overlaid base image during execution mode; here the model
     only ever sees the clean RGB scene.

Losses per step (one forward pass):

  - Action flow-matching (MSE on velocity field, same as pi05 / TraceVLA).
  - Completion progress regression (per-skill MLP, same head shape as the
    TraceVLA family). Loss is masked by ``has_skill`` so frames without a
    valid skill / target annotation contribute zero completion loss.

There is **no trace loss**.
"""
from __future__ import annotations

import logging

import einops
import flax.linen as nn  # noqa: F401  (kept for parity with the original modules)
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_target_vla_actionmoe_config as _config
from openpi.models import target_observation as _target_obs
import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.models.pi0_trace_vla_base import fourier_encode_2d, make_attn_mask, posemb_sincos

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Pi0TargetVLAActionMoe model
# ---------------------------------------------------------------------------

class Pi0TargetVLAActionMoe(_model.BaseModel):
    """Trace-free ablation with MoE action head and semantic-target AdaRMS."""

    def __init__(self, config: _config.Pi0TargetVLAActionMoeConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        if not self.pi05:
            raise ValueError("Pi0TargetVLAActionMoe assumes pi05=True (adaRMS pathway).")

        self.num_action_experts = int(config.num_action_experts)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Action expert is the MoE; pulled from gemmoe_trace's variants.
        action_expert_config = _gemma_trace.get_trace_config(config.action_expert_variant)

        if int(action_expert_config.num_local_experts) != self.num_action_experts:
            raise ValueError(
                f"action_expert_variant has {action_expert_config.num_local_experts} experts but "
                f"config requested {self.num_action_experts}."
            )

        self.paligemma_width = int(paligemma_config.width)
        self.action_width = int(action_expert_config.width)

        # ---------- Trunk: 2-stream Gemma. ----------
        # ``TraceModule`` is generic in the number of streams; passing two
        # configs gives us a (paligemma, action_MoE) trunk. Stream 0 is
        # dense (paligemma); stream 1 dispatches to ``HardMoeBlock`` because
        # ``num_local_experts > 1``.
        llm = nnx_bridge.ToNNX(
            _gemma_trace.TraceModule(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        # adaRMS on stream 1 only (the action stream).
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ---------- Action expert I/O + time MLP ----------
        self.action_in_proj = nnx.Linear(config.action_dim, self.action_width, rngs=rngs)
        self.action_out_proj = nnx.Linear(self.action_width, config.action_dim, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(self.action_width, self.action_width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(self.action_width, self.action_width, rngs=rngs)

        # ---------- Semantic-target Fourier MLP for AdaRMS conditioning ----------
        # The MLP outputs ``action_width``-sized vectors so the sum
        # ``time_emb + tgt_emb`` is a well-typed AdaRMS conditioning signal for
        # the action stream. Same architectural pattern as
        # ``Pi0TraceVLA._embed_trace_suffix``, just hooked to the action stream.
        target_fourier_dim = config.fourier_num_freqs * 2 * 2  # 2 coords * (sin+cos)
        self.target_mlp_in = nnx.Linear(target_fourier_dim, self.action_width, rngs=rngs)
        self.target_mlp_out = nnx.Linear(self.action_width, self.action_width, rngs=rngs)

        # ---------- Completion head (per-skill MLP) ----------
        # Same head as TraceVLA / actionmoe / moe variants: mean-pooled VLM
        # prefix output -> shared compression -> per-skill MLP, hard-routed by
        # ``atomic_token``.
        self.completion_shared_in = nnx.Linear(
            self.paligemma_width, config.completion_shared_dim, rngs=rngs
        )
        self.completion_shared_out = nnx.Linear(
            config.completion_shared_dim, config.completion_shared_dim, rngs=rngs
        )
        K = self.num_action_experts
        S = int(config.completion_shared_dim)
        H = int(config.completion_per_skill_hidden)
        rng1, rng2 = jax.random.split(rngs.params(), 2)
        std_in = (1.0 / S) ** 0.5
        std_out = (1.0 / H) ** 0.5
        self.cmp_w1 = nnx.Param(jax.random.normal(rng1, (K, S, H)) * std_in)
        self.cmp_b1 = nnx.Param(jnp.zeros((K, H)))
        self.cmp_w2 = nnx.Param(jax.random.normal(rng2, (K, H, 1)) * std_out)
        self.cmp_b2 = nnx.Param(jnp.zeros((K, 1)))

        self.deterministic = True

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------
    @at.typecheck
    def _embed_prefix_with_images(
        self,
        obs: _target_obs.TargetObservation,
        images: dict,
        image_masks: dict,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Same structure as the trace family's ``_embed_prefix_with_images``.

        Takes an explicit ``images`` dict so this method has a uniform call
        site between the training forward and the sampling endpoints. There is
        no overlay version of the image — both training and inference call
        sites pass ``obs.images`` directly.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        for name in images:
            image_tokens, _ = self.PaliGemma.img(images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask.append(0 * input_mask[-1])

        assert obs.tokenized_prompt is not None
        assert obs.tokenized_prompt_mask is not None
        assert obs.token_ar_mask is not None

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(txt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def _embed_action_suffix(
        self,
        noisy_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
        target_xy: at.Float[at.Array, "b 2"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"],
    ]:
        """Embed the noisy-action sequence and build the AdaRMS conditioning.

        Compared with ``Pi0TraceVLAActionMoe._embed_action_suffix`` (which fed
        the action stream with **time only**), this variant adds the
        Fourier-encoded semantic target to ``adarms_cond``:

            adarms_cond_action = time_emb + tgt_emb

        This is the only place ``target_xy`` enters the model — it modulates
        every block of the action stream via AdaRMS, never as a separate
        token. The same conditioning math (Fourier + 2-layer MLP, summed with
        the time embedding) is used by the trace stream in
        ``Pi0TraceVLA._embed_trace_suffix``.
        """
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding (Fourier sin/cos -> 2-layer MLP -> swish).
        time_emb = posemb_sincos(timestep, self.action_width, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(self.action_time_mlp_in(time_emb))
        time_emb = nnx.swish(self.action_time_mlp_out(time_emb))

        # Semantic-target Fourier embedding -> 2-layer MLP -> swish.
        tgt_feat = fourier_encode_2d(target_xy, num_freqs=self.config.fourier_num_freqs)
        tgt_emb = nnx.swish(self.target_mlp_in(tgt_feat))
        tgt_emb = nnx.swish(self.target_mlp_out(tgt_emb))

        adarms_cond = time_emb + tgt_emb  # (B, action_width)

        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.broadcast_to(
            jnp.array([True] + ([False] * (self.action_horizon - 1))),
            action_tokens.shape[:2],
        )
        return action_tokens, input_mask, ar_mask, adarms_cond

    # -----------------------------------------------------------------------
    # Completion head
    # -----------------------------------------------------------------------
    @at.typecheck
    def _completion_predict(
        self, prefix_out: at.Float[at.Array, "b p d"], prefix_mask: at.Bool[at.Array, "b p"], skill_id: at.Int[at.Array, " b"]
    ) -> at.Float[at.Array, " b"]:
        """Per-skill completion head — identical math to the TraceVLA family."""
        m = prefix_mask.astype(prefix_out.dtype)
        denom = jnp.maximum(jnp.sum(m, axis=-1, keepdims=True), 1.0)
        h_pool = jnp.sum(prefix_out * m[..., None], axis=1) / denom

        h = nnx.swish(self.completion_shared_in(h_pool))
        h = self.completion_shared_out(h)

        cmp_w1 = self.cmp_w1.value.astype(h.dtype)
        cmp_b1 = self.cmp_b1.value.astype(h.dtype)
        cmp_w2 = self.cmp_w2.value.astype(h.dtype)
        cmp_b2 = self.cmp_b2.value.astype(h.dtype)
        h1 = jnp.einsum("bs,ksh->bkh", h, cmp_w1) + cmp_b1[None, :, :]
        h1 = nnx.swish(h1)
        out = jnp.einsum("bkh,kho->bko", h1, cmp_w2) + cmp_b2[None, :, :]
        out = out[..., 0]

        K = self.num_action_experts
        skill_one_hot = jax.nn.one_hot(skill_id, K, dtype=out.dtype)
        logit = jnp.einsum("bk,bk->b", skill_one_hot, out)
        return jax.nn.sigmoid(logit)

    # -----------------------------------------------------------------------
    # Helpers: hard combine weights for the action MoE.
    # -----------------------------------------------------------------------
    def _action_combine_weights(self, batch_size: int, skill_id: at.Int[at.Array, " b"], length: int) -> at.Float[at.Array, "b _t _k"]:
        """Build ``(B, length, K)`` one-hot combine weights for the action MoE.

        Used during the joint forward (length = ``action_horizon``) and during
        the denoising loop in ``sample_actions`` / ``sample_actions_and_completion``.
        """
        skill_one_hot = jax.nn.one_hot(skill_id, self.num_action_experts)
        return jnp.broadcast_to(
            skill_one_hot[:, None, :], (batch_size, length, self.num_action_experts)
        )

    def _dummy_combine_weights(self, batch_size: int) -> at.Float[at.Array, "b _t _k"]:
        """Tiny placeholder combine_weights for calls where the MoE stream has no tokens.

        Used during prefix-only prefill calls — the action stream is ``None``
        there, so the ``HardMoeBlock`` is never invoked, but the call still
        needs a well-shaped tensor to satisfy ``nn.scan``'s broadcast contract.
        """
        return jnp.zeros((batch_size, 1, self.num_action_experts), dtype=jnp.dtype(self.config.dtype))

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    @at.typecheck
    def _forward(
        self,
        rng: at.KeyArrayLike,
        obs: _target_obs.TargetObservation,
        actions: _model.Actions,
    ) -> tuple[
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, "b ah ad"],
        at.Float[at.Array, " b"],
    ]:
        """Single forward pass returning (v_pred, u_target, progress_pred).

        Stream 1 (action MoE) is active; semantic-target AdaRMS is on. There
        is no overlay image (none in this variant) and no trace stream.
        """
        noise_rng, time_rng = jax.random.split(rng, 2)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_e = time[..., None, None]
        x_a_t = time_e * noise + (1.0 - time_e) * actions
        u_a_t = noise - actions

        # Prefix: SigLIP(images) + tokenized skill prompt. No overlay — just the
        # clean RGB scene.
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            obs, obs.images, obs.image_masks
        )

        # Action suffix with target+time AdaRMS.
        target_xy = obs.semantic_target_xy
        if target_xy is None:
            raise ValueError("semantic_target_xy is required for AdaRMS conditioning of the action stream.")
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_action = self._embed_action_suffix(
            x_a_t, time, target_xy
        )

        # Hard MoE routing for the action stream: skill_id -> one-hot, broadcast
        # over action_horizon. No second non-VLM stream in this variant.
        if obs.atomic_token is None:
            raise ValueError("atomic_token is required for hard MoE routing of the action expert.")
        skill_id = obs.atomic_token.astype(jnp.int32)
        batch_size = prefix_mask.shape[0]
        combine_weights = self._action_combine_weights(batch_size, skill_id, self.action_horizon)

        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1

        (prefix_out, action_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond_action],
            hard_combine_weights=combine_weights,
        )

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
        observation: _target_obs.TargetObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, " b"], dict[str, at.Array]]:
        """Action flow-matching loss + completion loss. No trace loss."""
        preprocess_rng, fwd_rng = jax.random.split(rng, 2)
        observation = _target_obs.preprocess_target_observation(
            preprocess_rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        v_a, u_a, progress_pred = self._forward(fwd_rng, observation, actions)
        action_loss_per_sample = jnp.mean(jnp.mean(jnp.square(v_a - u_a), axis=-1), axis=1)

        # ``has_skill`` masks the completion loss when the frame has no valid
        # annotation. Defaults to all-ones if the field is missing.
        has_skill = (observation.has_skill.astype(action_loss_per_sample.dtype)
                     if observation.has_skill is not None
                     else jnp.ones_like(action_loss_per_sample))
        progress_target = (observation.progress
                           if observation.progress is not None
                           else jnp.zeros_like(progress_pred))
        completion_loss_per_sample = jnp.square(progress_pred - progress_target) * has_skill

        total_loss = (
            self.config.action_loss_coeff * action_loss_per_sample
            + self.config.completion_loss_coeff * completion_loss_per_sample
        )
        info = {
            "action_loss": action_loss_per_sample,
            "completion_loss": completion_loss_per_sample,
            "progress_pred_mean": jnp.mean(progress_pred),
        }
        return total_loss, info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _target_obs.TargetObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions from the action MoE.

        The caller must set ``observation.semantic_target_xy`` (and
        ``observation.atomic_token`` for MoE routing). No overlay image is
        needed — the model never trained with one.
        """
        observation = _target_obs.preprocess_target_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        target_xy = observation.semantic_target_xy
        if target_xy is None:
            raise ValueError("semantic_target_xy is required for AdaRMS conditioning of the action stream.")

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, observation.images, observation.image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Prefix prefill: action stream is None here (no tokens => MoE block is
        # never invoked), so a placeholder combine_weights is enough.
        skill_id = observation.atomic_token.astype(jnp.int32)
        dummy_weights = self._dummy_combine_weights(batch_size)
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
            hard_combine_weights=dummy_weights,
        )

        # During the denoising loop the action MoE is active.
        action_combine_weights = self._action_combine_weights(
            batch_size, skill_id, self.action_horizon
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_action_suffix(
                x_t, jnp.broadcast_to(time, (batch_size,)), target_xy
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, action_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_s,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
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
        observation: _target_obs.TargetObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Float[at.Array, " b"]]:
        """Action sampling + completion prediction in one shared prefix prefill.

        Mirrors :meth:`Pi0TraceVLAActionMoe.sample_actions_and_completion` —
        same shared-prefill pattern, just with one fewer stream and the
        target-aware AdaRMS in the action-stream denoising step.
        """
        observation = _target_obs.preprocess_target_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        target_xy = observation.semantic_target_xy
        if target_xy is None:
            raise ValueError("semantic_target_xy is required for AdaRMS conditioning of the action stream.")

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, observation.images, observation.image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        skill_id = observation.atomic_token.astype(jnp.int32)
        dummy_weights = self._dummy_combine_weights(batch_size)
        (prefix_out, _action_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out

        progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)

        action_combine_weights = self._action_combine_weights(
            batch_size, skill_id, self.action_horizon
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_action_suffix(
                x_t, jnp.broadcast_to(time, (batch_size,)), target_xy
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_p_out, action_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_s,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                hard_combine_weights=action_combine_weights,
            )
            del _p_out
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
        observation: _target_obs.TargetObservation,
    ) -> at.Float[at.Array, " b"]:
        """Standalone completion-progress query (no action sampling)."""
        del rng
        observation = _target_obs.preprocess_target_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, observation.images, observation.image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        dummy_weights = self._dummy_combine_weights(batch_size)

        (prefix_out, _action_out), _kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out, _kv_cache

        skill_id = observation.atomic_token.astype(jnp.int32)
        return self._completion_predict(prefix_out, prefix_mask, skill_id)
