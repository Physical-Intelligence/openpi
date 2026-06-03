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
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_trace_vla_config as _config
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.models.pi0_trace_vla_base import fourier_encode_2d, make_attn_mask, posemb_sincos

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Pi0TraceVLA model
# ---------------------------------------------------------------------------

class Pi0TraceVLA(_model.BaseModel):
    """Trace-augmented Vision-Language-Action model.

    See module docstring for an architectural summary.
    """

    def __init__(self, config: _config.Pi0TraceVLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        if not self.pi05:
            raise ValueError("Pi0TraceVLA assumes pi05=True (adaRMS pathway).")

        self.num_trace_experts = int(config.num_trace_experts)
        self.trace_horizon = int(config.trace_horizon)
        self.trace_dim = int(config.trace_dim)
        # When the semantic-target anchor row is appended, the trace stream
        # internally carries ``trace_horizon + 1`` tokens (the extra one is
        # inpainting-clamped to ``p_tgt`` and masked from the flow loss). The
        # user-facing trace shape returned by ``sample_trace`` remains
        # ``(B, trace_horizon, 2)``.
        self.append_target_anchor = bool(getattr(config, "append_target_anchor", False))
        self.trace_seq_len = self.trace_horizon + (1 if self.append_target_anchor else 0)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        trace_expert_config = _gemma_trace.get_trace_config(config.trace_expert_variant)
        if int(trace_expert_config.num_local_experts) != self.num_trace_experts:
            raise ValueError(
                f"trace_expert_variant has {trace_expert_config.num_local_experts} experts but "
                f"config requested {self.num_trace_experts}."
            )

        self.paligemma_width = int(paligemma_config.width)
        self.action_width = int(action_expert_config.width)
        self.trace_width = int(trace_expert_config.width)

        # ---------- Trunk: 3-stream Gemma with hard-routed trace MoE ----------
        llm = nnx_bridge.ToNNX(
            _gemma_trace.TraceModule(
                configs=[paligemma_config, action_expert_config, trace_expert_config],
                embed_dtype=config.dtype,
                # adaRMS is enabled (set per-stream below).
            )
        )
        # adaRMS on streams 1 (action) and 2 (trace).
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True, True])
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

        # ---------- Trace expert I/O ----------
        self.trace_in_proj = nnx.Linear(self.trace_dim, self.trace_width, rngs=rngs)
        self.trace_out_proj = nnx.Linear(self.trace_width, self.trace_dim, rngs=rngs)
        # Trace expert time MLP (separate parameters from the action expert's).
        self.trace_time_mlp_in = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)
        self.trace_time_mlp_out = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)

        # ---------- Semantic-target Fourier MLP for trace AdaRMS conditioning ----------
        target_fourier_dim = config.fourier_num_freqs * 2 * 2  # 2 coords * (sin+cos)
        self.target_mlp_in = nnx.Linear(target_fourier_dim, self.trace_width, rngs=rngs)
        self.target_mlp_out = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)

        # ---------- Completion head ----------
        # Shared compression from VLM hidden state width to a smaller dim.
        self.completion_shared_in = nnx.Linear(
            self.paligemma_width, config.completion_shared_dim, rngs=rngs
        )
        self.completion_shared_out = nnx.Linear(
            config.completion_shared_dim, config.completion_shared_dim, rngs=rngs
        )
        # Per-skill stacked MLP head: (K, shared_dim, hidden) -> (K, hidden, 1)
        K = self.num_trace_experts
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
        obs: _trace_obs.TraceObservation,
        images: dict,
        image_masks: dict,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Same shape/structure as pi0_atomic.embed_prefix, but takes the images dict as
        an explicit argument so we can run the prefix once with clean images (planning)
        and once with overlay images (execution).
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
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"],
    ]:
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_width, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(self.action_time_mlp_in(time_emb))
        time_emb = nnx.swish(self.action_time_mlp_out(time_emb))
        adarms_cond = time_emb

        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        # Action tokens form a single block (causal-from-prefix, mutually visible internally).
        ar_mask = jnp.broadcast_to(
            jnp.array([True] + ([False] * (self.action_horizon - 1))),
            action_tokens.shape[:2],
        )
        return action_tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def _embed_trace_suffix(
        self,
        noisy_trace: at.Float[at.Array, "b n 2"],
        timestep: at.Float[at.Array, " b"],
        target_xy: at.Float[at.Array, "b 2"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "b s"],
        at.Float[at.Array, "b emb"],
    ]:
        trace_tokens = self.trace_in_proj(noisy_trace)

        # Time embedding (Fourier sin/cos -> 2-layer MLP -> swish).
        time_emb = posemb_sincos(timestep, self.trace_width, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(self.trace_time_mlp_in(time_emb))
        time_emb = nnx.swish(self.trace_time_mlp_out(time_emb))

        # Semantic-target Fourier embedding -> 2-layer MLP -> swish.
        tgt_feat = fourier_encode_2d(target_xy, num_freqs=self.config.fourier_num_freqs)
        tgt_emb = nnx.swish(self.target_mlp_in(tgt_feat))
        tgt_emb = nnx.swish(self.target_mlp_out(tgt_emb))

        adarms_cond = time_emb + tgt_emb  # (B, trace_width)

        # Build masks dynamically from the actual trace-stream length (which may
        # differ from ``self.trace_horizon`` when the semantic-target anchor row
        # is appended; see ``trace_seq_len``).
        seq_len = trace_tokens.shape[1]
        input_mask = jnp.ones(trace_tokens.shape[:2], dtype=jnp.bool_)
        # ar_mask = [True, False, ..., False] of length seq_len: the trace tokens
        # form a single non-causal block (all mutually visible) after the prefix.
        row0 = jnp.ones((1,), dtype=jnp.bool_)
        rest = jnp.zeros((seq_len - 1,), dtype=jnp.bool_)
        single_ar = jnp.concatenate([row0, rest], axis=0)
        ar_mask = jnp.broadcast_to(single_ar, trace_tokens.shape[:2])
        return trace_tokens, input_mask, ar_mask, adarms_cond

    # -----------------------------------------------------------------------
    # Completion head
    # -----------------------------------------------------------------------
    @at.typecheck
    def _completion_predict(
        self, prefix_out: at.Float[at.Array, "b p d"], prefix_mask: at.Bool[at.Array, "b p"], skill_id: at.Int[at.Array, " b"]
    ) -> at.Float[at.Array, " b"]:
        # Mean-pool the VLM prefix output, weighted by mask.
        m = prefix_mask.astype(prefix_out.dtype)
        denom = jnp.maximum(jnp.sum(m, axis=-1, keepdims=True), 1.0)
        h_pool = jnp.sum(prefix_out * m[..., None], axis=1) / denom  # (B, paligemma_width)

        # Shared compression.
        h = nnx.swish(self.completion_shared_in(h_pool))
        h = self.completion_shared_out(h)  # (B, S)

        # Per-skill MLP heads (stacked weights). Compute outputs for ALL K experts then gather.
        # cmp_w1: (K, S, H), cmp_b1: (K, H)
        # h: (B, S)
        cmp_w1 = self.cmp_w1.value.astype(h.dtype)
        cmp_b1 = self.cmp_b1.value.astype(h.dtype)
        cmp_w2 = self.cmp_w2.value.astype(h.dtype)
        cmp_b2 = self.cmp_b2.value.astype(h.dtype)
        h1 = jnp.einsum("bs,ksh->bkh", h, cmp_w1) + cmp_b1[None, :, :]
        h1 = nnx.swish(h1)
        out = jnp.einsum("bkh,kho->bko", h1, cmp_w2) + cmp_b2[None, :, :]
        out = out[..., 0]  # (B, K)

        K = self.num_trace_experts
        skill_one_hot = jax.nn.one_hot(skill_id, K, dtype=out.dtype)  # (B, K)
        logit = jnp.einsum("bk,bk->b", skill_one_hot, out)
        return jax.nn.sigmoid(logit)

    # -----------------------------------------------------------------------
    # Sub-routines for the two forward passes
    # -----------------------------------------------------------------------
    @at.typecheck
    def _forward_planning(
        self,
        rng: at.KeyArrayLike,
        obs: _trace_obs.TraceObservation,
    ) -> tuple[at.Float[at.Array, "b n 2"], at.Float[at.Array, "b n 2"], at.Bool[at.Array, "b n"]]:
        """Run the planning forward pass and return (v_pred, u_target, mask).

        When ``append_target_anchor`` is True, the trace stream is extended by
        one extra row whose flow-matching target is ``p_tgt`` (the semantic
        target). That row is inpainting-clamped at every step and masked from
        the loss, mirroring the row-0 (current-EE) treatment. The returned
        tensors carry the *extended* sequence length so the caller's masking
        math stays straightforward.
        """
        noise_rng, time_rng = jax.random.split(rng, 2)

        future_trace = obs.future_trace_xy  # (B, N, 2)
        if future_trace is None:
            raise ValueError("future_trace_xy is required for trace flow-matching loss.")
        batch_shape = future_trace.shape[:-2]  # (B,)

        target_xy = obs.semantic_target_xy
        if target_xy is None:
            raise ValueError("semantic_target_xy is required for trace conditioning.")

        # When configured, append ``p_tgt`` as the extra (N+1)th row of the
        # supervised trace. This row is NOT a true trace point — it has a
        # different role (semantic-target anchor) — but flow matching's joint
        # distribution over the (N+1, 2) tokens is well-defined, and the loss
        # mask below excludes this row so the model is only supervised on the
        # true trace.
        if self.append_target_anchor:
            future_trace_ext = jnp.concatenate(
                [future_trace, target_xy[:, None, :]], axis=1
            )  # (B, N+1, 2)
        else:
            future_trace_ext = future_trace  # (B, N, 2)

        noise = jax.random.normal(noise_rng, future_trace_ext.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # (B,)
        time_e = time[..., None, None]  # (B, 1, 1)
        x_t = time_e * noise + (1.0 - time_e) * future_trace_ext
        u_t = noise - future_trace_ext

        # Inpainting clamp on row 0 (start-anchor at current EE pixel).
        ee = obs.current_ee_xy  # (B, 2)
        # row 0 of x_t = (1-t) * p_ee + t * eps_row_0  where eps_row_0 = noise[:, 0, :]
        # This matches the forward-process convention: same noise level as other rows,
        # but mean centered at p_ee instead of future_trace[:, 0].
        x_t_row0 = (1.0 - time[:, None]) * ee + time[:, None] * noise[:, 0, :]  # (B, 2)
        x_t = x_t.at[:, 0, :].set(x_t_row0)

        # Inpainting clamp on the appended last row (semantic-target anchor).
        # Same forward-process convention: noise level matches the rest of x_t,
        # mean centered at p_tgt. Because ``future_trace_ext[:, -1, :] == p_tgt``
        # by construction above, this clamp is also exactly consistent with the
        # supervision target (just like row 0 is consistent with the dataset's
        # ``future_trace[:, 0] == current_ee`` invariant).
        if self.append_target_anchor:
            x_t_row_last = (
                (1.0 - time[:, None]) * target_xy + time[:, None] * noise[:, -1, :]
            )  # (B, 2)
            x_t = x_t.at[:, -1, :].set(x_t_row_last)

        # Build prefix (use clean images for planning).
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            obs, obs.images, obs.image_masks
        )

        # Build trace suffix (target -> AdaRMS, time -> AdaRMS).
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_trace = self._embed_trace_suffix(
            x_t, time, target_xy
        )

        # Hard-routed combine_weights for the trace stream: skill -> one-hot,
        # broadcast across all trace tokens (including the anchor row if any).
        if obs.atomic_token is None:
            raise ValueError("atomic_token is required for hard MoE routing.")
        skill_id = obs.atomic_token.astype(jnp.int32)  # (B,)
        skill_one_hot = jax.nn.one_hot(skill_id, self.num_trace_experts)  # (B, K)
        combine_weights = jnp.broadcast_to(
            skill_one_hot[:, None, :],
            (skill_one_hot.shape[0], self.trace_seq_len, self.num_trace_experts),
        )

        # Joint forward over [paligemma, None, trace_suffix].
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

        # Take last trace_seq_len tokens of stream 2 -> velocity prediction.
        v_t = self.trace_out_proj(trace_out[:, -self.trace_seq_len :])
        # Mask out the inpainted rows: row 0 always; the appended last row when present.
        # Build the static mask of shape (1, trace_seq_len) then broadcast.
        L = self.trace_seq_len
        mask_first = jnp.zeros((1, 1), dtype=jnp.bool_)  # row 0: inpainted, excluded
        if self.append_target_anchor:
            mask_middle = jnp.ones((1, L - 2), dtype=jnp.bool_)  # supervised rows
            mask_last = jnp.zeros((1, 1), dtype=jnp.bool_)       # anchor row: inpainted, excluded
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
        """Run the execution forward pass and return (action v_pred, action u_target, progress_pred)."""
        noise_rng, time_rng = jax.random.split(rng, 2)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_e = time[..., None, None]
        x_a_t = time_e * noise + (1.0 - time_e) * actions
        u_a_t = noise - actions

        # Build images for execution: overlay base image + clean wrist images.
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

        # Build action suffix.
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_action = self._embed_action_suffix(x_a_t, time)

        # Trace stream is None for the execution forward pass.
        # Joint forward over [paligemma, action_suffix, None].
        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1

        # combine_weights is unused (trace stream is None) but must be a valid array to please typecheck.
        # Use a tiny zero placeholder. Dtype matches the model's embed dtype to satisfy typechecking through scan.
        dummy_weights = jnp.zeros(
            (prefix_mask.shape[0], 1, self.num_trace_experts), dtype=jnp.dtype(self.config.dtype)
        )

        (prefix_out, action_out, trace_out_unused), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond_action, None],
            hard_combine_weights=dummy_weights,
        )
        del trace_out_unused

        v_a_t = self.action_out_proj(action_out[:, -self.action_horizon :])

        # Completion head: mean-pool prefix_out (only over valid prefix tokens).
        skill_id = obs.atomic_token.astype(jnp.int32) if obs.atomic_token is not None else jnp.zeros(
            (prefix_mask.shape[0],), dtype=jnp.int32
        )
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
        dummy_weights = jnp.zeros((batch_size, 1, self.num_trace_experts), dtype=jnp.dtype(self.config.dtype))
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
            (batch_size, 1, self.num_trace_experts), dtype=jnp.dtype(self.config.dtype)
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
            (batch_size, 1, self.num_trace_experts), dtype=jnp.dtype(self.config.dtype)
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
        skill_one_hot = jax.nn.one_hot(skill_id, self.num_trace_experts)
        combine_weights = jnp.broadcast_to(
            skill_one_hot[:, None, :], (batch_size, L, self.num_trace_experts)
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
