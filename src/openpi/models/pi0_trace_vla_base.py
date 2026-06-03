"""Shared base for the TraceVLA model family.

Home of the module-level helpers and (in later batches) ``TraceVLABase``, the shared trunk/head
construction and embed/loss/sample machinery for:

  - Pi0TraceVLA          (trace stream = MoE, action stream = dense)
  - Pi0TraceVLAMoe       (both streams MoE)
  - Pi0TraceVLAActionMoe (action stream = MoE, trace stream = dense)
  - Pi0TargetVLAActionMoe (no trace stream; target conditions the action MoE)

These helpers were previously copy-pasted (identically, modulo docstrings) into each of the four
model files; they are pure functions and carry no parameters, so this move does not touch the
flax param tree.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_trace_vla_config import Pi0TraceVLAConfig


def make_attn_mask(input_mask, mask_ar):
    """Build a (B, S, S) attention mask from a token mask and an autoregressive mask.

    ``mask_ar`` marks tokens that may only attend to earlier tokens; runs of equal cumulative
    value form mutually-visible blocks. (Same construction as ``pi0_atomic.make_attn_mask``.)
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Sin/cos positional embedding for scalar positions (matches pi0/pi05)."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def fourier_encode_2d(p: at.Float[at.Array, "*b 2"], num_freqs: int) -> at.Float[at.Array, "*b feat"]:
    """Map 2-D points (already in [0, 1]^2) to a Fourier feature vector.

    Returns ``(*b, 2 * 2 * num_freqs)`` features: for each of the 2 coords, sin and cos at
    ``num_freqs`` geometric frequencies. The trace/target configs all reuse this exact form so the
    AdaRMS conditioning signal matches across the family in both spectrum and scale.
    """
    freqs = 2.0 ** jnp.arange(num_freqs, dtype=jnp.float32)  # (num_freqs,)
    angles = p[..., None] * freqs * 2.0 * jnp.pi  # (..., 2, num_freqs)
    sins = jnp.sin(angles)
    coss = jnp.cos(angles)
    feat = jnp.concatenate([sins, coss], axis=-1)  # (..., 2, 2*num_freqs)
    new_shape = (*feat.shape[:-2], feat.shape[-2] * feat.shape[-1])
    return feat.reshape(new_shape)


class TraceVLABase(_model.BaseModel):
    """Shared base for the TraceVLA model family.

    Builds the PaliGemma trunk + the action/trace/target/completion heads, and provides the shared
    embed + completion methods. Variants subclass it and override:
      - ``_build_expert_configs`` — which streams are dense vs MoE, and ``self.num_skills`` (return
        ``trace_expert_config=None`` for a trace-free 2-stream variant).
      - the MoE-routing ``_forward_*`` / ``sample_*`` methods (genuinely variant-specific).

    A ``has_trace_stream`` flag (``trace_expert_config is not None``) gates the trace fields, trace
    heads, the 3rd trunk stream + adaRMS, and whether ``target_mlp`` conditions the trace or action
    stream — so the 3-stream variants (TraceVLA / Moe / ActionMoe) and the 2-stream target variant
    share one ``__init__``.

    IMPORTANT: the submodule-construction order here is load-bearing — it fixes the order in which
    ``rngs`` is consumed, hence the random initialization of every head not loaded from pi05_base.
    Reordering changes those weights (and the loss). The attribute names are equally load-bearing:
    they are the flax param-tree paths the weight loaders and checkpoints match on. The variant hook
    is rng-free, so overriding it never perturbs the init.
    """

    def __init__(self, config: "Pi0TraceVLAConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        if not self.pi05:
            raise ValueError("Pi0TraceVLA assumes pi05=True (adaRMS pathway).")

        # Variant hook: resolve the per-stream gemma configs and set self.num_skills. rng-free, so
        # overriding it does not perturb the random-init order of the heads built below. Trace-free
        # variants (target) return trace_expert_config=None.
        paligemma_config, action_expert_config, trace_expert_config = self._build_expert_configs(config)
        self.has_trace_stream = trace_expert_config is not None

        if self.has_trace_stream:
            self.trace_horizon = int(config.trace_horizon)
            self.trace_dim = int(config.trace_dim)
            # When the semantic-target anchor row is appended, the trace stream internally carries
            # ``trace_horizon + 1`` tokens (the extra one is inpainting-clamped to ``p_tgt`` and
            # masked from the flow loss). sample_trace still returns ``(B, trace_horizon, 2)``.
            self.append_target_anchor = bool(getattr(config, "append_target_anchor", False))
            self.trace_seq_len = self.trace_horizon + (1 if self.append_target_anchor else 0)

        self.paligemma_width = int(paligemma_config.width)
        self.action_width = int(action_expert_config.width)
        if self.has_trace_stream:
            self.trace_width = int(trace_expert_config.width)
        # The Fourier semantic-target MLP conditions the trace stream when present, else the action
        # stream (target variant, which has no trace stream).
        cond_width = self.trace_width if self.has_trace_stream else self.action_width

        # Whether each non-VLM stream is a hard-routed MoE (num_local_experts > 1) vs a dense FFN.
        # The forward passes use this to pass real one-hot combine weights to a MoE stream and a
        # dummy placeholder to a dense one (the dense FFN ignores combine weights).
        self.action_is_moe = int(getattr(action_expert_config, "num_local_experts", 1)) > 1
        self.trace_is_moe = self.has_trace_stream and int(getattr(trace_expert_config, "num_local_experts", 1)) > 1

        # ---------- Trunk: PaliGemma VLM + action expert (+ trace expert when present) ----------
        # adaRMS is enabled on every non-VLM stream: the action stream, and the trace stream when
        # the variant has one.
        trunk_configs = [paligemma_config, action_expert_config]
        use_adarms = [False, True]
        if self.has_trace_stream:
            trunk_configs.append(trace_expert_config)
            use_adarms.append(True)
        llm = nnx_bridge.ToNNX(
            _gemma_trace.TraceModule(configs=trunk_configs, embed_dtype=config.dtype)
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
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

        # ---------- Trace expert I/O (only when a trace stream exists) ----------
        if self.has_trace_stream:
            self.trace_in_proj = nnx.Linear(self.trace_dim, self.trace_width, rngs=rngs)
            self.trace_out_proj = nnx.Linear(self.trace_width, self.trace_dim, rngs=rngs)
            # Trace expert time MLP (separate parameters from the action expert's).
            self.trace_time_mlp_in = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)
            self.trace_time_mlp_out = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)

        # ---------- Semantic-target Fourier MLP for AdaRMS conditioning ----------
        target_fourier_dim = config.fourier_num_freqs * 2 * 2  # 2 coords * (sin+cos)
        self.target_mlp_in = nnx.Linear(target_fourier_dim, cond_width, rngs=rngs)
        self.target_mlp_out = nnx.Linear(cond_width, cond_width, rngs=rngs)

        # ---------- Completion head ----------
        # Shared compression from VLM hidden state width to a smaller dim.
        self.completion_shared_in = nnx.Linear(
            self.paligemma_width, config.completion_shared_dim, rngs=rngs
        )
        self.completion_shared_out = nnx.Linear(
            config.completion_shared_dim, config.completion_shared_dim, rngs=rngs
        )
        # Per-skill stacked MLP head: (K, shared_dim, hidden) -> (K, hidden, 1).
        K = self.num_skills
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
    # API to override, to define the architecture.
    # -----------------------------------------------------------------------
    def _build_expert_configs(self, config):
        """Resolve the gemma stream configs and set ``self.num_skills``. Override per variant.

        Returns ``(paligemma_config, action_expert_config, trace_expert_config)``. A trace-free
        variant (target) returns ``trace_expert_config=None``, which drives the ``has_trace_stream``
        branches in ``__init__``.

        Base = Pi0TraceVLA: dense action expert (``gemma`` config) + hard-routed trace MoE
        (``gemmoe_trace`` config). rng-free, so overriding it does not change the random init.
        """
        # One skill count for the whole model: routing is a hard one-hot over atomic skills (there
        # is no learned router), so the action MoE, the trace MoE, and the completion head all carry
        # ``num_skills`` experts. NOTE for future variants: if you ever need *different* numbers of
        # action vs trace experts, split this back into per-stream counts.
        self.num_skills = int(config.num_trace_experts)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        trace_expert_config = _gemma_trace.get_trace_config(config.trace_expert_variant)
        if int(trace_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"trace_expert_variant has {trace_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        return paligemma_config, action_expert_config, trace_expert_config

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------
    @at.typecheck
    def _embed_prefix_with_images(
        self,
        obs: _model.Observation,
        images: dict,
        image_masks: dict,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Embed the prefix (images + tokenized prompt). Takes the images dict explicitly so the
        prefix can be run with clean images (planning) or overlay images (execution). Annotated on
        the common ``Observation`` base so both Trace and Target variants share it.
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

        skill_one_hot = jax.nn.one_hot(skill_id, self.num_skills, dtype=out.dtype)  # (B, num_skills)
        logit = jnp.einsum("bk,bk->b", skill_one_hot, out)
        return jax.nn.sigmoid(logit)

    # -----------------------------------------------------------------------
    # Hard-routed MoE combine weights
    # -----------------------------------------------------------------------
    @at.typecheck
    def _combine_weights(
        self, batch_size: int, skill_id: at.Int[at.Array, " b"], length: int
    ) -> at.Float[at.Array, "b _t _k"]:
        """Hard one-hot routing weights for a skill-routed MoE stream: the skill one-hot broadcast
        over the token axis -> ``(B, length, num_skills)``."""
        skill_one_hot = jax.nn.one_hot(skill_id, self.num_skills)
        return jnp.broadcast_to(skill_one_hot[:, None, :], (batch_size, length, self.num_skills))

    @at.typecheck
    def _dummy_combine_weights(self, batch_size: int) -> at.Float[at.Array, "b _t _k"]:
        """Placeholder combine weights for a pass whose active stream is dense (its hard-MoE block
        is never invoked); shaped/typed to satisfy the typecheck through scan."""
        return jnp.zeros((batch_size, 1, self.num_skills), dtype=jnp.dtype(self.config.dtype))

    # -----------------------------------------------------------------------
    # Forward passes (planning = trace stream active; execution = action stream active)
    # -----------------------------------------------------------------------
    @at.typecheck
    def _forward_planning(
        self,
        rng: at.KeyArrayLike,
        obs: _trace_obs.TraceObservation,
    ) -> tuple[at.Float[at.Array, "b n 2"], at.Float[at.Array, "b n 2"], at.Bool[at.Array, "b n"]]:
        """Trace planning forward (stream 2 active, action stream None). Returns
        ``(v_pred, u_target, loss_mask)`` over the *extended* trace sequence (length
        ``trace_seq_len``); row 0 (current-EE inpaint) and, when ``append_target_anchor``, the
        appended last row (semantic-target inpaint) are masked out of the loss.
        """
        if obs.atomic_token is None:
            raise ValueError("atomic_token is required for hard MoE routing.")
        noise_rng, time_rng = jax.random.split(rng, 2)

        future_trace = obs.future_trace_xy  # (B, N, 2)
        if future_trace is None:
            raise ValueError("future_trace_xy is required for trace flow-matching loss.")
        batch_shape = future_trace.shape[:-2]  # (B,)

        target_xy = obs.semantic_target_xy
        if target_xy is None:
            raise ValueError("semantic_target_xy is required for trace conditioning.")

        # When configured, append ``p_tgt`` as the extra (N+1)th supervised row. flow matching's
        # joint distribution over the (N+1, 2) tokens is well-defined; the loss mask below excludes
        # this row so only the true trace is supervised.
        if self.append_target_anchor:
            future_trace_ext = jnp.concatenate([future_trace, target_xy[:, None, :]], axis=1)
        else:
            future_trace_ext = future_trace

        noise = jax.random.normal(noise_rng, future_trace_ext.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001  # (B,)
        time_e = time[..., None, None]  # (B, 1, 1)
        x_t = time_e * noise + (1.0 - time_e) * future_trace_ext
        u_t = noise - future_trace_ext

        # Inpainting clamp on row 0 (current EE): same noise level as the other rows, mean centered
        # at p_ee instead of future_trace[:, 0].
        ee = obs.current_ee_xy  # (B, 2)
        x_t_row0 = (1.0 - time[:, None]) * ee + time[:, None] * noise[:, 0, :]
        x_t = x_t.at[:, 0, :].set(x_t_row0)
        # Inpainting clamp on the appended last row (semantic target), consistent with the
        # supervision target since ``future_trace_ext[:, -1, :] == p_tgt``.
        if self.append_target_anchor:
            x_t_row_last = (1.0 - time[:, None]) * target_xy + time[:, None] * noise[:, -1, :]
            x_t = x_t.at[:, -1, :].set(x_t_row_last)

        # Prefix with the clean image (no overlay) for planning; trace suffix (time + target adaRMS).
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            obs, obs.images, obs.image_masks
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_trace = self._embed_trace_suffix(
            x_t, time, target_xy
        )

        # Real one-hot combine weights when the trace stream is a MoE, else a dummy placeholder.
        batch_size = prefix_mask.shape[0]
        if self.trace_is_moe:
            skill_id = obs.atomic_token.astype(jnp.int32)
            combine_weights = self._combine_weights(batch_size, skill_id, self.trace_seq_len)
        else:
            combine_weights = self._dummy_combine_weights(batch_size)

        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1

        # Joint forward over [paligemma, None, trace_suffix].
        (prefix_out, action_out_unused, trace_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None, adarms_cond_trace],
            hard_combine_weights=combine_weights,
        )
        del prefix_out, action_out_unused

        v_t = self.trace_out_proj(trace_out[:, -self.trace_seq_len :])
        # Loss mask: drop row 0 always; drop the appended last row when present.
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
        """Action execution forward (stream 1 active, trace stream None). Returns
        ``(action v_pred, action u_target, progress_pred)``."""
        if obs.atomic_token is None:
            raise ValueError("atomic_token is required for hard MoE routing of the action expert.")

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
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond_action = self._embed_action_suffix(x_a_t, time)

        # skill id: needed for the completion head always, and for action-stream MoE routing.
        batch_size = prefix_mask.shape[0]
        if self.action_is_moe:
            skill_id = obs.atomic_token.astype(jnp.int32)
        else:
            skill_id = jnp.zeros((batch_size,), dtype=jnp.int32)
        # Real one-hot combine weights when the action stream is a MoE, else a dummy placeholder.
        if self.action_is_moe:
            combine_weights = self._combine_weights(batch_size, skill_id, self.action_horizon)
        else:
            combine_weights = self._dummy_combine_weights(batch_size)

        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1

        # Joint forward over [paligemma, action_suffix, None].
        (prefix_out, action_out, trace_out_unused), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond_action, None],
            hard_combine_weights=combine_weights,
        )
        del trace_out_unused

        v_a_t = self.action_out_proj(action_out[:, -self.action_horizon :])
        progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)
        return v_a_t, u_a_t, progress_pred


    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Public sampling / inference API
    #
    # All four entrypoints share the execution/planning embed + prefill machinery and
    # differ only in which stream carries the hard-routed MoE — exactly the axis the
    # forward passes fold over. The action denoising loop uses real one-hot combine
    # weights when ``action_is_moe`` (else a placeholder); ``sample_trace`` uses real
    # weights when ``trace_is_moe``. A prefix-only prefill never invokes a MoE block, so
    # it always passes the placeholder.
    # -----------------------------------------------------------------------
    def _exec_images(self, observation: _trace_obs.TraceObservation) -> tuple[dict, dict]:
        """Execution-mode image set: overlay-rendered base image + clean wrist images."""
        exec_images = dict(observation.images)
        exec_image_masks = dict(observation.image_masks)
        if getattr(observation, "overlay_images", None) is not None:
            for k, v in observation.overlay_images.items():
                exec_images[k] = v
                if observation.overlay_image_masks is not None and k in observation.overlay_image_masks:
                    exec_image_masks[k] = observation.overlay_image_masks[k]
        return exec_images, exec_image_masks

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _trace_obs.TraceObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample an action chunk from the action expert (execution mode).

        The caller must set ``observation.overlay_images`` with the overlay-rendered base
        image (typically from a recent ``sample_trace``); the trace is not generated here.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.atomic_token is None:
            raise ValueError("atomic_token is required for the action expert.")
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        exec_images, exec_image_masks = self._exec_images(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, exec_images, exec_image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Prefill: no stream tokens -> the MoE block isn't invoked, placeholder weights.
        dummy_weights = self._dummy_combine_weights(batch_size)
        # Denoising-loop weights: real one-hot routing when the action stream is a MoE.
        if self.action_is_moe:
            skill_id = observation.atomic_token.astype(jnp.int32)
            action_combine_weights = self._combine_weights(batch_size, skill_id, self.action_horizon)
        else:
            action_combine_weights = dummy_weights

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
        """Sample actions and predict skill-completion progress in one shared forward.

        The recommended deployment path: both heads consume the *same* execution-mode
        prefix (overlay image + clean wrist + skill prompt + state), so running them
        together avoids redoing the SigLIP encode and Gemma prefill. Returns
        ``(actions (B, ah, ad), progress (B,) in [0, 1])``.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.atomic_token is None:
            raise ValueError("atomic_token is required for the action expert / completion head.")
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        exec_images, exec_image_masks = self._exec_images(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix_with_images(
            observation, exec_images, exec_image_masks
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        dummy_weights = self._dummy_combine_weights(batch_size)
        skill_id = observation.atomic_token.astype(jnp.int32)
        action_combine_weights = (
            self._combine_weights(batch_size, skill_id, self.action_horizon)
            if self.action_is_moe else dummy_weights
        )

        # Capture prefix_out (vs sample_actions which discards it) for the completion head;
        # the kv_cache is reused inside the action denoising loop.
        (prefix_out, _action_out, _trace_out), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None],
            mask=prefix_attn_mask,
            positions=positions,
            adarms_cond=[None, None, None],
            hard_combine_weights=dummy_weights,
        )
        del _action_out, _trace_out
        progress_pred = self._completion_predict(prefix_out, prefix_mask, skill_id)

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
        """Predict skill-completion progress only (no action sampling).

        Cheaper than ``sample_actions_and_completion`` when actions are not needed this
        step: runs the SigLIP encode + Gemma prefill + completion head, no denoising loop.
        ``rng`` is unused (kept for API symmetry).
        """
        del rng
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        if observation.atomic_token is None:
            raise ValueError("atomic_token is required for the completion head.")
        batch_size = observation.state.shape[0]

        exec_images, exec_image_masks = self._exec_images(observation)
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
        """Sample a trace from the trace expert (planning mode).

        ``observation.images`` should be the *clean* base image (no overlay), with
        ``semantic_target_xy`` / ``current_ee_xy`` provided. The trace stream's combine
        weights are real one-hot routing when ``trace_is_moe`` and a placeholder otherwise.
        """
        observation = _trace_obs.preprocess_trace_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # Noise lives on the *internal* trace sequence (includes the appended semantic-target
        # anchor row when enabled); the returned trace strips the anchor before handoff.
        L = self.trace_seq_len
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, L, self.trace_dim))

        target_xy = observation.semantic_target_xy
        ee = observation.current_ee_xy
        if self.trace_is_moe:
            skill_id = observation.atomic_token.astype(jnp.int32)
            combine_weights = self._combine_weights(batch_size, skill_id, L)
        else:
            combine_weights = self._dummy_combine_weights(batch_size)

        # Prefill the VLM with the clean (no-overlay) prefix.
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

        # Fix the row-0 (and, when present, appended last-row) noise to the corresponding
        # entries of the initial noise tensor, so the inpainted anchors stay on a continuous
        # interpolation in t that matches the training-time conditional distribution.
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
        # Defensive final clamps: at t=0 the inpainting math collapses the anchor rows
        # exactly to ``ee`` / ``target_xy``; re-assert to absorb accumulated float drift.
        x_0 = x_0.at[:, 0, :].set(ee)
        if append_anchor:
            x_0 = x_0.at[:, -1, :].set(target_xy)
            # Strip the anchor row: callers want the (N, 2) polyline, not (N+1, 2).
            x_0 = x_0[:, :self.trace_horizon, :]
        return x_0
