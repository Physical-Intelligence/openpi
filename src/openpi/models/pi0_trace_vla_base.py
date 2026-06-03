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
