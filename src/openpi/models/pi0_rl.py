"""Pi0 model extended with RL Token encoder-decoder (RLT Stage 1).

Implements the RL Token method from Xu et al. (2025). A small transformer
encoder-decoder is attached to the frozen VLA to produce a compact RL token
representation. The encoder compresses the VLA's final-layer prefix embeddings
into a single vector via a learned query. The decoder autoregressively
reconstructs the original embeddings from only this token, forcing it to act
as an information bottleneck.

Training loss:
    L_total = L_ro(phi) + alpha * L_vla(theta)

where L_ro is the autoregressive reconstruction loss (gradients to encoder-
decoder params phi only, via stop_gradient on VLA embeddings) and L_vla is
the standard flow-matching action prediction loss (gradients to VLA params
theta only).
"""

from collections.abc import Callable
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models.pi0_rl_config import Pi0RLConfig
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Lightweight transformer components for the RL token encoder-decoder
# ---------------------------------------------------------------------------


class RLTokenTransformerBlock(nnx.Module):
    """Pre-norm transformer block with SwiGLU FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, rngs=rngs)

        self.ffn_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.ffn_gate = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.ffn_up = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.ffn_down = nnx.Linear(mlp_dim, dim, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        b, s, d = x.shape

        # --- self-attention with pre-norm ---
        h = self.attn_norm(x)
        q = self.q_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(b, s, self.num_heads, self.head_dim)

        scale = jnp.float32(self.head_dim) ** -0.5
        logits = jnp.einsum("bsnh,btnh->bnst", q, k) * scale
        if mask is not None:
            # mask: (b, s, s) → (b, 1, s, s) for head broadcast
            logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(logits.dtype).min)
        attn_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(x.dtype)
        attn_out = jnp.einsum("bnst,btnh->bsnh", attn_weights, v).reshape(b, s, d)
        x = x + self.o_proj(attn_out)

        # --- SwiGLU FFN with pre-norm ---
        h = self.ffn_norm(x)
        x = x + self.ffn_down(nnx.silu(self.ffn_gate(h)) * self.ffn_up(h))

        return x


class RLTokenEncoder(nnx.Module):
    """Compresses VLA prefix embeddings into a single RL token via a learned query."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_layers: int, rngs: nnx.Rngs):
        self.rl_query = nnx.Param(jax.random.normal(rngs.params(), (1, 1, dim)) * 0.02)
        self.layers = {
            f"layer_{i}": RLTokenTransformerBlock(dim, num_heads, mlp_dim, rngs)
            for i in range(num_layers)
        }

    def __call__(self, vla_embeddings: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        Args:
            vla_embeddings: (b, M, dim) stop-gradiented VLA final-layer embeddings.
            mask: (b, M) True for valid tokens.

        Returns:
            rl_token: (b, dim).
        """
        b = vla_embeddings.shape[0]
        query = jnp.broadcast_to(self.rl_query.value, (b, 1, vla_embeddings.shape[-1]))
        x = jnp.concatenate([vla_embeddings, query], axis=1)  # (b, M+1, dim)

        if mask is not None:
            ext = jnp.concatenate([mask, jnp.ones((b, 1), dtype=jnp.bool_)], axis=1)
            attn_mask = ext[:, None, :] & ext[:, :, None]  # (b, M+1, M+1) bidirectional
        else:
            attn_mask = None

        for key in sorted(self.layers):
            x = self.layers[key](x, attn_mask)

        return x[:, -1, :]  # RL token at query position


class RLTokenDecoder(nnx.Module):
    """Autoregressively reconstructs VLA embeddings from the RL token."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_layers: int, rngs: nnx.Rngs):
        self.layers = {
            f"layer_{i}": RLTokenTransformerBlock(dim, num_heads, mlp_dim, rngs)
            for i in range(num_layers)
        }
        self.output_proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        rl_token: jax.Array,
        target_embeddings: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Teacher-forced autoregressive reconstruction.

        Decoder input:  [z_rl, z̄_1, z̄_2, ..., z̄_{M-1}]
        Target output:  [z̄_1, z̄_2, z̄_3, ..., z̄_M       ]

        Causal masking ensures position i only attends to positions ≤ i.

        Args:
            rl_token: (b, dim).
            target_embeddings: (b, M, dim) stop-gradiented targets.
            mask: (b, M) True for valid target tokens.

        Returns:
            predictions: (b, M, dim).
        """
        b, seq_len, _ = target_embeddings.shape

        decoder_input = jnp.concatenate(
            [rl_token[:, None, :], target_embeddings[:, :-1, :]], axis=1
        )  # (b, M, dim)

        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None]  # (1, M, M)
        if mask is not None:
            # Key validity: pos 0 is rl_token (always valid), pos 1..M-1 map to targets 0..M-2
            key_valid = jnp.concatenate([jnp.ones((b, 1), dtype=jnp.bool_), mask[:, :-1]], axis=1)
            attn_mask = causal & key_valid[:, None, :]  # (b, M, M)
        else:
            attn_mask = jnp.broadcast_to(causal, (b, seq_len, seq_len))

        x = decoder_input
        for key in sorted(self.layers):
            x = self.layers[key](x, attn_mask)

        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Reconstruction diagnostics
# ---------------------------------------------------------------------------


def _reconstruction_loss(
    predictions: jax.Array,
    target_embeddings: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    recon_sq = jnp.square(predictions - target_embeddings)
    per_token_l2 = recon_sq.sum(axis=-1)

    if mask is not None:
        per_token_l2 = per_token_l2 * mask
        num_valid = jnp.clip(mask.sum(axis=1), 1)
        per_example = per_token_l2.sum(axis=1) / num_valid
    else:
        per_example = jnp.mean(per_token_l2, axis=1)

    return jnp.mean(per_example)


def compute_reconstruction_ablation_metrics(
    decoder_fn: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array],
    rl_token: jax.Array,
    target_embeddings: jax.Array,
    mask: jax.Array | None = None,
    *,
    shuffle_perm: jax.Array | None = None,
) -> dict[str, float]:
    """Compare real, zeroed, and shuffled RL-token reconstruction losses."""
    real_predictions = decoder_fn(rl_token, target_embeddings, mask)
    real_loss = _reconstruction_loss(real_predictions, target_embeddings, mask)

    zero_token = jnp.zeros_like(rl_token)
    zero_predictions = decoder_fn(zero_token, target_embeddings, mask)
    zero_loss = _reconstruction_loss(zero_predictions, target_embeddings, mask)

    if shuffle_perm is None:
        shuffle_perm = jnp.roll(jnp.arange(rl_token.shape[0]), 1)
    shuffled_token = rl_token[shuffle_perm]
    shuffled_predictions = decoder_fn(shuffled_token, target_embeddings, mask)
    shuffled_loss = _reconstruction_loss(shuffled_predictions, target_embeddings, mask)

    return {
        "real_recon_loss": float(real_loss),
        "zero_recon_loss": float(zero_loss),
        "shuffled_recon_loss": float(shuffled_loss),
        "zero_recon_gap": float(zero_loss - real_loss),
        "shuffled_recon_gap": float(shuffled_loss - real_loss),
    }


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class Pi0RL(_pi0.Pi0):
    """Pi0 with RL Token encoder-decoder for RLT Stage 1.

    Adds a lightweight transformer encoder-decoder on top of the frozen VLA
    prefix embeddings. During training, the combined loss is:

        L_total = L_ro(phi) + alpha * L_vla(theta)

    where stop_gradient on VLA embeddings ensures L_ro does not backpropagate
    into the VLA, and L_vla does not involve encoder-decoder params phi.
    """

    def __init__(self, config: Pi0RLConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        dim = config.rl_embedding_dim

        self.rl_encoder = RLTokenEncoder(
            dim=dim,
            num_heads=config.rl_num_heads,
            mlp_dim=config.rl_mlp_dim,
            num_layers=config.rl_num_layers,
            rngs=rngs,
        )
        self.rl_decoder = RLTokenDecoder(
            dim=dim,
            num_heads=config.rl_num_heads,
            mlp_dim=config.rl_mlp_dim,
            num_layers=config.rl_num_layers,
            rngs=rngs,
        )

        self._rl_vla_loss_weight = config.rl_vla_loss_weight

    # ------------------------------------------------------------------
    # Public helpers for downstream stages
    # ------------------------------------------------------------------

    @at.typecheck
    def extract_rl_token(
        self, observation: _model.Observation
    ) -> at.Float[at.Array, "b emb"]:
        """Extract the RL token from an observation (no gradients into VLA).

        Runs only the VLA prefix (image + language) through the backbone,
        stop-gradients the output, and feeds it through the encoder.
        """
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        outputs, _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        prefix_out = jax.lax.stop_gradient(outputs[0])

        return self.rl_encoder(prefix_out, mask=prefix_mask)

    @at.typecheck
    def sample_actions_with_rl_token(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Float[at.Array, "b emb"]]:
        """Sample actions AND extract the RL token in a single VLA forward pass.

        Used during online RL (Stages 4+) where we need both the reference
        action chunk and the RL token for the actor-critic.

        Args:
            noise: If provided, use this fixed noise instead of sampling from rng.
                   Shape must be [batch_size, action_horizon, action_dim].
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Prefix forward pass: get embeddings + KV cache
        outputs, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        prefix_out = jax.lax.stop_gradient(outputs[0])
        rl_token = self.rl_encoder(prefix_out, mask=prefix_mask)

        # Denoise actions using the cached prefix (same as Pi0.sample_actions)
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_cross_mask = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1])
            )
            full_attn_mask = jnp.concatenate([prefix_cross_mask, suffix_attn_mask], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=pos,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, rl_token

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ---- VLA forward pass (identical to Pi0) ----
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # ---- VLA flow-matching loss (L_vla) ----
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        per_step_sq = jnp.square(v_t - u_t)

        if observation.action_dim_mask is not None:
            dim_mask = observation.action_dim_mask
            if dim_mask.ndim < per_step_sq.ndim:
                dim_mask = dim_mask[..., None, :]
            num_real_dims = jnp.clip(dim_mask.sum(axis=-1, keepdims=True), 1)
            per_step_sq = per_step_sq * dim_mask
            vla_per_step = per_step_sq.sum(axis=-1) / num_real_dims.squeeze(-1)
        else:
            vla_per_step = jnp.mean(per_step_sq, axis=-1)

        if observation.action_is_pad is not None:
            pad_mask = ~observation.action_is_pad
            vla_per_step = vla_per_step * pad_mask
            num_real = jnp.clip(pad_mask.sum(axis=-1, keepdims=True), 1)
            vla_per_step = vla_per_step * (pad_mask.shape[-1] / num_real)

        # ---- RL Token reconstruction loss (L_ro) ----
        # Stop-gradient: L_ro must NOT backpropagate into the VLA.
        sg_prefix = jax.lax.stop_gradient(prefix_out)

        rl_token = self.rl_encoder(sg_prefix, mask=prefix_mask)
        predictions = self.rl_decoder(rl_token, sg_prefix, mask=prefix_mask)

        recon_sq = jnp.square(predictions - sg_prefix)  # (b, M, dim)
        # Per-token squared L2 norm: sum over embedding dim (matching paper Eq. 2),
        # then mean over valid tokens to stay scale-independent of sequence length.
        per_token_l2 = recon_sq.sum(axis=-1)  # (b, M)

        if prefix_mask is not None:
            per_token_l2 = per_token_l2 * prefix_mask
            num_valid = jnp.clip(prefix_mask.sum(axis=1), 1)
            recon_loss = per_token_l2.sum(axis=1) / num_valid  # (b,)
        else:
            recon_loss = jnp.mean(per_token_l2, axis=1)  # (b,)

        # ---- Combined loss: L_ro + alpha * L_vla ----
        # recon_loss: (b,) → broadcast to (b, ah) via [..., None]
        # When averaged over (b, ah), this gives mean(L_ro) + alpha * mean(L_vla).
        return recon_loss[..., None] + self._rl_vla_loss_weight * vla_per_step
