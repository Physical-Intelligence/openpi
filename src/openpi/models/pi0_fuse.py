"""Pi0Fuse model: pi05-compatible architecture with joint textual reasoning and action loss.

Combines pi05's architecture (AdaRMSNorm, no state_proj) for weight compatibility
with "Do What You Say" paper's reasoning loss approach (text CE + action diffusion).
"""
import numpy
numpy.set_printoptions(threshold=numpy.inf)

import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos, put_along_last_axis
from openpi.models.pi0_fuse_config import Pi0FuseConfig

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1

class Pi0Fuse(Pi0):
    """Pi05-compatible model with joint text reasoning loss and action diffusion loss.

    Architecture matches pi05 (AdaRMSNorm, no state_proj) for loading pi05 base weights.
    Adds text cross-entropy loss on reasoning tokens alongside flow-matching action loss.
    """

    def __init__(self, config: Pi0FuseConfig, rngs: nnx.Rngs):
        assert config.pi05
        super().__init__(config, rngs)
        self.diffusion_loss_coeff = config.diffusion_loss_coeff

    def embed_prefix(self, obs):
        # Per-sample `ar_mask` version of `Pi0.embed_prefix`. `Pi0.embed_prefix`
        # collapses the text `ar_mask` to sample 0 only (`obs.token_ar_mask[0]`)
        # and returns a 1-D `[total_seq]` mask broadcast over the batch, which is
        # wrong for Fuse: each sample has its own prefix/suffix boundary (action
        # mode = 1-token suffix, reasoning mode = multi-token suffix starting at
        # a different position). Return a 2-D `[B, total_seq]` mask instead.
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            m = einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            input_mask.append(m)
            ar_mask.append(jnp.zeros_like(m, dtype=jnp.int32))

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(txt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask.astype(jnp.int32))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.FuseObservation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train,
            image_keys=list(observation.images.keys())
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        img_txt_tokens, img_txt_mask, img_txt_ar_mask = self.embed_prefix(observation)
        action_tokens, action_mask, action_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        input_mask = jnp.concatenate([img_txt_mask, action_mask], axis=1)
        B = img_txt_ar_mask.shape[0]
        action_ar_mask = jnp.broadcast_to(action_ar_mask, (B, action_ar_mask.shape[-1]))
        ar_mask = jnp.concatenate([img_txt_ar_mask, action_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (img_txt_pre_logits, action_out), _, intermediates = self.PaliGemma.llm(
            [img_txt_tokens, action_tokens], mask=attn_mask, positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # --- Text CE loss (reasoning loss) ---
        txt_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            _gemma.PALIGEMMA_VOCAB_SIZE,
        )
        txt_logits = self.PaliGemma.llm(
            img_txt_pre_logits[:, -1 - txt_targets.shape[1]: -1],
            method="deembed",
        )
        txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        txt_loss_mask = observation.token_loss_mask[:, 1:]
        txt_token_pplx = jnp.sum(txt_targets * txt_logp, axis=-1)
        txt_loss = (
            -jnp.sum(txt_token_pplx * txt_loss_mask, axis=-1) /
            jnp.clip(jnp.sum(txt_loss_mask, axis=-1), 1)
        )

        # --- Action diffusion loss (flow matching) ---
        # we only sample when diffusion_loss_mask is False to contribute text loss
        v_t = self.action_out_proj(action_out[:, -self.action_horizon:])
        action_loss = jnp.mean(
            jnp.square(v_t - u_t) * observation.diffusion_loss_mask[:, None, None],
            axis=(-2, -1),
        )

        loss = txt_loss + self.diffusion_loss_coeff * action_loss
        return loss

    @at.typecheck
    def prefill(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        temperature: float = 0.,
        debug: bool = False,
        bias = None
    ):
        """Prefill the prefix for inference. Returns KV cache and decision to think or act."""
        observation = _model.preprocess_observation(
            None, observation, train=False,
            image_keys=list(observation.images.keys())
        )

        original_observation = observation
        first_one_indices = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding_mask = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one_indices[..., jnp.newaxis]
        masked_tokenized_prompt = jnp.where(padding_mask, 0, observation.tokenized_prompt)
        masked_tokenized_prompt_mask = jnp.logical_not(padding_mask)
        observation = dataclasses.replace(
            observation,
            tokenized_prompt=masked_tokenized_prompt,
            tokenized_prompt_mask=masked_tokenized_prompt_mask,
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1

        (pre_logit, _), kv_cache, intermediates = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions,
            adarms_cond=[None, None], bias=bias
        )

        eop_indices = prefix_positions[:, -1]
        eop_pre_logit = jnp.take_along_axis(pre_logit, eop_indices[:, None, None], axis=1)
        eop_logit = self.PaliGemma.llm(eop_pre_logit, method="deembed")

        valid_tokens = jnp.array([_tokenizer.BEGIN_OF_ACTION, _tokenizer.BEGIN_OF_REASONING])
        valid_mask = jnp.full((1, 1, eop_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = eop_logit + valid_mask

        if temperature > 0.0:
            token = jax.random.categorical(rng, eop_logit / temperature, axis=-1)
        else:
            token = jnp.argmax(eop_logit, axis=-1)

        has_boa = jnp.any(token == _tokenizer.BEGIN_OF_ACTION, axis=1)

        return observation, kv_cache, token, eop_logit, prefix_mask, prefix_positions, has_boa, intermediates

    @at.typecheck
    def reason(
        self,
        rng: at.KeyArrayLike,
        last_logit: at.Float[at.Array, "b 1 v"],
        prefix_kv_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_positions: at.Int[at.Array, "b p"],
        *,
        temperature: float = 0.,
        max_decoding_steps: int = 256,
    ) -> at.Int[at.Array, "b _s"]:
        """Autoregressive reasoning token generation.

        Uses the LLM's built-in _update_cache for KV cache management.
        """
        batch_size = prefix_mask.shape[0]
        output_tokens = jnp.zeros((batch_size, max_decoding_steps), dtype=jnp.int32)
        # Do not sample control tokens after step 0; they are protocol markers, not reasoning text.

        idx, k_cache, v_cache = prefix_kv_cache
        k_cache = jnp.pad(k_cache, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0)))
        v_cache = jnp.pad(v_cache, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0)))
        kv_cache = (idx, k_cache, v_cache)

        def decode_step(carry):
            rng, last_logit, output_tokens, kv_cache, all_eos, step = carry
            step_rng = jax.random.fold_in(rng, step)
            sample_logit = last_logit

            if temperature > 0.0:
                token = jax.random.categorical(step_rng, sample_logit / temperature, axis=-1)
            else:
                token = jnp.argmax(sample_logit, axis=-1)

            # Force the first token generated to be a BOR token. Otherwise, this is just normal generation
            token = jnp.where(
                step == 0,
                jnp.full_like(token, _tokenizer.BEGIN_OF_REASONING),
                token,
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (batch_size, 1)), token
            )

            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefix_positions[:, [-1]] + step + 1

            decode_visible = jnp.arange(max_decoding_steps) <= step
            full_mask = jnp.concatenate(
                [
                    prefix_mask,
                    jnp.broadcast_to(decode_visible, (batch_size, max_decoding_steps))
                ],
                axis=-1
            )

            mask = full_mask[:, None, :]

            # Text head only (second output ignored)
            (last_pre_logit, _), kv_cache, intermediates = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions,
                kv_cache=kv_cache, adarms_cond=[None, None],
            )
            last_logit = self.PaliGemma.llm(last_pre_logit, method="deembed")

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def decode_cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            decode_cond, decode_step,
            (rng, last_logit, output_tokens, kv_cache, False, 0),
        )
        return output_tokens

    @at.typecheck
    def act(
        self,
        rng: at.KeyArrayLike,
        prefix_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_positions: at.Int[at.Array, "b p"],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> at.Float[at.Array, "b ah ad"]:
        """Sample actions after prefill, using diffusion denoising."""
        # Pad prefix KV cache by 1 to make room for the BOA token.
        # _init_cache sized the cache exactly to prefix_len, so there is no
        # spare slot for the additional token that act() needs to insert.
        idx, k_cache, v_cache = prefix_cache
        k_cache = jnp.pad(k_cache, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        v_cache = jnp.pad(v_cache, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        prefix_cache = (idx, k_cache, v_cache)

        batch_size = prefix_mask.shape[0]

        boa_token = jnp.broadcast_to(
            _tokenizer.BEGIN_OF_ACTION, (prefix_mask.shape[0], 1)
        )
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        boa_attn_mask = jnp.concatenate(
            [prefix_attn_mask, jnp.ones((prefix_attn_mask.shape[0], 1, 1), dtype=jnp.bool_)],
            axis=-1,
        )
        boa_positions = prefix_positions[:, [-1]] + 1
        boa_token_embedding = self.PaliGemma.llm(boa_token, method="embed")
        (_, _), img_txt_kv_cache, intermediates = self.PaliGemma.llm(
            [boa_token_embedding, None], mask=boa_attn_mask, positions=boa_positions,
            kv_cache=prefix_cache, adarms_cond=[None, None],
        )
        img_txt_mask = jnp.pad(prefix_mask, ((0, 0), (0, 1)), constant_values=1)

        dt = -1.0 / num_steps
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        return self._generate_action(noise, dt, img_txt_kv_cache, img_txt_mask, batch_size)
