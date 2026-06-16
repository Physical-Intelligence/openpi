"""model_pi05_loc.py — bbox: Pi0WithLocCE.

Subclass of openpi.models.pi0.Pi0 that adds a CE-on-loc-tokens loss term on
top of the existing flow-matching MSE term, sharing image tokens across two
language conditionings:

  L_total = β · MSE(action vector field, ε − A_t)        # parent's loss
          + α · CE(LM-head logits at target positions,
                   target_loc_tokens)                     # new

The CE term is masked per-sample by `obs.target_loc_mask` (= `has_bbox` from
the bbox sidecar). For frames with no bbox supervision the CE contribution
is zero — the parent's MSE anchor keeps acting on every frame.

Critical design choices (matching the plan + Step-0 diagnosis):

  1. The CE branch uses a SEPARATE prompt (`tokenized_prompt_detect`,
     "detect <obj>\\n<loc><loc><loc><loc>") because the π0.5 template can't
     emit loc tokens (verified: raw PaliGemma under the π0.5 template
     decodes "Box" garbage — see probe_raw_paligemma.py output).

  2. SigLIP runs ONCE per training step. `_embed_images(obs)` is called once;
     the same image tokens are reused for both the MSE-template forward and
     the CE-detect forward. This is why we factored `embed_prefix` in Phase 2.

  3. The CE forward is PREFIX-ONLY (`self.PaliGemma.llm([prefix, None], ...)`):
     no action expert involvement, no MSE-side gradients from the CE branch.
     The action expert stays frozen via the freeze filter (see
     experiments/config.py vlm_only_freeze_filter).

  4. CE is broadcast over the action_horizon axis so the trainer's
     `jnp.mean(loss)` produces the doc's intended `α·mean_B(ce) + β·mean_B(mean_H(mse))`
     scalar after reduction. No explicit `× action_horizon` scaling needed.

  5. Sentinel target tokens (-1, used when has_bbox=False) are clipped to 0
     before `take_along_axis` to avoid OOB indexing; the loss mask then
     zeroes their contribution.

The return shape is (B, action_horizon) — same as parent — so the trainer's
mean reduction works unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pi0WithLocCEConfig(_pi0_config.Pi0Config):
    """Pi0Config + α/β knobs for the joint CE+MSE loss."""

    # α: weight on the loc-CE term. Doc default 1.0.
    loc_loss_weight: float = 1.0
    # β: weight on the flow-matching MSE term. Doc default 10.0.
    mse_loss_weight: float = 10.0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0WithLocCE":
        return Pi0WithLocCE(self, rngs=nnx.Rngs(rng))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Pi0WithLocCE(_pi0.Pi0):
    """Joint CE-on-loc-tokens + flow-matching-MSE training for pi05_base."""

    def __init__(self, config: Pi0WithLocCEConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        # Store the two scalar weights as plain Python attributes (not nnx.Variable —
        # we never want to differentiate w.r.t. these, and the optimizer freeze
        # filter shouldn't see them as params).
        self.loc_loss_weight = float(config.loc_loss_weight)
        self.mse_loss_weight = float(config.mse_loss_weight)

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
        # When the CE branch is active (detect prompt present), base-camera
        # geometric augmentation (RandomCrop/Rotate in preprocess_observation,
        # model.py:172-187) would shift pixel positions and invalidate the
        # bbox coordinates in `target_loc_tokens`. Hard-disable augmentation
        # for those steps. ColorJitter is also disabled (small cost; we
        # prioritize correctness for the first bbox run).
        ce_active = observation.tokenized_prompt_detect is not None
        preprocess_train = train and (not ce_active)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=preprocess_train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # ------------------------------------------------------------------
        # Image tokens — computed ONCE, reused by both branches.
        # ------------------------------------------------------------------
        img_tokens, img_input_mask, img_ar_mask = self._embed_images(observation)

        # ==================================================================
        # MSE branch — π0.5 template prefix + flow-matching suffix.
        # Identical math to Pi0.compute_loss; we just rebuild the prefix from
        # the shared image tokens + the π0.5 language tokens.
        # ==================================================================
        lang_mse, lang_mse_mask, lang_mse_ar = self._embed_language(
            observation.tokenized_prompt, observation.tokenized_prompt_mask
        )
        prefix_tokens = jnp.concatenate([img_tokens, lang_mse], axis=1)
        prefix_mask = jnp.concatenate([img_input_mask, lang_mse_mask], axis=1)
        prefix_ar_mask = jnp.concatenate([img_ar_mask, lang_mse_ar], axis=0)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )
        full_input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(full_input_mask, full_ar_mask)
        positions = jnp.cumsum(full_input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        mse_per_dim_per_horizon = jnp.square(v_t - u_t)              # (B, H, action_dim)
        mse_per_horizon = jnp.mean(mse_per_dim_per_horizon, axis=-1)  # (B, H) — same as parent

        # ==================================================================
        # CE branch — detect prompt prefix-only forward, no action expert.
        # If detect fields are absent, skip CE entirely (MSE-only fallback).
        # ==================================================================
        if observation.tokenized_prompt_detect is None:
            return self.mse_loss_weight * mse_per_horizon

        lang_det, lang_det_mask, lang_det_ar = self._embed_language(
            observation.tokenized_prompt_detect, observation.tokenized_prompt_detect_mask
        )
        det_prefix_tokens = jnp.concatenate([img_tokens, lang_det], axis=1)
        det_prefix_mask = jnp.concatenate([img_input_mask, lang_det_mask], axis=1)
        det_prefix_ar = jnp.concatenate([img_ar_mask, lang_det_ar], axis=0)
        det_attn_mask = _pi0.make_attn_mask(det_prefix_mask, det_prefix_ar)
        det_positions = jnp.cumsum(det_prefix_mask, axis=1) - 1

        # Prefix-only forward (no action expert).
        (det_prefix_out, _), _ = self.PaliGemma.llm(
            [det_prefix_tokens, None],
            mask=det_attn_mask,
            positions=det_positions,
        )

        # target_loc_positions is relative to the detect language sequence
        # (positions 0..ld-1). The detect prefix is [img | lang], so positions
        # in the full prefix are target_loc_positions + n_img_tokens.
        n_img = img_tokens.shape[1]
        positions_in_prefix = observation.target_loc_positions + n_img  # (B, 4)

        # Gather hidden states at the 4 target positions.
        # det_prefix_out: (B, T, emb)  →  loc_hidden: (B, 4, emb)
        loc_hidden = jnp.take_along_axis(
            det_prefix_out, positions_in_prefix[:, :, None], axis=1
        )

        # Decode to vocab logits via the tied LM head.
        # Same hook probe_lm_head.py:_decode_logits_fn uses.
        loc_logits = self.PaliGemma.llm(
            loc_hidden, method=lambda m, x: m.embedder.decode(x)
        ).astype(jnp.float32)  # (B, 4, V)

        # Cross-entropy. Sentinel targets (-1) get clipped to 0 to keep
        # take_along_axis in-bounds; the loss mask zeroes those contributions.
        targets = observation.target_loc_tokens                       # (B, 4) int
        targets_safe = jnp.where(targets < 0, 0, targets)
        log_probs = jax.nn.log_softmax(loc_logits, axis=-1)           # (B, 4, V)
        ce_per_position = -jnp.take_along_axis(
            log_probs, targets_safe[..., None], axis=-1
        ).squeeze(-1)                                                  # (B, 4)
        ce_per_sample = jnp.mean(ce_per_position, axis=-1)            # (B,)

        # Per-sample mask: bool → float, broadcast.
        loss_mask = observation.target_loc_mask.astype(ce_per_sample.dtype)  # (B,)
        ce_per_sample = ce_per_sample * loss_mask                            # (B,)

        # Broadcast CE over the horizon so the trainer's jnp.mean reduction
        # gives the doc-intended α·mean_B(ce) + β·mean_B(mean_H(mse)).
        ce_broadcast = jnp.broadcast_to(
            ce_per_sample[..., None], mse_per_horizon.shape
        )                                                              # (B, H)

        return self.mse_loss_weight * mse_per_horizon + self.loc_loss_weight * ce_broadcast
