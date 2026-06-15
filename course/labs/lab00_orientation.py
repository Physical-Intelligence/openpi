"""Lab 00 — Orientation: see the data flow before you build it.

Goal: instantiate the REAL pi0.5 config, inspect the input/output specs, and (optionally)
build a tiny model and watch shapes flow through embed_prefix / embed_suffix.

Run:  uv run python course/labs/lab00_orientation.py

There is almost nothing to implement here — this lab is about *looking*. Follow the TODOs
and make sure every printed shape matches the diagram you drew in Module 00.
"""

import jax
import jax.numpy as jnp

from openpi.models import pi0_config


def main():
    # --- 1. The config -------------------------------------------------------
    # A real pi0.5 config. Note the toggles discussed in Module 00 / 06.
    cfg = pi0_config.Pi0Config(pi05=True)
    print("model_type        :", cfg.model_type)
    print("action_dim        :", cfg.action_dim)
    print("action_horizon    :", cfg.action_horizon)
    print("max_token_len     :", cfg.max_token_len, "(48 for pi0, 200 for pi05 — why?)")
    print("discrete_state_inp:", cfg.discrete_state_input)
    print("paligemma_variant :", cfg.paligemma_variant)
    print("action_expert     :", cfg.action_expert_variant)

    # --- 2. The I/O contract (no params needed: just shape specs) ------------
    obs_spec, act_spec = cfg.inputs_spec(batch_size=2)
    print("\n--- Observation spec ---")
    for k, v in obs_spec.images.items():
        print(f"  image[{k}]: {v.shape} {v.dtype}")
    print("  state          :", obs_spec.state.shape, obs_spec.state.dtype)
    print("  tokenized_prompt:", obs_spec.tokenized_prompt.shape)
    print("  Actions        :", act_spec.shape, act_spec.dtype)

    # TODO(you): How many image tokens will the three cameras produce in total,
    # given So400m/14 on 224x224? Write the number here as a comment and confirm
    # it in Lab 02.

    # --- 3. (Optional, heavier) build a tiny model and inspect intermediates --
    # The SigLIP encoder is fixed to So400m/14 (~400M params) even when the Gemma
    # experts are 'dummy', so creating a real Pi0 is memory-heavy. Prefer eval_shape
    # to inspect WITHOUT allocating params:
    tiny = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    obs = tiny.fake_obs(batch_size=1)
    act = tiny.fake_act(batch_size=1)
    print("\nfake_obs/fake_act built. action chunk shape:", act.shape)

    # TODO(you): Uncomment to actually build + run (slow on CPU due to SigLIP):
    # from openpi.models.pi0 import Pi0
    # import flax.nnx as nnx
    # model = Pi0(tiny, rngs=nnx.Rngs(0))
    # prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
    # print("prefix tokens:", prefix_tokens.shape, "ar_mask:", prefix_ar.shape)
    # loss = model.compute_loss(jax.random.key(0), obs, act, train=False)
    # print("loss per (b, horizon):", loss.shape)

    print("\nDone. Re-draw the Module-00 diagram from memory and label these shapes.")


if __name__ == "__main__":
    main()
