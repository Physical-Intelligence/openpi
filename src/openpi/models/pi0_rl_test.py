import math

import jax
import jax.numpy as jnp

from openpi.models import pi0_rl
from openpi.models.pi0_rl_config import Pi0RLConfig


def _token_only_decoder(
    rl_token: jnp.ndarray,
    target_embeddings: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    del mask
    return jnp.broadcast_to(rl_token[:, None, :], target_embeddings.shape)


def test_reconstruction_ablation_metrics_detects_zero_and_shuffled_tokens():
    rl_token = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target_embeddings = jnp.broadcast_to(rl_token[:, None, :], (2, 2, 2))
    mask = jnp.ones((2, 2), dtype=jnp.bool_)

    metrics = pi0_rl.compute_reconstruction_ablation_metrics(
        _token_only_decoder,
        rl_token,
        target_embeddings,
        mask,
        shuffle_perm=jnp.array([1, 0]),
    )

    assert math.isclose(metrics["real_recon_loss"], 0.0)
    assert math.isclose(metrics["zero_recon_loss"], 15.0)
    assert math.isclose(metrics["shuffled_recon_loss"], 8.0)
    assert math.isclose(metrics["zero_recon_gap"], 15.0)
    assert math.isclose(metrics["shuffled_recon_gap"], 8.0)


def _make_tiny_pi0rl_config():
    return Pi0RLConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        pi05=False,
        rl_num_layers=1,
        rl_num_heads=4,
        rl_mlp_dim=64,
        rl_vla_loss_weight=0.0,
    )


def test_sample_actions_with_rl_token_accepts_fixed_noise():
    """Fixed noise should produce deterministic actions regardless of rng."""
    config = _make_tiny_pi0rl_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    fixed_noise = jnp.ones((1, config.action_horizon, config.action_dim)) * 0.5

    actions_a, token_a = model.sample_actions_with_rl_token(
        jax.random.key(1), obs, num_steps=3, noise=fixed_noise
    )
    actions_b, token_b = model.sample_actions_with_rl_token(
        jax.random.key(99), obs, num_steps=3, noise=fixed_noise
    )

    assert actions_a.shape == (1, config.action_horizon, config.action_dim)
    assert token_a.shape[0] == 1
    assert jnp.allclose(actions_a, actions_b, atol=1e-5)
    assert jnp.allclose(token_a, token_b, atol=1e-5)


def test_sample_actions_with_rl_token_none_noise_uses_rng():
    """When noise=None, different rngs should (almost surely) give different actions."""
    config = _make_tiny_pi0rl_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    actions_a, _ = model.sample_actions_with_rl_token(
        jax.random.key(1), obs, num_steps=3, noise=None
    )
    actions_b, _ = model.sample_actions_with_rl_token(
        jax.random.key(2), obs, num_steps=3, noise=None
    )

    assert not jnp.allclose(actions_a, actions_b, atol=1e-5)
