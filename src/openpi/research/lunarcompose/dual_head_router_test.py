"""Tests for dual_head_router module."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from openpi.research.lunarcompose.dual_head_router import DualHeadRouter
from openpi.research.lunarcompose.dual_head_router import make_active_mask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_DIM = 32  # lang_dim(16) + vis_dim(16)
LANG_DIM = 16
VIS_DIM = 16
MAX_TASKS = 4
MAX_ENVS = 3
BATCH = 2


@pytest.fixture
def router() -> DualHeadRouter:
    """Fresh DualHeadRouter for each test."""
    return DualHeadRouter(
        input_dim=INPUT_DIM,
        max_tasks=MAX_TASKS,
        max_envs=MAX_ENVS,
        hidden_dim=64,
        num_layers=2,
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def dummy_inputs() -> tuple[jax.Array, jax.Array]:
    """Random lang/visual embeddings of shape (B, dim)."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    lang = jax.random.normal(k1, (BATCH, LANG_DIM))
    vis = jax.random.normal(k2, (BATCH, VIS_DIM))
    return lang, vis


# ---------------------------------------------------------------------------
# Smoke test (pre-existing — keep)
# ---------------------------------------------------------------------------


def test_dual_head_router_module_imports():
    """Verify the dual_head_router module can be imported."""
    from openpi.research.lunarcompose import dual_head_router  # noqa: F401


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_forward_shapes(
    router: DualHeadRouter,
    dummy_inputs: tuple[jax.Array, jax.Array],
) -> None:
    """Output shapes must match (B, max_tasks) and (B, max_envs)."""
    lang, vis = dummy_inputs
    task_probs, env_probs = router(lang, vis)

    assert task_probs.shape == (BATCH, MAX_TASKS)
    assert env_probs.shape == (BATCH, MAX_ENVS)


def test_softmax_validity(
    router: DualHeadRouter,
    dummy_inputs: tuple[jax.Array, jax.Array],
) -> None:
    """Both heads must produce valid probability distributions."""
    lang, vis = dummy_inputs
    task_probs, env_probs = router(lang, vis)

    # Non-negative
    assert jnp.all(task_probs >= 0.0)
    assert jnp.all(env_probs >= 0.0)

    # Each row sums to ~1
    npt.assert_allclose(jnp.sum(task_probs, axis=-1), jnp.ones(BATCH), atol=1e-5)
    npt.assert_allclose(jnp.sum(env_probs, axis=-1), jnp.ones(BATCH), atol=1e-5)


def test_independence_diagnostic(
    router: DualHeadRouter,
) -> None:
    """mutual_information_estimate returns a non-negative scalar."""
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    lang = jax.random.normal(k1, (8, LANG_DIM))
    vis = jax.random.normal(k2, (8, VIS_DIM))

    mi = router.mutual_information_estimate(lang, vis)

    assert mi.shape == ()  # scalar
    assert float(mi) >= 0.0


def test_counterfactual_task_swap(
    router: DualHeadRouter,
) -> None:
    """Different lang embeddings → different task routing distributions."""
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)

    vis = jax.random.normal(k1, (1, VIS_DIM))  # shared visual

    # Two very different language embeddings (large scale for clear separation)
    lang_a = jax.random.normal(k2, (1, LANG_DIM)) * 5.0
    lang_b = jax.random.normal(k3, (1, LANG_DIM)) * 5.0

    probs_a = router.counterfactual_task(lang_a, vis)
    probs_b = router.counterfactual_task(lang_b, vis)

    # Distributions should differ (not identical)
    assert not jnp.allclose(probs_a, probs_b, atol=1e-5)


def test_counterfactual_env_swap(
    router: DualHeadRouter,
) -> None:
    """Different visual summaries → different env routing distributions."""
    key = jax.random.PRNGKey(13)
    k1, k2, k3 = jax.random.split(key, 3)

    lang = jax.random.normal(k1, (1, LANG_DIM))  # shared language

    vis_a = jax.random.normal(k2, (1, VIS_DIM)) * 5.0
    vis_b = jax.random.normal(k3, (1, VIS_DIM)) * 5.0

    probs_a = router.counterfactual_env(lang, vis_a)
    probs_b = router.counterfactual_env(lang, vis_b)

    assert not jnp.allclose(probs_a, probs_b, atol=1e-5)


def test_route_returns_indices(
    router: DualHeadRouter,
    dummy_inputs: tuple[jax.Array, jax.Array],
) -> None:
    """route() must return integer index arrays in valid ranges."""
    lang, vis = dummy_inputs
    task_idx, env_idx = router.route(lang, vis, task_ids=["t0", "t1"], env_ids=["e0", "e1"])

    # Shape
    assert task_idx.shape == (BATCH,)
    assert env_idx.shape == (BATCH,)

    # Integer dtype
    assert jnp.issubdtype(task_idx.dtype, jnp.integer)
    assert jnp.issubdtype(env_idx.dtype, jnp.integer)

    # Valid range
    assert jnp.all(task_idx >= 0)
    assert jnp.all(task_idx < MAX_TASKS)
    assert jnp.all(env_idx >= 0)
    assert jnp.all(env_idx < MAX_ENVS)


def test_active_masks(
    router: DualHeadRouter,
    dummy_inputs: tuple[jax.Array, jax.Array],
) -> None:
    """Masked-out slots should receive ~0 probability."""
    lang, vis = dummy_inputs

    # Only first 2 of 4 tasks active, first 1 of 3 envs active
    task_mask = make_active_mask(2, MAX_TASKS)  # [T, T, F, F]
    env_mask = make_active_mask(1, MAX_ENVS)  # [T, F, F]

    task_probs, env_probs = router(lang, vis, task_mask=task_mask, env_mask=env_mask)

    # Inactive task slots should be ~0
    assert jnp.all(task_probs[:, 2:] < 1e-6)
    # Inactive env slots should be ~0
    assert jnp.all(env_probs[:, 1:] < 1e-6)

    # Active slots still form valid distributions
    npt.assert_allclose(jnp.sum(task_probs, axis=-1), jnp.ones(BATCH), atol=1e-5)
    npt.assert_allclose(jnp.sum(env_probs, axis=-1), jnp.ones(BATCH), atol=1e-5)


def test_route_soft_is_call_alias(
    router: DualHeadRouter,
    dummy_inputs: tuple[jax.Array, jax.Array],
) -> None:
    """route_soft must return identical results to __call__."""
    lang, vis = dummy_inputs

    call_task, call_env = router(lang, vis)
    soft_task, soft_env = router.route_soft(lang, vis)

    npt.assert_array_equal(call_task, soft_task)
    npt.assert_array_equal(call_env, soft_env)
