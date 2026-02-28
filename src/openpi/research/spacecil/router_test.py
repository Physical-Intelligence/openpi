"""Tests for router module."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.research.spacecil.router import TaskRouter
from openpi.research.spacecil.router import make_active_mask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Deterministic RNGs for reproducible tests."""
    return nnx.Rngs(0)


@pytest.fixture
def router(rngs: nnx.Rngs) -> TaskRouter:
    """Default router with input_dim=64, hidden_dim=32, max_tasks=8, 2 layers."""
    return TaskRouter(input_dim=64, hidden_dim=32, max_tasks=8, num_layers=2, rngs=rngs)


def _rand_inputs(batch_size: int, lang_dim: int, vis_dim: int, *, key: int = 42) -> tuple[jax.Array, jax.Array]:
    """Generate random language and visual embeddings."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    lang = jax.random.normal(k1, (batch_size, lang_dim))
    vis = jax.random.normal(k2, (batch_size, vis_dim))
    return lang, vis


# ---------------------------------------------------------------------------
# Forward pass shape tests
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_output_shape(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        probs = router(lang, vis)
        assert probs.shape == (4, 8)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, rngs: nnx.Rngs, batch_size: int) -> None:
        r = TaskRouter(input_dim=64, hidden_dim=32, max_tasks=8, rngs=rngs)
        lang, vis = _rand_inputs(batch_size, 32, 32)
        probs = r(lang, vis)
        assert probs.shape == (batch_size, 8)

    @pytest.mark.parametrize("input_dim", [64, 256, 4096])
    def test_input_dimensions(self, rngs: nnx.Rngs, input_dim: int) -> None:
        r = TaskRouter(input_dim=input_dim, hidden_dim=64, max_tasks=4, rngs=rngs)
        half = input_dim // 2
        lang, vis = _rand_inputs(2, half, half)
        probs = r(lang, vis)
        assert probs.shape == (2, 4)


# ---------------------------------------------------------------------------
# Probability distribution tests
# ---------------------------------------------------------------------------


class TestProbabilities:
    def test_softmax_sums_to_one(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        probs = router(lang, vis)
        sums = jnp.sum(probs, axis=-1)
        np.testing.assert_allclose(sums, jnp.ones(4), atol=1e-5)

    def test_all_probs_non_negative(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        probs = router(lang, vis)
        assert jnp.all(probs >= 0.0)

    def test_masked_probs_sum_to_one(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        mask = make_active_mask(3, 8)
        probs = router(lang, vis, active_mask=mask)
        sums = jnp.sum(probs, axis=-1)
        np.testing.assert_allclose(sums, jnp.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Masking tests
# ---------------------------------------------------------------------------


class TestMasking:
    def test_inactive_slots_near_zero(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        mask = make_active_mask(3, 8)
        probs = router(lang, vis, active_mask=mask)
        # Slots 3..7 should have near-zero probability
        inactive_probs = probs[:, 3:]
        assert jnp.all(inactive_probs < 1e-6)

    def test_single_task_mask_argmax_zero(self, router: TaskRouter) -> None:
        """With only 1 active task, argmax must always return 0."""
        lang, vis = _rand_inputs(8, 32, 32, key=99)
        mask = make_active_mask(1, 8)
        indices = router.route_hard(lang, vis, active_mask=mask)
        np.testing.assert_array_equal(indices, jnp.zeros(8, dtype=jnp.int32))

    def test_three_task_entropy_bounded(self, router: TaskRouter) -> None:
        """With 3 active tasks, entropy should be in [0, log(3)]."""
        lang, vis = _rand_inputs(16, 32, 32, key=7)
        mask = make_active_mask(3, 8)
        probs = router(lang, vis, active_mask=mask)
        # Only consider active slots for entropy
        active_probs = probs[:, :3]
        # Clip for numerical stability in log
        eps = 1e-10
        entropy = -jnp.sum(active_probs * jnp.log(active_probs + eps), axis=-1)
        max_entropy = jnp.log(3.0)
        assert jnp.all(entropy >= -1e-5), f"Negative entropy: {entropy}"
        assert jnp.all(entropy <= max_entropy + 1e-5), f"Entropy exceeds log(3): {entropy}"

    def test_no_mask_uses_all_slots(self, router: TaskRouter) -> None:
        """Without a mask, all max_tasks slots participate."""
        lang, vis = _rand_inputs(4, 32, 32)
        probs = router(lang, vis, active_mask=None)
        # At least some slots beyond index 0 should have non-trivial probability
        assert jnp.sum(probs[:, 1:]) > 0.01


# ---------------------------------------------------------------------------
# All-masked edge case
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_tasks_masked(self, router: TaskRouter) -> None:
        """With all tasks masked, softmax over all -1e9 → uniform over max_tasks."""
        lang, vis = _rand_inputs(2, 32, 32)
        mask = make_active_mask(0, 8)
        probs = router(lang, vis, active_mask=mask)
        # Softmax of equal values → uniform distribution
        expected = jnp.ones((2, 8)) / 8.0
        np.testing.assert_allclose(probs, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Routing method tests
# ---------------------------------------------------------------------------


class TestRoutingMethods:
    def test_route_hard_returns_int(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        indices = router.route_hard(lang, vis)
        assert indices.shape == (4,)
        assert jnp.issubdtype(indices.dtype, jnp.integer)

    def test_route_hard_within_active_range(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(8, 32, 32, key=123)
        mask = make_active_mask(3, 8)
        indices = router.route_hard(lang, vis, active_mask=mask)
        assert jnp.all(indices < 3)

    def test_route_soft_equals_call(self, router: TaskRouter) -> None:
        """route_soft should return identical output to __call__."""
        lang, vis = _rand_inputs(4, 32, 32)
        mask = make_active_mask(5, 8)
        call_out = router(lang, vis, active_mask=mask)
        soft_out = router.route_soft(lang, vis, active_mask=mask)
        np.testing.assert_array_equal(call_out, soft_out)

    def test_route_hard_consistent_with_soft(self, router: TaskRouter) -> None:
        lang, vis = _rand_inputs(4, 32, 32)
        soft = router.route_soft(lang, vis)
        hard = router.route_hard(lang, vis)
        expected = jnp.argmax(soft, axis=-1)
        np.testing.assert_array_equal(hard, expected)


# ---------------------------------------------------------------------------
# make_active_mask tests
# ---------------------------------------------------------------------------


class TestMakeActiveMask:
    def test_basic_mask(self) -> None:
        mask = make_active_mask(3, 8)
        expected = jnp.array([True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_zero_active(self) -> None:
        mask = make_active_mask(0, 4)
        np.testing.assert_array_equal(mask, jnp.zeros(4, dtype=bool))

    def test_all_active(self) -> None:
        mask = make_active_mask(8, 8)
        np.testing.assert_array_equal(mask, jnp.ones(8, dtype=bool))

    def test_shape(self) -> None:
        mask = make_active_mask(5, 16)
        assert mask.shape == (16,)
        assert mask.dtype == jnp.bool_


# ---------------------------------------------------------------------------
# Deterministic initialization test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_rngs_same_output(self) -> None:
        """Two routers initialized with the same seed should produce identical outputs."""
        r1 = TaskRouter(input_dim=64, hidden_dim=32, max_tasks=4, rngs=nnx.Rngs(42))
        r2 = TaskRouter(input_dim=64, hidden_dim=32, max_tasks=4, rngs=nnx.Rngs(42))
        lang, vis = _rand_inputs(2, 32, 32)
        out1 = r1(lang, vis)
        out2 = r2(lang, vis)
        np.testing.assert_array_equal(out1, out2)

    def test_different_rngs_different_output(self) -> None:
        """Two routers with different seeds should (almost surely) differ."""
        r1 = TaskRouter(input_dim=64, hidden_dim=32, max_tasks=4, rngs=nnx.Rngs(0))
        r2 = TaskRouter(input_dim=64, hidden_dim=32, max_tasks=4, rngs=nnx.Rngs(999))
        lang, vis = _rand_inputs(2, 32, 32)
        out1 = r1(lang, vis)
        out2 = r2(lang, vis)
        assert not jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Constructor variations
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_single_layer(self, rngs: nnx.Rngs) -> None:
        r = TaskRouter(input_dim=128, hidden_dim=64, max_tasks=4, num_layers=1, rngs=rngs)
        lang, vis = _rand_inputs(2, 64, 64)
        probs = r(lang, vis)
        assert probs.shape == (2, 4)
        np.testing.assert_allclose(jnp.sum(probs, axis=-1), jnp.ones(2), atol=1e-5)

    def test_many_layers(self, rngs: nnx.Rngs) -> None:
        r = TaskRouter(input_dim=32, hidden_dim=16, max_tasks=2, num_layers=5, rngs=rngs)
        lang, vis = _rand_inputs(1, 16, 16)
        probs = r(lang, vis)
        assert probs.shape == (1, 2)
        np.testing.assert_allclose(jnp.sum(probs, axis=-1), jnp.ones(1), atol=1e-5)

    def test_large_max_tasks(self, rngs: nnx.Rngs) -> None:
        r = TaskRouter(input_dim=64, hidden_dim=32, max_tasks=128, rngs=rngs)
        lang, vis = _rand_inputs(2, 32, 32)
        mask = make_active_mask(3, 128)
        probs = r(lang, vis, active_mask=mask)
        assert probs.shape == (2, 128)
        # Inactive slots negligible
        assert jnp.sum(probs[:, 3:]) < 1e-4
