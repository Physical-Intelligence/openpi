"""Dual-head router for separate task and environment routing.

Extends SpaceCIL's TaskRouter with a second environment routing head.
Includes independence diagnostics and counterfactual swap support.

The DualHeadRouter wraps two independent :class:`TaskRouter` instances —
one for task-adapter selection and one for environment-adapter selection.
Both heads receive the same concatenated ``(lang_embedding, visual_summary)``
input but maintain completely independent parameters.

Key diagnostics:
- **Counterfactual queries** let you probe what one head predicts while
  ignoring the other, useful for ablation studies.
- **Mutual information estimate** quantifies how entangled the two routing
  distributions are over a batch — lower MI means better factorisation.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.research.spacecil.router import TaskRouter
from openpi.research.spacecil.router import make_active_mask

__all__ = ["DualHeadRouter", "make_active_mask"]


class DualHeadRouter(nnx.Module):
    """Two-headed router producing independent task and environment distributions.

    Architecture (per head)::

        concat(lang, vis) -> [Linear -> LayerNorm -> GELU] x num_layers -> Linear -> masked softmax

    The task head outputs over ``max_tasks`` slots while the environment head
    outputs over ``max_envs`` slots.  Weights are **not** shared.

    Args:
        input_dim: Dimensionality of the concatenated input
            ``lang_dim + visual_dim``.  Must be identical for both heads.
        max_tasks: Pre-allocated task-adapter bank capacity.
        max_envs: Pre-allocated environment-adapter bank capacity.
        hidden_dim: Hidden size for each MLP block inside both heads.
        num_layers: Number of ``(Linear → LayerNorm → GELU)`` blocks per head.
        rngs: Flax NNX random number generators for parameter initialisation.
    """

    def __init__(
        self,
        input_dim: int,
        max_tasks: int = 16,
        max_envs: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.task_head = TaskRouter(
            input_dim,
            hidden_dim,
            max_tasks,
            num_layers,
            rngs=rngs,
        )
        self.env_head = TaskRouter(
            input_dim,
            hidden_dim,
            max_envs,
            num_layers,
            rngs=rngs,
        )
        self.max_tasks = max_tasks
        self.max_envs = max_envs

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        task_mask: jax.Array | None = None,
        env_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning softmax routing probabilities for both heads.

        Args:
            lang_embedding: Language embedding of shape ``(B, lang_dim)``.
            visual_summary: Visual summary of shape ``(B, vis_dim)``.
            task_mask: Optional boolean mask ``(max_tasks,)`` — ``True`` for
                active task slots.
            env_mask: Optional boolean mask ``(max_envs,)`` — ``True`` for
                active environment slots.

        Returns:
            ``(task_probs, env_probs)`` where ``task_probs`` has shape
            ``(B, max_tasks)`` and ``env_probs`` has shape ``(B, max_envs)``.
        """
        task_probs = self.task_head(lang_embedding, visual_summary, task_mask)
        env_probs = self.env_head(lang_embedding, visual_summary, env_mask)
        return task_probs, env_probs

    # ------------------------------------------------------------------
    # Hard / soft routing helpers
    # ------------------------------------------------------------------

    def route(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        task_ids: list[str],  # kept for API symmetry
        env_ids: list[str],
        task_mask: jax.Array | None = None,
        env_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Argmax routing for both heads.

        ``task_ids`` and ``env_ids`` are accepted for interface compatibility
        (e.g. logging) but are not used for the actual routing computation.

        Returns:
            ``(task_indices, env_indices)`` — integer arrays of shape ``(B,)``.
        """
        task_probs, env_probs = self(lang_embedding, visual_summary, task_mask, env_mask)
        return jnp.argmax(task_probs, axis=-1), jnp.argmax(env_probs, axis=-1)

    def route_soft(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        task_mask: jax.Array | None = None,
        env_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Alias for :meth:`__call__` — soft routing probabilities."""
        return self(lang_embedding, visual_summary, task_mask, env_mask)

    # ------------------------------------------------------------------
    # Counterfactual probes
    # ------------------------------------------------------------------

    def counterfactual_task(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        task_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Route the task head only, ignoring the environment head.

        Useful for ablation: see what task the router picks regardless of
        environment conditioning.

        Returns:
            Task routing probabilities of shape ``(B, max_tasks)``.
        """
        return self.task_head(lang_embedding, visual_summary, task_mask)

    def counterfactual_env(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        env_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Route the environment head only, ignoring the task head.

        Useful for ablation: see what environment the router picks regardless
        of task conditioning.

        Returns:
            Environment routing probabilities of shape ``(B, max_envs)``.
        """
        return self.env_head(lang_embedding, visual_summary, env_mask)

    # ------------------------------------------------------------------
    # Independence diagnostic
    # ------------------------------------------------------------------

    def mutual_information_estimate(
        self,
        lang_embeddings: jax.Array,
        visual_summaries: jax.Array,
        task_mask: jax.Array | None = None,
        env_mask: jax.Array | None = None,
    ) -> jax.Array:
        r"""Estimate mutual information between task and env routing.

        Computes MI from the empirical joint:

        .. math::

            P(t, e) = \frac{1}{B} \sum_b p^{\text{task}}_b(t) \, p^{\text{env}}_b(e)

        Then:

        .. math::

            \text{MI} = \sum_{t,e} P(t,e) \log \frac{P(t,e)}{P(t)\,P(e) + \epsilon}

        Lower MI → more factorised (independent) routing.

        Args:
            lang_embeddings: ``(B, lang_dim)`` batch of language embeddings.
            visual_summaries: ``(B, vis_dim)`` batch of visual summaries.
            task_mask: Optional ``(max_tasks,)`` boolean mask.
            env_mask: Optional ``(max_envs,)`` boolean mask.

        Returns:
            Scalar ``jax.Array`` — estimated mutual information (nats).
        """
        task_probs, env_probs = self(lang_embeddings, visual_summaries, task_mask, env_mask)
        # task_probs: (B, T), env_probs: (B, E)

        # Empirical joint P(t, e) = (1/B) sum_b  task_probs_b^T outer env_probs_b
        # Equivalent to: einsum("bt,be->te", task_probs, env_probs) / B
        batch_size = task_probs.shape[0]
        joint = jnp.einsum("bt,be->te", task_probs, env_probs) / batch_size

        # Marginals
        p_task = jnp.sum(joint, axis=1)  # (T,)
        p_env = jnp.sum(joint, axis=0)  # (E,)

        # Outer product of marginals
        outer = jnp.outer(p_task, p_env)  # (T, E)

        # MI = sum P(t,e) * log( P(t,e) / (P(t)*P(e) + eps) + eps )
        eps = 1e-10
        return jnp.sum(joint * jnp.log(joint / (outer + eps) + eps))

