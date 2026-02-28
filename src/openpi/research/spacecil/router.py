"""Language-visual router for task adapter selection.

Routes to task adapters without oracle task ID at inference time.
Input: language embedding + visual summary.
Output: routing weights over registered task adapters.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class TaskRouter(nnx.Module):
    """Lightweight MLP router that maps concatenated language + visual embeddings
    to a softmax distribution over task adapters.

    Architecture:
        input -> [Linear -> LayerNorm -> GELU] x num_layers -> Linear -> masked softmax

    The output head is pre-allocated to ``max_tasks`` and inactive slots are masked
    to ``-1e9`` before softmax so the router can grow without resizing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        max_tasks: int = 16,
        num_layers: int = 2,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.max_tasks = max_tasks

        # Hidden MLP blocks: Linear → LayerNorm → GELU
        self.hidden_layers: list[nnx.Linear] = []
        self.layer_norms: list[nnx.LayerNorm] = []
        in_dim = input_dim
        for _ in range(num_layers):
            self.hidden_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            self.layer_norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        # Output projection to max_tasks logits
        self.output_proj = nnx.Linear(hidden_dim, max_tasks, rngs=rngs)

    def __call__(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        active_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass returning softmax routing probabilities.

        Args:
            lang_embedding: Language embedding, shape ``(B, lang_dim)``.
            visual_summary: Visual summary embedding, shape ``(B, visual_dim)``.
            active_mask: Boolean mask of shape ``(max_tasks,)`` — ``True`` for
                registered (active) tasks. If ``None``, all tasks are active.

        Returns:
            Softmax probabilities of shape ``(B, max_tasks)``.
        """
        x = jnp.concatenate([lang_embedding, visual_summary], axis=-1)

        for linear, ln in zip(self.hidden_layers, self.layer_norms, strict=True):
            x = linear(x)
            x = ln(x)
            x = jax.nn.gelu(x)

        logits = self.output_proj(x)  # (B, max_tasks)

        if active_mask is not None:
            # Mask inactive task slots to large negative so softmax → ~0
            logits = jnp.where(active_mask, logits, jnp.float32(-1e9))

        return jax.nn.softmax(logits, axis=-1)

    def route_hard(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        active_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Argmax routing — returns selected task index per batch element.

        Returns:
            Integer indices of shape ``(B,)``.
        """
        probs = self(lang_embedding, visual_summary, active_mask)
        return jnp.argmax(probs, axis=-1)

    def route_soft(
        self,
        lang_embedding: jax.Array,
        visual_summary: jax.Array,
        active_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Soft routing — returns full probability distribution.

        Returns:
            Probabilities of shape ``(B, max_tasks)``.
        """
        return self(lang_embedding, visual_summary, active_mask)


def make_active_mask(num_active: int, max_tasks: int) -> jax.Array:
    """Create a boolean mask with the first ``num_active`` entries True.

    Args:
        num_active: Number of currently registered tasks.
        max_tasks: Total pre-allocated task slots.

    Returns:
        Boolean array of shape ``(max_tasks,)``.
    """
    return jnp.arange(max_tasks) < num_active
