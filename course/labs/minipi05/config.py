"""MiniConfig for the capstone minipi05 model. Keep everything tiny."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class MiniConfig:
    # Model dims (tiny on purpose).
    width: int = 128          # shared transformer width
    depth: int = 4
    num_heads: int = 2
    head_dim: int = 64
    mlp_dim: int = 256

    # Vision.
    image_size: int = 224
    patch: int = 32           # 224/32 = 7 -> 49 tokens per view

    # Action head.
    action_dim: int = 8
    action_horizon: int = 10

    # Language / state.
    vocab_size: int = 1024
    max_token_len: int = 64

    # The pi0.5 toggle (Module 06). True -> discrete state in prompt + adaRMS.
    pi05: bool = True

    # Flow inference.
    num_steps: int = 10
