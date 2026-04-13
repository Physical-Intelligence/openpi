"""MEM (Multi-Scale Embodied Memory) temporal attention module.

Implements temporal self-attention layers that allow each spatial position
(image patch) to attend to the same spatial position across multiple time frames.
Inserted after spatial attention at every N-th transformer layer.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemporalPositionEncoding(nn.Module):
    """Sinusoidal temporal position encoding for multi-frame inputs.

    Generates position embeddings based on frame index * frame_interval (in seconds),
    using sine-cosine encoding with logarithmically spaced frequencies.
    """

    def __init__(self, d_model: int, max_frames: int = 16, min_period: float = 0.01, max_period: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.max_frames = max_frames
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, num_frames: int, frame_interval: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Compute temporal position encoding.

        Args:
            num_frames: Number of frames N.
            frame_interval: Time interval between frames in seconds.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tensor of shape [1, N, 1, d_model] broadcastable over batch and spatial dims.
        """
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by 2")

        # Frame timestamps: [0, interval, 2*interval, ..., (N-1)*interval]
        frame_times = torch.arange(num_frames, device=device, dtype=torch.float64) * frame_interval

        # Logarithmically spaced frequencies
        fraction = torch.linspace(0.0, 1.0, self.d_model // 2, dtype=torch.float64, device=device)
        period = self.min_period * (self.max_period / self.min_period) ** fraction

        # Compute sinusoidal encoding
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * frame_times[:, None]  # [N, d_model//2]
        encoding = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)  # [N, d_model]

        return encoding.to(dtype=dtype).unsqueeze(0).unsqueeze(2)  # [1, N, 1, d_model]


class TemporalAttentionLayer(nn.Module):
    """Temporal self-attention layer for MEM.

    Applies multi-head self-attention across the temporal dimension for image tokens,
    allowing each spatial position to attend to the same position across all frames.

    Non-image tokens (language, action) pass through unchanged.
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        self.temporal_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

        self.temporal_pos_enc = TemporalPositionEncoding(d_model)

        # Zero-initialize o_proj so temporal attention starts as identity (residual passthrough)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        hidden_states: Tensor,
        num_image_tokens_per_frame: int,
        num_frames: int,
        frame_interval: float,
    ) -> Tensor:
        """Apply temporal attention to image tokens.

        Args:
            hidden_states: [B, seq_len, D] -- prefix embeddings (image tokens + language tokens).
            num_image_tokens_per_frame: Number of image tokens per frame (e.g. 768 for 3 cameras * 256 patches).
            num_frames: Number of frames N.
            frame_interval: Time interval between frames in seconds.

        Returns:
            [B, seq_len, D] -- hidden states with temporal attention applied to image tokens.
        """
        B, seq_len, D = hidden_states.shape
        num_image_tokens_total = num_image_tokens_per_frame * num_frames
        S = num_image_tokens_per_frame  # spatial positions per frame

        # Extract image tokens and non-image tokens
        img_tokens = hidden_states[:, :num_image_tokens_total, :]  # [B, N*S, D]
        other_tokens = hidden_states[:, num_image_tokens_total:, :]  # [B, rest, D]

        # Save for residual
        img_residual = img_tokens

        # Apply layer norm
        img_tokens = self.temporal_norm(img_tokens.float()).to(hidden_states.dtype)

        # Reshape to group same spatial position across frames: [B, N, S, D] -> [B*S, N, D]
        img_tokens = img_tokens.view(B, num_frames, S, D)

        # Add temporal position encoding
        pos_enc = self.temporal_pos_enc(num_frames, frame_interval, hidden_states.device, hidden_states.dtype)
        img_tokens = img_tokens + pos_enc  # [B, N, S, D] + [1, N, 1, D] -> [B, N, S, D]

        # Reshape: [B, N, S, D] -> [B, S, N, D] -> [B*S, N, D]
        img_tokens = img_tokens.permute(0, 2, 1, 3).reshape(B * S, num_frames, D)

        # Compute Q, K, V
        hidden_shape = (B * S, num_frames, self.num_heads, self.head_dim)
        q = self.q_proj(img_tokens).view(hidden_shape).transpose(1, 2)  # [B*S, heads, N, head_dim]
        k = self.k_proj(img_tokens).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(img_tokens).view(hidden_shape).transpose(1, 2)

        # Self-attention across temporal dimension (no mask -- all frames attend to all)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B*S, heads, N, N]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [B*S, heads, N, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B * S, num_frames, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)  # [B*S, N, D]

        # Reshape back to [B, N*S, D]: [B*S, N, D] -> [B, S, N, D] -> [B, N, S, D] -> [B, N*S, D]
        attn_output = attn_output.view(B, S, num_frames, D).permute(0, 2, 1, 3).reshape(B, num_frames * S, D)

        # Residual connection
        img_tokens_out = img_residual + attn_output

        # Concatenate with non-image tokens
        return torch.cat([img_tokens_out, other_tokens], dim=1)
