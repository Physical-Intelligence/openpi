"""Lab 03 — The two-expert transformer.

Implement make_attn_mask and a single multi-expert Block. Verify the mask against the
real pi0.make_attn_mask and the insulation property (prefix outputs independent of suffix).

Run:  uv run python course/labs/lab03_transformer.py
"""

import jax
import jax.numpy as jnp

from openpi.models import pi0 as _pi0


# ----------------------------------------------------------------------------
# Part 1: make_attn_mask (Module 03 §4)
# ----------------------------------------------------------------------------
def make_attn_mask(input_mask, mask_ar):
    """Token i may attend to j iff cumsum(mask_ar)[j] <= cumsum(mask_ar)[i] AND both
    are valid (input_mask). Return bool [B, N, N]. (See pi0.make_attn_mask.)"""
    # TODO(you): implement.
    raise NotImplementedError


def check_part1():
    input_mask = jnp.array([[True, True, True, False]])
    mask_ar = jnp.array([[False, False, True, False]])
    mine = make_attn_mask(input_mask, mask_ar)
    ref = _pi0.make_attn_mask(input_mask, mask_ar)
    assert jnp.array_equal(mine, ref), f"\nmine={mine}\nref={ref}"
    # Pure causal example from the docstring:
    ar = jnp.array([[True, True, True]])
    im = jnp.array([[True, True, True]])
    assert jnp.array_equal(make_attn_mask(im, ar), _pi0.make_attn_mask(im, ar))
    print("Part 1 OK: make_attn_mask matches openpi (bidirectional, causal, blockwise).")


# ----------------------------------------------------------------------------
# Part 2: a multi-expert Block (Module 03 §2-3)
# ----------------------------------------------------------------------------
# Implement a Block that takes xs = [x_prefix, x_suffix] (either may be None), does:
#   per-expert RMSNorm -> shared attention (concat over experts) -> gated residual
#   -> per-expert RMSNorm -> per-expert FFN -> gated residual.
# Use width per expert from a small config. RoPE optional for this lab (positions given).
#
# You can build this in flax.linen mirroring gemma.Block, or in plain functions for
# clarity. The key tests:
#   (a) shapes preserved per expert,
#   (b) with a mask where suffix cannot influence prefix, prefix output is invariant
#       to the suffix CONTENTS (the knowledge-insulation property).

def check_part2():
    # TODO(you): once your Block is implemented, fill in this test:
    # 1. Build two random expert inputs x_pre [b, Sp, d0], x_suf [b, Ss, d1] (same d ok).
    # 2. Build attn_mask via make_attn_mask with prefix ar=all False, suffix ar=[T,F,...].
    # 3. Run the block twice, changing ONLY x_suf the second time.
    # 4. assert the prefix output is identical both times (within fp tolerance).
    print("Part 2: implement your Block, then assert the insulation property here.")


if __name__ == "__main__":
    check_part1()
    check_part2()
    print("\nStretch: wrap your block in nn.scan over depth and reproduce gemma.Module "
          "('dummy' variant) output shapes for [prefix, suffix] inputs.")
