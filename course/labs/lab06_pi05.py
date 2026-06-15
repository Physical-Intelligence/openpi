"""Lab 06 — pi0.5 specifics: discrete state tokens + adaRMS.

Implement the pi0.5 discrete-state tokenization and adaptive RMSNorm, and verify against
openpi.

Run:  uv run python course/labs/lab06_pi05.py

NOTE: Part 1's cross-check downloads the PaliGemma tokenizer (gs://big_vision/...). If you
have no network/credentials, skip the download cross-check and just test the digitize step
(which needs no download).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Part 1: discrete state tokenization (Module 06 §1)
# ----------------------------------------------------------------------------
def discretize_state(state: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Bin a state vector assumed in [-1, 1] into n_bins. Match tokenizer.py:26 /
    np.digitize(state, bins=np.linspace(-1, 1, n_bins+1)[:-1]) - 1."""
    # TODO(you): implement.
    raise NotImplementedError


def build_pi05_prompt(prompt: str, state: np.ndarray) -> str:
    """Return the pi0.5 prompt string. Match tokenizer.py:28:
        f"Task: {cleaned}, State: {state_str};\\nAction: "
    where cleaned strips and replaces '_' and newlines with spaces, and state_str is the
    space-joined discretized state ints."""
    # TODO(you): implement.
    raise NotImplementedError


def check_part1():
    state = np.array([-1.0, -0.5, 0.0, 0.5, 0.999], dtype=np.float32)
    binned = discretize_state(state)
    assert binned.min() >= 0 and binned.max() <= 255
    assert binned[0] == 0 and binned[2] == 128, binned  # -1 -> bin 0, 0 -> bin 128
    print("Part 1a OK: digitize matches the [-1,1]->256-bin convention.")

    # Optional cross-check against the real tokenizer (needs download):
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        tok = PaligemmaTokenizer(max_len=200)
        ids, mask = tok.tokenize("pick up the cup", state)
        print("Part 1b OK: real tokenizer produced", int(mask.sum()), "real tokens.")
        # Your build_pi05_prompt(...) should equal the string the tokenizer encodes.
    except Exception as e:  # noqa: BLE001
        print("Part 1b skipped (no tokenizer download):", type(e).__name__)


# ----------------------------------------------------------------------------
# Part 2: adaptive RMSNorm (Module 06 §1b, Module 03 §2)
# ----------------------------------------------------------------------------
# Implement rms_norm(x, cond):
#   var = mean(x_f32**2, -1, keepdims); normed = x * rsqrt(var + 1e-6)
#   if cond is None: return normed * (1 + scale_param), None        # standard RMSNorm
#   else: scale, shift, gate = split(Dense(3d, zeros_init)(cond), 3)
#         return normed * (1 + scale) + shift, gate                 # adaRMS
# Test: with cond=None and scale_param=0, output == normed (i.e. reduces to standard).

def check_part2():
    print("Part 2: implement rms_norm and assert cond=None reduces to standard RMSNorm.")


if __name__ == "__main__":
    check_part1()
    check_part2()
    print("\nDiscussion (write your answer as a comment): where in compute_loss would a "
          "stop_gradient approximate knowledge insulation, and what would it cost/buy?")
