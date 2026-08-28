"""compare_pi05_vs_paligemma.py — same image + same prompt, both checkpoints.

Loads pi05_base (openpi/JAX) and raw PaliGemma-3b-mix-224 (HF/torch) and
runs greedy decode on the SAME 224×224 cam_high frame with the SAME prompt.
Prints a side-by-side terminal report and saves overlay PNGs so the
asymmetry is reproducible in one command.

Why this script: the load-bearing claim is "π0.5 Stage-2 destroyed
PaliGemma's detection." Showing both outputs together — same input, two
weights — is the cleanest single piece of evidence.

Cost: ~5-10 min wall-clock (first JIT of pi05_base dominates; raw PaliGemma
on CPU is fast). The model files are already cached on disk.

Usage:
  python experiments/diagnose/compare_pi05_vs_paligemma.py \\
      --image experiments/results/lm_head_probe/_inputs/data_1__ep0__f0045.png \\
      --prompt "detect box"
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

log = logging.getLogger("compare")
LOC_RE = re.compile(r"<loc(\d{4})>")
IMG_SIZE = 224
LOC_OFFSET = 256_000
LOC_COUNT = 1_024


@contextmanager
def stage(name: str):
    log.info("▶ %s ...", name)
    sys.stdout.flush(); sys.stderr.flush()
    t0 = time.time()
    try:
        yield
    finally:
        log.info("✓ %s done (%.1fs)", name, time.time() - t0)
        sys.stdout.flush(); sys.stderr.flush()


# ---------------------------------------------------------------------------
# pi05_base side (JAX / openpi)
# ---------------------------------------------------------------------------

def _tokenize_for_pi05(prompt: str, spm, max_len: int):
    import probe_lm_head as probe
    ids = spm.encode(prompt, add_bos=True)
    n = min(len(ids), max_len)
    arr = np.zeros(max_len, dtype=np.int32); arr[:n] = ids[:n]
    mask = np.zeros(max_len, dtype=bool); mask[:n] = True
    tp = probe.TokenisedPrompt(arr, mask, text=prompt)
    return tp


def run_pi05(image_float: np.ndarray, prompt: str, max_new_tokens: int) -> dict:
    import probe_lm_head as probe
    import openpi.models.pi0_config as pi0_config

    cfg = pi0_config.Pi0Config(pi05=True)
    with stage("[pi05_base] load checkpoint"):
        model = probe.load_pi05_base()
    with stage("[pi05_base] load SPM"):
        spm = probe.load_spm()

    tp = _tokenize_for_pi05(prompt, spm, cfg.max_token_len)
    obs = probe.make_observation(image_float, tp, state_dim=cfg.action_dim)

    with stage(f"[pi05_base] greedy decode {max_new_tokens} tokens"):
        tokens, per_step = probe.greedy_decode(model, obs, max_new_tokens=max_new_tokens)
    decoded = spm.decode(tokens.tolist())
    free_bboxes = probe.parse_loc_bboxes(tokens)

    with stage("[pi05_base] loc-constrained 4-step decode"):
        loc_tokens, loc_per_step = probe.greedy_decode_loc_constrained(model, obs)
    rels = [int(t) - LOC_OFFSET for t in loc_tokens]
    y1, x1, y2, x2 = (r / LOC_COUNT for r in rels)
    y_lo, y_hi = min(y1, y2), max(y1, y2)
    x_lo, x_hi = min(x1, x2), max(x1, x2)
    constrained_bbox = (y_lo, x_lo, y_hi, x_hi)

    return {
        "prompt": prompt,
        "tokens": [int(t) for t in tokens.tolist()],
        "decoded": decoded,
        "free_bboxes": [list(b) for b in free_bboxes],
        "constrained_bbox": list(constrained_bbox),
        "constrained_loc_rels": rels,
        "step0_loc_rank": per_step[0]["loc_top1_rank"],
        "step0_loc_mass": per_step[0]["loc_mass"],
        "step0_picked_id": per_step[0]["top1_id"],
        "step0_picked_logit": per_step[0]["top1_logit"],
    }


# ---------------------------------------------------------------------------
# raw PaliGemma side (HF / torch)
# ---------------------------------------------------------------------------

def run_paligemma(image_pil: Image.Image, prompt: str, max_new_tokens: int, device: str) -> dict:
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    model_id = "google/paligemma-3b-mix-224"

    with stage(f"[raw PaliGemma] AutoProcessor.from_pretrained({model_id})"):
        processor = AutoProcessor.from_pretrained(model_id)
    with stage(f"[raw PaliGemma] model.from_pretrained → {device}"):
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device).eval()

    with stage("[raw PaliGemma] processor + model.generate"):
        inputs = processor(images=image_pil, text=prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = out[0, input_len:]
        decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)

    nums = [int(m.group(1)) for m in LOC_RE.finditer(decoded)]
    bboxes = []
    for i in range(0, len(nums) - 3, 4):
        bboxes.append([n / 1024 for n in nums[i:i+4]])

    return {
        "prompt": prompt,
        "tokens": [int(t) for t in gen_ids.tolist()],
        "decoded": decoded,
        "bboxes": bboxes,
    }


# ---------------------------------------------------------------------------
# Overlay save + terminal report
# ---------------------------------------------------------------------------

def _save_overlay(img_pil: Image.Image, bboxes, color, label_lines, out_path: Path) -> None:
    pil = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(pil)
    for b in bboxes:
        y1, x1, y2, x2 = b
        y_lo, y_hi = sorted((y1, y2)); x_lo, x_hi = sorted((x1, x2))
        draw.rectangle(
            (x_lo*IMG_SIZE, y_lo*IMG_SIZE, x_hi*IMG_SIZE, y_hi*IMG_SIZE),
            outline=color, width=3,
        )
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = 4
    for line in label_lines:
        draw.text((4, y), line[:80], fill=(0, 255, 0), font=font)
        y += 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def _print_report(image_path: Path, prompt: str, pi: dict | None, pg: dict | None) -> None:
    bar = "=" * 78
    line = "─" * 78
    print()
    print(bar)
    print(" SAME-INPUT COMPARISON  —  pi05_base  vs  raw PaliGemma-3b-mix-224")
    print(bar)
    print(f" image  : {image_path}")
    print(f" prompt : {prompt!r}")

    if pg is not None:
        print()
        print(line)
        print(" raw PaliGemma-3b-mix-224  (no π0.5 fine-tuning)")
        print(line)
        print(f"   token ids  : {pg['tokens']}")
        print(f"   decoded    : {pg['decoded']!r}")
        print(f"   bboxes     : {len(pg['bboxes'])}")
        for i, b in enumerate(pg['bboxes']):
            y1, x1, y2, x2 = b
            print(f"     [{i}] YXYX [0,1] : ({y1:.3f}, {x1:.3f}, {y2:.3f}, {x2:.3f})")
            print(f"         224 px box : y=[{y1*IMG_SIZE:.1f}, {y2*IMG_SIZE:.1f}]  x=[{x1*IMG_SIZE:.1f}, {x2*IMG_SIZE:.1f}]")
        verdict = "DETECTS  ✓" if pg['bboxes'] else "no bbox emitted  ✗"
        print(f"   verdict    : {verdict}")

    if pi is not None:
        print()
        print(line)
        print(" pi05_base  (PaliGemma + π0.5 Stage-2 fine-tune)")
        print(line)
        print(f"   token ids  : {pi['tokens'][:max(8, len(pi['tokens']))]}")
        print(f"   decoded    : {pi['decoded']!r}")
        print(f"   step0 picked id    : {pi['step0_picked_id']}  (logit {pi['step0_picked_logit']:.2f})")
        print(f"   step0 loc-top1 rank: {pi['step0_loc_rank']}  in full 257152-vocab")
        print(f"   step0 loc-range mass: {pi['step0_loc_mass']:.3%}")
        print(f"   free-decode bboxes : {len(pi['free_bboxes'])}  (expect 0 — argmax goes to FAST tokens)")
        y1, x1, y2, x2 = pi['constrained_bbox']
        print(f"   forced-loc bbox YXYX : ({y1:.3f}, {x1:.3f}, {y2:.3f}, {x2:.3f})")
        print(f"     224 px box        : y=[{y1*IMG_SIZE:.1f}, {y2*IMG_SIZE:.1f}]  x=[{x1*IMG_SIZE:.1f}, {x2*IMG_SIZE:.1f}]")
        print(f"     loc rel indices   : y_min={pi['constrained_loc_rels'][0]} x_min={pi['constrained_loc_rels'][1]} y_max={pi['constrained_loc_rels'][2]} x_max={pi['constrained_loc_rels'][3]}")
        verdict = "NO LOC TOKEN in free decode  ✗" if not pi['free_bboxes'] else "free decode emitted bbox  ✓"
        print(f"   verdict    : {verdict}")

    if pi is not None and pg is not None:
        print()
        print(bar)
        print(" DIAGNOSIS")
        print(bar)
        same_image_same_prompt = "same image · same prompt · different checkpoint"
        if pg['bboxes'] and not pi['free_bboxes']:
            print(f"  {same_image_same_prompt}.")
            print("  → raw PaliGemma produces a valid bbox; pi05_base produces zero.")
            print("  → π0.5 Stage-2 fine-tuning overwrote PaliGemma's detection capability.")
            print("    The LM head was retrained to emit FAST action tokens (vocab ≈ 255k-257k).")
        elif not pg['bboxes'] and not pi['free_bboxes']:
            print("  Neither checkpoint produced a bbox. Possible occlusion / OOD scene /")
            print("  prompt-format issue. Try a frame where the object is clearly visible.")
        elif pg['bboxes'] and pi['free_bboxes']:
            print("  Both checkpoints produced bboxes. Compare positions for fidelity.")
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True,
                    help="Path to a 224x224 padded RGB PNG (e.g. from sweep_lm_head's _inputs/).")
    ap.add_argument("--prompt", type=str, default="detect box",
                    help="Raw prompt string fed to BOTH models (a trailing newline will be appended for the pi05 SPM side if not present).")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--device", type=str,
                    default="cpu",
                    help="Device for raw PaliGemma (torch). Use cpu on RTX 5090 until torch ships sm_120 kernels.")
    ap.add_argument("--out_dir", type=Path,
                    default=_HERE.parent / "results" / "compare")
    ap.add_argument("--skip_pi05", action="store_true")
    ap.add_argument("--skip_paligemma", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True, stream=sys.stdout)
    log.info("=" * 70)
    log.info("compare_pi05_vs_paligemma.py start")
    log.info("=" * 70)
    log.info("image=%s  prompt=%r  device=%s", args.image, args.prompt, args.device)

    # Load image once (same array fed to both)
    img_pil = Image.open(args.image).convert("RGB")
    if img_pil.size != (IMG_SIZE, IMG_SIZE):
        log.warning("image is %s, not %dx%d — pi05_base pipeline expects 224x224 padded.",
                    img_pil.size, IMG_SIZE, IMG_SIZE)
    image_float = np.asarray(img_pil, dtype=np.float32) / 255.0 * 2.0 - 1.0   # [-1, 1]

    # pi05_base expects prompts to end with \n (its SPM "start of answer" convention);
    # raw PaliGemma is happy either way but we'll feed the same string.
    prompt_for_pi05 = args.prompt if args.prompt.endswith("\n") else args.prompt + "\n"
    prompt_for_pg  = prompt_for_pi05  # same string, no surprises

    pg_result = None
    pi_result = None
    if not args.skip_paligemma:
        with stage("RAW PALIGEMMA RUN"):
            pg_result = run_paligemma(img_pil, prompt_for_pg, args.max_new_tokens, args.device)
    if not args.skip_pi05:
        with stage("PI05_BASE RUN"):
            pi_result = run_pi05(image_float, prompt_for_pi05, args.max_new_tokens)

    _print_report(args.image, prompt_for_pi05, pi_result, pg_result)

    # Save overlays
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    if pg_result is not None:
        _save_overlay(
            img_pil, pg_result['bboxes'], color=(0, 200, 255),
            label_lines=[f"raw PaliGemma  prompt={prompt_for_pi05!r}",
                         f"decoded: {pg_result['decoded']!r}",
                         f"n_bboxes={len(pg_result['bboxes'])}"],
            out_path=args.out_dir / f"{stem}__raw_paligemma.png",
        )
        log.info("wrote %s", args.out_dir / f"{stem}__raw_paligemma.png")
    if pi_result is not None:
        # Show the loc-constrained box on the pi05 overlay (free decode produces none)
        _save_overlay(
            img_pil, [pi_result['constrained_bbox']], color=(255, 0, 0),
            label_lines=[f"pi05_base  prompt={prompt_for_pi05!r}",
                         f"decoded(free): {pi_result['decoded'][:40]!r}",
                         f"free_bboxes={len(pi_result['free_bboxes'])}  loc_rank step0={pi_result['step0_loc_rank']}",
                         f"forced loc bbox shown in red"],
            out_path=args.out_dir / f"{stem}__pi05_base.png",
        )
        log.info("wrote %s", args.out_dir / f"{stem}__pi05_base.png")

    log.info("=" * 70)
    log.info("compare done — overlays in %s", args.out_dir)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
