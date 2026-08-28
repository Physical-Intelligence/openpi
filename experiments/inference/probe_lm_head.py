"""probe_lm_head.py - Step 0 of PI05_DIAGNOSE_AND_FINETUNE.md section 5.4.

Validates whether the vanilla `pi05_base` checkpoint can already emit
`<loc_NNNN>` tokens for "where is the box" queries on a robot image,
without any training.

For each prompt variant we:
  1. Load pi05_base into a `Pi0` instance.
  2. Build an Observation with a single overhead image (filled into
     `base_0_rgb`) and zero placeholders for the two wrist cameras
     (image_mask=False so they don't influence attention).
  3. Tokenise the prompt according to the variant.
  4. Run the prefix forward pass through `PaliGemma.llm`.
  5. Greedy-decode ~10 tokens from `embedder.decode(...)`, reusing the
     prefix KV cache.
  6. Parse any `<loc_NNNN>` quartets as YXYX bboxes in 1024-bin space.
  7. Save an overlay PNG to `experiments/results/lm_head_probe/`.

Run:
  python experiments/diagnose/probe_lm_head.py \\
      --image /path/to/cam_high.png \\
      --query "detect box" \\
      --max_new_tokens 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_OPENPI_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_OPENPI_ROOT / "src"))

import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0 as pi0_mod  # noqa: E402
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.shared.download as download  # noqa: E402
import openpi.shared.image_tools as image_tools  # noqa: E402

# PaliGemma <loc_NNNN> tokens occupy IDs [256000, 257024) (verified via
# sentencepiece). They encode coordinates in 1024-bin normalised space.
LOC_OFFSET = 256_000
LOC_COUNT = 1_024
IMG_SIZE = _model.IMAGE_RESOLUTION[0]  # 224

PI05_PARAMS_URI = "gs://openpi-assets/checkpoints/pi05_base/params"
PALIGEMMA_SPM_URI = "gs://big_vision/paligemma_tokenizer.model"

log = logging.getLogger("probe_lm_head")


@contextmanager
def stage(name: str):
    """Logs entry/exit + elapsed for a named stage. Flushes stdout/stderr so
    progress shows up in real time even when JAX is compiling."""
    log.info("▶ %s ...", name)
    sys.stdout.flush()
    sys.stderr.flush()
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        log.info("✓ %s done (%.1fs)", name, dt)
        sys.stdout.flush()
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------


def load_pi05_base(checkpoint_path: str | None = None, use_lora: bool = False) -> pi0_mod.Pi0:
    """Load a pi05 checkpoint into a Pi0 model.

    Args:
      checkpoint_path: Either a gs:// URI or a local path to an orbax params
        directory. Default = the public pi05_base checkpoint.
      use_lora: True when loading a trained bbox checkpoint (which has
        LoRA params on PaliGemma). Picks paligemma_variant accordingly so the
        param tree matches the saved checkpoint.
    """
    path = checkpoint_path or PI05_PARAMS_URI
    cfg = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora" if use_lora else "gemma_2b",
        action_expert_variant="gemma_300m",
    )
    log.info("Pi0Config(pi05=True, paligemma=%s, action_expert=%s) built. max_token_len=%d action_dim=%d action_horizon=%d",
             cfg.paligemma_variant, cfg.action_expert_variant,
             cfg.max_token_len, cfg.action_dim, cfg.action_horizon)

    with stage(f"resolving checkpoint URI {path} (uses ~/.cache/openpi if gs://, local path otherwise)"):
        if path.startswith("gs://"):
            params_path = download.maybe_download(path, gs={"token": "anon"})
        else:
            params_path = Path(path)
            if not params_path.exists():
                raise FileNotFoundError(f"checkpoint path does not exist: {params_path}")
        log.info("  local checkpoint path: %s", params_path)

    with stage("orbax restore_params from disk (~12 GB pytree of np.ndarrays)"):
        params = _model.restore_params(params_path, restore_type=np.ndarray)
        n_leaves = len(jax.tree.leaves(params))
        log.info("  restored %d param leaves", n_leaves)

    with stage("Pi0Config.load: nnx.eval_shape(create) + state.replace_by_pure_dict (first JIT trace of model graph)"):
        model = cfg.load(params)
    log.info("Pi0 model populated and ready.")
    return model


def load_spm() -> sentencepiece.SentencePieceProcessor:
    path = download.maybe_download(PALIGEMMA_SPM_URI, gs={"token": "anon"})
    return sentencepiece.SentencePieceProcessor(model_proto=path.open("rb").read())


# ---------------------------------------------------------------------------
# Prompt builders (3 variants from section 5.4 of the doc)
# ---------------------------------------------------------------------------


@dataclass
class TokenisedPrompt:
    ids: np.ndarray  # int32 [max_len]
    mask: np.ndarray  # bool   [max_len]
    text: str  # raw text used (for logging)


def _pad(ids: list[int], max_len: int) -> TokenisedPrompt:
    n = min(len(ids), max_len)
    out = np.zeros(max_len, dtype=np.int32)
    out[:n] = np.asarray(ids[:n], dtype=np.int32)
    mask = np.zeros(max_len, dtype=bool)
    mask[:n] = True
    return TokenisedPrompt(out, mask, text="")


def _state_bins_str(state: np.ndarray | None) -> str:
    if state is None:
        state = np.zeros(32, dtype=np.float32)
    bins = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    return " ".join(str(int(b)) for b in bins)


def build_prompt(variant: str, query: str, spm, max_len: int, state: np.ndarray | None) -> TokenisedPrompt:
    """Construct one of the named prompt variants.

    Original 3 (from PI05_DIAGNOSE_AND_FINETUNE.md §5.4):
      stock        — "<query>\\n"                                   PaliGemma canonical
      task         — "Task: <query>;\\nAction: "                    pi05 template w/o state
      task_state   — "Task: <query>, State: <bins>;\\nAction: "     full pi05 template

    Extended for prompt-tweaking experiments:
      hierarchical_table   — pi05 template with HIGH-level task "clean the table"
      hierarchical_kitchen — pi05 template with HIGH-level task "clean the kitchen"
      task_no_action       — pi05 template MINUS the trailing "Action: " suffix
      pg_multi             — PaliGemma multi-object canonical: "detect <q> ;"
      pg_multi_nl          — same but with trailing newline
      nl_locate            — natural language: "locate <q>"
      nl_where             — natural language: "What is the bounding box of the <q>?"
    """
    state_str = _state_bins_str(state)

    if variant == "stock":
        text = f"{query}\n"
        ids = spm.encode(query, add_bos=True) + spm.encode("\n")
    elif variant == "task":
        text = f"Task: {query};\nAction: "
        ids = spm.encode(text, add_bos=True)
    elif variant == "task_state":
        text = f"Task: {query}, State: {state_str};\nAction: "
        ids = spm.encode(text, add_bos=True)
    elif variant == "hierarchical_table":
        text = f"Task: clean the table, State: {state_str};\nAction: "
        ids = spm.encode(text, add_bos=True)
    elif variant == "hierarchical_kitchen":
        text = f"Task: clean the kitchen, State: {state_str};\nAction: "
        ids = spm.encode(text, add_bos=True)
    elif variant == "task_no_action":
        text = f"Task: {query}, State: {state_str};\n"
        ids = spm.encode(text, add_bos=True)
    elif variant == "pg_multi":
        text = f"detect {query} ;"
        ids = spm.encode(text, add_bos=True)
    elif variant == "pg_multi_nl":
        text = f"detect {query} ;\n"
        ids = spm.encode(text, add_bos=True)
    elif variant == "nl_locate":
        text = f"locate {query}\n"
        ids = spm.encode(text, add_bos=True)
    elif variant == "nl_where":
        text = f"What is the bounding box of the {query}?\n"
        ids = spm.encode(text, add_bos=True)
    else:
        raise ValueError(f"unknown prompt variant: {variant}")

    tp = _pad(ids, max_len)
    tp.text = text
    return tp


ALL_VARIANTS: tuple[str, ...] = (
    "stock",
    "task",
    "task_state",
    "hierarchical_table",
    "hierarchical_kitchen",
    "task_no_action",
    "pg_multi",
    "pg_multi_nl",
    "nl_locate",
    "nl_where",
)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def load_image_padded(path: Path) -> np.ndarray:
    """Load -> float32 in [-1, 1] -> 224x224 padded (same as preprocess_observation)."""
    raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 * 2.0 - 1.0
    padded = image_tools.resize_with_pad(jnp.asarray(raw), IMG_SIZE, IMG_SIZE)
    return np.asarray(padded)


def make_observation(image: np.ndarray, tp: TokenisedPrompt, state_dim: int) -> _model.Observation:
    """Build a single-frame Observation. Only base_0_rgb is active; wrist cams
    are zero-filled with image_mask=False so they're masked out of attention.
    All arrays are converted to jax.Array since model code is jaxtyping-checked."""
    img_b = jnp.asarray(image[None], dtype=jnp.float32)  # (1, 224, 224, 3) in [-1, 1]
    zero_img = jnp.full_like(img_b, -1.0)
    on = jnp.asarray([True], dtype=jnp.bool_)
    off = jnp.asarray([False], dtype=jnp.bool_)
    return _model.Observation(
        images={
            "base_0_rgb": img_b,
            "left_wrist_0_rgb": zero_img,
            "right_wrist_0_rgb": zero_img,
        },
        image_masks={
            "base_0_rgb": on,
            "left_wrist_0_rgb": off,
            "right_wrist_0_rgb": off,
        },
        state=jnp.zeros((1, state_dim), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray(tp.ids[None], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray(tp.mask[None], dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# Loc-range diagnostics on a single (V,) logit vector
# ---------------------------------------------------------------------------


def _loc_diagnostics(logits_1d: np.ndarray) -> dict:
    """Per-step diagnostics on the <loc_NNNN> sub-range of the LM head.

    For one logit vector over the full 257152-token PaliGemma vocab, returns:
      - top1_id / top1_logit  : the model's unconstrained argmax pick
      - top1_in_loc_range     : whether that pick is a <loc_NNNN> token
      - loc_top1_id / _rel    : best loc token (absolute id, and 0..1023 index)
      - loc_top1_logit        : its logit value
      - loc_top1_rank         : 1-based rank of that loc token within the FULL
                                vocab logits — the single most informative
                                "head alive vs atrophied" signal
      - loc_mass              : softmax probability mass on the loc range
                                (sum_{i in [256000,257024)} softmax(logits)[i])
    """
    loc_slice = logits_1d[LOC_OFFSET : LOC_OFFSET + LOC_COUNT]
    loc_rel = int(np.argmax(loc_slice))
    loc_id = LOC_OFFSET + loc_rel
    loc_logit = float(loc_slice[loc_rel])
    rank = int((logits_1d > loc_logit).sum()) + 1
    m = float(logits_1d.max())
    exp = np.exp(logits_1d.astype(np.float64) - m)
    Z = float(exp.sum())
    loc_mass = float(exp[LOC_OFFSET : LOC_OFFSET + LOC_COUNT].sum() / Z)
    top1_id = int(np.argmax(logits_1d))
    top1_logit = float(logits_1d[top1_id])
    return {
        "loc_top1_id": loc_id,
        "loc_top1_rel": loc_rel,
        "loc_top1_logit": loc_logit,
        "loc_top1_rank": rank,
        "loc_mass": loc_mass,
        "top1_id": top1_id,
        "top1_logit": top1_logit,
        "top1_in_loc_range": bool(LOC_OFFSET <= top1_id < LOC_OFFSET + LOC_COUNT),
    }


# ---------------------------------------------------------------------------
# LM-head greedy decoding (Pi0 / gemma.py API, not gemma_fast.py)
# ---------------------------------------------------------------------------


def _decode_logits_fn(model: pi0_mod.Pi0):
    """Helper that calls embedder.decode on the (b, T, D) post-final-norm
    embeddings to produce vocab logits, via the linen `method=` hook."""

    def decode(pre_logits):
        return model.PaliGemma.llm(
            pre_logits, method=lambda m, x: m.embedder.decode(x)
        )

    return decode


def greedy_decode(
    model: pi0_mod.Pi0,
    obs: _model.Observation,
    *,
    max_new_tokens: int,
) -> tuple[np.ndarray, list[dict]]:
    """Greedy decode max_new_tokens from the LM head, with per-step loc diagnostics.

    Returns:
        tokens: (max_new_tokens,) int64 array of picked token ids.
        per_step: list of dicts (length max_new_tokens), each with keys from
            _loc_diagnostics() PLUS `step` (int) and `wall_s` (float).
            picked_id == diag.top1_id since we are doing argmax decoding.
    """
    with stage("preprocess_observation (resize images to 224x224)"):
        obs = _model.preprocess_observation(None, obs, train=False)

    with stage("embed_prefix: SigLIP on 3 cams + token embed (first call JITs SigLIP+embedder, ~60-120s; cached after)"):
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        prefix_tokens.block_until_ready()
        log.info("  prefix_tokens shape=%s (b, T_prefix, D)", tuple(prefix_tokens.shape))

    prefix_attn_mask = pi0_mod.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    with stage(
        f"PaliGemma.llm prefix forward (T_prefix={int(prefix_tokens.shape[1])}); "
        "first call JITs the full 18-layer Gemma stack — expect ~60-180s the first time, fast afterwards"
    ):
        (prefix_out, _), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        prefix_out.block_until_ready()
        log.info("  prefix_out shape=%s", tuple(prefix_out.shape))

    decode_logits = _decode_logits_fn(model)
    with stage("embedder.decode on last prefix hidden state -> logits over 257152-token vocab (first call JITs)"):
        last_hidden = prefix_out[:, -1:, :]  # (1, 1, D)
        logits = decode_logits(last_hidden).astype(jnp.float32)  # (1, 1, V)
        logits.block_until_ready()

    logits_v = np.asarray(logits[0, 0])  # (V,) host-side
    diag = _loc_diagnostics(logits_v)
    diag["step"] = 0
    diag["wall_s"] = 0.0
    log.info(
        "  step 0: picked=%d (logit=%.3f) | loc_top1=%d (rel=%d, logit=%.3f, rank=%d, mass=%.2e)",
        diag["top1_id"], diag["top1_logit"],
        diag["loc_top1_id"], diag["loc_top1_rel"], diag["loc_top1_logit"],
        diag["loc_top1_rank"], diag["loc_mass"],
    )

    b = prefix_tokens.shape[0]
    t_prefix = int(prefix_tokens.shape[1])

    tokens_out = [diag["top1_id"]]
    per_step: list[dict] = [diag]
    token = jnp.asarray([[diag["top1_id"]]], dtype=jnp.int32)

    with stage(
        f"autoregressive greedy decode of {max_new_tokens - 1} more tokens "
        "(first AR step JITs the single-token decode path, then each step is ~tens of ms)"
    ):
        for step in range(max_new_tokens - 1):
            t_step = time.time()
            new_emb = model.PaliGemma.llm(token, method="embed")  # (1, 1, D)
            t_so_far = t_prefix + step + 1
            ar_mask = jnp.ones((b, 1, t_so_far), dtype=bool)
            pos = jnp.full((b, 1), t_prefix + step, dtype=jnp.int32)
            (out, _), kv_cache = model.PaliGemma.llm(
                [new_emb, None], mask=ar_mask, positions=pos, kv_cache=kv_cache
            )
            logits = decode_logits(out).astype(jnp.float32)
            logits_v = np.asarray(logits[0, 0])
            diag = _loc_diagnostics(logits_v)
            diag["step"] = step + 1
            diag["wall_s"] = float(time.time() - t_step)
            tokens_out.append(diag["top1_id"])
            per_step.append(diag)
            token = jnp.asarray([[diag["top1_id"]]], dtype=jnp.int32)
            log.info(
                "  step %d: picked=%d (logit=%.3f) | loc_rank=%d loc_mass=%.2e (%.2fs)",
                step + 1, diag["top1_id"], diag["top1_logit"],
                diag["loc_top1_rank"], diag["loc_mass"], diag["wall_s"],
            )

    return np.asarray(tokens_out, dtype=np.int64), per_step


def greedy_decode_loc_constrained(
    model: pi0_mod.Pi0,
    obs: _model.Observation,
) -> tuple[np.ndarray, list[dict]]:
    """Force the LM head to emit exactly 4 tokens from the <loc_NNNN> range
    [256000, 257024).

    Returns:
        tokens: (4,) int64 YXYX quartet, all in the loc range.
        per_step: list of 4 dicts with keys from _loc_diagnostics() PLUS
            `step`, `forced_loc_id`, `forced_loc_logit`.
            forced_loc_id == diag.loc_top1_id (since we mask everything else).
    """
    obs = _model.preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    prefix_tokens.block_until_ready()

    prefix_attn_mask = pi0_mod.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    decode_logits = _decode_logits_fn(model)
    last_hidden = prefix_out[:, -1:, :]
    logits = decode_logits(last_hidden).astype(jnp.float32)

    b = prefix_tokens.shape[0]
    t_prefix = int(prefix_tokens.shape[1])

    tokens_out: list[int] = []
    per_step: list[dict] = []

    for step in range(4):
        logits_v = np.asarray(logits[0, 0])  # (V,) host-side
        diag = _loc_diagnostics(logits_v)
        loc_id = diag["loc_top1_id"]
        diag["step"] = step
        diag["forced_loc_id"] = loc_id
        diag["forced_loc_logit"] = diag["loc_top1_logit"]
        tokens_out.append(loc_id)
        per_step.append(diag)

        log.info(
            "  loc_step %d: forced=%d (rel=%d, logit=%.2f, rank=%d, mass=%.2e) | "
            "unconstrained_top1=%d (logit=%.2f, in_loc=%s)",
            step, loc_id, diag["loc_top1_rel"], diag["loc_top1_logit"],
            diag["loc_top1_rank"], diag["loc_mass"],
            diag["top1_id"], diag["top1_logit"], diag["top1_in_loc_range"],
        )

        if step == 3:
            break

        token = jnp.asarray([[loc_id]], dtype=jnp.int32)
        new_emb = model.PaliGemma.llm(token, method="embed")
        t_so_far = t_prefix + step + 1
        ar_mask = jnp.ones((b, 1, t_so_far), dtype=bool)
        pos = jnp.full((b, 1), t_prefix + step, dtype=jnp.int32)
        (out, _), kv_cache = model.PaliGemma.llm(
            [new_emb, None], mask=ar_mask, positions=pos, kv_cache=kv_cache
        )
        logits = decode_logits(out).astype(jnp.float32)

    return np.asarray(tokens_out, dtype=np.int64), per_step


# ---------------------------------------------------------------------------
# Parsing <loc_NNNN> quartets -> YXYX bboxes in normalised [0, 1] coords
# ---------------------------------------------------------------------------


def parse_loc_bboxes(token_ids: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Find consecutive runs of 4 <loc_NNNN> tokens and return YXYX bboxes
    normalised to [0, 1] (1024-bin -> /1024)."""
    bboxes: list[tuple[float, float, float, float]] = []
    run: list[int] = []
    for tid in token_ids:
        if LOC_OFFSET <= tid < LOC_OFFSET + LOC_COUNT:
            run.append(int(tid) - LOC_OFFSET)
            if len(run) == 4:
                y1, x1, y2, x2 = (v / LOC_COUNT for v in run)
                bboxes.append((y1, x1, y2, x2))
                run = []
        else:
            run = []  # break the quartet on any non-loc token
    return bboxes


# ---------------------------------------------------------------------------
# Image overlay + save
# ---------------------------------------------------------------------------


def save_json(
    out_path: Path,
    *,
    frame_id: str,
    variant: str,
    mode: str,
    prompt_text: str,
    n_prompt_tokens: int,
    tokens: np.ndarray,
    decoded_text: str,
    per_step: list[dict],
    parsed_bboxes: list[tuple[float, float, float, float]],
) -> None:
    """Write a structured sidecar JSON next to the overlay PNG. One per
    (frame, variant, mode) — multi-frame sweep aggregators glob these."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "frame_id": frame_id,
        "variant": variant,
        "mode": mode,  # "free" or "loc_constrained"
        "prompt_text": prompt_text,
        "n_prompt_tokens": int(n_prompt_tokens),
        "tokens": [int(t) for t in tokens.tolist()],
        "decoded_text": decoded_text,
        "per_step": per_step,
        "parsed_bboxes_yxyx_01": [list(b) for b in parsed_bboxes],
        "n_parsed_bboxes": len(parsed_bboxes),
    }
    out_path.write_text(json.dumps(record, indent=2))
    log.info("wrote %s", out_path)


def save_overlay(
    image: np.ndarray,
    bboxes: list[tuple[float, float, float, float]],
    tokens: np.ndarray,
    spm: sentencepiece.SentencePieceProcessor,
    out_path: Path,
    *,
    variant: str,
    prompt_text: str,
) -> None:
    """image is float32 in [-1, 1] shape (224, 224, 3)."""
    rgb = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    pil = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for y1, x1, y2, x2 in bboxes:
        box = (x1 * IMG_SIZE, y1 * IMG_SIZE, x2 * IMG_SIZE, y2 * IMG_SIZE)
        draw.rectangle(box, outline=(255, 0, 0), width=3)
    decoded = spm.decode(tokens.tolist())
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    label_lines = [
        f"variant: {variant}",
        f"prompt: {prompt_text!r}",
        f"decoded: {decoded!r}",
        f"#bboxes: {len(bboxes)}",
    ]
    y = 4
    for line in label_lines:
        draw.text((4, y), line[:90], fill=(0, 255, 0), font=font)
        y += 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    log.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True, help="Overhead-camera image to probe.")
    ap.add_argument("--query", type=str, default="detect box", help='Object query, e.g. "detect box".')
    ap.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(ALL_VARIANTS),
        choices=list(ALL_VARIANTS),
        help="Which prompt variants to evaluate. See build_prompt() docstring for definitions.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=16, help="Greedy decode budget.")
    ap.add_argument(
        "--constrained_decode_loc",
        action="store_true",
        help=(
            "Also run a 4-step decode that masks all non-<loc_NNNN> logits to -inf, "
            "forcing the model to commit to a YXYX bbox. Diagnostic for whether the "
            "spatial-grounding embeddings retain meaning."
        ),
    )
    ap.add_argument(
        "--skip_free_decode",
        action="store_true",
        help="Skip the free greedy decode (only run constrained decoding).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_OPENPI_ROOT / "experiments" / "results" / "lm_head_probe",
    )
    ap.add_argument("--frame_id", type=str, default=None, help="Optional id used in output filenames.")
    ap.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path or gs:// URI of an orbax params dir to load. Default = pi05_base public checkpoint.",
    )
    ap.add_argument(
        "--use_lora", action="store_true",
        help="Set when the checkpoint has LoRA params (trained bbox output). "
             "Switches paligemma_variant to gemma_2b_lora so the param tree matches.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
        stream=sys.stdout,
    )
    # Make sure JAX prints its own progress messages too.
    logging.getLogger("jax").setLevel(logging.INFO)
    logging.getLogger("openpi").setLevel(logging.INFO)

    log.info("=" * 70)
    log.info("probe_lm_head.py start")
    log.info("=" * 70)
    log.info("JAX devices: %s", jax.devices())
    log.info("args: %s", vars(args))

    with stage("load_image_padded (resize+pad to 224x224, scale to [-1,1])"):
        image = load_image_padded(args.image)
        log.info("  image shape=%s dtype=%s range=[%.3f, %.3f]",
                 image.shape, image.dtype, float(image.min()), float(image.max()))
    frame_id = args.frame_id or args.image.stem

    with stage("load_pi05_base (download/restore checkpoint + build Pi0)"):
        model = load_pi05_base(checkpoint_path=args.checkpoint_path, use_lora=args.use_lora)

    cfg = pi0_config.Pi0Config(pi05=True)
    with stage("load_spm (PaliGemma sentencepiece, ~4 MB)"):
        spm = load_spm()

    for variant in args.variants:
        log.info("─" * 70)
        log.info("VARIANT: %s", variant)
        log.info("─" * 70)
        tp = build_prompt(variant, args.query, spm, max_len=cfg.max_token_len, state=None)
        log.info("prompt text: %r (n_tokens=%d)", tp.text, int(tp.mask.sum()))

        obs = make_observation(image, tp, state_dim=cfg.action_dim)

        if not args.skip_free_decode:
            tokens, per_step = greedy_decode(model, obs, max_new_tokens=args.max_new_tokens)
            decoded = spm.decode(tokens.tolist())
            log.info("decoded tokens (ids): %s", tokens.tolist())
            log.info("decoded text       : %r", decoded)
            log.info("loc_rank per step  : %s", [d["loc_top1_rank"] for d in per_step])
            log.info("loc_mass per step  : %s", [f"{d['loc_mass']:.2e}" for d in per_step])

            bboxes = parse_loc_bboxes(tokens)
            log.info("parsed bboxes (YXYX, [0,1]): %s", bboxes)

            out_png = args.out_dir / f"{frame_id}__{variant}.png"
            save_overlay(image, bboxes, tokens, spm, out_png, variant=variant, prompt_text=tp.text)
            save_json(
                args.out_dir / f"{frame_id}__{variant}.json",
                frame_id=frame_id, variant=variant, mode="free",
                prompt_text=tp.text, n_prompt_tokens=int(tp.mask.sum()),
                tokens=tokens, decoded_text=decoded,
                per_step=per_step, parsed_bboxes=bboxes,
            )

        if args.constrained_decode_loc:
            with stage(f"constrained loc decode for variant={variant} (4 forced <loc> tokens)"):
                loc_tokens, loc_per_step = greedy_decode_loc_constrained(model, obs)
            log.info("forced LOC token ids        : %s", loc_tokens.tolist())
            log.info("unconstrained top1 per step : %s",
                     [f"id={d['top1_id']} logit={d['top1_logit']:.2f}" for d in loc_per_step])
            log.info("loc rank per step           : %s", [d["loc_top1_rank"] for d in loc_per_step])
            log.info("loc mass per step           : %s",
                     [f"{d['loc_mass']:.2e}" for d in loc_per_step])
            # Convert 4-quartet to YXYX in [0, 1].
            rels = [int(t) - LOC_OFFSET for t in loc_tokens]
            y1, x1, y2, x2 = [r / LOC_COUNT for r in rels]
            # Normalize so y1<=y2, x1<=x2 for visualization regardless of model output order.
            y_lo, y_hi = min(y1, y2), max(y1, y2)
            x_lo, x_hi = min(x1, x2), max(x1, x2)
            log.info("constrained YXYX bbox raw   : (%.3f, %.3f, %.3f, %.3f)", y1, x1, y2, x2)
            log.info("constrained bbox normalised : y=[%.3f, %.3f] x=[%.3f, %.3f]  (1024-bins: y=[%d,%d] x=[%d,%d])",
                     y_lo, y_hi, x_lo, x_hi, rels[0], rels[2], rels[1], rels[3])

            out_png = args.out_dir / f"{frame_id}__{variant}__locconstrained.png"
            save_overlay(
                image, [(y_lo, x_lo, y_hi, x_hi)], loc_tokens, spm, out_png,
                variant=f"{variant} [loc-constrained]", prompt_text=tp.text,
            )
            save_json(
                args.out_dir / f"{frame_id}__{variant}__locconstrained.json",
                frame_id=frame_id, variant=variant, mode="loc_constrained",
                prompt_text=tp.text, n_prompt_tokens=int(tp.mask.sum()),
                tokens=loc_tokens,
                decoded_text=spm.decode(loc_tokens.tolist()),
                per_step=loc_per_step,
                parsed_bboxes=[(y_lo, x_lo, y_hi, x_hi)],
            )

    log.info("=" * 70)
    log.info("probe_lm_head.py done")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
