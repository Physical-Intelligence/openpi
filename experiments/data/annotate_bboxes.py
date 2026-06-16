"""annotate_bboxes.py — auto-annotate per-frame bounding boxes with raw PaliGemma.

Step 1 of PI05_DIAGNOSE_AND_FINETUNE.md §5.4: produce per-frame
`target_loc_tokens` (in PaliGemma's native <loc_NNNN> token format) for
every frame of every episode, so Approach C training has bbox supervision
aligned with action supervision.

Uses raw `google/paligemma-3b-mix-224` (NOT pi05_base — that head was
overwritten in π0.5 Stage-2; the raw weights still detect correctly on
cam_high, as proven by probe_raw_paligemma.py). PaliGemma's output tokens
are the supervision targets directly — no coordinate conversion needed.

Pipeline per dataset:
  1. Read meta/episodes/.../parquet to get episode boundaries.
  2. Open the concatenated mp4 (e.g. videos/observation.images.cam_high/
     chunk-000/file-000.mp4) and decode frames sequentially.
  3. Pad-resize each frame to 224×224 to match pi05_base's preprocessing,
     so bbox coordinates align with what the action model will see.
  4. Batch frames through PaliGemma.generate with `detect <query>\\n`.
  5. Parse loc-quartet from output; record YXYX in [0,1] + raw 1024-bin ints.
  6. Write a parquet sidecar to <dataset>/meta/bboxes/<camera>__detect_<q>.parquet.

Run (per workstation):
  /home/.../vlm_experiment/.venv-pg-gpu/bin/python \\
      experiments/data/annotate_bboxes.py \\
      --dataset_dirs ~/Desktop/.../datasets/data_1 \\
      --camera cam_high --query box --batch_size 16

After it finishes, check the printed detection rate per dataset and spot-check
a few rows by overlaying the bbox on the source frame.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

log = logging.getLogger("annotate_bboxes")
LOC_RE = re.compile(r"<loc(\d{4})>")
MODEL_ID = "google/paligemma-3b-mix-224"
IMG_SIZE = 224


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
# Dataset layout helpers (same conventions as sweep_lm_head.py)
# ---------------------------------------------------------------------------

def _episodes_parquet(dataset_dir: Path) -> Path:
    for cand in [
        dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        dataset_dir / "meta" / "episodes" / "file-000.parquet",
    ]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no episodes parquet under {dataset_dir / 'meta' / 'episodes'}")


def read_episodes(dataset_dir: Path) -> list[dict]:
    df = pq.read_table(_episodes_parquet(dataset_dir)).to_pandas()
    df = df.sort_values("episode_index").reset_index(drop=True)
    out, cum = [], 0
    for _, row in df.iterrows():
        out.append({
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "cum_start": cum,
        })
        cum += int(row["length"])
    return out


def _camera_mp4(dataset_dir: Path, camera: str) -> Path:
    p = dataset_dir / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
    if not p.exists():
        raise FileNotFoundError(f"missing mp4 at {p}")
    return p


# ---------------------------------------------------------------------------
# Frame decode (sequential) + 224×224 pad
# ---------------------------------------------------------------------------

def _pad_to_224(rgb_uint8: np.ndarray) -> Image.Image:
    """HxWx3 uint8 -> PIL 224x224 (letterbox-padded, preserving aspect ratio).
    Matches pi05_base's `image_tools.resize_with_pad` behaviour: scale so the
    long edge is 224, pad short edge with zeros (the openpi convention)."""
    h, w = rgb_uint8.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    pil = Image.fromarray(rgb_uint8).resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    canvas.paste(pil, ((IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2))
    return canvas


def iter_padded_frames(mp4: Path):
    """Yield (abs_idx, PIL 224×224) for every frame in the mp4, sequentially."""
    container = av.open(str(mp4))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    idx = 0
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                rgb = frame.to_ndarray(format="rgb24")
                yield idx, _pad_to_224(rgb)
                idx += 1
    finally:
        container.close()


# ---------------------------------------------------------------------------
# PaliGemma batched inference
# ---------------------------------------------------------------------------

def _parse_loc_quartet(text: str) -> tuple[bool, list[int]]:
    """Return (has_bbox, [y_min, x_min, y_max, x_max] in 0..1023 bins)."""
    nums = [int(m.group(1)) for m in LOC_RE.finditer(text)]
    if len(nums) >= 4:
        return True, nums[:4]
    return False, [-1, -1, -1, -1]


def run_batch(model, processor, pil_imgs: list[Image.Image], prompt: str, device: str, max_new_tokens: int) -> list[str]:
    import torch
    texts = [prompt] * len(pil_imgs)
    inputs = processor(images=pil_imgs, text=texts, return_tensors="pt", padding="longest").to(device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, input_len:]
    decoded = [processor.tokenizer.decode(g, skip_special_tokens=False) for g in gen]
    return decoded


# ---------------------------------------------------------------------------
# Main per-dataset pass
# ---------------------------------------------------------------------------

def annotate_dataset(
    *, dataset_dir: Path, camera: str, query: str,
    model, processor, device: str,
    batch_size: int, max_new_tokens: int,
    out_path: Path,
) -> dict:
    eps = read_episodes(dataset_dir)
    cum_starts = {e["episode_index"]: e["cum_start"] for e in eps}
    ep_lens = {e["episode_index"]: e["length"] for e in eps}
    total_frames = sum(ep_lens.values())
    log.info("dataset=%s  camera=%s  episodes=%d  total_frames=%d",
             dataset_dir.name, camera, len(eps), total_frames)

    prompt = f"detect {query}\n"

    # Stream frames in, run model in batches.
    rows: list[dict] = []
    batch_pils: list[Image.Image] = []
    batch_abs: list[int] = []

    def flush():
        if not batch_pils:
            return
        decoded_list = run_batch(model, processor, batch_pils, prompt, device, max_new_tokens)
        for abs_idx, decoded in zip(batch_abs, decoded_list):
            has, loc4 = _parse_loc_quartet(decoded)
            if has:
                y1, x1, y2, x2 = (v / 1024.0 for v in loc4)
            else:
                y1 = x1 = y2 = x2 = float("nan")
            ep = max((e for e in eps if abs_idx >= e["cum_start"]), key=lambda e: e["cum_start"])
            rows.append({
                "abs_frame_index": int(abs_idx),
                "episode_index": int(ep["episode_index"]),
                "frame_index": int(abs_idx - ep["cum_start"]),
                "has_bbox": bool(has),
                "y_min": float(y1), "x_min": float(x1),
                "y_max": float(y2), "x_max": float(x2),
                "loc_y_min": int(loc4[0]), "loc_x_min": int(loc4[1]),
                "loc_y_max": int(loc4[2]), "loc_x_max": int(loc4[3]),
                "decoded": decoded,
            })
        batch_pils.clear()
        batch_abs.clear()

    with stage(f"decode + annotate {total_frames} frames @ batch={batch_size}"):
        t0 = time.time()
        for abs_idx, pil in iter_padded_frames(_camera_mp4(dataset_dir, camera)):
            batch_pils.append(pil)
            batch_abs.append(abs_idx)
            if len(batch_pils) >= batch_size:
                flush()
                if abs_idx % (batch_size * 16) == 0:
                    rate = (abs_idx + 1) / (time.time() - t0)
                    log.info("  progress: %d/%d frames  (%.1f f/s, eta %.1fs)",
                             abs_idx + 1, total_frames, rate, (total_frames - abs_idx - 1) / rate)
        flush()

    # Write parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    n_hit = sum(1 for r in rows if r["has_bbox"])
    rate = n_hit / max(len(rows), 1)
    log.info("wrote %s  (%d/%d frames have bbox = %.1f%%)",
             out_path, n_hit, len(rows), 100 * rate)
    return {
        "dataset": dataset_dir.name,
        "n_frames": len(rows),
        "n_with_bbox": n_hit,
        "detection_rate": rate,
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--camera", type=str, default="cam_high",
                    choices=("cam_high", "cam_left_wrist", "cam_right_wrist"))
    ap.add_argument("--query", type=str, default="box",
                    help='Object name for the `detect <query>` prompt.')
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True, stream=sys.stdout)
    log.info("=" * 70)
    log.info("annotate_bboxes.py start")
    log.info("=" * 70)
    log.info("model=%s  device=%s  camera=%s  query=%r  batch=%d",
             MODEL_ID, args.device, args.camera, args.query, args.batch_size)
    log.info("dataset_dirs:")
    for d in args.dataset_dirs:
        log.info("  %s", d)

    # Import heavy deps now (so --help is fast)
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    import transformers.utils.logging as hf_log
    hf_log.set_verbosity_error()  # silence the "passing both text and images" warning
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    with stage(f"AutoProcessor.from_pretrained({MODEL_ID})"):
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    with stage(f"PaliGemmaForConditionalGeneration.from_pretrained({MODEL_ID}) → {args.device}"):
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        ).to(args.device).eval()

    summaries = []
    for d in args.dataset_dirs:
        out_path = d / "meta" / "bboxes" / f"{args.camera}__detect_{args.query.replace(' ', '_')}.parquet"
        try:
            s = annotate_dataset(
                dataset_dir=d, camera=args.camera, query=args.query,
                model=model, processor=processor, device=args.device,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                out_path=out_path,
            )
            summaries.append(s)
        except Exception as e:
            log.error("failed on %s: %s", d, e)
            summaries.append({"dataset": d.name, "error": str(e)})

    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    print(f"{'dataset':>16}  {'n_frames':>9}  {'n_bbox':>8}  {'rate':>6}  out")
    for s in summaries:
        if "error" in s:
            print(f"{s['dataset']:>16}  ERROR: {s['error']}")
            continue
        print(f"{s['dataset']:>16}  {s['n_frames']:>9}  {s['n_with_bbox']:>8}  "
              f"{s['detection_rate']*100:>5.1f}%  {s['out_path']}")
    log.info("done")


if __name__ == "__main__":
    main()
