"""validate_bboxes.py — visual review + manual correction for auto-annotated bboxes.

Two modes:

  grid (default) — sample N random frames from the annotation parquet, draw the
    auto-bbox on each, tile into a single composite PNG that you can open in
    any image viewer. Fast batch eyeballing — no clicking required.

  interactive — step through samples in a cv2 window. Per frame you can:
      a — accept the auto-bbox (mark as correct)
      b — mark as bad / no fix attempted
      d — redraw: opens a click-drag rectangle selector, your box overrides
      n — flag "no box visible" (for cases where annotation said yes but it's wrong)
      s — skip (don't record a decision)
      q — quit and save what you've reviewed so far
    Corrections are written to a sidecar parquet alongside the original.

Both modes read the parquet produced by annotate_bboxes.py and the same source
mp4 via the same PyAV path so coordinates match exactly.

Usage:
  # quick grid of 60 random hits on data_1
  python experiments/data/validate_bboxes.py grid \\
      --parquet datasets/data_1/meta/bboxes/cam_high__detect_box.parquet \\
      --dataset_dir datasets/data_1 --n 60

  # interactive review of 40 samples (mix of hits + misses)
  python experiments/data/validate_bboxes.py interactive \\
      --parquet datasets/data_1/meta/bboxes/cam_high__detect_box.parquet \\
      --dataset_dir datasets/data_1 --n 40
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

log = logging.getLogger("validate_bboxes")
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
# Frame extraction (same letterbox-pad as annotate_bboxes.py)
# ---------------------------------------------------------------------------

def _camera_mp4(dataset_dir: Path, camera: str) -> Path:
    p = dataset_dir / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
    if not p.exists():
        raise FileNotFoundError(f"missing mp4 at {p}")
    return p


def _pad_to_224(rgb_uint8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 224x224x3 uint8 letterboxed (zero-padded), matching
    annotate_bboxes.py and openpi's resize_with_pad convention."""
    h, w = rgb_uint8.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb_uint8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    oy, ox = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def extract_frames(mp4: Path, abs_indices: list[int]) -> dict[int, np.ndarray]:
    """Sequential decode, returns {abs_idx: 224x224 uint8 RGB}."""
    wanted = sorted(set(int(i) for i in abs_indices))
    if not wanted:
        return {}
    out: dict[int, np.ndarray] = {}
    container = av.open(str(mp4))
    stream = container.streams.video[0]; stream.thread_type = "AUTO"
    target_iter = iter(wanted); next_target = next(target_iter, None); idx = 0
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                if next_target is None:
                    return out
                if idx == next_target:
                    out[idx] = _pad_to_224(frame.to_ndarray(format="rgb24"))
                    next_target = next(target_iter, None)
                idx += 1
                if next_target is None:
                    return out
    finally:
        container.close()
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_bbox(img_rgb_224: np.ndarray, y1: float, x1: float, y2: float, x2: float,
               color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Draw a bbox (YXYX normalised) on a 224 uint8 RGB image, returns a NEW image."""
    out = img_rgb_224.copy()
    py1, px1 = int(y1 * IMG_SIZE), int(x1 * IMG_SIZE)
    py2, px2 = int(y2 * IMG_SIZE), int(x2 * IMG_SIZE)
    cv2.rectangle(out, (px1, py1), (px2, py2), color, thickness)
    return out


def _label(img_rgb_224: np.ndarray, lines: list[str]) -> np.ndarray:
    """Write small green text on top of the image, returns a NEW image."""
    out = img_rgb_224.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line[:40], (3, 12 + i * 11), cv2.FONT_HERSHEY_PLAIN,
                    0.7, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Grid mode
# ---------------------------------------------------------------------------

def _sample_rows(df, n: int, only_hits: bool, seed: int):
    pool = df[df["has_bbox"]] if only_hits else df
    if len(pool) == 0:
        return pool
    n = min(n, len(pool))
    return pool.sample(n, random_state=seed).reset_index(drop=True)


def _render_grid(rows, frames, cols: int, draw_bbox: bool) -> np.ndarray:
    rows_n = (len(rows) + cols - 1) // cols
    pad = 4
    cell = IMG_SIZE + 2 * pad
    H, W = rows_n * cell, cols * cell
    grid = np.full((H, W, 3), 32, dtype=np.uint8)
    for i, row in rows.iterrows():
        img = frames.get(int(row["abs_frame_index"]))
        if img is None:
            continue
        if draw_bbox and row["has_bbox"]:
            img = _draw_bbox(img, row["y_min"], row["x_min"], row["y_max"], row["x_max"])
        decoded = (row["decoded"] or "").replace("\n", " ")[:18]
        img = _label(img, [f"ep{row['episode_index']} f{row['frame_index']}", decoded])
        r, c = i // cols, i % cols
        y0, x0 = r * cell + pad, c * cell + pad
        grid[y0:y0 + IMG_SIZE, x0:x0 + IMG_SIZE] = img
    return grid


def run_grid(args) -> None:
    df = pq.read_table(args.parquet).to_pandas()
    log.info("loaded parquet: %d rows, %d hits, %d misses",
             len(df), int(df["has_bbox"].sum()), int((~df["has_bbox"]).sum()))

    todo: list[tuple[str, bool]] = []  # (label, only_hits)
    if args.show in ("hits", "both"):
        todo.append(("hits", True))
    if args.show in ("misses", "both"):
        todo.append(("misses", False))

    base_out = args.out or args.parquet.with_name(args.parquet.stem + "_grid.png")
    for label, only_hits in todo:
        if only_hits:
            rows = _sample_rows(df[df["has_bbox"]], args.n, only_hits=True, seed=args.seed)
        else:
            rows = _sample_rows(df[~df["has_bbox"]], args.n, only_hits=False, seed=args.seed)
        if len(rows) == 0:
            log.warning("no %s available, skipping", label)
            continue
        log.info("sampled %d %s rows for grid", len(rows), label)
        abs_idxs = rows["abs_frame_index"].tolist()
        with stage(f"extract {len(abs_idxs)} frames from mp4"):
            frames = extract_frames(_camera_mp4(args.dataset_dir, args.camera), abs_idxs)
        grid = _render_grid(rows, frames, cols=args.cols, draw_bbox=only_hits)
        if args.show == "both":
            out = base_out.with_name(base_out.stem + f"__{label}.png")
        else:
            out = base_out
        Image.fromarray(grid).save(out)
        log.info("wrote %s  (shape=%s, %d cells, %s)",
                 out, grid.shape, len(rows), label)
        print(f"\nOpen ({label}): {out}")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(args) -> None:
    df = pq.read_table(args.parquet).to_pandas()
    log.info("loaded parquet: %d rows, %d hits", len(df), int(df["has_bbox"].sum()))

    hit_share = args.hit_share
    n_hits = int(round(args.n * hit_share))
    n_miss = args.n - n_hits
    hits = _sample_rows(df[df["has_bbox"]], n_hits, only_hits=True, seed=args.seed)
    miss = _sample_rows(df[~df["has_bbox"]], n_miss, only_hits=False, seed=args.seed + 1)
    sample = (
        # interleave hits and misses so user sees both
        __import__("pandas").concat([hits, miss], ignore_index=True)
        .sample(frac=1, random_state=args.seed + 2).reset_index(drop=True)
    )
    log.info("interactive sample: %d hits + %d miss = %d rows", len(hits), len(miss), len(sample))

    abs_idxs = sample["abs_frame_index"].tolist()
    with stage(f"extract {len(abs_idxs)} frames"):
        frames = extract_frames(_camera_mp4(args.dataset_dir, args.camera), abs_idxs)

    decisions: list[dict] = []
    win = "validate_bboxes  (a=accept, b=bad, d=draw, n=no-box, s=skip, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, IMG_SIZE * 3, IMG_SIZE * 3)

    quit_flag = False
    for i, row in sample.iterrows():
        if quit_flag:
            break
        img_rgb = frames.get(int(row["abs_frame_index"]))
        if img_rgb is None:
            continue
        # cv2 uses BGR
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if row["has_bbox"]:
            bgr = _draw_bbox(bgr, row["y_min"], row["x_min"], row["y_max"], row["x_max"],
                             color=(0, 255, 255))
        bgr = _label(bgr, [
            f"[{i+1}/{len(sample)}]  ep{row['episode_index']} f{row['frame_index']}",
            f"hit={bool(row['has_bbox'])}  decoded={(row['decoded'] or '')[:30]}",
            "a=accept  b=bad  d=draw  n=no-box  s=skip  q=quit",
        ])
        # scale up for visibility (cv2 selectROI needs reasonably-sized window anyway)
        big = cv2.resize(bgr, (IMG_SIZE * 3, IMG_SIZE * 3), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, big)
        key = cv2.waitKey(0) & 0xFF
        ch = chr(key) if 0 <= key < 128 else ""

        if ch == "q":
            quit_flag = True; continue
        if ch == "s":
            continue
        if ch == "a":
            decisions.append({**_row_dict(row), "verdict": "accept", "fix_y_min": np.nan,
                              "fix_x_min": np.nan, "fix_y_max": np.nan, "fix_x_max": np.nan})
            continue
        if ch == "b":
            decisions.append({**_row_dict(row), "verdict": "bad", "fix_y_min": np.nan,
                              "fix_x_min": np.nan, "fix_y_max": np.nan, "fix_x_max": np.nan})
            continue
        if ch == "n":
            decisions.append({**_row_dict(row), "verdict": "no_box", "fix_y_min": np.nan,
                              "fix_x_min": np.nan, "fix_y_max": np.nan, "fix_x_max": np.nan})
            continue
        if ch == "d":
            # selectROI on the upscaled view, scale back to 224 normalised
            raw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            big_raw = cv2.resize(raw, (IMG_SIZE * 3, IMG_SIZE * 3), interpolation=cv2.INTER_NEAREST)
            r = cv2.selectROI("draw bbox (ENTER=ok ESC=cancel)", big_raw, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("draw bbox (ENTER=ok ESC=cancel)")
            x, y, w, h = r
            if w == 0 or h == 0:
                log.info("draw cancelled, skipping")
                continue
            # scale-back: big is 3× the 224
            y1, x1 = y / (IMG_SIZE * 3), x / (IMG_SIZE * 3)
            y2, x2 = (y + h) / (IMG_SIZE * 3), (x + w) / (IMG_SIZE * 3)
            decisions.append({**_row_dict(row), "verdict": "redrawn",
                              "fix_y_min": float(y1), "fix_x_min": float(x1),
                              "fix_y_max": float(y2), "fix_x_max": float(x2)})
            log.info("redrawn ep%d f%d: y=[%.3f, %.3f] x=[%.3f, %.3f]",
                     row["episode_index"], row["frame_index"], y1, y2, x1, x2)
            continue
        log.info("ignored key=%r", ch)

    cv2.destroyAllWindows()
    log.info("collected %d decisions", len(decisions))

    if decisions:
        out = args.out or args.parquet.with_name(args.parquet.stem + "_validation.parquet")
        pq.write_table(pa.Table.from_pylist(decisions), out)
        log.info("wrote %s", out)
        # quick verdict-tally
        from collections import Counter
        c = Counter(d["verdict"] for d in decisions)
        print("\nverdict tally:")
        for k in ("accept", "bad", "redrawn", "no_box"):
            print(f"  {k:>8} : {c.get(k, 0):3d}")
        if c.get("accept", 0) + c.get("bad", 0) > 0:
            rate = c.get("accept", 0) / (c.get("accept", 0) + c.get("bad", 0) + c.get("redrawn", 0))
            print(f"\n  acceptance rate (accept / (accept+bad+redrawn)) : {rate*100:.1f}%")


def _row_dict(row) -> dict:
    return {
        "abs_frame_index": int(row["abs_frame_index"]),
        "episode_index": int(row["episode_index"]),
        "frame_index": int(row["frame_index"]),
        "has_bbox_auto": bool(row["has_bbox"]),
        "y_min_auto": float(row["y_min"]) if row["has_bbox"] else float("nan"),
        "x_min_auto": float(row["x_min"]) if row["has_bbox"] else float("nan"),
        "y_max_auto": float(row["y_max"]) if row["has_bbox"] else float("nan"),
        "x_max_auto": float(row["x_max"]) if row["has_bbox"] else float("nan"),
        "decoded_auto": str(row["decoded"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--parquet", type=Path, required=True,
                        help="annotate_bboxes.py output parquet (cam_high__detect_box.parquet etc).")
    common.add_argument("--dataset_dir", type=Path, required=True)
    common.add_argument("--camera", type=str, default="cam_high",
                        choices=("cam_high", "cam_left_wrist", "cam_right_wrist"))
    common.add_argument("--n", type=int, default=60, help="how many samples to show / review.")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--out", type=Path, default=None)

    g = sub.add_parser("grid", parents=[common], help="non-interactive: tile N samples into one PNG.")
    g.add_argument("--cols", type=int, default=6)
    g.add_argument("--show", type=str, default="hits", choices=("hits", "misses", "both"),
                   help="which subset to tile: hits (default), misses (verify the 'no box' calls), or both (writes two PNGs).")

    i = sub.add_parser("interactive", parents=[common], help="cv2 click-through with manual draw.")
    i.add_argument("--hit_share", type=float, default=0.7,
                   help="fraction of samples that should be has_bbox=True (rest are misses).")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True, stream=sys.stdout)

    if args.mode == "grid":
        run_grid(args)
    elif args.mode == "interactive":
        run_interactive(args)


if __name__ == "__main__":
    main()
