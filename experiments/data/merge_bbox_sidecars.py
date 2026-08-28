"""merge_bbox_sidecars.py — re-key per-dataset bbox parquets into one merged file
that aligns with the LeRobot dataset produced by process_single_arm.merge_datasets.

Why this exists: annotate_bboxes.py wrote one bbox parquet per source dataset
(datasets/data_N/meta/bboxes/cam_high__detect_box.parquet) keyed by per-dataset
`(episode_index, frame_index)`. When process_single_arm.merge_datasets fuses
those 6 source datasets into one, it shifts `episode_index` by cumulative
`total_episodes` and the absolute frame index (`index`) by cumulative
`total_frames` — but NOT `frame_index` (which stays per-episode). This script
applies the same shifts to the bbox sidecars so the merged training set has a
single sidecar parquet to join on `(episode_index, frame_index)`.

Order of source dataset directories must match process_single_arm.merge_datasets
(it processes whatever you pass via --src_dirs in that script's order).

Output schema is identical to the inputs plus a `source_dataset` column for
provenance. The merged parquet is what the training loader's LocTargetsBuilder
should load.

Usage:
  python experiments/data/merge_bbox_sidecars.py \\
      --dataset_dirs ~/.../datasets/data_1 ~/.../datasets/data_2 ... \\
      --bbox_filename cam_high__detect_box.parquet \\
      --out_path experiments/data/merged_bbox_sidecar/cam_high__detect_box__merged.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("merge_bbox_sidecars")


def _read_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}")
    return json.loads(info_path.read_text())


def merge(*, dataset_dirs: list[Path], bbox_filename: str, out_path: Path) -> dict:
    """Apply the same episode/frame-index offsets process_single_arm.merge_datasets
    uses, concatenate the bbox parquets, write to out_path. Returns a small
    summary dict."""
    tables: list[pa.Table] = []
    episode_offset = 0
    frame_offset = 0
    summaries: list[dict] = []

    for src in dataset_dirs:
        bbox_path = src / "meta" / "bboxes" / bbox_filename
        if not bbox_path.exists():
            raise FileNotFoundError(f"missing {bbox_path}")
        df = pq.read_table(bbox_path).to_pandas()

        # Sanity: these columns must exist (finalize_bbox_schema.py guarantees them)
        for col in ("episode_index", "frame_index", "abs_frame_index", "has_bbox", "target_loc_tokens"):
            if col not in df.columns:
                raise ValueError(f"{bbox_path}: missing column {col!r}")

        # Apply the same shifts as process_single_arm.merge_datasets (lines 217-218):
        #   df["episode_index"] += episode_offset
        #   df["index"]         += frame_offset
        # Our column for the absolute frame index is `abs_frame_index`, not `index`.
        df["episode_index"] = df["episode_index"] + episode_offset
        df["abs_frame_index"] = df["abs_frame_index"] + frame_offset
        df["source_dataset"] = src.name

        info = _read_info(src)
        n_eps_before = episode_offset
        n_frames_before = frame_offset
        episode_offset += int(info["total_episodes"])
        frame_offset += int(info["total_frames"])

        summaries.append({
            "source": src.name,
            "n_rows": len(df),
            "n_hits": int(df["has_bbox"].sum()),
            "episode_range": (n_eps_before, episode_offset - 1),
            "abs_frame_range": (n_frames_before, frame_offset - 1),
        })
        # Convert back to pyarrow Table preserving target_loc_tokens type
        tables.append(pa.Table.from_pandas(df, preserve_index=False))
        log.info("read %s: %d rows (eps %d-%d, abs_frames %d-%d after shift)",
                 src.name, len(df),
                 summaries[-1]["episode_range"][0], summaries[-1]["episode_range"][1],
                 summaries[-1]["abs_frame_range"][0], summaries[-1]["abs_frame_range"][1])

    merged = pa.concat_tables(tables)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, out_path)

    n_total = sum(s["n_rows"] for s in summaries)
    n_hits = sum(s["n_hits"] for s in summaries)
    log.info("wrote %s  (total=%d rows, hits=%d, rate=%.1f%%, episodes=%d, frames=%d)",
             out_path, n_total, n_hits, 100 * n_hits / max(n_total, 1),
             episode_offset, frame_offset)
    return {
        "out_path": str(out_path),
        "n_total": n_total,
        "n_hits": n_hits,
        "total_episodes": episode_offset,
        "total_frames": frame_offset,
        "per_source": summaries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dirs", type=Path, nargs="+", required=True,
                    help="Source dataset roots in the SAME ORDER process_single_arm.merge_datasets consumed.")
    ap.add_argument("--bbox_filename", type=str, default="cam_high__detect_box.parquet",
                    help="Filename of the per-dataset bbox parquet under meta/bboxes/.")
    ap.add_argument("--out_path", type=Path, required=True,
                    help="Where to write the merged parquet.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True, stream=sys.stdout)
    log.info("=" * 70)
    log.info("merge_bbox_sidecars.py start")
    log.info("=" * 70)
    log.info("bbox_filename=%s  out=%s", args.bbox_filename, args.out_path)
    log.info("sources (order matters; must match process_single_arm.merge_datasets):")
    for d in args.dataset_dirs:
        log.info("  %s", d)

    s = merge(dataset_dirs=args.dataset_dirs, bbox_filename=args.bbox_filename, out_path=args.out_path)

    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    print(f"{'source':>16}  {'rows':>7}  {'hits':>7}  {'ep_range':>12}  {'abs_frame_range':>20}")
    for src in s["per_source"]:
        ep_lo, ep_hi = src["episode_range"]
        fr_lo, fr_hi = src["abs_frame_range"]
        ep_str = f"[{ep_lo},{ep_hi}]"
        fr_str = f"[{fr_lo},{fr_hi}]"
        print(f"{src['source']:>16}  {src['n_rows']:>7}  {src['n_hits']:>7}  "
              f"{ep_str:>12}  {fr_str:>20}")
    total_eps = s["total_episodes"]
    total_frames = s["total_frames"]
    total_eps_str = f"eps=0..{total_eps - 1}"
    total_frames_str = f"frames=0..{total_frames - 1}"
    print(f"{'TOTAL':>16}  {s['n_total']:>7}  {s['n_hits']:>7}  "
          f"{total_eps_str:>12}  {total_frames_str:>20}")


if __name__ == "__main__":
    main()
