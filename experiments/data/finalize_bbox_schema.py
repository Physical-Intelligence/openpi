"""finalize_bbox_schema.py — add training-ready columns to annotate_bboxes.py parquets.

annotate_bboxes.py writes a "raw" parquet with bbox in float [0,1] + raw 1024-bin
ints. For Approach C training, the loader needs the 4 PaliGemma vocab IDs that
go directly into the CE target tensor, plus provenance so we know what each
sample was supervised against.

This script adds these columns in-place (the raw data is cheap to regenerate
via annotate_bboxes.py):

  target_loc_tokens  list<int>[4]   PaliGemma vocab IDs: 256000 + 1024-bin index
                                    for [y_min, x_min, y_max, x_max].
                                    Sentinel [-1,-1,-1,-1] when has_bbox=False.
  source             str            'auto-paligemma-box-v1' (or whatever you tag).
  query              str            The prompt text used (e.g. 'box').
  model_id           str            Annotation model id (e.g. 'google/paligemma-3b-mix-224').

The training loader can then do:
    loc_tokens = row['target_loc_tokens']    # list[4] of int vocab IDs
    loc_mask   = row['has_bbox']             # bool: 1 = supervise CE, 0 = skip

Run:
  python experiments/data/finalize_bbox_schema.py \\
      --parquets <p1> <p2> ... \\
      [--source ...] [--query ...] [--model_id ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("finalize_bbox_schema")

LOC_OFFSET = 256_000
LOC_COUNT = 1_024
NEW_COLS = ("target_loc_tokens", "source", "query", "model_id")


def _build_target_tokens(loc_y_min, loc_x_min, loc_y_max, loc_x_max, has_bbox) -> list[int]:
    if not has_bbox:
        return [-1, -1, -1, -1]
    return [
        LOC_OFFSET + int(loc_y_min),
        LOC_OFFSET + int(loc_x_min),
        LOC_OFFSET + int(loc_y_max),
        LOC_OFFSET + int(loc_x_max),
    ]


def finalize_one(parquet: Path, *, source: str, query: str, model_id: str, force: bool) -> dict:
    df = pq.read_table(parquet).to_pandas()
    if not force and all(c in df.columns for c in NEW_COLS):
        log.info("%s already has target columns — skipping (use --force to overwrite)", parquet.name)
        return {"path": str(parquet), "n_rows": len(df), "status": "skipped"}

    # Sanity: required input columns
    required = ("has_bbox", "loc_y_min", "loc_x_min", "loc_y_max", "loc_x_max")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{parquet}: missing required columns {missing}")

    # Build new columns
    df["target_loc_tokens"] = df.apply(
        lambda r: _build_target_tokens(
            r["loc_y_min"], r["loc_x_min"], r["loc_y_max"], r["loc_x_max"], r["has_bbox"]
        ),
        axis=1,
    )
    df["source"] = source
    df["query"] = query
    df["model_id"] = model_id

    # Sanity check: all hit rows should have all 4 tokens in [256000, 257024)
    hit = df[df["has_bbox"]]
    if len(hit) > 0:
        all_tokens = [t for row in hit["target_loc_tokens"] for t in row]
        out_of_range = [t for t in all_tokens if not (LOC_OFFSET <= t < LOC_OFFSET + LOC_COUNT)]
        if out_of_range:
            raise ValueError(f"{parquet}: {len(out_of_range)} hit tokens out of [256000, 257024) range")

    # Schema spec to keep target_loc_tokens as list<int32> (vs. inferred large_list)
    schema = pa.Table.from_pandas(df).schema
    # Force target_loc_tokens to list<int32> for compactness + clarity
    fields = []
    for f in schema:
        if f.name == "target_loc_tokens":
            fields.append(pa.field("target_loc_tokens", pa.list_(pa.int32(), 4)))
        else:
            fields.append(f)
    table = pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)
    pq.write_table(table, parquet)
    n_hits = int(df["has_bbox"].sum())
    log.info("wrote %s  (%d rows, %d hits, %d misses)",
             parquet, len(df), n_hits, len(df) - n_hits)
    return {"path": str(parquet), "n_rows": len(df), "n_hits": n_hits, "status": "ok"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquets", type=Path, nargs="+", required=True,
                    help="annotation parquets produced by annotate_bboxes.py")
    ap.add_argument("--source", type=str, default="auto-paligemma-box-v1")
    ap.add_argument("--query", type=str, default="box",
                    help="The prompt text used during annotation. Must match what annotate_bboxes.py was run with.")
    ap.add_argument("--model_id", type=str, default="google/paligemma-3b-mix-224")
    ap.add_argument("--force", action="store_true",
                    help="Re-write target_loc_tokens etc. even if they already exist.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        force=True, stream=sys.stdout)
    log.info("=" * 70)
    log.info("finalize_bbox_schema.py start")
    log.info("=" * 70)
    log.info("source=%s  query=%r  model_id=%s  n_files=%d",
             args.source, args.query, args.model_id, len(args.parquets))

    summaries = []
    for p in args.parquets:
        try:
            s = finalize_one(p, source=args.source, query=args.query, model_id=args.model_id, force=args.force)
        except Exception as e:
            log.error("FAILED %s: %s", p, e)
            s = {"path": str(p), "status": "error", "error": str(e)}
        summaries.append(s)

    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    print(f"{'status':>8}  {'n_rows':>7}  {'n_hits':>7}  path")
    for s in summaries:
        if s["status"] == "ok":
            print(f"{s['status']:>8}  {s['n_rows']:>7}  {s['n_hits']:>7}  {s['path']}")
        elif s["status"] == "skipped":
            print(f"{s['status']:>8}  {s['n_rows']:>7}  {'-':>7}  {s['path']}")
        else:
            print(f"{s['status']:>8}  {'-':>7}  {'-':>7}  {s['path']}  ERROR: {s.get('error','')}")


if __name__ == "__main__":
    main()
