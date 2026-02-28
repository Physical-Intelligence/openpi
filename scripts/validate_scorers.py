#!/usr/bin/env python3
"""Scorer validation utility: computes precision/recall against manual labels."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from openpi.research.shared.episode_schema import Episode
from openpi.research.shared.scorer_base import (
    ConnectorMatingScorer,
    LatchActuationScorer,
    PayloadTransferScorer,
    Scorer,
    SurfaceCleaningScorer,
)

logger = logging.getLogger(__name__)

SCORERS: dict[str, type[Scorer]] = {
    "payload": PayloadTransferScorer,
    "latch": LatchActuationScorer,
    "clean": SurfaceCleaningScorer,
    "connector": ConnectorMatingScorer,
}


def main():
    parser = argparse.ArgumentParser(description="Validate scorer precision/recall against manual labels")
    parser.add_argument("--episodes-dir", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(SCORERS.keys()),
    )
    args = parser.parse_args()

    # Load labels
    labels = {}
    with open(args.labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["episode_id"]] = row["success"].lower() in (
                "true",
                "1",
                "yes",
            )

    # Instantiate scorer
    scorer = SCORERS[args.task]()

    # Load episodes and score
    tp, fp, tn, fn = 0, 0, 0, 0
    episodes_dir = Path(args.episodes_dir)
    for ep_file in sorted(episodes_dir.glob("*.json")):
        ep_id = ep_file.stem
        if ep_id not in labels:
            logger.warning(f"No label for episode {ep_id}, skipping")
            continue

        with open(ep_file) as f:
            episode = Episode.from_dict(json.load(f))

        result = scorer.score(episode)
        predicted = result.success
        actual = labels[ep_id]

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    # Compute metrics
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print confusion matrix
    print(f"\nConfusion Matrix (n={total}):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Threshold check
    if precision < 0.80 or recall < 0.80:
        print("\nWARNING: Below threshold!")
        if precision < 0.80:
            print(f"  Precision {precision:.4f} < 0.80 — consider tightening scorer criteria")
        if recall < 0.80:
            print(f"  Recall {recall:.4f} < 0.80 — consider loosening scorer thresholds")
        sys.exit(1)

    print("\nPASS: Both precision and recall >= 0.80")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
