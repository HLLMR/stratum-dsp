#!/usr/bin/env python3
"""
Analyze tempo error ratios (pred/gt) for a validation_results_*.csv file.

This is a lightweight helper for quickly spotting metrical-level / harmonic confusions
(e.g., 2×, 1/2×, 3:2, 4:3) in validation outputs.
"""

import argparse
import csv
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pred/gt ratio buckets in validation results")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to validation_results_*.csv",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.08,
        help="Absolute tolerance around ratio factors (default: 0.08)",
    )
    args = parser.parse_args()

    path = args.file
    tol = float(args.tol)

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ratios = []
    for r in rows:
        gt = float(r["bpm_gt"])
        pred = float(r["bpm_pred"])
        if gt > 0 and pred > 0:
            ratios.append(pred / gt)

    factors = [
        ("1x", 1.0),
        ("2x", 2.0),
        ("1/2x", 0.5),
        ("3/2x", 1.5),
        ("2/3x", 2 / 3),
        ("4/3x", 4 / 3),
        ("3/4x", 3 / 4),
    ]

    counts = {}
    for q in ratios:
        hit = None
        for name, f in factors:
            if abs(q - f) <= tol:
                hit = name
                break
        counts[hit or "other"] = counts.get(hit or "other", 0) + 1

    print(f"File: {os.path.basename(path)}")
    print(f"n={len(ratios)}")
    print(f"ratio buckets (±{tol}):")
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()


