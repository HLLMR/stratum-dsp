"""
Analyze a candidate diagnosis CSV produced by `diagnose_candidates.py`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="candidate_diagnosis_*.csv")
    ap.add_argument("--tol", type=float, default=2.0)
    args = ap.parse_args()

    path = Path(args.file)
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    rows = [r for r in rows if r.get("gt_in_topn") in ("YES", "NO")]
    if not rows:
        print("No usable rows")
        return 0

    hit = sum(1 for r in rows if r["gt_in_topn"] == "YES")
    print(f"File: {path.name}")
    print(f"n={len(rows)} GT-in-topN (±{args.tol}): {hit}/{len(rows)} ({100.0*hit/len(rows):.1f}%)")
    print()

    bins = [
        ("<=5", 0.0, 5.0),
        ("5-20", 5.0, 20.0),
        ("20-50", 20.0, 50.0),
        ("50-100", 50.0, 100.0),
        (">100", 100.0, 1e9),
    ]
    for name, a, b in bins:
        sub = [r for r in rows if a < float(r["prev_bpm_error"]) <= b]
        if not sub:
            continue
        hit = sum(1 for r in sub if r["gt_in_topn"] == "YES")
        print(f"{name:6s} n={len(sub):3d} GT-in-topN={hit/len(sub)*100:5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


