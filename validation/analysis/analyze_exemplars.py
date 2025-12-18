#!/usr/bin/env python3
"""
Generate a diagnostic exemplar report from a validation_results_*.csv file.

Outputs:
- Worst-N errors overall
- Worst-N errors per GT tempo band
- Ratio-bucket breakdown per GT tempo band
- "TAG wins vs Stratum" exemplars (TAG within ±2, Stratum not)
- "Stratum wins vs TAG" exemplars (Stratum within ±2, TAG not)

This is intended to make Phase 1F tuning surgical and reproducible.
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict


GT_BANDS = [
    ("lt60", 0.0, 60.0),
    ("60_90", 60.0, 90.0),
    ("90_120", 90.0, 120.0),
    ("120_150", 120.0, 150.0),
    ("150_180", 150.0, 180.0),
    ("gt180", 180.0, float("inf")),
]


RATIO_BUCKETS = [
    ("1x", 1.0),
    ("2x", 2.0),
    ("1/2x", 0.5),
    ("3/2x", 1.5),
    ("2/3x", 2 / 3),
    ("4/3x", 4 / 3),
    ("3/4x", 3 / 4),
]


def band_for_gt(gt: float) -> str:
    for name, lo, hi in GT_BANDS:
        if lo <= gt < hi:
            return name
    return "unknown"


def ratio_bucket(pred: float, gt: float, tol: float) -> str:
    if gt <= 0 or pred <= 0:
        return "N/A"
    r = pred / gt
    for name, f in RATIO_BUCKETS:
        if abs(r - f) <= tol:
            return name
    return "other"


def fnum(x, default="N/A") -> str:
    try:
        if x is None:
            return default
        if x == "":
            return default
        return f"{float(x):.2f}"
    except Exception:
        return default


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a Phase 1F exemplar report from results CSV")
    p.add_argument("--file", required=True, help="Path to validation_results_*.csv")
    p.add_argument("--tol", type=float, default=0.08, help="Ratio bucket tolerance (default: 0.08)")
    p.add_argument("--top", type=int, default=25, help="Number of exemplars per section (default: 25)")
    args = p.parse_args()

    path = args.file
    tol = float(args.tol)
    topn = int(args.top)

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Normalize numeric fields used below
    for r in rows:
        r["_gt"] = float(r["bpm_gt"])
        r["_pred"] = float(r["bpm_pred"])
        r["_err"] = float(r["bpm_error"])
        r["_band"] = band_for_gt(r["_gt"])
        r["_ratio_bucket"] = ratio_bucket(r["_pred"], r["_gt"], tol)
        r["_tag_err"] = float(r["bpm_tag_error"]) if r.get("bpm_tag_error", "") != "" else math.nan

    print(f"Exemplar report: {os.path.basename(path)}")
    print(f"n={len(rows)}")
    print()

    # Worst overall
    print("=" * 80)
    print(f"Worst {topn} Stratum BPM errors (overall)")
    print("=" * 80)
    worst = sorted(rows, key=lambda r: abs(r["_err"]), reverse=True)[:topn]
    for r in worst:
        print(
            f"track={r['track_id']:<6} band={r['_band']:<7} genre={r.get('genre',''):<12} "
            f"GT={r['_gt']:.3f} pred={r['_pred']:.2f} err={r['_err']:.2f} bucket={r['_ratio_bucket']:<6} "
            f"TAG={r.get('bpm_tag','') or '':>6} tag_err={fnum(r.get('bpm_tag_error','')):>6}"
        )
    print()

    # Worst per band
    print("=" * 80)
    print(f"Worst {min(10, topn)} per GT tempo band")
    print("=" * 80)
    for name, _, _ in GT_BANDS:
        sub = [r for r in rows if r["_band"] == name]
        if not sub:
            continue
        print(f"\n[{name}] n={len(sub)}")
        for r in sorted(sub, key=lambda r: abs(r["_err"]), reverse=True)[: min(10, topn)]:
            print(
                f"  track={r['track_id']:<6} genre={r.get('genre',''):<12} "
                f"GT={r['_gt']:.3f} pred={r['_pred']:.2f} err={r['_err']:.2f} bucket={r['_ratio_bucket']:<6} "
                f"TAG={r.get('bpm_tag','') or '':>6} tag_err={fnum(r.get('bpm_tag_error','')):>6}"
            )
    print()

    # Ratio buckets per band
    print("=" * 80)
    print("Ratio bucket breakdown per GT tempo band (Stratum pred/gt)")
    print("=" * 80)
    per_band = defaultdict(Counter)
    for r in rows:
        per_band[r["_band"]][r["_ratio_bucket"]] += 1
    for name, _, _ in GT_BANDS:
        c = per_band.get(name)
        if not c:
            continue
        total = sum(c.values())
        items = ", ".join([f"{k}={v}" for k, v in c.most_common()])
        print(f"{name:<8} n={total:<3} | {items}")
    print()

    # TAG vs Stratum "wins" in strict ±2
    tag_beats = [
        r
        for r in rows
        if not math.isnan(r["_tag_err"]) and r["_tag_err"] <= 2.0 and r["_err"] > 2.0
    ]
    stratum_beats = [
        r
        for r in rows
        if not math.isnan(r["_tag_err"]) and r["_err"] <= 2.0 and r["_tag_err"] > 2.0
    ]

    print("=" * 80)
    print(f"TAG within ±2, Stratum not (showing {topn})")
    print("=" * 80)
    for r in sorted(tag_beats, key=lambda r: abs(r["_err"]), reverse=True)[:topn]:
        print(
            f"track={r['track_id']:<6} band={r['_band']:<7} genre={r.get('genre',''):<12} "
            f"GT={r['_gt']:.3f} pred={r['_pred']:.2f} err={r['_err']:.2f} bucket={r['_ratio_bucket']:<6} "
            f"TAG={float(r.get('bpm_tag')):.2f} tag_err={r['_tag_err']:.2f}"
        )
    print()

    print("=" * 80)
    print(f"Stratum within ±2, TAG not (showing {topn})")
    print("=" * 80)
    for r in sorted(stratum_beats, key=lambda r: abs(r["_tag_err"]), reverse=True)[:topn]:
        print(
            f"track={r['track_id']:<6} band={r['_band']:<7} genre={r.get('genre',''):<12} "
            f"GT={r['_gt']:.3f} pred={r['_pred']:.2f} err={r['_err']:.2f} bucket={r['_ratio_bucket']:<6} "
            f"TAG={float(r.get('bpm_tag')):.2f} tag_err={r['_tag_err']:.2f}"
        )
    print()

    print("Summary:")
    print(f"  TAG within ±2, Stratum not: {len(tag_beats)}")
    print(f"  Stratum within ±2, TAG not: {len(stratum_beats)}")


if __name__ == "__main__":
    main()


