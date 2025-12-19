#!/usr/bin/env python3
"""
Analyze key detection errors from validation_results_*.csv.

Identifies:
- Mode confusion (major vs minor)
- Circle-of-fifths confusion patterns
- Most confused key pairs
- Systematic errors
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from validation._keys import normalize_key, key_name_to_echonest_key_mode


def key_to_tonic_mode(key_str: str) -> tuple:
    """Convert key string to (tonic, mode) tuple. Returns (None, None) if invalid."""
    if not key_str or key_str == "N/A":
        return (None, None)
    
    key_norm = normalize_key(key_str)
    ek = key_name_to_echonest_key_mode(key_norm)
    if ek is None:
        return (None, None)
    
    tonic, mode = ek
    return (tonic, mode)


def circle_of_fifths_distance(tonic1: int, tonic2: int) -> int:
    """Compute circle-of-fifths distance between two tonics (0-11)."""
    # Circle of fifths: C=0, G=1, D=2, A=3, E=4, B=5, F#=6, C#=7, G#=8, D#=9, A#=10, F=11
    cof_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C, G, D, A, E, B, F#, C#, G#, D#, A#, F
    try:
        idx1 = cof_order.index(tonic1)
        idx2 = cof_order.index(tonic2)
        return min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
    except ValueError:
        return 6  # Max distance if invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze key detection errors")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to validation_results_*.csv",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Filter to tracks with GT keys and predictions
    key_rows = []
    for row in rows:
        if row.get("key_ref") == "GT" and row.get("key_match") in ("YES", "NO"):
            key_gt = row.get("key_gt", "")
            key_pred = row.get("key_pred", "")
            if key_gt and key_pred:
                key_rows.append((row["track_id"], key_gt, key_pred, row["key_match"] == "YES"))

    if not key_rows:
        print("No key GT data found in results file")
        return

    print(f"Analyzing {len(key_rows)} tracks with key GT")
    print("=" * 60)

    # Confusion matrix
    confusion = defaultdict(int)
    mode_errors = 0
    tonic_errors = 0
    both_wrong = 0
    correct = 0

    # Circle-of-fifths distance distribution
    cof_distances = defaultdict(int)

    for track_id, key_gt, key_pred, is_match in key_rows:
        gt_tonic, gt_mode = key_to_tonic_mode(key_gt)
        pred_tonic, pred_mode = key_to_tonic_mode(key_pred)
        
        if gt_tonic is None or pred_tonic is None:
            continue

        if is_match:
            correct += 1
        else:
            confusion[(key_gt, key_pred)] += 1
            
            if gt_tonic == pred_tonic and gt_mode != pred_mode:
                mode_errors += 1
            elif gt_tonic != pred_tonic and gt_mode == pred_mode:
                tonic_errors += 1
            elif gt_tonic != pred_tonic and gt_mode != pred_mode:
                both_wrong += 1
            
            if gt_tonic != pred_tonic:
                cof_dist = circle_of_fifths_distance(gt_tonic, pred_tonic)
                cof_distances[cof_dist] += 1

    print(f"\nCorrect: {correct}/{len(key_rows)} ({100*correct/len(key_rows):.1f}%)")
    print(f"Errors: {len(key_rows) - correct}/{len(key_rows)} ({100*(len(key_rows)-correct)/len(key_rows):.1f}%)")
    
    print(f"\nError breakdown:")
    print(f"  Mode errors (same tonic, wrong mode): {mode_errors}")
    print(f"  Tonic errors (wrong tonic, same mode): {tonic_errors}")
    print(f"  Both wrong: {both_wrong}")

    print(f"\nCircle-of-fifths distance (tonic errors only):")
    for dist in sorted(cof_distances.keys()):
        print(f"  Distance {dist}: {cof_distances[dist]} errors")

    print(f"\nTop 10 most confused key pairs:")
    sorted_confusion = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
    for (gt, pred), count in sorted_confusion[:10]:
        print(f"  {gt} -> {pred}: {count}")

    # Mode-specific analysis
    print(f"\nMode confusion matrix:")
    mode_confusion = defaultdict(int)
    for track_id, key_gt, key_pred, is_match in key_rows:
        if not is_match:
            gt_tonic, gt_mode = key_to_tonic_mode(key_gt)
            pred_tonic, pred_mode = key_to_tonic_mode(key_pred)
            if gt_tonic is not None and pred_tonic is not None:
                mode_confusion[(gt_mode, pred_mode)] += 1
    
    for (gt_mode, pred_mode), count in sorted(mode_confusion.items(), key=lambda x: x[1], reverse=True):
        gt_str = "major" if gt_mode == 1 else "minor"
        pred_str = "major" if pred_mode == 1 else "minor"
        print(f"  GT {gt_str} -> Pred {pred_str}: {count}")


if __name__ == "__main__":
    main()

