#!/usr/bin/env python3
"""Analyze validation results (Stratum vs TAG if present)"""

import argparse
import csv
import glob
import os
import statistics


def find_latest_results_file() -> str:
    results_dir = "../validation-data/results"
    pattern = os.path.join(results_dir, "validation_results_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No validation results files found!")
    return max(files, key=os.path.getmtime)


def analyze_file(results_file: str) -> None:
    print(f"Using results file: {os.path.basename(results_file)}")

    with open(results_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors = [abs(float(row["bpm_error"])) for row in rows]
    has_tag = "bpm_tag_error" in (rows[0] if rows else {})
    tag_errors = []
    if has_tag:
        for row in rows:
            v = row.get("bpm_tag_error", "")
            if v != "":
                try:
                    tag_errors.append(abs(float(v)))
                except ValueError:
                    pass

    print("=" * 60)
    print("VALIDATION RESULTS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal tracks: {len(rows)}")
    print(f"Stratum MAE: {statistics.mean(errors):.2f} BPM")
    if tag_errors:
        print(f"TAG MAE: {statistics.mean(tag_errors):.2f} BPM (n={len(tag_errors)})")

    # Key summary (GT or TAG reference, depending on key_ref)
    if rows and "key_match" in rows[0]:
        key_rows = [r for r in rows if r.get("key_match") in ("YES", "NO")]
        if key_rows:
            ref_gt = [r for r in key_rows if r.get("key_ref") == "GT"]
            ref_tag = [r for r in key_rows if r.get("key_ref") == "TAG"]
            if ref_gt:
                acc_gt = sum(1 for r in ref_gt if r["key_match"] == "YES") / len(ref_gt) * 100
                print(f"Stratum Key accuracy vs GT: {acc_gt:.1f}% (n={len(ref_gt)})")
            if ref_tag:
                acc_tag = sum(1 for r in ref_tag if r["key_match"] == "YES") / len(ref_tag) * 100
                print(f"Stratum Key agreement vs TAG: {acc_tag:.1f}% (n={len(ref_tag)})")
        else:
            # Preserve old behavior: many batches have no key GT.
            pass
    print("\nAccuracy:")
    print(
        f"  Within ±2 BPM: {sum(1 for e in errors if e <= 2)}/{len(errors)} "
        f"({100*sum(1 for e in errors if e <= 2)/len(errors):.1f}%)"
    )
    if tag_errors:
        print(
            f"  TAG within ±2 BPM: {sum(1 for e in tag_errors if e <= 2)}/{len(tag_errors)} "
            f"({100*sum(1 for e in tag_errors if e <= 2)/len(tag_errors):.1f}%)"
        )
    print(
        f"  Within ±5 BPM: {sum(1 for e in errors if e <= 5)}/{len(errors)} "
        f"({100*sum(1 for e in errors if e <= 5)/len(errors):.1f}%)"
    )
    if tag_errors:
        print(
            f"  TAG within ±5 BPM: {sum(1 for e in tag_errors if e <= 5)}/{len(tag_errors)} "
            f"({100*sum(1 for e in tag_errors if e <= 5)/len(tag_errors):.1f}%)"
        )
    print(
        f"  Within ±10 BPM: {sum(1 for e in errors if e <= 10)}/{len(errors)} "
        f"({100*sum(1 for e in errors if e <= 10)/len(errors):.1f}%)"
    )
    if tag_errors:
        print(
            f"  TAG within ±10 BPM: {sum(1 for e in tag_errors if e <= 10)}/{len(tag_errors)} "
            f"({100*sum(1 for e in tag_errors if e <= 10)/len(tag_errors):.1f}%)"
        )
    print(
        f"  Within ±20 BPM: {sum(1 for e in errors if e <= 20)}/{len(errors)} "
        f"({100*sum(1 for e in errors if e <= 20)/len(errors):.1f}%)"
    )
    if tag_errors:
        print(
            f"  TAG within ±20 BPM: {sum(1 for e in tag_errors if e <= 20)}/{len(tag_errors)} "
            f"({100*sum(1 for e in tag_errors if e <= 20)/len(tag_errors):.1f}%)"
        )

    print("\nStratum error distribution:")
    print(f"  < 5 BPM: {sum(1 for e in errors if e < 5)}")
    print(f"  5-20 BPM: {sum(1 for e in errors if 5 <= e < 20)}")
    print(f"  20-50 BPM: {sum(1 for e in errors if 20 <= e < 50)}")
    print(f"  50-100 BPM: {sum(1 for e in errors if 50 <= e < 100)}")
    print(f"  > 100 BPM: {sum(1 for e in errors if e >= 100)}")
    if tag_errors:
        print("\nTAG error distribution:")
        print(f"  < 5 BPM: {sum(1 for e in tag_errors if e < 5)}")
        print(f"  5-20 BPM: {sum(1 for e in tag_errors if 5 <= e < 20)}")
        print(f"  20-50 BPM: {sum(1 for e in tag_errors if 20 <= e < 50)}")
        print(f"  50-100 BPM: {sum(1 for e in tag_errors if 50 <= e < 100)}")
        print(f"  > 100 BPM: {sum(1 for e in tag_errors if e >= 100)}")

    print("\nWorst Stratum errors:")
    worst = sorted(rows, key=lambda x: abs(float(x["bpm_error"])), reverse=True)[:10]
    for w in worst:
        print(
            f"  Track {w['track_id']}: GT={w['bpm_gt']}, Pred={w['bpm_pred']}, Error={w['bpm_error']}"
        )
    if tag_errors:
        print("\nWorst TAG errors:")
        worst_tag = sorted(
            [r for r in rows if r.get("bpm_tag_error", "") != ""],
            key=lambda x: abs(float(x["bpm_tag_error"])),
            reverse=True,
        )[:10]
        for w in worst_tag:
            print(
                f"  Track {w['track_id']}: GT={w['bpm_gt']}, TAG={w.get('bpm_tag','')}, Error={w.get('bpm_tag_error','')}"
            )

    print("\nPattern analysis:")
    print("Tracks with ~60 BPM predictions (likely floor effect):")
    floor60 = [row for row in rows if 59 <= float(row["bpm_pred"]) <= 61]
    print(f"  Count: {len(floor60)}")
    if floor60:
        examples = [f"{r['track_id']} (GT={r['bpm_gt']}, Pred={r['bpm_pred']})" for r in floor60[:5]]
        print(f"  Examples: {', '.join(examples)}")

    print("\nTracks with ~40-50 BPM predictions (subharmonics):")
    subharm = [row for row in rows if 40 <= float(row["bpm_pred"]) <= 50]
    print(f"  Count: {len(subharm)}")
    if subharm:
        examples = [f"{r['track_id']} (GT={r['bpm_gt']}, Pred={r['bpm_pred']})" for r in subharm[:5]]
        print(f"  Examples: {', '.join(examples)}")

    print("\nTracks with good predictions (<5 BPM error):")
    good = [row for row in rows if abs(float(row["bpm_error"])) < 5]
    print(f"  Count: {len(good)}")
    if good:
        examples = [
            f"{r['track_id']} (GT={r['bpm_gt']}, Pred={r['bpm_pred']}, Error={r['bpm_error']})"
            for r in good
        ]
        print(f"  All good tracks: {', '.join(examples)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one or more validation result CSV files")
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        default=[],
        help="Path to a validation_results_*.csv file. Can be provided multiple times.",
    )
    args = parser.parse_args()

    files = args.files or [find_latest_results_file()]
    for idx, fpath in enumerate(files):
        if idx > 0:
            print("\n")
        analyze_file(fpath)


if __name__ == "__main__":
    main()

