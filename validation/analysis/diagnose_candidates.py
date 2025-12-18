"""
Diagnose tempogram candidate quality on an existing validation batch.

Goal: determine whether the *ground truth BPM* appears in the tempogram's top-N candidates
when Stratum's final BPM prediction is wrong.

This lets us distinguish:
  - "selection problem" (GT is present but not chosen)
  - "generation problem" (GT not even among candidates)

Usage examples:
  python -m validation.analysis.diagnose_candidates --results-csv ..\\validation-data\\results\\validation_results_20251217_193119.csv
  python -m validation.analysis.diagnose_candidates --results-csv ..\\validation-data\\results\\validation_results_20251217_193119.csv --top-n 15 --only-misses
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # Allow running as a script: `python validation/analysis/diagnose_candidates.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation._paths import find_repo_root, resolve_data_path


@dataclass
class TrackInfo:
    track_id: str
    filename: Path
    bpm_gt: float


def _find_latest_test_batch(results_dir: Path) -> Path:
    batches = sorted(results_dir.glob("test_batch_*.csv"), reverse=True)
    if batches:
        return batches[0]
    fallback = results_dir / "test_batch.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No test batch found in {results_dir}")


def load_test_batch(results_dir: Path) -> dict[str, TrackInfo]:
    batch_csv = _find_latest_test_batch(results_dir)
    out: dict[str, TrackInfo] = {}
    with open(batch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row["track_id"])
            fn = Path(row["filename"])
            bpm_gt = float(row["bpm_gt"])
            out[tid] = TrackInfo(track_id=tid, filename=fn, bpm_gt=bpm_gt)
    print(f"Using test batch: {batch_csv.name} (n={len(out)})")
    return out


def run_stratum(binary_path: Path, audio_file: Path, top_n: int) -> dict:
    cmd = [
        str(binary_path),
        str(audio_file),
        "--json",
        "--bpm-candidates",
        "--bpm-candidates-top",
        str(top_n),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"binary failed: code={result.returncode} stderr={result.stderr[:500]}")

    output = result.stdout.strip()
    start = output.find("{")
    end = output.rfind("}") + 1
    if start < 0 or end <= start:
        raise RuntimeError("No JSON found in output")
    return json.loads(output[start:end])


def find_rank(candidates: list[dict], bpm_gt: float, tol: float) -> int | None:
    for i, c in enumerate(candidates, 1):
        try:
            bpm = float(c.get("bpm", 0.0))
        except Exception:
            continue
        if abs(bpm - bpm_gt) <= tol:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose tempogram candidate quality")
    ap.add_argument("--data-path", default="../validation-data")
    ap.add_argument("--results-csv", required=True, help="Existing validation_results_*.csv file")
    ap.add_argument("--binary", default=None, help="Path to analyze_file binary")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--only-misses", action="store_true", help="Only re-run tracks where bpm_error > tol")
    args = ap.parse_args()

    repo_root = find_repo_root()
    data_path = resolve_data_path(args.data_path, repo_root)
    results_dir = data_path / "results"
    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        print(f"ERROR: results csv not found: {results_csv}")
        return 1

    if args.binary:
        binary_path = Path(args.binary)
    else:
        if sys.platform == "win32":
            candidates = [
                repo_root / "target" / "release" / "examples" / "analyze_file.exe",
                repo_root / "target_alt" / "release" / "examples" / "analyze_file.exe",
                repo_root / "target-alt" / "release" / "examples" / "analyze_file.exe",
                repo_root / "target" / "debug" / "examples" / "analyze_file.exe",
            ]
            binary_path = next((p for p in candidates if p.exists()), candidates[0])
        else:
            candidates = [
                repo_root / "target" / "release" / "examples" / "analyze_file",
                repo_root / "target_alt" / "release" / "examples" / "analyze_file",
                repo_root / "target-alt" / "release" / "examples" / "analyze_file",
                repo_root / "target" / "debug" / "examples" / "analyze_file",
            ]
            binary_path = next((p for p in candidates if p.exists()), candidates[0])

    if not binary_path.exists():
        print(f"ERROR: analyze_file binary not found at {binary_path}")
        return 1

    batch = load_test_batch(results_dir)

    rows = list(csv.DictReader(open(results_csv, "r", encoding="utf-8")))
    if not rows:
        print("ERROR: empty results CSV")
        return 1

    # Select target tracks
    targets = []
    for r in rows:
        tid = str(r["track_id"])
        if tid not in batch:
            continue
        try:
            bpm_err = float(r["bpm_error"])
        except Exception:
            continue
        if args.only_misses and bpm_err <= args.tol:
            continue
        targets.append((tid, bpm_err))

    print(f"Diagnosing candidates for {len(targets)}/{len(rows)} tracks (only_misses={args.only_misses})")

    out_rows = []
    hit_top = 0
    miss_top = 0

    for i, (tid, bpm_err) in enumerate(targets, 1):
        info = batch[tid]
        print(f"[{i}/{len(targets)}] {tid} err={bpm_err:.2f} ...", end=" ", flush=True)
        try:
            res = run_stratum(binary_path, info.filename, args.top_n)
            cands = res.get("bpm_candidates", []) or []
            rank = find_rank(cands, info.bpm_gt, args.tol)
            in_top = rank is not None
            if in_top:
                hit_top += 1
            else:
                miss_top += 1
            out_rows.append(
                {
                    "track_id": tid,
                    "filename": str(info.filename),
                    "bpm_gt": info.bpm_gt,
                    "prev_bpm_error": bpm_err,
                    "gt_in_topn": "YES" if in_top else "NO",
                    "gt_rank": rank if rank is not None else "",
                    "bpm_candidates_json": json.dumps(cands),
                }
            )
            print(f"gt_in_topn={out_rows[-1]['gt_in_topn']} rank={out_rows[-1]['gt_rank'] or 'N/A'}")
        except Exception as e:
            print(f"ERROR ({e})")
            out_rows.append(
                {
                    "track_id": tid,
                    "filename": str(info.filename),
                    "bpm_gt": info.bpm_gt,
                    "prev_bpm_error": bpm_err,
                    "gt_in_topn": "ERROR",
                    "gt_rank": "",
                    "bpm_candidates_json": "",
                }
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = results_dir / f"candidate_diagnosis_{ts}.csv"
    if out_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

    print()
    print("========================================")
    print("CANDIDATE DIAGNOSIS SUMMARY")
    print("========================================")
    print(f"Output: {out_csv}")
    if hit_top + miss_top > 0:
        print(f"GT in top-{args.top_n} (±{args.tol}): {hit_top}/{hit_top+miss_top} ({100.0*hit_top/(hit_top+miss_top):.1f}%)")
    else:
        print("No valid tracks diagnosed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


