#!/usr/bin/env python3
"""
Sweep tempogram tuning knobs quickly on a subset of the test batch.

Strategy:
- Run `validation.tools.run_validation` with --max-tracks N and a small parameter grid.
- Parse the "Results saved to:" line to locate the output CSV.
- Compute MAE + accuracy-at-tolerances directly from the CSV (fast, no extra scripts).
- Print a sorted leaderboard and optionally re-run top configs on full batch.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    # Allow running as a script: `python validation/tools/sweep_tempogram_tuning.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation._paths import find_repo_root

RESULT_RE = re.compile(r"^Results saved to:\s+(?P<path>.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Metrics:
    n: int
    mae: float
    acc2: float
    acc5: float
    acc10: float


def compute_metrics(csv_path: Path) -> Metrics:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    errs = [abs(float(r["bpm_error"])) for r in rows]
    n = len(errs)
    if n == 0:
        return Metrics(n=0, mae=float("inf"), acc2=0.0, acc5=0.0, acc10=0.0)

    mae = sum(errs) / n
    acc2 = sum(1 for e in errs if e <= 2.0) / n
    acc5 = sum(1 for e in errs if e <= 5.0) / n
    acc10 = sum(1 for e in errs if e <= 10.0) / n
    return Metrics(n=n, mae=mae, acc2=acc2, acc5=acc5, acc10=acc10)


def run_validation(
    repo_root: Path,
    binary: Path,
    max_tracks: Optional[int],
    extra_args: List[str],
    timeout_s: int,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "validation.tools.run_validation",
        "--binary",
        str(binary),
    ]
    if max_tracks is not None:
        cmd += ["--max-tracks", str(max_tracks)]
    cmd += extra_args

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"run_validation failed ({proc.returncode}). Output:\n{out}")

    m = RESULT_RE.search(out)
    if not m:
        raise RuntimeError(f"Could not find results path in output:\n{out}")

    rel = m.group("path").strip().strip('"').strip("'")
    # run_validation prints Windows-style relative paths like "..\\validation-data\\results\\..."
    results_path = (repo_root / rel).resolve()
    if not results_path.exists():
        raise RuntimeError(f"Results CSV not found at {results_path}")
    return results_path


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep tempogram tuning knobs (fast subset runs).")
    ap.add_argument("--binary", type=str, default=None, help="Path to analyze_file binary.")
    ap.add_argument("--max-tracks", type=int, default=50, help="Tracks per sweep run (default: 50).")
    ap.add_argument("--timeout-s", type=int, default=600, help="Per-run timeout seconds (default: 600).")
    ap.add_argument("--top-k", type=int, default=5, help="How many configs to re-run on full batch.")
    ap.add_argument("--rerun-full", action="store_true", help="Re-run top-k configs on full batch.")

    # Band consensus sweep
    ap.add_argument("--support-thresholds", type=str, default="0.20,0.25,0.30")
    ap.add_argument("--consensus-bonuses", type=str, default="0.06,0.08,0.10,0.12,0.15")

    # Mel novelty sweep (these matter now that mel is included in consensus gating)
    ap.add_argument("--mel", type=str, default="on,off", help="Comma-separated: on/off")
    ap.add_argument("--mel-n-mels", type=str, default="24,40,64")
    ap.add_argument("--mel-max-filter-bins", type=str, default="1,2,3")
    ap.add_argument("--mel-fmax-hz", type=str, default="6000,8000,12000")

    args = ap.parse_args()

    repo_root = find_repo_root()
    if args.binary:
        binary = Path(args.binary)
    else:
        # Prefer target_alt to avoid Windows exe locks (if present)
        candidates = [
            repo_root / "target_alt" / "release" / "examples" / "analyze_file.exe",
            repo_root / "target" / "release" / "examples" / "analyze_file.exe",
            repo_root / "target_alt" / "release" / "examples" / "analyze_file",
            repo_root / "target" / "release" / "examples" / "analyze_file",
        ]
        binary = next((p for p in candidates if p.exists()), candidates[0])

    if not binary.exists():
        raise SystemExit(f"Binary not found at {binary}. Build with: cargo build --release --example analyze_file")

    support_thresholds = [float(x) for x in args.support_thresholds.split(",") if x.strip()]
    consensus_bonuses = [float(x) for x in args.consensus_bonuses.split(",") if x.strip()]
    mel_modes = [x.strip().lower() for x in args.mel.split(",") if x.strip()]
    mel_n_mels = [int(x) for x in args.mel_n_mels.split(",") if x.strip()]
    mel_maxf = [int(x) for x in args.mel_max_filter_bins.split(",") if x.strip()]
    mel_fmax = [float(x) for x in args.mel_fmax_hz.split(",") if x.strip()]

    sweep: List[Tuple[List[str], str]] = []
    for st, cb, mel_mode, nm, k, fmax in itertools.product(
        support_thresholds, consensus_bonuses, mel_modes, mel_n_mels, mel_maxf, mel_fmax
    ):
        extra = [
            "--band-support-threshold",
            str(st),
            "--band-consensus-bonus",
            str(cb),
            "--mel-n-mels",
            str(nm),
            "--mel-max-filter-bins",
            str(k),
            "--mel-fmax-hz",
            str(fmax),
        ]
        label = f"st={st:.2f} cb={cb:.2f} mel={mel_mode} nm={nm} k={k} fmax={int(fmax)}"
        if mel_mode == "off":
            extra = ["--no-tempogram-mel-novelty"] + extra
        sweep.append((extra, label))

    print(f"Binary: {binary}")
    print(f"Sweep runs: {len(sweep)} (max_tracks={args.max_tracks})")

    results: List[Tuple[Metrics, str, Path, List[str]]] = []
    for i, (extra, label) in enumerate(sweep, start=1):
        print(f"[{i}/{len(sweep)}] {label}")
        csv_path = run_validation(
            repo_root=repo_root,
            binary=binary,
            max_tracks=args.max_tracks,
            extra_args=extra,
            timeout_s=args.timeout_s,
        )
        m = compute_metrics(csv_path)
        print(f"  -> acc2={fmt_pct(m.acc2)} mae={m.mae:.2f} (csv={csv_path.name})")
        results.append((m, label, csv_path, extra))

    # Sort primarily by acc2, then mae
    results.sort(key=lambda t: (-t[0].acc2, t[0].mae))

    print("\n=== Leaderboard (subset) ===")
    for rank, (m, label, csv_path, _) in enumerate(results[:20], start=1):
        print(f"{rank:>2}. acc2={fmt_pct(m.acc2)} acc5={fmt_pct(m.acc5)} mae={m.mae:.2f}  {label}  ({csv_path.name})")

    if args.rerun_full:
        top = results[: max(1, args.top_k)]
        print("\n=== Re-running top configs on full batch ===")
        for rank, (m0, label, _, extra) in enumerate(top, start=1):
            print(f"[full {rank}/{len(top)}] {label}")
            csv_path = run_validation(
                repo_root=repo_root,
                binary=binary,
                max_tracks=None,
                extra_args=extra,
                timeout_s=max(args.timeout_s, 1800),
            )
            m = compute_metrics(csv_path)
            print(f"  -> FULL acc2={fmt_pct(m.acc2)} acc5={fmt_pct(m.acc5)} mae={m.mae:.2f} (csv={csv_path.name})")


if __name__ == "__main__":
    main()


