"""
Analyze fallback usage flags in a validation_results_*.csv.

Focus:
- How often multi-res escalation triggers vs is accepted
- How often percussive fallback triggers vs is accepted
- Conditional accuracy/MAE when used vs not used
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _to_bool(v: str):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--tol", type=float, default=2.0)
    args = ap.parse_args()

    path = Path(args.file)
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    if not rows:
        print("Empty file")
        return 0

    def subset(pred):
        return [r for r in rows if pred(r)]

    def mae(rs):
        errs = [float(r["bpm_error"]) for r in rs]
        return sum(errs) / len(errs) if errs else float("nan")

    def acc(rs, tol):
        errs = [float(r["bpm_error"]) for r in rs]
        return 100.0 * sum(1 for e in errs if e <= tol) / len(errs) if errs else float("nan")

    mr_trig = subset(lambda r: _to_bool(r.get("tempogram_multi_res_triggered", "")) is True)
    mr_used = subset(lambda r: _to_bool(r.get("tempogram_multi_res_used", "")) is True)
    perc_trig = subset(lambda r: _to_bool(r.get("tempogram_percussive_triggered", "")) is True)
    perc_used = subset(lambda r: _to_bool(r.get("tempogram_percussive_used", "")) is True)

    print(f"File: {path.name}")
    print(f"n={len(rows)} tol=±{args.tol}")
    print()

    def line(name, rs):
        print(f"{name:28s} n={len(rs):3d} acc={acc(rs, args.tol):5.1f}%  mae={mae(rs):6.2f}")

    # Overall
    line("OVERALL", rows)
    print()

    # Multi-res
    print("Multi-resolution:")
    print(f"  triggered: {len(mr_trig)}/{len(rows)} ({100.0*len(mr_trig)/len(rows):.1f}%)")
    print(f"  used:      {len(mr_used)}/{len(rows)} ({100.0*len(mr_used)/len(rows):.1f}%)")
    if mr_used:
        line("  when used", mr_used)
    line("  when NOT used", subset(lambda r: _to_bool(r.get("tempogram_multi_res_used", "")) is not True))
    print()

    # Percussive
    print("Percussive fallback:")
    print(f"  triggered: {len(perc_trig)}/{len(rows)} ({100.0*len(perc_trig)/len(rows):.1f}%)")
    print(f"  used:      {len(perc_used)}/{len(rows)} ({100.0*len(perc_used)/len(rows):.1f}%)")
    if perc_used:
        line("  when used", perc_used)
    line("  when NOT used", subset(lambda r: _to_bool(r.get("tempogram_percussive_used", "")) is not True))
    print()

    # Interaction
    both_used = subset(
        lambda r: _to_bool(r.get("tempogram_multi_res_used", "")) is True
        and _to_bool(r.get("tempogram_percussive_used", "")) is True
    )
    if both_used:
        line("Both MR+Perc used", both_used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


