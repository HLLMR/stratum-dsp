#!/usr/bin/env python3
"""
Run validation on test batch and compare results to ground truth.

This script runs stratum-dsp on each track in the test batch and compares
the results to the ground truth values from FMA metadata.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_key(key_str: str) -> str:
    """Parse key string to standard format (e.g., 'C', 'Am', 'F#', 'D#m')."""
    # FMA keys might be in various formats, normalize them
    key_str = key_str.strip().upper()
    
    # Handle common variations
    if key_str.endswith("MAJOR") or key_str.endswith("MAJ"):
        key_str = key_str.replace("MAJOR", "").replace("MAJ", "").strip()
    elif key_str.endswith("MINOR") or key_str.endswith("MIN"):
        key_str = key_str.replace("MINOR", "").replace("MIN", "").strip() + "m"
    
    # Handle flat notation (e.g., "Bb" -> "A#")
    key_map = {
        "DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#",
        "DBM": "C#M", "EBM": "D#M", "GBM": "F#M", "ABM": "G#M", "BBM": "A#M",
    }
    for old, new in key_map.items():
        if key_str.startswith(old):
            key_str = key_str.replace(old, new)
    
    return key_str


def run_stratum_dsp(binary_path: Path, audio_file: Path, extra_args=None) -> dict:
    """Run stratum-dsp on an audio file and return parsed results."""
    if extra_args is None:
        extra_args = []
    cmd = [str(binary_path), str(audio_file), "--json", *extra_args]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return {
                "error": f"Process exited with code {result.returncode}",
                "stderr": result.stderr,
            }
        
        # Parse JSON output
        try:
            # Extract JSON from output (might have other text)
            output = result.stdout.strip()
            # Find JSON object
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = output[start:end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON found in output"}
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON: {e}",
                "stdout": result.stdout,
            }
    
    except subprocess.TimeoutExpired:
        return {"error": "Process timed out after 5 minutes"}
    except Exception as e:
        return {"error": f"Failed to run command: {e}"}


def main():
    parser = argparse.ArgumentParser(
        description="Run validation on test batch"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../validation-data",
        help="Path to validation data directory (default: ../validation-data)",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to stratum-dsp binary (default: ../target/release/examples/analyze_file)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocessing (normalization + silence trimming) in the analyze_file binary",
    )
    parser.add_argument(
        "--no-onset-consensus",
        action="store_true",
        help="Disable onset consensus (use energy-flux-only onset list) in the analyze_file binary",
    )
    parser.add_argument(
        "--force-legacy-bpm",
        action="store_true",
        help="Force legacy BPM estimation (Phase 1B) in the analyze_file binary (skip tempogram)",
    )
    parser.add_argument(
        "--bpm-fusion",
        action="store_true",
        help="Enable BPM fusion (compute tempogram + legacy in parallel) in the analyze_file binary",
    )
    parser.add_argument("--legacy-preferred-min", type=float, default=None)
    parser.add_argument("--legacy-preferred-max", type=float, default=None)
    parser.add_argument("--legacy-soft-min", type=float, default=None)
    parser.add_argument("--legacy-soft-max", type=float, default=None)
    parser.add_argument("--legacy-mul-preferred", type=float, default=None)
    parser.add_argument("--legacy-mul-soft", type=float, default=None)
    parser.add_argument("--legacy-mul-extreme", type=float, default=None)
    
    args = parser.parse_args()
    
    # Paths
    data_path = Path(args.data_path)
    results_dir = data_path / "results"
    
    # Find the most recent test batch (prefer timestamped ones)
    test_batches = sorted(results_dir.glob("test_batch_*.csv"), reverse=True)
    if test_batches:
        test_batch_csv = test_batches[0]
        print(f"Using test batch: {test_batch_csv.name}")
    else:
        # Fallback to non-timestamped file
        test_batch_csv = results_dir / "test_batch.csv"
        if not test_batch_csv.exists():
            print(f"ERROR: Test batch not found")
            print("Run prepare_test_batch.py first")
            sys.exit(1)
    
    # Determine binary path
    if args.binary:
        binary_path = Path(args.binary)
    else:
        # Default: look for the example binary (relative to repo root)
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        if sys.platform == "win32":
            binary_path = repo_root / "target" / "release" / "examples" / "analyze_file.exe"
        else:
            binary_path = repo_root / "target" / "release" / "examples" / "analyze_file"
    
    if not test_batch_csv.exists():
        print(f"ERROR: Test batch not found at {test_batch_csv}")
        print("Run prepare_test_batch.py first")
        sys.exit(1)
    
    if not binary_path.exists():
        print(f"ERROR: stratum-dsp binary not found at {binary_path}")
        print("Build with: cargo build --release")
        sys.exit(1)
    
    # Load test batch
    print("Loading test batch...")
    test_batch = []
    with open(test_batch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_batch.append(row)
    
    print(f"Running validation on {len(test_batch)} tracks...")
    print(f"Using binary: {binary_path}")
    if args.no_preprocess:
        print("Preprocessing: DISABLED (--no-preprocess)")
    if args.no_onset_consensus:
        print("Onset consensus: DISABLED (--no-onset-consensus)")
    if args.force_legacy_bpm:
        print("BPM mode: LEGACY ONLY (--force-legacy-bpm)")
    if args.bpm_fusion:
        print("BPM mode: FUSION (--bpm-fusion)")
    print()
    
    extra_args = []
    if args.no_preprocess:
        extra_args.append("--no-preprocess")
    if args.no_onset_consensus:
        extra_args.append("--no-onset-consensus")
    if args.force_legacy_bpm:
        extra_args.append("--force-legacy-bpm")
    if args.bpm_fusion:
        extra_args.append("--bpm-fusion")

    # Pass-through tuning flags (if provided)
    if args.legacy_preferred_min is not None:
        extra_args += ["--legacy-preferred-min", str(args.legacy_preferred_min)]
    if args.legacy_preferred_max is not None:
        extra_args += ["--legacy-preferred-max", str(args.legacy_preferred_max)]
    if args.legacy_soft_min is not None:
        extra_args += ["--legacy-soft-min", str(args.legacy_soft_min)]
    if args.legacy_soft_max is not None:
        extra_args += ["--legacy-soft-max", str(args.legacy_soft_max)]
    if args.legacy_mul_preferred is not None:
        extra_args += ["--legacy-mul-preferred", str(args.legacy_mul_preferred)]
    if args.legacy_mul_soft is not None:
        extra_args += ["--legacy-mul-soft", str(args.legacy_mul_soft)]
    if args.legacy_mul_extreme is not None:
        extra_args += ["--legacy-mul-extreme", str(args.legacy_mul_extreme)]
    
    results = []
    
    for i, track in enumerate(test_batch, 1):
        track_id = track["track_id"]
        audio_file = Path(track["filename"])
        bpm_gt = float(track["bpm_gt"])
        key_gt = track["key_gt"]
        
        print(f"[{i}/{len(test_batch)}] Processing track {track_id}...", end=" ", flush=True)
        
        if not audio_file.exists():
            print("ERROR: Audio file not found")
            continue
        
        # Run stratum-dsp
        analysis_result = run_stratum_dsp(binary_path, audio_file, extra_args)
        
        if "error" in analysis_result:
            print(f"ERROR: {analysis_result['error']}")
            continue
        
        # Extract results
        pred_bpm = analysis_result.get("bpm")
        pred_key = analysis_result.get("key")
        
        if pred_bpm is None or pred_key is None:
            print("ERROR: Missing BPM or key in results")
            continue
        
        # Compare to ground truth
        bpm_error = abs(pred_bpm - bpm_gt)
        # Handle missing key ground truth (FMA doesn't have key annotations)
        if key_gt and key_gt.strip():
            key_match = "YES" if parse_key(pred_key) == parse_key(key_gt) else "NO"
        else:
            key_match = "N/A"  # No ground truth available
        
        results.append({
            "track_id": track_id,
            "genre": track["genre"],
            "bpm_gt": bpm_gt,
            "bpm_pred": pred_bpm,
            "bpm_error": bpm_error,
            "key_gt": key_gt,
            "key_pred": pred_key,
            "key_match": key_match,
            "bpm_confidence": analysis_result.get("bpm_confidence", 0.0),
            "key_confidence": analysis_result.get("key_confidence", 0.0),
            "key_clarity": analysis_result.get("key_clarity", 0.0),
            "grid_stability": analysis_result.get("grid_stability", 0.0),
        })
        
        print(f"BPM: {pred_bpm:.1f} (error: {bpm_error:.1f}), Key: {pred_key} ({key_match})")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = results_dir / f"validation_results_{timestamp}.csv"
    if results:
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Print summary
    print()
    print("=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    
    if results:
        avg_bpm_error = sum(r["bpm_error"] for r in results) / len(results)
        key_accuracy = sum(1 for r in results if r["key_match"] == "YES") / len(results) * 100
        
        # BPM accuracy within ±2 BPM
        bpm_accuracy_2 = sum(1 for r in results if r["bpm_error"] <= 2.0) / len(results) * 100
        
        print(f"Tracks tested: {len(results)}")
        print(f"BPM MAE: ±{avg_bpm_error:.2f}")
        print(f"BPM accuracy (±2 BPM): {bpm_accuracy_2:.1f}%")
        print(f"Key accuracy: {key_accuracy:.1f}%")
        print()
        print(f"Results saved to: {results_csv}")
    else:
        print("No results to summarize")
    
    print()
    print("Target accuracy:")
    print("  BPM: 88% (±2 BPM tolerance)")
    print("  Key: 77% (exact match)")


if __name__ == "__main__":
    main()

