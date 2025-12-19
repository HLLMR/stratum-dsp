#!/usr/bin/env python3
"""
Run validation on test batch and compare results to ground truth.

This script runs stratum-dsp on each track in the test batch and compares
the results to the ground truth values from FMA metadata.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    # Allow running as a script: `python validation/tools/run_validation.py`
    # (module imports require repo root on sys.path).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation._paths import find_repo_root, resolve_data_path
from validation._id3 import read_id3_text_fields
from validation._keys import normalize_key

def read_tag_bpm_key(mp3_path: Path) -> dict:
    """
    Read BPM and Key from ID3 tags (source label: TAG).

    Note: This is used as the 'TAG' baseline (e.g., after you run MIK and write tags).
    Ground truth comes from the prepared test batch CSV.
    """
    fields = read_id3_text_fields(mp3_path)
    return {"bpm_tag": fields.get("bpm"), "key_tag": fields.get("key", "")}


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
                data = json.loads(json_str)
                if result.stderr:
                    data["_stderr"] = result.stderr
                return data
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
    parser.add_argument(
        "--no-tempogram-multi-res",
        action="store_true",
        help="Disable true multi-resolution tempogram BPM estimation (use single hop_size only)",
    )
    parser.add_argument(
        "--no-tempogram-percussive",
        action="store_true",
        help="Disable HPSS percussive-only tempogram fallback (ambiguous-only)",
    )
    parser.add_argument(
        "--no-tempogram-band-fusion",
        action="store_true",
        help="Disable multi-band novelty fusion inside the tempogram estimator",
    )
    parser.add_argument("--band-low-max-hz", type=float, default=None)
    parser.add_argument("--band-mid-max-hz", type=float, default=None)
    parser.add_argument("--band-high-max-hz", type=float, default=None)
    parser.add_argument("--band-w-full", type=float, default=None)
    parser.add_argument("--band-w-low", type=float, default=None)
    parser.add_argument("--band-w-mid", type=float, default=None)
    parser.add_argument("--band-w-high", type=float, default=None)
    parser.add_argument("--band-support-threshold", type=float, default=None)
    parser.add_argument("--band-consensus-bonus", type=float, default=None)
    parser.add_argument("--superflux-max-filter-bins", type=int, default=None)
    parser.add_argument(
        "--band-score-fusion",
        action="store_true",
        help="Let bands affect scoring (not just candidate seeding). More aggressive; can increase metrical errors.",
    )
    parser.add_argument(
        "--no-tempogram-mel-novelty",
        action="store_true",
        help="Disable log-mel novelty tempogram variant",
    )
    parser.add_argument("--mel-n-mels", type=int, default=None)
    parser.add_argument("--mel-fmin-hz", type=float, default=None)
    parser.add_argument("--mel-fmax-hz", type=float, default=None)
    parser.add_argument("--mel-max-filter-bins", type=int, default=None)
    parser.add_argument("--mel-weight", type=float, default=None)
    parser.add_argument("--novelty-w-spectral", type=float, default=None)
    parser.add_argument("--novelty-w-energy", type=float, default=None)
    parser.add_argument("--novelty-w-hfc", type=float, default=None)
    parser.add_argument("--novelty-local-mean-window", type=int, default=None)
    parser.add_argument("--novelty-smooth-window", type=int, default=None)

    # Key HPSS tuning (pass-through to analyze_file)
    parser.add_argument(
        "--key-hpss",
        action="store_true",
        help="Enable median-filter HPSS harmonic mask for key detection (key-only).",
    )
    parser.add_argument(
        "--no-key-hpss",
        action="store_true",
        help="Disable key HPSS harmonic mask for key detection (key-only).",
    )
    parser.add_argument("--key-hpss-frame-step", type=int, default=None)
    parser.add_argument("--key-hpss-time-margin", type=int, default=None)
    parser.add_argument("--key-hpss-freq-margin", type=int, default=None)
    parser.add_argument("--key-hpss-mask-power", type=float, default=None)

    # Key mode heuristic + minor harmonic bonus (pass-through to analyze_file)
    parser.add_argument(
        "--key-mode-heuristic",
        action="store_true",
        help="Enable conservative key mode heuristic (parallel major/minor flip gate).",
    )
    parser.add_argument("--key-mode-third-margin", type=float, default=None)
    parser.add_argument("--key-mode-flip-min-score-ratio", type=float, default=None)
    parser.add_argument(
        "--key-minor-harmonic-bonus",
        action="store_true",
        help="Enable minor harmonic (leading-tone) bonus when scoring templates.",
    )
    parser.add_argument("--key-minor-leading-tone-bonus-weight", type=float, default=None)

    # Key segment voting (pass-through to analyze_file)
    parser.add_argument(
        "--no-key-segment-voting",
        action="store_true",
        help="Disable key segment voting (windowed key detection + score accumulation).",
    )
    parser.add_argument("--key-segment-len-frames", type=int, default=None)
    parser.add_argument("--key-segment-hop-frames", type=int, default=None)
    parser.add_argument("--key-segment-min-clarity", type=float, default=None)

    # Key HPCP (pass-through to analyze_file)
    parser.add_argument(
        "--key-hpcp",
        action="store_true",
        help="Enable HPCP-style pitch-class profile extraction for key detection.",
    )
    parser.add_argument("--key-hpcp-peaks", type=int, default=None)
    parser.add_argument("--key-hpcp-harmonics", type=int, default=None)
    parser.add_argument("--key-hpcp-harmonic-decay", type=float, default=None)
    parser.add_argument("--key-hpcp-mag-power", type=float, default=None)
    parser.add_argument(
        "--key-hpcp-whitening",
        action="store_true",
        help="Enable spectral whitening for HPCP peak picking (timbre suppression).",
    )
    parser.add_argument("--key-hpcp-whitening-smooth-bins", type=int, default=None)

    # Key-only STFT override (pass-through to analyze_file)
    parser.add_argument(
        "--key-stft-override",
        action="store_true",
        help="Compute a separate STFT for key detection (can increase frequency resolution for key).",
    )
    parser.add_argument(
        "--no-key-stft-override",
        action="store_true",
        help="Disable key-only STFT override (force shared STFT for key detection).",
    )
    parser.add_argument("--key-stft-frame-size", type=int, default=None)
    parser.add_argument("--key-stft-hop-size", type=int, default=None)

    # Key log-frequency spectrogram (pass-through to analyze_file)
    parser.add_argument(
        "--key-log-freq",
        action="store_true",
        help="Enable log-frequency (semitone-aligned) spectrogram for key detection.",
    )
    parser.add_argument(
        "--no-key-log-freq",
        action="store_true",
        help="Disable log-frequency spectrogram (use linear STFT with frequency-to-semitone mapping).",
    )

    # Key beat-synchronous chroma (pass-through to analyze_file)
    parser.add_argument(
        "--key-beat-sync",
        action="store_true",
        help="Enable beat-synchronous chroma extraction (align chroma windows to beat boundaries).",
    )
    parser.add_argument(
        "--no-key-beat-sync",
        action="store_true",
        help="Disable beat-synchronous chroma (use frame-based chroma extraction).",
    )

    # Key multi-scale detection (pass-through to analyze_file)
    parser.add_argument(
        "--key-multi-scale",
        action="store_true",
        help="Enable multi-scale key detection (ensemble voting across multiple time scales).",
    )
    parser.add_argument(
        "--no-key-multi-scale",
        action="store_true",
        help="Disable multi-scale key detection (use single-scale detection).",
    )
    parser.add_argument("--key-multi-scale-lengths", type=str, default=None)
    parser.add_argument("--key-multi-scale-hop", type=int, default=None)
    parser.add_argument("--key-multi-scale-min-clarity", type=float, default=None)
    parser.add_argument("--key-multi-scale-weights", type=str, default=None)

    # Key template set selection (pass-through to analyze_file)
    parser.add_argument(
        "--key-template-temperley",
        action="store_true",
        help="Use Temperley (1999) templates instead of Krumhansl-Kessler (1982).",
    )
    parser.add_argument(
        "--key-template-kk",
        action="store_true",
        help="Use Krumhansl-Kessler (1982) templates (default).",
    )

    # Key ensemble detection (pass-through to analyze_file)
    parser.add_argument(
        "--key-ensemble",
        action="store_true",
        help="Enable ensemble key detection (combine K-K and Temperley template scores).",
    )
    parser.add_argument(
        "--no-key-ensemble",
        action="store_true",
        help="Disable ensemble key detection (use single template set).",
    )
    parser.add_argument("--key-ensemble-kk-weight", type=float, default=None)
    parser.add_argument("--key-ensemble-temperley-weight", type=float, default=None)

    # Key median detection (pass-through to analyze_file)
    parser.add_argument(
        "--key-median",
        action="store_true",
        help="Enable median key detection (detect from multiple short segments, select median).",
    )
    parser.add_argument(
        "--no-key-median",
        action="store_true",
        help="Disable median key detection (use global key detection).",
    )
    parser.add_argument("--key-median-segment-length-frames", type=int, default=None)
    parser.add_argument("--key-median-segment-hop-frames", type=int, default=None)
    parser.add_argument("--key-median-min-segments", type=int, default=None)

    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit validation to the first N tracks of the loaded batch (useful for quick tuning)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Parallel workers for batch processing (default: CPU-1, keeping one core free). "
            "Use 1 to disable parallelism."
        ),
    )
    parser.add_argument(
        "--debug-track-ids",
        type=str,
        default="",
        help="Comma-separated track IDs to emit multi-res debug output for (e.g., '40244,11788')",
    )
    parser.add_argument("--multi-res-top-k", type=int, default=None)
    parser.add_argument("--multi-res-w512", type=float, default=None)
    parser.add_argument("--multi-res-w256", type=float, default=None)
    parser.add_argument("--multi-res-w1024", type=float, default=None)
    parser.add_argument("--multi-res-structural-discount", type=float, default=None)
    parser.add_argument("--multi-res-double-time-512-factor", type=float, default=None)
    parser.add_argument("--multi-res-margin-threshold", type=float, default=None)
    parser.add_argument("--multi-res-human-prior", action="store_true")
    parser.add_argument("--legacy-preferred-min", type=float, default=None)
    parser.add_argument("--legacy-preferred-max", type=float, default=None)
    parser.add_argument("--legacy-soft-min", type=float, default=None)
    parser.add_argument("--legacy-soft-max", type=float, default=None)
    parser.add_argument("--legacy-mul-preferred", type=float, default=None)
    parser.add_argument("--legacy-mul-soft", type=float, default=None)
    parser.add_argument("--legacy-mul-extreme", type=float, default=None)
    
    args = parser.parse_args()

    # Default parallelism: keep one core free for the system.
    if args.jobs is None:
        cpu_n = os.cpu_count() or 1
        args.jobs = max(1, cpu_n - 1)
    else:
        args.jobs = max(1, int(args.jobs))
    
    repo_root = find_repo_root()

    # Paths (treat relative --data-path as relative to repo root)
    data_path = resolve_data_path(args.data_path, repo_root)
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
        if sys.platform == "win32":
            candidates = [
                repo_root / "target" / "release" / "examples" / "analyze_file.exe",
                # Some workflows build into an alternate target dir to avoid Windows exe file locks.
                repo_root / "target_alt" / "release" / "examples" / "analyze_file.exe",
                repo_root / "target-alt" / "release" / "examples" / "analyze_file.exe",
                # Debug fallback (useful for quick iteration)
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
    
    if not test_batch_csv.exists():
        print(f"ERROR: Test batch not found at {test_batch_csv}")
        print("Run prepare_test_batch.py first")
        sys.exit(1)
    
    if not binary_path.exists():
        print(f"ERROR: stratum-dsp binary not found at {binary_path}")
        print("Build with: cargo build --release --example analyze_file")
        print("Or pass --binary PATH to override")
        sys.exit(1)
    
    # Load test batch
    print("Loading test batch...")
    test_batch = []
    with open(test_batch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_batch.append(row)

    if args.max_tracks is not None:
        n = max(1, int(args.max_tracks))
        test_batch = test_batch[:n]
        print(f"NOTE: Limiting run to first {len(test_batch)} tracks (--max-tracks)")
    
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
    if args.no_tempogram_multi_res:
        extra_args.append("--no-tempogram-multi-res")
    if args.no_tempogram_percussive:
        extra_args.append("--no-tempogram-percussive")
    if args.no_tempogram_band_fusion:
        extra_args.append("--no-tempogram-band-fusion")
    if args.band_score_fusion:
        extra_args.append("--band-score-fusion")
    if args.no_tempogram_mel_novelty:
        extra_args.append("--no-tempogram-mel-novelty")

    # Pass-through key HPSS flags (if provided)
    if args.no_key_hpss:
        extra_args.append("--no-key-hpss")
    if args.key_hpss:
        extra_args.append("--key-hpss")
    if args.key_hpss_frame_step is not None:
        extra_args += ["--key-hpss-frame-step", str(int(args.key_hpss_frame_step))]
    if args.key_hpss_time_margin is not None:
        extra_args += ["--key-hpss-time-margin", str(int(args.key_hpss_time_margin))]
    if args.key_hpss_freq_margin is not None:
        extra_args += ["--key-hpss-freq-margin", str(int(args.key_hpss_freq_margin))]
    if args.key_hpss_mask_power is not None:
        extra_args += ["--key-hpss-mask-power", str(float(args.key_hpss_mask_power))]

    # Pass-through key heuristic flags (if provided)
    if args.key_mode_heuristic:
        extra_args.append("--key-mode-heuristic")
    if args.key_mode_third_margin is not None:
        extra_args += ["--key-mode-third-margin", str(float(args.key_mode_third_margin))]
    if args.key_mode_flip_min_score_ratio is not None:
        extra_args += ["--key-mode-flip-min-score-ratio", str(float(args.key_mode_flip_min_score_ratio))]
    if args.key_minor_harmonic_bonus:
        extra_args.append("--key-minor-harmonic-bonus")
    if args.key_minor_leading_tone_bonus_weight is not None:
        extra_args += ["--key-minor-leading-tone-bonus-weight", str(float(args.key_minor_leading_tone_bonus_weight))]

    # Pass-through key segment voting flags (if provided)
    if args.no_key_segment_voting:
        extra_args.append("--no-key-segment-voting")
    if args.key_segment_len_frames is not None:
        extra_args += ["--key-segment-len-frames", str(int(args.key_segment_len_frames))]
    if args.key_segment_hop_frames is not None:
        extra_args += ["--key-segment-hop-frames", str(int(args.key_segment_hop_frames))]
    if args.key_segment_min_clarity is not None:
        extra_args += ["--key-segment-min-clarity", str(float(args.key_segment_min_clarity))]

    # Pass-through key HPCP flags (if provided)
    if args.key_hpcp:
        extra_args.append("--key-hpcp")
    if args.key_hpcp_peaks is not None:
        extra_args += ["--key-hpcp-peaks", str(int(args.key_hpcp_peaks))]
    if args.key_hpcp_harmonics is not None:
        extra_args += ["--key-hpcp-harmonics", str(int(args.key_hpcp_harmonics))]
    if args.key_hpcp_harmonic_decay is not None:
        extra_args += ["--key-hpcp-harmonic-decay", str(float(args.key_hpcp_harmonic_decay))]
    if args.key_hpcp_mag_power is not None:
        extra_args += ["--key-hpcp-mag-power", str(float(args.key_hpcp_mag_power))]
    if args.key_hpcp_whitening:
        extra_args.append("--key-hpcp-whitening")
    if args.key_hpcp_whitening_smooth_bins is not None:
        extra_args += ["--key-hpcp-whitening-smooth-bins", str(int(args.key_hpcp_whitening_smooth_bins))]

    # Pass-through key STFT override flags (if provided)
    if args.no_key_stft_override:
        extra_args.append("--no-key-stft-override")
    if args.key_stft_override:
        extra_args.append("--key-stft-override")
    if args.key_stft_frame_size is not None:
        extra_args += ["--key-stft-frame-size", str(int(args.key_stft_frame_size))]
    if args.key_stft_hop_size is not None:
        extra_args += ["--key-stft-hop-size", str(int(args.key_stft_hop_size))]

    # Pass-through key log-frequency flags (if provided)
    if args.key_log_freq:
        extra_args.append("--key-log-freq")
    if args.no_key_log_freq:
        extra_args.append("--no-key-log-freq")

    # Pass-through key beat-synchronous flags (if provided)
    if args.key_beat_sync:
        extra_args.append("--key-beat-sync")
    if args.no_key_beat_sync:
        extra_args.append("--no-key-beat-sync")

    # Pass-through key multi-scale flags (if provided)
    if args.key_multi_scale:
        extra_args.append("--key-multi-scale")
    if args.no_key_multi_scale:
        extra_args.append("--no-key-multi-scale")
    if args.key_multi_scale_lengths is not None:
        extra_args += ["--key-multi-scale-lengths", args.key_multi_scale_lengths]
    if args.key_multi_scale_hop is not None:
        extra_args += ["--key-multi-scale-hop", str(int(args.key_multi_scale_hop))]
    if args.key_multi_scale_min_clarity is not None:
        extra_args += ["--key-multi-scale-min-clarity", str(float(args.key_multi_scale_min_clarity))]
    if args.key_multi_scale_weights is not None:
        extra_args += ["--key-multi-scale-weights", args.key_multi_scale_weights]

    # Pass-through key template set flags (if provided)
    if args.key_template_temperley:
        extra_args.append("--key-template-temperley")
    if args.key_template_kk:
        extra_args.append("--key-template-kk")

    # Pass-through key ensemble flags (if provided)
    if args.key_ensemble:
        extra_args.append("--key-ensemble")
    if args.no_key_ensemble:
        extra_args.append("--no-key-ensemble")
    if args.key_ensemble_kk_weight is not None:
        extra_args += ["--key-ensemble-kk-weight", str(float(args.key_ensemble_kk_weight))]
    if args.key_ensemble_temperley_weight is not None:
        extra_args += ["--key-ensemble-temperley-weight", str(float(args.key_ensemble_temperley_weight))]

    # Pass-through multi-resolution tuning flags (if provided)
    if args.multi_res_top_k is not None:
        extra_args += ["--multi-res-top-k", str(args.multi_res_top_k)]
    if args.multi_res_w512 is not None:
        extra_args += ["--multi-res-w512", str(args.multi_res_w512)]
    if args.multi_res_w256 is not None:
        extra_args += ["--multi-res-w256", str(args.multi_res_w256)]
    if args.multi_res_w1024 is not None:
        extra_args += ["--multi-res-w1024", str(args.multi_res_w1024)]
    if args.multi_res_structural_discount is not None:
        extra_args += ["--multi-res-structural-discount", str(args.multi_res_structural_discount)]
    if args.multi_res_double_time_512_factor is not None:
        extra_args += ["--multi-res-double-time-512-factor", str(args.multi_res_double_time_512_factor)]
    if args.multi_res_margin_threshold is not None:
        extra_args += ["--multi-res-margin-threshold", str(args.multi_res_margin_threshold)]
    if args.multi_res_human_prior:
        extra_args.append("--multi-res-human-prior")

    # Pass-through band-fusion tuning flags (if provided)
    if args.band_low_max_hz is not None:
        extra_args += ["--band-low-max-hz", str(args.band_low_max_hz)]
    if args.band_mid_max_hz is not None:
        extra_args += ["--band-mid-max-hz", str(args.band_mid_max_hz)]
    if args.band_high_max_hz is not None:
        extra_args += ["--band-high-max-hz", str(args.band_high_max_hz)]
    if args.band_w_full is not None:
        extra_args += ["--band-w-full", str(args.band_w_full)]
    if args.band_w_low is not None:
        extra_args += ["--band-w-low", str(args.band_w_low)]
    if args.band_w_mid is not None:
        extra_args += ["--band-w-mid", str(args.band_w_mid)]
    if args.band_w_high is not None:
        extra_args += ["--band-w-high", str(args.band_w_high)]
    if args.band_support_threshold is not None:
        extra_args += ["--band-support-threshold", str(args.band_support_threshold)]
    if args.band_consensus_bonus is not None:
        extra_args += ["--band-consensus-bonus", str(args.band_consensus_bonus)]
    if args.superflux_max_filter_bins is not None:
        extra_args += ["--superflux-max-filter-bins", str(args.superflux_max_filter_bins)]
    if args.mel_n_mels is not None:
        extra_args += ["--mel-n-mels", str(args.mel_n_mels)]
    if args.mel_fmin_hz is not None:
        extra_args += ["--mel-fmin-hz", str(args.mel_fmin_hz)]
    if args.mel_fmax_hz is not None:
        extra_args += ["--mel-fmax-hz", str(args.mel_fmax_hz)]
    if args.mel_max_filter_bins is not None:
        extra_args += ["--mel-max-filter-bins", str(args.mel_max_filter_bins)]
    if args.mel_weight is not None:
        extra_args += ["--mel-weight", str(args.mel_weight)]
    if args.novelty_w_spectral is not None:
        extra_args += ["--novelty-w-spectral", str(args.novelty_w_spectral)]
    if args.novelty_w_energy is not None:
        extra_args += ["--novelty-w-energy", str(args.novelty_w_energy)]
    if args.novelty_w_hfc is not None:
        extra_args += ["--novelty-w-hfc", str(args.novelty_w_hfc)]
    if args.novelty_local_mean_window is not None:
        extra_args += ["--novelty-local-mean-window", str(args.novelty_local_mean_window)]
    if args.novelty_smooth_window is not None:
        extra_args += ["--novelty-smooth-window", str(args.novelty_smooth_window)]

    debug_ids = set()
    if args.debug_track_ids.strip():
        for part in args.debug_track_ids.split(","):
            part = part.strip()
            if part.isdigit():
                debug_ids.add(int(part))

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

    def _process_one(i, track):
        """Returns: (i, track_id, row_or_none, log_line_or_none)."""
        track_id = int(track["track_id"])
        audio_file = Path(track["filename"])
        bpm_gt = float(track["bpm_gt"])
        key_gt = track["key_gt"]

        if not audio_file.exists():
            return i, track_id, None, "ERROR: Audio file not found"

        # Run stratum-dsp (optionally with per-track debug flags)
        per_track_args = list(extra_args)
        if track_id in debug_ids:
            per_track_args += ["--debug-track-id", str(track_id), "--debug-gt-bpm", f"{bpm_gt:.6f}"]

        analysis_result = run_stratum_dsp(binary_path, audio_file, per_track_args)
        if "error" in analysis_result:
            return i, track_id, None, f"ERROR: {analysis_result['error']}"

        # Extract results
        pred_bpm = analysis_result.get("bpm")
        pred_key = analysis_result.get("key")
        if pred_bpm is None or pred_key is None:
            return i, track_id, None, "ERROR: Missing BPM or key in results"

        # Read TAG-based BPM/key (written externally into ID3 tags)
        tag_fields = read_tag_bpm_key(audio_file)
        bpm_tag = tag_fields.get("bpm_tag")
        key_tag = tag_fields.get("key_tag", "")

        # Compare to ground truth (from metadata CSVs via test batch)
        bpm_error = abs(pred_bpm - bpm_gt)
        bpm_tag_error = abs(float(bpm_tag) - bpm_gt) if bpm_tag is not None else ""

        key_gt_norm = normalize_key(key_gt)
        key_pred_norm = normalize_key(pred_key)
        key_tag_norm = normalize_key(key_tag)

        # Key comparison reference:
        # - Prefer GT if available.
        # - Otherwise, fall back to TAG as a baseline reference (requested), so we can track
        #   Stratum-vs-TAG agreement even when GT is missing.
        key_ref = "N/A"
        if key_gt_norm:
            key_ref = "GT"
            key_match = "YES" if key_pred_norm == key_gt_norm else "NO"
            key_tag_match = "YES" if key_tag_norm == key_gt_norm else "NO" if key_tag_norm else "NO"
        elif key_tag_norm:
            key_ref = "TAG"
            key_match = "YES" if key_pred_norm == key_tag_norm else "NO"
            # TAG is the reference here, so this field is not meaningful.
            key_tag_match = "N/A"
        else:
            key_match = "N/A"
            key_tag_match = "N/A"

        row = {
            "track_id": track_id,
            "genre": track["genre"],
            "bpm_gt": bpm_gt,
            "bpm_pred": pred_bpm,
            "bpm_error": bpm_error,
            "bpm_tag": bpm_tag if bpm_tag is not None else "",
            "bpm_tag_error": bpm_tag_error,
            "key_gt": key_gt,
            "key_pred": pred_key,
            "key_ref": key_ref,
            "key_match": key_match,
            "key_tag": key_tag,
            "key_tag_match": key_tag_match,
            "bpm_confidence": analysis_result.get("bpm_confidence", 0.0),
            "key_confidence": analysis_result.get("key_confidence", 0.0),
            "key_clarity": analysis_result.get("key_clarity", 0.0),
            "grid_stability": analysis_result.get("grid_stability", 0.0),
            "tempogram_multi_res_triggered": analysis_result.get("tempogram_multi_res_triggered", ""),
            "tempogram_multi_res_used": analysis_result.get("tempogram_multi_res_used", ""),
            "tempogram_percussive_triggered": analysis_result.get("tempogram_percussive_triggered", ""),
            "tempogram_percussive_used": analysis_result.get("tempogram_percussive_used", ""),
        }

        bpm_tag_str = f"{float(bpm_tag):.1f}" if bpm_tag is not None else "N/A"
        bpm_tag_err_str = f"{float(bpm_tag_error):.1f}" if bpm_tag is not None else "N/A"
        key_ref_disp = key_ref if key_ref != "N/A" else "N/A"
        log_line = (
            f"BPM: {pred_bpm:.1f} (error: {bpm_error:.1f}), TAG BPM: {bpm_tag_str} (error: {bpm_tag_err_str}), "
            f"Key: {pred_key} ({key_match}, ref={key_ref_disp}), TAG Key: {key_tag or 'N/A'} ({key_tag_match})"
        )

        # If this is a debug track, append stderr below the log line for easier diagnosis.
        if track_id in debug_ids and analysis_result.get("_stderr"):
            log_line = log_line + "\n" + str(analysis_result["_stderr"])

        return i, track_id, row, log_line

    total = len(test_batch)
    if args.jobs <= 1:
        for i, track in enumerate(test_batch, 1):
            print(f"[{i}/{total}] Processing track {track['track_id']}...", end=" ", flush=True)
            _i, _tid, row, log_line = _process_one(i, track)
            if row is None:
                print(log_line or "ERROR")
                continue
            results.append(row)
            print(log_line)
    else:
        print(f"Parallel batch: jobs={args.jobs} (CPU-1 default)")
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_process_one, i, track) for i, track in enumerate(test_batch, 1)]
            for fut in concurrent.futures.as_completed(futs):
                i, track_id, row, log_line = fut.result()
                completed += 1
                prefix = f"[{completed}/{total}] Track {track_id}"
                if row is None:
                    print(f"{prefix}: {log_line or 'ERROR'}")
                    continue
                results.append(row)
                print(f"{prefix}: {log_line}")
    
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

        # Key reference can be GT (preferred) or TAG (fallback baseline when GT is missing).
        key_rows = [r for r in results if r.get("key_match") in ("YES", "NO")]
        key_rows_ref_gt = [r for r in key_rows if r.get("key_ref") == "GT"]
        key_rows_ref_tag = [r for r in key_rows if r.get("key_ref") == "TAG"]
        key_accuracy = (
            sum(1 for r in key_rows if r["key_match"] == "YES") / len(key_rows) * 100
            if key_rows
            else 0.0
        )

        # TAG metrics (if present)
        tag_rows = [r for r in results if r.get("bpm_tag_error") != ""]
        avg_bpm_tag_error = (
            sum(float(r["bpm_tag_error"]) for r in tag_rows) / len(tag_rows)
            if tag_rows
            else None
        )
        bpm_tag_accuracy_2 = (
            sum(1 for r in tag_rows if float(r["bpm_tag_error"]) <= 2.0) / len(tag_rows) * 100
            if tag_rows
            else None
        )
        key_tag_rows = [r for r in results if r.get("key_tag_match") in ("YES", "NO")]
        key_tag_accuracy = (
            sum(1 for r in key_tag_rows if r["key_tag_match"] == "YES") / len(key_tag_rows) * 100
            if key_tag_rows
            else None
        )
        
        # BPM accuracy within ±2 BPM
        bpm_accuracy_2 = sum(1 for r in results if r["bpm_error"] <= 2.0) / len(results) * 100
        
        print(f"Tracks tested: {len(results)}")
        print(f"Stratum BPM MAE: ±{avg_bpm_error:.2f}")
        print(f"Stratum BPM accuracy (±2 BPM): {bpm_accuracy_2:.1f}%")
        if key_rows_ref_gt:
            acc_gt = sum(1 for r in key_rows_ref_gt if r["key_match"] == "YES") / len(key_rows_ref_gt) * 100
            print(f"Stratum Key accuracy vs GT: {acc_gt:.1f}% (n={len(key_rows_ref_gt)})")
        if key_rows_ref_tag:
            acc_tag = sum(1 for r in key_rows_ref_tag if r["key_match"] == "YES") / len(key_rows_ref_tag) * 100
            print(f"Stratum Key agreement vs TAG: {acc_tag:.1f}% (n={len(key_rows_ref_tag)})")
        if not key_rows_ref_gt and not key_rows_ref_tag:
            print("Stratum Key: N/A (no GT key and no TAG key available in batch)")

        if avg_bpm_tag_error is not None:
            print(f"TAG BPM MAE: ±{avg_bpm_tag_error:.2f} (n={len(tag_rows)})")
            print(f"TAG BPM accuracy (±2 BPM): {bpm_tag_accuracy_2:.1f}%")
        else:
            print("TAG BPM: N/A (no TBPM found in tags for this batch)")

        if key_tag_accuracy is not None:
            print(f"TAG Key accuracy vs GT: {key_tag_accuracy:.1f}% (n={len(key_tag_rows)})")
        else:
            print("TAG Key accuracy vs GT: N/A")
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

