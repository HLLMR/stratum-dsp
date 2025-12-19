# Validation

This directory contains scripts for validating stratum-dsp against ground truth datasets.

**Primary validation dataset**: Real-world DJ tracks (Beatport/ZipDJ) — 155 tracks with verified BPM/key ground truth.  
**Results**: 
- **BPM**: 87.7% ±2 BPM accuracy, 6.08 BPM MAE
- **Key**: 72.1% exact match vs GT (n=68), matching MIK performance

See `docs/progress-reports/PHASE_1F_VALIDATION.md` for BPM details and `docs/literature/stratum_2025_key_detection_real_world.md` for key detection improvements.

**Secondary dataset**: Free Music Archive (FMA) Small — used for algorithm development and tuning.

## Setup

### 1. Download FMA Small Dataset

Download the FMA Small dataset (~7 GB) from:
- https://github.com/mdeff/fma

Extract to the following structure:
```
../validation-data/
├── fma_small/          (7 GB - audio files)
└── fma_metadata/       (200 MB - metadata CSV files)
```

The `fma_small` directory should contain subdirectories like `000/`, `001/`, etc., each containing MP3 files named `000000.mp3`, `000001.mp3`, etc.

The `fma_metadata` directory should contain `tracks.csv` with metadata including BPM and key information.

### 2. Build stratum-dsp

Build the release version with the example binary:

```bash
cargo build --release --example analyze_file
```

This will create the `analyze_file` example binary at:
- Windows: `target/release/examples/analyze_file.exe`
- Linux/macOS: `target/release/examples/analyze_file`

### 3. Prepare Test Batch

Run the script to prepare a test batch from the FMA metadata:

```bash
python -m validation.tools.prepare_test_batch --num-tracks 20
```

Options:
- `--num-tracks N`: Number of tracks to include (default: 20)
- `--num-tracks 0` or `--all`: Include **all** eligible tracks with files
- `--data-path PATH`: Path to validation data directory (default: `../validation-data`)
- `--audio-dir NAME`: Audio directory under `--data-path` (default: `fma_small`)
- `--metadata-dir NAME`: Metadata directory under `--data-path` (default: `fma_metadata`)
- `--seed N`: Random seed for reproducible track selection

This will create `../validation-data/results/test_batch.csv` with the selected tracks.

### Optional: Real-world DJ dataset (`hllmr_small`)

If you have a folder of real DJ-use tracks with reliable ID3 tags (vendor/library GT), you can snapshot them into FMA-style metadata so the existing validation tooling can run on them.

Expected structure:

```
../validation-data/
├── hllmr_small/       (your MP3s, flat folder)
└── hllmr_metadata/    (generated GT snapshot)
```

1) **Build the GT snapshot (do this BEFORE writing MIK tags):**

```bash
python -m validation.tools.build_hllmr_metadata --data-path ../validation-data
```

2) **Prepare a test batch from that snapshot:**

```bash
python -m validation.tools.prepare_test_batch --data-path ../validation-data --audio-dir hllmr_small --metadata-dir hllmr_metadata --all
```

3) (Later) re-tag the MP3s with MIK, then run validation. The GT remains in `hllmr_metadata/*.csv`.

### 4. Run Validation

Run validation on the test batch:

```bash
python -m validation.tools.run_validation
```

Options:
- `--data-path PATH`: Path to validation data directory (default: `../validation-data`)
- `--binary PATH`: Path to stratum-dsp binary (default: auto-detected)
- `--jobs N`: Parallel workers for batch processing (default: CPU-1, keeping one core free). Use 1 to disable.
- `--no-preprocess`: Disable preprocessing (normalization + silence trimming) inside `analyze_file`
- `--no-onset-consensus`: Disable onset consensus (use energy-flux-only onset list)
- `--force-legacy-bpm`: Force legacy BPM estimation (Phase 1B) (skip tempogram)
- `--bpm-fusion`: Enable BPM fusion mode
- Legacy BPM guardrail tuning (pass-through to `analyze_file`):
  - `--legacy-preferred-min/--legacy-preferred-max`
  - `--legacy-soft-min/--legacy-soft-max`
  - `--legacy-mul-preferred/--legacy-mul-soft/--legacy-mul-extreme`

This will:
1. Run stratum-dsp on each track in the test batch
2. Compare results to ground truth (tempo and key when available in metadata)
3. Read **TAG** values from the audio file’s ID3 tags (if present) and compare those to ground truth as well
3. Generate a results CSV with metrics
4. Print a summary with accuracy statistics

### 5. Analyze Results

Summarize one or more results files:

```bash
python -m validation.analysis.analyze_results --file ../validation-data/results/validation_results_YYYYMMDD_HHMMSS.csv
```

If no `--file` is provided, the script analyzes the most recent `validation_results_*.csv`.

### 6. Analyze Ratio Buckets (metrical-level errors)

Quickly bucket pred/gt ratios to spot octave/harmonic confusions:

```bash
python -m validation.analysis.analyze_ratio_buckets --file ../validation-data/results/validation_results_YYYYMMDD_HHMMSS.csv
```

### 7. Generate an Exemplar Report (diagnostic deep-dive)

This produces a “surgical” report of the worst failures, per-tempo-band breakdowns, and
head-to-head cases where TAG is within ±2 BPM and Stratum is not (and vice versa):

```bash
python -m validation.analysis.analyze_exemplars --file ../validation-data/results/validation_results_YYYYMMDD_HHMMSS.csv --top 20
```

## Directory Structure

```
stratum-dsp/
├── validation/
│   ├── tools/                    ← In git (scripts you run)
│   ├── analysis/                 ← In git (post-run analysis scripts)
│   └── README.md                 ← In git
│
└── ../validation-data/           ← NOT in git (7+ GB)
    ├── fma_small/                ← YOU download this
    ├── fma_metadata/             ← YOU download this
    ├── hllmr_small/              ← Optional: your real-world MP3s
    ├── hllmr_metadata/           ← Optional: generated GT snapshot from ID3 tags
    └── results/                  ← Generated by scripts
        ├── test_batch.csv
        └── validation_results.csv
```

## Output

### test_batch.csv

Contains the selected tracks with ground truth values:
- `track_id`: FMA track ID
- `filename`: Full path to audio file
- `bpm_gt`: Ground truth BPM
- `key_gt`: Ground truth key
- `genre`: Genre classification

### validation_results.csv

Contains validation results for each track:
- `track_id`: FMA track ID
- `genre`: Genre classification
- `bpm_gt`: Ground truth BPM
- `bpm_pred`: Predicted BPM
- `bpm_error`: Absolute BPM error
- `bpm_tag`: TAG BPM (from ID3 TBPM), if present
- `bpm_tag_error`: Absolute TAG BPM error vs ground truth, if present
- `key_gt`: Ground truth key
- `key_pred`: Predicted key
- `key_match`: "YES" if keys match, "NO" otherwise
- `key_ref`: which reference was used for `key_match`:
  - `GT` if key ground truth exists in metadata
  - `TAG` if GT is missing but the file tag contains a key (fallback agreement mode)
  - `N/A` if neither is available
- `key_tag`: TAG key (from ID3 TKEY / common TXXX fallbacks), if present
- `key_tag_match`: "YES" if TAG key matches ground truth, "NO" otherwise
- `bpm_confidence`: BPM confidence score
- `key_confidence`: Key confidence score
- `key_clarity`: Key clarity score
- `grid_stability`: Beat grid stability score

## Target Accuracy

The validation compares results against these targets:
- **BPM**: 88% accuracy (±2 BPM tolerance)
- **Key**: 77% accuracy (exact match) — **Current: 72.1%** (matches MIK performance)

## Notes

- All paths are relative, so scripts work from any location
- The scripts are cross-platform (Windows, Linux, macOS)
- Python 3.6+ is required (no external dependencies)
- The validation uses the `analyze_file` example binary which outputs JSON for easy parsing
- TAG extraction currently supports **ID3v2.3/2.4** and reads `TBPM`, `TKEY`, plus common `TXXX` key fields (e.g., `initialkey`).

