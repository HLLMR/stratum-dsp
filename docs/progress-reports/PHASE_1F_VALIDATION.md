# Phase 1F: Tempogram BPM Pivot - Validation Report

**Date**: 2025-12-18  
**Status**: ⚠️ **TUNING IN PROGRESS (STABLE BASELINE, NOT YET AT TARGET)**  
**Dataset**: FMA Small (Echonest tempo)  

---

## Overview

Phase 1F replaces the legacy period estimation (Phase 1B autocorrelation + comb filterbank) with a dual tempogram approach (FFT + autocorrelation tempogram) based on Grosche et al. (2012). This report records **post-implementation validation runs** on an FMA Small test batch and summarizes observed failure modes and tuning progress.

**Important**: Initial validation was far below the Phase 1F target. After tuning (novelty conditioning + tempo-octave folding), accuracy improved substantially on the same batch, but Phase 1F is still **not yet at the 88% (±2 BPM) target**.

### Update (2025-12-18): 200-track tuning (novelty weighting)

We added **configurable novelty mixing** for the tempogram novelty curve (spectral / energy / HFC) and validated on a 200-track batch.

- **Baseline (200 tracks)**: `validation_results_20251218_082236.csv`
  - Stratum **±2 BPM**: **53.5%**
  - Stratum **MAE**: **25.21 BPM**
- **Best run so far (200 tracks)**: `validation_results_20251218_083250.csv`
  - Flags: `--novelty-w-spectral 0.30 --novelty-w-energy 0.35 --novelty-w-hfc 0.35`
  - Stratum **±2 BPM**: **54.0%**
  - Stratum **MAE**: **23.61 BPM**

Net: **+0.5% absolute** at ±2 BPM and **-1.6 BPM MAE** on this batch.

---

## Validation Setup

### Dataset

The validation scripts use the Free Music Archive (FMA) Small dataset, with BPM ground truth taken from Echonest metadata:

- Audio: `../validation-data/fma_small/`
- Metadata: `../validation-data/fma_metadata/`
  - `tracks.csv`
  - `echonest.csv` (tempo)

**Note**: FMA does **not** provide key ground truth; key fields are **not evaluated** by these scripts.

### Binary

- Built: `cargo build --release --example analyze_file`
- Binary used by validation: `target/release/examples/analyze_file.exe`

### Batch + Results Files (Primary)

- Test batch (30 tracks):
  - `../validation-data/results/test_batch_20251217_152521.csv`
- Current baseline (tempogram primary, same batch):
  - `../validation-data/results/validation_results_20251217_175146.csv`

---

## How to Reproduce

From repo root:

1. Build:
   - `cargo build --release --example analyze_file`
2. Create a new batch:
   - `python -m validation.tools.prepare_test_batch --num-tracks 30`
3. Run validation:
   - `python -m validation.tools.run_validation`
4. Summarize results:
   - `python -m validation.analysis.analyze_results`

Optional (A/B modes used during tuning):
- Legacy-only BPM (Phase 1B): `python -m validation.tools.run_validation --force-legacy-bpm`
- Disable preprocessing: `python -m validation.tools.run_validation --no-preprocess`
- Disable onset consensus: `python -m validation.tools.run_validation --no-onset-consensus`
- Enable BPM fusion validator: `python -m validation.tools.run_validation --bpm-fusion`
- Ratio bucket analysis: `python -m validation.analysis.analyze_ratio_buckets --file <results_csv>`

---

## Results (Batch: 30 tracks)

### Summary Metrics (Current Baseline)

- **BPM MAE**: **13.42 BPM**
- **BPM accuracy (±2 BPM)**: **56.7% (17/30)**
- **BPM accuracy (±5 BPM)**: **63.3% (19/30)**
- **BPM accuracy (±10 BPM)**: **66.7% (20/30)**
- **BPM accuracy (±20 BPM)**: **66.7% (20/30)**

### Distribution Notes (Current Baseline)

From `validation.analysis.analyze_results` on the baseline results:

- **<5 BPM**: 19
- **5–20 BPM**: 1
- **20–50 BPM**: 7
- **50–100 BPM**: 3
- **>100 BPM**: 0

---

## Tuning Progress (Run History)

All runs below use the **same** batch: `test_batch_20251217_152521.csv`.

| Run | Results File | ±2 BPM | MAE |
|-----|--------------|--------|-----|
| Baseline (post-impl) | `validation_results_20251217_152528.csv` | 16.7% | 57.55 |
| Selection scoring (pre-fold) | `validation_results_20251217_154704.csv` | 23.3% | 47.94 |
| Novelty conditioning + octave folding | `validation_results_20251217_155634.csv` | 56.7% | 13.42 |
| Preprocessing ablation (no change) | `validation_results_20251217_162748.csv` | 56.7% | 13.42 |
| Post-onset-consensus wiring (no change) | `validation_results_20251217_163800.csv` | 56.7% | 13.42 |
| Sprint end baseline (tempogram primary) | `validation_results_20251217_175146.csv` | 56.7% | 13.42 |

Additional experimental runs (for diagnosis, not promoted to defaults):
- Legacy-only BPM (Phase 1B) w/ consensus onsets: `validation_results_20251217_164203.csv` (10.0% ±2, MAE 34.50)
- Legacy-only BPM w/ energy-flux onsets: `validation_results_20251217_164154.csv` (16.7% ±2, MAE 45.43)
- Fusion chooser experiment (rejected; degraded): `validation_results_20251217_164805.csv` (6.7% ±2, MAE 56.91)
- Fusion validator mode (keeps tempogram BPM, adjusts confidence): available via `--bpm-fusion` (see pipeline notes)

---

## Failure Mode Analysis (Key Observations)

### 1) Dominant error: metrical-level / harmonic tempo selection

For many tracks, predictions are close to **2× the ground truth** (e.g., ~100 BPM ground truth → ~200 BPM predicted).

On this batch (baseline `validation_results_20251217_163800.csv`, analyzed via ratio buckets):

- **~1×**: 19/30
- **~4/3×**: 3/30
- **~3/2×**: 1/30
- **~3/4×**: 1/30
- **~2/3×**: 1/30
- **~2×**: 1/30
- **other**: 4/30

This indicates the system is often identifying a valid periodicity, but selecting the wrong metrical level (tatum/double-time vs beat-level).

### 2) Systematic bias (early runs)

Predictions skew high relative to ground truth:

- Predicted BPM > 180: 9/30
- Ground truth BPM > 180: 3/30

### 3) “Octave-only” correction materially improves results

As a diagnostic, applying a tempo-octave folding rule (prefer /2 when BPM is above ~180 unless strongly justified) yields a large improvement on this batch, and has now been incorporated into Phase 1F selection logic.

- **If pred > 180, set pred = pred / 2**

improves (on this batch):

- ±2 BPM: 16.7% → **40.0%**
- ±5 BPM: 30.0% → **53.3%**
- MAE: 57.6 → **26.7 BPM**

Remaining errors suggest additional issues beyond pure octave errors (notably 4:3 and 3:2 family confusions, and a small number of persistent hard cases).

### 4) Preprocessing is not the culprit (on this batch)

Running the same batch with preprocessing disabled (`--no-preprocess`) produced **no measurable change** vs the baseline on this batch. This suggests the dominant errors are downstream (selection/scoring), not normalization/silence trimming artifacts.

### 5) Onset consensus improves legacy readiness, not tempogram BPM

We wired multi-detector onset consensus into `analyze_audio()` for beat tracking and legacy BPM robustness. Tempogram BPM uses an STFT-derived novelty curve and does **not** consume the onset list, so tempogram accuracy did not change on this batch. Legacy-only metrics improved slightly in some regimes but remain far below Phase 1F target accuracy.

---

## Next Steps (Planned Remediation)

### 1) Metrical-level selection (priority: high)

Introduce explicit tempo-level resolution instead of “prefer higher BPM” when harmonic relationships exist:

- Evaluate \{b, b/2, b/3, 2b, 3b\} candidates using novelty alignment / beat-grid stability.
- Prefer the metrical level that maximizes beat-grid stability and onset alignment (leveraging Phase 1C).

### 2) Novelty curve conditioning (priority: high)

Improve novelty curve robustness:

- Log compression of magnitudes before flux
- Local mean subtraction / adaptive thresholding on novelty
- Band emphasis (reduce dominance of harmonic energy)

### 3) Confidence and peak-prominence calibration (priority: medium)

Make confidence meaningful and avoid “confidently wrong” metrical levels:

- Peak prominence vs neighborhood
- Peak-to-median ratio
- Cross-resolution agreement bonus

---

## References

- Grosche, P., Müller, M., & Serrà, J. (2012). Robust Local Features for Remote Folk Music Identification. *IEEE Transactions on Audio, Speech, and Language Processing*.
- See also: `docs/progress-reports/TEMPOGRAM_PIVOT_EVALUATION.md`

---

## Conclusion

Phase 1F tempogram BPM detection is **implemented and integrated**, and validation tooling runs end-to-end (including A/B modes for legacy-only and preprocessing ablations). Accuracy has improved significantly vs the initial post-implementation baseline, but remains **below target** on this batch. Remaining work is concentrated in metrical-level / harmonic-family selection and confidence calibration.

**Status**: ⚠️ **NOT VALIDATED**


