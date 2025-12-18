# Phase 1F: Tempogram BPM Pivot - Implementation Summary

**Date**: 2025-12-17  
**Status**: ⚠️ **IMPLEMENTED, NOT VALIDATED (TUNING + TOOLING ITERATION COMPLETE FOR SPRINT)**  

---

## Executive Summary

Phase 1F implements the planned tempogram BPM pivot (dual method: FFT + autocorrelation tempogram) and integrates it into the primary analysis pipeline. Empirical validation on an initial FMA Small batch indicates the current implementation is **not yet meeting accuracy targets**; the dominant failure mode is **tempo octave / metrical-level selection**.

**See**:
- `docs/progress-reports/PHASE_1F_VALIDATION.md` (results and failure mode analysis)
- `docs/progress-reports/PHASE_1F_LITERATURE_REVIEW.md` (literature foundation)
- `docs/progress-reports/TEMPOGRAM_PIVOT_EVALUATION.md` (technical specification)

---

## Implemented Modules (Phase 1F)

### 1) Novelty Curve Extraction

**File**: `src/features/period/novelty.rs` (NEW)

Implemented novelty curves:
- Spectral flux novelty
- Energy flux novelty
- High-frequency content (HFC) novelty
- Weighted combined novelty curve

### 2) Autocorrelation Tempogram

**File**: `src/features/period/tempogram_autocorr.rs` (NEW)

Implements hypothesis testing over a BPM range by scoring autocorrelation at tempo-specific lags.

### 3) FFT Tempogram

**File**: `src/features/period/tempogram_fft.rs` (NEW)

Implements FFT-based periodicity analysis of the novelty curve and maps dominant frequencies to BPM.

### 4) Comparison / Selection Entry Point

**File**: `src/features/period/tempogram.rs` (NEW)

Runs both tempogram methods and selects/combines results. Includes logic for:
- agreement (ensemble)
- disagreement handling (including harmonic relationship handling)

### 5) Multi-Resolution Wrapper

**File**: `src/features/period/multi_resolution.rs` (NEW)

Runs tempogram across multiple hop sizes and attempts agreement-based selection.

---

## Integration Changes

### Period module exports

**File**: `src/features/period/mod.rs`

- Adds and exports Phase 1F tempogram modules
- Retains legacy Phase 1B APIs for fallback/comparison during transition

### Pipeline switch

**File**: `src/lib.rs`

- Uses tempogram as the primary BPM estimator
- Retains legacy period estimation as a fallback if tempogram fails
- Computes STFT magnitudes for tempogram using shared STFT implementation

### Post-implementation integration & experimentation (this sprint)

- **Onset consensus wired into runtime pipeline (legacy + beat tracking readiness)**:
  - `analyze_audio()` now computes energy-flux onsets and (optionally) a multi-detector consensus onset list (spectral flux + HFC + optional HPSS).
  - These improved onsets feed **beat tracking** and **legacy BPM**.
  - Tempogram BPM remains STFT/novelty-driven and does not consume the onset list.

- **Preprocessing toggles for controlled ablation tests**:
  - `AnalysisConfig` supports disabling normalization and/or silence trimming.
  - Validation runs showed no measurable effect on the current FMA batch when disabling preprocessing.

- **Legacy-only and fusion modes (for A/B testing, not for production defaults)**:
  - Added flags to run legacy-only BPM (`--force-legacy-bpm`) and a BPM fusion mode (`--bpm-fusion`).
  - Fusion has been stabilized into a **validator mode** that does not override tempogram BPM (it only adjusts confidence + logs diagnostics).

- **Legacy guardrails (soft biasing)**:
  - Added configurable legacy BPM confidence multipliers by tempo range to discourage extreme tempos from winning.
  - These are tunable via config and via validation CLI passthrough (see `validation/README.md`).

### Shared STFT access

**File**: `src/features/chroma/extractor.rs`

- `compute_stft(...)` exposed for reuse by the tempogram pipeline

---

## Testing Status

- New Phase 1F unit tests exist for novelty and tempogram components.
- The overall library test suite currently reports **2 failing tests** in legacy `features::period::candidate_filter` (investigation pending).

---

## Empirical Validation Status (FMA Small)

Validation runs completed on a fixed 30-track batch. Results are improved vs initial post-implementation baseline, but remain below target.

- Initial post-implementation baseline: 16.7% (±2 BPM), MAE 57.55
- Tuned baseline (current): 56.7% (±2 BPM), MAE 13.42

Dominant remaining failure modes include metrical-level / harmonic-family confusions (e.g., 4:3, 3:2) and a small number of persistent hard cases.

Full details: `docs/progress-reports/PHASE_1F_VALIDATION.md` (run history + current baseline).

---

## Next Steps

1. **Metrical-level selection** (tempo folding, beat-level inference):
   - Explicitly score \{T, 2T, T/2, 3T, T/3\} and choose the most musically consistent level.
2. **Novelty conditioning**:
   - Improve robustness via standard novelty preprocessing (e.g., log compression, local mean subtraction).
3. **Confidence calibration**:
   - Peak prominence and agreement-based confidence to avoid “confidently wrong” octave choices.
4. Re-run FMA validation across multiple batches and report aggregate metrics.

5. Evaluate multi-resolution tempogram correctly (recompute STFT per hop size) if used as a decision signal.

---

## Conclusion

Phase 1F is fully implemented and integrated, but remains **not validated** due to poor initial empirical results. The code is in place; the remaining work is focused on metrical-level selection and novelty/selection calibration to achieve the documented accuracy targets.

**Status**: ⚠️ **IMPLEMENTED, NOT VALIDATED**


