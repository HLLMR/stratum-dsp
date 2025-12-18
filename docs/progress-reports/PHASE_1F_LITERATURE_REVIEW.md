# Phase 1F: Tempogram BPM Pivot - Literature Review

**Date**: 2025-12-17  
**Status**: ✅ **COMPLETE**  

---

## Overview

Phase 1F pivots BPM estimation from a legacy onset-list-based, frame-oriented approach to a tempogram-based approach built on novelty curves and global periodicity analysis. This review summarizes the core literature supporting:

- Novelty curve design for tempo estimation
- Tempogram (Fourier/autocorrelation) approaches to tempo estimation
- Multi-resolution strategies for robustness
- Practical considerations: metrical-level ambiguity (tempo octave / double-time)

This document complements the technical specification in:
- `docs/progress-reports/TEMPOGRAM_PIVOT_EVALUATION.md`

---

## Primary References

### 1) Grosche, Müller, & Serrà (2012) — Fourier Tempogram

**Contribution**:
- Establishes a robust tempo estimation pipeline using a novelty function and a tempogram representation that emphasizes global periodicity.

**Key takeaways for implementation**:
- Tempo is a global property; compute periodicity on a novelty curve over a longer segment rather than making per-frame decisions.
- Autocorrelation at tempo-specific lags provides a principled “test each BPM hypothesis” scoring function.

**Where we use it**:
- Autocorrelation tempogram scoring (hypothesis testing over BPM range)
- Overall architecture: novelty → periodicity representation → peak selection

**Project references**:
- Literature note: `docs/literature/grosche_2012_tempogram.md`

### 2) Klapuri, Eronen, & Astola (2006) — Meter/Tempo and Novelty Design

**Contribution**:
- Motivates robust onset/novelty design and clarifies that correct tempo inference often requires careful feature engineering and metrical interpretation.

**Key takeaways for implementation**:
- Spectral flux-style novelty measures often outperform simple energy changes for complex musical mixtures.
- Meter/tempo estimation is vulnerable to **metrical-level ambiguity** (e.g., beat vs tatum vs bar-level periodicity).

**Project references**:
- Literature note: `docs/literature/klapuri_2006_meter_analysis.md`

### 3) Ellis (2007) — Global vs Local Tempo Decisions

**Contribution**:
- Argues for global optimization in beat/tempo inference to avoid local ambiguities and subharmonic/harmonic confusions.

**Key takeaways for implementation**:
- Global methods reduce noisy local decisions and better resolve ambiguous periodicities.
- The same motivation applies to tempo estimation: treat tempo as a global optimum.

**Project references**:
- Literature note: `docs/literature/ellis_2007_beat_tracking_dp.md`

### 4) Schreiber & Müller (2018) — Multi-Resolution Strategies

**Contribution**:
- Demonstrates that multi-resolution analysis can improve tempo robustness compared with a single time-frequency resolution.

**Key takeaways for implementation**:
- Multi-resolution estimates can be combined via agreement/consensus to reduce sensitivity to hop-size artifacts.
- Even if the model is ML-based, the multi-resolution insight generalizes to classical DSP tempo representations.

**Project references**:
- Literature note: `docs/literature/schreiber_2018_blstm_tempo.md`

---

## Supporting References (Feature Design and Practice)

### Bello et al. (2005) — Onset Detection Foundations

**Contribution**:
- Canonical overview of onset detection methods, including spectral flux and high-frequency content measures.

**Key takeaways**:
- Half-wave rectification (positive differences) and appropriate normalization are standard choices for spectral flux novelty.
- HFC-like measures can emphasize percussive transients.

**Project references**:
- `docs/literature/bello_2005_onset_detection_tutorial.md`

---

## Practical Implication: Metrical-Level (Tempo Octave) Ambiguity

Across the tempo literature, a recurring theme is that tempo estimation is not only “find the dominant periodicity,” but also “choose the musically meaningful metrical level.”

Common confusions:
- **2× (double-time)**: predicting \(2T\) instead of \(T\)
- **½× (half-time)**: predicting \(T/2\) instead of \(T\)
- **3:2 relationships** (less common but observed in practice)

Implication for Phase 1F:
- A tempogram peak can represent a valid periodicity while still being the wrong metrical interpretation.
- Robust systems often include explicit “tempo folding” or metrical-level selection logic.

This is consistent with the initial empirical findings in:
- `docs/progress-reports/PHASE_1F_VALIDATION.md`

---

## How This Literature Maps to the Codebase

Phase 1F code aligns with the literature as follows:

- **Novelty functions** (Bello 2005; Klapuri 2006; Grosche 2012):
  - `src/features/period/novelty.rs`
- **Autocorrelation tempogram scoring** (Grosche 2012):
  - `src/features/period/tempogram_autocorr.rs`
- **FFT-based periodicity view** (common in practice; complements Grosche’s scoring):
  - `src/features/period/tempogram_fft.rs`
- **Method comparison / selection** (method agreement as a confidence cue):
  - `src/features/period/tempogram.rs`
- **Multi-resolution agreement heuristic** (Schreiber & Müller 2018 inspiration):
  - `src/features/period/multi_resolution.rs`

---

## Open Literature-Driven Follow-Ups

Based on the same literature, the highest-impact next improvements are:

1. **Metrical-level selection**:
   - Explicitly evaluate tempo-related candidates \(\{T, 2T, T/2, 3T, T/3\}\)
   - Select the most musically consistent level using novelty alignment or beat-grid stability.

2. **Novelty conditioning**:
   - Log-compression of magnitudes before spectral flux
   - Local mean subtraction / adaptive thresholding of novelty curve
   - Band emphasis to reduce non-percussive dominance

3. **Confidence calibration**:
   - Peak prominence and peak-to-median scoring for stronger reliability measures
   - Agreement bonuses across resolutions and/or methods

---

## References

- Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., & Sandler, M. B. (2005). A Tutorial on Onset Detection in Music Signals. *IEEE Transactions on Speech and Audio Processing*, 13(5), 1035–1047.
- Ellis, D. P. W. (2007). Beat Tracking by Dynamic Programming. *Journal of New Music Research*.
- Grosche, P., Müller, M., & Serrà, J. (2012). Robust Local Features for Remote Folk Music Identification. *IEEE Transactions on Audio, Speech, and Language Processing*.
- Klapuri, A., Eronen, A., & Astola, J. (2006). Analysis of the Meter of Audio Signals. *IEEE Transactions on Audio, Speech, and Language Processing*, 14(1), 342–355.
- Schreiber, H., & Müller, M. (2018). A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network. *ISMIR*.

---

**Last Updated**: 2025-12-17  
**Status**: Complete


