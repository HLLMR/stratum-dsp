# Stratum Notes (2025): Real-World DJ Key Detection Fixes

**Full Citation**: Internal engineering note (Stratum DSP project), Dec 2025.  
**Category**: Key Detection (Practical)  
**Relevance**: Documents the literature-aligned upgrades that improved key detection accuracy on real-world DJ tracks from near-zero to 72.1% (matching MIK performance).

---

## Context

During Phase 1F validation on a real-world DJ dataset (MP3s with ID3 tags and a GT snapshot), the
template-matching key detector (Krumhansl–Kessler) collapsed to a single key when using naive
full-track chroma averaging.

This note captures the upgrades that improved key accuracy from near-zero to **72.1%** (matching MIK's 72.1% on the same dataset) without touching the tempo pipeline (BPM accuracy remained at 87.7% ±2 BPM).

## Changes (Summary)

### Major Improvements (Final Accuracy: 72.1%)

1. **Score normalization (separate major/minor normalization)**:
   - Normalize major and minor key scores separately before comparison to address scale differences.
   - This improved **Stratum key accuracy vs GT from 63.2% → 66.2%** (+3.0 pp) on the hllmr DJ validation batch (n=68).

2. **Circle-of-fifths distance weighting**:
   - Apply a bonus to keys that are close on the circle of fifths (e.g., C-G, C-F) to the top-scoring keys.
   - Keys close on the circle of fifths are harmonically related and often confused, so this helps when the true key is a neighbor.
   - This improved **Stratum key accuracy vs GT from 66.2% → 72.1%** (+5.9 pp) on the hllmr DJ validation batch (n=68).

### Foundation Techniques (Baseline: 63.2%)

3. **Key-only higher-resolution STFT (separate key STFT override)**:
   - Compute a separate STFT for **key detection** using a larger FFT size (default **8192**) while keeping the
     tempo/onset pipeline unchanged.
   - Rationale: for low frequencies, semitone spacing is small; a larger FFT size materially improves pitch-class
     precision for chroma/HPCP mapping.
   - On the hllmr DJ validation batch, using key-STFT override (8192) + HPSS improved
     **Stratum key accuracy vs GT from 45.6% → 63.2%** (n=68), with BPM accuracy unchanged.

4. **Harmonic emphasis (median-filter HPSS harmonic mask, key-only)**:
   - Apply a median-filter HPSS-style harmonic soft mask on the STFT magnitude spectrogram used for
     **key** chroma/HPCP extraction.
   - Implementation is practical for batch runs by band-limiting to the tonal range and computing
     medians on a time-downsampled spectrogram, then applying the mask at full resolution.
   - On the hllmr DJ validation batch this improved **Stratum key accuracy vs GT from 42.6% → 45.6%** (n=68)
     without changing tempo accuracy.

5. **Mode heuristic (3rd scale degree discrimination)**:
   - Use the 3rd scale degree (minor 3rd vs major 3rd) to reduce minor→major mistakes.
   - Applied conservatively with a score-ratio gate to avoid harming major tracks.

6. **HPCP-style pitch-class profiles (peak + harmonics)**:
   - Use spectral peak picking and harmonic summation to build a more robust pitch-class profile than
     naive STFT-bin chroma.

### Experimental/Disabled Features

7. **Segment voting (windowed key detection)**:
   - Run key detection on overlapping chroma windows (e.g., 1024-frame windows with 50% overlap).
   - Compute a clarity score per window and accumulate template scores weighted by clarity.
   - **Status**: Implemented but neutral on this dataset (63.2% baseline).

8. **Tuning compensation (experimental, disabled by default)**:
   - Implemented a lightweight detuning estimator (circular mean of semitone residuals), but initial
     results were unstable on real mixes. Kept for future work (likely needs a more peak/partial-aware
     frontend like HPCP/CQT).

9. **Bass-band HPCP blending (experimental, disabled by default)**:
   - Tried a low-frequency (bass) band PCP blended with the full-band profile to reinforce tonic.
   - On this dataset it regressed overall accuracy (likely because bass is not reliably pitched in all tracks).

10. **Multi-scale key detection, ensemble templates, log-frequency spectrogram, beat-synchronous chroma**:
    - Various advanced techniques were implemented and tested, but did not improve accuracy beyond the baseline.

## References (Project Literature Alignment)

- **Chroma feature extraction**:
  - Müller, M., & Ewert, S. (2010). *Chroma Toolbox: MATLAB Implementations for Extracting Variants of Chroma-Based Audio Features.*
    Proceedings of ISMIR.  
  - See: `docs/literature/mueller_2010_chroma_tutorial.md`

- **Key templates**:
  - Krumhansl, C. L., & Kessler, E. J. (1982). *Tracing the Dynamic Changes in Perceived Tonal Organization in a Spatial Representation of Musical Keys.*
    Psychological Review, 89(4), 334–368.

- **Harmonic/percussive separation background**:
  - Driedger, J., & Müller, M. (2014). *Extending Harmonic-Percussive Separation of Audio Signals.*
    Proceedings of ISMIR.  
  - See: `docs/literature/driedger_mueller_2014_hpss.md`

- **Modern key detection evaluation / modulations as an edge case**:
  - Gomtsyan, M., et al. (2019). *Music Key and Scale Detection.*
    Proceedings of ISMIR.  
  - See: `docs/literature/gomtsyan_2019_key_detection.md`

## Implementation Notes (Where in Code)

- Median-filter HPSS harmonic mask (key-only spectrogram conditioning):
  - `src/features/chroma/extractor.rs`: `harmonic_spectrogram_hpss_median_mask(...)`
  - Wired in `src/lib.rs` key detection path; toggled by `src/config.rs` (`enable_key_hpss_harmonic`).
  - Validation runner supports pass-through flags via `validation/tools/run_validation.py`:
    - `--key-hpss`
    - `--no-key-hpss`
    - `--key-hpss-frame-step`
    - `--key-hpss-time-margin`
    - `--key-hpss-freq-margin`
    - `--key-hpss-mask-power`

- Key-only STFT override (key detection only; tempo unchanged):
  - `src/lib.rs` key detection path computes a separate STFT when `src/config.rs` enables it:
    - `enable_key_stft_override`
    - `key_stft_frame_size`
    - `key_stft_hop_size`
  - CLI/validation pass-through flags:
    - `--key-stft-override`
    - `--no-key-stft-override`
    - `--key-stft-frame-size`
    - `--key-stft-hop-size`

- Harmonic mask (key-only spectrogram conditioning):
  - `src/features/chroma/extractor.rs`: `harmonic_spectrogram_time_mask(...)`
  - Wired in `src/lib.rs` key detection path, reusing the already-computed STFT.

- HPCP-style PCP (key-only feature option):
  - `src/features/chroma/extractor.rs`: `extract_hpcp_from_spectrogram_with_options_and_energy_tuned(...)`
  - Enabled by default in `src/config.rs` (`enable_key_hpcp=true`).

- Score normalization and circle-of-fifths weighting:
  - `src/features/key/detector.rs`: `detect_key_weighted(...)` applies score normalization and circle-of-fifths weighting.
  - These are always enabled (no config flags) as they are core improvements.

- Mode heuristic:
  - `src/features/key/detector.rs`: `detect_key_weighted_mode_heuristic(...)` uses 3rd scale degree discrimination.
  - Enabled by default via `src/config.rs` (`enable_key_mode_heuristic=true`).

- Segment voting (experimental):
  - `src/lib.rs` key detection path (accumulates per-window scores weighted by clarity).
  - Config knobs live in `src/config.rs` (`enable_key_segment_voting`, `key_segment_len_frames`, etc.).
  - **Note**: Neutral on this dataset, kept for future evaluation.

- Experimental tuning:
  - `src/features/chroma/extractor.rs`: `estimate_tuning_offset_semitones_from_spectrogram(...)`
  - Disabled by default in `src/config.rs` (`enable_key_tuning_compensation=false`).

- Experimental bass-band HPCP blend:
  - `src/features/chroma/extractor.rs`: `extract_hpcp_bass_blend_from_spectrogram_with_options_and_energy_tuned(...)`
  - Disabled by default in `src/config.rs` (`enable_key_hpcp_bass_blend=false`).


