# Literature Review Index

This directory contains individual literature review documents for each research paper that informs the algorithms implemented in Stratum DSP. Each document provides a summary, key contributions, relevance to the project, implementation notes, and performance characteristics.

## Onset Detection

- **[Bello et al. (2005): A Tutorial on Onset Detection in Music Signals](bello_2005_onset_detection_tutorial.md)**
  - Comprehensive survey of onset detection methods
  - Energy Flux, Spectral Flux, and HFC algorithms
  - Multi-method consensus approach

- **[Pecan et al. (2017): A Comparison of Onset Detection Methods](pecan_2017_onset_comparison.md)**
  - Comparative evaluation of modern onset detection algorithms
  - Validates multi-method consensus approach
  - Performance benchmarks

- **[McFee & Ellis (2014): Better Beat Tracking Through Robust Onset Aggregation](mcfee_ellis_2014_onset_aggregation.md)**
  - Median aggregation techniques for robust onset detection
  - Improves beat tracking through better onset quality
  - Spectrogram decomposition methods

- **[Bello & Sandler (2003): Phase-Based Note Onset Detection](bello_sandler_2003_phase_onset.md)**
  - Alternative onset detection using phase information
  - Excellent for soft attacks (piano, strings)
  - Complementary to energy/spectral methods

## Beat Tracking & Tempo Estimation

- **[Ellis & Pikrakis (2006): Real-time Beat Induction](ellis_pikrakis_2006_beat_induction.md)**
  - Autocorrelation-based tempo estimation
  - Core algorithm for period detection
  - FFT-accelerated autocorrelation

- **[Ellis (2007): Beat Tracking by Dynamic Programming](ellis_2007_beat_tracking_dp.md)**
  - Foundation for onset-aligned beat tracking
  - Global analysis approach (why global > local)
  - Theoretical foundation for tempogram methods

- **[Gkiokas et al. (2012): Dimensionality Reduction for BPM Estimation](gkiokas_2012_bpm_estimation.md)**
  - Comb filterbank approach for robust BPM estimation
  - Tests hypothesis tempos (80-180 BPM)
  - Alternative to autocorrelation method

- **[Grosche et al. (2012): Robust Local Features for Remote Folk Music Identification](grosche_2012_tempogram.md)**
  - Fourier tempogram algorithm (85%+ accuracy)
  - Global analysis of spectral flux novelty curve
  - Industry gold standard for tempo estimation
  - Core algorithm for tempogram-based BPM detection

- **[Klapuri et al. (2006): Analysis of the Meter in Acoustic Music Signals](klapuri_2006_meter_analysis.md)**
  - Spectral novelty curves and spectral flux
  - Why spectral flux > energy flux for BPM estimation
  - Foundation for tempogram novelty curve computation

- **[Schreiber & Müller (2018): A Single-Layer BLSTM Acoustic Model for Music Tempo Estimation](schreiber_2018_blstm_tempo.md)**
  - Multi-resolution analysis for robust tempo estimation
  - Temporal constraint modeling
  - Enhances tempogram with multi-resolution aggregation

- **[Böck et al. (2016): Joint Beat and Downbeat Tracking](boeck_2016_beat_downbeat_tracking.md)**
  - HMM-based beat tracking using Viterbi algorithm
  - State-of-the-art performance on MIREX
  - Handles tempo variations and syncopation

## Key Detection

- **[Krumhansl & Kessler (1982): Key Detection Templates](krumhansl_kessler_1982_key_templates.md)**
  - Empirical key profiles (templates) from listening experiments
  - 24 templates (12 major + 12 minor keys)
  - Foundation for template-matching key detection

- **[Gomtsyan et al. (2019): Music Key and Scale Detection](gomtsyan_2019_key_detection.md)**
  - Modern evaluation of key detection methods
  - Validates Krumhansl-Kessler approach
  - Performance benchmarks and key clarity metrics

## Chroma Extraction

- **[Müller et al. (2010): Chroma-Based Audio Analysis Tutorial](mueller_2010_chroma_tutorial.md)**
  - Comprehensive tutorial on chroma feature extraction
  - Frequency-to-semitone mapping
  - Octave summation and normalization

- **[Ellis & Poliner (2007): Identifying Cover Songs from Audio](ellis_poliner_2007_cover_songs.md)**
  - Temporal smoothing of chroma features
  - Improves key detection accuracy (5-10% improvement)
  - Median/average filtering across time

## Harmonic-Percussive Source Separation

- **[Driedger & Müller (2014): Harmonic-Percussive Source Separation](driedger_mueller_2014_hpss.md)**
  - HPSS algorithm using median filtering
  - Separates harmonic (sustained) from percussive (transient) components
  - Improves onset detection in complex material

## Loudness Normalization

- **[ITU-R BS.1770-4: Loudness Measurement Standard](itu_bs1770_4_loudness.md)**
  - Standard for LUFS (Loudness Units relative to Full Scale) measurement
  - K-weighting filter and gating algorithm
  - Industry standard for broadcast and streaming

## Machine Learning (Phase 2)

- **[Schlüter & Böck (2014): Improved Musical Onset Detection with CNNs](schlueter_boeck_2014_cnn_onset.md)**
  - CNN-based onset detection for ML refinement
  - Outperforms traditional methods
  - Reference for Phase 2 ML enhancement

---

## Quick Reference by Algorithm

### Onset Detection
- **Energy Flux**: Bello et al. (2005)
- **Spectral Flux**: Bello et al. (2005)
- **HFC**: Bello et al. (2005)
- **HPSS**: Driedger & Müller (2014)
- **Phase-Based**: Bello & Sandler (2003) - Optional
- **Consensus Voting**: Bello et al. (2005), Pecan et al. (2017), McFee & Ellis (2014)
- **ML Refinement**: Schlüter & Böck (2014) - Phase 2

### BPM Estimation
- **Autocorrelation**: Ellis & Pikrakis (2006) - Current implementation
- **Comb Filterbank**: Gkiokas et al. (2012) - Current implementation
- **Tempogram (Fourier)**: Grosche et al. (2012) - Pivot implementation
- **Spectral Flux Novelty**: Klapuri et al. (2006) - Foundation for tempogram
- **Multi-Resolution**: Schreiber & Müller (2018) - Enhancement for tempogram
- **Global Analysis**: Ellis (2007) - Theoretical foundation

### Beat Tracking
- **HMM Viterbi**: Böck et al. (2016)

### Key Detection
- **Templates**: Krumhansl & Kessler (1982)
- **Performance**: Gomtsyan et al. (2019)

### Chroma Extraction
- **Extraction**: Müller et al. (2010)
- **Smoothing**: Ellis & Poliner (2007)

### Normalization
- **LUFS**: ITU-R BS.1770-4

---

## Document Format

Each literature review document follows this structure:

1. **Summary**: Brief overview of the paper
2. **Key Contributions**: Main findings and algorithms
3. **Relevance to Stratum DSP**: How it informs our implementation
4. **Key Algorithms**: Formulas and pseudocode
5. **Implementation Notes**: Practical implementation details
6. **Performance Characteristics**: Speed, accuracy, strengths, weaknesses
7. **References in Code**: Where the algorithm is implemented
8. **Additional Notes**: Other relevant information

---

**Last Updated**: 2025-01-XX  
**Purpose**: Reference for Cursor AI and developers implementing audio analysis algorithms

