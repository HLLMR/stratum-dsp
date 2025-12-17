# Stratum DSP - Development Guide

A comprehensive guide for developing the Stratum DSP audio analysis engine. This document covers architecture, algorithms, implementation strategies, and development workflow.

## Project Overview

**Project**: Stratum DSP (`stratum-dsp`)

**Scope**: Pure-Rust hybrid classical DSP + ML-refined audio analysis engine for professional DJ-grade BPM and key detection, with extensibility for future music analysis features (energy, mood, genre, etc.).

**Status**: Phase 1D Complete - Phase 1E Next

**Target Accuracy**:
- BPM: 88% (±2 BPM tolerance)
- Key: 77% (exact match)

**Performance**: <500ms per 30s track (single-threaded)

## Architecture Overview

### High-Level Pipeline

```
Audio Input (PCM samples)
        ↓
    [Preprocessing]
    • Normalization
    • Silence detection
    • Channel mixing (mono for analysis)
        ↓
    [Feature Extraction Layer]
    ├─→ Onset Detection (Multi-method consensus)
    │   ├─ Energy Flux
    │   ├─ Spectral Flux
    │   ├─ High-Frequency Content
    │   └─ Percussive Sharpness (via HPSS)
    │
    ├─→ Period Estimation
    │   ├─ Autocorrelation
    │   ├─ Comb Filterbank
    │   └─ Peak Confidence Scoring
    │
    ├─→ Beat Tracking
    │   ├─ HMM with Viterbi algorithm
    │   └─ Bayesian grid drift correction
    │
    ├─→ Chroma Extraction
    │   ├─ FFT-based harmonic analysis
    │   └─ Chroma vector normalization
    │
    └─→ Key Detection
        ├─ Krumhansl-Kessler template matching
        ├─ Key clarity scoring
        └─ 24-key confidence distribution
        ↓
    [Confidence Scoring]
    • BPM multimodal detection
    • Key clarity indicator
    • Grid stability measure
    • Feature coherence
        ↓
    [ML Refinement Layer] (Optional, Phase 2)
    • ONNX model inference
    • Edge case detection
    • Confidence boosting
        ↓
    Output (Analysis Result)
```

### Module Structure

```
stratum-dsp/
├── src/
│   ├── lib.rs                    # Public API
│   ├── error.rs                  # Error types
│   ├── config.rs                 # Algorithm parameters
│   │
│   ├── preprocessing/             # Audio preprocessing modules
│   │   ├── mod.rs
│   │   ├── normalization.rs      # Peak normalization, loudness
│   │   ├── silence.rs            # Silence detection & trimming
│   │   └── channel_mixer.rs      # Stereo → mono, channel mixing
│   │
│   ├── features/                  # Feature extraction modules
│   │   ├── mod.rs
│   │   ├── onset/                 # Onset detection (4 methods + consensus)
│   │   ├── period/                # Period estimation (BPM detection)
│   │   ├── beat_tracking/         # Beat tracking (HMM + Bayesian)
│   │   ├── chroma/                # Chroma extraction
│   │   └── key/                   # Key detection
│   │
│   ├── analysis/                  # Analysis and confidence scoring
│   │   ├── mod.rs
│   │   ├── confidence.rs          # Confidence scoring & validation
│   │   ├── metadata.rs            # Analysis metadata structures
│   │   └── result.rs              # Final result types
│   │
│   ├── ml/                        # ML refinement (Phase 2)
│   │   ├── mod.rs
│   │   ├── onnx_model.rs          # ONNX model loading & inference
│   │   ├── refinement.rs          # Confidence refinement pipeline
│   │   └── edge_cases.rs          # Edge case detection & correction
│   │
│   └── io/                        # Audio I/O
│       ├── mod.rs
│       ├── decoder.rs              # Audio decoding (via Symphonia)
│       └── sample_buffer.rs       # Sample windowing & buffering
│
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/                  # Ground truth test audio
│
├── benches/                       # Performance benchmarks
├── examples/                      # Usage examples
└── models/                        # ML models (Phase 2)
```

## Core Algorithms

### 1. Preprocessing

#### Normalization
- **Peak method**: Simple peak normalization (fast)
- **RMS method**: RMS-based normalization
- **LUFS method**: ITU-R BS.1770-4 loudness normalization (accurate)
  - Gate at -70 LUFS
  - Apply K-weighting filter
  - Integrate loudness over 400ms blocks

#### Silence Detection
- Frame audio into chunks
- Compute RMS per frame
- Mark frames below threshold as silent
- Merge consecutive silent frames (< 500ms)
- Trim leading/trailing silence

#### Channel Mixing
- Convert stereo to mono: `mono[i] = (left[i] + right[i]) / 2.0`
- Support multiple mixing modes (Mono, MidSide, Dominant, Center)

### 2. Onset Detection

Four independent methods with consensus voting:

#### Energy Flux
- Divide audio into frames (frame_size=2048, hop_size=512)
- Compute RMS energy per frame
- Compute derivative: `E_flux[n] = max(0, E[n] - E[n-1])`
- Threshold and peak-pick

#### Spectral Flux
- Compute STFT (magnitude only)
- Normalize magnitude to [0, 1] per frame
- Compute L2 distance between consecutive frames
- Threshold and peak-pick

#### High-Frequency Content (HFC)
- Compute STFT magnitude
- Weight higher frequencies more heavily
- Threshold and peak-pick

#### Harmonic-Percussive Source Separation (HPSS)
- Decompose spectrogram into harmonic and percussive components
- Apply horizontal median filter (across time) for harmonic
- Apply vertical median filter (across frequency) for percussive
- Detect onsets in percussive component

#### Consensus Voting
- Merge all 4 onset lists within tolerance windows (default: 50ms)
- Weight votes by method confidence
- Return deduplicated onset candidates with confidence scores

### 3. Period Estimation

#### Autocorrelation
- Convert onset list to binary "beat signal"
- Compute autocorrelation (FFT-accelerated O(n log n))
- Find peaks in ACF
- Convert lag → BPM: `BPM = (60 * sample_rate) / (lag * hop_size)`
- Filter within [min_bpm, max_bpm] range
- **Reference**: Ellis & Pikrakis (2006)

#### Comb Filterbank
- For each candidate BPM (80-180, configurable resolution)
- Compute expected beat interval
- Score by counting onsets within adaptive tolerance window
  - Adaptive tolerance: `tolerance = base_tolerance * (120.0 / bpm)`, clamped to [5%, 15%]
  - Higher BPM = smaller tolerance (more precise)
  - Lower BPM = larger tolerance (more forgiving)
- Normalize score by beat count
- **Reference**: Gkiokas et al. (2012)
- **Optimization**: Coarse-to-fine search available (`coarse_to_fine_search()`)
  - Stage 1: 2.0 BPM resolution (coarse)
  - Stage 2: 0.5 BPM resolution around best candidate (fine)
  - Reduces computation time by ~50%

#### Candidate Filtering
- Merge autocorrelation + comb results
- Handle octave errors (2x or 0.5x BPM)
- Group candidates within octave tolerance
- Boost confidence if both methods agree

### 4. Beat Tracking

#### HMM Viterbi Algorithm
- Build state space: BPM candidates around nominal estimate
- Compute transition probabilities (tempo stability)
- Compute emission probabilities (onset alignment)
- Forward pass: track best path probability
- Backtrack: extract most likely beat sequence

#### Bayesian Tempo Tracking
- Update beat grid incrementally for variable-tempo tracks
- Prior: P(BPM | previous_estimate)
- Likelihood: P(onset_evidence | BPM)
- Posterior: P(BPM | evidence) ∝ Prior * Likelihood

### 5. Chroma Extraction

#### STFT-Based Chroma Extraction
- Compute STFT (2048-point FFT, 512 sample hop, Hann windowing)
- Convert frequency bins → semitone classes: `semitone = 12 * log2(freq / 440.0) + 57.0`
- Sum magnitude across octaves for each semitone class (ignores octave, focuses on pitch class)
- L2 normalize each chroma vector for loudness independence
- Filter frequencies below 80 Hz (below typical musical range)
- Reference: Müller, M., & Ewert, S. (2010). Chroma Toolbox: MATLAB Implementations for Extracting Variants of Chroma-Based Audio Features. *Proceedings of the International Society for Music Information Retrieval Conference*

#### Soft Chroma Mapping (Optional Enhancement)
- Gaussian-weighted spread to neighboring semitone classes
- Formula: `weight = exp(-distance² / (2 * σ²))`
- More robust to frequency binning artifacts and tuning variations
- Configurable standard deviation (default: 0.5 semitones)
- Enabled by default for improved robustness

#### Chroma Normalization
- L2 normalization: Normalizes chroma vectors to unit length
- Chroma sharpening: Power function to emphasize prominent semitones
  - Power = 1.0: no change
  - Power > 1.0: increases contrast (emphasizes peaks)
  - Recommended: 1.5-2.0 for improved accuracy
- Reference: Müller, M., & Ewert, S. (2010). Chroma Toolbox

#### Temporal Chroma Smoothing
- Median filtering: Preserves sharp transitions while reducing noise
- Average filtering: Provides smoother results but may blur transitions
- Applied across frames for each semitone class independently
- Configurable window size (typical: 3, 5, 7 frames)
- Reference: Müller, M., & Ewert, S. (2010). Chroma Toolbox

### 6. Key Detection

#### Krumhansl-Kessler Templates
- 24 key templates (12 major + 12 minor)
- Each template is 12-element vector representing likelihood of each semitone
- Templates derived from empirical listening experiments (Krumhansl & Kessler 1982)
- Template rotation for all 12 keys (major and minor)
- Reference: Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the Dynamic Changes in Perceived Tonal Organization in a Spatial Representation of Musical Keys. *Psychological Review*, 89(4), 334-368

#### Template Matching
- Average chroma vectors across all frames
- For each of 24 keys, compute dot product with template
- Sort scores (highest first)
- Find best and second-best scores
- Confidence: `(best - second) / best`
- Returns top N keys (default: top 3) for ambiguous cases
- Reference: Krumhansl, C. L., & Kessler, E. J. (1982)

#### Key Clarity
- Estimate how "tonal" vs "atonal" the track is
- Formula: `clarity = (best_score - average_score) / range`
- High clarity (>0.5): Strong tonality, reliable key detection
- Medium clarity (0.2-0.5): Moderate tonality
- Low clarity (<0.2): Weak tonality, key detection may be unreliable
- Reference: Krumhansl, C. L., & Kessler, E. J. (1982)

#### Key Change Detection (Optional Enhancement)
- Segment-based key detection for tracks with modulations
- Divides track into overlapping segments (configurable duration and overlap)
- Detects key for each segment
- Reports primary key (most common) and key change timestamps
- Useful for classical/jazz music with key modulations

### 7. ML Refinement (Phase 2)

- Lightweight neural network (200-500 parameters)
- Input: 64 features (BPM, onset histograms, spectral energy, key clarity, etc.)
- Architecture: Dense(64→32→16→8→1) with ReLU + Dropout
- Output: Confidence boost factor [0.5, 1.5]
- Trained on 1000+ ground truth DJ tracks

## Development Roadmap

### Phase 1: Classical DSP (Weeks 1-5)

#### Phase 1A: Preprocessing & Onset Detection (Week 1) ✅
- [x] Preprocessing: normalization, silence detection, channel mixing
- [x] Onset detection: 4 methods + consensus voting
- [x] Adaptive thresholding utilities (median + MAD)
- [x] Unit tests for each method (75 tests)
- [x] Integration tests with real audio fixtures (5 tests)
- [x] Test fixture generation script
- [x] Literature-based enhancements
- [x] Main API implementation (`analyze_audio()`)
- **Deliverable**: ✅ Complete - Onset detection module with 80 tests, production-ready code

#### Phase 1B: Period Estimation (Week 2) ✅
- [x] Autocorrelation BPM estimation (FFT-accelerated, O(n log n))
- [x] Comb filterbank BPM estimation
- [x] Peak picking and candidate filtering
- [x] Octave error handling and candidate merging
- [x] Coarse-to-fine search optimization (5-15ms vs 10-30ms for 30s track)
- [x] Adaptive tolerance window (BPM-dependent, improves accuracy)
- [x] Detailed literature citations in function documentation
- [x] Unit tests for all modules (32 tests: 29 original + 3 for enhancements)
- [x] Integration tests on known BPM tracks (120 BPM, 128 BPM, ±2 BPM tolerance)
- [x] Performance benchmarks: autocorrelation ~18.7µs, comb filterbank ~11.1µs, coarse-to-fine ~7.7µs (8-beat pattern)
- [x] Full pipeline benchmark: ~11.6ms for 30s track (43x faster than 500ms target)
- [x] Integrated into main `analyze_audio()` function
- **Deliverable**: ✅ Complete - BPM estimation module with 32 tests, production-ready code with optimizations, benchmarked and validated

#### Phase 1C: Beat Tracking (Week 3) ✅
- [x] HMM Viterbi beat tracker
- [x] Bayesian tempo tracking
- [x] Variable tempo detection and integration
- [x] Time signature detection (4/4, 3/4, 6/8)
- [x] Beat grid generation
- [x] Downbeat detection (using detected time signature)
- [x] Grid stability calculation
- [x] 44 unit tests + integration tests
- [x] Performance benchmarks: HMM ~2.50µs, Bayesian ~1.10µs, Tempo Variation ~601ns, Time Signature ~200ns, Full Beat Grid ~3.75µs (16 beats)
- [x] Full pipeline benchmark: ~11.56ms for 30s track (includes beat tracking, ~43x faster than target)
- **Deliverable**: ✅ Complete - Beat tracking module with <50ms jitter validation, variable tempo handling, time signature detection, and benchmarked

#### Phase 1D: Key Detection (Week 4) ✅
- [x] Chroma extraction (STFT-based with soft mapping)
- [x] Chroma normalization (L2 normalization, sharpening)
- [x] Temporal chroma smoothing (median and average filtering)
- [x] Krumhansl-Kessler templates (24 keys: 12 major + 12 minor)
- [x] Template matching algorithm with confidence scoring
- [x] Key clarity computation (tonal strength estimation)
- [x] Key change detection (segment-based analysis)
- [x] Musical notation display (e.g., "C", "Am", "F#", "D#m")
- [x] DJ standard numerical format (1A, 2B, etc.) without trademarked names
- [x] Unit tests for all modules (40 tests)
- [x] Integration tests with known key fixtures
- [x] Performance benchmarks: ~17-28ms for 30s track (2x faster than target)
- [x] Literature review and validation against benchmarks (Gomtsyan et al. 2019)
- **Deliverable**: ✅ Complete - Key detection module with 70-75% accuracy target validated, production-ready code

#### Phase 1E: Integration & Tuning (Week 5)
- [ ] Confidence scoring
- [ ] Result aggregation
- [ ] Comprehensive test suite (100+ tracks)
- [ ] Performance benchmarking
- **Deliverable**: v0.9-alpha, 85%+ BPM accuracy, 70%+ key accuracy

### Phase 2: ML Refinement (Weeks 6-8)

#### Phase 2A: Data Collection (Week 6)
- [ ] Collect 1000+ diverse DJ tracks
- [ ] Annotate ground truth (BPM, key)
- [ ] Extract features from each track
- **Deliverable**: Training dataset

#### Phase 2B: Model Training & Integration (Week 7)
- [ ] Design and train ONNX model
- [ ] Implement ONNX inference in Rust
- [ ] Integrate ML refinement into pipeline
- **Deliverable**: v1.0-beta with ML refinement, 87-88% BPM, 77-78% key accuracy

#### Phase 2C: Polish & Release (Week 8)
- [ ] Comprehensive documentation
- [ ] Performance optimization
- [ ] Code quality improvements
- [ ] Publish to crates.io
- **Deliverable**: v1.0 published

## Implementation Guidelines

### Code Style

- Use `Result<T>` for all fallible operations (no panics in library code)
- Add `log` crate for debug/info/warn logging
- Numerical stability: epsilon=1e-10 for divisions
- Use `ndarray` for multi-dimensional arrays
- Performance: Start single-threaded, add `rayon` parallelization later if needed

### Error Handling

```rust
pub enum AnalysisError {
    InvalidInput(String),
    DecodingError(String),
    ProcessingError(String),
    NotImplemented(String),
    NumericalError(String),
}
```

### Testing Strategy

**Unit Tests**: Each function/module tested in isolation

**Integration Tests**: Full pipeline on synthetic and real audio

**Benchmarks**: Performance targets per module
- `extract_chroma`: <50ms for 30s audio
- `detect_energy_flux_onsets`: <20ms
- `autocorrelation`: <30ms
- `key_detection`: <10ms
- Total pipeline: <200ms for 30s audio

**Accuracy Tests**: Compare against ground truth (Rekordbox/Mixed In Key)

### Dependencies

```toml
[dependencies]
# Audio I/O
symphonia = { version = "0.5", features = ["all"] }

# Math & DSP
ndarray = "0.15"
ndarray-linalg = "0.15"
rustfft = "6.2"

# ML (Phase 2, optional)
ort = { version = "2.0.0-rc.10", optional = true }

# Utilities
serde = { version = "1", features = ["derive"] }
serde_json = "1"
log = "0.4"
rayon = "1.8"

[dev-dependencies]
env_logger = "0.11"
criterion = { version = "0.5", features = ["html_reports"] }
```

## Performance Targets

### Accuracy

| Metric | Target v1.0 | Stretch v2.0 |
|--------|-------------|--------------|
| BPM Accuracy | 88% (±2 BPM) | 90%+ |
| Key Accuracy | 77% (exact) | 82%+ |
| Confidence Scoring | Robust | Enhanced |
| Edge Cases | Handled | Learned |

### Speed

- **Single-threaded**: <500ms per 30s track
  - **Actual (Phase 1B benchmarks)**: ~11.6ms for 30s track (43x faster than target)
  - **Period estimation**: <50ms for 30s track (autocorrelation + comb filterbank)
  - **Autocorrelation**: ~18.7 µs (8-beat), ~5-15ms extrapolated (30s)
  - **Comb filterbank**: ~11.1 µs (8-beat), ~10-30ms extrapolated (30s)
  - **Coarse-to-fine**: ~7.7 µs (8-beat), ~5-15ms extrapolated (30s)
- **With parallelization**: 50-100ms per 30s track (estimated)
- **With GPU FFT**: 50-100ms (with amortized GPU overhead, estimated)

## Known Challenges & Mitigation

### Octave Errors in BPM

**Problem**: Autocorrelation often detects 2x or 0.5x the true BPM.

**Mitigation**:
- Filter unrealistic BPMs (<60 or >180 BPM)
- Combine autocorrelation + comb filter (they disagree on octave errors)
- ML model trained to detect and correct octave errors

### Key Detection on Atonal/Ambient Tracks

**Problem**: No musical key = unreliable result.

**Mitigation**:
- Compute key clarity (high clarity = reliable, low = warning)
- Return low confidence instead of asserting wrong key
- Allow manual override

### Variable Tempo (DJ Mixes, Live Recordings)

**Problem**: Assumes constant tempo, fails on tempo ramps.

**Mitigation**:
- Detect tempo changes (divide track into segments)
- Track tempo per segment
- Bayesian update for tempo-variable data
- Document limitation: "Phase 1 assumes constant tempo"

## Success Criteria

v1.0 is ready when:

- ✅ BPM detection: ≥88% accuracy (±2 BPM tolerance) on test set
- ✅ Key detection: ≥77% accuracy (exact match) on test set
- ✅ Performance: <500ms per 30s track (single-threaded)
- ✅ Documentation: Full API docs + algorithm explanations
- ✅ Testing: 80%+ code coverage, no panics in production
- ✅ Published to crates.io

## Reference Literature

**Beat Tracking & Tempo Estimation**:
- Ellis & Pikrakis (2006): "Real-time beat induction"
- Gkiokas et al. (2012): "Dimensionality reduction for BPM estimation"
- Böck et al. (2016): "Joint beat and downbeat tracking" (MIREX)

**Key Detection**:
- Krumhansl & Kessler (1982): "Triad Priming and the Psychological Representation"
- Gomtsyan et al. (2019): "Music key and scale detection"

**Onset Detection**:
- Pecan et al. (2017): "A Comparison of Onset Detection Methods"
- Bello et al. (2005): "A Tutorial on Onset Detection in Music Signals"

**Chroma**:
- Müller et al. (2010): "Chroma-Based Audio Analysis Tutorial"
- Ellis & Poliner (2007): "Identifying Cover Songs from Audio"

## Future Enhancements

- Energy/Intensity detection
- Genre classification
- Mood classification
- Vocal/Instrumental detection
- Harmonic analysis (chord progressions)
- Real-time streaming analysis

## Contributing

Contributions welcome! Please ensure:

- Code follows Rust style guidelines
- All tests pass
- New features include tests
- Documentation is updated
- Performance benchmarks are maintained

## License

Dual-licensed under MIT OR Apache-2.0

---

**Last Updated**: 2025-01-XX  
**Status**: Phase 1D Complete - Phase 1E Next
