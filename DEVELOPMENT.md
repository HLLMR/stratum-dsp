# Stratum DSP - Development Guide

A comprehensive guide for developing the Stratum DSP audio analysis engine. This document covers architecture, algorithms, implementation strategies, and development workflow.

## Project Overview

**Project**: Stratum DSP (`stratum-dsp`)

**Scope**: Pure-Rust hybrid classical DSP + ML-refined audio analysis engine for professional DJ-grade BPM and key detection, with extensibility for future music analysis features (energy, mood, genre, etc.).

**Status**: Phase 1B Complete - Phase 1C In Progress

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

#### Comb Filterbank
- For each candidate BPM (80-180, step=0.5 BPM)
- Compute expected beat interval
- Score by counting onsets within ±10% timing tolerance
- Normalize score by onset count

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

- Compute STFT (2048-point FFT, 512 sample hop)
- Convert frequency bins → semitone classes: `semitone = 12 * log2(freq / 440.0) + 57.0`
- Sum magnitude across octaves for each semitone
- Normalize to L2 unit norm
- Apply soft mapping (spread to neighboring semitones)

### 6. Key Detection

#### Krumhansl-Kessler Templates
- 24 key templates (12 major + 12 minor)
- Each template is 12-element vector representing likelihood of each semitone
- Templates derived from empirical listening tests

#### Template Matching
- Average chroma across all frames
- For each of 24 keys, compute dot product with template
- Find best and second-best scores
- Confidence: `(best - second) / best`

#### Key Clarity
- Estimate how "tonal" vs "atonal" the track is
- Formula: `(best_score - average_score) / range`
- High clarity = sharp tonal center
- Low clarity = ambiguous/atonal

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
- [x] Integration tests on known BPM tracks (120 BPM, 128 BPM)
- [x] Integrated into main `analyze_audio()` function
- **Deliverable**: ✅ Complete - BPM estimation module with 32 tests, production-ready code with optimizations

#### Phase 1C: Beat Tracking (Week 3)
- [ ] HMM Viterbi beat tracker
- [ ] Bayesian tempo tracking
- [ ] Beat grid generation
- **Deliverable**: Beat tracking module, <50ms jitter

#### Phase 1D: Key Detection (Week 4)
- [ ] Chroma extraction
- [ ] Krumhansl-Kessler templates
- [ ] Template matching and key clarity
- **Deliverable**: Key detection module, 70-75% accuracy

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
- **With parallelization**: 50-100ms per 30s track
- **With GPU FFT**: 50-100ms (with amortized GPU overhead)

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
**Status**: Phase 1B Complete - Phase 1C In Progress

