---
name: Stratum Audio Analysis Engine
overview: "Build a professional-grade Rust audio analysis engine for DJ applications with classical DSP algorithms (onset detection, BPM, key detection, beat tracking) plus ML refinement, targeting 88% BPM and 77% key accuracy. Deliverable: v1.0 crate published to crates.io in 8 weeks."
todos:
  - id: setup-repo
    content: Create stratum-audio-analysis repository structure with Cargo.toml, module directories, and CI/CD setup
    status: pending
  - id: preprocessing
    content: "Implement preprocessing module: normalization (peak/RMS/LUFS), silence detection, channel mixing"
    status: pending
    dependencies:
      - setup-repo
  - id: onset-detection
    content: "Implement all 4 onset detection methods: energy flux, spectral flux, HFC, HPSS, plus consensus voting"
    status: pending
    dependencies:
      - preprocessing
  - id: period-estimation
    content: "Implement period estimation: autocorrelation, comb filterbank, peak picking, candidate filtering with octave error handling"
    status: pending
    dependencies:
      - onset-detection
  - id: beat-tracking
    content: "Implement beat tracking: HMM Viterbi algorithm and Bayesian tempo tracking for grid generation"
    status: pending
    dependencies:
      - period-estimation
  - id: key-detection
    content: "Implement key detection: chroma extraction, normalization, Krumhansl-Kessler templates, template matching, key clarity"
    status: pending
    dependencies:
      - beat-tracking
  - id: integration-tuning
    content: Integrate all modules, implement confidence scoring, create comprehensive test suite, tune accuracy to 85%+ BPM
    status: pending
    dependencies:
      - key-detection
  - id: ml-data-collection
    content: Collect 1000+ ground truth DJ tracks, annotate BPM/key, extract features for ML training dataset
    status: pending
    dependencies:
      - integration-tuning
  - id: ml-model-training
    content: Design and train lightweight ONNX model, implement ONNX inference in Rust, integrate ML refinement into pipeline
    status: pending
    dependencies:
      - ml-data-collection
  - id: polish-release
    content: Write comprehensive documentation, achieve 80%+ test coverage, publish v1.0 to crates.io with MIT/Apache license
    status: pending
    dependencies:
      - ml-model-training
  - id: desktop-integration
    content: Integrate stratum-audio-analysis into stratum-desktop as drop-in replacement, update command handlers, validate accuracy
    status: pending
    dependencies:
      - polish-release
---

# Stratum Audio Analysis Engine - Development Plan

## Project Overview

Build `stratum-audio-analysis`, a pure-Rust audio analysis crate for professional DJ-grade BPM and key detection. The project follows a phased approach: Phase 1 implements classical DSP algorithms (5 weeks), Phase 2 adds ML refinement (3 weeks), resulting in v1.0 ready for integration into `stratum-desktop`.

**Target Accuracy**: 88% BPM (±2 BPM tolerance), 77% key detection  
**Performance**: <500ms per 30s track (single-threaded)  
**License**: MIT/Apache dual (open-source)  
**Timeline**: 8 weeks to v1.0

## Architecture

The engine follows a modular pipeline architecture as specified in [audio-analysis-engine-spec.md](audio-analysis-engine-spec.md):

```
Audio Input → Preprocessing → Feature Extraction → Analysis → ML Refinement → Output
```

**Key Modules**:
- `preprocessing/`: Normalization, silence detection, channel mixing
- `features/onset/`: 4 onset detection methods + consensus voting
- `features/period/`: Autocorrelation + comb filter BPM estimation
- `features/beat_tracking/`: HMM Viterbi + Bayesian tempo tracking
- `features/chroma/`: Chroma extraction and normalization
- `features/key/`: Krumhansl-Kessler template matching
- `analysis/`: Confidence scoring and result aggregation
- `ml/`: ONNX model inference (Phase 2)

## Phase 0: Project Setup (Week 0)

**Goal**: Establish project structure and development environment

### Repository Structure

Create workspace structure:
```
stratum-audio-analysis/
├── Cargo.toml              # Main crate manifest
├── src/
│   ├── lib.rs              # Public API entry point
│   ├── error.rs            # Error types
│   ├── config.rs           # Algorithm parameters
│   ├── preprocessing/       # Audio preprocessing modules
│   ├── features/           # Feature extraction modules
│   ├── analysis/            # Analysis and confidence scoring
│   ├── ml/                  # ML refinement (Phase 2)
│   └── io/                  # Audio I/O (Symphonia integration)
├── tests/
│   ├── integration_tests.rs
│   └── fixtures/            # Test audio files
├── benches/                 # Performance benchmarks
├── examples/                # Usage examples
├── README.md
└── CHANGELOG.md
```

### Dependencies

Configure [Cargo.toml](Cargo.toml) with dependencies from [audio-analysis-engine-spec.md Section 7.1](audio-analysis-engine-spec.md):
- `symphonia` (audio decoding)
- `rustfft` (FFT computation)
- `ndarray` (multi-dimensional arrays)
- `log` (logging)
- `serde` (serialization)
- `rayon` (parallelization, optional)
- `ort` (ONNX Runtime, Phase 2, optional feature)

### CI/CD Setup

- GitHub Actions for testing and benchmarking
- Cargo fmt/clippy checks
- Automated test coverage reporting

## Phase 1: Classical DSP Implementation (Weeks 1-5)

### Phase 1A: Preprocessing & Onset Detection (Week 1)

**Goal**: Implement audio preprocessing and all 4 onset detection methods with consensus voting.

**Modules to implement** (reference [audio-analysis-engine-spec.md Section 2.1-2.2](audio-analysis-engine-spec.md)):

1. **Preprocessing** ([Section 2.1](audio-analysis-engine-spec.md)):
   - `preprocessing/normalization.rs`: Peak, RMS, and LUFS normalization (ITU-R BS.1770-4)
   - `preprocessing/silence.rs`: Silence detection and trimming
   - `preprocessing/channel_mixer.rs`: Stereo to mono conversion

2. **Onset Detection** ([Section 2.2](audio-analysis-engine-spec.md)):
   - `features/onset/energy_flux.rs`: Energy-based onset detection
   - `features/onset/spectral_flux.rs`: Spectral change detection (requires STFT)
   - `features/onset/hfc.rs`: High-frequency content method
   - `features/onset/hpss.rs`: Harmonic-percussive source separation
   - `features/onset/consensus.rs`: Multi-method voting algorithm

**Testing**: Unit tests for each method, synthetic kick pattern validation (120 BPM 4-on-floor)

**Deliverable**: Onset detection module with 90%+ code coverage

### Phase 1B: Period Estimation (Week 2)

**Goal**: Convert onset list to BPM candidates using dual-method approach.

**Modules** ([Section 2.3](audio-analysis-engine-spec.md)):

1. `features/period/autocorrelation.rs`: FFT-accelerated autocorrelation for period detection
2. `features/period/comb_filter.rs`: Hypothesis-based tempo testing (80-180 BPM range)
3. `features/period/peak_picking.rs`: Robust peak detection in ACF/comb scores
4. `features/period/candidate_filter.rs`: Merge results, handle octave errors (2x/0.5x BPM)

**Testing**: Known BPM tracks (synthetic and real), accuracy target: 75%+ on real data

**Deliverable**: BPM estimation module, ready for beat tracking

### Phase 1C: Beat Tracking (Week 3)

**Goal**: Generate precise beat grid from BPM estimate.

**Modules** ([Section 2.4](audio-analysis-engine-spec.md)):

1. `features/beat_tracking/hmm.rs`: HMM Viterbi algorithm for beat sequence tracking
   - State space: BPM candidates around nominal estimate
   - Transition probabilities (tempo stability)
   - Emission probabilities (onset alignment)
   - Backtracking for most likely path

2. `features/beat_tracking/bayesian.rs`: Incremental tempo update for variable-tempo tracks

**Testing**: Beat grid validation against manually annotated tracks, <50ms jitter target

**Deliverable**: Beat tracking module with grid generation

### Phase 1D: Key Detection (Week 4)

**Goal**: Detect musical key using chroma analysis and template matching.

**Modules** ([Section 2.5-2.6](audio-analysis-engine-spec.md)):

1. `features/chroma/extractor.rs`: FFT → chroma vector (12 semitones)
   - Frequency to semitone mapping
   - Octave summation
   - L2 normalization

2. `features/chroma/normalization.rs`: Chroma sharpening and smoothing
3. `features/chroma/smoothing.rs`: Temporal smoothing (median/average filtering)

4. `features/key/templates.rs`: Krumhansl-Kessler profiles (24 keys: 12 major + 12 minor)
5. `features/key/detector.rs`: Template matching algorithm
6. `features/key/key_clarity.rs`: Tonal clarity scoring

**Testing**: Known key tracks (C major, A minor, etc.), target: 70-75% accuracy

**Deliverable**: Key detection module with confidence scoring

### Phase 1E: Integration & Tuning (Week 5)

**Goal**: Integrate all modules, implement confidence scoring, tune accuracy.

**Modules** ([Section 2.7-2.8](audio-analysis-engine-spec.md)):

1. `analysis/confidence.rs`: Multi-factor confidence scoring
   - BPM confidence (onset strength + method agreement)
   - Key confidence (key clarity + chroma consistency)
   - Grid stability (HMM likelihood)

2. `analysis/result.rs`: Final result types (`AnalysisResult`, `BeatGrid`, etc.)
3. `analysis/metadata.rs`: Analysis metadata and flags
4. `lib.rs`: Public API (`analyze_audio()` function)

**Testing**:
- Comprehensive test suite (100+ real DJ tracks)
- Accuracy report generation
- Performance benchmarking
- Edge case identification

**Deliverable**: v0.9-alpha with 85%+ BPM accuracy, 70%+ key accuracy, full pipeline working

## Phase 2: ML Refinement (Weeks 6-8)

### Phase 2A: Data Collection (Week 6)

**Goal**: Build ground truth dataset for ML model training.

**Tasks**:
- Collect 1000+ diverse DJ tracks (electronic, hip-hop, breakbeats, live)
- Annotate ground truth: BPM (from Rekordbox/manual), Key (from Mixed In Key/manual)
- Extract features from each track using Phase 1 pipeline
- Build training dataset: (features) → (correction_factor)

**Data Sources**: Public DJ mixes, test collection, MusicBrainz/AcousticBrainz

**Deliverable**: Training dataset (1000 tracks, feature vectors, labels)

### Phase 2B: ML Model Training & Integration (Week 7)

**Goal**: Train small ONNX model and integrate into pipeline.

**Tasks**:
1. Design lightweight neural network (200-500 parameters)
   - Input: 64 features (BPM, onset histograms, spectral energy, key clarity, etc.)
   - Architecture: Dense(64→32→16→8→1) with ReLU + Dropout
   - Output: Confidence boost factor [0.5, 1.5]

2. Train in Python (PyTorch/scikit-learn), export to ONNX
3. Implement `ml/onnx_model.rs`: Model loading and inference
4. Implement `ml/refinement.rs`: Apply ML corrections to analysis results
5. A/B testing: Classical vs ML-refined accuracy

**Testing**: Validation set accuracy improvement, inference latency <50ms

**Deliverable**: v1.0-beta with ML refinement, 87-88% BPM, 77-78% key accuracy

### Phase 2C: Polish & Release (Week 8)

**Goal**: Production-ready v1.0 release.

**Tasks**:
1. **Documentation**:
   - Comprehensive README.md with examples
   - API documentation (cargo doc)
   - Algorithm explanations ([docs/ALGORITHM_EXPLANATION.md](docs/ALGORITHM_EXPLANATION.md))
   - Performance benchmarks ([docs/PERFORMANCE.md](docs/PERFORMANCE.md))

2. **Code Quality**:
   - Code review and refactoring
   - 80%+ test coverage
   - No panics in release mode
   - Clippy and fmt compliance

3. **Publishing**:
   - Create CHANGELOG.md
   - Set up crates.io publishing
   - Choose license (MIT/Apache dual)
   - Tag v1.0 release on GitHub

4. **Examples**:
   - `examples/analyze_file.rs`: Full pipeline example
   - `examples/batch_process.rs`: Batch processing example

**Deliverable**: **v1.0 published to crates.io**, ready for production use

## Phase 3: Integration (Week 9+)

**Goal**: Integrate into `stratum-desktop` as drop-in replacement.

**Tasks**:
- Add dependency in desktop `Cargo.toml`
- Update command handlers to use new API
- Migrate database schema if needed
- Re-run accuracy tests on real DJ library
- Performance testing (batch processing)
- User acceptance testing

**Deliverable**: `stratum-desktop` v2.0 with new audio analysis engine

## Implementation Strategy

### Working with Cursor

Use prompts from [cursor-prompts.md](cursor-prompts.md) for each module:
1. Read relevant section from [audio-analysis-engine-spec.md](audio-analysis-engine-spec.md)
2. Paste corresponding prompt from [cursor-prompts.md](cursor-prompts.md)
3. Review generated implementation
4. Write/request tests
5. Iterate until tests pass

### Code Style Guidelines

- Use `Result<T>` for all fallible operations (no panics in library code)
- Add `log` crate for debug/info/warn logging
- Numerical stability: epsilon=1e-10 for divisions
- Use `ndarray` for multi-dimensional arrays (Phase 1), consider `Vec<Vec<f32>>` for simple cases
- Performance: Start single-threaded, add `rayon` parallelization later if needed

### Testing Strategy

**Unit Tests**: Each function/module tested in isolation  
**Integration Tests**: Full pipeline on synthetic and real audio  
**Benchmarks**: Performance targets per module (see [audio-analysis-engine-spec.md Section 5.2](audio-analysis-engine-spec.md))  
**Accuracy Tests**: Compare against Rekordbox/Mixed In Key ground truth

## Success Criteria

v1.0 is ready when:
- ✅ BPM detection: ≥88% accuracy (±2 BPM tolerance) on test set
- ✅ Key detection: ≥77% accuracy (exact match) on test set
- ✅ Performance: <500ms per 30s track (single-threaded)
- ✅ Integration: Compiles cleanly into `stratum-desktop`
- ✅ Documentation: Full API docs + algorithm explanations
- ✅ Testing: 80%+ code coverage, no panics in production
- ✅ Open source: Published to crates.io, MIT/Apache dual license

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Accuracy < 85% | Iterate algorithms, compare against reference implementations, add ML early |
| Performance > 500ms | Optimize hot paths, use FFT-accelerated algorithms, parallelize with rayon |
| Octave errors in BPM | Dual-method approach (autocorr + comb), octave tolerance filtering |
| Low key clarity tracks | Return low confidence instead of wrong key, allow manual override |
| Training data collection | Use MusicBrainz/AcousticBrainz, start with smaller dataset (500 tracks) |

## Future Enhancements (Post-v1.0)

- Energy/Intensity detection
- Genre classification
- Mood classification
- Vocal/Instrumental detection
- Harmonic analysis (chord progressions)
- Real-time streaming analysis

## References

- [audio-analysis-engine-spec.md](audio-analysis-engine-spec.md): Complete technical specification
- [development-strategy.md](development-strategy.md): Strategic decisions and workflow
- [cursor-prompts.md](cursor-prompts.md): Ready-to-use Cursor AI prompts
- [EXECUTIVE-SUMMARY.md](EXECUTIVE-SUMMARY.md): Business case and justification
- [open-source-licensing-strategy.md](open-source-licensing-strategy.md): Licensing model
- [QUICK-REFERENCE.md](QUICK-REFERENCE.md): Quick reference guide