# Stratum DSP - Development Roadmap

**Target**: v1.0 release in 8 weeks  
**Goal**: 88% BPM accuracy, 77% key detection accuracy  
**Performance**: <500ms per 30s track

---

## Phase 0: Project Setup ✅

- [x] Repository structure
- [x] Cargo.toml with dependencies
- [x] Module scaffolding
- [x] CI/CD workflow
- [x] Documentation framework

**Status**: Complete

---

## Phase 1: Classical DSP (Weeks 1-5)

### Phase 1A: Preprocessing & Onset Detection (Week 1) ✅

- [x] **Preprocessing**
  - [x] Normalization (peak, RMS, LUFS)
  - [x] Silence detection and trimming
  - [x] Channel mixing (stereo to mono)

- [x] **Onset Detection**
  - [x] Energy flux method
  - [x] Spectral flux method
  - [x] High-frequency content (HFC)
  - [x] Harmonic-percussive source separation (HPSS)
  - [x] Consensus voting algorithm

- [x] **Testing**
  - [x] Unit tests for each method (75 tests)
  - [x] Integration tests with real audio fixtures (5 tests)
  - [x] Synthetic kick pattern validation (120 BPM, 128 BPM)
  - [x] Comprehensive test coverage (80 tests passing)
  - [x] Test fixture generation script

- [x] **Documentation**
  - [x] Academic literature references
  - [x] Comprehensive module documentation
  - [x] Phase 1A completion report
  - [x] Validation report

- [x] **Enhancements**
  - [x] Adaptive thresholding (median + MAD)
  - [x] K-weighting filter verification
  - [x] Code review improvements

**Status**: ✅ Complete - All modules implemented, tested, validated, and documented

---

### Phase 1B: Period Estimation (Week 2) ✅

- [x] **BPM Detection**
  - [x] Autocorrelation-based estimation (FFT-accelerated)
  - [x] Comb filterbank estimation
  - [x] Peak picking algorithm
  - [x] Candidate filtering and merging
  - [x] Octave error handling

- [x] **Testing**
  - [x] Unit tests for all modules (29 tests)
  - [x] Integration tests with known BPM fixtures (120 BPM, 128 BPM)
  - [x] Validation on real audio data
  - [x] Comprehensive test coverage

- [x] **Integration**
  - [x] Integrated into main `analyze_audio()` function
  - [x] BPM and confidence returned in `AnalysisResult`
  - [x] Error handling for edge cases

- [x] **Documentation**
  - [x] Academic literature references (Ellis & Pikrakis 2006, Gkiokas et al. 2012)
  - [x] Comprehensive module documentation
  - [x] Public API documentation

**Status**: ✅ Complete - All modules implemented, tested, validated, and documented

---

### Phase 1C: Beat Tracking (Week 3)

- [ ] **Beat Grid Generation**
  - [ ] HMM Viterbi algorithm
  - [ ] Bayesian tempo tracking
  - [ ] Beat grid generation
  - [ ] Downbeat detection

- [ ] **Testing**
  - [ ] Beat grid validation
  - [ ] <50ms jitter target

**Deliverable**: Beat tracking module with grid generation

---

### Phase 1D: Key Detection (Week 4)

- [ ] **Chroma Analysis**
  - [ ] Chroma vector extraction
  - [ ] Chroma normalization
  - [ ] Temporal smoothing

- [ ] **Key Detection**
  - [ ] Krumhansl-Kessler templates (24 keys)
  - [ ] Template matching algorithm
  - [ ] Key clarity scoring

- [ ] **Testing**
  - [ ] Known key track validation
  - [ ] 70-75% accuracy target

**Deliverable**: Key detection module with confidence scoring

---

### Phase 1E: Integration & Tuning (Week 5)

- [ ] **Integration**
  - [ ] Confidence scoring system
  - [ ] Result aggregation
  - [ ] Public API (`analyze_audio()`)
  - [ ] Error handling refinement

- [ ] **Testing & Validation**
  - [ ] Comprehensive test suite (100+ tracks)
  - [ ] Accuracy report generation
  - [ ] Performance benchmarking
  - [ ] Edge case identification

- [ ] **Target Metrics**
  - [ ] 85%+ BPM accuracy (±2 BPM tolerance)
  - [ ] 70%+ key accuracy (exact match)
  - [ ] <500ms per 30s track

**Deliverable**: v0.9-alpha with full classical DSP pipeline

---

## Phase 2: ML Refinement (Weeks 6-8)

### Phase 2A: Data Collection (Week 6)

- [ ] **Dataset Preparation**
  - [ ] Collect 1000+ diverse DJ tracks
  - [ ] Annotate ground truth (BPM, key)
  - [ ] Extract features using Phase 1 pipeline
  - [ ] Build training dataset

**Deliverable**: Training dataset (1000 tracks, feature vectors, labels)

---

### Phase 2B: Model Training & Integration (Week 7)

- [ ] **ML Model**
  - [ ] Design lightweight neural network (200-500 params)
  - [ ] Train ONNX model (Python)
  - [ ] Implement ONNX inference (Rust)
  - [ ] Integrate ML refinement pipeline
  - [ ] A/B testing (classical vs ML-refined)

- [ ] **Target Metrics**
  - [ ] 87-88% BPM accuracy
  - [ ] 77-78% key accuracy
  - [ ] <50ms inference latency

**Deliverable**: v1.0-beta with ML refinement

---

### Phase 2C: Polish & Release (Week 8)

- [ ] **Documentation**
  - [ ] Comprehensive README with examples
  - [ ] API documentation (cargo doc)
  - [ ] Algorithm explanations
  - [ ] Performance benchmarks

- [ ] **Code Quality**
  - [ ] 80%+ test coverage
  - [ ] Code review and refactoring
  - [ ] Clippy and fmt compliance
  - [ ] No panics in release mode

- [ ] **Publishing**
  - [ ] CHANGELOG.md updates
  - [ ] crates.io publishing setup
  - [ ] v1.0 release tag
  - [ ] GitHub release notes

**Deliverable**: v1.0 published to crates.io

---

## Phase 3: Integration (Week 9+)

- [ ] **Desktop Integration**
  - [ ] Add dependency to stratum-desktop
  - [ ] Update command handlers
  - [ ] Database schema migration (if needed)
  - [ ] Accuracy validation on real library
  - [ ] Performance testing
  - [ ] User acceptance testing

**Deliverable**: stratum-desktop v2.0 with new analysis engine

---

## Success Criteria

v1.0 is ready when:

- ✅ BPM detection: ≥88% accuracy (±2 BPM tolerance)
- ✅ Key detection: ≥77% accuracy (exact match)
- ✅ Performance: <500ms per 30s track
- ✅ Documentation: Full API docs + algorithm explanations
- ✅ Testing: 80%+ code coverage, no panics
- ✅ Published: crates.io, MIT/Apache dual license

---

## Current Status

**Last Updated**: 2025-01-XX  
**Current Phase**: Phase 1B (Complete) → Phase 1C (Next: Beat Tracking)  
**Overall Progress**: 25% (2/8 weeks complete)

---

## Notes

- See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed implementation guidelines
- Reference `.reference/` directory for internal planning documents
- All placeholder functions are marked with `_` prefix for easy identification

