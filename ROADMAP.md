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
  - [x] Unit tests for all modules (32 tests)
  - [x] Integration tests with known BPM fixtures (120 BPM, 128 BPM)
  - [x] Validation on real audio data
  - [x] Comprehensive test coverage

- [x] **Integration**
  - [x] Integrated into main `analyze_audio()` function
  - [x] BPM and confidence returned in `AnalysisResult`
  - [x] Error handling for edge cases

- [x] **Enhancements & Optimizations**
  - [x] Coarse-to-fine search optimization (reduces computation time by ~50%)
  - [x] Adaptive tolerance window (BPM-dependent, improves accuracy)
  - [x] Detailed literature citations in function documentation

- [x] **Performance Benchmarks**
  - [x] Period estimation total: 15-45ms for 30s track (autocorrelation + comb filterbank)
    - Autocorrelation: ~18.7 µs (8-beat pattern), ~5-15ms extrapolated (30s track)
    - Comb filterbank: ~11.1 µs (8-beat pattern), ~10-30ms extrapolated (30s track)
    - Combined: 15-45ms total (<50ms target ✓)
  - [x] Coarse-to-fine (optional optimization): ~7.7 µs (8-beat pattern), ~5-15ms extrapolated (30s track)
    - Can replace comb filterbank for 10-30ms total when used
  - [x] Full pipeline: ~11.6ms for 30s track (43x faster than 500ms target)
  - [x] All methods exceed performance targets
  - [x] Comprehensive benchmark suite (`benches/audio_analysis_bench.rs`)
  - [x] Benchmark report documented (`docs/progress-reports/PHASE_1B_BENCHMARKS.md`)

- [x] **Documentation**
  - [x] Academic literature references (Ellis & Pikrakis 2006, Gkiokas et al. 2012)
  - [x] Comprehensive module documentation
  - [x] Public API documentation
  - [x] Enhancement documentation
  - [x] Benchmark results and validation reports

**Status**: ✅ Complete - All modules implemented, tested, validated, benchmarked, and documented

---

### Phase 1C: Beat Tracking (Week 3) ✅

- [x] **Beat Grid Generation**
  - [x] HMM Viterbi algorithm
  - [x] Bayesian tempo tracking
  - [x] Beat grid generation
  - [x] Downbeat detection

- [x] **Enhancements**
  - [x] Variable tempo detection and integration
  - [x] Time signature detection (4/4, 3/4, 6/8)
  - [x] Automatic Bayesian refinement for variable-tempo segments

- [x] **Testing**
  - [x] Beat grid validation
  - [x] <50ms jitter target (validated in integration tests)

- [x] **Performance Benchmarks**
  - [x] HMM Viterbi: ~2.50 µs (16 beats), ~20-50ms extrapolated (30s)
  - [x] Bayesian Update: ~1.10 µs (16 beats), ~10-20ms extrapolated (30s)
  - [x] Tempo Variation: ~601 ns (16 beats), ~5-10ms extrapolated (30s)
  - [x] Time Signature: ~200 ns (16 beats), ~1-5ms extrapolated (30s)
  - [x] Full Beat Grid: ~3.75 µs (16 beats), ~20-50ms extrapolated (30s)
  - [x] Full Pipeline: ~11.56ms for 30s track (includes beat tracking, ~43x faster than target)
  - [x] All methods exceed performance targets
  - [x] Comprehensive benchmark suite (`benches/audio_analysis_bench.rs`)
  - [x] Benchmark report documented (`docs/progress-reports/PHASE_1C_BENCHMARKS.md`)
  - [x] Variable tempo handling validation
  - [x] Time signature detection validation

**Status**: ✅ Complete - All modules implemented, tested, validated, benchmarked, and documented with enhancements

---

### Phase 1D: Key Detection (Week 4) ✅

- [x] **Chroma Analysis**
  - [x] Chroma vector extraction (STFT-based)
  - [x] Chroma normalization (L2 normalization, sharpening)
  - [x] Temporal smoothing (median and average filtering)

- [x] **Key Detection**
  - [x] Krumhansl-Kessler templates (24 keys: 12 major + 12 minor)
  - [x] Template matching algorithm
  - [x] Key clarity scoring

- [x] **Testing**
  - [x] Comprehensive unit tests (40 tests)
  - [x] Integration tests with known key fixtures
  - [x] 70-75% accuracy target (validated against literature)

- [x] **Enhancements**
  - [x] Musical notation display (e.g., "C", "Am", "F#", "D#m")
  - [x] DJ standard numerical format (1A, 2B, etc.) without trademarked names
  - [x] Comprehensive documentation and literature references

**Status**: ✅ Complete - All modules implemented, tested, validated, and documented

---

### Phase 1E: Integration & Tuning (Week 5) ✅

- [x] **Integration**
  - [x] Confidence scoring system
  - [x] Result aggregation
  - [x] Public API (`analyze_audio()`)
  - [x] Error handling refinement

- [x] **Testing & Validation**
  - [x] Comprehensive test suite (211+ tests)
  - [x] Accuracy validation framework
  - [x] Performance benchmarking
  - [x] Edge case identification

- [x] **Target Metrics**
  - [x] Performance: ~75-150ms per 30s track (3-6x faster than 500ms target)
  - [x] Confidence scoring: <1ms overhead
  - [x] All tests passing (100%)

- [x] **Documentation**
  - [x] Phase 1E completion report
  - [x] Validation report
  - [x] Benchmark report
  - [x] Literature review

**Deliverable**: ✅ v0.9-alpha with full classical DSP pipeline

**Status**: ✅ Complete - All modules integrated, confidence scoring implemented, validated, and documented

---

### Phase 1F: Tempogram BPM Pivot (Critical Fix) 🔄

**Status**: ⚠️ Implemented - Initial Validation Poor (Tuning Required)

**Problem Identified**: Current period estimation (autocorrelation + comb filterbank) is fundamentally limited to ~30% accuracy due to frame-by-frame analysis approach. This is a critical architectural flaw.

**Solution**: Complete replacement with Fourier tempogram (Grosche et al. 2012), the industry standard achieving 85-92% accuracy through global temporal analysis.

- [x] **Novelty Curve Implementation**
  - [x] Spectral flux novelty detection
  - [x] Energy flux novelty detection
  - [x] High-frequency content (HFC) novelty detection
  - [x] Combined novelty curve with weighted voting
  - [x] File: `src/features/period/novelty.rs` (NEW)

- [x] **Autocorrelation Tempogram**
  - [x] Autocorrelation-based periodicity analysis (test each BPM hypothesis)
  - [x] For each BPM (40-240, 0.5 resolution): compute autocorrelation at tempo lag
  - [x] Peak detection: find BPM with highest autocorrelation strength
  - [x] Confidence scoring based on peak prominence
  - [x] File: `src/features/period/tempogram_autocorr.rs` (NEW)

- [x] **FFT Tempogram**
  - [x] FFT-based periodicity analysis (research shows more consistent)
  - [x] Apply FFT to novelty curve, convert frequencies to BPM
  - [x] Peak detection: find BPM with highest FFT power
  - [x] Confidence scoring based on peak prominence
  - [x] File: `src/features/period/tempogram_fft.rs` (NEW)

- [x] **Tempogram Comparison & Selection**
  - [x] Compare both methods on same input
  - [x] Selection logic: use best method or ensemble
  - [ ] A/B testing framework for validation (old vs new methods; report results)
  - [x] File: `src/features/period/tempogram.rs` (NEW - main entry point)

- [x] **Multi-Resolution Validation**
  - [x] Tempogram at 3 hop sizes (256, 512, 1024)
  - [x] Cross-resolution agreement validation
  - [x] Consensus BPM selection
  - [x] File: `src/features/period/multi_resolution.rs` (NEW)

- [x] **Integration & Migration**
  - [x] Add tempogram methods to `mod.rs` (keep old methods for comparison)
  - [x] Update main analysis pipeline in `lib.rs` to use tempogram (legacy fallback retained)
  - [ ] A/B testing: run old and new methods side-by-side
  - [ ] Document comparison results

- [ ] **Deprecation Plan (After Validation)**
  - [ ] Mark old methods as `#[deprecated]` after validation confirms tempogram superiority
  - [ ] Add deprecation warnings in documentation
  - [ ] Keep functional for 1-2 releases for transition
  - [ ] Remove in v0.9.2 or later (final cleanup)

- [x] **Testing (Unit)**
  - [x] Unit tests for new tempogram components
  - [ ] Fix failing legacy unit tests (`features::period::candidate_filter`) before declaring full suite “green”

- [ ] **Validation (Empirical)**
  - [x] Initial FMA Small batch run (30 tracks) completed
  - [x] Results captured in `docs/progress-reports/PHASE_1F_VALIDATION.md`
  - [ ] Address dominant failure mode: tempo octave / metrical-level selection (~2× errors)
  - [ ] Improve novelty conditioning and confidence calibration
  - [ ] Re-run FMA validation on multiple batches (N ≥ 5) and report mean/variance

- [ ] **Performance Benchmarks**
  - [ ] Single-resolution tempogram: <50ms for 30s track
  - [ ] Multi-resolution tempogram: <150ms for 30s track
  - [ ] Full pipeline performance validation

- [ ] **Documentation**
  - [x] Algorithm documentation (tempogram approach) and technical spec
  - [x] Validation report (initial run; tuning required)
  - [x] Literature review (Phase 1F)
  - [ ] Migration guide from old to new system (post-validation)
  - [ ] Performance benchmarks report (post-benchmark)
  - [ ] Accuracy validation report (post-tuning)

**Expected Results**:
- Accuracy (±2 BPM): 20% → 80%+
- Accuracy (±5 BPM): 30% → 85-92%
- Subharmonic Errors: 10-15% → <1%
- MAE: 34 BPM → 3-4 BPM

**Timeline**: Implementation complete; tuning + validation TBD (driven by empirical results)

**Implementation Strategy**:
- Implement BOTH FFT and autocorrelation tempogram for maximum accuracy
- Compare empirically on test batch
- Choose best method or use ensemble
- Document hybrid approach (FFT coarse + autocorr fine) for future enhancement

**Deliverable**: ✅ v0.9.1-alpha with dual tempogram BPM detection implemented; ⚠️ not yet meeting accuracy targets

**Future Enhancement**: Hybrid approach (FFT coarse + autocorr fine) documented for future implementation

**Status**: ⚠️ Implemented - Requires tuning and re-validation to meet targets

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
**Current Phase**: Phase 1F (Tempogram Pivot) - Implemented, tuning/validation in progress  
**Overall Progress**: 62.5% (5/8 weeks complete) + Critical Pivot Required

---

## Critical Pivot: Tempogram BPM Detection

**Status**: Phase 1F - Implemented, initial validation indicates tuning required

The current BPM detection system (Phase 1B) is fundamentally limited to ~30% accuracy due to frame-by-frame analysis. A complete replacement with dual tempogram approach (FFT + Autocorrelation, Grosche et al. 2012) is required before Phase 2. Both methods will be implemented and compared empirically for maximum accuracy.

**See**:
- `docs/progress-reports/TEMPOGRAM_PIVOT_EVALUATION.md` for complete technical specification
- `docs/progress-reports/PHASE_1F_VALIDATION.md` for the initial empirical validation run (FMA Small)

**Timeline**: 3-4 hours for complete implementation (both FFT and autocorr tempogram + comparison)

**Expected**: 85-92% accuracy (vs current 30%) - using best of both methods

## Notes

- See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed implementation guidelines
- See `PIPELINE.md` for the authoritative end-to-end processing logic and decision points
- See `docs/progress-reports/TEMPOGRAM_PIVOT_EVALUATION.md` for tempogram pivot specification
- Reference `docs/literature/` for academic foundations
- All placeholder functions are marked with `_` prefix for easy identification

Validation / A-B tooling note:
- The validation harness and `examples/analyze_file` support controlled modes (`--force-legacy-bpm`, `--bpm-fusion`, `--no-preprocess`, `--no-onset-consensus`) to support Phase 1F tuning and regression testing.

