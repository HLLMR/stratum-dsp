# Stratum Audio Analysis - Development Strategy & Workflow

## Part A: STRATEGIC DECISIONS

### Why a Separate Crate?

**The case FOR building separately:**

1. **Modularity**: Audio analysis is a distinct, reusable domain
   - Can be used in stratum-desktop, stratum-web, stratum-cloud
   - Can be published to crates.io (establishes Stratum as DSP authority in DJ space)
   - Independent CI/CD pipeline

2. **Development Velocity**: Decouple from desktop UI iterations
   - Desktop can have v1.0 ready while audio engine is still on v0.8
   - Can iterate on algorithm without blocking app releases
   - Easier to test in isolation

3. **Open Source Positioning**: Creates competitive moat
   - Publish as open-source → community contribution + visibility
   - Position Stratum as "the DJ-focused audio analysis library for Rust"
   - Leverage for recruiting + credibility

4. **Quality Bar**: Forces clarity on API boundaries
   - Separate crate = public API design from day 1
   - Can't have messy internal coupling
   - Results in cleaner integration into desktop

**The case AGAINST:**

- Initial setup overhead (separate repo, CI/CD)
- Requires clear API contract upfront
- Cannot iterate rapidly on both crate + desktop together

**VERDICT**: Build separately. The benefits outweigh the overhead, especially since you're aiming for professional-grade quality.

---

### Repository Structure Recommendation

```
stratum-workspace/
├── stratum-audio-analysis/          ← NEW CRATE (this project)
│   ├── Cargo.toml
│   ├── src/
│   ├── tests/
│   ├── benches/
│   ├── README.md
│   ├── CHANGELOG.md
│   └── CI config (.github/workflows/)
│
├── stratum-shared/                  ← EXISTING (common types)
│   ├── Cargo.toml
│   └── src/
│
├── stratum-desktop/                 ← EXISTING (app integration later)
│   └── src-tauri/
│       └── Cargo.toml
│           [depends on stratum-audio-analysis = { path = "../../stratum-audio-analysis" }]
│
└── Cargo.workspace.toml
    [members = ["stratum-audio-analysis", "stratum-shared", "stratum-desktop"]]
```

**Benefit**: `cargo build` from workspace root builds all 3 crates with unified dependency resolution.

---

## Part B: BUILD STRATEGY

### Option 1: Start from Scratch (RECOMMENDED)

**Approach**: Implement all algorithms from first principles, using research papers as reference.

**Pros**:
- ✅ Optimized for your specific use case
- ✅ No external dependencies (just symphonia, rustfft, rayon)
- ✅ Full control over accuracy tuning
- ✅ Learnable codebase for your team
- ✅ Can optimize hot paths aggressively

**Cons**:
- ❌ Longer development time (8-10 weeks vs 4-6)
- ❌ More risk of subtle bugs in complex algorithms

**Timeline**: 10 weeks, v1.0 release with 85%+ accuracy

---

### Option 2: Leverage Academic Libraries (NOT RECOMMENDED)

**Approach**: Port existing algorithms (librosa, aubio, essentia paper implementations).

**Pros**:
- ✅ Battle-tested algorithms
- ✅ Faster initial implementation

**Cons**:
- ❌ Essentia = licensing nightmare (AGPL)
- ❌ librosa = Python (not Rust)
- ❌ aubio = C (FFI overhead)
- ❌ Still need to validate for DJ use case

**Verdict**: Skip this. The "from scratch" approach is actually cleaner.

---

### Option 3: Hybrid (BEST APPROACH)

**Approach**: 
- Study research papers deeply (they're free, open)
- Implement algorithms in Rust from scratch
- Use academic implementations as reference for validation only
- Add DJ-specific tuning on top

**Timeline**: 9-10 weeks, v1.0 with 86-88% accuracy + ML roadmap

This is the recommended path.

---

## Part C: PHASED DELIVERY PLAN

### Phase 0: Setup (Week -1, before you start coding)

- [ ] Create `stratum-audio-analysis` repo
- [ ] Set up workspace with stratum-shared dependency
- [ ] Create Cargo.toml with target dependencies
- [ ] Set up GitHub Actions CI
- [ ] Create test fixtures directory structure
- [ ] Write preliminary literature review doc (for Cursor reference)

**Deliverable**: Buildable, empty skeleton crate

---

### Phase 1a: Onset Detection (Weeks 1-1.5)

**Goal**: Detect beat transients with 4 independent methods.

**Work Items**:
- [ ] Implement `preprocessing/normalization.rs` (peak + LUFS)
- [ ] Implement `preprocessing/silence.rs`
- [ ] Implement `preprocessing/channel_mixer.rs`
- [ ] Implement `features/onset/energy_flux.rs`
- [ ] Implement `features/onset/spectral_flux.rs` (requires basic FFT)
- [ ] Implement `features/onset/hfc.rs`
- [ ] Implement `features/onset/hpss.rs` (median filtering)
- [ ] Implement `features/onset/consensus.rs` (voting)
- [ ] Write unit tests for each method
- [ ] Create test fixtures: synthetic drum patterns, real tracks

**Testing**:
```
Input: Simple kick pattern (120 BPM, 4-on-floor)
Expected: 4 onsets per bar detected consistently
Accuracy: 100% (synthetic data)

Input: Real EDM track
Expected: 500-1000 onsets for 30s track
Accuracy: Reasonable clustering (visual inspection)
```

**Deliverable**: Onset detection module, 90%+ code coverage

---

### Phase 1b: Period Estimation (Weeks 1.5-2.5)

**Goal**: Convert onset list → BPM candidates.

**Work Items**:
- [ ] Implement `features/period/autocorrelation.rs`
- [ ] Implement `features/period/comb_filter.rs`
- [ ] Implement `features/period/peak_picking.rs` (robust peak detection)
- [ ] Implement `features/period/candidate_filter.rs` (merge results)
- [ ] Write integration tests: known BPM tracks
- [ ] Benchmark performance

**Testing**:
```
Input: 120 BPM kick pattern
Expected: BPM estimate 119-121 BPM
Accuracy: >95% (synthetic data)

Input: Real EDM track
Expected: BPM within ±5 of ground truth
Accuracy: ~80% (real data, needs tuning)
```

**Deliverable**: Period estimation, ready for beat tracking

---

### Phase 1c: Beat Tracking & Grid (Weeks 2.5-3.5)

**Goal**: Generate precise beat grid.

**Work Items**:
- [ ] Implement `features/beat_tracking/hmm.rs` (Viterbi algorithm)
- [ ] Implement `features/beat_tracking/bayesian.rs` (tempo updates)
- [ ] Implement grid generation (downbeats, bars)
- [ ] Write tests: compare grid to manually annotated tracks
- [ ] Benchmark HMM performance

**Testing**:
```
Input: 120 BPM track with onset list
Expected: Beat grid with <50ms jitter
Accuracy: Visual inspection against Rekordbox grid
```

**Deliverable**: Beat tracking module, ~85% BPM accuracy

---

### Phase 1d: Key Detection (Weeks 3.5-4.5)

**Goal**: Detect musical key.

**Work Items**:
- [ ] Implement `features/chroma/extractor.rs` (FFT → chroma)
- [ ] Implement `features/chroma/normalization.rs` (sharpening)
- [ ] Implement `features/chroma/smoothing.rs`
- [ ] Implement `features/key/templates.rs` (Krumhansl-Kessler 24 keys)
- [ ] Implement `features/key/detector.rs` (template matching)
- [ ] Implement `features/key/key_clarity.rs`
- [ ] Write tests: know key tracks (C major, A minor, etc.)

**Testing**:
```
Input: 30s C major audio loop
Expected: Detect as C major with >0.7 confidence
Accuracy: ~70-75% on real tracks
```

**Deliverable**: Key detection module, confidence scoring

---

### Phase 1e: Integration & Tuning (Week 4.5-5)

**Goal**: Bring it all together, tune accuracy.

**Work Items**:
- [ ] Implement `analysis/confidence.rs` (scoring system)
- [ ] Implement `analysis/result.rs` (output types)
- [ ] Implement `analysis/metadata.rs`
- [ ] Create comprehensive test suite (100+ real tracks)
- [ ] Benchmark full pipeline end-to-end
- [ ] Generate accuracy report (BPM %, Key %)
- [ ] Document any edge cases found

**Testing**:
```
Full dataset (100+ tracks):
- BPM accuracy: ? %
- Key accuracy: ? %
- Performance: ? ms per 30s track

Identify failing categories (fast, slow, breakbeats, live, etc.)
```

**Deliverable**: v0.9 alpha, accuracy metrics, list of known issues

---

### Phase 2a: ML Refinement - Data Collection (Week 5-6)

**Goal**: Build ground truth dataset for model training.

**Work Items**:
- [ ] Collect 1000 diverse DJ tracks (electronic, hip-hop, breakbeats, live)
- [ ] Annotate ground truth: BPM (manual or from Rekordbox), Key (manual or from Mixed In Key)
- [ ] Extract features from each track
- [ ] Build training dataset: (features) → (correction_factor)

**Data sources**:
- Public DJ mixes (with known BPM from metadata)
- Your own test collection
- Freesound.org curated collections

**Deliverable**: Training dataset (1000 tracks, feature vectors, labels)

---

### Phase 2b: ML Model Training & Integration (Week 6-7)

**Goal**: Train small ONNX model, integrate into pipeline.

**Work Items**:
- [ ] Design small neural net (200-500 parameters)
- [ ] Train in Python (scikit-learn or PyTorch)
- [ ] Export to ONNX format
- [ ] Implement `ml/onnx_model.rs` (model loading)
- [ ] Implement `ml/refinement.rs` (inference + result adjustment)
- [ ] Test refinement accuracy improvement
- [ ] Benchmark inference latency

**Testing**:
```
v0.9 accuracy: 85% BPM, 75% Key
v1.0 accuracy (with ML): 87-88% BPM, 77-78% Key
Inference overhead: <50ms per track
```

**Deliverable**: v1.0 beta with ML refinement

---

### Phase 2c: Polish & Release (Week 7.5-8)

**Goal**: Production-ready release.

**Work Items**:
- [ ] Write comprehensive README.md (with examples)
- [ ] Document all public APIs (cargo doc)
- [ ] Write PERFORMANCE.md (benchmarks, accuracy, limitations)
- [ ] Create CHANGELOG.md
- [ ] Set up crates.io publishing
- [ ] Choose license (Apache 2.0 / MIT dual)
- [ ] Tag v1.0 release on GitHub

**Deliverable**: **v1.0 release to crates.io**

---

### Phase 3: Integration into stratum-desktop (Week 8-9)

**Goal**: Replace current DSP with new analysis engine.

**Work Items**:
- [ ] Add dependency in desktop Cargo.toml
- [ ] Update command handlers to use new API
- [ ] Migrate database schema (if needed)
- [ ] Re-run accuracy tests on real DJ library
- [ ] Performance testing (batch processing)
- [ ] User acceptance testing (with beta testers)

**Deliverable**: stratum-desktop v2.0 with new audio analysis

---

## Part D: TECHNICAL DECISIONS

### FFT Choice: rustfft vs GPU FFT

**For initial development**:
- Use `rustfft` (pure Rust, no GPU complexity)
- CPU FFT for 2048-point: ~2-5ms (acceptable)
- Can optimize later with GPU if needed

**After Phase 1, if too slow**:
- Add GPU FFT via OpenCL (you already have infrastructure)
- Use `rustfft` as fallback

**Decision**: Start with rustfft, optimize later if needed.

---

### Data Structures: ndarray vs Vec

**Option A**: Use `ndarray` crate (numpy-like interface)
```rust
use ndarray::{Array1, Array2};
let spectrogram = Array2::<f32>::zeros((n_frames, n_bins));
```

**Pros**: Elegant, familiar to ML folks, built-in linear algebra

**Cons**: Extra dependency, slight performance overhead

**Option B**: Use `Vec<Vec<f32>>` (raw Rust)
```rust
let spectrogram: Vec<Vec<f32>> = vec![vec![0.0; n_bins]; n_frames];
```

**Pros**: Zero overhead, simple, no extra dependencies

**Cons**: Less ergonomic, manual memory management

**Decision**: Start with Vec, migrate to ndarray in Phase 2 if doing ML.

---

### Parallelization Strategy

**Single-threaded for Phase 1**: 
- Focus on correctness first
- Parallelize hot paths later if needed
- Use `rayon` for embarrassingly parallel operations (batch processing)

**Hot paths to parallelize**:
1. STFT computation across frames
2. Chroma extraction (per-frame)
3. Autocorrelation (can use FFT-based fast version)

**Decision**: Single-threaded initially, add rayon after Phase 1.

---

### Error Handling

**Use Result<T> for all fallible operations**:

```rust
pub enum AudioAnalysisError {
    DecodingError(String),
    InvalidInput(String),
    ProcessingError(String),
}

pub type AnalysisResult<T> = Result<T, AudioAnalysisError>;
```

**Never panic in library code** (except during setup/configuration).

---

### Logging & Debugging

**Use `log` crate** for diagnostic output:

```rust
use log::{debug, info, warn};

debug!("Computing FFT for frame {}", frame_idx);
info!("Detected {} onsets with consensus voting", onset_count);
warn!("Low key clarity: {}, consider manual override", clarity);
```

**Enable via**:
```rust
// In tests/bin:
env_logger::init();
```

---

## Part E: WORKING WITH CURSOR

### How to Use This Spec with Cursor

**Setup**:
1. Put `audio-analysis-engine-spec.md` in repo root
2. Tell Cursor: "Read `TECHNICAL.md` for implementation details"
3. Let it reference algorithms when implementing

**Per-module workflow**:
1. Cursor implements one module (e.g., `energy_flux.rs`)
2. You review implementation
3. You write tests
4. Cursor fixes any issues
5. Move to next module

**Key prompts for Cursor**:
- "Implement Section 2.2.2 (Energy Flux) from TECHNICAL.md"
- "Write comprehensive unit tests for energy_flux detection"
- "Add benchmarking code for autocorrelation performance"
- "Refactor chroma extraction to use ndarray (Section 2.5.1)"

---

## Part F: SUCCESS MILESTONES

### v0.1-alpha (End of Week 1)
- Preprocessing + onset detection (energy flux only)
- Basic tests passing
- Accuracy: Unknown (just checking it doesn't crash)

### v0.5-alpha (End of Week 3)
- All onset methods + period estimation
- BPM detection working
- Accuracy: ~75% BPM (real data)

### v0.9-alpha (End of Week 5)
- Full classical DSP pipeline (all 4 modules)
- ~85% BPM accuracy
- ~70% Key accuracy
- Ready for ML training data collection

### v1.0 (End of Week 8)
- ML refinement integrated
- ~88% BPM accuracy
- ~77% Key accuracy
- Published to crates.io
- Full documentation
- Ready for production

---

## Part G: KNOWN CHALLENGES & MITIGATION

### Challenge 1: Octave Errors in BPM

**Problem**: Autocorrelation often detects 2x or 0.5x the true BPM.

**Mitigation**:
- Filter unrealistic BPMs (humans can't make music <60 or >180 BPM)
- Combine autocorrelation + comb filter (they disagree on octave errors)
- ML model trained to detect and correct octave errors

---

### Challenge 2: Key Detection on Atonal/Ambient Tracks

**Problem**: No musical key = garbage result.

**Mitigation**:
- Compute key clarity (high clarity = reliable, low = warning)
- Return "key_confidence: 0.2" instead of asserting wrong key
- User can manually override if needed

---

### Challenge 3: Variable Tempo (DJ Mixes, Live Recordings)

**Problem**: Assumes constant tempo, fails on tempo ramps.

**Mitigation**:
- Detect tempo changes (divide track into segments)
- Track tempo per segment
- For Phase 2: Train ML model on tempo-variable data
- Document limitation: "Phase 1 assumes constant tempo"

---

### Challenge 4: Training Data Annotation

**Problem**: Finding 1000 ground-truth DJ tracks is hard.

**Mitigation**:
- Use Rekordbox library (extract BPM from analyzing users' collections)
- Use MusicBrainz + AcousticBrainz (free, annotated, 1M+ tracks)
- Hire 1-2 DJs to manually annotate 100 test tracks for validation
- Start with smaller dataset (500 tracks), expand later

---

## Part H: AFTER V1.0: FUTURE ENHANCEMENTS

### Extension Ideas (Phase 3+)

1. **Energy/Intensity detection**: Classify track intensity (quiet → loud)
2. **Genre classification**: Lightweight ONNX model per genre
3. **Mood classification**: Happy, sad, aggressive, chill
4. **Vocal/Instrumental detection**: Does track have vocals?
5. **Harmonic analysis**: Chord progression detection
6. **Mastering analysis**: Loudness, dynamic range, frequency balance
7. **DJ remix detection**: Identify if remix or bootleg

**Strategy**: Build modular so new analyses are easy to add.

---

## Part I: ESTIMATED EFFORT & RESOURCE PLANNING

**Solo developer (or 1 person + Cursor AI)**:
- Phase 1 (classical DSP): 4-5 weeks
- Phase 2 (ML): 3-4 weeks
- Total: **8-9 weeks** to production v1.0

**Small team (2-3 people)**:
- Can parallelize Phase 1 modules
- Total: **4-5 weeks** to production v1.0

**Resource requirements**:
- 1x powerful CPU (for training ML model, Phase 2)
- Free: All libraries are open-source
- Cost: $0 for tools (except if you buy Mixed In Key for validation)

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-15  
**Status**: Ready for Execution
