# Cursor Agent Prompts for Audio Analysis Engine Development

## Quick Reference: How to Use These Prompts

Each section below is a ready-to-use prompt for Cursor. Customize them with your specific needs and paste directly into Cursor's chat.

---

## PHASE 1A: ONSET DETECTION

### Prompt 1.1: Preprocessing - Normalization Module

```
You are implementing the Stratum Audio Analysis Engine, a professional-grade
DJ BPM and key detection library in pure Rust.

TASK: Implement src-tauri/src/audio/preprocessing/normalization.rs

REQUIREMENTS from TECHNICAL.md Section 2.1.1:
- Struct: NormalizationConfig with target_loudness_lufs, max_headroom_db, method enum
- Support Peak, RMS, and LUFS (ITU-R BS.1770-4) normalization
- Implement loudness metering (gate at -70 LUFS, K-weighting filter)
- Return normalized PCM samples + loudness metadata
- Handle edge cases: silent audio, very quiet/loud audio
- Comprehensive error handling (no panics)

CODE STYLE:
- Use ndarray for multi-dimensional arrays
- Add debug! and warn! logging at decision points
- Add numerical stability: epsilon=1e-10 for divisions
- Include inline documentation for complex math

TESTS:
- Unit test: peak normalization on known signal
- Unit test: LUFS calculation on reference audio (match algorithm)
- Unit test: edge case (silent input, ultra-quiet input)

Return complete implementation with docstrings and tests.
```

### Prompt 1.2: Onset Detection - Energy Flux

```
TASK: Implement src/features/onset/energy_flux.rs

REQUIREMENTS from Section 2.2.2:
- Function: detect_energy_flux_onsets(samples, frame_size, hop_size, threshold_db) → Vec<usize>
- Algorithm:
  1. Divide audio into overlapping frames
  2. Compute RMS energy per frame
  3. Compute energy derivative (flux)
  4. Threshold and peak-pick
- Output: Onset times in samples

PERFORMANCE TARGETS:
- 30s audio in <50ms (single-threaded)
- Use efficient FFT-based convolution if possible

VALIDATE:
- Test on synthetic kick pattern (120 BPM 4-on-floor)
- Test on real electronic music

Return implementation with performance comments and tests.
```

### Prompt 1.3: Onset Consensus Voting

```
TASK: Implement src/features/onset/consensus.rs

This module combines all 4 onset detection methods via weighted voting.

REQUIREMENTS from Section 2.2.6:
- Struct: OnsetConsensus { energy_flux, spectral_flux, hfc, hpss }
- Function: vote_onsets(consensus, weights, tolerance_ms, sample_rate) → Vec<OnsetCandidate>
- OnsetCandidate fields: time_samples, time_seconds, confidence (0.0-1.0), voted_by (count)
- Merge onsets within tolerance_ms windows (default 50ms)
- Confidence = sum of weights from methods that detected this onset
- Higher voted_by = higher confidence

ALGORITHM PSEUDOCODE:
1. Sort all onsets from 4 methods
2. Cluster onsets within 50ms tolerance
3. For each cluster, sum weights of method that detected it
4. Normalize confidence to [0, 1]
5. Return sorted by confidence

Return implementation with clustering algorithm and tests.
```

---

## PHASE 1B: PERIOD ESTIMATION

### Prompt 1.4: Autocorrelation BPM Estimation

```
TASK: Implement src/features/period/autocorrelation.rs

REQUIREMENTS from Section 2.3.2:
- Function: estimate_bpm_from_autocorrelation(onsets, sample_rate, hop_size, min_bpm, max_bpm)
- Returns: Vec<BpmCandidate> with fields { bpm: f32, confidence: f32 }
- Algorithm:
  1. Convert onset list to beat signal (binary: 1 if onset, 0 else)
  2. Compute autocorrelation (use FFT-based O(n log n) algorithm)
  3. Find peaks in ACF
  4. Filter peaks within [min_bpm, max_bpm] range
  5. Convert lag values to BPM
  6. Score by ACF peak height (normalize by max)

NUMERICAL STABILITY:
- Handle empty onset list
- Avoid division by zero
- Add 1e-10 epsilon to denominators

BPM RANGE:
- min_bpm: 60.0 (slowest DJ tempo)
- max_bpm: 180.0 (fastest DJ tempo)

Return complete implementation. Include comments explaining
lag-to-BPM conversion formula.
```

### Prompt 1.5: Comb Filterbank BPM Estimation

```
TASK: Implement src/features/period/comb_filter.rs

REQUIREMENTS from Section 2.3.3:
- Function: estimate_bpm_from_comb_filter(onsets, sample_rate, hop_size, min_bpm, max_bpm, bpm_resolution)
- Returns: Vec<BpmCandidate> { bpm, confidence }
- Algorithm:
  1. For each candidate BPM (80-180, step=0.5 BPM)
  2. Compute expected beat interval in samples
  3. For each onset, find nearest beat
  4. Score: count onsets within ±10% timing tolerance
  5. Normalize score by onset count
  6. Sort by score

OPTIMIZATION:
- Use BPM resolution (e.g., 0.5 for half-BPM precision) to limit candidates
- Default resolution: 1.0 BPM

Return implementation. This should be slower but more robust than autocorr.
```

### Prompt 1.6: Merge BPM Candidates

```
TASK: Implement src/features/period/candidate_filter.rs

REQUIREMENTS from Section 2.3.4:
- Function: merge_bpm_candidates(autocorr_results, comb_results, octave_tolerance_cents) → Vec<BpmEstimate>
- Output struct: BpmEstimate { bpm, confidence, method_agreement }
- Octave error handling: 120 BPM and 60 BPM are same underlying beat
- Algorithm:
  1. Group candidates within octave tolerance (50 cents = semitone)
  2. If autocorr and comb agree within tolerance, boost confidence
  3. Filter outliers (keep top 3-5 candidates)
  4. Return sorted by confidence

OCTAVE TOLERANCE LOGIC:
- 50 cents = 50/1200 = 2^(50/1200) ratio
- If ratio between two BPMs is close to 2.0 or 0.5, they're octave errors

Return implementation with detailed comments on octave detection.
```

---

## PHASE 1C: BEAT TRACKING

### Prompt 1.7: HMM Viterbi Beat Tracker

```
TASK: Implement src/features/beat_tracking/hmm.rs

This is the most complex module. Use a simple Viterbi implementation.

REQUIREMENTS from Section 2.4.1:
- Struct: HmmBeatTracker { bpm_estimate, onsets, sample_rate }
- Method: track_beats() → Vec<BeatPosition>
- Viterbi algorithm:
  1. Build state space: BPM candidates [bpm*0.9, bpm*0.95, bpm, bpm*1.05, bpm*1.1]
  2. Compute transition probabilities (high prob for stable tempo)
  3. Compute emission probabilities (high prob if onset near beat)
  4. Forward pass: compute best path probabilities
  5. Backtrack: extract most likely beat sequence

STATE SPACE: 5 BPM states around nominal BPM
TRANSITIONS: Higher probability for state consistency
EMISSIONS: Higher prob if onset within tolerance

BeatPosition output fields:
- beat_index: 0, 1, 2, 3 (within bar)
- time_seconds: float
- confidence: 0.0-1.0

Return implementation with pseudocode comments. This will be ~200 lines.
```

---

## PHASE 1D: KEY DETECTION

### Prompt 1.8: Chroma Extraction

```
TASK: Implement src/features/chroma/extractor.rs

REQUIREMENTS from Section 2.5.1:
- Function: extract_chroma(samples, sample_rate, frame_size, hop_size) → Vec<Vec<f32>>
- Returns: Vec of 12-element chroma vectors (one per frame)
- Algorithm:
  1. Compute STFT (use rustfft)
  2. For each FFT frame, convert frequency bins to semitone classes
  3. Sum magnitude across octaves for each semitone
  4. Normalize to L2 unit norm
  5. Apply soft mapping (spread to neighboring semitones for smoothness)

SEMITONE MAPPING:
- Formula: semitone = 12 * log2(freq / 440.0) + 57.0
- Then modulo 12 to get 0-11 class
- Soft mapping: spread +/- 1 semitone with linear falloff

FRAME PARAMETERS:
- frame_size: 2048 (standard for music)
- hop_size: 512 (25% overlap)

Return implementation with detailed comments on semitone math.
```

### Prompt 1.9: Key Templates & Detection

```
TASK: Implement src/features/key/templates.rs and src/features/key/detector.rs

REQUIREMENTS from Section 2.6.1-2.6.2:

Part A: templates.rs
- Struct: KeyTemplates
- Fields: major[12], minor[12] (each is Vec<f32> of 12 elements)
- Initialize with Krumhansl-Kessler profiles (hardcode the 24 profiles)
- Include documentation: which semitones are tonic/third/fifth for each key

Part B: detector.rs
- Function: detect_key(chroma_vectors, templates) → KeyDetectionResult
- Algorithm:
  1. Average chroma across all frames
  2. For each of 24 keys, compute dot product with template
  3. Find best and second-best scores
  4. Confidence = (best - second) / best
  5. Return best key and confidence

KeyDetectionResult struct:
- key: Key enum (Major(0-11) or Minor(0-11))
- confidence: 0.0-1.0
- all_scores: Vec<(Key, f32)> (sorted by score)

Return both files. Use semitone order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
```

---

## PHASE 2: ML INTEGRATION

### Prompt 2.1: ONNX Model Loading

```
TASK: Implement src/ml/onnx_model.rs

REQUIREMENTS from Section 3.1:
- Struct: OnnxModel
- Load pre-trained ONNX model from file
- Method: infer(features: &[f32]) → f32 (confidence boost factor)
- Output should be in range [0.5, 1.5]
- Error handling: If model fails to load, return confidence boost of 1.0 (no effect)

Dependencies:
- Use ort crate (ONNX Runtime)

Return implementation that gracefully degrades if model unavailable.
```

---

## COMPREHENSIVE TESTS & VALIDATION

### Prompt 3.1: Create Test Fixtures and Test Suite

```
TASK: Set up comprehensive test suite

Create:
1. tests/fixtures/ directory with synthetic test audio
2. Generate test signals:
   - Pure 120 BPM kick pattern (4-on-floor, 30 seconds)
   - C major chord loop (chroma validation)
   - Silence (edge case)
   - White noise (low-confidence case)

3. Create tests/integration_tests.rs with:
   - Full pipeline test on each fixture
   - Accuracy assertions (e.g., assert! BPM within ±2 of expected)
   - Performance benchmarks

Generate synthetic audio using pseudocode:
- Kick: 200ms impulse at 120 BPM intervals
- Chord: Sine waves at C(261), E(329), G(392) Hz, 30 sec duration
- Use wav crate to write to .wav files

Return test setup code + generated fixtures.
```

---

## PERFORMANCE & DOCUMENTATION

### Prompt 3.2: Performance Optimization & Benchmarking

```
TASK: Add performance benchmarking and optimization

Create benches/audio_analysis_bench.rs:
- Benchmark each major function:
  * extract_chroma: target <50ms for 30s audio
  * detect_energy_flux_onsets: target <20ms
  * autocorrelation: target <30ms
  * key detection: target <10ms
  
- Total pipeline: target <200ms for 30s audio

Use criterion crate:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

Profile hot paths and optimize as needed.
Return benchmark harness.
```

### Prompt 3.3: API Documentation & README

```
TASK: Write comprehensive API documentation

Deliverables:
1. docs/ALGORITHM_EXPLANATION.md
   - Explain each major step in plain English
   - Diagrams (ASCII art) showing data flow
   - Accuracy expectations per module

2. README.md
   - What this library does
   - Quickstart code example
   - Feature comparison to pro tools
   - Accuracy metrics
   - Performance metrics

3. examples/ directory
   - example_analyze_file.rs: Full pipeline on real audio file
   - example_batch_process.rs: Process 100 files

Return all documentation. README should be compelling for crates.io.
```

---

## WORKFLOW TIPS

### When Implementing a Module:

1. **Cursor first pass**: Generate implementation from prompt above
2. **You review**: Read code, check algorithm correctness
3. **Write tests**: You or Cursor write comprehensive tests
4. **Run tests**: `cargo test --lib`
5. **Bench if hot path**: `cargo bench`
6. **Document**: Cursor adds doc comments and examples

### Common Cursor Adjustments:

After first implementation, you might ask:
- "Add more numerical stability (epsilon=1e-10) throughout"
- "Add logging with the log crate at debug level"
- "Refactor to use rayon for parallelization in hot paths"
- "Add error handling instead of unwrap()"
- "Optimize this loop to use SIMD (std::simd)"

### Validation Workflow:

1. Run against synthetic data (should be perfect)
2. Run against real DJ tracks (80-85% accuracy expected in Phase 1)
3. Compare results to Rekordbox / Mixed In Key
4. Identify failure modes
5. Adjust thresholds / algorithm parameters

---

## QUICK REFERENCE: Key Dependencies

Add to Cargo.toml:

```toml
[dependencies]
symphonia = { version = "0.5", features = ["all"] }
rustfft = "6.2"
ndarray = "0.15"
log = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
env_logger = "0.11"
criterion = { version = "0.5", features = ["html_reports"] }

[features]
ml = ["ort"]  # Optional ML support

[profile.release]
opt-level = 3
lto = true
```

---

**Last Updated**: 2025-12-15  
**For use with**: Cursor AI code agent
