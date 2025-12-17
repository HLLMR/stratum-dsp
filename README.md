# Stratum DSP

A professional-grade audio analysis engine for DJ applications, providing accurate BPM detection, key detection, and beat tracking in pure Rust.

## Features

- **BPM Detection**: Multi-method onset detection with autocorrelation and comb filterbank
- **Key Detection**: Chroma-based analysis with Krumhansl-Kessler template matching
- **Beat Tracking**: HMM-based beat grid generation with tempo drift correction
- **ML Refinement**: Optional ONNX model for edge case correction (Phase 2)

## Status

✅ **Phase 1A Complete** - Preprocessing & Onset Detection implemented and tested  
✅ **Phase 1B Complete** - Period Estimation (BPM Detection) implemented and tested  
✅ **Phase 1C Complete** - Beat Tracking (HMM Viterbi) implemented and tested  
✅ **Phase 1D Complete** - Key Detection (Chroma + Templates) implemented and tested  
✅ **Phase 1E Complete** - Integration & Tuning with confidence scoring

Target accuracy:
- BPM: 88% (±2 BPM tolerance)
- Key: 77% (exact match)

**Current Progress**: 62.5% (5/8 weeks) - Classical DSP pipeline complete, ready for Phase 2 ML refinement

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
stratum-dsp = { git = "https://github.com/HLLMR/stratum-dsp" }
```

Basic usage:

```rust
use stratum_dsp::{analyze_audio, AnalysisConfig, compute_confidence};

// Load audio samples (mono, f32, normalized)
let samples: Vec<f32> = vec![]; // Your audio data
let sample_rate = 44100;

// Analyze
let result = analyze_audio(&samples, sample_rate, AnalysisConfig::default())?;

// Compute comprehensive confidence scores
let confidence = compute_confidence(&result);

println!("BPM: {:.2} (confidence: {:.2})", result.bpm, confidence.bpm_confidence);
println!("Key: {} (confidence: {:.2}, clarity: {:.2})", 
         result.key.name(), 
         confidence.key_confidence,
         result.key_clarity);
println!("Overall confidence: {:.2} ({})", 
         confidence.overall_confidence, 
         confidence.confidence_level());
# Ok::<(), stratum_dsp::AnalysisError>(())
```

## Architecture

The analysis pipeline follows this flow:

```
Audio Input → Preprocessing → Feature Extraction → Analysis → ML Refinement → Output
```

### Modules

- **Preprocessing**: Normalization, silence detection, channel mixing
- **Onset Detection**: Energy flux, spectral flux, HFC, HPSS with consensus voting
- **Period Estimation**: Autocorrelation + comb filterbank BPM estimation
- **Beat Tracking**: HMM Viterbi algorithm + Bayesian tempo tracking
- **Chroma Extraction**: FFT → 12-semitone chroma vectors
- **Key Detection**: Krumhansl-Kessler template matching (24 keys)
- **Confidence Scoring**: Comprehensive confidence computation for all components
- **ML Refinement**: Optional ONNX model for edge case correction (Phase 2)

## Development Roadmap

### Phase 1: Classical DSP (Weeks 1-5)
- [x] **Phase 1A**: Preprocessing & Onset Detection ✅
  - [x] Normalization (peak, RMS, LUFS with K-weighting)
  - [x] Silence detection and trimming
  - [x] Channel mixing (stereo to mono)
  - [x] Onset detection (energy flux, spectral flux, HFC, HPSS)
  - [x] Consensus voting algorithm
  - [x] 80 tests (75 unit + 5 integration)
- [x] **Phase 1B**: Period Estimation (BPM Detection) ✅
  - [x] Autocorrelation-based BPM estimation (FFT-accelerated)
  - [x] Comb filterbank BPM estimation
  - [x] Peak picking and candidate filtering
  - [x] Octave error handling
  - [x] Coarse-to-fine search optimization (5-15ms vs 10-30ms)
  - [x] Adaptive tolerance window (BPM-dependent)
  - [x] 32 unit tests + integration tests
- [x] **Phase 1C**: Beat Tracking (HMM) ✅
  - [x] HMM Viterbi algorithm for beat sequence tracking
  - [x] Bayesian tempo tracking for variable-tempo tracks
  - [x] Variable tempo detection and automatic refinement
  - [x] Time signature detection (4/4, 3/4, 6/8)
  - [x] Beat grid generation with downbeat detection
  - [x] Grid stability calculation
  - [x] 44 unit tests + integration tests with <50ms jitter validation
- [x] **Phase 1D**: Key Detection (Chroma + Templates) ✅
  - [x] STFT-based chroma extraction with soft mapping
  - [x] Chroma normalization (L2 normalization, sharpening)
  - [x] Temporal chroma smoothing (median and average filtering)
  - [x] Krumhansl-Kessler templates (24 keys: 12 major + 12 minor)
  - [x] Template matching algorithm with confidence scoring
  - [x] Key clarity computation (tonal strength estimation)
  - [x] Key change detection (segment-based analysis)
  - [x] Musical notation display (e.g., "C", "Am", "F#", "D#m")
  - [x] DJ standard numerical format (1A, 2B, etc.) without trademarked names
  - [x] 40 unit tests + integration tests with known key fixtures
  - [x] Performance: ~17-28ms for 30s track (2x faster than target)
- [x] **Phase 1E**: Integration & Tuning ✅
  - [x] Comprehensive confidence scoring system
  - [x] Result aggregation and error handling refinement
  - [x] Full pipeline integration in `analyze_audio()`
  - [x] Key clarity added to `AnalysisResult`
  - [x] Confidence helper methods (`is_high_confidence()`, etc.)
  - [x] 8 new confidence tests + all existing tests passing (219+ total)
  - [x] Performance: ~75-150ms for 30s track (3-6x faster than 500ms target)

### Phase 2: ML Refinement (Weeks 6-8)
- [ ] Data collection (1000+ tracks)
- [ ] ONNX model training
- [ ] ML inference integration
- [ ] Accuracy validation

### Phase 3: Release (Week 8)
- [ ] Documentation
- [ ] Performance optimization
- [ ] Publish to crates.io

## Performance Targets

- **Speed**: <500ms per 30s track (single-threaded)
- **Accuracy**: 88% BPM, 77% key detection
- **Memory**: Efficient streaming processing

## License

Dual-licensed under MIT OR Apache-2.0

## Contributing

Contributions welcome! See [DEVELOPMENT.md](DEVELOPMENT.md) for comprehensive development guidelines, algorithms, and implementation details.

