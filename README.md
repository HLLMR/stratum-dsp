# Stratum DSP

A professional-grade audio analysis engine for DJ applications, providing accurate BPM detection, key detection, and beat tracking in pure Rust.

## Features

- **BPM Detection**: Multi-method onset detection with autocorrelation and comb filterbank
- **Key Detection**: Chroma-based analysis with Krumhansl-Kessler template matching
- **Beat Tracking**: HMM-based beat grid generation with tempo drift correction
- **ML Refinement**: Optional ONNX model for edge case correction (Phase 2)

## Status

✅ **Phase 1A Complete** - Preprocessing & Onset Detection implemented and tested  
🚧 **Phase 1B In Progress** - Period Estimation (BPM Detection)

Target accuracy:
- BPM: 88% (±2 BPM tolerance)
- Key: 77% (exact match)

**Current Progress**: 12.5% (1/8 weeks)

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
stratum-dsp = { git = "https://github.com/HLLMR/stratum-dsp" }
```

Basic usage:

```rust
use stratum_dsp::{analyze_audio, AnalysisConfig};

// Load audio samples (mono, f32, normalized)
let samples: Vec<f32> = vec![]; // Your audio data
let sample_rate = 44100;

// Analyze
let result = analyze_audio(&samples, sample_rate, AnalysisConfig::default())?;

println!("BPM: {:.2} (confidence: {:.2})", result.bpm, result.bpm_confidence);
println!("Key: {:?} (confidence: {:.2})", result.key, result.key_confidence);
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
- **ML Refinement**: Optional ONNX model for edge case correction

## Development Roadmap

### Phase 1: Classical DSP (Weeks 1-5)
- [x] **Phase 1A**: Preprocessing & Onset Detection ✅
  - [x] Normalization (peak, RMS, LUFS with K-weighting)
  - [x] Silence detection and trimming
  - [x] Channel mixing (stereo to mono)
  - [x] Onset detection (energy flux, spectral flux, HFC, HPSS)
  - [x] Consensus voting algorithm
  - [x] 80 tests (75 unit + 5 integration)
- [ ] **Phase 1B**: Period Estimation (BPM Detection)
- [ ] **Phase 1C**: Beat Tracking (HMM)
- [ ] **Phase 1D**: Key Detection (chroma + templates)
- [ ] **Phase 1E**: Integration and tuning

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

