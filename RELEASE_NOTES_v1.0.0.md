# Stratum DSP v1.0.0 - Production Release

**Release Date**: December 18, 2025

## 🎉 Production Ready

Stratum DSP v1.0.0 is the first stable release of a professional-grade audio analysis engine for DJ applications. Built in pure Rust with zero FFI dependencies, it delivers production-ready BPM detection, key detection, and beat tracking.

## 📊 Performance Highlights

### BPM Detection
- **87.7% accuracy** (±2 BPM) on 155 verified Beatport/ZipDJ tracks
- **6.08 BPM MAE** (Mean Absolute Error)
- Dual tempogram approach (FFT + autocorrelation) with multi-resolution escalation

### Key Detection
- **72.1% accuracy** (exact match vs ground truth)
- **Matches Mixed In Key performance** on the same dataset
- Krumhansl-Kessler template matching with HPSS preprocessing

### Performance
- **~200ms** per 3-minute track
- **21 tracks/sec** batch throughput (7.7× speedup with parallel processing)

## ✨ Core Features

- **Tempogram BPM** (Grosche et al. 2012): FFT + autocorrelation tempogram with multi-resolution escalation
- **Krumhansl-Kessler key detection**: Chroma-based analysis with circle-of-fifths weighting
- **HMM-based beat tracking**: Beat grid generation with stability scoring
- **Confidence scoring**: Comprehensive confidence metrics for all analysis components
- **Parallel batch processing**: CPU-1 workers default for high-throughput library scans
- **Preprocessing**: Peak/RMS/LUFS normalization (ITU-R BS.1770-4) + silence trimming

## 📚 Documentation

- Complete pipeline documentation (`PIPELINE.md`)
- Validation reports and benchmarks
- Literature reviews for all algorithms
- Development and contribution guides

## 🔗 Links

- **Crates.io**: https://crates.io/crates/stratum-dsp
- **Documentation**: https://docs.rs/stratum-dsp
- **Repository**: https://github.com/HLLMR/stratum-dsp

## 📦 Installation

```toml
[dependencies]
stratum-dsp = "1.0"
```

## 🚀 Quick Start

```rust
use stratum_dsp::{analyze_audio, AnalysisConfig};

let result = analyze_audio(&samples, 44100, AnalysisConfig::default())?;
println!("BPM: {:.1} | Key: {} ({})", 
    result.bpm, 
    result.key.name(), 
    result.key.numerical()
);
```

## 🎯 Validation

Validated on 155 real-world DJ tracks (Beatport, ZipDJ) with verified ground truth. Full validation results documented in `docs/progress-reports/PHASE_1F_VALIDATION.md`.

**Reference baseline**: Mixed In Key achieves 98.1% ±2 BPM and 72.1% key accuracy on the same dataset.

## 🙏 Acknowledgments

Built with academic rigor, implementing algorithms from:
- Grosche, P., Müller, M., & Kurth, F. (2012) - Tempogram BPM
- Krumhansl, C. L., & Kessler, E. J. (1982) - Key detection
- Ellis, D. P. W. (2007) - Beat tracking
- And many more (see `docs/literature/`)

---

**Full Changelog**: See [CHANGELOG.md](CHANGELOG.md)

