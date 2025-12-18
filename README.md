# Stratum DSP

Stratum DSP is a Rust audio analysis engine aimed at DJ/library workflows:

- **Tempo (BPM)** + confidence
- **Key** (musical name + **numerical** DJ notation like `1A/1B`)
- **Beat grid** + stability

The library API is sample-based (`analyze_audio(samples, sample_rate, ...)`). File decoding is provided via the example CLIs.

## What’s implemented today

- **BPM**: novelty curve → dual tempogram (FFT + autocorrelation) with metrical-family selection and optional multi-resolution escalation.
  - Primary reference: Grosche et al. (2012) Fourier tempogram (see `docs/literature/grosche_2012_tempogram.md`).
- **Key**: chroma → template matching (Krumhansl & Kessler, 1982).
- **Preprocessing**: Peak/RMS/LUFS normalization (ITU-R BS.1770-4) + silence trimming.
- **Beat grid**: HMM-based beat tracking + stability.
- **Validation tooling**: FMA Small harness under `validation/`.

## Quick start (library)

Add to your `Cargo.toml`:

```toml
[dependencies]
stratum-dsp = { git = "https://github.com/HLLMR/stratum-dsp" }
```

Example:

```rust
use stratum_dsp::{analyze_audio, compute_confidence, AnalysisConfig};

let samples: Vec<f32> = vec![]; // mono, f32 in [-1, 1]
let sample_rate = 44_100;

let result = analyze_audio(&samples, sample_rate, AnalysisConfig::default())?;
let conf = compute_confidence(&result);

println!("BPM: {:.2} (conf={:.2})", result.bpm, conf.bpm_confidence);
println!("Key: {} (conf={:.2})", result.key.name(), conf.key_confidence);
# Ok::<(), stratum_dsp::AnalysisError>(())
```

## Example CLIs

- **Single file**: `examples/analyze_file.rs`

```bash
cargo build --release --example analyze_file
target/release/examples/analyze_file --json <audio_file>
```

- **Batch (parallel, CPU-1 workers)**: `examples/analyze_batch.rs`

```bash
cargo build --release --example analyze_batch
target/release/examples/analyze_batch --jobs 7 <file1> <file2> ...
```

## Validation (FMA Small)

See `validation/README.md`. Canonical commands:

```bash
python -m validation.tools.prepare_test_batch --num-tracks 200
python -m validation.tools.run_validation --jobs 15
python -m validation.analysis.analyze_results
```

## Documentation map

- **Pipeline (authoritative)**: `PIPELINE.md`
- **Dev workflow (concise)**: `DEVELOPMENT.md`
- **Roadmap (high-level only)**: `ROADMAP.md`
- **Progress reports / deep history**: `docs/progress-reports/`

## License

Dual-licensed under MIT OR Apache-2.0.

