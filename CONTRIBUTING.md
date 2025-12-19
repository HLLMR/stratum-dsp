# Contributing to Stratum DSP

Thank you for your interest in contributing to Stratum DSP! This document outlines how to get started, run tests, and submit changes.

## Development Setup

### Prerequisites

- **Rust** (stable toolchain, 1.70+)
- **Python 3.x** (for validation tooling)
- **Git**

### Building

```bash
# Debug build
cargo build

# Release build (for benchmarks/validation)
cargo build --release

# Build example CLIs
cargo build --release --example analyze_file
cargo build --release --example analyze_batch
```

### Running Tests

```bash
# Unit tests
cargo test

# Integration tests (requires test fixtures)
cargo test --test integration_tests

# Run with output
cargo test -- --nocapture
```

### Running Benchmarks

```bash
cargo bench --bench audio_analysis_bench
```

Results are saved to `target/criterion/` with HTML reports.

## Project Structure

- **`src/lib.rs`**: Main entry point (`analyze_audio` function)
- **`src/config.rs`**: Configuration defaults and tuning knobs
- **`src/features/`**: Core algorithms (onset, tempo, beat, key)
- **`src/preprocessing/`**: Normalization and silence trimming
- **`src/analysis/`**: Result types and confidence scoring
- **`examples/`**: CLI tools (single-file and batch processing)
- **`validation/`**: Python validation harness (FMA Small dataset)
- **`docs/`**: Literature reviews, progress reports, pipeline docs
- **`tests/`**: Integration tests with fixtures

## Code Style

- Follow Rust standard formatting: `cargo fmt`
- Run clippy: `cargo clippy`
- Document public APIs with `///` doc comments
- Use `log::debug!` for diagnostic output (not `println!`)

## Adding New Features

### Algorithm Changes

1. **Document the reference**: Add a literature review in `docs/literature/` if introducing a new algorithm
2. **Update `PIPELINE.md`**: Document where your change fits in the runtime flow
3. **Add tests**: Unit tests for the algorithm, integration tests for end-to-end behavior
4. **Benchmark**: Add a Criterion benchmark if the change affects performance
5. **Validate**: Run validation on FMA Small or hllmr batch to check for regressions

### Configuration Options

- Add new options to `AnalysisConfig` in `src/config.rs`
- Document defaults and rationale
- Make options tunable via CLI flags in `examples/analyze_file.rs` if relevant

## Validation Workflow

To validate changes against ground truth:

```bash
# 1. Build release binary
cargo build --release --example analyze_file

# 2. Prepare test batch (FMA Small)
python -m validation.tools.prepare_test_batch --num-tracks 50

# 3. Run validation
python -m validation.tools.run_validation

# 4. Analyze results
python -m validation.analysis.analyze_results
```

See `validation/README.md` for detailed validation documentation.

## Submitting Changes

1. **Fork the repository** (if external contributor)
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the guidelines above
4. **Run tests**: `cargo test` and validation if applicable
5. **Update documentation**: README, PIPELINE.md, or progress reports as needed
6. **Commit**: Use clear commit messages
7. **Push and open a PR**: Describe what changed and why

### Commit Message Format

- Use present tense: "Add X" not "Added X"
- First line should be a short summary (<72 chars)
- Optional body explaining the "why" for complex changes

Examples:
- `feat: add HPSS percussive-only tempogram fallback`
- `fix: correct minor key template alignment in K-K profiles`
- `docs: update README with validation results table`

## Current Focus Areas

### Phase 1 (DSP-first)

- **BPM accuracy**: Target ≥88% within ±2 BPM (currently 87.7% on hllmr batch)
- **Key accuracy**: Target ≥77% exact match (currently 17.6%, improvements in progress)
- **Performance**: Maintain fast single-track processing + high batch throughput

### Phase 2 (Future)

- Optional ML refinement (feature-gated `ml`)
- GPU acceleration (if needed for ML runtime)

## Questions?

- Check `DEVELOPMENT.md` for build/test/bench workflows
- Check `PIPELINE.md` for algorithm details and runtime flow
- Check `docs/progress-reports/` for implementation history and tuning notes
- Open an issue for questions or discussion

