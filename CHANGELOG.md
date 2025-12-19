# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Detailed phase-by-phase implementation history and tuning logs live in `docs/progress-reports/`.

## [Unreleased]

### Planned
- Phase 2 ML refinement (feature-gated `ml`)
- Key detection improvements (harmonic-only chroma, better aggregation)

### Added
- `examples/analyze_batch.rs`: parallel batch processing (CPU-1 workers default)
- `docs/progress-reports/PHASE_1F_BENCHMARKS.md`: batch throughput + outlier analysis
- `CONTRIBUTING.md`: contributor guidelines and development workflow
- Validation tooling cleanup:
  - `validation/tools/` (run scripts) and `validation/analysis/` (post-run analysis)
  - `validation/_id3.py`, `validation/_keys.py`: shared ID3/key parsing utilities
  - `validation/tools/build_hllmr_metadata.py`: GT snapshot tool for real-world DJ tracks
- `archive/`: archived "construction debris" not compiled as part of the crate

### Changed
- **README.md**: Major update with validation results table, performance benchmarks, known limitations
- Documentation: top-level docs focus on the current pipeline and canonical workflows
- Defaults: HPSS percussive tempogram fallback is opt-in (avoids multi-second outliers)
- Key detection: Fixed Krumhansl-Kessler template alignment (canonical profiles + L2 normalization)
  - Minor keys now correctly detected (was previously biased toward major)
  - Key accuracy improved from 1.5% to 17.6% vs GT (further improvements in progress)

### Removed
- Unused dependencies: `ndarray`, `ndarray-linalg`
- Unimplemented public IO stubs moved out of the crate (archived under `archive/`)


