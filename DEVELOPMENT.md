# Stratum DSP - Development Guide

This document is intentionally **workflow-focused** and describes how to build, test, benchmark, and validate the codebase.

For algorithm details and runtime branching, use:
- `PIPELINE.md` (authoritative runtime flow)
- `docs/literature/` (paper summaries)
- `docs/progress-reports/` (implementation + validation logs)

---

## Prerequisites

- Rust (stable toolchain)
- Python 3.x (validation tooling)

---

## Build / test / bench

```bash
cargo build
cargo test
cargo bench --bench audio_analysis_bench
```

---

## CLI examples

### Analyze a single file

```bash
cargo build --release --example analyze_file
target/release/examples/analyze_file --json <audio_file>
```

### Analyze a batch (parallel, CPU-1 workers)

```bash
cargo build --release --example analyze_batch
target/release/examples/analyze_batch --jobs 7 <file1> <file2> ...
```

---

## Validation (FMA Small)

Setup and directory layout: `validation/README.md`.

Canonical workflow (repo root):

```bash
python -m validation.tools.prepare_test_batch --num-tracks 200
python -m validation.tools.run_validation --jobs 15
python -m validation.analysis.analyze_results
```

---

## Where things live

- `src/lib.rs`: pipeline orchestration (`analyze_audio`)
- `src/config.rs`: defaults + tuning knobs
- `src/features/`: onset/tempo/beat/key implementations
- `validation/tools/`: scripts that produce artifacts (batches/results/sweeps)
- `validation/analysis/`: scripts that interpret artifacts
- `docs/progress-reports/`: phase reports + detailed run history
- `archive/`: intentionally archived “construction debris” (not compiled)

---

## Conventions

- Keep the core crate API **sample-based**; file decoding belongs in example binaries.
- Prefer clean, cited implementations; keep experiment history in progress reports.
- Key notation: support musical names + **numerical** (`1A/1B`) without trademarked naming.


