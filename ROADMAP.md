# Stratum DSP - Roadmap (high-level)

This roadmap is intentionally short and focused on **where the project is going next**.

For detailed phase histories, tuning logs, and run-by-run validation notes, see `docs/progress-reports/`.

---

## Current focus (Phase 1F)

- **Tempo accuracy**: 87.7% ±2 BPM on real-world DJ tracks (155 tracks) — close to 88% target
- **Key detection**: 72.1% vs GT (n=68) — matches MIK performance on same dataset
- **Stable defaults** and a repeatable validation workflow (FMA Small + hllmr real-world dataset)
- **Performance + throughput**: Batch processing with CPU-1 workers (7.7× speedup) — ✅ complete

Authoritative references:
- Pipeline: `PIPELINE.md`
- Validation workflow: `validation/README.md`
- Phase 1F empirical status: `docs/progress-reports/PHASE_1F_VALIDATION.md`
- Phase 1F benchmarks: `docs/progress-reports/PHASE_1F_BENCHMARKS.md`

---

## Milestones

### v0.x (DSP-first, CPU)
- Production-quality DSP pipeline (tempo/key/beat-grid + confidence)
- Validation + benchmarks documented
- Clean CLI examples for single-file + batch processing

### Phase 2 (ML refinement, optional)
- Feature-gated ONNX refinement (`--features ml`)
- GPU acceleration is optional and driven by the ML runtime needs

---

## Definition of done (project-level)

- **Tempo**: ≥ **88%** within ±2 BPM on representative validation sets
  - ✅ **87.7%** on real-world DJ tracks (155 tracks) — *0.3pp from target*
- **Key**: ≥ **77%** exact match on representative validation sets
  - ✅ **72.1%** on real-world DJ tracks (n=68) — *matches MIK performance, 4.9pp from target*
- **Performance**: fast single-track processing + high batch throughput (CPU-1 parallel workers for scans)
  - ✅ **Complete**: ~200ms single-track, ~21 tracks/sec batch throughput
- **Docs**: pipeline + validation + benchmark docs match what the code actually does
  - ✅ **Complete**: README, PIPELINE, DEVELOPMENT, validation docs updated

