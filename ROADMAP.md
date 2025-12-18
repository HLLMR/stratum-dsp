# Stratum DSP - Roadmap (high-level)

This roadmap is intentionally short and focused on **where the project is going next**.

For detailed phase histories, tuning logs, and run-by-run validation notes, see `docs/progress-reports/`.

---

## Current focus (Phase 1F)

- **Tempo accuracy + validation** for the current Phase 1F tempogram-based BPM estimator.
- **Stable defaults** and a repeatable validation workflow (FMA Small).
- **Performance + throughput** for batch library scans (CPU-only).

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

- **Tempo**: ≥ **88%** within ±2 BPM on representative validation sets (FMA Small baseline + additional batches)
- **Key**: ≥ **77%** exact match on representative validation sets
- **Performance**: fast single-track processing + high batch throughput (CPU-1 parallel workers for scans)
- **Docs**: pipeline + validation + benchmark docs match what the code actually does

