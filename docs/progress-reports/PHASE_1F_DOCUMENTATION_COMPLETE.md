# Phase 1F: Tempogram Pivot - Documentation Alignment (Post-Sprint)

**Date**: 2025-12-17  
**Status**: ✅ Documentation Updated and Aligned  
**Implementation Status**: ⚠️ Implemented; tuning/validation in progress (baseline stable)

---

## Summary

This sprint included significant implementation + experimentation around Phase 1F tuning, validation tooling, onset consensus wiring, and BPM fusion/legacy A/B modes. This document records the **documentation alignment** work performed afterward so the project remains readable and traceable.

---

## Current Status (Post-Sprint)

Phase 1F is implemented and integrated. Validation tooling has been expanded (A/B modes, ablation toggles, multi-file analysis). Accuracy remains **below target**, but the baseline is stable and fully reproducible.

**See**:
- `docs/progress-reports/PHASE_1F_VALIDATION.md` (run history + current baseline + failure modes)
- `docs/progress-reports/PHASE_1F_LITERATURE_REVIEW.md` (Phase 1F literature review)
- `PIPELINE.md` (authoritative end-to-end pipeline + decision points)

---

## Documentation Updates (this sprint)

### 1) Authoritative pipeline reference
- ✅ `PIPELINE.md`
  - Updated to reflect current runtime wiring and decision points
  - Includes debugging/log guidance

### 2) Validation workflow documentation
- ✅ `validation/README.md`
  - Updated to document current validation flags and analysis scripts

### 3) Phase 1F progress reports updated
- ✅ `PHASE_1F_VALIDATION.md`
  - Updated run history and baseline results
  - Includes A/B modes used during tuning
- ✅ `PHASE_1F_COMPLETE.md`
  - Updated to include post-implementation wiring & tooling changes

### 4) Root docs alignment
- ✅ `README.md`, `ROADMAP.md`, `DEVELOPMENT.md`, `CHANGELOG.md`
  - Updated references and tooling details so readers can reproduce the current state

### 5. Project README
- ✅ `README.md` - Project status
  - Updated features section
  - Updated status section
  - Updated roadmap section
  - Added deprecation notes

### 6. Summary Document
- ✅ `TEMPOGRAM_PIVOT_SUMMARY.md` - Cross-reference
  - Updated implementation plan
  - Added deprecation timeline
  - Updated file structure

---

## Implementation Strategy

### Dual Tempogram Approach

**Phase 2A: Autocorrelation Tempogram**
- Test each BPM hypothesis (40-240, 0.5 BPM resolution)
- Direct periodicity testing
- Expected: 75-85% accuracy, 20-40ms

**Phase 2B: FFT Tempogram**
- FFT the novelty curve
- Convert frequencies to BPM
- Expected: 75-85% accuracy, 10-20ms

**Phase 2C: Comparison & Selection**
- Run both methods
- Compare results
- Use best or ensemble
- Expected: 85-92% accuracy, 30-60ms

### Hybrid Approach (Future)

Documented for future implementation:
- FFT: Fast coarse estimate (2 BPM resolution)
- Autocorr: Precise fine estimate (±5 BPM around FFT, 0.5 BPM resolution)
- Benefits: Speed + Precision
- Status: Documented, not implemented yet

---

## Deprecation Plan

### Old Methods (Phase 1B)

**Files to Deprecate**:
- `src/features/period/autocorrelation.rs`
- `src/features/period/comb_filter.rs`
- `src/features/period/candidate_filter.rs`

**Timeline**:
1. **Phase 1F**: Keep active for A/B comparison
2. **After Validation**: Mark as `#[deprecated]`
3. **v0.9.2**: Remove entirely

**Rationale**: Old methods are fundamentally broken (30% accuracy). Once tempogram is validated, no reason to keep them.

---

## File Structure

### New Files (To Be Created)
- `src/features/period/novelty.rs` - Novelty curve extraction
- `src/features/period/tempogram_autocorr.rs` - Autocorrelation tempogram
- `src/features/period/tempogram_fft.rs` - FFT tempogram
- `src/features/period/tempogram.rs` - Main entry point (comparison)
- `src/features/period/multi_resolution.rs` - Multi-resolution validation

### Files to Update
- `src/features/period/mod.rs` - Add tempogram methods
- `src/lib.rs` - Update pipeline

### Files to Deprecate (After Validation)
- `src/features/period/autocorrelation.rs`
- `src/features/period/comb_filter.rs`
- `src/features/period/candidate_filter.rs`

---

## Expected Results

| Metric | Current | Tempogram (Single) | Tempogram (Dual) |
|--------|---------|-------------------|------------------|
| Accuracy (±2 BPM) | ~20% | 70-80% | 80%+ |
| Accuracy (±5 BPM) | 30% | 75-85% | 85-92% |
| Subharmonic Errors | 10-15% | 2-3% | <1% |
| MAE | 34 BPM | 4-6 BPM | 3-4 BPM |
| Speed | 15-45ms | 20-40ms | 30-60ms |

---

## Implementation Timeline

- **Phase 1**: Novelty curve (20 min)
- **Phase 2A**: Autocorrelation tempogram (40 min)
- **Phase 2B**: FFT tempogram (30 min)
- **Phase 2C**: Comparison & selection (20 min)
- **Phase 3**: Hybrid approach documented (5 min)
- **Phase 4**: Multi-resolution (20 min)
- **Phase 5**: Integration (20 min)
- **Phase 6**: Testing & validation (40 min)

**Total: 3-4 hours**

---

## Consistency Verification

### ✅ All Documentation Aligned

- **Accuracy Claims**: 30% → 85-92% (consistent)
- **Timeline**: 3-4 hours (consistent)
- **Methods**: Both FFT and autocorr (consistent)
- **Deprecation**: Plan documented (consistent)
- **Hybrid**: Documented for future (consistent)

### ✅ Cross-References Verified

- ROADMAP → EVALUATION ✓
- DEVELOPMENT → EVALUATION ✓
- README → All docs ✓
- EVALUATION → HYBRID ✓
- SUMMARY → All docs ✓

---

## Next Steps

Documentation work is caught up. Next engineering effort should focus on:
- Improving Phase 1F metrical-level / harmonic-family selection without regressions
- Expanding validation beyond a single small batch (multiple batches, aggregate metrics)
- Keeping A/B modes (legacy-only, ablations) for regression testing until Phase 1F is validated

---

**Status**: ✅ **ALL DOCUMENTATION COMPLETE AND ALIGNED**

**Ready for Implementation**: Yes

**Recommendation**: Proceed with dual tempogram implementation following the technical specification.

---

**Last Updated**: 2025-01-XX  
**Next Action**: Begin Phase 1 implementation (Novelty curve)

