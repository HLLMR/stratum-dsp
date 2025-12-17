# Hybrid Tempogram Approach: Future Enhancement

**Date**: 2025-01-XX  
**Status**: Documented for Future Implementation  
**Purpose**: Hybrid approach combining FFT (fast, coarse) + Autocorrelation (precise, fine)

---

## Overview

The hybrid approach combines the speed of FFT tempogram with the precision of autocorrelation tempogram. This is documented as a future enhancement to be implemented after empirical comparison of the two independent methods.

## Algorithm

### Phase 1: FFT Tempogram (Fast, Coarse)

```rust
// Step 1: FFT the novelty curve
let fft_output = compute_fft(&novelty_curve);

// Step 2: Convert frequencies to BPM
let bpms = frequencies_to_bpm(&fft_output, hop_time);

// Step 3: Find approximate BPM (2 BPM resolution)
let coarse_bpm = find_peak_in_range(&bpms, min_bpm, max_bpm);
```

**Result**: Approximate BPM with ~2 BPM resolution (fast, O(n log n))

### Phase 2: Autocorrelation Refinement (Precise, Fine)

```rust
// Step 1: Define refinement window around FFT estimate
let refinement_window = 5.0; // ±5 BPM
let min_refine = (coarse_bpm - refinement_window).max(min_bpm);
let max_refine = (coarse_bpm + refinement_window).min(max_bpm);

// Step 2: Test BPM candidates in refinement window with fine resolution
let fine_bpm = autocorrelation_tempogram(
    &novelty_curve,
    sample_rate,
    hop_size,
    min_refine,
    max_refine,
    0.5, // Fine resolution: 0.5 BPM
);
```

**Result**: Precise BPM with 0.5 BPM resolution (slower, but only tests ~20 candidates)

## Benefits

1. **Speed**: FFT narrows search space (400 candidates → ~20 candidates)
2. **Precision**: Autocorrelation refines to 0.5 BPM resolution
3. **Best of Both**: Consistency (FFT) + Precision (autocorr)
4. **Efficiency**: ~15-25ms total (vs 30-60ms for both methods independently)

## Expected Performance

- **Speed**: 15-25ms for 30s track (vs 30-60ms for both methods)
- **Accuracy**: 85-92% (±5 BPM tolerance)
- **Resolution**: 0.5 BPM (fine-grained)

## Implementation Notes

### When to Implement

- **After**: Empirical comparison of FFT vs autocorrelation tempogram
- **If**: Both methods show similar accuracy, hybrid provides speed optimization
- **If**: One method clearly superior, hybrid may not be needed

### Optimization Opportunities

1. **Early Termination**: If FFT confidence is very high, skip refinement
2. **Adaptive Window**: Adjust refinement window based on FFT confidence
3. **Parallel Execution**: Run FFT and autocorr in parallel, use FFT result to narrow autocorr search

## Comparison with Independent Methods

| Aspect | FFT Only | Autocorr Only | Hybrid |
|--------|----------|---------------|--------|
| Speed | 10-20ms | 20-40ms | 15-25ms |
| Resolution | ~2 BPM | 0.5 BPM | 0.5 BPM |
| Accuracy | 75-85% | 75-85% | 85-92% |
| Consistency | High | Medium | High |

## Status

**Current**: Documented for future implementation  
**Priority**: Low (after empirical comparison)  
**Dependencies**: FFT tempogram + Autocorrelation tempogram must be implemented first

---

**Last Updated**: 2025-01-XX  
**Next Action**: Implement after Phase 1F validation and comparison

