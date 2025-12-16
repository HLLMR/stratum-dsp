# Gkiokas et al. (2012): Dimensionality Reduction for BPM Estimation

**Full Citation**: Gkiokas, A., Katsouros, V., & Carayannis, G. (2012). Dimensionality Reduction for BPM Estimation. *IEEE Transactions on Audio, Speech, and Language Processing*.

**Category**: Beat Tracking & Tempo Estimation  
**Relevance**: Comb filterbank approach for robust BPM estimation

---

## Summary

This paper presents an efficient BPM estimation method using dimensionality reduction and a comb filterbank approach. It tests hypothesis tempos by counting onsets aligned with expected beat times, providing robustness when autocorrelation fails.

## Key Contributions

1. **Comb Filterbank Approach**:
   - Test candidate BPM values (typically 80-180 BPM)
   - For each candidate, compute expected beat intervals
   - Score by counting onsets aligned with expected beats

2. **Dimensionality Reduction**:
   - Efficient computation through dimensionality reduction
   - Performance optimization techniques
   - Scalable to large onset lists

3. **Robustness**:
   - Works when autocorrelation fails (noisy, sparse onsets)
   - Handles variable tempo better than autocorrelation
   - Provides alternative to autocorrelation method

## Relevance to Stratum DSP

This paper informs our comb filterbank implementation:

### Comb Filterbank Period Estimation
- **Source**: Core algorithm from this paper
- **Location**: `src/features/period/comb_filter.rs`
- **Algorithm**: Test hypothesis tempos → score by onset alignment → select best

### Dual-Method Approach
- **Autocorrelation**: Fast, finds periodicity directly (from Ellis & Pikrakis 2006)
- **Comb Filterbank**: Robust, tests hypotheses (from this paper)
- **Merge**: Combine results to handle octave errors and boost confidence

## Key Algorithms

### Comb Filterbank BPM Estimation

**Step 1: Generate Candidate BPMs**
```
candidates = [80, 81, 82, ..., 180]  // 1 BPM resolution
// Or: [80, 82, 84, ..., 180]  // 2 BPM resolution for speed
```

**Step 2: For Each Candidate BPM**
```
period_samples = (60 * sample_rate) / (BPM * hop_size)
expected_beats = [0, period, 2*period, 3*period, ...]
```

**Step 3: Score by Onset Alignment**
```
score = 0
for each onset:
    distance_to_nearest_beat = min(|onset - expected_beat| for all expected_beats)
    if distance_to_nearest_beat < tolerance:
        score += 1

normalized_score = score / total_onsets
```

**Step 4: Select Best BPM**
```
best_BPM = argmax(normalized_score)
confidence = normalized_score[best] / max(normalized_score[others])
```

### Tolerance Window

Typical tolerance: ±10% of beat interval
```
tolerance = 0.1 * period_samples
```

## Implementation Notes

### Candidate BPM Generation
- **Resolution**: 1 BPM (80-180) = 101 candidates
- **Alternative**: 2 BPM resolution for speed (50 candidates)
- **Range**: Adjustable based on genre (e.g., 60-200 for broader coverage)

### Onset Alignment Scoring
- **Tolerance**: 10% of beat interval (typical)
- **Normalization**: Divide by total onset count
- **Weighting**: Optional weighting by onset strength/confidence

### Performance Optimization
- **Early Termination**: Stop if score is clearly best
- **Coarse-to-Fine**: Test 2 BPM resolution first, refine around best
- **Parallelization**: Test candidates in parallel (rayon)

### Dimensionality Reduction
- **Onset Clustering**: Group nearby onsets to reduce computation
- **Temporal Binning**: Bin onsets into frames
- **Sparse Representation**: Only test candidates near autocorrelation result

## Performance Characteristics

**Accuracy**:
- **Synthetic Data**: 90-95% accuracy
- **Real Music**: 70-80% accuracy (±2 BPM tolerance)
- **Robustness**: Better than autocorrelation on noisy/sparse onsets

**Speed**:
- **Naive**: O(n * m) where n=onsets, m=candidates
- **Optimized**: O(n * log(m)) with early termination
- **Typical Performance**: 10-30ms for 30s track (101 candidates)

**Strengths**:
- Robust to octave errors (tests actual BPM values)
- Works with sparse/noisy onset lists
- Handles variable tempo better than autocorrelation
- Provides confidence scores

**Weaknesses**:
- Slower than autocorrelation (tests many candidates)
- Requires sufficient onsets for reliable scoring
- Can be sensitive to tolerance window size

## Comparison with Autocorrelation

| Aspect | Autocorrelation | Comb Filterbank |
|--------|----------------|-----------------|
| Speed | Fast (O(n log n)) | Slower (O(n*m)) |
| Octave Errors | Prone | Robust |
| Sparse Onsets | Struggles | Better |
| Variable Tempo | Assumes constant | More flexible |
| Confidence | Indirect | Direct (score) |

**Best Approach**: Combine both methods (our implementation)

## Tolerance Window Selection

**Too Small** (<5%):
- Misses onsets with timing jitter
- Lower scores, less robust

**Too Large** (>15%):
- Accepts false positives
- Less discriminative between candidates

**Optimal** (10%):
- Balances robustness and discrimination
- Works well across genres

## References in Code

- `src/features/period/comb_filter.rs`: Comb filterbank implementation
- `src/features/period/candidate_filter.rs`: Merging autocorrelation and comb results
- `src/features/period/peak_picking.rs`: Finding best candidate from scores

## Additional Notes

- Provides robustness when autocorrelation fails
- Complements autocorrelation (different strengths)
- Can be optimized with dimensionality reduction
- Direct confidence scores useful for quality assessment

---

**Last Updated**: 2025-01-XX  
**Status**: Core reference for comb filterbank BPM estimation

