# McFee & Ellis (2014): Better Beat Tracking Through Robust Onset Aggregation

**Full Citation**: McFee, B., & Ellis, D. P. W. (2014). Better Beat Tracking Through Robust Onset Aggregation. *Proceedings of the International Society for Music Information Retrieval Conference*.

**Category**: Onset Detection & Beat Tracking  
**Relevance**: Robust onset aggregation techniques for improving beat tracking accuracy

---

## Summary

This paper investigates methods to improve the robustness of beat tracking by enhancing onset detection and aggregation. The authors propose techniques such as median onset aggregation and spectrogram decomposition to separate rhythmic elements, leading to improved beat tracking accuracy.

## Key Contributions

1. **Robust Onset Aggregation**:
   - Median aggregation across multiple onset detection methods
   - Reduces false positives and improves reliability
   - Better than simple averaging or voting

2. **Spectrogram Decomposition**:
   - Separate rhythmic (percussive) from harmonic content
   - Similar to HPSS but with different approach
   - Improves onset detection in complex material

3. **Beat Tracking Improvements**:
   - Better onset lists → better beat tracking
   - Validates importance of onset quality
   - Provides framework for evaluating onset methods

## Relevance to Stratum DSP

This paper directly informs our consensus voting approach:

### Median Onset Aggregation
- **Source**: Core technique from this paper
- **Location**: `src/features/onset/consensus.rs`
- **Algorithm**: Use median instead of mean for aggregating onset scores

### Robust Voting
- **Insight**: Median aggregation more robust than mean
- **Implementation**: Can use median of onset strengths before voting
- **Benefit**: Reduces impact of outliers

### Onset Quality → Beat Tracking
- **Finding**: Better onsets → better beat tracking
- **Implication**: Validates our multi-method approach
- **Use Case**: Quality of onset detection critical for downstream tasks

## Key Algorithms

### Median Onset Aggregation

**Process**:
```
for each time frame:
    onset_scores = [energy_flux[t], spectral_flux[t], hfc[t], hpss[t]]
    median_score = median(onset_scores)
    if median_score > threshold:
        add_onset(t)
```

**Why Median?**:
- **Robust to Outliers**: Single method failure doesn't corrupt result
- **Better than Mean**: Mean sensitive to extreme values
- **Better than Max**: Max too sensitive to any method

### Spectrogram Decomposition

**Process**:
1. Decompose spectrogram into rhythmic and harmonic components
2. Detect onsets in rhythmic component
3. More reliable than detecting in full spectrogram

**Similar to HPSS**: But uses different decomposition method

## Implementation Notes

### Median Aggregation in Consensus

**Current Approach** (simple voting):
```
for each method:
    if method detects onset at time t:
        votes[t] += 1

if votes[t] >= threshold:
    final_onset[t] = true
```

**Improved Approach** (median aggregation):
```
for each time t:
    scores = [energy_flux[t], spectral_flux[t], hfc[t], hpss[t]]
    median_score = median(scores)
    if median_score > threshold:
        final_onset[t] = true
```

**Hybrid Approach** (combine both):
```
for each time t:
    scores = [energy_flux[t], spectral_flux[t], hfc[t], hpss[t]]
    median_score = median(scores)
    vote_count = count(scores > threshold)
    
    if median_score > threshold AND vote_count >= 2:
        final_onset[t] = true
```

### Threshold Selection

**Adaptive Threshold**:
```
threshold = median(all_scores) + k * MAD(all_scores)
```
where MAD = Median Absolute Deviation, k = 2-3 typical

**Fixed Threshold**:
- Genre-dependent
- Typically 0.1-0.3 of peak value

## Performance Characteristics

**Accuracy Improvement**:
- **Median Aggregation**: 5-10% improvement over mean
- **Spectrogram Decomposition**: 10-15% improvement
- **Combined**: 15-20% improvement in beat tracking

**Robustness**:
- **Outlier Handling**: Median much better than mean
- **Method Failure**: Single method failure doesn't corrupt result
- **Noise**: More robust to noisy conditions

## Comparison with Our Approach

| Aspect | Our Current | McFee & Ellis |
|--------|------------|--------------|
| Aggregation | Voting (count) | Median (score) |
| Threshold | Fixed/adaptive | Adaptive (median + MAD) |
| Methods | 4 methods | Multiple methods |
| HPSS | Separate method | Similar decomposition |

**Recommendation**: Incorporate median aggregation into our consensus voting

## References in Code

- `src/features/onset/consensus.rs`: Consensus voting (can add median aggregation)
- `src/features/onset/mod.rs`: Onset detection module
- `src/features/beat_tracking/hmm.rs`: Benefits from better onset lists

## Additional Notes

- Validates importance of onset quality for beat tracking
- Median aggregation more robust than mean
- Can improve our consensus voting implementation
- Provides framework for evaluating onset methods

---

**Last Updated**: 2025-01-XX  
**Status**: Reference for improving onset aggregation

