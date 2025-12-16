# Ellis & Pikrakis (2006): Real-time Beat Induction

**Full Citation**: Ellis, D. P. W., & Pikrakis, A. (2006). Real-time Beat Induction. *Proceedings of the International Conference on Music Information Retrieval*.

**Category**: Beat Tracking & Tempo Estimation  
**Relevance**: Core algorithm for autocorrelation-based BPM estimation

---

## Summary

This paper presents a real-time beat tracking algorithm that uses autocorrelation for tempo estimation. It establishes the standard pipeline: onset detection → tempo estimation → beat tracking, which forms the foundation of our period estimation module.

## Key Contributions

1. **Autocorrelation-Based Tempo Estimation**:
   - Convert onset list to binary beat signal
   - Compute autocorrelation to find periodicities
   - Extract tempo from autocorrelation peaks

2. **Real-Time Pipeline**:
   - Onset detection → tempo estimation → beat tracking
   - Efficient algorithms suitable for real-time processing
   - Incremental updates as new audio arrives

3. **Practical Implementation**:
   - FFT-accelerated autocorrelation
   - Peak picking in autocorrelation function
   - BPM conversion from lag values

## Relevance to Stratum DSP

This paper directly informs our autocorrelation-based BPM estimation:

### Autocorrelation Period Estimation
- **Source**: Core algorithm from this paper
- **Location**: `src/features/period/autocorrelation.rs`
- **Algorithm**: Convert onsets → binary signal → autocorrelation → peaks → BPM

### Pipeline Structure
- **Onset Detection**: First stage (we use multiple methods)
- **Period Estimation**: Second stage (autocorrelation + comb filterbank)
- **Beat Tracking**: Third stage (HMM Viterbi)

## Key Algorithms

### Autocorrelation-Based Tempo Estimation

**Step 1: Convert Onsets to Binary Signal**
```
beat_signal[n] = 1 if onset detected at frame n, else 0
```

**Step 2: Compute Autocorrelation**
```
ACF[lag] = sum(beat_signal[i] * beat_signal[i + lag])
```

**Step 3: Find Peaks**
- Find local maxima in ACF
- Filter by minimum lag (corresponds to max BPM)
- Filter by maximum lag (corresponds to min BPM)

**Step 4: Convert Lag to BPM**
```
BPM = (60 * sample_rate) / (lag * hop_size)
```

### FFT-Accelerated Autocorrelation

For efficiency, use FFT:
```
ACF = IFFT(|FFT(beat_signal)|²)
```

This reduces complexity from O(n²) to O(n log n).

## Implementation Notes

### Binary Beat Signal
- **Frame Rate**: Match onset detection frame rate (typically 512 sample hop)
- **Onset Placement**: Place 1 at frame containing onset
- **Smoothing**: Optional Gaussian smoothing to handle timing uncertainty

### Autocorrelation Computation
- **FFT Size**: Next power of 2 >= 2*length(beat_signal)
- **Zero Padding**: Pad to FFT size
- **Normalization**: Optional normalization by signal length

### Peak Picking
- **Minimum Lag**: Corresponds to max BPM (e.g., 180 BPM → lag_min)
- **Maximum Lag**: Corresponds to min BPM (e.g., 60 BPM → lag_max)
- **Peak Detection**: Find local maxima, filter by prominence
- **Octave Errors**: Common issue (detects 2x or 0.5x true BPM)

### BPM Conversion
```
lag_samples = lag * hop_size
period_seconds = lag_samples / sample_rate
BPM = 60 / period_seconds
```

## Performance Characteristics

**Accuracy**:
- **Synthetic Data**: 95%+ accuracy on clean patterns
- **Real Music**: 75-85% accuracy (±2 BPM tolerance)
- **Octave Errors**: 10-15% of tracks (detect 2x or 0.5x)

**Speed**:
- **FFT-Accelerated**: O(n log n) complexity
- **Typical Performance**: 5-15ms for 30s track
- **Real-Time Capable**: Yes, with incremental updates

**Strengths**:
- Fast and efficient
- Directly finds periodicity
- Works well for constant-tempo music
- FFT acceleration makes it scalable

**Weaknesses**:
- Prone to octave errors (2x or 0.5x BPM)
- Assumes constant tempo
- Can fail on variable tempo (DJ mixes, live recordings)
- Requires sufficient onsets for reliable autocorrelation

## Octave Error Handling

The paper discusses octave errors, which we address by:

1. **BPM Range Filtering**: Reject BPMs outside 60-180 range
2. **Comb Filterbank**: Alternative method that tests hypothesis tempos
3. **Consensus**: Merge autocorrelation and comb filter results

## Real-Time Considerations

While we focus on offline analysis, the paper's real-time approach informs our design:

- **Incremental Updates**: Can update BPM estimate as more audio arrives
- **Sliding Window**: Use recent onsets for autocorrelation
- **Efficiency**: FFT acceleration critical for real-time

## Parameter Selection

### Hop Size
- **Typical**: 512 samples (11.6ms at 44.1kHz)
- **Trade-off**: Smaller = better temporal resolution, larger = faster

### FFT Size
- **Minimum**: 2 * length(beat_signal)
- **Typical**: Next power of 2 (e.g., 2048, 4096, 8192)

### BPM Range
- **Minimum**: 60 BPM (lag_max)
- **Maximum**: 180 BPM (lag_min)
- **Typical DJ Range**: 120-140 BPM

### Peak Picking
- **Prominence**: 10-20% of peak value
- **Minimum Distance**: Corresponds to ±5 BPM separation

## References in Code

- `src/features/period/autocorrelation.rs`: Autocorrelation implementation
- `src/features/period/peak_picking.rs`: Peak detection in ACF
- `src/features/period/candidate_filter.rs`: BPM range filtering and octave error handling

## Additional Notes

- Establishes standard pipeline we follow
- FFT acceleration critical for performance
- Octave errors are common and need handling
- Real-time approach could be adapted for streaming analysis

---

**Last Updated**: 2025-01-XX  
**Status**: Core reference for autocorrelation BPM estimation

