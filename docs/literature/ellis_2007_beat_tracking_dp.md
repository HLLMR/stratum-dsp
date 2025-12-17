# Ellis (2007): Beat Tracking by Dynamic Programming

**Full Citation**: Ellis, D. P. W. (2007). Beat Tracking by Dynamic Programming. *Journal of New Music Research*.

**Category**: Beat Tracking & Tempo Estimation  
**Relevance**: Foundation for onset-aligned beat tracking and global analysis approach

---

## Summary

This paper presents a beat tracking algorithm using dynamic programming to find the globally optimal beat sequence aligned with detected onsets. Unlike frame-by-frame methods, this approach analyzes the entire song to find the best beat path, establishing why global analysis beats local frame-by-frame methods. This work provides the theoretical foundation for tempogram-based approaches that analyze entire songs rather than individual frames.

## Key Contributions

1. **Dynamic Programming Beat Tracking**:
   - Finds globally optimal beat sequence
   - Aligns beats with detected onsets
   - Handles tempo variations through state space
   - More robust than greedy frame-by-frame methods

2. **Global Analysis Philosophy**:
   - Analyzes entire song, not individual frames
   - Finds best overall solution, not local optima
   - Prevents subharmonic errors (common in per-frame methods)
   - Establishes foundation for tempogram approach

3. **Onset-Aligned Beat Tracking**:
   - Uses onset detection as foundation
   - Finds beat sequence that best aligns with onsets
   - Handles syncopation and timing jitter
   - More accurate than energy-based methods

## Relevance to Stratum DSP

This paper provides the theoretical foundation for our tempogram pivot:

### Global Analysis Approach
- **Philosophy**: Analyze entire song, not frame-by-frame
- **Rationale**: Prevents subharmonic errors, finds global optimum
- **Application**: Tempogram analyzes entire novelty curve globally

### Onset-Aligned Beat Tracking
- **Foundation**: Onset detection → beat tracking pipeline
- **Location**: `src/features/beat_tracking/hmm.rs` (already implemented)
- **Connection**: Tempogram provides better BPM estimate for beat tracking

### Why Global > Local
- **Current Problem**: Per-frame analysis causes subharmonic errors
- **Solution**: Global analysis (like this paper) prevents this
- **Tempogram**: Implements global analysis for BPM detection

## Key Algorithms

### Dynamic Programming Beat Tracking

**Step 1: Onset Detection**
```
onsets = detect_onsets(audio)
```

**Step 2: Build State Space**
```
states = [bpm * 0.9, bpm * 0.95, bpm, bpm * 1.05, bpm * 1.10]
// Represents tempo variations around nominal BPM
```

**Step 3: Compute Costs**
```
for each time t:
    for each state s:
        expected_beat_time = compute_from_state(s, t)
        cost[t][s] = distance_to_nearest_onset(expected_beat_time)
```

**Step 4: Dynamic Programming**
```
dp[0][s] = initial_cost[0][s]

for t = 1 to T:
    for each state s:
        dp[t][s] = min(dp[t-1][s_prev] + transition_cost + cost[t][s])
        backpointer[t][s] = argmin(...)
```

**Step 5: Backtrack**
```
best_path = backtrack(backpointer)
beats = extract_beats(best_path)
```

### Why Global Analysis Works

**Frame-by-Frame (Broken)**:
- Each frame analyzed independently
- Subharmonics appear in every frame
- No global consistency check
- Result: Subharmonics get high confidence

**Global Analysis (Correct)**:
- Analyzes entire song
- Finds globally optimal solution
- Subharmonics less consistent globally
- Result: True tempo wins

## Implementation Notes

### Global vs Local Analysis

**Local (Current System)**:
- Autocorrelation per frame → merge
- Comb filter per frame → merge
- Problem: Subharmonics consistent locally

**Global (Tempogram)**:
- Analyze entire novelty curve
- Single autocorrelation over full song
- Result: True tempo more consistent globally

### Dynamic Programming Connection

While we use HMM Viterbi (from Böck 2016) for beat tracking, the philosophy is the same:
- **Global Optimization**: Find best overall solution
- **State Space**: Model tempo variations
- **Alignment**: Align with onsets

The tempogram provides the BPM estimate that feeds into this beat tracking.

## Performance Characteristics

**Accuracy**:
- **Beat Tracking**: <50ms jitter on constant-tempo tracks
- **Tempo Estimation**: Better when combined with global BPM detection
- **Robustness**: Handles syncopation and timing jitter

**Speed**:
- **Complexity**: O(T * S²) where T=frames, S=states
- **Typical Performance**: 20-50ms for 30s track
- **Scalability**: Linear in track length

**Strengths**:
- Globally optimal solution
- Handles tempo variations
- Robust to timing jitter
- Prevents subharmonic errors

**Weaknesses**:
- Requires good initial BPM estimate
- Assumes relatively constant tempo
- More complex than greedy methods

## Connection to Tempogram

This paper establishes why global analysis works:

1. **Problem**: Frame-by-frame methods fail on subharmonics
2. **Solution**: Global analysis finds true tempo
3. **Tempogram**: Implements global analysis for BPM detection
4. **Result**: Single unambiguous peak (like global DP solution)

The tempogram is essentially applying the global analysis philosophy to BPM estimation:
- Instead of analyzing frames independently
- Analyze entire novelty curve globally
- Find BPM with strongest global periodicity

## References in Code

- `src/features/beat_tracking/hmm.rs`: HMM Viterbi (similar global optimization)
- `src/features/period/tempogram.rs`: Global tempogram analysis (to be created)
- `src/features/period/novelty.rs`: Novelty curve for global analysis (to be created)

## Additional Notes

- **Foundation**: Establishes why global > local analysis
- **Philosophy**: Applied to tempogram approach
- **Connection**: Tempogram provides BPM for beat tracking
- **Legacy**: Influenced all modern beat tracking systems

---

**Last Updated**: 2025-01-XX  
**Status**: Theoretical foundation for global analysis approach (tempogram pivot)

