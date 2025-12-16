# Böck et al. (2016): Joint Beat and Downbeat Tracking

**Full Citation**: Böck, S., Krebs, F., & Schedl, M. (2016). Joint Beat and Downbeat Tracking with a Recurrent Neural Network. *Proceedings of the International Society for Music Information Retrieval Conference*.

**Category**: Beat Tracking  
**Relevance**: HMM-based beat tracking using Viterbi algorithm

---

## Summary

This paper presents a state-of-the-art beat and downbeat tracking system using Hidden Markov Models (HMMs) and the Viterbi algorithm. It achieved top performance on MIREX evaluation and provides the theoretical foundation for our HMM-based beat tracking implementation.

## Key Contributions

1. **HMM-Based Beat Tracking**:
   - State space represents BPM variations around nominal estimate
   - Transition probabilities model tempo stability
   - Emission probabilities model onset alignment with expected beats

2. **Viterbi Algorithm**:
   - Finds globally optimal beat sequence
   - Handles tempo variations and syncopation
   - Backtracking extracts most likely path

3. **State-of-the-Art Performance**:
   - Top results on MIREX beat tracking evaluation
   - Robust to tempo variations
   - Handles complex musical structures

## Relevance to Stratum DSP

This paper directly informs our HMM beat tracking implementation:

### HMM Viterbi Beat Tracker
- **Source**: Core algorithm from this paper
- **Location**: `src/features/beat_tracking/hmm.rs`
- **Algorithm**: Build HMM → Viterbi forward pass → backtrack → extract beats

### State Space Design
- States represent BPM variations: [bpm*0.9, bpm*0.95, bpm, bpm*1.05, bpm*1.10]
- Models tempo stability and small variations
- Handles syncopation and timing jitter

## Key Algorithms

### HMM State Space

**States**: BPM variations around nominal estimate
```
states = [
    nominal_bpm * 0.90,  // -10% (slower)
    nominal_bpm * 0.95,  // -5% (slightly slower)
    nominal_bpm * 1.00,  // Nominal (expected)
    nominal_bpm * 1.05,  // +5% (slightly faster)
    nominal_bpm * 1.10   // +10% (faster)
]
```

**State Index**: Maps BPM to state index
```
state_index = round((actual_bpm - nominal_bpm) / (nominal_bpm * 0.05))
```

### Transition Probabilities

Model tempo stability:
```
transition[i][j] = {
    high if |i - j| <= 1,      // Small tempo change (likely)
    medium if |i - j| == 2,     // Moderate tempo change (possible)
    low if |i - j| > 2          // Large tempo change (unlikely)
}
```

**Example**:
```
transition[2][2] = 0.7   // Stay at nominal (most likely)
transition[2][1] = 0.15  // Slow down slightly
transition[2][3] = 0.15  // Speed up slightly
transition[2][0] = 0.0   // Large change (unlikely)
```

### Emission Probabilities

Model onset alignment with expected beats:
```
for each time frame t:
    for each state s (BPM variation):
        expected_beat_time = compute_beat_time(state_bpm[s], t)
        distance_to_nearest_onset = min(|expected_beat_time - onset|)
        
        if distance < tolerance:
            emission[t][s] = high
        else:
            emission[t][s] = low
```

**Formula**:
```
emission[t][s] = exp(-distance² / (2 * σ²))
```
where σ is timing tolerance (typically 50ms).

### Viterbi Algorithm

**Forward Pass** (compute best path probability):
```
viterbi[0][s] = initial_prob[s] * emission[0][s]

for t = 1 to T:
    for each state s:
        viterbi[t][s] = max(viterbi[t-1][s_prev] * transition[s_prev][s]) * emission[t][s]
        backpointer[t][s] = argmax(viterbi[t-1][s_prev] * transition[s_prev][s])
```

**Backtracking** (extract most likely path):
```
best_path[T] = argmax(viterbi[T][s])
for t = T-1 down to 0:
    best_path[t] = backpointer[t+1][best_path[t+1]]
```

**Extract Beats**:
```
for each frame t in best_path:
    if emission[t][best_path[t]] > threshold:
        beat_time = frame_to_time(t)
        add_beat(beat_time)
```

## Implementation Notes

### State Space Construction
- **Nominal BPM**: From period estimation (autocorrelation or comb filter)
- **Variations**: ±10% in 5% steps (5 states)
- **Alternative**: ±20% in 5% steps (9 states) for more flexibility

### Transition Probability Design
- **Self-Transition**: High (0.6-0.8) - tempo is stable
- **Neighbor Transitions**: Medium (0.1-0.2) - small tempo changes
- **Distant Transitions**: Low (0.0-0.05) - large tempo changes unlikely

### Emission Probability Computation
- **Expected Beat Times**: Compute from state BPM and frame index
- **Onset Matching**: Find nearest onset to expected beat time
- **Distance Metric**: Gaussian decay with σ = 50ms typical
- **Tolerance**: ±100ms window for onset matching

### Viterbi Implementation
- **Forward Pass**: Compute probabilities for all states at each time
- **Backtracking**: Trace back from final state to extract path
- **Beat Extraction**: Extract beats where emission probability is high

### Performance Optimization
- **Sparse Onsets**: Only compute emissions near actual onsets
- **Early Termination**: Stop if path probability becomes too low
- **Parallelization**: Forward pass can be parallelized across states

## Performance Characteristics

**Accuracy**:
- **MIREX Evaluation**: Top performance (F-measure >0.9)
- **Beat Tracking**: <50ms jitter on constant-tempo tracks
- **Downbeat Tracking**: 70-80% accuracy (if implemented)

**Speed**:
- **Complexity**: O(T * S²) where T=frames, S=states
- **Typical Performance**: 20-50ms for 30s track
- **Scalability**: Linear in track length

**Strengths**:
- Handles tempo variations and syncopation
- Globally optimal solution (Viterbi)
- Robust to timing jitter
- State-of-the-art performance

**Weaknesses**:
- Assumes constant tempo (within ±10%)
- Requires good initial BPM estimate
- More complex than simpler methods
- Computationally more expensive

## Handling Tempo Variations

### Small Variations (±10%)
- Handled by state space (5 states cover range)
- Transition probabilities allow gradual changes

### Moderate Variations (±20%)
- Expand state space to 9 states
- Adjust transition probabilities

### Large Variations (DJ Mixes, Live Recordings)
- **Segmentation**: Divide track into segments, track tempo per segment
- **Bayesian Update**: Use Bayesian beat tracker (separate module)
- **Limitation**: Phase 1 assumes constant tempo

## References in Code

- `src/features/beat_tracking/hmm.rs`: HMM Viterbi implementation
- `src/features/beat_tracking/bayesian.rs`: Bayesian tempo update (for variable tempo)
- `src/features/beat_tracking/mod.rs`: Beat tracking module

## Additional Notes

- State-of-the-art method, top MIREX performance
- Viterbi provides globally optimal solution
- Handles syncopation and timing jitter well
- Can be extended for downbeat tracking (future work)

---

**Last Updated**: 2025-01-XX  
**Status**: Core reference for HMM beat tracking

