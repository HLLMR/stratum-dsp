# Bello et al. (2005): A Tutorial on Onset Detection in Music Signals

**Full Citation**: Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., & Sandler, M. B. (2005). A Tutorial on Onset Detection in Music Signals. *IEEE Transactions on Speech and Audio Processing*, 13(5), 1035-1047.

**Category**: Onset Detection  
**Relevance**: Core algorithm reference for multiple onset detection methods

---

## Summary

This paper provides a comprehensive tutorial and survey of onset detection methods in music signals. It classifies methods into four categories (energy-based, spectral-based, phase-based, and complex-domain) and provides a framework for evaluating and comparing different approaches.

## Key Contributions

1. **Taxonomy of Onset Detection Methods**:
   - Energy-based methods: Detect changes in signal energy
   - Spectral-based methods: Detect changes in frequency content
   - Phase-based methods: Detect phase discontinuities
   - Complex-domain methods: Combine magnitude and phase information

2. **Performance Evaluation Framework**:
   - Metrics for comparing methods
   - Discussion of challenges (noise, reverberation, polyphonic music)
   - Genre-specific performance considerations

3. **Practical Implementation Guidance**:
   - Algorithm descriptions with formulas
   - Parameter selection guidelines
   - Trade-offs between methods

## Relevance to Stratum DSP

This paper directly informs three of our four onset detection methods:

### Energy Flux
- **Source**: Energy-based methods from this paper
- **Algorithm**: Compute frame-by-frame energy derivative
- **Formula**: `E_flux[n] = max(0, E[n] - E[n-1])`
- **Use Case**: Fast, works well for clean electronic music

### Spectral Flux
- **Source**: Spectral-based methods from this paper
- **Algorithm**: Compute L2 distance between consecutive magnitude spectra
- **Formula**: `flux[n] = sqrt(sum((M[n] - M[n-1])²))`
- **Use Case**: Catches spectral changes, robust to compression artifacts

### High-Frequency Content (HFC)
- **Source**: Spectral methods with frequency weighting
- **Algorithm**: Weight higher frequencies more heavily (typical of percussive attacks)
- **Use Case**: Excellent for drums and percussion

### Multi-Method Consensus
- **Insight**: Paper emphasizes that different methods work better for different genres
- **Implementation**: We combine all methods with weighted voting for robustness

## Key Algorithms

### Energy-Based Onset Detection

**Energy Flux**:
```
E[n] = sqrt(sum(x[n*hop..(n+1)*hop]²))  // RMS energy
E_flux[n] = max(0, E[n] - E[n-1])        // Energy derivative
```

**Peak Picking**:
- Apply threshold (adaptive or fixed)
- Find local maxima
- Merge peaks within tolerance window

### Spectral-Based Onset Detection

**Spectral Flux**:
```
M[n] = |FFT(x[n*hop..(n+1)*hop])|       // Magnitude spectrum
M[n] = M[n] / ||M[n]||                  // Normalize
flux[n] = sqrt(sum((M[n] - M[n-1])²))   // L2 distance
```

**High-Frequency Content**:
```
HFC[n] = sum(k * |X[n][k]|²)            // Weight by frequency bin k
```

## Implementation Notes

### Energy Flux Implementation
1. Compute RMS energy per frame
2. Compute first-order difference (derivative)
3. Half-wave rectification (keep only positive changes)
4. Peak picking with adaptive threshold

### Spectral Flux Implementation
1. Compute STFT (2048-point FFT, 512 sample hop typical)
2. Normalize magnitude spectrum (L2 norm)
3. Compute L2 distance between consecutive frames
4. Peak picking

### HFC Implementation
1. Compute STFT
2. Weight magnitude squared by frequency bin index
3. Sum weighted magnitudes
4. Peak picking

### Consensus Voting
- Merge onset lists from all methods within tolerance window (50ms default)
- Weight votes by method confidence
- Higher agreement → higher confidence in final onset

## Performance Characteristics

**Energy Flux**:
- **Speed**: Very fast (O(n))
- **Accuracy**: 70-80% recall on clean electronic music
- **Strengths**: Fast, simple, works for clear beats
- **Weaknesses**: Fails on complex polyphonic music, sensitive to noise

**Spectral Flux**:
- **Speed**: Moderate (O(n log n) due to FFT)
- **Accuracy**: 75-85% recall
- **Strengths**: Robust to compression, catches spectral changes
- **Weaknesses**: Slower, can miss subtle onsets

**HFC**:
- **Speed**: Moderate (O(n log n))
- **Accuracy**: 80-90% recall on percussive music
- **Strengths**: Excellent for drums, emphasizes transients
- **Weaknesses**: Can miss low-frequency onsets

## Challenges Addressed

1. **Noise**: Spectral methods more robust than energy methods
2. **Reverberation**: Phase-based methods can help, but complex
3. **Polyphonic Music**: Multiple simultaneous onsets → consensus voting helps
4. **Genre Variation**: Different methods excel in different genres

## Parameter Selection

### Frame Size
- **Energy Flux**: 512-1024 samples (11-23ms at 44.1kHz)
- **Spectral Flux**: 2048 samples (46ms) for good frequency resolution
- **Hop Size**: 256-512 samples (5.8-11.6ms) for temporal resolution

### Threshold
- **Adaptive**: Use median + multiple of MAD (median absolute deviation)
- **Fixed**: Genre-dependent, typically 0.1-0.3 of peak value

### Peak Picking
- **Minimum Distance**: 30-50ms between onsets
- **Tolerance Window**: 50ms for consensus voting

## References in Code

- `src/features/onset/energy_flux.rs`: Energy-based onset detection
- `src/features/onset/spectral_flux.rs`: Spectral-based onset detection
- `src/features/onset/hfc.rs`: High-frequency content method
- `src/features/onset/consensus.rs`: Multi-method voting

## Additional Notes

- Paper emphasizes that no single method is best for all music
- Recommends combining methods for robustness
- Provides evaluation framework we can use for testing
- Discusses real-time considerations (we focus on offline analysis)

---

**Last Updated**: 2025-01-XX  
**Status**: Core reference for onset detection implementation

