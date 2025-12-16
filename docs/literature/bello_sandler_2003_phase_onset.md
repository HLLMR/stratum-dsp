# Bello & Sandler (2003): Phase-Based Note Onset Detection

**Full Citation**: Bello, J. P., & Sandler, M. B. (2003). Phase-Based Note Onset Detection for Music Signals. *Proceedings of the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics*.

**Category**: Onset Detection  
**Relevance**: Alternative onset detection method using phase information

---

## Summary

This paper proposes an alternative to standard energy-based onset detection by using phase information. By observing the distribution of differential phase angles frame-by-frame, the method accurately detects the precise moment of onsets in complex musical recordings, providing a complementary approach to energy and spectral methods.

## Key Contributions

1. **Phase-Based Onset Detection**:
   - Uses phase information instead of magnitude
   - Detects phase discontinuities at onsets
   - Complementary to energy/spectral methods

2. **Differential Phase Angles**:
   - Compute phase difference between consecutive frames
   - Phase jumps indicate onsets
   - More precise timing than energy methods

3. **Robustness**:
   - Works well for soft attacks (piano, strings)
   - Less sensitive to noise than energy methods
   - Good for polyphonic music

## Relevance to Stratum DSP

This paper provides an alternative onset detection method:

### Potential 5th Onset Method
- **Current**: Energy Flux, Spectral Flux, HFC, HPSS
- **Addition**: Phase-based method (optional)
- **Benefit**: Better for soft attacks, complementary to existing methods

### Soft Attack Detection
- **Problem**: Energy methods miss soft attacks (piano, strings)
- **Solution**: Phase method detects phase discontinuities
- **Use Case**: Classical music, jazz, complex arrangements

## Key Algorithms

### Phase-Based Onset Detection

**Step 1: Compute STFT with Phase**
```
X = STFT(audio, window=2048, hop=512)
magnitude = |X|
phase = angle(X)
```

**Step 2: Compute Differential Phase**
```
for each frequency bin f:
    phase_diff[f] = phase[t, f] - phase[t-1, f]
    # Unwrap phase to handle 2π jumps
    phase_diff[f] = unwrap(phase_diff[f])
```

**Step 3: Compute Phase Deviation**
```
phase_deviation[t] = std(phase_diff[t, :])
```

**Step 4: Detect Onsets**
```
if phase_deviation[t] > threshold:
    onset[t] = true
```

### Phase Unwrapping

**Problem**: Phase wraps at ±π

**Solution**: Unwrap to continuous phase
```
if phase_diff > π:
    phase_diff -= 2π
if phase_diff < -π:
    phase_diff += 2π
```

## Implementation Notes

### Phase Computation
- **STFT**: Same parameters as spectral flux (2048 window, 512 hop)
- **Phase Extraction**: Use `atan2(imag, real)` for full range
- **Unwrapping**: Handle 2π discontinuities

### Phase Deviation
- **Method**: Standard deviation of phase differences
- **Threshold**: Adaptive (median + k*MAD)
- **Normalization**: Optional normalization by frequency

### Combining with Other Methods
- **Consensus Voting**: Add phase method to existing 4 methods
- **Weighting**: Lower weight (soft attacks less common in DJ music)
- **Optional**: Only enable for specific genres

## Performance Characteristics

**Accuracy**:
- **Soft Attacks**: 80-90% recall (better than energy methods)
- **Hard Attacks**: 70-80% recall (worse than energy methods)
- **Overall**: 75-85% recall (complementary to other methods)

**Speed**:
- **Complexity**: O(n log n) due to STFT
- **Typical Performance**: 5-10ms for 30s track
- **Overhead**: Similar to spectral flux

**Strengths**:
- Excellent for soft attacks
- Precise timing (phase discontinuities)
- Robust to noise
- Complementary to energy methods

**Weaknesses**:
- More complex than energy methods
- Requires phase unwrapping
- Less effective for hard attacks (drums)
- May have false positives from vibrato/tremolo

## When to Use

**Recommended For**:
- Classical music (soft attacks)
- Jazz (complex harmony)
- Acoustic instruments
- Polyphonic music

**Not Recommended For**:
- Electronic music (energy methods better)
- Drum-heavy tracks (HFC better)
- Real-time applications (more expensive)

## Integration with Existing Methods

### As 5th Method
```
methods = [energy_flux, spectral_flux, hfc, hpss, phase]
consensus = vote(methods)
```

### Weighted Voting
```
weights = {
    energy_flux: 1.0,    # Good for electronic
    spectral_flux: 1.0, # Good general purpose
    hfc: 1.0,          # Good for drums
    hpss: 1.0,         # Good for mixed material
    phase: 0.5,        # Good for soft attacks (less common)
}
```

### Genre-Dependent
```
if genre == "classical" or genre == "jazz":
    enable_phase_method()
else:
    disable_phase_method()
```

## References in Code

- `src/features/onset/phase.rs`: Phase-based onset detection (if implemented)
- `src/features/onset/consensus.rs`: Would include phase method in voting
- `src/features/onset/mod.rs`: Onset detection module

## Additional Notes

- Complementary to existing methods
- Best for soft attacks (piano, strings)
- More complex but provides additional coverage
- Optional addition for Phase 1, or Phase 2 enhancement

---

**Last Updated**: 2025-01-XX  
**Status**: Optional reference for phase-based onset detection

