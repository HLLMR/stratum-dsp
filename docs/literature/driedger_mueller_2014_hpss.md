# Driedger & Müller (2014): Harmonic-Percussive Source Separation

**Full Citation**: Driedger, J., & Müller, M. (2014). Extending Harmonic-Percussive Separation of Audio Signals. *Proceedings of the International Society for Music Information Retrieval Conference*.

**Category**: Harmonic-Percussive Source Separation  
**Relevance**: HPSS algorithm using median filtering for improved onset detection

---

## Summary

This paper presents the Harmonic-Percussive Source Separation (HPSS) algorithm using median filtering. It separates audio into harmonic (sustained, tonal) and percussive (transient, rhythmic) components, which improves onset detection in complex musical material.

## Key Contributions

1. **HPSS Algorithm**:
   - Horizontal median filter (across time) → harmonic component
   - Vertical median filter (across frequency) → percussive component
   - State-of-the-art separation quality

2. **Median Filtering**:
   - Key technique for separation
   - Preserves harmonic content (broad in frequency, sustained in time)
   - Preserves percussive content (narrow in frequency, sharp in time)

3. **Applications**:
   - Onset detection in mixed material
   - Drum extraction
   - Harmonic analysis

## Relevance to Stratum DSP

This paper directly informs our HPSS implementation:

### HPSS Onset Detection
- **Source**: Core algorithm from this paper
- **Location**: `src/features/onset/hpss.rs`
- **Algorithm**: Separate harmonic/percussive → detect onsets in percussive component

### Why It Works
- **Harmonic Content**: Broad in frequency, sustained in time → horizontal median preserves
- **Percussive Content**: Narrow in frequency, sharp in time → vertical median preserves
- **Onset Detection**: Percussive component has clearer onsets

## Key Algorithms

### HPSS Separation

**Step 1: Compute STFT Magnitude Spectrogram**
```
X = STFT(audio, window=2048, hop=512)
magnitude = |X|  // Magnitude spectrogram
```

**Step 2: Horizontal Median Filter (Harmonic)**
```
H[n, f] = median(magnitude[n-margin..n+margin, f])
```
- Filters across time (horizontal)
- Preserves sustained tones (harmonic content)
- Removes sharp transients (percussive content)

**Step 3: Vertical Median Filter (Percussive)**
```
P[n, f] = median(magnitude[n, f-margin..f+margin])
```
- Filters across frequency (vertical)
- Preserves sharp transients (percussive content)
- Removes sustained tones (harmonic content)

**Step 4: Normalize**
```
H = H / (H + P + ε)
P = P / (H + P + ε)
```
- Normalize so H + P = 1 (approximately)
- ε prevents division by zero

**Step 5: Reconstruct Components**
```
harmonic = ISTFT(H * phase)
percussive = ISTFT(P * phase)
```

## Implementation Notes

### Median Filter Parameters

**Horizontal Filter (Harmonic)**:
- **Margin**: 5-10 frames typical (58-116ms)
- **Purpose**: Preserve sustained tones
- **Effect**: Removes sharp transients

**Vertical Filter (Percussive)**:
- **Margin**: 5-10 frequency bins typical
- **Purpose**: Preserve sharp transients
- **Effect**: Removes sustained tones

### Filter Size Selection

**Small Margin (3-5)**:
- **Separation**: Less aggressive
- **Artifacts**: Fewer
- **Use Case**: Subtle separation

**Medium Margin (5-10)**:
- **Separation**: Good balance
- **Artifacts**: Minimal
- **Use Case**: Default (our choice)

**Large Margin (10-20)**:
- **Separation**: Very aggressive
- **Artifacts**: More noticeable
- **Use Case**: Strong separation needed

### Onset Detection in Percussive Component

**Process**:
1. Extract percussive component using HPSS
2. Apply onset detection (energy flux, spectral flux, etc.) to percussive
3. Results: Clearer onsets, less interference from harmonic content

**Benefit**: 
- Harmonic content (sustained tones) doesn't interfere
- Percussive content (drums, transients) is emphasized
- Better onset detection in complex material

## Performance Characteristics

**Separation Quality**:
- **Subjective**: Good separation, minimal artifacts
- **Objective**: High SDR (Signal-to-Distortion Ratio)
- **State-of-the-Art**: Competitive with other methods

**Speed**:
- **Complexity**: O(n * log n) due to STFT
- **Typical Performance**: 20-30ms for 30s track
- **Most Expensive**: Among our onset methods

**Strengths**:
- Excellent for complex/mixed material
- Preserves both harmonic and percussive content
- Minimal artifacts
- Well-established method

**Weaknesses**:
- More expensive than other onset methods
- Requires STFT computation
- May introduce slight artifacts

## Why HPSS Improves Onset Detection

### Problem: Mixed Material
- **Harmonic Content**: Sustained tones (piano, strings) create false onsets
- **Percussive Content**: Sharp transients (drums) are the real onsets
- **Solution**: Separate components, detect onsets in percussive

### Harmonic Component
- **Characteristics**: Broad in frequency, sustained in time
- **Horizontal Median**: Preserves (filters across time, keeps sustained)
- **Onset Detection**: Not used (would create false positives)

### Percussive Component
- **Characteristics**: Narrow in frequency, sharp in time
- **Vertical Median**: Preserves (filters across frequency, keeps sharp)
- **Onset Detection**: Used (clear onsets, minimal interference)

## Applications

### Onset Detection
- **Input**: Mixed audio
- **Process**: HPSS → detect onsets in percussive
- **Output**: Clearer onset list

### Drum Extraction
- **Input**: Mixed audio
- **Process**: HPSS → extract percussive component
- **Output**: Drum-only audio

### Harmonic Analysis
- **Input**: Mixed audio
- **Process**: HPSS → extract harmonic component
- **Output**: Harmonic-only audio for analysis

## References in Code

- `src/features/onset/hpss.rs`: HPSS implementation
- `src/features/onset/consensus.rs`: Uses HPSS in consensus voting
- `src/features/onset/mod.rs`: Onset detection module

## Additional Notes

- State-of-the-art separation method
- Essential for complex/mixed material
- Most expensive onset method but worth it
- Well-established, widely used

---

**Last Updated**: 2025-01-XX  
**Status**: Core reference for HPSS onset detection

