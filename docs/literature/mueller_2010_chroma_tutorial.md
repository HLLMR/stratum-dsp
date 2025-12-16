# Müller et al. (2010): Chroma-Based Audio Analysis Tutorial

**Full Citation**: Müller, M., & Ewert, S. (2010). Chroma Toolbox: MATLAB Implementations for Extracting Variants of Chroma-Based Audio Features. *Proceedings of the International Society for Music Information Retrieval Conference*.

**Category**: Chroma Extraction  
**Relevance**: Comprehensive tutorial on chroma feature extraction for key detection

---

## Summary

This paper provides a comprehensive tutorial on chroma feature extraction, including multiple variants (standard, logarithmic, CQT-based) and applications. It forms the foundation for our chroma extraction implementation, which is essential for key detection.

## Key Contributions

1. **Chroma Extraction Methods**:
   - Standard chroma (STFT-based)
   - Logarithmic chroma
   - Constant-Q Transform (CQT) chroma
   - Comparison of different approaches

2. **Implementation Details**:
   - Frequency-to-semitone mapping
   - Octave summation
   - Normalization techniques
   - Temporal smoothing

3. **Applications**:
   - Key detection
   - Chord recognition
   - Cover song identification
   - Harmonic analysis

## Relevance to Stratum DSP

This paper directly informs our chroma extraction:

### Chroma Extractor
- **Source**: Core algorithm from this paper
- **Location**: `src/features/chroma/extractor.rs`
- **Algorithm**: STFT → frequency bins → semitone classes → octave summation → normalization

### Chroma Normalization
- **Source**: Normalization techniques from this paper
- **Location**: `src/features/chroma/normalization.rs`
- **Algorithm**: L2 normalization, sharpening, etc.

### Chroma Smoothing
- **Source**: Temporal smoothing from this paper
- **Location**: `src/features/chroma/smoothing.rs`
- **Algorithm**: Median/average filtering across time

## Key Algorithms

### Frequency-to-Semitone Mapping

**Formula**:
```
semitone = 12 * log2(freq / 440.0) + 57.0
semitone_class = semitone mod 12
```

**Explanation**:
- **440.0 Hz**: Reference frequency (A4)
- **57.0**: Offset to map A4 to semitone 57 (middle of piano range)
- **12 * log2**: Converts frequency ratio to semitones (12 semitones per octave)
- **mod 12**: Maps to semitone class (0-11, ignoring octave)

**Example**:
```
A4 (440 Hz): semitone = 12*log2(440/440) + 57 = 57, class = 57 mod 12 = 9
C4 (261.6 Hz): semitone = 12*log2(261.6/440) + 57 ≈ 52, class = 52 mod 12 = 4
```

### Chroma Extraction

**Step 1: Compute STFT**
```
X = STFT(audio, window=2048, hop=512)
magnitude = |X|
```

**Step 2: Map Frequencies to Semitone Classes**
```
for each frequency bin f:
    freq = bin_to_frequency(f, sample_rate, fft_size)
    semitone = 12 * log2(freq / 440.0) + 57.0
    semitone_class = semitone mod 12
    magnitude[semitone_class] += magnitude[f]
```

**Step 3: Normalize**
```
chroma = chroma / ||chroma||  // L2 normalization
```

### Soft Mapping

Instead of hard assignment, use soft mapping (spread to neighboring semitones):
```
for each frequency bin f:
    semitone = 12 * log2(freq / 440.0) + 57.0
    semitone_class = round(semitone) mod 12
    distance = |semitone - round(semitone)|
    weight = exp(-distance² / (2 * σ²))
    
    chroma[semitone_class] += weight * magnitude[f]
    if distance < threshold:
        chroma[(semitone_class + 1) mod 12] += (1 - weight) * magnitude[f]
```

## Implementation Notes

### STFT Parameters
- **Window Size**: 2048 samples (46ms at 44.1kHz)
- **Hop Size**: 512 samples (11.6ms)
- **Window Type**: Hann or Hamming window
- **FFT Size**: 2048 (matches window size)

### Frequency-to-Semitone Mapping
- **Reference**: A4 = 440 Hz
- **Offset**: 57.0 (maps A4 to semitone 57)
- **Range**: Typically covers 5-6 octaves (semitone 24-84)

### Octave Summation
- **Process**: Sum magnitude across all octaves for each semitone class
- **Result**: 12-element vector (one per semitone class: C, C#, D, ..., B)

### Normalization
- **L2 Normalization**: `chroma = chroma / ||chroma||`
- **Purpose**: Makes chroma independent of overall loudness
- **Alternative**: L1 normalization (sum to 1)

### Temporal Smoothing
- **Method**: Median or average filter across time
- **Window Size**: 5 frames typical (58ms at 11.6ms hop)
- **Purpose**: Reduces frame-to-frame variance

## Chroma Variants

### Standard Chroma
- **Method**: STFT → semitone mapping → octave sum
- **Use Case**: General purpose, fast
- **Our Implementation**: This variant

### Logarithmic Chroma
- **Method**: Use logarithmic frequency scale
- **Use Case**: Better frequency resolution at low frequencies
- **Trade-off**: More complex computation

### CQT Chroma
- **Method**: Constant-Q Transform (logarithmic frequency bins)
- **Use Case**: Better frequency resolution, matches human perception
- **Trade-off**: More expensive computation

## Performance Characteristics

**Accuracy** (for key detection):
- **Standard Chroma**: 70-80% accuracy
- **CQT Chroma**: 75-85% accuracy (slight improvement)
- **Trade-off**: Small accuracy gain for higher complexity

**Speed**:
- **Standard Chroma**: 10-50ms for 30s track
- **CQT Chroma**: 50-100ms (slower)
- **Our Choice**: Standard chroma (good balance)

**Strengths**:
- Captures harmonic content (pitch class)
- Invariant to octave (C4 and C5 both map to C)
- Normalized (independent of loudness)
- Fast computation

**Weaknesses**:
- Loses octave information (C4 = C5)
- Assumes equal temperament (12-TET)
- May miss microtonal music

## Applications

### Key Detection
- **Input**: Chroma vector (12 elements)
- **Process**: Compare to key templates (Krumhansl-Kessler)
- **Output**: Detected key and confidence

### Chord Recognition
- **Input**: Chroma vector
- **Process**: Compare to chord templates
- **Output**: Detected chord

### Cover Song Identification
- **Input**: Chroma sequences
- **Process**: Compare chroma sequences (DTW, etc.)
- **Output**: Similarity score

## References in Code

- `src/features/chroma/extractor.rs`: Chroma extraction implementation
- `src/features/chroma/normalization.rs`: Chroma normalization
- `src/features/chroma/smoothing.rs`: Temporal smoothing
- `src/features/key/detector.rs`: Uses chroma for key detection

## Additional Notes

- Comprehensive tutorial with implementation details
- Multiple variants discussed (we use standard)
- Applications beyond key detection
- Well-established method, widely used

---

**Last Updated**: 2025-01-XX  
**Status**: Core reference for chroma extraction

