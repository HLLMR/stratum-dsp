# ITU-R BS.1770-4: Loudness Measurement Standard

**Full Citation**: ITU-R BS.1770-4 (2015). Algorithms to measure audio programme loudness and true-peak audio level.

**Category**: Loudness Normalization  
**Relevance**: Standard for LUFS (Loudness Units relative to Full Scale) measurement

---

## Summary

This ITU (International Telecommunication Union) standard defines algorithms for measuring audio programme loudness and true-peak audio level. It provides the K-weighting filter and gating algorithm used in our normalization module for LUFS measurement.

## Key Contributions

1. **LUFS Measurement**:
   - Loudness Units relative to Full Scale
   - Standardized loudness measurement
   - Used in broadcast and streaming (YouTube, Spotify normalization)

2. **K-Weighting Filter**:
   - Pre-filter before loudness measurement
   - Models human perception of loudness
   - Frequency-dependent weighting

3. **Gating Algorithm**:
   - Gate at -70 LUFS
   - Excludes silence and very quiet passages
   - Provides stable loudness measurement

## Relevance to Stratum DSP

This standard informs our loudness normalization:

### LUFS Measurement
- **Source**: Algorithm from this standard
- **Location**: `src/preprocessing/normalization.rs`
- **Algorithm**: K-weighting → mean square → gate → LUFS conversion

### Normalization Target
- **Typical Target**: -14 LUFS (streaming platforms)
- **DJ Applications**: -6 to -12 LUFS (louder, club-ready)
- **Our Default**: -12 LUFS (balanced)

## Key Algorithms

### K-Weighting Filter

**Purpose**: Model human perception of loudness

**Frequency Response**:
- **Low Frequencies**: Boost (human hearing less sensitive)
- **Mid Frequencies**: Flat
- **High Frequencies**: Slight boost

**Implementation**:
- **IIR Filter**: High-shelf + low-shelf filters
- **Coefficients**: Defined in standard
- **Purpose**: Pre-filter before loudness measurement

### Mean Square Calculation

**Process**:
```
for each sample:
    k_weighted = k_filter(sample)
    mean_square += k_weighted²

mean_square = mean_square / num_samples
```

### Gating Algorithm

**Gate Threshold**: -70 LUFS

**Process**:
```
loudness_gated = []
for each block:
    block_loudness = compute_loudness(block)
    if block_loudness > -70 LUFS:
        loudness_gated.append(block_loudness)

final_loudness = mean(loudness_gated)
```

**Purpose**: Exclude silence and very quiet passages

### LUFS Conversion

**Formula**:
```
LUFS = -0.691 + 10 * log10(mean_square)
```

**Explanation**:
- **-0.691**: Offset to match reference level
- **10 * log10**: Convert to decibels
- **Result**: LUFS value

## Implementation Notes

### K-Weighting Filter

**Filter Type**: IIR (Infinite Impulse Response)
- **High-Shelf**: Boosts high frequencies
- **Low-Shelf**: Boosts low frequencies
- **Coefficients**: Defined in standard (fixed values)

**Alternative**: Use pre-computed filter coefficients
- **Biquad Sections**: Implement as cascaded biquad filters
- **Efficiency**: Fast, O(n) per sample

### Block-Based Processing

**Block Size**: 400ms typical (17640 samples at 44.1kHz)

**Process**:
1. Apply K-weighting to block
2. Compute mean square
3. Convert to LUFS
4. Apply gate
5. Average gated blocks

### True-Peak Measurement

**Purpose**: Measure peak level (not just RMS)

**Process**:
1. Upsample signal (4x typical)
2. Find maximum absolute value
3. Convert to dBTP (decibels True Peak)

**Use Case**: Prevent clipping in normalization

## Performance Characteristics

**Accuracy**:
- **Standard Compliance**: Matches ITU-R BS.1770-4
- **Measurement**: ±0.1 LUFS typical accuracy
- **Reliability**: Stable across different audio content

**Speed**:
- **K-Weighting**: O(n) per sample
- **Mean Square**: O(n) per block
- **Typical Performance**: 5-10ms for 30s track

**Strengths**:
- Standardized measurement (industry standard)
- Perceptually accurate (K-weighting)
- Stable (gating algorithm)
- Widely used (broadcast, streaming)

**Weaknesses**:
- More complex than simple RMS
- Requires K-weighting filter implementation
- Gating adds complexity

## Normalization Process

### Measure LUFS
1. Apply K-weighting filter
2. Compute mean square
3. Apply gate (-70 LUFS)
4. Convert to LUFS

### Compute Gain
```
target_lufs = -12.0  // Target loudness
current_lufs = measure_lufs(audio)
gain_db = target_lufs - current_lufs
gain_linear = 10^(gain_db / 20)
```

### Apply Gain
```
normalized = audio * gain_linear
```

### True-Peak Limiting
```
if max(|normalized|) > 1.0:
    limit_gain = 1.0 / max(|normalized|)
    normalized = normalized * limit_gain
```

## References in Code

- `src/preprocessing/normalization.rs`: LUFS measurement and normalization
- **Reference**: ITU-R BS.1770-4 Annex 2 (algorithm specification)

## Additional Notes

- Industry standard for loudness measurement
- Used by YouTube, Spotify, broadcast
- Perceptually accurate (K-weighting)
- Essential for professional audio processing

---

**Last Updated**: 2025-01-XX  
**Status**: Standard reference for loudness normalization

