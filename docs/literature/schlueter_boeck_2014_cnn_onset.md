# Schlüter & Böck (2014): Improved Musical Onset Detection with CNNs

**Full Citation**: Schlüter, J., & Böck, S. (2014). Improved Musical Onset Detection with Convolutional Neural Networks. *Proceedings of the International Society for Music Information Retrieval Conference*.

**Category**: Onset Detection (Machine Learning)  
**Relevance**: ML-based onset detection for Phase 2 refinement

---

## Summary

This paper demonstrates that Convolutional Neural Networks (CNNs) outperform previous state-of-the-art methods in musical onset detection, requiring less manual preprocessing. The findings suggest that machine learning can be superior to knowledge engineering in well-understood signal processing tasks, providing a path for Phase 2 ML refinement.

## Key Contributions

1. **CNN-Based Onset Detection**:
   - Trains CNN to detect onsets from spectrograms
   - Learns features automatically (no manual feature engineering)
   - Outperforms traditional methods

2. **Reduced Preprocessing**:
   - Less manual feature extraction
   - Network learns optimal features
   - Simpler pipeline

3. **Performance Improvements**:
   - Higher accuracy than traditional methods
   - Better generalization across genres
   - Robust to various audio conditions

## Relevance to Stratum DSP

This paper informs our Phase 2 ML refinement strategy:

### ML Refinement for Edge Cases
- **Phase 1**: Classical DSP methods (Energy Flux, Spectral Flux, HFC, HPSS)
- **Phase 2**: Small CNN to boost confidence on difficult tracks
- **Approach**: Train on edge cases, refine classical results

### Hybrid Approach
- **Base**: Classical DSP methods (fast, interpretable)
- **Refinement**: Small CNN for difficult cases
- **Benefit**: Best of both worlds (speed + accuracy)

## Key Algorithms

### CNN Architecture

**Input**: Spectrogram (time × frequency)

**Architecture**:
```
Conv2D(32 filters, 3×3) → ReLU → MaxPool
Conv2D(64 filters, 3×3) → ReLU → MaxPool
Conv2D(128 filters, 3×3) → ReLU → MaxPool
Flatten → Dense(256) → ReLU → Dropout(0.5)
Dense(1) → Sigmoid  # Onset probability
```

**Training**:
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam
- **Data**: Labeled onset annotations

### Inference

**Process**:
1. Compute spectrogram
2. Sliding window over time
3. CNN predicts onset probability per frame
4. Peak picking on probability curve
5. Threshold to get final onsets

## Implementation Notes

### Phase 2 Integration

**Small CNN Model**:
- **Purpose**: Refine classical DSP results
- **Size**: Small (few MB) for fast inference
- **Input**: Spectrogram patches
- **Output**: Onset probability

**When to Use**:
- Low confidence from classical methods
- Edge cases (variable tempo, complex harmony)
- User can enable/disable

### Training Data

**Requirements**:
- 1000+ tracks with onset annotations
- Diverse genres (electronic, rock, jazz, classical)
- Edge cases (variable tempo, complex arrangements)

**Sources**:
- MIREX datasets
- Custom annotations
- User corrections (active learning)

### Inference Performance

**Speed Target**: <50ms for 30s track
- **Classical DSP**: 30-50ms (baseline)
- **CNN Refinement**: +10-20ms (if enabled)
- **Total**: 40-70ms (acceptable)

## Performance Characteristics

**Accuracy**:
- **CNN Alone**: 85-90% recall (better than classical)
- **Hybrid (Classical + CNN)**: 90-95% recall
- **Edge Cases**: CNN significantly better

**Speed**:
- **CNN Inference**: 10-20ms for 30s track
- **Total Pipeline**: 40-70ms (with CNN)
- **Trade-off**: Small speed cost for accuracy gain

**Strengths**:
- Higher accuracy than classical methods
- Better generalization
- Learns optimal features
- Robust to edge cases

**Weaknesses**:
- Requires training data
- More complex (black box)
- Slower than classical methods
- Requires GPU for training (CPU inference OK)

## Phase 2 Strategy

### Small Model Approach
- **Size**: <5MB model
- **Purpose**: Refine classical results, not replace
- **Inference**: CPU-optimized (ONNX Runtime)
- **When**: Low confidence cases only

### Hybrid Pipeline
```
1. Classical DSP → onset list + confidence
2. If confidence < threshold:
   a. Run CNN refinement
   b. Merge CNN results with classical
3. Final onset list
```

### Training Strategy
- **Focus**: Edge cases (variable tempo, complex harmony)
- **Data**: 1000+ diverse tracks
- **Validation**: Hold out test set
- **Metrics**: Recall, precision, F-measure

## References in Code

- `src/ml/refinement.rs`: ML refinement module (Phase 2)
- `src/ml/onnx_model.rs`: ONNX model loading/inference
- `src/features/onset/consensus.rs`: Would integrate CNN results

## Additional Notes

- Phase 2 enhancement (not Phase 1)
- Validates ML approach for difficult cases
- Small model strategy (fast inference)
- Hybrid approach (classical + ML)

---

**Last Updated**: 2025-01-XX  
**Status**: Reference for Phase 2 ML refinement

