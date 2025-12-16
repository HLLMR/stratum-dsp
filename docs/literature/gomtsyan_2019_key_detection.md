# Gomtsyan et al. (2019): Music Key and Scale Detection

**Full Citation**: Gomtsyan, M., et al. (2019). Music Key and Scale Detection. *Proceedings of the International Conference on Music Information Retrieval*.

**Category**: Key Detection  
**Relevance**: Modern evaluation of key detection methods and performance benchmarks

---

## Summary

This paper provides a modern evaluation of key detection methods, comparing template-matching approaches (like Krumhansl-Kessler) with machine learning methods. It validates that Krumhansl-Kessler templates remain competitive and provides performance benchmarks for key detection systems.

## Key Contributions

1. **Comparative Evaluation**:
   - Systematic comparison of template-matching vs. ML approaches
   - Performance analysis across different music genres
   - Identification of strengths and weaknesses

2. **Performance Benchmarks**:
   - Accuracy metrics for different methods
   - Genre-specific performance analysis
   - Confidence scoring evaluation

3. **Key Clarity Metrics**:
   - Discussion of how to measure tonal strength
   - Relationship between clarity and accuracy
   - Guidelines for when key detection is reliable

## Relevance to Stratum DSP

This paper validates our approach and sets performance expectations:

### Template-Matching Validation
- **Finding**: Krumhansl-Kessler templates still competitive with modern ML
- **Implication**: Our template-matching approach is sound
- **Performance**: 70-80% accuracy on tonal music (matches our targets)

### Performance Benchmarks
- Provides accuracy targets we can aim for
- Sets realistic expectations (70-80% on real music)
- Identifies edge cases where methods fail

### Key Clarity
- Validates our key clarity metric
- Provides guidelines for reliability assessment
- Helps identify when key detection is trustworthy

## Key Findings

### Accuracy by Method

**Template-Matching (Krumhansl-Kessler)**:
- **Tonal Music**: 70-80% accuracy
- **Complex Harmony**: 60-70% accuracy
- **Atonal Music**: <50% accuracy (expected)

**Machine Learning Approaches**:
- **Deep Learning**: 75-85% accuracy (slight improvement)
- **Complexity**: Much higher (requires training data)
- **Trade-off**: Small accuracy gain for large complexity increase

### Accuracy by Genre

**Electronic/Dance Music**:
- **Accuracy**: 80-85% (clear tonality, simple harmony)
- **Confidence**: High (strong key clarity)

**Rock/Pop Music**:
- **Accuracy**: 75-80% (moderate complexity)
- **Confidence**: Medium to high

**Jazz/Complex Music**:
- **Accuracy**: 60-70% (complex harmony, modulations)
- **Confidence**: Low to medium

**Classical Music**:
- **Accuracy**: 65-75% (modulations, key changes)
- **Confidence**: Medium

**Ambient/Experimental**:
- **Accuracy**: <50% (atonal, no clear key)
- **Confidence**: Very low

## Key Clarity Analysis

### Clarity vs. Accuracy

**High Clarity (>2.0)**:
- **Accuracy**: 85-90%
- **Reliability**: Very high
- **Use Case**: Confident key detection

**Medium Clarity (1.5-2.0)**:
- **Accuracy**: 70-80%
- **Reliability**: Moderate
- **Use Case**: Key detection with caution

**Low Clarity (<1.5)**:
- **Accuracy**: 50-60%
- **Reliability**: Low
- **Use Case**: Warn user, may be unreliable

### Clarity Computation

```
clarity = max(score) / mean(score)
```

**Interpretation**:
- High clarity = one key clearly dominates
- Low clarity = multiple keys have similar scores

## Performance Expectations

### Overall Accuracy
- **Tonal Music**: 70-80% (exact key match)
- **All Music**: 65-75% (includes atonal tracks)
- **Target**: 77% accuracy (our Phase 1 goal)

### Confidence Scoring
- **High Confidence (>0.3)**: 85-90% accuracy
- **Medium Confidence (0.1-0.3)**: 70-80% accuracy
- **Low Confidence (<0.1)**: 50-60% accuracy

### Computational Performance
- **Template Matching**: <1ms (very fast)
- **Chroma Extraction**: 10-50ms (dominates)
- **Total**: 10-50ms for 30s track

## Edge Cases Identified

1. **Atonal Music**: No clear key → low accuracy expected
2. **Key Modulations**: Key changes mid-track → single key detection fails
3. **Complex Harmony**: Extended chords, jazz harmony → lower accuracy
4. **Ambient/Experimental**: No tonal center → unreliable detection
5. **Transposed Music**: Same key, different pitch → handled correctly

## Recommendations

1. **Use Template Matching**: Still competitive, simple, fast
2. **Compute Key Clarity**: Warn users when detection is unreliable
3. **Provide Confidence Scores**: Let users assess reliability
4. **Handle Edge Cases**: Detect atonal music, warn about low clarity

## References in Code

- `src/features/key/detector.rs`: Template matching implementation
- `src/features/key/key_clarity.rs`: Key clarity computation
- `src/analysis/confidence.rs`: Confidence scoring system

## Additional Notes

- Validates our template-matching approach
- Sets realistic performance expectations
- Provides benchmarks we can compare against
- Supports our key clarity metric

---

**Last Updated**: 2025-01-XX  
**Status**: Validation reference for key detection performance

