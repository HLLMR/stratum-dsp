//! Key detection algorithm
//!
//! Matches chroma distribution against Krumhansl-Kessler templates to detect
//! the musical key of an audio track.
//!
//! # Reference
//!
//! Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the Dynamic Changes in Perceived
//! Tonal Organization in a Spatial Representation of Musical Keys. *Psychological Review*,
//! 89(4), 334-368.

use crate::analysis::result::Key;
use crate::error::AnalysisError;
use super::{templates::KeyTemplates, KeyDetectionResult};

/// Detect musical key from chroma vectors
///
/// Averages chroma vectors across all frames, then computes dot product with
/// each of the 24 key templates. The key with the highest score is selected.
///
/// # Arguments
///
/// * `chroma_vectors` - Vector of 12-element chroma vectors (one per frame)
/// * `templates` - Key templates (Krumhansl-Kessler profiles)
///
/// # Returns
///
/// Key detection result with:
/// - Detected key (major or minor, 0-11)
/// - Confidence score (0.0-1.0)
/// - All 24 key scores (ranked)
///
/// # Errors
///
/// Returns `AnalysisError` if:
/// - Chroma vectors are empty
/// - Chroma vectors have incorrect dimensions
///
/// # Example
///
/// ```no_run
/// use stratum_dsp::features::key::{detector::detect_key, templates::KeyTemplates};
/// use stratum_dsp::features::chroma::extractor::extract_chroma;
///
/// let samples = vec![0.0f32; 44100 * 5];
/// let chroma_vectors = extract_chroma(&samples, 44100, 2048, 512)?;
/// let templates = KeyTemplates::new();
/// let result = detect_key(&chroma_vectors, &templates)?;
///
/// println!("Detected key: {:?}, confidence: {:.2}", result.key, result.confidence);
/// # Ok::<(), stratum_dsp::AnalysisError>(())
/// ```
pub fn detect_key(
    chroma_vectors: &[Vec<f32>],
    templates: &KeyTemplates,
) -> Result<KeyDetectionResult, AnalysisError> {
    log::debug!("Detecting key from {} chroma vectors", chroma_vectors.len());
    
    if chroma_vectors.is_empty() {
        return Err(AnalysisError::InvalidInput(
            "Empty chroma vectors".to_string()
        ));
    }
    
    // Validate chroma vector dimensions
    let n_semitones = chroma_vectors[0].len();
    if n_semitones != 12 {
        return Err(AnalysisError::InvalidInput(format!(
            "Chroma vectors must have 12 elements, got {}",
            n_semitones
        )));
    }
    
    for (i, chroma) in chroma_vectors.iter().enumerate() {
        if chroma.len() != 12 {
            return Err(AnalysisError::InvalidInput(format!(
                "Chroma vector at index {} has {} elements, expected 12",
                i, chroma.len()
            )));
        }
    }
    
    // Step 1: Average chroma across all frames
    let avg_chroma = average_chroma(chroma_vectors);
    
    // Step 2: Compute scores for all 24 keys
    let mut scores = Vec::with_capacity(24);
    
    // Major keys (0-11)
    for key_idx in 0..12 {
        let template = templates.get_major_template(key_idx);
        let score = dot_product(&avg_chroma, template);
        scores.push((Key::Major(key_idx), score));
    }
    
    // Minor keys (12-23, mapped to 0-11)
    for key_idx in 0..12 {
        let template = templates.get_minor_template(key_idx);
        let score = dot_product(&avg_chroma, template);
        scores.push((Key::Minor(key_idx), score));
    }
    
    // Step 3: Sort by score (highest first)
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Step 4: Select best key and compute confidence
    let (best_key, best_score) = scores[0];
    let second_score = if scores.len() > 1 { scores[1].1 } else { 0.0 };
    
    // Confidence: (best - second) / best
    // Higher difference = higher confidence
    let confidence = if best_score > 0.0 {
        ((best_score - second_score) / best_score).max(0.0).min(1.0)
    } else {
        0.0
    };
    
    // Step 5: Extract top N keys (default: top 3)
    let top_n = 3;
    let top_keys: Vec<(Key, f32)> = scores.iter()
        .take(top_n)
        .cloned()
        .collect();
    
    log::debug!("Detected key: {:?}, score: {:.4}, confidence: {:.4}",
                best_key, best_score, confidence);
    
    Ok(KeyDetectionResult {
        key: best_key,
        confidence,
        all_scores: scores,
        top_keys,
    })
}

/// Average chroma vectors across all frames
///
/// Computes element-wise average of all chroma vectors.
///
/// # Arguments
///
/// * `chroma_vectors` - Vector of 12-element chroma vectors
///
/// # Returns
///
/// 12-element averaged chroma vector
fn average_chroma(chroma_vectors: &[Vec<f32>]) -> Vec<f32> {
    if chroma_vectors.is_empty() {
        return vec![0.0; 12];
    }
    
    let n_frames = chroma_vectors.len() as f32;
    let mut avg = vec![0.0f32; 12];
    
    for chroma in chroma_vectors {
        for (i, &value) in chroma.iter().enumerate() {
            if i < 12 {
                avg[i] += value;
            }
        }
    }
    
    // Normalize by number of frames
    for x in &mut avg {
        *x /= n_frames;
    }
    
    // L2 normalize
    let norm: f32 = avg.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut avg {
            *x /= norm;
        }
    }
    
    avg
}

/// Compute dot product between two vectors
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Dot product (scalar)
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_key_empty() {
        let templates = KeyTemplates::new();
        let result = detect_key(&[], &templates);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_detect_key_basic() {
        let templates = KeyTemplates::new();
        
        // Create chroma vectors that match C major
        // C major template has high values at indices 0 (C), 4 (E), 7 (G)
        let mut chroma_vectors = Vec::new();
        for _ in 0..10 {
            let mut chroma = vec![0.0f32; 12];
            chroma[0] = 0.3; // C
            chroma[4] = 0.3; // E
            chroma[7] = 0.3; // G
            // Normalize
            let norm: f32 = chroma.iter().map(|&x| x * x).sum::<f32>().sqrt();
            for x in &mut chroma {
                *x /= norm;
            }
            chroma_vectors.push(chroma);
        }
        
        let result = detect_key(&chroma_vectors, &templates);
        assert!(result.is_ok());
        
        let detection = result.unwrap();
        assert!(detection.confidence >= 0.0 && detection.confidence <= 1.0);
        assert_eq!(detection.all_scores.len(), 24);
        
        // Best key should be C major (index 0)
        assert_eq!(detection.key, Key::Major(0));
        
        // Check top_keys is populated
        assert!(!detection.top_keys.is_empty());
        assert!(detection.top_keys.len() <= 3);
        assert_eq!(detection.top_keys[0].0, Key::Major(0));
    }
    
    #[test]
    fn test_detect_key_wrong_dimensions() {
        let templates = KeyTemplates::new();
        let chroma_vectors = vec![vec![0.0f32; 10]]; // Wrong size
        let result = detect_key(&chroma_vectors, &templates);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_average_chroma() {
        let chroma_vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        
        let avg = average_chroma(&chroma_vectors);
        assert_eq!(avg.len(), 12);
        
        // Should be approximately [0.5, 0.5, 0.0, ...] after normalization
        // (exact values depend on normalization)
        assert!(avg[0] > 0.0);
        assert!(avg[1] > 0.0);
    }
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
}
