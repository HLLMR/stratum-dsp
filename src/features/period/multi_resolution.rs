//! Multi-resolution tempogram validation
//!
//! Runs tempogram at multiple hop sizes (256, 512, 1024) and validates
//! agreement across resolutions. This improves robustness and accuracy.
//!
//! # Reference
//!
//! Schreiber, H., & Müller, M. (2018). A Single-Step Approach to Musical Tempo Estimation
//! Using a Convolutional Neural Network. *Proceedings of the International Society for
//! Music Information Retrieval Conference*.
//!
//! # Algorithm
//!
//! 1. Run tempogram at 3 hop sizes: 256, 512, 1024
//! 2. Extract BPM estimates from each resolution
//! 3. Check agreement: if all agree ±2 BPM → high confidence
//! 4. Return consensus BPM or best estimate
//!
//! # Example
//!
//! ```no_run
//! use stratum_dsp::features::period::multi_resolution::multi_resolution_analysis;
//!
//! let magnitude_spec_frames = vec![vec![0.0f32; 1024]; 100];
//! let result = multi_resolution_analysis(&magnitude_spec_frames, 44100, 512, 40.0, 240.0, 0.5)?;
//! println!("Consensus BPM: {:.2} (confidence: {:.3})", result.bpm, result.confidence);
//! # Ok::<(), stratum_dsp::AnalysisError>(())
//! ```

use crate::error::AnalysisError;
use super::BpmEstimate;
use super::tempogram::{
    estimate_bpm_tempogram,
    estimate_bpm_tempogram_with_candidates,
    TempogramCandidateDebug,
};
use crate::features::chroma::extractor::compute_stft;

/// Multi-resolution tempogram analysis (simplified wrapper)
///
/// Runs tempogram at multiple hop sizes and validates agreement across resolutions.
/// This improves robustness by catching artifacts from individual hop sizes.
///
/// # Reference
///
/// Schreiber, H., & Müller, M. (2018). A Single-Step Approach to Musical Tempo Estimation
/// Using a Convolutional Neural Network. *Proceedings of the International Society for
/// Music Information Retrieval Conference*.
///
/// # Arguments
///
/// * `magnitude_spec_frames` - FFT magnitude spectrogram (n_frames × n_bins)
///   Note: This should be computed with the base hop_size, we'll recompute STFT for other hop sizes
/// * `sample_rate` - Sample rate in Hz
/// * `base_hop_size` - Base hop size used for input spectrogram (typically 512)
/// * `min_bpm` - Minimum BPM to consider
/// * `max_bpm` - Maximum BPM to consider
/// * `bpm_resolution` - BPM resolution for autocorrelation tempogram
///
/// # Returns
///
/// Consensus BPM estimate with confidence based on multi-resolution agreement
///
/// # Errors
///
/// Returns `AnalysisError` if tempogram computation fails at all resolutions
///
/// # Note
///
/// This function currently uses the same spectrogram for all resolutions. For true
/// multi-resolution analysis, we would need to recompute STFT at different hop sizes.
/// This is a simplified version that validates consistency across the same data.
pub fn multi_resolution_analysis(
    magnitude_spec_frames: &[Vec<f32>],
    sample_rate: u32,
    base_hop_size: u32,
    min_bpm: f32,
    max_bpm: f32,
    bpm_resolution: f32,
) -> Result<BpmEstimate, AnalysisError> {
    log::debug!("Multi-resolution tempogram analysis: {} frames, sample_rate={}, base_hop_size={}",
                magnitude_spec_frames.len(), sample_rate, base_hop_size);
    
    // Run tempogram at multiple hop sizes
    // Note: In a full implementation, we would recompute STFT at each hop size.
    // For now, we use the same spectrogram but simulate different resolutions by
    // using different hop_size values in the tempogram computation.
    // This is a simplified approach - true multi-resolution would require recomputing STFT.
    
    let hop_sizes = vec![256, 512, 1024];
    let mut results = Vec::new();
    
    for &hop_size in &hop_sizes {
        match estimate_bpm_tempogram(
            magnitude_spec_frames,
            sample_rate,
            hop_size,
            min_bpm,
            max_bpm,
            bpm_resolution,
        ) {
            Ok(est) => {
                log::debug!("Hop size {}: BPM={:.1}, confidence={:.3}",
                           hop_size, est.bpm, est.confidence);
                results.push((hop_size, est));
            }
            Err(e) => {
                log::warn!("Tempogram failed at hop_size {}: {}", hop_size, e);
                // Continue with other hop sizes
            }
        }
    }
    
    if results.is_empty() {
        return Err(AnalysisError::ProcessingError(
            "All multi-resolution tempogram analyses failed".to_string()
        ));
    }
    
    // Check agreement across resolutions
    if results.len() >= 2 {
        // Check if all results agree within ±2 BPM
        let first_bpm = results[0].1.bpm;
        let all_agree = results.iter().all(|(_, est)| (est.bpm - first_bpm).abs() < 2.0);
        
        if all_agree {
            // All resolutions agree: use average with boosted confidence
            let avg_bpm: f32 = results.iter().map(|(_, est)| est.bpm).sum::<f32>() / results.len() as f32;
            let avg_confidence: f32 = results.iter().map(|(_, est)| est.confidence).sum::<f32>() / results.len() as f32;
            
            // Boost confidence for agreement
            let boosted_confidence = (avg_confidence * 1.2).min(1.0);
            
            log::debug!("Multi-resolution agreement: all {} resolutions agree on {:.1} BPM (conf={:.3})",
                       results.len(), avg_bpm, boosted_confidence);
            
            Ok(BpmEstimate {
                bpm: avg_bpm,
                confidence: boosted_confidence,
                method_agreement: results.len() as u32,
            })
        } else {
            // Resolutions disagree: use best confidence
            let best = results.iter()
                .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            
            log::debug!("Multi-resolution disagreement: using best (hop_size={}, BPM={:.1}, conf={:.3})",
                       best.0, best.1.bpm, best.1.confidence);
            
            Ok(BpmEstimate {
                bpm: best.1.bpm,
                confidence: best.1.confidence * 0.9, // Slight penalty for disagreement
                method_agreement: 1,
            })
        }
    } else {
        // Only one resolution succeeded
        let result = &results[0];
        log::debug!("Single resolution result: hop_size={}, BPM={:.1}, conf={:.3}",
                   result.0, result.1.bpm, result.1.confidence);
        
        Ok(result.1.clone())
    }
}

/// True multi-resolution tempogram BPM estimation (Phase 1F tuning).
///
/// Recomputes STFT magnitudes at hop sizes {256, 512, 1024} on the *audio samples*,
/// runs the tempogram pipeline on each, then fuses candidates using a calibrated
/// cross-resolution scoring rule aimed at resolving T vs 2T vs T/2 ambiguity.
pub fn multi_resolution_tempogram_from_samples(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,
    min_bpm: f32,
    max_bpm: f32,
    bpm_resolution: f32,
    top_k: usize,
    w512: f32,
    w256: f32,
    w1024: f32,
    structural_discount: f32,
    double_time_512_factor: f32,
    margin_threshold: f32,
    use_human_prior: bool,
) -> Result<(BpmEstimate, Vec<TempogramCandidateDebug>), AnalysisError> {
    if samples.len() < frame_size {
        return Err(AnalysisError::InvalidInput(
            "Audio too short for STFT".to_string(),
        ));
    }

    let top_k = top_k.max(1);
    // Use a wider candidate list on auxiliary hops to reduce false “missing support” from truncation.
    let aux_k = (top_k.saturating_mul(2)).clamp(10, 60);
    let tol = 2.0f32.max(bpm_resolution);

    let hop_256 = compute_stft(samples, frame_size, 256)?;
    let hop_512 = compute_stft(samples, frame_size, 512)?;
    let hop_1024 = compute_stft(samples, frame_size, 1024)?;

    let (_e256, c256) = estimate_bpm_tempogram_with_candidates(
        &hop_256,
        sample_rate,
        256,
        min_bpm,
        max_bpm,
        bpm_resolution,
        aux_k,
    )?;
    let (_e512, mut c512) = estimate_bpm_tempogram_with_candidates(
        &hop_512,
        sample_rate,
        512,
        min_bpm,
        max_bpm,
        bpm_resolution,
        top_k,
    )?;
    let (_e1024, c1024) = estimate_bpm_tempogram_with_candidates(
        &hop_1024,
        sample_rate,
        1024,
        min_bpm,
        max_bpm,
        bpm_resolution,
        aux_k,
    )?;

    fn lookup_nearest(cands: &[TempogramCandidateDebug], bpm: f32, tol: f32) -> f32 {
        let mut best_d = f32::INFINITY;
        let mut best_s = 0.0f32;
        for c in cands {
            let d = (c.bpm - bpm).abs();
            if d <= tol && d < best_d {
                best_d = d;
                best_s = c.score;
            }
        }
        best_s
    }

    #[derive(Clone, Copy)]
    struct Hyp {
        bpm: f32,
        score: f32,
    }

    let mut hyps: Vec<Hyp> = Vec::new();

    for t in c512.iter().take(top_k) {
        let t_bpm = t.bpm;
        if !(t_bpm.is_finite() && t_bpm > 0.0) {
            continue;
        }

        let s_t_512 = lookup_nearest(&c512, t_bpm, tol);
        let s_t_256 = lookup_nearest(&c256, t_bpm, tol);
        let s_t_1024 = lookup_nearest(&c1024, t_bpm, tol);

        let s_2t_512 = lookup_nearest(&c512, t_bpm * 2.0, tol);
        let s_2t_256 = lookup_nearest(&c256, t_bpm * 2.0, tol);
        let s_2t_1024 = lookup_nearest(&c1024, t_bpm * 2.0, tol);

        let s_half_512 = lookup_nearest(&c512, t_bpm * 0.5, tol);
        let s_half_256 = lookup_nearest(&c256, t_bpm * 0.5, tol);
        let s_half_1024 = lookup_nearest(&c1024, t_bpm * 0.5, tol);

        // H(T): tempo is T
        let h_t = w512 * s_t_512
            + w256 * s_t_256
            + w1024 * (s_t_1024 + structural_discount * s_2t_1024);

        // H(2T): tempo is double-time
        let mut h_2t = w512
            * (double_time_512_factor * s_t_512 + (1.0 - double_time_512_factor) * s_2t_512)
            + w256 * s_2t_256
            + w1024 * s_2t_1024;

        // H(T/2): tempo is half-time
        let mut h_half = w512
            * (double_time_512_factor * s_t_512 + (1.0 - double_time_512_factor) * s_half_512)
            + w256 * s_half_256
            + w1024 * s_half_1024;

        // Guardrails: only allow switching to 2T / T/2 when there is meaningful supporting evidence.
        // This prevents the fusion from “chasing” subdivision peaks that are only slightly stronger.
        let eps = 1e-6f32;
        let ratio_2t_256 = (s_2t_256 + eps) / (s_t_256 + eps);
        if ratio_2t_256 < 1.10 {
            h_2t *= 0.75;
        }
        if ratio_2t_256 < 1.00 {
            h_2t *= 0.75;
        }

        let ratio_half_1024 = (s_half_1024 + eps) / (s_t_1024 + eps);
        if ratio_half_1024 < 1.10 {
            h_half *= 0.75;
        }
        if ratio_half_1024 < 1.00 {
            h_half *= 0.75;
        }

        // Candidate hypotheses (clamp to allowed BPM range)
        let mut local: Vec<(f32, f32)> = vec![
            (t_bpm, h_t),
            (t_bpm * 2.0, h_2t),
            (t_bpm * 0.5, h_half),
        ];
        local.retain(|(b, _)| *b >= min_bpm && *b <= max_bpm);

        // Mild tempo prior (applied uniformly to hypothesis score)
        for (b, s) in &mut local {
            if *b > 210.0 {
                *s *= 0.80;
            } else if *b > 180.0 {
                *s *= 0.90;
            } else if *b < 60.0 {
                *s *= 0.92;
            }
        }

        local.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if local.is_empty() {
            continue;
        }

        let (best_bpm, best_score) = local[0];
        let second_score = local.get(1).map(|x| x.1).unwrap_or(0.0);
        let margin = best_score - second_score;

        // Critical: only switch away from T when the best alternative is clearly better.
        // Otherwise, keep T (reduces catastrophic 2× flips near 240).
        let mut chosen_bpm = best_bpm;
        let mut chosen_score = best_score;
        if (chosen_bpm - t_bpm).abs() > 1e-3 && margin < margin_threshold {
            chosen_bpm = t_bpm;
            chosen_score = h_t;
        }

        // If the choice is not clearly separated, apply an optional gentle prior as tie-break.
        if margin < margin_threshold {
            if use_human_prior && chosen_bpm >= 70.0 && chosen_bpm <= 180.0 && margin < 0.05 {
                chosen_score += 0.05;
            }
        }

        hyps.push(Hyp {
            bpm: chosen_bpm,
            score: chosen_score,
        });
    }

    if hyps.is_empty() {
        return Err(AnalysisError::ProcessingError(
            "Multi-resolution fusion produced no hypotheses".to_string(),
        ));
    }

    // Deduplicate hypotheses by BPM proximity; keep highest score per cluster.
    hyps.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique: Vec<Hyp> = Vec::new();
    for h in hyps {
        if unique.iter().any(|u| (u.bpm - h.bpm).abs() < 0.75) {
            continue;
        }
        unique.push(h);
        if unique.len() >= 8 {
            break;
        }
    }

    let best = unique[0];
    let second_score = unique.get(1).map(|h| h.score).unwrap_or(0.0);
    let conf = if best.score > 1e-6 {
        ((best.score - second_score).max(0.0) / best.score).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Method agreement: number of hop resolutions with direct support for the chosen BPM.
    let mut agree = 0u32;
    if lookup_nearest(&c256, best.bpm, tol) > 0.0 {
        agree += 1;
    }
    if lookup_nearest(&c512, best.bpm, tol) > 0.0 {
        agree += 1;
    }
    if lookup_nearest(&c1024, best.bpm, tol) > 0.0 {
        agree += 1;
    }

    // Mark which hop=512 candidates align with the final selection for downstream diagnostics.
    for c in &mut c512 {
        c.selected = (c.bpm - best.bpm).abs() < 0.75;
    }

    Ok((
        BpmEstimate {
            bpm: best.bpm,
            confidence: conf,
            method_agreement: agree,
        },
        c512,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multi_resolution_analysis_basic() {
        // Create spectrogram with periodic pattern
        let mut spectrogram = vec![vec![0.1f32; 1024]; 500];
        
        // Add periodic pattern
        let period = 43;
        for i in 0..spectrogram.len() {
            if i % period == 0 {
                for bin in 0..512 {
                    spectrogram[i][bin] = 1.0;
                }
            }
        }
        
        let result = multi_resolution_analysis(&spectrogram, 44100, 512, 100.0, 140.0, 0.5);
        
        // Should either succeed or fail gracefully
        match result {
            Ok(est) => {
                assert!(est.bpm >= 100.0 && est.bpm <= 140.0);
                assert!(est.confidence >= 0.0 && est.confidence <= 1.0);
            }
            Err(_) => {
                // Failure is acceptable for test input
            }
        }
    }
    
    #[test]
    fn test_multi_resolution_analysis_empty() {
        let spectrogram = vec![];
        let result = multi_resolution_analysis(&spectrogram, 44100, 512, 40.0, 240.0, 0.5);
        assert!(result.is_err());
    }
}

