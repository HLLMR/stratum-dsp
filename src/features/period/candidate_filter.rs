//! BPM candidate filtering and merging
//!
//! Merges results from autocorrelation and comb filter, handles octave errors.
//!
//! # Algorithm
//!
//! This module merges BPM candidates from multiple estimation methods
//! (autocorrelation and comb filterbank) and handles common issues:
//!
//! 1. **Octave Errors**: Autocorrelation often detects 2x or 0.5x the true BPM
//!    - Detect when candidates are related by octave (2x or 0.5x)
//!    - Prefer candidates that both methods agree on
//!
//! 2. **Candidate Merging**: Combine results from both methods
//!    - Group candidates within tolerance (e.g., ±2 BPM)
//!    - Boost confidence when both methods agree
//!    - Track method agreement count
//!
//! 3. **Final Selection**: Return best candidates with confidence scores
//!
//! # Example
//!
//! ```no_run
//! use stratum_dsp::features::period::candidate_filter::merge_bpm_candidates;
//! use stratum_dsp::features::period::{BpmCandidate, BpmEstimate};
//!
//! let autocorr = vec![
//!     BpmCandidate { bpm: 120.0, confidence: 0.9 },
//!     BpmCandidate { bpm: 60.0, confidence: 0.7 }, // Octave error
//! ];
//! let comb = vec![
//!     BpmCandidate { bpm: 120.0, confidence: 0.85 },
//! ];
//! let merged = merge_bpm_candidates(autocorr, comb, 50.0)?;
//! // Returns BpmEstimate with bpm=120.0, confidence boosted, method_agreement=2
//! # Ok::<(), stratum_dsp::AnalysisError>(())
//! ```

use super::{BpmCandidate, BpmEstimate};

// EPSILON not currently used, but kept for consistency with other modules
const DEFAULT_BPM_TOLERANCE: f32 = 2.0; // ±2 BPM for grouping

/// Merge BPM candidates from multiple methods
///
/// Combines results from autocorrelation and comb filterbank, handling
/// octave errors and boosting confidence when methods agree.
///
/// # Arguments
///
/// * `autocorr` - Candidates from autocorrelation (sorted by confidence)
/// * `comb` - Candidates from comb filterbank (sorted by confidence)
/// * `octave_tolerance_cents` - Tolerance for octave detection (default: 50 cents ≈ 2.9%)
///
/// # Returns
///
/// Merged BPM estimates with method agreement, sorted by confidence (highest first)
///
/// # Errors
///
/// Returns `AnalysisError` if numerical errors occur during merging
///
/// # Algorithm Details
///
/// 1. **Octave Detection**: Check if candidates are related by 2x or 0.5x
///    - If autocorr finds 60 BPM and comb finds 120 BPM, prefer 120 BPM
///    - If autocorr finds 240 BPM and comb finds 120 BPM, prefer 120 BPM
///
/// 2. **Candidate Grouping**: Group candidates within ±2 BPM tolerance
///    - Average BPM within group
///    - Sum confidence scores
///    - Count method agreement
///
/// 3. **Confidence Boosting**: Boost confidence when both methods agree
///    - Single method: confidence = original
///    - Both methods: confidence = min(1.0, (conf1 + conf2) * 1.2)
pub fn merge_bpm_candidates(
    autocorr: Vec<BpmCandidate>,
    comb: Vec<BpmCandidate>,
    octave_tolerance_cents: f32,
) -> Result<Vec<BpmEstimate>, crate::error::AnalysisError> {
    log::debug!(
        "Merging BPM candidates: {} autocorr, {} comb, octave_tolerance={:.1} cents",
        autocorr.len(),
        comb.len(),
        octave_tolerance_cents
    );

    if autocorr.is_empty() && comb.is_empty() {
        return Ok(vec![]);
    }

    // Convert octave tolerance from cents to BPM ratio
    // 50 cents ≈ 2.9% ≈ 3.5 BPM at 120 BPM
    let octave_tolerance_ratio = (octave_tolerance_cents / 1200.0).exp2();

    // Step 1: Handle octave errors
    // Check if autocorr candidates are octave-related to comb candidates
    let mut autocorr_corrected = autocorr;
    let comb_corrected = comb;

    // Check for 2x octave errors (autocorr finds 2x BPM)
    for autocorr_candidate in &mut autocorr_corrected {
        for comb_candidate in &comb_corrected {
            let ratio = autocorr_candidate.bpm / comb_candidate.bpm;
            if (ratio - 2.0).abs() < octave_tolerance_ratio {
                // Autocorr found 2x, prefer comb's value
                log::debug!(
                    "Octave error detected: autocorr={:.2}, comb={:.2}, correcting to {:.2}",
                    autocorr_candidate.bpm,
                    comb_candidate.bpm,
                    comb_candidate.bpm
                );
                autocorr_candidate.bpm = comb_candidate.bpm;
                break;
            }
        }
    }

    // Check for 0.5x octave errors (autocorr finds 0.5x BPM)
    for autocorr_candidate in &mut autocorr_corrected {
        for comb_candidate in &comb_corrected {
            let ratio = comb_candidate.bpm / autocorr_candidate.bpm;
            if (ratio - 2.0).abs() < octave_tolerance_ratio {
                // Autocorr found 0.5x, prefer comb's value
                log::debug!(
                    "Octave error detected: autocorr={:.2}, comb={:.2}, correcting to {:.2}",
                    autocorr_candidate.bpm,
                    comb_candidate.bpm,
                    comb_candidate.bpm
                );
                autocorr_candidate.bpm = comb_candidate.bpm;
                break;
            }
        }
    }

    // Step 2: Group candidates within tolerance
    let mut groups: Vec<(f32, f32, u32)> = Vec::new(); // (bpm, total_confidence, method_count)

    // Process autocorr candidates
    for candidate in &autocorr_corrected {
        let mut found_group = false;
        for group in &mut groups {
            if (candidate.bpm - group.0).abs() <= DEFAULT_BPM_TOLERANCE {
                // Add to existing group
                let count = group.2;
                group.0 = (group.0 * (count as f32) + candidate.bpm) / ((count + 1) as f32);
                group.1 += candidate.confidence;
                group.2 += 1;
                found_group = true;
                break;
            }
        }
        if !found_group {
            groups.push((candidate.bpm, candidate.confidence, 1));
        }
    }

    // Process comb candidates
    for candidate in &comb_corrected {
        let mut found_group = false;
        for group in &mut groups {
            if (candidate.bpm - group.0).abs() <= DEFAULT_BPM_TOLERANCE {
                // Add to existing group
                let count = group.2;
                group.0 = (group.0 * (count as f32) + candidate.bpm) / ((count + 1) as f32);
                group.1 += candidate.confidence;
                group.2 += 1;
                found_group = true;
                break;
            }
        }
        if !found_group {
            groups.push((candidate.bpm, candidate.confidence, 1));
        }
    }

    // Step 3: Convert groups to BpmEstimate with confidence boosting
    let mut estimates: Vec<BpmEstimate> = groups
        .into_iter()
        .map(|(bpm, total_conf, method_count)| {
            // Boost confidence when both methods agree
            let confidence = if method_count >= 2 {
                // Both methods agree: boost by 20%
                (total_conf * 1.2).min(1.0)
            } else {
                // Single method: use original confidence
                total_conf.min(1.0)
            };

            BpmEstimate {
                bpm,
                confidence,
                method_agreement: method_count,
            }
        })
        .collect();

    // Sort by confidence (highest first)
    estimates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    log::debug!(
        "Merged to {} BPM estimates (best: {:.2} BPM, confidence: {:.3}, agreement: {})",
        estimates.len(),
        estimates.first().map(|e| e.bpm).unwrap_or(0.0),
        estimates.first().map(|e| e.confidence).unwrap_or(0.0),
        estimates.first().map(|e| e.method_agreement).unwrap_or(0)
    );

    Ok(estimates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_candidates_agreement() {
        // Both methods agree on 120 BPM
        let autocorr = vec![
            BpmCandidate {
                bpm: 120.0,
                confidence: 0.9,
            },
        ];
        let comb = vec![
            BpmCandidate {
                bpm: 120.0,
                confidence: 0.85,
            },
        ];

        let merged = merge_bpm_candidates(autocorr, comb, 50.0).unwrap();

        assert!(!merged.is_empty());
        let best = &merged[0];
        assert!((best.bpm - 120.0).abs() < 1.0);
        assert!(best.confidence > 0.9, "Confidence should be boosted");
        assert_eq!(best.method_agreement, 2);
    }

    #[test]
    fn test_merge_candidates_octave_error() {
        // Autocorr finds 2x (octave error), comb finds correct
        let autocorr = vec![
            BpmCandidate {
                bpm: 240.0, // 2x error
                confidence: 0.8,
            },
        ];
        let comb = vec![
            BpmCandidate {
                bpm: 120.0, // Correct
                confidence: 0.9,
            },
        ];

        let merged = merge_bpm_candidates(autocorr, comb, 50.0).unwrap();

        assert!(!merged.is_empty());
        let best = &merged[0];
        // Should prefer comb's value (120 BPM)
        assert!((best.bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_merge_candidates_octave_error_half() {
        // Autocorr finds 0.5x (octave error), comb finds correct
        let autocorr = vec![
            BpmCandidate {
                bpm: 60.0, // 0.5x error
                confidence: 0.8,
            },
        ];
        let comb = vec![
            BpmCandidate {
                bpm: 120.0, // Correct
                confidence: 0.9,
            },
        ];

        let merged = merge_bpm_candidates(autocorr, comb, 50.0).unwrap();

        assert!(!merged.is_empty());
        let best = &merged[0];
        // Should prefer comb's value (120 BPM)
        assert!((best.bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_merge_candidates_grouping() {
        // Candidates within tolerance should be grouped
        let autocorr = vec![
            BpmCandidate {
                bpm: 120.0,
                confidence: 0.8,
            },
            BpmCandidate {
                bpm: 121.0, // Within ±2 BPM
                confidence: 0.7,
            },
        ];
        let comb = vec![
            BpmCandidate {
                bpm: 120.5, // Within ±2 BPM
                confidence: 0.85,
            },
        ];

        let merged = merge_bpm_candidates(autocorr, comb, 50.0).unwrap();

        // Should group into one estimate
        assert_eq!(merged.len(), 1);
        let best = &merged[0];
        // Average should be around 120.2
        assert!((best.bpm - 120.0).abs() < 2.0);
        assert_eq!(best.method_agreement, 3); // All 3 candidates grouped
    }

    #[test]
    fn test_merge_candidates_empty() {
        let merged = merge_bpm_candidates(vec![], vec![], 50.0).unwrap();
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_candidates_single_method() {
        // Only autocorr has candidates
        let autocorr = vec![
            BpmCandidate {
                bpm: 120.0,
                confidence: 0.8,
            },
        ];
        let comb = vec![];

        let merged = merge_bpm_candidates(autocorr, comb, 50.0).unwrap();

        assert!(!merged.is_empty());
        let best = &merged[0];
        assert_eq!(best.method_agreement, 1);
        // Confidence should not be boosted (only one method)
        assert!(best.confidence <= 0.8);
    }

    #[test]
    fn test_merge_candidates_sorted() {
        let autocorr = vec![
            BpmCandidate {
                bpm: 120.0,
                confidence: 0.9,
            },
            BpmCandidate {
                bpm: 130.0,
                confidence: 0.7,
            },
        ];
        let comb = vec![
            BpmCandidate {
                bpm: 120.0,
                confidence: 0.85,
            },
        ];

        let merged = merge_bpm_candidates(autocorr, comb, 50.0).unwrap();

        // Should be sorted by confidence (highest first)
        for i in 1..merged.len() {
            assert!(
                merged[i - 1].confidence >= merged[i].confidence,
                "Should be sorted by confidence"
            );
        }
    }
}

