//! Period estimation modules
//!
//! Convert onset list to BPM candidates using:
//! - Autocorrelation
//! - Comb filterbank
//! - Candidate filtering and merging
//!
//! # Example
//!
//! ```no_run
//! use stratum_dsp::features::period::estimate_bpm;
//!
//! let onsets = vec![0, 11025, 22050, 33075]; // 120 BPM at 44.1kHz
//! if let Some(estimate) = estimate_bpm(&onsets, 44100, 512, 60.0, 180.0, 1.0)? {
//!     println!("BPM: {:.2} (confidence: {:.3})", estimate.bpm, estimate.confidence);
//! }
//! # Ok::<(), stratum_dsp::AnalysisError>(())
//! ```

pub mod autocorrelation;
pub mod candidate_filter;
pub mod comb_filter;
pub mod peak_picking;

pub use autocorrelation::estimate_bpm_from_autocorrelation;
pub use comb_filter::estimate_bpm_from_comb_filter;
pub use candidate_filter::merge_bpm_candidates;
pub use peak_picking::find_peaks;

/// BPM candidate with confidence
#[derive(Debug, Clone)]
pub struct BpmCandidate {
    /// BPM estimate
    pub bpm: f32,
    
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Final BPM estimate with method agreement
#[derive(Debug, Clone)]
pub struct BpmEstimate {
    /// BPM estimate
    pub bpm: f32,
    
    /// Confidence score
    pub confidence: f32,
    
    /// Number of methods that agree
    pub method_agreement: u32,
}

/// Estimate BPM from onset list using both autocorrelation and comb filterbank
///
/// This is the main public API for period estimation. It combines results from
/// both autocorrelation and comb filterbank methods, handling octave errors
/// and boosting confidence when methods agree.
///
/// # Arguments
///
/// * `onsets` - Onset times in samples
/// * `sample_rate` - Sample rate in Hz
/// * `hop_size` - Hop size used for onset detection
/// * `min_bpm` - Minimum BPM to consider (default: 60.0)
/// * `max_bpm` - Maximum BPM to consider (default: 180.0)
/// * `bpm_resolution` - BPM resolution for comb filterbank (default: 1.0)
///
/// # Returns
///
/// Best BPM estimate with confidence and method agreement, or None if no valid estimate found
///
/// # Errors
///
/// Returns `AnalysisError` if estimation fails
pub fn estimate_bpm(
    onsets: &[usize],
    sample_rate: u32,
    hop_size: usize,
    min_bpm: f32,
    max_bpm: f32,
    bpm_resolution: f32,
) -> Result<Option<BpmEstimate>, crate::error::AnalysisError> {
    use candidate_filter::merge_bpm_candidates;
    
    // Get candidates from both methods
    let autocorr_candidates = autocorrelation::estimate_bpm_from_autocorrelation(
        onsets,
        sample_rate,
        hop_size,
        min_bpm,
        max_bpm,
    )?;

    let comb_candidates = comb_filter::estimate_bpm_from_comb_filter(
        onsets,
        sample_rate,
        hop_size,
        min_bpm,
        max_bpm,
        bpm_resolution,
    )?;

    // Merge candidates
    let merged = merge_bpm_candidates(autocorr_candidates, comb_candidates, 50.0)?;

    // Return best estimate
    Ok(merged.into_iter().next())
}

