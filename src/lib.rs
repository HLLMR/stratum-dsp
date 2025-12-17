//! # Stratum DSP
//!
//! A professional-grade audio analysis engine for DJ applications, providing
//! accurate BPM detection, key detection, and beat tracking.
//!
//! ## Features
//!
//! - **BPM Detection**: Multi-method onset detection with autocorrelation and comb filterbank
//! - **Key Detection**: Chroma-based analysis with Krumhansl-Kessler template matching
//! - **Beat Tracking**: HMM-based beat grid generation with tempo drift correction
//! - **ML Refinement**: Optional ONNX model for edge case correction (Phase 2)
//!
//! ## Quick Start
//!
//! ```no_run
//! use stratum_dsp::{analyze_audio, AnalysisConfig};
//!
//! // Load audio samples (mono, f32, normalized)
//! let samples: Vec<f32> = vec![]; // Your audio data
//! let sample_rate = 44100;
//!
//! // Analyze
//! let result = analyze_audio(&samples, sample_rate, AnalysisConfig::default())?;
//!
//! println!("BPM: {:.2} (confidence: {:.2})", result.bpm, result.bpm_confidence);
//! println!("Key: {:?} (confidence: {:.2})", result.key, result.key_confidence);
//! # Ok::<(), stratum_dsp::AnalysisError>(())
//! ```
//!
//! ## Architecture
//!
//! The analysis pipeline follows this flow:
//!
//! ```text
//! Audio Input -> Preprocessing -> Feature Extraction -> Analysis -> ML Refinement -> Output
//! ```
//!
//! See the [module documentation](https://docs.rs/stratum-dsp) for details.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod analysis;
pub mod config;
pub mod error;
pub mod features;
pub mod io;
pub mod preprocessing;

#[cfg(feature = "ml")]
pub mod ml;

// Re-export main types
pub use analysis::result::{AnalysisResult, AnalysisMetadata, BeatGrid, Key, KeyType};
pub use analysis::confidence::{AnalysisConfidence, compute_confidence};
pub use config::AnalysisConfig;
pub use error::AnalysisError;

/// Main analysis function
///
/// Analyzes audio samples and returns comprehensive analysis results including
/// BPM, key, beat grid, and confidence scores.
///
/// # Arguments
///
/// * `samples` - Mono audio samples, normalized to [-1.0, 1.0]
/// * `sample_rate` - Sample rate in Hz (typically 44100 or 48000)
/// * `config` - Analysis configuration parameters
///
/// # Returns
///
/// `AnalysisResult` containing BPM, key, beat grid, and confidence metrics
///
/// # Errors
///
/// Returns `AnalysisError` if analysis fails (invalid input, processing error, etc.)
///
/// # Example
///
/// ```no_run
/// use stratum_dsp::{analyze_audio, AnalysisConfig};
///
/// let samples = vec![0.0f32; 44100 * 30]; // 30 seconds of silence
/// let result = analyze_audio(&samples, 44100, AnalysisConfig::default())?;
/// # Ok::<(), stratum_dsp::AnalysisError>(())
/// ```
pub fn analyze_audio(
    samples: &[f32],
    sample_rate: u32,
    config: AnalysisConfig,
) -> Result<AnalysisResult, AnalysisError> {
    use std::time::Instant;
    let start_time = Instant::now();
    
    log::debug!("Starting audio analysis: {} samples at {} Hz", samples.len(), sample_rate);
    
    if samples.is_empty() {
        return Err(AnalysisError::InvalidInput("Empty audio samples".to_string()));
    }
    
    if sample_rate == 0 {
        return Err(AnalysisError::InvalidInput("Invalid sample rate".to_string()));
    }
    
    // Phase 1A: Preprocessing
    let mut processed_samples = samples.to_vec();
    
    // 1. Normalization
    use preprocessing::normalization::{normalize, NormalizationConfig};
    let norm_config = NormalizationConfig {
        method: config.normalization,
        target_loudness_lufs: -14.0, // Default target
        max_headroom_db: 1.0,
    };
    let _loudness_metadata = normalize(&mut processed_samples, norm_config, sample_rate as f32)?;
    
    // 2. Silence detection and trimming
    use preprocessing::silence::{detect_and_trim, SilenceDetector};
    let silence_detector = SilenceDetector {
        threshold_db: config.min_amplitude_db,
        min_duration_ms: 500,
        frame_size: config.frame_size,
    };
    let (trimmed_samples, _silence_regions) = detect_and_trim(&processed_samples, sample_rate, silence_detector)?;
    
    if trimmed_samples.is_empty() {
        return Err(AnalysisError::ProcessingError("Audio is entirely silent after trimming".to_string()));
    }
    
    // Phase 1A: Onset Detection
    // Note: For now, we only have energy flux working directly on samples
    // Spectral methods (spectral_flux, hfc, hpss) require STFT which will be added in Phase 1B
    use features::onset::energy_flux::detect_energy_flux_onsets;
    
    let energy_onsets = detect_energy_flux_onsets(
        &trimmed_samples,
        config.frame_size,
        config.hop_size,
        -20.0, // threshold_db
    )?;
    
    log::debug!("Detected {} onsets using energy flux", energy_onsets.len());
    
    // Phase 1B: Period Estimation (BPM Detection)
    use features::period::estimate_bpm;
    let bpm_estimate = if energy_onsets.len() >= 2 {
        estimate_bpm(
            &energy_onsets,
            sample_rate,
            config.hop_size,
            config.min_bpm,
            config.max_bpm,
            config.bpm_resolution,
        )?
    } else {
        None
    };
    
    let (bpm, bpm_confidence) = if let Some(estimate) = bpm_estimate {
        (estimate.bpm, estimate.confidence)
    } else {
        log::warn!("Could not estimate BPM: insufficient onsets or estimation failed");
        (0.0, 0.0)
    };
    
    log::debug!("Estimated BPM: {:.2} (confidence: {:.3})", bpm, bpm_confidence);
    
    // Phase 1C: Beat Tracking
    let (beat_grid, grid_stability) = if bpm > 0.0 && energy_onsets.len() >= 2 {
        // Convert onsets from sample indices to seconds
        let onsets_seconds: Vec<f32> = energy_onsets.iter()
            .map(|&sample_idx| sample_idx as f32 / sample_rate as f32)
            .collect();
        
        // Generate beat grid using HMM Viterbi algorithm
        use features::beat_tracking::generate_beat_grid;
        match generate_beat_grid(bpm, bpm_confidence, &onsets_seconds, sample_rate) {
            Ok((grid, stability)) => {
                log::debug!("Beat grid generated: {} beats, {} downbeats, stability={:.3}",
                           grid.beats.len(), grid.downbeats.len(), stability);
                (grid, stability)
            }
            Err(e) => {
                log::warn!("Beat tracking failed: {}, using empty grid", e);
                (BeatGrid {
                    downbeats: vec![],
                    beats: vec![],
                    bars: vec![],
                }, 0.0)
            }
        }
    } else {
        log::debug!("Skipping beat tracking: BPM={:.2}, onsets={}", bpm, energy_onsets.len());
        (BeatGrid {
            downbeats: vec![],
            beats: vec![],
            bars: vec![],
        }, 0.0)
    };
    
    // Phase 1D: Key Detection
    let (key, key_confidence, key_clarity) = if trimmed_samples.len() >= config.frame_size {
        // Extract chroma vectors with configurable options
        use features::chroma::extractor::extract_chroma_with_options;
        use features::chroma::normalization::sharpen_chroma;
        use features::chroma::smoothing::smooth_chroma;
        use features::key::{detect_key, compute_key_clarity, KeyTemplates};
        
        match extract_chroma_with_options(
            &trimmed_samples,
            sample_rate,
            config.frame_size,
            config.hop_size,
            config.soft_chroma_mapping,
            config.soft_mapping_sigma,
        ) {
            Ok(mut chroma_vectors) => {
                // Apply chroma sharpening if enabled (power > 1.0)
                if config.chroma_sharpening_power > 1.0 {
                    for chroma in &mut chroma_vectors {
                        *chroma = sharpen_chroma(chroma, config.chroma_sharpening_power);
                    }
                    log::debug!("Applied chroma sharpening with power {:.2}", config.chroma_sharpening_power);
                }
                
                // Apply temporal smoothing (optional but recommended)
                if chroma_vectors.len() > 5 {
                    chroma_vectors = smooth_chroma(&chroma_vectors, 5);
                }
                
                // Detect key using Krumhansl-Kessler templates
                let templates = KeyTemplates::new();
                match detect_key(&chroma_vectors, &templates) {
                    Ok(key_result) => {
                        // Compute key clarity
                        let clarity = compute_key_clarity(&key_result.all_scores);
                        
                        log::debug!("Detected key: {:?}, confidence: {:.3}, clarity: {:.3}",
                                   key_result.key, key_result.confidence, clarity);
                        
                        (key_result.key, key_result.confidence, clarity)
                    }
                    Err(e) => {
                        log::warn!("Key detection failed: {}, using default", e);
                        (Key::Major(0), 0.0, 0.0)
                    }
                }
            }
            Err(e) => {
                log::warn!("Chroma extraction failed: {}, using default key", e);
                (Key::Major(0), 0.0, 0.0)
            }
        }
    } else {
        log::debug!("Skipping key detection: insufficient samples (need at least {} samples)",
                   config.frame_size);
        (Key::Major(0), 0.0, 0.0)
    };
    
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
    
    // Build confidence warnings
    let mut confidence_warnings = Vec::new();
    let mut flags = Vec::new();
    
    if bpm == 0.0 {
        confidence_warnings.push("BPM detection failed: insufficient onsets or estimation error".to_string());
    }
    if grid_stability < 0.5 {
        confidence_warnings.push(format!("Low beat grid stability: {:.2} (may indicate tempo variation)", grid_stability));
    }
    if key_confidence < 0.3 {
        confidence_warnings.push(format!("Low key detection confidence: {:.2} (may indicate ambiguous or atonal music)", key_confidence));
    }
    if key_clarity < 0.2 {
        confidence_warnings.push(format!("Low key clarity: {:.2} (track may be atonal or have weak tonality)", key_clarity));
        flags.push(crate::analysis::result::AnalysisFlag::WeakTonality);
    }
    
    // Phase 1E: Build result and compute comprehensive confidence scores
    let result = AnalysisResult {
        bpm,
        bpm_confidence,
        key,
        key_confidence,
        key_clarity,
        beat_grid,
        grid_stability,
        metadata: AnalysisMetadata {
            duration_seconds: trimmed_samples.len() as f32 / sample_rate as f32,
            sample_rate,
            processing_time_ms,
            algorithm_version: "0.1.0-alpha".to_string(),
            onset_method_consensus: if energy_onsets.is_empty() { 0.0 } else { 1.0 },
            methods_used: vec!["energy_flux".to_string(), "chroma_extraction".to_string(), "key_detection".to_string()],
            flags,
            confidence_warnings,
        },
    };
    
    // Phase 1E: Compute comprehensive confidence scores
    use analysis::confidence::compute_confidence;
    let confidence = compute_confidence(&result);
    log::debug!(
        "Analysis complete: BPM={:.2} (conf={:.3}), Key={:?} (conf={:.3}), Overall confidence={:.3}",
        result.bpm,
        confidence.bpm_confidence,
        result.key,
        confidence.key_confidence,
        confidence.overall_confidence
    );
    
    // Return result with Phase 1E confidence scoring integrated
    Ok(result)
}

