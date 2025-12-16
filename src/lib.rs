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
//! ```
//! Audio Input → Preprocessing → Feature Extraction → Analysis → ML Refinement → Output
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
    
    // Phase 1B-1E: Not yet implemented
    // TODO: Period estimation (Phase 1B)
    // TODO: Beat tracking (Phase 1C)
    // TODO: Key detection (Phase 1D)
    
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
    
    // Return result with placeholder values for unimplemented features
    // This allows the API to work while we build out the rest
    Ok(AnalysisResult {
        bpm: 0.0, // TODO: Phase 1B
        bpm_confidence: 0.0,
        key: Key::Major(0), // TODO: Phase 1D (C major placeholder)
        key_confidence: 0.0,
        beat_grid: BeatGrid {
            downbeats: vec![],
            beats: vec![],
            bars: vec![],
        },
        grid_stability: 0.0,
        metadata: AnalysisMetadata {
            duration_seconds: trimmed_samples.len() as f32 / sample_rate as f32,
            sample_rate,
            processing_time_ms,
            algorithm_version: "0.1.0-alpha".to_string(),
            onset_method_consensus: if energy_onsets.is_empty() { 0.0 } else { 1.0 },
            methods_used: vec!["energy_flux".to_string()],
            flags: vec![],
            confidence_warnings: vec!["BPM detection not yet implemented (Phase 1B)".to_string(),
                                       "Key detection not yet implemented (Phase 1D)".to_string()],
        },
    })
}

