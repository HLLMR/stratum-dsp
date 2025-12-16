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
//! use stratum_audio_analysis::{analyze_audio, AnalysisConfig};
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
//! # Ok::<(), stratum_audio_analysis::AnalysisError>(())
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
//! See the [module documentation](https://docs.rs/stratum-audio-analysis) for details.

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
pub use analysis::result::{AnalysisResult, BeatGrid, Key, KeyType};
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
/// use stratum_audio_analysis::{analyze_audio, AnalysisConfig};
///
/// let samples = vec![0.0f32; 44100 * 30]; // 30 seconds of silence
/// let result = analyze_audio(&samples, 44100, AnalysisConfig::default())?;
/// # Ok::<(), stratum_audio_analysis::AnalysisError>(())
/// ```
pub fn analyze_audio(
    samples: &[f32],
    sample_rate: u32,
    _config: AnalysisConfig,
) -> Result<AnalysisResult, AnalysisError> {
    // TODO: Implement full analysis pipeline
    // Phase 1A: Preprocessing
    // Phase 1B: Onset detection
    // Phase 1C: Period estimation
    // Phase 1D: Beat tracking
    // Phase 1E: Key detection
    // Phase 2: ML refinement (if enabled)
    
    log::debug!("Starting audio analysis: {} samples at {} Hz", samples.len(), sample_rate);
    
    Err(AnalysisError::NotImplemented("Analysis pipeline not yet implemented".to_string()))
}

