//! Configuration parameters for audio analysis

use crate::preprocessing::normalization::NormalizationMethod;

/// Analysis configuration parameters
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    // Preprocessing
    /// Silence detection threshold in dB (default: -40.0)
    /// Frames with RMS below this threshold are considered silent
    pub min_amplitude_db: f32,
    
    /// Normalization method to use (default: Peak)
    pub normalization: NormalizationMethod,
    
    // BPM detection
    /// Minimum BPM to consider (default: 60.0)
    pub min_bpm: f32,
    
    /// Maximum BPM to consider (default: 180.0)
    pub max_bpm: f32,
    
    /// BPM resolution for comb filterbank (default: 1.0)
    pub bpm_resolution: f32,
    
    // STFT parameters
    /// Frame size for STFT (default: 2048)
    pub frame_size: usize,
    
    /// Hop size for STFT (default: 512)
    pub hop_size: usize,
    
    // Key detection
    /// Center frequency for chroma extraction (default: 440.0 Hz, A4)
    pub center_frequency: f32,
    
    /// Enable soft chroma mapping (default: true)
    /// Soft mapping spreads frequency bins to neighboring semitones for robustness
    pub soft_chroma_mapping: bool,
    
    /// Soft mapping standard deviation in semitones (default: 0.5)
    /// Lower values = sharper mapping, higher values = more spread
    pub soft_mapping_sigma: f32,
    
    /// Chroma sharpening power (default: 1.0 = no sharpening, 1.5-2.0 recommended)
    /// Power > 1.0 emphasizes prominent semitones, improving key detection
    pub chroma_sharpening_power: f32,
    
    // ML refinement
    /// Enable ML refinement (requires ml feature)
    #[cfg(feature = "ml")]
    pub enable_ml_refinement: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_amplitude_db: -40.0,
            normalization: NormalizationMethod::Peak,
            min_bpm: 60.0,
            max_bpm: 180.0,
            bpm_resolution: 1.0,
            frame_size: 2048,
            hop_size: 512,
            center_frequency: 440.0,
            soft_chroma_mapping: true,
            soft_mapping_sigma: 0.5,
            chroma_sharpening_power: 1.0, // No sharpening by default (can be enabled with 1.5-2.0)
            #[cfg(feature = "ml")]
            enable_ml_refinement: false,
        }
    }
}

