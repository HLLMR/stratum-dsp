//! Configuration parameters for audio analysis

/// Analysis configuration parameters
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Minimum BPM to consider (default: 60.0)
    pub min_bpm: f32,
    
    /// Maximum BPM to consider (default: 180.0)
    pub max_bpm: f32,
    
    /// BPM resolution for comb filterbank (default: 1.0)
    pub bpm_resolution: f32,
    
    /// Frame size for STFT (default: 2048)
    pub frame_size: usize,
    
    /// Hop size for STFT (default: 512)
    pub hop_size: usize,
    
    /// Enable ML refinement (requires ml feature)
    #[cfg(feature = "ml")]
    pub enable_ml_refinement: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_bpm: 60.0,
            max_bpm: 180.0,
            bpm_resolution: 1.0,
            frame_size: 2048,
            hop_size: 512,
            #[cfg(feature = "ml")]
            enable_ml_refinement: false,
        }
    }
}

