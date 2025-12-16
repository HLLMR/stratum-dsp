//! Beat tracking modules
//!
//! Generate precise beat grid from BPM estimate:
//! - HMM Viterbi algorithm
//! - Bayesian tempo tracking

pub mod bayesian;
pub mod hmm;

/// Beat position in a bar
#[derive(Debug, Clone)]
pub struct BeatPosition {
    /// Beat index within bar (0, 1, 2, 3)
    pub beat_index: u32,
    
    /// Time in seconds
    pub time_seconds: f32,
    
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

