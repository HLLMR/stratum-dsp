//! Period estimation modules
//!
//! Convert onset list to BPM candidates using:
//! - Autocorrelation
//! - Comb filterbank
//! - Candidate filtering and merging

pub mod autocorrelation;
pub mod candidate_filter;
pub mod comb_filter;
pub mod peak_picking;

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

