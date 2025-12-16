//! Key detection modules
//!
//! Detect musical key using:
//! - Krumhansl-Kessler templates (24 keys)
//! - Template matching
//! - Key clarity scoring

pub mod detector;
pub mod key_clarity;
pub mod templates;

use crate::analysis::result::Key;

/// Key detection result
#[derive(Debug, Clone)]
pub struct KeyDetectionResult {
    /// Detected key
    pub key: Key,
    
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    
    /// All 24 key scores (ranked)
    pub all_scores: Vec<(Key, f32)>,
}

