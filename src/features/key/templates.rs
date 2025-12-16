//! Krumhansl-Kessler key templates
//!
//! Defines tonal profiles for 24 keys (12 major + 12 minor).

/// Key templates for all 24 keys
#[derive(Debug, Clone)]
pub struct KeyTemplates {
    /// Major key templates (12 keys: C, C#, D, ..., B)
    pub major: [Vec<f32>; 12],
    
    /// Minor key templates (12 keys: C, C#, D, ..., B)
    pub minor: [Vec<f32>; 12],
}

impl KeyTemplates {
    /// Create new key templates with Krumhansl-Kessler profiles
    pub fn new() -> Self {
        // TODO: Initialize with Krumhansl-Kessler profiles
        // See audio-analysis-engine-spec.md Section 2.6.1
        // C major profile: [0.15, 0.01, 0.12, 0.01, 0.13, 0.11, 0.01, 0.13, 0.01, 0.12, 0.01, 0.10]
        // A minor profile: [0.10, 0.01, 0.15, 0.01, 0.12, 0.13, 0.01, 0.11, 0.13, 0.01, 0.12, 0.01]
        
        Self {
            major: [
                vec![0.0; 12], // TODO: Fill with actual profiles
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
            ],
            minor: [
                vec![0.0; 12], // TODO: Fill with actual profiles
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
                vec![0.0; 12],
            ],
        }
    }
}

impl Default for KeyTemplates {
    fn default() -> Self {
        Self::new()
    }
}

