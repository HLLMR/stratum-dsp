//! Analysis result types

use serde::{Deserialize, Serialize};

/// Musical key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Key {
    /// Major key (0 = C, 1 = C#, ..., 11 = B)
    Major(u32),
    /// Minor key (0 = C, 1 = C#, ..., 11 = B)
    Minor(u32),
}

impl Key {
    /// Get key name as string (e.g., "C major", "A minor")
    pub fn name(&self) -> String {
        let note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
        match self {
            Key::Major(i) => format!("{} major", note_names[*i as usize % 12]),
            Key::Minor(i) => format!("{} minor", note_names[*i as usize % 12]),
        }
    }
}

/// Beat grid structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatGrid {
    /// Downbeat times (beat 1) in seconds
    pub downbeats: Vec<f32>,
    
    /// All beat times in seconds
    pub beats: Vec<f32>,
    
    /// Bar boundaries in seconds
    pub bars: Vec<f32>,
}

/// Analysis flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisFlag {
    /// Multiple BPM peaks equally strong
    MultimodalBpm,
    /// Low key clarity (atonal/ambiguous)
    WeakTonality,
    /// Track has tempo drift
    TempoVariation,
    /// Multiple onset interpretations
    OnsetDetectionAmbiguous,
}

/// Complete analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// BPM estimate
    pub bpm: f32,
    
    /// BPM confidence (0.0-1.0)
    pub bpm_confidence: f32,
    
    /// Detected key
    pub key: Key,
    
    /// Key confidence (0.0-1.0)
    pub key_confidence: f32,
    
    /// Beat grid
    pub beat_grid: BeatGrid,
    
    /// Grid stability (0.0-1.0)
    pub grid_stability: f32,
    
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Audio duration in seconds
    pub duration_seconds: f32,
    
    /// Sample rate in Hz
    pub sample_rate: u32,
    
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    
    /// Algorithm version
    pub algorithm_version: String,
    
    /// Onset method consensus score
    pub onset_method_consensus: f32,
    
    /// Methods used
    pub methods_used: Vec<String>,
    
    /// Analysis flags
    pub flags: Vec<AnalysisFlag>,
    
    /// Confidence warnings (low confidence, ambiguous results, etc.)
    pub confidence_warnings: Vec<String>,
}

// Re-export for convenience
pub use Key as KeyType;

