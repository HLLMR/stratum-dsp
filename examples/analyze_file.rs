//! Example: Analyze a single audio file
//!
//! This example demonstrates how to analyze an audio file and print the results.

use stratum_audio_analysis::{analyze_audio, AnalysisConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();
    
    // TODO: Load audio file
    // For now, use placeholder
    let samples: Vec<f32> = vec![]; // Load from file
    let sample_rate = 44100;
    
    // Configure analysis
    let config = AnalysisConfig::default();
    
    // Analyze
    let result = analyze_audio(&samples, sample_rate, config)?;
    
    // Print results
    println!("Analysis Results:");
    println!("  BPM: {:.2} (confidence: {:.2})", result.bpm, result.bpm_confidence);
    println!("  Key: {} (confidence: {:.2})", result.key.name(), result.key_confidence);
    println!("  Grid stability: {:.2}", result.grid_stability);
    println!("  Processing time: {} ms", result.processing_time_ms);
    
    Ok(())
}

