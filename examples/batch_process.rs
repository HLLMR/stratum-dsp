//! Example: Batch process multiple audio files
//!
//! This example demonstrates how to process multiple files efficiently.

use stratum_dsp::{analyze_audio, AnalysisConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // TODO: Get list of audio files
    let audio_files: Vec<&Path> = vec![]; // Load from directory
    
    let config = AnalysisConfig::default();
    
    println!("Processing {} files...", audio_files.len());
    
    for (i, file_path) in audio_files.iter().enumerate() {
        println!("[{}/{}] Processing: {:?}", i + 1, audio_files.len(), file_path);
        
        // TODO: Load and analyze
        let samples: Vec<f32> = vec![];
        let sample_rate = 44100;
        
        match analyze_audio(&samples, sample_rate, config.clone()) {
            Ok(result) => {
                println!("  BPM: {:.2}, Key: {}", result.bpm, result.key.name());
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }
    
    Ok(())
}

