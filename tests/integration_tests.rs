//! Integration tests for audio analysis engine

use stratum_dsp::{analyze_audio, AnalysisConfig};
use std::path::PathBuf;

/// Load a WAV file and return (samples, sample_rate)
fn load_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Int => {
            let max_value = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>()
                .map(|s| s.map(|s| s as f32 / max_value))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    
    // Convert to mono if stereo
    let mono_samples = if spec.channels == 2 {
        samples.chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect()
    } else {
        samples
    };
    
    Ok((mono_samples, spec.sample_rate))
}

fn fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_120bpm_kick() {
        let path = fixture_path("120bpm_4bar.wav");
        let (samples, sample_rate) = load_wav(path.to_str().unwrap())
            .expect("Failed to load 120bpm_4bar.wav");
        
        let config = AnalysisConfig::default();
        let result = analyze_audio(&samples, sample_rate, config)
            .expect("Analysis should succeed");
        
        // Verify basic results
        assert!(result.metadata.duration_seconds > 7.0 && result.metadata.duration_seconds < 9.0);
        assert!(result.metadata.processing_time_ms > 0.0);
        assert_eq!(result.metadata.sample_rate, sample_rate);
        
        // Note: BPM detection not yet implemented (Phase 1B)
        // When implemented, we should check: result.bpm ≈ 120.0 ± 2 BPM
        println!("120 BPM test: duration={:.2}s, processing={:.2}ms", 
                 result.metadata.duration_seconds, result.metadata.processing_time_ms);
    }

    #[test]
    fn test_analyze_128bpm_kick() {
        let path = fixture_path("128bpm_4bar.wav");
        let (samples, sample_rate) = load_wav(path.to_str().unwrap())
            .expect("Failed to load 128bpm_4bar.wav");
        
        let config = AnalysisConfig::default();
        let result = analyze_audio(&samples, sample_rate, config)
            .expect("Analysis should succeed");
        
        // Verify basic results
        assert!(result.metadata.duration_seconds > 7.0 && result.metadata.duration_seconds < 8.0);
        assert!(result.metadata.processing_time_ms > 0.0);
        
        println!("128 BPM test: duration={:.2}s, processing={:.2}ms", 
                 result.metadata.duration_seconds, result.metadata.processing_time_ms);
    }

    #[test]
    fn test_analyze_cmajor_scale() {
        let path = fixture_path("cmajor_scale.wav");
        let (samples, sample_rate) = load_wav(path.to_str().unwrap())
            .expect("Failed to load cmajor_scale.wav");
        
        let config = AnalysisConfig::default();
        let result = analyze_audio(&samples, sample_rate, config)
            .expect("Analysis should succeed");
        
        // Verify basic results
        assert!(result.metadata.duration_seconds > 3.0 && result.metadata.duration_seconds < 5.0);
        
        // Note: Key detection not yet implemented (Phase 1D)
        // When implemented, we should check: result.key == Key::Major(0) (C major)
        println!("C major scale test: duration={:.2}s", result.metadata.duration_seconds);
    }

    #[test]
    fn test_silence_detection_and_trimming() {
        let path = fixture_path("mixed_silence.wav");
        let (samples, sample_rate) = load_wav(path.to_str().unwrap())
            .expect("Failed to load mixed_silence.wav");
        
        // Original duration should be ~15 seconds (5s silence + 5s audio + 5s silence)
        let original_duration = samples.len() as f32 / sample_rate as f32;
        assert!(original_duration > 14.0 && original_duration < 16.0);
        
        let config = AnalysisConfig::default();
        let result = analyze_audio(&samples, sample_rate, config)
            .expect("Analysis should succeed");
        
        // After silence trimming, duration should be ~5 seconds (just the audio content)
        // The analyze_audio function trims silence, so metadata.duration_seconds should reflect trimmed length
        assert!(result.metadata.duration_seconds > 4.0 && result.metadata.duration_seconds < 6.0,
                "Expected trimmed duration ~5s, got {:.2}s", result.metadata.duration_seconds);
        
        println!("Silence trimming test: original={:.2}s, trimmed={:.2}s", 
                 original_duration, result.metadata.duration_seconds);
    }

    #[test]
    fn test_analyze_audio_placeholder() {
        // Test with silence (edge case)
        let samples = vec![0.0f32; 44100 * 30]; // 30 seconds of silence
        let config = AnalysisConfig::default();
        
        // This should fail because audio is entirely silent after trimming
        let result = analyze_audio(&samples, 44100, config);
        assert!(result.is_err(), "Silent audio should return error");
        
        if let Err(e) = result {
            assert!(e.to_string().contains("silent"), 
                   "Error should mention silence: {}", e);
        }
    }
}
