//! Example: Analyze a single audio file
//!
//! This example demonstrates how to analyze an audio file and print the results.
//! Can be used as a CLI tool for validation scripts.

use stratum_dsp::{analyze_audio, AnalysisConfig, compute_confidence};
use std::env;
use std::fs::File;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::i24;
use symphonia::default::get_probe;

/// Convert i24 to f32
fn i24_to_f32(sample: i24) -> f32 {
    // i24 uses the lower 24 bits of an i32
    let val = sample.inner();
    val as f32
}

fn decode_audio_file(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    // Open the media source
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    
    // Create a probe hint using the file extension
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    
    // Use the default probe to get the format
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    
    let probed = get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    
    // Get the default track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or("No supported audio tracks found")?;
    
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs().make(
        &track.codec_params,
        &DecoderOptions::default(),
    )?;
    
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let mut all_samples = Vec::new();
    
    // Decode all samples
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };
        
        if packet.track_id() != track_id {
            continue;
        }
        
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let channels = spec.channels.count();
                
                // Convert to f32 samples and mix to mono
                let samples_f32: Vec<f32> = match decoded {
                    AudioBufferRef::F32(buf) => {
                        if channels == 1 {
                            buf.chan(0).to_vec()
                        } else {
                            // Mix to mono
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| buf.chan(ch)[i])
                                        .sum::<f32>() / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        if channels == 1 {
                            buf.chan(0).iter().map(|&s| s as f32).collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| buf.chan(ch)[i] as f32)
                                        .sum::<f32>() / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        if channels == 1 {
                            buf.chan(0).iter().map(|&s| s as f32 / 32768.0).collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| buf.chan(ch)[i] as f32 / 32768.0)
                                        .sum::<f32>() / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        if channels == 1 {
                            buf.chan(0).iter().map(|&s| i24_to_f32(s) / 8388608.0).collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| i24_to_f32(buf.chan(ch)[i]) / 8388608.0)
                                        .sum::<f32>() / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        if channels == 1 {
                            buf.chan(0).iter().map(|&s| s as f32 / 2147483648.0).collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| buf.chan(ch)[i] as f32 / 2147483648.0)
                                        .sum::<f32>() / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        if channels == 1 {
                            buf.chan(0).iter().map(|&s| (s as f32 - 128.0) / 128.0).collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| (buf.chan(ch)[i] as f32 - 128.0) / 128.0)
                                        .sum::<f32>() / channels as f32
                                })
                                .collect()
                        }
                    }
                    _ => {
                        // Unsupported format
                        return Err("Unsupported audio format".into());
                    }
                };
                
                all_samples.extend(samples_f32);
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                // Skip decode errors (can happen with corrupted packets)
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        }
    }
    
    Ok((all_samples, sample_rate))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <audio_file> [--json] [--debug] [--no-preprocess] [--no-normalize] [--no-trim] [--no-onset-consensus] [--force-legacy-bpm] [--bpm-fusion] [--legacy-preferred-min X] [--legacy-preferred-max X] [--legacy-soft-min X] [--legacy-soft-max X] [--legacy-mul-preferred X] [--legacy-mul-soft X] [--legacy-mul-extreme X]", args[0]);
        std::process::exit(1);
    }
    
    let audio_file = &args[1];
    let json_output = args.contains(&"--json".to_string());
    let debug_mode = args.contains(&"--debug".to_string());
    let no_preprocess = args.contains(&"--no-preprocess".to_string());
    let no_normalize = args.contains(&"--no-normalize".to_string());
    let no_trim = args.contains(&"--no-trim".to_string());
    let no_onset_consensus = args.contains(&"--no-onset-consensus".to_string());
    let force_legacy_bpm = args.contains(&"--force-legacy-bpm".to_string());
    let bpm_fusion = args.contains(&"--bpm-fusion".to_string());

    fn arg_value(args: &[String], name: &str) -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    }

    fn parse_f32(args: &[String], name: &str) -> Option<f32> {
        arg_value(args, name).and_then(|v| v.parse::<f32>().ok())
    }
    
    // Initialize logger - set debug level if requested or if RUST_LOG is set
    let filter = if debug_mode {
        "debug"
    } else {
        "info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(filter)).init();
    
    // Decode audio file
    let (samples, sample_rate) = decode_audio_file(audio_file)?;
    
    if samples.is_empty() {
        eprintln!("ERROR: No audio samples decoded from file");
        std::process::exit(1);
    }
    
    // Configure analysis
    let mut config = AnalysisConfig::default();
    if no_preprocess {
        config.enable_normalization = false;
        config.enable_silence_trimming = false;
    }
    if no_normalize {
        config.enable_normalization = false;
    }
    if no_trim {
        config.enable_silence_trimming = false;
    }
    if no_onset_consensus {
        config.enable_onset_consensus = false;
    }
    if force_legacy_bpm {
        config.force_legacy_bpm = true;
    }
    if bpm_fusion {
        config.enable_bpm_fusion = true;
    }

    // Optional tuning overrides for legacy BPM guardrails (confidence multipliers by tempo range)
    if let Some(v) = parse_f32(&args, "--legacy-preferred-min") {
        config.legacy_bpm_preferred_min = v;
    }
    if let Some(v) = parse_f32(&args, "--legacy-preferred-max") {
        config.legacy_bpm_preferred_max = v;
    }
    if let Some(v) = parse_f32(&args, "--legacy-soft-min") {
        config.legacy_bpm_soft_min = v;
    }
    if let Some(v) = parse_f32(&args, "--legacy-soft-max") {
        config.legacy_bpm_soft_max = v;
    }
    if let Some(v) = parse_f32(&args, "--legacy-mul-preferred") {
        config.legacy_bpm_conf_mul_preferred = v;
    }
    if let Some(v) = parse_f32(&args, "--legacy-mul-soft") {
        config.legacy_bpm_conf_mul_soft = v;
    }
    if let Some(v) = parse_f32(&args, "--legacy-mul-extreme") {
        config.legacy_bpm_conf_mul_extreme = v;
    }
    
    if debug_mode {
        println!("=== DEBUG MODE ===");
        println!("Audio file: {}", audio_file);
        println!("Samples: {}, Sample rate: {} Hz", samples.len(), sample_rate);
        println!("Duration: {:.2} seconds", samples.len() as f32 / sample_rate as f32);
        println!();
    }
    
    // Analyze
    let result = match analyze_audio(&samples, sample_rate, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR: Analysis failed: {}", e);
            std::process::exit(1);
        }
    };
    
    // Compute confidence scores
    let confidence = compute_confidence(&result);
    
    // Output results
    if json_output {
        // JSON output for parsing by validation scripts
        println!("{{");
        println!("  \"bpm\": {:.2},", result.bpm);
        println!("  \"bpm_confidence\": {:.2},", confidence.bpm_confidence);
        println!("  \"key\": \"{}\",", result.key.name());
        println!("  \"key_confidence\": {:.2},", confidence.key_confidence);
        println!("  \"key_clarity\": {:.2},", result.key_clarity);
        println!("  \"grid_stability\": {:.2},", result.grid_stability);
        println!("  \"processing_time_ms\": {:.2}", result.metadata.processing_time_ms);
        println!("}}");
    } else {
        // Human-readable output
        println!("Analysis Results:");
        println!("  BPM: {:.2} (confidence: {:.2})", result.bpm, confidence.bpm_confidence);
        println!("  Key: {} (confidence: {:.2}, clarity: {:.2})", 
                 result.key.name(), 
                 confidence.key_confidence,
                 result.key_clarity);
        println!("  Grid stability: {:.2}", result.grid_stability);
        println!("  Processing time: {:.2} ms", result.metadata.processing_time_ms);
    }
    
    Ok(())
}
