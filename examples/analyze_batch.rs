//! Example: Analyze multiple audio files in parallel
//!
//! Usage:
//!   cargo run --release --example analyze_batch -- [--jobs N] [--json] <file1> <file2> ...
//!
//! Notes:
//! - Parallelism is across files (batch-level). Each file analysis is still single-threaded.
//! - Default workers: (available CPU threads - 1), keeping one core free for the system.

use rayon::prelude::*;
use stratum_dsp::{analyze_audio, compute_confidence, AnalysisConfig};
use std::env;
use std::fs::File;
use std::time::Instant;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::sample::i24;
use symphonia::default::get_probe;

/// Convert i24 to f32.
fn i24_to_f32(sample: i24) -> f32 {
    let val = sample.inner();
    val as f32
}

fn decode_audio_file(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or("No supported audio tracks found")?;

    let track_id = track.id;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let mut all_samples: Vec<f32> = Vec::new();

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

                let samples_f32: Vec<f32> = match decoded {
                    AudioBufferRef::F32(buf) => {
                        if channels == 1 {
                            buf.chan(0).to_vec()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels).map(|ch| buf.chan(ch)[i]).sum::<f32>()
                                        / channels as f32
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
                                        .sum::<f32>()
                                        / channels as f32
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
                                        .sum::<f32>()
                                        / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        if channels == 1 {
                            buf.chan(0)
                                .iter()
                                .map(|&s| i24_to_f32(s) / 8388608.0)
                                .collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| i24_to_f32(buf.chan(ch)[i]) / 8388608.0)
                                        .sum::<f32>()
                                        / channels as f32
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
                                        .sum::<f32>()
                                        / channels as f32
                                })
                                .collect()
                        }
                    }
                    AudioBufferRef::U8(buf) => {
                        if channels == 1 {
                            buf.chan(0)
                                .iter()
                                .map(|&s| (s as f32 - 128.0) / 128.0)
                                .collect()
                        } else {
                            (0..buf.frames())
                                .map(|i| {
                                    (0..channels)
                                        .map(|ch| (buf.chan(ch)[i] as f32 - 128.0) / 128.0)
                                        .sum::<f32>()
                                        / channels as f32
                                })
                                .collect()
                        }
                    }
                    _ => {
                        // Unsupported format in this lightweight example.
                        return Err("Unsupported sample format".into());
                    }
                };

                all_samples.extend_from_slice(&samples_f32);
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                // Skip decode errors (can happen with corrupted packets).
                continue;
            }
            Err(e) => return Err(Box::new(e)),
        }
    }

    Ok((all_samples, sample_rate))
}

fn default_jobs() -> usize {
    let n = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    std::cmp::max(1, n.saturating_sub(1))
}

fn percentile(mut xs: Vec<f32>, p: f32) -> Option<f32> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((xs.len() - 1) as f32 * p.clamp(0.0, 1.0)).round() as usize;
    Some(xs[idx.min(xs.len() - 1)])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().skip(1).collect();

    let mut json = false;
    let mut jobs: Option<usize> = None;
    let mut paths: Vec<String> = Vec::new();

    while let Some(a) = args.first().cloned() {
        args.remove(0);
        match a.as_str() {
            "--json" => json = true,
            "--jobs" => {
                let v = args
                    .first()
                    .ok_or("--jobs requires a value")?
                    .parse::<usize>()?;
                args.remove(0);
                jobs = Some(std::cmp::max(1, v));
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: analyze_batch [--jobs N] [--json] <file1> <file2> ...\n\
                     \n\
                     --jobs N   Parallel workers (default: CPU-1)\n\
                     --json     Emit one JSON object per line (JSONL)\n"
                );
                return Ok(());
            }
            _ => paths.push(a),
        }
    }

    if paths.is_empty() {
        eprintln!("ERROR: Provide at least one audio file path. Use --help for usage.");
        std::process::exit(2);
    }

    let jobs = jobs.unwrap_or_else(default_jobs);
    eprintln!("Batch: {} files, jobs={}", paths.len(), jobs);

    let config = AnalysisConfig::default();

    let t0 = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .expect("Failed to build rayon thread pool");

    #[derive(Clone)]
    struct ItemOut {
        path: String,
        ok: bool,
        bpm: f32,
        bpm_conf: f32,
        key: String,
        key_conf: f32,
        processing_ms: f32,
        tempogram_multi_res_triggered: Option<bool>,
        tempogram_multi_res_used: Option<bool>,
        tempogram_percussive_triggered: Option<bool>,
        tempogram_percussive_used: Option<bool>,
        error: Option<String>,
    }

    let outs: Vec<ItemOut> = pool.install(|| {
        paths
            .par_iter()
            .map(|path| {
                let path_s = path.clone();
                let decoded = decode_audio_file(&path_s);
                match decoded {
                    Ok((samples, sr)) => {
                        let r = analyze_audio(&samples, sr, config.clone());
                        match r {
                            Ok(res) => {
                                let conf = compute_confidence(&res);
                                ItemOut {
                                    path: path_s,
                                    ok: true,
                                    bpm: res.bpm,
                                    bpm_conf: conf.bpm_confidence,
                                    key: res.key.name().to_string(),
                                    key_conf: conf.key_confidence,
                                    processing_ms: res.metadata.processing_time_ms,
                                    tempogram_multi_res_triggered: res.metadata.tempogram_multi_res_triggered,
                                    tempogram_multi_res_used: res.metadata.tempogram_multi_res_used,
                                    tempogram_percussive_triggered: res.metadata.tempogram_percussive_triggered,
                                    tempogram_percussive_used: res.metadata.tempogram_percussive_used,
                                    error: None,
                                }
                            }
                            Err(e) => ItemOut {
                                path: path_s,
                                ok: false,
                                bpm: 0.0,
                                bpm_conf: 0.0,
                                key: "".to_string(),
                                key_conf: 0.0,
                                processing_ms: 0.0,
                                tempogram_multi_res_triggered: None,
                                tempogram_multi_res_used: None,
                                tempogram_percussive_triggered: None,
                                tempogram_percussive_used: None,
                                error: Some(format!("analysis failed: {e}")),
                            },
                        }
                    }
                    Err(e) => ItemOut {
                        path: path_s,
                        ok: false,
                        bpm: 0.0,
                        bpm_conf: 0.0,
                        key: "".to_string(),
                        key_conf: 0.0,
                        processing_ms: 0.0,
                        tempogram_multi_res_triggered: None,
                        tempogram_multi_res_used: None,
                        tempogram_percussive_triggered: None,
                        tempogram_percussive_used: None,
                        error: Some(format!("decode failed: {e}")),
                    },
                }
            })
            .collect()
    });

    if json {
        for o in &outs {
            if o.ok {
                println!(
                    "{{\"file\":{},\"bpm\":{:.2},\"bpm_confidence\":{:.4},\"key\":{},\"key_confidence\":{:.4},\"processing_time_ms\":{:.2},\"tempogram_multi_res_triggered\":{},\"tempogram_multi_res_used\":{},\"tempogram_percussive_triggered\":{},\"tempogram_percussive_used\":{}}}",
                    serde_json::to_string(&o.path).unwrap(),
                    o.bpm,
                    o.bpm_conf,
                    serde_json::to_string(&o.key).unwrap(),
                    o.key_conf,
                    o.processing_ms,
                    o.tempogram_multi_res_triggered.map(|v| v.to_string()).unwrap_or("null".to_string()),
                    o.tempogram_multi_res_used.map(|v| v.to_string()).unwrap_or("null".to_string()),
                    o.tempogram_percussive_triggered.map(|v| v.to_string()).unwrap_or("null".to_string()),
                    o.tempogram_percussive_used.map(|v| v.to_string()).unwrap_or("null".to_string()),
                );
            } else {
                println!(
                    "{{\"file\":{},\"error\":{}}}",
                    serde_json::to_string(&o.path).unwrap(),
                    serde_json::to_string(o.error.as_deref().unwrap_or("unknown error")).unwrap()
                );
            }
        }
    } else {
        for (idx, o) in outs.iter().enumerate() {
            if o.ok {
                println!(
                    "[{}/{}] {}: BPM={:.2} (conf={:.3}) Key={} (conf={:.3}) time={:.2}ms",
                    idx + 1,
                    outs.len(),
                    o.path,
                    o.bpm,
                    o.bpm_conf,
                    o.key,
                    o.key_conf,
                    o.processing_ms
                );
            } else {
                println!(
                    "[{}/{}] {}: ERROR: {}",
                    idx + 1,
                    outs.len(),
                    o.path,
                    o.error.as_deref().unwrap_or("unknown error")
                );
            }
        }
    }

    let ok_times: Vec<f32> = outs.iter().filter(|o| o.ok).map(|o| o.processing_ms).collect();
    let wall = t0.elapsed();
    let wall_ms = wall.as_secs_f64() * 1000.0;

    eprintln!(
        "Done: ok={}/{} wall={:.0}ms",
        ok_times.len(),
        outs.len(),
        wall_ms
    );
    if !ok_times.is_empty() {
        let mean = ok_times.iter().sum::<f32>() / ok_times.len() as f32;
        let p50 = percentile(ok_times.clone(), 0.50).unwrap_or(mean);
        let p90 = percentile(ok_times.clone(), 0.90).unwrap_or(mean);
        let min = ok_times.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = ok_times.iter().cloned().fold(0.0, f32::max);
        eprintln!(
            "processing_time_ms: mean={:.2} p50={:.2} p90={:.2} min={:.2} max={:.2}",
            mean, p50, p90, min, max
        );
    }

    Ok(())
}


