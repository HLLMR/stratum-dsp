//! Performance benchmarks for audio analysis

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stratum_dsp::preprocessing::normalization::{normalize, NormalizationConfig, NormalizationMethod};
use stratum_dsp::preprocessing::silence::{detect_and_trim, SilenceDetector};
use stratum_dsp::features::onset::energy_flux::detect_energy_flux_onsets;
use stratum_dsp::features::period::autocorrelation::estimate_bpm_from_autocorrelation;
use stratum_dsp::features::period::comb_filter::{estimate_bpm_from_comb_filter, coarse_to_fine_search};
use stratum_dsp::{analyze_audio, AnalysisConfig};

/// Generate synthetic test audio (sine wave)
fn generate_test_audio(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
        .collect()
}

fn normalization_benchmarks(c: &mut Criterion) {
    let audio = generate_test_audio(44100 * 30); // 30 seconds
    
    let mut group = c.benchmark_group("normalization");
    
    // Peak normalization
    group.bench_function("normalize_peak_30s", |b| {
        b.iter(|| {
            let mut samples = black_box(audio.clone());
            let config = NormalizationConfig {
                method: NormalizationMethod::Peak,
                target_loudness_lufs: -14.0,
                max_headroom_db: 1.0,
            };
            let _ = normalize(&mut samples, config, 44100.0);
        });
    });
    
    // RMS normalization
    group.bench_function("normalize_rms_30s", |b| {
        b.iter(|| {
            let mut samples = black_box(audio.clone());
            let config = NormalizationConfig {
                method: NormalizationMethod::RMS,
                target_loudness_lufs: -14.0,
                max_headroom_db: 1.0,
            };
            let _ = normalize(&mut samples, config, 44100.0);
        });
    });
    
    // LUFS normalization
    group.bench_function("normalize_lufs_30s", |b| {
        b.iter(|| {
            let mut samples = black_box(audio.clone());
            let config = NormalizationConfig {
                method: NormalizationMethod::Loudness,
                target_loudness_lufs: -14.0,
                max_headroom_db: 1.0,
            };
            let _ = normalize(&mut samples, config, 44100.0);
        });
    });
    
    group.finish();
}

fn silence_detection_benchmarks(c: &mut Criterion) {
    let audio = generate_test_audio(44100 * 30); // 30 seconds
    let detector = SilenceDetector::default();
    
    c.bench_function("detect_and_trim_30s", |b| {
        b.iter(|| {
            let _ = detect_and_trim(black_box(&audio), black_box(44100), black_box(detector.clone()));
        });
    });
}

fn onset_detection_benchmarks(c: &mut Criterion) {
    let audio = generate_test_audio(44100 * 30); // 30 seconds
    
    c.bench_function("energy_flux_onsets_30s", |b| {
        b.iter(|| {
            let _ = detect_energy_flux_onsets(
                black_box(&audio),
                black_box(2048),
                black_box(512),
                black_box(-20.0),
            );
        });
    });
}

fn period_estimation_benchmarks(c: &mut Criterion) {
    // Generate synthetic onsets for 120 BPM at 44.1kHz
    let sample_rate = 44100;
    let hop_size = 512;
    let bpm = 120.0;
    let period_samples = (60.0 * sample_rate as f32) / bpm;
    
    let mut onsets = Vec::new();
    // Generate 8 beats worth of onsets (4 seconds)
    for beat in 0..8 {
        let sample = (beat as f32 * period_samples).round() as usize;
        onsets.push(sample);
    }
    
    let mut group = c.benchmark_group("period_estimation");
    
    // Autocorrelation BPM estimation
    group.bench_function("autocorrelation_bpm_8beats", |b| {
        b.iter(|| {
            let _ = estimate_bpm_from_autocorrelation(
                black_box(&onsets),
                black_box(sample_rate),
                black_box(hop_size),
                black_box(60.0),
                black_box(180.0),
            );
        });
    });
    
    // Comb filterbank BPM estimation (full resolution)
    group.bench_function("comb_filterbank_bpm_8beats", |b| {
        b.iter(|| {
            let _ = estimate_bpm_from_comb_filter(
                black_box(&onsets),
                black_box(sample_rate),
                black_box(hop_size),
                black_box(60.0),
                black_box(180.0),
                black_box(1.0),
            );
        });
    });
    
    // Coarse-to-fine search (optimized)
    group.bench_function("coarse_to_fine_bpm_8beats", |b| {
        b.iter(|| {
            let _ = coarse_to_fine_search(
                black_box(&onsets),
                black_box(sample_rate),
                black_box(hop_size),
                black_box(60.0),
                black_box(180.0),
                black_box(5.0),
            );
        });
    });
    
    group.finish();
}

fn full_analysis_benchmark(c: &mut Criterion) {
    let samples = generate_test_audio(44100 * 30); // 30 seconds
    let config = AnalysisConfig::default();
    
    c.bench_function("analyze_audio_30s", |b| {
        b.iter(|| {
            let _ = analyze_audio(black_box(&samples), black_box(44100), black_box(config.clone()));
        });
    });
}

criterion_group!(
    benches,
    normalization_benchmarks,
    silence_detection_benchmarks,
    onset_detection_benchmarks,
    period_estimation_benchmarks,
    full_analysis_benchmark
);
criterion_main!(benches);

