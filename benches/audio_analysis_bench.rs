//! Performance benchmarks for audio analysis

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stratum_dsp::preprocessing::normalization::{normalize, NormalizationConfig, NormalizationMethod};
use stratum_dsp::preprocessing::silence::{detect_and_trim, SilenceDetector};
use stratum_dsp::features::onset::energy_flux::detect_energy_flux_onsets;
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
    full_analysis_benchmark
);
criterion_main!(benches);

