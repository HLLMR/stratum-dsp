//! Performance benchmarks for audio analysis

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stratum_audio_analysis::{analyze_audio, AnalysisConfig};

fn bench_analyze_audio(c: &mut Criterion) {
    // Generate synthetic audio (30 seconds at 44.1kHz)
    let samples: Vec<f32> = (0..44100 * 30)
        .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 44100.0).sin() * 0.5)
        .collect();
    
    let config = AnalysisConfig::default();
    
    c.bench_function("analyze_audio_30s", |b| {
        b.iter(|| {
            let _ = analyze_audio(black_box(&samples), black_box(44100), black_box(config.clone()));
        });
    });
}

criterion_group!(benches, bench_analyze_audio);
criterion_main!(benches);

