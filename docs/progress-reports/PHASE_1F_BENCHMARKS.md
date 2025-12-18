# Phase 1F: Tempogram BPM Pivot - Benchmark Results

**Date**: 2025-12-18  
**Status**: ✅ **COMPLETE**  
**Benchmark Suite**: `benches/audio_analysis_bench.rs` + batch throughput (`examples/analyze_batch.rs`)

---

## Overview

This document captures performance benchmarks after Phase 1F tempogram pivot work and the addition of **batch-level parallel processing** (CPU-only). The primary goals were:

- Confirm the pipeline remains performant in single-track mode.
- Measure throughput improvements when processing many tracks in a batch.
- Identify and eliminate pathological multi-second outliers during batch runs.

---

## Key Findings (Executive Summary)

- ✅ **Batch throughput (40 real tracks)** improved from **~2.8 tracks/sec** (single worker) to **~21.3 tracks/sec** (CPU-1 workers), a **~7.7×** improvement in wall-clock throughput.
- ✅ A previous ~8–9s outlier class was explained by **HPSS percussive fallback** (expensive) being triggered in ambiguous cases.
- ✅ Default configuration was updated to keep the HPSS percussive fallback **opt-in** to prevent multi-second outliers during validation/batch runs.
- ✅ Criterion synthetic benchmark `analyze_audio_30s` remained stable at ~**203–208ms** for a 30s synthetic sine track (close to prior ~200ms; small fluctuations within typical noise for microbench reruns).

---

## Benchmark Configuration

### Criterion microbench (`cargo bench`)

- **Tool**: Criterion.rs
- **Benchmark**: `benches/audio_analysis_bench.rs`
- **Mode**: Release (optimized)
- **Synthetic test pattern**: 30-second sine wave at 440 Hz
- **Sample rate**: 44,100 Hz
- **Frame size**: 2048 samples
- **Hop size**: 512 samples

### Batch throughput test (real audio)

- **Tool**: `examples/analyze_batch.rs` (Rayon thread pool)
- **Input**: first 40 tracks of `validation-data/results/test_batch_20251218_103538.csv` (FMA small paths)
- **Metric**:
  - **Wall time** for the full batch (end-to-end)
  - **Per-track `processing_time_ms`** as reported by Stratum analysis metadata

---

## Results

### 1) Batch Throughput (Real Audio, 40 tracks)

Command pattern:

```bash
# single worker
.\target_alt\release\examples\analyze_batch.exe --jobs 1 <40 files...>

# CPU-1 workers (keeps one core free)
.\target_alt\release\examples\analyze_batch.exe --jobs (CPU-1) <40 files...>
```

**Run A: jobs=1**

- **Wall**: **14462 ms** for 40 tracks
- **Throughput**: **~2.77 tracks/sec**
- **Per-track `processing_time_ms`**:
  - **mean**: 334.40 ms
  - **p50**: 221.43 ms
  - **p90**: 508.34 ms
  - **min/max**: 115.79 ms / 525.15 ms

**Run B: jobs=CPU-1 (15 on this machine)**

- **Wall**: **1881 ms** for 40 tracks
- **Throughput**: **~21.27 tracks/sec**
- **Effective speedup**: **~7.7×** vs jobs=1
- **Per-track `processing_time_ms`**:
  - **mean**: 528.32 ms
  - **p50**: 388.63 ms
  - **p90**: 861.87 ms
  - **min/max**: 150.31 ms / 893.96 ms

**Interpretation**

- Batch parallelism is working as intended: **wall time drops sharply** as work is spread across cores.
- `processing_time_ms` increases under contention (expected), because many tracks are running concurrently (decode + STFT/FFT compete for CPU and memory bandwidth).
- For user-facing batch jobs (library scans/validation), the **wall-clock throughput** is the primary metric; this is a significant win.

---

### 2) Outlier Diagnosis: Multi-second Track Processing

Observed outliers were consistently in the **~8–9 second** range for some tracks (30s MP3s).

Root cause:

- The pipeline was triggering **HPSS percussive fallback**, which runs an HPSS decomposition over spectrogram frames and is extremely expensive.
- In the debug trace, HPSS was executed even in cases where the percussive result was **not selected**, meaning we paid the cost without benefit.

Mitigation (default behavior):

- The percussive fallback was changed to **opt-in** by default:
  - `AnalysisConfig::default().enable_tempogram_percussive_fallback` → **false**

After this change, a representative previous outlier track dropped from ~8.6s to **~0.55–0.80s** processing time.

---

### 3) Criterion Benchmark (Synthetic, 30s)

Command:

```bash
cargo bench --bench audio_analysis_bench
```

Key datapoint:

- **`analyze_audio_30s`**: ~**203–208 ms**

Interpretation:

- This is broadly consistent with prior measurements (~200 ms) and indicates no major regression in single-track synthetic performance. Minor deltas are within typical variance across reruns.

---

## Performance Summary

| Category | Scenario | Result |
|---------|----------|--------|
| Batch throughput | 40 real tracks, jobs=1 | ~2.77 tracks/sec |
| Batch throughput | 40 real tracks, jobs=CPU-1 | ~21.27 tracks/sec (**~7.7×**) |
| Pathological outliers | HPSS percussive fallback enabled by default | ~8–9s spikes (unacceptable for batch runs) |
| Pathological outliers | HPSS percussive fallback opt-in (default off) | Outliers removed in default batch runs |
| Synthetic microbench | `analyze_audio_30s` (Criterion) | ~203–208 ms |

---

## Recommendations

### Production / Batch Runs

- Prefer batch parallelism at **CPU-1 workers** (keep 1 core free for system responsiveness).
- Keep **HPSS percussive fallback disabled by default** for batch runs unless explicitly needed for targeted experiments.

### Future (Phase 2+)

- Consider adding internal per-track parallelism only if needed (e.g., parallelizing independent candidate-generation branches), but batch-level parallelism already provides strong throughput wins.
- GPU acceleration remains out of scope until Phase 2 ML integration.

---

## Conclusion

Phase 1F now has a **fast batch processing protocol** with strong throughput scaling across CPU cores. The main performance risk (HPSS percussive fallback) was identified and mitigated by defaulting it to opt-in, eliminating multi-second outliers during validation and bulk scans.

**Status**: ✅ **READY FOR BATCH VALIDATION / LIBRARY SCANS (CPU)**  

---

**Last Updated**: 2025-12-18  
**Benchmarked By**: AI Assistant  
**Status**: Complete


