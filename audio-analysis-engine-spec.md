# Stratum Audio Analysis Engine - Technical Specification

## Executive Summary

**Project**: Stratum Audio Analysis Engine (working name: `stratum-audio-analysis`)

**Scope**: Pure-Rust hybrid classical DSP + ML-refined audio analysis engine for professional DJ-grade BPM and key detection, with extensibility for future music analysis features (energy, mood, genre, etc.).

**Status**: New crate, separate development lifecycle, eventual integration into `stratum-desktop`

**Why separate?**: 
- Reusable across multiple Stratum projects (desktop, web, cloud)
- Open-source opportunity (competitive moat + community contribution)
- Independent testing & iteration
- Clear separation of concerns
- Can be published to crates.io as reference implementation

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 High-Level Pipeline

```
Audio Input (PCM samples)
        ↓
    [Preprocessing]
    • Normalization
    • Silence detection
    • Channel mixing (mono for analysis)
        ↓
    [Feature Extraction Layer]
    ├─→ Onset Detection (Multi-method consensus)
    │   ├─ Energy Flux
    │   ├─ Spectral Flux
    │   ├─ High-Frequency Content
    │   └─ Percussive Sharpness (via HPSS)
    │
    ├─→ Period Estimation
    │   ├─ Autocorrelation
    │   ├─ Comb Filterbank
    │   └─ Peak Confidence Scoring
    │
    ├─→ Beat Tracking
    │   ├─ HMM with Viterbi algorithm
    │   └─ Bayesian grid drift correction
    │
    ├─→ Chroma Extraction
    │   ├─ FFT-based harmonic analysis
    │   └─ Chroma vector normalization
    │
    └─→ Key Detection
        ├─ Krumhansl-Kessler template matching
        ├─ Key clarity scoring
        └─ 24-key confidence distribution
        ↓
    [Confidence Scoring]
    • BPM multimodal detection
    • Key clarity indicator
    • Grid stability measure
    • Feature coherence
        ↓
    [ML Refinement Layer] (Optional, Phase 2)
    • ONNX model inference
    • Edge case detection
    • Confidence boosting
        ↓
    Output (Analysis Result)
    {
        bpm: f32,
        bpm_confidence: f32,
        key: Key,
        key_confidence: f32,
        grid: BeatGrid,
        grid_stability: f32,
        metadata: AnalysisMetadata
    }
```

### 1.2 Module Structure

```
stratum-audio-analysis/
├── src/
│   ├── lib.rs                    # Public API
│   ├── error.rs                  # Error types
│   │
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── normalization.rs      # Peak normalization, loudness
│   │   ├── silence.rs            # Silence detection & trimming
│   │   └── channel_mixer.rs      # Stereo → mono, channel mixing
│   │
│   ├── features/
│   │   ├── mod.rs
│   │   ├── onset/
│   │   │   ├── mod.rs
│   │   │   ├── energy_flux.rs    # Energy-based onset detection
│   │   │   ├── spectral_flux.rs  # Spectral flux onsets
│   │   │   ├── hfc.rs            # High-frequency content
│   │   │   ├── hpss.rs           # Harmonic-percussive source separation
│   │   │   └── consensus.rs      # Multi-method voting
│   │   │
│   │   ├── period/
│   │   │   ├── mod.rs
│   │   │   ├── autocorrelation.rs    # Autocorrelation-based BPM
│   │   │   ├── comb_filter.rs        # Multi-comb BPM estimator
│   │   │   ├── peak_picking.rs       # Robust peak detection
│   │   │   └── candidate_filter.rs   # Tempo candidate refinement
│   │   │
│   │   ├── beat_tracking/
│   │   │   ├── mod.rs
│   │   │   ├── hmm.rs               # HMM Viterbi beat tracker
│   │   │   └─── bayesian.rs         # Bayesian tempo & grid tracking
│   │   │
│   │   ├── chroma/
│   │   │   ├── mod.rs
│   │   │   ├── extractor.rs         # Chroma vector computation
│   │   │   ├── normalization.rs     # Chroma normalization strategies
│   │   │   └── smoothing.rs         # Temporal chroma smoothing
│   │   │
│   │   └── key/
│   │       ├── mod.rs
│   │       ├── templates.rs         # Krumhansl-Kessler templates (24-key)
│       ├── detector.rs          # Key detection algorithm
│   │       ├── key_clarity.rs       # Key confidence estimation
│   │       └── profiles.rs          # Tonal profiles per track
│   │
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── confidence.rs        # Confidence scoring & validation
│   │   ├── metadata.rs          # Analysis metadata structures
│   │   └── result.rs            # Final result types
│   │
│   ├── ml/ (Phase 2)
│   │   ├── mod.rs
│   │   ├── onnx_model.rs        # ONNX model loading & inference
│   │   ├── refinement.rs        # Confidence refinement pipeline
│   │   └── edge_cases.rs        # Edge case detection & correction
│   │
│   ├── io/
│   │   ├── mod.rs
│   │   ├── decoder.rs           # Audio decoding (via Symphonia)
│   │   └── sample_buffer.rs     # Sample windowing & buffering
│   │
│   └── config.rs                # Algorithm parameters & tuning
│
├── tests/
│   ├── integration_tests.rs
│   ├── fixtures/
│   │   └── test_tracks/         # Ground truth test audio
│   └── benchmarks.rs
│
├── models/ (Phase 2)
│   └── refinement.onnx          # Pre-trained ML model
│
├── Cargo.toml
├── README.md
└── TECHNICAL.md                 # This file
```

---

## 2. COMPONENT SPECIFICATIONS

### 2.1 Preprocessing Module

#### 2.1.1 Normalization (`normalization.rs`)

**Purpose**: Normalize audio to consistent loudness level without clipping.

**Algorithm**:
```rust
pub struct NormalizationConfig {
    pub target_loudness_lufs: f32,  // -14 LUFS (YouTube standard)
    pub max_headroom_db: f32,        // 1.0 dB peak margin
    pub method: NormalizationMethod,
}

pub enum NormalizationMethod {
    Peak,                           // Simple peak normalization
    RMS,                            // RMS-based normalization
    Loudness(LoudnessStandard),     // ITU-R BS.1770-4 (LUFS)
}
```

**Implementation**:
1. **Peak method** (fast):
   - Find absolute peak sample value
   - Scale: `samples *= target_level / peak`
   - Cost: O(n) single pass

2. **LUFS method** (accurate):
   - Gate at -70 LUFS (ITU standard)
   - Apply K-weighting filter
   - Integrate loudness over 400ms blocks
   - Target adjustment: `gain_db = target_lufs - measured_lufs`

**Output**: Normalized PCM samples, loudness metadata

---

#### 2.1.2 Silence Detection (`silence.rs`)

**Purpose**: Detect and trim silence/padding at track boundaries.

**Algorithm**:
```rust
pub struct SilenceDetector {
    threshold_db: f32,              // -40 dB default
    min_duration_ms: u32,           // 500 ms minimum gap
    frame_size: usize,              // 2048 samples typical
}
```

**Implementation**:
1. Frame audio into chunks
2. Compute RMS per frame
3. Mark frames below threshold as silent
4. Merge consecutive silent frames (< 500ms)
5. Trim leading/trailing silence, mark interior gaps

**Output**: Silence-trimmed samples, silence map (for visualization)

---

#### 2.1.3 Channel Mixer (`channel_mixer.rs`)

**Purpose**: Convert stereo to mono for analysis (DSP algorithms need monophonic input).

**Algorithm**:
```rust
pub enum ChannelMixMode {
    Mono,                   // Downmix: (L + R) / 2
    MidSide,                // (L + R)/2 (ignores side info)
    Dominant,               // Keep louder channel
    Center,                 // Center image only
}
```

**Implementation**:
- Simple average for mono: `mono[i] = (left[i] + right[i]) / 2.0`
- Store original stereo for parallel key detection if needed

**Output**: Monophonic PCM, sample rate

---

### 2.2 Onset Detection Module

#### 2.2.1 Overview

**Purpose**: Identify percussive transients (likely beat locations).

**Why multiple methods?**: Different audio characteristics break different detectors:
- Clean EDM: All methods work equally
- Heavy compression: Energy flux fails, spectral flux still works
- Live recordings: HFC better than energy
- Breakbeats: HPSS (percussive) superior to spectral

**Multi-method consensus** (Section 2.2.5) votes across all 4 methods.

---

#### 2.2.2 Energy Flux (`energy_flux.rs`)

**Algorithm**: Detect peaks in frame-by-frame energy derivative.

```
1. Divide audio into frames (frame_size=2048, hop_size=512)
2. Compute RMS energy per frame: E[n] = sqrt(mean(frame[n]²))
3. Compute derivative: E_flux[n] = max(0, E[n] - E[n-1])
4. Threshold (relative to max): peaks where flux > 0.3 * max(flux)
5. Return: Onset times (in samples)
```

**Pseudocode**:
```rust
pub fn detect_energy_flux_onsets(
    samples: &[f32],
    frame_size: usize,
    hop_size: usize,
    threshold_db: f32,
) -> Vec<usize> {
    let energy = compute_frame_energy(&samples, frame_size, hop_size);
    let flux = energy.windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .collect::<Vec<_>>();
    
    let threshold = threshold_db_to_linear(threshold_db) * flux.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    flux.iter()
        .enumerate()
        .filter(|(_, &f)| f > threshold)
        .map(|(i, _)| i * hop_size)
        .collect()
}
```

**Strengths**: Fast, robust to quiet passages
**Weaknesses**: Fails on highly compressed audio, smooth pads

---

#### 2.2.3 Spectral Flux (`spectral_flux.rs`)

**Algorithm**: Detect changes in magnitude spectrogram across frequency bins.

```
1. Compute STFT (magnitude only) with frame_size=2048, hop=512
2. Normalize magnitude to [0, 1] per frame
3. Compute L2 distance between consecutive frames:
   flux[n] = sqrt(sum((M[n] - M[n-1])²))
4. Threshold and peak-pick
5. Return: Onset times
```

**Why this works**: Captures spectral changes (e.g., filter sweeps, new instrument entries) that pure energy misses.

**Pseudocode**:
```rust
pub fn detect_spectral_flux_onsets(
    fft_magnitudes: &[Vec<f32>],  // Shape: (n_frames, n_bins)
    threshold_percentile: f32,     // e.g., 0.8 for 80th percentile
) -> Vec<usize> {
    let mut flux = Vec::new();
    for i in 1..fft_magnitudes.len() {
        let prev = &fft_magnitudes[i - 1];
        let curr = &fft_magnitudes[i];
        
        let dist = prev.iter().zip(curr.iter())
            .map(|(p, c)| (c - p).powi(2))
            .sum::<f32>()
            .sqrt();
        flux.push(dist);
    }
    
    let threshold = percentile(&flux, threshold_percentile);
    peak_pick(&flux, threshold)
}
```

**Strengths**: Catches spectral shifts, doesn't depend on loudness
**Weaknesses**: Can create false positives on smooth frequency sweeps

---

#### 2.2.4 High-Frequency Content (`hfc.rs`)

**Algorithm**: Detect energy concentration in high frequencies (typical of percussive attacks).

```
1. Compute STFT magnitude
2. Weight higher frequencies more heavily:
   HFC[n] = sum(i * magnitude[i]) for i in 0..n_bins
3. Threshold and peak-pick
4. Return: Onset times
```

**Why this works**: Percussion (kicks, snares, hi-hats) has energy burst in 2-10kHz range.

**Pseudocode**:
```rust
pub fn detect_hfc_onsets(
    fft_magnitudes: &[Vec<f32>],
    sample_rate: u32,
    threshold_percentile: f32,
) -> Vec<usize> {
    let nyquist = sample_rate / 2;
    let n_bins = fft_magnitudes[0].len();
    
    let mut hfc = Vec::new();
    for frame in fft_magnitudes {
        let hf_content = frame.iter().enumerate()
            .map(|(i, &mag)| {
                let freq = (i as f32 / n_bins as f32) * nyquist as f32;
                if freq > 2000.0 {  // Only high frequencies
                    freq * mag
                } else {
                    0.0
                }
            })
            .sum::<f32>();
        hfc.push(hf_content);
    }
    
    let threshold = percentile(&hfc, threshold_percentile);
    peak_pick(&hfc, threshold)
}
```

**Strengths**: Excellent for drums, electronic percussion
**Weaknesses**: Misses bass-heavy drops, fails on acoustic guitar

---

#### 2.2.5 Harmonic-Percussive Source Separation (`hpss.rs`)

**Algorithm**: Decompose spectrogram into harmonic (sustained) and percussive (transient) components using median filtering.

```
1. Compute STFT magnitude
2. Apply horizontal median filter (across time):
   H[n,f] = median(magnitude[n-k..n+k, f])
3. Apply vertical median filter (across frequency):
   P[n,f] = median(magnitude[n, f-k..f+k])
4. Normalize and combine: H = H / (H + P + ε), P = P / (H + P + ε)
5. Percussive onsets = energy peaks in P component
6. Return: Onset times
```

**Why this works**: Mathematically separates percussive (narrow in frequency, sharp in time) from harmonic (broad in frequency, sustained in time).

**Pseudocode**:
```rust
pub fn hpss_decompose(
    magnitude_spec: &[Vec<f32>],
    margin: usize,  // Median filter window: ±margin
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_frames = magnitude_spec.len();
    let n_bins = magnitude_spec[0].len();
    
    // Horizontal (time) median for harmonic
    let harmonic = (0..n_frames).map(|n| {
        (0..n_bins).map(|f| {
            let window: Vec<f32> = (n.saturating_sub(margin)..=(n+margin).min(n_frames-1))
                .map(|nn| magnitude_spec[nn][f])
                .collect();
            median(&window)
        }).collect()
    }).collect::<Vec<_>>();
    
    // Vertical (frequency) median for percussive
    let percussive = (0..n_frames).map(|n| {
        (0..n_bins).map(|f| {
            let window: Vec<f32> = (f.saturating_sub(margin)..=(f+margin).min(n_bins-1))
                .map(|ff| magnitude_spec[n][ff])
                .collect();
            median(&window)
        }).collect()
    }).collect::<Vec<_>>();
    
    (harmonic, percussive)
}

pub fn detect_hpss_onsets(
    percussive_component: &[Vec<f32>],
    threshold_percentile: f32,
) -> Vec<usize> {
    let energy = percussive_component.iter()
        .map(|frame| frame.iter().sum::<f32>())
        .collect::<Vec<_>>();
    
    let threshold = percentile(&energy, threshold_percentile);
    peak_pick(&energy, threshold)
}
```

**Strengths**: State-of-the-art for mixed material (live+studio), excellent on varied genres
**Weaknesses**: Computationally expensive, slower than other methods

---

#### 2.2.6 Consensus Voting (`consensus.rs`)

**Algorithm**: Combine all 4 methods with weighted voting.

```rust
pub struct OnsetConsensus {
    energy_flux: Vec<usize>,
    spectral_flux: Vec<usize>,
    hfc: Vec<usize>,
    hpss: Vec<usize>,
}

pub fn vote_onsets(
    consensus: OnsetConsensus,
    weights: [f32; 4],  // e.g., [0.25, 0.25, 0.25, 0.25]
    tolerance_ms: u32,  // e.g., 50ms for clustering
    sample_rate: u32,
) -> Vec<OnsetCandidate> {
    // Merge all 4 onset lists within tolerance_ms windows
    // Vote: Higher weighted sum → higher confidence
    // Return: Deduplicated onset candidates with confidence scores
}
```

**Result**: `Vec<OnsetCandidate>` where:
```rust
pub struct OnsetCandidate {
    pub time_samples: usize,
    pub time_seconds: f32,
    pub confidence: f32,  // 0.0-1.0, higher = more detectors agreed
    pub voted_by: u32,    // How many methods detected this onset
}
```

---

### 2.3 Period Estimation Module

#### 2.3.1 Overview

**Purpose**: Convert onset list → BPM and tempo candidates.

**Why two methods?**:
- **Autocorrelation**: Finds periodicity directly in onset signal
- **Comb filterbank**: Tests hypothesis tempos across range, picks best

These often disagree; we use both and confidence-score the result.

---

#### 2.3.2 Autocorrelation (`autocorrelation.rs`)

**Algorithm**: Find periodic pattern in onset times.

```
1. Convert onset list to binary "beat signal":
   beat_signal[i] = 1 if onset at frame i, else 0
2. Compute autocorrelation:
   ACF[lag] = sum(beat_signal[i] * beat_signal[i + lag])
3. Find peaks in ACF (periodicity candidates)
4. Convert lag → BPM:
   BPM = (60 * sample_rate) / (lag * hop_size)
5. Return: BPM candidates ranked by ACF peak height
```

**Pseudocode**:
```rust
pub fn estimate_bpm_from_autocorrelation(
    onsets: &[usize],
    sample_rate: u32,
    hop_size: usize,
    min_bpm: f32,
    max_bpm: f32,
) -> Vec<BpmCandidate> {
    let max_samples = (onsets.last().unwrap_or(&0) + sample_rate as usize * 60 / min_bpm as usize / 4);
    let mut beat_signal = vec![0.0; max_samples / hop_size];
    
    for &onset in onsets {
        let frame = onset / hop_size;
        if frame < beat_signal.len() {
            beat_signal[frame] += 1.0;
        }
    }
    
    // Autocorrelation
    let mut acf = vec![0.0; beat_signal.len() / 2];
    for lag in 0..acf.len() {
        for i in 0..(beat_signal.len() - lag) {
            acf[lag] += beat_signal[i] * beat_signal[i + lag];
        }
    }
    
    // Peak pick in ACF, convert to BPM
    let min_lag = (60.0 * sample_rate as f32) / (max_bpm * hop_size as f32);
    let max_lag = (60.0 * sample_rate as f32) / (min_bpm * hop_size as f32);
    
    peak_pick_range(&acf, min_lag as usize, max_lag as usize)
        .into_iter()
        .map(|(lag, strength)| {
            let bpm = (60.0 * sample_rate as f32) / (lag as f32 * hop_size as f32);
            BpmCandidate { bpm, confidence: strength / acf[acf.len() / 2].max(1e-10) }
        })
        .collect()
}
```

**Strengths**: Fast (FFT-accelerated autocorrelation is O(n log n))
**Weaknesses**: Prone to octave errors (2x or 0.5x BPM), sensitive to onset detection quality

---

#### 2.3.3 Comb Filterbank (`comb_filter.rs`)

**Algorithm**: Test candidate tempos, score by match quality.

```
1. For each candidate BPM in range (80-180, 1 BPM resolution):
   - Compute expected beat times: beat_times = [0, 60/BPM, 120/BPM, ...]
   - Create comb filter (impulses at beat times)
   - Convolve with onset signal
   - Score: correlate expected beats vs. actual onsets
   
2. Peak score → best BPM estimate
3. Return: BPM with confidence (correlation strength)
```

**Pseudocode**:
```rust
pub fn estimate_bpm_from_comb_filter(
    onsets: &[usize],
    sample_rate: u32,
    hop_size: usize,
    min_bpm: f32,
    max_bpm: f32,
    bpm_resolution: f32,  // e.g., 0.5 for half-BPM precision
) -> Vec<BpmCandidate> {
    let mut candidates = Vec::new();
    
    for bpm in (min_bpm as i32..=max_bpm as i32)
        .step_by(bpm_resolution as usize)
        .map(|b| b as f32)
    {
        let beat_interval_samples = (60.0 / bpm) * sample_rate as f32;
        
        // Score this BPM by checking onset alignment
        let mut score = 0.0;
        let tolerance = beat_interval_samples * 0.1; // ±10% timing tolerance
        
        for &onset in onsets {
            // Find nearest beat
            let nearest_beat = (onset as f32 / beat_interval_samples).round() * beat_interval_samples;
            let distance = (onset as f32 - nearest_beat).abs();
            
            if distance < tolerance {
                score += (1.0 - distance / tolerance); // Closer matches score higher
            }
        }
        
        candidates.push(BpmCandidate {
            bpm,
            confidence: score / onsets.len().max(1) as f32,
        });
    }
    
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    candidates
}
```

**Strengths**: Robust to octave errors, interpretable scoring
**Weaknesses**: Slower (O(n * n_candidates)), can miss syncopated rhythms

---

#### 2.3.4 Candidate Filtering (`candidate_filter.rs`)

**Algorithm**: Merge autocorrelation + comb results, resolve conflicts.

```rust
pub struct BpmEstimate {
    pub bpm: f32,
    pub confidence: f32,
    pub method_agreement: u32,  // How many methods agree
}

pub fn merge_bpm_candidates(
    autocorr: Vec<BpmCandidate>,
    comb: Vec<BpmCandidate>,
    octave_tolerance_cents: f32,  // e.g., 50 cents
) -> Vec<BpmEstimate> {
    // Group candidates within octave tolerance
    // Higher score if both methods agree
    // Filter out outliers
}
```

**Octave tolerance**: BPM estimates often off by 2x. If autocorr says 90 and comb says 180, that's the same underlying beat (one missed half-beats).

---

### 2.4 Beat Tracking Module

#### 2.4.1 Hidden Markov Model Beat Tracker (`hmm.rs`)

**Purpose**: Refine tempo estimate and establish precise beat grid.

**Why HMM?**: Allows probabilistic tracking of tempo variations, handles rubato (tempo drift) and syncopation.

**States**: Beat position (0 = expected beat, ±10% tempo deviation)

**Observations**: Onsets (match expected beat or not)

**Algorithm** (Viterbi):
```
1. Build HMM:
   - States: All possible beat phases (BPM ± 10%)
   - Transition prob: P(tempo_t | tempo_t-1) = high if stable tempo
   - Emission prob: P(onset | beat) = high if onset near expected beat time
   
2. Forward pass: Track best path through state space
3. Backtrack to find most likely beat sequence
4. Output: Refined BPM, beat times, confidence
```

**Pseudocode**:
```rust
pub struct HmmBeatTracker {
    bpm_estimate: f32,
    onsets: Vec<f32>,  // in seconds
    sample_rate: u32,
}

impl HmmBeatTracker {
    pub fn track_beats(&self) -> Vec<BeatPosition> {
        // Viterbi algorithm:
        // 1. Initialize probabilities for first frame
        // 2. Forward pass: compute best path probability
        // 3. Backtrack: extract best path (beat sequence)
        
        let states = self.build_state_space();
        let transitions = self.compute_transitions();
        let emissions = self.compute_emissions();
        
        let viterbi = self.viterbi_algorithm(&states, &transitions, &emissions);
        let path = self.backtrack(&viterbi);
        
        path
    }
    
    fn build_state_space(&self) -> Vec<f32> {
        // BPM candidates: [bpm * 0.9, bpm * 0.95, bpm, bpm * 1.05, bpm * 1.10]
        vec![
            self.bpm_estimate * 0.9,
            self.bpm_estimate * 0.95,
            self.bpm_estimate,
            self.bpm_estimate * 1.05,
            self.bpm_estimate * 1.10,
        ]
    }
    
    fn viterbi_algorithm(
        &self,
        states: &[f32],
        transitions: &[Vec<f32>],
        emissions: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let mut viterbi = vec![vec![0.0; states.len()]; emissions.len()];
        
        // Base case
        for (s, &emission) in emissions[0].iter().enumerate() {
            viterbi[0][s] = emission;
        }
        
        // Recursion
        for t in 1..emissions.len() {
            for s in 0..states.len() {
                let best = (0..states.len())
                    .map(|s_prev| viterbi[t - 1][s_prev] * transitions[s_prev][s])
                    .fold(f32::NEG_INFINITY, f32::max);
                viterbi[t][s] = best * emissions[t][s];
            }
        }
        
        viterbi
    }
}
```

**Output**: Precise beat times (grid points), refined BPM, grid stability metric

---

#### 2.4.2 Bayesian Tempo & Grid Tracking (`bayesian.rs`)

**Purpose**: Update beat grid as analysis progresses (real-time or streaming scenario).

**Algorithm**:
```
Prior: P(BPM | previous_estimate)
Likelihood: P(onset_evidence | BPM)
Posterior: P(BPM | evidence) ∝ Prior * Likelihood

Update: BPM_new = weighted_average(BPM_old, BPM_evidence)
```

**Use case**: If analyzing a 10-minute DJ mix with tempo changes, update grid incrementally rather than recomputing everything.

```rust
pub struct BayesianBeatTracker {
    current_bpm: f32,
    current_confidence: f32,
    history: Vec<f32>,  // Track BPM changes
}

impl BayesianBeatTracker {
    pub fn update(&mut self, new_evidence: f32) {
        let posterior = self.compute_posterior(&new_evidence);
        self.current_bpm = self.update_estimate(posterior);
        self.current_confidence = posterior.confidence;
    }
}
```

---

### 2.5 Chroma Extraction Module

#### 2.5.1 Chroma Vector Computation (`extractor.rs`)

**Purpose**: Extract pitch-class distribution (12 semitones: C, C#, D, ..., B).

**Algorithm**:
```
1. Compute STFT (2048-point FFT, 512 sample hop)
2. Convert frequency bins → semitone classes:
   semitone = 12 * log2(freq / tuning_freq)
   Assuming tuning_freq = 440 Hz (standard A4)
   
3. Sum magnitude across octaves:
   chroma[semitone_class] = sum all bins matching that semitone
   (E.g., chroma[0] = C0 + C1 + C2 + ... + C8)
   
4. Normalize to [0, 1] L2 norm
5. Return: 12-element chroma vector
```

**Pseudocode**:
```rust
pub fn extract_chroma(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,  // 2048
    hop_size: usize,    // 512
) -> Vec<Vec<f32>> {
    let fft = compute_stft(samples, frame_size, hop_size);
    let magnitude = fft.iter()
        .map(|frame| frame.iter().map(|c| c.norm()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    
    let mut chroma_vectors = Vec::new();
    
    for frame_mag in magnitude {
        let mut chroma = vec![0.0; 12];
        
        for (bin, &mag) in frame_mag.iter().enumerate() {
            let freq = (bin as f32 / frame_size as f32) * sample_rate as f32;
            
            if freq > 0.0 {
                // Convert frequency to semitone (relative to A4=440Hz)
                let semitone = 12.0 * (freq / 440.0).log2() + 57.0;  // +57 to shift to [0, 12)
                let semitone_class = ((semitone % 12.0) + 12.0) % 12.0;  // Ensure [0, 12)
                
                // Soft mapping: spread to nearby semitones
                for offset in -1..=1 {
                    let target_class = ((semitone_class as i32 + offset + 12) % 12) as usize;
                    let weight = 1.0 - 0.5 * (offset as f32).abs();
                    chroma[target_class] += mag * weight;
                }
            }
        }
        
        // Normalize
        let norm = chroma.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for x in &mut chroma {
            *x /= norm;
        }
        
        chroma_vectors.push(chroma);
    }
    
    chroma_vectors
}
```

**Output**: `Vec<Vec<f32>>` where each inner vec is 12-element chroma

---

#### 2.5.2 Chroma Normalization (`normalization.rs`)

**Purpose**: Reduce variance due to production choices, amplify tonal signal.

**Methods**:
1. **L2 Normalization** (used above): Already normalized to unit length
2. **Sharpening**: Emphasize prominent semitones:
   ```
   chroma_sharp[i] = chroma[i]^2  or  chroma[i]^1.5
   ```
3. **Downsampling**: Average chroma vectors over 3-5 frames to smooth noise

```rust
pub fn sharpen_chroma(chroma: &[f32], power: f32) -> Vec<f32> {
    let mut sharp = chroma.to_vec();
    for x in &mut sharp {
        *x = x.powf(power);
    }
    let norm = sharp.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    for x in &mut sharp {
        *x /= norm;
    }
    sharp
}
```

---

#### 2.5.3 Temporal Chroma Smoothing (`smoothing.rs`)

**Purpose**: Reduce frame-to-frame variance using median or average filtering.

```rust
pub fn smooth_chroma(
    chroma_vectors: &[Vec<f32>],
    window_size: usize,  // e.g., 5 frames
) -> Vec<Vec<f32>> {
    let half_window = window_size / 2;
    
    (0..chroma_vectors.len())
        .map(|i| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(chroma_vectors.len());
            
            let avg = (0..12)
                .map(|semitone| {
                    let sum: f32 = chroma_vectors[start..end]
                        .iter()
                        .map(|c| c[semitone])
                        .sum();
                    sum / (end - start) as f32
                })
                .collect::<Vec<_>>();
            
            // Renormalize
            let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            avg.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}
```

---

### 2.6 Key Detection Module

#### 2.6.1 Krumhansl-Kessler Templates (`templates.rs`)

**Purpose**: Define tonal profile for each of 24 keys (12 major + 12 minor).

**Template**: 12-element vector representing likelihood of each semitone in a key.

**Major key (C major)**: `[0.15, 0.01, 0.12, 0.01, 0.13, 0.11, 0.01, 0.13, 0.01, 0.12, 0.01, 0.10]`
- C: 0.15 (tonic, strongest)
- E: 0.13 (major third)
- G: 0.13 (perfect fifth)
- Others: minor contributions

**Minor key (A minor)**: `[0.10, 0.01, 0.12, 0.01, 0.13, 0.11, 0.01, 0.13, 0.01, 0.12, 0.01, 0.15]`
- Note: Rotated and adjusted relative to major

```rust
pub struct KeyTemplates {
    major: [Vec<f32>; 12],  // C, C#, D, ..., B
    minor: [Vec<f32>; 12],
}

impl KeyTemplates {
    pub fn new() -> Self {
        // Krumhansl-Kessler profiles (empirically derived from listening tests)
        Self {
            major: [
                vec![0.15, 0.01, 0.12, 0.01, 0.13, 0.11, 0.01, 0.13, 0.01, 0.12, 0.01, 0.10],
                // ... repeat for all 12 major keys (rotated)
            ],
            minor: [
                vec![0.10, 0.01, 0.15, 0.01, 0.12, 0.13, 0.01, 0.11, 0.13, 0.01, 0.12, 0.01],
                // ... repeat for all 12 minor keys
            ],
        }
    }
}
```

---

#### 2.6.2 Key Detection Algorithm (`detector.rs`)

**Purpose**: Match chroma distribution against templates.

**Algorithm**:
```
For each of 24 keys:
    correlation = dot_product(chroma_vector, key_template)
    
Best key = argmax(correlation)
Confidence = (best_score - second_best_score) / max_possible_score
```

**Pseudocode**:
```rust
pub fn detect_key(
    chroma_vectors: &[Vec<f32>],
    templates: &KeyTemplates,
) -> KeyDetectionResult {
    // Average chroma across entire track
    let avg_chroma = average_chroma(chroma_vectors);
    
    // Test all 24 keys
    let mut scores = Vec::new();
    
    for (i, template_major) in templates.major.iter().enumerate() {
        let score = dot_product(&avg_chroma, template_major);
        scores.push((KeyType::Major(i), score));
    }
    
    for (i, template_minor) in templates.minor.iter().enumerate() {
        let score = dot_product(&avg_chroma, template_minor);
        scores.push((KeyType::Minor(i), score));
    }
    
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let best_key = scores[0].0.clone();
    let best_score = scores[0].1;
    let second_score = scores[1].1;
    let confidence = (best_score - second_score) / best_score.max(1e-10);
    
    KeyDetectionResult {
        key: best_key,
        confidence,
        scores,  // Full 24-key distribution for analysis
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

**Output**:
```rust
pub struct KeyDetectionResult {
    pub key: Key,
    pub confidence: f32,
    pub all_scores: Vec<(Key, f32)>,  // All 24 keys ranked
}

pub enum Key {
    Major(u32),  // 0 = C, 1 = C#, ..., 11 = B
    Minor(u32),
}
```

---

#### 2.6.3 Key Clarity (`key_clarity.rs`)

**Purpose**: Estimate how "tonal" vs "atonal" the track is.

**Algorithm**:
```
Clarity = (best_key_score - average_score) / range
High clarity: Sharp tonal center (e.g., most notes in one key)
Low clarity: Ambiguous/atonal (e.g., noise, chromatic)
```

```rust
pub fn compute_key_clarity(scores: &[(Key, f32)]) -> f32 {
    let best = scores[0].1;
    let avg: f32 = scores.iter().map(|s| s.1).sum::<f32>() / scores.len() as f32;
    let worst = scores.last().unwrap().1;
    
    if (worst - best).abs() < 1e-10 {
        return 0.0;  // All scores equal = no tonality
    }
    
    (best - avg) / (best - worst).max(1e-10)
}
```

---

### 2.7 Confidence Scoring Module

#### 2.7.1 Scoring Strategy (`confidence.rs`)

**Purpose**: Generate trustworthiness scores for each analysis result.

```rust
pub struct AnalysisConfidence {
    pub bpm_confidence: f32,        // 0.0-1.0
    pub key_confidence: f32,        // 0.0-1.0
    pub grid_stability: f32,        // 0.0-1.0
    pub overall_confidence: f32,    // Weighted average
    pub flags: Vec<AnalysisFlag>,   // Warnings/notes
}

pub enum AnalysisFlag {
    MultimodalBpm,              // Multiple BPM peaks equally strong
    WeakTonality,               // Low key clarity
    TempoVariation,             // Track has tempo drift
    OnsetDetectionAmbiguous,    // Multiple onset interpretations
}
```

**Formula**:
```
BPM Confidence = 0.7 * onset_strength + 0.2 * method_agreement + 0.1 * autocorr_dominance
Key Confidence = key_clarity * 0.8 + chroma_consistency * 0.2
Grid Stability = hmm_likelihood / max_likelihood

Overall = 0.5 * bpm + 0.3 * key + 0.2 * grid
```

---

### 2.8 Output Result Types

```rust
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    // Tempo analysis
    pub bpm: f32,
    pub bpm_confidence: f32,
    pub tempo_changes: Vec<TempoChange>,  // If variable tempo detected
    
    // Key analysis
    pub key: Key,
    pub key_confidence: f32,
    pub all_key_scores: Vec<(Key, f32)>,  // Top 5 candidates
    pub key_clarity: f32,
    
    // Beat grid
    pub beat_grid: BeatGrid,
    pub grid_stability: f32,
    
    // Metadata
    pub analysis_metadata: AnalysisMetadata,
    pub processing_time_ms: u32,
}

pub struct BeatGrid {
    pub downbeats: Vec<f32>,        // Beat 1 times (seconds)
    pub beats: Vec<f32>,            // All beat times (seconds)
    pub bars: Vec<f32>,             // Bar boundaries (seconds)
}

pub struct AnalysisMetadata {
    pub algorithm_version: String,
    pub onset_method_consensus: f32,
    pub methods_used: Vec<String>,
    pub flags: Vec<AnalysisFlag>,
}
```

---

## 3. PHASE 2: ML REFINEMENT LAYER

### 3.1 ONNX Model Integration

**Purpose**: Train a small neural network to correct common edge cases and boost accuracy 2-5%.

**Model architecture** (lightweight):
```
Input: (features: 64)
├─ Dense(64 → 32, ReLU)
├─ Dropout(0.2)
├─ Dense(32 → 16, ReLU)
├─ Dense(16 → 8, ReLU)
└─ Output(8 → 1, Sigmoid)

Output: Confidence boost factor [0.5, 1.5]
```

**Features**:
- BPM from all 4 methods
- Onset histograms
- Spectral energy distribution
- Key clarity
- Harmonic-to-percussive ratio

**Training data**:
- Collect 1000+ DJ tracks
- Get ground truth from Rekordbox/Serato/manual expert
- Compute features
- Label as "correct" or "error_magnitude"
- Train regression model on error_magnitude

**Inference**:
```rust
pub fn refine_with_ml(
    initial_result: &AnalysisResult,
    features: &[f32],
    model: &OnnxModel,
) -> AnalysisResult {
    let boost = model.infer(features);
    
    // Adjust confidence by boost factor
    let mut refined = initial_result.clone();
    refined.bpm_confidence *= boost;
    refined.key_confidence *= boost;
    refined.overall_confidence *= boost;
    
    refined
}
```

---

## 4. TESTING & VALIDATION

### 4.1 Unit Tests

**For each module**: Test individual functions with synthetic inputs.

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_energy_flux_detects_kick() {
        let kick_pattern = vec![1.0; 2048]; // Impulse
        let onsets = detect_energy_flux_onsets(&kick_pattern, 2048, 512, 0.3);
        assert!(onsets.len() > 0);
    }
    
    #[test]
    fn test_chroma_extraction_c_major() {
        let c_major_notes = generate_c_major_chord();
        let chroma = extract_chroma(&c_major_notes, 44100, 2048, 512);
        let avg = average_chroma(&chroma);
        
        // C (index 0) should dominate
        assert!(avg[0] > avg[2] && avg[0] > avg[4]);
    }
}
```

### 4.2 Integration Tests

**Full pipeline on test tracks**.

```rust
#[test]
fn test_edm_track_bpm_detection() {
    let audio = load_test_audio("fixtures/edm_120bpm.wav");
    let result = analyze_audio(&audio, 44100).unwrap();
    
    assert!((result.bpm - 120.0).abs() < 2.0);  // ±2 BPM tolerance
    assert!(result.bpm_confidence > 0.7);
}

#[test]
fn test_key_detection_c_major_track() {
    let audio = load_test_audio("fixtures/c_major_432.wav");
    let result = analyze_audio(&audio, 44100).unwrap();
    
    assert_eq!(result.key, Key::Major(0));  // C major
    assert!(result.key_confidence > 0.6);
}
```

### 4.3 Ground Truth Dataset

Build collection of test tracks with known BPM/key:
```
fixtures/
├── edm/
│   ├── 120bpm_cmajor.wav
│   ├── 128bpm_aminor.wav
│   └── ...
├── breakbeats/
│   ├── funk_95bpm_gmajor.wav
│   └── ...
├── live/
│   ├── jazz_rubato_dmajor.wav
│   └── ...
```

### 4.4 Accuracy Metrics

```rust
pub struct AccuracyReport {
    pub bpm_accuracy_percent: f32,    // % within ±2 BPM
    pub key_accuracy_percent: f32,    // % exact match
    pub false_positive_rate: f32,     // Octave errors
    pub mean_error_bpm: f32,
    pub median_error_bpm: f32,
}

pub fn compute_accuracy(
    predicted: &[AnalysisResult],
    ground_truth: &[GroundTruth],
) -> AccuracyReport {
    // Compute metrics
}
```

---

## 5. PERFORMANCE TARGETS

### 5.1 Accuracy

**Goal**: Match or exceed professional tools.

```
Baseline (Essentia):
- BPM: ~82% (±2 BPM tolerance)
- Key: ~70%

Target (Stratum):
- BPM: 85-88% (with tuning)
- Key: 75-80% (with tuning)

Stretch (With ML):
- BPM: 90%+
- Key: 82%+
```

### 5.2 Speed

**Single-threaded Rust (no GPU)**:
- 30s track: ~200-500ms (real-time factor 0.15-0.35x)
- CPU i7 2024: Parallelizable to 50-100ms with rayon

**With GPU FFT**:
- 30s track: ~50-100ms (with amortized GPU overhead)

---

## 6. ROADMAP & MILESTONES

### Phase 1: Classical DSP (Weeks 1-4)

- **Week 1**: Onset detection (4 methods + consensus)
- **Week 2**: Period estimation (autocorr + comb filter)
- **Week 3**: Beat tracking (HMM) + chroma extraction
- **Week 4**: Key detection (Krumhansl-Kessler) + confidence scoring
- **Deliverable**: Core algorithm, unit tests, 80%+ accuracy

### Phase 2: ML Refinement (Weeks 5-8)

- **Week 5**: Generate training dataset (1000+ tracks)
- **Week 6**: Train ONNX model
- **Week 7**: Integrate ONNX inference
- **Week 8**: Tuning & validation
- **Deliverable**: ML-enhanced algorithm, 88%+ accuracy

### Phase 3: Integration & Polish (Weeks 9-10)

- **Week 9**: Integrate into stratum-desktop
- **Week 10**: Performance optimization, crate polish
- **Deliverable**: Published crate, integrated into app

---

## 7. IMPLEMENTATION NOTES

### 7.1 Dependencies

```toml
[dependencies]
# Audio I/O
symphonia = { version = "0.5", features = ["all"] }

# Math & DSP
ndarray = "0.15"                    # Multi-dimensional arrays
ndarray-linalg = "0.15"             # Linear algebra
rustfft = "6.2"                     # FFT
# OR use GPU FFT via opencl3 = "0.12"

# ML (Phase 2)
ort = "2.0"                         # ONNX Runtime

# Utilities
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rayon = "1.8"                       # Parallelization
```

### 7.2 Numerical Stability

- Use **log-domain math** for very small/large values (avoid underflow)
- Add epsilon (1e-10) to denominators
- Normalize vectors to unit norm before comparisons

### 7.3 Real-Time Streaming

For future DJ software that analyzes during playback:
- Process in chunks (2048 sample frames)
- Cache FFT plan across frames
- Incremental chroma computation
- Bayesian update for BPM/key

---

## 8. REFERENCE LITERATURE

**Beat Tracking & Tempo Estimation**:
- Ellis & Pikrakis (2006): "Real-time beat induction"
- Gkiokas et al. (2012): "Dimensionality reduction for BPM estimation"
- Böck et al. (2016): "Joint beat and downbeat tracking" (MIREX)

**Key Detection**:
- Krumhansl & Kessler (1982): "Triad Priming and the Psychological Representation"
- Gomtsyan et al. (2019): "Music key and scale detection"

**Onset Detection**:
- Pecan et al. (2017): "A Comparison of Onset Detection Methods"
- Bello et al. (2005): "A Tutorial on Onset Detection in Music Signals"

**Chroma**:
- Müller et al. (2010): "Chroma-Based Audio Analysis Tutorial"
- Ellis & Poliner (2007): "Identifying Cover Songs from Audio"

---

## 9. OPEN QUESTIONS & FUTURE WORK

1. **Real-time key detection**: Should we support streaming (incremental chroma)?
2. **Multi-tempo tracks**: How to handle DJ mixes with bpm ramps?
3. **Acapella vs instrumental**: Should we handle vocal-only differently?
4. **Transposition detection**: Detect if a remix is transposed from original?
5. **Cross-validation**: Build test corpus, compare to Rekordbox/Serato
6. **Hardware acceleration**: GPU-accelerated FFT for batch processing
7. **Model training**: Where to get 1000+ ground truth DJ tracks?

---

## 10. SUCCESS CRITERIA

A release is ready when:

✅ BPM detection: ≥85% accuracy (±2 BPM tolerance) on test set
✅ Key detection: ≥75% accuracy on test set
✅ Performance: <500ms per 30s track (single-threaded)
✅ Integration: Compiles cleanly into stratum-desktop
✅ Documentation: Full API docs + algorithm explanations
✅ Testing: 80%+ code coverage, no panics in production
✅ Open source: Published to crates.io, MIT/Apache dual license

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-15  
**Status**: Ready for Implementation
