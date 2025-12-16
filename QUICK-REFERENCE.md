# Quick Reference: Audio Analysis Engine Build Plan

## TL;DR

✅ **Build separate Rust crate** (not in desktop app)  
✅ **Classic DSP + ML** (not just Essentia wrapper)  
✅ **8 weeks to v1.0** (Phase 1: DSP, Phase 2: ML)  
✅ **Target: 88% BPM, 77% key** (vs Essentia 70%, Rekordbox 80%)  
✅ **Publish to crates.io** (establish authority, open-source)  

---

## Why Not Essentia?

| Problem | Essentia | Your Build |
|---------|----------|-----------|
| Licensing | AGPL (reciprocal) | MIT/Apache ✅ |
| Distribution | Requires system deps | Single binary ✅ |
| Accuracy | 70% | 88% (goal) ✅ |
| DJ-optimized | No | Yes ✅ |
| Tauri integration | FFI nightmare | Native Rust ✅ |

---

## 8-Week Sprint Schedule

```
PHASE 1A (Week 1): Onset Detection (4 methods)
  └─ Energy flux, spectral flux, HFC, HPSS

PHASE 1B (Week 2): Period Estimation
  └─ Autocorrelation + Comb filter → BPM

PHASE 1C (Week 3-4): Beat Tracking + Key Detection
  └─ HMM beat tracking, Krumhansl-Kessler key matching

PHASE 1E (Week 5): Integration & Tuning
  └─ v0.9-alpha ready, 85%+ BPM accuracy

PHASE 2A (Week 6): Data Collection
  └─ Gather 1000 ground-truth tracks

PHASE 2B (Week 7): ML Model Training
  └─ Train ONNX refinement model

PHASE 2C (Week 8): Polish & Publish
  └─ v1.0 released to crates.io
```

---

## Core Algorithm Stack

### 1. Onset Detection (Detect Beat Transients)
- Energy Flux: Peak in frame energy derivative
- Spectral Flux: Change in magnitude spectrum
- HFC: High-frequency energy content (drums)
- HPSS: Percussive components (median filtering)
- **Vote**: Majority agreement = robust detection

### 2. Period Estimation (Find BPM)
- Autocorrelation: Find periodicity in onset signal
- Comb Filter: Test hypothesis tempos, score by match
- **Merge**: Octave-error correction, confidence scoring

### 3. Beat Tracking (Generate Grid)
- HMM Viterbi: Track most likely beat sequence
- Bayesian: Update for tempo drift (DJ mixes)
- Output: Beat times, bar boundaries

### 4. Key Detection (Find Musical Key)
- Chroma Extraction: FFT → 12-semitone distribution
- Template Matching: Compare to Krumhansl-Kessler profiles (24 keys)
- Clarity Scoring: How "tonal" is the track?

### 5. ML Refinement (Boost Accuracy)
- Small ONNX model (Phase 2)
- Input: Features from all 4 modules
- Output: Confidence boost factor
- Result: +2-3% accuracy on edge cases

---

## Accuracy Targets

| Metric | Current | Essentia | Target v1.0 | Stretch v2.0 |
|--------|---------|----------|-------------|--------------|
| **BPM Accuracy** | ~60% | ~82% | **88%** | **90%+** |
| **Key Accuracy** | ~50% | ~70% | **77%** | **82%+** |
| **Confidence Scoring** | No | Basic | Yes | Robust |
| **Edge Cases** | Fails | Mixed | Handled | Learned |

---

## Files Provided

| File | Purpose | Use |
|------|---------|-----|
| `audio-analysis-engine-spec.md` | Full technical spec | Reference during implementation |
| `development-strategy.md` | Roadmap + decisions | Strategic planning + Cursor briefing |
| `cursor-prompts.md` | AI prompts | Paste into Cursor for each module |
| `EXECUTIVE-SUMMARY.md` | Business case | Justify to stakeholders |

---

## Repository Setup

```
stratum-workspace/
├── stratum-audio-analysis/     ← NEW (this project)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── preprocessing/
│   │   ├── features/
│   │   ├── analysis/
│   │   └── ml/                 # Phase 2
│   ├── tests/
│   └── models/                 # Phase 2
├── stratum-shared/             ← EXISTING
└── stratum-desktop/            ← EXISTING
    └── Integrates in Week 9
```

---

## Key Dependencies

```toml
symphonia = "0.5"           # Audio decode
rustfft = "6.2"             # FFT
ndarray = "0.15"            # Multi-dim arrays
log = "0.4"                 # Logging
ort = "2.0"                 # ONNX (Phase 2)
```

---

## Testing Strategy

### Phase 1 (Weeks 1-5)
- Unit tests for each function
- Integration tests on synthetic audio
- Accuracy tests on real DJ tracks
- Target: 85% BPM on test set

### Phase 2 (Weeks 6-8)
- Train ML model on 1000-track dataset
- A/B test: Classical vs ML-refined
- Final accuracy report
- Edge case analysis

---

## How to Use Cursor

1. Read `audio-analysis-engine-spec.md` Section 2 (algorithms)
2. For each module, paste corresponding prompt from `cursor-prompts.md`
3. Cursor generates implementation (~200-400 lines per module)
4. You review implementation for correctness
5. Write tests (Cursor can help)
6. Run: `cargo test`
7. Next module

**Typical module cycle**: 1-2 hours with Cursor

---

## Success Criteria (When to Ship v1.0)

✅ BPM detection: ≥88% accuracy (±2 BPM tolerance)  
✅ Key detection: ≥77% accuracy (exact match)  
✅ Performance: <500ms per 30s track  
✅ Full API documentation  
✅ 80%+ test coverage  
✅ No panics in release mode  
✅ Published to crates.io  

---

## Integration into Desktop (Week 9)

After v1.0 release:

```rust
// stratum-desktop/src-tauri/Cargo.toml
stratum-audio-analysis = "1.0"

// src-tauri/src/commands/analyze.rs
use stratum_audio_analysis::analyze_audio;

pub async fn analyze_file(path: String) -> Result<AnalysisResult> {
    let audio = decode_audio(&path)?;
    let result = analyze_audio(&audio, 44100)?;
    Ok(result)
}
```

Done. Drop-in replacement.

---

## Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| Accuracy < 85% | Medium | Iterate algorithms, add ML early |
| Performance > 500ms | Low | Use rustfft optimization, parallelize |
| Licensing issues | Low | Use MIT/Apache, not AGPL |
| Data collection hard | Medium | Use Rekordbox library + MusicBrainz |
| Viterbi/HMM complexity | Medium | Start simple, add complexity iteratively |

---

## Competitive Positioning

After v1.0, you have:

✅ Open-source reference implementation  
✅ Better accuracy than Essentia  
✅ DJ-specific optimizations  
✅ Extensible architecture for Phase 3 (energy, mood, genre)  
✅ Platform for community contributions  

**Result**: Stratum becomes "the Rust DJ audio analysis standard"

---

## Timeline Summary

```
Week 1-5:   Implement classical DSP (Phase 1)
            Deliverable: v0.9-alpha, 85% BPM accuracy
            
Week 6-8:   Add ML refinement (Phase 2)
            Deliverable: v1.0, 88% BPM accuracy, published

Week 9:     Integrate into desktop
            Deliverable: stratum-desktop with new analysis

Week 10+:   User feedback iteration, Phase 3 planning
            (energy, mood, genre classification)
```

---

## Start Here

1. ✅ Read this document (5 min)
2. ✅ Read `EXECUTIVE-SUMMARY.md` (10 min)
3. ✅ Skim `audio-analysis-engine-spec.md` Section 1-2 (30 min)
4. ✅ Create repo: `cargo new stratum-audio-analysis --lib`
5. ✅ Start with Cursor: Use Prompt 1.1 from `cursor-prompts.md`

**Go time.**

---

**Last Updated**: 2025-12-15  
**Status**: Ready to execute  
**Confidence**: 🟢 High (88%+ accuracy achievable in 8 weeks)
