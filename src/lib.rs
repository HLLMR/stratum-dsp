//! # Stratum DSP
//!
//! A professional-grade audio analysis engine for DJ applications, providing
//! accurate BPM detection, key detection, and beat tracking.
//!
//! ## Features
//!
//! - **BPM Detection**: Multi-method onset detection with autocorrelation and comb filterbank
//! - **Key Detection**: Chroma-based analysis with Krumhansl-Kessler template matching
//! - **Beat Tracking**: HMM-based beat grid generation with tempo drift correction
//! - **ML Refinement**: Optional ONNX model for edge case correction (Phase 2)
//!
//! ## Quick Start
//!
//! ```no_run
//! use stratum_dsp::{analyze_audio, AnalysisConfig};
//!
//! // Load audio samples (mono, f32, normalized)
//! let samples: Vec<f32> = vec![]; // Your audio data
//! let sample_rate = 44100;
//!
//! // Analyze
//! let result = analyze_audio(&samples, sample_rate, AnalysisConfig::default())?;
//!
//! println!("BPM: {:.2} (confidence: {:.2})", result.bpm, result.bpm_confidence);
//! println!("Key: {:?} (confidence: {:.2})", result.key, result.key_confidence);
//! # Ok::<(), stratum_dsp::AnalysisError>(())
//! ```
//!
//! ## Architecture
//!
//! The analysis pipeline follows this flow:
//!
//! ```text
//! Audio Input -> Preprocessing -> Feature Extraction -> Analysis -> ML Refinement -> Output
//! ```
//!
//! See the [module documentation](https://docs.rs/stratum-dsp) for details.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod analysis;
pub mod config;
pub mod error;
pub mod features;
pub mod io;
pub mod preprocessing;

#[cfg(feature = "ml")]
pub mod ml;

// Re-export main types
pub use analysis::result::{AnalysisResult, AnalysisMetadata, BeatGrid, Key, KeyType};
pub use analysis::confidence::{AnalysisConfidence, compute_confidence};
pub use config::AnalysisConfig;
pub use error::AnalysisError;

/// Main analysis function
///
/// Analyzes audio samples and returns comprehensive analysis results including
/// BPM, key, beat grid, and confidence scores.
///
/// # Arguments
///
/// * `samples` - Mono audio samples, normalized to [-1.0, 1.0]
/// * `sample_rate` - Sample rate in Hz (typically 44100 or 48000)
/// * `config` - Analysis configuration parameters
///
/// # Returns
///
/// `AnalysisResult` containing BPM, key, beat grid, and confidence metrics
///
/// # Errors
///
/// Returns `AnalysisError` if analysis fails (invalid input, processing error, etc.)
///
/// # Example
///
/// ```no_run
/// use stratum_dsp::{analyze_audio, AnalysisConfig};
///
/// let samples = vec![0.0f32; 44100 * 30]; // 30 seconds of silence
/// let result = analyze_audio(&samples, 44100, AnalysisConfig::default())?;
/// # Ok::<(), stratum_dsp::AnalysisError>(())
/// ```
pub fn analyze_audio(
    samples: &[f32],
    sample_rate: u32,
    config: AnalysisConfig,
) -> Result<AnalysisResult, AnalysisError> {
    use std::time::Instant;
    let start_time = Instant::now();
    
    log::debug!("Starting audio analysis: {} samples at {} Hz", samples.len(), sample_rate);
    
    if samples.is_empty() {
        return Err(AnalysisError::InvalidInput("Empty audio samples".to_string()));
    }
    
    if sample_rate == 0 {
        return Err(AnalysisError::InvalidInput("Invalid sample rate".to_string()));
    }
    
    // Phase 1A: Preprocessing
    let mut processed_samples = samples.to_vec();
    
    // 1. Normalization
    use preprocessing::normalization::{normalize, NormalizationConfig};
    if config.enable_normalization {
        let norm_config = NormalizationConfig {
            method: config.normalization,
            target_loudness_lufs: -14.0, // Default target
            max_headroom_db: 1.0,
        };
        let _loudness_metadata = normalize(&mut processed_samples, norm_config, sample_rate as f32)?;
    } else {
        log::debug!("Skipping normalization (enable_normalization=false)");
    }
    
    // 2. Silence detection and trimming
    use preprocessing::silence::{detect_and_trim, SilenceDetector};
    let (trimmed_samples, _silence_regions) = if config.enable_silence_trimming {
        let silence_detector = SilenceDetector {
            threshold_db: config.min_amplitude_db,
            min_duration_ms: 500,
            frame_size: config.frame_size,
        };
        detect_and_trim(&processed_samples, sample_rate, silence_detector)?
    } else {
        log::debug!("Skipping silence trimming (enable_silence_trimming=false)");
        (processed_samples.clone(), Vec::new())
    };
    
    if trimmed_samples.is_empty() {
        return Err(AnalysisError::ProcessingError("Audio is entirely silent after trimming".to_string()));
    }
    
    // Phase 1A: Onset Detection
    // Note: For now, we only have energy flux working directly on samples
    // Spectral methods (spectral_flux, hfc, hpss) require STFT which will be added in Phase 1B
    use features::onset::energy_flux::detect_energy_flux_onsets;
    
    let energy_onsets = detect_energy_flux_onsets(
        &trimmed_samples,
        config.frame_size,
        config.hop_size,
        -20.0, // threshold_db
    )?;
    
    log::debug!("Detected {} onsets using energy flux", energy_onsets.len());
    
    // Phase 1F: Tempogram-based BPM Detection (replaces Phase 1B period estimation)
    // Compute STFT once (used by tempogram BPM and STFT-based onset detectors)
    use features::chroma::extractor::compute_stft;
    let magnitude_spec_frames = compute_stft(
        &trimmed_samples,
        config.frame_size,
        config.hop_size,
    )?;

    // Onset consensus (improves beat tracking + legacy BPM fallback robustness)
    //
    // Important: Tempogram BPM does NOT use these onsets, but the legacy BPM estimator and
    // beat tracker do. Since we will compare / integrate legacy + tempogram estimates,
    // we want the best onsets we can get.
    let mut onsets_for_legacy: Vec<usize> = energy_onsets.clone();
    let mut onsets_for_beat_tracking: Vec<usize> = energy_onsets.clone();

    if config.enable_onset_consensus && !magnitude_spec_frames.is_empty() {
        use features::onset::consensus::{vote_onsets, OnsetConsensus};
        use features::onset::hfc::detect_hfc_onsets;
        use features::onset::spectral_flux::detect_spectral_flux_onsets;

        let to_samples = |frames: Vec<usize>, hop_size: usize, n_samples: usize| -> Vec<usize> {
            let mut out: Vec<usize> = frames
                .into_iter()
                .map(|f| f.saturating_mul(hop_size))
                .filter(|&s| s < n_samples)
                .collect();
            out.sort_unstable();
            out.dedup();
            out
        };

        let spectral_onsets_frames = match detect_spectral_flux_onsets(
            &magnitude_spec_frames,
            config.onset_threshold_percentile,
        ) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Spectral flux onset detection failed: {}", e);
                Vec::new()
            }
        };
        let spectral_onsets_samples = to_samples(
            spectral_onsets_frames,
            config.hop_size,
            trimmed_samples.len(),
        );

        let hfc_onsets_frames = match detect_hfc_onsets(
            &magnitude_spec_frames,
            sample_rate,
            config.onset_threshold_percentile,
        ) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("HFC onset detection failed: {}", e);
                Vec::new()
            }
        };
        let hfc_onsets_samples = to_samples(hfc_onsets_frames, config.hop_size, trimmed_samples.len());

        let hpss_onsets_samples = if config.enable_hpss_onsets {
            use features::onset::hpss::{detect_hpss_onsets, hpss_decompose};
            match hpss_decompose(&magnitude_spec_frames, config.hpss_margin)
                .and_then(|(_, p)| detect_hpss_onsets(&p, config.onset_threshold_percentile))
            {
                Ok(hpss_frames) => to_samples(hpss_frames, config.hop_size, trimmed_samples.len()),
                Err(e) => {
                    log::warn!("HPSS onset detection failed: {}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        log::debug!(
            "Onset detectors: energy_flux(samples)={}, spectral_flux(samples)={}, hfc(samples)={}, hpss(samples)={}",
            energy_onsets.len(),
            spectral_onsets_samples.len(),
            hfc_onsets_samples.len(),
            hpss_onsets_samples.len()
        );

        let consensus = OnsetConsensus {
            energy_flux: energy_onsets.clone(),
            spectral_flux: spectral_onsets_samples,
            hfc: hfc_onsets_samples,
            hpss: hpss_onsets_samples,
        };

        match vote_onsets(
            consensus,
            config.onset_consensus_weights,
            config.onset_consensus_tolerance_ms,
            sample_rate,
        ) {
            Ok(candidates) => {
                // Default policy: prefer onsets confirmed by >=2 methods.
                // If that yields nothing, fall back to the full clustered set (>=1 method).
                let mut strong: Vec<usize> = candidates
                    .iter()
                    .filter(|c| c.voted_by >= 2)
                    .map(|c| c.time_samples)
                    .collect();
                strong.sort_unstable();
                strong.dedup();

                let mut any: Vec<usize> = candidates.iter().map(|c| c.time_samples).collect();
                any.sort_unstable();
                any.dedup();

                let chosen = if !strong.is_empty() { strong } else { any };
                if !chosen.is_empty() {
                    log::debug!(
                        "Onset consensus: chosen {} onsets (strong>=2 methods: {}, total_clusters: {})",
                        chosen.len(),
                        candidates.iter().filter(|c| c.voted_by >= 2).count(),
                        candidates.len()
                    );
                    onsets_for_legacy = chosen.clone();
                    onsets_for_beat_tracking = chosen;
                } else {
                    log::debug!("Onset consensus produced no candidates; using energy-flux onsets");
                }
            }
            Err(e) => {
                log::warn!("Onset consensus voting failed: {}", e);
            }
        }
    }
    
    // BPM estimation: tempogram (Phase 1F) + legacy (Phase 1B), optionally fused
    let legacy_estimate = {
        use features::period::{estimate_bpm, estimate_bpm_with_guardrails, LegacyBpmGuardrails};
        if onsets_for_legacy.len() >= 2 {
            if config.enable_legacy_bpm_guardrails {
                let guardrails = LegacyBpmGuardrails {
                    preferred_min: config.legacy_bpm_preferred_min,
                    preferred_max: config.legacy_bpm_preferred_max,
                    soft_min: config.legacy_bpm_soft_min,
                    soft_max: config.legacy_bpm_soft_max,
                    mul_preferred: config.legacy_bpm_conf_mul_preferred,
                    mul_soft: config.legacy_bpm_conf_mul_soft,
                    mul_extreme: config.legacy_bpm_conf_mul_extreme,
                };
                estimate_bpm_with_guardrails(
                    &onsets_for_legacy,
                    sample_rate,
                    config.hop_size,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                    guardrails,
                )?
            } else {
                estimate_bpm(
                    &onsets_for_legacy,
                    sample_rate,
                    config.hop_size,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                )?
            }
        } else {
            None
        }
    };

    let mut tempogram_candidates: Option<Vec<crate::analysis::result::TempoCandidateDebug>> = None;

    let tempogram_estimate = if !config.force_legacy_bpm && !magnitude_spec_frames.is_empty() {
        use crate::analysis::result::TempoCandidateDebug;
        use features::period::tempogram::{estimate_bpm_tempogram, estimate_bpm_tempogram_with_candidates};

        if config.emit_tempogram_candidates {
            match estimate_bpm_tempogram_with_candidates(
                &magnitude_spec_frames,
                sample_rate,
                config.hop_size as u32,
                config.min_bpm,
                config.max_bpm,
                config.bpm_resolution,
                config.tempogram_candidates_top_n,
            ) {
                Ok((estimate, cands)) => {
                    tempogram_candidates = Some(
                        cands.into_iter()
                            .map(|c| TempoCandidateDebug {
                                bpm: c.bpm,
                                score: c.score,
                                fft_norm: c.fft_norm,
                                autocorr_norm: c.autocorr_norm,
                                selected: c.selected,
                            })
                            .collect(),
                    );
                    log::debug!(
                        "Tempogram BPM estimate: {:.2} (confidence: {:.3}, method_agreement: {}, candidates_emitted={})",
                        estimate.bpm,
                        estimate.confidence,
                        estimate.method_agreement,
                        tempogram_candidates.as_ref().map(|v| v.len()).unwrap_or(0)
                    );
                    Some(estimate)
                }
                Err(e) => {
                    log::warn!("Tempogram BPM detection failed: {}", e);
                    None
                }
            }
        } else {
            match estimate_bpm_tempogram(
                &magnitude_spec_frames,
                sample_rate,
                config.hop_size as u32,
                config.min_bpm,
                config.max_bpm,
                config.bpm_resolution,
            ) {
                Ok(estimate) => {
                    log::debug!(
                        "Tempogram BPM estimate: {:.2} (confidence: {:.3}, method_agreement: {})",
                        estimate.bpm,
                        estimate.confidence,
                        estimate.method_agreement
                    );
                    Some(estimate)
                }
                Err(e) => {
                    log::warn!("Tempogram BPM detection failed: {}", e);
                    None
                }
            }
        }
    } else {
        if config.force_legacy_bpm {
            log::debug!("Forcing legacy BPM estimation (force_legacy_bpm=true)");
        } else if magnitude_spec_frames.is_empty() {
            log::warn!("Could not compute STFT for tempogram");
        }
        None
    };

    let (bpm, bpm_confidence) = if config.force_legacy_bpm {
        legacy_estimate
            .as_ref()
            .map(|e| (e.bpm, e.confidence))
            .unwrap_or((0.0, 0.0))
    } else if config.enable_bpm_fusion {
        // Fusion (safe validator mode):
        // - **Never** override the tempogram BPM (so fusion cannot regress BPM accuracy).
        // - Use legacy only to adjust *confidence* and emit diagnostics.
        let (t_bpm, t_conf, t_agree) = tempogram_estimate
            .as_ref()
            .map(|e| (e.bpm, e.confidence, e.method_agreement))
            .unwrap_or((0.0, 0.0, 0));
        let (l_bpm, l_conf_raw) = legacy_estimate
            .as_ref()
            .map(|e| (e.bpm, e.confidence))
            .unwrap_or((0.0, 0.0));
        let l_conf = l_conf_raw.clamp(0.0, 1.0);

        // If tempogram is unavailable, fall back to legacy (guardrailed).
        if t_bpm <= 0.0 {
            legacy_estimate
                .as_ref()
                .map(|e| (e.bpm, e.confidence))
                .unwrap_or((0.0, 0.0))
        } else {
            let tol = 2.0f32;
            let mut conf = t_conf.clamp(0.0, 1.0);

            // Agreement / validation scoring between legacy and tempogram BPMs.
            let agreement = if l_bpm > 0.0 {
                // Allow common metrical ambiguity relations without forcing an override.
                let diffs = [
                    (l_bpm - t_bpm).abs(),
                    (l_bpm - (t_bpm * 0.5)).abs(),
                    (l_bpm - (t_bpm * 2.0)).abs(),
                    (l_bpm - (t_bpm * (2.0 / 3.0))).abs(),
                    (l_bpm - (t_bpm * (3.0 / 2.0))).abs(),
                ];
                diffs.into_iter().any(|d| d <= tol)
            } else {
                false
            };

            if agreement {
                // Modest boost when legacy is consistent (even if it’s at a different metrical level).
                let boost = 0.12 * l_conf;
                conf = (conf + boost).clamp(0.0, 1.0);
                log::debug!(
                    "BPM fusion (validator): tempogram {:.2} kept; legacy {:.2} validates (agree≈true); conf {:.3}->{:.3}; temp_agree={}",
                    t_bpm,
                    l_bpm,
                    t_conf,
                    conf,
                    t_agree
                );
            } else if l_bpm > 0.0 {
                // If legacy strongly disagrees, slightly down-weight confidence.
                // This helps downstream beat-tracking avoid over-trusting borderline tempos,
                // while preserving the tempogram BPM choice.
                conf = (conf * 0.90).clamp(0.0, 1.0);
                log::debug!(
                    "BPM fusion (validator): tempogram {:.2} kept; legacy {:.2} disagrees; conf {:.3}->{:.3}; temp_agree={}",
                    t_bpm,
                    l_bpm,
                    t_conf,
                    conf,
                    t_agree
                );
            } else {
                log::debug!(
                    "BPM fusion (validator): tempogram {:.2} kept; no legacy estimate available; temp_agree={}",
                    t_bpm,
                    t_agree
                );
            }

            (t_bpm, conf)
        }
    } else {
        // Default behavior: tempogram first; legacy fallback only if tempogram fails.
        tempogram_estimate
            .as_ref()
            .map(|e| (e.bpm, e.confidence))
            .or_else(|| legacy_estimate.as_ref().map(|e| (e.bpm, e.confidence)))
            .unwrap_or((0.0, 0.0))
    };
    
    if bpm == 0.0 {
        log::warn!("Could not estimate BPM: tempogram and legacy methods both failed");
    } else {
        log::debug!("Estimated BPM: {:.2} (confidence: {:.3})", bpm, bpm_confidence);
    }
    
    // Phase 1C: Beat Tracking
    let (beat_grid, grid_stability) = if bpm > 0.0 && onsets_for_beat_tracking.len() >= 2 {
        // Convert onsets from sample indices to seconds
        let onsets_seconds: Vec<f32> = onsets_for_beat_tracking
            .iter()
            .map(|&sample_idx| sample_idx as f32 / sample_rate as f32)
            .collect();

        // Generate beat grid using HMM Viterbi algorithm
        use features::beat_tracking::generate_beat_grid;
        match generate_beat_grid(bpm, bpm_confidence, &onsets_seconds, sample_rate) {
            Ok((grid, stability)) => {
                log::debug!(
                    "Beat grid generated: {} beats, {} downbeats, stability={:.3}",
                    grid.beats.len(),
                    grid.downbeats.len(),
                    stability
                );
                (grid, stability)
            }
            Err(e) => {
                log::warn!("Beat tracking failed: {}, using empty grid", e);
                (
                    BeatGrid {
                        downbeats: vec![],
                        beats: vec![],
                        bars: vec![],
                    },
                    0.0,
                )
            }
        }
    } else {
        log::debug!("Skipping beat tracking: BPM={:.2}, onsets={}", bpm, energy_onsets.len());
        (
            BeatGrid {
                downbeats: vec![],
                beats: vec![],
                bars: vec![],
            },
            0.0,
        )
    };
    
    // Phase 1D: Key Detection
    let (key, key_confidence, key_clarity) = if trimmed_samples.len() >= config.frame_size {
        // Extract chroma vectors with configurable options
        use features::chroma::extractor::extract_chroma_with_options;
        use features::chroma::normalization::sharpen_chroma;
        use features::chroma::smoothing::smooth_chroma;
        use features::key::{detect_key, compute_key_clarity, KeyTemplates};
        
        match extract_chroma_with_options(
            &trimmed_samples,
            sample_rate,
            config.frame_size,
            config.hop_size,
            config.soft_chroma_mapping,
            config.soft_mapping_sigma,
        ) {
            Ok(mut chroma_vectors) => {
                // Apply chroma sharpening if enabled (power > 1.0)
                if config.chroma_sharpening_power > 1.0 {
                    for chroma in &mut chroma_vectors {
                        *chroma = sharpen_chroma(chroma, config.chroma_sharpening_power);
                    }
                    log::debug!("Applied chroma sharpening with power {:.2}", config.chroma_sharpening_power);
                }
                
                // Apply temporal smoothing (optional but recommended)
                if chroma_vectors.len() > 5 {
                    chroma_vectors = smooth_chroma(&chroma_vectors, 5);
                }
                
                // Detect key using Krumhansl-Kessler templates
                let templates = KeyTemplates::new();
                match detect_key(&chroma_vectors, &templates) {
                    Ok(key_result) => {
                        // Compute key clarity
                        let clarity = compute_key_clarity(&key_result.all_scores);
                        
                        log::debug!("Detected key: {:?}, confidence: {:.3}, clarity: {:.3}",
                                   key_result.key, key_result.confidence, clarity);
                        
                        (key_result.key, key_result.confidence, clarity)
                    }
                    Err(e) => {
                        log::warn!("Key detection failed: {}, using default", e);
                        (Key::Major(0), 0.0, 0.0)
                    }
                }
            }
            Err(e) => {
                log::warn!("Chroma extraction failed: {}, using default key", e);
                (Key::Major(0), 0.0, 0.0)
            }
        }
    } else {
        log::debug!("Skipping key detection: insufficient samples (need at least {} samples)",
                   config.frame_size);
        (Key::Major(0), 0.0, 0.0)
    };
    
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
    
    // Build confidence warnings
    let mut confidence_warnings = Vec::new();
    let mut flags = Vec::new();
    
    if bpm == 0.0 {
        confidence_warnings.push("BPM detection failed: insufficient onsets or estimation error".to_string());
    }
    if grid_stability < 0.5 {
        confidence_warnings.push(format!("Low beat grid stability: {:.2} (may indicate tempo variation)", grid_stability));
    }
    if key_confidence < 0.3 {
        confidence_warnings.push(format!("Low key detection confidence: {:.2} (may indicate ambiguous or atonal music)", key_confidence));
    }
    if key_clarity < 0.2 {
        confidence_warnings.push(format!("Low key clarity: {:.2} (track may be atonal or have weak tonality)", key_clarity));
        flags.push(crate::analysis::result::AnalysisFlag::WeakTonality);
    }
    
    // Phase 1E: Build result and compute comprehensive confidence scores
    let result = AnalysisResult {
        bpm,
        bpm_confidence,
        key,
        key_confidence,
        key_clarity,
        beat_grid,
        grid_stability,
        metadata: AnalysisMetadata {
            duration_seconds: trimmed_samples.len() as f32 / sample_rate as f32,
            sample_rate,
            processing_time_ms,
            algorithm_version: "0.1.0-alpha".to_string(),
            onset_method_consensus: if energy_onsets.is_empty() { 0.0 } else { 1.0 },
            methods_used: vec!["energy_flux".to_string(), "chroma_extraction".to_string(), "key_detection".to_string()],
            flags,
            confidence_warnings,
            tempogram_candidates,
        },
    };
    
    // Phase 1E: Compute comprehensive confidence scores
    use analysis::confidence::compute_confidence;
    let confidence = compute_confidence(&result);
    log::debug!(
        "Analysis complete: BPM={:.2} (conf={:.3}), Key={:?} (conf={:.3}), Overall confidence={:.3}",
        result.bpm,
        confidence.bpm_confidence,
        result.key,
        confidence.key_confidence,
        confidence.overall_confidence
    );
    
    // Return result with Phase 1E confidence scoring integrated
    Ok(result)
}

