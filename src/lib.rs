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
    let mut tempogram_multi_res_triggered: Option<bool> = None;
    let mut tempogram_multi_res_used: Option<bool> = None;
    let mut tempogram_percussive_triggered: Option<bool> = None;
    let mut tempogram_percussive_used: Option<bool> = None;

    let tempogram_estimate = if !config.force_legacy_bpm && !magnitude_spec_frames.is_empty() {
        use crate::analysis::result::TempoCandidateDebug;
        use features::period::multi_resolution::multi_resolution_tempogram_from_samples;
        use features::period::tempogram::{
            estimate_bpm_tempogram,
            estimate_bpm_tempogram_band_fusion,
            estimate_bpm_tempogram_with_candidates,
            estimate_bpm_tempogram_with_candidates_band_fusion,
            TempogramBandFusionConfig,
        };

        let band_cfg = TempogramBandFusionConfig {
            enabled: config.enable_tempogram_band_fusion,
            low_max_hz: config.tempogram_band_low_max_hz,
            mid_max_hz: config.tempogram_band_mid_max_hz,
            high_max_hz: config.tempogram_band_high_max_hz,
            w_full: config.tempogram_band_w_full,
            w_low: config.tempogram_band_w_low,
            w_mid: config.tempogram_band_w_mid,
            w_high: config.tempogram_band_w_high,
            seed_only: config.tempogram_band_seed_only,
            support_threshold: config.tempogram_band_support_threshold,
            consensus_bonus: config.tempogram_band_consensus_bonus,
            enable_mel: config.enable_tempogram_mel_novelty,
            mel_n_mels: config.tempogram_mel_n_mels,
            mel_fmin_hz: config.tempogram_mel_fmin_hz,
            mel_fmax_hz: config.tempogram_mel_fmax_hz,
            mel_max_filter_bins: config.tempogram_mel_max_filter_bins,
            w_mel: config.tempogram_mel_weight,
            novelty_w_spectral: config.tempogram_novelty_w_spectral,
            novelty_w_energy: config.tempogram_novelty_w_energy,
            novelty_w_hfc: config.tempogram_novelty_w_hfc,
            novelty_local_mean_window: config.tempogram_novelty_local_mean_window,
            novelty_smooth_window: config.tempogram_novelty_smooth_window,
            debug_track_id: config.debug_track_id,
            debug_gt_bpm: config.debug_gt_bpm,
            debug_top_n: config.debug_top_n,
            superflux_max_filter_bins: config.tempogram_superflux_max_filter_bins,
        };

        let use_aux_variants = config.enable_tempogram_band_fusion
            || config.enable_tempogram_mel_novelty
            || config.tempogram_band_consensus_bonus > 0.0;

        if config.enable_tempogram_multi_resolution {
            // Run single-resolution tempogram first; only escalate to multi-resolution
            // when the result looks ambiguous (prevents global regressions).
            let base_top_n = config
                .tempogram_candidates_top_n
                .max(config.tempogram_multi_res_top_k)
                .max(10);

            let base_call = if use_aux_variants {
                estimate_bpm_tempogram_with_candidates_band_fusion(
                    &magnitude_spec_frames,
                    sample_rate,
                    config.hop_size as u32,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                    base_top_n,
                    band_cfg.clone(),
                )
            } else {
                estimate_bpm_tempogram_with_candidates(
                    &magnitude_spec_frames,
                    sample_rate,
                    config.hop_size as u32,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                    base_top_n,
                )
            };

            match base_call {
                Ok((base_est, base_cands)) => {
                    let trap_low = base_est.bpm >= 55.0 && base_est.bpm <= 80.0;
                    let trap_high = base_est.bpm >= 170.0 && base_est.bpm <= 200.0;

                    // Additional ambiguity detection: if the single-resolution candidate list already
                    // contains strong tempo-family alternatives (2× or 1/2×), escalate to multi-res.
                    //
                    // This catches cases like GT~184 predicted ~92 where base BPM is not in our
                    // original “trap” window but is a classic half-time error.
                    fn cand_support(
                        cands: &[features::period::tempogram::TempogramCandidateDebug],
                        bpm: f32,
                        tol: f32,
                    ) -> f32 {
                        let mut best = 0.0f32;
                        for c in cands {
                            if (c.bpm - bpm).abs() <= tol {
                                best = best.max(c.score);
                            }
                        }
                        best
                    }

                    let tol = 2.0f32.max(config.bpm_resolution);
                    let s_base = cand_support(&base_cands, base_est.bpm, tol);
                    let s_2x = cand_support(&base_cands, base_est.bpm * 2.0, tol);
                    let s_half = cand_support(&base_cands, base_est.bpm * 0.5, tol);
                    let family_competes = (s_2x > 0.0 && s_2x >= s_base * 0.90)
                        || (s_half > 0.0 && s_half >= s_base * 0.90);

                    // IMPORTANT: Our tempogram "confidence" is currently conservative and can be low even
                    // for correct tempos. If we use a generic confidence threshold, we end up escalating
                    // on nearly every track (catastrophic for performance, especially with HPSS).
                    //
                    // For now, only escalate in the known tempo-family trap zones. We'll widen this later
                    // once we have a better uncertainty measure.
                    // Escape hatch: if confidence/agreement is poor and a 2× fold would land in the
                    // high trap zone (or a 1/2× fold would land in the low trap zone), escalate even
                    // if the candidate list didn’t surface it strongly (prevents missing half-time errors).
                    // Only use the "fold_into_trap" escape hatch for the missed half-time case:
                    // base ~90, true ~180. Do NOT trigger on base ~120 (since 120/2=60 is common and
                    // would cause unnecessary multi-res runs / regressions).
                    let fold_into_trap = base_est.bpm * 2.0 >= 170.0 && base_est.bpm * 2.0 <= 200.0;
                    let weak_base = base_est.method_agreement == 0 || base_est.confidence < 0.06;

                    let ambiguous = trap_low || trap_high || family_competes || (weak_base && fold_into_trap);
                    // Instrumentation: "triggered" means we *considered* escalation, not just "base looks ambiguous".
                    tempogram_multi_res_triggered = Some(ambiguous);

                    if let Some(track_id) = config.debug_track_id {
                        eprintln!("\n=== DEBUG base tempogram (track_id={}) ===", track_id);
                        if let Some(gt) = config.debug_gt_bpm {
                            eprintln!("GT bpm: {:.3}", gt);
                        }
                        eprintln!(
                            "base_est: bpm={:.2} conf={:.4} agree={} (trap_low={} trap_high={} ambiguous={})",
                            base_est.bpm,
                            base_est.confidence,
                            base_est.method_agreement,
                            trap_low,
                            trap_high,
                            ambiguous
                        );
                        eprintln!(
                            "ambiguity signals: family_competes={} (s_base={:.4} s_2x={:.4} s_half={:.4}) weak_base={} fold_into_trap={}",
                            family_competes,
                            s_base,
                            s_2x,
                            s_half,
                            weak_base,
                            fold_into_trap
                        );
                        if !ambiguous {
                            eprintln!("NOTE: multi-res not run (outside trap zones).");
                        }
                    }

                    let mut chosen_est = base_est.clone();
                    let mut chosen_cands = base_cands;
                    let mut used_mr = false;

                    if ambiguous {
                        match multi_resolution_tempogram_from_samples(
                            &trimmed_samples,
                            sample_rate,
                            config.frame_size,
                            config.min_bpm,
                            config.max_bpm,
                            config.bpm_resolution,
                            config.tempogram_multi_res_top_k,
                            config.tempogram_multi_res_w512,
                            config.tempogram_multi_res_w256,
                            config.tempogram_multi_res_w1024,
                            config.tempogram_multi_res_structural_discount,
                            config.tempogram_multi_res_double_time_512_factor,
                            config.tempogram_multi_res_margin_threshold,
                            config.tempogram_multi_res_use_human_prior,
                            Some(band_cfg.clone()),
                        ) {
                            Ok((mr_est, mr_cands_512)) => {
                                let mr_est_log = mr_est.clone();
                                // Choose multi-res only if it provides stronger evidence or
                                // a safer tempo-family choice in the trap regions.
                                let rel = if base_est.bpm > 1e-6 {
                                    (mr_est.bpm / base_est.bpm).max(base_est.bpm / mr_est.bpm)
                                } else {
                                    1.0
                                };
                                let family_related =
                                    (rel - 2.0).abs() < 0.05 || (rel - 1.5).abs() < 0.05 || (rel - (4.0 / 3.0)).abs() < 0.05;

                                // Hard safety rule: do not “promote” a sane in-range tempo into an extreme
                                // high tempo (e.g., 120 -> 240). Multi-res should primarily resolve
                                // octave *folding* errors, not create them.
                                let forbid_promote_high = base_est.bpm <= 180.0 && mr_est.bpm > 180.0;

                                let mr_better = !forbid_promote_high
                                    && (mr_est.confidence >= (base_est.confidence + 0.05)
                                        || (mr_est.method_agreement > base_est.method_agreement
                                            && mr_est.confidence >= base_est.confidence * 0.90)
                                        || ((trap_low || trap_high)
                                            && family_related
                                            && mr_est.confidence >= base_est.confidence * 0.88
                                            // Additional safety: only accept family moves that land in a
                                            // typical music/DJ tempo band unless base was already extreme.
                                            && ((mr_est.bpm >= 70.0 && mr_est.bpm <= 180.0) || base_est.bpm > 180.0)));

                                if mr_better {
                                    chosen_est = mr_est;
                                    chosen_cands = mr_cands_512;
                                    used_mr = true;
                                }

                                if let Some(track_id) = config.debug_track_id {
                                    eprintln!("\n=== DEBUG multi-res decision (track_id={}) ===", track_id);
                                    if let Some(gt) = config.debug_gt_bpm {
                                        eprintln!("GT bpm: {:.3}", gt);
                                    }
                                    eprintln!("base_est: bpm={:.2} conf={:.4} agree={}", base_est.bpm, base_est.confidence, base_est.method_agreement);
                                    eprintln!("mr_est:   bpm={:.2} conf={:.4} agree={}", mr_est_log.bpm, mr_est_log.confidence, mr_est_log.method_agreement);
                                    eprintln!("ambiguous(trap_low||trap_high)={}", ambiguous);
                                    eprintln!("rel={:.3} family_related={} forbid_promote_high={}", rel, family_related, forbid_promote_high);
                                    eprintln!("mr_better={} used_mr={}", mr_better, used_mr);
                                }
                            }
                            Err(e) => {
                                log::debug!("Multi-resolution escalation skipped (failed): {}", e);
                            }
                        }
                    }
                    tempogram_multi_res_used = Some(used_mr);

                    // Percussive-only fallback (HPSS) for ambiguous cases (generation improvement).
                    //
                    // Important: HPSS is expensive. We only run it when we are in the classic
                    // low-tempo ambiguity zone where sustained harmonic content commonly causes
                    // half/double-time traps.
                    let percussive_needed = ambiguous && trap_low;
                    tempogram_percussive_triggered = Some(percussive_needed);

                    if config.enable_tempogram_percussive_fallback && percussive_needed {
                        use features::onset::hpss::hpss_decompose;

                        // Decompose the already computed spectrogram at the base hop_size.
                        match hpss_decompose(&magnitude_spec_frames, config.hpss_margin) {
                            Ok((_h, p)) => {
                                // Re-run tempogram on percussive component.
                                let p_call = if use_aux_variants {
                                    estimate_bpm_tempogram_with_candidates_band_fusion(
                                        &p,
                                        sample_rate,
                                        config.hop_size as u32,
                                        config.min_bpm,
                                        config.max_bpm,
                                        config.bpm_resolution,
                                        base_top_n,
                                        band_cfg.clone(),
                                    )
                                } else {
                                    estimate_bpm_tempogram_with_candidates(
                                        &p,
                                        sample_rate,
                                        config.hop_size as u32,
                                        config.min_bpm,
                                        config.max_bpm,
                                        config.bpm_resolution,
                                        base_top_n,
                                    )
                                };

                                match p_call {
                                    Ok((p_est, p_cands)) => {
                                        // Accept percussive estimate only when it is a tempo-family move
                                        // and does not promote sane tempos into extremes.
                                        let rel = if chosen_est.bpm > 1e-6 {
                                            (p_est.bpm / chosen_est.bpm).max(chosen_est.bpm / p_est.bpm)
                                        } else {
                                            1.0
                                        };
                                        let family_related = (rel - 2.0).abs() < 0.05
                                            || (rel - 1.5).abs() < 0.05
                                            || (rel - (4.0 / 3.0)).abs() < 0.05
                                            || (rel - (3.0 / 2.0)).abs() < 0.05
                                            || (rel - (2.0 / 3.0)).abs() < 0.05
                                            || (rel - (3.0 / 4.0)).abs() < 0.05;

                                        let forbid_promote_high = chosen_est.bpm <= 180.0 && p_est.bpm > 180.0;

                                        // Slightly more permissive acceptance in the low-tempo trap region:
                                        // if percussive yields a coherent 2× tempo in a common range, take it
                                        // even if confidence is only marginally better.
                                        let base_low_trap = trap_low || base_est.bpm < 95.0;
                                        let percussive_in_common = p_est.bpm >= 70.0 && p_est.bpm <= 180.0;

                                        let p_better = !forbid_promote_high
                                            && family_related
                                            && percussive_in_common
                                            && (p_est.confidence >= chosen_est.confidence + 0.04
                                                || (base_low_trap && p_est.confidence >= chosen_est.confidence * 0.85)
                                                || (p_est.method_agreement > chosen_est.method_agreement
                                                    && p_est.confidence >= chosen_est.confidence * 0.92));

                                        if p_better {
                                            chosen_est = p_est;
                                            chosen_cands = p_cands;
                                            tempogram_percussive_used = Some(true);
                                        } else {
                                            tempogram_percussive_used = Some(false);
                                        }
                                    }
                                    Err(e) => {
                                        log::debug!("Percussive tempogram fallback failed: {}", e);
                                        tempogram_percussive_used = Some(false);
                                    }
                                }
                            }
                            Err(e) => {
                                log::debug!("HPSS decomposition for percussive tempogram failed: {}", e);
                                tempogram_percussive_used = Some(false);
                            }
                        }
                    } else if config.enable_tempogram_percussive_fallback {
                        tempogram_percussive_used = Some(false);
                    }

                    if config.emit_tempogram_candidates {
                        tempogram_candidates = Some(
                            chosen_cands
                                .into_iter()
                                .map(|c| TempoCandidateDebug {
                                    bpm: c.bpm,
                                    score: c.score,
                                    fft_norm: c.fft_norm,
                                    autocorr_norm: c.autocorr_norm,
                                    selected: c.selected,
                                })
                                .collect(),
                        );
                    }

                    log::debug!(
                        "Tempogram BPM estimate: {:.2} (confidence: {:.3}, method_agreement: {}, multi_res={})",
                        chosen_est.bpm,
                        chosen_est.confidence,
                        chosen_est.method_agreement,
                        ambiguous
                    );

                    Some(chosen_est)
                }
                Err(e) => {
                    log::warn!("Tempogram BPM detection failed: {}", e);
                    None
                }
            }
        } else if config.emit_tempogram_candidates {
            let call = if use_aux_variants {
                estimate_bpm_tempogram_with_candidates_band_fusion(
                    &magnitude_spec_frames,
                    sample_rate,
                    config.hop_size as u32,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                    config.tempogram_candidates_top_n,
                    band_cfg.clone(),
                )
            } else {
                estimate_bpm_tempogram_with_candidates(
                    &magnitude_spec_frames,
                    sample_rate,
                    config.hop_size as u32,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                    config.tempogram_candidates_top_n,
                )
            };

            match call {
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
            let call = if use_aux_variants {
                estimate_bpm_tempogram_band_fusion(
                    &magnitude_spec_frames,
                    sample_rate,
                    config.hop_size as u32,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                    band_cfg.clone(),
                )
            } else {
                estimate_bpm_tempogram(
                    &magnitude_spec_frames,
                    sample_rate,
                    config.hop_size as u32,
                    config.min_bpm,
                    config.max_bpm,
                    config.bpm_resolution,
                )
            };

            match call {
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
            tempogram_multi_res_triggered,
            tempogram_multi_res_used,
            tempogram_percussive_triggered,
            tempogram_percussive_used,
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

