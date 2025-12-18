//! Configuration parameters for audio analysis

use crate::preprocessing::normalization::NormalizationMethod;

/// Analysis configuration parameters
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    // Preprocessing
    /// Silence detection threshold in dB (default: -40.0)
    /// Frames with RMS below this threshold are considered silent
    pub min_amplitude_db: f32,
    
    /// Normalization method to use (default: Peak)
    pub normalization: NormalizationMethod,

    /// Enable normalization step (default: true)
    pub enable_normalization: bool,

    /// Enable silence detection + trimming step (default: true)
    pub enable_silence_trimming: bool,

    // Onset detection (used by beat tracking and legacy BPM fallback)
    /// Enable multi-detector onset consensus (spectral flux + HFC + optional HPSS) (default: true)
    ///
    /// Note: Tempogram BPM does not use this onset list, but legacy BPM + beat tracking do.
    pub enable_onset_consensus: bool,

    /// Threshold percentile for STFT-based onset detectors (spectral flux / HFC / HPSS) (default: 0.80)
    /// Range: [0.0, 1.0]
    pub onset_threshold_percentile: f32,

    /// Onset clustering tolerance window in milliseconds for consensus voting (default: 50 ms)
    pub onset_consensus_tolerance_ms: u32,

    /// Consensus method weights [energy_flux, spectral_flux, hfc, hpss] (default: equal weights)
    pub onset_consensus_weights: [f32; 4],

    /// Enable HPSS-based onset detector inside consensus (default: false; more expensive)
    pub enable_hpss_onsets: bool,

    /// HPSS median-filter margin (default: 10). Typical values: 5–20.
    pub hpss_margin: usize,
    
    // BPM detection
    /// Force legacy BPM estimation (Phase 1B autocorrelation + comb filter) and skip tempogram.
    /// Default: false.
    ///
    /// Intended for A/B validation and hybrid/consensus experimentation.
    pub force_legacy_bpm: bool,

    /// Enable BPM fusion (compute tempogram + legacy in parallel, then choose using consensus logic).
    /// Default: false (tempogram-only unless it fails, then legacy fallback).
    pub enable_bpm_fusion: bool,

    /// Enable legacy BPM guardrails (soft confidence caps by tempo range).
    /// Default: true.
    pub enable_legacy_bpm_guardrails: bool,

    /// Enable **true** multi-resolution tempogram BPM estimation.
    ///
    /// When enabled, BPM estimation recomputes STFT at hop sizes {256, 512, 1024} and fuses
    /// candidates using a cross-resolution scoring rule. This is intended to reduce
    /// metrical-level (T vs 2T vs T/2) errors.
    ///
    /// Default: true (Phase 1F tuning path).
    pub enable_tempogram_multi_resolution: bool,

    /// Multi-resolution fusion: number of hop=512 candidates to consider as anchors.
    /// Default: 10.
    pub tempogram_multi_res_top_k: usize,

    /// Multi-resolution fusion weight for hop=512 (global beat).
    pub tempogram_multi_res_w512: f32,
    /// Multi-resolution fusion weight for hop=256 (fine transients).
    pub tempogram_multi_res_w256: f32,
    /// Multi-resolution fusion weight for hop=1024 (structural/metre level).
    pub tempogram_multi_res_w1024: f32,

    /// Structural discount factor applied when hop=1024 supports 2T instead of T.
    pub tempogram_multi_res_structural_discount: f32,

    /// Factor applied to hop=512 support when evaluating the 2T / T/2 hypotheses.
    pub tempogram_multi_res_double_time_512_factor: f32,

    /// Minimum score margin (absolute) required to switch between T / 2T / T/2 hypotheses.
    pub tempogram_multi_res_margin_threshold: f32,

    /// Enable a gentle human-tempo prior as a tie-breaker (only when scores are very close).
    /// Default: false.
    pub tempogram_multi_res_use_human_prior: bool,

    /// Enable HPSS percussive-only tempogram fallback (ambiguous-only).
    ///
    /// This computes an HPSS decomposition on the (already computed) STFT magnitudes and re-runs
    /// tempogram on the percussive component. Intended to reduce low-tempo half/double-time traps
    /// caused by sustained harmonic energy.
    ///
    /// Default: true (Phase 1F tuning path).
    pub enable_tempogram_percussive_fallback: bool,

    /// Enable multi-band novelty fusion inside the tempogram estimator.
    ///
    /// This computes novelty curves over low/mid/high frequency bands, runs the tempogram
    /// on each, then fuses their support when scoring BPM candidates. This is primarily
    /// intended to improve **candidate generation** (getting GT into top-N candidates),
    /// which is currently the limiting factor after metrical selection improvements.
    ///
    /// Default: true (Phase 1F tuning path).
    pub enable_tempogram_band_fusion: bool,

    /// Band split cutoffs (Hz). Bands are: low=[~0..low_max], mid=[low_max..mid_max], high=[mid_max..high_max].
    /// If `tempogram_band_high_max_hz <= 0`, high extends to Nyquist.
    pub tempogram_band_low_max_hz: f32,
    /// Upper cutoff for the mid band (Hz).
    pub tempogram_band_mid_max_hz: f32,
    /// Upper cutoff for the high band (Hz). If <= 0, uses Nyquist.
    pub tempogram_band_high_max_hz: f32,

    /// Weight for the full-band tempogram contribution when band-score fusion is enabled.
    pub tempogram_band_w_full: f32,
    /// Weight for the low band contribution.
    pub tempogram_band_w_low: f32,
    /// Weight for the mid band contribution.
    pub tempogram_band_w_mid: f32,
    /// Weight for the high band contribution.
    pub tempogram_band_w_high: f32,

    /// If true, multi-band tempograms contribute **only to candidate seeding** (peak proposals),
    /// while final candidate scoring remains full-band-only.
    ///
    /// This is the safer default: high-frequency bands often emphasize subdivisions (hi-hats),
    /// which can otherwise increase 2× / 3:2 metrical errors if they directly affect scoring.
    pub tempogram_band_seed_only: bool,

    /// Minimum per-band normalized support required to count as "supporting" a BPM candidate
    /// for band-consensus scoring.
    ///
    /// Range: [0, 1]. Default: 0.25.
    pub tempogram_band_support_threshold: f32,

    /// Bonus multiplier applied when **multiple bands** support the same BPM candidate.
    ///
    /// This is a lightweight "consensus" heuristic intended to reduce metrical/subdivision errors
    /// (e.g., a 2× tempo supported only by the high band should not win over a tempo supported by
    /// low+mid bands).
    ///
    /// Score adjustment: `score *= (1 + bonus * max(0, support_bands - 1))`.
    pub tempogram_band_consensus_bonus: f32,

    /// Tempogram novelty weights for combining {spectral, energy, HFC}.
    pub tempogram_novelty_w_spectral: f32,
    /// Tempogram novelty weight for energy flux.
    pub tempogram_novelty_w_energy: f32,
    /// Tempogram novelty weight for HFC.
    pub tempogram_novelty_w_hfc: f32,
    /// Tempogram novelty conditioning windows.
    pub tempogram_novelty_local_mean_window: usize,
    /// Tempogram novelty moving-average smoothing window (frames). Use 0/1 to disable.
    pub tempogram_novelty_smooth_window: usize,

    /// Debug: if set, the `analyze_file` example will pass this track ID through to the
    /// multi-resolution fusion so it can print detailed scoring diagnostics.
    pub debug_track_id: Option<u32>,
    /// Debug: optional ground-truth BPM passed alongside `debug_track_id`.
    pub debug_gt_bpm: Option<f32>,
    /// Debug: number of top candidates per hop to print when `debug_track_id` is set.
    pub debug_top_n: usize,

    /// Enable log-mel novelty tempogram as an additional candidate generator/support signal.
    ///
    /// This computes a log-mel SuperFlux-style novelty curve, then runs the tempogram on it.
    /// The resulting candidates are used for seeding and for the consensus bonus logic.
    pub enable_tempogram_mel_novelty: bool,
    /// Mel band count used by log-mel novelty.
    pub tempogram_mel_n_mels: usize,
    /// Minimum mel frequency (Hz).
    pub tempogram_mel_fmin_hz: f32,
    /// Maximum mel frequency (Hz). If <= 0, uses Nyquist.
    pub tempogram_mel_fmax_hz: f32,
    /// Max-filter neighborhood radius in mel bins (SuperFlux-style reference).
    pub tempogram_mel_max_filter_bins: usize,
    /// Weight for mel variant when band scoring fusion is enabled (`seed_only=false`).
    pub tempogram_mel_weight: f32,

    /// SuperFlux max-filter neighborhood radius (bins) used by the tempogram novelty extractor.
    pub tempogram_superflux_max_filter_bins: usize,

    /// Emit tempogram BPM candidate list (top-N) into `AnalysisMetadata` for validation/tuning.
    ///
    /// Default: false (avoid bloating outputs in normal use).
    pub emit_tempogram_candidates: bool,

    /// Number of tempogram candidates to emit when `emit_tempogram_candidates` is enabled.
    /// Default: 10.
    pub tempogram_candidates_top_n: usize,

    /// Legacy guardrails: preferred BPM range (default: 75–150).
    pub legacy_bpm_preferred_min: f32,
    /// Legacy guardrails: preferred BPM range upper bound (default: 150).
    pub legacy_bpm_preferred_max: f32,

    /// Legacy guardrails: soft BPM range (default: 60–180).
    /// Values in [soft_min, preferred_min) or (preferred_max, soft_max] get a medium cap.
    pub legacy_bpm_soft_min: f32,
    /// Legacy guardrails: soft BPM range upper bound (default: 180).
    pub legacy_bpm_soft_max: f32,

    /// Legacy guardrails: confidence caps by range.
    /// - preferred: inside [preferred_min, preferred_max]
    /// - soft: inside [soft_min, soft_max] but outside preferred
    /// - extreme: outside [soft_min, soft_max]
    ///
    /// **Multiplier semantics**: these are applied as `confidence *= multiplier` to legacy
    /// candidates/estimates (softly biasing the selection).
    pub legacy_bpm_conf_mul_preferred: f32,
    /// Legacy guardrails: confidence multiplier for the soft band (default: 0.50).
    pub legacy_bpm_conf_mul_soft: f32,
    /// Legacy guardrails: confidence multiplier for extremes (default: 0.10).
    pub legacy_bpm_conf_mul_extreme: f32,

    /// Minimum BPM to consider (default: 60.0)
    pub min_bpm: f32,
    
    /// Maximum BPM to consider (default: 180.0)
    pub max_bpm: f32,
    
    /// BPM resolution for comb filterbank (default: 1.0)
    pub bpm_resolution: f32,
    
    // STFT parameters
    /// Frame size for STFT (default: 2048)
    pub frame_size: usize,
    
    /// Hop size for STFT (default: 512)
    pub hop_size: usize,
    
    // Key detection
    /// Center frequency for chroma extraction (default: 440.0 Hz, A4)
    pub center_frequency: f32,
    
    /// Enable soft chroma mapping (default: true)
    /// Soft mapping spreads frequency bins to neighboring semitones for robustness
    pub soft_chroma_mapping: bool,
    
    /// Soft mapping standard deviation in semitones (default: 0.5)
    /// Lower values = sharper mapping, higher values = more spread
    pub soft_mapping_sigma: f32,
    
    /// Chroma sharpening power (default: 1.0 = no sharpening, 1.5-2.0 recommended)
    /// Power > 1.0 emphasizes prominent semitones, improving key detection
    pub chroma_sharpening_power: f32,
    
    // ML refinement
    /// Enable ML refinement (requires ml feature)
    #[cfg(feature = "ml")]
    pub enable_ml_refinement: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_amplitude_db: -40.0,
            normalization: NormalizationMethod::Peak,
            enable_normalization: true,
            enable_silence_trimming: true,
            enable_onset_consensus: true,
            onset_threshold_percentile: 0.80,
            onset_consensus_tolerance_ms: 50,
            onset_consensus_weights: [0.25, 0.25, 0.25, 0.25],
            enable_hpss_onsets: false,
            hpss_margin: 10,
            force_legacy_bpm: false,
            enable_bpm_fusion: false,
            enable_legacy_bpm_guardrails: true,
            enable_tempogram_multi_resolution: true,
            tempogram_multi_res_top_k: 25,
            tempogram_multi_res_w512: 0.45,
            tempogram_multi_res_w256: 0.35,
            tempogram_multi_res_w1024: 0.20,
            tempogram_multi_res_structural_discount: 0.85,
            tempogram_multi_res_double_time_512_factor: 0.92,
            tempogram_multi_res_margin_threshold: 0.08,
            tempogram_multi_res_use_human_prior: false,
            // HPSS percussive fallback is very expensive and (so far) has not shown consistent gains.
            // Keep it opt-in to avoid multi-second outliers during batch runs.
            enable_tempogram_percussive_fallback: false,
            enable_tempogram_band_fusion: true,
            // Default cutoffs (Hz): ~kick/bass fundamentals, then body/rhythm textures, then attacks.
            tempogram_band_low_max_hz: 200.0,
            tempogram_band_mid_max_hz: 2000.0,
            tempogram_band_high_max_hz: 8000.0,
            // Default weights: keep full-band as anchor, but allow bands to pull candidates into view.
            tempogram_band_w_full: 0.40,
            tempogram_band_w_low: 0.25,
            tempogram_band_w_mid: 0.20,
            tempogram_band_w_high: 0.15,
            tempogram_band_seed_only: true,
            tempogram_band_support_threshold: 0.25,
            tempogram_band_consensus_bonus: 0.08,
            // Novelty weighting defaults (tuned on 200-track validation):
            // shift weight toward transient-heavy signals (energy/HFC) to reduce octave/subdivision traps.
            tempogram_novelty_w_spectral: 0.30,
            tempogram_novelty_w_energy: 0.35,
            tempogram_novelty_w_hfc: 0.35,
            tempogram_novelty_local_mean_window: 16,
            tempogram_novelty_smooth_window: 5,
            debug_track_id: None,
            debug_gt_bpm: None,
            debug_top_n: 5,
            enable_tempogram_mel_novelty: true,
            tempogram_mel_n_mels: 40,
            tempogram_mel_fmin_hz: 30.0,
            tempogram_mel_fmax_hz: 8000.0,
            tempogram_mel_max_filter_bins: 2,
            tempogram_mel_weight: 0.15,
            tempogram_superflux_max_filter_bins: 4,
            emit_tempogram_candidates: false,
            tempogram_candidates_top_n: 10,
            // Tuned defaults (empirical, small-batch): slightly wider preferred band and
            // slightly less aggressive down-weighting while keeping a strong extreme penalty.
            legacy_bpm_preferred_min: 72.0,
            legacy_bpm_preferred_max: 168.0,
            legacy_bpm_soft_min: 60.0,
            legacy_bpm_soft_max: 210.0,
            legacy_bpm_conf_mul_preferred: 1.30,
            legacy_bpm_conf_mul_soft: 0.70,
            legacy_bpm_conf_mul_extreme: 0.01,
            min_bpm: 40.0,  // Lowered from 60.0 to catch slower tracks (ballads, ambient, etc.)
            max_bpm: 240.0, // Raised from 180.0 to catch high-tempo tracks (drum & bass, etc.)
            bpm_resolution: 1.0,
            frame_size: 2048,
            hop_size: 512,
            center_frequency: 440.0,
            soft_chroma_mapping: true,
            soft_mapping_sigma: 0.5,
            chroma_sharpening_power: 1.0, // No sharpening by default (can be enabled with 1.5-2.0)
            #[cfg(feature = "ml")]
            enable_ml_refinement: false,
        }
    }
}

