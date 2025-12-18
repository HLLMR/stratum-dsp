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

