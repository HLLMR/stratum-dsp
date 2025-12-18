//! Archived: Audio decoding using Symphonia (placeholder)
//!
//! This module was originally added as a future-facing library API for file decoding, but the
//! project currently performs decoding in the example CLI (`examples/analyze_file.rs`) and keeps
//! the core library API sample-based (`analyze_audio(samples, sample_rate, ...)`).
//!
//! The original placeholder was removed from the compiled crate to avoid exposing a
//! `NotImplemented` public API surface.

/// Decode audio file to PCM samples (ARCHIVED PLACEHOLDER)
///
/// Note: This is intentionally not wired into the crate. Kept for future review only.
pub fn decode_audio(
    _path: &str,
) -> Result<(Vec<f32>, u32, u32), String> {
    Err("Archived placeholder: audio decoding API not implemented".to_string())
}


