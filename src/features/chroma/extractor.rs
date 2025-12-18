//! Chroma vector extraction
//!
//! Converts FFT magnitude spectrogram to 12-element chroma vectors.
//!
//! Algorithm:
//! 1. Compute STFT (Short-Time Fourier Transform)
//! 2. Convert frequency bins → semitone classes: `semitone = 12 * log2(freq / 440.0) + 57.0`
//! 3. Sum magnitude across octaves for each semitone class
//! 4. Normalize to L2 unit norm
//!
//! # Reference
//!
//! Müller, M., & Ewert, S. (2010). Chroma Toolbox: MATLAB Implementations for Extracting
//! Variants of Chroma-Based Audio Features. *Proceedings of the International Society for
//! Music Information Retrieval Conference*.
//!
//! # Example
//!
//! ```no_run
//! use stratum_dsp::features::chroma::extractor::extract_chroma;
//!
//! let samples = vec![0.0f32; 44100 * 5]; // 5 seconds of audio
//! let chroma_vectors = extract_chroma(&samples, 44100, 2048, 512)?;
//! // chroma_vectors is Vec<Vec<f32>> where each inner Vec is a 12-element chroma vector
//! # Ok::<(), stratum_dsp::AnalysisError>(())
//! ```

use crate::error::AnalysisError;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Numerical stability epsilon
const EPSILON: f32 = 1e-10;

/// Reference frequency for semitone calculation (A4)
const A4_FREQ: f32 = 440.0;

/// Semitone offset to map A4 to semitone 57 (middle of piano range)
const SEMITONE_OFFSET: f32 = 57.0;

/// Extract chroma vectors from audio samples
///
/// Computes STFT, maps frequencies to semitone classes, and sums across octaves
/// to produce 12-element chroma vectors (one per frame).
///
/// # Arguments
///
/// * `samples` - Audio samples (mono, normalized to [-1.0, 1.0])
/// * `sample_rate` - Sample rate in Hz (typically 44100 or 48000)
/// * `frame_size` - FFT frame size (default: 2048)
/// * `hop_size` - Hop size between frames (default: 512)
///
/// # Returns
///
/// Vector of 12-element chroma vectors (one per frame)
/// Each chroma vector represents the pitch-class distribution for that frame
///
/// # Errors
///
/// Returns `AnalysisError` if:
/// - Samples are empty
/// - Invalid parameters (frame_size or hop_size == 0)
/// - STFT computation fails
///
/// # Example
///
/// ```no_run
/// use stratum_dsp::features::chroma::extractor::extract_chroma;
///
/// let samples = vec![0.0f32; 44100 * 5]; // 5 seconds
/// let chroma_vectors = extract_chroma(&samples, 44100, 2048, 512)?;
///
/// // Each vector has 12 elements (one per semitone class: C, C#, D, ..., B)
/// assert_eq!(chroma_vectors[0].len(), 12);
/// # Ok::<(), stratum_dsp::AnalysisError>(())
/// ```
pub fn extract_chroma(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
) -> Result<Vec<Vec<f32>>, AnalysisError> {
    extract_chroma_with_options(samples, sample_rate, frame_size, hop_size, true, 0.5)
}

/// Extract chroma vectors with configurable options
///
/// Same as `extract_chroma` but allows configuration of soft mapping.
pub fn extract_chroma_with_options(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    soft_mapping: bool,
    soft_mapping_sigma: f32,
) -> Result<Vec<Vec<f32>>, AnalysisError> {
    log::debug!("Extracting chroma: {} samples at {} Hz, frame_size={}, hop_size={}, soft_mapping={}",
                samples.len(), sample_rate, frame_size, hop_size, soft_mapping);
    
    if samples.is_empty() {
        return Err(AnalysisError::InvalidInput("Empty audio samples".to_string()));
    }
    
    if frame_size == 0 {
        return Err(AnalysisError::InvalidInput("Frame size must be > 0".to_string()));
    }
    
    if hop_size == 0 {
        return Err(AnalysisError::InvalidInput("Hop size must be > 0".to_string()));
    }
    
    if sample_rate == 0 {
        return Err(AnalysisError::InvalidInput("Sample rate must be > 0".to_string()));
    }
    
    // Step 1: Compute STFT
    let stft_magnitudes = compute_stft(samples, frame_size, hop_size)?;
    
    if stft_magnitudes.is_empty() {
        return Ok(vec![]);
    }
    
    // Step 2: Convert each frame to chroma vector
    let mut chroma_vectors = Vec::with_capacity(stft_magnitudes.len());
    
    for frame in &stft_magnitudes {
        let chroma = frame_to_chroma(frame, sample_rate, frame_size, soft_mapping, soft_mapping_sigma)?;
        chroma_vectors.push(chroma);
    }
    
    log::debug!("Extracted {} chroma vectors", chroma_vectors.len());
    
    Ok(chroma_vectors)
}

/// Compute STFT (Short-Time Fourier Transform)
///
/// Computes magnitude spectrogram using windowed FFT.
///
/// # Arguments
///
/// * `samples` - Audio samples
/// * `frame_size` - FFT frame size (window size)
/// * `hop_size` - Hop size between frames
///
/// # Returns
///
/// Vector of magnitude spectra (one per frame)
/// Each inner vector has `frame_size / 2 + 1` frequency bins
pub fn compute_stft(
    samples: &[f32],
    frame_size: usize,
    hop_size: usize,
) -> Result<Vec<Vec<f32>>, AnalysisError> {
    let n_samples = samples.len();
    
    if n_samples < frame_size {
        // Not enough samples for even one frame
        return Ok(vec![]);
    }
    
    // Compute number of frames
    let n_frames = (n_samples - frame_size) / hop_size + 1;
    let mut magnitudes = Vec::with_capacity(n_frames);
    
    // Create Hann window
    let window: Vec<f32> = (0..frame_size)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect();
    
    // FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(frame_size);
    
    // Process each frame
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let end = start + frame_size;
        
        if end > n_samples {
            break;
        }
        
        // Window the frame
        let mut fft_input: Vec<Complex<f32>> = samples[start..end]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        
        // Forward FFT
        fft.process(&mut fft_input);
        
        // Compute magnitude spectrum (only need first frame_size/2 + 1 bins for real FFT)
        let n_bins = frame_size / 2 + 1;
        let magnitude: Vec<f32> = fft_input[..n_bins]
            .iter()
            .map(|x| (x.re * x.re + x.im * x.im).sqrt())
            .collect();
        
        magnitudes.push(magnitude);
    }
    
    Ok(magnitudes)
}

/// Convert a single FFT magnitude frame to chroma vector
///
/// Maps frequency bins to semitone classes and sums across octaves.
///
/// # Arguments
///
/// * `magnitude_frame` - FFT magnitude spectrum for one frame
/// * `sample_rate` - Sample rate in Hz
/// * `fft_size` - FFT size (same as frame_size)
/// * `soft_mapping` - Enable soft mapping (spread to neighboring semitones)
/// * `soft_mapping_sigma` - Standard deviation for soft mapping (in semitones)
///
/// # Returns
///
/// 12-element chroma vector (one per semitone class)
fn frame_to_chroma(
    magnitude_frame: &[f32],
    sample_rate: u32,
    fft_size: usize,
    soft_mapping: bool,
    soft_mapping_sigma: f32,
) -> Result<Vec<f32>, AnalysisError> {
    // Initialize chroma vector (12 semitone classes)
    let mut chroma = vec![0.0f32; 12];
    
    // Frequency resolution: sample_rate / fft_size
    let freq_resolution = sample_rate as f32 / fft_size as f32;
    
    // Process each frequency bin
    for (bin_idx, &magnitude) in magnitude_frame.iter().enumerate() {
        // Convert bin index to frequency
        let freq = bin_idx as f32 * freq_resolution;
        
        // Skip DC component and very low frequencies (< 80 Hz, below typical musical range)
        if freq < 80.0 {
            continue;
        }
        
        // Skip Nyquist and above
        if freq >= sample_rate as f32 / 2.0 {
            break;
        }
        
        // Convert frequency to semitone
        // Formula: semitone = 12 * log2(freq / 440.0) + 57.0
        let semitone = 12.0 * (freq / A4_FREQ).log2() + SEMITONE_OFFSET;
        
        if soft_mapping {
            // Soft mapping: spread magnitude to neighboring semitone classes
            // Use Gaussian weighting: weight = exp(-distance² / (2 * σ²))
            let mut semitone_fractional = semitone % 12.0;
            if semitone_fractional < 0.0 {
                semitone_fractional += 12.0;
            }
            
            // Distribute to primary semitone class and neighbors
            let primary_class = semitone_fractional.round() as i32;
            let primary_class = ((primary_class % 12) + 12) % 12;
            
            // Compute weights for primary and neighboring classes
            for offset in -1..=1 {
                let target_class = ((primary_class + offset) % 12 + 12) % 12;
                let target_semitone = primary_class as f32 + offset as f32;
                let distance = (semitone_fractional - target_semitone).abs();
                
                // Gaussian weight
                let weight = (-distance * distance / (2.0 * soft_mapping_sigma * soft_mapping_sigma)).exp();
                
                chroma[target_class as usize] += magnitude * weight;
            }
        } else {
            // Hard assignment: assign to nearest semitone class
            let semitone_class = (semitone.round() as i32) % 12;
            // Handle negative modulo
            let semitone_class = if semitone_class < 0 {
                semitone_class + 12
            } else {
                semitone_class
            } as usize;
            
            // Sum magnitude across octaves for this semitone class
            chroma[semitone_class] += magnitude;
        }
    }
    
    // L2 normalize chroma vector
    let norm: f32 = chroma.iter().map(|&x| x * x).sum::<f32>().sqrt();
    
    if norm > EPSILON {
        for x in &mut chroma {
            *x /= norm;
        }
    }
    
    Ok(chroma)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_chroma_empty() {
        let result = extract_chroma(&[], 44100, 2048, 512);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_extract_chroma_short() {
        // Very short audio (less than one frame)
        let samples = vec![0.0f32; 1000];
        let result = extract_chroma(&samples, 44100, 2048, 512);
        assert!(result.is_ok());
        let chroma_vectors = result.unwrap();
        assert_eq!(chroma_vectors.len(), 0);
    }
    
    #[test]
    fn test_extract_chroma_basic() {
        // Generate a simple sine wave at A4 (440 Hz)
        let sample_rate = 44100;
        let duration_samples = sample_rate * 2; // 2 seconds
        let mut samples = Vec::with_capacity(duration_samples);
        
        for i in 0..duration_samples {
            let t = i as f32 / sample_rate as f32;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        
        let result = extract_chroma(&samples, sample_rate as u32, 2048, 512);
        assert!(result.is_ok());
        
        let chroma_vectors = result.unwrap();
        assert!(!chroma_vectors.is_empty());
        
        // Check that each chroma vector has 12 elements
        for chroma in &chroma_vectors {
            assert_eq!(chroma.len(), 12);
            
            // Check normalization (L2 norm should be ~1.0)
            let norm: f32 = chroma.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01 || norm < EPSILON);
        }
        
        // A4 should map to semitone class 9 (A)
        // Check that A (index 9) has significant energy
        let avg_chroma: Vec<f32> = (0..12)
            .map(|i| {
                chroma_vectors.iter().map(|v| v[i]).sum::<f32>() / chroma_vectors.len() as f32
            })
            .collect();
        
        // A (index 9) should be prominent
        assert!(avg_chroma[9] > 0.1, "A semitone class should be prominent for A4 tone");
    }
    
    #[test]
    fn test_frame_to_chroma() {
        let sample_rate = 44100;
        let fft_size = 2048;
        
        // Create a magnitude spectrum with energy at A4 (440 Hz)
        let mut magnitude = vec![0.0f32; fft_size / 2 + 1];
        let bin_a4 = (440.0 * fft_size as f32 / sample_rate as f32) as usize;
        if bin_a4 < magnitude.len() {
            magnitude[bin_a4] = 1.0;
        }
        
        let chroma = frame_to_chroma(&magnitude, sample_rate, fft_size, false, 0.5).unwrap();
        assert_eq!(chroma.len(), 12);
        
        // Check normalization
        let norm: f32 = chroma.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01 || norm < EPSILON);
    }
    
    #[test]
    fn test_invalid_parameters() {
        let samples = vec![0.0f32; 10000];
        
        // Zero frame size
        assert!(extract_chroma(&samples, 44100, 0, 512).is_err());
        
        // Zero hop size
        assert!(extract_chroma(&samples, 44100, 2048, 0).is_err());
        
        // Zero sample rate
        assert!(extract_chroma(&samples, 0, 2048, 512).is_err());
    }
    
    #[test]
    fn test_soft_chroma_mapping() {
        let sample_rate = 44100;
        let duration_samples = sample_rate * 2; // 2 seconds
        let mut samples = Vec::with_capacity(duration_samples);
        
        for i in 0..duration_samples {
            let t = i as f32 / sample_rate as f32;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }
        
        // Test with soft mapping enabled
        let result_soft = extract_chroma_with_options(&samples, sample_rate as u32, 2048, 512, true, 0.5);
        assert!(result_soft.is_ok());
        
        // Test with soft mapping disabled
        let result_hard = extract_chroma_with_options(&samples, sample_rate as u32, 2048, 512, false, 0.5);
        assert!(result_hard.is_ok());
        
        // Both should produce valid chroma vectors
        let chroma_soft = result_soft.unwrap();
        let chroma_hard = result_hard.unwrap();
        
        assert_eq!(chroma_soft.len(), chroma_hard.len());
        assert!(!chroma_soft.is_empty());
    }
}
