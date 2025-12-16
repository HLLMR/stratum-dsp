#!/usr/bin/env python3
"""
Generate synthetic test audio fixtures for Stratum DSP testing.

This script generates the test fixtures described in tests/fixtures/README.md:
- 120bpm_4bar.wav: Kick drum pattern at 120 BPM
- 128bpm_4bar.wav: Kick drum pattern at 128 BPM
- cmajor_scale.wav: C major scale for key detection
- mixed_silence.wav: Audio with leading/trailing silence

Requirements:
    pip install -r scripts/requirements.txt
    OR
    pip install numpy soundfile
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44100


def generate_kick_drum(bpm: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate a simple kick drum pattern at the specified BPM.
    
    Args:
        bpm: Beats per minute
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Normalized audio samples as numpy array
    """
    beat_interval = 60.0 / bpm
    samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
    
    for beat_time in np.arange(0, duration, beat_interval):
        beat_sample = int(beat_time * sample_rate)
        # Simple kick: low frequency sine wave with exponential decay
        kick_duration = int(0.1 * sample_rate)  # 100ms kick
        t = np.arange(kick_duration) / sample_rate
        
        # 60Hz fundamental with harmonics
        kick = (np.sin(2 * np.pi * 60 * t) * 0.6 +
                np.sin(2 * np.pi * 120 * t) * 0.3 +
                np.sin(2 * np.pi * 180 * t) * 0.1)
        
        # Exponential decay envelope
        envelope = np.exp(-t * 10)
        kick *= envelope
        
        # Add to samples
        end = min(beat_sample + kick_duration, len(samples))
        samples[beat_sample:end] += kick[:end - beat_sample]
    
    # Normalize to [-1.0, 1.0]
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val
    
    return samples


def generate_c_major_scale(duration_per_note: float = 0.5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate a C major scale (C-D-E-F-G-A-B-C).
    
    Args:
        duration_per_note: Duration of each note in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Normalized audio samples as numpy array
    """
    # C major scale frequencies (starting from C4 = 261.63 Hz)
    frequencies = [
        261.63,  # C
        293.66,  # D
        329.63,  # E
        349.23,  # F
        392.00,  # G
        440.00,  # A
        493.88,  # B
        523.25,  # C (octave)
    ]
    
    samples_list = []
    
    for freq in frequencies:
        note_samples = int(duration_per_note * sample_rate)
        t = np.arange(note_samples) / sample_rate
        
        # Generate sine wave with fade in/out to avoid clicks
        note = np.sin(2 * np.pi * freq * t)
        
        # Apply envelope (fade in/out)
        fade_samples = int(0.05 * sample_rate)  # 50ms fade
        envelope = np.ones(note_samples)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        note *= envelope
        samples_list.append(note)
    
    # Concatenate all notes
    samples = np.concatenate(samples_list).astype(np.float32)
    
    # Normalize
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val
    
    return samples


def generate_mixed_silence(silence_duration: float = 5.0, 
                          audio_duration: float = 5.0,
                          sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate audio with leading silence, audio content, and trailing silence.
    
    Args:
        silence_duration: Duration of leading/trailing silence in seconds
        audio_duration: Duration of audio content in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Audio samples with silence and content
    """
    silence_samples = int(silence_duration * sample_rate)
    audio_samples = int(audio_duration * sample_rate)
    
    # Generate silence
    leading_silence = np.zeros(silence_samples, dtype=np.float32)
    trailing_silence = np.zeros(silence_samples, dtype=np.float32)
    
    # Generate audio content (simple 440Hz tone)
    t = np.arange(audio_samples) / sample_rate
    audio_content = np.sin(2 * np.pi * 440.0 * t).astype(np.float32) * 0.5
    
    # Concatenate: silence + audio + silence
    samples = np.concatenate([leading_silence, audio_content, trailing_silence])
    
    return samples


def main():
    """Generate all test fixtures."""
    print("Generating test fixtures for Stratum DSP...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # 1. 120 BPM kick pattern (4 bars = 8 seconds at 120 BPM)
    print("Generating 120bpm_4bar.wav...")
    kick_120 = generate_kick_drum(120.0, 8.0)
    sf.write(OUTPUT_DIR / "120bpm_4bar.wav", kick_120, SAMPLE_RATE)
    print(f"  [OK] Created: {len(kick_120) / SAMPLE_RATE:.2f} seconds, {len(kick_120)} samples")
    
    # 2. 128 BPM kick pattern (4 bars = 7.5 seconds at 128 BPM)
    print("Generating 128bpm_4bar.wav...")
    kick_128 = generate_kick_drum(128.0, 7.5)
    sf.write(OUTPUT_DIR / "128bpm_4bar.wav", kick_128, SAMPLE_RATE)
    print(f"  [OK] Created: {len(kick_128) / SAMPLE_RATE:.2f} seconds, {len(kick_128)} samples")
    
    # 3. C major scale (~4 seconds: 8 notes × 0.5s each)
    print("Generating cmajor_scale.wav...")
    scale = generate_c_major_scale(duration_per_note=0.5)
    sf.write(OUTPUT_DIR / "cmajor_scale.wav", scale, SAMPLE_RATE)
    print(f"  [OK] Created: {len(scale) / SAMPLE_RATE:.2f} seconds, {len(scale)} samples")
    
    # 4. Mixed silence (5s silence + 5s audio + 5s silence = 15s total)
    print("Generating mixed_silence.wav...")
    mixed = generate_mixed_silence(silence_duration=5.0, audio_duration=5.0)
    sf.write(OUTPUT_DIR / "mixed_silence.wav", mixed, SAMPLE_RATE)
    print(f"  [OK] Created: {len(mixed) / SAMPLE_RATE:.2f} seconds, {len(mixed)} samples")
    
    print()
    print("[OK] All test fixtures generated successfully!")
    print(f"  Location: {OUTPUT_DIR}")
    print()
    print("Expected results:")
    print("  - 120bpm_4bar.wav: ~16 onsets (4 beats/bar × 4 bars)")
    print("  - 128bpm_4bar.wav: ~16 onsets (4 beats/bar × 4 bars)")
    print("  - cmajor_scale.wav: Key = C Major, confidence > 0.8")
    print("  - mixed_silence.wav: Trimmed duration = ~5 seconds")


if __name__ == "__main__":
    main()

