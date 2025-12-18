#!/usr/bin/env python3
"""
Prepare a test batch from FMA Small dataset metadata.

This script reads FMA metadata, filters for tracks with BPM ground truth (and key if available),
and creates a test batch CSV file for validation.

Ground truth comes from `fma_metadata/echonest.csv` audio features (tempo, key, mode) when present.
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path


def find_track_file(fma_path: Path, track_id: int) -> Path:
    """Find the audio file for a given track ID."""
    # FMA uses directory structure: XXX/XXXXXX.mp3 where XXX = track_id // 1000
    subdir = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return fma_path / subdir / filename


def read_fma_tracks_csv(path: Path) -> dict:
    """Read FMA tracks.csv with hierarchical structure."""
    tracks = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Skip first 2 header rows
        next(reader)  # Column category row
        next(reader)  # Column name row
        header_row = next(reader)  # Actual column names
        
        # Find track_id column index
        track_id_idx = header_row.index("track_id") if "track_id" in header_row else 0
        
        # Find genre_top column (it's in the track.* hierarchy)
        genre_idx = None
        for i, col in enumerate(header_row):
            if col == "track.genre_top":
                genre_idx = i
                break
        
        # Read data rows
        for row in reader:
            if len(row) > track_id_idx:
                try:
                    track_id = int(row[track_id_idx])
                    genre = row[genre_idx] if genre_idx and len(row) > genre_idx else ""
                    tracks[track_id] = {"genre": genre}
                except (ValueError, IndexError):
                    continue
    
    return tracks


def echonest_key_mode_to_name(key: int, mode: int) -> str:
    """
    Convert Echonest key/mode to a key name.

    Echonest convention:
      - key: 0..11 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
      - mode: 1=major, 0=minor
    """
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if key < 0 or key >= len(names):
        return ""
    note = names[key]
    if mode == 0:
        return f"{note}m"
    if mode == 1:
        return note
    return ""


def read_fma_echonest_csv(path: Path) -> dict:
    """Read FMA echonest.csv with hierarchical structure to get BPM (tempo) and key/mode (when available)."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Read hierarchical header rows
        h1 = next(reader)  # Column category row (echonest, ...)
        h2 = next(reader)  # Column subcategory row (audio_features, ...)
        h3 = next(reader)  # Column name row (tempo, ...)
        h4 = next(reader)  # track_id row
        
        # Find track_id column index
        track_id_idx = h4.index("track_id") if "track_id" in h4 else 0
        
        # Find columns by matching all three header levels
        tempo_idx = None
        key_idx = None
        mode_idx = None
        for i in range(len(h1)):
            if (h1[i] == "echonest" and 
                i < len(h2) and h2[i] == "audio_features" and
                i < len(h3) and h3[i] == "tempo"):
                tempo_idx = i
            if (h1[i] == "echonest" and 
                i < len(h2) and h2[i] == "audio_features" and
                i < len(h3) and h3[i] == "key"):
                key_idx = i
            if (h1[i] == "echonest" and 
                i < len(h2) and h2[i] == "audio_features" and
                i < len(h3) and h3[i] == "mode"):
                mode_idx = i
        
        if tempo_idx is None:
            print("WARNING: Could not find tempo column in echonest.csv")
            return data
        
        # Read data rows
        for row in reader:
            if len(row) > max(track_id_idx, tempo_idx):
                try:
                    track_id = int(row[track_id_idx])
                    tempo_str = row[tempo_idx].strip()
                    if tempo_str:
                        tempo = float(tempo_str)
                        if tempo > 0:
                            if track_id not in data:
                                data[track_id] = {}
                            data[track_id]["tempo"] = tempo

                    # Key/mode are optional (some metadata distributions may omit them)
                    if key_idx is not None and mode_idx is not None:
                        if len(row) > max(key_idx, mode_idx):
                            key_str = row[key_idx].strip()
                            mode_str = row[mode_idx].strip()
                            if key_str and mode_str:
                                try:
                                    k = int(float(key_str))
                                    m = int(float(mode_str))
                                    key_name = echonest_key_mode_to_name(k, m)
                                    if key_name:
                                        if track_id not in data:
                                            data[track_id] = {}
                                        data[track_id]["key"] = key_name
                                except ValueError:
                                    pass
                except (ValueError, IndexError):
                    continue
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test batch from FMA Small dataset"
    )
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=20,
        help="Number of tracks to include in test batch (default: 20)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../validation-data",
        help="Path to validation data directory (default: ../validation-data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible track selection",
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Paths
    data_path = Path(args.data_path)
    fma_path = data_path / "fma_small"
    metadata_path = data_path / "fma_metadata"
    results_dir = data_path / "results"
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("FMA Validation - Prepare Test Batch")
    print("=" * 40)
    print(f"Data path: {data_path}")
    print(f"FMA path: {fma_path}")
    print(f"Metadata path: {metadata_path}")
    print(f"Tracks to test: {args.num_tracks}")
    print()
    
    # Check if metadata exists
    tracks_csv = metadata_path / "tracks.csv"
    echonest_csv = metadata_path / "echonest.csv"
    
    if not tracks_csv.exists():
        print(f"ERROR: FMA metadata not found at {tracks_csv}")
        print("Please download FMA Small dataset and extract to:")
        print(f"  {fma_path}")
        print(f"  {metadata_path}")
        sys.exit(1)
    
    if not echonest_csv.exists():
        print(f"ERROR: FMA echonest metadata not found at {echonest_csv}")
        sys.exit(1)
    
    # Read metadata
    print("Reading FMA tracks metadata...")
    try:
        tracks_data = read_fma_tracks_csv(tracks_csv)
        print(f"Loaded {len(tracks_data)} tracks from tracks.csv")
    except Exception as e:
        print(f"ERROR: Could not read tracks.csv: {e}")
        sys.exit(1)
    
    print("Reading FMA echonest metadata (tempo + key/mode when available)...")
    try:
        echonest_data = read_fma_echonest_csv(echonest_csv)
        tempo_count = sum(1 for v in echonest_data.values() if "tempo" in v)
        key_count = sum(1 for v in echonest_data.values() if "key" in v)
        print(f"Loaded tempo data for {tempo_count} tracks from echonest.csv")
        print(f"Loaded key data for {key_count} tracks from echonest.csv")
    except Exception as e:
        print(f"ERROR: Could not read echonest.csv: {e}")
        sys.exit(1)
    
    # Combine data and filter for tracks with BPM
    valid_tracks = []
    for track_id, track_info in tracks_data.items():
        if track_id in echonest_data and "tempo" in echonest_data[track_id]:
            valid_tracks.append({
                "track_id": track_id,
                "bpm": echonest_data[track_id]["tempo"],
                "key": echonest_data[track_id].get("key", ""),
                "genre": track_info.get("genre", ""),
            })
    
    print(f"Found {len(valid_tracks)} tracks with tempo ground truth")
    key_available = sum(1 for t in valid_tracks if t.get("key"))
    if key_available == 0:
        print("Note: No key ground truth found in echonest metadata (key evaluation will be N/A).")
    else:
        print(f"Key ground truth available for {key_available}/{len(valid_tracks)} tracks in this pool.")
    
    # Keep sampling until we find enough tracks with actual files
    test_batch = []
    max_attempts = min(len(valid_tracks) * 2, 10000)  # Don't try forever
    attempts = 0
    seen_track_ids = set()
    
    print(f"Searching for {args.num_tracks} tracks with actual files...")
    
    while len(test_batch) < args.num_tracks and attempts < max_attempts:
        # Randomly select a track we haven't tried yet
        remaining_tracks = [t for t in valid_tracks if t["track_id"] not in seen_track_ids]
        if not remaining_tracks:
            break  # No more tracks to try
        
        track = random.choice(remaining_tracks)
        track_id = track["track_id"]
        seen_track_ids.add(track_id)
        attempts += 1
        
        track_file = find_track_file(fma_path, track_id)
        
        if track_file.exists():
            test_batch.append({
                "track_id": track_id,
                "filename": str(track_file.resolve()),
                "bpm_gt": track["bpm"],
                "key_gt": track.get("key", ""),
                "genre": track["genre"],
            })
            if len(test_batch) % 10 == 0:
                print(f"  Found {len(test_batch)}/{args.num_tracks} tracks...")
    
    if len(test_batch) < args.num_tracks:
        print(f"WARNING: Only found {len(test_batch)} tracks with files out of {args.num_tracks} requested")
    
    print(f"\nPrepared {len(test_batch)} tracks for validation")
    
    # Save test batch with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_batch_csv = results_dir / f"test_batch_{timestamp}.csv"
    with open(test_batch_csv, "w", newline="", encoding="utf-8") as f:
        if test_batch:
            writer = csv.DictWriter(f, fieldnames=test_batch[0].keys())
            writer.writeheader()
            writer.writerows(test_batch)
    
    print(f"Test batch saved to: {test_batch_csv}")
    print()
    print("Next steps:")
    print("1. Build stratum-dsp: cargo build --release")
    print("2. Run validation: python run_validation.py")


if __name__ == "__main__":
    main()
