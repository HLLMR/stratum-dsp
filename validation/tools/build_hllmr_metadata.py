#!/usr/bin/env python3
"""
Build GT metadata for the hllmr_small dataset by scraping current ID3 tags.

Goal:
- Capture vendor-provided BPM/Key/etc. as GT **before** re-tagging files with MIK.
- Write metadata in a minimal FMA-compatible pattern so existing tooling can reuse it.

Outputs (under --data-path):
  - hllmr_metadata/tracks.csv
  - hllmr_metadata/echonest.csv

Notes:
- We intentionally keep this dependency-free (no mutagen).
- Track IDs are derived from filename suffixes when present; otherwise a stable hash is used.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from zlib import crc32

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation._id3 import read_id3_text_fields
from validation._keys import normalize_key, key_name_to_echonest_key_mode
from validation._paths import find_repo_root, resolve_data_path


_RE_TRAILING_ID = re.compile(r"(?:-|_)(\d{5,})$")


def derive_track_id(file_path: Path, used: set[int]) -> int:
    """
    Prefer a trailing numeric token in the filename (e.g. *-69424441.mp3).
    Otherwise fall back to a stable hash of the filename.
    """
    stem = file_path.stem
    m = _RE_TRAILING_ID.search(stem)
    if m:
        tid = int(m.group(1))
    else:
        tid = crc32(file_path.name.encode("utf-8")) & 0x7FFFFFFF
        # keep it in a human-friendly range
        if tid < 100000:
            tid += 100000000

    # Resolve collisions deterministically
    base = tid
    bump = 1
    while tid in used:
        tid = base + bump
        bump += 1
    used.add(tid)
    return tid


def write_tracks_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Two header rows (hierarchical style), then the actual header row.
        w.writerow(["", "track", "track"])
        w.writerow(["", "genre_top", "filepath"])
        w.writerow(["track_id", "track.genre_top", "filepath"])
        for r in rows:
            w.writerow([r["track_id"], r.get("genre", ""), r["filepath"]])


def write_echonest_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "echonest", "echonest", "echonest"])
        w.writerow(["", "audio_features", "audio_features", "audio_features"])
        w.writerow(["", "tempo", "key", "mode"])
        w.writerow(["track_id", "", "", ""])
        for r in rows:
            # tempo is required; key/mode optional
            tempo = r.get("tempo")
            if tempo is None:
                continue
            key = r.get("key_echonest", "")
            mode = r.get("mode_echonest", "")
            w.writerow([r["track_id"], f"{tempo:.6f}", key, mode])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build hllmr_small metadata (GT) from ID3 tags.")
    ap.add_argument("--data-path", type=str, default="../validation-data")
    ap.add_argument("--audio-dir", type=str, default="hllmr_small")
    ap.add_argument("--metadata-dir", type=str, default="hllmr_metadata")
    args = ap.parse_args()

    repo_root = find_repo_root()
    data_path = resolve_data_path(args.data_path, repo_root)
    audio_dir = data_path / args.audio_dir
    meta_dir = data_path / args.metadata_dir

    if not audio_dir.exists():
        print(f"ERROR: audio dir not found: {audio_dir}")
        return 1

    mp3s = sorted(audio_dir.glob("*.mp3"))
    if not mp3s:
        print(f"ERROR: no .mp3 files found in: {audio_dir}")
        return 1

    used_ids: set[int] = set()
    track_rows = []
    echo_rows = []

    missing_bpm = 0
    missing_key = 0

    for i, mp3 in enumerate(mp3s, 1):
        fields = read_id3_text_fields(mp3)
        tid = derive_track_id(mp3, used_ids)

        bpm = fields.get("bpm")
        key_raw = fields.get("key", "")
        key_norm = normalize_key(key_raw)
        genre = fields.get("genre", "") or ""

        if bpm is None:
            missing_bpm += 1
        if not key_norm:
            missing_key += 1

        track_rows.append(
            {
                "track_id": tid,
                "filepath": str(mp3.resolve()),
                "genre": genre,
                "title": fields.get("title", ""),
                "artist": fields.get("artist", ""),
            }
        )

        echo = {"track_id": tid, "tempo": bpm}
        km = key_name_to_echonest_key_mode(key_norm) if key_norm else None
        if km is not None:
            echo["key_echonest"] = km[0]
            echo["mode_echonest"] = km[1]
        echo_rows.append(echo)

        if i % 50 == 0 or i == len(mp3s):
            print(f"[{i}/{len(mp3s)}] scanned...")

    write_tracks_csv(meta_dir / "tracks.csv", track_rows)
    write_echonest_csv(meta_dir / "echonest.csv", echo_rows)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme = meta_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "hllmr_metadata (GT capture)",
                "",
                f"Generated: {stamp}",
                f"Audio dir: {audio_dir}",
                "Source: current ID3 tags on files (vendor/library GT snapshot).",
                "",
                "Files:",
                "- tracks.csv   (track_id, track.genre_top, filepath)",
                "- echonest.csv (track_id, tempo, key, mode) in an FMA-compatible header pattern",
                "",
                "Note: Run this BEFORE writing MIK tags, so GT is preserved in CSV form.",
            ]
        ),
        encoding="utf-8",
    )

    print()
    print("DONE")
    print(f"  tracks: {len(track_rows)}")
    print(f"  missing_bpm: {missing_bpm}")
    print(f"  missing_key: {missing_key}")
    print(f"  wrote: {meta_dir / 'tracks.csv'}")
    print(f"  wrote: {meta_dir / 'echonest.csv'}")
    print(f"  wrote: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


