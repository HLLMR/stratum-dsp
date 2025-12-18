from __future__ import annotations

import re
from pathlib import Path


def _synchsafe_to_int(b: bytes) -> int:
    return ((b[0] & 0x7F) << 21) | ((b[1] & 0x7F) << 14) | ((b[2] & 0x7F) << 7) | (b[3] & 0x7F)


def _decode_text_frame(payload: bytes) -> str:
    """
    Decode an ID3v2 text frame payload.

    Payload is usually: [encoding][text bytes...]
    """
    if not payload:
        return ""
    enc = payload[0]
    data = payload[1:]
    try:
        if enc == 0:  # ISO-8859-1
            return data.decode("latin1", errors="replace").strip("\x00").strip()
        if enc == 1:  # UTF-16 with BOM
            return data.decode("utf-16", errors="replace").strip("\x00").strip()
        if enc == 2:  # UTF-16BE without BOM
            return data.decode("utf-16-be", errors="replace").strip("\x00").strip()
        if enc == 3:  # UTF-8
            return data.decode("utf-8", errors="replace").strip("\x00").strip()
    except Exception:
        pass
    return data.decode("latin1", errors="replace").strip("\x00").strip()


def read_id3v2_tag_data(mp3_path: Path) -> tuple[int, bytes] | None:
    """
    Read raw ID3v2 tag data from the start of an MP3 file.

    Returns: (ver_major, tag_bytes) or None if no ID3v2 tag found.
    """
    try:
        with open(mp3_path, "rb") as f:
            header = f.read(10)
            if len(header) != 10 or header[0:3] != b"ID3":
                return None
            ver_major = header[3]
            tag_size = _synchsafe_to_int(header[6:10])
            tag_data = f.read(tag_size)
            return ver_major, tag_data
    except Exception:
        return None


def iter_id3v2_frames(tag_data: bytes, ver_major: int):
    """
    Iterate ID3v2.3/2.4 frames and yield (frame_id, payload_bytes).
    """
    pos = 0
    while pos + 10 <= len(tag_data):
        frame_id = tag_data[pos:pos + 4].decode("latin1", errors="ignore")
        if frame_id.strip("\x00") == "":
            break

        size_bytes = tag_data[pos + 4:pos + 8]
        if ver_major == 4:
            frame_size = _synchsafe_to_int(size_bytes)
        else:
            frame_size = int.from_bytes(size_bytes, "big", signed=False)

        pos += 10  # skip header (ID + size + flags)
        if frame_size <= 0 or pos + frame_size > len(tag_data):
            break

        payload = tag_data[pos:pos + frame_size]
        pos += frame_size
        yield frame_id, payload


_RE_FLOAT = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_bpm(value: str) -> float | None:
    """
    Parse BPM from a tag value like '128', '128.0', or '128 BPM'.
    """
    if not value:
        return None
    m = _RE_FLOAT.search(value)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    if v <= 0:
        return None
    return v


def read_id3_text_fields(mp3_path: Path) -> dict:
    """
    Extract a small set of ID3v2 text fields from an MP3.

    Returns dict with:
      - title, artist, album, genre (strings)
      - bpm (float|None)
      - key (string)
      - raw: a small debug dict of seen fields
    """
    out = {
        "title": "",
        "artist": "",
        "album": "",
        "genre": "",
        "bpm": None,
        "key": "",
        "raw": {},
    }

    tag = read_id3v2_tag_data(mp3_path)
    if tag is None:
        return out
    ver_major, tag_data = tag

    for frame_id, payload in iter_id3v2_frames(tag_data, ver_major):
        if frame_id == "TIT2":
            out["title"] = _decode_text_frame(payload)
        elif frame_id == "TPE1":
            out["artist"] = _decode_text_frame(payload)
        elif frame_id == "TALB":
            out["album"] = _decode_text_frame(payload)
        elif frame_id == "TCON":
            out["genre"] = _decode_text_frame(payload)
        elif frame_id == "TBPM":
            txt = _decode_text_frame(payload)
            out["bpm"] = parse_bpm(txt)
        elif frame_id == "TKEY":
            out["key"] = _decode_text_frame(payload)
        elif frame_id == "TXXX":
            # TXXX: [encoding][desc]\x00[ value ]
            if not payload:
                continue
            enc = payload[0]
            rest = payload[1:]
            if enc in (0, 3):  # single-byte encodings
                parts = rest.split(b"\x00", 1)
            else:  # UTF-16 variants
                parts = rest.split(b"\x00\x00", 1)
            if len(parts) != 2:
                continue
            desc = _decode_text_frame(bytes([enc]) + parts[0]).strip()
            val = _decode_text_frame(bytes([enc]) + parts[1]).strip()
            if desc:
                out["raw"].setdefault("TXXX", []).append((desc, val))

            d = desc.strip().lower()
            if d in ("initialkey", "initial key", "key") and val and not out["key"]:
                out["key"] = val
            if d in ("bpm", "tempo", "tbpm") and out["bpm"] is None:
                out["bpm"] = parse_bpm(val)

    return out


