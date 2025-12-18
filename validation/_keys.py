from __future__ import annotations


def _numerical_to_key(numerical: str) -> str:
    """
    Convert numerical notation (e.g., '8A', '8B') to a key name (e.g., 'Am', 'C').

    Mapping follows the common DJ numerical wheel convention.
    """
    c = numerical.strip().upper()
    mapping = {
        "1A": "G#m",
        "1B": "B",
        "2A": "D#m",
        "2B": "F#",
        "3A": "A#m",
        "3B": "C#",
        "4A": "Fm",
        "4B": "G#",
        "5A": "Cm",
        "5B": "D#",
        "6A": "Gm",
        "6B": "A#",
        "7A": "Dm",
        "7B": "F",
        "8A": "Am",
        "8B": "C",
        "9A": "Em",
        "9B": "G",
        "10A": "Bm",
        "10B": "D",
        "11A": "F#m",
        "11B": "A",
        "12A": "C#m",
        "12B": "E",
    }
    return mapping.get(c, "")


def normalize_key(key_str: str) -> str:
    """
    Normalize a key string into a canonical form for comparison: e.g., 'C', 'Am', 'F#', 'D#m'.

    Also supports numerical notation (e.g., '8A' -> 'Am') and common shorthand like:
      - 'Cmin', 'Cmaj', 'Ebmaj'
      - Unicode accidentals (♭, ♯)
    """
    if not key_str:
        return ""

    s_raw = key_str.strip()
    if not s_raw:
        return ""

    # Numerical notation detection (1A..12B)
    s_upper = s_raw.upper().replace(" ", "")
    if (
        len(s_upper) in (2, 3)
        and s_upper[-1] in ("A", "B")
        and s_upper[:-1].isdigit()
        and 1 <= int(s_upper[:-1]) <= 12
    ):
        mapped = _numerical_to_key(s_upper)
        if mapped:
            return mapped

    # Normalize unicode accidentals and whitespace
    s = s_raw.replace("♭", "b").replace("♯", "#").strip()
    lower = s.lower()

    # Identify minor/major descriptors
    is_minor = False
    if "minor" in lower or lower.endswith("min") or lower.endswith(" minor"):
        is_minor = True
    if lower.endswith("m") and not lower.endswith("maj") and not lower.endswith(" major"):
        # Common shorthand, e.g., "Am", "C#m"
        is_minor = True

    # Strip descriptors
    for suffix in (" major", " maj", "major", "maj", " minor", " min", "minor", "min"):
        suf = suffix.strip()
        if lower.endswith(suf):
            s = s[: -len(suf)].strip()
            lower = s.lower()
            break

    s = s.replace(" ", "")
    if not s:
        return ""

    # Parse note token
    note = s[0].upper()
    if note < "A" or note > "G":
        return ""

    accidental = ""
    if len(s) >= 2 and s[1] in ("#", "b", "B"):
        accidental = s[1]
        if accidental == "B":
            accidental = "b"

    base = note + accidental
    # Convert flats to sharps
    flat_map = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    base = flat_map.get(base, base)

    return base + ("m" if is_minor else "")


def key_name_to_echonest_key_mode(key_name: str) -> tuple[int, int] | None:
    """
    Convert a normalized key name (from normalize_key) to Echonest key/mode:
      - key: 0..11 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
      - mode: 1=major, 0=minor
    """
    s = normalize_key(key_name)
    if not s:
        return None

    is_minor = s.endswith("m")
    tonic = s[:-1] if is_minor else s

    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    try:
        k = names.index(tonic)
    except ValueError:
        return None

    mode = 0 if is_minor else 1
    return k, mode


