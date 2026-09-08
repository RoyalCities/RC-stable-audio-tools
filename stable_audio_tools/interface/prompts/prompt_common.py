import hashlib
import random
import re

from typing import List, Tuple, Dict, Optional


def clamp_list(rng: random.Random, xs: List[str], max_n: int) -> List[str]:
    """
    Clamp list length to max_n, preserving determinism.
    If too long, randomly samples a subset.
    """
    xs = dedupe_keep_order(xs)
    if max_n <= 0:
        return []
    if len(xs) <= max_n:
        return xs
    return rng.sample(xs, k=max_n)


def sha_seed(*parts: str) -> int:
    h = hashlib.sha256(("|".join(parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def weighted_choice(rng: random.Random, items: List[str], weights: List[int]) -> str:
    return rng.choices(items, weights=weights, k=1)[0]


def weighted_sample_unique(rng: random.Random, items: List[str], weights: List[int], k: int) -> List[str]:
    if k <= 0 or not items:
        return []
    out, seen = [], set()
    tries = 0
    while len(out) < k and tries < 5000:
        tries += 1
        pick = rng.choices(items, weights=weights, k=1)[0]
        if pick in seen:
            continue
        seen.add(pick)
        out.append(pick)
    return out


def dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def join_prompt(tokens: List[str]) -> str:
    return ", ".join([t for t in tokens if isinstance(t, str) and t.strip()])


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


FX_CATS = ["reverb", "delay", "distortion", "phaser", "bitcrush"]


FX_BY_CAT: Dict[str, Tuple[List[str], List[int]]] = {
    "reverb": (
        ["Low Reverb", "Medium Reverb", "High Reverb", "Plate Reverb"],
        [37, 45, 17, 1],
    ),
    "delay": (
        ["Low Delay", "Medium Delay", "Ping Pong Delay", "Stereo Delay", "Cross Delay", "Delay", "High Delay", "Mono Delay"],
        [28, 25, 27, 10, 3, 4, 2, 1],
    ),
    "distortion": (
        ["Low Distortion", "Medium Distortion", "High Distortion", "Distortion"],
        [35, 34, 20, 11],
    ),
    "phaser": (
        ["Phaser", "Low Phaser", "Medium Phaser", "High Phaser"],
        [38, 24, 19, 19],
    ),
    "bitcrush": (
        ["Bitcrush", "High Bitcrush"],
        [95, 5],
    ),
}


def choose_fx_for_wet(
    rng: random.Random,
    *,
    allow_two: bool = True,
) -> List[str]:
    """
    Wet => ALWAYS 1–2 FX tokens.
    Prefer reverb/delay as the primary space FX (but not always reverb).
    """
    # 1 vs 2 FX categories (no kitchen sink)
    k = 1
    if allow_two and rng.random() < 0.25:
        k = 2

    chosen_cats: List[str] = []

    # Primary: reverb/delay mix, not forced reverb
    primary = rng.choices(["reverb", "delay"], weights=[55, 45], k=1)[0]
    chosen_cats.append(primary)

    if k == 2:
        remaining = [c for c in FX_CATS if c not in chosen_cats]
        # Secondary: weighted toward "another space" or light color, but avoids too many reverbs
        # If primary was reverb, let delay be common secondary; if primary was delay, let reverb be common secondary.
        if primary == "reverb":
            weights = [55 if c == "delay" else 20 if c == "distortion" else 15 if c == "phaser" else 10 if c == "bitcrush" else 0 for c in remaining]
        else:
            weights = [55 if c == "reverb" else 20 if c == "distortion" else 15 if c == "phaser" else 10 if c == "bitcrush" else 0 for c in remaining]
        secondary = rng.choices(remaining, weights=weights, k=1)[0]
        chosen_cats.append(secondary)

    fx_tokens: List[str] = []
    for c in chosen_cats:
        items, w = FX_BY_CAT[c]
        fx_tokens.append(weighted_choice(rng, items, w))

    return fx_tokens


SHARED_WET_ALIASES = {
    True: True,
    False: False,

    "wet": True,
    "with fx": True,
    "fx": True,
    "effects": True,
    "effected": True,
    "true": True,
    "yes": True,
    "1": True,

    "dry": False,
    "no fx": False,
    "none": False,
    "clean": False,
    "false": False,
    "no": False,
    "0": False,
}


def normalize_shared_wet(value=True) -> bool:
    """
    Binary wet/dry only.

    True / "wet" / "fx" -> Wet + FX tokens
    False / "dry"       -> Dry + no FX
    """
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = value

    return SHARED_WET_ALIASES.get(key, True)


def resolve_shared_wet(*, wet=True, wetdry=None) -> bool:
    """
    Accept either:
      wet=True/False
      wetdry="Wet"/"Dry"

    wetdry wins if supplied.
    """
    return normalize_shared_wet(wetdry if wetdry is not None else wet)


def choose_shared_fx(
    rng: random.Random,
    *,
    wet: bool,
) -> List[str]:
    """
    Binary wet/dry behavior:
      wet=True  -> always 1-2 FX tokens
      wet=False -> no FX tokens
    """
    if not wet:
        return []

    return choose_fx_for_wet(rng, allow_two=True)


def wetdry_tokens(*, wet: bool, include_wetdry: bool = True) -> List[str]:
    if not include_wetdry:
        return []

    return ["Wet" if wet else "Dry"]


def choose_wet_and_fx(
    rng: random.Random,
    *,
    wet_p: float,
) -> Tuple[bool, List[str]]:
    """
    Internal wet/dry decision.
    - Dry => []
    - Wet => 1–2 FX tokens (always)
    """
    wet = (rng.random() < wet_p)
    if not wet:
        return False, []
    return True, choose_fx_for_wet(rng, allow_two=True)


def enforce_mutex_group(
    rng: random.Random,
    tokens: List[str],
    group: set,
) -> List[str]:
    """
    Ensure at most ONE token from `group` exists in `tokens`.
    If multiple exist, keep exactly one (chosen deterministically via rng) and drop the rest.
    """
    hits = [t for t in tokens if t in group]
    if len(hits) <= 1:
        return tokens

    keep = rng.choice(hits)
    out = []
    kept = False
    for t in tokens:
        if t in group:
            if (not kept) and t == keep:
                out.append(t)
                kept = True
            # else: drop it
        else:
            out.append(t)
    return out


def normalize_mode_to_profile(mode: str) -> str:
    m = (mode or "standard").strip().lower()
    # Backwards compat: your old code used "experimental"
    if m in ("experimental", "mix", "mixmatch", "mix_and_match", "mix-and-match"):
        return "mix"
    return "standard"


NOTE_NAME_TO_PC = {
    "C": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11,
}


def note_name_to_midi(note: str) -> Optional[int]:
    """
    Converts note names like C0, F#4, Bb3 to MIDI.
    Uses standard MIDI convention: C4 = 60.
    """
    if note is None:
        return None

    s = str(note).strip().upper().replace("♯", "#").replace("♭", "B")
    m = re.match(r"^([A-G](?:#|B)?)(-?\d+)$", s)

    if not m:
        return None

    name = m.group(1)
    octave = int(m.group(2))

    if name not in NOTE_NAME_TO_PC:
        return None

    return (octave + 1) * 12 + NOTE_NAME_TO_PC[name]
