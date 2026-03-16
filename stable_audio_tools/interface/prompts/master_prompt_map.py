
import random
import re
import hashlib

from typing import List, Tuple, Dict, Optional


FOUNDATION_VARIANT_CHOICES = ["auto", "M1", "T1"]

FOUNDATION_VARIANT_LABELS = {
    "auto": "Auto (by mode)",
    "M1": "M1 – Standard (anchor / coherent)",
    "T1": "T1 – Mix & Match (synth-heavy / richer)",
}

FOUNDATION_VARIANT_HELP = {
    "auto": "Standard mode => M1. Experimental/mix mode => T1.",
    "M1": "Single family/sub; tags stay near anchor; melody rebuilt (family-aware).",
    "T1": "Synth-heavy bias; richer tags; optional 2nd family for timbre mixing; melody family-aware.",
}

MAX_TAGS_STANDARD = 14
MAX_TAGS_MIX = 18 

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


def prompt_generator_foundation(
    *,
    seed=None,
    variant="auto",
    mode="standard",            
    allow_timbre_mix=True,      
    family_hint=None,           
    **_,
):
    # seed may arrive as str from gradio textbox
    if seed in ("", None, -1, "-1"):
        seed = None
    else:
        seed = int(seed)

    if variant not in FOUNDATION_VARIANT_CHOICES:
        variant = "auto"

    return prompt_generator_variants(
        seed=seed,
        mode=mode,
        variant=variant,
        allow_timbre_mix=allow_timbre_mix,
        family_hint=family_hint,
    )


# -------------------------
# Deterministic helpers
# -------------------------
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


# -------------------------
# Core vocab
# -------------------------

FAMILIES = [
    "Synth", "Keys", "Bass", "Bowed Strings", "Mallet",
    "Wind", "Guitar", "Brass", "Vocal", "Plucked Strings"
]

# balanced distribution for anchored mode (M1)
FAMILY_W_STANDARD = [36, 22, 14, 8, 6, 4, 4, 3, 2, 1]  # sums ~100
#timbre experiments 
FAMILY_W_MIX = [55, 14, 18, 3, 2, 2, 2, 1, 1, 0]  # sums ~98; close enough

SUBFAMILIES: Dict[str, List[Tuple[str, int]]] = {
    "Synth": [
        ("Synth Lead", 40), ("Pluck", 15), ("Pad", 12), ("Supersaw", 10),
        ("FM Synth", 8), ("Wavetable Synth", 8), ("Atmosphere", 4), ("Texture", 3),
    ],
    "Keys": [
        ("Grand Piano", 20), ("Digital Piano", 25), ("Rhodes Piano", 20), ("Felt Piano", 8),
        ("Wurlitzer Piano", 8), ("Clavinet", 6), ("Hammond Organ", 6), ("Church Organ", 4), ("Harpsichord", 3),
    ],
    "Bass": [
        ("Wavetable Bass", 25), ("Reese Bass", 20), ("Sub Bass", 18), ("Electric Bass", 12),
        ("Analog Bass", 8), ("FM Bass", 7), ("Picked Bass", 5), ("Digital Bass", 5),
    ],
    "Bowed Strings": [
        ("Violin", 35), ("Cello", 30), ("Viola", 10), ("Fiddle", 10), ("Digital Strings", 15),
    ],
    "Mallet": [
        ("Bell", 30), ("Marimba", 25), ("Vibraphone", 15), ("Glockenspiel", 10),
        ("Kalimba", 10), ("Xylophone", 10),
    ],
    "Wind": [
        ("Flute", 40), ("Pan Flute", 20), ("Piccolo", 8), ("Clarinet", 8),
        ("Oboe", 6), ("Bassoon", 4), ("Ocarina", 6), ("World Winds", 8),
    ],
    "Guitar": [
        ("Electric Guitar", 50), ("Acoustic Guitar", 30), ("Nylon Guitar", 20),
    ],
    "Brass": [
        ("Trumpet", 40), ("Brass", 25), ("French Horn", 10), ("Tuba", 8),
        ("Tenor Trombone", 9), ("Bass Trombone", 8),
    ],
    "Vocal": [
        ("Texture", 45), ("Choir", 25), ("Ensemble", 15), ("Synthetic Choir", 15),
    ],
    "Plucked Strings": [
        ("Harp", 45), ("Concert Harp", 20), ("Celtic Harp", 15), ("Koto", 10), ("Sitar", 10),
    ],
}

# Tag buckets
BAND_TAGS = ["sub", "sub bass", "bass", "low mids", "mids", "upper mids", "highs", "air"]
BAND_W    = [5,     11,        12,     11,        10,     10,           10,       9]

SPATIAL_TAGS = ["wide", "mono", "near", "far", "spacey", "ambient", "distant", "intimate", "small", "big", "deep"]
SPATIAL_W    = [14,     6,      14,     10,    10,       10,        6,         10,         8,       8,    4]

WAVE_TECH_TAGS = [
    "saw", "square", "sine", "triangle", "pulse",
    "analog", "digital", "fm", "supersaw", "reese", 
    "pitch bend", "white noise", "filter"
]
WAVE_TECH_W    = [12,   12,      12,     6,         7,
                  10,     11,       8,    8,         7,      
                  3,          2,        2
]

STYLE_TAGS = ["dubstep", "chiptune", "acid", "303", "retro", "vintage", "laser", "siren", "fx", "formant vocal", "growl"]
STYLE_W    = [10,        10,        6,      8,     16,      14,       8,       6,      8,    10,             12]

RESTRICT_STYLE_TO_SYNTH_BASS = {"303", "acid"}  

TIMBRE_TAGS = [
    "warm","bright","tight","thick","airy","rich","clean","gritty","crisp","focused","metallic","dark","shiny",
    "present","silky","sparkly","smooth","cold","buzzy","round","fat","punchy","thin","soft","woody","hollow",
    "nasal","biting","overdriven","subdued","breathy","glassy",
    "pizzicato","staccato","snappy",

    
    "full","harsh","knock","muddy","steel","veiled","rubbery","rumble","noisy","boomy","crispy","dreamy","heavy","tiny",
    "spiccato"
]


TIMBRE_W = [
    12,    10,     11,     11,     9,     11,    9,      9,      9,      8,       8,      8,      7,
    8,      8,      7,      6,      5,     5,     5,     5,     5,      4,     4,     4,      4,
    3,      3,      3,      2,      2,     2,
    2,      2,      2,

    
    2,      1,      1,      1,      1,     1,      1,      1,      1,      1,      1,      1,      1,      1,
    2
]

FAMILY_TAG_BOOST: Dict[str, List[str]] = {
    "Brass": ["nasal", "present", "biting", "bright", "big"],
    "Wind":  ["hollow", "airy", "breathy", "thin", "woody"],
    "Mallet":["woody", "sparkly", "shiny", "crisp", "bright"],
    "Bass":  ["fat", "punchy", "tight", "gritty", "dark", "sub bass", "bass"],
    "Synth": ["digital", "analog", "fm", "supersaw", "wide", "laser", "saw", "square"],
    "Keys":  ["warm", "clean", "soft", "rich", "smooth"],
    "Guitar":["crisp", "woody", "bright", "clean", "gritty"],
    "Vocal": ["formant vocal", "breathy", "intimate", "airy"],
}


# -------------------------
# Melody builders (family-aware)
# -------------------------
SPEED = ["slow speed", "medium speed", "fast speed"]
RHYTHM = ["off beat", "alternating", "triplets", "strummed", "arp"]
CONTOUR = ["rising", "falling", "bounce", "rolling", "sustained", "choppy", "top"]
DENSITY = ["simple", "repeating", "catchy", "complex", "epic"]

STRUCTURE_GENERIC = ["chord progression", "dance chord progression", "arp", "melody"]
STRUCTURE_BASS = STRUCTURE_GENERIC + ["bassline"]  # ONLY for Bass family

def pick_structure(rng: random.Random, family: str) -> str:
    items = STRUCTURE_BASS if family == "Bass" else STRUCTURE_GENERIC
    return rng.choice(items)

def maybe_add_speed(rng: random.Random, parts: List[str], p: float) -> None:
    if rng.random() < p:
        parts.append(rng.choice(SPEED))

def style_items_for_family(family: str) -> Tuple[List[str], List[int]]:
    # Only allow certain style tokens for Synth/Bass
    if family in ("Synth", "Bass"):
        return STYLE_TAGS, STYLE_W

    items: List[str] = []
    weights: List[int] = []
    for t, w in zip(STYLE_TAGS, STYLE_W):
        if t in RESTRICT_STYLE_TO_SYNTH_BASS:
            continue
        items.append(t)
        weights.append(w)
    return items, weights

def build_melody_coherent(rng: random.Random, family: str, *, speed_p: float) -> str:
    parts: List[str] = []
    maybe_add_speed(rng, parts, p=speed_p)

    # 0–2 rhythmic modifiers
    parts += rng.sample(RHYTHM, k=rng.choice([0, 1, 2]))

    parts.append(pick_structure(rng, family))

    # 0–2 contours
    parts += rng.sample(CONTOUR, k=rng.choice([0, 1, 2]))

    # 0–2 density words
    parts += rng.sample(DENSITY, k=rng.choice([0, 1, 2]))

    return join_prompt(dedupe_keep_order(parts))

def build_melody_density_ladder(rng: random.Random, family: str, *, speed_p: float) -> str:
    parts: List[str] = []
    maybe_add_speed(rng, parts, p=speed_p)

    if rng.random() < 0.7:
        parts.append(rng.choice(["off beat", "alternating", "triplets"]))

    parts.append(pick_structure(rng, family))

    if rng.random() < 0.6:
        parts.append(rng.choice(["rising", "falling", "bounce", "rolling", "sustained"]))

    parts.append(rng.choice(DENSITY))
    return join_prompt(dedupe_keep_order(parts))

def build_melody_weird(rng: random.Random, family: str) -> str:
    # still family-aware so "bassline" won't leak into trumpet
    parts: List[str] = []
    maybe_add_speed(rng, parts, p=0.65)

    parts.append(rng.choice(["off beat", "triplets", "16th note", "quarter note"]))
    parts.append(pick_structure(rng, family))
    parts.append(rng.choice(["sustained", "choppy"]))
    parts += rng.sample(["simple", "complex", "repeating", "catchy"], k=2)
    return join_prompt(dedupe_keep_order(parts))


# -------------------------
# FX sampler (internal wet/dry; no wet/dry token emitted)
# -------------------------
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


# -------------------------
# Tag sampler (profile-aware)
# -------------------------
def sample_tags(
    rng: random.Random,
    family: str,
    *,
    profile: str,  # "standard" | "mix"
) -> List[str]:
    """
    Collapsed tag sampler with family bias.
    Mix profile increases richness for Synth/Bass and slightly increases spatial/style variety.
    """
    profile = (profile or "standard").strip().lower()
    is_mix = (profile in ("mix", "mixmatch", "experimental"))
    is_synthy = family in ("Synth", "Bass")

    # --- Choose counts by profile/family ---
    # Standard: modest, closer to older defaults.
    if not is_mix:
        k_timbre  = rng.choice([3, 4, 5])
        k_spatial = rng.choice([0, 1, 2])
        k_wave    = rng.choice([0, 1, 2])
        k_style   = rng.choice([0, 1])
        k_band    = rng.choice([0, 1])
    else:
        # Mix: richer tags in synth/bass; a bit more spatial/style overall.
        if is_synthy:
            k_timbre  = rng.choice([4, 5, 6, 7])
            k_spatial = rng.choice([1, 2, 3])
            k_wave    = rng.choice([2, 3, 4])
            k_style   = rng.choice([1, 1, 2])     # usually at least 1 style token for synth/bass in mix
            k_band    = rng.choice([0, 1, 1])     # lightly more likely
        else:
            k_timbre  = rng.choice([3, 4, 5, 6])
            k_spatial = rng.choice([0, 1, 2, 2])
            k_wave    = rng.choice([0, 1, 2])     # still conservative for acoustic families
            k_style   = rng.choice([0, 1, 1])
            k_band    = rng.choice([0, 1])

    out: List[str] = []

    # band tags are light (don’t dominate)
    out += weighted_sample_unique(rng, BAND_TAGS, BAND_W, k_band)

    # base timbre
    out += weighted_sample_unique(rng, TIMBRE_TAGS, TIMBRE_W, k_timbre)

    # articulation mutual exclusion
    ARTICULATION_MUTEX = {"pizzicato", "staccato", "spiccato"}
    out = enforce_mutex_group(rng, out, ARTICULATION_MUTEX)

    # family boosts (0–2)
    boosts = FAMILY_TAG_BOOST.get(family, [])
    if boosts:
        kb = rng.choice([0, 1, 2])
        kb = clamp_int(kb, 0, len(boosts))
        if kb > 0:
            out += rng.sample(boosts, k=kb)

    # spatial
    out += weighted_sample_unique(rng, SPATIAL_TAGS, SPATIAL_W, k_spatial)

    # wave-tech
    if is_synthy:
        out += weighted_sample_unique(rng, WAVE_TECH_TAGS, WAVE_TECH_W, k_wave)
    else:
        # non-synth families: occasional tech tag only (keeps realism)
        if k_wave > 0 and rng.random() < (0.25 if not is_mix else 0.35):
            out += weighted_sample_unique(rng, WAVE_TECH_TAGS, WAVE_TECH_W, 1)

    # style (family-aware: prevents 303 leaking into real instruments)
    if k_style > 0:
        p_style = 0.6 if not is_mix else (0.8 if is_synthy else 0.65)
        if rng.random() < p_style:
            s_items, s_w = style_items_for_family(family)
            out += weighted_sample_unique(rng, s_items, s_w, k_style)

    return dedupe_keep_order(out)


# -------------------------
# Family/sub helpers
# -------------------------
def pick_family(rng: random.Random, *, profile: str) -> str:
    profile = (profile or "standard").strip().lower()
    if profile in ("mix", "mixmatch", "experimental"):
        return weighted_choice(rng, FAMILIES, FAMILY_W_MIX)
    return weighted_choice(rng, FAMILIES, FAMILY_W_STANDARD)

def pick_subfamily(rng: random.Random, family: str) -> str:
    subs = SUBFAMILIES.get(family, [])
    if not subs:
        return ""
    items = [s for s, _ in subs]
    w = [w for _, w in subs]
    return weighted_choice(rng, items, w)


# -------------------------
# Descriptor assembly
# -------------------------
def shuffle_blocks(rng: random.Random, blocks: List[List[str]]) -> List[str]:
    # family first most of the time
    family_block = blocks[0]
    other = blocks[1:]

    for b in blocks:
        rng.shuffle(b)

    if rng.random() < 0.75:
        rng.shuffle(other)
        ordered = [family_block] + other
    else:
        rng.shuffle(blocks)
        ordered = blocks

    flat: List[str] = []
    for b in ordered:
        flat.extend(b)

    return dedupe_keep_order(flat)

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


def build_descriptor_string(
    rng: random.Random,
    families_and_subs: List[str],
    tags: List[str],
    fx: List[str],
    melody: str,
) -> str:
    family_block = families_and_subs[:]
    tags_block = tags[:]
    fx_block = fx[:]
    melody_block = [m.strip() for m in melody.split(",") if m.strip()]

    tokens = shuffle_blocks(rng, [family_block, tags_block, fx_block, melody_block])
    return join_prompt(tokens)


# -------------------------
# Anchor + Variant Engine (ONLY M1 + T1)
# -------------------------
VARIANT_TYPES = ["M1", "T1"]

def build_anchor(
    rng: random.Random,
    *,
    profile: str,
    family_hint: Optional[str] = None,
) -> Dict[str, object]:
    fam = family_hint or pick_family(rng, profile=profile)
    sub = pick_subfamily(rng, fam)

    # internal wet/fx — token not emitted
    wet_p = 0.70 if profile == "standard" else 0.80
    wet, fx = choose_wet_and_fx(rng, wet_p=wet_p)

    # melody: speed optional (a bit higher in mix)
    speed_p = 0.55 if profile == "standard" else 0.65
    melody = build_melody_coherent(rng, fam, speed_p=speed_p)

    tags = sample_tags(rng, fam, profile=profile)

    return {
        "family": fam,
        "sub": sub,
        "wet": wet,      # internal only
        "fx": fx,        # emitted (if any)
        "melody": melody,
        "tags": tags,
        "profile": profile,
    }

def normalize_mode_to_profile(mode: str) -> str:
    m = (mode or "standard").strip().lower()
    # Backwards compat: your old code used "experimental"
    if m in ("experimental", "mix", "mixmatch", "mix_and_match", "mix-and-match"):
        return "mix"
    return "standard"

def choose_variant_type(mode: str, variant: str) -> str:
    """
    You said you’ll end up with 2:
      - Standard => M1
      - Mix & Match => T1
    So auto is deterministic by mode.
    """
    if variant in VARIANT_TYPES:
        return variant
    profile = normalize_mode_to_profile(mode)
    return "T1" if profile == "mix" else "M1"

def maybe_add_second_family_for_mix(
    rng: random.Random,
    *,
    fam1: str,
    sub1: str,
) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Optional timbre mixing within T1.
    Returns (fam_tokens, tag_borrow_families, fam2_name_or_none)
    """
    # If fam1 already Synth/Bass, 2nd family less frequent (keep variety but avoid constant dual-instrument)
    p_second = 0.18 if fam1 in ("Synth", "Bass") else 0.28
    if rng.random() >= p_second:
        tokens = [fam1] + ([sub1] if sub1 else [])
        return tokens, [fam1], None

    # Choose second family with a bias toward Synth if first is "real", and toward Bass/Synth if first is Synth.
    candidates = [f for f in FAMILIES if f != fam1]
    if not candidates:
        tokens = [fam1] + ([sub1] if sub1 else [])
        return tokens, [fam1], None

    if fam1 in ("Synth", "Bass"):
        # if already synthy, pick a "real" family sometimes
        weights_map = {f: (6 if f in ("Keys", "Wind", "Guitar", "Brass", "Bowed Strings") else 2) for f in candidates}
        # still allow another synth-related family
        for f in candidates:
            if f in ("Keys", "Wind", "Guitar", "Brass", "Bowed Strings"):
                weights_map[f] += 0
            elif f in ("Vocal", "Mallet", "Plucked Strings"):
                weights_map[f] += 0
            elif f == "Synth":
                weights_map[f] += 2
            elif f == "Bass":
                weights_map[f] += 2
    else:
        # real instrument: favor Synth, then Bass, then Keys
        weights_map = {f: 2 for f in candidates}
        if "Synth" in weights_map: weights_map["Synth"] = 10
        if "Bass" in weights_map:  weights_map["Bass"] = 6
        if "Keys" in weights_map:  weights_map["Keys"] = 5

    cand_list = list(weights_map.keys())
    w_list = list(weights_map.values())
    fam2 = rng.choices(cand_list, weights=w_list, k=1)[0]
    sub2 = pick_subfamily(rng, fam2)

    tokens = [fam1] + ([sub1] if sub1 else []) + [fam2] + ([sub2] if sub2 else [])
    return tokens, [fam1, fam2], fam2

def prompt_generator_variants(
    *,
    seed: Optional[int] = None,
    mode: str = "standard",
    variant: str = "auto",
    allow_timbre_mix: bool = True,
    family_hint: Optional[str] = None,
) -> str:
    """
    Returns a single comma-separated descriptor string.
    No bpm/key/bars. Pure string builder.
    """
    seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
    base_rng = random.Random(seed)

    vt = choose_variant_type(mode=mode, variant=variant)
    profile = "mix" if vt == "T1" else "standard"
    profile = "mix" if normalize_mode_to_profile(mode) == "mix" and vt == "T1" else profile

    # Build anchor with variant-specific family distribution
    anchor = build_anchor(base_rng, profile=profile, family_hint=family_hint)

    # per-variant deterministic RNG so “same seed + vt” is stable
    vrng = random.Random(sha_seed(str(seed), vt, str(anchor["family"])))

    fam_a = str(anchor["family"])
    sub_a = str(anchor["sub"])
    fx_a = list(anchor["fx"])
    tags_a = list(anchor["tags"])

    # M1: keep timbre near anchor; rebuild melody (family-aware); wet/fx stay tied to anchor behavior
    if vt == "M1":
        speed_p = 0.55
        melody = build_melody_coherent(vrng, fam_a, speed_p=speed_p)

        fam_tokens = [fam_a] + ([sub_a] if sub_a else [])
        tags = tags_a[:]  # keep
        tags = clamp_list(vrng, tags, MAX_TAGS_STANDARD)
        fx = fx_a[:]      # keep

        return build_descriptor_string(vrng, fam_tokens, tags, fx, melody)

    # T1: Mix & Match
    # - synth-heavy family selection already baked into anchor profile="mix"
    # - richer tag sampling, especially for Synth/Bass
    # - optional 2nd family/sub (timbre-mix) if allow_timbre_mix
    if vt == "T1":
        # optionally inject a second family/sub
        if allow_timbre_mix:
            fam_tokens, borrow_fams, _fam2 = maybe_add_second_family_for_mix(vrng, fam1=fam_a, sub1=sub_a)
        else:
            fam_tokens, borrow_fams = [fam_a] + ([sub_a] if sub_a else []), [fam_a]

        # Re-sample tags with richer mix profile.
        # If we have two families, merge tag pools (deduped) to encourage blend.
        tags_pool: List[str] = []
        for f in borrow_fams:
            tags_pool += sample_tags(vrng, f, profile="mix")
        tags = dedupe_keep_order(tags_pool)
        tags = clamp_list(vrng, tags, MAX_TAGS_MIX)

        # FX: dry/wet internal only; but emitted tokens come from fx list.
        # In mix, we want wet more often (handled by anchor wet_p=0.80), and wet always yields 1–2 FX tokens.
        # Re-roll FX for more variety instead of anchoring it.
        wet, fx = choose_wet_and_fx(vrng, wet_p=0.80)  # dry => [], wet => 1–2

        # Melody: keep coherent-ish but allow more variety; still family-aware, bassline restricted to Bass
        speed_p = 0.65
        melody = build_melody_coherent(vrng, fam_a, speed_p=speed_p)

        return build_descriptor_string(vrng, fam_tokens, tags, fx, melody)

    # fallback (shouldn't hit)
    fam_tokens = [fam_a] + ([sub_a] if sub_a else [])
    melody = build_melody_coherent(vrng, fam_a, speed_p=0.55)
    return build_descriptor_string(vrng, fam_tokens, tags_a, fx_a, melody)

def prompt_generator_piano():
    piano_types = ["Soft E. Piano", "Medium E. Piano", "Grand Piano"]
    tremolo_effects = ["Low Tremolo", "Medium Tremolo", "High Tremolo", "No Tremolo"]
    non_tremolo_effects = ["No Reverb", "Low Reverb", "Medium Reverb", "High Reverb", "High Spacey Reverb"]

    chord_progressions = ["simple", "complex", "dance plucky", "fast", "jazzy", "low", "simple strummed", "rising strummed", "complex strummed", "jazzy strummed", "slow strummed", "plucky dance",
                          "rising", "falling", "slow", "slow jazzy", "fast jazzy", "smooth", "strummed", "plucky"]
    melodies = [
        "catchy melody", "complex melody", "complex top melody", "catchy top melody", "top melody", "smooth melody", "catchy complex melody",
        "jazzy melody", "smooth catchy melody", "plucky dance melody", "dance melody", "alternating low melody", "alternating top arp melody", "alternating top melody", "top arp melody", "alternating melody", "falling arp melody",
        "rising arp melody", "top catchy melody"
    ]

    # Choose the piano type first to ensure an even split
    piano = random.choice(piano_types)

    # Choose effect based on piano type
    if piano == "Grand Piano":
        effect = random.choice(non_tremolo_effects)
    else:
        effect = random.choice(tremolo_effects + non_tremolo_effects)

    # Decide category for generation
    category_choice = random.choice(["chord progression only", "chord progression with melody", "melody only"])
    
    if category_choice == "chord progression only":
        chord_progression = random.choice(chord_progressions) + " chord progression only,"
        descriptor = f"{piano}, {chord_progression} {effect}"
    elif category_choice == "chord progression with melody":
        chord_progression = random.choice(chord_progressions) + " chord progression,"
        melody = "with " + random.choice(melodies) + ","
        descriptor = f"{piano}, {chord_progression} {melody} {effect}"
    else:
        melody = random.choice(melodies) + " only,"
        descriptor = f"{piano}, {melody} {effect}"

    return descriptor

def prompt_generator_edm():
    # Note: Key signatures are handled in the UI code and should not be included in the prompt descriptors.

    # ---------------------------
    # 1. Define Descriptor Categories
    # ---------------------------

    # Polyphonic Presets
    poly_presets = [
        ['Pluck', 'Sine', 'Bright', 'Clean', 'Bell'],
        ['Pluck', 'Sine', 'Bell'],
        ['Saw', 'Synth', 'Warm'], 
        ['Lead', 'Saw', 'Synth', 'Warm', 'Supersaw']
        # Add more polyphonic presets as needed
    ]

    # Monophonic Presets
    mono_presets = [
        ['Lead', 'Square', 'Synth', 'Buzzy', 'Legato'],    # Preset 1
        ['Lead', 'Square', 'Clean', 'Warm'],               # Preset 2
        ['Bass', 'Punchy', 'Sub']                          # Bass Preset
        # Add more monophonic presets as needed
    ]

    # Corresponding Weights for Mono Presets
    mono_weights = [30, 30, 40]  # Preset 1: 30%, Preset 2: 30%, Bass Preset: 40%

    # Arpeggio Descriptors
    arpeggio_prompts = [
        "medium speed, alternating arp",
        "medium speed, alternating arp, triplets",
        "fast speed, alternating arp",
        "fast speed, alternating arp, triplets",
        "medium speed, alternating arp melody",
        "medium speed, alternating arp melody, triplets",
        "fast speed, alternating arp melody",
        "fast speed, alternating arp melody, triplets",
        "alternating arp",
        "slow simple melody",
        "rising melody",
        "repeating, simple melody",
        "repeating, catchy melody",
        "repeating catchy, bounce melody",
        "repeating, bounce, catchy, melody",
        "catchy, bounce, melody",
        "catchy, triplets, bounce melody",
        "bounce, top, catchy melody",
        "slow rising arp",
        "slow alternating arp",
        "slow falling arp",
        "arp chord progression",
        "arp melody",
        "arp melody, triplets",
        "arp rising melody",
        "arp catchy melody",
        "arp catchy melody, triplets",
        "alternating arp, triplets",
        "alternating arp melody",
        "alternating arp melody, triplets",
        "fast speed, rising arp melody",
        "fast speed, rising arp melody, triplets",
        "medium speed, rising arp melody",
        "medium speed, rising arp melody, triplets",
        "fast speed falling arp melody",
        "fast speed falling arp melody, triplets",
        "medium speed, falling arp melody",
        "catchy, top simple melody",
        "catchy, repeating, off beat melody",
        "medium speed, falling arp melody, triplets",
        "simple, off beat, catchy melody"
        # Add more arpeggio descriptors as needed
    ]

    # Chord Progressions
    chord_progressions = [
        "dance", "complex", "", "catchy dance", "fast speed", "medium speed", "fast speed, strummed",
        "medium speed, strummed", "pluck", "rising dance",
        "simple dance", "slow strummed", "slow speed"
        # Add more chord progressions as needed
    ]

    # Melodies
    melodies = [
        "alternating arp", "alternating catchy melody",
        "alternating arp melody", "catchy melody", 
        "melody", "off beat simple catchy melody", "repeating catchy melody",
        "repeating melody",
        "simple alternating arp melody", "simple catchy melody", "simple falling melody", "simple melody", 
        "simple slow melody", "simple off beat catchy melody", "slow top melody",
        "top catchy melody", "top slow melody", "top repeating catchy melody", "top slow melody", 
        # Add more melodies as needed
    ]

    # New Specific Chord Progression Descriptors
    chord_progression_specific_prompts = [
        "chord progression with catchy melody",
        "chord progression with catchy repeating melody",
        "chord progression with complex melody",
        "chord progression with melody",
        "chord progression with off beat simple catchy melody",
        "chord progression with repeating catchy melody",
        "chord progression with repeating melody",
        "complex chord progression with melody",
        "complex chord progression with top simple melody",
        "dance chord progression",
        "dance chord progression with catchy melody",
        "dance chord progression with complex dance melody",
        "dance chord progression with complex rising arp",
        "dance chord progression with dance catchy melody",
        "dance chord progression with off beat",
        "dance chord progression with off beat catchy melody",
        "dance chord progression with off beat melody",
        "dance chord progression with off beat simple melody",
        "dance chord progression with off beat top melody",
        "dance chord progression with rising arp melody",
        "dance chord progression with simple catchy melody",
        "dance chord progression with simple melody",
        "dance chord progression with simple top catchy melody",
        "dance chord progression with slow beat simple melody",
        "dance chord progression with top",
        "dance chord progression with top catchy melody",
        "dance chord progression with top catchy repeating melody",
        "dance chord progression with top dance melody",
        "dance chord progression with top melody",
        "dance chord progression with top repeating melody",
        "dance chord progression with top simple melody",
        "dance chord progression with top slow melody",
        "dance progression",
        "dance progression with simple catchy melody",
        "dance progression with top melody",
        "dance simple chord progression",
        "dance slow chord progression",
        "medium speed chord progression with top catchy melody",
        "pluck chord progression with top alternating melody",
        "pluck chord progression with top catchy melody",
        "pluck chord progression with top slow melody",
        "rising dance chord progression",
        "rising dance chord progression with off beat repeating melody",
        "rising dance chord progression with top catchy melody",
        "simple dance chord progression",
        "simple dance chord progression with alternating arp melody",
        "simple dance chord progression with simple melody",
        "simple dance chord progression with simple off beat melody",
        "simple dance chord progression with simple top melody",
        "simple dance chord progression with slow top melody",
        "simple dance chord progression with top dance melody",
        "simple dance chord progression with top melody"
    ]

    # ---------------------------
    # 3. Define Prompt Categories and Probabilities
    # ---------------------------

    prompt_categories = [
        "arpeggio_only",
        "chord_progression_only",
        "chord_progression_with_melody",
        "chord_progression_specific",   # New Category Added
        "melody_only"
    ]

    # Probability Percentages for selecting prompt categories
    prompt_probabilities = [20, 15, 25, 20, 20]  # Sum = 100

    # ---------------------------
    # 4. Decide Prompt Type
    # ---------------------------
    prompt_type = random.choices(
        prompt_categories,
        weights=prompt_probabilities,
        k=1
    )[0]

    initial_descriptors = []
    specific_descriptors = []

    # ---------------------------
    # 5. Generate Descriptors Based on Prompt Type
    # ---------------------------

    if prompt_type == "arpeggio_only":
        # Arpeggio Only: Must include exactly three descriptors from either polyphonic or monophonic presets

        # Decide whether to use polyphonic or monophonic presets
        # 50% polyphonic, 50% monophonic
        preset_type = random.choices(
            ['polyphonic', 'monophonic'],
            weights=[50, 50],
            k=1
        )[0]

        if preset_type == 'polyphonic':
            preset = random.choice(poly_presets)
        else:
            # Use weighted selection for mono presets
            preset = random.choices(
                mono_presets,
                weights=mono_weights,
                k=1
            )[0]

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add an arpeggio descriptor
        arpeggio = random.choice(arpeggio_prompts)
        specific_descriptors.append(arpeggio)

    elif prompt_type == "chord_progression_only":
        # Chord Progression Only: Only polyphonic presets

        # Select a polyphonic preset
        preset = random.choice(poly_presets)

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a chord progression descriptor
        chord_prog = random.choice(chord_progressions)
        if chord_prog:
            chord_prog += " chord progression"
        else:
            chord_prog = "chord progression"
        specific_descriptors.append(chord_prog)

    elif prompt_type == "chord_progression_with_melody":
        # Chord Progression with Melodies: Only polyphonic presets

        # Select a polyphonic preset
        preset = random.choice(poly_presets)

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a chord progression descriptor
        chord_prog = random.choice(chord_progressions)
        if chord_prog:
            chord_prog += " chord progression"
        else:
            chord_prog = "chord progression"
        specific_descriptors.append(chord_prog)

        # Add a melody descriptor
        melody = random.choice(melodies)
        specific_descriptors.append(f"with {melody}")

    elif prompt_type == "chord_progression_specific":
        # Chord Progression Specific: Only polyphonic presets and use specific chord progression descriptors

        # Select a polyphonic preset
        preset = random.choice(poly_presets)

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a specific chord progression descriptor
        chord_prog_specific = random.choice(chord_progression_specific_prompts)
        specific_descriptors.append(chord_prog_specific)

    elif prompt_type == "melody_only":
        # Melody Only: Must include exactly three descriptors from either polyphonic or monophonic presets

        # Decide whether to use polyphonic or monophonic presets
        # 60% polyphonic, 40% monophonic
        preset_type = random.choices(
            ['polyphonic', 'monophonic'],
            weights=[60, 40],
            k=1
        )[0]

        if preset_type == 'polyphonic':
            preset = random.choice(poly_presets)
        else:
            # Use weighted selection for mono presets
            preset = random.choices(
                mono_presets,
                weights=mono_weights,
                k=1
            )[0]

        # Select exactly three descriptors from the chosen preset
        selected_descriptors = random.sample(preset, min(3, len(preset)))
        initial_descriptors.extend(selected_descriptors)

        # Add a melody descriptor
        melody = random.choice(melodies) + " only"
        specific_descriptors.append(melody)

    # ---------------------------
    # 7. Handle Effects with Probabilities
    # ---------------------------
    effects = []

    # Decide how many effect categories to apply (0 to 3)
    # Probabilities: 0 effects (10%), 1 effect (60%), 2 effects (25%), 3 effects (5%)
    num_effect_categories = random.choices(
        [0, 1, 2, 3],
        weights=[10, 60, 25, 5],
        k=1
    )[0]

    if num_effect_categories > 0:
        # Define effects categories
        effect_categories = {
            'reverb': ["small reverb", "medium reverb", "high reverb"],
            'filter_sweep': ["with falling high-cut", "with rising low-pass"],
            'gate': ["with half-beat gate", "with quarter-beat gate"]
        }

        # Define weights for effect categories
        effect_category_weights = {
            'reverb': 45,
            'filter_sweep': 45,
            'gate': 10
        }

        # Create a list of categories weighted by their probabilities
        categories = list(effect_categories.keys())
        weights = [effect_category_weights[cat] for cat in categories]

        # To select multiple categories without replacement, perform weighted sampling manually
        selected_categories = []
        available_categories = categories.copy()
        available_weights = weights.copy()

        for _ in range(num_effect_categories):
            if not available_categories:
                break
            chosen = random.choices(
                available_categories,
                weights=available_weights,
                k=1
            )[0]
            selected_categories.append(chosen)
            # Remove the chosen category to avoid duplication
            index = available_categories.index(chosen)
            del available_categories[index]
            del available_weights[index]

        # Now, select one effect from each chosen category
        for category in selected_categories:
            effect = random.choice(effect_categories[category])
            effects.append(effect)

    # ---------------------------
    # 8. Assemble the Descriptor
    # ---------------------------
    # Ensure that initial descriptors are first, followed by specific descriptors
    descriptor = ", ".join(initial_descriptors + specific_descriptors)

    if effects:
        descriptor += ", " + ", ".join(effects)

    return descriptor

def prompt_generator_vocal_textures():
    vocal_types = ["Male Vocal Texture", "Female Vocal Texture", "Ensemble Vocal Texture"]
    vocal = random.choice(vocal_types)
    descriptor = f"{vocal}, chord progression,"
    return descriptor

def default_prompt_generator():

    # Generic descriptors
    descriptors = [
        "arp",
        "chord progression",
        "catchy melody",
        "chord progression with top melody",
        "top melody"        
    ]

    descriptor = random.choice(descriptors)
    return descriptor

def get_prompt_generator(model_name):
    model_name = (model_name or "")

    patterns = [
        (r'foundation.*\.(ckpt|safetensors)$', prompt_generator_foundation),
        (r'piano[s]?.*\.(ckpt|safetensors)$', prompt_generator_piano),
        (r'edm.*elements.*\.(ckpt|safetensors)$', prompt_generator_edm),
        (r'vocal.*textures.*\.(ckpt|safetensors)$', prompt_generator_vocal_textures),
    ]

    for pattern, generator in patterns:
        if re.search(pattern, model_name, re.IGNORECASE):
            return generator

    return default_prompt_generator