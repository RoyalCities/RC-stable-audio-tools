import random

from typing import List, Tuple, Dict, Optional

from .prompt_common import (
    sha_seed,
    weighted_choice,
    weighted_sample_unique,
    dedupe_keep_order,
    join_prompt,
    clamp_int,
    clamp_list,
    enforce_mutex_group,
    choose_shared_fx,
    resolve_shared_wet,
    wetdry_tokens,
    normalize_mode_to_profile,
)


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


def prompt_generator_foundation(
    *,
    seed=None,
    variant="auto",
    mode="standard",
    allow_timbre_mix=True,
    family_hint=None,
    wet=True,
    wetdry=None,
    include_wetdry=True,
    return_plan=False,
    **_,
):
    """
    Foundation / loop prompt generator.

    Wet/dry behavior is now UI-controlled:
      wet=True / wetdry="Wet" -> emits Wet + FX tokens
      wet=False / wetdry="Dry" -> emits Dry + no FX tokens

    This returns the descriptor body only.
    UI/inference layer should append:
      key signature, bars, BPM
    """
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
        wet=wet,
        wetdry=wetdry,
        include_wetdry=include_wetdry,
        return_plan=return_plan,
    )


FAMILIES = [
    "Synth", "Keys", "Bass", "Bowed Strings", "Mallet",
    "Wind", "Guitar", "Brass", "Vocal", "Plucked Strings"
]


FAMILY_W_STANDARD = [36, 22, 14, 8, 6, 4, 4, 3, 2, 1]  # sums ~100


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


FOUNDATION_NO_SOURCE = "None"
FOUNDATION_EXPERIMENTAL_SECOND_SOURCE_PROBABILITY = 0.80


def _flatten_foundation_source_vocabulary() -> List[str]:
    values: List[str] = []
    for family in FAMILIES:
        values.append(str(family))
        values.extend(
            str(source)
            for source, _weight in SUBFAMILIES.get(family, [])
            if str(source or "").strip()
        )
    return dedupe_keep_order(values)


FOUNDATION_FLAT_SOURCE_CHOICES = _flatten_foundation_source_vocabulary()


def foundation_flat_source_choices(*, include_none: bool = False) -> List[str]:
    """Return alphabetized display choices without changing generation weights."""
    choices = sorted(FOUNDATION_FLAT_SOURCE_CHOICES, key=str.casefold)
    return ([FOUNDATION_NO_SOURCE] + choices) if include_none else choices


def foundation_source_contexts(source: Optional[str]) -> List[Tuple[str, str, int]]:
    """Return every family/subfamily context associated with one flat token.

    Ambiguous tokens such as Texture intentionally retain every matching
    context instead of being forced into the first dictionary match.
    """
    value = str(source or "").strip()
    if not value or value.casefold() == FOUNDATION_NO_SOURCE.casefold():
        return []

    key = value.casefold()
    contexts: List[Tuple[str, str, int]] = []
    for family, family_weight in zip(FAMILIES, FAMILY_W_STANDARD):
        if family.casefold() == key:
            contexts.append((family, "", int(family_weight)))
        for subfamily, weight in SUBFAMILIES.get(family, []):
            if str(subfamily).casefold() == key:
                contexts.append((family, str(subfamily), int(weight)))
    return contexts


def foundation_primary_source_context(source: Optional[str]) -> Tuple[str, str]:
    contexts = foundation_source_contexts(source)
    if not contexts:
        return "", ""
    family, subfamily, _weight = max(contexts, key=lambda item: int(item[2]))
    return family, subfamily


def _source_token_for_context(family: str, subfamily: str) -> str:
    return str(subfamily or family or "").strip()


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


SPEED = ["slow speed", "medium speed", "fast speed"]


RHYTHM = ["off beat", "alternating", "triplets", "strummed", "arp"]


CONTOUR = ["rising", "falling", "bounce", "rolling", "sustained", "choppy", "top"]


DENSITY = ["simple", "repeating", "catchy", "complex", "epic"]


STRUCTURE_GENERIC = ["chord progression", "dance chord progression", "arp", "melody"]


STRUCTURE_BASS = STRUCTURE_GENERIC + ["bassline"]  # ONLY for Bass family


FOUNDATION_TIMBRE_TAG_CHOICES = dedupe_keep_order(
    BAND_TAGS
    + TIMBRE_TAGS
    + SPATIAL_TAGS
    + WAVE_TECH_TAGS
    + STYLE_TAGS
    + [tag for values in FAMILY_TAG_BOOST.values() for tag in values]
)
FOUNDATION_MUSICAL_STRUCTURE_CHOICES = dedupe_keep_order(STRUCTURE_BASS)
FOUNDATION_MUSICAL_TAG_CHOICES = dedupe_keep_order(
    SPEED + RHYTHM + CONTOUR + DENSITY + ["16th note", "quarter note"]
)


def foundation_timbre_tag_choices() -> List[str]:
    """Alphabetized display list; weighted generation pools remain unchanged."""
    return sorted(FOUNDATION_TIMBRE_TAG_CHOICES, key=str.casefold)


def foundation_musical_structure_choices() -> List[str]:
    """Alphabetized display list; generation structure weights remain unchanged."""
    return sorted(FOUNDATION_MUSICAL_STRUCTURE_CHOICES, key=str.casefold)


def foundation_musical_tag_choices() -> List[str]:
    """Alphabetized display list; generation tag weights remain unchanged."""
    return sorted(FOUNDATION_MUSICAL_TAG_CHOICES, key=str.casefold)


def split_foundation_melody_tokens(melody: str) -> Tuple[Optional[str], List[str]]:
    tokens = [part.strip() for part in str(melody or "").split(",") if part.strip()]
    structure_keys = {value.casefold(): value for value in FOUNDATION_MUSICAL_STRUCTURE_CHOICES}

    # RHYTHM also contains "arp", so the first structure-looking token is not
    # always the actual structure. The generator inserts its one structure after
    # rhythmic modifiers; choosing the last matching token resolves that overlap.
    structure_index: Optional[int] = None
    structure: Optional[str] = None
    for index, token in enumerate(tokens):
        canonical = structure_keys.get(token.casefold())
        if canonical is not None:
            structure_index = index
            structure = canonical

    modifiers = [
        token for index, token in enumerate(tokens)
        if index != structure_index
    ]
    return structure, dedupe_keep_order(modifiers)


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


def build_loop_descriptor_string(
    rng: random.Random,
    families_and_subs: List[str],
    tags: List[str],
    melody: str,
    *,
    wet: bool,
    fx: List[str],
    include_wetdry: bool = True,
) -> str:
    """
    Loop prompt order:

      Instrument / family block
      Timbre tags
      Melodic / structure block
      Wet or Dry + FX block

    UI appends key signature, bars, and BPM after this.
    """
    family_block = families_and_subs[:]
    tags_block = tags[:]
    melody_block = [m.strip() for m in melody.split(",") if m.strip()]
    wetdry_block = wetdry_tokens(wet=wet, include_wetdry=include_wetdry)
    fx_block = fx[:]

    tokens: List[str] = []
    tokens.extend(family_block)
    tokens.extend(tags_block)
    tokens.extend(melody_block)
    tokens.extend(wetdry_block)
    tokens.extend(fx_block)

    return join_prompt(dedupe_keep_order(tokens))


VARIANT_TYPES = ["M1", "T1"]


def build_anchor(
    rng: random.Random,
    *,
    profile: str,
    family_hint: Optional[str] = None,
    wet: bool = True,
) -> Dict[str, object]:
    fam = family_hint or pick_family(rng, profile=profile)
    sub = pick_subfamily(rng, fam)

    # UI-controlled wet/dry.
    # Wet => FX tokens.
    # Dry => no FX tokens.
    fx = choose_shared_fx(rng, wet=wet)

    # melody: speed optional, a bit higher in mix
    speed_p = 0.55 if profile == "standard" else 0.65
    melody = build_melody_coherent(rng, fam, speed_p=speed_p)

    tags = sample_tags(rng, fam, profile=profile)

    return {
        "family": fam,
        "sub": sub,
        "wet": wet,
        "fx": fx,
        "melody": melody,
        "tags": tags,
        "profile": profile,
    }


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


def _pick_second_foundation_context_for_mix(
    rng: random.Random,
    *,
    fam1: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Choose one optional secondary context using the existing mix biases.

    Experimental remains dataset-weighted in Source 1, while Source 2 appears
    often enough to make the mode meaningfully hybrid. The secondary-family
    weighting below is unchanged; only the optional-slot gate is raised.
    """
    if rng.random() >= FOUNDATION_EXPERIMENTAL_SECOND_SOURCE_PROBABILITY:
        return None, None

    candidates = [family for family in FAMILIES if family != fam1]
    if not candidates:
        return None, None

    if fam1 in ("Synth", "Bass"):
        weights_map = {
            family: (6 if family in ("Keys", "Wind", "Guitar", "Brass", "Bowed Strings") else 2)
            for family in candidates
        }
        if "Synth" in weights_map:
            weights_map["Synth"] += 2
        if "Bass" in weights_map:
            weights_map["Bass"] += 2
    else:
        weights_map = {family: 2 for family in candidates}
        if "Synth" in weights_map:
            weights_map["Synth"] = 10
        if "Bass" in weights_map:
            weights_map["Bass"] = 6
        if "Keys" in weights_map:
            weights_map["Keys"] = 5

    fam2 = rng.choices(list(weights_map.keys()), weights=list(weights_map.values()), k=1)[0]
    sub2 = pick_subfamily(rng, fam2)
    return fam2, sub2


def maybe_add_second_family_for_mix(
    rng: random.Random,
    *,
    fam1: str,
    sub1: str,
) -> Tuple[List[str], List[str], Optional[str]]:
    """Compatibility helper retaining the original return structure."""
    fam2, sub2 = _pick_second_foundation_context_for_mix(rng, fam1=fam1)
    tokens = [fam1] + ([sub1] if sub1 else [])
    borrow_families = [fam1]
    if fam2:
        tokens.extend([fam2] + ([sub2] if sub2 else []))
        borrow_families.append(fam2)
    return tokens, borrow_families, fam2


def _foundation_prompt_plan(
    *,
    prompt: str,
    seed: int,
    variant: str,
    profile: str,
    wet: bool,
    source_1: str,
    source_2: Optional[str],
    tags: List[str],
    melody: str,
    fx: List[str],
    primary_family: str,
    primary_subfamily: str,
    secondary_family: Optional[str] = None,
    secondary_subfamily: Optional[str] = None,
) -> Dict[str, object]:
    structure, musical_tags = split_foundation_melody_tokens(melody)
    return {
        "prompt": prompt,
        "seed": int(seed),
        "variant": str(variant),
        "profile": str(profile),
        "wetdry": "Wet" if wet else "Dry",
        "source_1": source_1,
        "source_2": source_2 or None,
        "instrument_1": source_1,
        "instrument_2": source_2 or None,
        "tags": list(tags),
        "timbre_tags": list(tags),
        "musical_structure": structure,
        "musical_tags": list(musical_tags),
        "melody": str(melody),
        "fx": list(fx),
        "family": primary_family,
        "subfamily": primary_subfamily or None,
        "primary_family": primary_family,
        "primary_subfamily": primary_subfamily or None,
        "secondary_family": secondary_family or None,
        "secondary_subfamily": secondary_subfamily or None,
    }


def prompt_generator_variants(
    *,
    seed: Optional[int] = None,
    mode: str = "standard",
    variant: str = "auto",
    allow_timbre_mix: bool = True,
    family_hint: Optional[str] = None,
    wet=True,
    wetdry=None,
    include_wetdry: bool = True,
    return_plan: bool = False,
):
    """Build one Foundation loop descriptor or its structured UI plan.

    The user-facing instrument block is intentionally flat: one primary sound
    source plus one optional secondary source. Family/subfamily hierarchy is
    retained only as hidden context for coherent random tags and musical ideas.
    """
    seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
    base_rng = random.Random(seed)
    is_wet = resolve_shared_wet(wet=wet, wetdry=wetdry)

    vt = choose_variant_type(mode=mode, variant=variant)
    profile = "mix" if vt == "T1" else "standard"
    if normalize_mode_to_profile(mode) == "mix" and vt == "T1":
        profile = "mix"

    # Resolve an optional flat source hint back into its hidden context. A broad
    # family hint still chooses one of its weighted subfamilies; a specific
    # subfamily hint stays specific.
    resolved_family_hint = family_hint
    hinted_subfamily = ""
    if family_hint:
        hint_family, hint_subfamily = foundation_primary_source_context(family_hint)
        if hint_family:
            resolved_family_hint = hint_family
            hinted_subfamily = hint_subfamily

    anchor = build_anchor(
        base_rng,
        profile=profile,
        family_hint=resolved_family_hint,
        wet=is_wet,
    )
    if hinted_subfamily:
        anchor["sub"] = hinted_subfamily

    vrng = random.Random(sha_seed(str(seed), vt, str(anchor["family"])))

    fam_a = str(anchor["family"])
    sub_a = str(anchor["sub"] or "")
    source_1 = _source_token_for_context(fam_a, sub_a)
    source_2: Optional[str] = None
    fam_b: Optional[str] = None
    sub_b: Optional[str] = None

    if vt == "M1":
        melody = build_melody_coherent(vrng, fam_a, speed_p=0.55)
        tags = clamp_list(vrng, list(anchor["tags"]), MAX_TAGS_STANDARD)
        fx = list(anchor["fx"])

    elif vt == "T1":
        borrow_families = [fam_a]
        if allow_timbre_mix:
            fam_b, sub_b = _pick_second_foundation_context_for_mix(vrng, fam1=fam_a)
            if fam_b:
                source_2 = _source_token_for_context(fam_b, sub_b or "")
                borrow_families.append(fam_b)

        tags_pool: List[str] = []
        for family in borrow_families:
            tags_pool.extend(sample_tags(vrng, family, profile="mix"))
        tags = clamp_list(vrng, dedupe_keep_order(tags_pool), MAX_TAGS_MIX)
        fx = choose_shared_fx(vrng, wet=is_wet)
        melody = build_melody_coherent(vrng, fam_a, speed_p=0.65)

    else:
        melody = build_melody_coherent(vrng, fam_a, speed_p=0.55)
        tags = list(anchor["tags"])
        fx = list(anchor["fx"])

    source_tokens = [source_1] + ([source_2] if source_2 else [])
    prompt = build_loop_descriptor_string(
        vrng,
        source_tokens,
        tags,
        melody,
        wet=is_wet,
        fx=fx,
        include_wetdry=include_wetdry,
    )

    if return_plan:
        return _foundation_prompt_plan(
            prompt=prompt,
            seed=seed,
            variant=vt,
            profile=profile,
            wet=is_wet,
            source_1=source_1,
            source_2=source_2,
            tags=tags,
            melody=melody,
            fx=fx,
            primary_family=fam_a,
            primary_subfamily=sub_a,
            secondary_family=fam_b,
            secondary_subfamily=sub_b,
        )
    return prompt

