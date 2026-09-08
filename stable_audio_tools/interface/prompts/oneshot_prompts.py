import random

from typing import List, Tuple, Dict, Optional

from .prompt_common import (
    weighted_choice,
    weighted_sample_unique,
    dedupe_keep_order,
    join_prompt,
    clamp_list,
    enforce_mutex_group,
    choose_shared_fx,
    wetdry_tokens,
    resolve_shared_wet,
    normalize_mode_to_profile,
    note_name_to_midi,
)
from .foundation_prompts import prompt_generator_foundation


ONESHOT_MODEL_SAMPLE_TYPE_ALIASES = {
    "one shot": "oneshot",
    "one-shot": "oneshot",
    "oneshot": "oneshot",
    "shot": "oneshot",
    "single": "oneshot",
    "single shot": "oneshot",

    "loop": "loop",
    "loops": "loop",
    "foundation": "loop",
    "phrase": "loop",
    "melodic": "loop",
    "music": "loop",
}


def normalize_oneshot_model_sample_type(value=None) -> str:
    """
    Default to ONESHOT if the UI has not been updated yet.
    Once the UI passes sample_type="loop", the same model can use the
    existing Foundation prompt builder.
    """
    if value is None:
        return "oneshot"

    key = str(value).strip().lower()
    return ONESHOT_MODEL_SAMPLE_TYPE_ALIASES.get(key, "oneshot")


def prompt_generator_oneshot_model_router(
    *,
    seed=None,
    sample_type="oneshot",
    mode="standard",
    variant="auto",
    allow_timbre_mix=True,
    family_hint=None,

    # shared Wet/Dry UI arg
    wet=True,
    wetdry=None,
    include_wetdry=True,

    # ONESHOT-only UI args
    note=None,
    midi_note=None,
    register_hint=None,
    include_prefix=True,
    include_note=False,
    return_plan=False,

    **kwargs,
) -> str:
    """
    Router for ONESHOT-named models.

    UI-level logic:
      sample_type="loop"
        -> use existing Foundation prompt generator
        -> descriptor includes Wet/Dry after melodic block
        -> UI/inference layer appends BPM / bars / key / major-minor

      sample_type="oneshot"
        -> use ONESHOT prompt generator
        -> note/midi_note derives register
        -> descriptor order:
             One Shot, source 1, optional source 2, tags, Wet/Dry + FX, Register, Note
    """
    st = normalize_oneshot_model_sample_type(sample_type)

    if st == "loop":
        return prompt_generator_foundation(
            seed=seed,
            variant=variant,
            mode=mode,
            allow_timbre_mix=allow_timbre_mix,
            family_hint=family_hint,
            wet=wet,
            wetdry=wetdry,
            include_wetdry=include_wetdry,
            return_plan=return_plan,
            **kwargs,
        )

    return prompt_generator_oneshot_core(
        seed=seed,
        mode=mode,
        family_hint=family_hint,
        note=note,
        midi_note=midi_note,
        register_hint=register_hint,
        wet=wet,
        wetdry=wetdry,
        include_prefix=include_prefix,
        include_wetdry=include_wetdry,
        include_note=include_note,
        return_plan=return_plan,
        **kwargs,
    )


ONESHOT_FAMILIES = [
    "Synth",
    "Keys",
    "Bass",
    "Spectral",
    "Bowed Strings",
    "Wind",
    "Mallet",
    "Brass",
    "FX",
    "Vocal",
    "Guitar",
    "Plucked Strings",
]


ONESHOT_FAMILY_W_STANDARD = [
    28,  # Synth
    20,  # Keys
    14,  # Bass
    10,  # Spectral
    6,   # Bowed Strings
    5,   # Wind
    5,   # Mallet
    4,   # Brass
    4,   # FX
    2,   # Vocal
    1,   # Guitar
    1,   # Plucked Strings
]


ONESHOT_FAMILY_W_EXPERIMENTAL = [
    30,  # Synth
    10,  # Keys
    14,  # Bass
    20,  # Spectral
    4,   # Bowed Strings
    4,   # Wind
    4,   # Mallet
    4,   # Brass
    8,   # FX
    2,   # Vocal
    0,   # Guitar
    0,   # Plucked Strings
]


ONESHOT_SUBFAMILIES: Dict[str, List[Tuple[str, int]]] = {
    "Synth": [
        ("Synth Lead", 160),
        ("FX", 116),
        ("Pluck", 98),
        ("Supersaw", 52),
        ("Synth Bass", 36),
        ("Pad", 11),
        ("Wavetable Synth", 11),
        ("Bell", 10),
        ("Ensemble", 4),
        ("Texture", 4),
        ("Atmosphere", 3),
    ],
    "Keys": [
        ("Grand Piano", 108),
        ("Digital Piano", 61),
        ("Church Organ", 16),
        ("Digital Organ", 16),
        ("Ensemble", 14),
        ("Rhodes Piano", 9),
        ("Harpsichord", 9),
        ("Bell", 8),
        ("Synth Lead", 5),
        ("Pipe Organ", 5),
        ("Clavinet", 4),
        ("Hammond Organ", 4),
        ("Felt Piano", 3),
        ("Celesta", 2),
    ],
    "Bass": [
        ("Reese Bass", 55),
        ("FX", 55),
        ("Sub Bass", 25),
        ("Pluck", 24),
        ("808", 16),
        ("Electric Bass", 15),
        ("Synth Bass", 2),
        ("Wavetable Bass", 2),
        ("Synth Lead", 1),
    ],
    "Spectral": [
        ("Artifact", 89),
        ("Drone", 45),
        ("Atmosphere", 16),
        ("Spectral", 2),
        ("FX", 2),
        ("Impact", 1),
    ],
    "Bowed Strings": [
        ("Cello", 37),
        ("Violin", 35),
        ("Digital Strings", 9),
        ("Ensemble", 7),
        ("Viola", 2),
    ],
    "Wind": [
        ("Woodwinds", 10),
        ("Alto Sax", 10),
        ("Pan Flute", 10),
        ("Flute", 7),
        ("Saxophones", 6),
        ("Saxophone", 4),
        ("Clarinet", 3),
        ("Irish Flute", 3),
        ("Ocarina", 3),
        ("Oboe", 2),
        ("Bassoon", 2),
        ("Baritone Sax", 2),
        ("Tenor Sax", 1),
        ("Ensemble", 1),
        ("Soprano Sax", 1),
    ],
    "Mallet": [
        ("Marimba", 21),
        ("Bell", 15),
        ("Tubular Bells", 10),
        ("Kalimba", 3),
        ("Steel Drums", 3),
        ("Church Bell", 3),
        ("Music Box", 2),
    ],
    "Brass": [
        ("Trumpet", 23),
        ("Tuba", 8),
        ("Trombone", 4),
        ("French Horn", 3),
        ("Tenor Trombone", 2),
        ("Ensemble", 2),
        ("War Horn", 1),
    ],
    "FX": [
        ("Impact", 31),
        ("Swell", 5),
        ("Atmosphere", 1),
    ],
    "Vocal": [
        ("Choir", 7),
        ("FX", 6),
        ("Synth Lead", 4),
        ("Synthetic", 1),
    ],
    "Guitar": [
        ("Electric Guitar", 9),
        ("Acoustic Guitar", 6),
        ("Nylon Guitar", 2),
    ],
    "Plucked Strings": [
        ("Koto", 9),
        ("Celtic Harp", 3),
    ],
}


def oneshot_subfamily_choices(family: Optional[str], *, include_none: bool = False) -> List[str]:
    """Legacy hierarchical choices retained for older callers."""
    family_name = normalize_oneshot_family_hint(family) or str(family or "").strip()
    choices = [str(name) for name, _weight in ONESHOT_SUBFAMILIES.get(family_name, []) if str(name or "").strip()]
    choices = dedupe_keep_order(choices)
    if include_none:
        return ["None"] + choices
    return choices


def _flatten_oneshot_source_vocabulary() -> List[str]:
    """Families and subfamilies exposed as one unrestricted source vocabulary."""
    choices: List[str] = []
    for family, pairs in ONESHOT_SUBFAMILIES.items():
        family_name = str(family or "").strip()
        if family_name:
            choices.append(family_name)
        for subfamily, _weight in pairs:
            subfamily_name = str(subfamily or "").strip()
            if subfamily_name:
                choices.append(subfamily_name)
    return dedupe_keep_order(choices)


ONESHOT_FLAT_SOURCE_CHOICES = _flatten_oneshot_source_vocabulary()


def oneshot_flat_source_choices(*, include_none: bool = False) -> List[str]:
    """Alphabetized display choices without changing random source selection."""
    choices = sorted(ONESHOT_FLAT_SOURCE_CHOICES, key=str.casefold)
    if include_none:
        return ["None"] + choices
    return choices


def oneshot_source_contexts(source: Optional[str]) -> List[Tuple[str, str, int]]:
    """
    Resolve one flat source token to every compatible hierarchy context.

    A family token resolves to that broad family. Repeated subfamily names such
    as Bell, FX, Ensemble, or Synth Lead retain all of their trained contexts
    rather than being forced into whichever dictionary entry appears first.
    """
    token = str(source or "").strip()
    if not token or token.casefold() == "none":
        return []

    token_key = token.casefold()
    for family in ONESHOT_FAMILIES:
        if str(family).casefold() == token_key:
            family_total = sum(max(1, int(weight)) for _name, weight in ONESHOT_SUBFAMILIES.get(family, []))
            return [(str(family), "", max(1, family_total))]

    contexts: List[Tuple[str, str, int]] = []
    for family, pairs in ONESHOT_SUBFAMILIES.items():
        for subfamily, weight in pairs:
            if str(subfamily or "").strip().casefold() == token_key:
                contexts.append((str(family), str(subfamily), max(1, int(weight))))
    return contexts


def oneshot_primary_source_context(source: Optional[str]) -> Tuple[str, str]:
    """Return the strongest compatibility context for legacy metadata fields."""
    contexts = oneshot_source_contexts(source)
    if not contexts:
        token = str(source or "").strip()
        return (token, "") if token else ("", "")
    family, subfamily, _weight = max(contexts, key=lambda item: int(item[2]))
    return family, subfamily


def oneshot_timbre_tag_choices() -> List[str]:
    """Return an alphabetized UI vocabulary without changing tag weights."""
    choices = dedupe_keep_order([str(name) for name, _weight in ONESHOT_TIMBRE_TAGS])
    return sorted(choices, key=str.casefold)


def infer_oneshot_sources_from_descriptor(descriptor: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer up to two leading flat source tokens without enforcing hierarchy."""
    tokens = [part.strip() for part in str(descriptor or "").split(",") if part and part.strip()]
    source_lookup = {choice.casefold(): choice for choice in ONESHOT_FLAT_SOURCE_CHOICES}
    found: List[str] = []
    for token in tokens:
        match = source_lookup.get(token.casefold())
        if not match:
            break
        if match.casefold() not in {item.casefold() for item in found}:
            found.append(match)
        if len(found) >= 2:
            break
    return (found[0] if found else None, found[1] if len(found) > 1 else None)


def infer_oneshot_family_and_subfamily_from_descriptor(descriptor: str) -> Tuple[Optional[str], Optional[str]]:
    """Backward-compatible best hierarchy context for the first flat source."""
    source_1, _source_2 = infer_oneshot_sources_from_descriptor(descriptor)
    family, subfamily = oneshot_primary_source_context(source_1)
    return family or None, subfamily or None


ONESHOT_REGISTERS = [
    "Sub Register",
    "Low Register",
    "Medium Register",
    "High Register",
    "Top Register",
]


ONESHOT_REGISTER_ALIASES = {
    "reg_sub": "Sub Register",
    "sub": "Sub Register",
    "sub register": "Sub Register",

    "reg_low": "Low Register",
    "low": "Low Register",
    "low register": "Low Register",

    "reg_mid": "Medium Register",
    "mid": "Medium Register",
    "medium": "Medium Register",
    "medium register": "Medium Register",

    "reg_high": "High Register",
    "high": "High Register",
    "high register": "High Register",

    "reg_top": "Top Register",
    "top": "Top Register",
    "top register": "Top Register",
}


ONESHOT_REGISTER_BANDS = {
    "REG_SUB":  ("C0", "B1",  "Sub Register"),
    "REG_LOW":  ("C2", "B2",  "Low Register"),
    "REG_MID":  ("C3", "B4",  "Medium Register"),
    "REG_HIGH": ("C5", "B5",  "High Register"),
    "REG_TOP":  ("C6", "G#7", "Top Register"),
}


ONESHOT_REGISTER_KEYS_IN_ORDER = [
    "REG_SUB",
    "REG_LOW",
    "REG_MID",
    "REG_HIGH",
    "REG_TOP",
]


ONESHOT_DEFAULT_REGISTER_W = [4, 28, 40, 24, 4]


ONESHOT_REGISTER_WEIGHTS_BY_FAMILY: Dict[str, List[int]] = {
    "Bass":            [10, 42, 35, 10, 3],
    "Spectral":        [8, 30, 36, 20, 6],
    "FX":              [5, 25, 40, 25, 5],
    "Bowed Strings":   [3, 22, 40, 30, 5],
    "Wind":            [2, 16, 43, 34, 5],
    "Brass":           [3, 28, 42, 23, 4],
    "Keys":            [2, 20, 45, 28, 5],
    "Mallet":          [2, 18, 45, 30, 5],
    "Vocal":           [3, 18, 44, 30, 5],
    "Guitar":          [3, 24, 45, 24, 4],
    "Plucked Strings": [3, 18, 44, 30, 5],
}


ONESHOT_REGISTER_WEIGHTS_BY_SUB: Dict[str, List[int]] = {
    "Sub Bass":      [20, 50, 25, 4, 1],
    "808":           [18, 45, 30, 6, 1],
    "Reese Bass":    [12, 45, 32, 9, 2],
    "Electric Bass": [8, 42, 35, 12, 3],

    "Cello":         [5, 38, 38, 16, 3],
    "Violin":        [1, 10, 42, 40, 7],
    "Viola":         [2, 20, 42, 31, 5],

    "Flute":         [1, 8, 40, 43, 8],
    "Pan Flute":     [1, 10, 42, 40, 7],
    "Alto Sax":      [1, 15, 45, 34, 5],
    "Saxophone":     [1, 15, 45, 34, 5],
    "Saxophones":    [1, 15, 45, 34, 5],
    "Clarinet":      [1, 18, 46, 31, 4],

    "Bell":          [1, 8, 38, 45, 8],
    "Tubular Bells": [1, 8, 36, 46, 9],
    "Music Box":     [1, 8, 38, 45, 8],
}



# ============================================================
# Context-aware one-shot register + note selection
# ============================================================
# Distilled from register_recommendations_by_family.csv and
# register_recommendations_by_pair.csv.
#
# These are softened dataset weights, not hard rules:
# - exact family/subfamily pairs win when available
# - broad instruments keep multiple useful bands
# - unsupported bands get only a tiny escape-hatch weight
# - Spectral/FX/noise-style catchalls stay deliberately neutral
ONESHOT_CONTEXT_REGISTER_WEIGHTS_BY_PAIR: Dict[Tuple[str, str], List[int]] = {('Bass', '808'): [407, 243, 350, 1, 1],
 ('Bass', 'Electric Bass'): [1, 353, 647, 1, 1],
 ('Bass', 'FX'): [1000, 1, 1, 1, 1],
 ('Bass', 'Pluck'): [291, 300, 409, 1, 1],
 ('Bass', 'Reese Bass'): [308, 389, 303, 1, 1],
 ('Bass', 'Synth Bass'): [1, 650, 350, 1, 1],
 ('Bass', 'Synth Lead'): [1, 1000, 1, 1, 1],
 ('Bass', 'Wavetable Bass'): [351, 239, 411, 1, 1],
 ('Bowed Strings', 'Cello'): [1, 320, 680, 1, 1],
 ('Bowed Strings', 'Digital Strings'): [1, 1, 360, 270, 371],
 ('Bowed Strings', 'Viola'): [1, 1, 1, 386, 614],
 ('Bowed Strings', 'Violin'): [1, 1, 1, 372, 628],
 ('Brass', 'French Horn'): [1, 1, 429, 243, 328],
 ('Brass', 'Tenor Trombone'): [1, 1, 656, 344, 1],
 ('Brass', 'Trombone'): [1, 576, 424, 1, 1],
 ('Brass', 'Trumpet'): [1, 1, 406, 293, 301],
 ('Brass', 'Tuba'): [184, 431, 385, 1, 1],
 ('Brass', 'War Horn'): [641, 359, 1, 1, 1],
 ('Guitar', 'Acoustic Guitar'): [1, 181, 311, 176, 333],
 ('Guitar', 'Electric Guitar'): [1, 1, 473, 256, 270],
 ('Keys', 'Bell'): [1, 1, 429, 243, 328],
 ('Keys', 'Celesta'): [1, 1, 656, 344, 1],
 ('Keys', 'Church Organ'): [1, 1, 504, 273, 224],
 ('Keys', 'Clavinet'): [1, 1, 451, 244, 305],
 ('Keys', 'Digital Organ'): [1, 161, 336, 190, 313],
 ('Keys', 'Digital Piano'): [174, 1, 367, 199, 261],
 ('Keys', 'Ensemble'): [1, 1, 429, 243, 328],
 ('Keys', 'Felt Piano'): [1, 1, 429, 243, 328],
 ('Keys', 'Grand Piano'): [254, 178, 255, 139, 174],
 ('Keys', 'Hammond Organ'): [1, 198, 342, 193, 266],
 ('Keys', 'Harpsichord'): [143, 171, 295, 167, 225],
 ('Keys', 'Pipe Organ'): [224, 156, 267, 151, 201],
 ('Keys', 'Rhodes Piano'): [1, 1, 656, 344, 1],
 ('Mallet', 'Bell'): [294, 205, 351, 1, 150],
 ('Mallet', 'Kalimba'): [1, 1, 357, 329, 314],
 ('Mallet', 'Marimba'): [131, 186, 321, 155, 207],
 ('Mallet', 'Steel Drums'): [1, 1, 429, 243, 328],
 ('Mallet', 'Tubular Bells'): [252, 282, 297, 168, 1],
 ('Plucked Strings', 'Celtic Harp'): [1, 1, 429, 243, 328],
 ('Synth', 'Bell'): [1, 1, 1, 327, 673],
 ('Synth', 'Ensemble'): [1, 1, 1, 386, 614],
 ('Synth', 'Pad'): [205, 1, 345, 195, 255],
 ('Synth', 'Pluck'): [1, 1, 298, 254, 448],
 ('Synth', 'Supersaw'): [158, 187, 323, 171, 161],
 ('Synth', 'Synth Bass'): [1, 340, 660, 1, 1],
 ('Synth', 'Synth Lead'): [1, 1, 287, 250, 463],
 ('Synth', 'Texture'): [641, 359, 1, 1, 1],
 ('Synth', 'Wavetable Synth'): [116, 190, 327, 185, 182],
 ('Synth', 'nan'): [1, 257, 488, 255, 1],
 ('Vocal', 'Choir'): [1, 1, 373, 280, 347],
 ('Vocal', 'Synthetic'): [1, 1, 648, 1, 352],
 ('Wind', 'Alto Sax'): [1, 1, 310, 301, 389],
 ('Wind', 'Bassoon'): [1, 1, 1000, 1, 1],
 ('Wind', 'Clarinet'): [1, 1, 292, 320, 387],
 ('Wind', 'Flute'): [1, 1, 429, 243, 328],
 ('Wind', 'Irish Flute'): [1, 1, 357, 329, 314],
 ('Wind', 'Oboe'): [1, 1, 1, 446, 554],
 ('Wind', 'Ocarina'): [1, 1, 429, 243, 328],
 ('Wind', 'Pan Flute'): [1, 1, 473, 256, 270],
 ('Wind', 'Saxophones'): [1, 1, 453, 406, 141],
 ('Wind', 'Soprano Sax'): [1, 1, 1000, 1, 1],
 ('Wind', 'Tenor Sax'): [1, 1, 1, 1000, 1],
 ('Wind', 'Woodwinds'): [1, 1, 462, 225, 312]}

ONESHOT_CONTEXT_REGISTER_WEIGHTS_BY_FAMILY: Dict[str, List[int]] = {'Bass': [345, 337, 319, 1, 1],
 'Bowed Strings': [1, 182, 341, 200, 277],
 'Brass': [1, 1, 652, 1, 348],
 'FX': [8, 22, 40, 22, 8],
 'Guitar': [1, 139, 368, 190, 303],
 'Keys': [183, 155, 292, 165, 204],
 'Mallet': [170, 179, 299, 154, 198],
 'Plucked Strings': [1, 1, 429, 243, 328],
 'Pure Tone': [8, 22, 40, 22, 8],
 'Spectral': [8, 22, 40, 22, 8],
 'Synth': [125, 1, 329, 225, 320],
 'Vocal': [1, 1, 399, 263, 338],
 'White Noise': [8, 22, 40, 22, 8],
 'Wind': [1, 1, 429, 271, 300]}

ONESHOT_CONTEXT_NEUTRAL_REGISTER_W = [8, 22, 40, 22, 8]

ONESHOT_NOTE_NAMES_IN_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_note_name(midi_note: int) -> str:
    midi_note = int(midi_note)
    octave = (midi_note // 12) - 1
    name = ONESHOT_NOTE_NAMES_IN_ORDER[midi_note % 12]
    return f"{name}{octave}"


def _register_key_for_label(register_label: Optional[str]) -> Optional[str]:
    label = normalize_oneshot_register_hint(register_label) or register_label
    for band_key in ONESHOT_REGISTER_KEYS_IN_ORDER:
        _lo_note, _hi_note, band_label = ONESHOT_REGISTER_BANDS[band_key]
        if str(band_label).strip().lower() == str(label or "").strip().lower():
            return band_key
    return None


def pick_oneshot_note_for_register(
    rng: random.Random,
    register_label: str,
) -> str:
    """
    Pick a concrete note inside the already-selected register band.

    This is the missing half of context-aware random:
      1. family/subfamily -> register
      2. register -> note dropdown value

    The note is intentionally uniform inside the chosen band. The musical
    intelligence lives in the register weights, while the exact semitone still
    varies naturally.
    """
    band_key = _register_key_for_label(register_label) or "REG_MID"
    lo_note, hi_note, _label = ONESHOT_REGISTER_BANDS[band_key]
    lo = note_name_to_midi(lo_note)
    hi = note_name_to_midi(hi_note)

    if lo is None or hi is None:
        lo = note_name_to_midi("C3")
        hi = note_name_to_midi("B4")

    return _midi_to_note_name(rng.randint(int(lo), int(hi)))


def context_register_weights_for_prompt(
    *,
    family: str,
    subfamily: str,
) -> List[int]:
    """
    Register priority:
      1. exact family + subfamily pair from dataset heuristics
      2. family-level dataset heuristic
      3. older hand-written subfamily hints
      4. older hand-written family/default hints

    Exact pair wins so duplicate subfamily names can behave differently:
      Synth/Bell -> top-ish
      Mallet/Bell -> broader/sub-ish
      Keys/Bell -> mid/top-ish
    """
    pair_key = (str(family or "").strip(), str(subfamily or "").strip())
    if pair_key in ONESHOT_CONTEXT_REGISTER_WEIGHTS_BY_PAIR:
        return ONESHOT_CONTEXT_REGISTER_WEIGHTS_BY_PAIR[pair_key]

    family_key = str(family or "").strip()
    if family_key in ONESHOT_CONTEXT_REGISTER_WEIGHTS_BY_FAMILY:
        return ONESHOT_CONTEXT_REGISTER_WEIGHTS_BY_FAMILY[family_key]

    if subfamily in ONESHOT_REGISTER_WEIGHTS_BY_SUB:
        return ONESHOT_REGISTER_WEIGHTS_BY_SUB[subfamily]

    return ONESHOT_REGISTER_WEIGHTS_BY_FAMILY.get(family, ONESHOT_DEFAULT_REGISTER_W)


def _normalize_register_weight_vector(weights: List[int]) -> List[float]:
    values = [max(0.0, float(value)) for value in list(weights or [])[:len(ONESHOT_REGISTERS)]]
    while len(values) < len(ONESHOT_REGISTERS):
        values.append(0.0)
    total = sum(values)
    if total <= 0:
        values = [float(value) for value in ONESHOT_CONTEXT_NEUTRAL_REGISTER_W]
        total = sum(values)
    return [value / total for value in values]


def oneshot_register_weights_for_source(source: Optional[str]) -> List[float]:
    """Blend every hierarchy context associated with one flat source token."""
    contexts = oneshot_source_contexts(source)
    if not contexts:
        return _normalize_register_weight_vector(ONESHOT_CONTEXT_NEUTRAL_REGISTER_W)

    accumulated = [0.0] * len(ONESHOT_REGISTERS)
    context_total = float(sum(max(1, int(weight)) for _family, _subfamily, weight in contexts))
    for family, subfamily, context_weight in contexts:
        vector = _normalize_register_weight_vector(
            context_register_weights_for_prompt(family=family, subfamily=subfamily)
        )
        share = max(1.0, float(context_weight)) / context_total
        for index, value in enumerate(vector):
            accumulated[index] += value * share
    return _normalize_register_weight_vector(accumulated)


def combine_oneshot_register_weights(
    instrument_1: Optional[str],
    instrument_2: Optional[str] = None,
    *,
    primary_weight: float = 0.70,
) -> List[float]:
    """
    Combine flat-source register tendencies while keeping Source 1 authoritative.

    The chosen UI note remains the final source of truth. These weights are used
    only when the note is unlocked and the prompt builder is suggesting one.
    """
    source_1 = str(instrument_1 or "").strip()
    source_2 = str(instrument_2 or "").strip()
    if not source_1 and source_2:
        source_1, source_2 = source_2, ""

    first = oneshot_register_weights_for_source(source_1)
    if not source_2 or source_2.casefold() == "none":
        return first

    second = oneshot_register_weights_for_source(source_2)
    first_share = max(0.0, min(1.0, float(primary_weight)))
    second_share = 1.0 - first_share
    return _normalize_register_weight_vector([
        (first[index] * first_share) + (second[index] * second_share)
        for index in range(len(ONESHOT_REGISTERS))
    ])


def pick_oneshot_register_for_sources(
    rng: random.Random,
    *,
    instrument_1: Optional[str],
    instrument_2: Optional[str] = None,
) -> str:
    weights = combine_oneshot_register_weights(instrument_1, instrument_2)
    return weighted_choice(rng, ONESHOT_REGISTERS, weights)


def pick_oneshot_note_for_sources(
    rng: random.Random,
    *,
    instrument_1: Optional[str],
    instrument_2: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (register, concrete note) for an unlocked flat-source prompt."""
    register = pick_oneshot_register_for_sources(
        rng,
        instrument_1=instrument_1,
        instrument_2=instrument_2,
    )
    return register, pick_oneshot_note_for_register(rng, register)


def normalize_oneshot_register_hint(register_hint: Optional[str]) -> Optional[str]:
    if not register_hint:
        return None

    h = str(register_hint).strip().lower()
    return ONESHOT_REGISTER_ALIASES.get(h)


def midi_to_oneshot_register_label(midi_note: int) -> str:
    """
    Maps MIDI pitch to the trained ONESHOT register label.

    Bands:
      C0-B1   -> Sub Register
      C2-B2   -> Low Register
      C3-B4   -> Medium Register
      C5-B5   -> High Register
      C6-G#7  -> Top Register
    """
    midi_note = int(midi_note)

    band_ranges = []

    for band_key in ONESHOT_REGISTER_KEYS_IN_ORDER:
        lo_note, hi_note, label = ONESHOT_REGISTER_BANDS[band_key]
        lo = note_name_to_midi(lo_note)
        hi = note_name_to_midi(hi_note)
        band_ranges.append((lo, hi, label))

    first_lo, _, first_label = band_ranges[0]
    _, last_hi, last_label = band_ranges[-1]

    if midi_note <= first_lo:
        return first_label

    if midi_note >= last_hi:
        return last_label

    for lo, hi, label in band_ranges:
        if lo <= midi_note <= hi:
            return label

    return "Medium Register"


def resolve_oneshot_register_from_ui(
    *,
    note: Optional[str] = None,
    midi_note: Optional[int] = None,
    register_hint: Optional[str] = None,
) -> Optional[str]:
    """
    Priority:
      1. midi_note from UI
      2. note name from UI
      3. explicit register_hint
      4. None, meaning generator should roll weighted random
    """
    if midi_note not in ("", None):
        try:
            return midi_to_oneshot_register_label(int(midi_note))
        except Exception:
            pass

    if note not in ("", None):
        parsed = note_name_to_midi(str(note))
        if parsed is not None:
            return midi_to_oneshot_register_label(parsed)

    return normalize_oneshot_register_hint(register_hint)


ONESHOT_TIMBRE_TAGS = [
    ("Warm", 12),
    ("Bright", 14),
    ("Dark", 9),
    ("Airy", 8),
    ("Rich", 10),
    ("Clean", 8),
    ("Gritty", 9),
    ("Crisp", 8),
    ("Focused", 7),
    ("Metallic", 13),
    ("Smooth", 8),
    ("Cold", 6),
    ("Buzzy", 6),
    ("Round", 6),
    ("Fat", 7),
    ("Punchy", 8),
    ("Thin", 7),
    ("Soft", 8),
    ("Woody", 8),
    ("Hollow", 12),
    ("Nasal", 9),
    ("Biting", 7),
    ("Overdriven", 8),
    ("Subdued", 9),
    ("Breathy", 7),
    ("Glassy", 7),
    ("Sparkly", 8),
    ("Shiny", 6),
    ("Noisy", 7),
    ("Muffled", 9),
    ("Distant", 9),
    ("Wide", 8),
    ("Mono", 4),
    ("Near", 6),
    ("Far", 6),
    ("Spacey", 7),
    ("Ambient", 7),
    ("Intimate", 5),
    ("Small", 4),
    ("Big", 6),
    ("Deep", 8),
    ("Rumble", 6),
    ("Growl", 8),
    ("Neuro", 6),
    ("Wobble", 5),
    ("Wavetable", 6),
    ("Digital", 7),
    ("Analog", 6),
    ("Retro", 8),
    ("Vintage", 7),
    ("Dubstep", 6),
    ("Chiptune", 5),
    ("White Noise", 5),
    ("Formant Vocal", 4),
    ("Synthetic Vox", 4),
    ("Choir", 4),
    ("Pluck", 8),
    ("Sustained", 8),
    ("Short", 6),
    ("Staccato", 5),
    ("Snappy", 6),
    ("Pizzicato", 4),
    ("Spiccato", 4),
    ("Impact", 6),
    ("Hit", 5),
    ("Swell", 5),
    ("Thick", 8),
    ("Present", 7),
    ("Sharp", 5),
    ("Harsh", 5),
    ("Bell", 6),
]


ONESHOT_FAMILY_TAG_BOOST: Dict[str, List[str]] = {
    "Bass": [
        "Deep", "Subdued", "Fat", "Punchy", "Rumble", "Growl",
        "Dark", "Gritty", "Wobble", "Dubstep"
    ],
    "Synth": [
        "Wavetable", "Digital", "Analog", "Bright", "Buzzy", "Wide",
        "Growl", "Neuro", "Dubstep", "Metallic"
    ],
    "Spectral": [
        "Hollow", "Distant", "Muffled", "Metallic", "Glassy",
        "Subdued", "Noisy", "Ambient", "Cold"
    ],
    "FX": [
        "Impact", "Hit", "Swell", "Noisy", "Wide", "Distant",
        "Metallic", "Big", "Bright"
    ],
    "Keys": [
        "Warm", "Clean", "Rich", "Bright", "Smooth", "Soft",
        "Sparkly", "Retro", "Vintage"
    ],
    "Bowed Strings": [
        "Warm", "Smooth", "Nasal", "Hollow", "Sustained",
        "Pizzicato", "Spiccato", "Bright", "Dark"
    ],
    "Wind": [
        "Airy", "Breathy", "Hollow", "Woody", "Thin",
        "Nasal", "Bright", "Smooth"
    ],
    "Mallet": [
        "Woody", "Bright", "Metallic", "Sparkly", "Crisp",
        "Bell", "Soft", "Pluck"
    ],
    "Brass": [
        "Bright", "Nasal", "Biting", "Big", "Present",
        "Metallic", "Harsh"
    ],
    "Vocal": [
        "Choir", "Formant Vocal", "Synthetic Vox", "Breathy",
        "Airy", "Intimate", "Distant"
    ],
    "Guitar": [
        "Woody", "Clean", "Crisp", "Bright", "Gritty", "Pluck"
    ],
    "Plucked Strings": [
        "Pluck", "Woody", "Bright", "Sparkly", "Clean", "Soft"
    ],
}


ONESHOT_REGISTER_TAG_BOOST: Dict[str, List[str]] = {
    "Sub Register": [
        "Deep", "Subdued", "Muffled", "Dark", "Rumble", "Distant"
    ],
    "Low Register": [
        "Warm", "Dark", "Thick", "Round", "Hollow", "Muffled"
    ],
    "Medium Register": [
        "Focused", "Warm", "Rich", "Clean", "Smooth", "Present"
    ],
    "High Register": [
        "Bright", "Airy", "Thin", "Nasal", "Sparkly", "Crisp"
    ],
    "Top Register": [
        "Bright", "Thin", "Airy", "Sparkly", "Sharp", "Metallic"
    ],
}


ONESHOT_TAG_MUTEX_GROUPS = [
    {"Sustained", "Short", "Staccato"},
    {"Pizzicato", "Spiccato", "Sustained"},
]


ONESHOT_DRY_TAG_BLOCKLIST = {
    "Distant",
    "Far",
    "Spacey",
    "Ambient",
}


def normalize_oneshot_family_hint(family_hint: Optional[str]) -> Optional[str]:
    if not family_hint:
        return None

    h = str(family_hint).strip().lower()

    for fam in ONESHOT_FAMILIES:
        if fam.lower() == h:
            return fam

    return None


def pick_oneshot_family(
    rng: random.Random,
    *,
    profile: str,
    family_hint: Optional[str] = None,
) -> str:
    hinted = normalize_oneshot_family_hint(family_hint)

    if hinted:
        return hinted

    if profile in ("mix", "experimental", "mixmatch"):
        return weighted_choice(rng, ONESHOT_FAMILIES, ONESHOT_FAMILY_W_EXPERIMENTAL)

    return weighted_choice(rng, ONESHOT_FAMILIES, ONESHOT_FAMILY_W_STANDARD)


def pick_oneshot_subfamily(rng: random.Random, family: str) -> str:
    pairs = ONESHOT_SUBFAMILIES.get(family, [])

    if not pairs:
        return ""

    items = [x for x, _ in pairs]
    weights = [w for _, w in pairs]

    return weighted_choice(rng, items, weights)


def pick_oneshot_flat_sources(
    rng: random.Random,
    *,
    profile: str,
    family_hint: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Plan user-facing flat sources while retaining hierarchy only as hidden context.

    Standard chooses one specific source from the trained hierarchy. Experimental
    chooses two unrestricted, distinct tokens from the full family/subfamily
    vocabulary to encourage free-form sound combinations.
    """
    normalized_profile = normalize_mode_to_profile(profile)
    hinted_family = normalize_oneshot_family_hint(family_hint)

    if normalized_profile != "mix":
        family = hinted_family or pick_oneshot_family(rng, profile="standard")
        subfamily = pick_oneshot_subfamily(rng, family)
        return subfamily or family, None

    choices = list(ONESHOT_FLAT_SOURCE_CHOICES)
    if len(choices) < 2:
        fallback = hinted_family or (choices[0] if choices else "Synth")
        return fallback, None

    instrument_1 = hinted_family or rng.choice(choices)
    remaining = [choice for choice in choices if choice.casefold() != instrument_1.casefold()]
    instrument_2 = rng.choice(remaining) if remaining else None
    return instrument_1, instrument_2


def pick_oneshot_register(
    rng: random.Random,
    *,
    family: str,
    subfamily: str,
    register_hint: Optional[str] = None,
    note: Optional[str] = None,
    midi_note: Optional[int] = None,
) -> str:
    forced_register = resolve_oneshot_register_from_ui(
        note=note,
        midi_note=midi_note,
        register_hint=register_hint,
    )

    if forced_register:
        return forced_register

    weights = context_register_weights_for_prompt(
        family=family,
        subfamily=subfamily,
    )

    return weighted_choice(rng, ONESHOT_REGISTERS, weights)


def filter_oneshot_tags_for_wetdry(tags: List[str], *, wet: bool) -> List[str]:
    if wet:
        return tags

    return [t for t in tags if t not in ONESHOT_DRY_TAG_BLOCKLIST]


def _oneshot_subfamily_nudges(rng: random.Random, subfamily: str) -> List[str]:
    token = str(subfamily or "").strip()
    if token in {"Reese Bass", "Sub Bass", "808", "Wavetable Bass", "Synth Bass"}:
        return rng.sample(["Deep", "Fat", "Rumble", "Growl"], k=1)
    if token in {"Bell", "Tubular Bells", "Music Box", "Church Bell"}:
        return rng.sample(["Bright", "Sparkly", "Metallic", "Bell"], k=1)
    if token in {"Violin", "Cello", "Viola", "Digital Strings"}:
        return rng.sample(["Sustained", "Smooth", "Warm", "Nasal"], k=1)
    if token in {"Artifact", "Drone", "Atmosphere", "Spectral"}:
        return rng.sample(["Hollow", "Distant", "Muffled", "Glassy"], k=1)
    return []


def _sample_oneshot_tags_from_contexts(
    rng: random.Random,
    *,
    primary_contexts: List[Tuple[str, str, int]],
    secondary_contexts: List[Tuple[str, str, int]],
    register: str,
    profile: str,
    wet: bool,
) -> List[str]:
    profile = (profile or "standard").strip().lower()
    is_experimental = profile in ("mix", "experimental", "mixmatch")

    base_items = [x for x, _ in ONESHOT_TIMBRE_TAGS]
    base_weights = [w for _, w in ONESHOT_TIMBRE_TAGS]
    if is_experimental:
        k_base = rng.choice([5, 6, 7, 8])
        max_tags = 10
    else:
        k_base = rng.choice([4, 5, 6])
        max_tags = 8

    tags: List[str] = weighted_sample_unique(rng, base_items, base_weights, k_base)

    primary_families = dedupe_keep_order([family for family, _subfamily, _weight in primary_contexts])
    secondary_families = dedupe_keep_order([family for family, _subfamily, _weight in secondary_contexts])

    if primary_families:
        family = rng.choice(primary_families)
        boosts = ONESHOT_FAMILY_TAG_BOOST.get(family, [])
        if boosts:
            k_family = rng.choice([1, 2, 2, 3]) if is_experimental else rng.choice([1, 1, 2])
            tags += rng.sample(boosts, k=min(k_family, len(boosts)))

    if secondary_families:
        family = rng.choice(secondary_families)
        boosts = ONESHOT_FAMILY_TAG_BOOST.get(family, [])
        if boosts:
            k_secondary = rng.choice([0, 1, 1, 2]) if is_experimental else rng.choice([0, 1])
            if k_secondary > 0:
                tags += rng.sample(boosts, k=min(k_secondary, len(boosts)))

    register_boosts = ONESHOT_REGISTER_TAG_BOOST.get(register, [])
    if register_boosts:
        k_reg = rng.choice([0, 1, 1, 2]) if is_experimental else rng.choice([0, 1, 1])
        if k_reg > 0:
            tags += rng.sample(register_boosts, k=min(k_reg, len(register_boosts)))

    primary_subs = dedupe_keep_order([subfamily for _family, subfamily, _weight in primary_contexts if subfamily])
    secondary_subs = dedupe_keep_order([subfamily for _family, subfamily, _weight in secondary_contexts if subfamily])
    if primary_subs:
        tags += _oneshot_subfamily_nudges(rng, rng.choice(primary_subs))
    if secondary_subs and rng.random() < (0.75 if is_experimental else 0.45):
        tags += _oneshot_subfamily_nudges(rng, rng.choice(secondary_subs))

    tags = dedupe_keep_order(tags)
    for group in ONESHOT_TAG_MUTEX_GROUPS:
        tags = enforce_mutex_group(rng, tags, group)
    tags = filter_oneshot_tags_for_wetdry(tags, wet=wet)
    return clamp_list(rng, tags, max_tags)


def sample_oneshot_tags(
    rng: random.Random,
    *,
    family: str,
    subfamily: str,
    register: str,
    profile: str,
    wet: bool,
    secondary_family: str = "",
    secondary_subfamily: str = "",
) -> List[str]:
    """Backward-compatible hierarchy-based tag sampler."""
    primary_contexts = [(str(family or ""), str(subfamily or ""), 1)] if family or subfamily else []
    secondary_contexts = (
        [(str(secondary_family or ""), str(secondary_subfamily or ""), 1)]
        if secondary_family or secondary_subfamily
        else []
    )
    return _sample_oneshot_tags_from_contexts(
        rng,
        primary_contexts=primary_contexts,
        secondary_contexts=secondary_contexts,
        register=register,
        profile=profile,
        wet=wet,
    )


def sample_oneshot_tags_for_sources(
    rng: random.Random,
    *,
    instrument_1: Optional[str],
    instrument_2: Optional[str],
    register: str,
    profile: str,
    wet: bool,
) -> List[str]:
    """Flat-source tag sampler used by the modern two-source one-shot builder."""
    return _sample_oneshot_tags_from_contexts(
        rng,
        primary_contexts=oneshot_source_contexts(instrument_1),
        secondary_contexts=oneshot_source_contexts(instrument_2),
        register=register,
        profile=profile,
        wet=wet,
    )


def build_oneshot_descriptor_string(
    *,
    include_prefix: bool,
    family: str,
    subfamily: str,
    tags: List[str],
    wet: bool,
    fx: List[str],
    register: str,
    note=None,
    include_wetdry: bool = True,
    include_note: bool = False,
) -> str:
    """
    One-shot prompt order:

      One Shot
      Flat sound-source block
      Timbre tags
      Wet or Dry + FX block
      Register
      Note
    """
    tokens: List[str] = []

    if include_prefix:
        tokens.append("One Shot")

    tokens.append(family)

    if subfamily:
        tokens.append(subfamily)

    tokens.extend(tags)
    tokens.extend(wetdry_tokens(wet=wet, include_wetdry=include_wetdry))
    tokens.extend(fx)

    if register:
        tokens.append(register)

    if include_note and note not in ("", None):
        tokens.append(str(note).strip())

    return join_prompt(dedupe_keep_order(tokens))


def prompt_generator_oneshot_core(
    *,
    seed=None,
    mode="standard",
    family_hint=None,
    instrument_1=None,
    instrument_2=None,
    register_hint=None,
    note=None,
    midi_note=None,
    wet=True,
    wetdry=None,
    include_prefix=True,
    include_wetdry=True,
    include_note=False,
    return_plan=False,
    **_,
) -> str:
    """
    ONESHOT prompt generator using a flat one- or two-source display model.

    The family/subfamily ontology remains internal. It guides random source
    selection, timbre tags, and unlocked note/register suggestions, but the
    visible prompt contains only the actual source tokens selected by the user
    or random planner.

    The UI-selected note is authoritative. Register inference from that note is
    always applied at conditioning time; source-aware planning is used only when
    a note has not been locked or supplied.
    """
    if seed in ("", None, -1, "-1"):
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed)

    rng = random.Random(seed)
    profile = normalize_mode_to_profile(mode)
    is_wet = resolve_shared_wet(wet=wet, wetdry=wetdry)

    selected_1 = str(instrument_1 or "").strip()
    selected_2 = str(instrument_2 or "").strip()
    if selected_2.casefold() == "none":
        selected_2 = ""

    if not selected_1:
        selected_1, random_second = pick_oneshot_flat_sources(
            rng,
            profile=profile,
            family_hint=family_hint,
        )
        if not selected_2:
            selected_2 = str(random_second or "").strip()

    if selected_2 and selected_2.casefold() == selected_1.casefold():
        selected_2 = ""

    family, subfamily = oneshot_primary_source_context(selected_1)
    secondary_family, secondary_subfamily = oneshot_primary_source_context(selected_2)

    forced_register = resolve_oneshot_register_from_ui(
        note=note,
        midi_note=midi_note,
        register_hint=register_hint,
    )
    register = forced_register or pick_oneshot_register_for_sources(
        rng,
        instrument_1=selected_1,
        instrument_2=selected_2,
    )

    tags = sample_oneshot_tags_for_sources(
        rng,
        instrument_1=selected_1,
        instrument_2=selected_2,
        register=register,
        profile=profile,
        wet=is_wet,
    )
    fx = choose_shared_fx(rng, wet=is_wet)

    planned_note = note
    if planned_note in ("", None) and (include_note or return_plan):
        planned_note = pick_oneshot_note_for_register(rng, register)

    prompt = build_oneshot_descriptor_string(
        include_prefix=include_prefix,
        family=selected_1,
        subfamily=selected_2,
        tags=tags,
        wet=is_wet,
        fx=fx,
        register=register,
        note=planned_note,
        include_wetdry=include_wetdry,
        include_note=include_note,
    )

    if return_plan:
        return {
            "prompt": prompt,
            "instrument_1": selected_1,
            "instrument_2": selected_2 or None,
            "family": family,
            "subfamily": subfamily,
            "secondary_family": secondary_family,
            "secondary_subfamily": secondary_subfamily,
            "tags": list(tags),
            "fx": list(fx),
            "register": register,
            "note": planned_note,
            "wetdry": "Wet" if is_wet else "Dry",
            "profile": profile,
            "seed": seed,
        }

    return prompt

