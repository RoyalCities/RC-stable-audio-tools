import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .prompt_common import (
    clamp_list,
    dedupe_keep_order,
    join_prompt,
    note_name_to_midi,
    normalize_mode_to_profile,
    resolve_shared_wet,
    weighted_choice,
    weighted_sample_unique,
    choose_shared_fx,
    wetdry_tokens,
    sha_seed,
)


# ============================================================
# KEYBED constants / note-position mapping
# ============================================================

KEYBED_PREFIX_TOKEN = "Keybed"
KEYBED_TIMBRE_PROFILE_TOKEN = "Timbre Profile"
KEYBED_TARGET_NOTE_TOKEN = "Target Note"
# Legacy alias accepted when cleaning pasted/debug prompts. Do not emit this in new prompts.
KEYBED_TARGET_POSITION_TOKEN = "Target Position"
KEYBED_SEQUENCE_TOKEN = "Sequence"
KEYBED_CHROMATIC_CHUNK_TOKEN = "Chromatic Chunk"
KEYBED_NOTE_SEQUENCE_TOKEN = "Note Sequence"

KEYBED_MIN_NOTE = "C0"
KEYBED_MAX_NOTE = "C8"
KEYBED_MIN_MIDI = note_name_to_midi(KEYBED_MIN_NOTE)
KEYBED_MAX_MIDI = note_name_to_midi(KEYBED_MAX_NOTE)
KEYBED_POSITION_COUNT = 97

KEYBED_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEYBED_NOTE_CHOICES = [
    f"{name}{octave}"
    for octave in range(0, 9)
    for name in KEYBED_NOTE_NAMES
    if not (octave == 8 and name != "C")
]

KEYBED_PREVIEW_SHAPES: Dict[str, List[int]] = {
    "Minor Triad": [0, 3, 7],
    "Major Triad": [0, 4, 7],
    "Octave": [0, 12],
    "Chromatic 5": [0, 1, 2, 3, 4],
}

KEYBED_DEFAULT_PREVIEW_ROOT_BY_FAMILY: Dict[str, str] = {
    "Bass": "C2",
    "Keys": "C4",
    "Synth": "C4",
    "Bowed Strings": "C4",
    "Mallet": "C5",
    "Wind": "C4",
    "Brass": "C3",
    "Guitar": "C3",
    "Vocal": "C4",
    "Pure Tone": "C4",
    "White Noise": "C4",
    "Plucked Strings": "C4",
}

# Context-aware preview-root picking for the Keybed tab.
# The UI preview range starts at C2, so true sub-register instruments are
# intentionally mapped into the lowest available preview octave rather than C0/C1.
KEYBED_PREVIEW_ROOT_KEYS_IN_ORDER = ["ROOT_LOW", "ROOT_MID", "ROOT_HIGH", "ROOT_TOP"]

KEYBED_PREVIEW_ROOT_BANDS: Dict[str, Tuple[str, str]] = {
    "ROOT_LOW": ("C2", "B2"),
    "ROOT_MID": ("C3", "B4"),
    "ROOT_HIGH": ("C5", "B5"),
    "ROOT_TOP": ("C6", "G6"),
}

# Weights are deliberately soft, like ONESHOT register planning. Exact
# family/subfamily pairs win, family-level weights are the fallback.
KEYBED_CONTEXT_PREVIEW_ROOT_WEIGHTS_BY_PAIR: Dict[Tuple[str, str], List[int]] = {
    ("Bass", "808"): [100, 0, 0, 0],
    ("Bass", "Sub Bass"): [100, 0, 0, 0],
    ("Bass", "Electric Bass"): [90, 10, 0, 0],
    ("Bass", "Fingered Bass"): [90, 10, 0, 0],
    ("Bass", "Pluck"): [90, 10, 0, 0],
    ("Bass", "Reese Bass"): [95, 5, 0, 0],
    ("Bass", "Synth Bass"): [95, 5, 0, 0],
    ("Bass", "Wavetable Bass"): [92, 8, 0, 0],

    ("Synth", "Synth Bass"): [85, 15, 0, 0],
    ("Synth", "FM Synth"): [8, 62, 25, 5],
    ("Synth", "Wavetable Synth"): [8, 58, 27, 7],
    ("Synth", "Synth Lead"): [3, 55, 34, 8],
    ("Synth", "Supersaw"): [6, 54, 32, 8],
    ("Synth", "Pluck"): [5, 50, 35, 10],
    ("Synth", "Pad"): [8, 58, 27, 7],
    ("Synth", "Bell"): [1, 22, 52, 25],
    ("Synth", "Atmosphere"): [8, 52, 30, 10],
    ("Synth", "Texture"): [18, 54, 22, 6],

    ("Keys", "Grand Piano"): [12, 58, 24, 6],
    ("Keys", "Digital Piano"): [10, 60, 24, 6],
    ("Keys", "Rhodes Piano"): [8, 64, 23, 5],
    ("Keys", "Digital Organ"): [12, 58, 23, 7],
    ("Keys", "Church Organ"): [16, 55, 22, 7],
    ("Keys", "Pipe Organ"): [20, 52, 22, 6],
    ("Keys", "Bell"): [2, 38, 43, 17],
    ("Keys", "Celesta"): [1, 32, 48, 19],
    ("Keys", "Harpsichord"): [7, 54, 31, 8],

    ("Bowed Strings", "Cello"): [35, 55, 9, 1],
    ("Bowed Strings", "Viola"): [5, 58, 32, 5],
    ("Bowed Strings", "Violin"): [1, 48, 42, 9],
    ("Bowed Strings", "Fiddle"): [1, 45, 43, 11],
    ("Bowed Strings", "Digital Strings"): [3, 54, 34, 9],
    ("Bowed Strings", "Ensemble"): [5, 58, 31, 6],

    ("Mallet", "Marimba"): [8, 48, 35, 9],
    ("Mallet", "Bell"): [2, 33, 45, 20],
    ("Mallet", "Kalimba"): [3, 45, 39, 13],
    ("Mallet", "Steel Drums"): [3, 42, 42, 13],
    ("Mallet", "Tubular Bells"): [3, 30, 47, 20],
    ("Mallet", "Xylophone"): [2, 38, 45, 15],
    ("Mallet", "Vibraphone"): [3, 43, 41, 13],
    ("Mallet", "Glockenspiel"): [1, 25, 48, 26],
    ("Mallet", "Music Box"): [1, 25, 48, 26],
    ("Mallet", "Toy Bell"): [1, 25, 48, 26],

    ("Wind", "Bassoon"): [28, 62, 9, 1],
    ("Wind", "Clarinet"): [4, 58, 32, 6],
    ("Wind", "Flute"): [1, 45, 43, 11],
    ("Wind", "Irish Flute"): [1, 43, 44, 12],
    ("Wind", "Pan Flute"): [1, 45, 43, 11],
    ("Wind", "Ocarina"): [2, 48, 39, 11],
    ("Wind", "Saxophones"): [4, 58, 32, 6],
    ("Wind", "Sax"): [4, 58, 32, 6],
    ("Wind", "Alto Sax"): [3, 55, 35, 7],
    ("Wind", "Tenor Sax"): [8, 62, 26, 4],
    ("Wind", "Soprano Sax"): [1, 38, 47, 14],
    ("Wind", "Oboe"): [1, 42, 45, 12],

    ("Brass", "Tuba"): [60, 38, 2, 1],
    ("Brass", "War Horn"): [58, 39, 3, 1],
    ("Brass", "Bass Trombone"): [48, 48, 3, 1],
    ("Brass", "Trombone"): [35, 58, 6, 1],
    ("Brass", "Tenor Trombone"): [28, 62, 9, 1],
    ("Brass", "French Horn"): [10, 64, 22, 4],
    ("Brass", "Trumpet"): [2, 54, 34, 10],

    ("Guitar", "Acoustic Guitar"): [10, 58, 26, 6],
    ("Guitar", "Electric Guitar"): [8, 58, 27, 7],
    ("Guitar", "Koto"): [4, 48, 36, 12],
    ("Guitar", "Sitar"): [4, 48, 36, 12],
    ("Guitar", "Ukulele"): [1, 40, 45, 14],
    ("Guitar", "Lute"): [5, 54, 32, 9],
    ("Guitar", "Banjo"): [3, 48, 38, 11],

    ("Vocal", "Choir"): [2, 48, 38, 12],
    ("Vocal", "Synthetic"): [2, 50, 36, 12],
    ("Vocal", "Pad"): [4, 54, 34, 8],

    ("Plucked Strings", "Concert Harp"): [2, 42, 43, 13],
    ("Plucked Strings", "Celtic Harp"): [2, 42, 43, 13],
}

KEYBED_CONTEXT_PREVIEW_ROOT_WEIGHTS_BY_FAMILY: Dict[str, List[int]] = {
    "Bass": [90, 10, 0, 0],
    "Synth": [7, 55, 30, 8],
    "Keys": [12, 58, 24, 6],
    "Bowed Strings": [8, 56, 29, 7],
    "Mallet": [4, 42, 40, 14],
    "Wind": [3, 50, 38, 9],
    "Brass": [22, 58, 17, 3],
    "Guitar": [7, 54, 31, 8],
    "Vocal": [3, 50, 36, 11],
    "Pure Tone": [6, 64, 24, 6],
    "White Noise": [6, 64, 24, 6],
    "Plucked Strings": [3, 43, 41, 13],
}

KEYBED_CONTEXT_NEUTRAL_PREVIEW_ROOT_W = [6, 64, 24, 6]

KEYBED_POS_RE = re.compile(r"^keybed_pos_\d{3}$", re.IGNORECASE)
NOTE_TOKEN_RE = re.compile(r"^[A-G](?:#|b)?-?\d+$", re.IGNORECASE)

# Legacy prompt-style labels kept only so older callers do not break.
# New user-facing code should not expose these; canonical KEYBED grammar emits
# Target Note for singles and Note Sequence for chromatic chunks.
KEYBED_PROMPT_STYLE_TARGET_ONLY = "Target Position Only"
KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER = "Target + Register + Note"
KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE = "Register + Note + Target"
KEYBED_PROMPT_STYLE_CHOICES = [
    KEYBED_PROMPT_STYLE_TARGET_ONLY,
    KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER,
    KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE,
]
KEYBED_PROMPT_STYLE_SLUGS = {
    KEYBED_PROMPT_STYLE_TARGET_ONLY: "target_only",
    KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER: "target_then_register_note",
    KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE: "register_note_then_target",
}

KEYBED_REGISTER_LABELS = [
    "Sub Register",
    "Low Register",
    "Medium Register",
    "High Register",
    "Top Register",
]
KEYBED_REGISTER_BANDS = [
    ("C0", "B1", "Sub Register"),
    ("C2", "B2", "Low Register"),
    ("C3", "B4", "Medium Register"),
    ("C5", "B5", "High Register"),
    ("C6", "C8", "Top Register"),
]


# ============================================================
# KEYBED ontology
# Curated for useful random prompts. It is based on the analyzed
# KEYBED data, but intentionally downweights one-off/weird pairs.
# ============================================================

KEYBED_SUBFAMILIES: Dict[str, List[Tuple[str, int]]] = {
    "Synth": [
        ("Synth Lead", 70),
        ("Supersaw", 21),
        ("Pluck", 20),
        ("Pad", 17),
        ("Atmosphere", 15),
        ("Synth Bass", 11),
        ("Ensemble", 6),
        ("Bell", 4),
        ("Texture", 4),
        ("FM Synth", 3),
        ("Wavetable Synth", 2),
        ("Flute", 1),
        ("Digital Organ", 1),
        ("Digital Strings", 1),
    ],
    "Keys": [
        ("Grand Piano", 21),
        ("Digital Piano", 17),
        ("Digital Organ", 11),
        ("Rhodes Piano", 10),
        ("Bell", 6),
        ("Ensemble", 6),
        ("Synth Lead", 4),
        ("Pad", 3),
        ("Pipe Organ", 2),
        ("Church Organ", 2),
        ("Pluck", 2),
        ("Harpsichord", 2),
        ("Hammond Organ", 2),
        ("Digital Strings", 2),
        ("Clavinet", 1),
        ("Celesta", 1),
        ("Felt Piano", 1),
        ("Tubular Bells", 1),
        ("Harp", 1),
        ("Wurlitzer Piano", 1),
        ("Tack Piano", 1),
        ("Harmonium", 1),
    ],
    "Bass": [
        ("Pluck", 20),
        ("Reese Bass", 9),
        ("808", 5),
        ("Electric Bass", 5),
        ("Wavetable Bass", 3),
        ("Synth Bass", 2),
        ("Fingered Bass", 1),
        ("FX", 1),
    ],
    "Bowed Strings": [
        ("Ensemble", 18),
        ("Violin", 17),
        ("Cello", 15),
        ("Digital Strings", 13),
        ("Fiddle", 2),
        ("Viola", 1),
        ("Pad", 1),
    ],
    "Mallet": [
        ("Marimba", 9),
        ("Bell", 4),
        ("Kalimba", 2),
        ("Ensemble", 2),
        ("Steel Drums", 2),
        ("Tubular Bells", 2),
        ("Xylophone", 1),
        ("Vibraphone", 1),
        ("Glockenspiel", 1),
        ("Music Box", 1),
        ("Toy Bell", 1),
    ],
    "Wind": [
        ("Woodwinds", 11),
        ("Flute", 6),
        ("Saxophones", 5),
        ("World Winds", 5),
        ("Pan Flute", 5),
        ("Clarinet", 3),
        ("Sax", 3),
        ("Alto Sax", 2),
        ("Oboe", 1),
        ("Bassoon", 1),
        ("Tenor Sax", 1),
        ("Irish Flute", 1),
        ("Soprano Sax", 1),
        ("Ensemble", 1),
        ("Ocarina", 1),
    ],
    "Brass": [
        ("Trumpet", 7),
        ("French Horn", 3),
        ("Tuba", 2),
        ("War Horn", 2),
        ("Tenor Trombone", 1),
        ("Bass Trombone", 1),
        ("Trombone", 1),
    ],
    "Guitar": [
        ("Electric Guitar", 4),
        ("Acoustic Guitar", 3),
        ("Koto", 3),
        ("Sitar", 2),
        ("Ukulele", 1),
        ("Lute", 1),
        ("Banjo", 1),
    ],
    "Vocal": [
        ("Synthetic", 7),
        ("Choir", 4),
        ("Pad", 1),
        ("Atmosphere", 1),
    ],
    "Pure Tone": [
        ("Sine", 1),
        ("Saw", 1),
        ("Triangle", 1),
        ("Pulse", 1),
    ],
    "White Noise": [
        ("", 1),
    ],
    "Plucked Strings": [
        ("Concert Harp", 1),
        ("Celtic Harp", 1),
    ],
}

KEYBED_FAMILIES_SIMPLE = [
    "Synth", "Keys", "Bowed Strings", "Bass", "Mallet", "Wind", "Brass", "Guitar", "Vocal", "Pure Tone"
]
KEYBED_FAMILY_W_SIMPLE = [30, 24, 13, 12, 7, 6, 3, 2, 2, 1]

KEYBED_FAMILIES_EXPERIMENTAL = [
    "Synth", "Keys", "Bass", "Bowed Strings", "Mallet", "Wind", "Brass", "Guitar",
    "Vocal", "Pure Tone", "White Noise", "Plucked Strings"
]
KEYBED_FAMILY_W_EXPERIMENTAL = [32, 16, 14, 9, 7, 7, 4, 4, 3, 2, 1, 1]


def _flatten_keybed_instrument_vocabulary() -> List[str]:
    """Families and subfamilies as one unrestricted experimental vocabulary."""
    choices: List[str] = []
    for family, pairs in KEYBED_SUBFAMILIES.items():
        family_name = str(family or "").strip()
        if family_name and family_name != "White Noise":
            choices.append(family_name)
        for subfamily, _weight in pairs:
            subfamily_name = str(subfamily or "").strip()
            if subfamily_name and subfamily_name != "White Noise":
                choices.append(subfamily_name)
    return dedupe_keep_order(choices)


KEYBED_EXPERIMENTAL_INSTRUMENTS = _flatten_keybed_instrument_vocabulary()


def keybed_flat_instrument_choices() -> List[str]:
    """Alphabetized UI choices without changing random instrument selection."""
    return sorted(KEYBED_EXPERIMENTAL_INSTRUMENTS, key=str.casefold)


KEYBED_INSTRUMENT_MODE_SINGLE = "single"
KEYBED_INSTRUMENT_MODE_HYBRID = "hybrid"
KEYBED_INSTRUMENT_MODE_CHOICES = (
    KEYBED_INSTRUMENT_MODE_SINGLE,
    KEYBED_INSTRUMENT_MODE_HYBRID,
)


def normalize_keybed_instrument_mode(value=None) -> str:
    """Normalize the UI's Single/Hybrid choice without tying it to prompt profile."""
    key = str(value or KEYBED_INSTRUMENT_MODE_HYBRID).strip().lower()
    if key in {"single", "one", "solo", "1"}:
        return KEYBED_INSTRUMENT_MODE_SINGLE
    return KEYBED_INSTRUMENT_MODE_HYBRID


def keybed_instrument_context(instrument: str) -> Tuple[str, str]:
    """Resolve a flat instrument token back to its best family/subfamily context."""
    token = str(instrument or "").strip()
    if not token:
        return "", ""

    token_key = token.casefold()
    for family in KEYBED_SUBFAMILIES.keys():
        if str(family).casefold() == token_key:
            return str(family), ""

    candidates: List[Tuple[str, int]] = []
    for family, pairs in KEYBED_SUBFAMILIES.items():
        for subfamily, weight in pairs:
            if str(subfamily or "").strip().casefold() == token_key:
                candidates.append((str(family), int(weight)))

    if not candidates:
        return "", ""

    family = max(candidates, key=lambda item: item[1])[0]
    return family, token


def pick_keybed_experimental_instruments(
    rng: random.Random,
    *,
    family_hint: Optional[str] = None,
) -> Tuple[str, str]:
    """Pick two distinct free-form instrument tokens for Experimental mode."""
    if len(KEYBED_EXPERIMENTAL_INSTRUMENTS) < 2:
        raise ValueError("Experimental keybed vocabulary requires at least two instruments.")

    hinted = normalize_keybed_family_hint(family_hint)
    instrument_1 = hinted or rng.choice(KEYBED_EXPERIMENTAL_INSTRUMENTS)
    remaining = [
        item for item in KEYBED_EXPERIMENTAL_INSTRUMENTS
        if str(item).casefold() != str(instrument_1).casefold()
    ]
    instrument_2 = rng.choice(remaining)
    return instrument_1, instrument_2

KEYBED_TIMBRE_TAGS = [
    "Warm", "Bright", "Dark", "Airy", "Rich", "Clean", "Gritty", "Crisp", "Focused",
    "Metallic", "Smooth", "Cold", "Buzzy", "Round", "Fat", "Punchy", "Thin", "Soft",
    "Woody", "Hollow", "Nasal", "Biting", "Overdriven", "Subdued", "Breathy", "Glassy",
    "Sparkly", "Shiny", "Noisy", "Muffled", "Distant", "Wide", "Mono", "Near", "Far",
    "Spacey", "Ambient", "Intimate", "Small", "Big", "Deep", "Rumble", "Growl",
    "Wavetable", "Digital", "Analog", "Retro", "Vintage", "Dubstep", "Chiptune",
    "Formant Vocal", "Synthetic Vox", "Choir", "Pluck", "Sustained", "Short", "Staccato",
    "Snappy", "Pizzicato", "Spiccato", "Impact", "Hit", "Swell", "Thick", "Present",
    "Sharp", "Harsh", "Bell", "Full", "Silky", "Square", "Pulse", "Saw", "Sine", "Triangle",
    "White Noise", "Pure Tone", "FM", "Supersaw", "Reese", "Filter",
]

KEYBED_TIMBRE_W = [
    13, 12, 8, 9, 11, 12, 9, 8, 8,
    8, 8, 5, 5, 6, 8, 8, 5, 7,
    6, 6, 5, 5, 5, 4, 4, 5,
    7, 5, 4, 4, 3, 6, 3, 3, 3,
    4, 4, 3, 2, 4, 6, 5, 5,
    5, 6, 6, 6, 5, 4, 4,
    2, 2, 3, 6, 6, 4, 4,
    5, 3, 3, 3, 3, 3, 9, 6,
    4, 4, 7, 8, 6, 7, 6, 5,
    2, 5, 5, 4, 4, 5, 4, 4,
]


def keybed_timbre_tag_choices() -> List[str]:
    """Alphabetized UI choices without changing timbre-tag weights."""
    return sorted(dedupe_keep_order(KEYBED_TIMBRE_TAGS), key=str.casefold)


KEYBED_OSCILLATOR_TAGS = ["Pure Tone", "Sine", "Saw", "Triangle", "Pulse", "Square", "White Noise"]
KEYBED_OSCILLATOR_W = [9, 4, 5, 3, 6, 6, 1]

KEYBED_FAMILY_TAG_BOOST: Dict[str, List[str]] = {
    "Bass": ["Warm", "Thick", "Fat", "Deep", "Punchy", "Gritty", "Clean", "Full", "Dark", "Rumble"],
    "Keys": ["Warm", "Clean", "Bright", "Rich", "Smooth", "Sparkly", "Bell", "Soft", "Full"],
    "Synth": ["Warm", "Bright", "Thick", "Fat", "Digital", "Analog", "Pulse", "Square", "Saw", "Wavetable", "Supersaw", "Clean", "Gritty"],
    "Bowed Strings": ["Warm", "Rich", "Smooth", "Airy", "Sustained", "Dark", "Bright", "Full"],
    "Mallet": ["Bright", "Sparkly", "Metallic", "Bell", "Clean", "Pluck", "Woody", "Crisp"],
    "Wind": ["Airy", "Breathy", "Hollow", "Woody", "Thin", "Bright", "Smooth"],
    "Brass": ["Bright", "Nasal", "Biting", "Big", "Present", "Warm", "Harsh"],
    "Guitar": ["Clean", "Bright", "Woody", "Pluck", "Gritty", "Warm"],
    "Vocal": ["Airy", "Choir", "Synthetic Vox", "Warm", "Smooth", "Distant"],
    "Pure Tone": ["Clean", "Full", "Bright", "Warm", "Thin"],
    "White Noise": ["Noisy", "Bright", "Wide", "Airy"],
    "Plucked Strings": ["Pluck", "Bright", "Clean", "Sparkly", "Warm"],
}

KEYBED_TAG_MUTEX_GROUPS = [
    {"Sustained", "Short", "Staccato"},
    {"Pizzicato", "Spiccato", "Sustained"},
    {"Sine", "Saw", "Triangle", "Pulse", "Square", "White Noise"},
]

KEYBED_DRY_TAG_BLOCKLIST = {"Distant", "Far", "Spacey", "Ambient"}

KEYBED_FX_WORDS = ("reverb", "delay", "distortion", "phaser", "bitcrush")
KEYBED_CONTROL_TOKENS = {
    KEYBED_PREFIX_TOKEN.lower(),
    KEYBED_SEQUENCE_TOKEN.lower(),
    KEYBED_TIMBRE_PROFILE_TOKEN.lower(),
    KEYBED_TARGET_NOTE_TOKEN.lower(),
    KEYBED_TARGET_POSITION_TOKEN.lower(),
    KEYBED_CHROMATIC_CHUNK_TOKEN.lower(),
    KEYBED_NOTE_SEQUENCE_TOKEN.lower(),
    "wet",
    "dry",
}
KEYBED_CONTROL_TOKENS.update(label.lower() for label in KEYBED_REGISTER_LABELS)


# ============================================================
# Note / position helpers
# ============================================================


def midi_to_note_name(midi_note: int) -> str:
    midi_note = int(midi_note)
    pc = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{KEYBED_NOTE_NAMES[pc]}{octave}"


def normalize_keybed_note(note: str) -> Optional[str]:
    midi = note_name_to_midi(note)
    if midi is None:
        return None
    if midi < KEYBED_MIN_MIDI or midi > KEYBED_MAX_MIDI:
        return None
    return midi_to_note_name(midi)


def keybed_note_to_midi(note: str) -> int:
    midi = note_name_to_midi(note)
    if midi is None:
        raise ValueError(f"Invalid note name: {note}")
    if midi < KEYBED_MIN_MIDI or midi > KEYBED_MAX_MIDI:
        raise ValueError(f"Keybed note outside {KEYBED_MIN_NOTE}-{KEYBED_MAX_NOTE}: {note}")
    return int(midi)


def midi_to_keybed_pos(midi_note: int) -> int:
    midi_note = int(midi_note)
    if midi_note < KEYBED_MIN_MIDI or midi_note > KEYBED_MAX_MIDI:
        raise ValueError(f"MIDI note outside keybed range {KEYBED_MIN_MIDI}-{KEYBED_MAX_MIDI}: {midi_note}")
    return midi_note - KEYBED_MIN_MIDI + 1


def midi_to_keybed_pos_token(midi_note: int) -> str:
    return f"keybed_pos_{midi_to_keybed_pos(midi_note):03d}"


def note_to_keybed_pos_token(note: str) -> str:
    return midi_to_keybed_pos_token(keybed_note_to_midi(note))


def notes_between(start_note: str, end_note: str) -> List[str]:
    start_midi = keybed_note_to_midi(start_note)
    end_midi = keybed_note_to_midi(end_note)
    if end_midi < start_midi:
        start_midi, end_midi = end_midi, start_midi
    return [midi_to_note_name(m) for m in range(start_midi, end_midi + 1)]


def preview_notes_from_shape(root_note: str, shape: str) -> List[str]:
    root_midi = keybed_note_to_midi(root_note)
    intervals = KEYBED_PREVIEW_SHAPES.get(shape, KEYBED_PREVIEW_SHAPES["Minor Triad"])
    out = []
    for interval in intervals:
        midi = root_midi + int(interval)
        if KEYBED_MIN_MIDI <= midi <= KEYBED_MAX_MIDI:
            out.append(midi_to_note_name(midi))
    return out


def midi_to_keybed_register_label(midi_note: int) -> str:
    """Map KEYBED MIDI pitch to the same broad register labels used by ONESHOT tests."""
    midi_note = int(midi_note)

    first_lo = keybed_note_to_midi(KEYBED_REGISTER_BANDS[0][0])
    last_hi = keybed_note_to_midi(KEYBED_REGISTER_BANDS[-1][1])

    if midi_note <= first_lo:
        return KEYBED_REGISTER_BANDS[0][2]
    if midi_note >= last_hi:
        return KEYBED_REGISTER_BANDS[-1][2]

    for lo_note, hi_note, label in KEYBED_REGISTER_BANDS:
        lo = keybed_note_to_midi(lo_note)
        hi = keybed_note_to_midi(hi_note)
        if lo <= midi_note <= hi:
            return label

    return "Medium Register"


def note_to_keybed_register_label(note: str) -> str:
    return midi_to_keybed_register_label(keybed_note_to_midi(note))


def normalize_keybed_prompt_style(prompt_style: Optional[str] = None) -> str:
    value = str(prompt_style or KEYBED_PROMPT_STYLE_TARGET_ONLY).strip().lower()
    aliases = {
        "target": KEYBED_PROMPT_STYLE_TARGET_ONLY,
        "target only": KEYBED_PROMPT_STYLE_TARGET_ONLY,
        "target_position_only": KEYBED_PROMPT_STYLE_TARGET_ONLY,
        "target position only": KEYBED_PROMPT_STYLE_TARGET_ONLY,
        KEYBED_PROMPT_STYLE_TARGET_ONLY.lower(): KEYBED_PROMPT_STYLE_TARGET_ONLY,

        "after": KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER,
        "target + register + note": KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER,
        "target_then_register_note": KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER,
        KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER.lower(): KEYBED_PROMPT_STYLE_REGISTER_NOTE_AFTER,

        "before": KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE,
        "register + note + target": KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE,
        "register_note_then_target": KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE,
        KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE.lower(): KEYBED_PROMPT_STYLE_REGISTER_NOTE_BEFORE,
    }
    return aliases.get(value, KEYBED_PROMPT_STYLE_TARGET_ONLY)


def keybed_prompt_style_slug(prompt_style: Optional[str] = None) -> str:
    # Modern KEYBED prompts use Target Note for singles. Legacy prompt-style
    # labels are accepted by wrappers, but no longer affect emitted grammar.
    return "target_note"



# ============================================================
# Descriptor cleanup / assembly
# ============================================================


def normalize_keybed_mode(mode: str) -> str:
    return "experimental" if normalize_mode_to_profile(mode) == "mix" else "simple"


def is_keybed_fx_token(token: str) -> bool:
    value = str(token or "").strip().lower()
    return any(word in value for word in KEYBED_FX_WORDS)


def normalize_keybed_tokens(prompt: str) -> List[str]:
    return [part.strip() for part in str(prompt or "").split(",") if part and part.strip()]


def split_keybed_descriptor_tokens(descriptor: str) -> Tuple[List[str], List[str]]:
    """
    Returns (body_tokens, explicit_fx_tokens).

    Accepts either a clean descriptor or a pasted full single/sequence prompt.
    Removes Keybed control grammar so final builders can reassemble the modern
    training schema:
      Keybed, Timbre Profile, <descriptor>, Wet/Dry + FX, Target Note, C4
      Keybed, Sequence, Timbre Profile, <descriptor>, Wet/Dry + FX,
      Chromatic Chunk, Note Sequence, C4, C#4, ...

    Legacy debug grammar such as Target Position, keybed_pos_049 is stripped,
    but it is never emitted by the modern builders.
    """
    body: List[str] = []
    fx: List[str] = []

    skip_next_value = False
    stop_after_note_sequence = False

    for raw in normalize_keybed_tokens(descriptor):
        token = raw.strip()
        low = token.lower()

        if stop_after_note_sequence:
            # All following note tokens belong to the sequence metadata, not the descriptor.
            continue

        if skip_next_value:
            skip_next_value = False
            continue

        if low in {
            KEYBED_PREFIX_TOKEN.lower(),
            KEYBED_SEQUENCE_TOKEN.lower(),
            KEYBED_TIMBRE_PROFILE_TOKEN.lower(),
            KEYBED_CHROMATIC_CHUNK_TOKEN.lower(),
        }:
            continue

        if low in {KEYBED_TARGET_NOTE_TOKEN.lower(), KEYBED_TARGET_POSITION_TOKEN.lower()}:
            skip_next_value = True
            continue

        if low == KEYBED_NOTE_SEQUENCE_TOKEN.lower():
            stop_after_note_sequence = True
            continue

        if KEYBED_POS_RE.match(token):
            continue
        if NOTE_TOKEN_RE.match(token):
            continue
        if low in KEYBED_CONTROL_TOKENS:
            continue

        if is_keybed_fx_token(token):
            fx.append(token)
        else:
            body.append(token)

    return dedupe_keep_order(body), dedupe_keep_order(fx)

def clean_keybed_descriptor(descriptor: str) -> str:
    body, fx = split_keybed_descriptor_tokens(descriptor)
    return join_prompt(body + fx)



def keybed_pos_token_to_midi(keybed_pos: str) -> int:
    match = re.match(r"^keybed_pos_(\d{3})$", str(keybed_pos or "").strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid keybed position token: {keybed_pos}")
    pos = int(match.group(1))
    if pos < 1 or pos > KEYBED_POSITION_COUNT:
        raise ValueError(f"Keybed position outside 001-{KEYBED_POSITION_COUNT:03d}: {keybed_pos}")
    return KEYBED_MIN_MIDI + pos - 1


def keybed_pos_token_to_note(keybed_pos: str) -> str:
    return midi_to_note_name(keybed_pos_token_to_midi(keybed_pos))


def _resolve_keybed_fx_tokens(
    descriptor: str,
    *,
    wet=True,
    wetdry=None,
    fx: Optional[List[str]] = None,
    seed=None,
) -> Tuple[bool, List[str], List[str]]:
    is_wet = resolve_shared_wet(wet=wet, wetdry=wetdry)
    body_tokens, explicit_fx = split_keybed_descriptor_tokens(descriptor)

    if not is_wet:
        return is_wet, body_tokens, []

    if fx is not None:
        return is_wet, body_tokens, list(fx)
    if explicit_fx:
        return is_wet, body_tokens, explicit_fx

    # Wet/Dry is a conditioning control, not an FX-policy validator.
    # If the user selected Wet but supplied no explicit FX descriptors, inject
    # only the Wet token later in the prompt builder. Random prompt generation
    # may still deliberately create FX descriptors before reaching this path.
    return is_wet, body_tokens, []


def build_keybed_base_prompt(
    descriptor: str,
    *,
    sequence: bool = False,
    wet=True,
    wetdry=None,
    fx: Optional[List[str]] = None,
    seed=None,
    include_prefix: bool = True,
    include_timbre_profile: bool = True,
    include_wetdry: bool = True,
) -> str:
    is_wet, body_tokens, fx_tokens = _resolve_keybed_fx_tokens(
        descriptor,
        wet=wet,
        wetdry=wetdry,
        fx=fx,
        seed=seed,
    )

    tokens: List[str] = []
    if include_prefix:
        tokens.append(KEYBED_PREFIX_TOKEN)
    if sequence:
        tokens.append(KEYBED_SEQUENCE_TOKEN)
    if include_timbre_profile:
        tokens.append(KEYBED_TIMBRE_PROFILE_TOKEN)

    tokens.extend(body_tokens)
    tokens.extend(wetdry_tokens(wet=is_wet, include_wetdry=include_wetdry))
    tokens.extend(fx_tokens if is_wet else [])
    return join_prompt(dedupe_keep_order(tokens))


def build_keybed_single_note_prompt(
    descriptor: str,
    note: str,
    *,
    wet=True,
    wetdry=None,
    fx: Optional[List[str]] = None,
    seed=None,
    include_prefix: bool = True,
    include_timbre_profile: bool = True,
    include_wetdry: bool = True,
) -> str:
    normalized_note = normalize_keybed_note(note)
    if normalized_note is None:
        raise ValueError(f"Invalid KEYBED target note: {note}")

    base = build_keybed_base_prompt(
        descriptor,
        sequence=False,
        wet=wet,
        wetdry=wetdry,
        fx=fx,
        seed=seed,
        include_prefix=include_prefix,
        include_timbre_profile=include_timbre_profile,
        include_wetdry=include_wetdry,
    )
    return join_prompt([base, KEYBED_TARGET_NOTE_TOKEN, normalized_note])


def build_keybed_note_sequence_prompt(
    descriptor: str,
    notes: Iterable[str],
    *,
    wet=True,
    wetdry=None,
    fx: Optional[List[str]] = None,
    seed=None,
    chromatic_chunk: bool = True,
    include_prefix: bool = True,
    include_timbre_profile: bool = True,
    include_wetdry: bool = True,
) -> str:
    normalized_notes: List[str] = []
    for note in notes:
        normalized = normalize_keybed_note(str(note))
        if normalized is None:
            raise ValueError(f"Invalid KEYBED sequence note: {note}")
        normalized_notes.append(normalized)

    base = build_keybed_base_prompt(
        descriptor,
        sequence=True,
        wet=wet,
        wetdry=wetdry,
        fx=fx,
        seed=seed,
        include_prefix=include_prefix,
        include_timbre_profile=include_timbre_profile,
        include_wetdry=include_wetdry,
    )

    tail: List[str] = []
    if chromatic_chunk:
        tail.append(KEYBED_CHROMATIC_CHUNK_TOKEN)
    tail.append(KEYBED_NOTE_SEQUENCE_TOKEN)
    tail.extend(normalized_notes)
    # Do not dedupe notes here. Repeated notes can be intentional sequence metadata.
    return join_prompt([base] + tail)


def build_keybed_sequence_rows(
    descriptor: str,
    note_groups: Iterable[Iterable[str]],
    *,
    wet=True,
    wetdry=None,
    seed=None,
    chromatic_chunk: bool = True,
) -> List[Dict[str, object]]:
    resolved_seed = resolve_keybed_seed(seed)
    rows: List[Dict[str, object]] = []
    for idx, group in enumerate(note_groups, start=1):
        normalized_notes = [normalize_keybed_note(str(n)) for n in group]
        normalized_notes = [n for n in normalized_notes if n]
        if not normalized_notes:
            continue
        prompt = build_keybed_note_sequence_prompt(
            descriptor,
            normalized_notes,
            wet=wet,
            wetdry=wetdry,
            seed=resolved_seed,
            chromatic_chunk=chromatic_chunk,
        )
        rows.append({
            "prompt_schema": "note_sequence",
            "sequence_index": idx,
            "chromatic_chunk": bool(chromatic_chunk),
            "note_count": len(normalized_notes),
            "notes": normalized_notes,
            "prompt": prompt,
            "seed": sha_seed(str(resolved_seed), "keybed_sequence", str(idx), "|".join(normalized_notes)),
            "base_seed": resolved_seed,
        })
    return rows


def build_keybed_prompt_for_position(
    descriptor: str,
    keybed_pos: str,
    *,
    wet=True,
    wetdry=None,
    fx: Optional[List[str]] = None,
    note: Optional[str] = None,
    midi_note: Optional[int] = None,
    prompt_style: Optional[str] = None,
    include_prefix: bool = True,
    include_timbre_profile: bool = True,
    include_wetdry: bool = True,
) -> str:
    """
    Backward-compatible wrapper for older callers.

    Older debug tabs passed keybed_pos_XXX and optionally prompt_style. The
    modern KEYBED grammar does not emit Target Position; it resolves the note
    and emits:
      Keybed, Timbre Profile, ..., Target Note, C4
    """
    if note in (None, ""):
        if midi_note is not None:
            note = midi_to_note_name(int(midi_note))
        else:
            note = keybed_pos_token_to_note(keybed_pos)

    return build_keybed_single_note_prompt(
        descriptor,
        str(note),
        wet=wet,
        wetdry=wetdry,
        fx=fx,
        include_prefix=include_prefix,
        include_timbre_profile=include_timbre_profile,
        include_wetdry=include_wetdry,
    )

def build_keybed_prompt_rows(
    descriptor: str,
    notes: Iterable[str],
    *,
    wet=True,
    wetdry=None,
    seed=None,
    prompt_style: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Build full prompts for a note list. Notes are always returned for UI/manifest.

    Prompt style controls whether the note/register are also injected as an
    experimental conditioning test:
      - Target Position Only: trained KEYBED grammar
      - Target + Register + Note: appends register/note after target position
      - Register + Note + Target: places register/note before target position
    """
    is_wet = resolve_shared_wet(wet=wet, wetdry=wetdry)
    style = KEYBED_TARGET_NOTE_TOKEN
    _body, explicit_fx = split_keybed_descriptor_tokens(descriptor)
    # Preserve only FX descriptors that are actually present in the prompt.
    # Selecting Wet by itself must not synthesize an arbitrary FX tag.
    fx = explicit_fx if is_wet else []

    rows: List[Dict[str, object]] = []
    for note in notes:
        midi = keybed_note_to_midi(note)
        normalized_note = midi_to_note_name(midi)
        pos = midi_to_keybed_pos(midi)
        pos_token = f"keybed_pos_{pos:03d}"
        rows.append({
            "prompt_style": style,
            "note": normalized_note,
            "register": midi_to_keybed_register_label(midi),
            "midi": midi,
            "keybed_pos": pos_token,
            "prompt": build_keybed_prompt_for_position(
                descriptor,
                pos_token,
                wet=is_wet,
                fx=fx,
                note=normalized_note,
                midi_note=midi,
                prompt_style=style,
            ),
            "seed": keybed_seed_for_position(resolved_seed, pos_token),
            "base_seed": resolved_seed,
        })
    return rows


# ============================================================
# Random descriptor generation
# ============================================================


def resolve_keybed_seed(seed=None) -> int:
    if seed in ("", None, -1, "-1"):
        return random.randint(0, 2**31 - 1)
    return int(seed)


def keybed_seed_for_position(base_seed: int, keybed_pos: str) -> int:
    return sha_seed(str(int(base_seed)), str(keybed_pos))


def normalize_keybed_family_hint(family_hint: Optional[str]) -> Optional[str]:
    if not family_hint:
        return None
    h = str(family_hint).strip().lower()
    for fam in KEYBED_SUBFAMILIES.keys():
        if fam.lower() == h:
            return fam
    return None


def pick_keybed_family(rng: random.Random, *, profile: str, family_hint: Optional[str] = None) -> str:
    hinted = normalize_keybed_family_hint(family_hint)
    if hinted:
        return hinted

    if normalize_keybed_mode(profile) == "experimental":
        return weighted_choice(rng, KEYBED_FAMILIES_EXPERIMENTAL, KEYBED_FAMILY_W_EXPERIMENTAL)
    return weighted_choice(rng, KEYBED_FAMILIES_SIMPLE, KEYBED_FAMILY_W_SIMPLE)


def pick_keybed_subfamily(rng: random.Random, family: str) -> str:
    pairs = KEYBED_SUBFAMILIES.get(family, [])
    if not pairs:
        return ""
    items = [item for item, _ in pairs]
    weights = [weight for _, weight in pairs]
    return weighted_choice(rng, items, weights)


def filter_keybed_tags_for_wetdry(tags: List[str], *, wet: bool) -> List[str]:
    if wet:
        return tags
    return [t for t in tags if t not in KEYBED_DRY_TAG_BLOCKLIST]


def enforce_keybed_mutexes(rng: random.Random, tokens: List[str]) -> List[str]:
    out = list(tokens)
    for group in KEYBED_TAG_MUTEX_GROUPS:
        hits = [t for t in out if t in group]
        if len(hits) <= 1:
            continue
        keep = rng.choice(hits)
        kept = False
        next_out: List[str] = []
        for token in out:
            if token in group:
                if token == keep and not kept:
                    next_out.append(token)
                    kept = True
            else:
                next_out.append(token)
        out = next_out
    return out


def _keybed_subfamily_nudge(
    rng: random.Random,
    *,
    family: str,
    subfamily: str,
) -> List[str]:
    if subfamily in {"Reese Bass", "808", "Wavetable Bass", "Synth Bass"}:
        return rng.sample(["Deep", "Fat", "Rumble", "Growl", "Thick"], k=1)
    if subfamily in {"Bell", "Tubular Bells", "Music Box", "Church Bell", "Toy Bell"}:
        return rng.sample(["Bright", "Sparkly", "Metallic", "Bell"], k=1)
    if subfamily in {"Violin", "Cello", "Viola", "Digital Strings", "Ensemble"} and family == "Bowed Strings":
        return rng.sample(["Sustained", "Smooth", "Warm", "Nasal", "Rich"], k=1)
    if subfamily in {"Synth Lead", "Supersaw", "FM Synth", "Wavetable Synth"}:
        return rng.sample(["Bright", "Digital", "Analog", "Wide", "Present"], k=1)
    return []


def sample_keybed_tags(
    rng: random.Random,
    *,
    family: str,
    subfamily: str,
    profile: str,
    wet: bool,
    secondary_family: str = "",
    secondary_subfamily: str = "",
) -> List[str]:
    profile = normalize_keybed_mode(profile)
    is_experimental = profile == "experimental"
    has_secondary_context = bool(secondary_family or secondary_subfamily)

    # Experimental hybrids use the full timbre budget even when one side is a
    # primitive. Simple mode keeps sparse Pure Tone / White Noise behavior.
    if is_experimental and has_secondary_context:
        k_base = rng.choice([5, 6, 7, 8])
        max_tags = 10
    elif family == "Pure Tone":
        k_base = rng.choice([0, 1, 1, 2]) if not is_experimental else rng.choice([1, 2, 2, 3])
        max_tags = 4 if not is_experimental else 5
    elif family == "White Noise":
        k_base = rng.choice([1, 2, 3])
        max_tags = 5
    elif is_experimental:
        k_base = rng.choice([5, 6, 7, 8])
        max_tags = 10
    else:
        k_base = rng.choice([4, 5, 6])
        max_tags = 7

    tags: List[str] = []
    tags += weighted_sample_unique(rng, KEYBED_TIMBRE_TAGS, KEYBED_TIMBRE_W, k_base)

    context_families = dedupe_keep_order([family, secondary_family])
    for context_family in context_families:
        boosts = KEYBED_FAMILY_TAG_BOOST.get(context_family, [])
        if boosts:
            k_boost = rng.choice([1, 1, 2]) if not is_experimental else rng.choice([1, 1, 2])
            tags += rng.sample(boosts, k=min(k_boost, len(boosts)))

    tags += _keybed_subfamily_nudge(rng, family=family, subfamily=subfamily)
    if secondary_family or secondary_subfamily:
        tags += _keybed_subfamily_nudge(
            rng,
            family=secondary_family,
            subfamily=secondary_subfamily,
        )

    # Pure Tone as a timbre/oscillator modifier, not a separate UI mode.
    if "Pure Tone" not in {family, secondary_family}:
        p_osc = 0.07 if not is_experimental else 0.22
        if rng.random() < p_osc:
            k_osc = 1 if rng.random() < 0.65 else 2
            tags += weighted_sample_unique(rng, KEYBED_OSCILLATOR_TAGS, KEYBED_OSCILLATOR_W, k_osc)

    tags = dedupe_keep_order(tags)
    tags = enforce_keybed_mutexes(rng, tags)
    tags = filter_keybed_tags_for_wetdry(tags, wet=wet)
    return clamp_list(rng, tags, max_tags)


def build_keybed_descriptor_string(
    *,
    family: str,
    subfamily: str,
    tags: List[str],
    wet: bool,
    fx: List[str],
    include_wetdry: bool = False,
) -> str:
    """
    Descriptor body for the UI textbox. By default, this does not include Wet/Dry
    or FX because the Keybed tab keeps wet/dry UI-authoritative and adds the
    final FX tokens only when building per-position prompts.
    """
    tokens: List[str] = [family]
    if subfamily:
        tokens.append(subfamily)
    tokens.extend(tags)
    if include_wetdry:
        tokens.extend(wetdry_tokens(wet=wet, include_wetdry=True))
        tokens.extend(fx if wet else [])
    return join_prompt(dedupe_keep_order(tokens))


def context_preview_root_weights_for_prompt(
    *,
    family: str,
    subfamily: str,
) -> List[int]:
    """
    Resolve a soft preview-root distribution for a random KEYBED descriptor.

    Priority mirrors the ONESHOT note planner:
      1. exact family + subfamily pair
      2. family-level fallback
      3. neutral fallback
    """
    pair_key = (str(family or "").strip(), str(subfamily or "").strip())
    if pair_key in KEYBED_CONTEXT_PREVIEW_ROOT_WEIGHTS_BY_PAIR:
        return KEYBED_CONTEXT_PREVIEW_ROOT_WEIGHTS_BY_PAIR[pair_key]

    family_key = str(family or "").strip()
    if family_key in KEYBED_CONTEXT_PREVIEW_ROOT_WEIGHTS_BY_FAMILY:
        return KEYBED_CONTEXT_PREVIEW_ROOT_WEIGHTS_BY_FAMILY[family_key]

    return KEYBED_CONTEXT_NEUTRAL_PREVIEW_ROOT_W


def pick_keybed_preview_root_for_context(
    rng: random.Random,
    *,
    family: str,
    subfamily: str,
) -> str:
    """Pick a concrete C2-G6 preview root from family/subfamily context."""
    weights = context_preview_root_weights_for_prompt(family=family, subfamily=subfamily)
    band_key = weighted_choice(rng, KEYBED_PREVIEW_ROOT_KEYS_IN_ORDER, weights)
    lo_note, hi_note = KEYBED_PREVIEW_ROOT_BANDS.get(band_key, KEYBED_PREVIEW_ROOT_BANDS["ROOT_MID"])
    lo = keybed_note_to_midi(lo_note)
    hi = keybed_note_to_midi(hi_note)
    return midi_to_note_name(rng.randint(int(lo), int(hi)))


def prompt_generator_keybed_descriptor(
    *,
    seed=None,
    mode="standard",
    instrument_mode="hybrid",
    family_hint=None,
    wet=True,
    wetdry=None,
    include_wetdry: bool = False,
    return_plan: bool = False,
    **_,
):
    """
    Random KEYBED descriptor generator.

    Prompt mode and instrument mode are intentionally independent:
      - Simple / Experimental controls hierarchy, vocabulary, and tag richness.
      - Single / Hybrid controls how many instrument concepts are shown.

    Simple keeps the tuned family -> subfamily hierarchy internally even when
    Single emits only the more specific subfamily token. Experimental Single
    picks one unrestricted vocabulary token and resolves its hidden context for
    tags and preview-root planning. Hybrid preserves the existing two-token
    behavior in both profiles.
    """
    resolved_seed = resolve_keybed_seed(seed)
    rng = random.Random(resolved_seed)
    profile = normalize_keybed_mode(mode)
    resolved_instrument_mode = normalize_keybed_instrument_mode(instrument_mode)
    is_hybrid = resolved_instrument_mode == KEYBED_INSTRUMENT_MODE_HYBRID
    is_wet = resolve_shared_wet(wet=wet, wetdry=wetdry)

    if profile == "experimental":
        if is_hybrid:
            instrument_1, instrument_2 = pick_keybed_experimental_instruments(
                rng,
                family_hint=family_hint,
            )
        else:
            instrument_1 = (
                normalize_keybed_family_hint(family_hint)
                or rng.choice(KEYBED_EXPERIMENTAL_INSTRUMENTS)
            )
            instrument_2 = ""

        family, subfamily = keybed_instrument_context(instrument_1)
        secondary_family, secondary_subfamily = (
            keybed_instrument_context(instrument_2) if instrument_2 else ("", "")
        )

        # Generated vocabulary should always resolve, but neutral fallbacks keep
        # pasted/custom future tokens safe for downstream planning.
        family = family or instrument_1
        subfamily = subfamily or ""
        secondary_family = secondary_family or (instrument_2 if instrument_2 else "")
        secondary_subfamily = secondary_subfamily or ""

        tags = sample_keybed_tags(
            rng,
            family=family,
            subfamily=subfamily,
            secondary_family=secondary_family if is_hybrid else "",
            secondary_subfamily=secondary_subfamily if is_hybrid else "",
            profile=profile,
            wet=is_wet,
        )
        descriptor = build_keybed_descriptor_string(
            family=instrument_1,
            subfamily=instrument_2 if is_hybrid else "",
            tags=tags,
            wet=is_wet,
            fx=[],
            include_wetdry=False,
        )
    else:
        # The hierarchy remains the hidden training-aware context in both modes.
        family = pick_keybed_family(rng, profile=profile, family_hint=family_hint)
        subfamily = pick_keybed_subfamily(rng, family)
        secondary_family = ""
        secondary_subfamily = ""

        tags = sample_keybed_tags(
            rng,
            family=family,
            subfamily=subfamily,
            profile=profile,
            wet=is_wet,
        )

        if is_hybrid:
            instrument_1 = family
            instrument_2 = subfamily
        else:
            # Show the user one useful instrument concept while retaining the
            # broad family + specific subfamily as hidden metadata/context.
            instrument_1 = subfamily or family
            instrument_2 = ""

        descriptor = build_keybed_descriptor_string(
            family=instrument_1,
            subfamily=instrument_2 if is_hybrid else "",
            tags=tags,
            wet=is_wet,
            fx=[],
            include_wetdry=False,
        )

    fx = choose_shared_fx(rng, wet=is_wet)
    if include_wetdry:
        descriptor = join_prompt(
            dedupe_keep_order(
                normalize_keybed_tokens(descriptor)
                + wetdry_tokens(wet=is_wet, include_wetdry=True)
                + (fx if is_wet else [])
            )
        )

    # Preview-root planning always uses the hidden hierarchy/context, not merely
    # the reduced Single-mode display string.
    preview_root = pick_keybed_preview_root_for_context(
        rng,
        family=family,
        subfamily=subfamily,
    )

    if return_plan:
        return {
            "prompt": descriptor,
            "instrument_1": instrument_1,
            "instrument_2": instrument_2 or None,
            "instrument_mode": resolved_instrument_mode,
            "family": family,
            "subfamily": subfamily,
            "secondary_family": secondary_family,
            "secondary_subfamily": secondary_subfamily,
            "tags": tags,
            "wet": is_wet,
            "fx": fx,
            "profile": profile,
            "seed": resolved_seed,
            "preview_root": preview_root,
            "preview_root_weights": context_preview_root_weights_for_prompt(
                family=family,
                subfamily=subfamily,
            ),
        }

    return descriptor


def infer_keybed_family_and_subfamily_from_descriptor(descriptor: str) -> Tuple[Optional[str], Optional[str]]:
    body, _fx = split_keybed_descriptor_tokens(descriptor)
    if not body:
        return None, None

    lowered = [token.strip().lower() for token in body]

    family: Optional[str] = None
    family_index = -1
    for idx, token in enumerate(lowered):
        for fam in KEYBED_SUBFAMILIES.keys():
            if fam.lower() == token:
                family = fam
                family_index = idx
                break
        if family:
            break

    # Useful fallback for pasted descriptors like:
    #   Sub Bass, Bass, Warm, Analog, Pluck
    # where the strongest register clue can appear before the family token.
    bass_aliases = {
        "sub bass": "Sub Bass",
        "808": "808",
        "reese bass": "Reese Bass",
        "synth bass": "Synth Bass",
        "wavetable bass": "Wavetable Bass",
    }
    alias_subfamily: Optional[str] = None
    for token in lowered:
        if token in bass_aliases:
            alias_subfamily = bass_aliases[token]
            if family is None:
                family = "Bass"
            break

    if not family:
        return None, None

    subfamily: Optional[str] = alias_subfamily

    # Prefer an exact listed subfamily after/near the family token for normal
    # generated descriptors: Family, Subfamily, tags...
    search_tokens = lowered[family_index + 1:] if family_index >= 0 else lowered
    for token in search_tokens:
        for sub, _weight in KEYBED_SUBFAMILIES.get(family, []):
            if str(sub).strip().lower() == token:
                subfamily = sub
                break
        if subfamily:
            break

    return family, subfamily


def infer_keybed_family_from_descriptor(descriptor: str) -> Optional[str]:
    family, _subfamily = infer_keybed_family_and_subfamily_from_descriptor(descriptor)
    return family


def default_preview_root_for_descriptor(descriptor: str, *, seed=None) -> str:
    family, subfamily = infer_keybed_family_and_subfamily_from_descriptor(descriptor)
    if family:
        resolved_seed = resolve_keybed_seed(seed)
        rng = random.Random(sha_seed(str(resolved_seed), descriptor, "preview_root_fallback"))
        return pick_keybed_preview_root_for_context(
            rng,
            family=family,
            subfamily=subfamily or "",
        )
    return "C4"
