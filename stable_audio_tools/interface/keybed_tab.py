import gc
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import gradio as gr
import torch
import torchaudio
from einops import rearrange

from ..inference.generation import generate_diffusion_cond
from .prompts import keybed_prompts
from .keybed_exporter import export_keybed_run


KEYBED_SECONDS_PER_NOTE = 3.0
KEYBED_SEQUENCE_GAP_SECONDS = 0.25
KEYBED_OUTPUT_SUBDIR = "Keybed_Source"
KEYBED_PREVIEW_OUTPUT_SUBDIR = "Keybed_Previews"
KEYBED_PREVIEW_GAP_MS = 120.0
KEYBED_PREVIEW_FADE_MS = 8.0
KEYBED_DEFAULT_STEPS = 80
KEYBED_DEFAULT_CFG = 6.0
KEYBED_DEFAULT_SAMPLER = "dpmpp-3m-sde"
KEYBED_DEFAULT_SIGMA_MIN = 0.03
KEYBED_DEFAULT_SIGMA_MAX = 500.0
KEYBED_DEFAULT_CFG_RESCALE = 0.0
KEYBED_INSTRUMENT_NAME_MAX_CHARS = 30

KEYBED_USER_ROOT_MIN = "C2"
KEYBED_USER_ROOT_MAX = "C6"
KEYBED_SEQUENCE_MAX_NOTE = "G#7"
KEYBED_CHUNK_SIZE = 6

KEYBED_ROOT_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEYBED_ROOT_OCTAVES = ["2", "3", "4", "5", "6"]
KEYBED_OCTAVE_CHOICES = ["0.5", "1", "2"]

# Fixed export ranges for full sampler generation.
# Preview remains context-aware; full sampler export intentionally ignores the
# preview root so the final instrument has a predictable keyboard spread.
KEYBED_FULL_SAMPLER_RANGES = {
    "Compact — C2 to B5 (48 notes / 8 chunks)": ("C2", "B5"),
    "Extended — C2 to F6 (54 notes / 9 chunks)": ("C2", "F6"),
    "Wide — C2 to B6 (60 notes / 10 chunks)": ("C2", "B6"),
}
KEYBED_FULL_SAMPLER_RANGE_CHOICES = list(KEYBED_FULL_SAMPLER_RANGES.keys())
KEYBED_FULL_SAMPLER_DEFAULT_RANGE = KEYBED_FULL_SAMPLER_RANGE_CHOICES[0]

VISIBLE_WETDRY_TOKENS = {"wet", "dry"}

def _build_keybed_instrument_picker_choices() -> List[str]:
    """Use the prompt module's alphabetized flat instrument vocabulary."""
    try:
        return list(keybed_prompts.keybed_flat_instrument_choices())
    except Exception:
        choices: List[str] = []
        for family, pairs in keybed_prompts.KEYBED_SUBFAMILIES.items():
            family_name = str(family or "").strip()
            if family_name and family_name.casefold() != "white noise":
                choices.append(family_name)
            for subfamily, _weight in pairs:
                subfamily_name = str(subfamily or "").strip()
                if subfamily_name and subfamily_name.casefold() != "white noise":
                    choices.append(subfamily_name)
        return sorted(dict.fromkeys(choices), key=str.casefold)


KEYBED_TAG_PICKER_INSTRUMENTS = _build_keybed_instrument_picker_choices()
KEYBED_TAG_PICKER_INSTRUMENT_KEYS = {
    value.casefold(): value for value in KEYBED_TAG_PICKER_INSTRUMENTS
}
KEYBED_NO_SECOND_INSTRUMENT = "None"
KEYBED_TAG_PICKER_OPTIONAL_INSTRUMENTS = [
    KEYBED_NO_SECOND_INSTRUMENT,
    *KEYBED_TAG_PICKER_INSTRUMENTS,
]
try:
    KEYBED_TAG_PICKER_TIMBRES = list(keybed_prompts.keybed_timbre_tag_choices())
except Exception:
    KEYBED_TAG_PICKER_TIMBRES = sorted(
        dict.fromkeys(keybed_prompts.KEYBED_TIMBRE_TAGS),
        key=str.casefold,
    )
KEYBED_TAG_PICKER_TIMBRE_KEYS = {
    value.casefold(): value for value in KEYBED_TAG_PICKER_TIMBRES
}

KEYBED_EXPORT_FORMAT_CHOICES = ["DecentSampler", "SFZ"]
KEYBED_DEFAULT_EXPORT_FORMATS = ["DecentSampler"]

KEYBED_BUTTON_LABELS = {
    "preview": "Generate Keybed Preview",
    "full": "Generate & Export Keybed",
}


def _normalize_keybed_export_formats(selected_formats) -> Tuple[str, ...]:
    if isinstance(selected_formats, str):
        selected = [selected_formats]
    else:
        selected = list(selected_formats or [])

    formats: List[str] = []
    for value in selected:
        key = str(value or "").strip().casefold()
        if key in {"decentsampler", "dspreset", "decent sampler"}:
            normalized = "dspreset"
        elif key == "sfz":
            normalized = "sfz"
        else:
            continue
        if normalized not in formats:
            formats.append(normalized)

    if not formats:
        raise gr.Error("Select at least one export type: DecentSampler or SFZ.")
    return tuple(formats)


def _keybed_export_format_label(formats: Iterable[str]) -> str:
    labels = ["DecentSampler" if fmt == "dspreset" else "SFZ" for fmt in formats]
    return " + ".join(labels)


def _keybed_export_completion_label(formats: Iterable[str]) -> str:
    format_list = list(formats)
    label = _keybed_export_format_label(format_list)
    noun = "instruments" if len(format_list) > 1 else "instrument"
    return f"{label} {noun} exported"


def _generation_id_from_path(path: str) -> str:
    name = os.path.basename(os.path.normpath(str(path or "")))
    match = re.search(r"(?:^|[_-])([0-9a-fA-F]{8})$", name)
    return match.group(1).upper() if match else name


def open_keybed_export_folder_action(export_dir):
    """Open the latest export folder on the desktop hosting Gradio.

    Generation/export itself is platform-independent. This helper uses the
    native desktop opener on Windows/macOS and tries common freedesktop/Linux
    openers without assuming a particular desktop environment.
    """
    folder = os.path.abspath(os.path.expanduser(str(export_dir or "").strip()))
    if not folder or not os.path.isdir(folder):
        raise gr.Error("No completed Keybed export folder is available yet.")

    try:
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
            return

        if sys.platform == "darwin":
            opener = shutil.which("open") or "/usr/bin/open"
            subprocess.Popen(
                [opener, folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return

        linux_openers = [
            ("xdg-open", ["xdg-open", folder]),
            ("gio", ["gio", "open", folder]),
            ("kde-open5", ["kde-open5", folder]),
            ("kde-open", ["kde-open", folder]),
            ("gnome-open", ["gnome-open", folder]),
        ]
        for executable, command in linux_openers:
            if shutil.which(executable):
                subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return

        raise RuntimeError(
            "No desktop folder opener was found. Open this folder manually: "
            f"{folder}"
        )
    except Exception as exc:
        raise gr.Error(f"Could not open the export folder: {exc}") from exc


def _strip_visible_wetdry_tokens(prompt: str) -> str:
    """Keep descriptor/FX visible in the textbox, but hide Wet/Dry control tokens."""
    tokens = []
    for part in str(prompt or "").split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() in VISIBLE_WETDRY_TOKENS:
            continue
        tokens.append(token)
    return ", ".join(tokens)


def _keybed_cli_status(message: str) -> None:
    print(f"[Keybed] {message}", flush=True)


def _searchable_dropdown(*args, **kwargs):
    """Use optional dropdown features when supported without forcing an upgrade."""
    try:
        parameters = inspect.signature(gr.Dropdown).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "filterable" in parameters:
        kwargs.setdefault("filterable", True)
    else:
        kwargs.pop("filterable", None)
    if "allow_custom_value" not in parameters:
        kwargs.pop("allow_custom_value", None)
    return gr.Dropdown(*args, **kwargs)


def _instrument_name_textbox(*, elem_id: str):
    """Create the optional name field without requiring a newer Gradio build."""
    kwargs = {
        "label": "Instrument Name (Optional)",
        "placeholder": f"Optional custom name — max {KEYBED_INSTRUMENT_NAME_MAX_CHARS} characters",
        "lines": 1,
        "max_lines": 1,
        "elem_id": elem_id,
    }
    try:
        parameters = inspect.signature(gr.Textbox).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "max_length" in parameters:
        kwargs["max_length"] = KEYBED_INSTRUMENT_NAME_MAX_CHARS
    return gr.Textbox(**kwargs)


def _dedupe_casefold(tokens: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for token in tokens:
        value = str(token or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _normalize_optional_instrument(value) -> Optional[str]:
    """Translate the Instrument 2 UI's explicit None choice into no token."""
    text = str(value or "").strip()
    if not text or text.casefold() == KEYBED_NO_SECOND_INSTRUMENT.casefold():
        return None
    return text


def _remove_exact_tokens(tokens: List[str], values: Iterable[str]) -> List[str]:
    blocked = {str(value or "").strip().casefold() for value in values if str(value or "").strip()}
    if not blocked:
        return list(tokens)
    return [token for token in tokens if str(token).strip().casefold() not in blocked]


def _leading_instrument_tokens(tokens: Iterable[str], *, max_count: int = 2) -> List[str]:
    """Read only leading picker-known instruments so later Bell/Pluck tags stay timbre."""
    out: List[str] = []
    for token in tokens:
        value = str(token or "").strip()
        if not value:
            continue
        if value.casefold() not in KEYBED_TAG_PICKER_INSTRUMENT_KEYS:
            break
        out.append(value)
        if len(out) >= int(max_count):
            break
    return out


def _instrument_parent_candidates(instrument: str) -> List[Tuple[str, int]]:
    target = str(instrument or "").strip().casefold()
    if not target:
        return []
    candidates: List[Tuple[str, int]] = []
    for family, pairs in keybed_prompts.KEYBED_SUBFAMILIES.items():
        if str(family).casefold() == "white noise":
            continue
        for subfamily, weight in pairs:
            if str(subfamily or "").strip().casefold() == target:
                candidates.append((str(family), int(weight)))
    return candidates


def _picker_preview_context(instrument_1, instrument_2) -> Tuple[Optional[str], Optional[str]]:
    """Use Instrument 1 as the primary register clue, with Instrument 2 as fallback."""
    selected = _dedupe_casefold([
        _normalize_optional_instrument(instrument_1),
        _normalize_optional_instrument(instrument_2),
    ])
    if not selected:
        return None, None

    family_names = {
        str(family).casefold(): str(family)
        for family in keybed_prompts.KEYBED_SUBFAMILIES.keys()
        if str(family).casefold() != "white noise"
    }

    for instrument in selected:
        key = instrument.casefold()
        if key in family_names:
            family = family_names[key]
            valid_subfamilies = {
                str(subfamily).casefold(): str(subfamily)
                for subfamily, _weight in keybed_prompts.KEYBED_SUBFAMILIES.get(family, [])
                if str(subfamily or "").strip()
            }
            subfamily = next(
                (valid_subfamilies[other.casefold()] for other in selected if other.casefold() in valid_subfamilies),
                None,
            )
            return family, subfamily

        parents = _instrument_parent_candidates(instrument)
        if parents:
            # Highest ontology weight wins for ambiguous tags such as Bell/Pluck.
            family = max(parents, key=lambda item: item[1])[0]
            return family, instrument

    return None, None


def _suggest_preview_root_from_picker(
    descriptor: str,
    instrument_1,
    instrument_2,
    seed_str,
) -> str:
    family, subfamily = _picker_preview_context(instrument_1, instrument_2)
    if family:
        context_tokens = [family]
        if subfamily:
            context_tokens.append(subfamily)
        context_tokens.append(str(descriptor or ""))
        context_descriptor = keybed_prompts.join_prompt(context_tokens)
    else:
        context_descriptor = str(descriptor or "")
    return keybed_prompts.default_preview_root_for_descriptor(
        context_descriptor,
        seed=_prompt_seed_from_text(seed_str),
    )


def _enforce_picker_mutexes_keep_last(tokens: List[str]) -> List[str]:
    """Keep the most recently selected tag when manual picker choices conflict."""
    out = list(tokens)
    for group in keybed_prompts.KEYBED_TAG_MUTEX_GROUPS:
        group_keys = {str(value).casefold() for value in group}
        hit_indexes = [idx for idx, token in enumerate(out) if str(token).casefold() in group_keys]
        if len(hit_indexes) <= 1:
            continue
        keep_index = hit_indexes[-1]
        out = [
            token
            for idx, token in enumerate(out)
            if str(token).casefold() not in group_keys or idx == keep_index
        ]
    return out


def apply_keybed_tag_picker_action(
    current_descriptor,
    instrument_1,
    instrument_2,
    timbre_tags,
    wetdry_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    """Apply two unrestricted instrument tags plus the selected timbre vocabulary."""
    body_tokens, fx_tokens = keybed_prompts.split_keybed_descriptor_tokens(current_descriptor)

    # Instrument picker values always occupy the first two descriptor positions.
    # Only remove leading known instruments so words such as Bell/Pluck later in
    # a manually edited prompt are not accidentally reclassified.
    existing_instruments = _leading_instrument_tokens(body_tokens)
    body_tokens = body_tokens[len(existing_instruments):]

    # The timbre picker is authoritative for known timbre tags. Custom/manual
    # descriptor words remain untouched, while deselecting a known tag removes it.
    manual_tokens = [
        token
        for token in body_tokens
        if str(token).strip().casefold() not in KEYBED_TAG_PICKER_TIMBRE_KEYS
    ]

    selected_instruments = _dedupe_casefold([
        _normalize_optional_instrument(instrument_1),
        _normalize_optional_instrument(instrument_2),
    ])[:2]
    selected_tags = [str(tag).strip() for tag in (timbre_tags or []) if str(tag or "").strip()]
    is_wet = _wetdry_from_label(wetdry_label) == "Wet"
    selected_tags = keybed_prompts.filter_keybed_tags_for_wetdry(selected_tags, wet=is_wet)

    merged = selected_instruments + manual_tokens + selected_tags
    merged = _enforce_picker_mutexes_keep_last(_dedupe_casefold(merged))

    descriptor = keybed_prompts.join_prompt(merged + fx_tokens)
    if not descriptor:
        return (
            str(current_descriptor or ""),
            gr.update(value=str(current_root_note or "C")),
            gr.update(value=str(current_root_octave or "4")),
            "Choose an instrument or one or more timbre tags first.",
        )

    if bool(root_locked):
        root_name = str(current_root_note or "C")
        root_octave = str(current_root_octave or "4")
        root_note = _note_from_parts(root_name, root_octave)
        root_message = f"Preview root remains locked at `{root_note}`."
    else:
        suggested_root = _suggest_preview_root_from_picker(
            descriptor,
            instrument_1,
            instrument_2,
            seed_str,
        )
        root_name, root_octave = _split_note(suggested_root)
        root_message = f"Preview root updated from the selected prompt context to `{suggested_root}`."

    return (
        descriptor,
        gr.update(value=root_name),
        gr.update(value=root_octave),
        f"Applied prompt tags. {root_message}",
    )


def auto_sync_keybed_tag_picker_action(
    current_descriptor,
    instrument_1,
    instrument_2,
    timbre_tags,
    wetdry_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    """Immediately mirror user picker edits into the descriptor textbox."""
    descriptor, root_name, root_octave, _status = apply_keybed_tag_picker_action(
        current_descriptor,
        instrument_1,
        instrument_2,
        timbre_tags,
        wetdry_label,
        seed_str,
        root_locked,
        current_root_note,
        current_root_octave,
    )
    return descriptor, root_name, root_octave


def clear_keybed_tag_picker_action(
    current_descriptor,
    current_root_note,
    current_root_octave,
):
    """Clear picker-controlled tokens while preserving custom text and explicit FX."""
    body_tokens, fx_tokens = keybed_prompts.split_keybed_descriptor_tokens(current_descriptor)
    existing_instruments = _leading_instrument_tokens(body_tokens)
    remaining_body = body_tokens[len(existing_instruments):]
    manual_tokens = [
        token
        for token in remaining_body
        if str(token).strip().casefold() not in KEYBED_TAG_PICKER_TIMBRE_KEYS
    ]
    descriptor = keybed_prompts.join_prompt(_dedupe_casefold(manual_tokens + fx_tokens))
    return (
        gr.update(value=None),
        gr.update(value=KEYBED_NO_SECOND_INSTRUMENT),
        gr.update(value=[]),
        descriptor,
        gr.update(value=str(current_root_note or "C")),
        gr.update(value=str(current_root_octave or "4")),
    )


def _busy_button_updates(active: str):
    labels = {
        "preview": "Generating Preview...",
        "full": "Building Keybed...",
    }
    return tuple(
        gr.update(
            interactive=False,
            value=labels[key] if key == active else KEYBED_BUTTON_LABELS[key],
        )
        for key in ("preview", "full")
    )


def _ready_button_updates():
    return tuple(
        gr.update(interactive=True, value=KEYBED_BUTTON_LABELS[key])
        for key in ("preview", "full")
    )


def begin_keybed_preview_action():
    _keybed_cli_status("Building keybed preview...")
    return (*_busy_button_updates("preview"), "**Building Keybed Preview...**")


def begin_full_keybed_action(selected_formats):
    formats = _normalize_keybed_export_formats(selected_formats)
    format_label = _keybed_export_format_label(formats)
    _keybed_cli_status(f"Building full keybed source chunks for {format_label} export...")
    return (
        *_busy_button_updates("full"),
        f"**Building {format_label} Keybed...** Source chunks are being generated.",
    )


def _normalize_instrument_name(value, *, max_chars: int = KEYBED_INSTRUMENT_NAME_MAX_CHARS) -> str:
    """Normalize an optional user-facing instrument name and clamp its length."""
    text = " ".join(str(value or "").strip().split())[: int(max_chars)].strip()
    if not text or not re.search(r"[A-Za-z0-9]", text):
        return ""

    # Keep explicit acronyms such as FM/AI intact while presenting ordinary
    # words consistently in the UI and exported folder name.
    words = []
    for word in text.split(" "):
        words.append(word if word.isupper() else word[:1].upper() + word[1:].lower())
    return " ".join(words)


def _instrument_name_slug(value) -> str:
    display = _normalize_instrument_name(value)
    if not display:
        return ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", display)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:KEYBED_INSTRUMENT_NAME_MAX_CHARS].rstrip("_")


def _safe_filename_part(value: str, *, max_chars: int = 48) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("#", "sharp")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value[:max_chars].strip("_") or "keybed")


def _note_filename(note: str) -> str:
    return str(note).replace("#", "sharp")


def _get_runtime(get_runtime: Callable[[], Dict]) -> Dict:
    return get_runtime() or {}


def _runtime_model(runtime: Dict):
    return runtime.get("model")


def _runtime_sample_rate(runtime: Dict) -> int:
    return int(runtime.get("sample_rate") or 32000)


def _runtime_output_directory(runtime: Dict) -> str:
    return str(runtime.get("output_directory") or "generations")


def _runtime_model_name(runtime: Dict) -> str:
    return str(runtime.get("model_name") or "")


def _runtime_status_md(
    get_runtime: Callable[[], Dict],
    runtime_details: str = "",
    fallback_model_name: str = "",
) -> str:
    runtime = _get_runtime(get_runtime)
    model = _runtime_model(runtime)
    model_name = _runtime_model_name(runtime)

    runtime_line = ""
    details_model_name = ""
    for raw_line in str(runtime_details or "").splitlines():
        cleaned = raw_line.strip().replace("**", "")
        if cleaned.lower().startswith("runtime:"):
            runtime_line = raw_line.strip()
        elif cleaned.lower().startswith("model:"):
            details_model_name = cleaned.split(":", 1)[1].strip().strip("`")

    model_name = model_name or details_model_name or str(fallback_model_name or "").strip() or "n/a"

    if not runtime_line:
        if model is None:
            runtime_line = "**Runtime:** not ready"
        else:
            device = _model_device(model)
            try:
                dtype = next(model.parameters()).dtype
                dtype_name = str(dtype).replace("torch.", "")
            except Exception:
                dtype_name = "unknown"
            runtime_line = f"**Runtime:** `{device}` | dtype: `{dtype_name}`"

    return (
        f"**Loaded model:** `{model_name}`  \n"
        f"{runtime_line}  \n"
        "For consistent keybeds, use a pitch-synced / keybed-supported checkpoint."
    )


def _require_keybed_runtime(get_runtime: Callable[[], Dict]) -> Dict:
    runtime = _get_runtime(get_runtime)
    if _runtime_model(runtime) is None:
        raise gr.Error("No model is loaded yet. Load a checkpoint from this tab, Batch Generation, or the main Generation tab.")
    return runtime


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seconds_total_int(seconds_total: float) -> int:
    return int(math.ceil(float(seconds_total)))


def _sequence_actual_duration(note_count: int) -> float:
    note_count = max(1, int(note_count))
    if note_count <= 1:
        return float(KEYBED_SECONDS_PER_NOTE)
    return (note_count * float(KEYBED_SECONDS_PER_NOTE)) + ((note_count - 1) * float(KEYBED_SEQUENCE_GAP_SECONDS))


def _target_samples_for_keybed(model, sample_rate: int, seconds_total: float) -> int:
    seconds_int = _seconds_total_int(seconds_total)
    target_samples = int(seconds_int * int(sample_rate))
    min_input = getattr(model, "min_input_length", None)
    if isinstance(min_input, int) and min_input > 0 and (target_samples % min_input) != 0:
        target_samples += min_input - (target_samples % min_input)
    return int(target_samples)


def _apply_short_fade(audio: torch.Tensor, sample_rate: int, fade_ms: float) -> torch.Tensor:
    fade_len = int(round((float(fade_ms) / 1000.0) * int(sample_rate)))
    if fade_len <= 1 or audio.shape[-1] <= 1:
        return audio
    fade_len = min(fade_len, int(audio.shape[-1]))
    ramp_in = torch.linspace(0.0, 1.0, steps=fade_len, device=audio.device, dtype=audio.dtype)
    ramp_out = torch.linspace(1.0, 0.0, steps=fade_len, device=audio.device, dtype=audio.dtype)
    audio = audio.clone()
    audio[:, :fade_len] *= ramp_in
    audio[:, -fade_len:] *= ramp_out
    return audio


def _tensor_to_i16_cpu(audio: torch.Tensor) -> torch.Tensor:
    return audio.to(torch.float32).clamp(-1, 1).mul(32767.0).to(torch.int16).cpu()


def _stable_int_seed(*parts: object) -> int:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8", errors="replace")).hexdigest()
    return int(h[:8], 16)


def _resolve_seed(seed_str) -> int:
    return keybed_prompts.resolve_keybed_seed(seed_str)


def _optional_int_seed(value) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return None
        return int(text)
    except Exception:
        return None


def _normalize_descriptor_for_seed_reuse(descriptor: str) -> str:
    return re.sub(r"\s+", " ", str(descriptor or "").strip())


def _note_from_parts(note_name: str, octave: str | int) -> str:
    name = str(note_name or "C").strip().upper()
    if name not in KEYBED_ROOT_NOTE_NAMES:
        name = "C"
    octave = str(octave if octave not in (None, "") else "4")
    if octave not in KEYBED_ROOT_OCTAVES:
        octave = "4"
    return f"{name}{octave}"


def _split_note(note: str) -> Tuple[str, str]:
    match = re.match(r"^([A-G](?:#)?)(-?\d+)$", str(note or "C4").strip(), re.IGNORECASE)
    if not match:
        return "C", "4"
    name = match.group(1).upper()
    octave = match.group(2)
    if name not in KEYBED_ROOT_NOTE_NAMES:
        name = "C"
    if octave not in KEYBED_ROOT_OCTAVES:
        octave = "4"
    return name, octave


def _note_count_for_octaves(number_of_octaves: str | float) -> int:
    value = str(number_of_octaves or "0.5").strip()
    if value == "2":
        return 24
    if value == "1":
        return 12
    return 6


def _clamp_preview_root(note_name: str, octave: str | int, number_of_octaves: str | float) -> Tuple[str, bool]:
    requested = _note_from_parts(note_name, octave)
    requested_midi = keybed_prompts.keybed_note_to_midi(requested)
    lo = keybed_prompts.keybed_note_to_midi(KEYBED_USER_ROOT_MIN)
    user_hi = keybed_prompts.keybed_note_to_midi(KEYBED_USER_ROOT_MAX)
    sequence_hi = keybed_prompts.keybed_note_to_midi(KEYBED_SEQUENCE_MAX_NOTE)
    note_count = _note_count_for_octaves(number_of_octaves)
    max_start = min(user_hi, sequence_hi - note_count + 1)

    clamped = max(lo, min(int(requested_midi), int(max_start)))
    return keybed_prompts.midi_to_note_name(clamped), clamped != requested_midi


def _chromatic_notes_from_root(root_note: str, note_count: int) -> List[str]:
    start = keybed_prompts.keybed_note_to_midi(root_note)
    return [keybed_prompts.midi_to_note_name(start + idx) for idx in range(int(note_count))]


def _chromatic_notes_between(start_note: str, end_note: str) -> List[str]:
    start = keybed_prompts.keybed_note_to_midi(start_note)
    end = keybed_prompts.keybed_note_to_midi(end_note)
    if end < start:
        start, end = end, start
    return [keybed_prompts.midi_to_note_name(midi) for midi in range(start, end + 1)]


def _resolve_full_sampler_range(range_label: str | None) -> Tuple[str, str, str, List[str], List[List[str]]]:
    label = str(range_label or KEYBED_FULL_SAMPLER_DEFAULT_RANGE).strip()
    if label not in KEYBED_FULL_SAMPLER_RANGES:
        label = KEYBED_FULL_SAMPLER_DEFAULT_RANGE
    start_note, end_note = KEYBED_FULL_SAMPLER_RANGES[label]
    notes = _chromatic_notes_between(start_note, end_note)
    chunks = _chunk_notes(notes, KEYBED_CHUNK_SIZE)
    range_key = _safe_filename_part(f"{start_note}_to_{end_note}", max_chars=32)
    return label, start_note, end_note, notes, chunks


def _chunk_notes(notes: List[str], chunk_size: int = KEYBED_CHUNK_SIZE) -> List[List[str]]:
    chunks: List[List[str]] = []
    for idx in range(0, len(notes), int(chunk_size)):
        chunk = notes[idx:idx + int(chunk_size)]
        if len(chunk) >= 2:
            chunks.append(chunk)
    return chunks


def _descriptor_slug_from_keybed_descriptor(descriptor: str, *, max_chars: int = 48) -> str:
    """Compact family/subfamily-ish slug for KEYBED run folders.

    Uses the canonical keybed prompt cleaner when available so pasted full
    prompts do not leak Target Note / Note Sequence / FX grammar into folder
    names. This mirrors the short Loops / One_Shots naming style.
    """
    body_tokens: List[str] = []
    try:
        body_tokens, _fx_tokens = keybed_prompts.split_keybed_descriptor_tokens(descriptor)
    except Exception:
        for part in str(descriptor or "").split(","):
            token = part.strip()
            if not token:
                continue
            low = token.lower()
            if low in {"keybed", "sequence", "timbre profile", "target note", "target position", "chromatic chunk", "note sequence", "wet", "dry"}:
                continue
            if re.match(r"^[A-G](?:#|b)?-?\d+$", token, re.IGNORECASE):
                continue
            if any(word in low for word in ("reverb", "delay", "distortion", "phaser", "bitcrush")):
                continue
            body_tokens.append(token)

    return _safe_filename_part("_".join(body_tokens[:2]) or "keybed", max_chars=max_chars)


def _make_run_dir(
    output_root: str,
    descriptor: str,
    *,
    root_note: str | None = None,
    seed: int | None = None,
    instrument_name: str | None = None,
    output_subdir: str = KEYBED_OUTPUT_SUBDIR,
) -> str:
    slug = _instrument_name_slug(instrument_name) or _descriptor_slug_from_keybed_descriptor(descriptor)
    digest = hashlib.sha1(
        f"{time.time_ns()}|{seed}|{root_note}|{descriptor}|{_normalize_instrument_name(instrument_name)}".encode("utf-8", errors="replace")
    ).hexdigest()[:8]
    base_dir = os.path.join(output_root, str(output_subdir), f"{slug}_{digest}")
    run_dir = base_dir
    counter = 1
    while os.path.exists(run_dir):
        counter += 1
        run_dir = f"{base_dir}_{counter:02d}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _note_slices_for_sequence(notes: List[str]) -> List[Dict[str, object]]:
    """Timing hints for future SFZ/DecentSampler slicing.

    Generated keybed chunks are sequential note renders: each note occupies
    KEYBED_SECONDS_PER_NOTE and adjacent notes are separated by
    KEYBED_SEQUENCE_GAP_SECONDS. Keeping this in JSONL makes later cutting
    deterministic without parsing filenames.
    """
    out: List[Dict[str, object]] = []
    cursor = 0.0
    for note in notes:
        start = cursor
        end = start + float(KEYBED_SECONDS_PER_NOTE)
        out.append({
            "note": str(note),
            "start_sec": round(start, 6),
            "end_sec": round(end, 6),
        })
        cursor = end + float(KEYBED_SEQUENCE_GAP_SECONDS)
    return out


def _write_keybed_run_metadata(
    run_dir: str,
    *,
    descriptor: str,
    root_note: str,
    number_of_octaves: str,
    all_notes: List[str],
    wetdry: str,
    resolved_seed: int,
    model_name: str,
    sample_rate: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    cfg_rescale: float,
    manifest_rows: List[Dict[str, object]],
) -> Tuple[str, str]:
    metadata_path = os.path.join(run_dir, "metadata.txt")
    jsonl_path = os.path.join(run_dir, "manifest.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines: List[str] = [
        "type: keybed_preview",
        f"model: {model_name or 'n/a'}",
        f"descriptor: {descriptor}",
        f"root_note: {root_note}",
        f"number_of_octaves: {number_of_octaves}",
        f"notes: {', '.join(all_notes)}",
        f"wetdry: {wetdry}",
        f"base_seed: {resolved_seed}",
        f"sample_rate: {sample_rate}",
        "",
        f"steps: {steps}",
        f"cfg_scale: {cfg_scale}",
        f"sampler_type: {sampler_type}",
        f"sigma_min: {sigma_min}",
        f"sigma_max: {sigma_max}",
        f"cfg_rescale: {cfg_rescale}",
        "",
        f"note_seconds: {KEYBED_SECONDS_PER_NOTE}",
        f"gap_seconds: {KEYBED_SEQUENCE_GAP_SECONDS}",
        "",
        "chunk_prompts:",
    ]

    for row in manifest_rows:
        if row.get("role") != "chunk":
            continue
        lines.append(f"- {row.get('name')}: seed={row.get('seed')} notes={', '.join(row.get('notes') or [])}")
        lines.append(f"  prompt: {row.get('prompt')}")
        lines.append(f"  audio_path: {row.get('audio_path')}")

    lines.extend([
        "",
        f"manifest_jsonl: {jsonl_path}",
    ])

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return metadata_path, jsonl_path


def _wetdry_from_label(wetdry_label: str | None) -> str:
    return "Wet" if str(wetdry_label or "Dry").strip().lower() == "wet" else "Dry"


def _mode_from_label(mode_label: str | None) -> str:
    return "experimental" if str(mode_label or "Simple").strip().lower() == "experimental" else "standard"


def _instrument_mode_from_label(instrument_mode_label: str | None) -> str:
    return (
        "single"
        if str(instrument_mode_label or "Hybrid").strip().lower() == "single"
        else "hybrid"
    )


def _single_primary_instrument(instrument_1, instrument_2, descriptor="") -> Optional[str]:
    """Choose the best one-token display instrument when switching to Single."""
    first = str(instrument_1 or "").strip()
    second = _normalize_optional_instrument(instrument_2)

    if not first:
        body_tokens, _fx_tokens = keybed_prompts.split_keybed_descriptor_tokens(descriptor)
        existing = _leading_instrument_tokens(body_tokens)
        first = existing[0] if existing else ""
        second = second or (existing[1] if len(existing) > 1 else None)

    if second:
        first_family, first_subfamily = keybed_prompts.keybed_instrument_context(first)
        second_family, second_subfamily = keybed_prompts.keybed_instrument_context(second)
        # Simple Hybrid commonly presents Family + Subfamily. Single should show
        # the more useful specific token (Grand Piano rather than Keys).
        if (
            first_family
            and not first_subfamily
            and second_family == first_family
            and second_subfamily
        ):
            return second

    return first or second


def set_keybed_instrument_mode_action(
    instrument_mode_label,
    current_descriptor,
    instrument_1,
    instrument_2,
    timbre_tags,
    wetdry_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    """Apply Single/Hybrid to the manual builder without hiding Instrument 2."""
    instrument_mode = _instrument_mode_from_label(instrument_mode_label)

    if instrument_mode == "hybrid":
        return (
            str(current_descriptor or ""),
            gr.update(value=instrument_1),
            gr.update(
                value=_normalize_optional_instrument(instrument_2) or KEYBED_NO_SECOND_INSTRUMENT,
                interactive=True,
            ),
            gr.update(value=str(current_root_note or "C")),
            gr.update(value=str(current_root_octave or "4")),
        )

    primary = _single_primary_instrument(instrument_1, instrument_2, current_descriptor)
    if not primary:
        return (
            str(current_descriptor or ""),
            gr.update(value=instrument_1),
            gr.update(value=KEYBED_NO_SECOND_INSTRUMENT, interactive=False),
            gr.update(value=str(current_root_note or "C")),
            gr.update(value=str(current_root_octave or "4")),
        )

    descriptor, root_name, root_octave, _status = apply_keybed_tag_picker_action(
        current_descriptor,
        primary,
        KEYBED_NO_SECOND_INSTRUMENT,
        timbre_tags,
        wetdry_label,
        seed_str,
        root_locked,
        current_root_note,
        current_root_octave,
    )
    return (
        descriptor,
        gr.update(value=primary),
        gr.update(value=KEYBED_NO_SECOND_INSTRUMENT, interactive=False),
        root_name,
        root_octave,
    )


def _prompt_seed_from_text(seed_str) -> int:
    # Keep Random Prompt behavior aligned with Batch Generation:
    # -1 rolls fresh; any explicit seed gives repeatable prompt + preview root.
    return keybed_prompts.resolve_keybed_seed(seed_str)


def random_keybed_descriptor_action(
    mode_label,
    wetdry_label,
    instrument_mode_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    prompt_seed = _prompt_seed_from_text(seed_str)
    mode = _mode_from_label(mode_label)
    wetdry = _wetdry_from_label(wetdry_label)
    instrument_mode = _instrument_mode_from_label(instrument_mode_label)

    try:
        plan = keybed_prompts.prompt_generator_keybed_descriptor(
            seed=prompt_seed,
            mode=mode,
            instrument_mode=instrument_mode,
            wetdry=wetdry,
            # Build with Wet/Dry enabled so Wet random prompts visibly include FX,
            # then hide the Wet/Dry control token from the textbox.
            include_wetdry=True,
            return_plan=True,
        )
    except TypeError:
        # Backwards-compatible fallback if an older prompt builder is imported.
        plan = keybed_prompts.prompt_generator_keybed_descriptor(
            seed=prompt_seed,
            mode=mode,
            wetdry=wetdry,
            include_wetdry=True,
        )

    if isinstance(plan, dict):
        descriptor = _strip_visible_wetdry_tokens(plan.get("prompt", ""))
        family = str(plan.get("family") or "").strip() or None
        subfamily = str(plan.get("subfamily") or "").strip() or None
        planned_instrument_1 = str(plan.get("instrument_1") or family or "").strip() or None
        planned_instrument_2 = str(plan.get("instrument_2") or subfamily or "").strip() or None
        tags = list(plan.get("tags") or [])
        suggested_root = plan.get("preview_root") or keybed_prompts.default_preview_root_for_descriptor(
            descriptor,
            seed=prompt_seed,
        )
    else:
        descriptor = _strip_visible_wetdry_tokens(plan)
        family, subfamily = keybed_prompts.infer_keybed_family_and_subfamily_from_descriptor(descriptor)
        body_tokens, _fx_tokens = keybed_prompts.split_keybed_descriptor_tokens(descriptor)
        existing_instruments = _leading_instrument_tokens(body_tokens)
        family = existing_instruments[0] if existing_instruments else family
        subfamily = existing_instruments[1] if len(existing_instruments) > 1 else subfamily
        tags = [
            token
            for token in body_tokens[len(existing_instruments):]
            if str(token).casefold() in KEYBED_TAG_PICKER_TIMBRE_KEYS
        ]
        suggested_root = keybed_prompts.default_preview_root_for_descriptor(
            descriptor,
            seed=prompt_seed,
        )

    if not isinstance(plan, dict):
        planned_instrument_1 = family
        planned_instrument_2 = subfamily

    instruments = [
        value
        for value in _dedupe_casefold([planned_instrument_1, planned_instrument_2])
        if value.casefold() in KEYBED_TAG_PICKER_INSTRUMENT_KEYS
    ][:2]
    instrument_1 = instruments[0] if instruments else None
    instrument_2 = instruments[1] if len(instruments) > 1 else None

    if instrument_mode == "single":
        primary = _single_primary_instrument(instrument_1, instrument_2, descriptor)
        if primary:
            body_tokens, fx_tokens = keybed_prompts.split_keybed_descriptor_tokens(descriptor)
            existing = _leading_instrument_tokens(body_tokens)
            remaining_body = body_tokens[len(existing):]
            descriptor = keybed_prompts.join_prompt(
                _dedupe_casefold([primary] + remaining_body + fx_tokens)
            )
            instrument_1 = primary
        instrument_2 = None

    # White Noise is intentionally not an instrument choice; keep it represented
    # in the timbre picker when the experimental random builder selects it.
    if family and family.casefold() in KEYBED_TAG_PICKER_TIMBRE_KEYS and not instrument_1:
        tags = [family] + tags
    tags = [
        KEYBED_TAG_PICKER_TIMBRE_KEYS[str(tag).casefold()]
        for tag in _dedupe_casefold(tags)
        if str(tag).casefold() in KEYBED_TAG_PICKER_TIMBRE_KEYS
    ]

    instrument_1_update = gr.update(value=instrument_1)
    instrument_2_update = gr.update(
        value=instrument_2 or KEYBED_NO_SECOND_INSTRUMENT,
        interactive=(instrument_mode == "hybrid"),
    )
    tags_update = gr.update(value=tags)

    if bool(root_locked):
        return (
            descriptor,
            gr.update(value=str(current_root_note or "C")),
            gr.update(value=str(current_root_octave or "4")),
            instrument_1_update,
            instrument_2_update,
            tags_update,
        )

    root_name, root_octave = _split_note(suggested_root)
    return (
        descriptor,
        gr.update(value=root_name),
        gr.update(value=root_octave),
        instrument_1_update,
        instrument_2_update,
        tags_update,
    )


def _generate_keybed_sequence_tensor(
    *,
    model,
    sample_rate: int,
    prompt: str,
    seed: int,
    note_count: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    cfg_rescale: float,
) -> torch.Tensor:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    device = _model_device(model)
    actual_duration = _sequence_actual_duration(note_count)
    seconds_total = _seconds_total_int(actual_duration)
    input_sample_size = _target_samples_for_keybed(model, sample_rate, seconds_total)

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0.0,
        "seconds_total": float(seconds_total),
    }]

    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=None,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        batch_size=1,
        sample_size=int(input_sample_size),
        sample_rate=int(sample_rate),
        seed=int(seed),
        device=device,
        sampler_type=str(sampler_type),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        init_audio=None,
        init_noise_level=1.0,
        mask_args=None,
        callback=None,
        scale_phi=float(cfg_rescale),
    )

    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).clamp(-1, 1)
    clip_samples = int(round(actual_duration * int(sample_rate)))
    audio = audio[:, :max(1, min(int(audio.shape[-1]), clip_samples))].contiguous()
    return _apply_short_fade(audio, sample_rate, KEYBED_PREVIEW_FADE_MS)


def _concat_audio(parts: List[torch.Tensor], sample_rate: int) -> torch.Tensor:
    if not parts:
        return torch.zeros((2, 1), dtype=torch.float32)
    gap_len = int(round((KEYBED_PREVIEW_GAP_MS / 1000.0) * sample_rate))
    out: List[torch.Tensor] = []
    for part in parts:
        out.append(part.cpu())
        if gap_len > 0:
            out.append(torch.zeros((part.shape[0], gap_len), dtype=torch.float32))
    if gap_len > 0 and out:
        out = out[:-1]
    return _apply_short_fade(torch.cat(out, dim=-1), sample_rate, KEYBED_PREVIEW_FADE_MS)


def generate_keybed_preview_action(
    descriptor,
    root_note_name,
    root_octave,
    number_of_octaves,
    wetdry_label,
    seed_str,
    steps,
    cfg_scale,
    sampler_type,
    sigma_min,
    sigma_max,
    cfg_rescale,
    get_runtime: Callable[[], Dict],
):
    if not descriptor or not str(descriptor).strip():
        raise gr.Error("Add a Keybed descriptor or use Random Keybed Prompt first.")

    runtime = _require_keybed_runtime(get_runtime)
    model = _runtime_model(runtime)
    sample_rate = _runtime_sample_rate(runtime)
    model_name = _runtime_model_name(runtime)
    output_root = _runtime_output_directory(runtime)

    resolved_seed = _resolve_seed(seed_str)
    wetdry = _wetdry_from_label(wetdry_label)
    root_note, adjusted = _clamp_preview_root(root_note_name, root_octave, number_of_octaves)
    root_name_out, root_octave_out = _split_note(root_note)

    note_count = _note_count_for_octaves(number_of_octaves)
    all_notes = _chromatic_notes_from_root(root_note, note_count)
    chunks = _chunk_notes(all_notes, KEYBED_CHUNK_SIZE)

    run_dir = _make_run_dir(
        output_root,
        descriptor,
        root_note=root_note,
        seed=resolved_seed,
        output_subdir=KEYBED_PREVIEW_OUTPUT_SUBDIR,
    )
    chunk_dir = os.path.join(run_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    run_id = os.path.basename(run_dir)
    _keybed_cli_status(
        f"Preview run {run_id}: generating {len(chunks)} chunk(s) for {all_notes[0]}-{all_notes[-1]}."
    )

    chunk_paths: List[str] = []
    chunk_audio: List[torch.Tensor] = []
    chunk_labels: List[str] = []
    manifest_rows: List[Dict[str, object]] = []

    for idx, chunk in enumerate(chunks, start=1):
        _keybed_cli_status(
            f"Preview run {run_id}: chunk {idx}/{len(chunks)} ({chunk[0]}-{chunk[-1]})."
        )
        prompt = keybed_prompts.build_keybed_note_sequence_prompt(
            descriptor,
            chunk,
            wetdry=wetdry,
            seed=resolved_seed,
            chromatic_chunk=True,
        )
        # Use the resolved preview seed directly for every preview chunk.
        # If the UI seed is -1, _resolve_seed() rolls one random seed for this
        # preview run; that same seed is then captured for full sampler export.
        seed = int(resolved_seed)
        audio = _generate_keybed_sequence_tensor(
            model=model,
            sample_rate=sample_rate,
            prompt=prompt,
            seed=seed,
            note_count=len(chunk),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
        )
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:02d}_{_note_filename(chunk[0])}_to_{_note_filename(chunk[-1])}.wav")
        torchaudio.save(chunk_path, _tensor_to_i16_cpu(audio), sample_rate)
        chunk_paths.append(chunk_path)
        chunk_audio.append(audio.cpu())
        chunk_labels.append(f"{chunk[0]}-{chunk[-1]}")
        manifest_rows.append({
            "type": "keybed_preview",
            "role": "chunk",
            "name": f"chunk_{idx:02d}",
            "descriptor": str(descriptor),
            "root_note": root_note,
            "number_of_octaves": str(number_of_octaves),
            "notes": list(chunk),
            "note_slices": _note_slices_for_sequence(list(chunk)),
            "wetdry": wetdry,
            "seed": int(seed),
            "base_seed": int(resolved_seed),
            "seed_mode": "same_seed_all_preview_chunks",
            "prompt": prompt,
            "audio_path": chunk_path,
            "sample_rate": int(sample_rate),
            "actual_duration_sec": _sequence_actual_duration(len(chunk)),
            "seconds_total": _seconds_total_int(_sequence_actual_duration(len(chunk))),
            "note_seconds": KEYBED_SECONDS_PER_NOTE,
            "gap_seconds": KEYBED_SEQUENCE_GAP_SECONDS,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "sampler_type": str(sampler_type),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "cfg_rescale": float(cfg_rescale),
            "model": model_name,
            "export_hint": "slice_by_note_slices",
        })

    preview_paths: List[str] = []

    if note_count <= 12:
        preview_audio = _concat_audio(chunk_audio, sample_rate)
        preview_path = os.path.join(run_dir, f"preview_{_note_filename(all_notes[0])}_to_{_note_filename(all_notes[-1])}.wav")
        torchaudio.save(preview_path, _tensor_to_i16_cpu(preview_audio), sample_rate)
        preview_paths.append(preview_path)
        manifest_rows.append({
            "type": "keybed_preview",
            "role": "preview",
            "name": "preview_full",
            "descriptor": str(descriptor),
            "root_note": root_note,
            "number_of_octaves": str(number_of_octaves),
            "notes": list(all_notes),
            "source_chunks": list(chunk_paths),
            "audio_path": preview_path,
            "sample_rate": int(sample_rate),
            "model": model_name,
            "export_hint": "audition_only",
        })
        audio_1 = gr.update(
            value=preview_path,
            label=(f"Half-Octave Preview — {all_notes[0]} to {all_notes[-1]}" if note_count == 6 else f"1-Octave Preview — {all_notes[0]} to {all_notes[-1]}"),
            visible=True,
        )
        audio_2 = gr.update(value=None, visible=False)
    else:
        octave_1_audio = _concat_audio(chunk_audio[:2], sample_rate)
        octave_2_audio = _concat_audio(chunk_audio[2:4], sample_rate)
        octave_1_notes = all_notes[:12]
        octave_2_notes = all_notes[12:24]
        octave_1_path = os.path.join(run_dir, f"preview_octave_1_{_note_filename(octave_1_notes[0])}_to_{_note_filename(octave_1_notes[-1])}.wav")
        octave_2_path = os.path.join(run_dir, f"preview_octave_2_{_note_filename(octave_2_notes[0])}_to_{_note_filename(octave_2_notes[-1])}.wav")
        torchaudio.save(octave_1_path, _tensor_to_i16_cpu(octave_1_audio), sample_rate)
        torchaudio.save(octave_2_path, _tensor_to_i16_cpu(octave_2_audio), sample_rate)
        preview_paths.extend([octave_1_path, octave_2_path])
        manifest_rows.append({
            "type": "keybed_preview",
            "role": "preview",
            "name": "preview_octave_1",
            "descriptor": str(descriptor),
            "root_note": root_note,
            "number_of_octaves": str(number_of_octaves),
            "notes": list(octave_1_notes),
            "source_chunks": list(chunk_paths[:2]),
            "audio_path": octave_1_path,
            "sample_rate": int(sample_rate),
            "model": model_name,
            "export_hint": "audition_only",
        })
        manifest_rows.append({
            "type": "keybed_preview",
            "role": "preview",
            "name": "preview_octave_2",
            "descriptor": str(descriptor),
            "root_note": root_note,
            "number_of_octaves": str(number_of_octaves),
            "notes": list(octave_2_notes),
            "source_chunks": list(chunk_paths[2:4]),
            "audio_path": octave_2_path,
            "sample_rate": int(sample_rate),
            "model": model_name,
            "export_hint": "audition_only",
        })
        audio_1 = gr.update(value=octave_1_path, label=f"Octave 1 — {octave_1_notes[0]} to {octave_1_notes[-1]}", visible=True)
        audio_2 = gr.update(value=octave_2_path, label=f"Octave 2 — {octave_2_notes[0]} to {octave_2_notes[-1]}", visible=True)

    metadata_path, jsonl_path = _write_keybed_run_metadata(
        run_dir,
        descriptor=str(descriptor),
        root_note=root_note,
        number_of_octaves=str(number_of_octaves),
        all_notes=list(all_notes),
        wetdry=wetdry,
        resolved_seed=int(resolved_seed),
        model_name=model_name,
        sample_rate=int(sample_rate),
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        sampler_type=str(sampler_type),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        cfg_rescale=float(cfg_rescale),
        manifest_rows=manifest_rows,
    )
    preview_id = _generation_id_from_path(run_dir)
    status = (
        f"**Preview ready** — `{all_notes[0]}` to `{all_notes[-1]}` "
        f"(`{len(all_notes)}` notes) · ID `{preview_id}` · Seed `{resolved_seed}`"
    )
    if adjusted:
        status += f"  \nPreview root was adjusted to `{root_note}` to stay inside the trained range."

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    _keybed_cli_status(f"Preview complete - ID {run_id}.")

    return (
        audio_1,
        audio_2,
        gr.update(value=root_name_out),
        gr.update(value=root_octave_out),
        status,
        run_dir,
        jsonl_path,
        int(resolved_seed),
        _normalize_descriptor_for_seed_reuse(descriptor),
    )


def generate_full_keybed_sampler_action(
    descriptor,
    full_sampler_range_label,
    export_formats,
    wetdry_label,
    seed_str,
    latest_preview_seed,
    latest_preview_descriptor,
    steps,
    cfg_scale,
    sampler_type,
    sigma_min,
    sigma_max,
    cfg_rescale,
    instrument_name,
    get_runtime: Callable[[], Dict],
):
    """Generate a deterministic fixed-range keybed and export selected sampler formats.

    This is intentionally separate from Generate Keybed Preview:
      - Preview uses the context-aware root picker for quick auditioning.
      - Full sampler generation ignores preview root and uses a fixed keyboard
        range so exports are predictable and easy to test.
      - If a matching preview exists, its resolved seed is reused for every
        full-sampler chunk; otherwise the seed box is resolved once.
    """
    if not descriptor or not str(descriptor).strip():
        raise gr.Error("Add a Keybed descriptor or use Random Keybed Prompt first.")

    normalized_export_formats = _normalize_keybed_export_formats(export_formats)
    export_format_label = _keybed_export_format_label(normalized_export_formats)

    runtime = _require_keybed_runtime(get_runtime)
    model = _runtime_model(runtime)
    sample_rate = _runtime_sample_rate(runtime)
    model_name = _runtime_model_name(runtime)
    output_root = _runtime_output_directory(runtime)

    descriptor = str(descriptor).strip()
    instrument_name = _normalize_instrument_name(instrument_name)
    preview_seed = _optional_int_seed(latest_preview_seed)
    preview_descriptor = _normalize_descriptor_for_seed_reuse(latest_preview_descriptor)
    current_descriptor = _normalize_descriptor_for_seed_reuse(descriptor)
    if preview_seed is not None and preview_descriptor == current_descriptor:
        resolved_seed = int(preview_seed)
        seed_source = "captured_preview_seed"
    else:
        resolved_seed = _resolve_seed(seed_str)
        seed_source = "new_seed_from_seed_control"
    wetdry = _wetdry_from_label(wetdry_label)
    range_label, start_note, end_note, all_notes, chunks = _resolve_full_sampler_range(full_sampler_range_label)
    range_key = _safe_filename_part(f"{start_note}_to_{end_note}", max_chars=32)

    if not chunks:
        raise gr.Error(f"Could not build any keybed chunks for range {start_note} to {end_note}.")

    run_dir = _make_run_dir(
        output_root,
        descriptor,
        root_note=f"{start_note}_to_{end_note}",
        seed=resolved_seed,
        instrument_name=instrument_name,
    )
    chunk_dir = os.path.join(run_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    run_id = os.path.basename(run_dir)
    _keybed_cli_status(
        f"Full run {run_id}: generating {len(chunks)} chunk(s) for {start_note}-{end_note}."
    )

    chunk_paths: List[str] = []
    manifest_rows: List[Dict[str, object]] = []

    for idx, chunk in enumerate(chunks, start=1):
        _keybed_cli_status(
            f"Full run {run_id}: chunk {idx}/{len(chunks)} ({chunk[0]}-{chunk[-1]})."
        )
        prompt = keybed_prompts.build_keybed_note_sequence_prompt(
            descriptor,
            chunk,
            wetdry=wetdry,
            seed=resolved_seed,
            chromatic_chunk=True,
        )
        # Full sampler export prioritizes timbre continuity: every chunk uses
        # the same resolved seed. If a matching preview was generated first,
        # this is the preview's captured seed; otherwise -1 rolls once here.
        seed = int(resolved_seed)
        audio = _generate_keybed_sequence_tensor(
            model=model,
            sample_rate=sample_rate,
            prompt=prompt,
            seed=seed,
            note_count=len(chunk),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
        )
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:02d}_{_note_filename(chunk[0])}_to_{_note_filename(chunk[-1])}.wav")
        torchaudio.save(chunk_path, _tensor_to_i16_cpu(audio), sample_rate)
        chunk_paths.append(chunk_path)

        manifest_rows.append({
            "type": "keybed_full_sampler",
            "role": "chunk",
            "name": f"chunk_{idx:02d}",
            "descriptor": str(descriptor),
            "root_note": start_note,
            "end_note": end_note,
            "range_label": range_label,
            "number_of_octaves": range_label,
            "notes": list(chunk),
            "note_slices": _note_slices_for_sequence(list(chunk)),
            "wetdry": wetdry,
            "seed": int(seed),
            "base_seed": int(resolved_seed),
            "seed_mode": "same_seed_all_full_sampler_chunks",
            "seed_source": seed_source,
            "prompt": prompt,
            "audio_path": chunk_path,
            "sample_rate": int(sample_rate),
            "actual_duration_sec": _sequence_actual_duration(len(chunk)),
            "seconds_total": _seconds_total_int(_sequence_actual_duration(len(chunk))),
            "note_seconds": KEYBED_SECONDS_PER_NOTE,
            "gap_seconds": KEYBED_SEQUENCE_GAP_SECONDS,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "sampler_type": str(sampler_type),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "cfg_rescale": float(cfg_rescale),
            "model": model_name,
            "export_hint": "slice_by_note_slices",
        })

    metadata_path, jsonl_path = _write_keybed_run_metadata(
        run_dir,
        descriptor=str(descriptor),
        root_note=start_note,
        number_of_octaves=range_label,
        all_notes=list(all_notes),
        wetdry=wetdry,
        resolved_seed=int(resolved_seed),
        model_name=model_name,
        sample_rate=int(sample_rate),
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        sampler_type=str(sampler_type),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        cfg_rescale=float(cfg_rescale),
        manifest_rows=manifest_rows,
    )

    _keybed_cli_status(
        f"Full run {run_id}: source generation complete; starting {export_format_label} export."
    )
    export_result = export_keybed_run(
        run_dir,
        formats=normalized_export_formats,
        trim_tail=True,
        instrument_name=instrument_name,
    )

    display_name = str(export_result.get("display_name") or "Foundation-1 Keybed")
    instrument_id = str(export_result.get("instrument_id") or _generation_id_from_path(run_dir))
    export_dir = str(export_result.get("export_dir") or "")
    completion_label = _keybed_export_completion_label(normalized_export_formats)

    export_dirs_by_format = dict(export_result.get("export_dirs_by_format") or {})
    if len(export_dirs_by_format) > 1:
        saved_to = "  \n".join(
            f"{('DecentSampler' if fmt == 'dspreset' else 'SFZ')}: `{path}`"
            for fmt, path in export_dirs_by_format.items()
        )
        saved_text = f"Saved to separate sampler folders:  \n{saved_to}"
    else:
        saved_text = f"Saved to: `{export_dir}`"

    status = (
        f"**{completion_label}** — {display_name} · ID `{instrument_id}`  \n"
        f"Range: `{all_notes[0]}` to `{all_notes[-1]}` (`{len(all_notes)}` notes)  \n"
        f"{saved_text}"
    )

    instrument_files: List[str] = []
    if "dspreset" in normalized_export_formats and export_result.get("dspreset_path"):
        instrument_files.append(str(export_result["dspreset_path"]))
    if "sfz" in normalized_export_formats and export_result.get("sfz_path"):
        instrument_files.append(str(export_result["sfz_path"]))

    sampler_file_update = gr.update(
        value=instrument_files,
        label=f"Exported {export_format_label} Instrument" + ("s" if len(instrument_files) > 1 else ""),
        visible=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    _keybed_cli_status(
        f"Keybed Exported - ID {export_result.get('instrument_id') or run_id} "
        f"({export_result.get('preset_id') or run_id})."
    )

    return (
        status,
        run_dir,
        jsonl_path,
        int(resolved_seed),
        _normalize_descriptor_for_seed_reuse(descriptor),
        sampler_file_update,
        export_dir,
        gr.update(visible=True, interactive=True),
    )


def _refresh_runtime_action(
    get_runtime: Callable[[], Dict],
    runtime_details: str = "",
):
    return _runtime_status_md(get_runtime, runtime_details)


def _cleanup_after_keybed_action() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def safe_generate_keybed_preview_action(*args, get_runtime: Callable[[], Dict]):
    """Always restore the UI controls, including when inference raises."""
    if len(args) < 4:
        raise ValueError("Preview wrapper did not receive the expected state inputs.")
    core_args = args[:-4]
    previous_run_dir, previous_manifest, previous_seed, previous_descriptor = args[-4:]
    try:
        result = generate_keybed_preview_action(*core_args, get_runtime=get_runtime)
        return (*result, *_ready_button_updates())
    except Exception as exc:
        traceback.print_exc()
        _keybed_cli_status(f"Preview failed: {type(exc).__name__}: {exc}")
        error_status = (
            f"**Keybed preview failed:** `{type(exc).__name__}: {exc}`  \n"
            "The controls have been reset; you can adjust the prompt or settings and try again."
        )
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            error_status,
            previous_run_dir,
            previous_manifest,
            previous_seed,
            previous_descriptor,
            *_ready_button_updates(),
        )
    finally:
        _cleanup_after_keybed_action()


def safe_generate_full_keybed_sampler_action(*args, get_runtime: Callable[[], Dict]):
    """Run the long full-keybed path without leaving the Gradio controls stuck."""
    if len(args) < 3:
        raise ValueError("Full sampler wrapper did not receive the expected state inputs.")
    core_args = args[:-3]
    previous_run_dir, previous_manifest, previous_export_dir = args[-3:]
    try:
        result = generate_full_keybed_sampler_action(*core_args, get_runtime=get_runtime)
        return (*result, *_ready_button_updates())
    except Exception as exc:
        traceback.print_exc()
        _keybed_cli_status(f"Full keybed failed: {type(exc).__name__}: {exc}")
        error_status = (
            f"**Full keybed failed:** `{type(exc).__name__}: {exc}`  \n"
            "The controls have been reset. Existing preview/export state was preserved."
        )
        return (
            error_status,
            previous_run_dir,
            previous_manifest,
            core_args[5],  # latest_preview_seed
            core_args[6],  # latest_preview_descriptor
            gr.update(),
            previous_export_dir,
            gr.update(),
            *_ready_button_updates(),
        )
    finally:
        _cleanup_after_keybed_action()


def build_keybed_tab(
    get_runtime: Callable[[], Dict],
    *,
    config: Optional[Dict] = None,
    initial_ckpt=None,
    get_models_and_configs: Optional[Callable] = None,
    get_config_files: Optional[Callable] = None,
    update_config_dropdown: Optional[Callable] = None,
    load_model_action: Optional[Callable] = None,
    runtime_status_md: Optional[Callable] = None,
    torchao_int4_supported: bool = False,
):
    """
    Build the modern Keybed tab.

    This tab shares the one global loaded model with Generation and Batch
    Generation. If model-loading callbacks are supplied, the tab exposes the
    same checkpoint picker style as Batch Generation; loading from here still
    updates the shared global runtime rather than creating a second model.
    """
    can_load_models = all([
        config is not None,
        get_models_and_configs is not None,
        get_config_files is not None,
        update_config_dropdown is not None,
        load_model_action is not None,
    ])
    ckpt_files = get_models_and_configs(config["models_directory"]) if can_load_models else []
    initial_name = os.path.basename(initial_ckpt or "")
    initial_configs = get_config_files(initial_ckpt) if can_load_models and initial_ckpt else []
    initial_config = initial_configs[0] if initial_configs else None

    def _current_runtime_summary(runtime_details: str = "") -> str:
        details = runtime_details
        if not details and runtime_status_md:
            details = runtime_status_md()
        return _runtime_status_md(
            get_runtime,
            details,
            fallback_model_name=os.path.basename(initial_ckpt or ""),
        )

    def _load_keybed_model(selected_ckpt, selected_config, int4_requested):
        result = load_model_action(selected_ckpt, selected_config, ckpt_files, int4_requested)
        details = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else ""
        return _current_runtime_summary(details)

    with gr.Column(elem_id="keybed_tab_root"):

        # Keep the shared runtime visible without letting model management
        # dominate the creative workflow.
        runtime_md = gr.Markdown(
            _current_runtime_summary(),
            elem_id="keybed_runtime_summary",
        )

        with gr.Accordion("Model & Runtime", open=False, elem_id="keybed_model_runtime_accordion"):
            if can_load_models:
                with gr.Row(elem_id="keybed_model_load_row"):
                    model_dropdown = gr.Dropdown(
                        ["Select Model"] + [file[0] for file in ckpt_files],
                        value=initial_name if initial_name else "Select Model",
                        label="Select Model",
                        scale=3,
                    )
                    config_dropdown = gr.Dropdown(
                        initial_configs if initial_configs else ["Select Config"],
                        value=initial_config if initial_config else "Select Config",
                        label="Select Config",
                        scale=3,
                    )
                    load_button = gr.Button(
                        "Load Model",
                        variant="primary",
                        scale=1,
                    )

                if bool(torchao_int4_supported):
                    with gr.Accordion("Advanced model load options", open=False):
                        gr.Markdown(
                            "INT4 requires TorchAO to be installed and supported on this device. "
                            "Enable it only when lower VRAM usage is necessary."
                        )
                        int4_checkbox = gr.Checkbox(
                            label="Enable INT4 on load (TorchAO)",
                            value=False,
                            interactive=True,
                        )
                else:
                    int4_checkbox = gr.State(value=False)
            else:
                refresh_runtime_button = gr.Button(
                    "Refresh Runtime",
                    variant="secondary",
                )

        if can_load_models:
            model_dropdown.change(
                fn=lambda x: update_config_dropdown(x, ckpt_files),
                inputs=model_dropdown,
                outputs=config_dropdown,
            )
            load_button.click(
                fn=_load_keybed_model,
                inputs=[model_dropdown, config_dropdown, int4_checkbox],
                outputs=[runtime_md],
            )
        else:
            refresh_runtime_button.click(
                fn=lambda: _current_runtime_summary(),
                inputs=[],
                outputs=[runtime_md],
            )

        # Scoped layout polish for the Keybed workflow. Keeping this styling
        # here avoids requiring any changes to the parent gradio.py file.
        gr.HTML(
            """
            <style>
                #keybed_top_prompt_row {
                    align-items: stretch !important;
                }

                #keybed_prompt_left_col {
                    display: flex !important;
                    flex-direction: column !important;
                    height: 100% !important;
                }

                #keybed_descriptor_box {
                    flex: 1 1 auto !important;
                    height: 100% !important;
                    min-height: 0 !important;
                }

                #keybed_descriptor_box > .wrap,
                #keybed_descriptor_box > div {
                    height: 100% !important;
                }

                #keybed_descriptor_box textarea {
                    height: 100% !important;
                    min-height: 18rem !important;
                    resize: none !important;
                }

                #keybed_prompt_action_col {
                    display: flex !important;
                    flex-direction: column !important;
                    height: 100% !important;
                }

                #keybed_section_preview,
                #keybed_section_export,
                #keybed_export_seed_note {
                    width: 100% !important;
                    text-align: center !important;
                }

                #keybed_section_preview > div,
                #keybed_section_export > div,
                #keybed_export_seed_note > div {
                    width: 100% !important;
                    text-align: center !important;
                }

                .keybed-section-title {
                    margin: 0.85rem 0 0.45rem;
                    font-size: 1.05rem;
                    font-weight: 650;
                    text-align: center;
                }

                .keybed-section-note {
                    margin: 0 0 0.75rem;
                    text-align: center;
                    opacity: 0.82;
                }
            </style>
            """
        )

        # CREATE -----------------------------------------------------------
        # The top row now mirrors the main Generation tab: a single tall text
        # entry surface on the left, with actions and prompt toggles on the right.
        # The expandable Prompt Builder lives below this row so opening it cannot
        # stretch the buttons or toggle panel.
        with gr.Row(equal_height=True, elem_id="keybed_top_prompt_row"):
            with gr.Column(scale=7, elem_id="keybed_prompt_left_col"):
                descriptor = gr.Textbox(
                    show_label=False,
                    placeholder=(
                        "Describe the instrument — for example: Synth, Synth Lead, "
                        "Gritty, Chiptune, Square, Buzzy, Bitcrushed, Digital, Pure Tone"
                    ),
                    lines=9,
                    elem_id="keybed_descriptor_box",
                )

            with gr.Column(scale=3, elem_id="keybed_prompt_action_col"):
                generate_preview_button = gr.Button(
                    "Generate Keybed Preview",
                    variant="primary",
                    elem_id="keybed_generate_preview_button",
                )
                random_button = gr.Button(
                    "Random Keybed Prompt",
                    variant="secondary",
                )

                with gr.Group(elem_id="keybed_prompt_toggle_group"):
                    prompt_mode_radio = gr.Radio(
                        ["Simple", "Experimental"],
                        value="Simple",
                        label="Prompt Mode",
                    )
                    wetdry_radio = gr.Radio(
                        ["Dry", "Wet"],
                        value="Dry",
                        label="FX Toggle",
                    )
                    instrument_mode_radio = gr.Radio(
                        ["Single", "Hybrid"],
                        value="Hybrid",
                        label="Instrument Blend",
                    )

        # Full-width strip beneath the prompt row. It starts collapsed and can
        # expand independently without changing the dimensions of the top row.
        with gr.Accordion(
            "Prompt Builder",
            open=False,
            elem_id="keybed_prompt_builder",
        ):
            gr.Markdown(
                "Single keeps Instrument 2 visible at None and emits one instrument token. "
                "Hybrid enables the existing two-slot hierarchy/mix behavior. Picker changes "
                "update the descriptor automatically."
            )
            with gr.Row():
                keybed_instrument_1_picker = _searchable_dropdown(
                    KEYBED_TAG_PICKER_INSTRUMENTS,
                    label="Instrument 1",
                    value=None,
                )
                keybed_instrument_2_picker = _searchable_dropdown(
                    KEYBED_TAG_PICKER_OPTIONAL_INSTRUMENTS,
                    label="Instrument 2 (Optional)",
                    value=KEYBED_NO_SECOND_INSTRUMENT,
                )
            keybed_timbre_picker = _searchable_dropdown(
                KEYBED_TAG_PICKER_TIMBRES,
                label="Timbre Tags",
                value=[],
                multiselect=True,
            )
            clear_keybed_tags_button = gr.Button(
                "Clear Selected Tags",
                variant="secondary",
            )

        # PREVIEW SETUP ----------------------------------------------------
        with gr.Accordion(
            "Preview Setup",
            open=False,
            elem_id="keybed_preview_setup",
        ):
            with gr.Row(elem_id="keybed_preview_settings_row"):
                root_lock_checkbox = gr.Checkbox(
                    label="Lock Preview Root",
                    value=False,
                    scale=1,
                )
                root_note_dropdown = gr.Dropdown(
                    KEYBED_ROOT_NOTE_NAMES,
                    label="Preview Root",
                    value="C",
                    scale=1,
                )
                root_octave_dropdown = gr.Dropdown(
                    KEYBED_ROOT_OCTAVES,
                    label="Root Octave",
                    value="4",
                    scale=1,
                )
                number_octaves_dropdown = gr.Dropdown(
                    KEYBED_OCTAVE_CHOICES,
                    label="Preview Length (Octaves)",
                    value="0.5",
                    scale=1,
                )
                seed_textbox = gr.Textbox(
                    label="Seed (-1 for random)",
                    value="-1",
                    scale=1,
                )

            gr.Markdown(
                "The preview root is automatically clamped so the generated sequence stays inside C2–G#7."
            )

            with gr.Accordion("Advanced Generation Settings", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=500,
                        step=1,
                        value=KEYBED_DEFAULT_STEPS,
                        label="Steps",
                    )
                    cfg_scale_slider = gr.Slider(
                        minimum=0.0,
                        maximum=25.0,
                        step=0.1,
                        value=KEYBED_DEFAULT_CFG,
                        label="CFG Scale",
                    )
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(
                        [
                            "dpmpp-2m-sde",
                            "dpmpp-3m-sde",
                            "k-heun",
                            "k-lms",
                            "k-dpmpp-2s-ancestral",
                            "k-dpm-2",
                            "k-dpm-fast",
                        ],
                        label="Sampler Type",
                        value=KEYBED_DEFAULT_SAMPLER,
                    )
                    sigma_min_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=KEYBED_DEFAULT_SIGMA_MIN,
                        label="Sigma Min",
                    )
                    sigma_max_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1000.0,
                        step=0.1,
                        value=KEYBED_DEFAULT_SIGMA_MAX,
                        label="Sigma Max",
                    )
                    cfg_rescale_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1,
                        step=0.01,
                        value=KEYBED_DEFAULT_CFG_RESCALE,
                        label="CFG Rescale",
                    )

        # AUDITION ---------------------------------------------------------
        gr.HTML(
            '<div class="keybed-section-title">Preview</div>',
            elem_id="keybed_section_preview",
        )
        status_output = gr.Markdown("", elem_id="keybed_status_output")
        with gr.Row(elem_id="keybed_preview_audio_row"):
            preview_audio_1 = gr.Audio(
                label="Keybed Preview",
                interactive=False,
            )
            preview_audio_2 = gr.Audio(
                label="Octave 2 Preview",
                interactive=False,
                visible=False,
            )

        latest_keybed_run_dir_state = gr.State(value="")
        latest_keybed_manifest_state = gr.State(value="")
        latest_keybed_seed_state = gr.State(value="")
        latest_keybed_descriptor_state = gr.State(value="")
        latest_keybed_export_dir_state = gr.State(value="")

        # EXPORT -----------------------------------------------------------
        # Keep range, format, and the final commit action together as one clear
        # final stage after the user has auditioned a preview.
        gr.HTML(
            '<div class="keybed-section-title">Export Instrument</div>',
            elem_id="keybed_section_export",
        )
        gr.HTML(
            '<div class="keybed-section-note">Export uses the selected fixed range and reuses the latest matching preview seed when available.</div>',
            elem_id="keybed_export_seed_note",
        )
        instrument_name_textbox = _instrument_name_textbox(
            elem_id="keybed_instrument_name",
        )
        with gr.Row(equal_height=True, elem_id="keybed_full_sampler_strip"):
            full_sampler_range_dropdown = gr.Dropdown(
                KEYBED_FULL_SAMPLER_RANGE_CHOICES,
                value=KEYBED_FULL_SAMPLER_DEFAULT_RANGE,
                label="Sampler Range",
                scale=4,
            )
            export_formats_checkbox = gr.CheckboxGroup(
                KEYBED_EXPORT_FORMAT_CHOICES,
                value=KEYBED_DEFAULT_EXPORT_FORMATS,
                label="Export Type",
                scale=3,
            )
            generate_full_sampler_button = gr.Button(
                "Generate & Export Keybed",
                variant="primary",
                scale=2,
                elem_id="keybed_generate_export_button",
            )

        with gr.Row(elem_id="keybed_export_result_row"):
            sampler_files_output = gr.File(
                label="Exported Instrument",
                file_count="multiple",
                type="filepath",
                interactive=False,
                visible=False,
                scale=4,
            )
            open_export_folder_button = gr.Button(
                "Open Export Folder",
                variant="secondary",
                visible=False,
                interactive=False,
                scale=1,
            )

        random_button.click(
            fn=random_keybed_descriptor_action,
            inputs=[
                prompt_mode_radio,
                wetdry_radio,
                instrument_mode_radio,
                seed_textbox,
                root_lock_checkbox,
                root_note_dropdown,
                root_octave_dropdown,
            ],
            outputs=[
                descriptor,
                root_note_dropdown,
                root_octave_dropdown,
                keybed_instrument_1_picker,
                keybed_instrument_2_picker,
                keybed_timbre_picker,
            ],
        )

        instrument_mode_radio.change(
            fn=set_keybed_instrument_mode_action,
            inputs=[
                instrument_mode_radio,
                descriptor,
                keybed_instrument_1_picker,
                keybed_instrument_2_picker,
                keybed_timbre_picker,
                wetdry_radio,
                seed_textbox,
                root_lock_checkbox,
                root_note_dropdown,
                root_octave_dropdown,
            ],
            outputs=[
                descriptor,
                keybed_instrument_1_picker,
                keybed_instrument_2_picker,
                root_note_dropdown,
                root_octave_dropdown,
            ],
            queue=False,
            show_progress="hidden",
        )

        picker_sync_inputs = [
            descriptor,
            keybed_instrument_1_picker,
            keybed_instrument_2_picker,
            keybed_timbre_picker,
            wetdry_radio,
            seed_textbox,
            root_lock_checkbox,
            root_note_dropdown,
            root_octave_dropdown,
        ]
        picker_sync_outputs = [
            descriptor,
            root_note_dropdown,
            root_octave_dropdown,
        ]

        # .input() runs only for direct user edits. Random Prompt updates the
        # picker programmatically, so it will not trigger a second descriptor
        # rebuild or create a feedback loop.
        for picker_component in (
            keybed_instrument_1_picker,
            keybed_instrument_2_picker,
            keybed_timbre_picker,
        ):
            picker_component.input(
                fn=auto_sync_keybed_tag_picker_action,
                inputs=picker_sync_inputs,
                outputs=picker_sync_outputs,
                queue=False,
                show_progress="hidden",
            )

        clear_keybed_tags_button.click(
            fn=clear_keybed_tag_picker_action,
            inputs=[
                descriptor,
                root_note_dropdown,
                root_octave_dropdown,
            ],
            outputs=[
                keybed_instrument_1_picker,
                keybed_instrument_2_picker,
                keybed_timbre_picker,
                descriptor,
                root_note_dropdown,
                root_octave_dropdown,
            ],
            queue=False,
            show_progress="hidden",
        )

        preview_event = generate_preview_button.click(
            fn=begin_keybed_preview_action,
            inputs=[],
            outputs=[
                generate_preview_button,
                generate_full_sampler_button,
                status_output,
            ],
        )
        preview_event.then(
            fn=lambda *args: safe_generate_keybed_preview_action(*args, get_runtime=get_runtime),
            inputs=[
                descriptor,
                root_note_dropdown,
                root_octave_dropdown,
                number_octaves_dropdown,
                wetdry_radio,
                seed_textbox,
                steps_slider,
                cfg_scale_slider,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                cfg_rescale_slider,
                latest_keybed_run_dir_state,
                latest_keybed_manifest_state,
                latest_keybed_seed_state,
                latest_keybed_descriptor_state,
            ],
            outputs=[
                preview_audio_1,
                preview_audio_2,
                root_note_dropdown,
                root_octave_dropdown,
                status_output,
                latest_keybed_run_dir_state,
                latest_keybed_manifest_state,
                latest_keybed_seed_state,
                latest_keybed_descriptor_state,
                generate_preview_button,
                generate_full_sampler_button,
            ],
        )

        full_sampler_event = generate_full_sampler_button.click(
            fn=begin_full_keybed_action,
            inputs=[export_formats_checkbox],
            outputs=[
                generate_preview_button,
                generate_full_sampler_button,
                status_output,
            ],
        )
        full_sampler_event.then(
            fn=lambda *args: safe_generate_full_keybed_sampler_action(*args, get_runtime=get_runtime),
            inputs=[
                descriptor,
                full_sampler_range_dropdown,
                export_formats_checkbox,
                wetdry_radio,
                seed_textbox,
                latest_keybed_seed_state,
                latest_keybed_descriptor_state,
                steps_slider,
                cfg_scale_slider,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                cfg_rescale_slider,
                instrument_name_textbox,
                latest_keybed_run_dir_state,
                latest_keybed_manifest_state,
                latest_keybed_export_dir_state,
            ],
            outputs=[
                status_output,
                latest_keybed_run_dir_state,
                latest_keybed_manifest_state,
                latest_keybed_seed_state,
                latest_keybed_descriptor_state,
                sampler_files_output,
                latest_keybed_export_dir_state,
                open_export_folder_button,
                generate_preview_button,
                generate_full_sampler_button,
            ],
        )

        open_export_folder_button.click(
            fn=open_keybed_export_folder_action,
            inputs=[latest_keybed_export_dir_state],
            outputs=[],
        )

    def refresh_keybed_tab_from_runtime():
        return _current_runtime_summary()

    return {
        "refresh_fn": refresh_keybed_tab_from_runtime,
        "refresh_inputs": [],
        "refresh_outputs": [runtime_md],
    }

