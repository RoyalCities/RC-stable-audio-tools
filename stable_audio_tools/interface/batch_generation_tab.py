import hashlib
import inspect
import json
import os
import random
import re
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import gradio as gr

from .prompts import (
    foundation_prompts, master_prompt_map, oneshot_prompts,
    piano_prompts, edm_elements_prompts, vocal_textures_prompts,
)


# ============================================================
# Modern Batch Generation: Loop Variations + One-Shot Variations
# ============================================================
# This tab intentionally shares the app's single global runtime. It does not
# route UI availability by checkpoint filename; users should load the intended
# model before starting a batch.

FOUNDATION_MODE_SIMPLE = "Simple"
FOUNDATION_MODE_EXPERIMENTAL = "Experimental"

FOUNDATION_MODE_TO_VARIANT = {
    FOUNDATION_MODE_SIMPLE: "M1",
    FOUNDATION_MODE_EXPERIMENTAL: "T1",
}

LOOP_BATCH_AUDIO_SLOT_COUNT = 6
LOOP_BATCH_GEN_CHOICES = [1, 2, 3, 4, 5, 6]

ONESHOT_BATCH_AUDIO_SLOT_COUNT = 6
ONESHOT_BATCH_GEN_CHOICES = [2, 3, 4, 5, 6]
ONESHOT_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ONESHOT_OCTAVE_CHOICES = [str(octave) for octave in range(0, 8)]
ONESHOT_REGISTER_LABELS = [
    "Sub Register",
    "Low Register",
    "Medium Register",
    "High Register",
    "Top Register",
]
ONESHOT_NO_SOURCE = "None"
ONESHOT_SECONDS_TOTAL = 2.0

BATCH_OUTPUT_SUBDIR = "Batch_Generation"
NOTE_TOKEN_RE = re.compile(r"^[A-G](?:#|b)?-?\d+$", re.IGNORECASE)
VISIBLE_WETDRY_TOKENS = {"wet", "dry"}

# The model is global/shared, so a second batch should never start while the
# first batch is still using it. The UI also disables both generation buttons,
# while this lock closes the tiny race window before that update reaches the
# browser.
_BATCH_GENERATION_LOCK = threading.Lock()


# ============================================================
# Small helpers
# ============================================================

def _searchable_dropdown(*args, **kwargs):
    """Use optional Gradio dropdown features without forcing a version bump."""
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


def _normalize_prompt_tokens(prompt: str) -> List[str]:
    return [part.strip() for part in str(prompt or "").split(",") if part and part.strip()]


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


def _is_visible_wetdry_token(token: str) -> bool:
    return str(token or "").strip().lower() in VISIBLE_WETDRY_TOKENS


def _is_fx_token(token: str) -> bool:
    value = str(token or "").strip().lower()
    return any(word in value for word in ("reverb", "delay", "distortion", "phaser", "bitcrush"))


def _strip_wetdry_tokens(prompt: str) -> str:
    return ", ".join([
        token for token in _normalize_prompt_tokens(prompt)
        if not _is_visible_wetdry_token(token)
    ])


def _strip_oneshot_pitch_tokens(prompt: str) -> str:
    tokens = []
    for token in _normalize_prompt_tokens(prompt):
        if NOTE_TOKEN_RE.match(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if _is_visible_wetdry_token(token):
            continue
        if token.strip().lower() in {"one shot", "oneshot", "one-shot"}:
            continue
        tokens.append(token)
    return ", ".join(tokens)


def _build_oneshot_note(note_name: str | None, octave: str | int | None) -> str:
    note_name = str(note_name or "F#").strip()
    octave = str(octave if octave not in (None, "") else "4").strip()
    return f"{note_name}{octave}"


def _split_oneshot_note(note: str | None) -> Tuple[str, str]:
    match = re.match(r"^([A-G](?:#|b)?)(-?\d+)$", str(note or "F#4").strip(), re.IGNORECASE)
    if not match:
        return "F#", "4"

    name = match.group(1).upper().replace("B", "b")
    if len(name) == 2 and name.endswith("b"):
        return "F#", "4"

    octave = match.group(2)
    if name not in ONESHOT_NOTE_NAMES:
        name = "F#"
    if octave not in ONESHOT_OCTAVE_CHOICES:
        octave = "4"
    return name, octave


def _safe_filename_part(value: str, *, max_chars: int = 48) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("#", "sharp")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value[:max_chars].strip("_") or "batch")


def _batch_style_to_prompt_args(style_label: str):
    value = str(style_label or FOUNDATION_MODE_SIMPLE).strip().lower()
    experimental = value == FOUNDATION_MODE_EXPERIMENTAL.lower()
    mode_label = FOUNDATION_MODE_EXPERIMENTAL if experimental else FOUNDATION_MODE_SIMPLE
    mode_arg = "experimental" if experimental else "standard"
    variant = FOUNDATION_MODE_TO_VARIANT[mode_label]
    allow_timbre_mix = experimental
    return mode_arg, variant, allow_timbre_mix


def _batch_wetdry_label(wetdry_label: str | None) -> str:
    return "Dry" if str(wetdry_label or "Wet").strip().lower() == "dry" else "Wet"


def _batch_wetdry_checks(wetdry_label: str | None) -> tuple[bool, bool]:
    wetdry = _batch_wetdry_label(wetdry_label)
    return wetdry == "Dry", wetdry == "Wet"


def _batch_seed(seed_str) -> str:
    seed = str(seed_str if seed_str not in (None, "") else "-1").strip()
    return seed or "-1"


def _batch_prompt_seed(seed_str) -> int:
    seed = _batch_seed(seed_str)
    if seed == "-1":
        return int(time.time_ns() % (2**31 - 1))
    return int(seed)


def _stable_int_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16)


def _runtime_model(runtime: Dict):
    return runtime.get("model")


def _runtime_sample_rate(runtime: Dict) -> int:
    return int(runtime.get("sample_rate") or 32000)


def _runtime_output_directory(runtime: Dict) -> str:
    return str(runtime.get("output_directory") or "generations")


def _runtime_model_name(runtime: Dict) -> str:
    return str(runtime.get("model_name") or "")


def _batch_require_model_loaded(get_runtime: Callable[[], Dict]) -> Dict:
    runtime = get_runtime() or {}
    if _runtime_model(runtime) is None:
        raise gr.Error("No model is loaded yet. Load a model from this tab or the main Generation tab.")
    return runtime


def _batch_runtime_status(
    get_runtime: Callable[[], Dict],
    runtime_details: str = "",
    fallback_model_name: str = "",
) -> str:
    runtime = get_runtime() or {}
    model = _runtime_model(runtime)
    model_name = _runtime_model_name(runtime)
    details_model_name = ""
    runtime_line = ""

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
            try:
                parameter = next(model.parameters())
                device_name = str(parameter.device)
                dtype_name = str(parameter.dtype).replace("torch.", "")
            except Exception:
                device_name = "loaded"
                dtype_name = "unknown"
            runtime_line = f"**Runtime:** `{device_name}` | dtype: `{dtype_name}`"

    return (
        f"**Loaded model:** `{model_name}`  \n"
        f"{runtime_line}  \n"
        f"**Sample rate:** `{_runtime_sample_rate(runtime):,} Hz` | "
        f"**Output:** `{_runtime_output_directory(runtime)}`"
    )


def _disable_batch_generation_buttons():
    return (
        gr.update(interactive=False, value="Generating Loops…"),
        gr.update(interactive=False),
    )


def _disable_batch_generation_buttons_for_oneshot():
    return (
        gr.update(interactive=False),
        gr.update(interactive=False, value="Generating One-Shots…"),
    )


def _enable_batch_generation_buttons():
    return (
        gr.update(interactive=True, value="Generate Loop Variations"),
        gr.update(interactive=True, value="Generate One-Shot Variations"),
    )


def _run_batch_exclusive(action: Callable, *args, **kwargs):
    if not _BATCH_GENERATION_LOCK.acquire(blocking=False):
        raise gr.Error("Another batch generation is already running. Wait for it to finish before starting a new batch.")
    try:
        return action(*args, **kwargs)
    finally:
        _BATCH_GENERATION_LOCK.release()


# ============================================================
# Random prompt actions
# ============================================================

def batch_random_loop_prompt_action(style_label, wetdry_label, seed_str):
    plan = _batch_random_loop_plan(style_label, wetdry_label, seed_str)
    return str(plan.get("prompt") or "")


def batch_random_loop_prompt_and_controls_action(
    style_label,
    wetdry_label,
    seed_str,
    lock_bpm,
    bars,
    bpm,
    lock_key,
    note,
    scale,
):
    plan = _batch_random_loop_plan(style_label, wetdry_label, seed_str)
    new_prompt = str(plan.get("prompt") or "")

    if not bool(lock_bpm):
        bars = random.choice([4, 8])
        bpm = random.choice([100, 110, 120, 128, 130, 140, 150])

    if not bool(lock_key):
        note = random.choice(["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
        scale = random.choice(["major", "minor"])

    return (
        new_prompt,
        bars,
        bpm,
        note,
        scale,
        gr.update(value=_normalize_foundation_source(plan.get("source_1"))),
        gr.update(value=_normalize_foundation_source(plan.get("source_2")) or FOUNDATION_PROMPT_BUILDER_NONE),
        gr.update(value=_dedupe_casefold(plan.get("timbre_tags") or [])),
        gr.update(value=str(plan.get("musical_structure") or FOUNDATION_PROMPT_BUILDER_NONE)),
        gr.update(value=_dedupe_casefold(plan.get("musical_tags") or [])),
    )




def _foundation_flat_source_choices(*, include_none: bool = False) -> List[str]:
    try:
        return list(foundation_prompts.foundation_flat_source_choices(include_none=include_none))
    except Exception:
        values: List[str] = []
        for family in getattr(foundation_prompts, "FAMILIES", []):
            values.append(str(family))
            values.extend(
                str(source)
                for source, _weight in getattr(foundation_prompts, "SUBFAMILIES", {}).get(family, [])
                if str(source or "").strip()
            )
        values = _dedupe_casefold(values)
        return ([foundation_prompts.FOUNDATION_NO_SOURCE] + values) if include_none else values


def _foundation_timbre_choices() -> List[str]:
    try:
        return list(foundation_prompts.foundation_timbre_tag_choices())
    except Exception:
        return _dedupe_casefold(list(getattr(foundation_prompts, "TIMBRE_TAGS", [])))


def _foundation_structure_choices() -> List[str]:
    try:
        return list(foundation_prompts.foundation_musical_structure_choices())
    except Exception:
        return _dedupe_casefold(list(getattr(foundation_prompts, "STRUCTURE_BASS", [])))


def _foundation_musical_choices() -> List[str]:
    try:
        return list(foundation_prompts.foundation_musical_tag_choices())
    except Exception:
        return _dedupe_casefold(
            list(getattr(foundation_prompts, "SPEED", []))
            + list(getattr(foundation_prompts, "RHYTHM", []))
            + list(getattr(foundation_prompts, "CONTOUR", []))
            + list(getattr(foundation_prompts, "DENSITY", []))
        )


FOUNDATION_FLAT_SOURCE_CHOICES = _foundation_flat_source_choices()
FOUNDATION_FLAT_SOURCE_KEYS = {source.casefold(): source for source in FOUNDATION_FLAT_SOURCE_CHOICES}
FOUNDATION_SECOND_SOURCE_CHOICES = [foundation_prompts.FOUNDATION_NO_SOURCE] + FOUNDATION_FLAT_SOURCE_CHOICES
FOUNDATION_TIMBRE_CHOICES = _foundation_timbre_choices()
FOUNDATION_TIMBRE_KEYS = {tag.casefold(): tag for tag in FOUNDATION_TIMBRE_CHOICES}
FOUNDATION_STRUCTURE_CHOICES = _foundation_structure_choices()
FOUNDATION_STRUCTURE_KEYS = {value.casefold(): value for value in FOUNDATION_STRUCTURE_CHOICES}
FOUNDATION_MUSICAL_CHOICES = _foundation_musical_choices()
FOUNDATION_MUSICAL_KEYS = {value.casefold(): value for value in FOUNDATION_MUSICAL_CHOICES}
FOUNDATION_PROMPT_BUILDER_NONE = foundation_prompts.FOUNDATION_NO_SOURCE


def _normalize_foundation_source(value) -> Optional[str]:
    text = str(value or "").strip()
    if not text or text.casefold() == FOUNDATION_PROMPT_BUILDER_NONE.casefold():
        return None
    return FOUNDATION_FLAT_SOURCE_KEYS.get(text.casefold(), text)


def _consume_leading_foundation_sources(body: List[str], *, expected_source_1=None, expected_source_2=None) -> Tuple[List[str], List[str]]:
    remaining = list(body or [])
    sources: List[str] = []
    expected = [
        _normalize_foundation_source(expected_source_1),
        _normalize_foundation_source(expected_source_2),
    ]

    for slot in range(2):
        if not remaining:
            break
        token = str(remaining[0])
        key = token.casefold()
        canonical = FOUNDATION_FLAT_SOURCE_KEYS.get(key)
        if canonical is None:
            break

        expected_value = expected[slot]
        if expected_value and key == expected_value.casefold():
            sources.append(expected_value)
            remaining.pop(0)
            continue

        if slot == 0:
            sources.append(canonical)
            remaining.pop(0)
            continue

        if key not in FOUNDATION_TIMBRE_KEYS:
            sources.append(canonical)
            remaining.pop(0)
            continue
        break

    return remaining, sources


def _split_foundation_builder_prompt(prompt: str, expected_source_1=None, expected_source_2=None):
    body: List[str] = []
    fx: List[str] = []
    for token in _normalize_prompt_tokens(prompt):
        if _is_visible_wetdry_token(token):
            continue
        if _is_fx_token(token):
            fx.append(token)
        else:
            body.append(token)

    body, sources = _consume_leading_foundation_sources(
        body,
        expected_source_1=expected_source_1,
        expected_source_2=expected_source_2,
    )

    structure_index = None
    structure = None
    for index, token in enumerate(body):
        canonical = FOUNDATION_STRUCTURE_KEYS.get(token.casefold())
        if canonical is not None:
            structure_index = index
            structure = canonical

    manual: List[str] = []
    timbre: List[str] = []
    musical: List[str] = []
    for index, token in enumerate(body):
        if index == structure_index:
            continue
        key = token.casefold()
        if key in FOUNDATION_TIMBRE_KEYS:
            timbre.append(FOUNDATION_TIMBRE_KEYS[key])
        elif key in FOUNDATION_MUSICAL_KEYS:
            musical.append(FOUNDATION_MUSICAL_KEYS[key])
        else:
            manual.append(token)
    return manual, fx, sources, timbre, structure, musical


def _enforce_foundation_mutexes_keep_last(tokens: List[str]) -> List[str]:
    out = list(tokens or [])
    groups = [{"pizzicato", "staccato", "spiccato"}]
    for group in groups:
        hit_indexes = [index for index, token in enumerate(out) if str(token).casefold() in group]
        if len(hit_indexes) <= 1:
            continue
        keep_index = hit_indexes[-1]
        out = [
            token
            for index, token in enumerate(out)
            if str(token).casefold() not in group or index == keep_index
        ]
    return out


def apply_foundation_prompt_builder_action(
    current_prompt,
    source_1,
    source_2,
    timbre_tags,
    musical_structure,
    musical_tags,
):
    manual, fx, _old_sources, _old_timbre, _old_structure, _old_musical = _split_foundation_builder_prompt(
        current_prompt,
        expected_source_1=source_1,
        expected_source_2=source_2,
    )
    selected_1 = _normalize_foundation_source(source_1)
    selected_2 = _normalize_foundation_source(source_2)
    if selected_1 and selected_2 and selected_1.casefold() == selected_2.casefold():
        selected_2 = None

    selected_timbre = [
        FOUNDATION_TIMBRE_KEYS[str(tag).casefold()]
        for tag in _dedupe_casefold(timbre_tags)
        if str(tag).casefold() in FOUNDATION_TIMBRE_KEYS
    ]
    selected_timbre = _enforce_foundation_mutexes_keep_last(selected_timbre)

    structure = str(musical_structure or "").strip()
    structure = FOUNDATION_STRUCTURE_KEYS.get(structure.casefold()) if structure else None
    selected_musical = [
        FOUNDATION_MUSICAL_KEYS[str(tag).casefold()]
        for tag in _dedupe_casefold(musical_tags)
        if str(tag).casefold() in FOUNDATION_MUSICAL_KEYS
    ]

    tokens: List[str] = []
    if selected_1:
        tokens.append(selected_1)
    if selected_2:
        tokens.append(selected_2)
    tokens.extend(manual)
    tokens.extend(selected_timbre)
    if structure:
        tokens.append(structure)
    tokens.extend(selected_musical)
    tokens.extend(fx)
    return ", ".join(_dedupe_casefold(tokens))


def clear_foundation_prompt_builder_tags_action(current_prompt, source_1, source_2, musical_structure):
    prompt = apply_foundation_prompt_builder_action(
        current_prompt,
        source_1,
        source_2,
        [],
        musical_structure,
        [],
    )
    return prompt, gr.update(value=[]), gr.update(value=[])


def _batch_random_loop_plan(style_label, wetdry_label, seed_str) -> Dict[str, object]:
    mode_arg, variant, allow_timbre_mix = _batch_style_to_prompt_args(style_label)
    wetdry = _batch_wetdry_label(wetdry_label)
    try:
        plan = foundation_prompts.prompt_generator_foundation(
            seed=_batch_prompt_seed(seed_str),
            variant=variant,
            mode=mode_arg,
            allow_timbre_mix=allow_timbre_mix,
            wetdry=wetdry,
            return_plan=True,
        )
    except TypeError:
        plan = foundation_prompts.prompt_generator_foundation(
            seed=_batch_prompt_seed(seed_str),
            variant=variant,
            mode=mode_arg,
            allow_timbre_mix=allow_timbre_mix,
            wetdry=wetdry,
        )

    if isinstance(plan, dict):
        prompt = _strip_wetdry_tokens(plan.get("prompt", ""))
        source_1 = str(
            plan.get("source_1")
            or plan.get("instrument_1")
            or plan.get("subfamily")
            or plan.get("family")
            or ""
        ).strip() or None
        source_2 = str(plan.get("source_2") or plan.get("instrument_2") or "").strip() or None
        timbre_tags = [
            FOUNDATION_TIMBRE_KEYS[str(tag).casefold()]
            for tag in _dedupe_casefold(plan.get("timbre_tags") or plan.get("tags") or [])
            if str(tag).casefold() in FOUNDATION_TIMBRE_KEYS
        ]
        structure = str(plan.get("musical_structure") or "").strip()
        structure = FOUNDATION_STRUCTURE_KEYS.get(structure.casefold(), structure) if structure else None
        musical_tags = [
            FOUNDATION_MUSICAL_KEYS[str(tag).casefold()]
            for tag in _dedupe_casefold(plan.get("musical_tags") or [])
            if str(tag).casefold() in FOUNDATION_MUSICAL_KEYS
        ]
    else:
        prompt = _strip_wetdry_tokens(plan)
        _manual, _fx, sources, timbre_tags, structure, musical_tags = _split_foundation_builder_prompt(prompt)
        source_1 = sources[0] if sources else None
        source_2 = sources[1] if len(sources) > 1 else None

    return {
        "prompt": prompt,
        "source_1": source_1,
        "source_2": source_2,
        "timbre_tags": _dedupe_casefold(timbre_tags),
        "musical_structure": structure or FOUNDATION_PROMPT_BUILDER_NONE,
        "musical_tags": _dedupe_casefold(musical_tags),
    }


def _oneshot_flat_source_choices(*, include_none: bool = False) -> List[str]:
    try:
        return list(oneshot_prompts.oneshot_flat_source_choices(include_none=include_none))
    except Exception:
        values: List[str] = []
        for family in oneshot_prompts.ONESHOT_FAMILIES:
            values.append(str(family))
            values.extend(
                str(source)
                for source, _weight in oneshot_prompts.ONESHOT_SUBFAMILIES.get(family, [])
                if str(source or "").strip()
            )
        values = _dedupe_casefold(values)
        return ([ONESHOT_NO_SOURCE] + values) if include_none else values


def _oneshot_timbre_choices() -> List[str]:
    try:
        return list(oneshot_prompts.oneshot_timbre_tag_choices())
    except Exception:
        return _dedupe_casefold([name for name, _weight in oneshot_prompts.ONESHOT_TIMBRE_TAGS])


ONESHOT_FLAT_SOURCE_CHOICES = _oneshot_flat_source_choices()
ONESHOT_FLAT_SOURCE_KEYS = {source.casefold(): source for source in ONESHOT_FLAT_SOURCE_CHOICES}
ONESHOT_SECOND_SOURCE_CHOICES = [ONESHOT_NO_SOURCE] + ONESHOT_FLAT_SOURCE_CHOICES
ONESHOT_TIMBRE_CHOICES = _oneshot_timbre_choices()
ONESHOT_TIMBRE_KEYS = {tag.casefold() for tag in ONESHOT_TIMBRE_CHOICES}


def _normalize_oneshot_source(value) -> Optional[str]:
    text = str(value or "").strip()
    if not text or text.casefold() == ONESHOT_NO_SOURCE.casefold():
        return None
    return ONESHOT_FLAT_SOURCE_KEYS.get(text.casefold(), text)


def _enforce_oneshot_mutexes_keep_last(tokens: List[str]) -> List[str]:
    out = list(tokens)
    for group in oneshot_prompts.ONESHOT_TAG_MUTEX_GROUPS:
        group_keys = {str(value).casefold() for value in group}
        hit_indexes = [index for index, token in enumerate(out) if str(token).casefold() in group_keys]
        if len(hit_indexes) <= 1:
            continue
        keep_index = hit_indexes[-1]
        out = [
            token
            for index, token in enumerate(out)
            if str(token).casefold() not in group_keys or index == keep_index
        ]
    return out


def _split_visible_oneshot_descriptor(prompt: str) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """Split visible prompt into custom body, FX, and up to two leading flat sources."""
    body: List[str] = []
    fx: List[str] = []
    for token in _normalize_prompt_tokens(prompt):
        if NOTE_TOKEN_RE.match(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if _is_visible_wetdry_token(token):
            continue
        if token.casefold() in {"one shot", "oneshot", "one-shot"}:
            continue
        if _is_fx_token(token):
            fx.append(token)
        else:
            body.append(token)

    sources: List[str] = []
    while body and len(sources) < 2:
        match = ONESHOT_FLAT_SOURCE_KEYS.get(body[0].casefold())
        if not match:
            break
        sources.append(match)
        body = body[1:]

    return body, fx, (sources[0] if sources else None), (sources[1] if len(sources) > 1 else None)


def apply_oneshot_prompt_builder_action(
    current_prompt,
    instrument_1,
    instrument_2,
    timbre_tags,
    wetdry_label,
):
    """
    Mirror the flat picker into the prompt while preserving free-form user text.

    Only the two leading recognized source slots and recognized timbre tags are
    picker-controlled. Extra tokens such as Guitar, Vocal, Choir remain valid
    prompt text and are sent to generation without hierarchy enforcement.
    """
    body_tokens, fx_tokens, _old_1, _old_2 = _split_visible_oneshot_descriptor(current_prompt)
    manual_tokens = [
        token
        for token in body_tokens
        if str(token).casefold() not in ONESHOT_TIMBRE_KEYS
    ]

    selected_1 = _normalize_oneshot_source(instrument_1)
    selected_2 = _normalize_oneshot_source(instrument_2)
    if selected_1 and selected_2 and selected_1.casefold() == selected_2.casefold():
        selected_2 = None

    selected_tags = [str(tag).strip() for tag in (timbre_tags or []) if str(tag or "").strip()]
    is_wet = _batch_wetdry_label(wetdry_label) == "Wet"
    selected_tags = oneshot_prompts.filter_oneshot_tags_for_wetdry(selected_tags, wet=is_wet)
    selected_tags = _enforce_oneshot_mutexes_keep_last(_dedupe_casefold(selected_tags))

    tokens: List[str] = []
    if selected_1:
        tokens.append(selected_1)
    if selected_2:
        tokens.append(selected_2)
    tokens.extend(manual_tokens)
    tokens.extend(selected_tags)
    tokens.extend(fx_tokens)
    return ", ".join(_dedupe_casefold(tokens))


def _suggest_oneshot_note_for_sources(instrument_1, instrument_2, seed_str) -> Tuple[str, str]:
    rng = random.Random(_batch_prompt_seed(seed_str))
    try:
        _register, note = oneshot_prompts.pick_oneshot_note_for_sources(
            rng,
            instrument_1=_normalize_oneshot_source(instrument_1),
            instrument_2=_normalize_oneshot_source(instrument_2),
        )
    except Exception:
        note = "F#4"
    return _split_oneshot_note(note)


def oneshot_source_change_action(
    current_prompt,
    instrument_1,
    instrument_2,
    timbre_tags,
    wetdry_label,
    lock_note,
    current_note_name,
    current_octave,
    seed_str,
):
    prompt = apply_oneshot_prompt_builder_action(
        current_prompt,
        instrument_1,
        instrument_2,
        timbre_tags,
        wetdry_label,
    )
    if bool(lock_note):
        return prompt, str(current_note_name or "F#"), str(current_octave or "4")
    note_name, octave = _suggest_oneshot_note_for_sources(instrument_1, instrument_2, seed_str)
    return prompt, note_name, octave


def oneshot_wetdry_change_action(current_prompt, instrument_1, instrument_2, timbre_tags, wetdry_label):
    is_wet = _batch_wetdry_label(wetdry_label) == "Wet"
    filtered_tags = oneshot_prompts.filter_oneshot_tags_for_wetdry(
        [str(tag) for tag in (timbre_tags or [])],
        wet=is_wet,
    )
    filtered_tags = _enforce_oneshot_mutexes_keep_last(_dedupe_casefold(filtered_tags))
    prompt = apply_oneshot_prompt_builder_action(
        current_prompt,
        instrument_1,
        instrument_2,
        filtered_tags,
        wetdry_label,
    )
    return prompt, gr.update(value=filtered_tags)


def clear_oneshot_timbre_action(current_prompt, instrument_1, instrument_2, wetdry_label):
    prompt = apply_oneshot_prompt_builder_action(
        current_prompt,
        instrument_1,
        instrument_2,
        [],
        wetdry_label,
    )
    return prompt, gr.update(value=[])


def _batch_random_oneshot_plan(
    style_label,
    wetdry_label,
    seed_str,
    *,
    locked_note: str | None = None,
) -> Dict[str, object]:
    mode_arg, variant, allow_timbre_mix = _batch_style_to_prompt_args(style_label)
    wetdry = _batch_wetdry_label(wetdry_label)

    try:
        plan = master_prompt_map.prompt_generator_oneshot_model_router(
            seed=_batch_prompt_seed(seed_str),
            sample_type="oneshot",
            mode=mode_arg,
            variant=variant,
            allow_timbre_mix=allow_timbre_mix,
            note=locked_note,
            wetdry=wetdry,
            include_prefix=False,
            include_note=False,
            return_plan=True,
        )
    except TypeError:
        plan = master_prompt_map.prompt_generator_oneshot_model_router(
            seed=_batch_prompt_seed(seed_str),
            sample_type="oneshot",
            mode=mode_arg,
            variant=variant,
            allow_timbre_mix=allow_timbre_mix,
            note=locked_note or "F#4",
            wetdry=wetdry,
            include_prefix=False,
            include_note=False,
        )

    if isinstance(plan, dict):
        prompt = _strip_oneshot_pitch_tokens(plan.get("prompt", ""))
        note = plan.get("note") or locked_note or "F#4"
        instrument_1 = str(
            plan.get("instrument_1")
            or plan.get("subfamily")
            or plan.get("family")
            or ""
        ).strip() or None
        instrument_2 = str(plan.get("instrument_2") or "").strip() or None
        tags = [str(tag) for tag in (plan.get("tags") or []) if str(tag or "").strip()]
    else:
        prompt = _strip_oneshot_pitch_tokens(plan)
        note = locked_note or "F#4"
        body, _fx, instrument_1, instrument_2 = _split_visible_oneshot_descriptor(prompt)
        tags = [token for token in body if token.casefold() in ONESHOT_TIMBRE_KEYS]

    if not tags:
        tags = [
            token
            for token in _normalize_prompt_tokens(prompt)
            if token.casefold() in ONESHOT_TIMBRE_KEYS
        ]

    instrument_1 = _normalize_oneshot_source(instrument_1)
    instrument_2 = _normalize_oneshot_source(instrument_2)
    note_name, octave = _split_oneshot_note(note)
    return {
        "prompt": prompt,
        "note_name": note_name,
        "octave": octave,
        "instrument_1": instrument_1,
        "instrument_2": instrument_2,
        "tags": _dedupe_casefold(tags),
    }


def batch_random_oneshot_prompt_action(style_label, wetdry_label, seed_str, note_name, octave):
    locked_note = _build_oneshot_note(note_name, octave)
    return str(_batch_random_oneshot_plan(
        style_label,
        wetdry_label,
        seed_str,
        locked_note=locked_note,
    )["prompt"])


def batch_random_oneshot_prompt_with_lock_action(
    style_label,
    wetdry_label,
    seed_str,
    lock_note,
    note_name,
    octave,
):
    locked_note = _build_oneshot_note(note_name, octave) if bool(lock_note) else None
    plan = _batch_random_oneshot_plan(
        style_label,
        wetdry_label,
        seed_str,
        locked_note=locked_note,
    )
    return (
        plan["prompt"],
        plan["note_name"],
        plan["octave"],
        gr.update(value=plan.get("instrument_1")),
        gr.update(value=plan.get("instrument_2") or ONESHOT_NO_SOURCE),
        gr.update(value=plan.get("tags") or []),
    )


# ============================================================
# Lightweight Audio output helpers
# ============================================================

def _empty_loop_audio_slot(slot_index: int):
    return gr.update(value=None, label=f"Loop Gen {int(slot_index):02d}", visible=False)


def _loop_audio_slot_update(slot_index: int, audio_path: str, seed: int):
    return gr.update(
        value=audio_path,
        label=f"Loop Gen {int(slot_index):02d} | seed {int(seed)}",
        visible=True,
    )


def _loop_audio_slot_updates(audio_paths: List[str], seeds: List[int]) -> List[object]:
    updates: List[object] = []
    for index in range(1, LOOP_BATCH_AUDIO_SLOT_COUNT + 1):
        if index <= len(audio_paths):
            seed = int(seeds[index - 1]) if index - 1 < len(seeds) else -1
            updates.append(_loop_audio_slot_update(index, audio_paths[index - 1], seed))
        else:
            # Slots are cleared before diffusion starts. Leave unused players
            # untouched in the completion payload so Gradio only processes the
            # WAV paths that were actually generated.
            updates.append(gr.update())
    return updates


def _empty_oneshot_audio_slot(slot_index: int):
    return gr.update(value=None, label=f"One-Shot Gen {int(slot_index):02d}", visible=False)


def _oneshot_audio_slot_update(slot_index: int, audio_path: str, seed: int):
    return gr.update(
        value=audio_path,
        label=f"One-Shot Gen {int(slot_index):02d} | seed {int(seed)}",
        visible=True,
    )


def _oneshot_audio_slot_updates(audio_paths: List[str], seeds: List[int]) -> List[object]:
    updates: List[object] = []
    for index in range(1, ONESHOT_BATCH_AUDIO_SLOT_COUNT + 1):
        if index <= len(audio_paths):
            seed = int(seeds[index - 1]) if index - 1 < len(seeds) else -1
            updates.append(_oneshot_audio_slot_update(index, audio_paths[index - 1], seed))
        else:
            # Slots are cleared before diffusion starts. Leave unused players
            # untouched in the completion payload so Gradio only processes the
            # WAV paths that were actually generated.
            updates.append(gr.update())
    return updates


def _prepare_loop_batch_ui():
    """Disable both generators and clear old loop players before inference."""
    return (
        gr.update(interactive=False, value="Generating Loops…"),
        gr.update(interactive=False),
        *[_empty_loop_audio_slot(index) for index in range(1, LOOP_BATCH_AUDIO_SLOT_COUNT + 1)],
    )


def _prepare_oneshot_batch_ui():
    """Disable both generators and clear old one-shot players before inference."""
    return (
        gr.update(interactive=False),
        gr.update(interactive=False, value="Generating One-Shots…"),
        *[_empty_oneshot_audio_slot(index) for index in range(1, ONESHOT_BATCH_AUDIO_SLOT_COUNT + 1)],
    )


def _verified_batch_audio_path(audio_path: str) -> str:
    """Return one absolute, closed-on-disk WAV path suitable for gr.Audio."""
    resolved = os.path.abspath(os.path.expanduser(str(audio_path or "").strip()))
    if not resolved or not os.path.isfile(resolved):
        raise gr.Error(f"Generated audio file was not found: {resolved or audio_path}")
    try:
        size_bytes = os.path.getsize(resolved)
    except OSError as exc:
        raise gr.Error(f"Generated audio file could not be inspected: {resolved}") from exc
    if size_bytes <= 44:
        raise gr.Error(f"Generated audio file is empty: {resolved}")
    return resolved


def _resolve_loop_batch_base_seed(seed_str) -> int:
    seed = _batch_seed(seed_str)
    if seed == "-1":
        return int(time.time_ns() % (2**31 - 1))
    return int(seed)


def _loop_seed_for_index(base_seed: int, index: int, prompt: str, bars, bpm, note, scale) -> int:
    if int(index) <= 1:
        return int(base_seed)
    return _stable_int_seed(base_seed, "loop_batch", index, prompt, bars, bpm, note, scale)


def _resolve_oneshot_batch_base_seed(seed_str) -> int:
    seed = _batch_seed(seed_str)
    if seed == "-1":
        return int(time.time_ns() % (2**31 - 1))
    return int(seed)


def _oneshot_seed_for_index(base_seed: int, index: int, prompt: str, note: str) -> int:
    if int(index) <= 1:
        return int(base_seed)
    return _stable_int_seed(base_seed, "oneshot_batch", index, prompt, note)


def _batch_generation_sample_folder(sample_type: str) -> str:
    return "One_Shots" if str(sample_type).strip().lower() == "oneshot" else "Loops"


def _is_loop_control_token(token: str) -> bool:
    value = str(token or "").strip()
    return bool(
        re.match(r"^\d+\s*BPM$", value, re.IGNORECASE)
        or re.match(r"^\d+\s*Bars?$", value, re.IGNORECASE)
        or re.match(r"^[A-G](?:#|b)?\s+(?:major|minor)$", value, re.IGNORECASE)
    )


def _batch_descriptor_slug(prompt: str, sample_type: str, *, max_chars: int = 36) -> str:
    usable_tokens: List[str] = []
    for token in _normalize_prompt_tokens(prompt):
        low = token.strip().lower()
        if low in {"one shot", "oneshot", "one-shot"}:
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if NOTE_TOKEN_RE.match(token):
            continue
        if _is_visible_wetdry_token(token) or _is_fx_token(token) or _is_loop_control_token(token):
            continue
        usable_tokens.append(token)

    fallback = "oneshot" if str(sample_type).strip().lower() == "oneshot" else "loop"
    return _safe_filename_part("_".join(usable_tokens[:2]) or fallback, max_chars=max_chars)


def _make_batch_generation_run_dir(
    output_root: str,
    sample_type: str,
    prompt: str,
    *,
    base_seed: int,
    settings_key: str,
) -> Tuple[str, str]:
    sample_folder = _batch_generation_sample_folder(sample_type)
    parent = os.path.join(str(output_root or "generations"), BATCH_OUTPUT_SUBDIR, sample_folder)
    os.makedirs(parent, exist_ok=True)

    slug = _batch_descriptor_slug(prompt, sample_type)
    digest = hashlib.sha1(
        f"{time.time_ns()}|{sample_type}|{base_seed}|{settings_key}|{prompt}".encode("utf-8", errors="replace")
    ).hexdigest()[:8]
    base = f"{slug}_{digest}"

    run_name = base
    run_dir = os.path.join(parent, run_name)
    counter = 1
    while os.path.exists(run_dir):
        counter += 1
        run_name = f"{base}_{counter}"
        run_dir = os.path.join(parent, run_name)

    os.makedirs(run_dir, exist_ok=True)
    return run_name, run_dir


def _batch_loop_conditioning_prompt(prompt: str, note, scale, bars, bpm, wetdry_label: str) -> str:
    wetdry = _batch_wetdry_label(wetdry_label)
    is_wet = wetdry == "Wet"
    body_tokens: List[str] = []
    fx_tokens: List[str] = []

    for token in _normalize_prompt_tokens(prompt):
        if _is_visible_wetdry_token(token):
            continue
        if _is_fx_token(token):
            fx_tokens.append(token)
        else:
            body_tokens.append(token)

    tokens = body_tokens + [wetdry]
    if is_wet:
        tokens.extend(fx_tokens)
    tokens.extend([f"{note} {scale}", f"{bars} Bars", f"{bpm} BPM"])
    return ", ".join(tokens)


def _batch_oneshot_register_label(note: str) -> Optional[str]:
    try:
        return master_prompt_map.resolve_oneshot_register_from_ui(note=note)
    except Exception:
        return None


def _batch_oneshot_conditioning_prompt(prompt: str, note: str, wetdry_label: str) -> str:
    wetdry = _batch_wetdry_label(wetdry_label)
    is_wet = wetdry == "Wet"
    register_label = _batch_oneshot_register_label(note)
    body_tokens: List[str] = []
    fx_tokens: List[str] = []

    for token in _normalize_prompt_tokens(prompt):
        if NOTE_TOKEN_RE.match(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if _is_visible_wetdry_token(token):
            continue
        if token.strip().lower() in {"one shot", "oneshot", "one-shot"}:
            continue
        if _is_fx_token(token):
            fx_tokens.append(token)
        else:
            body_tokens.append(token)

    tokens = ["One Shot"] + body_tokens + [wetdry]
    if is_wet:
        tokens.extend(fx_tokens)
    if register_label:
        tokens.append(register_label)
    tokens.append(note)
    return ", ".join(tokens)


def _write_batch_generation_manifest_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_batch_generation_metadata_txt(
    path: str,
    *,
    sample_type: str,
    run_dir: str,
    model_name: str,
    visible_prompt: str,
    conditioning_prompt: str,
    wetdry: str,
    base_seed: int,
    seeds: List[int],
    generation_count: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    cfg_rescale: float,
    rows: List[Dict[str, object]],
    bars=None,
    bpm=None,
    key=None,
    scale=None,
    oneshot_note=None,
) -> None:
    lines = [
        f"sample_type: {sample_type}",
        f"run_dir: {run_dir}",
        f"model: {model_name or 'n/a'}",
        f"generation_count: {int(generation_count)}",
        f"base_seed: {int(base_seed)}",
        f"seeds: {', '.join(str(seed) for seed in seeds)}",
        f"wetdry: {wetdry}",
        "",
        f"visible_prompt: {visible_prompt}",
        f"conditioning_prompt: {conditioning_prompt}",
        "",
    ]

    if str(sample_type).strip().lower() == "loop":
        lines.extend([
            f"key_signature: {key} {scale}",
            f"bars: {bars}",
            f"bpm: {bpm}",
        ])
    else:
        lines.append(f"oneshot_note: {oneshot_note}")
        lines.append(f"seconds_total: {ONESHOT_SECONDS_TOTAL}")

    lines.extend([
        "",
        f"steps: {steps}",
        f"cfg_scale: {cfg_scale}",
        f"sampler_type: {sampler_type}",
        f"sigma_min: {sigma_min}",
        f"sigma_max: {sigma_max}",
        f"cfg_rescale: {cfg_rescale}",
        "",
        "outputs:",
    ])

    for index, row in enumerate(rows, start=1):
        lines.append(
            f"  {index:02d}. seed={row.get('seed')} audio={row.get('audio_path')}"
        )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")


# ============================================================
# Generation actions
# ============================================================

def batch_generate_loop_action(
    prompt,
    style_label,
    wetdry_label,
    seed_str,
    bars,
    bpm,
    note,
    scale,
    generation_count,
    steps,
    cfg_scale,
    sampler_type,
    sigma_min,
    sigma_max,
    cfg_rescale,
    *,
    get_runtime: Callable[[], Dict],
    generate_cond: Callable,
):
    runtime = _batch_require_model_loaded(get_runtime)
    prompt = str(prompt or "").strip()
    if not prompt:
        prompt = batch_random_loop_prompt_action(style_label, wetdry_label, seed_str)

    dry_checked, wet_checked = _batch_wetdry_checks(wetdry_label)
    wetdry = _batch_wetdry_label(wetdry_label)
    generation_count = max(1, min(LOOP_BATCH_AUDIO_SLOT_COUNT, int(generation_count or 1)))
    base_seed = _resolve_loop_batch_base_seed(seed_str)
    output_root = _runtime_output_directory(runtime)
    model_name = _runtime_model_name(runtime)
    conditioning_prompt = _batch_loop_conditioning_prompt(prompt, note, scale, bars, bpm, wetdry)
    run_subdir, run_dir = _make_batch_generation_run_dir(
        output_root,
        "loop",
        prompt,
        base_seed=base_seed,
        settings_key=f"{bars}|{bpm}|{note}|{scale}|{generation_count}|{steps}|{cfg_scale}|{sampler_type}|{sigma_min}|{sigma_max}|{cfg_rescale}",
    )

    audio_paths: List[str] = []
    seeds: List[int] = []
    rows: List[Dict[str, object]] = []

    for index in range(1, generation_count + 1):
        generation_seed = _loop_seed_for_index(base_seed, index, prompt, bars, bpm, note, scale)
        audio_path, _spectrograms, _piano_roll, _midi_path = generate_cond(
            prompt=prompt,
            negative_prompt="",
            bars=int(bars),
            bpm=int(bpm),
            note=str(note),
            scale=str(scale),
            sample_type_loop_checked=True,
            sample_type_oneshot_checked=False,
            oneshot_note_name="F#",
            oneshot_octave="4",
            wetdry_dry_checked=dry_checked,
            wetdry_wet_checked=wet_checked,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            preview_every=0,
            seed=str(generation_seed),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
            use_init=False,
            init_audio=None,
            init_noise_level=1.0,
            batch_size=1,
            output_parent_subdir=BATCH_OUTPUT_SUBDIR,
            output_run_subdir=run_subdir,
            output_write_sidecar=False,
            output_generate_midi=False,
            output_generate_spectrogram=False,
        )
        audio_path = _verified_batch_audio_path(audio_path)
        audio_paths.append(audio_path)
        seeds.append(generation_seed)
        rows.append({
            "type": "batch_loop",
            "index": index,
            "seed": int(generation_seed),
            "base_seed": int(base_seed),
            "visible_prompt": _strip_wetdry_tokens(prompt),
            "conditioning_prompt": conditioning_prompt,
            "wetdry": wetdry,
            "key": str(note),
            "scale": str(scale),
            "bars": int(bars),
            "bpm": int(bpm),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "sampler_type": str(sampler_type),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "cfg_rescale": float(cfg_rescale),
            "model": model_name,
            "audio_path": audio_path,
            "midi_path": None,
        })

    metadata_path = os.path.join(run_dir, "metadata.txt")
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    _write_batch_generation_manifest_jsonl(manifest_path, rows)
    _write_batch_generation_metadata_txt(
        metadata_path,
        sample_type="loop",
        run_dir=run_dir,
        model_name=model_name,
        visible_prompt=_strip_wetdry_tokens(prompt),
        conditioning_prompt=conditioning_prompt,
        wetdry=wetdry,
        base_seed=base_seed,
        seeds=seeds,
        generation_count=generation_count,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        sampler_type=str(sampler_type),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        cfg_rescale=float(cfg_rescale),
        rows=rows,
        bars=int(bars),
        bpm=int(bpm),
        key=str(note),
        scale=str(scale),
    )

    status = (
        f"Generated `{len(audio_paths)}` loop variation(s) for `{note} {scale}`, `{bars}` bars, `{bpm}` BPM.  \n"
        f"Run folder: `{run_dir}`  \n"
        f"Metadata: `{metadata_path}`  \n"
        f"Manifest: `{manifest_path}`  \n"
        f"Base seed: `{base_seed}`  \n"
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`"
    )

    return (
        prompt,
        *_loop_audio_slot_updates(audio_paths, seeds),
        status,
    )


def batch_generate_oneshot_action(
    prompt,
    style_label,
    wetdry_label,
    seed_str,
    note_name,
    octave,
    generation_count,
    steps,
    cfg_scale,
    sampler_type,
    sigma_min,
    sigma_max,
    cfg_rescale,
    *,
    get_runtime: Callable[[], Dict],
    generate_cond: Callable,
):
    runtime = _batch_require_model_loaded(get_runtime)
    prompt = str(prompt or "").strip()
    if not prompt:
        prompt = batch_random_oneshot_prompt_action(style_label, wetdry_label, seed_str, note_name, octave)

    dry_checked, wet_checked = _batch_wetdry_checks(wetdry_label)
    wetdry = _batch_wetdry_label(wetdry_label)
    oneshot_note = _build_oneshot_note(note_name, octave)
    conditioning_prompt = _batch_oneshot_conditioning_prompt(prompt, oneshot_note, wetdry)
    generation_count = max(2, min(ONESHOT_BATCH_AUDIO_SLOT_COUNT, int(generation_count or 3)))
    base_seed = _resolve_oneshot_batch_base_seed(seed_str)
    output_root = _runtime_output_directory(runtime)
    model_name = _runtime_model_name(runtime)
    run_subdir, run_dir = _make_batch_generation_run_dir(
        output_root,
        "oneshot",
        prompt,
        base_seed=base_seed,
        settings_key=f"{oneshot_note}|{generation_count}|{steps}|{cfg_scale}|{sampler_type}|{sigma_min}|{sigma_max}|{cfg_rescale}",
    )

    audio_paths: List[str] = []
    seeds: List[int] = []
    rows: List[Dict[str, object]] = []

    for index in range(1, generation_count + 1):
        generation_seed = _oneshot_seed_for_index(base_seed, index, prompt, oneshot_note)
        audio_path, _spectrograms, _piano_roll, _midi_path = generate_cond(
            prompt=prompt,
            negative_prompt="",
            bars=4,
            bpm=100,
            note="C",
            scale="major",
            sample_type_loop_checked=False,
            sample_type_oneshot_checked=True,
            oneshot_note_name=str(note_name),
            oneshot_octave=str(octave),
            wetdry_dry_checked=dry_checked,
            wetdry_wet_checked=wet_checked,
            cfg_scale=float(cfg_scale),
            steps=int(steps),
            preview_every=0,
            seed=str(generation_seed),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
            use_init=False,
            init_audio=None,
            init_noise_level=1.0,
            batch_size=1,
            output_parent_subdir=BATCH_OUTPUT_SUBDIR,
            output_run_subdir=run_subdir,
            output_write_sidecar=False,
            output_generate_midi=False,
            output_generate_spectrogram=False,
        )
        audio_path = _verified_batch_audio_path(audio_path)
        audio_paths.append(audio_path)
        seeds.append(generation_seed)
        rows.append({
            "type": "batch_oneshot",
            "index": index,
            "seed": int(generation_seed),
            "base_seed": int(base_seed),
            "visible_prompt": _strip_oneshot_pitch_tokens(prompt),
            "conditioning_prompt": conditioning_prompt,
            "wetdry": wetdry,
            "oneshot_note": oneshot_note,
            "seconds_total": float(ONESHOT_SECONDS_TOTAL),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "sampler_type": str(sampler_type),
            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "cfg_rescale": float(cfg_rescale),
            "model": model_name,
            "audio_path": audio_path,
            "midi_path": None,
        })

    metadata_path = os.path.join(run_dir, "metadata.txt")
    manifest_path = os.path.join(run_dir, "manifest.jsonl")
    _write_batch_generation_manifest_jsonl(manifest_path, rows)
    _write_batch_generation_metadata_txt(
        metadata_path,
        sample_type="oneshot",
        run_dir=run_dir,
        model_name=model_name,
        visible_prompt=_strip_oneshot_pitch_tokens(prompt),
        conditioning_prompt=conditioning_prompt,
        wetdry=wetdry,
        base_seed=base_seed,
        seeds=seeds,
        generation_count=generation_count,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        sampler_type=str(sampler_type),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        cfg_rescale=float(cfg_rescale),
        rows=rows,
        oneshot_note=oneshot_note,
    )

    status = (
        f"Generated `{len(audio_paths)}` one-shot variation(s) for `{oneshot_note}`.  \n"
        f"Run folder: `{run_dir}`  \n"
        f"Metadata: `{metadata_path}`  \n"
        f"Manifest: `{manifest_path}`  \n"
        f"Base seed: `{base_seed}`  \n"
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`"
    )

    return (
        prompt,
        *_oneshot_audio_slot_updates(audio_paths, seeds),
        status,
    )


# ============================================================
# Gradio builder
# ============================================================

def build_batch_generation_tab(
    *,
    config: Dict,
    initial_ckpt,
    get_runtime: Callable[[], Dict],
    get_models_and_configs: Callable,
    get_config_files: Callable,
    update_config_dropdown: Callable,
    load_model_action: Callable,
    runtime_status_md: Callable,
    generate_cond: Callable,
    torchao_int4_supported: bool = False,
):
    """Build the two-column Loop / One-Shot batch workflow."""
    ckpt_files = get_models_and_configs(config["models_directory"])
    initial_name = os.path.basename(initial_ckpt or "")
    initial_configs = get_config_files(initial_ckpt) if initial_ckpt else []
    initial_config = initial_configs[0] if initial_configs else None

    def _current_runtime_summary(runtime_details: str = "") -> str:
        details = runtime_details or (runtime_status_md() if runtime_status_md else "")
        return _batch_runtime_status(
            get_runtime,
            details,
            fallback_model_name=os.path.basename(initial_ckpt or ""),
        )

    def _loop_family(model_name=None):
        runtime = get_runtime() or {}
        name = model_name or runtime.get("model_name") or initial_name
        return master_prompt_map.get_loop_prompt_family(name)

    def _loop_builder_visibility(model_name=None):
        family = _loop_family(model_name)
        return (
            gr.update(visible=family == "foundation"),
            gr.update(visible=family == "piano"),
            gr.update(visible=family == "edm_elements"),
            gr.update(visible=family == "vocal_textures"),
        )

    def _load_batch_model(selected_ckpt, selected_config, int4_requested):
        result = load_model_action(selected_ckpt, selected_config, ckpt_files, int4_requested)
        info = result[0] if isinstance(result, (tuple, list)) and len(result) > 0 else "Loaded model."
        details = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else ""
        return info, _current_runtime_summary(details), *_loop_builder_visibility(selected_ckpt)

    def _random_loop_for_runtime(style_label, wetdry_label, seed_str, lock_bpm, bars, bpm, lock_key, note, scale):
        family = _loop_family()

        # All return paths include every loop-builder control. Only the active
        # family's controls receive values; inactive builders are left untouched.
        legacy_noops = tuple(gr.update() for _ in range(11))
        if family == "foundation":
            foundation_result = batch_random_loop_prompt_and_controls_action(
                style_label, wetdry_label, seed_str, lock_bpm, bars, bpm, lock_key, note, scale
            )
            return (*foundation_result, *legacy_noops)

        generator = master_prompt_map.get_loop_prompt_generator((get_runtime() or {}).get("model_name") or initial_name)
        try:
            plan = generator(return_plan=True)
        except TypeError:
            plan = generator()
        prompt = plan.get("prompt", "") if isinstance(plan, dict) else str(plan or "")
        if not bool(lock_bpm):
            bars = random.choice([4, 8])
            bpm = random.choice([100, 110, 120, 128, 130, 140, 150])
        if not bool(lock_key):
            note = random.choice(["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
            scale = random.choice(["major", "minor"])

        foundation_noops = tuple(gr.update() for _ in range(5))
        piano_updates = [gr.update() for _ in range(5)]
        edm_updates = [gr.update() for _ in range(4)]
        vocal_updates = [gr.update() for _ in range(2)]

        if isinstance(plan, dict):
            if family == "piano":
                piano_type = plan.get("piano_type") or piano_prompts.PIANO_TYPES[0]
                allowed = piano_prompts.effect_choices_for_piano(piano_type)
                effect = plan.get("effect") if plan.get("effect") in allowed else allowed[0]
                piano_updates = [
                    gr.update(value=piano_type),
                    gr.update(value=plan.get("structure") or piano_prompts.STRUCTURE_CHOICES[0]),
                    gr.update(value=plan.get("chord_style") or "None"),
                    gr.update(value=plan.get("melody_style") or "None"),
                    gr.update(choices=allowed, value=effect),
                ]
            elif family == "edm_elements":
                edm_updates = [
                    gr.update(value=list(plan.get("sound_tags") or [])),
                    gr.update(value=plan.get("structure") or "None"),
                    gr.update(value=list(plan.get("musical_tags") or [])),
                    gr.update(value=list(plan.get("effects") or [])),
                ]
            elif family == "vocal_textures":
                vocal_updates = [
                    gr.update(value=plan.get("vocal_type") or vocal_textures_prompts.VOCAL_TYPES[0]),
                    gr.update(value=plan.get("structure") or vocal_textures_prompts.STRUCTURE_CHOICES[0]),
                ]

        return (
            prompt, bars, bpm, note, scale,
            *foundation_noops,
            *piano_updates,
            *edm_updates,
            *vocal_updates,
        )

    with gr.Column(elem_id="batch_generation_tab_root"):
        runtime_md = gr.Markdown(
            _current_runtime_summary(),
            elem_id="batch_runtime_summary",
        )

        with gr.Accordion("Model & Runtime", open=False, elem_id="batch_model_runtime_accordion"):
            with gr.Row(elem_id="batch_model_load_row"):
                batch_model_dropdown = gr.Dropdown(
                    ["Select Model"] + [file[0] for file in ckpt_files],
                    value=initial_name if initial_name else "Select Model",
                    label="Select Model",
                    scale=3,
                )
                batch_config_dropdown = gr.Dropdown(
                    initial_configs if initial_configs else ["Select Config"],
                    value=initial_config if initial_config else "Select Config",
                    label="Select Config",
                    scale=3,
                )
                batch_load_button = gr.Button("Load Model", variant="primary", scale=1)

            if bool(torchao_int4_supported):
                with gr.Accordion("Advanced model load options", open=False):
                    gr.Markdown(
                        "INT4 requires TorchAO support. Enable it only when reduced VRAM usage is necessary."
                    )
                    batch_int4_checkbox = gr.Checkbox(
                        label="Enable INT4 on load (TorchAO)",
                        value=False,
                        interactive=True,
                    )
            else:
                batch_int4_checkbox = gr.State(value=False)

            batch_model_info = gr.Markdown("")

        batch_model_dropdown.change(
            fn=lambda selected: update_config_dropdown(selected, ckpt_files),
            inputs=batch_model_dropdown,
            outputs=batch_config_dropdown,
        )

        with gr.Row(elem_id="batch_shared_controls_row"):
            batch_style_radio = gr.Radio(
                [FOUNDATION_MODE_SIMPLE, FOUNDATION_MODE_EXPERIMENTAL],
                value=FOUNDATION_MODE_SIMPLE,
                label="Prompt Mode",
                scale=1,
            )
            batch_wetdry_radio = gr.Radio(
                ["Dry", "Wet"],
                value="Wet",
                label="FX Toggle",
                scale=1,
            )
            batch_seed_textbox = gr.Textbox(
                label="Seed (-1 for random)",
                value="-1",
                scale=1,
            )

        with gr.Accordion("Advanced Generation Settings", open=False, elem_id="batch_advanced_settings"):
            with gr.Row():
                loop_steps_slider = gr.Slider(
                    minimum=1,
                    maximum=500,
                    step=1,
                    value=75,
                    label="Loop Steps",
                )
                oneshot_steps_slider = gr.Slider(
                    minimum=1,
                    maximum=500,
                    step=1,
                    value=80,
                    label="One-Shot Steps",
                )
                batch_cfg_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=25.0,
                    step=0.1,
                    value=6.0,
                    label="CFG Scale",
                )
            with gr.Row():
                batch_sampler_type_dropdown = gr.Dropdown(
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
                    value="dpmpp-3m-sde",
                )
                batch_sigma_min_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.01,
                    value=0.03,
                    label="Sigma Min",
                )
                batch_sigma_max_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1000.0,
                    step=0.1,
                    value=500,
                    label="Sigma Max",
                )
                batch_cfg_rescale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    label="CFG Rescale",
                )

        gr.HTML(
            """
            <style>
                #batch_generation_columns {
                    align-items: flex-start !important;
                    gap: 1rem !important;
                }

                #batch_loop_column,
                #batch_oneshot_column {
                    min-width: 0 !important;
                }

                #batch_loop_prompt textarea,
                #batch_oneshot_prompt textarea {
                    min-height: 9rem !important;
                    resize: vertical !important;
                }

                #batch_loop_title,
                #batch_oneshot_title {
                    width: 100% !important;
                    text-align: center !important;
                }

                #batch_loop_title > div,
                #batch_oneshot_title > div {
                    width: 100% !important;
                    text-align: center !important;
                }

                .batch-section-title {
                    margin: 0.35rem 0 0.15rem;
                    font-size: 1.08rem;
                    font-weight: 650;
                    text-align: center;
                }

                .batch-section-note {
                    margin: 0 0 0.8rem;
                    opacity: 0.8;
                    text-align: center;
                }
            </style>
            """
        )

        with gr.Row(equal_height=False, elem_id="batch_generation_columns"):
            # ------------------------------------------------------------
            # Loop Variations
            # ------------------------------------------------------------
            with gr.Column(scale=1, elem_id="batch_loop_column"):
                gr.HTML(
                    "<div class='batch-section-title'>Loop Variations</div>"
                    "<div class='batch-section-note'>Generate several interpretations of one musical prompt.</div>",
                    elem_id="batch_loop_title",
                )

                loop_prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe the loop, musical idea, sound palette, rhythm, or melody…",
                    lines=5,
                    elem_id="batch_loop_prompt",
                )

                with gr.Row():
                    random_loop_button = gr.Button("Random Loop Prompt", variant="secondary")
                    generate_loop_button = gr.Button("Generate Loop Variations", variant="primary")

                initial_loop_family = master_prompt_map.get_loop_prompt_family(initial_name)
                with gr.Accordion("Loop Prompt Builder", open=False, elem_id="batch_loop_prompt_builder"):
                    with gr.Column(visible=(initial_loop_family == "foundation")) as batch_foundation_builder_group:
                        gr.Markdown(
                            "Choose up to two unrestricted sound sources, then add timbre and musical tags. "
                            "The prompt textbox remains authoritative, so manually typed ideas are preserved."
                        )
                        with gr.Row():
                            foundation_source_1_picker = _searchable_dropdown(FOUNDATION_FLAT_SOURCE_CHOICES, label="Sound Source 1", value=None)
                            foundation_source_2_picker = _searchable_dropdown(FOUNDATION_SECOND_SOURCE_CHOICES, label="Sound Source 2 (Optional)", value=FOUNDATION_PROMPT_BUILDER_NONE)
                        foundation_timbre_picker = _searchable_dropdown(FOUNDATION_TIMBRE_CHOICES, label="Timbre / Sound Tags", value=[], multiselect=True)
                        with gr.Row():
                            foundation_structure_picker = _searchable_dropdown([FOUNDATION_PROMPT_BUILDER_NONE] + FOUNDATION_STRUCTURE_CHOICES, label="Musical Structure", value=FOUNDATION_PROMPT_BUILDER_NONE)
                            foundation_musical_picker = _searchable_dropdown(FOUNDATION_MUSICAL_CHOICES, label="Musical Tags", value=[], multiselect=True)
                        clear_foundation_builder_tags_button = gr.Button("Clear Selected Tags", variant="secondary")

                    with gr.Column(visible=(initial_loop_family == "piano")) as batch_piano_builder_group:
                        gr.Markdown("Legacy Infinite Pianos builder.")
                        with gr.Row():
                            batch_piano_type = _searchable_dropdown(piano_prompts.PIANO_TYPES, label="Piano Type", value=piano_prompts.PIANO_TYPES[0])
                            batch_piano_structure = _searchable_dropdown(piano_prompts.STRUCTURE_CHOICES, label="Structure", value=piano_prompts.STRUCTURE_CHOICES[0])
                        with gr.Row():
                            batch_piano_chord = _searchable_dropdown(["None"] + piano_prompts.CHORD_STYLES, label="Chord Style", value="None")
                            batch_piano_melody = _searchable_dropdown(["None"] + piano_prompts.MELODY_STYLES, label="Melody Style", value="None")
                        batch_piano_effect = _searchable_dropdown(piano_prompts.PIANO_EFFECT_CHOICES, label="Effect", value=piano_prompts.PIANO_EFFECT_CHOICES[0])

                    with gr.Column(visible=(initial_loop_family == "edm_elements")) as batch_edm_builder_group:
                        gr.Markdown("Legacy EDM Elements builder.")
                        batch_edm_sound = _searchable_dropdown(edm_elements_prompts.SOUND_TAG_CHOICES, label="Sound Tags", value=[], multiselect=True)
                        with gr.Row():
                            batch_edm_structure = _searchable_dropdown(["None"] + edm_elements_prompts.STRUCTURE_CHOICES, label="Musical Structure", value="None")
                            batch_edm_musical = _searchable_dropdown(edm_elements_prompts.MUSICAL_TAG_CHOICES, label="Musical Tags", value=[], multiselect=True)
                        batch_edm_effects = _searchable_dropdown(edm_elements_prompts.EFFECT_CHOICES, label="Legacy Effects", value=[], multiselect=True)

                    with gr.Column(visible=(initial_loop_family == "vocal_textures")) as batch_vocal_builder_group:
                        gr.Markdown("Legacy Vocal Textures builder.")
                        with gr.Row():
                            batch_vocal_type = _searchable_dropdown(vocal_textures_prompts.VOCAL_TYPES, label="Vocal Type", value=vocal_textures_prompts.VOCAL_TYPES[0])
                            batch_vocal_structure = _searchable_dropdown(vocal_textures_prompts.STRUCTURE_CHOICES, label="Structure", value=vocal_textures_prompts.STRUCTURE_CHOICES[0])

                with gr.Group():
                    with gr.Row():
                        loop_lock_bpm_checkbox = gr.Checkbox(label="Lock BPM Settings", value=True)
                        loop_lock_key_checkbox = gr.Checkbox(label="Lock Key Signature", value=True)
                    with gr.Row():
                        loop_bars_dropdown = gr.Dropdown([4, 8], label="Bars", value=8)
                        loop_bpm_dropdown = gr.Dropdown(
                            [100, 110, 120, 128, 130, 140, 150],
                            label="BPM",
                            value=128,
                        )
                    with gr.Row():
                        loop_key_dropdown = gr.Dropdown(
                            ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],
                            label="Key",
                            value="F",
                        )
                        loop_scale_dropdown = gr.Dropdown(
                            ["major", "minor"],
                            label="Scale",
                            value="minor",
                        )
                        loop_generation_count_dropdown = gr.Dropdown(
                            LOOP_BATCH_GEN_CHOICES,
                            label="Variations",
                            value=3,
                        )

                loop_audio_outputs: List[gr.Audio] = []
                for row_start in range(0, LOOP_BATCH_AUDIO_SLOT_COUNT, 2):
                    with gr.Row():
                        for slot in range(row_start + 1, min(row_start + 3, LOOP_BATCH_AUDIO_SLOT_COUNT + 1)):
                            loop_audio_outputs.append(
                                gr.Audio(
                                    label=f"Loop Gen {slot:02d}",
                                    interactive=False,
                                    visible=False,
                                )
                            )

                loop_status_output = gr.Markdown("")

            # ------------------------------------------------------------
            # One-Shot Variations
            # ------------------------------------------------------------
            with gr.Column(scale=1, elem_id="batch_oneshot_column"):
                gr.HTML(
                    "<div class='batch-section-title'>One-Shot Variations</div>"
                    "<div class='batch-section-note'>Generate several samples from one or two sound ideas and a note.</div>",
                    elem_id="batch_oneshot_title",
                )

                oneshot_prompt = gr.Textbox(
                    show_label=False,
                    placeholder="Describe the one-shot — for example: Grand Piano, Choir, Warm, Soft, Rich…",
                    lines=5,
                    elem_id="batch_oneshot_prompt",
                )

                with gr.Row():
                    random_oneshot_button = gr.Button("Random One-Shot Prompt", variant="secondary")
                    generate_oneshot_button = gr.Button("Generate One-Shot Variations", variant="primary")

                with gr.Accordion("One-Shot Prompt Builder", open=False, elem_id="oneshot_prompt_builder"):
                    gr.Markdown(
                        "Choose one or two unrestricted sound sources plus timbre tags. "
                        "One Shot generation is intended for models trained with one-shot support, though other checkpoints can still be used experimentally."
                    )
                    with gr.Row():
                        oneshot_instrument_1_picker = _searchable_dropdown(
                            ONESHOT_FLAT_SOURCE_CHOICES,
                            label="Sound Source 1",
                            value=None,
                        )
                        oneshot_instrument_2_picker = _searchable_dropdown(
                            ONESHOT_SECOND_SOURCE_CHOICES,
                            label="Sound Source 2 (Optional)",
                            value=ONESHOT_NO_SOURCE,
                        )
                    oneshot_timbre_picker = _searchable_dropdown(
                        ONESHOT_TIMBRE_CHOICES,
                        label="Timbre Tags",
                        value=[],
                        multiselect=True,
                    )
                    with gr.Row():
                        randomize_oneshot_sound_button = gr.Button("Randomize Sound", variant="secondary")
                        clear_oneshot_timbre_button = gr.Button("Clear Timbre Tags", variant="secondary")

                with gr.Group():
                    with gr.Row():
                        oneshot_lock_note_checkbox = gr.Checkbox(label="Lock Note", value=False)
                        oneshot_note_name_dropdown = gr.Dropdown(
                            ONESHOT_NOTE_NAMES,
                            label="Note",
                            value="F#",
                        )
                        oneshot_octave_dropdown = gr.Dropdown(
                            ONESHOT_OCTAVE_CHOICES,
                            label="Octave",
                            value="4",
                        )
                        oneshot_generation_count_dropdown = gr.Dropdown(
                            ONESHOT_BATCH_GEN_CHOICES,
                            label="Variations",
                            value=3,
                        )

                oneshot_audio_outputs: List[gr.Audio] = []
                for row_start in range(0, ONESHOT_BATCH_AUDIO_SLOT_COUNT, 2):
                    with gr.Row():
                        for slot in range(row_start + 1, min(row_start + 3, ONESHOT_BATCH_AUDIO_SLOT_COUNT + 1)):
                            oneshot_audio_outputs.append(
                                gr.Audio(
                                    label=f"One-Shot Gen {slot:02d}",
                                    interactive=False,
                                    visible=False,
                                )
                            )

                oneshot_status_output = gr.Markdown("")

        # Model-specific legacy loop builders. One-Shot remains available regardless of checkpoint.
        batch_piano_inputs = [batch_piano_type, batch_piano_structure, batch_piano_chord, batch_piano_melody, batch_piano_effect]

        def _batch_change_piano_type(piano_type, structure, chord_style, melody_style, effect):
            allowed = piano_prompts.effect_choices_for_piano(piano_type)
            resolved_effect = effect if effect in allowed else allowed[0]
            return (
                piano_prompts.build_prompt(piano_type, structure, chord_style, melody_style, resolved_effect),
                gr.update(choices=allowed, value=resolved_effect),
            )

        batch_piano_type.change(
            fn=_batch_change_piano_type,
            inputs=batch_piano_inputs,
            outputs=[loop_prompt, batch_piano_effect],
            queue=False,
            show_progress="hidden",
        )
        for component in batch_piano_inputs[1:]:
            component.input(fn=piano_prompts.build_prompt, inputs=batch_piano_inputs, outputs=loop_prompt, queue=False, show_progress="hidden")

        batch_edm_inputs = [batch_edm_sound, batch_edm_structure, batch_edm_musical, batch_edm_effects]
        for component in batch_edm_inputs:
            component.input(fn=edm_elements_prompts.build_prompt, inputs=batch_edm_inputs, outputs=loop_prompt, queue=False, show_progress="hidden")

        batch_vocal_inputs = [batch_vocal_type, batch_vocal_structure]
        for component in batch_vocal_inputs:
            component.input(fn=vocal_textures_prompts.build_prompt, inputs=batch_vocal_inputs, outputs=loop_prompt, queue=False, show_progress="hidden")

        batch_load_button.click(
            fn=_load_batch_model,
            inputs=[batch_model_dropdown, batch_config_dropdown, batch_int4_checkbox],
            outputs=[batch_model_info, runtime_md, batch_foundation_builder_group, batch_piano_builder_group, batch_edm_builder_group, batch_vocal_builder_group],
        )

        # Prompt-generation callbacks ------------------------------------
        random_loop_button.click(
            fn=_random_loop_for_runtime,
            inputs=[
                batch_style_radio,
                batch_wetdry_radio,
                batch_seed_textbox,
                loop_lock_bpm_checkbox,
                loop_bars_dropdown,
                loop_bpm_dropdown,
                loop_lock_key_checkbox,
                loop_key_dropdown,
                loop_scale_dropdown,
            ],
            outputs=[
                loop_prompt,
                loop_bars_dropdown,
                loop_bpm_dropdown,
                loop_key_dropdown,
                loop_scale_dropdown,
                foundation_source_1_picker,
                foundation_source_2_picker,
                foundation_timbre_picker,
                foundation_structure_picker,
                foundation_musical_picker,
                batch_piano_type,
                batch_piano_structure,
                batch_piano_chord,
                batch_piano_melody,
                batch_piano_effect,
                batch_edm_sound,
                batch_edm_structure,
                batch_edm_musical,
                batch_edm_effects,
                batch_vocal_type,
                batch_vocal_structure,
            ],
        )

        foundation_builder_inputs = [
            loop_prompt,
            foundation_source_1_picker,
            foundation_source_2_picker,
            foundation_timbre_picker,
            foundation_structure_picker,
            foundation_musical_picker,
        ]
        foundation_source_1_picker.input(
            fn=apply_foundation_prompt_builder_action,
            inputs=foundation_builder_inputs,
            outputs=loop_prompt,
        )
        foundation_source_2_picker.input(
            fn=apply_foundation_prompt_builder_action,
            inputs=foundation_builder_inputs,
            outputs=loop_prompt,
        )
        foundation_timbre_picker.input(
            fn=apply_foundation_prompt_builder_action,
            inputs=foundation_builder_inputs,
            outputs=loop_prompt,
        )
        foundation_structure_picker.input(
            fn=apply_foundation_prompt_builder_action,
            inputs=foundation_builder_inputs,
            outputs=loop_prompt,
        )
        foundation_musical_picker.input(
            fn=apply_foundation_prompt_builder_action,
            inputs=foundation_builder_inputs,
            outputs=loop_prompt,
        )
        clear_foundation_builder_tags_button.click(
            fn=clear_foundation_prompt_builder_tags_action,
            inputs=[
                loop_prompt,
                foundation_source_1_picker,
                foundation_source_2_picker,
                foundation_structure_picker,
            ],
            outputs=[loop_prompt, foundation_timbre_picker, foundation_musical_picker],
        )

        oneshot_random_inputs = [
            batch_style_radio,
            batch_wetdry_radio,
            batch_seed_textbox,
            oneshot_lock_note_checkbox,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
        ]
        oneshot_random_outputs = [
            oneshot_prompt,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
            oneshot_instrument_1_picker,
            oneshot_instrument_2_picker,
            oneshot_timbre_picker,
        ]
        random_oneshot_button.click(
            fn=batch_random_oneshot_prompt_with_lock_action,
            inputs=oneshot_random_inputs,
            outputs=oneshot_random_outputs,
        )
        randomize_oneshot_sound_button.click(
            fn=batch_random_oneshot_prompt_with_lock_action,
            inputs=oneshot_random_inputs,
            outputs=oneshot_random_outputs,
        )

        # .input() runs only for direct user edits. Programmatic Random Prompt
        # updates will not trigger a second note roll or descriptor rebuild.
        oneshot_source_change_inputs = [
            oneshot_prompt,
            oneshot_instrument_1_picker,
            oneshot_instrument_2_picker,
            oneshot_timbre_picker,
            batch_wetdry_radio,
            oneshot_lock_note_checkbox,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
            batch_seed_textbox,
        ]
        oneshot_source_change_outputs = [
            oneshot_prompt,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
        ]
        oneshot_instrument_1_picker.input(
            fn=oneshot_source_change_action,
            inputs=oneshot_source_change_inputs,
            outputs=oneshot_source_change_outputs,
        )
        oneshot_instrument_2_picker.input(
            fn=oneshot_source_change_action,
            inputs=oneshot_source_change_inputs,
            outputs=oneshot_source_change_outputs,
        )
        oneshot_timbre_picker.input(
            fn=apply_oneshot_prompt_builder_action,
            inputs=[
                oneshot_prompt,
                oneshot_instrument_1_picker,
                oneshot_instrument_2_picker,
                oneshot_timbre_picker,
                batch_wetdry_radio,
            ],
            outputs=oneshot_prompt,
        )
        clear_oneshot_timbre_button.click(
            fn=clear_oneshot_timbre_action,
            inputs=[
                oneshot_prompt,
                oneshot_instrument_1_picker,
                oneshot_instrument_2_picker,
                batch_wetdry_radio,
            ],
            outputs=[oneshot_prompt, oneshot_timbre_picker],
        )
        batch_wetdry_radio.change(
            fn=oneshot_wetdry_change_action,
            inputs=[
                oneshot_prompt,
                oneshot_instrument_1_picker,
                oneshot_instrument_2_picker,
                oneshot_timbre_picker,
                batch_wetdry_radio,
            ],
            outputs=[oneshot_prompt, oneshot_timbre_picker],
        )


        # Exclusive generation callbacks ---------------------------------
        loop_start = generate_loop_button.click(
            fn=_prepare_loop_batch_ui,
            inputs=[],
            outputs=[
                generate_loop_button,
                generate_oneshot_button,
                *loop_audio_outputs,
            ],
            queue=True,
            show_progress="hidden",
        )
        loop_run = loop_start.then(
            fn=lambda *args: _run_batch_exclusive(
                batch_generate_loop_action,
                *args,
                get_runtime=get_runtime,
                generate_cond=generate_cond,
            ),
            inputs=[
                loop_prompt,
                batch_style_radio,
                batch_wetdry_radio,
                batch_seed_textbox,
                loop_bars_dropdown,
                loop_bpm_dropdown,
                loop_key_dropdown,
                loop_scale_dropdown,
                loop_generation_count_dropdown,
                loop_steps_slider,
                batch_cfg_scale_slider,
                batch_sampler_type_dropdown,
                batch_sigma_min_slider,
                batch_sigma_max_slider,
                batch_cfg_rescale_slider,
            ],
            outputs=[
                loop_prompt,
                *loop_audio_outputs,
                loop_status_output,
            ],
            concurrency_id="batch_generation_diffusion",
            concurrency_limit=1,
            show_progress="full",
        )
        loop_run.then(
            fn=_enable_batch_generation_buttons,
            inputs=[],
            outputs=[generate_loop_button, generate_oneshot_button],
            queue=True,
            show_progress="hidden",
        )

        oneshot_start = generate_oneshot_button.click(
            fn=_prepare_oneshot_batch_ui,
            inputs=[],
            outputs=[
                generate_loop_button,
                generate_oneshot_button,
                *oneshot_audio_outputs,
            ],
            queue=True,
            show_progress="hidden",
        )
        oneshot_run = oneshot_start.then(
            fn=lambda *args: _run_batch_exclusive(
                batch_generate_oneshot_action,
                *args,
                get_runtime=get_runtime,
                generate_cond=generate_cond,
            ),
            inputs=[
                oneshot_prompt,
                batch_style_radio,
                batch_wetdry_radio,
                batch_seed_textbox,
                oneshot_note_name_dropdown,
                oneshot_octave_dropdown,
                oneshot_generation_count_dropdown,
                oneshot_steps_slider,
                batch_cfg_scale_slider,
                batch_sampler_type_dropdown,
                batch_sigma_min_slider,
                batch_sigma_max_slider,
                batch_cfg_rescale_slider,
            ],
            outputs=[
                oneshot_prompt,
                *oneshot_audio_outputs,
                oneshot_status_output,
            ],
            concurrency_id="batch_generation_diffusion",
            concurrency_limit=1,
            show_progress="full",
        )
        oneshot_run.then(
            fn=_enable_batch_generation_buttons,
            inputs=[],
            outputs=[generate_loop_button, generate_oneshot_button],
            queue=True,
            show_progress="hidden",
        )

    def refresh_batch_tab_from_runtime():
        return _current_runtime_summary(), "", *_loop_builder_visibility()

    return {
        "refresh_fn": refresh_batch_tab_from_runtime,
        "refresh_inputs": [],
        "refresh_outputs": [runtime_md, batch_model_info, batch_foundation_builder_group, batch_piano_builder_group, batch_edm_builder_group, batch_vocal_builder_group],
    }
