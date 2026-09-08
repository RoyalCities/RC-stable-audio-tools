import gc
import platform
import os
import time
import numpy as np
import gradio as gr
import json
import torch
import torchaudio
import random
import math
import re
import hashlib
import inspect



from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T


from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict
from .prompts import (
    master_prompt_map, foundation_prompts, oneshot_prompts,
    piano_prompts, edm_elements_prompts, vocal_textures_prompts,
)
from .keybed_tab import build_keybed_tab
from .layered_keybed_tab import build_layered_keybed_tab
from .batch_generation_tab import build_batch_generation_tab

import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH
from huggingface_hub import snapshot_download

# Load config file
with open("config.json") as config_file:
    config = json.load(config_file)


# Keep model downloads lean: checkpoints + lightweight repo metadata only.
MODEL_DOWNLOAD_PATTERNS = [
    "*.safetensors",
    "*.ckpt",
    "*.json",
    "README",
    "README.*",
    "LICENSE",
    "LICENSE.*",
]


def _model_download_options():
    """Return the configured Hugging Face repo choices and destination root."""
    entries = config.get("hffs") or []
    if not entries:
        return [], config.get("models_directory", "models")

    entry = entries[0] or {}
    repos = [str(repo).strip() for repo in (entry.get("options") or []) if str(repo).strip()]
    destination_root = entry.get("path") or config.get("models_directory", "models")
    return repos, destination_root


def download_model_repo(repo_id: str):
    """Download only the files needed to use/document a model repo."""
    repo_id = str(repo_id or "").strip()
    repos, destination_root = _model_download_options()

    if not repo_id or repo_id not in repos:
        return "**Download failed:** Select a model repository first."

    # Preserve the folder naming used by the existing HFFS downloader.
    folder_name = repo_id.replace("/", "-")
    local_dir = os.path.join(destination_root, folder_name)
    os.makedirs(local_dir, exist_ok=True)

    try:
        print(f"Downloading {repo_id} to {local_dir}")
        print(f"Allowed files: {MODEL_DOWNLOAD_PATTERNS}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=MODEL_DOWNLOAD_PATTERNS,
        )

        downloaded = []
        for root, _, files in os.walk(local_dir):
            for filename in files:
                # Do not report Hugging Face's tiny local cache metadata as model files.
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, local_dir)
                if rel_path.startswith(".cache" + os.sep):
                    continue
                downloaded.append(rel_path)

        downloaded.sort()
        file_summary = "  \n".join(f"- `{path}`" for path in downloaded) if downloaded else "- No matching files found."
        return (
            f"**Download complete:** `{repo_id}`  \n"
            f"Saved to: `{local_dir}`  \n\n"
            f"{file_summary}  \n\n"
            "Restart the app to detect/load the downloaded model."
        )
    except Exception as exc:
        print(f"Model download failed: {type(exc).__name__}: {exc}")
        return f"**Download failed:** `{type(exc).__name__}: {exc}`"


def build_model_download_ui(*, initialize=False):
    """Build the lightweight model downloader used by both startup and the tab."""
    repos, _ = _model_download_options()

    if initialize:
        gr.HTML("<h2>Initialize</h2><div>No checkpoint was found. Download a model, then restart the app.</div>")
    else:
        gr.HTML("<h2>Download</h2><div>Download a model and restart the app to apply.</div>")

    repo_dropdown = gr.Dropdown(
        choices=repos,
        value=repos[0] if repos else None,
        label="Model Repository",
    )
    download_button = gr.Button("Download Model", variant="primary")
    download_status = gr.Markdown()

    download_button.click(
        fn=download_model_repo,
        inputs=[repo_dropdown],
        outputs=[download_status],
    )

model = None
sample_rate = 32000
sample_size = 1920000
DEVICE = None
global_model_half = False
BEATS_PER_BAR = 4

#torch ao int4 /model controls
# --- runtime / precision globals ---
PREFERRED_DTYPE = torch.float32
TORCHAO_INT4_SUPPORTED = False
INT4_ENABLED = False

LAST_CKPT_PATH = None
LAST_CONFIG_PATH = None
LAST_CKPT_NAME = None
LAST_MODEL_CONFIG = None  

# torchao int preset
TORCHAO_WBITS = 4   

# --- Foundation prompt modes (user-facing) ---
FOUNDATION_MODE_SIMPLE = "Simple"
FOUNDATION_MODE_EXPERIMENTAL = "Experimental"

FOUNDATION_MODE_TO_VARIANT = {
    FOUNDATION_MODE_SIMPLE: "M1",          
    FOUNDATION_MODE_EXPERIMENTAL: "T1",    
}

FOUNDATION_MODE_HELP = {
    FOUNDATION_MODE_SIMPLE: (
        "`Predictable Prompts - less timbre-mix/chaos`"
    ),
    FOUNDATION_MODE_EXPERIMENTAL: (
        "`Adventurous Prompts - more timbre-mix`"
    ),
}

# --- Dual-capability / ONESHOT UI controls ---
SAMPLE_TYPE_LOOP = "Loop"
SAMPLE_TYPE_ONESHOT = "One Shot"

ONESHOT_SECONDS_TOTAL = 2.0

# One-shot tail trim is intentionally lightweight and torch-only.
# It only trims silence from the END of the generated 2s clip, then applies
# a tiny fade so plucks/hits do not click after trimming. Loops remain
# sample-exact to bars/BPM and do not use this trim.
ONESHOT_TAIL_TRIM_DB = -60.0
ONESHOT_TAIL_TRIM_FRAME_MS = 10.0
ONESHOT_TAIL_TRIM_PAD_MS = 60.0
ONESHOT_TAIL_TRIM_FADE_MS = 8.0
ONESHOT_TAIL_TRIM_MIN_KEEP_MS = 120.0

ONESHOT_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ONESHOT_OCTAVE_CHOICES = [str(octave) for octave in range(0, 8)]
ONESHOT_RANDOM_NOTE_CHOICES = [
    f"{name}{octave}"
    for octave in range(0, 8)
    for name in ONESHOT_NOTE_NAMES
    if not (octave == 7 and name in {"A", "A#", "B"})
]

ONESHOT_REGISTER_LABELS = [
    "Sub Register",
    "Low Register",
    "Medium Register",
    "High Register",
    "Top Register",
]

NOTE_TOKEN_RE = re.compile(r"^[A-G](?:#|b)?-?\d+$", re.IGNORECASE)
VISIBLE_WETDRY_TOKENS = {"wet", "dry"}


def _protect_mps_audio_range(audio: torch.Tensor) -> torch.Tensor:
    """Prevent MPS-only out-of-range inference peaks from being hard-clipped.

    CUDA/CPU behavior remains unchanged. On Apple MPS, audio is attenuated only
    when the generated waveform actually exceeds full scale; already-valid MPS
    output is left untouched. The final clamp remains a last-resort safety guard.
    """
    audio = audio.to(torch.float32)

    if audio.device.type == "mps":
        peak = audio.abs().max()
        peak_is_finite = bool(torch.isfinite(peak).item())
        peak_value = float(peak.item()) if peak_is_finite else 0.0

        if peak_is_finite and peak_value > 1.0:
            print(
                f"[MPS Audio] Peak {peak_value:.4f} exceeds full scale; "
                "applying overflow protection.",
                flush=True,
            )
            audio = audio * (0.999 / peak)

    return audio.clamp(-1.0, 1.0)


def is_visible_wetdry_token(token: str) -> bool:
    return str(token or "").strip().lower() in VISIBLE_WETDRY_TOKENS


def is_oneshot_fx_token(token: str) -> bool:
    value = str(token or "").strip().lower()
    return any(word in value for word in ("reverb", "delay", "distortion", "phaser", "bitcrush"))


def is_oneshot_model_name(model_name: str | None) -> bool:
    return master_prompt_map.is_oneshot_capable_model(model_name)

def is_foundation_model_name(model_name: str | None) -> bool:
    return master_prompt_map.is_foundation_model(model_name)


def normalize_ui_sample_type(sample_type: str | None) -> str:
    value = str(sample_type or SAMPLE_TYPE_LOOP).strip().lower()
    if value in {"one shot", "one-shot", "oneshot", "shot", "single"}:
        return "oneshot"
    return "loop"


def prompt_mode_from_style(simple_checked: bool, experimental_checked: bool) -> tuple[str, str, bool]:
    mode_label = FOUNDATION_MODE_EXPERIMENTAL if experimental_checked else FOUNDATION_MODE_SIMPLE
    mode_arg = "experimental" if mode_label == FOUNDATION_MODE_EXPERIMENTAL else "standard"
    variant = FOUNDATION_MODE_TO_VARIANT[mode_label]
    allow_timbre_mix = (mode_label == FOUNDATION_MODE_EXPERIMENTAL)
    return mode_arg, variant, allow_timbre_mix


def sample_type_from_toggles(loop_checked: bool, oneshot_checked: bool) -> str:
    return SAMPLE_TYPE_ONESHOT if bool(oneshot_checked) else SAMPLE_TYPE_LOOP


def wetdry_from_toggles(dry_checked: bool, wet_checked: bool) -> str:
    return "Wet" if bool(wet_checked) else "Dry"


def normalize_prompt_tokens(prompt: str) -> list[str]:
    return [part.strip() for part in str(prompt or "").split(",") if part and part.strip()]


def strip_wetdry_tokens(prompt: str) -> str:
    """
    Textbox prompt should stay descriptor-only.
    Wet/Dry is selected by the FX Toggle and injected into conditioning later.
    FX detail tokens like Medium Reverb are intentionally preserved.
    """
    return ", ".join([
        token for token in normalize_prompt_tokens(prompt)
        if not is_visible_wetdry_token(token)
    ])


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


def _dedupe_casefold(tokens):
    seen = set()
    out = []
    for token in tokens or []:
        value = str(token or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


PROMPT_BUILDER_NONE = "None"


def _safe_prompt_builder_choices(getter, fallback):
    try:
        return list(getter())
    except Exception:
        return list(fallback)


FOUNDATION_BUILDER_SOURCES = _safe_prompt_builder_choices(
    foundation_prompts.foundation_flat_source_choices,
    list(getattr(foundation_prompts, "FAMILIES", [])),
)
FOUNDATION_BUILDER_SOURCE_KEYS = {value.casefold(): value for value in FOUNDATION_BUILDER_SOURCES}
FOUNDATION_BUILDER_OPTIONAL_SOURCES = [PROMPT_BUILDER_NONE] + FOUNDATION_BUILDER_SOURCES
FOUNDATION_BUILDER_TIMBRE_TAGS = _safe_prompt_builder_choices(
    foundation_prompts.foundation_timbre_tag_choices,
    list(getattr(foundation_prompts, "TIMBRE_TAGS", [])),
)
FOUNDATION_BUILDER_TIMBRE_KEYS = {value.casefold(): value for value in FOUNDATION_BUILDER_TIMBRE_TAGS}
FOUNDATION_BUILDER_STRUCTURES = _safe_prompt_builder_choices(
    foundation_prompts.foundation_musical_structure_choices,
    list(getattr(foundation_prompts, "STRUCTURE_BASS", [])),
)
FOUNDATION_BUILDER_STRUCTURE_KEYS = {value.casefold(): value for value in FOUNDATION_BUILDER_STRUCTURES}
FOUNDATION_BUILDER_MUSICAL_TAGS = _safe_prompt_builder_choices(
    foundation_prompts.foundation_musical_tag_choices,
    list(getattr(foundation_prompts, "SPEED", []))
    + list(getattr(foundation_prompts, "RHYTHM", []))
    + list(getattr(foundation_prompts, "CONTOUR", []))
    + list(getattr(foundation_prompts, "DENSITY", [])),
)
FOUNDATION_BUILDER_MUSICAL_KEYS = {value.casefold(): value for value in FOUNDATION_BUILDER_MUSICAL_TAGS}

ONESHOT_BUILDER_SOURCES = _safe_prompt_builder_choices(
    oneshot_prompts.oneshot_flat_source_choices,
    list(getattr(oneshot_prompts, "ONESHOT_FAMILIES", [])),
)
ONESHOT_BUILDER_SOURCE_KEYS = {value.casefold(): value for value in ONESHOT_BUILDER_SOURCES}
ONESHOT_BUILDER_OPTIONAL_SOURCES = [PROMPT_BUILDER_NONE] + ONESHOT_BUILDER_SOURCES
ONESHOT_BUILDER_TIMBRE_TAGS = _safe_prompt_builder_choices(
    oneshot_prompts.oneshot_timbre_tag_choices,
    [name for name, _weight in getattr(oneshot_prompts, "ONESHOT_TIMBRE_TAGS", [])],
)
ONESHOT_BUILDER_TIMBRE_KEYS = {value.casefold(): value for value in ONESHOT_BUILDER_TIMBRE_TAGS}


PIANO_BUILDER_TYPES = list(piano_prompts.PIANO_TYPES)
PIANO_BUILDER_STRUCTURES = list(piano_prompts.STRUCTURE_CHOICES)
PIANO_BUILDER_CHORD_STYLES = [PROMPT_BUILDER_NONE] + list(piano_prompts.CHORD_STYLES)
PIANO_BUILDER_MELODY_STYLES = [PROMPT_BUILDER_NONE] + list(piano_prompts.MELODY_STYLES)
PIANO_BUILDER_EFFECTS = list(piano_prompts.PIANO_EFFECT_CHOICES)

EDM_BUILDER_SOUND_TAGS = list(edm_elements_prompts.SOUND_TAG_CHOICES)
EDM_BUILDER_STRUCTURES = [PROMPT_BUILDER_NONE] + list(edm_elements_prompts.STRUCTURE_CHOICES)
EDM_BUILDER_MUSICAL_TAGS = list(edm_elements_prompts.MUSICAL_TAG_CHOICES)
EDM_BUILDER_EFFECTS = list(edm_elements_prompts.EFFECT_CHOICES)

VOCAL_BUILDER_TYPES = list(vocal_textures_prompts.VOCAL_TYPES)
VOCAL_BUILDER_STRUCTURES = list(vocal_textures_prompts.STRUCTURE_CHOICES)


def _normalize_builder_source(value, source_keys):
    text = str(value or "").strip()
    if not text or text.casefold() == PROMPT_BUILDER_NONE.casefold():
        return None
    return source_keys.get(text.casefold(), text)


def _enforce_last_mutex(tokens, groups):
    out = list(tokens or [])
    for group in groups or []:
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


def _consume_leading_builder_sources(
    body,
    source_keys,
    tag_keys,
    *,
    expected_source_1=None,
    expected_source_2=None,
):
    remaining = list(body or [])
    sources = []
    expected = [
        _normalize_builder_source(expected_source_1, source_keys),
        _normalize_builder_source(expected_source_2, source_keys),
    ]

    for slot in range(2):
        if not remaining:
            break
        token = str(remaining[0])
        key = token.casefold()
        canonical = source_keys.get(key)
        if canonical is None:
            break

        expected_value = expected[slot]
        if expected_value and key == expected_value.casefold():
            sources.append(expected_value)
            remaining.pop(0)
            continue

        if slot == 0:
            # The first recognized token is the primary source.
            sources.append(canonical)
            remaining.pop(0)
            continue

        # Some vocab overlaps are legitimate tags (Bass/Sub Bass/Supersaw on
        # Foundation; Choir/Bell/Pluck on One Shot). Only auto-consume an
        # unselected second source when it is not also a known tag.
        if key not in tag_keys:
            sources.append(canonical)
            remaining.pop(0)
            continue
        break

    return remaining, sources


def _split_foundation_builder_prompt(prompt, expected_source_1=None, expected_source_2=None):
    body = []
    fx = []
    for token in normalize_prompt_tokens(prompt):
        if is_visible_wetdry_token(token):
            continue
        if is_oneshot_fx_token(token):
            fx.append(token)
        else:
            body.append(token)

    body, sources = _consume_leading_builder_sources(
        body,
        FOUNDATION_BUILDER_SOURCE_KEYS,
        FOUNDATION_BUILDER_TIMBRE_KEYS,
        expected_source_1=expected_source_1,
        expected_source_2=expected_source_2,
    )

    structure_index = None
    structure = None
    for index, token in enumerate(body):
        canonical = FOUNDATION_BUILDER_STRUCTURE_KEYS.get(token.casefold())
        if canonical is not None:
            structure_index = index
            structure = canonical

    manual = []
    timbre = []
    musical = []
    for index, token in enumerate(body):
        if index == structure_index:
            continue
        key = token.casefold()
        if key in FOUNDATION_BUILDER_TIMBRE_KEYS:
            timbre.append(FOUNDATION_BUILDER_TIMBRE_KEYS[key])
        elif key in FOUNDATION_BUILDER_MUSICAL_KEYS:
            musical.append(FOUNDATION_BUILDER_MUSICAL_KEYS[key])
        else:
            manual.append(token)
    return manual, fx, sources, timbre, structure, musical


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
    selected_1 = _normalize_builder_source(source_1, FOUNDATION_BUILDER_SOURCE_KEYS)
    selected_2 = _normalize_builder_source(source_2, FOUNDATION_BUILDER_SOURCE_KEYS)
    if selected_1 and selected_2 and selected_1.casefold() == selected_2.casefold():
        selected_2 = None

    selected_timbre = [
        FOUNDATION_BUILDER_TIMBRE_KEYS[str(tag).casefold()]
        for tag in _dedupe_casefold(timbre_tags)
        if str(tag).casefold() in FOUNDATION_BUILDER_TIMBRE_KEYS
    ]
    selected_timbre = _enforce_last_mutex(
        selected_timbre,
        [{"pizzicato", "staccato", "spiccato"}],
    )

    structure = str(musical_structure or "").strip()
    structure = FOUNDATION_BUILDER_STRUCTURE_KEYS.get(structure.casefold()) if structure else None
    selected_musical = [
        FOUNDATION_BUILDER_MUSICAL_KEYS[str(tag).casefold()]
        for tag in _dedupe_casefold(musical_tags)
        if str(tag).casefold() in FOUNDATION_BUILDER_MUSICAL_KEYS
    ]

    tokens = []
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


def _split_oneshot_builder_prompt(prompt, expected_source_1=None, expected_source_2=None):
    body = []
    fx = []
    for token in normalize_prompt_tokens(prompt):
        if NOTE_TOKEN_RE.match(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if is_visible_wetdry_token(token):
            continue
        if token.casefold() in {"one shot", "oneshot", "one-shot"}:
            continue
        if is_oneshot_fx_token(token):
            fx.append(token)
        else:
            body.append(token)

    body, sources = _consume_leading_builder_sources(
        body,
        ONESHOT_BUILDER_SOURCE_KEYS,
        ONESHOT_BUILDER_TIMBRE_KEYS,
        expected_source_1=expected_source_1,
        expected_source_2=expected_source_2,
    )
    return body, fx, sources


def apply_oneshot_prompt_builder_action(current_prompt, source_1, source_2, timbre_tags):
    body, fx, _old_sources = _split_oneshot_builder_prompt(
        current_prompt,
        expected_source_1=source_1,
        expected_source_2=source_2,
    )
    manual = [token for token in body if token.casefold() not in ONESHOT_BUILDER_TIMBRE_KEYS]

    selected_1 = _normalize_builder_source(source_1, ONESHOT_BUILDER_SOURCE_KEYS)
    selected_2 = _normalize_builder_source(source_2, ONESHOT_BUILDER_SOURCE_KEYS)
    if selected_1 and selected_2 and selected_1.casefold() == selected_2.casefold():
        selected_2 = None

    selected_tags = [
        ONESHOT_BUILDER_TIMBRE_KEYS[str(tag).casefold()]
        for tag in _dedupe_casefold(timbre_tags)
        if str(tag).casefold() in ONESHOT_BUILDER_TIMBRE_KEYS
    ]
    selected_tags = _enforce_last_mutex(
        selected_tags,
        getattr(oneshot_prompts, "ONESHOT_TAG_MUTEX_GROUPS", []),
    )

    tokens = []
    if selected_1:
        tokens.append(selected_1)
    if selected_2:
        tokens.append(selected_2)
    tokens.extend(manual)
    tokens.extend(selected_tags)
    tokens.extend(fx)
    return ", ".join(_dedupe_casefold(tokens))


def _suggest_main_oneshot_note(source_1, source_2, seed_str):
    try:
        seed = None if str(seed_str or "").strip() in {"", "-1"} else int(seed_str)
    except Exception:
        seed = None
    rng = random.Random(seed)
    try:
        _register, note = oneshot_prompts.pick_oneshot_note_for_sources(
            rng,
            instrument_1=_normalize_builder_source(source_1, ONESHOT_BUILDER_SOURCE_KEYS),
            instrument_2=_normalize_builder_source(source_2, ONESHOT_BUILDER_SOURCE_KEYS),
        )
    except Exception:
        note = "F#4"
    return split_oneshot_note(note)


def main_oneshot_source_change_action(
    current_prompt,
    source_1,
    source_2,
    timbre_tags,
    lock_note,
    current_note_name,
    current_octave,
    seed_str,
):
    prompt = apply_oneshot_prompt_builder_action(current_prompt, source_1, source_2, timbre_tags)
    if bool(lock_note):
        return prompt, str(current_note_name or "F#"), str(current_octave or "4")
    note_name, octave = _suggest_main_oneshot_note(source_1, source_2, seed_str)
    return prompt, note_name, octave


def clear_oneshot_prompt_builder_tags_action(current_prompt, source_1, source_2):
    prompt = apply_oneshot_prompt_builder_action(current_prompt, source_1, source_2, [])
    return prompt, gr.update(value=[])



def get_generation_output_dir(
    resolved_sample_type: str,
    parent_subdir: str | None = None,
    run_subdir: str | None = None,
) -> str:
    """Return and create the final save folder for this generation type.

    Normal Generation writes to:
      generations/Loops
      generations/One_Shots

    Batch Generation can pass parent_subdir="Batch_Generation" and an
    optional run_subdir so grouped runs land under:
      generations/Batch_Generation/Loops/<run>/
      generations/Batch_Generation/One_Shots/<run>/
    """
    subdir = ONESHOT_OUTPUT_SUBDIR if resolved_sample_type == "oneshot" else LOOP_OUTPUT_SUBDIR
    parts = [output_directory]
    if parent_subdir:
        parts.append(str(parent_subdir))
    parts.append(subdir)
    if run_subdir:
        # Keep caller-provided grouping local to the sample type folder.
        parts.append(_safe_filename_part(str(run_subdir), max_chars=72))
    save_dir = os.path.join(*parts)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _safe_filename_part(value: str, *, max_chars: int = 40) -> str:
    """Cross-platform safe filename chunk. Keeps names short and readable."""
    value = str(value or "").strip().lower()
    value = value.replace("#", "sharp")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value[:max_chars].strip("_") or "sample")


def _is_loop_control_token(token: str) -> bool:
    t = str(token or "").strip()
    if re.match(r"^\d+\s*BPM$", t, re.IGNORECASE):
        return True
    if re.match(r"^\d+\s*Bars?$", t, re.IGNORECASE):
        return True
    if re.match(r"^[A-G](?:#|b)?\s+(?:major|minor)$", t, re.IGNORECASE):
        return True
    return False


def descriptor_slug_from_prompt(amended_prompt: str, resolved_sample_type: str) -> str:
    """
    Build a short readable slug from the instrument/sub-instrument area.
    Control tokens stay out of filenames; full prompt goes in the sidecar.
    """
    usable_tokens = []
    for token in normalize_prompt_tokens(amended_prompt):
        low = token.strip().lower()
        if low in {"one shot", "oneshot", "one-shot"}:
            continue
        if is_visible_wetdry_token(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if NOTE_TOKEN_RE.match(token):
            continue
        if _is_loop_control_token(token):
            continue
        if is_oneshot_fx_token(token):
            continue
        usable_tokens.append(token)

    # Usually this becomes e.g. synth_pluck, keys_grand_piano, bass_reese_bass.
    return _safe_filename_part("_".join(usable_tokens[:2]) or resolved_sample_type)


def unique_generation_stem(save_dir: str, amended_prompt: str, seed: int, resolved_sample_type: str) -> str:
    slug = descriptor_slug_from_prompt(amended_prompt, resolved_sample_type)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"{time.time_ns()}|{seed}|{amended_prompt}".encode("utf-8")).hexdigest()[:8]
    base = f"{stamp}_{slug}_{digest}"

    stem = base
    counter = 1
    while any(os.path.exists(os.path.join(save_dir, f"{stem}{ext}")) for ext in (".wav", ".mid", ".txt")):
        counter += 1
        stem = f"{base}_{counter}"
    return stem


def find_latest_midi_for_stem(save_dir: str, file_stem: str) -> str | None:
    candidates = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.lower().endswith(".mid") and file_stem in f
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getctime)
    return candidates[-1]


def normalize_midi_filename(midi_path: str | None, save_dir: str, file_stem: str) -> str | None:
    """Rename the Basic Pitch MIDI to match the WAV/TXT stem when possible."""
    if midi_path is None:
        return None

    target_path = os.path.join(save_dir, f"{file_stem}.mid")
    try:
        if os.path.abspath(midi_path) != os.path.abspath(target_path):
            if os.path.exists(target_path):
                os.remove(target_path)
            os.replace(midi_path, target_path)
        return target_path
    except Exception as e:
        print(f"Could not normalize MIDI filename: {e}")
        return midi_path


def write_generation_sidecar(
    sidecar_path: str,
    *,
    file_stem: str,
    resolved_sample_type: str,
    visible_prompt: str,
    conditioning_prompt: str,
    seed: int,
    wetdry: str,
    audio_path: str,
    midi_path: str | None,
    bars=None,
    bpm=None,
    note=None,
    scale=None,
    oneshot_note=None,
    steps=None,
    cfg_scale=None,
    sampler_type=None,
    sigma_min=None,
    sigma_max=None,
    cfg_rescale=None,
    model_name=None,
):
    """Plain-text metadata sidecar with the same stem as the WAV/MIDI pair."""
    lines = [
        f"file_stem: {file_stem}",
        f"sample_type: {resolved_sample_type}",
        f"model: {model_name or LAST_CKPT_NAME or 'n/a'}",
        f"seed: {seed}",
        f"wetdry: {wetdry}",
        "",
        f"visible_prompt: {visible_prompt}",
        f"conditioning_prompt: {conditioning_prompt}",
        "",
    ]

    if resolved_sample_type == "oneshot":
        lines.append(f"oneshot_note: {oneshot_note}")
        lines.append(f"seconds_total: {ONESHOT_SECONDS_TOTAL}")
    else:
        lines.append(f"key_signature: {note} {scale}")
        lines.append(f"bars: {bars}")
        lines.append(f"bpm: {bpm}")

    lines.extend([
        "",
        f"steps: {steps}",
        f"cfg_scale: {cfg_scale}",
        f"sampler_type: {sampler_type}",
        f"sigma_min: {sigma_min}",
        f"sigma_max: {sigma_max}",
        f"cfg_rescale: {cfg_rescale}",
        "",
        f"audio_path: {audio_path}",
        f"midi_path: {midi_path or ''}",
    ])

    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
    except Exception as e:
        print(f"Could not write sidecar metadata: {e}")


def build_oneshot_note(note_name: str | None, octave: str | int | None) -> str:
    note_name = str(note_name or "F#").strip()
    octave = str(octave if octave not in (None, "") else "4").strip()
    return f"{note_name}{octave}"


def split_oneshot_note(note: str | None) -> tuple[str, str]:
    match = re.match(r"^([A-G](?:#|b)?)(-?\d+)$", str(note or "F#4").strip(), re.IGNORECASE)
    if not match:
        return "F#", "4"

    name = match.group(1).upper().replace("B", "b")
    if len(name) == 2 and name.endswith("b"):
        # UI is sharp-only; fall back rather than introducing flats into the dropdown.
        return "F#", "4"

    octave = match.group(2)
    if name not in ONESHOT_NOTE_NAMES:
        name = "F#"
    if octave not in ONESHOT_OCTAVE_CHOICES:
        octave = "4"
    return name, octave


def strip_oneshot_pitch_tokens(prompt: str) -> str:
    """
    Textbox prompt should remain descriptor-only.
    ONESHOT prefix, register, pitch, and Wet/Dry are injected at generation time
    from the UI controls.
    """
    tokens = []
    for token in normalize_prompt_tokens(prompt):
        if NOTE_TOKEN_RE.match(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if is_visible_wetdry_token(token):
            continue
        if token.strip().lower() in {"one shot", "oneshot", "one-shot"}:
            continue
        tokens.append(token)
    return ", ".join(tokens)


def get_oneshot_register_label(note: str | None) -> str | None:
    try:
        return master_prompt_map.resolve_oneshot_register_from_ui(note=note)
    except Exception:
        return None


def amend_prompt_oneshot(prompt: str, note: str | None, wetdry: str | None) -> str:
    """
    One-shot generation keeps UI Note and Wet/Dry authoritative.

    Expected final order:
      One Shot, family/sub, tags, Wet/Dry + FX, Register, Note

    If the prompt already contains an old note/register/Wet-Dry/FX set, this
    normalizes those pieces to the current UI values before conditioning.
    """
    note = str(note or "F#4").strip()
    is_wet = str(wetdry or "Wet").strip().lower() == "wet"
    wetdry_token = "Wet" if is_wet else "Dry"
    register_label = get_oneshot_register_label(note)

    body_tokens = []
    fx_tokens = []

    for token in normalize_prompt_tokens(prompt):
        if NOTE_TOKEN_RE.match(token):
            continue
        if is_visible_wetdry_token(token):
            continue
        if token in ONESHOT_REGISTER_LABELS:
            continue
        if token.strip().lower() in {"one shot", "oneshot", "one-shot"}:
            continue

        if is_oneshot_fx_token(token):
            fx_tokens.append(token)
        else:
            body_tokens.append(token)

    tokens = ["One Shot"] + body_tokens + [wetdry_token] + (fx_tokens if is_wet else [])

    if register_label:
        tokens.append(register_label)

    tokens.append(note)

    return ", ".join(tokens)


def apply_short_fade_out(audio: torch.Tensor, sample_rate: int, fade_ms: float) -> torch.Tensor:
    """Apply a short fade-out in-place-safe form to avoid end clicks."""
    fade_len = int(round((float(fade_ms) / 1000.0) * int(sample_rate)))
    if fade_len > 1 and audio.shape[-1] > 1:
        fade_len = min(fade_len, audio.shape[-1])
        ramp = torch.linspace(1.0, 0.0, steps=fade_len, device=audio.device, dtype=audio.dtype)
        audio = audio.clone()
        audio[:, -fade_len:] *= ramp
    return audio


def trim_oneshot_trailing_silence(
    audio: torch.Tensor,
    sample_rate: int,
    *,
    threshold_db: float = ONESHOT_TAIL_TRIM_DB,
    frame_ms: float = ONESHOT_TAIL_TRIM_FRAME_MS,
    tail_pad_ms: float = ONESHOT_TAIL_TRIM_PAD_MS,
    min_keep_ms: float = ONESHOT_TAIL_TRIM_MIN_KEEP_MS,
) -> torch.Tensor:
    """
    Trim silence only from the tail of a one-shot clip.

    This uses torch only, so it works anywhere the app already runs. It scans
    from the end by finding the last short frame whose peak amplitude crosses
    the threshold. Middle gaps do not matter because we never search from the
    beginning.
    """
    if audio is None or audio.numel() == 0 or audio.shape[-1] <= 1:
        return audio

    n = int(audio.shape[-1])
    sr = int(sample_rate)
    frame_len = max(1, int(round(sr * float(frame_ms) / 1000.0)))
    tail_pad = max(0, int(round(sr * float(tail_pad_ms) / 1000.0)))
    min_keep = max(1, int(round(sr * float(min_keep_ms) / 1000.0)))
    threshold_amp = float(10 ** (float(threshold_db) / 20.0))

    # Collapse to a mono peak envelope, preserving device/dtype.
    mono_peak = audio.abs().amax(dim=0)

    n_frames = int(math.ceil(n / frame_len))
    pad = (n_frames * frame_len) - n
    if pad > 0:
        mono_peak = F.pad(mono_peak, (0, pad))

    frame_peaks = mono_peak.view(n_frames, frame_len).amax(dim=1)
    active_frames = torch.nonzero(frame_peaks > threshold_amp, as_tuple=False).flatten()

    if active_frames.numel() == 0:
        # If the whole file is under threshold, keep a tiny safety slice rather
        # than returning a zero-length file.
        end = min(n, min_keep)
    else:
        last_active_frame = int(active_frames[-1].item())
        end = ((last_active_frame + 1) * frame_len) + tail_pad
        end = min(n, max(end, min_keep))

    return audio[:, :max(1, int(end))].contiguous()

def is_keybed_model_name(model_name: str | None) -> bool:
    return master_prompt_map.is_keybed_capable_model(model_name)

def get_runtime_for_keybed():
    return {
        "model": model,
        "sample_rate": sample_rate,
        "device": DEVICE,
        "model_name": LAST_CKPT_NAME,
        "model_config": LAST_MODEL_CONFIG,
        "output_directory": output_directory,
    }


output_directory = config['generations_directory']

# Generated files are saved into short-name subfolders so prompt text never
# becomes the filename. This avoids Windows/path-length issues and keeps
# audio/MIDI/metadata pairs together.
LOOP_OUTPUT_SUBDIR = "Loops"
ONESHOT_OUTPUT_SUBDIR = "One_Shots"
GENERATION_FILENAME_MAX_SLUG_CHARS = 40

current_prompt_generator = master_prompt_map.default_prompt_generator

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

def pick_preferred_dtype(device: torch.device) -> torch.dtype:
    """
    User-facing policy:
      - CUDA: bf16 if supported else fp16
      - MPS: fp16
      - CPU: fp32
    """
    if device.type == "cuda":
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def check_torchao_int4_support(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        from torchao.quantization import quantize_  # noqa
        from torchao.quantization import Int4WeightOnlyConfig  # noqa
        from torchao.quantization import Int8WeightOnlyConfig  # noqa
        return True
    except Exception:
        return False


def toggle_int4_action(enable: bool):
    global INT4_ENABLED
    try:
        enable = bool(enable)

        if enable:
            if model is None:
                raise RuntimeError("No model loaded yet.")
            if not TORCHAO_INT4_SUPPORTED:
                raise RuntimeError("TorchAO INT4 not available.")
            if not INT4_ENABLED:
                apply_int4_inplace(model)
            return runtime_status_md(), gr.update(value=True)

        else:
            if INT4_ENABLED:
                if LAST_CKPT_PATH is None or LAST_MODEL_CONFIG is None:
                    raise RuntimeError("Can't disable INT4 without a reloadable checkpoint/config.")
                reload_last_model(int4_requested=False)

            return runtime_status_md(), gr.update(value=False)

    except Exception as e:
        print("INT4 toggle error:", e)
        return runtime_status_md(), gr.update(value=INT4_ENABLED)
    

def _build_weight_only_cfg(wbits: int, group_size: int = 128):
    errs = []

    if wbits == 4:
        from torchao.quantization import Int4WeightOnlyConfig as Cfg
        # int4 signatures vary; multi-try approach
        for kwargs in (
            dict(group_size=group_size, use_hqq=True, version=1),
            dict(group_size=group_size, use_hqq=True),
            dict(group_size=group_size),
            dict(),
        ):
            try:
                return Cfg(**kwargs)
            except TypeError as e:
                errs.append((kwargs, repr(e)))
        raise RuntimeError("Failed to construct Int4WeightOnlyConfig. Tried: " + str(errs))

    if wbits == 8:
        from torchao.quantization import Int8WeightOnlyConfig as Cfg
        # int8 is usually simpler
        for kwargs in (
            dict(group_size=group_size),
            dict(),
        ):
            try:
                return Cfg(**kwargs)
            except TypeError as e:
                errs.append((kwargs, repr(e)))
        raise RuntimeError("Failed to construct Int8WeightOnlyConfig. Tried: " + str(errs))

    raise ValueError(f"Unsupported wbits={wbits} (expected 4 or 8)")


def _get_transformer_root(m):
    # Matches test loader
    try:
        return m.model.model.transformer
    except Exception:
        return None


def apply_int4_inplace(m) -> None:
    global INT4_ENABLED
    if m is None:
        raise RuntimeError("No model is loaded.")

    tx = _get_transformer_root(m)
    if tx is None:
        raise RuntimeError("Could not find transformer module at model.model.model.transformer")

    from torchao.quantization import quantize_

    qcfg = _build_weight_only_cfg(TORCHAO_WBITS, group_size=64)
    quantize_(tx, qcfg)

    INT4_ENABLED = True   


def unload_current_model():
    """
    Drop the currently loaded model before loading another checkpoint.
    This keeps the app in a single-model runtime and avoids old+new model
    overlap during rapid checkpoint testing.
    """
    global model, INT4_ENABLED

    try:
        if model is not None:
            old_model = model
            model = None
            del old_model

        INT4_ENABLED = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Warning: could not fully unload model before reload: {e}")


def runtime_status_md() -> str:
    dev = DEVICE.type if DEVICE is not None else "unknown"
    dtype = "n/a"
    if model is not None:
        try:
            dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        except Exception:
            dtype = "n/a"

    int4_avail = TORCHAO_INT4_SUPPORTED
    int4_state = "on" if INT4_ENABLED else ("available" if int4_avail else "unavailable")

    name = LAST_CKPT_NAME or "n/a"
    return (
        f"**Runtime:** `{dev}` | dtype: `{dtype}` | INT4: **{int4_state}**  \n"
        f"**Model:** `{name}`"
    )


def reload_last_model(int4_requested: bool):
    """
    Re-load last model from disk. Used when disabling INT4 (since quant is in-place).
    """
    global model, LAST_MODEL_CONFIG
    if LAST_CKPT_PATH is None or LAST_MODEL_CONFIG is None:
        raise RuntimeError("No previous model to reload.")

    # reload base weights (non-quant)
    unload_current_model()
    model, _mc = load_model(
        model_config=LAST_MODEL_CONFIG,
        model_ckpt_path=LAST_CKPT_PATH,
        device=DEVICE,
        preferred_dtype=PREFERRED_DTYPE,
    )

    # optionally re-apply int4
    if int4_requested:
        apply_int4_inplace(model)
    else:
        # ensure flag is correct
        global INT4_ENABLED
        INT4_ENABLED = False

    return model

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None,
               pretransform_ckpt_path=None, device=None, preferred_dtype=None):
    global model, sample_rate, sample_size, global_model_half
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)
        
        # Load checkpoint
        state_dict = load_ckpt_state_dict(model_ckpt_path)

        # Detect dtype safely (state_dict may contain non-tensors)
        tensors = [v for v in state_dict.values() if torch.is_tensor(v)]
        is_fp16 = (len(tensors) > 0) and all(t.dtype == torch.float16 for t in tensors)

        if is_fp16:
            print("Model is in float16 format. Enabling half-precision inference.")
            global_model_half = True
            model.to(torch.float16)
        else:
            print("Model is in full precision format.")
            global_model_half = False

        model.load_state_dict(state_dict)

        
        # Print parameter types after loading into the model
        #print("Parameter types after loading into the model:")
        #for name, param in model.named_parameters():
        #    print(f"Parameter {name} has dtype {param.dtype}")
    
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    
    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        pretransform_state_dict = load_ckpt_state_dict(pretransform_ckpt_path)
        
        # Check if the pretransform model is in float16 format before loading into the pretransform model
        pt_tensors = [v for v in pretransform_state_dict.values() if torch.is_tensor(v)]
        is_float16_pretransform = (len(pt_tensors) > 0) and all(t.dtype == torch.float16 for t in pt_tensors)
                
        if is_float16_pretransform:
            print("Model is in float16 format. Enabling half-precision inference.")
            model.pretransform.to(torch.float16)  # Convert the pretransform model to half precision before loading state dict
        else:
            print("Model is in full precision format.")
        
        model.pretransform.load_state_dict(pretransform_state_dict, strict=False)
        #print(f"Done loading pretransform")
    
    # Move the model to the specified device
    model.to(device).eval().requires_grad_(False)

    # Cast to preferred compute dtype (bf16/fp16 on GPU, fp32 on CPU)
    if preferred_dtype is not None and device is not None:
        if device.type in ("cuda", "mps"):
            model.to(preferred_dtype)
            # treat bf16 as "half" for your global flag
            global_model_half = preferred_dtype in (torch.float16, torch.bfloat16)
        else:
            global_model_half = False

    print(f"Done loading model")
    return model, model_config

def torchao_backend_status():
    try:
        import torchao
        ver = getattr(torchao, "__version__", "unknown")
        try:
            import torchao._C  # compiled extension (fast path indicator)
            return True, "torchao._C: compiled extension loaded", ver
        except Exception as e:
            return False, f"torchao._C: no compiled extension ({e})", ver
    except Exception as e:
        return False, f"torchao import failed: {e}", "n/a"


def calculate_seconds_total(bars, bpm):
    bar_duration = 60 / bpm * 4
    return bar_duration * bars

def clip_samples_from_bars_bpm(bars: int, bpm: float, sample_rate: int, beats_per_bar: int = BEATS_PER_BAR):
    clip_seconds = (60.0 / float(bpm)) * float(beats_per_bar) * float(bars)
    clip_samples = int(round(clip_seconds * sample_rate))
    return clip_samples, clip_seconds

def seconds_total_int_from_clip_samples(n_samples: int, sample_rate: int) -> int:
    return int(math.ceil(int(n_samples) / int(sample_rate)))

def target_samples_for_generation(clip_samples: int, sample_rate: int, min_input_length: int | None):
    """
    Model gets a sample_size corresponding to ceil(seconds)*sr, then padded to min_input_length.
    """
    seconds_total_int = seconds_total_int_from_clip_samples(clip_samples, sample_rate)
    target_samples = int(seconds_total_int * sample_rate)

    if isinstance(min_input_length, int) and min_input_length > 0 and (target_samples % min_input_length) != 0:
        target_samples = target_samples + (min_input_length - (target_samples % min_input_length))

    return seconds_total_int, target_samples

def amend_prompt(prompt, note, scale, bars, bpm, wetdry=None):
    """
    Loop generation keeps Key/Scale/Bars/BPM and Wet/Dry authoritative.
    The textbox stays descriptor-only; Wet/Dry is injected only into conditioning.
    FX detail tokens are preserved for Wet and removed for Dry.
    """
    wetdry_token = str(wetdry or "Wet").strip().title()
    if wetdry_token not in {"Wet", "Dry"}:
        wetdry_token = "Wet"

    is_wet = wetdry_token == "Wet"
    body_tokens = []
    fx_tokens = []

    for token in normalize_prompt_tokens(prompt):
        if is_visible_wetdry_token(token):
            continue
        if is_oneshot_fx_token(token):
            fx_tokens.append(token)
        else:
            body_tokens.append(token)

    tokens = body_tokens + [wetdry_token]
    if is_wet:
        tokens.extend(fx_tokens)

    tokens.extend([f"{note} {scale}", f"{bars} Bars", f"{bpm} BPM"])
    return ", ".join(tokens)

def convert_audio_to_midi(audio_path, output_dir):
    predict_and_save(
        [audio_path],
        output_directory=output_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        save_notes=False
    )

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    plt.figure(figsize=(12, 6))
    piano_roll = pm.get_piano_roll(fs=fs)[start_pitch:end_pitch]

    # This is a MIDI pitch x time matrix, not a CQT spectrogram.
    # Avoid librosa's CQT frequency-axis assumptions (and the resulting
    # Nyquist warning) by plotting a generic Y axis and labeling MIDI notes.
    librosa.display.specshow(
        piano_roll,
        hop_length=1,
        sr=fs,
        x_axis="time",
        y_axis=None,
    )

    plt.colorbar()
    plt.title("Piano Roll Visualization")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")

    pitch_count = max(0, int(end_pitch) - int(start_pitch))
    tick_positions = np.arange(0, pitch_count, 12)
    if len(tick_positions):
        plt.yticks(
            tick_positions,
            [
                pretty_midi.note_number_to_name(int(start_pitch) + int(i))
                for i in tick_positions
            ],
        )

    plt.savefig("piano_roll.png")
    plt.close()
    return "piano_roll.png"

def generate_cond(
        prompt,
        negative_prompt=None,
        bars=4,
        bpm=100,
        note='C',
        scale='major',
        sample_type_loop_checked=True,
        sample_type_oneshot_checked=False,
        oneshot_note_name='F#',
        oneshot_octave='4',
        wetdry_dry_checked=False,
        wetdry_wet_checked=True,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1,
        output_parent_subdir=None,
        output_run_subdir=None,
        output_write_sidecar=True,
        output_generate_midi=True,
        output_generate_spectrogram=True
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    sample_type = sample_type_from_toggles(sample_type_loop_checked, sample_type_oneshot_checked)
    wetdry = wetdry_from_toggles(wetdry_dry_checked, wetdry_wet_checked)
    resolved_sample_type = normalize_ui_sample_type(sample_type)

    if resolved_sample_type == "oneshot":
        oneshot_note = build_oneshot_note(oneshot_note_name, oneshot_octave)
        amended_prompt = amend_prompt_oneshot(prompt, oneshot_note, wetdry)
        clip_seconds = float(ONESHOT_SECONDS_TOTAL)
        clip_samples = int(round(clip_seconds * sample_rate))
    else:
        amended_prompt = amend_prompt(prompt, note, scale, bars, bpm, wetdry)
        clip_samples, clip_seconds = clip_samples_from_bars_bpm(bars, bpm, sample_rate)

    print(f"Prompt: {amended_prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0 or not bool(output_generate_spectrogram):
        preview_every = None

    seconds_start = 0.0

    # --- SAT conditioner expects integer seconds_total ---
    min_input = getattr(model, "min_input_length", None)
    seconds_total_int, input_sample_size = target_samples_for_generation(
        clip_samples=clip_samples,
        sample_rate=sample_rate,
        min_input_length=min_input
    )

    conditioning = [{"prompt": amended_prompt, "seconds_start": seconds_start, "seconds_total": float(seconds_total_int)}] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": float(seconds_total_int)}] * batch_size
    else:
        negative_conditioning = None

    # Get the device from the model
    device = next(model.parameters()).device
    seed = int(seed)

    if not use_init:
        init_audio = None

    # ---------- init audio handling ----------
    # If init audio is provided and is longer than the computed input_sample_size,
    # we expand input_sample_size to fit it (and keep min_input_length alignment).
    if init_audio is not None:
        in_sr, init_audio_arr = init_audio

        # Convert numpy audio to torch float32 mono/stereo handling
        if init_audio_arr.dtype == np.float32:
            init_audio_t = torch.from_numpy(init_audio_arr)
        elif init_audio_arr.dtype == np.int16:
            init_audio_t = torch.from_numpy(init_audio_arr).float().div(32767)
        elif init_audio_arr.dtype == np.int32:
            init_audio_t = torch.from_numpy(init_audio_arr).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio_arr.dtype}")

        if init_audio_t.dim() == 1:
            init_audio_t = init_audio_t.unsqueeze(0)
        elif init_audio_t.dim() == 2:
            init_audio_t = init_audio_t.transpose(0, 1)

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio_t.device)
            init_audio_t = resample_tf(init_audio_t)

        audio_length = int(init_audio_t.shape[-1])

        # expand generation size if init audio longer than planned
        if audio_length > input_sample_size:
            if isinstance(min_input, int) and min_input > 0:
                pad = (min_input - (audio_length % min_input)) % min_input
                input_sample_size = audio_length + pad
            else:
                input_sample_size = audio_length

        init_audio = (sample_rate, init_audio_t)

    # ---------- preview callback ----------
    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if preview_every is None:
            return

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # ---------- mask args ----------
    if mask_cropfrom is not None:
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None

    # ---------- generation ----------
    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=int(input_sample_size),
        sample_rate=sample_rate,
        seed=seed,
        device=device,
        sampler_type=sampler_type,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        mask_args=mask_args,
        callback=progress_callback if preview_every is not None else None,
        scale_phi=cfg_rescale
    )

    # ---------- tensor trim + short fade ----------
    audio = rearrange(audio, "b d n -> d (b n)")  # [ch, n] or [d, n]
    audio = _protect_mps_audio_range(audio)

    # Always cap to the planned generation window first.
    # - Loops stay sample-exact to bars/BPM.
    # - One-shots then get an additional end-only silence trim.
    end = min(int(audio.shape[-1]), int(clip_samples))
    audio = audio[:, :max(1, end)].contiguous()

    if resolved_sample_type == "oneshot":
        audio = trim_oneshot_trailing_silence(audio, sample_rate)
        audio = apply_short_fade_out(audio, sample_rate, ONESHOT_TAIL_TRIM_FADE_MS)
    else:
        audio = apply_short_fade_out(audio, sample_rate, 15.0)

    wav_i16 = (audio * 32767.0).to(torch.int16).cpu()

    # Spectrograms are useful in the main Generation tab, but expensive and
    # discarded by Batch Generation. Keep the default enabled for backwards
    # compatibility and let lightweight callers skip the work entirely.
    audio_spectrogram = None
    if bool(output_generate_spectrogram):
        audio_spectrogram = audio_spectrogram_image(wav_i16, sample_rate=sample_rate)
    else:
        print("Spectrogram generation skipped for this generation.")

    # ---------- save WAV ----------
    # Use short, stable filenames and save directly into the final folder.
    # This keeps Basic Pitch output in the same place and avoids filename/path
    # length issues from using the full prompt as the filename.
    save_dir = get_generation_output_dir(
        resolved_sample_type,
        parent_subdir=output_parent_subdir,
        run_subdir=output_run_subdir,
    )
    file_stem = unique_generation_stem(save_dir, amended_prompt, seed, resolved_sample_type)
    file_path = os.path.join(save_dir, f"{file_stem}.wav")
    sidecar_path = os.path.join(save_dir, f"{file_stem}.txt")

    torchaudio.save(file_path, wav_i16, sample_rate)

    # ---------- optional MIDI conversion ----------
    # Main Generation keeps MIDI enabled by default. Batch Generation disables
    # it so multi-inference runs do not repeatedly invoke Basic Pitch or create
    # piano-roll images that the Batch UI never displays.
    midi_output_path = None
    piano_roll_path = None

    if bool(output_generate_midi):
        try:
            convert_audio_to_midi(file_path, save_dir)
            time.sleep(1)

            midi_output_path = find_latest_midi_for_stem(save_dir, file_stem)
            midi_output_path = normalize_midi_filename(midi_output_path, save_dir, file_stem)

            if midi_output_path is not None:
                print(f"MIDI file saved successfully as {midi_output_path}.")
                midi_data = pretty_midi.PrettyMIDI(midi_output_path)
                print("MIDI file loaded successfully.")
                piano_roll_path = plot_piano_roll(midi_data, 21, 109)
            else:
                print("MIDI file was not found. Please check the conversion process.")

        except Exception as e:
            print(f"An error occurred during MIDI conversion: {e}")
            midi_output_path = None
            piano_roll_path = None
    else:
        print("MIDI extraction skipped for this generation.")

    # Write the metadata sidecar after MIDI conversion so the pair is recorded.
    # Batch Generation can disable per-file sidecars and write one grouped
    # metadata file for the whole run instead.
    if bool(output_write_sidecar):
        write_generation_sidecar(
            sidecar_path,
            file_stem=file_stem,
            resolved_sample_type=resolved_sample_type,
            visible_prompt=strip_oneshot_pitch_tokens(prompt) if resolved_sample_type == "oneshot" else strip_wetdry_tokens(prompt),
            conditioning_prompt=amended_prompt,
            seed=seed,
            wetdry=wetdry,
            audio_path=file_path,
            midi_path=midi_output_path,
            bars=bars,
            bpm=bpm,
            note=note,
            scale=scale,
            oneshot_note=oneshot_note if resolved_sample_type == "oneshot" else None,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            cfg_rescale=cfg_rescale,
            model_name=LAST_CKPT_NAME,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Return the same four-item structure used by existing callers. When
    # spectrogram generation is disabled, the gallery payload is simply empty.
    spectrogram_outputs = list(preview_images)
    if audio_spectrogram is not None:
        spectrogram_outputs.insert(0, audio_spectrogram)

    # returning file_path (already trimmed)
    return (file_path, spectrogram_outputs, piano_roll_path, midi_output_path)



def get_models_and_configs(models_path):
    ckpt_files = []
    for root, _, files in os.walk(models_path):
        for file in files:
            if file.endswith((".ckpt", ".safetensors")):
                ckpt_files.append((file, os.path.join(root, file)))

    # Deterministic startup/order: prefer the flagship Foundation-1.2 Samples
    # checkpoint when installed, then fall back to a normal alphabetical order.
    def _model_sort_key(item):
        name = str(item[0] or "").casefold()
        preferred = bool(re.search(r"foundation[-_.\s]*1\.2[-_.\s]*samples", name, re.IGNORECASE))
        return (0 if preferred else 1, name)

    ckpt_files.sort(key=_model_sort_key)
    return ckpt_files


def get_config_files(ckpt_path):
    config_files = []
    folder = os.path.dirname(ckpt_path)
    for file in os.listdir(folder):
        if file.endswith(".json"):
            config_files.append(file)

    # Keep selection deterministic and prefer the conventional model_config.json
    # when it lives beside the checkpoint.
    config_files.sort(key=lambda name: (name.lower() != "model_config.json", name.lower()))
    return config_files


def get_preferred_config_for_checkpoint(ckpt_path):
    """Return the preferred JSON config beside a checkpoint, or None."""
    if not ckpt_path:
        return None
    configs = get_config_files(ckpt_path)
    return configs[0] if configs else None


def update_config_dropdown(selected_ckpt, ckpt_files):
    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)
        configs = get_config_files(ckpt_path)
        preferred = configs[0] if configs else "Select Config"
        return gr.update(
            choices=configs if configs else ["Select Config"],
            value=preferred,
        )
    except Exception as e:
        print(f"Error updating config dropdown: {e}")  # Debugging output
        return gr.update(choices=["Error finding configs"], value="Error finding configs")


def load_model_action(selected_ckpt, selected_config, ckpt_files, int4_requested: bool):
    """Load a checkpoint without mutating the user's Generation-tab UI state.

    Checkpoint loading is a runtime operation. Loop/One Shot, prompt style,
    Wet/Dry, prompt text, and prompt-builder state belong to the user's current
    session and must survive a model swap unchanged.
    """
    global DEVICE, current_prompt_generator
    global LAST_CKPT_PATH, LAST_CONFIG_PATH, LAST_CKPT_NAME, LAST_MODEL_CONFIG, INT4_ENABLED
    global model

    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)

        # The normal case is model_config.json sitting beside the checkpoint.
        # If the UI still contains a placeholder (or a stale/missing value),
        # resolve the preferred config automatically instead of trying to open a
        # literal file named "Select Config".
        config_dir = os.path.dirname(ckpt_path)
        selected_config_path = (
            os.path.join(config_dir, selected_config)
            if selected_config and selected_config not in {"Select Config", "Error finding configs"}
            else None
        )

        if not selected_config_path or not os.path.isfile(selected_config_path):
            selected_config = get_preferred_config_for_checkpoint(ckpt_path)
            if not selected_config:
                raise FileNotFoundError(
                    f"No JSON config found beside checkpoint: {ckpt_path}"
                )

        config_path = os.path.join(config_dir, selected_config)

        with open(config_path, "r") as f:
            cfg = json.load(f)

        # Remember what we loaded for the shared runtime and other tabs.
        LAST_CKPT_PATH = ckpt_path
        LAST_CONFIG_PATH = config_path
        LAST_CKPT_NAME = selected_ckpt
        LAST_MODEL_CONFIG = cfg

        # Loading remains exclusive at the model/runtime level, but it no
        # longer resets any user-facing generation controls.
        unload_current_model()
        _m, _mc = load_model(
            model_config=cfg,
            model_ckpt_path=ckpt_path,
            device=DEVICE,
            preferred_dtype=PREFERRED_DTYPE,
        )
        model = _m

        INT4_ENABLED = False
        if bool(int4_requested):
            if not TORCHAO_INT4_SUPPORTED:
                raise RuntimeError("INT4 requested but TorchAO INT4 is not available on this device/install.")
            apply_int4_inplace(model)

        # Keep this for legacy/internal callers that still inspect the current
        # generator. Main-tab UI routing itself is now sample-type driven.
        current_prompt_generator = master_prompt_map.get_prompt_generator(selected_ckpt)

        info = f"Loaded model {selected_ckpt} with config {selected_config}"
        return info, runtime_status_md()

    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error loading model: {e}", runtime_status_md()

def create_sampling_ui(model_config, initial_ckpt, inpainting=False):
    ckpt_files = get_models_and_configs(config['models_directory'])
    initial_name = os.path.basename(initial_ckpt or "")
    initial_configs = get_config_files(initial_ckpt) if initial_ckpt else []
    initial_config = initial_configs[0] if initial_configs else None

    selected_ckpt = gr.State(value=initial_name)
    selected_config = gr.State(value=initial_config)
    # Main Generation UI is capability-agnostic: Loop / One Shot remain user
    # choices. The loaded model only selects which *loop prompt builder* is shown.
    prompt_style_initial = True
    initial_loop_family = master_prompt_map.get_loop_prompt_family(initial_name)

    prompt_style_active = gr.State(value=True)
    # Sample type is now a user-selected Generation mode, not a capability
    # inferred from the checkpoint filename.
    oneshot_active = gr.State(value=True)
    loop_prompt_state = gr.State(value="")
    oneshot_prompt_state = gr.State(value="")

    with gr.Row(elem_id="top_prompt_row"):
        with gr.Column(scale=8, elem_id="prompt_left_col"):
            prompt = gr.Textbox(show_label=False, placeholder="Prompt", elem_id="prompt_box", lines=4)
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt", visible=False, value="")

        with gr.Column(scale=2):
            with gr.Column():
                generate_button = gr.Button("Generate", variant="primary", scale=1)
                random_prompt_button = gr.Button("Random Prompt", variant="secondary", scale=1)

                # ✅ wrapper that hides/shows everything, but looks seamless
                with gr.Column(visible=(prompt_style_initial and initial_loop_family == "foundation"), elem_id="foundation_mode_group") as foundation_mode_group:
                    with gr.Row():
                        foundation_simple_cb = gr.Checkbox(label="Simple", value=True)
                        foundation_experimental_cb = gr.Checkbox(label="Experimental", value=False)

                    foundation_mode_help = gr.Markdown(
                        FOUNDATION_MODE_HELP[FOUNDATION_MODE_SIMPLE],
                        elem_id="foundation_mode_help",
                    )



                def _toggle_simple(is_checked: bool):
                    """
                    If user checks Simple => turn off Experimental.
                    If user unchecks Simple => force Experimental on (so one is always active).
                    """
                    if is_checked:
                        mode = FOUNDATION_MODE_SIMPLE
                        return (
                            gr.update(value=True),
                            gr.update(value=False),
                            gr.update(value=FOUNDATION_MODE_HELP[mode], visible=True),
                        )
                    else:
                        mode = FOUNDATION_MODE_EXPERIMENTAL
                        return (
                            gr.update(value=False),
                            gr.update(value=True),
                            gr.update(value=FOUNDATION_MODE_HELP[mode], visible=True),
                        )

                def _toggle_experimental(is_checked: bool):
                    """
                    If user checks Experimental => turn off Simple.
                    If user unchecks Experimental => force Simple on.
                    """
                    if is_checked:
                        mode = FOUNDATION_MODE_EXPERIMENTAL
                        return (
                            gr.update(value=False),
                            gr.update(value=True),
                            gr.update(value=FOUNDATION_MODE_HELP[mode], visible=True),
                        )
                    else:
                        mode = FOUNDATION_MODE_SIMPLE
                        return (
                            gr.update(value=True),
                            gr.update(value=False),
                            gr.update(value=FOUNDATION_MODE_HELP[mode], visible=True),
                        )

                foundation_simple_cb.change(
                    fn=_toggle_simple,
                    inputs=[foundation_simple_cb],
                    outputs=[foundation_simple_cb, foundation_experimental_cb, foundation_mode_help],
                )

                foundation_experimental_cb.change(
                    fn=_toggle_experimental,
                    inputs=[foundation_experimental_cb],
                    outputs=[foundation_simple_cb, foundation_experimental_cb, foundation_mode_help],
                )


    
    with gr.Accordion(
        "Prompt Builder",
        open=False,
        visible=prompt_style_initial,
        elem_id="main_prompt_builder",
    ) as prompt_builder_group:
        with gr.Column(visible=(prompt_style_initial and initial_loop_family == "foundation")) as foundation_loop_builder_group:
            gr.Markdown(
                "Choose up to two unrestricted sound sources, then add timbre and musical tags. "
                "The prompt textbox remains authoritative, so unusual manually typed ideas are preserved."
            )
            with gr.Row():
                foundation_source_1_picker = _searchable_dropdown(
                    FOUNDATION_BUILDER_SOURCES,
                    label="Sound Source 1",
                    value=None,
                )
                foundation_source_2_picker = _searchable_dropdown(
                    FOUNDATION_BUILDER_OPTIONAL_SOURCES,
                    label="Sound Source 2 (Optional)",
                    value=PROMPT_BUILDER_NONE,
                )
            foundation_timbre_picker = _searchable_dropdown(
                FOUNDATION_BUILDER_TIMBRE_TAGS,
                label="Timbre / Sound Tags",
                value=[],
                multiselect=True,
            )
            with gr.Row():
                foundation_structure_picker = _searchable_dropdown(
                    [PROMPT_BUILDER_NONE] + FOUNDATION_BUILDER_STRUCTURES,
                    label="Musical Structure",
                    value=PROMPT_BUILDER_NONE,
                )
                foundation_musical_picker = _searchable_dropdown(
                    FOUNDATION_BUILDER_MUSICAL_TAGS,
                    label="Musical Tags",
                    value=[],
                    multiselect=True,
                )
            clear_foundation_builder_tags_button = gr.Button(
                "Clear Selected Tags",
                variant="secondary",
            )

        with gr.Column(visible=(initial_loop_family == "piano")) as piano_loop_builder_group:
            gr.Markdown("Legacy Infinite Pianos builder — preserves piano type, structure, tremolo, and reverb vocabulary.")
            with gr.Row():
                piano_type_picker = _searchable_dropdown(PIANO_BUILDER_TYPES, label="Piano Type", value=PIANO_BUILDER_TYPES[0])
                piano_structure_picker = _searchable_dropdown(PIANO_BUILDER_STRUCTURES, label="Structure", value=PIANO_BUILDER_STRUCTURES[0])
            with gr.Row():
                piano_chord_style_picker = _searchable_dropdown(PIANO_BUILDER_CHORD_STYLES, label="Chord Style", value=PROMPT_BUILDER_NONE)
                piano_melody_style_picker = _searchable_dropdown(PIANO_BUILDER_MELODY_STYLES, label="Melody Style", value=PROMPT_BUILDER_NONE)
            piano_effect_picker = _searchable_dropdown(PIANO_BUILDER_EFFECTS, label="Effect", value=PIANO_BUILDER_EFFECTS[0])

        with gr.Column(visible=(initial_loop_family == "edm_elements")) as edm_loop_builder_group:
            gr.Markdown("Legacy EDM Elements builder — model vocabulary exposed as composable sound, structure, movement, and FX controls.")
            edm_sound_picker = _searchable_dropdown(EDM_BUILDER_SOUND_TAGS, label="Sound Tags", value=[], multiselect=True)
            with gr.Row():
                edm_structure_picker = _searchable_dropdown(EDM_BUILDER_STRUCTURES, label="Musical Structure", value=PROMPT_BUILDER_NONE)
                edm_musical_picker = _searchable_dropdown(EDM_BUILDER_MUSICAL_TAGS, label="Musical Tags", value=[], multiselect=True)
            edm_effect_picker = _searchable_dropdown(EDM_BUILDER_EFFECTS, label="Legacy Effects", value=[], multiselect=True)

        with gr.Column(visible=(initial_loop_family == "vocal_textures")) as vocal_loop_builder_group:
            gr.Markdown("Legacy Vocal Textures builder.")
            with gr.Row():
                vocal_type_picker = _searchable_dropdown(VOCAL_BUILDER_TYPES, label="Vocal Type", value=VOCAL_BUILDER_TYPES[0])
                vocal_structure_picker = _searchable_dropdown(VOCAL_BUILDER_STRUCTURES, label="Structure", value=VOCAL_BUILDER_STRUCTURES[0])

        with gr.Column(visible=False) as oneshot_prompt_builder_group:
            gr.Markdown(
                "Choose one or two unrestricted sound sources plus timbre tags. "
                "One Shot generation is intended for models trained with one-shot support, though other checkpoints can still be used experimentally."
            )
            with gr.Row():
                oneshot_source_1_picker = _searchable_dropdown(
                    ONESHOT_BUILDER_SOURCES,
                    label="Sound Source 1",
                    value=None,
                )
                oneshot_source_2_picker = _searchable_dropdown(
                    ONESHOT_BUILDER_OPTIONAL_SOURCES,
                    label="Sound Source 2 (Optional)",
                    value=PROMPT_BUILDER_NONE,
                )
            oneshot_timbre_picker = _searchable_dropdown(
                ONESHOT_BUILDER_TIMBRE_TAGS,
                label="Timbre Tags",
                value=[],
                multiselect=True,
            )
            clear_oneshot_builder_tags_button = gr.Button(
                "Clear Selected Tags",
                variant="secondary",
            )

    model_conditioning_config = model_config["model"].get("conditioning", None)

    has_seconds_start = False
    has_seconds_total = False

    if model_conditioning_config is not None:
        for conditioning_config in model_conditioning_config["configs"]:
            if conditioning_config["id"] == "seconds_start":
                has_seconds_start = True
            if conditioning_config["id"] == "seconds_total":
                has_seconds_total = True

    with gr.Row(equal_height=False):
        with gr.Column():
            current_model_info = gr.Markdown(f"Current Model: {selected_ckpt.value}")

            # Model and Config dropdowns
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    ["Select Model"] + [file[0] for file in ckpt_files],
                    value=initial_name if initial_name else "Select Model",
                    label="Select Model",
                )
                config_dropdown = gr.Dropdown(
                    initial_configs if initial_configs else ["Select Config"],
                    value=initial_config if initial_config else "Select Config",
                    label="Select Config",
                )

            model_dropdown.change(
                fn=lambda x: update_config_dropdown(x, ckpt_files),
                inputs=model_dropdown,
                outputs=config_dropdown
            )

            # Status stays in the "model" area so users always see device/dtype/int4 state
            status_md = gr.Markdown(runtime_status_md())

            load_model_button = gr.Button("Load Model")

            with gr.Column(visible=prompt_style_initial, elem_id="model_configurator_group") as model_configurator_group:
                with gr.Column(elem_id="wetdry_group") as wetdry_group:
                    gr.Markdown("FX Toggle", elem_id="wetdry_label")
                    with gr.Row():
                        wetdry_dry_cb = gr.Checkbox(label="Dry", value=False)
                        wetdry_wet_cb = gr.Checkbox(label="Wet", value=True)

            # Optional post-generation work. These controls intentionally live
            # outside the Loop / One-Shot visibility groups so their values
            # persist when users switch sample type.
            with gr.Column(elem_id="generation_extras_group"):
                gr.Markdown("Extras", elem_id="generation_extras_label")
                with gr.Row():
                    generate_midi_checkbox = gr.Checkbox(
                        label="MIDI Output",
                        value=True,
                    )
                    generate_spectrogram_checkbox = gr.Checkbox(
                        label="Spectrogram Output",
                        value=True,
                    )

            with gr.Column(visible=True) as loop_controls_group:
                lock_bpm_checkbox = gr.Checkbox(label="Lock BPM Settings", value=True)
                with gr.Row(visible=has_seconds_start or has_seconds_total):
                    bars_dropdown = gr.Dropdown([4, 8], label="Bars", value=8, visible=has_seconds_total)
                    bpm_dropdown = gr.Dropdown([100, 110, 120, 128, 130, 140, 150],
                                               label="BPM", value=128, visible=has_seconds_total)

                # Lock Key Signature + key signature dropdowns
                lock_key_checkbox = gr.Checkbox(label="Lock Key Signature", value=True)
                with gr.Row():
                    note_dropdown = gr.Dropdown(
                        ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],
                        label="Key",
                        value="F"
                    )
                    scale_dropdown = gr.Dropdown(["major", "minor"], label="Scale", value="minor")

            with gr.Column(visible=False) as oneshot_controls_group:
                # Default OFF so Random Prompt can use the ONESHOT builder's
                # family/subfamily-aware register heuristic and update the UI
                # note/octave dropdowns to a sensible range. When enabled, the
                # current dropdown note remains authoritative.
                lock_oneshot_note_checkbox = gr.Checkbox(label="Lock Note", value=False)
                with gr.Row():
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

            # Seed moved
            seed_textbox = gr.Textbox(label="Seed (set to -1 for random)", value="-1")

            # Sampler params accordion now contains: steps / preview / cfg / int4 / sampler knobs
            with gr.Accordion("Sampler params", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=75, label="Steps")
                    cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG scale")
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")

                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(
                        ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms",
                         "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                        label="Sampler type",
                        value="dpmpp-3m-sde"
                    )
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")

                # Experimental INT4 - only if supported
                with gr.Accordion("Experimental", open=False, visible=TORCHAO_INT4_SUPPORTED):
                    win_note = " **Very slow on Windows** (often Triton fallback)." if platform.system() == "Windows" else ""
                    gr.Markdown(
                        "INT4 is for low-VRAM systems and can be painfully slow depending on your setup."
                        + win_note
                        + " Disabling INT4 reloads the model."
                    )

                    int4_checkbox = gr.Checkbox(
                        label="Enable INT4 (TorchAO weight-only)",
                        value=False,
                        interactive=TORCHAO_INT4_SUPPORTED,
                        visible=TORCHAO_INT4_SUPPORTED,
                    )

                    int4_checkbox.change(
                        fn=toggle_int4_action,
                        inputs=[int4_checkbox],
                        outputs=[status_md, int4_checkbox],
                    )

            if inpainting:
                with gr.Accordion("Inpainting", open=False):
                    sigma_max_slider.maximum = 1000

                    init_audio_checkbox = gr.Checkbox(label="Do inpainting")
                    init_audio_input = gr.Audio(label="Init audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.1, value=80,
                                                       label="Init audio noise level", visible=False)

                    mask_cropfrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Crop From %")
                    mask_pastefrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Paste From %")
                    mask_pasteto_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Paste To %")

                    mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50, label="Mask Start %")
                    mask_maskend_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Mask End %")
                    mask_softnessL_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Left Crossfade Length %")
                    mask_softnessR_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Right Crossfade Length %")
                    mask_marination_slider = gr.Slider(minimum=0.0, maximum=1, step=0.0001, value=0, label="Marination level", visible=False)

        with gr.Column():
            with gr.Column(visible=True, elem_id="sample_type_group") as sample_type_group:
                gr.Markdown("Sample Type", elem_id="sample_type_label")
                with gr.Row():
                    sample_type_loop_cb = gr.Checkbox(label="Loop", value=True)
                    sample_type_oneshot_cb = gr.Checkbox(label="One Shot", value=False)

            audio_output = gr.Audio(label="Output audio", interactive=False)
            send_to_init_button = gr.Button("Send to Style Transfer", scale=1)
            with gr.Accordion("AI Style Transfer", open=False):
                init_audio_checkbox = gr.Checkbox(label="Use for Style Transfer")
                init_audio_input = gr.Audio(label="Input audio")
                init_noise_level_slider = gr.Slider(minimum=0.1, maximum=5.0, step=0.01, value=0.9, label="Init noise level")

        with gr.Column():
            midi_piano_roll_output = gr.Image(label="MIDI Piano Roll", interactive=False)
            midi_download_button = gr.File(label="Download MIDI", file_count="single", type="filepath", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)

    # IMPORTANT: int4_checkbox exists only if TORCHAO_INT4_SUPPORTED.
    # For load_model_button inputs we need a safe placeholder when it's not supported.
    if TORCHAO_INT4_SUPPORTED:
        int4_for_load = int4_checkbox
    else:
        int4_for_load = gr.State(value=False)

    def _apply_piano_builder(piano_type, structure, chord_style, melody_style, effect):
        return piano_prompts.build_prompt(piano_type, structure, chord_style, melody_style, effect)

    def _apply_edm_builder(sound_tags, structure, musical_tags, effects):
        return edm_elements_prompts.build_prompt(sound_tags, structure, musical_tags, effects)

    def _apply_vocal_builder(vocal_type, structure):
        return vocal_textures_prompts.build_prompt(vocal_type, structure)

    piano_builder_inputs = [piano_type_picker, piano_structure_picker, piano_chord_style_picker, piano_melody_style_picker, piano_effect_picker]

    def _change_piano_type(piano_type, structure, chord_style, melody_style, effect):
        allowed = piano_prompts.effect_choices_for_piano(piano_type)
        resolved_effect = effect if effect in allowed else allowed[0]
        return (
            piano_prompts.build_prompt(piano_type, structure, chord_style, melody_style, resolved_effect),
            gr.update(choices=allowed, value=resolved_effect),
        )

    piano_type_picker.change(
        fn=_change_piano_type,
        inputs=piano_builder_inputs,
        outputs=[prompt, piano_effect_picker],
        queue=False,
        show_progress="hidden",
    )
    for component in piano_builder_inputs[1:]:
        component.input(fn=_apply_piano_builder, inputs=piano_builder_inputs, outputs=prompt, queue=False, show_progress="hidden")

    edm_builder_inputs = [edm_sound_picker, edm_structure_picker, edm_musical_picker, edm_effect_picker]
    for component in edm_builder_inputs:
        component.input(fn=_apply_edm_builder, inputs=edm_builder_inputs, outputs=prompt, queue=False, show_progress="hidden")

    vocal_builder_inputs = [vocal_type_picker, vocal_structure_picker]
    for component in vocal_builder_inputs:
        component.input(fn=_apply_vocal_builder, inputs=vocal_builder_inputs, outputs=prompt, queue=False, show_progress="hidden")

    def _loop_builder_visibility(is_loop):
        family = master_prompt_map.get_loop_prompt_family(LAST_CKPT_NAME or initial_name)
        return (
            gr.update(visible=bool(is_loop) and family == "foundation"),
            gr.update(visible=bool(is_loop) and family == "piano"),
            gr.update(visible=bool(is_loop) and family == "edm_elements"),
            gr.update(visible=bool(is_loop) and family == "vocal_textures"),
        )

    def _toggle_sample_type_loop(
        is_checked: bool,
        is_oneshot_capable: bool,
        current_prompt,
        saved_loop_prompt,
        saved_oneshot_prompt,
    ):
        # `is_oneshot_capable` is retained only for callback compatibility.
        # UI routing is determined exclusively by the user's sample-type choice.
        target_loop = bool(is_checked)
        if target_loop:
            saved_oneshot_prompt = str(current_prompt or "")
            next_prompt = str(saved_loop_prompt or "")
        else:
            saved_loop_prompt = str(current_prompt or "")
            next_prompt = str(saved_oneshot_prompt or "")

        family = master_prompt_map.get_loop_prompt_family(LAST_CKPT_NAME or initial_name)
        return (
            gr.update(value=target_loop),
            gr.update(value=not target_loop),
            gr.update(visible=target_loop),
            gr.update(visible=not target_loop),
            *_loop_builder_visibility(target_loop),
            gr.update(visible=not target_loop),
            gr.update(visible=(not target_loop) or family == "foundation"),
            saved_loop_prompt,
            saved_oneshot_prompt,
            next_prompt,
        )

    def _toggle_sample_type_oneshot(
        is_checked: bool,
        is_oneshot_capable: bool,
        current_prompt,
        saved_loop_prompt,
        saved_oneshot_prompt,
    ):
        # `is_oneshot_capable` is retained only for callback compatibility.
        # UI routing is determined exclusively by the user's sample-type choice.
        target_oneshot = bool(is_checked)
        if target_oneshot:
            saved_loop_prompt = str(current_prompt or "")
            next_prompt = str(saved_oneshot_prompt or "")
        else:
            saved_oneshot_prompt = str(current_prompt or "")
            next_prompt = str(saved_loop_prompt or "")

        family = master_prompt_map.get_loop_prompt_family(LAST_CKPT_NAME or initial_name)
        return (
            gr.update(value=not target_oneshot),
            gr.update(value=target_oneshot),
            gr.update(visible=not target_oneshot),
            gr.update(visible=target_oneshot),
            *_loop_builder_visibility(not target_oneshot),
            gr.update(visible=target_oneshot),
            gr.update(visible=target_oneshot or family == "foundation"),
            saved_loop_prompt,
            saved_oneshot_prompt,
            next_prompt,
        )

    def _toggle_wetdry_dry(is_checked: bool):
        if is_checked:
            return gr.update(value=True), gr.update(value=False)
        return gr.update(value=False), gr.update(value=True)

    def _toggle_wetdry_wet(is_checked: bool):
        if is_checked:
            return gr.update(value=False), gr.update(value=True)
        return gr.update(value=True), gr.update(value=False)

    sample_type_switch_inputs = [
        prompt,
        loop_prompt_state,
        oneshot_prompt_state,
    ]
    sample_type_switch_outputs = [
        sample_type_loop_cb,
        sample_type_oneshot_cb,
        loop_controls_group,
        oneshot_controls_group,
        foundation_loop_builder_group,
        piano_loop_builder_group,
        edm_loop_builder_group,
        vocal_loop_builder_group,
        oneshot_prompt_builder_group,
        foundation_mode_group,
        loop_prompt_state,
        oneshot_prompt_state,
        prompt,
    ]

    sample_type_loop_cb.change(
        fn=_toggle_sample_type_loop,
        inputs=[sample_type_loop_cb, oneshot_active, *sample_type_switch_inputs],
        outputs=sample_type_switch_outputs,
        queue=False,
        show_progress="hidden",
    )

    sample_type_oneshot_cb.change(
        fn=_toggle_sample_type_oneshot,
        inputs=[sample_type_oneshot_cb, oneshot_active, *sample_type_switch_inputs],
        outputs=sample_type_switch_outputs,
        queue=False,
        show_progress="hidden",
    )

    wetdry_dry_cb.change(
        fn=_toggle_wetdry_dry,
        inputs=[wetdry_dry_cb],
        outputs=[wetdry_dry_cb, wetdry_wet_cb],
    )

    wetdry_wet_cb.change(
        fn=_toggle_wetdry_wet,
        inputs=[wetdry_wet_cb],
        outputs=[wetdry_dry_cb, wetdry_wet_cb],
    )

    foundation_builder_inputs = [
        prompt,
        foundation_source_1_picker,
        foundation_source_2_picker,
        foundation_timbre_picker,
        foundation_structure_picker,
        foundation_musical_picker,
    ]
    for builder_component in (
        foundation_source_1_picker,
        foundation_source_2_picker,
        foundation_timbre_picker,
        foundation_structure_picker,
        foundation_musical_picker,
    ):
        builder_component.input(
            fn=apply_foundation_prompt_builder_action,
            inputs=foundation_builder_inputs,
            outputs=prompt,
            queue=False,
            show_progress="hidden",
        )

    clear_foundation_builder_tags_button.click(
        fn=clear_foundation_prompt_builder_tags_action,
        inputs=[
            prompt,
            foundation_source_1_picker,
            foundation_source_2_picker,
            foundation_structure_picker,
        ],
        outputs=[prompt, foundation_timbre_picker, foundation_musical_picker],
        queue=False,
        show_progress="hidden",
    )

    oneshot_builder_source_inputs = [
        prompt,
        oneshot_source_1_picker,
        oneshot_source_2_picker,
        oneshot_timbre_picker,
        lock_oneshot_note_checkbox,
        oneshot_note_name_dropdown,
        oneshot_octave_dropdown,
        seed_textbox,
    ]
    for source_component in (oneshot_source_1_picker, oneshot_source_2_picker):
        source_component.input(
            fn=main_oneshot_source_change_action,
            inputs=oneshot_builder_source_inputs,
            outputs=[prompt, oneshot_note_name_dropdown, oneshot_octave_dropdown],
            queue=False,
            show_progress="hidden",
        )

    oneshot_timbre_picker.input(
        fn=apply_oneshot_prompt_builder_action,
        inputs=[
            prompt,
            oneshot_source_1_picker,
            oneshot_source_2_picker,
            oneshot_timbre_picker,
        ],
        outputs=prompt,
        queue=False,
        show_progress="hidden",
    )

    clear_oneshot_builder_tags_button.click(
        fn=clear_oneshot_prompt_builder_tags_action,
        inputs=[prompt, oneshot_source_1_picker, oneshot_source_2_picker],
        outputs=[prompt, oneshot_timbre_picker],
        queue=False,
        show_progress="hidden",
    )

    # Define inputs list after UI elements exist
    if inpainting:
        inputs = [
            prompt,
            negative_prompt,
            bars_dropdown,
            bpm_dropdown,
            note_dropdown,
            scale_dropdown,
            sample_type_loop_cb,
            sample_type_oneshot_cb,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
            wetdry_dry_cb,
            wetdry_wet_cb,
            cfg_scale_slider,
            steps_slider,
            preview_every_slider,
            seed_textbox,
            sampler_type_dropdown,
            sigma_min_slider,
            sigma_max_slider,
            cfg_rescale_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
            mask_cropfrom_slider,
            mask_pastefrom_slider,
            mask_pasteto_slider,
            mask_maskstart_slider,
            mask_maskend_slider,
            mask_softnessL_slider,
            mask_softnessR_slider,
            mask_marination_slider,
            generate_midi_checkbox,
            generate_spectrogram_checkbox,
        ]
    else:
        inputs = [
            prompt,
            negative_prompt,
            bars_dropdown,
            bpm_dropdown,
            note_dropdown,
            scale_dropdown,
            sample_type_loop_cb,
            sample_type_oneshot_cb,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
            wetdry_dry_cb,
            wetdry_wet_cb,
            cfg_scale_slider,
            steps_slider,
            preview_every_slider,
            seed_textbox,
            sampler_type_dropdown,
            sigma_min_slider,
            sigma_max_slider,
            cfg_rescale_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider,
            generate_midi_checkbox,
            generate_spectrogram_checkbox,
        ]

    def _generate_cond_from_main_ui(*args):
        """Forward main-tab inputs while keeping optional extras keyword-only.

        The two final UI inputs are intentionally stripped before the call so
        they cannot slide into the older positional mask/output arguments.
        """
        if len(args) < 2:
            raise gr.Error("Generation extras were not supplied by the UI.")

        generation_args = args[:-2]
        output_generate_midi = bool(args[-2])
        output_generate_spectrogram = bool(args[-1])
        return generate_cond(
            *generation_args,
            output_generate_midi=output_generate_midi,
            output_generate_spectrogram=output_generate_spectrogram,
        )

    generate_button.click(
        fn=_generate_cond_from_main_ui,
        inputs=inputs,
        outputs=[
            audio_output,
            audio_spectrogram_output,
            midi_piano_roll_output,
            midi_download_button
        ],
        api_name="generate"
    )

    send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

    def _load_main_model(selected_ckpt_name, selected_config_name, int4_requested, loop_checked):
        result = load_model_action(selected_ckpt_name, selected_config_name, ckpt_files, int4_requested)
        info = result[0] if isinstance(result, (tuple, list)) and len(result) > 0 else "Loaded model."
        details = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else runtime_status_md()
        family = master_prompt_map.get_loop_prompt_family(selected_ckpt_name)
        is_loop = bool(loop_checked)
        return (
            info, details,
            gr.update(visible=is_loop and family == "foundation"),
            gr.update(visible=is_loop and family == "piano"),
            gr.update(visible=is_loop and family == "edm_elements"),
            gr.update(visible=is_loop and family == "vocal_textures"),
            # Foundation Simple/Experimental is a Foundation-loop control, but
            # remains available in One Shot mode where that style selection is
            # still used by the one-shot prompt generator.
            gr.update(visible=(not is_loop) or family == "foundation"),
        )

    load_model_button.click(
        fn=_load_main_model,
        inputs=[model_dropdown, config_dropdown, int4_for_load, sample_type_loop_cb],
        # Model swaps preserve prompt/mode state; only the loop builder panel changes.
        outputs=[current_model_info, status_md, foundation_loop_builder_group, piano_loop_builder_group, edm_loop_builder_group, vocal_loop_builder_group, foundation_mode_group],
    )


    def update_prompt(prompt, lock_bpm, bars, bpm, lock_key, note, scale, seed_str,
                    simple_cb, experimental_cb, prompt_style_is_active, is_oneshot_capable,
                    sample_type_loop_checked, sample_type_oneshot_checked,
                    lock_oneshot_note, oneshot_note_name, oneshot_octave,
                    wetdry_dry_checked, wetdry_wet_checked):

        mode_arg, variant, allow_timbre_mix = prompt_mode_from_style(simple_cb, experimental_cb)
        sample_type = sample_type_from_toggles(sample_type_loop_checked, sample_type_oneshot_checked)
        wetdry = wetdry_from_toggles(wetdry_dry_checked, wetdry_wet_checked)
        resolved_sample_type = normalize_ui_sample_type(sample_type)

        foundation_source_1_update = gr.update()
        foundation_source_2_update = gr.update()
        foundation_timbre_update = gr.update()
        foundation_structure_update = gr.update()
        foundation_musical_update = gr.update()
        oneshot_source_1_update = gr.update()
        oneshot_source_2_update = gr.update()
        oneshot_timbre_update = gr.update()

        piano_type_update = gr.update()
        piano_structure_update = gr.update()
        piano_chord_update = gr.update()
        piano_melody_update = gr.update()
        piano_effect_update = gr.update()
        edm_sound_update = gr.update()
        edm_structure_update = gr.update()
        edm_musical_update = gr.update()
        edm_effect_update = gr.update()
        vocal_type_update = gr.update()
        vocal_structure_update = gr.update()

        if resolved_sample_type == "oneshot":
            note_is_locked = bool(lock_oneshot_note)
            locked_note = build_oneshot_note(oneshot_note_name, oneshot_octave) if note_is_locked else None

            try:
                oneshot_plan = oneshot_prompts.prompt_generator_oneshot_model_router(
                    seed=seed_str,
                    sample_type="oneshot",
                    variant=variant,
                    mode=mode_arg,
                    allow_timbre_mix=allow_timbre_mix,
                    note=locked_note,
                    wetdry=wetdry,
                    include_prefix=False,
                    include_note=False,
                    return_plan=True,
                )
            except TypeError:
                if not note_is_locked:
                    random_note = random.choice(ONESHOT_RANDOM_NOTE_CHOICES)
                    oneshot_note_name, oneshot_octave = split_oneshot_note(random_note)
                oneshot_note = build_oneshot_note(oneshot_note_name, oneshot_octave)
                oneshot_plan = oneshot_prompts.prompt_generator_oneshot_model_router(
                    seed=seed_str,
                    sample_type="oneshot",
                    variant=variant,
                    mode=mode_arg,
                    allow_timbre_mix=allow_timbre_mix,
                    note=oneshot_note,
                    wetdry=wetdry,
                    include_prefix=False,
                    include_note=False,
                )

            if isinstance(oneshot_plan, dict):
                new_prompt = oneshot_plan.get("prompt", "")
                if not note_is_locked:
                    planned_note = oneshot_plan.get("note")
                    if planned_note not in (None, ""):
                        oneshot_note_name, oneshot_octave = split_oneshot_note(planned_note)

                source_1 = str(
                    oneshot_plan.get("instrument_1")
                    or oneshot_plan.get("subfamily")
                    or oneshot_plan.get("family")
                    or ""
                ).strip() or None
                source_2 = str(oneshot_plan.get("instrument_2") or "").strip() or None
                selected_tags = [
                    ONESHOT_BUILDER_TIMBRE_KEYS[str(tag).casefold()]
                    for tag in _dedupe_casefold(oneshot_plan.get("tags") or [])
                    if str(tag).casefold() in ONESHOT_BUILDER_TIMBRE_KEYS
                ]
            else:
                new_prompt = oneshot_plan
                body, _fx, sources = _split_oneshot_builder_prompt(new_prompt)
                source_1 = sources[0] if sources else None
                source_2 = sources[1] if len(sources) > 1 else None
                selected_tags = [
                    ONESHOT_BUILDER_TIMBRE_KEYS[token.casefold()]
                    for token in body
                    if token.casefold() in ONESHOT_BUILDER_TIMBRE_KEYS
                ]

            new_prompt = strip_oneshot_pitch_tokens(new_prompt)
            oneshot_source_1_update = gr.update(value=_normalize_builder_source(source_1, ONESHOT_BUILDER_SOURCE_KEYS))
            oneshot_source_2_update = gr.update(
                value=_normalize_builder_source(source_2, ONESHOT_BUILDER_SOURCE_KEYS) or PROMPT_BUILDER_NONE
            )
            oneshot_timbre_update = gr.update(value=_dedupe_casefold(selected_tags))

        elif prompt_style_is_active:
            loop_family = master_prompt_map.get_loop_prompt_family(LAST_CKPT_NAME or initial_name)
            if loop_family != "foundation":
                legacy_generator = master_prompt_map.get_loop_prompt_generator(LAST_CKPT_NAME or initial_name)
                try:
                    legacy_plan = legacy_generator(return_plan=True)
                except TypeError:
                    legacy_plan = legacy_generator()
                new_prompt = legacy_plan.get("prompt", "") if isinstance(legacy_plan, dict) else str(legacy_plan or "")

                if isinstance(legacy_plan, dict):
                    if loop_family == "piano":
                        piano_type = legacy_plan.get("piano_type")
                        piano_structure = legacy_plan.get("structure")
                        piano_chord = legacy_plan.get("chord_style") or PROMPT_BUILDER_NONE
                        piano_melody = legacy_plan.get("melody_style") or PROMPT_BUILDER_NONE
                        allowed_effects = piano_prompts.effect_choices_for_piano(piano_type)
                        piano_effect = legacy_plan.get("effect")
                        if piano_effect not in allowed_effects:
                            piano_effect = allowed_effects[0]
                        piano_type_update = gr.update(value=piano_type)
                        piano_structure_update = gr.update(value=piano_structure)
                        piano_chord_update = gr.update(value=piano_chord)
                        piano_melody_update = gr.update(value=piano_melody)
                        piano_effect_update = gr.update(choices=allowed_effects, value=piano_effect)
                    elif loop_family == "edm_elements":
                        edm_sound_update = gr.update(value=_dedupe_casefold(legacy_plan.get("sound_tags") or []))
                        edm_structure_update = gr.update(value=legacy_plan.get("structure") or PROMPT_BUILDER_NONE)
                        edm_musical_update = gr.update(value=_dedupe_casefold(legacy_plan.get("musical_tags") or []))
                        edm_effect_update = gr.update(value=_dedupe_casefold(legacy_plan.get("effects") or []))
                    elif loop_family == "vocal_textures":
                        vocal_type_update = gr.update(value=legacy_plan.get("vocal_type") or VOCAL_BUILDER_TYPES[0])
                        vocal_structure_update = gr.update(value=legacy_plan.get("structure") or VOCAL_BUILDER_STRUCTURES[0])
            else:
                legacy_plan = None

            call_kwargs = dict(
                seed=seed_str,
                variant=variant,
                mode=mode_arg,
                allow_timbre_mix=allow_timbre_mix,
                wetdry=wetdry,
                return_plan=True,
            )

            if loop_family == "foundation":
                try:
                    foundation_plan = foundation_prompts.prompt_generator_foundation(**call_kwargs)
                except TypeError:
                    call_kwargs.pop("return_plan", None)
                    foundation_plan = foundation_prompts.prompt_generator_foundation(**call_kwargs)
    
                if isinstance(foundation_plan, dict):
                    new_prompt = foundation_plan.get("prompt", "")
                    source_1 = str(
                        foundation_plan.get("source_1")
                        or foundation_plan.get("instrument_1")
                        or foundation_plan.get("subfamily")
                        or foundation_plan.get("family")
                        or ""
                    ).strip() or None
                    source_2 = str(
                        foundation_plan.get("source_2")
                        or foundation_plan.get("instrument_2")
                        or ""
                    ).strip() or None
                    timbre_tags = [
                        FOUNDATION_BUILDER_TIMBRE_KEYS[str(tag).casefold()]
                        for tag in _dedupe_casefold(
                            foundation_plan.get("timbre_tags")
                            or foundation_plan.get("tags")
                            or []
                        )
                        if str(tag).casefold() in FOUNDATION_BUILDER_TIMBRE_KEYS
                    ]
                    structure = str(foundation_plan.get("musical_structure") or "").strip()
                    musical_tags = [
                        FOUNDATION_BUILDER_MUSICAL_KEYS[str(tag).casefold()]
                        for tag in _dedupe_casefold(foundation_plan.get("musical_tags") or [])
                        if str(tag).casefold() in FOUNDATION_BUILDER_MUSICAL_KEYS
                    ]
                else:
                    new_prompt = foundation_plan
                    _manual, _fx, sources, timbre_tags, structure, musical_tags = _split_foundation_builder_prompt(new_prompt)
                    source_1 = sources[0] if sources else None
                    source_2 = sources[1] if len(sources) > 1 else None
    
                new_prompt = strip_wetdry_tokens(new_prompt)
                foundation_source_1_update = gr.update(
                    value=_normalize_builder_source(source_1, FOUNDATION_BUILDER_SOURCE_KEYS)
                )
                foundation_source_2_update = gr.update(
                    value=_normalize_builder_source(source_2, FOUNDATION_BUILDER_SOURCE_KEYS) or PROMPT_BUILDER_NONE
                )
                foundation_timbre_update = gr.update(value=_dedupe_casefold(timbre_tags))
                foundation_structure_update = gr.update(
                    value=FOUNDATION_BUILDER_STRUCTURE_KEYS.get(str(structure or "").casefold())
                    or PROMPT_BUILDER_NONE
                )
                foundation_musical_update = gr.update(value=_dedupe_casefold(musical_tags))

        else:
            new_prompt = current_prompt_generator()

        if resolved_sample_type == "loop":
            if not lock_bpm:
                bars = random.choice([4, 8])
                bpm = random.choice([100, 110, 120, 128, 130, 140, 150])

            if not lock_key:
                note = random.choice(["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
                scale = random.choice(["major", "minor"])

        return (
            new_prompt,
            bars,
            bpm,
            note,
            scale,
            oneshot_note_name,
            oneshot_octave,
            foundation_source_1_update,
            foundation_source_2_update,
            foundation_timbre_update,
            foundation_structure_update,
            foundation_musical_update,
            oneshot_source_1_update,
            oneshot_source_2_update,
            oneshot_timbre_update,
            piano_type_update,
            piano_structure_update,
            piano_chord_update,
            piano_melody_update,
            piano_effect_update,
            edm_sound_update,
            edm_structure_update,
            edm_musical_update,
            edm_effect_update,
            vocal_type_update,
            vocal_structure_update,
        )



    random_prompt_button.click(
        fn=update_prompt,
        inputs=[
            prompt,
            lock_bpm_checkbox,
            bars_dropdown,
            bpm_dropdown,
            lock_key_checkbox,
            note_dropdown,
            scale_dropdown,
            seed_textbox,
            foundation_simple_cb,
            foundation_experimental_cb,
            prompt_style_active,
            oneshot_active,
            sample_type_loop_cb,
            sample_type_oneshot_cb,
            lock_oneshot_note_checkbox,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
            wetdry_dry_cb,
            wetdry_wet_cb,
        ],
        outputs=[
            prompt,
            bars_dropdown,
            bpm_dropdown,
            note_dropdown,
            scale_dropdown,
            oneshot_note_name_dropdown,
            oneshot_octave_dropdown,
            foundation_source_1_picker,
            foundation_source_2_picker,
            foundation_timbre_picker,
            foundation_structure_picker,
            foundation_musical_picker,
            oneshot_source_1_picker,
            oneshot_source_2_picker,
            oneshot_timbre_picker,
            piano_type_picker,
            piano_structure_picker,
            piano_chord_style_picker,
            piano_melody_style_picker,
            piano_effect_picker,
            edm_sound_picker,
            edm_structure_picker,
            edm_musical_picker,
            edm_effect_picker,
            vocal_type_picker,
            vocal_structure_picker,
        ]
    )

    def refresh_generation_tab_from_runtime(sample_type_loop_checked, sample_type_oneshot_checked, simple_checked, experimental_checked):
        """Refresh Generation-tab UI from the shared loaded-model globals.

        Loading a checkpoint from Keybed or Batch Generation updates the shared
        model/runtime, but Gradio does not automatically re-render components
        in other tabs. This is called when the Generation tab is selected so
        model status, prompt controls, and One Shot visibility match the real
        loaded model.
        """
        loaded_name = LAST_CKPT_NAME or os.path.basename(initial_ckpt or "") or "n/a"
        # Prompt UI availability is no longer inferred from checkpoint filenames.
        prompt_style_visible = True

        loop_checked = bool(sample_type_loop_checked)
        oneshot_checked = bool(sample_type_oneshot_checked)
        if not loop_checked and not oneshot_checked:
            loop_checked = True
        if loop_checked and oneshot_checked:
            # Keep the UI mutually exclusive if stale component state ever
            # reports both as active.
            oneshot_checked = False
        sample_type_visible = True

        show_oneshot_controls = bool(oneshot_checked)
        mode = FOUNDATION_MODE_EXPERIMENTAL if bool(experimental_checked) else FOUNDATION_MODE_SIMPLE
        help_text = FOUNDATION_MODE_HELP.get(mode, FOUNDATION_MODE_HELP[FOUNDATION_MODE_SIMPLE])

        loop_family = master_prompt_map.get_loop_prompt_family(loaded_name)
        show_foundation_mode = show_oneshot_controls or (not show_oneshot_controls and loop_family == "foundation")

        return (
            f"Current Model: `{loaded_name}`",
            runtime_status_md(),
            gr.update(visible=show_foundation_mode),
            gr.update(value=help_text if show_foundation_mode else "", visible=show_foundation_mode),
            prompt_style_visible,
            True,
            gr.update(visible=prompt_style_visible),
            gr.update(visible=sample_type_visible),
            # Read the current sample-type values to choose visibility, but do
            # not write them back. Re-emitting checkbox values during a tab
            # refresh can fire their .change() handlers and swap prompt state.
            gr.update(),
            gr.update(),
            gr.update(visible=not show_oneshot_controls),
            gr.update(visible=show_oneshot_controls),
            gr.update(visible=True),
            gr.update(visible=(not show_oneshot_controls) and master_prompt_map.get_loop_prompt_family(loaded_name) == "foundation"),
            gr.update(visible=(not show_oneshot_controls) and master_prompt_map.get_loop_prompt_family(loaded_name) == "piano"),
            gr.update(visible=(not show_oneshot_controls) and master_prompt_map.get_loop_prompt_family(loaded_name) == "edm_elements"),
            gr.update(visible=(not show_oneshot_controls) and master_prompt_map.get_loop_prompt_family(loaded_name) == "vocal_textures"),
            gr.update(visible=show_oneshot_controls),
        )

    return {
        "refresh_fn": refresh_generation_tab_from_runtime,
        "refresh_inputs": [
            sample_type_loop_cb,
            sample_type_oneshot_cb,
            foundation_simple_cb,
            foundation_experimental_cb,
        ],
        "refresh_outputs": [
            current_model_info,
            status_md,
            foundation_mode_group,
            foundation_mode_help,
            prompt_style_active,
            oneshot_active,
            model_configurator_group,
            sample_type_group,
            sample_type_loop_cb,
            sample_type_oneshot_cb,
            loop_controls_group,
            oneshot_controls_group,
            prompt_builder_group,
            foundation_loop_builder_group,
            piano_loop_builder_group,
            edm_loop_builder_group,
            vocal_loop_builder_group,
            oneshot_prompt_builder_group,
        ],
    }


def create_txt2audio_ui(model_config, initial_ckpt):
    css = """
    /* Make the wrapper seamless (no panel look) */
    #foundation_mode_group,
    #model_configurator_group,
    #sample_type_group,
    #wetdry_group {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }

    /* Make the markdown itself seamless */
    #foundation_mode_help {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin-top: 0.25rem !important;
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
    }

    /* Gradio markdown inner wrapper(s) sometimes hold the scroll */
    #foundation_mode_help .prose,
    #foundation_mode_help > div {
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
    }

    /* Text sizing (adjust these) */
    #foundation_mode_help { font-size: 0.85rem; line-height: 1.2; }
    #foundation_mode_help h3 { font-size: 0.95rem; margin: 0.15rem 0; }
    #foundation_mode_help ul { margin: 0.15rem 0 0.15rem 1.0rem; }

    #sample_type_label,
    #wetdry_label {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0.35rem 0 0.15rem !important;
        width: 100% !important;
        text-align: center !important;
        text-decoration: underline !important;
        font-weight: 600 !important;
        font-size: 0.85rem;
        line-height: 1.2;
    }

    #sample_type_label .prose,
    #sample_type_label > div,
    #wetdry_label .prose,
    #wetdry_label > div {
        width: 100% !important;
        text-align: center !important;
    }
    
    /* Make the top row stretch children to the tallest column */
#top_prompt_row { align-items: stretch !important; }

    /* Make the left column fill the row and behave like a vertical flex stack */
    #prompt_left_col {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    }

    /* Make the prompt component take all remaining vertical space in the left column */
    #prompt_box {
    flex: 1 1 auto !important;
    min-height: 0 !important; /* important so it can shrink when right side shrinks */
    }

    /* Gradio wraps components; force wrappers to stretch too */
    #prompt_box > .wrap {
    height: 100% !important;
    }

    /* Make the actual textarea fill the component */
    #prompt_box textarea {
    height: 100% !important;
    min-height: 0 !important;
    resize: none; /* optional */
    }
        
    """

    with gr.Blocks(css=css) as ui:
        with gr.Tab("Generation") as generation_tab:
            generation_tab_handles = create_sampling_ui(model_config, initial_ckpt)
        with gr.Tab("Keybed") as keybed_tab:
            keybed_tab_handles = build_keybed_tab(
                get_runtime_for_keybed,
                config=config,
                initial_ckpt=initial_ckpt,
                get_models_and_configs=get_models_and_configs,
                get_config_files=get_config_files,
                update_config_dropdown=update_config_dropdown,
                load_model_action=load_model_action,
                runtime_status_md=runtime_status_md,
                torchao_int4_supported=TORCHAO_INT4_SUPPORTED,
            )
        with gr.Tab("Layered Keybeds") as layered_keybed_tab:
            layered_keybed_tab_handles = build_layered_keybed_tab(
                get_runtime_for_keybed,
                config=config,
                initial_ckpt=initial_ckpt,
                get_models_and_configs=get_models_and_configs,
                get_config_files=get_config_files,
                update_config_dropdown=update_config_dropdown,
                load_model_action=load_model_action,
                runtime_status_md=runtime_status_md,
                torchao_int4_supported=TORCHAO_INT4_SUPPORTED,
            )
        with gr.Tab("Batch Generation") as batch_generation_tab:
            batch_generation_tab_handles = build_batch_generation_tab(
                config=config,
                initial_ckpt=initial_ckpt,
                get_runtime=get_runtime_for_keybed,
                get_models_and_configs=get_models_and_configs,
                get_config_files=get_config_files,
                update_config_dropdown=update_config_dropdown,
                load_model_action=load_model_action,
                runtime_status_md=runtime_status_md,
                generate_cond=generate_cond,
                torchao_int4_supported=TORCHAO_INT4_SUPPORTED,
            )
        with gr.Tab("Download Models"):
            build_model_download_ui()

        # Cross-tab runtime sync. The model object/global runtime is shared, but
        # Gradio component visibility/status is per-tab. Refresh each tab from
        # the shared runtime whenever it is selected.
        if generation_tab_handles:
            generation_tab.select(
                fn=generation_tab_handles["refresh_fn"],
                inputs=generation_tab_handles["refresh_inputs"],
                outputs=generation_tab_handles["refresh_outputs"],
            )

        if keybed_tab_handles:
            keybed_tab.select(
                fn=keybed_tab_handles["refresh_fn"],
                inputs=keybed_tab_handles.get("refresh_inputs", []),
                outputs=keybed_tab_handles["refresh_outputs"],
            )

        if layered_keybed_tab_handles:
            layered_keybed_tab.select(
                fn=layered_keybed_tab_handles["refresh_fn"],
                inputs=layered_keybed_tab_handles.get("refresh_inputs", []),
                outputs=layered_keybed_tab_handles["refresh_outputs"],
            )

        if batch_generation_tab_handles:
            batch_generation_tab.select(
                fn=batch_generation_tab_handles["refresh_fn"],
                inputs=batch_generation_tab_handles.get("refresh_inputs", []),
                outputs=batch_generation_tab_handles["refresh_outputs"],
            )
    return ui

def create_diffusion_uncond_ui(model_config):
    with gr.Blocks() as ui:
        create_uncond_sampling_ui(model_config)

    return ui

def autoencoder_process(audio, latent_noise, n_quantizers):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.transpose(0, 1)

    audio = model.preprocess_audio_for_encoder(audio, in_sr)
    # Note: If you need to do chunked encoding, to reduce VRAM,
    # then add these arguments to encode_audio and decode_audio: chunked=True, overlap=32, chunk_size=128
    # To turn it off, do chunked=False
    # Optimal overlap and chunk_size values will depend on the model.
    # See encode_audio & decode_audio in autoencoders.py for more info
    # Get dtype of model
    dtype = next(model.parameters()).dtype

    audio = audio.to(dtype)

    if n_quantizers > 0:
        latents = model.encode_audio(audio, chunked=False, n_quantizers=n_quantizers)
    else:
        latents = model.encode_audio(audio, chunked=False)

    if latent_noise > 0:
        latents = latents + torch.randn_like(latents) * latent_noise

    audio = model.decode_audio(latents, chunked=False)

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_autoencoder_ui(model_config):

    is_dac_rvq = "model" in model_config and "bottleneck" in model_config["model"] and model_config["model"]["bottleneck"]["type"] in ["dac_rvq","dac_rvq_vae"]

    if is_dac_rvq:
        n_quantizers = model_config["model"]["bottleneck"]["config"]["n_codebooks"]
    else:
        n_quantizers = 0

    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        n_quantizers_slider = gr.Slider(minimum=1, maximum=n_quantizers, step=1, value=n_quantizers, label="# quantizers", visible=is_dac_rvq)
        latent_noise_slider = gr.Slider(minimum=0.0, maximum=10.0, step=0.001, value=0.0, label="Add latent noise")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=autoencoder_process, inputs=[input_audio, latent_noise_slider, n_quantizers_slider], outputs=output_audio, api_name="process")

    return ui

def diffusion_prior_process(audio, steps, sampler_type, sigma_min, sigma_max):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Get the device from the model
    device = next(model.parameters()).device

    in_sr, audio = audio

    audio = torch.from_numpy(audio).float().div(32767).to(device)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1) # [n, 2] -> [2, n]

    audio = audio.unsqueeze(0)

    audio = model.stereoize(audio, in_sr, steps, sampler_kwargs={"sampler_type": sampler_type, "sigma_min": sigma_min, "sigma_max": sigma_max})

    audio = rearrange(audio, "b d n -> d (b n)")

    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", audio, sample_rate)

    return "output.wav"

def create_diffusion_prior_ui(model_config):
    with gr.Blocks() as ui:
        input_audio = gr.Audio(label="Input audio")
        output_audio = gr.Audio(label="Output audio", interactive=False)
        # Sampler params
        with gr.Row():
            steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
            sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
            sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
            sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
        process_button = gr.Button("Process", variant='primary', scale=1)
        process_button.click(fn=diffusion_prior_process, inputs=[input_audio, steps_slider, sampler_type_dropdown, sigma_min_slider, sigma_max_slider], outputs=output_audio, api_name="process")

    return ui

def create_lm_ui(model_config):
    with gr.Blocks() as ui:
        output_audio = gr.Audio(label="Output audio", interactive=False)
        audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
        midi_piano_roll_output = gr.Image(label="MIDI Piano Roll", interactive=False)

        # Sampling params
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0, maximum=5, step=0.01, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.95, label="Top p")
            top_k_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Top k")

        generate_button = gr.Button("Generate", variant='primary', scale=1)
        generate_button.click(
            fn=generate_lm,
            inputs=[
                temperature_slider,
                top_p_slider,
                top_k_slider
            ],
            outputs=[output_audio, audio_spectrogram_output, midi_piano_roll_output],
            api_name="generate"
        )

    return ui

def create_ui(
    model_config_path=None,
    ckpt_path=None,
    pretrained_name=None,
    pretransform_ckpt_path=None,
    model_half=False,
    gradio_title=None,
    **kwargs
):
    global global_model_half
    global current_prompt_generator
    global model
    global DEVICE
    global PREFERRED_DTYPE, TORCHAO_INT4_SUPPORTED
    global LAST_CKPT_PATH, LAST_CONFIG_PATH, LAST_CKPT_NAME, LAST_MODEL_CONFIG, INT4_ENABLED

    global_model_half = model_half  # keep for upstream compatibility

    # If nothing specified, try default model in models dir
    if pretrained_name is None and model_config_path is None and ckpt_path is None:
        print("checking the models folder for a default checkpoint")
        try:
            ckpt_files = get_models_and_configs(config['models_directory'])
            ckpt_path = ckpt_files[0][1]
            configs = get_config_files(ckpt_path)
            model_config_path = os.path.join(os.path.dirname(ckpt_path), configs[0])
        except IndexError:
            print("no default checkpoint.")
            with gr.Blocks() as ui:
                build_model_download_ui(initialize=True)
            return ui

    # Exactly one of: pretrained_name OR (model_config_path + ckpt_path)
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), \
        "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    # Load model config dict if using local ckpt
    if model_config_path is not None:
        with open(model_config_path) as f:
            cfg_dict = json.load(f)   # <-- keep the dict around for LAST_MODEL_CONFIG
    else:
        cfg_dict = None

    # Device selection
    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    DEVICE = device

    # Precision / capability checks (cheap)
    PREFERRED_DTYPE = pick_preferred_dtype(device)
    TORCHAO_INT4_SUPPORTED = check_torchao_int4_support(device)
    
    ok, ext, ver = torchao_backend_status()
    print(f"[torchao] version={ver} | {ext}")

    # This is just for display / prompt generator mapping
    initial_ckpt = ckpt_path if ckpt_path is not None else pretrained_name
    initial_name = os.path.basename(initial_ckpt) if initial_ckpt else None

    # --- load model (IMPORTANT: assign to global `model`) ---
    model, loaded_model_config = load_model(
        model_config=cfg_dict,
        model_ckpt_path=ckpt_path,
        pretrained_name=pretrained_name,
        pretransform_ckpt_path=pretransform_ckpt_path,
        device=device,
        preferred_dtype=PREFERRED_DTYPE,
    )

    # --- set LAST_* so INT4 toggle can reload even before user hits "Load Model" ---
    LAST_CKPT_PATH = ckpt_path
    LAST_CONFIG_PATH = model_config_path
    LAST_CKPT_NAME = initial_name

    # Use the dict we loaded from disk (for reload); if pretrained, keep None
    LAST_MODEL_CONFIG = cfg_dict
    INT4_ENABLED = False

    # prompt generator based on initial model name
    current_prompt_generator = master_prompt_map.get_prompt_generator(initial_name)

    model_type = loaded_model_config["model_type"]

    if model_type == "diffusion_cond":
        ui = create_txt2audio_ui(loaded_model_config, initial_ckpt)
    elif model_type == "diffusion_uncond":
        ui = create_diffusion_uncond_ui(loaded_model_config)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(loaded_model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(loaded_model_config)
    elif model_type == "lm":
        ui = create_lm_ui(loaded_model_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return ui
