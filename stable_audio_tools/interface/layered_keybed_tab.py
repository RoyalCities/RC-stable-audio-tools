from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import re
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import torch
import torchaudio

from .keybed_exporter.tri_layered_exporter import (
    TRI_LAYER_DEFAULT_VOLUMES,
    TRI_LAYER_MASTER_VOLUME,
    export_tri_layered_keybed_run,
)
from .keybed_tab import (
    KEYBED_CHUNK_SIZE,
    KEYBED_DEFAULT_CFG,
    KEYBED_DEFAULT_CFG_RESCALE,
    KEYBED_DEFAULT_SAMPLER,
    KEYBED_DEFAULT_SIGMA_MAX,
    KEYBED_DEFAULT_SIGMA_MIN,
    KEYBED_DEFAULT_STEPS,
    KEYBED_FULL_SAMPLER_DEFAULT_RANGE,
    KEYBED_FULL_SAMPLER_RANGE_CHOICES,
    KEYBED_OCTAVE_CHOICES,
    KEYBED_ROOT_NOTE_NAMES,
    KEYBED_ROOT_OCTAVES,
    KEYBED_NO_SECOND_INSTRUMENT,
    KEYBED_TAG_PICKER_INSTRUMENTS,
    KEYBED_TAG_PICKER_OPTIONAL_INSTRUMENTS,
    KEYBED_TAG_PICKER_TIMBRES,
    _chromatic_notes_from_root,
    _chunk_notes,
    _clamp_preview_root,
    _concat_audio,
    _generate_keybed_sequence_tensor,
    _get_runtime,
    _model_device,
    _normalize_instrument_name,
    _instrument_name_slug,
    _instrument_name_textbox,
    _normalize_descriptor_for_seed_reuse,
    _note_filename,
    _note_slices_for_sequence,
    _require_keybed_runtime,
    _resolve_full_sampler_range,
    _resolve_seed,
    _runtime_model,
    _runtime_model_name,
    _runtime_output_directory,
    _runtime_sample_rate,
    _runtime_status_md,
    _safe_filename_part,
    _searchable_dropdown,
    _split_note,
    _stable_int_seed,
    _tensor_to_i16_cpu,
    _wetdry_from_label,
    _write_keybed_run_metadata,
    apply_keybed_tag_picker_action,
    clear_keybed_tag_picker_action,
    open_keybed_export_folder_action,
    random_keybed_descriptor_action,
    set_keybed_instrument_mode_action,
)
from .prompts import keybed_prompts


TRI_OUTPUT_SUBDIR = "Layered_Keybed_Source"
TRI_PREVIEW_OUTPUT_SUBDIR = "Keybed_Previews"
TRI_LAYER_ROLES = ("Main Instrument", "Support Instrument 1", "Support Instrument 2")
TRI_LAYER_DIR_NAMES = ("layer_1_main", "layer_2_support", "layer_3_support")
TRI_BUTTON_LABELS = {
    "preview": "Generate Layered Preview",
    "full": "Generate & Export Layered Keybed",
}


def _tri_cli_status(message: str) -> None:
    print(f"[Layered Keybed] {message}", flush=True)


def _descriptor_title(descriptor: str, fallback: str) -> str:
    tokens = [part.strip() for part in str(descriptor or "").split(",") if part.strip()]
    return " + ".join(tokens[:2]) if tokens else fallback


def _descriptor_bundle(descriptors: Sequence[str]) -> str:
    normalized = [_normalize_descriptor_for_seed_reuse(value) for value in descriptors]
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _layer_generation_seeds(base_seed: int) -> Tuple[int, int, int]:
    return tuple(
        int(_stable_int_seed(base_seed, key))
        for key in ("tri_main", "tri_support_1", "tri_support_2")
    )


def _prompt_seed_for_layer(seed_str, layer_index: int):
    text = str(seed_str if seed_str is not None else "-1").strip()
    if text in {"", "-1"}:
        return "-1"
    try:
        base = int(text)
    except Exception:
        return text
    return str(_stable_int_seed(base, f"tri_prompt_{layer_index + 1}"))


def _make_tri_parent_dir(
    output_root: str,
    descriptors: Sequence[str],
    *,
    seed: int,
    range_hint: str,
    instrument_name: str | None = None,
    output_subdir: str = TRI_OUTPUT_SUBDIR,
) -> str:
    title_parts = [
        _safe_filename_part(_descriptor_title(descriptor, f"layer_{index + 1}"), max_chars=24)
        for index, descriptor in enumerate(descriptors)
    ]
    title_slug = _instrument_name_slug(instrument_name) or _safe_filename_part("_".join(title_parts), max_chars=72)
    digest = hashlib.sha1(
        f"{time.time_ns()}|{seed}|{range_hint}|{'|'.join(descriptors)}|{_normalize_instrument_name(instrument_name)}".encode(
            "utf-8", errors="replace"
        )
    ).hexdigest()[:8]
    base = os.path.join(output_root, str(output_subdir), f"{title_slug}_{digest}")
    candidate = base
    suffix = 1
    while os.path.exists(candidate):
        suffix += 1
        candidate = f"{base}_{suffix:02d}"
    os.makedirs(candidate, exist_ok=True)
    return candidate


def _mix_aligned_layers(
    layer_audio: Sequence[torch.Tensor],
    *,
    volumes: Sequence[float] = TRI_LAYER_DEFAULT_VOLUMES,
    master_volume: float = TRI_LAYER_MASTER_VOLUME,
) -> torch.Tensor:
    if len(layer_audio) != 3:
        raise ValueError("Layered preview requires exactly three audio tensors.")
    min_channels = min(int(audio.shape[0]) for audio in layer_audio)
    min_samples = min(int(audio.shape[-1]) for audio in layer_audio)
    if min_channels <= 0 or min_samples <= 0:
        raise ValueError("One or more preview layers contain no audio.")

    mix = torch.zeros((min_channels, min_samples), dtype=torch.float32)
    for audio, volume in zip(layer_audio, volumes):
        mix += audio[:min_channels, :min_samples].to(torch.float32).cpu() * float(volume)
    return (mix * float(master_volume)).clamp(-1.0, 1.0).contiguous()



def _gain_percent(value) -> float:
    """Convert a 0-100 UI value into the 0-1 gain used by preview/export."""
    try:
        percent = float(value)
    except Exception:
        percent = 0.0
    return max(0.0, min(100.0, percent)) / 100.0


def _default_gain_percent(value) -> float:
    """Translate the exporter's normalized defaults into slider percentages."""
    try:
        gain = float(value)
    except Exception:
        gain = 0.0
    if gain <= 1.0:
        gain *= 100.0
    return max(0.0, min(100.0, gain))


def _mix_settings_from_ui(
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
) -> Tuple[Tuple[float, float, float], float]:
    return (
        (
            _gain_percent(main_volume_percent),
            _gain_percent(support_1_volume_percent),
            _gain_percent(support_2_volume_percent),
        ),
        _gain_percent(master_volume_percent),
    )


def _layer_preview_path(parent_dir: str, layer_index: int) -> str:
    return os.path.join(
        str(parent_dir),
        TRI_LAYER_DIR_NAMES[int(layer_index)],
        "layer_preview.wav",
    )


def _cleanup_legacy_mixed_previews(parent_dir: str) -> None:
    """Remove obsolete one-file-per-adjustment mixes from older UI revisions.

    The current mixer uses only two stable ping-pong files. Legacy files are
    safe to prune because none of them is returned by this revision.
    """
    parent = os.path.abspath(os.path.expanduser(str(parent_dir or "").strip()))
    try:
        names = os.listdir(parent)
    except OSError:
        return

    protected = {
        "tri_layer_preview_mix_a.wav",
        "tri_layer_preview_mix_b.wav",
    }
    for name in names:
        lower = name.lower()
        is_legacy_mix = (
            lower == "tri_layer_preview_current_mix.wav"
            or (
                lower.startswith("tri_layer_preview_mix_")
                and lower.endswith(".wav")
                and lower not in protected
            )
        )
        if not is_legacy_mix:
            continue
        try:
            os.remove(os.path.join(parent, name))
        except OSError:
            # An older file may still be open in the browser on Windows. It can
            # be left alone without affecting the two-slot mixer.
            pass


def _mixed_preview_path(parent_dir: str) -> str:
    """Choose the inactive file from a two-slot preview-mix ring.

    Reusing one exact filepath can leave Gradio/the browser serving cached
    media, while creating a new timestamped filename for every adjustment
    leaves redundant files behind. Alternating between A and B keeps refreshes
    cache-safe and caps the current mixer at two WAV files.
    """
    parent = os.path.abspath(os.path.expanduser(str(parent_dir or "").strip()))
    if not parent:
        raise gr.Error("The layered preview folder is not available.")
    os.makedirs(parent, exist_ok=True)
    _cleanup_legacy_mixed_previews(parent)

    slot_a = os.path.join(parent, "tri_layer_preview_mix_a.wav")
    slot_b = os.path.join(parent, "tri_layer_preview_mix_b.wav")

    if not os.path.exists(slot_a):
        return slot_a
    if not os.path.exists(slot_b):
        return slot_b

    # The newest file is normally the one currently displayed. Overwrite the
    # older/inactive slot so Windows is not asked to replace an open media file.
    try:
        a_mtime = os.path.getmtime(slot_a)
    except OSError:
        a_mtime = 0.0
    try:
        b_mtime = os.path.getmtime(slot_b)
    except OSError:
        b_mtime = 0.0
    return slot_a if a_mtime <= b_mtime else slot_b


def _write_mixed_preview_from_tensors(
    parent_dir: str,
    layer_audio: Sequence[torch.Tensor],
    *,
    sample_rate: int,
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
) -> Tuple[str, torch.Tensor]:
    volumes, master_volume = _mix_settings_from_ui(
        main_volume_percent,
        support_1_volume_percent,
        support_2_volume_percent,
        master_volume_percent,
    )
    combined = _mix_aligned_layers(
        layer_audio,
        volumes=volumes,
        master_volume=master_volume,
    )

    preview_path = _mixed_preview_path(parent_dir)
    temp_path = preview_path + ".partial.wav"
    try:
        torchaudio.save(temp_path, _tensor_to_i16_cpu(combined), int(sample_rate))
        os.replace(temp_path, preview_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if not os.path.isfile(preview_path):
        raise gr.Error(f"The remixed preview was not created: {preview_path}")
    file_size = os.path.getsize(preview_path)
    if file_size <= 44:
        raise gr.Error(f"The remixed preview file is empty: {preview_path}")

    # The two-slot ring preserves the currently served file and overwrites
    # only the inactive slot. No timestamped mix-file buildup is created.
    _tri_cli_status(
        f"Preview mix WAV written: {preview_path} ({file_size:,} bytes)."
    )
    return preview_path, combined


def update_tri_preview_mix_action(
    parent_dir,
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
):
    """Rebuild only the audition mix from the three previews already on disk."""
    parent_dir = os.path.abspath(os.path.expanduser(str(parent_dir or "").strip()))
    if not parent_dir or not os.path.isdir(parent_dir):
        return (
            gr.update(),
            "Generate a layered preview before updating the mix.",
        )

    layer_paths = [_layer_preview_path(parent_dir, index) for index in range(3)]
    missing = [path for path in layer_paths if not os.path.isfile(path)]
    if missing:
        missing_text = "<br>".join(f"`{path}`" for path in missing)
        return (
            gr.update(),
            "The saved layer previews were not found. Generate a new layered preview first.<br>"
            + missing_text,
        )

    layer_audio: List[torch.Tensor] = []
    sample_rate = None
    for path in layer_paths:
        audio, current_sr = torchaudio.load(path)
        if sample_rate is None:
            sample_rate = int(current_sr)
        elif int(current_sr) != int(sample_rate):
            raise gr.Error("The saved layer previews do not share one sample rate.")
        layer_audio.append(audio.to(torch.float32).cpu())

    resolved_sample_rate = int(sample_rate or 32000)
    preview_path, combined = _write_mixed_preview_from_tensors(
        parent_dir,
        layer_audio,
        sample_rate=resolved_sample_rate,
        main_volume_percent=main_volume_percent,
        support_1_volume_percent=support_1_volume_percent,
        support_2_volume_percent=support_2_volume_percent,
        master_volume_percent=master_volume_percent,
    )
    peak = float(combined.abs().max().item()) if combined.numel() else 0.0
    rms = float(torch.sqrt(torch.mean(combined.square())).item()) if combined.numel() else 0.0
    file_size = os.path.getsize(preview_path)
    _tri_cli_status(
        "Preview mix updated from saved layers: "
        f"Main={float(main_volume_percent):.0f}% "
        f"Support1={float(support_1_volume_percent):.0f}% "
        f"Support2={float(support_2_volume_percent):.0f}% "
        f"Master={float(master_volume_percent):.0f}% "
        f"peak={peak:.6f} rms={rms:.6f}."
    )

    status = (
        "Preview mix updated from disk — no diffusion rerun. "
        f"Main `{float(main_volume_percent):.0f}%` · "
        f"Support 1 `{float(support_1_volume_percent):.0f}%` · "
        f"Support 2 `{float(support_2_volume_percent):.0f}%` · "
        f"Master `{float(master_volume_percent):.0f}%`  \n"
        f"Output peak: `{peak:.5f}` · RMS: `{rms:.5f}` · `{file_size:,}` bytes  \n"
        f"Saved to: `{preview_path}`"
    )
    # Return the inactive ping-pong filepath directly to the Audio component.
    # The alternating path forces a media refresh without accumulating files.
    return (
        gr.update(
            value=preview_path,
            visible=True,
            label="Layered Preview — Current Mix",
        ),
        status,
    )


MIX_CHANNEL_NAMES = ("Main", "Support 1", "Support 2", "Master")


def _clamp_mix_percent(value, fallback=0.0) -> float:
    try:
        resolved = float(value)
    except Exception:
        resolved = float(fallback)
    return max(0.0, min(100.0, resolved))


def _normalize_mix_saved_volumes(saved_values, current_values) -> List[float]:
    saved = []
    for index in range(4):
        current = _clamp_mix_percent(current_values[index])
        fallback = current if current > 0.0 else 100.0
        try:
            value = saved_values[index]
        except Exception:
            value = fallback
        saved.append(_clamp_mix_percent(value, fallback=fallback))
    return saved


def _remember_nonzero_mix_volumes(current_values, saved_values) -> List[float]:
    saved = _normalize_mix_saved_volumes(saved_values, current_values)
    for index, value in enumerate(current_values):
        resolved = _clamp_mix_percent(value)
        if resolved > 0.0:
            saved[index] = resolved
    return saved


def _normalize_mix_bool_list(values) -> List[bool]:
    normalized = [bool(value) for value in list(values or [])[:4]]
    while len(normalized) < 4:
        normalized.append(False)
    return normalized


def _normalize_active_solo(value) -> int:
    try:
        index = int(value)
    except Exception:
        return -1
    return index if 0 <= index < 4 else -1


def _remix_after_mixer_control(parent_dir, volumes):
    return update_tri_preview_mix_action(
        parent_dir,
        volumes[0],
        volumes[1],
        volumes[2],
        volumes[3],
    )


def update_tri_preview_mix_with_memory_action(
    parent_dir,
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
    saved_main_volume,
    saved_support_1_volume,
    saved_support_2_volume,
    saved_master_volume,
):
    """Rebuild the mix and remember every channel's latest non-zero level."""
    volumes = [
        _clamp_mix_percent(main_volume_percent),
        _clamp_mix_percent(support_1_volume_percent),
        _clamp_mix_percent(support_2_volume_percent),
        _clamp_mix_percent(master_volume_percent),
    ]
    saved = _remember_nonzero_mix_volumes(
        volumes,
        [
            saved_main_volume,
            saved_support_1_volume,
            saved_support_2_volume,
            saved_master_volume,
        ],
    )
    preview_update, status = _remix_after_mixer_control(parent_dir, volumes)
    return preview_update, status, *saved


def toggle_tri_mix_solo_action(
    channel_index,
    requested_solo,
    parent_dir,
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
    main_solo,
    support_1_solo,
    support_2_solo,
    master_solo,
    main_mute,
    support_1_mute,
    support_2_mute,
    master_mute,
    saved_main_volume,
    saved_support_1_volume,
    saved_support_2_volume,
    saved_master_volume,
    active_solo,
    pre_solo_mutes,
):
    """Apply one-at-a-time solo while retaining every channel's prior level.

    Master Solo means audition the complete saved mix through the master bus.
    This avoids the impossible/silent interpretation of muting every source
    while leaving only the master bus active.
    """
    index = max(0, min(3, int(channel_index)))
    volumes = [
        _clamp_mix_percent(main_volume_percent),
        _clamp_mix_percent(support_1_volume_percent),
        _clamp_mix_percent(support_2_volume_percent),
        _clamp_mix_percent(master_volume_percent),
    ]
    solos = _normalize_mix_bool_list(
        [main_solo, support_1_solo, support_2_solo, master_solo]
    )
    mutes = _normalize_mix_bool_list(
        [main_mute, support_1_mute, support_2_mute, master_mute]
    )
    saved = _remember_nonzero_mix_volumes(
        volumes,
        [
            saved_main_volume,
            saved_support_1_volume,
            saved_support_2_volume,
            saved_master_volume,
        ],
    )
    active = _normalize_active_solo(active_solo)
    original_mutes = _normalize_mix_bool_list(pre_solo_mutes)

    if bool(requested_solo):
        if active < 0:
            original_mutes = list(mutes)
        active = index
        solos = [slot == index for slot in range(4)]

        if index < 3:
            # Solo one actual source. Keep the master bus audible at its saved
            # level while the two non-solo sources become effective mutes.
            for slot in range(3):
                if slot == index:
                    volumes[slot] = saved[slot]
                    mutes[slot] = False
                else:
                    volumes[slot] = 0.0
                    mutes[slot] = True
            volumes[3] = saved[3]
            mutes[3] = False
        else:
            # Audition the master bus as the complete saved mix.
            volumes = list(saved)
            mutes = [False, False, False, False]
    elif active == index:
        active = -1
        solos = [False, False, False, False]
        mutes = list(original_mutes)
        volumes = [
            0.0 if mutes[slot] else saved[slot]
            for slot in range(4)
        ]
        original_mutes = list(mutes)
    else:
        solos = [slot == active for slot in range(4)] if active >= 0 else [False] * 4

    preview_update, status = _remix_after_mixer_control(parent_dir, volumes)
    solo_text = MIX_CHANNEL_NAMES[active] if active >= 0 else "Off"
    status = f"{status}  \nSolo: `{solo_text}`"

    return (
        *[gr.update(value=value) for value in volumes],
        *[gr.update(value=value) for value in solos],
        *[gr.update(value=value) for value in mutes],
        *saved,
        active,
        original_mutes,
        preview_update,
        status,
    )


def toggle_tri_mix_mute_action(
    channel_index,
    requested_mute,
    parent_dir,
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
    main_solo,
    support_1_solo,
    support_2_solo,
    master_solo,
    main_mute,
    support_1_mute,
    support_2_mute,
    master_mute,
    saved_main_volume,
    saved_support_1_volume,
    saved_support_2_volume,
    saved_master_volume,
    active_solo,
    pre_solo_mutes,
):
    """Mute one mixer channel by setting its effective slider level to zero."""
    index = max(0, min(3, int(channel_index)))
    volumes = [
        _clamp_mix_percent(main_volume_percent),
        _clamp_mix_percent(support_1_volume_percent),
        _clamp_mix_percent(support_2_volume_percent),
        _clamp_mix_percent(master_volume_percent),
    ]
    solos = _normalize_mix_bool_list(
        [main_solo, support_1_solo, support_2_solo, master_solo]
    )
    mutes = _normalize_mix_bool_list(
        [main_mute, support_1_mute, support_2_mute, master_mute]
    )
    saved = _remember_nonzero_mix_volumes(
        volumes,
        [
            saved_main_volume,
            saved_support_1_volume,
            saved_support_2_volume,
            saved_master_volume,
        ],
    )
    active = _normalize_active_solo(active_solo)
    original_mutes = _normalize_mix_bool_list(pre_solo_mutes)

    mutes[index] = bool(requested_mute)
    volumes[index] = 0.0 if mutes[index] else saved[index]

    # Source solo remains exclusive. Programmatically trying to unmute another
    # source while one is soloed leaves that non-solo source silent.
    if 0 <= active < 3 and index < 3 and index != active:
        mutes[index] = True
        volumes[index] = 0.0

    preview_update, status = _remix_after_mixer_control(parent_dir, volumes)
    muted_names = [MIX_CHANNEL_NAMES[i] for i, muted in enumerate(mutes) if muted]
    mute_text = ", ".join(muted_names) if muted_names else "None"
    status = f"{status}  \nMuted: `{mute_text}`"

    return (
        *[gr.update(value=value) for value in volumes],
        *[gr.update(value=value) for value in solos],
        *[gr.update(value=value) for value in mutes],
        *saved,
        active,
        original_mutes,
        preview_update,
        status,
    )


def _layer_random_action(
    layer_index: int,
    mode_label,
    wetdry_label,
    instrument_mode_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    # Only the Main Instrument is allowed to update the one shared preview root.
    lock_for_this_layer = bool(root_locked) or layer_index != 0
    return random_keybed_descriptor_action(
        mode_label,
        wetdry_label,
        instrument_mode_label,
        _prompt_seed_for_layer(seed_str, layer_index),
        lock_for_this_layer,
        current_root_note,
        current_root_octave,
    )


def randomize_all_tri_layers_action(
    mode_label,
    wetdry_label,
    instrument_mode_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    # Resolve one base prompt seed so -1 still produces three different but
    # internally related prompt draws in a single button press.
    base_prompt_seed = keybed_prompts.resolve_keybed_seed(seed_str)
    results = []
    root_name = current_root_note
    root_octave = current_root_octave
    for layer_index in range(3):
        layer_seed = str(_stable_int_seed(base_prompt_seed, f"tri_prompt_{layer_index + 1}"))
        result = random_keybed_descriptor_action(
            mode_label,
            wetdry_label,
            instrument_mode_label,
            layer_seed,
            bool(root_locked) or layer_index != 0,
            root_name,
            root_octave,
        )
        results.append(result)
        if layer_index == 0:
            root_name = result[1]
            root_octave = result[2]

    # Descriptor/pickers for each layer, followed by the shared root controls.
    return (
        results[0][0], results[0][3], results[0][4], results[0][5],
        results[1][0], results[1][3], results[1][4], results[1][5],
        results[2][0], results[2][3], results[2][4], results[2][5],
        results[0][1], results[0][2],
    )


def set_all_tri_layer_instrument_modes_action(
    instrument_mode_label,
    descriptor_0, instrument_1_0, instrument_2_0, timbres_0,
    descriptor_1, instrument_1_1, instrument_2_1, timbres_1,
    descriptor_2, instrument_1_2, instrument_2_2, timbres_2,
    wetdry_label,
    seed_str,
    root_locked,
    current_root_note,
    current_root_octave,
):
    """Apply one shared Single/Hybrid choice to all three visible builders."""
    layer_values = [
        (descriptor_0, instrument_1_0, instrument_2_0, timbres_0),
        (descriptor_1, instrument_1_1, instrument_2_1, timbres_1),
        (descriptor_2, instrument_1_2, instrument_2_2, timbres_2),
    ]
    results = []
    for layer_index, (descriptor, instrument_1, instrument_2, timbres) in enumerate(layer_values):
        results.append(
            set_keybed_instrument_mode_action(
                instrument_mode_label,
                descriptor,
                instrument_1,
                instrument_2,
                timbres,
                wetdry_label,
                _prompt_seed_for_layer(seed_str, layer_index),
                bool(root_locked) or layer_index != 0,
                current_root_note,
                current_root_octave,
            )
        )

    return (
        results[0][0], results[0][1], results[0][2],
        results[1][0], results[1][1], results[1][2],
        results[2][0], results[2][1], results[2][2],
        results[0][3], results[0][4],
    )


def _sync_layer_picker_action(
    layer_index: int,
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
    descriptor, root_name, root_octave, _status = apply_keybed_tag_picker_action(
        current_descriptor,
        instrument_1,
        instrument_2,
        timbre_tags,
        wetdry_label,
        _prompt_seed_for_layer(seed_str, layer_index),
        bool(root_locked) or layer_index != 0,
        current_root_note,
        current_root_octave,
    )
    return descriptor, root_name, root_octave


def _write_layer_run(
    layer_dir: str,
    *,
    descriptor: str,
    root_note: str,
    range_label: str,
    all_notes: Sequence[str],
    wetdry: str,
    layer_seed: int,
    model_name: str,
    sample_rate: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    sigma_min: float,
    sigma_max: float,
    cfg_rescale: float,
    chunks: Sequence[Sequence[str]],
    model,
    run_kind: str,
) -> Tuple[List[torch.Tensor], str]:
    chunk_dir = os.path.join(layer_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_audio: List[torch.Tensor] = []
    manifest_rows: List[Dict[str, object]] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        _tri_cli_status(
            f"{os.path.basename(layer_dir)}: chunk {chunk_index}/{len(chunks)} "
            f"({chunk[0]}-{chunk[-1]})."
        )
        prompt = keybed_prompts.build_keybed_note_sequence_prompt(
            descriptor,
            list(chunk),
            wetdry=wetdry,
            seed=int(layer_seed),
            chromatic_chunk=True,
        )
        audio = _generate_keybed_sequence_tensor(
            model=model,
            sample_rate=int(sample_rate),
            prompt=prompt,
            seed=int(layer_seed),
            note_count=len(chunk),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
        )
        chunk_path = os.path.join(
            chunk_dir,
            f"chunk_{chunk_index:02d}_{_note_filename(chunk[0])}_to_{_note_filename(chunk[-1])}.wav",
        )
        torchaudio.save(chunk_path, _tensor_to_i16_cpu(audio), int(sample_rate))
        chunk_audio.append(audio.cpu())
        manifest_rows.append(
            {
                "type": run_kind,
                "role": "chunk",
                "name": f"chunk_{chunk_index:02d}",
                "descriptor": str(descriptor),
                "root_note": root_note,
                "number_of_octaves": range_label,
                "notes": list(chunk),
                "note_slices": _note_slices_for_sequence(list(chunk)),
                "wetdry": wetdry,
                "seed": int(layer_seed),
                "base_seed": int(layer_seed),
                "seed_mode": "same_seed_all_layer_chunks",
                "prompt": prompt,
                "audio_path": chunk_path,
                "sample_rate": int(sample_rate),
                "note_seconds": 3.0,
                "gap_seconds": 0.25,
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "sampler_type": str(sampler_type),
                "sigma_min": float(sigma_min),
                "sigma_max": float(sigma_max),
                "cfg_rescale": float(cfg_rescale),
                "model": model_name,
                "export_hint": "slice_by_note_slices",
            }
        )

    _metadata_path, jsonl_path = _write_keybed_run_metadata(
        layer_dir,
        descriptor=str(descriptor),
        root_note=root_note,
        number_of_octaves=range_label,
        all_notes=list(all_notes),
        wetdry=wetdry,
        resolved_seed=int(layer_seed),
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
    return chunk_audio, jsonl_path


def generate_tri_layer_preview_action(
    main_descriptor,
    support_1_descriptor,
    support_2_descriptor,
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
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
    get_runtime: Callable[[], Dict],
):
    descriptors = [
        str(main_descriptor or "").strip(),
        str(support_1_descriptor or "").strip(),
        str(support_2_descriptor or "").strip(),
    ]
    if any(not descriptor for descriptor in descriptors):
        raise gr.Error("Add a descriptor for Main, Support 1, and Support 2 before previewing.")

    runtime = _require_keybed_runtime(get_runtime)
    model = _runtime_model(runtime)
    sample_rate = _runtime_sample_rate(runtime)
    model_name = _runtime_model_name(runtime)
    output_root = _runtime_output_directory(runtime)

    base_seed = int(_resolve_seed(seed_str))
    layer_seeds = _layer_generation_seeds(base_seed)
    wetdry = _wetdry_from_label(wetdry_label)
    root_note, adjusted = _clamp_preview_root(root_note_name, root_octave, number_of_octaves)
    root_name_out, root_octave_out = _split_note(root_note)
    note_count = 6 if str(number_of_octaves) == "0.5" else 12 if str(number_of_octaves) == "1" else 24
    all_notes = _chromatic_notes_from_root(root_note, note_count)
    chunks = _chunk_notes(all_notes, KEYBED_CHUNK_SIZE)

    parent_dir = _make_tri_parent_dir(
        output_root,
        descriptors,
        seed=base_seed,
        range_hint=f"preview_{root_note}_{number_of_octaves}",
        output_subdir=TRI_PREVIEW_OUTPUT_SUBDIR,
    )
    _tri_cli_status(
        f"Preview {os.path.basename(parent_dir)}: {len(all_notes)} notes x 3 layers, root {root_note}."
    )

    mixed_layer_previews: List[torch.Tensor] = []
    for layer_index in range(3):
        layer_dir = os.path.join(parent_dir, TRI_LAYER_DIR_NAMES[layer_index])
        os.makedirs(layer_dir, exist_ok=True)
        chunk_audio, _manifest_path = _write_layer_run(
            layer_dir,
            descriptor=descriptors[layer_index],
            root_note=root_note,
            range_label=f"Preview {number_of_octaves} octave(s)",
            all_notes=all_notes,
            wetdry=wetdry,
            layer_seed=layer_seeds[layer_index],
            model_name=model_name,
            sample_rate=sample_rate,
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
            chunks=chunks,
            model=model,
            run_kind="tri_layer_preview",
        )
        layer_preview = _concat_audio(chunk_audio, sample_rate)
        torchaudio.save(
            _layer_preview_path(parent_dir, layer_index),
            _tensor_to_i16_cpu(layer_preview),
            int(sample_rate),
        )
        mixed_layer_previews.append(layer_preview)

    preview_path, _combined_preview = _write_mixed_preview_from_tensors(
        parent_dir,
        mixed_layer_previews,
        sample_rate=int(sample_rate),
        main_volume_percent=main_volume_percent,
        support_1_volume_percent=support_1_volume_percent,
        support_2_volume_percent=support_2_volume_percent,
        master_volume_percent=master_volume_percent,
    )

    preview_id = os.path.basename(parent_dir).rsplit("_", 1)[-1].upper()
    status = (
        f"**Layered preview ready** — `{all_notes[0]}` to `{all_notes[-1]}` "
        f"(`{len(all_notes)}` notes x 3 layers) · ID `{preview_id}` · Base seed `{base_seed}`  \n"
        f"Mix: Main `{float(main_volume_percent):.0f}%` · "
        f"Support 1 `{float(support_1_volume_percent):.0f}%` · "
        f"Support 2 `{float(support_2_volume_percent):.0f}%` · "
        f"Master `{float(master_volume_percent):.0f}%`"
    )
    if adjusted:
        status += f"  \nPreview root was adjusted to `{root_note}` to stay inside the trained range."

    _tri_cli_status(f"Layered preview complete - ID {preview_id}.")
    return (
        gr.update(
            value=preview_path,
            label=f"Layered Preview — {all_notes[0]} to {all_notes[-1]}",
            visible=True,
        ),
        gr.update(value=root_name_out),
        gr.update(value=root_octave_out),
        status,
        parent_dir,
        int(base_seed),
        _descriptor_bundle(descriptors),
    )


def generate_and_export_tri_layer_action(
    main_descriptor,
    support_1_descriptor,
    support_2_descriptor,
    full_sampler_range_label,
    wetdry_label,
    seed_str,
    latest_preview_seed,
    latest_preview_descriptor_bundle,
    steps,
    cfg_scale,
    sampler_type,
    sigma_min,
    sigma_max,
    cfg_rescale,
    main_volume_percent,
    support_1_volume_percent,
    support_2_volume_percent,
    master_volume_percent,
    instrument_name,
    get_runtime: Callable[[], Dict],
):
    descriptors = [
        str(main_descriptor or "").strip(),
        str(support_1_descriptor or "").strip(),
        str(support_2_descriptor or "").strip(),
    ]
    instrument_name = _normalize_instrument_name(instrument_name)
    if any(not descriptor for descriptor in descriptors):
        raise gr.Error("Add a descriptor for Main, Support 1, and Support 2 before exporting.")

    runtime = _require_keybed_runtime(get_runtime)
    model = _runtime_model(runtime)
    sample_rate = _runtime_sample_rate(runtime)
    model_name = _runtime_model_name(runtime)
    output_root = _runtime_output_directory(runtime)

    current_bundle = _descriptor_bundle(descriptors)
    try:
        preview_seed = int(latest_preview_seed)
    except Exception:
        preview_seed = None
    if preview_seed is not None and str(latest_preview_descriptor_bundle or "") == current_bundle:
        base_seed = int(preview_seed)
        seed_source = "captured layered preview"
    else:
        base_seed = int(_resolve_seed(seed_str))
        seed_source = "seed control"
    layer_seeds = _layer_generation_seeds(base_seed)
    wetdry = _wetdry_from_label(wetdry_label)

    range_label, start_note, end_note, all_notes, chunks = _resolve_full_sampler_range(
        full_sampler_range_label
    )
    parent_dir = _make_tri_parent_dir(
        output_root,
        descriptors,
        seed=base_seed,
        range_hint=f"{start_note}_to_{end_note}",
        instrument_name=instrument_name,
    )
    run_id = os.path.basename(parent_dir)
    _tri_cli_status(
        f"Full run {run_id}: {len(chunks)} chunks x 3 layers for {start_note}-{end_note}."
    )

    layer_dirs: List[str] = []
    for layer_index in range(3):
        layer_dir = os.path.join(parent_dir, TRI_LAYER_DIR_NAMES[layer_index])
        os.makedirs(layer_dir, exist_ok=True)
        _write_layer_run(
            layer_dir,
            descriptor=descriptors[layer_index],
            root_note=start_note,
            range_label=range_label,
            all_notes=all_notes,
            wetdry=wetdry,
            layer_seed=layer_seeds[layer_index],
            model_name=model_name,
            sample_rate=sample_rate,
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sampler_type=str(sampler_type),
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            cfg_rescale=float(cfg_rescale),
            chunks=chunks,
            model=model,
            run_kind="tri_layer_full_sampler",
        )
        layer_dirs.append(layer_dir)

    _tri_cli_status(f"Full run {run_id}: source generation complete; starting tri-layer export.")
    layer_volumes, master_volume = _mix_settings_from_ui(
        main_volume_percent,
        support_1_volume_percent,
        support_2_volume_percent,
        master_volume_percent,
    )
    export_result = export_tri_layered_keybed_run(
        layer_dirs,
        preset_id=run_id,
        layer_titles=[
            _descriptor_title(descriptors[0], "Main"),
            _descriptor_title(descriptors[1], "Support 1"),
            _descriptor_title(descriptors[2], "Support 2"),
        ],
        layer_volumes=layer_volumes,
        master_volume=master_volume,
        trim_tail=True,
        instrument_name=instrument_name,
    )

    export_dir = str(export_result["export_dir"])
    instrument_id = str(export_result.get("instrument_id") or run_id)
    status = (
        f"**Tri-layer DecentSampler instrument exported** — "
        f"`{export_result.get('display_name') or run_id}` · ID `{instrument_id}`  \n"
        f"Range: `{start_note}` to `{end_note}` (`{len(all_notes)}` notes x 3 layers) · "
        f"Seed: `{base_seed}` ({seed_source})  \n"
        f"Saved to: `{export_dir}`"
    )
    _tri_cli_status(f"Tri-layer Keybed Exported - ID {instrument_id} ({run_id}).")

    # Export state is intentionally kept separate from preview state. The mixer
    # must continue pointing at the last preview run after a full instrument
    # export, because only preview runs contain layer_preview.wav files.
    return (
        status,
        gr.update(
            value=[str(export_result["dspreset_path"])],
            label="Exported Tri-Layer DecentSampler Instrument",
            visible=True,
        ),
        export_dir,
        gr.update(visible=True, interactive=True),
    )


def _busy_updates(active: str):
    labels = {
        "preview": "Generating 3-Layer Preview...",
        "full": "Building 3-Layer Keybed...",
    }
    return tuple(
        gr.update(
            interactive=False,
            value=labels[key] if key == active else TRI_BUTTON_LABELS[key],
        )
        for key in ("preview", "full")
    )


def _ready_updates():
    return tuple(
        gr.update(interactive=True, value=TRI_BUTTON_LABELS[key])
        for key in ("preview", "full")
    )


def begin_tri_preview_action():
    _tri_cli_status("Building layered preview...")
    return (*_busy_updates("preview"), "**Building Layered Preview...** Three keybeds will be generated.")


def begin_tri_full_action():
    _tri_cli_status("Building full tri-layer keybed...")
    return (
        *_busy_updates("full"),
        "**Building Tri-Layer Keybed...** Generating three complete source keybeds before export.",
    )


def _cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def safe_generate_tri_preview_action(*args, get_runtime: Callable[[], Dict]):
    # Previous state: parent dir, preview seed, descriptor bundle.
    core_args = args[:-3]
    previous_parent, previous_seed, previous_bundle = args[-3:]
    try:
        result = generate_tri_layer_preview_action(*core_args, get_runtime=get_runtime)
        return (*result, *_ready_updates())
    except Exception as exc:
        traceback.print_exc()
        _tri_cli_status(f"Layered preview failed: {type(exc).__name__}: {exc}")
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"**Layered preview failed:** `{type(exc).__name__}: {exc}`  \nControls were reset.",
            previous_parent,
            previous_seed,
            previous_bundle,
            *_ready_updates(),
        )
    finally:
        _cleanup()


def safe_generate_tri_full_action(*args, get_runtime: Callable[[], Dict]):
    # Only export-directory state belongs to the full-generation path. Preview
    # folder/seed/descriptor state is deliberately untouched so the existing
    # audition remains remixable after export.
    core_args = args[:-1]
    previous_export_dir = args[-1]
    try:
        result = generate_and_export_tri_layer_action(*core_args, get_runtime=get_runtime)
        return (*result, *_ready_updates())
    except Exception as exc:
        traceback.print_exc()
        _tri_cli_status(f"Tri-layer export failed: {type(exc).__name__}: {exc}")
        return (
            f"**Tri-layer export failed:** `{type(exc).__name__}: {exc}`  \nControls were reset; existing preview/export state was preserved.",
            gr.update(),
            previous_export_dir,
            gr.update(),
            *_ready_updates(),
        )
    finally:
        _cleanup()


def build_layered_keybed_tab(
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
    """Build the cohesive three-layer Keybed workflow using the shared runtime."""
    can_load_models = all(
        [
            config is not None,
            get_models_and_configs is not None,
            get_config_files is not None,
            update_config_dropdown is not None,
            load_model_action is not None,
        ]
    )
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

    def _load_model(selected_ckpt, selected_config, int4_requested):
        result = load_model_action(selected_ckpt, selected_config, ckpt_files, int4_requested)
        details = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else ""
        return _current_runtime_summary(details)

    default_main_percent = _default_gain_percent(TRI_LAYER_DEFAULT_VOLUMES[0])
    default_support_1_percent = _default_gain_percent(TRI_LAYER_DEFAULT_VOLUMES[1])
    default_support_2_percent = _default_gain_percent(TRI_LAYER_DEFAULT_VOLUMES[2])
    default_master_percent = _default_gain_percent(TRI_LAYER_MASTER_VOLUME)

    with gr.Column(elem_id="layered_keybed_tab_root"):
        runtime_md = gr.Markdown(
            _current_runtime_summary(),
            elem_id="layered_keybed_runtime_summary",
        )

        with gr.Accordion(
            "Model & Runtime",
            open=False,
            elem_id="layered_keybed_model_runtime_accordion",
        ):
            if can_load_models:
                with gr.Row(elem_id="layered_keybed_model_load_row"):
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
                fn=lambda value: update_config_dropdown(value, ckpt_files),
                inputs=model_dropdown,
                outputs=config_dropdown,
            )
            load_button.click(
                fn=_load_model,
                inputs=[model_dropdown, config_dropdown, int4_checkbox],
                outputs=[runtime_md],
            )
        else:
            refresh_runtime_button.click(
                fn=lambda: _current_runtime_summary(),
                inputs=[],
                outputs=[runtime_md],
            )

        gr.HTML(
            """
            <style>
                #layered_keybed_top_prompt_row {
                    align-items: stretch !important;
                }

                #layered_keybed_prompt_left_col,
                #layered_keybed_prompt_action_col {
                    display: flex !important;
                    flex-direction: column !important;
                    height: 100% !important;
                }

                #layered_keybed_prompt_action_col {
                    justify-content: flex-start !important;
                }

                #layered_keybed_section_preview,
                #layered_keybed_section_export,
                #layered_keybed_preview_note,
                #layered_keybed_export_note,
                #layered_keybed_mix_note {
                    width: 100% !important;
                    text-align: center !important;
                }

                #layered_keybed_section_preview > div,
                #layered_keybed_section_export > div,
                #layered_keybed_preview_note > div,
                #layered_keybed_export_note > div,
                #layered_keybed_mix_note > div {
                    width: 100% !important;
                    text-align: center !important;
                }

                .layered-keybed-section-title {
                    margin: 0.9rem 0 0.45rem;
                    font-size: 1.05rem;
                    font-weight: 650;
                    text-align: center;
                }

                .layered-keybed-section-note {
                    margin: 0 0 0.75rem;
                    text-align: center;
                    opacity: 0.82;
                }
            </style>
            """
        )

        # CREATE -----------------------------------------------------------
        # Three persistent descriptors make the layered relationship obvious.
        # Their detailed builders are separate collapsed strips below, so opening
        # a builder never distorts the top action/toggle column.
        with gr.Row(equal_height=True, elem_id="layered_keybed_top_prompt_row"):
            with gr.Column(scale=7, elem_id="layered_keybed_prompt_left_col"):
                descriptors = [
                    gr.Textbox(
                        label="Main Instrument",
                        placeholder="Grand Piano, Warm, Soft",
                        lines=3,
                        elem_id="layered_keybed_main_descriptor",
                    ),
                    gr.Textbox(
                        label="Support Instrument 1",
                        placeholder="Music Box, Sparkly, Metallic",
                        lines=3,
                        elem_id="layered_keybed_support_1_descriptor",
                    ),
                    gr.Textbox(
                        label="Support Instrument 2",
                        placeholder="Warm Pad, Airy, Wide",
                        lines=3,
                        elem_id="layered_keybed_support_2_descriptor",
                    ),
                ]

            with gr.Column(scale=3, elem_id="layered_keybed_prompt_action_col"):
                generate_preview_button = gr.Button(
                    TRI_BUTTON_LABELS["preview"],
                    variant="primary",
                    elem_id="layered_keybed_generate_preview_button",
                )
                randomize_all_button = gr.Button(
                    "Randomize All Three",
                    variant="secondary",
                )
                with gr.Group(elem_id="layered_keybed_prompt_toggle_group"):
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

        instrument_1_pickers = []
        instrument_2_pickers = []
        timbre_pickers = []
        random_buttons = []
        clear_buttons = []

        for layer_index, role in enumerate(TRI_LAYER_ROLES):
            with gr.Accordion(
                f"{role} Prompt Builder",
                open=False,
                elem_id=f"layered_keybed_builder_{layer_index + 1}",
            ):
                gr.Markdown(
                    "Single keeps Instrument 2 visible at None. Hybrid enables the existing "
                    "two-slot hierarchy/mix behavior. Picker changes update this layer automatically."
                )
                with gr.Row():
                    instrument_1 = _searchable_dropdown(
                        KEYBED_TAG_PICKER_INSTRUMENTS,
                        label="Instrument 1",
                        value=None,
                    )
                    instrument_2 = _searchable_dropdown(
                        KEYBED_TAG_PICKER_OPTIONAL_INSTRUMENTS,
                        label="Instrument 2 (Optional)",
                        value=KEYBED_NO_SECOND_INSTRUMENT,
                    )
                timbres = _searchable_dropdown(
                    KEYBED_TAG_PICKER_TIMBRES,
                    label="Timbre Tags",
                    value=[],
                    multiselect=True,
                )
                with gr.Row():
                    random_button = gr.Button(
                        f"Randomize {role}",
                        variant="secondary",
                    )
                    clear_button = gr.Button(
                        "Clear Selected Tags",
                        variant="secondary",
                    )

                instrument_1_pickers.append(instrument_1)
                instrument_2_pickers.append(instrument_2)
                timbre_pickers.append(timbres)
                random_buttons.append(random_button)
                clear_buttons.append(clear_button)

        # Shared preview range and seed remain normal workflow controls.
        with gr.Accordion(
            "Preview Setup",
            open=False,
            elem_id="layered_keybed_preview_setup",
        ):
            with gr.Row():
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
                    label="Base Seed (-1 for random)",
                    value="-1",
                    scale=1,
                )
            gr.Markdown(
                "All three passes use the same preview notes. Only the Main Instrument can "
                "automatically move the shared preview root."
            )

            # Match the single-Keybed layout: advanced inference controls live
            # inside Preview Setup, directly above the preview output.
            with gr.Accordion(
                "Advanced Generation Settings",
                open=False,
                elem_id="layered_keybed_generation_settings",
            ):
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
                        maximum=1.0,
                        step=0.01,
                        value=KEYBED_DEFAULT_CFG_RESCALE,
                        label="CFG Rescale",
                    )
    
        # PREVIEW + MIX ----------------------------------------------------
        gr.HTML(
            '<div class="layered-keybed-section-title">Preview</div>',
            elem_id="layered_keybed_section_preview",
        )
        gr.HTML(
            '<div class="layered-keybed-section-note">The three passes are still generated sequentially. The mixer below only rebuilds the audition from saved audio.</div>',
            elem_id="layered_keybed_preview_note",
        )
        preview_status_output = gr.Markdown("")
        preview_audio = gr.Audio(
            label="Layered Preview",
            interactive=False,
            visible=False,
        )

        gr.HTML(
            '<div class="layered-keybed-section-note"><strong>Layer Mix</strong></div>',
            elem_id="layered_keybed_mix_note",
        )
        with gr.Row(elem_id="layered_keybed_mix_controls"):
            with gr.Column(scale=1, min_width=170):
                with gr.Row():
                    main_solo_checkbox = gr.Checkbox(label="Solo", value=False)
                    main_mute_checkbox = gr.Checkbox(label="Mute", value=False)
                main_volume_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=default_main_percent,
                    label="Main (%)",
                )
            with gr.Column(scale=1, min_width=170):
                with gr.Row():
                    support_1_solo_checkbox = gr.Checkbox(label="Solo", value=False)
                    support_1_mute_checkbox = gr.Checkbox(label="Mute", value=False)
                support_1_volume_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=default_support_1_percent,
                    label="Support 1 (%)",
                )
            with gr.Column(scale=1, min_width=170):
                with gr.Row():
                    support_2_solo_checkbox = gr.Checkbox(label="Solo", value=False)
                    support_2_mute_checkbox = gr.Checkbox(label="Mute", value=False)
                support_2_volume_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=default_support_2_percent,
                    label="Support 2 (%)",
                )
            with gr.Column(scale=1, min_width=170):
                with gr.Row():
                    master_solo_checkbox = gr.Checkbox(label="Solo", value=False)
                    master_mute_checkbox = gr.Checkbox(label="Mute", value=False)
                master_volume_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=default_master_percent,
                    label="Master (%)",
                )
        mix_status_output = gr.Markdown(
            "Mix updates automatically. Solo remembers the previous mix; Mute sets that channel to 0%. "
            "Master Solo restores the complete saved mix through the master bus. No diffusion is rerun."
        )

        # Mixer memory is session-local. Zeroing a slider through Solo/Mute does
        # not overwrite the channel's latest non-zero level.
        saved_main_volume_state = gr.State(value=default_main_percent)
        saved_support_1_volume_state = gr.State(value=default_support_1_percent)
        saved_support_2_volume_state = gr.State(value=default_support_2_percent)
        saved_master_volume_state = gr.State(value=default_master_percent)
        active_solo_state = gr.State(value=-1)
        pre_solo_mutes_state = gr.State(value=[False, False, False, False])

        # Preview-only folder state. Full export must never overwrite this.
        latest_parent_state = gr.State(value="")
        latest_preview_seed_state = gr.State(value="")
        latest_preview_bundle_state = gr.State(value="")
        latest_export_dir_state = gr.State(value="")

        # EXPORT -----------------------------------------------------------
        gr.HTML(
            '<div class="layered-keybed-section-title">Export Instrument</div>',
            elem_id="layered_keybed_section_export",
        )
        gr.HTML(
            '<div class="layered-keybed-section-note">Export uses the selected fixed range, the current layer mix, and the latest matching layered-preview seed when available.</div>',
            elem_id="layered_keybed_export_note",
        )
        instrument_name_textbox = _instrument_name_textbox(
            elem_id="layered_keybed_instrument_name",
        )
        with gr.Row(equal_height=True, elem_id="layered_keybed_export_strip"):
            full_sampler_range_dropdown = gr.Dropdown(
                KEYBED_FULL_SAMPLER_RANGE_CHOICES,
                value=KEYBED_FULL_SAMPLER_DEFAULT_RANGE,
                label="Sampler Range",
                scale=5,
            )
            generate_full_button = gr.Button(
                "Generate & Export Layered Keybed",
                variant="primary",
                scale=2,
                elem_id="layered_keybed_generate_export_button",
            )

        export_status_output = gr.Markdown("")
        with gr.Row(elem_id="layered_keybed_export_result_row"):
            exported_file = gr.File(
                label="Exported Tri-Layer DecentSampler Instrument",
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

        # Prompt and picker callbacks -------------------------------------
        for layer_index in range(3):
            random_buttons[layer_index].click(
                fn=lambda *args, idx=layer_index: _layer_random_action(idx, *args),
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
                    descriptors[layer_index],
                    root_note_dropdown,
                    root_octave_dropdown,
                    instrument_1_pickers[layer_index],
                    instrument_2_pickers[layer_index],
                    timbre_pickers[layer_index],
                ],
                queue=False,
            )

            sync_inputs = [
                descriptors[layer_index],
                instrument_1_pickers[layer_index],
                instrument_2_pickers[layer_index],
                timbre_pickers[layer_index],
                wetdry_radio,
                seed_textbox,
                root_lock_checkbox,
                root_note_dropdown,
                root_octave_dropdown,
            ]
            sync_outputs = [
                descriptors[layer_index],
                root_note_dropdown,
                root_octave_dropdown,
            ]
            for picker in (
                instrument_1_pickers[layer_index],
                instrument_2_pickers[layer_index],
                timbre_pickers[layer_index],
            ):
                picker.input(
                    fn=lambda *args, idx=layer_index: _sync_layer_picker_action(idx, *args),
                    inputs=sync_inputs,
                    outputs=sync_outputs,
                    queue=False,
                    show_progress="hidden",
                )

            clear_buttons[layer_index].click(
                fn=clear_keybed_tag_picker_action,
                inputs=[
                    descriptors[layer_index],
                    root_note_dropdown,
                    root_octave_dropdown,
                ],
                outputs=[
                    instrument_1_pickers[layer_index],
                    instrument_2_pickers[layer_index],
                    timbre_pickers[layer_index],
                    descriptors[layer_index],
                    root_note_dropdown,
                    root_octave_dropdown,
                ],
                queue=False,
                show_progress="hidden",
            )

        instrument_mode_radio.change(
            fn=set_all_tri_layer_instrument_modes_action,
            inputs=[
                instrument_mode_radio,
                descriptors[0], instrument_1_pickers[0], instrument_2_pickers[0], timbre_pickers[0],
                descriptors[1], instrument_1_pickers[1], instrument_2_pickers[1], timbre_pickers[1],
                descriptors[2], instrument_1_pickers[2], instrument_2_pickers[2], timbre_pickers[2],
                wetdry_radio,
                seed_textbox,
                root_lock_checkbox,
                root_note_dropdown,
                root_octave_dropdown,
            ],
            outputs=[
                descriptors[0], instrument_1_pickers[0], instrument_2_pickers[0],
                descriptors[1], instrument_1_pickers[1], instrument_2_pickers[1],
                descriptors[2], instrument_1_pickers[2], instrument_2_pickers[2],
                root_note_dropdown, root_octave_dropdown,
            ],
            queue=False,
            show_progress="hidden",
        )

        randomize_all_button.click(
            fn=randomize_all_tri_layers_action,
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
                descriptors[0], instrument_1_pickers[0], instrument_2_pickers[0], timbre_pickers[0],
                descriptors[1], instrument_1_pickers[1], instrument_2_pickers[1], timbre_pickers[1],
                descriptors[2], instrument_1_pickers[2], instrument_2_pickers[2], timbre_pickers[2],
                root_note_dropdown, root_octave_dropdown,
            ],
            queue=False,
        )

        # Existing sequential preview generation. The four mix values are used
        # only after the same three layer passes have completed.
        preview_event = generate_preview_button.click(
            fn=begin_tri_preview_action,
            inputs=[],
            outputs=[generate_preview_button, generate_full_button, preview_status_output],
        )
        preview_event.then(
            fn=lambda *args: safe_generate_tri_preview_action(*args, get_runtime=get_runtime),
            inputs=[
                descriptors[0],
                descriptors[1],
                descriptors[2],
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
                main_volume_slider,
                support_1_volume_slider,
                support_2_volume_slider,
                master_volume_slider,
                latest_parent_state,
                latest_preview_seed_state,
                latest_preview_bundle_state,
            ],
            outputs=[
                preview_audio,
                root_note_dropdown,
                root_octave_dropdown,
                preview_status_output,
                latest_parent_state,
                latest_preview_seed_state,
                latest_preview_bundle_state,
                generate_preview_button,
                generate_full_button,
            ],
        )

        # Rebuild the saved audition when the user finishes moving any mix
        # control. Slider.release avoids running once per drag tick. The shared
        # concurrency ID serializes rapid adjustments, and always_last is used
        # when the installed Gradio version supports it.
        mix_volume_components = [
            main_volume_slider,
            support_1_volume_slider,
            support_2_volume_slider,
            master_volume_slider,
        ]
        mix_solo_components = [
            main_solo_checkbox,
            support_1_solo_checkbox,
            support_2_solo_checkbox,
            master_solo_checkbox,
        ]
        mix_mute_components = [
            main_mute_checkbox,
            support_1_mute_checkbox,
            support_2_mute_checkbox,
            master_mute_checkbox,
        ]
        mix_saved_volume_states = [
            saved_main_volume_state,
            saved_support_1_volume_state,
            saved_support_2_volume_state,
            saved_master_volume_state,
        ]

        slider_mix_inputs = [
            latest_parent_state,
            *mix_volume_components,
            *mix_saved_volume_states,
        ]
        slider_mix_outputs = [
            preview_audio,
            mix_status_output,
            *mix_saved_volume_states,
        ]

        def _bind_mix_slider_event(slider):
            event_method = getattr(slider, "release", None) or slider.change
            kwargs = {
                "fn": update_tri_preview_mix_with_memory_action,
                "inputs": slider_mix_inputs,
                "outputs": slider_mix_outputs,
                "queue": True,
                "show_progress": "hidden",
            }
            try:
                parameters = inspect.signature(event_method).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "trigger_mode" in parameters:
                kwargs["trigger_mode"] = "always_last"
            if "concurrency_limit" in parameters:
                kwargs["concurrency_limit"] = 1
            if "concurrency_id" in parameters:
                kwargs["concurrency_id"] = "layered_keybed_preview_mix"
            event_method(**kwargs)

        for mix_slider in mix_volume_components:
            _bind_mix_slider_event(mix_slider)

        mixer_toggle_shared_inputs = [
            latest_parent_state,
            *mix_volume_components,
            *mix_solo_components,
            *mix_mute_components,
            *mix_saved_volume_states,
            active_solo_state,
            pre_solo_mutes_state,
        ]
        mixer_toggle_outputs = [
            *mix_volume_components,
            *mix_solo_components,
            *mix_mute_components,
            *mix_saved_volume_states,
            active_solo_state,
            pre_solo_mutes_state,
            preview_audio,
            mix_status_output,
        ]

        def _bind_mix_checkbox_event(component, fn, channel_index):
            event_method = component.input
            kwargs = {
                "fn": lambda requested, *args, channel_index=channel_index, fn=fn: fn(
                    channel_index,
                    requested,
                    *args,
                ),
                "inputs": [component, *mixer_toggle_shared_inputs],
                "outputs": mixer_toggle_outputs,
                "queue": True,
                "show_progress": "hidden",
            }
            try:
                parameters = inspect.signature(event_method).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "trigger_mode" in parameters:
                kwargs["trigger_mode"] = "always_last"
            if "concurrency_limit" in parameters:
                kwargs["concurrency_limit"] = 1
            if "concurrency_id" in parameters:
                kwargs["concurrency_id"] = "layered_keybed_preview_mix"
            event_method(**kwargs)

        for channel_index, solo_checkbox in enumerate(mix_solo_components):
            _bind_mix_checkbox_event(
                solo_checkbox,
                toggle_tri_mix_solo_action,
                channel_index,
            )

        for channel_index, mute_checkbox in enumerate(mix_mute_components):
            _bind_mix_checkbox_event(
                mute_checkbox,
                toggle_tri_mix_mute_action,
                channel_index,
            )

        full_event = generate_full_button.click(
            fn=begin_tri_full_action,
            inputs=[],
            outputs=[generate_preview_button, generate_full_button, export_status_output],
        )
        full_event.then(
            fn=lambda *args: safe_generate_tri_full_action(*args, get_runtime=get_runtime),
            inputs=[
                descriptors[0],
                descriptors[1],
                descriptors[2],
                full_sampler_range_dropdown,
                wetdry_radio,
                seed_textbox,
                latest_preview_seed_state,
                latest_preview_bundle_state,
                steps_slider,
                cfg_scale_slider,
                sampler_type_dropdown,
                sigma_min_slider,
                sigma_max_slider,
                cfg_rescale_slider,
                main_volume_slider,
                support_1_volume_slider,
                support_2_volume_slider,
                master_volume_slider,
                instrument_name_textbox,
                latest_export_dir_state,
            ],
            outputs=[
                export_status_output,
                exported_file,
                latest_export_dir_state,
                open_export_folder_button,
                generate_preview_button,
                generate_full_button,
            ],
        )

        open_export_folder_button.click(
            fn=open_keybed_export_folder_action,
            inputs=[latest_export_dir_state],
            outputs=[],
        )

    def refresh_layered_keybed_tab_from_runtime():
        return _current_runtime_summary()

    return {
        "refresh_fn": refresh_layered_keybed_tab_from_runtime,
        "refresh_inputs": [],
        "refresh_outputs": [runtime_md],
    }


__all__ = ["build_layered_keybed_tab"]
