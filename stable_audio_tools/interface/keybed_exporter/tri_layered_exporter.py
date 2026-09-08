from __future__ import annotations

import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio

from .exporter import (
    DEFAULT_FADE_MS,
    DEFAULT_FRAME_MS,
    DEFAULT_MIN_KEEP_MS,
    DEFAULT_MIN_TRIM_MS,
    DEFAULT_TAIL_PAD_MS,
    DEFAULT_TERMINAL_FADE_MS,
    DEFAULT_THRESHOLD_DB,
    DS_FILTER_CONTROL_MAX,
    DS_FILTER_CONTROL_MIN,
    DS_FILTER_Q_CONTROL_MAX,
    DS_FILTER_Q_CONTROL_MIN,
    DS_FILTER_Q_DEFAULT,
    DS_FILTER_RESONANCE_MAX,
    DS_FILTER_RESONANCE_MIN,
    DS_UI_BG_COLOR,
    DS_UI_MUTED_TEXT_COLOR,
    DS_UI_TEXT_COLOR,
    DS_UI_TRACK_BACKGROUND,
    DS_UI_TRACK_FOREGROUND,
    ExportedSample,
    _apply_fade_out,
    _centered_section_label,
    _catalog_metadata_from_rows,
    _find_manifest,
    _is_exportable_chunk_row,
    _knob,
    _label,
    _log_translation_table,
    _normalize_instrument_name,
    _preset_id_with_instrument_name,
    _parse_note_slices,
    _parse_notes,
    _read_jsonl,
    _relpath,
    _resolve_audio_path,
    _safe_slug,
    _save_wav,
    _to_float_audio,
    midi_to_note,
    note_filename,
    note_to_midi,
    trim_tail_conservative,
)

TRI_LAYER_ROLES = ("Main", "Support 1", "Support 2")
TRI_LAYER_DEFAULT_VOLUMES = (0.90, 0.60, 0.35)
TRI_LAYER_MASTER_VOLUME = 0.55
TRI_LAYER_DEFAULT_ATTACK = 0.005
TRI_LAYER_DEFAULT_DECAY = 0.0
TRI_LAYER_DEFAULT_SUSTAIN = 1.0
TRI_LAYER_DEFAULT_RELEASE = 0.25

# Volume policy:
# - mapped WAVs and base DecentSampler sample/group/instrument volumes stay at unity;
# - the four UI knobs are the only attenuation stages;
# - at knob value 1.0, one layer equals its mapped WAV level (before layer summing).

# Optional tri-layer artwork. Place Tri_Background.png (or JPG/JPEG/WEBP) in
# keybed_exporter/decent_sampler_asset beside the existing single-keybed artwork.
TRI_BACKGROUND_ASSET_DIR = Path(__file__).resolve().parent / "decent_sampler_asset"
TRI_BACKGROUND_BASENAME = "Tri_Background"
TRI_UI_WIDTH = 980
TRI_UI_HEIGHT = 600

# Native DecentSampler glide/portamento. This is intentionally one shared
# instrument-level control for the complete layered instrument: all three
# sample groups follow the same pitch transition. "On" maps to legato mode;
# the always-glide mode is deliberately not exposed.
TRI_GLIDE_DEFAULT_MODE = "off"
TRI_GLIDE_DEFAULT_TIME = 0.15
TRI_GLIDE_MAX_TIME = 2.0


def _status(message: str, callback: Optional[Callable[[str], None]] = None) -> None:
    text = f"[Tri-Layer Exporter] {message}"
    print(text, flush=True)
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


def _display_name_and_id(preset_id: str) -> Tuple[str, str]:
    raw = str(preset_id or "").strip()
    match = re.search(r"(?:^|[_-])([0-9a-fA-F]{8})$", raw)
    serial = match.group(1).upper() if match else ""
    name_part = raw[: match.start()].rstrip("_- ") if match else raw
    display = re.sub(r"[_-]+", " ", name_part).strip()
    display = re.sub(r"\s+", " ", display).title() or "Foundation-1 Layered Keybed"
    return display, serial


def _default_export_root(layer_run_folder: Path) -> Path:
    """Resolve <generations>/Sampler_Exports without hard-coded paths."""
    folder = layer_run_folder.resolve()
    for parent in (folder, *folder.parents):
        if parent.name == "Layered_Keybed_Source":
            return parent.parent / "Sampler_Exports"
    # Fallback for custom layouts.
    return layer_run_folder.parent / "Sampler_Exports"


def _preset_id_from_parent(layer_run_folders: Sequence[Path]) -> str:
    first = layer_run_folders[0]
    parent_name = first.parent.name.strip()
    if parent_name and parent_name not in {"Layered_Keybed_Source", "layer_1_main"}:
        return _safe_slug(parent_name, fallback="tri_layer_keybed")
    return _safe_slug(first.name, fallback="tri_layer_keybed")


def _slice_layer_run(
    run_folder: Path,
    *,
    layer_index: int,
    raw_dir: Optional[Path],
    sample_dir: Path,
    trim_tail: bool,
    threshold_db: float,
    frame_ms: float,
    tail_pad_ms: float,
    fade_ms: float,
    terminal_fade_ms: float,
    min_keep_ms: float,
    min_trim_ms: float,
    progress_callback: Optional[Callable[[str], None]],
) -> Tuple[List[ExportedSample], List[Dict[str, Any]]]:
    manifest_path = _find_manifest(run_folder)
    rows = _read_jsonl(manifest_path)
    chunk_rows = [(idx, row) for idx, row in enumerate(rows) if _is_exportable_chunk_row(row)]
    if not chunk_rows:
        raise ValueError(f"No exportable chunk rows found in {manifest_path}")

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    exported: List[ExportedSample] = []
    seen_midis: Dict[int, int] = {}

    for chunk_number, (row_index, row) in enumerate(chunk_rows, start=1):
        source_path = _resolve_audio_path(run_folder, str(row.get("audio_path") or ""))
        _status(
            f"Layer {layer_index + 1}: slicing chunk {chunk_number}/{len(chunk_rows)} ({source_path.name}).",
            progress_callback,
        )
        audio, sample_rate = torchaudio.load(str(source_path))
        audio = _to_float_audio(audio)
        sample_rate = int(sample_rate)

        notes = _parse_notes(row)
        slices = _parse_note_slices(row, notes)
        for slice_index, note_slice in enumerate(slices):
            note = str(
                note_slice.get("note")
                or (notes[slice_index] if slice_index < len(notes) else "")
            ).strip()
            if not note:
                continue
            midi = note_to_midi(note)
            if midi in seen_midis:
                seen_midis[midi] += 1
                continue
            seen_midis[midi] = 1

            scheduled_start = float(note_slice.get("start_sec") or 0.0)
            scheduled_end = float(
                note_slice.get("end_sec")
                or (scheduled_start + float(row.get("note_seconds") or 3.0))
            )
            start_sample = max(0, int(round(scheduled_start * sample_rate)))
            end_sample = min(
                int(audio.shape[-1]),
                max(start_sample + 1, int(round(scheduled_end * sample_rate))),
            )
            sliced = audio[:, start_sample:end_sample].contiguous()

            sample_name = f"{note_filename(midi_to_note(midi))}.wav"
            raw_path: Optional[Path] = None
            if raw_dir is not None:
                raw_path = raw_dir / sample_name
                _save_wav(raw_path, sliced, sample_rate)

            if trim_tail:
                mapped_audio, did_trim, method, duration_sec = trim_tail_conservative(
                    sliced,
                    sample_rate,
                    threshold_db=threshold_db,
                    frame_ms=frame_ms,
                    tail_pad_ms=tail_pad_ms,
                    fade_ms=fade_ms,
                    min_keep_ms=min_keep_ms,
                    min_trim_ms=min_trim_ms,
                )
            else:
                mapped_audio = sliced
                did_trim = False
                method = "none_disabled"
                duration_sec = sliced.shape[-1] / sample_rate

            terminal_fade_applied = False
            if float(terminal_fade_ms) > 0 and mapped_audio.shape[-1] > 1:
                mapped_audio = _apply_fade_out(mapped_audio, sample_rate, float(terminal_fade_ms))
                terminal_fade_applied = True

            mapped_path = sample_dir / sample_name
            _save_wav(mapped_path, mapped_audio, sample_rate)
            final_source_end = scheduled_start + float(duration_sec)

            exported.append(
                ExportedSample(
                    note=midi_to_note(midi),
                    midi=midi,
                    source_chunk_path=str(source_path),
                    scheduled_start_sec=round(scheduled_start, 6),
                    scheduled_end_sec=round(scheduled_end, 6),
                    final_source_start_sec=round(scheduled_start, 6),
                    final_source_end_sec=round(final_source_end, 6),
                    export_duration_sec=round(float(duration_sec), 6),
                    raw_sample_path=str(raw_path) if raw_path is not None else "",
                    sample_paths_by_format={"dspreset": str(mapped_path)},
                    tail_trimmed=bool(did_trim),
                    trim_method=method,
                    terminal_fade_applied=terminal_fade_applied,
                    terminal_fade_ms=float(terminal_fade_ms) if terminal_fade_applied else 0.0,
                    rootNote=midi,
                    pitch_keycenter=midi,
                    loNote=midi,
                    hiNote=midi,
                    source_row_name=str(row.get("name") or row.get("test_name") or ""),
                    source_row_index=int(row_index),
                )
            )

    exported.sort(key=lambda item: item.midi)
    if not exported:
        raise ValueError(f"Layer {layer_index + 1} produced no mapped samples.")
    return exported, rows


def _layer_title_from_rows(rows: Sequence[Dict[str, Any]], fallback: str) -> str:
    for row in rows:
        descriptor = str(row.get("descriptor") or "").strip()
        if descriptor:
            tokens = [token.strip() for token in descriptor.split(",") if token.strip()]
            if tokens:
                return " + ".join(tokens[:2])
    return fallback


def _group_sample_lines(samples: Sequence[ExportedSample], export_dir: Path) -> List[str]:
    lines: List[str] = []
    for sample in samples:
        path = _relpath(Path(sample.sample_paths_by_format["dspreset"]), export_dir)
        lines.append(
            "      "
            f'<sample path="{html.escape(path)}" rootNote="{sample.rootNote}" '
            f'loNote="{sample.loNote}" hiNote="{sample.hiNote}" '
            f'volume="1.0" loVel="1" hiVel="127" />'
        )
    return lines


def _group_amp_binding(group_index: int, parameter: str) -> str:
    attrs = [
        'type="amp"',
        'level="group"',
        f'position="{group_index}"',
        f'parameter="{parameter}"',
        'translation="linear"',
    ]
    # AMP_VOLUME is a modulation/control gain in DecentSampler. Keep its UI
    # range explicitly at 0..1 so 1.0 means unity relative to the mapped WAV.
    # The base <group> volume is intentionally left at DecentSampler's 1.0
    # default; otherwise the base volume and knob gain multiply together.
    if parameter == "AMP_VOLUME":
        attrs.extend([
            'translationOutputMin="0.0"',
            'translationOutputMax="1.0"',
        ])
    return f"<binding {' '.join(attrs)} />"


def _group_filter_binding(
    group_index: int,
    effect_index: int,
    parameter: str,
    *,
    translation: str,
    translation_table: str = "",
    output_min: str = "",
    output_max: str = "",
) -> str:
    attrs = [
        'type="effect"',
        'level="group"',
        f'groupIndex="{group_index}"',
        f'effectIndex="{effect_index}"',
        f'parameter="{parameter}"',
        f'translation="{translation}"',
    ]
    if translation_table:
        attrs.append(f'translationTable="{translation_table}"')
    if output_min:
        attrs.append(f'translationOutputMin="{output_min}"')
    if output_max:
        attrs.append(f'translationOutputMax="{output_max}"')
    return f"<binding {' '.join(attrs)} />"


def _copy_tri_background(export_dir: Path) -> Optional[Path]:
    """Copy optional Tri_Background artwork into one exported preset folder.

    Lookup is case-insensitive and accepts PNG/JPG/JPEG/WEBP. When no matching
    asset exists, the preset simply uses the normal dark bgColor fallback.
    """
    asset_dir = TRI_BACKGROUND_ASSET_DIR
    if not asset_dir.exists():
        return None

    supported = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = [
        path
        for path in asset_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported
        and path.stem.lower() == TRI_BACKGROUND_BASENAME.lower()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: (path.suffix.lower() != ".png", path.name.lower()))
    source = candidates[0]
    destination = export_dir / f"Tri_Background{source.suffix.lower()}"
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _horizontal_slider(
    *,
    x: int,
    y: int,
    label: str,
    min_value: str | float,
    max_value: str | float,
    value: str | float,
    binding: str,
    width: int = 120,
    height: int = 36,
    text_size: int = 13,
    slider_type: str = "float",
    value_type: Optional[str] = None,
    parameter_name: Optional[str] = None,
) -> List[str]:
    """Compact DecentSampler horizontal slider using labeled-knob syntax.

    DecentSampler renders a labeled-knob as a horizontal slider when
    style="linear_horizontal". This lets the tri-layer UI expose the full
    single-keybed FX set without turning the lower panel into another row of
    large rotaries.
    """
    attrs = [
        f'x="{int(x)}"',
        f'y="{int(y)}"',
        f'width="{int(width)}"',
        f'height="{int(height)}"',
        f'label="{html.escape(str(label))}"',
        f'parameterName="{html.escape(str(parameter_name or label))}"',
        f'type="{html.escape(str(slider_type))}"',
        f'minValue="{min_value}"',
        f'maxValue="{max_value}"',
        f'value="{value}"',
        f'textSize="{int(text_size)}"',
        f'textColor="{DS_UI_TEXT_COLOR}"',
        f'trackForegroundColor="{DS_UI_TRACK_FOREGROUND}"',
        f'trackBackgroundColor="{DS_UI_TRACK_BACKGROUND}"',
        'style="linear_horizontal"',
    ]
    if value_type:
        attrs.append(f'valueType="{html.escape(str(value_type))}"')
    return [
        f'      <labeled-knob {" ".join(attrs)}>',
        f'        {binding}',
        '      </labeled-knob>',
    ]


def _write_tri_dspreset(
    path: Path,
    layer_samples: Sequence[Sequence[ExportedSample]],
    *,
    export_dir: Path,
    layer_titles: Sequence[str],
    layer_volumes: Sequence[float],
    master_volume: float,
    instrument_id: str,
    instrument_name: str = "",
    background_filename: Optional[str] = None,
) -> None:
    filter_table = _log_translation_table()
    custom_instrument_name = _normalize_instrument_name(instrument_name)

    # Keep the proven 980px-wide coordinate system, but request a slightly taller
    # default plugin window. All controls still finish well above the keyboard.
    canvas_width = TRI_UI_WIDTH
    canvas_height = TRI_UI_HEIGHT
    column_width = 280

    # Visual order is Support 1 | Main | Support 2. Group order remains
    # Main=0, Support 1=1, Support 2=2 so existing bindings remain stable.
    visual_columns = (
        (1, 20),
        (0, 350),
        (2, 680),
    )

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<DecentSampler minVersion="1.29.0">',
        (
            f'  <ui width="{canvas_width}" height="{canvas_height}" layoutMode="relative" '
            f'bgMode="top_left" bgColor="{DS_UI_BG_COLOR}"'
            + (f' bgImage="{html.escape(background_filename)}"' if background_filename else '')
            + '>'
        ),
        '    <tab name="main">',
        # A custom catalog name occupies the primary title line. The product
        # label becomes a smaller subtitle beneath it. Without a custom name,
        # preserve the original single-line Foundation-1 title.
        *(
            [
                _label(0, 2, custom_instrument_name, width=canvas_width, height=24, size=22),
                _label(0, 25, "Foundation-1 Layered Keybed", width=canvas_width, height=14, size=11, color=DS_UI_MUTED_TEXT_COLOR),
            ]
            if custom_instrument_name
            else [
                _label(0, 9, "Foundation-1 Layered Keybed", width=canvas_width, height=24, size=21),
            ]
        ),
    ]

    # Top mixer. Main remains larger and centred.
    mixer_positions = (
        (1, 123, 47, 64, "SUPPORT 1", layer_volumes[1]),
        (0, 451, 38, 80, "MAIN", layer_volumes[0]),
        (2, 781, 47, 64, "SUPPORT 2", layer_volumes[2]),
    )
    for group_index, x, y, size, label, volume in mixer_positions:
        lines += _knob(
            x=x,
            y=y,
            label=label,
            min_value="0.0",
            max_value="1.0",
            value=f"{float(volume):.3f}",
            binding=_group_amp_binding(group_index, "AMP_VOLUME"),
            width=size,
            height=size + 8,
            text_size=16 if group_index == 0 else 14,
        )

    lines += _knob(
        x=898,
        y=20,
        label="MASTER",
        min_value="0.0",
        max_value="1.0",
        value=f"{float(master_volume):.3f}",
        binding=(
            '<binding type="amp" level="instrument" position="0" parameter="AMP_VOLUME" '
            'translation="linear" translationOutputMin="0.0" translationOutputMax="1.0" />'
        ),
        width=68,
        height=76,
        text_size=14,
    )

    for group_index, left in visual_columns:
        role = TRI_LAYER_ROLES[group_index]
        title = str(layer_titles[group_index] or role)
        lines.append(_centered_section_label(left, left + column_width, 112, role, size=19))
        lines.append(
            _label(
                left,
                136,
                title,
                width=column_width,
                height=18,
                size=14,
                color=DS_UI_MUTED_TEXT_COLOR,
            )
        )

        # Per-layer ADSR.
        adsr_specs = (
            ("Attack", "ENV_ATTACK", 0.0, 10.0, TRI_LAYER_DEFAULT_ATTACK),
            ("Decay", "ENV_DECAY", 0.0, 25.0, TRI_LAYER_DEFAULT_DECAY),
            ("Sustain", "ENV_SUSTAIN", 0.0, 1.0, TRI_LAYER_DEFAULT_SUSTAIN),
            ("Release", "ENV_RELEASE", 0.0, 25.0, TRI_LAYER_DEFAULT_RELEASE),
        )
        for x_offset, (label, parameter, minimum, maximum, value) in zip(
            (0, 70, 140, 210), adsr_specs
        ):
            lines += _knob(
                x=left + x_offset,
                y=156,
                label=label,
                min_value=f"{minimum:g}",
                max_value=f"{maximum:g}",
                value=f"{float(value):.4f}",
                binding=_group_amp_binding(group_index, parameter),
                width=68,
                height=68,
                text_size=16,
            )

        # Filters follow the ordinary exporter: large cutoff rotaries, with the
        # smaller Q controls tucked onto the outside shoulders.
        lines += _knob(
            x=left + 55,
            y=224,
            label="Lowpass",
            min_value=f"{DS_FILTER_CONTROL_MIN:g}",
            max_value=f"{DS_FILTER_CONTROL_MAX:g}",
            value=f"{DS_FILTER_CONTROL_MAX:g}",
            binding=_group_filter_binding(
                group_index,
                0,
                "FX_FILTER_FREQUENCY",
                translation="table",
                translation_table=filter_table,
            ),
            width=72,
            height=72,
            text_size=15,
        )
        lines += _knob(
            x=left + 20,
            y=230,
            label="Q",
            parameter_name="Lowpass Q",
            min_value=f"{DS_FILTER_Q_CONTROL_MIN:g}",
            max_value=f"{DS_FILTER_Q_CONTROL_MAX:g}",
            value=f"{DS_FILTER_Q_DEFAULT:.3f}",
            binding=_group_filter_binding(
                group_index,
                0,
                "FX_FILTER_RESONANCE",
                translation="linear",
                output_min=f"{DS_FILTER_RESONANCE_MIN:g}",
                output_max=f"{DS_FILTER_RESONANCE_MAX:g}",
            ),
            width=42,
            height=52,
            text_size=11,
        )
        lines += _knob(
            x=left + 153,
            y=224,
            label="Highpass",
            min_value=f"{DS_FILTER_CONTROL_MIN:g}",
            max_value=f"{DS_FILTER_CONTROL_MAX:g}",
            value=f"{DS_FILTER_CONTROL_MIN:g}",
            binding=_group_filter_binding(
                group_index,
                1,
                "FX_FILTER_FREQUENCY",
                translation="table",
                translation_table=filter_table,
            ),
            width=72,
            height=72,
            text_size=15,
        )
        lines += _knob(
            x=left + 225,
            y=230,
            label="Q",
            parameter_name="Highpass Q",
            min_value=f"{DS_FILTER_Q_CONTROL_MIN:g}",
            max_value=f"{DS_FILTER_Q_CONTROL_MAX:g}",
            value=f"{DS_FILTER_Q_DEFAULT:.3f}",
            binding=_group_filter_binding(
                group_index,
                1,
                "FX_FILTER_RESONANCE",
                translation="linear",
                output_min=f"{DS_FILTER_RESONANCE_MIN:g}",
                output_max=f"{DS_FILTER_RESONANCE_MAX:g}",
            ),
            width=42,
            height=52,
            text_size=11,
        )

    # Full global FX set, grouped like the ordinary single-instrument export.
    # Reverb and Delay retain one primary rotary each. Secondary parameters use
    # compact horizontal controls, as demonstrated by third-party DecentSampler
    # presets, allowing all controls to fit above the keyboard.
    # The taller 980x600 canvas leaves a genuine lower panel above the built-in
    # keyboard. The shared title is vertically centred between the layer filters and the module headings.
    lines.append(_centered_section_label(18, 962, 299, "Global Effects", size=18))

    # GLIDE — one shared instrument-level utility for all three layers. Keep it
    # in the unused upper-left corner so it stays visually separate from the
    # Global Effects modules. "On" maps to native legato glide; there is no
    # per-layer glide control and the always mode is intentionally omitted.
    lines.append(
        _label(
            8, 16, "Glide",
            width=48, height=20, size=13,
            color=DS_UI_TEXT_COLOR, h_align="right",
        )
    )
    lines += [
        (
            f'      <menu x="56" y="13" width="78" height="24" value="1" requireSelection="true" '
            f'textColor="{DS_UI_TEXT_COLOR}" backgroundColor="FF25262A" '
            f'highlightedTextColor="FF000000" highlightedBackgroundColor="FFD8D8D8">'
        ),
        '        <option name="Off">',
        '          <binding type="amp" level="instrument" position="0" parameter="GLIDE_MODE" translation="fixed_value" translationValue="off" />',
        '        </option>',
        '        <option name="On">',
        '          <binding type="amp" level="instrument" position="0" parameter="GLIDE_MODE" translation="fixed_value" translationValue="legato" />',
        '        </option>',
        '      </menu>',
    ]
    lines += _knob(
        x=140, y=2, label="Time",
        min_value="0.0", max_value=f"{TRI_GLIDE_MAX_TIME:g}", value=f"{TRI_GLIDE_DEFAULT_TIME:.3f}",
        parameter_name="Glide Time",
        binding='<binding type="amp" level="instrument" position="0" parameter="GLIDE_TIME" />',
        width=42, height=52, text_size=11,
    )

    # Global FX module headings remain unchanged; Glide now lives in the upper-left utility area.
    lines.append(_centered_section_label(18, 302, 334, "Space", size=17))
    lines.append(_centered_section_label(322, 626, 334, "Delay", size=17))
    lines.append(_centered_section_label(640, 962, 334, "Color & Distortion", size=17))

    # Space: Reverb + Room + Damping.
    lines += _knob(
        x=34,
        y=362,
        label="Reverb",
        min_value="0.0",
        max_value="1.0",
        value="0.10",
        binding='<binding type="effect" level="instrument" tags="tri-reverb" parameter="FX_REVERB_WET_LEVEL" translation="linear" />',
        width=74,
        height=74,
        text_size=15,
    )
    lines += _horizontal_slider(
        x=108,
        y=362,
        label="Room",
        min_value="0.0",
        max_value="1.0",
        value="0.70",
        binding='<binding type="effect" level="instrument" tags="tri-reverb" parameter="FX_REVERB_ROOM_SIZE" translation="linear" />',
        width=174,
        height=36,
        text_size=14,
    )
    lines += _horizontal_slider(
        x=108,
        y=406,
        label="Damping",
        min_value="0.0",
        max_value="1.0",
        value="0.30",
        binding='<binding type="effect" level="instrument" tags="tri-reverb" parameter="FX_REVERB_DAMPING" translation="linear" />',
        width=174,
        height=36,
        text_size=14,
    )

    # Delay: Mix + Time + Feedback.
    lines += _knob(
        x=338,
        y=362,
        label="Delay",
        min_value="0.0",
        max_value="1.0",
        value="0.0",
        binding='<binding type="effect" level="instrument" tags="tri-delay" parameter="FX_WET_LEVEL" translation="linear" />',
        width=74,
        height=74,
        text_size=15,
    )
    lines += _horizontal_slider(
        x=412,
        y=362,
        label="Time",
        min_value="0",
        max_value="20",
        value="10",
        slider_type="integer",
        value_type="musical_time",
        binding='<binding type="effect" level="instrument" tags="tri-delay" parameter="FX_DELAY_TIME" />',
        width=194,
        height=36,
        text_size=14,
    )
    lines += _horizontal_slider(
        x=412,
        y=406,
        label="Feedback",
        min_value="0.0",
        max_value="0.95",
        value="0.20",
        binding='<binding type="effect" level="instrument" tags="tri-delay" parameter="FX_FEEDBACK" translation="linear" />',
        width=194,
        height=36,
        text_size=14,
    )

    # Color & Distortion: two modulation mixes over the complete bit-crusher row.
    lines += _horizontal_slider(
        x=646,
        y=362,
        label="Chorus",
        min_value="0.0",
        max_value="1.0",
        value="0.0",
        binding='<binding type="effect" level="instrument" tags="tri-chorus" parameter="FX_MIX" translation="linear" />',
        width=146,
        height=36,
        text_size=14,
    )
    lines += _horizontal_slider(
        x=806,
        y=362,
        label="Phaser",
        min_value="0.0",
        max_value="1.0",
        value="0.0",
        binding='<binding type="effect" level="instrument" tags="tri-phaser" parameter="FX_MIX" translation="linear" />',
        width=146,
        height=36,
        text_size=14,
    )
    lines += _horizontal_slider(
        x=646,
        y=406,
        label="Bit Mix",
        min_value="0.0",
        max_value="1.0",
        value="0.0",
        binding='<binding type="effect" level="instrument" tags="tri-bitcrush" parameter="FX_MIX" translation="linear" />',
        width=94,
        height=36,
        text_size=12,
    )
    lines += _horizontal_slider(
        x=752,
        y=406,
        label="Bits",
        min_value="1",
        max_value="16",
        value="8",
        slider_type="integer",
        binding='<binding type="effect" level="instrument" tags="tri-bitcrush" parameter="FX_BIT_DEPTH" />',
        width=94,
        height=36,
        text_size=12,
    )
    lines += _horizontal_slider(
        x=858,
        y=406,
        label="Rate",
        min_value="1",
        max_value="32",
        value="4",
        slider_type="integer",
        binding='<binding type="effect" level="instrument" tags="tri-bitcrush" parameter="FX_SAMPLE_RATE_REDUCTION" />',
        width=94,
        height=36,
        text_size=12,
    )

    # Serial metadata sits in the lower-left, immediately beneath the global
    # effects and above DecentSampler's built-in keyboard.
    lines.append(
        _label(
            24,
            448,
            f"ID: {instrument_id or 'N/A'}",
            width=300,
            height=16,
            size=10,
            color="99FFFFFF",
            h_align="left",
        )
    )

    lines.extend(['    </tab>', '  </ui>'])

    # Group order is deliberately Main, Support 1, Support 2. Base volumes remain
    # at unity; the four mixer knobs are the only attenuation stages.
    lines.append(
        f'  <groups glideTime="{TRI_GLIDE_DEFAULT_TIME:.3f}" glideMode="{TRI_GLIDE_DEFAULT_MODE}">'
    )
    for group_index, samples in enumerate(layer_samples):
        role = TRI_LAYER_ROLES[group_index]
        lines.append(
            f'    <group name="{html.escape(role)}" tags="tri-layer-{group_index + 1}" '
            f'ampVelTrack="0.0" '
            f'attack="{TRI_LAYER_DEFAULT_ATTACK:.4f}" decay="{TRI_LAYER_DEFAULT_DECAY:.4f}" '
            f'sustain="{TRI_LAYER_DEFAULT_SUSTAIN:.4f}" release="{TRI_LAYER_DEFAULT_RELEASE:.4f}">'
        )
        lines.extend(_group_sample_lines(samples, export_dir))
        lines.extend(
            [
                '      <effects>',
                f'        <effect type="lowpass" tags="tri-layer-{group_index + 1}-lowpass" frequency="22000" resonance="0.7" />',
                f'        <effect type="highpass" tags="tri-layer-{group_index + 1}-highpass" frequency="20" resonance="0.7" />',
                '      </effects>',
                '    </group>',
            ]
        )
    lines.append('  </groups>')

    lines.extend(
        [
            '  <effects>',
            '    <effect type="reverb" tags="tri-reverb" roomSize="0.70" damping="0.30" wetLevel="0.10" />',
            '    <effect type="delay" tags="tri-delay" delayTimeFormat="musical_time" delayTime="10" stereoOffset="0.01" feedback="0.20" wetLevel="0.0" />',
            '    <effect type="chorus" tags="tri-chorus" mix="0.0" modDepth="0.75" modRate="0.35" />',
            '    <effect type="phaser" tags="tri-phaser" mix="0.0" modDepth="0.85" modRate="0.25" centerFrequency="700" feedback="0.70" />',
            '    <effect type="bit_crusher" tags="tri-bitcrush" bitDepth="8" sampleRateReduction="4" mix="0.0" />',
            '  </effects>',
            '  <midi/>',
            '  <tags/>',
            '</DecentSampler>',
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def _write_layered_manifest(
    path: Path,
    *,
    preset_id: str,
    layer_run_folders: Sequence[Path],
    layer_titles: Sequence[str],
    layer_volumes: Sequence[float],
    master_volume: float,
    layer_samples: Sequence[Sequence[ExportedSample]],
    layer_rows: Sequence[Sequence[Dict[str, Any]]],
    export_dir: Path,
    dspreset_path: Path,
) -> None:
    """Write the original compact tri-layer catalog shape plus prompts."""
    payload = {
        "type": "tri_layered_keybed",
        "preset_id": preset_id,
        "dspreset_path": _relpath(dspreset_path, export_dir),
        "layer_count": 3,
        "master_volume": float(master_volume),
        "layers": [],
    }
    for index in range(3):
        catalog = _catalog_metadata_from_rows(layer_rows[index])
        payload["layers"].append(
            {
                "index": index,
                "role": TRI_LAYER_ROLES[index],
                "title": layer_titles[index],
                "visible_prompt": catalog["visible_prompt"],
                "conditioning_prompt": catalog["conditioning_prompt"],
                "default_volume": float(layer_volumes[index]),
                "run_folder": str(layer_run_folders[index]),
                "sample_count": len(layer_samples[index]),
                "sample_paths": [
                    _relpath(Path(sample.sample_paths_by_format["dspreset"]), export_dir)
                    for sample in layer_samples[index]
                ],
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_tri_layered_keybed_run(
    layer_run_folders: Sequence[str | os.PathLike[str]],
    *,
    preset_id: Optional[str] = None,
    layer_titles: Optional[Sequence[str]] = None,
    layer_volumes: Sequence[float] = TRI_LAYER_DEFAULT_VOLUMES,
    master_volume: float = TRI_LAYER_MASTER_VOLUME,
    trim_tail: bool = True,
    keep_raw_samples: bool = False,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    frame_ms: float = DEFAULT_FRAME_MS,
    tail_pad_ms: float = DEFAULT_TAIL_PAD_MS,
    fade_ms: float = DEFAULT_FADE_MS,
    terminal_fade_ms: float = DEFAULT_TERMINAL_FADE_MS,
    min_keep_ms: float = DEFAULT_MIN_KEEP_MS,
    min_trim_ms: float = DEFAULT_MIN_TRIM_MS,
    clean_existing: bool = False,
    instrument_name: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Export three generated keybed runs as one layered DecentSampler preset.

    By default only the mapped DecentSampler WAVs are written. The original
    sequence chunks remain in each generation run, so a second folder of raw
    scheduled note slices is unnecessary. Set ``keep_raw_samples=True`` only
    when debugging the slicer or tail cleanup.
    """
    folders = [Path(folder) for folder in layer_run_folders]
    if len(folders) != 3:
        raise ValueError("Tri-layer export requires exactly three Keybed run folders.")
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Layer run folder does not exist: {folder}")

    volumes = tuple(float(value) for value in layer_volumes)
    if len(volumes) != 3:
        raise ValueError("layer_volumes must contain exactly three values.")
    if any(value < 0.0 or value > 1.0 for value in volumes):
        raise ValueError("Each initial layer volume must be between 0.0 and 1.0.")
    master_volume = float(master_volume)
    if not 0.0 <= master_volume <= 1.0:
        raise ValueError("master_volume must be between 0.0 and 1.0.")

    source_preset_id = _safe_slug(
        preset_id or _preset_id_from_parent(folders),
        fallback="tri_layer_keybed",
    )
    instrument_name = _normalize_instrument_name(instrument_name)
    resolved_preset_id = _preset_id_with_instrument_name(source_preset_id, instrument_name)
    export_root = _default_export_root(folders[0])
    export_dir = export_root / f"{resolved_preset_id}_decent_sampler"
    if clean_existing and export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    _status(f"Building {resolved_preset_id} from three Keybed runs.", progress_callback)

    all_layer_samples: List[List[ExportedSample]] = []
    all_rows: List[List[Dict[str, Any]]] = []
    for layer_index, run_folder in enumerate(folders):
        role_slug = ("main", "support_1", "support_2")[layer_index]
        legacy_raw_dir = export_dir / f"layer_{layer_index + 1}_{role_slug}_raw_samples"
        raw_dir = legacy_raw_dir if bool(keep_raw_samples) else None
        if raw_dir is None and legacy_raw_dir.exists():
            # Older exporter versions always wrote a second scheduled-slice copy.
            # Source chunks remain in the generation run, so remove that redundant
            # debug folder when re-exporting with the new default.
            shutil.rmtree(legacy_raw_dir)

        samples, rows = _slice_layer_run(
            run_folder,
            layer_index=layer_index,
            raw_dir=raw_dir,
            sample_dir=export_dir / f"layer_{layer_index + 1}_{role_slug}_samples",
            trim_tail=trim_tail,
            threshold_db=threshold_db,
            frame_ms=frame_ms,
            tail_pad_ms=tail_pad_ms,
            fade_ms=fade_ms,
            terminal_fade_ms=terminal_fade_ms,
            min_keep_ms=min_keep_ms,
            min_trim_ms=min_trim_ms,
            progress_callback=progress_callback,
        )
        all_layer_samples.append(samples)
        all_rows.append(rows)

    midi_sets = [{sample.midi for sample in samples} for samples in all_layer_samples]
    if not (midi_sets[0] == midi_sets[1] == midi_sets[2]):
        differences = [len(values) for values in midi_sets]
        raise ValueError(
            "All three layers must contain the same mapped MIDI notes. "
            f"Layer sample counts were {differences}."
        )

    titles = list(layer_titles or [])
    while len(titles) < 3:
        index = len(titles)
        titles.append(_layer_title_from_rows(all_rows[index], TRI_LAYER_ROLES[index]))
    titles = titles[:3]

    dspreset_path = export_dir / f"{resolved_preset_id}.dspreset"
    display_name, instrument_id = _display_name_and_id(resolved_preset_id)
    background_path = _copy_tri_background(export_dir)
    _status("Writing tri-layer DecentSampler mapping and controls.", progress_callback)
    _write_tri_dspreset(
        dspreset_path,
        all_layer_samples,
        export_dir=export_dir,
        layer_titles=titles,
        layer_volumes=volumes,
        master_volume=master_volume,
        instrument_id=instrument_id,
        instrument_name=instrument_name,
        background_filename=background_path.name if background_path else None,
    )

    layered_manifest_path = export_dir / "tri_layer_manifest.json"
    _write_layered_manifest(
        layered_manifest_path,
        preset_id=resolved_preset_id,
        layer_run_folders=folders,
        layer_titles=titles,
        layer_volumes=volumes,
        master_volume=master_volume,
        layer_samples=all_layer_samples,
        layer_rows=all_rows,
        export_dir=export_dir,
        dspreset_path=dspreset_path,
    )

    sample_count = len(all_layer_samples[0])
    tail_trimmed_count = sum(
        1
        for samples in all_layer_samples
        for sample in samples
        if sample.tail_trimmed
    )
    _status(
        f"Tri-layer Keybed Exported - ID {instrument_id or resolved_preset_id}; "
        f"{sample_count} notes x 3 layers.",
        progress_callback,
    )

    return {
        "export_root": str(export_root),
        "export_dir": str(export_dir),
        "export_dirs_by_format": {"dspreset": str(export_dir)},
        "preset_id": resolved_preset_id,
        "display_name": display_name,
        "instrument_id": instrument_id,
        "dspreset_path": str(dspreset_path),
        "layered_manifest": str(layered_manifest_path),
        "layer_run_folders": [str(folder) for folder in folders],
        "layer_titles": titles,
        "layer_volumes": list(volumes),
        "master_volume": master_volume,
        "notes_per_layer": sample_count,
        "total_mapped_samples": sample_count * 3,
        "tail_trimmed_count": tail_trimmed_count,
        "keep_raw_samples": bool(keep_raw_samples),
        "dspreset_background_path": str(background_path) if background_path else None,
        "dspreset_background_source_dir": str(TRI_BACKGROUND_ASSET_DIR),
        "ui_files": [str(dspreset_path)],
        "support_files": [
            str(layered_manifest_path),
            *([str(background_path)] if background_path else []),
        ],
    }


__all__ = [
    "TRI_LAYER_DEFAULT_VOLUMES",
    "TRI_LAYER_MASTER_VOLUME",
    "TRI_BACKGROUND_ASSET_DIR",
    "TRI_BACKGROUND_BASENAME",
    "TRI_UI_WIDTH",
    "TRI_UI_HEIGHT",
    "TRI_GLIDE_DEFAULT_MODE",
    "TRI_GLIDE_DEFAULT_TIME",
    "TRI_GLIDE_MAX_TIME",
    "export_tri_layered_keybed_run",
]
