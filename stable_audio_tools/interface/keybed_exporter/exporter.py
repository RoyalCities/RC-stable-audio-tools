from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio


NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
PC_TO_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DEFAULT_THRESHOLD_DB = -60.0
DEFAULT_FRAME_MS = 10.0
DEFAULT_TAIL_PAD_MS = 80.0
DEFAULT_FADE_MS = 8.0
DEFAULT_TERMINAL_FADE_MS = 120.0
DEFAULT_MIN_KEEP_MS = 250.0
DEFAULT_MIN_TRIM_MS = 40.0

FORMAT_SFZ = "sfz"
FORMAT_DSPRESET = "dspreset"
INSTRUMENT_NAME_MAX_CHARS = 30

# DecentSampler native glide/portamento defaults. Glide is opt-in so existing
# Foundation-1 exports retain their normal chromatic sample playback until the
# user selects Legato or Always in the preset UI.
DS_GLIDE_DEFAULT_MODE = "off"
DS_GLIDE_DEFAULT_TIME = 0.15
DS_GLIDE_MAX_TIME = 2.0


def _export_status(message: str, callback: Optional[Callable[[str], None]] = None) -> None:
    text = f"[Keybed Exporter] {message}"
    print(text, flush=True)
    if callback is not None:
        try:
            callback(message)
        except Exception:
            # A UI/status callback should never be able to break the actual export.
            pass

# DecentSampler UI asset bundled beside this exporter:
# stable_audio_tools/interface/keybed_exporter/decent_sampler_asset/Background.png
DECENT_SAMPLER_ASSET_DIR = Path(__file__).resolve().parent / "decent_sampler_asset"
DECENT_SAMPLER_BACKGROUND_BASENAME = "Background"

# Foundation-1 DecentSampler UI palette.
# bgColor is intentionally dark so white controls remain readable even when
# Background.png is missing or cannot be loaded.
DS_UI_BG_COLOR = "FF1A1B1E"
DS_UI_TEXT_COLOR = "FFFFFFFF"
DS_UI_MUTED_TEXT_COLOR = "CCFFFFFF"
DS_UI_TRACK_FOREGROUND = "FFFFFFFF"
DS_UI_TRACK_BACKGROUND = "44FFFFFF"

# Filter controls use a normalized continuous 0-1000 UI range. The binding
# translates that range onto a logarithmic 20 Hz - 22 kHz frequency curve.
DS_FILTER_CONTROL_MIN = 0.0
DS_FILTER_CONTROL_MAX = 1000.0
DS_FILTER_MIN_HZ = 20.0
DS_FILTER_MAX_HZ = 22000.0
DS_FILTER_TABLE_SEGMENTS = 40

# Small Q knobs use the same automation-friendly 0-1000 macro scale.
# They translate linearly onto DecentSampler's low/high-pass resonance range.
DS_FILTER_Q_CONTROL_MIN = 0.0
DS_FILTER_Q_CONTROL_MAX = 1000.0
DS_FILTER_RESONANCE_MIN = 0.001
DS_FILTER_RESONANCE_MAX = 5.0
DS_FILTER_RESONANCE_DEFAULT = 0.7
DS_FILTER_Q_DEFAULT = (
    (DS_FILTER_RESONANCE_DEFAULT - DS_FILTER_RESONANCE_MIN)
    / (DS_FILTER_RESONANCE_MAX - DS_FILTER_RESONANCE_MIN)
    * DS_FILTER_Q_CONTROL_MAX
)


@dataclass
class ExportedSample:
    note: str
    midi: int
    source_chunk_path: str
    scheduled_start_sec: float
    scheduled_end_sec: float
    final_source_start_sec: float
    final_source_end_sec: float
    export_duration_sec: float
    raw_sample_path: str
    sample_paths_by_format: Dict[str, str] = field(default_factory=dict)
    tail_trimmed: bool = False
    trim_method: str = "none"
    terminal_fade_applied: bool = False
    terminal_fade_ms: float = 0.0
    rootNote: int = 60
    pitch_keycenter: int = 60
    loNote: int = 60
    hiNote: int = 60
    source_row_name: str = ""
    source_row_index: int = -1

    @property
    def sample_path(self) -> str:
        if self.sample_paths_by_format:
            return next(iter(self.sample_paths_by_format.values()))
        return ""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _find_manifest(run_folder: Path) -> Path:
    preferred = run_folder / "manifest.jsonl"
    if preferred.exists():
        return preferred
    candidates = sorted(run_folder.rglob("manifest.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No manifest.jsonl found under: {run_folder}")


def note_to_midi(note: str) -> int:
    match = NOTE_RE.match(str(note or "").strip())
    if not match:
        raise ValueError(f"Invalid note name: {note!r}")
    name = match.group(1).upper()
    accidental = match.group(2).replace("♯", "#").replace("♭", "b")
    key = name + accidental
    octave = int(match.group(3))
    if key not in NOTE_TO_PC:
        raise ValueError(f"Unsupported note name: {note!r}")
    return (octave + 1) * 12 + int(NOTE_TO_PC[key])


def midi_to_note(midi: int) -> str:
    midi = int(midi)
    return f"{PC_TO_SHARP[midi % 12]}{(midi // 12) - 1}"


def note_filename(note: str) -> str:
    return str(note).replace("#", "sharp").replace("b", "flat")


def _safe_slug(value: str, fallback: str = "instrument", *, max_chars: int = 80) -> str:
    value = str(value or "").strip().lower().replace("#", "sharp")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    value = value[:max_chars].strip("_")
    return value or fallback


def _normalize_instrument_name(value: Optional[str], *, max_chars: int = INSTRUMENT_NAME_MAX_CHARS) -> str:
    """Normalize and clamp an optional user-facing instrument name."""
    text = " ".join(str(value or "").strip().split())[: int(max_chars)].strip()
    if not text or not re.search(r"[A-Za-z0-9]", text):
        return ""
    words: List[str] = []
    for word in text.split(" "):
        words.append(word if word.isupper() else word[:1].upper() + word[1:].lower())
    return " ".join(words)


def _instrument_name_slug(value: Optional[str]) -> str:
    display = _normalize_instrument_name(value)
    if not display:
        return ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", display)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:INSTRUMENT_NAME_MAX_CHARS].rstrip("_")


def _trailing_instrument_id(value: str) -> str:
    match = re.search(r"(?:^|[_-])([0-9a-fA-F]{8})(?:_\d{2})?$", str(value or "").strip())
    return match.group(1) if match else ""


def _preset_id_with_instrument_name(base_preset_id: str, instrument_name: Optional[str]) -> str:
    """Replace only the readable preset prefix while preserving its hash ID."""
    custom_slug = _instrument_name_slug(instrument_name)
    if not custom_slug:
        return str(base_preset_id or "").strip()
    instrument_id = _trailing_instrument_id(base_preset_id)
    return f"{custom_slug}_{instrument_id.lower()}" if instrument_id else custom_slug


def _normalize_formats(formats: Sequence[str]) -> List[str]:
    out: List[str] = []
    for fmt in formats:
        value = str(fmt or "").strip().lower()
        if value in {"sfz", ".sfz"}:
            norm = FORMAT_SFZ
        elif value in {"dspreset", "decentsampler", "decent_sampler", "ds", ".dspreset"}:
            norm = FORMAT_DSPRESET
        else:
            continue
        if norm not in out:
            out.append(norm)
    if not out:
        raise ValueError("No supported export formats requested. Use 'sfz' and/or 'dspreset'.")
    return out


def _parse_notes(row: Dict[str, Any]) -> List[str]:
    notes = row.get("notes_list") or row.get("notes") or []
    if isinstance(notes, str):
        notes = [n.strip() for n in notes.split(",") if n.strip()]
    return [str(n).strip() for n in notes if str(n).strip()]


def _parse_note_slices(row: Dict[str, Any], notes: Sequence[str]) -> List[Dict[str, Any]]:
    note_slices = row.get("note_slices") or []
    if isinstance(note_slices, str):
        try:
            note_slices = json.loads(note_slices)
        except Exception:
            note_slices = []

    if note_slices:
        return list(note_slices)

    # Fallback for older manifests: rebuild from row timing values.
    note_seconds = float(row.get("note_seconds") or 3.0)
    gap_seconds = float(row.get("gap_seconds") or 0.25)
    out: List[Dict[str, Any]] = []
    cursor = 0.0
    for note in notes:
        start = cursor
        end = start + note_seconds
        out.append({"note": note, "start_sec": start, "end_sec": end})
        cursor = end + gap_seconds
    return out


def _resolve_audio_path(run_folder: Path, raw_path: str) -> Path:
    if not raw_path:
        raise FileNotFoundError("Manifest row is missing audio_path")

    p = Path(raw_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.extend([
        run_folder / raw_path,
        run_folder / p.name,
        run_folder / "chunks" / p.name,
        run_folder / "audio" / p.name,
    ])

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue

    raise FileNotFoundError(f"Could not resolve chunk audio path from manifest: {raw_path}")


def _is_exportable_chunk_row(row: Dict[str, Any]) -> bool:
    if row.get("audio_path") in (None, ""):
        return False
    if row.get("export_hint") == "audition_only":
        return False
    role = str(row.get("role") or "").strip().lower()
    if role in {"chunk", "keybed_output"}:
        return True
    return row.get("note_slices") is not None and row.get("audio_path") is not None


def _to_float_audio(audio: torch.Tensor) -> torch.Tensor:
    if audio.dtype.is_floating_point:
        return audio.to(torch.float32).clamp(-1.0, 1.0)
    if audio.dtype == torch.int16:
        return audio.to(torch.float32).div(32768.0).clamp(-1.0, 1.0)
    if audio.dtype == torch.int32:
        return audio.to(torch.float32).div(2147483648.0).clamp(-1.0, 1.0)
    return audio.to(torch.float32).clamp(-1.0, 1.0)


def _save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_i16 = audio.to(torch.float32).clamp(-1.0, 1.0).mul(32767.0).to(torch.int16).cpu()
    torchaudio.save(str(path), audio_i16, int(sample_rate))


def _apply_fade_out(audio: torch.Tensor, sample_rate: int, fade_ms: float) -> torch.Tensor:
    fade_len = int(round(int(sample_rate) * float(fade_ms) / 1000.0))
    if fade_len <= 1 or audio.shape[-1] <= 1:
        return audio
    fade_len = min(fade_len, int(audio.shape[-1]))
    ramp = torch.linspace(1.0, 0.0, steps=fade_len, dtype=audio.dtype, device=audio.device)
    out = audio.clone()
    out[:, -fade_len:] *= ramp
    return out


def trim_tail_conservative(
    audio: torch.Tensor,
    sample_rate: int,
    *,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    frame_ms: float = DEFAULT_FRAME_MS,
    tail_pad_ms: float = DEFAULT_TAIL_PAD_MS,
    fade_ms: float = DEFAULT_FADE_MS,
    terminal_fade_ms: float = DEFAULT_TERMINAL_FADE_MS,
    min_keep_ms: float = DEFAULT_MIN_KEEP_MS,
    min_trim_ms: float = DEFAULT_MIN_TRIM_MS,
) -> Tuple[torch.Tensor, bool, str, float]:
    """Trim only trailing silence inside a pre-sliced note.

    Returns: (audio, tail_trimmed, trim_method, final_duration_sec)
    """
    if audio is None or audio.numel() == 0 or audio.shape[-1] <= 1:
        return audio, False, "none_empty", 0.0

    sr = int(sample_rate)
    n = int(audio.shape[-1])
    frame_len = max(1, int(round(sr * float(frame_ms) / 1000.0)))
    tail_pad = max(0, int(round(sr * float(tail_pad_ms) / 1000.0)))
    min_keep = max(1, int(round(sr * float(min_keep_ms) / 1000.0)))
    min_trim = max(1, int(round(sr * float(min_trim_ms) / 1000.0)))
    threshold_amp = float(10 ** (float(threshold_db) / 20.0))

    mono_peak = audio.abs().amax(dim=0)
    n_frames = int(math.ceil(n / frame_len))
    pad = (n_frames * frame_len) - n
    if pad > 0:
        mono_peak = F.pad(mono_peak, (0, pad))

    frame_peaks = mono_peak.view(n_frames, frame_len).amax(dim=1)
    active_frames = torch.nonzero(frame_peaks > threshold_amp, as_tuple=False).flatten()

    if active_frames.numel() == 0:
        proposed_end = min(n, min_keep)
        method = f"tail_peak_{threshold_db:g}db_all_below_threshold"
    else:
        last_active_frame = int(active_frames[-1].item())
        proposed_end = min(n, max(min_keep, ((last_active_frame + 1) * frame_len) + tail_pad))
        method = f"tail_peak_{threshold_db:g}db_frame_{frame_ms:g}ms_pad_{tail_pad_ms:g}ms"

    # Conservative guard: do nothing unless the trim is large enough to matter.
    if proposed_end >= n - min_trim:
        return audio.contiguous(), False, "none_tail_not_silent", n / sr

    trimmed = audio[:, :max(1, int(proposed_end))].contiguous()
    trimmed = _apply_fade_out(trimmed, sr, fade_ms)
    return trimmed, True, method, trimmed.shape[-1] / sr


def _relpath(path: Path, start: Path) -> str:
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except Exception:
        return os.path.relpath(str(path), str(start)).replace(os.sep, "/")


def _write_export_manifest(
    path: Path,
    samples: Sequence[ExportedSample],
    export_dir: Path,
    *,
    preset_id: str,
    run_folder: Path,
    rows: Sequence[Dict[str, Any]],
    normalized_formats: Sequence[str],
    sfz_path: Optional[Path],
    dspreset_path: Optional[Path],
) -> None:
    """Write one readable catalog manifest for the finished instrument.

    Keep this intentionally parallel to the tri-layer manifest. The original
    generation manifest remains the technical source of truth for note slicing;
    this file only records the identity, prompts, output files, source run, and
    mapped sample paths an end user may need when cataloging the instrument.
    """
    display_name, _instrument_id = _display_name_and_serial(preset_id)
    catalog = _catalog_metadata_from_rows(rows)

    sample_paths: Dict[str, List[str]] = {}
    for fmt in normalized_formats:
        paths = [
            _relpath(Path(sample.sample_paths_by_format[fmt]), export_dir)
            for sample in samples
            if fmt in sample.sample_paths_by_format
        ]
        if paths:
            sample_paths[str(fmt)] = paths

    payload = {
        "type": "single_keybed",
        "preset_id": preset_id,
        "sfz_path": _relpath(sfz_path, export_dir) if sfz_path is not None else None,
        "dspreset_path": _relpath(dspreset_path, export_dir) if dspreset_path is not None else None,
        "layer_count": 1,
        "master_volume": 1.0,
        "layers": [
            {
                "index": 0,
                "role": "Main",
                "title": display_name,
                "visible_prompt": catalog["visible_prompt"],
                "conditioning_prompt": catalog["conditioning_prompt"],
                "default_volume": 1.0,
                "run_folder": str(run_folder),
                "sample_count": len(samples),
                "sample_paths": sample_paths,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def _write_sfz(path: Path, samples: Sequence[ExportedSample], *, attack: float, release: float, export_dir: Path) -> None:
    lines = [
        "// Auto-generated Foundation-1 keybed export",
        "<group>",
        f"ampeg_attack={float(attack):.4f}",
        "ampeg_decay=0.0000",
        "ampeg_sustain=100",
        f"ampeg_release={float(release):.4f}",
        "",
    ]
    for s in samples:
        sample_path = _relpath(Path(s.sample_paths_by_format[FORMAT_SFZ]), export_dir)
        lines.append(
            f"<region> sample={sample_path} lokey={s.loNote} hikey={s.hiNote} pitch_keycenter={s.pitch_keycenter}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")



def _log_translation_table(
    *,
    input_min: float = DS_FILTER_CONTROL_MIN,
    input_max: float = DS_FILTER_CONTROL_MAX,
    output_min: float = DS_FILTER_MIN_HZ,
    output_max: float = DS_FILTER_MAX_HZ,
    segments: int = DS_FILTER_TABLE_SEGMENTS,
) -> str:
    """Build a dense piecewise-linear approximation of a logarithmic curve.

    DecentSampler's table translation interpolates between input/output pairs.
    Forty segments across 0-1000 gives quarter-octave-ish resolution over the
    roughly ten-octave 20 Hz - 22 kHz range while keeping automation continuous.
    """
    if segments < 1:
        raise ValueError("segments must be at least 1")
    if input_max <= input_min:
        raise ValueError("input_max must be greater than input_min")
    if output_min <= 0 or output_max <= output_min:
        raise ValueError("logarithmic output range must be positive and increasing")

    ratio = output_max / output_min
    pairs: List[str] = []

    for index in range(segments + 1):
        fraction = index / segments
        control_value = input_min + ((input_max - input_min) * fraction)
        frequency = output_min * (ratio ** fraction)

        control_text = f"{control_value:.6f}".rstrip("0").rstrip(".")
        frequency_text = f"{frequency:.3f}".rstrip("0").rstrip(".")
        pairs.append(f"{control_text},{frequency_text}")

    # Endpoint guard, matching the pattern used in DecentSampler's own
    # nonlinear filter examples.
    endpoint_guard = input_max + max(0.0001, (input_max - input_min) * 0.0001)
    endpoint_text = f"{endpoint_guard:.6f}".rstrip("0").rstrip(".")
    output_max_text = f"{output_max:.3f}".rstrip("0").rstrip(".")
    pairs.append(f"{endpoint_text},{output_max_text}")

    return ";".join(pairs)



def _knob(
    *,
    x: int,
    y: int,
    label: str,
    min_value: str | float,
    max_value: str | float,
    value: str | float,
    binding: str,
    knob_type: str = "float",
    value_type: Optional[str] = None,
    parameter_name: Optional[str] = None,
    text_size: int = 15,
    width: int = 90,
    height: int = 92,
) -> List[str]:
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'width="{width}"',
        f'height="{height}"',
        f'textSize="{int(text_size)}"',
        f'textColor="{DS_UI_TEXT_COLOR}"',
        f'trackForegroundColor="{DS_UI_TRACK_FOREGROUND}"',
        f'trackBackgroundColor="{DS_UI_TRACK_BACKGROUND}"',
        f'label="{html.escape(label)}"',
        f'parameterName="{html.escape(parameter_name or label)}"',
        f'type="{knob_type}"',
        f'minValue="{min_value}"',
        f'maxValue="{max_value}"',
        f'value="{value}"',
    ]
    if value_type:
        attrs.append(f'valueType="{value_type}"')
    return [
        f'      <labeled-knob {" ".join(attrs)}>',
        f'        {binding}',
        '      </labeled-knob>',
    ]


def _label(
    x: int,
    y: int,
    text: str,
    *,
    width: int = 160,
    height: int = 24,
    size: int = 15,
    color: str = DS_UI_TEXT_COLOR,
    h_align: str = "center",
    v_align: str = "center",
) -> str:
    return (
        f'      <label x="{x}" y="{y}" width="{width}" height="{height}" '
        f'text="{html.escape(text)}" textSize="{size}" textColor="{color}" '
        f'hAlign="{h_align}" vAlign="{v_align}"/>'
    )


def _centered_section_label(
    left: int,
    right: int,
    y: int,
    text: str,
    *,
    size: int = 16,
) -> str:
    """Center a section heading across the exact span of its knob group."""
    return _label(
        left,
        y,
        text,
        width=max(1, right - left),
        height=24,
        size=size,
        h_align="center",
        v_align="center",
    )


def _display_name_and_serial(preset_id: str) -> Tuple[str, str]:
    """Split a trailing eight-character hexadecimal generation ID from a preset name.

    Example:
        keys_digital_piano_3c909cc5
        -> ("Keys Digital Piano", "3C909CC5")
    """
    raw = str(preset_id or "").strip()
    match = re.search(r"(?:^|[_-])([0-9a-fA-F]{8})$", raw)

    serial = ""
    name_part = raw
    if match:
        serial = match.group(1).upper()
        name_part = raw[:match.start()].rstrip("_- ")

    display_name = re.sub(r"[_-]+", " ", name_part).strip()
    display_name = re.sub(r"\s+", " ", display_name)
    if display_name:
        display_name = display_name.title()
    else:
        display_name = "Foundation-1 Keybed"

    return display_name, serial


def _prompt_metadata_for_ui(text: str, *, max_chars: int = 155) -> str:
    """Format comma-separated prompt tags as one polished bullet-separated line."""
    parts = [
        " ".join(part.strip().split())
        for part in str(text or "").split(",")
        if part.strip()
    ]
    formatted = " • ".join(parts)
    if len(formatted) <= max_chars:
        return formatted
    return formatted[: max_chars - 1].rstrip(" •") + "…"


def _is_note_token(token: str) -> bool:
    return NOTE_RE.match(str(token or "").strip()) is not None


def _prompt_flavour_from_rows(rows: Sequence[Dict[str, Any]]) -> str:
    """Return the user-facing keybed flavour prompt for the DecentSampler UI.

    Prefer the actual conditioning prompt because it includes Wet/Dry and any FX
    tokens. Strip KEYBED control grammar and note-sequence values so the UI line
    stays descriptive rather than chunk-specific.
    """
    control_tokens = {
        "keybed",
        "sequence",
        "timbre profile",
        "chromatic chunk",
    }

    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        if not prompt:
            continue

        tokens: List[str] = []
        skip_next = False
        stop = False
        for raw in prompt.split(","):
            token = raw.strip()
            if not token or stop:
                continue
            low = token.lower()

            if skip_next:
                skip_next = False
                continue
            if low in control_tokens:
                continue
            if low in {"target note", "target position"}:
                skip_next = True
                continue
            if low == "note sequence":
                stop = True
                continue
            if low.startswith("keybed_pos_"):
                continue
            if _is_note_token(token):
                continue
            tokens.append(token)

        out = ", ".join(tokens).strip()
        if out:
            return out

    # Fallback for older manifests that have descriptor/wetdry but no prompt.
    for row in rows:
        descriptor = str(row.get("descriptor") or "").strip()
        if descriptor:
            wetdry = str(row.get("wetdry") or "").strip()
            if wetdry and wetdry.lower() not in descriptor.lower():
                return f"{descriptor}, {wetdry}"
            return descriptor

    return ""


def _conditioning_prompt_template(prompt: str) -> str:
    """Collapse chunk-specific note values into one readable prompt template."""
    tokens = [part.strip() for part in str(prompt or "").split(",") if part.strip()]
    if not tokens:
        return ""

    out: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        low = token.casefold()
        out.append(token)

        if low == "note sequence":
            out.append("[notes]")
            break
        if low in {"target note", "target position"}:
            out.append("[note]")
            break
        index += 1

    return ", ".join(out)


def _catalog_metadata_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract one human-readable identity record from chunk-level metadata."""
    visible_prompt = ""
    conditioning_prompt = ""
    wetdry = ""
    seed: Any = None
    model = ""

    for row in rows:
        if not visible_prompt:
            visible_prompt = str(
                row.get("visible_prompt")
                or row.get("descriptor")
                or ""
            ).strip()
        if not conditioning_prompt:
            raw_prompt = str(
                row.get("conditioning_prompt")
                or row.get("prompt")
                or ""
            ).strip()
            if raw_prompt:
                conditioning_prompt = _conditioning_prompt_template(raw_prompt)
        if not wetdry:
            wetdry = str(row.get("wetdry") or "").strip()
        if seed is None:
            seed = row.get("base_seed")
            if seed is None:
                seed = row.get("seed")
        if not model:
            model = str(row.get("model") or "").strip()

        if visible_prompt and conditioning_prompt and wetdry and seed is not None and model:
            break

    if not visible_prompt:
        visible_prompt = _prompt_flavour_from_rows(rows)

    return {
        "visible_prompt": visible_prompt,
        "conditioning_prompt": conditioning_prompt,
        "wetdry": wetdry,
        "seed": seed,
        "model": model,
    }


def _shorten_for_ui(text: str, *, max_chars: int = 150) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip(" ,") + "…"



def _copy_decentsampler_background(export_dir: Path) -> Optional[Path]:
    """Copy the bundled DecentSampler background into one exported instrument.

    Preferred source:
        keybed_exporter/decent_sampler_asset/Background.png

    The lookup is case-insensitive and accepts PNG/JPG/JPEG/WEBP. If no
    background exists, export continues without a bgImage rather than failing.
    """
    asset_dir = DECENT_SAMPLER_ASSET_DIR
    if not asset_dir.exists():
        return None

    supported = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = [
        p for p in asset_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in supported
        and p.stem.lower() == DECENT_SAMPLER_BACKGROUND_BASENAME.lower()
    ]
    if not candidates:
        return None

    # Prefer PNG, then stable alphabetical ordering.
    candidates.sort(key=lambda p: (p.suffix.lower() != ".png", p.name.lower()))
    source = candidates[0]

    destination = export_dir / f"Background{source.suffix.lower()}"
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_dspreset(
    path: Path,
    samples: Sequence[ExportedSample],
    *,
    attack: float,
    release: float,
    export_dir: Path,
    instrument_id: str = "",
    prompt_flavour: str = "",
    instrument_name: str = "",
    background_filename: Optional[str] = None,
) -> None:
    prompt_metadata = _prompt_metadata_for_ui(prompt_flavour)
    esc_prompt = html.escape(prompt_metadata)
    esc_id = html.escape(str(instrument_id or "").strip().upper())
    custom_instrument_name = _normalize_instrument_name(instrument_name)
    filter_translation_table = _log_translation_table()

    # Top-level effects are instrument/global effects. Keep defaults neutral:
    # filters open, reverb/delay/modulation/bitcrush at 0 wet. Bitcrush uses an audible preset depth/rate with mix=0 for clean default playback. No Tone/Drive/Shape macro.
    # v13 removes the extra gain effect and explicit group volume attr so raw playback uses DecentSampler defaults.
    # UI v13 keeps the corrected reverb and centered lower row, and lowers the main title slightly further to align with the final background artwork.
    # Group ampVelTrack is set to 0.0 so MIDI note velocity does not quietly attenuate normalized generated samples.
    # Tags are used for bindings so future effect-order changes do not break knobs.
    effects = [
        '  <effects>',
        # Filter resonance must stay inside DecentSampler's valid 0.001-5.0 range.
        # Using the documented/default 0.7 avoids the weird low-level behavior seen with resonance=0.0.
        '    <effect type="lowpass" tags="f1-lowpass" frequency="22000" resonance="0.7" />',
        '    <effect type="highpass" tags="f1-highpass" frequency="20" resonance="0.7" />',
        '    <effect type="reverb" tags="f1-reverb" roomSize="0.70" damping="0.30" wetLevel="0.0" />',
        '    <effect type="delay" tags="f1-delay" delayTimeFormat="musical_time" delayTime="10" stereoOffset="0.01" feedback="0.20" wetLevel="0.0" />',
        # v5 only exposed mix while depth/rate stayed at mild defaults. Higher neutral depth/rate makes the knob useful at 1.0.
        '    <effect type="chorus" tags="f1-chorus" mix="0.0" modDepth="0.75" modRate="0.35" />',
        '    <effect type="phaser" tags="f1-phaser" mix="0.0" modDepth="0.85" modRate="0.25" centerFrequency="700" feedback="0.70" />',
        # Keep Bit Mix at 0.0 for clean load, but expose the two tonal parameters so the user can dial in real crush depth.
        '    <effect type="bit_crusher" tags="f1-bitcrush" bitDepth="8" sampleRateReduction="4" mix="0.0" />',
        '  </effects>',
    ]


    # Keep the UI canvas tall enough that DecentSampler's built-in keyboard
    # attaches below the controls instead of overlaying the lower control row.
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<DecentSampler minVersion="1.29.0">',
        (
            f'  <ui width="980" height="540" layoutMode="relative" bgMode="top_left" bgColor="{DS_UI_BG_COLOR}"'
            + (f' bgImage="{html.escape(background_filename)}"' if background_filename else '')
            + '>'
        ),
        '    <tab name="main">',

        # Optional catalog name above a smaller product subtitle. When no custom
        # name is supplied, preserve the original single-line header.
        *(
            [
                _label(0, 18, custom_instrument_name, width=980, height=28, size=23, color=DS_UI_TEXT_COLOR, h_align="center"),
                _label(0, 47, "Foundation-1 Keybed Export", width=980, height=18, size=12, color=DS_UI_MUTED_TEXT_COLOR, h_align="center"),
            ]
            if custom_instrument_name
            else [
                _label(0, 33, "Foundation-1 Keybed Export", width=980, height=34, size=23, color=DS_UI_TEXT_COLOR, h_align="center"),
            ]
        ),

        # Top row: three module centres at x=160, 490, and 820.
        _centered_section_label(65, 255, 74, "Main", size=18),
    ]

    # MAIN — two uniform 80px controls, centred at x=160.
    lines += _knob(
        x=65, y=102, label="Attack", min_value="0.0", max_value="4.0", value=f"{float(attack):.4f}",
        binding='<binding type="amp" level="instrument" position="0" parameter="ENV_ATTACK" />',
        width=90,
    )
    lines += _knob(
        x=165, y=102, label="Release", min_value="0.0", max_value="8.0", value=f"{float(release):.4f}",
        binding='<binding type="amp" level="instrument" position="0" parameter="ENV_RELEASE" />',
        width=90,
    )


    # DELAY — three uniform controls centred around the Time knob at x=490.
    lines.append(_centered_section_label(345, 635, 74, "Delay", size=18))
    lines += _knob(
        x=345, y=102, label="Delay", min_value="0.0", max_value="1.0", value="0.0",
        binding='<binding type="effect" level="instrument" tags="f1-delay" parameter="FX_WET_LEVEL" translation="linear" />',
        width=90,
    )
    lines += _knob(
        x=445, y=102, label="Time", min_value="0", max_value="20", value="10",
        value_type="musical_time", knob_type="integer",
        binding='<binding type="effect" level="instrument" tags="f1-delay" parameter="FX_DELAY_TIME" />',
        width=90,
    )
    lines += _knob(
        x=545, y=102, label="Feedback", min_value="0.0", max_value="0.95", value="0.20",
        binding='<binding type="effect" level="instrument" tags="f1-delay" parameter="FX_FEEDBACK" translation="linear" />',
        width=90,
    )

    # FILTERS — large logarithmic cutoff controls plus compact resonance/Q
    # controls tucked against the upper outside shoulders of each main dial.
    lines.append(_centered_section_label(688, 952, 74, "Filters", size=18))

    # Draw the main cutoff controls first. The Q controls are emitted afterward
    # so they remain visually on top where their control bounds slightly overlap.
    lines += _knob(
        x=725, y=102, label="Lowpass",
        min_value=f"{DS_FILTER_CONTROL_MIN:g}",
        max_value=f"{DS_FILTER_CONTROL_MAX:g}",
        value=f"{DS_FILTER_CONTROL_MAX:g}",
        binding=(
            '<binding type="effect" level="instrument" tags="f1-lowpass" '
            'parameter="FX_FILTER_FREQUENCY" translation="table" '
            f'translationTable="{filter_translation_table}" />'
        ),
        width=90,
    )
    lines += _knob(
        x=825, y=102, label="Highpass",
        min_value=f"{DS_FILTER_CONTROL_MIN:g}",
        max_value=f"{DS_FILTER_CONTROL_MAX:g}",
        value=f"{DS_FILTER_CONTROL_MIN:g}",
        binding=(
            '<binding type="effect" level="instrument" tags="f1-highpass" '
            'parameter="FX_FILTER_FREQUENCY" translation="table" '
            f'translationTable="{filter_translation_table}" />'
        ),
        width=90,
    )

    lines += _knob(
        x=688, y=108, label="Q",
        min_value=f"{DS_FILTER_Q_CONTROL_MIN:g}",
        max_value=f"{DS_FILTER_Q_CONTROL_MAX:g}",
        value=f"{DS_FILTER_Q_DEFAULT:.3f}",
        parameter_name="Lowpass Q",
        text_size=14,
        binding=(
            '<binding type="effect" level="instrument" tags="f1-lowpass" '
            'parameter="FX_FILTER_RESONANCE" translation="linear" '
            f'translationOutputMin="{DS_FILTER_RESONANCE_MIN:g}" '
            f'translationOutputMax="{DS_FILTER_RESONANCE_MAX:g}" />'
        ),
        width=48,
        height=66,
    )
    lines += _knob(
        x=904, y=108, label="Q",
        min_value=f"{DS_FILTER_Q_CONTROL_MIN:g}",
        max_value=f"{DS_FILTER_Q_CONTROL_MAX:g}",
        value=f"{DS_FILTER_Q_DEFAULT:.3f}",
        parameter_name="Highpass Q",
        text_size=14,
        binding=(
            '<binding type="effect" level="instrument" tags="f1-highpass" '
            'parameter="FX_FILTER_RESONANCE" translation="linear" '
            f'translationOutputMin="{DS_FILTER_RESONANCE_MIN:g}" '
            f'translationOutputMax="{DS_FILTER_RESONANCE_MAX:g}" />'
        ),
        width=48,
        height=66,
    )

    # Bottom row: two module centres at x=245 and x=735.
    lines.append(_centered_section_label(67, 357, 220, "Space", size=18))
    lines += _knob(
        x=67, y=248, label="Reverb", min_value="0.0", max_value="1.0", value="0.0",
        binding='<binding type="effect" level="instrument" tags="f1-reverb" parameter="FX_REVERB_WET_LEVEL" translation="linear" />',
        width=90,
    )
    lines += _knob(
        x=167, y=248, label="Room", min_value="0.0", max_value="1.0", value="0.70",
        binding='<binding type="effect" level="instrument" tags="f1-reverb" parameter="FX_REVERB_ROOM_SIZE" translation="linear" />',
        width=90,
    )
    lines += _knob(
        x=267, y=248, label="Damping", min_value="0.0", max_value="1.0", value="0.30",
        binding='<binding type="effect" level="instrument" tags="f1-reverb" parameter="FX_REVERB_DAMPING" translation="linear" />',
        width=90,
    )

    # COLOR & DISTORTION — all controls now use the same 80px geometry.
    lines.append(_centered_section_label(422, 912, 220, "Color & Distortion", size=18))
    lines += _knob(
        x=422, y=248, label="Chorus", min_value="0.0", max_value="1.0", value="0.0",
        binding='<binding type="effect" level="instrument" tags="f1-chorus" parameter="FX_MIX" translation="linear" />',
        width=90,
    )
    lines += _knob(
        x=522, y=248, label="Phaser", min_value="0.0", max_value="1.0", value="0.0",
        binding='<binding type="effect" level="instrument" tags="f1-phaser" parameter="FX_MIX" translation="linear" />',
        width=90,
    )
    lines += _knob(
        x=622, y=248, label="Bit Mix", min_value="0.0", max_value="1.0", value="0.0",
        binding='<binding type="effect" level="instrument" tags="f1-bitcrush" parameter="FX_MIX" translation="linear" />',
        width=90,
    )
    lines += _knob(
        x=722, y=248, label="Bits", min_value="1", max_value="16", value="8",
        knob_type="integer",
        binding='<binding type="effect" level="instrument" tags="f1-bitcrush" parameter="FX_BIT_DEPTH" />',
        width=90,
    )
    lines += _knob(
        x=822, y=248, label="Rate", min_value="1", max_value="32", value="4",
        knob_type="integer",
        binding='<binding type="effect" level="instrument" tags="f1-bitcrush" parameter="FX_SAMPLE_RATE_REDUCTION" />',
        width=90,
    )

    # Keyboard-edge utility strip. Metadata stays on the lower-left while Glide
    # sits on the lower-right, just above DecentSampler's built-in keyboard.
    # "On" intentionally maps to native legato glide; the "always" mode is not
    # exposed because it is not useful for the Foundation-1 keybed workflow.
    if esc_prompt:
        lines.append(
            _label(
                24, 342, f"Prompt: {prompt_metadata}",
                width=610, height=20, size=11,
                color=DS_UI_MUTED_TEXT_COLOR, h_align="left",
            )
        )
    if esc_id:
        lines.append(
            _label(
                24, 361, f"ID: {instrument_id.upper()}",
                width=300, height=17, size=10,
                color="99FFFFFF", h_align="left",
            )
        )

    # GLIDE — compact utility control opposite Prompt / ID.
    lines.append(
        _label(
            750, 353, "Glide",
            width=48, height=20, size=13,
            color=DS_UI_TEXT_COLOR, h_align="right",
        )
    )
    lines += [
        (
            f'      <menu x="806" y="350" width="78" height="24" value="1" requireSelection="true" '
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
        x=906, y=332, label="Time",
        min_value="0.0", max_value=f"{DS_GLIDE_MAX_TIME:g}", value=f"{DS_GLIDE_DEFAULT_TIME:.3f}",
        parameter_name="Glide Time",
        binding='<binding type="amp" level="instrument" position="0" parameter="GLIDE_TIME" />',
        width=62, height=68, text_size=12,
    )

    lines += [
        '    </tab>',
        '  </ui>',
        f'  <groups attack="{float(attack):.4f}" decay="25" sustain="1.0" release="{float(release):.4f}" ampVelTrack="0.0" glideTime="{DS_GLIDE_DEFAULT_TIME:.3f}" glideMode="{DS_GLIDE_DEFAULT_MODE}">',
        '    <group>',
    ]

    for s in samples:
        sample_path = html.escape(_relpath(Path(s.sample_paths_by_format[FORMAT_DSPRESET]), export_dir))
        lines.append(
            f'      <sample path="{sample_path}" loNote="{s.loNote}" hiNote="{s.hiNote}" rootNote="{s.rootNote}" volume="1.0" />'
        )

    lines.extend([
        '    </group>',
        '  </groups>',
        *effects,
        '</DecentSampler>',
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def _preset_id_from_rows(rows: Sequence[Dict[str, Any]], run_folder: Path) -> str:
    # Use the actual keybed run folder name so the preset, .sfz/.dspreset, and
    # sample folders carry the same compact slug+hash identity.
    folder_slug = _safe_slug(run_folder.name, fallback="foundation_keybed")
    if folder_slug:
        return folder_slug

    for row in rows:
        descriptor = row.get("descriptor")
        if descriptor:
            parts = [p.strip() for p in str(descriptor).split(",") if p.strip()]
            if parts:
                return _safe_slug("_".join(parts[:2]), fallback="foundation_keybed")
    return "foundation_keybed"


def _default_sampler_export_root(run_folder: Path) -> Path:
    """Resolve the top-level Generations/Sampler_Exports folder.

    Normal keybeds usually live at:
        <generations>/Keybed_Source/<preset_id>
    Batch keybeds may live at:
        <generations>/Batch_Generation/Keybed_Source/<preset_id>

    Finished sampler instruments should be easy to find, so they are written to:
        <generations>/Sampler_Exports/<preset_id>/
    """
    run_folder = Path(run_folder)
    parent = run_folder.parent
    if parent.name == "Keybed_Source" and parent.parent.name == "Batch_Generation":
        return parent.parent.parent / "Sampler_Exports"
    if parent.name == "Keybed_Source":
        return parent.parent / "Sampler_Exports"
    if parent.name in {"sampler_export", "Sampler_Exports"}:
        return parent
    return run_folder.parent / "Sampler_Exports"


def export_keybed_run(
    run_folder: str | os.PathLike[str],
    *,
    formats: Sequence[str] = ("sfz", "dspreset"),
    trim_tail: bool = True,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    frame_ms: float = DEFAULT_FRAME_MS,
    tail_pad_ms: float = DEFAULT_TAIL_PAD_MS,
    fade_ms: float = DEFAULT_FADE_MS,
    terminal_fade_ms: float = DEFAULT_TERMINAL_FADE_MS,
    min_keep_ms: float = DEFAULT_MIN_KEEP_MS,
    min_trim_ms: float = DEFAULT_MIN_TRIM_MS,
    attack: float = 0.005,
    release: float = 0.250,
    clean_existing: bool = False,
    keep_raw_samples: bool = False,
    instrument_name: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Export a generated keybed run to separate self-contained sampler folders.

    Finished packages are written beneath ``Sampler_Exports`` as:

      <preset_id>_sfz/
      <preset_id>_decent_sampler/

    When both formats are requested they remain completely separate, while the
    source generation run and its manifest.jsonl stay in ``Keybed_Source``.
    """
    run_folder = Path(run_folder)
    if not run_folder.exists():
        raise FileNotFoundError(f"Keybed run folder does not exist: {run_folder}")

    normalized_formats = _normalize_formats(formats)
    manifest_path = _find_manifest(run_folder)
    rows = _read_jsonl(manifest_path)
    chunk_rows = [(idx, row) for idx, row in enumerate(rows) if _is_exportable_chunk_row(row)]
    if not chunk_rows:
        raise ValueError(f"No exportable chunk rows found in {manifest_path}")

    source_preset_id = _preset_id_from_rows(rows, run_folder)
    instrument_name = _normalize_instrument_name(instrument_name)
    preset_id = _preset_id_with_instrument_name(source_preset_id, instrument_name)
    _export_status(
        f"Building keybed {preset_id}: {len(chunk_rows)} source chunk(s), formats={', '.join(normalized_formats)}.",
        progress_callback,
    )

    export_root = _default_sampler_export_root(run_folder)
    export_root.mkdir(parents=True, exist_ok=True)
    export_dirs_by_format: Dict[str, Path] = {
        FORMAT_SFZ: export_root / f"{preset_id}_sfz",
        FORMAT_DSPRESET: export_root / f"{preset_id}_decent_sampler",
    }
    export_dirs_by_format = {
        fmt: path for fmt, path in export_dirs_by_format.items() if fmt in normalized_formats
    }

    if clean_existing:
        for export_dir in export_dirs_by_format.values():
            if export_dir.exists():
                shutil.rmtree(export_dir)

    raw_dirs_by_format: Dict[str, Path] = (
        {
            fmt: export_dir / f"{preset_id}_raw_scheduled_samples"
            for fmt, export_dir in export_dirs_by_format.items()
        }
        if bool(keep_raw_samples)
        else {}
    )
    sample_dirs_by_format: Dict[str, Path] = {
        FORMAT_SFZ: export_dirs_by_format[FORMAT_SFZ] / f"{preset_id}_sfz_samples"
        for _ in [0] if FORMAT_SFZ in export_dirs_by_format
    }
    if FORMAT_DSPRESET in export_dirs_by_format:
        sample_dirs_by_format[FORMAT_DSPRESET] = (
            export_dirs_by_format[FORMAT_DSPRESET] / f"{preset_id}_dspreset_samples"
        )

    for fmt, export_dir in export_dirs_by_format.items():
        export_dir.mkdir(parents=True, exist_ok=True)
        if fmt in raw_dirs_by_format:
            raw_dirs_by_format[fmt].mkdir(parents=True, exist_ok=True)
        sample_dirs_by_format[fmt].mkdir(parents=True, exist_ok=True)

    exported: List[ExportedSample] = []
    seen_midis: Dict[int, int] = {}

    for chunk_number, (row_index, row) in enumerate(chunk_rows, start=1):
        source_path = _resolve_audio_path(run_folder, str(row.get("audio_path") or ""))
        _export_status(
            f"Slicing source chunk {chunk_number}/{len(chunk_rows)}: {source_path.name}",
            progress_callback,
        )
        audio, sr = torchaudio.load(str(source_path))
        audio = _to_float_audio(audio)
        sr = int(sr)

        notes = _parse_notes(row)
        slices = _parse_note_slices(row, notes)

        for slice_index, note_slice in enumerate(slices):
            note = str(note_slice.get("note") or (notes[slice_index] if slice_index < len(notes) else "")).strip()
            if not note:
                continue
            midi = note_to_midi(note)
            if midi in seen_midis:
                seen_midis[midi] += 1
                continue
            seen_midis[midi] = 1

            scheduled_start = float(note_slice.get("start_sec") or 0.0)
            scheduled_end = float(note_slice.get("end_sec") or (scheduled_start + float(row.get("note_seconds") or 3.0)))
            start_sample = max(0, int(round(scheduled_start * sr)))
            end_sample = min(int(audio.shape[-1]), max(start_sample + 1, int(round(scheduled_end * sr))))
            sliced = audio[:, start_sample:end_sample].contiguous()

            sample_name = f"{note_filename(midi_to_note(midi))}.wav"
            raw_paths: Dict[str, str] = {}
            for fmt, raw_dir in raw_dirs_by_format.items():
                raw_path = raw_dir / sample_name
                _save_wav(raw_path, sliced, sr)
                raw_paths[fmt] = str(raw_path)

            if trim_tail:
                mapped_audio, did_trim, method, duration_sec = trim_tail_conservative(
                    sliced,
                    sr,
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
                duration_sec = sliced.shape[-1] / sr

            terminal_fade_applied = False
            if float(terminal_fade_ms) > 0 and mapped_audio is not None and mapped_audio.shape[-1] > 1:
                mapped_audio = _apply_fade_out(mapped_audio, sr, float(terminal_fade_ms))
                terminal_fade_applied = True

            sample_paths_by_format: Dict[str, str] = {}
            for fmt, sample_dir in sample_dirs_by_format.items():
                mapped_path = sample_dir / sample_name
                _save_wav(mapped_path, mapped_audio, sr)
                sample_paths_by_format[fmt] = str(mapped_path)

            final_source_end = scheduled_start + float(duration_sec)
            first_raw_path = next(iter(raw_paths.values()), "")
            exported.append(ExportedSample(
                note=midi_to_note(midi),
                midi=midi,
                source_chunk_path=str(source_path),
                scheduled_start_sec=round(scheduled_start, 6),
                scheduled_end_sec=round(scheduled_end, 6),
                final_source_start_sec=round(scheduled_start, 6),
                final_source_end_sec=round(final_source_end, 6),
                export_duration_sec=round(float(duration_sec), 6),
                raw_sample_path=first_raw_path,
                sample_paths_by_format=sample_paths_by_format,
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
            ))

    exported.sort(key=lambda sample: sample.midi)
    if not exported:
        raise ValueError("No samples were exported from the keybed manifest.")

    written_files: List[str] = []
    ui_files: List[str] = []
    support_files: List[str] = []
    export_manifests_by_format: Dict[str, str] = {}

    sfz_path: Optional[Path] = None
    dspreset_path: Optional[Path] = None
    background_path: Optional[Path] = None

    if FORMAT_SFZ in normalized_formats:
        sfz_export_dir = export_dirs_by_format[FORMAT_SFZ]
        _export_status("Writing SFZ mapping.", progress_callback)
        sfz_path = sfz_export_dir / f"{preset_id}.sfz"
        _write_sfz(sfz_path, exported, attack=attack, release=release, export_dir=sfz_export_dir)
        written_files.append(str(sfz_path))
        ui_files.append(str(sfz_path))

        sfz_manifest = sfz_export_dir / "export_manifest.json"
        _write_export_manifest(
            sfz_manifest,
            exported,
            sfz_export_dir,
            preset_id=preset_id,
            run_folder=run_folder,
            rows=rows,
            normalized_formats=[FORMAT_SFZ],
            sfz_path=sfz_path,
            dspreset_path=None,
        )
        export_manifests_by_format[FORMAT_SFZ] = str(sfz_manifest)
        support_files.append(str(sfz_manifest))
        written_files.append(str(sfz_manifest))

    if FORMAT_DSPRESET in normalized_formats:
        dspreset_export_dir = export_dirs_by_format[FORMAT_DSPRESET]
        _export_status("Writing DecentSampler preset and UI assets.", progress_callback)
        background_path = _copy_decentsampler_background(dspreset_export_dir)
        dspreset_path = dspreset_export_dir / f"{preset_id}.dspreset"
        display_name, instrument_id = _display_name_and_serial(preset_id)
        _write_dspreset(
            dspreset_path,
            exported,
            attack=attack,
            release=release,
            export_dir=dspreset_export_dir,
            instrument_id=instrument_id,
            prompt_flavour=_prompt_flavour_from_rows(rows),
            instrument_name=instrument_name,
            background_filename=background_path.name if background_path else None,
        )
        written_files.append(str(dspreset_path))
        ui_files.append(str(dspreset_path))
        if background_path:
            written_files.append(str(background_path))
            support_files.append(str(background_path))

        dspreset_manifest = dspreset_export_dir / "export_manifest.json"
        _write_export_manifest(
            dspreset_manifest,
            exported,
            dspreset_export_dir,
            preset_id=preset_id,
            run_folder=run_folder,
            rows=rows,
            normalized_formats=[FORMAT_DSPRESET],
            sfz_path=None,
            dspreset_path=dspreset_path,
        )
        export_manifests_by_format[FORMAT_DSPRESET] = str(dspreset_manifest)
        support_files.append(str(dspreset_manifest))
        written_files.append(str(dspreset_manifest))

    for sample in exported:
        for fmt in normalized_formats:
            path = sample.sample_paths_by_format.get(fmt)
            if path:
                written_files.append(path)

    display_name, instrument_id = _display_name_and_serial(preset_id)
    _export_status(
        f"Keybed Exported - ID {instrument_id} ({preset_id}); {len(exported)} mapped sample(s).",
        progress_callback,
    )

    # Preserve a useful single path for older UI callers. If both formats were
    # requested, opening Sampler_Exports itself is clearer than arbitrarily
    # choosing one of the two sibling packages.
    primary_export_dir = (
        next(iter(export_dirs_by_format.values()))
        if len(export_dirs_by_format) == 1
        else export_root
    )
    first_raw_dir = next(iter(raw_dirs_by_format.values()), None)
    first_manifest = next(iter(export_manifests_by_format.values()), None)

    return {
        "run_folder": str(run_folder),
        "manifest_path": str(manifest_path),
        "export_root": str(export_root),
        "export_dir": str(primary_export_dir),
        "export_dirs_by_format": {fmt: str(path) for fmt, path in export_dirs_by_format.items()},
        "preset_id": preset_id,
        "display_name": display_name,
        "instrument_id": instrument_id,
        "raw_dir": str(first_raw_dir) if first_raw_dir else None,
        "raw_dirs_by_format": {fmt: str(path) for fmt, path in raw_dirs_by_format.items()},
        "sample_dirs_by_format": {fmt: str(path) for fmt, path in sample_dirs_by_format.items()},
        "sfz_sample_dir": str(sample_dirs_by_format.get(FORMAT_SFZ)) if FORMAT_SFZ in sample_dirs_by_format else None,
        "dspreset_sample_dir": str(sample_dirs_by_format.get(FORMAT_DSPRESET)) if FORMAT_DSPRESET in sample_dirs_by_format else None,
        "export_manifest": first_manifest,
        "export_manifests_by_format": export_manifests_by_format,
        "sfz_path": str(sfz_path) if sfz_path else None,
        "dspreset_path": str(dspreset_path) if dspreset_path else None,
        "dspreset_background_path": str(background_path) if background_path else None,
        "dspreset_background_source_dir": str(DECENT_SAMPLER_ASSET_DIR),
        "sample_count": len(exported),
        "notes": [sample.note for sample in exported],
        "files": written_files,
        "ui_files": ui_files,
        "support_files": support_files,
        "tail_trimmed_count": sum(1 for sample in exported if sample.tail_trimmed),
        "keep_raw_samples": bool(keep_raw_samples),
    }
