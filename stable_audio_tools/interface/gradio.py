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
import hffs
import math


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
from .prompts import master_prompt_map

import pretty_midi
import matplotlib.pyplot as plt
import librosa.display
from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH

# Load config file
with open("config.json") as config_file:
    config = json.load(config_file)

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



output_directory = config['generations_directory']

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
        # int4 signatures vary; keep your multi-try approach
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


def runtime_status_md() -> str:
    dev = DEVICE.type if DEVICE is not None else "unknown"
    dtype = "n/a"
    if model is not None:
        try:
            dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        except Exception:
            dtype = "n/a"

    int4_avail = TORCHAO_INT4_SUPPORTED
    int4_state = "✅ on" if INT4_ENABLED else ("available" if int4_avail else "unavailable")

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
            return True, "torchao._C: ✅ compiled extension loaded", ver
        except Exception as e:
            return False, f"torchao._C: ❌ no compiled extension ({e})", ver
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

def amend_prompt(prompt, note, scale, bars, bpm):
    return f"{prompt}, {note} {scale}, {bars} bars, {bpm}BPM,"

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
    librosa.display.specshow(piano_roll, hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))
    plt.colorbar(format='%+2.0f dB')
    plt.title('Piano Roll Visualization')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch')
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
        batch_size=1
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    amended_prompt = amend_prompt(prompt, note, scale, bars, bpm)
    print(f"Prompt: {amended_prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    seconds_start = 0.0

    # --- deterministic grid length (sample-exact) ---
    clip_samples, clip_seconds = clip_samples_from_bars_bpm(bars, bpm, sample_rate)

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

    # ---------- tensor trim (sample-exact) + short fade ----------
    audio = rearrange(audio, "b d n -> d (b n)")  # [ch, n] or [d, n]
    audio = audio.to(torch.float32).clamp(-1, 1)

    # trim to deterministic grid length
    end = min(int(audio.shape[-1]), int(clip_samples))
    audio = audio[:, :max(1, end)].contiguous()

    # tiny fade out to avoid clicks
    fade_ms = 15.0
    fade_len = int(round((fade_ms / 1000.0) * sample_rate))
    if fade_len > 1 and audio.shape[-1] > 1:
        fade_len = min(fade_len, audio.shape[-1])
        ramp = torch.linspace(1.0, 0.0, steps=fade_len, device=audio.device, dtype=audio.dtype)
        audio[:, -fade_len:] *= ramp

    wav_i16 = (audio * 32767.0).to(torch.int16).cpu()

    # Create spectrogram BEFORE returning (spectrogram function expects int16)
    audio_spectrogram = audio_spectrogram_image(wav_i16, sample_rate=sample_rate)

    # ---------- save WAV ----------
    def get_unique_filename(base_name, seed, directory):
        filename = f"{base_name}_{seed}.wav"
        file_path = os.path.join(directory, filename)
        counter = 1
        while os.path.exists(file_path):
            filename = f"{base_name}_{seed}_{counter}.wav"
            file_path = os.path.join(directory, filename)
            counter += 1
        return file_path

    base_name = amended_prompt.replace(" ", "_").replace(",", "").replace(":", "").replace(";", "")
    file_path = get_unique_filename(base_name, seed, output_directory)

    torchaudio.save(file_path, wav_i16, sample_rate)

    # ---------- MIDI conversion ----------
    try:
        convert_audio_to_midi(file_path, output_directory)
        time.sleep(1)

        midi_files = [f for f in os.listdir(output_directory) if f.endswith('.mid') and base_name in f]
        if midi_files:
            midi_files.sort(key=lambda x: os.path.getctime(os.path.join(output_directory, x)))
            midi_output_path = os.path.join(output_directory, midi_files[-1])
            print(f"MIDI file saved successfully as {midi_output_path}.")
        else:
            print("MIDI file was not found. Please check the conversion process.")
            midi_output_path = None

        if midi_output_path is not None:
            midi_data = pretty_midi.PrettyMIDI(midi_output_path)
            print("MIDI file loaded successfully.")
            piano_roll_path = plot_piano_roll(midi_data, 21, 109)
        else:
            piano_roll_path = None

    except Exception as e:
        print(f"An error occurred during MIDI conversion: {e}")
        midi_output_path = None
        piano_roll_path = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    #returning file_path (already trimmed)
    return (file_path, [audio_spectrogram, *preview_images], piano_roll_path, midi_output_path)



def get_models_and_configs(models_path):
    ckpt_files = []
    for root, _, files in os.walk(models_path):
        for file in files:
            if file.endswith((".ckpt", ".safetensors")):
                ckpt_files.append((file, os.path.join(root, file)))
    return ckpt_files


def get_config_files(ckpt_path):
    config_files = []
    folder = os.path.dirname(ckpt_path)
    print(f"Looking for config files in folder: {folder}")  # Debugging output
    for file in os.listdir(folder):
        if file.endswith(".json"):
            config_files.append(file)
    print(f"Found config files: {config_files}")  # Debugging output
    return config_files

def update_config_dropdown(selected_ckpt, ckpt_files):
    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)
        configs = get_config_files(ckpt_path)
        return gr.update(choices=configs, value=configs[0] if configs else "Select Config")
    except Exception as e:
        print(f"Error updating config dropdown: {e}")  # Debugging output
        return gr.update(choices=["Error finding configs"], value="Error finding configs")

def load_model_action(selected_ckpt, selected_config, ckpt_files, int4_requested: bool):
    global DEVICE, current_prompt_generator
    global LAST_CKPT_PATH, LAST_CONFIG_PATH, LAST_CKPT_NAME, LAST_MODEL_CONFIG, INT4_ENABLED
    global model 
    
    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)
        config_path = os.path.join(os.path.dirname(ckpt_path), selected_config)

        with open(config_path, "r") as f:
            cfg = json.load(f)

        # remember what we loaded
        LAST_CKPT_PATH = ckpt_path
        LAST_CONFIG_PATH = config_path
        LAST_CKPT_NAME = selected_ckpt
        LAST_MODEL_CONFIG = cfg

        # load base model (non-quant)
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

        current_prompt_generator = master_prompt_map.get_prompt_generator(selected_ckpt)

        info = f"Loaded model {selected_ckpt} with config {selected_config}"
        return info, runtime_status_md()

    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error loading model: {e}", runtime_status_md()



def create_sampling_ui(model_config, initial_ckpt, inpainting=False):
    ckpt_files = get_models_and_configs(config['models_directory'])
    selected_ckpt = gr.State(value=os.path.basename(initial_ckpt))
    selected_config = gr.State()

    with gr.Row():
        with gr.Column(scale=8):  # Input fields take more space
            prompt = gr.Textbox(show_label=False, placeholder="Prompt")
            negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
        with gr.Column(scale=2):  # Buttons take less space
            with gr.Column():
                generate_button = gr.Button("Generate", variant='primary', scale=1)
                random_prompt_button = gr.Button("Random Prompt", variant='secondary', scale=1)

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
                model_dropdown = gr.Dropdown(["Select Model"] + [file[0] for file in ckpt_files], label="Select Model")
                config_dropdown = gr.Dropdown(["Select Config"], label="Select Config")

            model_dropdown.change(
                fn=lambda x: update_config_dropdown(x, ckpt_files),
                inputs=model_dropdown,
                outputs=config_dropdown
            )

            # Status stays in the "model" area so users always see device/dtype/int4 state
            status_md = gr.Markdown(runtime_status_md())

            load_model_button = gr.Button("Load Model")

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

    # Define inputs list after UI elements exist
    if inpainting:
        inputs = [
            prompt,
            negative_prompt,
            bars_dropdown,
            bpm_dropdown,
            note_dropdown,
            scale_dropdown,
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
            mask_marination_slider
        ]
    else:
        inputs = [
            prompt,
            negative_prompt,
            bars_dropdown,
            bpm_dropdown,
            note_dropdown,
            scale_dropdown,
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
            init_noise_level_slider
        ]

    generate_button.click(
        fn=generate_cond,
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

    load_model_button.click(
        fn=lambda x, y, q: load_model_action(x, y, ckpt_files, q),
        inputs=[model_dropdown, config_dropdown, int4_for_load],
        outputs=[current_model_info, status_md]
    )

    def update_prompt(prompt, lock_bpm, bars, bpm, lock_key, note, scale):
        new_prompt = current_prompt_generator()

        if not lock_bpm:
            bars = random.choice([4, 8])
            bpm = random.choice([100, 110, 120, 128, 130, 140, 150])

        if not lock_key:
            note = random.choice(["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
            scale = random.choice(["major", "minor"])

        return new_prompt, bars, bpm, note, scale

    random_prompt_button.click(
        fn=update_prompt,
        inputs=[
            prompt,
            lock_bpm_checkbox,
            bars_dropdown,
            bpm_dropdown,
            lock_key_checkbox,
            note_dropdown,
            scale_dropdown
        ],
        outputs=[
            prompt,
            bars_dropdown,
            bpm_dropdown,
            note_dropdown,
            scale_dropdown
        ]
    )



def create_txt2audio_ui(model_config, initial_ckpt):
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config, initial_ckpt)
        # with gr.Tab("Inpainting"): removing for now no inpainting yet.
        #     create_sampling_ui(model_config, initial_ckpt, inpainting=True)
        with gr.Tab("Download Models"):
            gr.HTML("<h2>Download</h2><div>Download a model and restart the app to apply.</div>")
            hffs.from_config(config)
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
                gr.HTML("<h2>Initialize</h2><div>Download a model first, and restart the app.</div>")
                hffs.from_config(config)
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

