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

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T
from pydub import AudioSegment

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

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

output_directory = config['generations_directory']

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device=None):
    global model, sample_rate, sample_size, global_model_half
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)
        
        # Load checkpoint
        state_dict = load_ckpt_state_dict(model_ckpt_path)
        
        # Check if the model is in float16 format before loading into the model
        is_float16 = all(param.dtype == torch.float16 for param in state_dict.values())
        
        if is_float16:
            print("Model is in float16 format. Enabling half-precision inference.")
            global_model_half = True
            model.to(torch.float16)  # Convert the model to half precision before loading state dict
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
        is_float16_pretransform = all(param.dtype == torch.float16 for param in pretransform_state_dict.values())
        
        if is_float16_pretransform:
            print("Model is in float16 format. Enabling half-precision inference.")
            model.pretransform.to(torch.float16)  # Convert the pretransform model to half precision before loading state dict
        else:
            print("Model is in full precision format.")
        
        model.pretransform.load_state_dict(pretransform_state_dict, strict=False)
        #print(f"Done loading pretransform")
    
    # Move the model to the specified device
    model.to(device).eval().requires_grad_(False)
    
    print(f"Done loading model")
    return model, model_config

def calculate_seconds_total(bars, bpm):
    bar_duration = 60 / bpm * 4
    return bar_duration * bars

def amend_prompt(prompt, bars, bpm):
    return f"{prompt}, {bars} bars, {bpm}BPM,"

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

    amended_prompt = amend_prompt(prompt, bars, bpm)
    print(f"Prompt: {amended_prompt}")

    global preview_images
    preview_images = []
    if preview_every == 0:
        preview_every = None

    seconds_start = 0
    seconds_total = calculate_seconds_total(bars, bpm)
    trim_end = seconds_total + 0.1  # Adding 100ms buffer

    # Return fake stereo audio
    conditioning = [{"prompt": amended_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size

    if negative_prompt:
        negative_conditioning = [{"prompt": negative_prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}] * batch_size
    else:
        negative_conditioning = None

    # Get the device from the model
    device = next(model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None

    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        
        # Turn into torch tensor, converting to float32 # Non-16bit audio support added from PR #157 by lyramakesmusic
        if init_audio.dtype == np.float32:
            init_audio = torch.from_numpy(init_audio)
        elif init_audio.dtype == np.int16:
            init_audio = torch.from_numpy(init_audio).float().div(32767)
        elif init_audio.dtype == np.int32:
            init_audio = torch.from_numpy(init_audio).float().div(2147483647)
        else:
            raise ValueError(f"Unsupported audio data type: {init_audio.dtype}")

        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        global preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    # If inpainting, send mask args
    # This will definitely change in the future
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

    # Do the audio generation
    audio = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        steps=steps,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sample_size=input_sample_size,
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

    # Convert to WAV file
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    # Create spectrogram before saving the WAV file
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)

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
    
    torchaudio.save(file_path, audio, sample_rate)

    # Trim the audio file using pydub
    audio_segment = AudioSegment.from_wav(file_path)
    trimmed_audio_segment = audio_segment[:trim_end * 1000]  # trim_end is in seconds, converting to milliseconds
    trimmed_file_path = file_path.replace(".wav", "_trimmed.wav")
    trimmed_audio_segment.export(trimmed_file_path, format="wav")

    # Convert audio to MIDI
    try:
        convert_audio_to_midi(trimmed_file_path, output_directory)
        time.sleep(1)  # Wait to ensure the MIDI file is saved

        midi_files = [f for f in os.listdir(output_directory) if f.endswith('.mid') and base_name in f]
        if midi_files:
            midi_files.sort(key=lambda x: os.path.getctime(os.path.join(output_directory, x)))
            midi_output_path = os.path.join(output_directory, midi_files[-1])
            print(f"MIDI file saved successfully as {midi_output_path}.")
        else:
            print("MIDI file was not found. Please check the conversion process.")

        # Load the generated MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_output_path)
        print("MIDI file loaded successfully.")
        piano_roll_path = plot_piano_roll(midi_data, 21, 109)
    except Exception as e:
        print(f"An error occurred during MIDI conversion: {e}")
        midi_output_path = None
        piano_roll_path = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return (trimmed_file_path, [audio_spectrogram, *preview_images], piano_roll_path, midi_output_path)



def get_models_and_configs(models_path):
    ckpt_files = []
    for root, _, files in os.walk(models_path):
        for file in files:
            if file.endswith(".ckpt"):
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

def load_model_action(selected_ckpt, selected_config, ckpt_files):
    global DEVICE
    try:
        ckpt_path = next(path for name, path in ckpt_files if name == selected_ckpt)
        config_path = os.path.join(os.path.dirname(ckpt_path), selected_config)
        
        model, model_config = load_model(
            model_config=json.load(open(config_path)),
            model_ckpt_path=ckpt_path,
            device=DEVICE
        )
        
        return f"Loaded model {selected_ckpt} with config {selected_config}"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error loading model: {e}"
    
def generate_random_filename():
    piano_types = ["Soft E. Piano", "Medium E. Piano", "Grand Piano"]
    tremolo_effects = ["Low Tremolo", "Medium Tremolo", "High Tremolo", "No Tremolo"]
    non_tremolo_effects = ["No Reverb", "Low Reverb", "Medium Reverb", "High Reverb", "High Spacey Reverb"]

    chord_progressions = ["simple", "complex", "dance plucky", "fast", "jazzy", "low", "simple strummed", "rising strummed", "complex strummed", "jazzy strummed", "slow strummed", "plucky dance",
                          "rising", "falling", "slow", "slow jazzy", "fast jazzy", "smooth", "strummed", "plucky"]
    melodies = [
        "catchy melody", "complex melody", "complex top melody", "catchy top melody", "top melody", "smooth melody", "catchy complex melody",
        "jazzy melody", "smooth catchy melody", "plucky dance melody", "dance melody", "alternating low melody", "alternating top arp melody", "alternating top melody", "top arp melody", "alternating melody", "falling arp melody",
        "rising arp melody", "top catchy melody"
    ]
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    scales = ["major,", "minor,"]

    # Choose the piano type first to ensure an even split
    piano = random.choice(piano_types)

    # Choose effect based on piano type
    if piano == "Grand Piano":
        effect = random.choice(non_tremolo_effects)
    else:
        effect = random.choice(tremolo_effects + non_tremolo_effects)

    note = random.choice(notes)
    scale = random.choice(scales)

    # Decide category for generation
    category_choice = random.choice(["chord progression only", "chord progression with melody", "melody only"])
    
    if category_choice == "chord progression only":
        chord_progression = random.choice(chord_progressions) + " chord progression only,"
        descriptor = f"{piano}, {chord_progression} {note} {scale} {effect}"
    elif category_choice == "chord progression with melody":
        chord_progression = random.choice(chord_progressions) + " chord progression,"
        melody = "with " + random.choice(melodies) + ","
        descriptor = f"{piano}, {chord_progression} {melody} {note} {scale} {effect}"
    else:
        melody = random.choice(melodies) + " only,"
        descriptor = f"{piano}, {melody} {note} {scale} {effect}"

    return descriptor

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
            
            # comment out the model and config dropdowns / load model button for demos
            with gr.Row():
                # Model and Config dropdowns
                model_dropdown = gr.Dropdown(["Select Model"] + [file[0] for file in ckpt_files], label="Select Model")
                config_dropdown = gr.Dropdown(["Select Config"], label="Select Config")
            
            load_model_button = gr.Button("Load Model")

            model_dropdown.change(fn=lambda x: update_config_dropdown(x, ckpt_files), inputs=model_dropdown, outputs=config_dropdown)

            lock_bpm_checkbox = gr.Checkbox(label="Lock BPM Settings", value=True)
            with gr.Row(visible=has_seconds_start or has_seconds_total):
                bars_dropdown = gr.Dropdown([4, 8], label="Bars", value=8, visible=has_seconds_total)  # Set default value here
                bpm_dropdown = gr.Dropdown([100, 110, 120, 128, 130, 140, 145, 150], label="BPM", value=128, visible=has_seconds_total)  # Set default value here
                
            with gr.Row():
                # Steps slider
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")

                # Preview Every slider
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")

                # CFG scale
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG scale")

            with gr.Accordion("Sampler params", open=False):
                # Seed
                seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")

                # Sampler params
                with gr.Row():
                    sampler_type_dropdown = gr.Dropdown(["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"], label="Sampler type", value="dpmpp-3m-sde")
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG rescale amount")

            if inpainting:
                # Inpainting Tab
                with gr.Accordion("Inpainting", open=False):
                    sigma_max_slider.maximum = 1000

                    init_audio_checkbox = gr.Checkbox(label="Do inpainting")
                    init_audio_input = gr.Audio(label="Init audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.1, value=80, label="Init audio noise level", visible=False)  # hide this

                    mask_cropfrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Crop From %")
                    mask_pastefrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Paste From %")
                    mask_pasteto_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Paste To %")

                    mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50, label="Mask Start %")
                    mask_maskend_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Mask End %")
                    mask_softnessL_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Left Crossfade Length %")
                    mask_softnessR_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Right Crossfade Length %")
                    mask_marination_slider = gr.Slider(minimum=0.0, maximum=1, step=0.0001, value=0, label="Marination level", visible=False)  # still working on the usefulness of this

                    inputs = [prompt,
                        negative_prompt,
                        bars_dropdown,
                        bpm_dropdown,
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
                # Default generation tab
                with gr.Accordion("Init audio - Full Precision Models Only)", open=False):
                    init_audio_checkbox = gr.Checkbox(label="Use init audio")
                    init_audio_input = gr.Audio(label="Init audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init noise level")

                    inputs = [prompt,
                        negative_prompt,
                        bars_dropdown,
                        bpm_dropdown,
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

        with gr.Column():
            audio_output = gr.Audio(label="Output audio", interactive=False)
            audio_spectrogram_output = gr.Gallery(label="Output spectrogram", show_label=False)
            send_to_init_button = gr.Button("Send to init audio", scale=1)

        with gr.Column():
            midi_piano_roll_output = gr.Image(label="MIDI Piano Roll", interactive=False)
            midi_download_button = gr.File(label="Download MIDI", file_count="single", type="filepath", interactive=False)

    generate_button.click(fn=generate_cond,
        inputs=inputs,
        outputs=[
            audio_output,
            audio_spectrogram_output,
            midi_piano_roll_output,
            midi_download_button
        ],
        api_name="generate")

    send_to_init_button.click(fn=lambda audio: audio, inputs=[audio_output], outputs=[init_audio_input])

    # Comment out the load model button click event
    load_model_button.click(fn=lambda x, y: load_model_action(x, y, ckpt_files), inputs=[model_dropdown, config_dropdown], outputs=[current_model_info])

    def update_prompt(prompt, lock_bpm, bars, bpm):
        new_prompt = generate_random_filename()
        # Preserve the original bars and bpm in the prompt only if lock_bpm is True
        if lock_bpm:
            return new_prompt, bars, bpm
        else:
            # Randomize bars and bpm if not locked
            bars = random.choice([4, 8])
            bpm = random.choice([100, 110, 120, 128, 130, 140, 150])
            return new_prompt, bars, bpm

    random_prompt_button.click(
        fn=update_prompt,
        inputs=[prompt, lock_bpm_checkbox, bars_dropdown, bpm_dropdown],
        outputs=[prompt, bars_dropdown, bpm_dropdown]
    )


def create_txt2audio_ui(model_config, initial_ckpt):
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_config, initial_ckpt)
        with gr.Tab("Inpainting"):
            create_sampling_ui(model_config, initial_ckpt, inpainting=True)
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

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global global_model_half
    global_model_half = model_half  # Initialize model_half with the provided value

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
    assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_config = json.load(f)
    else:
        model_config = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    global DEVICE
    DEVICE = device

    initial_ckpt = ckpt_path if ckpt_path is not None else pretrained_name

    _, model_config = load_model(model_config, ckpt_path, pretrained_name=pretrained_name, pretransform_ckpt_path=pretransform_ckpt_path, device=device)

    model_type = model_config["model_type"]

    if model_type == "diffusion_cond":
        ui = create_txt2audio_ui(model_config, initial_ckpt)
    elif model_type == "diffusion_uncond":
        ui = create_diffusion_uncond_ui(model_config)
    elif model_type == "autoencoder" or model_type == "diffusion_autoencoder":
        ui = create_autoencoder_ui(model_config)
    elif model_type == "diffusion_prior":
        ui = create_diffusion_prior_ui(model_config)
    elif model_type == "lm":
        ui = create_lm_ui(model_config)

    return ui
