# RC Stable Audio Tools

RC Stable Audio Tools is RoyalCities' fork of Stability AI's `stable-audio-tools` project. It adds a friendlier Gradio workflow for model discovery, dynamic model loading, structured music prompting, automatic trimming, and optional sample-to-MIDI conversion.

If you are coming here for Foundation-1 specifically, this repo is a practical local runner for models like:

- `RoyalCities/Foundation-1`
- `RoyalCities/RC_Infinite_Pianos`
- `RoyalCities/Vocal_Textures_Main`
- `adlb/Audialab_EDM_Elements`

It can be used in three ways:

1. As a local Gradio UI for prompting and auditioning models.
2. As a headless local generator from the command line.
3. As the basis for an HTTP inference service you can deploy to a GPU provider.

## What This Fork Adds

Compared with upstream `stable-audio-tools`, this fork focuses on usability:

- Dynamic model loading from local checkpoints.
- A Hugging Face downloader flow when `models/` is empty.
- Prompt helpers tailored to music production workflows.
- BPM, bar-count, and key controls wired into prompt conditioning.
- Random prompt generation based on model metadata.
- Automatic sample trimming to the requested musical grid length.
- Automatic sample-to-MIDI conversion with Basic Pitch.
- Apple Silicon support in the verified local setup path.

Examples from the current UI:

- Dynamic model loading

  <p align="center">
    <img src="https://i.imgur.com/kB8CQ3J.gif" alt="Dynamic model loading" width="50%">
  </p>

- Random prompt generation

  <p align="center">
    <img src="https://i.imgur.com/fNEE8cR.gif" alt="Random prompt generation" width="95%">
  </p>

- BPM and bar-aware prompting

  <p align="center">
    <img src="https://i.imgur.com/hcedPl5.png" alt="BPM and bar controls" width="50%">
  </p>

- Key locking

  <p align="center">
    <img src="https://i.imgur.com/7IXXDSZ.jpeg" alt="Key locking" width="50%">
  </p>

- MIDI conversion

  <p align="center">
    <img src="https://i.imgur.com/R9ipGiq.gif" alt="MIDI conversion" width="50%">
  </p>

- Automatic trimming

  <p align="center">
    <img src="https://i.imgur.com/ApH5SOM.gif" alt="Automatic trimming" width="75%">
  </p>

## How It Works

At a high level, the generation pipeline is:

1. Pick a model checkpoint and matching `model_config.json`.
2. Convert your musical intent into conditioning inputs.
   - prompt text
   - bar count
   - BPM
   - key / scale
   - optional negative prompt
3. Run diffusion sampling for a chosen number of steps.
4. Decode latent audio back into waveform audio.
5. Trim to the requested musical duration.
6. Save a WAV file.
7. Optionally run Basic Pitch to create a MIDI file and preview piano roll.

The most important files for that flow are:

- `run_gradio.py`: launches the web UI.
- `stable_audio_tools/interface/gradio.py`: model loading, prompting, generation, and post-processing.
- `scripts/generate_foundation1.py`: simple headless generation example.
- `deploy/api_server.py`: minimal HTTP inference server for hosted deployments.
- `config.json`: model locations, output locations, and pre-listed Hugging Face repos.

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/RoyalCities/RC-stable-audio-tools.git
cd RC-stable-audio-tools
```

### 2. Use Python 3.10

This project still depends on older pinned packages. Python `3.10` is the safest choice.

### 3. Install and download a model

### macOS / Apple Silicon

This is the most verified path in this repo right now:

```bash
chmod +x scripts/setup_foundation1_macos.sh
./scripts/setup_foundation1_macos.sh
```

That script:

- creates `.venv`
- installs PyTorch
- installs this repo
- applies the compatibility fixes needed on macOS
- downloads `RoyalCities/Foundation-1`

More detail lives in [LOCAL_MAC_SETUP.md](LOCAL_MAC_SETUP.md).

### Windows / Linux

The original workflow still applies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install .
```

If you are on NVIDIA hardware, install the matching CUDA build of PyTorch for your system. On Windows and Linux, optional TorchAO INT4 inference is also supported when the environment is compatible.

## Running The UI

With the environment active:

```bash
source .venv/bin/activate
python run_gradio.py
```

The first launch behaves like this:

1. If `models/` is empty, you get a Hugging Face downloader UI.
2. Download a model such as `RoyalCities/Foundation-1`.
3. Restart `python run_gradio.py`.
4. The full generation UI appears.

If you already have a local checkpoint and config, you can also pass them directly:

```bash
python run_gradio.py \
  --model-config models/Foundation-1/model_config.json \
  --ckpt-path models/Foundation-1/Foundation_1.safetensors
```

## Using The UI Well

The UI is built around musical structure, not just abstract text prompts.

A strong prompt usually combines:

- source or family
- timbre
- musical behavior
- bars
- BPM
- key / scale

Example:

```text
Bass, Wavetable Bass, Acid, Gritty, 4 Bars, 140 BPM, F minor
```

Practical notes:

- Low step counts are good for smoke tests.
- `50-100` steps is a more realistic interactive range.
- Higher step counts improve quality at the cost of latency.
- Different models respond differently to prompt density, so start simple and add detail gradually.

## Headless Usage

If you want a reproducible local command instead of the UI:

```bash
source .venv/bin/activate
python scripts/generate_foundation1.py \
  --prompt "Bass, Wavetable Bass, Acid, Gritty, 4 Bars, 140 BPM, F minor" \
  --steps 12 \
  --seconds-total 8 \
  --output generations/bass-test.wav
```

This is the simplest way to understand the core inference flow without thinking about the Gradio layer.

## Deploying As An API

This repo now includes a small FastAPI server:

```bash
source .venv/bin/activate
uvicorn deploy.api_server:app --host 0.0.0.0 --port 8000
```

It exposes:

- `GET /health`
- `POST /generate`

That is the shape most hosted model deployments take: load the model once, keep it warm on a GPU, and send prompt requests to an HTTP endpoint.

More detail and provider notes live in [DEPLOYMENT_NOTES.md](DEPLOYMENT_NOTES.md).

## Which Hosted Option Fits What

If you are new to model deployment, the terms you are looking for are usually:

- inference endpoint
- model hosting
- inference server

The practical options are:

- Hugging Face Spaces
  - best for a public demo UI
- Hugging Face Inference Endpoints
  - best for a managed hosted API
- Modal
  - best for a Python-first deployment workflow
- RunPod
  - best when you want more direct control over GPU infrastructure

## Model Layout

Models should live in their own subdirectory inside `models/`, alongside the matching config file.

For example:

```text
models/
  Foundation-1/
    Foundation_1.safetensors
    model_config.json
```

This is what lets the UI discover checkpoints and offer dynamic model switching.

## Repository Layout

Useful entry points:

- `run_gradio.py`
- `config.json`
- `models/`
- `generations/`
- `scripts/setup_foundation1_macos.sh`
- `scripts/generate_foundation1.py`
- `deploy/api_server.py`
- `LOCAL_MAC_SETUP.md`
- `DEPLOYMENT_NOTES.md`

## Known Notes

- Python `3.10` is recommended.
- On current PyTorch builds, `torchaudio.save()` may expect `torchcodec`.
- This repo's verified local path uses `soundfile` for output writing instead.
- You will still see some dependency warnings from older transitive packages. They are noisy, but they do not block generation in the verified path.

## Credits

This project stands on a lot of upstream work:

- Stability AI for the original [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools)
- RoyalCities for this fork and the Foundation-1 ecosystem
- [@cocktailpeanut](https://github.com/cocktailpeanut) for Apple Silicon support help
- Hugging Face for model distribution and hosting infrastructure
- Descript's [Basic Pitch](https://github.com/spotify/basic-pitch) for the MIDI conversion workflow
- PyTorch, torchaudio, Gradio, k-diffusion, x-transformers, and the rest of the open-source stack this repo builds on

If you use a model from Hugging Face, check that model's own card and license as well. For example, `RoyalCities/Foundation-1` is based on `stabilityai/stable-audio-open-1.0` and carries its own model-card and license context.

## Acknowledgement

This repo is maintained as a practical, creator-friendly layer on top of a powerful but fairly technical audio generation stack. The goal is simple: make it easier to load a model, prompt it musically, generate something useful, and move quickly.
