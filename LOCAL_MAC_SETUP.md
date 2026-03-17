# Foundation-1 On macOS

This repo now has a verified local path for Apple Silicon Macs.

What was tested:

- Machine: Apple Silicon MacBook Pro
- Runtime: `python3.10`
- Backend: PyTorch `mps`
- Model: `RoyalCities/Foundation-1`
- Proof: local Gradio app booted and a headless smoke-test WAV was generated

## Quick Start

From the repo root:

```bash
chmod +x scripts/setup_foundation1_macos.sh
./scripts/setup_foundation1_macos.sh
```

Then either:

```bash
source .venv/bin/activate
python run_gradio.py
```

Or generate a short sample from the terminal:

```bash
source .venv/bin/activate
python scripts/generate_foundation1.py
```

The smoke-test output goes to `generations/foundation1-smoketest.wav`.

## Why The Setup Looks A Bit Weird

The upstream README is mostly right, but two macOS compatibility fixes were needed during verification:

- `setuptools<81`: older `librosa` still imports `pkg_resources`
- `soundfile>=0.13,<0.14`: the pinned `SoundFile 0.10.2` wheel did not load `libsndfile` correctly on this machine

Also note:

- `torchaudio.save()` in the current PyTorch stack expects `torchcodec`
- the headless generator script uses `soundfile.write()` instead, which avoids that extra dependency

## Learning Map

If you're new to ML tooling, the moving parts here are:

1. Model weights
   `Foundation_1.safetensors` is the learned parameter file.

2. Model config
   `model_config.json` describes the architecture and conditioning schema.

3. Runtime
   PyTorch loads the graph and executes it on `mps`, `cuda`, or `cpu`.

4. Sampler
   Diffusion models do not emit audio in one pass. They iteratively denoise random noise over multiple steps.

5. Conditioning
   Foundation-1 uses prompt text plus timing metadata like `seconds_total`.

## Quality vs Speed

- `steps=8` is only a smoke test
- `steps=50-100` is a more realistic interactive range
- higher step counts usually improve quality but increase latency

## Recommended First Commands

Open the UI:

```bash
source .venv/bin/activate
python run_gradio.py
```

Run a custom prompt:

```bash
source .venv/bin/activate
python scripts/generate_foundation1.py \
  --prompt "Bass, Wavetable Bass, Acid, Gritty, 4 Bars, 140 BPM, F minor" \
  --steps 12 \
  --seconds-total 8 \
  --output generations/bass-test.wav
```
