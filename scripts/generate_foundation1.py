#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from einops import rearrange

from stable_audio_tools.interface.gradio import load_model, pick_preferred_dtype
from stable_audio_tools.inference.generation import generate_diffusion_cond


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short Foundation-1 sample locally.")
    parser.add_argument(
        "--model-dir",
        default="models/Foundation-1",
        help="Directory containing Foundation_1.safetensors and model_config.json.",
    )
    parser.add_argument(
        "--prompt",
        default="Synth Pad, Warm, Airy, Chord Progression, 4 Bars, 120 BPM, D minor",
        help="Prompt sent to the model.",
    )
    parser.add_argument(
        "--seconds-total",
        type=int,
        default=8,
        help="Conditioning length in seconds.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Sampling steps. Low values are good for smoke tests; higher values improve quality.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance strength.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for repeatable output.",
    )
    parser.add_argument(
        "--output",
        default="generations/foundation1-smoketest.wav",
        help="Where to write the generated WAV.",
    )
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    config_path = model_dir / "model_config.json"
    ckpt_path = model_dir / "Foundation_1.safetensors"

    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Expected both {config_path} and {ckpt_path}. Run scripts/setup_foundation1_macos.sh first."
        )

    with config_path.open() as f:
        cfg = json.load(f)

    device = pick_device()
    dtype = pick_preferred_dtype(device)
    print(f"Using device={device} dtype={dtype}")

    model, _ = load_model(
        model_config=cfg,
        model_ckpt_path=str(ckpt_path),
        device=device,
        preferred_dtype=dtype,
    )

    conditioning = [
        {
            "prompt": args.prompt,
            "seconds_start": 0,
            "seconds_total": args.seconds_total,
        }
    ]

    audio = generate_diffusion_cond(
        model=model,
        conditioning=conditioning,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        batch_size=1,
        sample_size=cfg["sample_size"],
        seed=args.seed,
        device=device,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.3,
        sigma_max=50,
        rho=1.0,
    )

    audio = audio[:, :, : cfg["sample_rate"] * args.seconds_total]
    audio = rearrange(audio, "b d n -> (b n) d")
    audio = audio.to(torch.float32)
    audio = audio.div(torch.max(torch.abs(audio))).clamp(-1, 1).cpu().numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, cfg["sample_rate"])
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
