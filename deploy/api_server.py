from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from einops import rearrange
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from stable_audio_tools.interface.gradio import load_model, pick_preferred_dtype
from stable_audio_tools.inference.generation import generate_diffusion_cond


MODEL = None
MODEL_CONFIG = None
DEVICE = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Foundation-1 prompt text.")
    seconds_total: int = Field(default=8, ge=1, le=30)
    steps: int = Field(default=12, ge=1, le=250)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=25.0)
    seed: int = Field(default=42)
    sampler_type: str = Field(default="dpmpp-2m-sde")
    sigma_min: float = Field(default=0.3, gt=0.0)
    sigma_max: float = Field(default=50.0, gt=0.0)
    rho: float = Field(default=1.0, gt=0.0)


def pick_device() -> torch.device:
    forced = os.environ.get("FOUNDATION1_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def model_paths() -> tuple[Path, Path]:
    model_dir = Path(os.environ.get("FOUNDATION1_MODEL_DIR", "models/Foundation-1"))
    config_path = model_dir / "model_config.json"
    ckpt_path = model_dir / "Foundation_1.safetensors"
    return config_path, ckpt_path


def load_runtime() -> None:
    global MODEL, MODEL_CONFIG, DEVICE

    config_path, ckpt_path = model_paths()
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Expected {config_path} and {ckpt_path}. Download the model before starting the API."
        )

    with config_path.open() as f:
        MODEL_CONFIG = json.load(f)

    DEVICE = pick_device()
    dtype = pick_preferred_dtype(DEVICE)
    MODEL, _ = load_model(
        model_config=MODEL_CONFIG,
        model_ckpt_path=str(ckpt_path),
        device=DEVICE,
        preferred_dtype=dtype,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime()
    yield


app = FastAPI(title="Foundation-1 API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if MODEL is None or MODEL_CONFIG is None or DEVICE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "ok",
        "device": DEVICE.type,
        "sample_rate": str(MODEL_CONFIG["sample_rate"]),
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> Response:
    if MODEL is None or MODEL_CONFIG is None or DEVICE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    conditioning = [
        {
            "prompt": req.prompt,
            "seconds_start": 0,
            "seconds_total": req.seconds_total,
        }
    ]

    audio = generate_diffusion_cond(
        model=MODEL,
        conditioning=conditioning,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        batch_size=1,
        sample_size=MODEL_CONFIG["sample_size"],
        seed=req.seed,
        device=DEVICE,
        sampler_type=req.sampler_type,
        sigma_min=req.sigma_min,
        sigma_max=req.sigma_max,
        rho=req.rho,
    )

    audio = audio[:, :, : MODEL_CONFIG["sample_rate"] * req.seconds_total]
    audio = rearrange(audio, "b d n -> (b n) d")
    audio = audio.to(torch.float32)
    audio = audio.div(torch.max(torch.abs(audio))).clamp(-1, 1).cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, audio, MODEL_CONFIG["sample_rate"], format="WAV")
    headers = {
        "X-Foundation1-Device": DEVICE.type,
        "X-Foundation1-Sample-Rate": str(MODEL_CONFIG["sample_rate"]),
        "X-Foundation1-Seconds": str(req.seconds_total),
    }
    return Response(content=buffer.getvalue(), media_type="audio/wav", headers=headers)
