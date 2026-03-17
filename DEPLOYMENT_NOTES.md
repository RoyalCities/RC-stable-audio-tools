# Foundation-1 Deployment Notes

If you're new to ML infrastructure, the phrase you usually want is one of these:

- model hosting
- inference server
- inference endpoint
- GPU inference deployment

For this model, the simplest mental model is:

1. Load the model once onto a GPU.
2. Expose an HTTP API.
3. Send prompts to that API.
4. Return generated audio.

The included API server lives at `deploy/api_server.py` and exposes:

- `GET /health`
- `POST /generate`

Example local run:

```bash
source .venv/bin/activate
uvicorn deploy.api_server:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Bass, Wavetable Bass, Acid, 4 Bars, 140 BPM, F minor",
    "seconds_total": 8,
    "steps": 12,
    "cfg_scale": 7.0,
    "seed": 42
  }' \
  --output foundation1-api.wav
```

## Which Deployment Option Fits What

### 1. Hugging Face Spaces

Best when you want a public demo UI.

- Spaces can host Gradio apps and Docker apps.
- Spaces can be upgraded to GPU hardware.
- Good for sharing, testing prompts, and showing the model to people.
- Not my first choice for a production API.

### 2. Hugging Face Inference Endpoints

Best when you want a managed API and are okay paying for a hosted endpoint.

- Hugging Face supports custom Docker containers for models with custom inference logic.
- Their custom container guide explicitly recommends exposing `/health` and `/generate`.
- The selected model repository is mounted at `/repository` inside the container.

This is the cleanest managed option if you want Hugging Face to own most of the hosting layer.

### 3. Modal

Best when you want a Python-first deployment workflow.

- Modal lets you deploy Python functions and web endpoints.
- You choose a GPU in code, for example `gpu="A10G"` or `gpu="L40S"`.
- Good ergonomics for developers and fast iteration.

This is usually the easiest "real deployment" for a solo developer who wants less Docker pain.

### 4. RunPod Serverless

Best when you want flexible GPU hosting and don't mind Docker.

- RunPod Serverless scales from zero and bills only for active compute time.
- Queue-based endpoints use a handler function.
- Load-balancing endpoints let you run your own HTTP server, including FastAPI.

For this project, a FastAPI container is a good fit for a RunPod load-balancing endpoint.

## My Recommendation

If your goal is learning:

1. Run locally first.
2. Deploy the Gradio UI to a Hugging Face Space if you want a shareable demo.
3. Deploy the FastAPI app to Modal or RunPod when you want a real API.

If your goal is shipping an app quickly:

- Modal is the easiest developer experience.
- RunPod is attractive if you want more direct infrastructure control.
- Hugging Face Inference Endpoints is the most managed path if you want to stay close to the Hub.

## Provider Notes

The current official docs I used for this recommendation:

- Hugging Face custom containers: https://huggingface.co/docs/inference-endpoints/engines/custom_container
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/en/spaces-overview
- Modal GPU guide: https://modal.com/docs/guide/gpu
- RunPod Serverless overview: https://docs.runpod.io/serverless/overview
- RunPod endpoints overview: https://docs.runpod.io/serverless/endpoints/overview
