#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL_REPO="${MODEL_REPO:-RoyalCities/Foundation-1}"
MODEL_DIR="${MODEL_DIR:-models/Foundation-1}"
export MODEL_REPO MODEL_DIR

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.10 first."
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel
python -m pip install 'setuptools<81'
python -m pip install torch torchvision torchaudio
python -m pip install .
python -m pip install 'soundfile>=0.13,<0.14'

mkdir -p "$MODEL_DIR"
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import os

repo = os.environ["MODEL_REPO"]
out = Path(os.environ["MODEL_DIR"])
out.mkdir(parents=True, exist_ok=True)

for filename in ("Foundation_1.safetensors", "model_config.json"):
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(out))
    print(f"downloaded {path}")
PY

cat <<'EOF'

Setup complete.

Next steps:
  1. source .venv/bin/activate
  2. python run_gradio.py
  3. Or run a headless smoke test:
     python scripts/generate_foundation1.py

EOF
