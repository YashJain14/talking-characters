#!/bin/bash
# ---------------------------------------------------------------------------
# Setup Script: Run on the LOGIN NODE before submitting run_talking_characters.pbs
# ---------------------------------------------------------------------------

echo "Setting up talking_characters_env ..."

CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"

ENV_NAME="talking_characters_env"
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2"

# Video decode (CPU path used in fractional-GPU Ray actors)
pip install av
# GPU decode available for standalone use; not used in Ray actors — see docs
pip install pynvvideocodec
# numpy<2 constraint: pin opencv below 4.11 (4.11+ uses NumPy 2 ABI)
pip install "opencv-python-headless<4.11"

# YouTube download
pip install yt-dlp

# Audio extraction / processing
pip install soundfile
pip install python_speech_features

# Face detection + tracking
pip install insightface onnxruntime-gpu
pip install boxmot                        # ByteTrack

# Active speaker detection
# LoCoNet (recommended — best multi-speaker accuracy):
pip install loconet
# Light-ASD fallback (faster, 0.84M params):
pip install light-asd

# Ray distributed
pip install "ray[default]"

# Orchestration
pip install prefect

# Experiment tracking
pip install wandb

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. python prefetch_models_talking_characters.py   # download model weights (needs internet)"
echo "  2. export WANDB_API_KEY=<your_key>"
echo "  3. qsub run_talking_characters.pbs"