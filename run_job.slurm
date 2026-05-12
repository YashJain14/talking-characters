#!/bin/bash
#SBATCH --job-name=talking_characters
#SBATCH --output=talking_characters_%j.log
#SBATCH --error=talking_characters_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=UGGPU-TC1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source talking_character/bin/activate

echo "   Environment activated"
echo "   Python: $(python --version)"
echo "   CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Change to project directory
cd "$HOME/talking-characters" || exit 1

echo "GPUs:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:  $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "| GPUs:", torch.cuda.device_count())')"

export WANDB_API_KEY=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
# ── Credentials ───────────────────────────────────────────────────────────────
# WANDB_API_KEY must be exported on the login node before qsub.
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set. Export it on the login node before qsub." >&2
    exit 1
fi

# ── Offline mode (compute nodes have no internet) ─────────────────────────────
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── Scratch storage ───────────────────────────────────────────────────────────
export SCRATCH_TC="$HOME/scratch/talking-characters"
mkdir -p "$SCRATCH_TC/raw_videos"
echo "Scratch: $SCRATCH_TC"

# ── Config ────────────────────────────────────────────────────────────────────
# Path to your input CSV (see ingest.py for format: url, label columns)
INPUT_CSV="$SLURM_SUBMIT_DIR/videos_test.csv"

NUM_GPUS=1
ACTORS_PER_GPU=1      # 1 GPU total — single actor per stage
INGEST_WORKERS=4       # parallel yt-dlp download threads

# ASD model: "loconet" (recommended, +3% mAP over TalkNet in multi-speaker)
# or "light_asd" (faster, 0.84M params, 94.1% mAP)
ASD_MODEL="loconet"

# LoCoNet speaking score threshold.
# 0.5 = default. Raise to 0.6 if both speakers are simultaneously marked active.
SPEAK_THRESHOLD=0.5

GAP_BRIDGE_S=0.5   # merge speaking gaps shorter than this (natural pauses)
MIN_CLIP_S=3.0     # discard clips shorter than this
MAX_CLIP_S=60.0    # discard clips longer than this

FROM_STAGE="segment_export"   # resume: ingest | detect_track | active_speaker | segment_export

# ── GPU concurrency summary (1x V100-32GB) ───────────────────────────────────
# ingest:         CPU-only → 4 yt-dlp threads
# detect_track:   0.25 GPU/actor → 4 concurrent actors   (SCRFD-10GF, CPU decode)
# active_speaker: 0.5  GPU/actor → 2 concurrent actors   (LoCoNet ~34M params)
# segment_export: CPU-only       → 16 workers             (ffmpeg re-encode)

# ── Run ───────────────────────────────────────────────────────────────────────
python dag_talking_characters.py \
    --csv             $INPUT_CSV       \
    --num_gpus        $NUM_GPUS        \
    --actors_per_gpu  $ACTORS_PER_GPU  \
    --ingest_workers  $INGEST_WORKERS  \
    --asd_model       $ASD_MODEL       \
    --speak_threshold $SPEAK_THRESHOLD \
    --gap_bridge_s    $GAP_BRIDGE_S    \
    --min_clip_s      $MIN_CLIP_S      \
    --max_clip_s      $MAX_CLIP_S      \
    --from_stage      $FROM_STAGE

echo "Job completed at $(date)"