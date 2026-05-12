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

source talking_character/bin/activate

echo "   Environment activated"
echo "   Python: $(python --version)"
echo "   CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH')"
echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

cd "$HOME/talking-characters" || exit 1

echo "GPUs:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:  $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "| GPUs:", torch.cuda.device_count())')"

export WANDB_API_KEY=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
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set." >&2
    exit 1
fi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export SCRATCH_TC="$HOME/scratch/talking-characters"
mkdir -p "$SCRATCH_TC/raw_videos"
echo "Scratch: $SCRATCH_TC"

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV="$SLURM_SUBMIT_DIR/videos_test.csv"

NUM_GPUS=1
ACTORS_PER_GPU=1
INGEST_WORKERS=4

# SyncNet confidence threshold: median_dist - min_dist.
# solo close-ups typically score 1.0-3.0; two-person shots 0.1-0.9
# 0.3 captures most two-person shots while filtering pure noise
SYNC_THRESHOLD=0.3

GAP_BRIDGE_S=0.5   # bridge speaking gaps shorter than this
MIN_CLIP_S=3.0
MAX_CLIP_S=60.0

# Minimum face size: 64px captures two-person and some panel shots
# (was 128 — that dropped 56k panel shot detections)
MIN_FACE_PX=64

FROM_STAGE=ingest   # resume: ingest | detect_track | syncnet_score | segment_export

# Keeping existing clips — only new videos will be processed
echo "Keeping existing clips — running from $FROM_STAGE"

# ── GPU concurrency summary (1x V100-32GB) ───────────────────────────────────
# ingest:         CPU-only → 4 yt-dlp threads
# detect_track:   0.25 GPU/actor → up to 4 concurrent actors  (SCRFD-10GF)
# syncnet_score:  1.0  GPU/actor → 1 actor                    (SyncNet ~5M params)
# segment_export: CPU-only → 4 workers                        (ffmpeg)

python dag_talking_characters.py \
    --csv             $INPUT_CSV       \
    --num_gpus        $NUM_GPUS        \
    --actors_per_gpu  $ACTORS_PER_GPU  \
    --ingest_workers  $INGEST_WORKERS  \
    --sync_threshold  $SYNC_THRESHOLD  \
    --gap_bridge_s    $GAP_BRIDGE_S    \
    --min_clip_s      $MIN_CLIP_S      \
    --max_clip_s      $MAX_CLIP_S      \
    --min_face_px     $MIN_FACE_PX     \
    --from_stage      $FROM_STAGE

echo "Job completed at $(date)"

# ── Copy clips into repo for inspection ───────────────────────────────────────
CLIPS_DEST="$HOME/talking-characters/output_clips"
mkdir -p "$CLIPS_DEST"
cp -r "$SCRATCH_TC/clips/"*/*.mp4 "$CLIPS_DEST/" 2>/dev/null || true
cp "$SCRATCH_TC/clips/segments.json" "$CLIPS_DEST/" 2>/dev/null || true
echo "Clips copied to $CLIPS_DEST  ($(ls "$CLIPS_DEST"/*.mp4 2>/dev/null | wc -l) mp4s)"
