# Talking Characters — Video Data Curation Pipeline

End-to-end pipeline that takes a CSV of YouTube video URLs and produces clean, single-speaker talking-head clips ready for video diffusion training.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NSCC HPC · PBS scheduler · 4× A100 GPUs per node

---

## Pipeline

```
videos.csv  (YouTube URLs)
       │
       ▼
[ingest.py]           yt-dlp download → raw_videos/<label>/<id>.mp4
       │                                CPU, 4 parallel threads
       ▼
[detect_track.py]     SCRFD-10GF face detection + ByteTrack
       │               0.25 GPU/actor → 16 concurrent actors
       ▼
[active_speaker.py]   LoCoNet active speaker detection
       │               per-frame speaking score per face track
       │               0.5 GPU/actor → 8 concurrent actors
       ▼
[segment_export.py]   Find single-speaker segments
                       remove black bars, split frame if extra speaker present
                       re-encode via ffmpeg → clips/<stem>/<clip>.mp4
```

Orchestrated by `dag_talking_characters.py` (Prefect flow). Resume from any stage with `--from_stage`.

---

## Input CSV Format

One video per row. Supported formats:

```csv
url
https://www.youtube.com/watch?v=abc123
https://youtu.be/xyz789
```

```csv
url,label
https://www.youtube.com/watch?v=abc123,elon_musk
https://youtu.be/xyz789,lex_fridman
```

If no `label` column is present, all videos go into `raw_videos/unlabelled/`.

---

## Setup & Running

### One-time setup (login node)

```bash
bash setup_env.sh
conda activate talking_characters_env
python prefetch_models_talking_characters.py
```

Compute nodes have no internet (`HF_HUB_OFFLINE=1`). All model weights must be prefetched on the login node.

### Full pipeline

```bash
export WANDB_API_KEY=<your_key>
# Edit INPUT_CSV in run_talking_characters.pbs to point to your CSV
qsub run_talking_characters.pbs
```

### Direct invocation / resume

```bash
# Full run
python dag_talking_characters.py --csv videos.csv --num_gpus 4

# Resume from a stage
python dag_talking_characters.py --csv videos.csv --num_gpus 4 --from_stage detect_track
python dag_talking_characters.py --csv videos.csv --num_gpus 4 --from_stage active_speaker
python dag_talking_characters.py --csv videos.csv --num_gpus 4 --from_stage segment_export
```

Valid `--from_stage`: `ingest  detect_track  active_speaker  segment_export`

---

## Output Layout

```
$SCRATCH_TC/                                    # ~/scratch/talking-characters
├── raw_videos/<label>/<video_id>.mp4           # downloaded videos
├── tracks/<stem>.tracks.json                   # face bboxes + ByteTrack IDs per frame
├── asd/<stem>.asd.json                         # LoCoNet speaking scores per track per frame
├── clips/<stem>/<stem>_<tid>_<s>_<e>.mp4      # exported single-speaker clips
├── clips/<stem>/<stem>_<tid>_<s>_<e>.json     # per-clip metadata
└── clips/segments.json                         # aggregated manifest of all clips
```

---

## GPU Concurrency

| Stage | GPU fraction | Concurrent actors (4 GPUs) |
|-------|-------------|---------------------------|
| ingest | CPU only | 4 yt-dlp threads |
| detect_track | 0.25 | 16 (SCRFD-10GF, CPU decode) |
| active_speaker | 0.5 | 8 (LoCoNet 34M params) |
| segment_export | CPU only | 16 workers (ffmpeg) |

All Ray GPU actors use PyAV CPU decode — fractional actors share a physical GPU, so PyNvVideoCodec CUDA contexts race.

---

## Models

| Model | Used in | Notes |
|-------|---------|-------|
| SCRFD-10GF (InsightFace buffalo_sc) | `detect_track.py` | 95.2% AP WIDER FACE hard, 3× faster than RetinaFace |
| LoCoNet | `active_speaker.py` | 95.2% mAP AVA, +3% over TalkNet in multi-speaker scenes |
| Light-ASD | `active_speaker.py` | Fallback — 94.1% mAP, 0.84M params, faster |

---

## Quality Gates

| Gate | Default | Notes |
|------|---------|-------|
| Min face size | 128px | Tracks with smaller faces are dropped |
| Min track length | fps/2 frames | Drops sub-0.5s flickers |
| Active ratio | ≥ 60% | Clip frames must have exactly one speaker active |
| Face present ratio | ≥ 80% | Face must be visible most of the clip |
| Clip duration | 3–60s | |

## Crop Logic

Black bars are detected and removed on every clip by scanning row/column mean luminance. If a second speaker's face is tracked in the same segment, the frame is split at the midpoint between the two face centres and only the active-speaker half is kept. No fixed margin is applied — content that isn't junk is preserved.

---

## Key Constraints

- **`numpy < 2`** — some dependencies are built against NumPy 1.x ABI
- **`opencv-python-headless < 4.11`** — OpenCV ≥ 4.11 uses NumPy 2 ABI
- **`WANDB_API_KEY` must be exported before `qsub`** — compute nodes have no interactive login
- **`HF_HUB_OFFLINE=1`** on compute nodes — run `prefetch_models_talking_characters.py` first