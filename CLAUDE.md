# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source talking_character/bin/activate
python prefetch_models_talking_characters.py   # login node only — downloads weights
export WANDB_API_KEY=<your_key>
sbatch run_job.slurm
```

Compute nodes have no internet — all model weights must be prefetched on the login node before submitting.

## Running the Pipeline

```bash
# Full run from CSV
python dag_talking_characters.py --csv videos.csv --num_gpus 1

# Resume from a specific stage
python dag_talking_characters.py --csv videos.csv --from_stage detect_track
python dag_talking_characters.py --csv videos.csv --from_stage syncnet_score
python dag_talking_characters.py --csv videos.csv --from_stage segment_export
```

Valid `--from_stage`: `ingest  detect_track  syncnet_score  segment_export`

## Architecture

### Pipeline DAG

```
ingest → detect_track → syncnet_score → segment_export
```

### Stage Scripts

| Script | Model | GPU | Purpose |
|--------|-------|-----|---------|
| `ingest.py` | — | CPU | yt-dlp download from CSV → `raw_videos/<label>/<id>.mp4` |
| `detect_track.py` | SCRFD-10GF (InsightFace buffalo_sc) | 0.25/actor | Per-frame face detection + ByteTrack |
| `syncnet_score.py` | SyncNet (~5M params) | 1.0/actor | Audio-visual sync confidence per face track |
| `segment_export.py` | — | CPU | Clip extraction + content-aware crop via ffmpeg |

### How it works

1. **ingest**: Download videos from CSV via yt-dlp
2. **detect_track**: SCRFD detects faces every frame; ByteTrack assigns persistent IDs. Outputs `<stem>.tracks.json` per video.
3. **syncnet_score**: For each face track, extract lip crops + MFCC audio windows, run SyncNet in a sliding 25-frame window. A track **passes** if mean confidence ≥ `SYNC_THRESHOLD` (default 5.0). Outputs `<stem>.sync.json` per video.
4. **segment_export**: For each passing track, split on long gaps, apply duration/face-ratio gates, export via ffmpeg with black-bar and multi-speaker crop.

### SyncNet scoring

SyncNet measures whether a face's lip movements are in sync with the audio. High confidence (>5) = person is actively speaking. This replaces LoCoNet ASD — it's more reliable and doesn't require a custom model implementation.

- `sync_threshold=5.0` — lower to 3.0 if too few clips, raise to 7.0 to be stricter
- Sliding window: 25 frames (1s at 25fps), stride 5 frames
- CPU fallback works if GPU is unavailable (slower)

### Input CSV Format

```csv
url,label
https://www.youtube.com/watch?v=abc123,speaker_name
```

`label` is optional — videos go into `raw_videos/unlabelled/` if omitted.

### Crop Logic (segment_export.py)

1. **Black bars** — always: scan row/col mean luminance < 16, crop if found
2. **Other speakers** — only when another face track overlaps this clip's time range: bisect at midpoint between face centres, keep active-speaker half
3. **Slides/no-face** — discarded automatically (no face tracks → no sync score → no clips)

### Caching

Each stage writes per-video result files:
- `tracks/<stem>.tracks.json` — detect_track output
- `sync/<stem>.sync.json` — syncnet_score output
- `clips/<stem>/*.mp4` + `*.json` — exported clips

### Quality Gates

| Gate | Value | Where |
|------|-------|-------|
| Min face size | 128px | detect_track.py |
| Min track length | fps/2 frames | detect_track.py |
| SyncNet confidence | ≥ 5.0 | syncnet_score.py |
| Face present ratio | ≥ 80% of clip frames | segment_export.py |
| Clip duration | 3–60s | segment_export.py |

## Key Constraints

- **`numpy < 2`** and **`opencv-python-headless < 4.11`** — ABI constraints
- **`WANDB_API_KEY`** must be exported before `sbatch`
- **`HF_HUB_OFFLINE=1`** on compute nodes
- SyncNet weights must be at `~/.cache/talking_characters/syncnet.pth`
