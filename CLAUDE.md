# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
bash setup_env.sh                                   # creates conda env: talking_characters_env
conda activate talking_characters_env
python prefetch_models_talking_characters.py        # download weights (login node only)
export WANDB_API_KEY=<your_key>
qsub run_talking_characters.pbs
```

Compute nodes have no internet — all model weights must be prefetched on the login node before submitting.

## Running the Pipeline

```bash
# Full run from CSV
python dag_talking_characters.py --csv videos.csv --num_gpus 4

# Resume from a specific stage
python dag_talking_characters.py --csv videos.csv --num_gpus 4 --from_stage detect_track
python dag_talking_characters.py --csv videos.csv --num_gpus 4 --from_stage active_speaker
python dag_talking_characters.py --csv videos.csv --num_gpus 4 --from_stage segment_export
```

Valid `--from_stage`: `ingest  detect_track  active_speaker  segment_export`

## Architecture

### Pipeline DAG

`dag_talking_characters.py` is a Prefect `@flow` that shells out to each stage script.

```
ingest → detect_track → active_speaker → segment_export
```

### Stage Scripts

| Script | Model | Ray GPU fraction | Purpose |
|--------|-------|-----------------|---------|
| `ingest.py` | — | CPU (4 threads) | yt-dlp download from CSV → `raw_videos/<label>/<id>.mp4` |
| `detect_track.py` | SCRFD-10GF (InsightFace buffalo_sc) | 0.25 (16 actors) | Per-frame face detection + ByteTrack ID assignment |
| `active_speaker.py` | LoCoNet (34M) or Light-ASD (0.84M) | 0.5 (8 actors) | Per-frame speaking probability per face track |
| `segment_export.py` | — | CPU (16 workers) | Single-speaker segment finding + ffmpeg export |

### Input CSV Format

```csv
url,label
https://www.youtube.com/watch?v=abc123,speaker_name
```

`label` is optional — omit it and videos go into `raw_videos/unlabelled/`. The `url` column header is also optional if the CSV has no header (first column assumed to be URL).

### Video Decode

All Ray actors use PyAV (CPU) decode. Fractional-GPU actors share a physical GPU; PyNvVideoCodec CUDA contexts race under fractional sharing → `CUDA_ERROR_CONTEXT_IS_DESTROYED`. PyAV decode is fast enough since model inference dominates per-video time.

### Crop Logic (segment_export.py)

Content-aware crop — not a fixed padding:
1. **Black bars** — always: scan row/col mean luminance < `BLACK_THRESHOLD=16`, remove if found
2. **Extra speaker** — only when detected: bisect frame at midpoint between the two face centres, keep active-speaker half
3. **Slides/no-face scenes** — discarded automatically because `detect_track.py` requires at least one face track with ≥ `MIN_FACE_PX=128` pixels; `active_speaker.py` requires a face track to assign a speaking score; `segment_export.py` requires exactly one track above `SPEAK_THRESHOLD` — scenes with no visible speaker produce no clips

### Caching

Each stage writes a per-video result file before aggregating:
- `tracks/<stem>.tracks.json` — detect_track cache
- `asd/<stem>.asd.json` — active_speaker cache
- `clips/<stem>/*.json` + `.done` markers — segment_export cache
- `raw_videos/**/.*.done` — ingest re-run markers

### Quality Gates

| Gate | Value | Where |
|------|-------|-------|
| Min face px | 128 | detect_track.py |
| Min track frames | fps/2 | detect_track.py |
| Active ratio | ≥ 0.60 | segment_export.py |
| Face present ratio | ≥ 0.80 | segment_export.py |
| Clip duration | 3–60s | segment_export.py |
| Speak threshold | 0.5 | segment_export.py (raise to 0.6 if over-triggering) |

## Key Constraints

- **`numpy < 2`** and **`opencv-python-headless < 4.11`** — same ABI constraints as sibling pipeline
- **`WANDB_API_KEY`** must be exported before `qsub`
- **`HF_HUB_OFFLINE=1`** on compute nodes