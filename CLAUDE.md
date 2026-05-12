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
python dag_talking_characters.py --csv videos_test.csv --num_gpus 1

# Resume from a specific stage
python dag_talking_characters.py --csv videos_test.csv --from_stage detect_track
python dag_talking_characters.py --csv videos_test.csv --from_stage syncnet_score
python dag_talking_characters.py --csv videos_test.csv --from_stage segment_export
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
| `detect_track.py` | SCRFD-10GF (InsightFace buffalo_sc) | 1.0/actor | Per-frame face detection + ByteTrack |
| `syncnet_score.py` | SyncNet (~5M params) | 1.0/actor | Audio-visual sync confidence per face track |
| `segment_export.py` | — | CPU | Quality gates + clip extraction via ffmpeg |

### How it works

1. **ingest**: Download videos from CSV via yt-dlp
2. **detect_track**: SCRFD detects faces every frame; ByteTrack assigns persistent IDs. Outputs `<stem>.tracks.json` with per-track stats (face size, concurrent face count).
3. **syncnet_score**: For each face track, extract lip crops + MFCC audio windows, run SyncNet in a sliding 25-frame window. Outputs `<stem>.sync.json` with confidence, reject_reason, face size per track.
4. **segment_export**: Winner-takes-all per time segment, panel/nodder gates, ffmpeg export with black-bar and two-person crop.

### SyncNet scoring

- `confidence = median(pairwise_dist) - min(pairwise_dist)` with vshift=15
- Solo close-ups score 1.0–3.0; two-person shots 0.3–0.9; panel shots 0.0–0.4
- Weights: `~/.cache/talking_characters/syncnet.pth`

### Quality Gates (segment_export.py)

| Gate | Value | Catches |
|------|-------|---------|
| Panel shot | n\_other ≥ 2 → reject | 5-person wide shots |
| Two-person confidence | n\_other=1 → conf ≥ 0.8 | Nodders, listeners |
| Solo confidence | n\_other=0 → conf ≥ 0.3 | Non-speakers |
| Winner-takes-all | highest conf per overlap group | Duplicate time ranges |
| Face present ratio | ≥ 80% | Partial tracks |
| Clip duration | 3–60s | Too short/long |

### segments.json format

```json
{
  "clips": [
    {
      "track_id": "26",
      "start_s": 54.1, "end_s": 59.0, "duration_s": 4.9,
      "sync_confidence": 3.064,
      "n_faces_in_frame": 0,
      "median_face_px": 317.0,
      "had_black_bars": false,
      "had_extra_speaker": false
    }
  ],
  "rejected_clips": [
    {
      "track_id": "33",
      "reject_reason": "overlap_lost_to_higher_conf"
    }
  ]
}
```

Reject reasons: `too_short` · `too_long` · `low_face_ratio` · `overlap_lost_to_higher_conf` · `panel_shot` · `low_conf_for_scene_type`

### Input CSV Format

```csv
url,label
https://www.youtube.com/watch?v=abc123,speaker_name
```

`label` is optional — videos go into `raw_videos/unlabelled/` if omitted.

### Caching

Each stage writes per-video result files and skips if already present:
- `tracks/<stem>.tracks.json` — detect_track output
- `sync/<stem>.sync.json` — syncnet_score output
- `clips/<stem>/.done` — segment_export done marker
- `clips/<stem>/*.mp4` + `*.json` — exported clips

## Key Constraints

- **`numpy < 2`** and **`opencv-python-headless < 4.11`** — ABI constraints
- **`WANDB_API_KEY`** must be exported before `sbatch`
- **`HF_HUB_OFFLINE=1`** on compute nodes
- SyncNet weights must be at `~/.cache/talking_characters/syncnet.pth`
- `MIN_FACE_PX=64` in detect_track — captures two-person shots (was 128, dropped 56k panel detections)
- `SYNC_THRESHOLD=0.3` in run_job.slurm — per-scene thresholds applied in segment_export