# Talking Characters — Single-Speaker Clip Curation from Multi-Speaker Videos

End-to-end pipeline that takes a CSV of YouTube URLs and produces clean, single-speaker talking-head clips ready for video diffusion training.

**Hardware target:** NVIDIA V100-32GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** SLURM · `UGGPU-TC1` partition

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
       │               1 GPU/actor, min face size 64px
       ▼
[syncnet_score.py]    SyncNet audio-visual lip-sync scoring
       │               1 GPU/actor, confidence = median_dist - min_dist
       ▼
[segment_export.py]   Quality gates + ffmpeg clip export
                       winner-takes-all per time segment
                       panel shot rejection, nodder filtering
```

Orchestrated by `dag_talking_characters.py`. Resume from any stage with `--from_stage`.

---

## Quick Start

### One-time setup (login node only)

```bash
python -m venv talking_character
source talking_character/bin/activate
pip install -r requirements.txt
python prefetch_models_talking_characters.py   # downloads SyncNet weights
export WANDB_API_KEY=<your_key>
```

Compute nodes have no internet — weights must be prefetched on the login node first.

### Run

```bash
sbatch run_job.slurm
```

### Resume from a stage

```bash
python dag_talking_characters.py --csv videos_test.csv --from_stage detect_track
python dag_talking_characters.py --csv videos_test.csv --from_stage syncnet_score
python dag_talking_characters.py --csv videos_test.csv --from_stage segment_export
```

---

## Input CSV Format

```csv
url,label
https://www.youtube.com/watch?v=abc123,speaker_name
https://youtu.be/xyz789,another_speaker
```

`label` is optional — goes into `raw_videos/unlabelled/` if omitted.

---

## Output Layout

```
$SCRATCH_TC/                                       # ~/scratch/talking-characters
├── raw_videos/<label>/<video_id>.mp4              # downloaded videos
├── tracks/<stem>.tracks.json                      # face bboxes + track IDs + stats
├── sync/<stem>.sync.json                          # SyncNet confidence per track
└── clips/
    ├── <stem>/<stem>_<tid>_<sf>_<ef>.mp4         # exported clips
    ├── <stem>/<stem>_<tid>_<sf>_<ef>.json        # per-clip metadata
    └── segments.json                              # aggregated manifest
                                                   #   {"clips": [...], "rejected_clips": [...]}
```

---

## Quality Gates

Applied in order during `segment_export.py`:

| Gate | Value | Reason |
|------|-------|--------|
| Min face size (detect) | 64px | Smaller faces unreliable for SyncNet |
| Min track length | fps/2 frames | Drop sub-0.5s flickers |
| SyncNet confidence (solo) | ≥ 0.3 | Solo close-up, reliable scoring |
| SyncNet confidence (2-person) | ≥ 0.8 | Filter nodders from two-person shots |
| Panel shot rejection | n\_other ≥ 2 | Too small to crop cleanly |
| Winner-takes-all | highest conf wins | One clip per time segment |
| Face present ratio | ≥ 80% | Face visible most of clip |
| Clip duration | 3–60s | |

---

## SyncNet Scoring

SyncNet (Chung & Zisserman, ECCV 2016) measures whether lip movements match the audio.

- `confidence = median(pairwise_dist) - min(pairwise_dist)`
- Solo close-ups typically score 1.0–3.0
- Two-person shots score 0.3–0.9 (real speakers) or 0.1–0.5 (nodders)
- Panel shots (5 people, face ~65px) score 0.0–0.4 → rejected by panel gate

Weights: `~/.cache/talking_characters/syncnet.pth`

---

## Crop Logic

1. **Black bars** — scan row/col mean luminance < 16, crop if found
2. **Two-person shot** — bisect frame at midpoint between face centres, keep active-speaker half
3. **Panel shots** — rejected entirely (face too small to crop cleanly at export resolution)

---

## Tested On

| Video | Label | Clips Extracted | Notes |
|-------|-------|-----------------|-------|
| [youtube.com/watch?v=57lDpTwiW6g](https://www.youtube.com/watch?v=57lDpTwiW6g) | YC | 101 | Human characters, multi-speaker |
| [youtube.com/watch?v=UPGB-hsAoVY](https://www.youtube.com/watch?v=UPGB-hsAoVY) | YC | 38 | Human characters, multi-speaker |
| [youtube.com/watch?v=8kkosuO2AII](https://www.youtube.com/watch?v=8kkosuO2AII) | Naruto | 6 | Animated — pipeline struggles |

---

## Results & Limitations

### Human Characters
Results on human characters were generally good. The main failure mode was sideways/profile shots — SCRFD detects the face but SyncNet relies on frontal lip visibility, so active speaker identification degrades significantly when the character is not facing the camera.

### Animated Characters
The pipeline does not work for animated characters. SCRFD is trained on real human faces and misses or misdetects cartoon/animated faces. SyncNet similarly fails as it learned lip-sync from real video. Animated characters need separate tooling that operates beyond human face detection — e.g. character-specific detectors or audio-driven active speaker methods that don't rely on facial geometry.

### Future Enhancements
- **Watermark and overlay cropping** — YouTube watermarks, on-screen text, and graphical overlays are not currently handled. Detecting and cropping these regions out would improve clip quality for training.
- **Animated character support** — requires tooling beyond human face detection; character-specific detectors or audio-driven active speaker methods not reliant on facial geometry.

---

## Key Constraints

- **`numpy < 2`** and **`opencv-python-headless < 4.11`** — ABI constraints
- **`WANDB_API_KEY`** must be set before `sbatch`
- **`HF_HUB_OFFLINE=1`** on compute nodes
- SyncNet weights at `~/.cache/talking_characters/syncnet.pth`
