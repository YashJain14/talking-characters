"""
dag_talking_characters.py
--------------------------
Pipeline runner for the talking-characters pipeline.

Stages:
  0. ingest        — download YouTube videos from CSV via yt-dlp        (CPU, threaded)
  1. detect_track  — SCRFD-10GF face detection + ByteTrack  (0.25 GPU/actor → 16 actors)
  2. active_speaker — LoCoNet ASD scoring per face track    (0.5  GPU/actor →  8 actors)
  3. segment_export — single-speaker segments + content-aware crop export (CPU, 16 workers)

Each stage shells out to the corresponding script.
Resume from any stage with --from_stage (skips all earlier stages).

Usage:
  # Full pipeline from CSV
  python dag_talking_characters.py --csv videos.csv --num_gpus 1

  # Resume from a stage (videos already downloaded)
  python dag_talking_characters.py --csv videos.csv --num_gpus 1 --from_stage detect_track
  python dag_talking_characters.py --csv videos.csv --num_gpus 1 --from_stage active_speaker
  python dag_talking_characters.py --csv videos.csv --num_gpus 1 --from_stage segment_export

  # Use Light-ASD instead of LoCoNet (faster, slightly lower accuracy)
  python dag_talking_characters.py --csv videos.csv --num_gpus 1 --asd_model light_asd

Valid --from_stage: ingest  detect_track  active_speaker  segment_export
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STAGES = ["ingest", "detect_track", "active_speaker", "segment_export"]


def _paths():
    scratch = Path(os.environ.get("SCRATCH_TC") or
                   os.path.expanduser("~/scratch/talking-characters"))
    return {
        "data":    scratch,
        "videos":  scratch / "raw_videos",
        "tracks":  scratch / "tracks",
        "asd":     scratch / "asd",
        "clips":   scratch / "clips",
    }


def _run(cmd: list[str], stage: str, retries: int = 1):
    log.info(f"[{stage}] Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - t0
            log.info(f"[{stage}] Done  {elapsed:.1f}s")
            wandb.log({f"{stage}/status": 1, f"{stage}/duration_s": elapsed})
            return
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            if attempt < retries:
                log.warning(f"[{stage}] Failed (exit={e.returncode}), retrying ({attempt+1}/{retries})...")
            else:
                log.error(f"[{stage}] Failed (exit={e.returncode})  {elapsed:.1f}s")
                wandb.log({f"{stage}/status": 0, f"{stage}/duration_s": elapsed})
                raise


def run_pipeline(
    csv_path:        str,
    num_gpus:        int   = 1,
    actors_per_gpu:  int   = 1,
    ingest_workers:  int   = 4,
    asd_model:       str   = "loconet",
    speak_threshold: float = 0.5,
    gap_bridge_s:    float = 0.5,
    min_clip_s:      float = 3.0,
    max_clip_s:      float = 60.0,
    from_stage:      str   = "ingest",
):
    wandb.init(
        project="talking-characters",
        entity="rlx-labs",
        name=f"pipeline-{from_stage}",
        config={
            "csv_path":        csv_path,
            "num_gpus":        num_gpus,
            "ingest_workers":  ingest_workers,
            "asd_model":       asd_model,
            "speak_threshold": speak_threshold,
            "gap_bridge_s":    gap_bridge_s,
            "min_clip_s":      min_clip_s,
            "max_clip_s":      max_clip_s,
            "from_stage":      from_stage,
        },
    )

    p = _paths()
    stage_idx = STAGES.index(from_stage)
    here = Path(__file__).parent

    try:
        if stage_idx <= STAGES.index("ingest"):
            _run([
                sys.executable, str(here / "ingest.py"),
                "--csv",     csv_path,
                "--out_dir", str(p["videos"]),
                "--workers", str(ingest_workers),
            ], "ingest")

        if stage_idx <= STAGES.index("detect_track"):
            _run([
                sys.executable, str(here / "detect_track.py"),
                "--video_dir",      str(p["videos"]),
                "--out_dir",        str(p["tracks"]),
                "--num_gpus",       str(num_gpus),
                "--actors_per_gpu", str(actors_per_gpu),
            ], "detect_track")

        if stage_idx <= STAGES.index("active_speaker"):
            _run([
                sys.executable, str(here / "active_speaker.py"),
                "--video_dir",      str(p["videos"]),
                "--track_dir",      str(p["tracks"]),
                "--out_dir",        str(p["asd"]),
                "--num_gpus",       str(num_gpus),
                "--actors_per_gpu", str(actors_per_gpu),
                "--asd_model",      asd_model,
            ], "active_speaker")

        if stage_idx <= STAGES.index("segment_export"):
            _run([
                sys.executable, str(here / "segment_export.py"),
                "--video_dir",       str(p["videos"]),
                "--asd_dir",         str(p["asd"]),
                "--track_dir",       str(p["tracks"]),
                "--out_dir",         str(p["clips"]),
                "--num_workers",     str(num_gpus * 4),
                "--speak_threshold", str(speak_threshold),
                "--gap_bridge_s",    str(gap_bridge_s),
                "--min_clip_s",      str(min_clip_s),
                "--max_clip_s",      str(max_clip_s),
            ], "segment_export")

    finally:
        wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",             required=True,
                    help="CSV of YouTube URLs to download (columns: url, label[optional])")
    ap.add_argument("--num_gpus",        type=int,   default=1)
    ap.add_argument("--actors_per_gpu",  type=int,   default=1,
                    help="Ray actors per GPU (1=safe for single GPU, "
                         "4 for detect_track on A100, 2 for active_speaker on A100).")
    ap.add_argument("--ingest_workers",  type=int,   default=4,
                    help="Parallel yt-dlp download threads")
    ap.add_argument("--asd_model",       default="loconet",
                    choices=["loconet", "light_asd"])
    ap.add_argument("--speak_threshold", type=float, default=0.5,
                    help="LoCoNet speaking score threshold (raise to 0.6 if "
                         "both speakers are simultaneously marked active)")
    ap.add_argument("--gap_bridge_s",    type=float, default=0.5,
                    help="Merge speaking gaps shorter than this (seconds)")
    ap.add_argument("--min_clip_s",      type=float, default=3.0)
    ap.add_argument("--max_clip_s",      type=float, default=60.0)
    ap.add_argument("--from_stage",      default="ingest", choices=STAGES)
    args = ap.parse_args()

    run_pipeline(
        csv_path=args.csv,
        num_gpus=args.num_gpus,
        actors_per_gpu=args.actors_per_gpu,
        ingest_workers=args.ingest_workers,
        asd_model=args.asd_model,
        speak_threshold=args.speak_threshold,
        gap_bridge_s=args.gap_bridge_s,
        min_clip_s=args.min_clip_s,
        max_clip_s=args.max_clip_s,
        from_stage=args.from_stage,
    )


if __name__ == "__main__":
    main()