"""
detect_track.py
---------------
Stage 1: Face detection + tracking for every video.

For each video:
  - Decode all frames at native FPS via PyNvVideoCodec (standalone, 1 GPU per worker)
  - Run SCRFD-10GF face detector on every frame
  - Associate detections across frames using ByteTrack
  - Discard tracks where face crop is < MIN_FACE_PX in either dimension
  - Write per-video track file: <stem>.tracks.json

Track file format:
  {
    "path": str,
    "fps": float,
    "n_frames": int,
    "tracks": {
      "<track_id>": [
        {"frame": int, "bbox": [x1, y1, x2, y2], "score": float},
        ...
      ]
    }
  }

Why SCRFD-10GF:
  95.2% AP on WIDER FACE hard set — beats RetinaFace-R50 (91.4%) at 3x
  the speed. Part of InsightFace, same package as ArcFace — one dep.

Why ByteTrack:
  171 FPS (CPU-side) with lowest ID switches. For frontal talking-head
  video (stable camera, no heavy occlusion) it outperforms heavier
  trackers. BoT-SORT's camera motion compensation is irrelevant here.

GPU assignment:
  0.25 GPU/actor → 16 concurrent actors across 4 A100s, matching the main
  pipeline's embed/score stages. SCRFD-10GF is small enough to share.
  Decode uses PyAV (CPU) for the same reason as the main pipeline: fractional
  actors share a physical GPU, so PyNvVideoCodec CUDA contexts would race —
  see docs/09-decode.md. CPU decode is fast enough; SCRFD inference dominates.

Usage:
  python detect_track.py --video_dir $SCRATCH_TC/raw_videos \\
                         --out_dir   $SCRATCH_TC/tracks \\
                         --num_gpus  4
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ray
import wandb


MIN_FACE_PX    = 128   # discard faces smaller than this in either dimension
ACTORS_PER_GPU = 4     # 0.25 GPU/actor → 16 concurrent across 4 A100s

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("detect_track")


def _load_scrfd(device: str):
    """Load SCRFD-10GF via InsightFace."""
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_sc",          # SCRFD-10GF + 2d106 landmark
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _decode_all_frames(video_path: str) -> tuple[list, float]:
    """
    Decode all frames via PyAV (CPU).
    Must be CPU decode: fractional-GPU actors (0.25 each) share a physical GPU,
    so PyNvVideoCodec CUDA contexts would race → CUDA_ERROR_CONTEXT_IS_DESTROYED.
    See docs/09-decode.md. PyAV is fast enough — SCRFD inference dominates.
    Returns (frames_list, fps) where each frame is [H, W, 3] uint8 numpy.
    """
    import av
    frames = []
    fps    = 25.0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        if stream.average_rate:
            fps = float(stream.average_rate)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames, fps


class _ByteTracker:
    """
    Minimal ByteTrack wrapper using the boxmot library.
    Falls back to a simple IoU tracker if boxmot is not installed.
    """
    def __init__(self, fps: float):
        try:
            from boxmot import ByteTrack
            self._tracker = ByteTrack(frame_rate=fps)
            self._mode = "boxmot"
        except ImportError:
            self._tracker = None
            self._mode    = "iou"
            self._tracks: dict[int, dict] = {}
            self._next_id = 1

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        dets: [N, 5] float32 — [x1, y1, x2, y2, score]
        Returns [M, 6] — [x1, y1, x2, y2, track_id, score]
        """
        if self._mode == "boxmot":
            if len(dets) == 0:
                return np.empty((0, 6), dtype=np.float32)
            result = self._tracker.update(dets, frame)
            return result  # boxmot returns [x1,y1,x2,y2,id,score,cls,idx]
        else:
            return self._iou_update(dets)

    def _iou_update(self, dets: np.ndarray) -> np.ndarray:
        """Simple greedy IoU tracker — fallback when boxmot is unavailable."""
        if len(dets) == 0:
            return np.empty((0, 6), dtype=np.float32)
        out = []
        matched_ids = set()
        new_tracks  = {}
        for det in dets:
            x1, y1, x2, y2, score = det
            best_iou = 0.4
            best_id  = None
            for tid, prev in self._tracks.items():
                if tid in matched_ids:
                    continue
                px1, py1, px2, py2 = prev["bbox"]
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = ((x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter)
                iou   = inter / (union + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid
            if best_id is not None:
                matched_ids.add(best_id)
                new_tracks[best_id] = {"bbox": [x1, y1, x2, y2]}
                out.append([x1, y1, x2, y2, best_id, score])
            else:
                tid = self._next_id
                self._next_id += 1
                new_tracks[tid] = {"bbox": [x1, y1, x2, y2]}
                out.append([x1, y1, x2, y2, tid, score])
        self._tracks = new_tracks
        return np.array(out, dtype=np.float32)


@ray.remote(num_gpus=1.0 / ACTORS_PER_GPU)
class DetectTrackWorker:
    def __init__(self):
        self._log = logging.getLogger("detect_track.worker")
        self._log.info(f"Worker init pid={os.getpid()}")
        self._detector = _load_scrfd("cuda:0")
        self._log.info("SCRFD-10GF loaded")

    def process(self, video_path: str, out_dir: str) -> dict:
        out_path = Path(out_dir) / (Path(video_path).stem + ".tracks.json")
        if out_path.exists():
            return {"path": video_path, "status": "cached"}

        t0 = time.perf_counter()
        try:
            frames, fps = _decode_all_frames(video_path)
            if not frames:
                return {"path": video_path, "status": "failed: no frames",
                        "n_tracks": 0}

            tracker  = _ByteTracker(fps)
            tracks: dict[str, list] = {}

            for frame_idx, frame in enumerate(frames):
                faces = self._detector.get(frame)
                if not faces:
                    tracker.update(np.empty((0, 5), dtype=np.float32), frame)
                    continue

                dets = np.array(
                    [[*f.bbox.tolist(), float(f.det_score)] for f in faces],
                    dtype=np.float32
                )
                result = tracker.update(dets, frame)

                for row in result:
                    x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                    track_id = int(row[4])
                    score    = float(row[5]) if len(row) > 5 else 1.0

                    # Discard undersized faces — too small for ASD input
                    if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
                        continue

                    key = str(track_id)
                    if key not in tracks:
                        tracks[key] = []
                    tracks[key].append({
                        "frame": frame_idx,
                        "bbox":  [round(x1, 1), round(y1, 1),
                                  round(x2, 1), round(y2, 1)],
                        "score": round(score, 4),
                    })

            # Drop tracks with fewer than fps/2 detections (~0.5s)
            min_len = max(1, int(fps / 2))
            tracks  = {k: v for k, v in tracks.items() if len(v) >= min_len}

            result_data = {
                "path":     video_path,
                "fps":      round(fps, 3),
                "n_frames": len(frames),
                "n_tracks": len(tracks),
                "tracks":   tracks,
                "status":   "ok",
                "time_s":   round(time.perf_counter() - t0, 3),
            }
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result_data))
            return {"path": video_path, "status": "ok", "n_tracks": len(tracks)}

        except Exception as e:
            import traceback
            self._log.error(f"FAILED {Path(video_path).name}\n{traceback.format_exc()}")
            return {"path": video_path, "status": f"failed: {e}", "n_tracks": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--num_gpus",  type=int, default=1)
    args = ap.parse_args()

    videos   = sorted(Path(args.video_dir).rglob("*.mp4"))
    n_actors = args.num_gpus * ACTORS_PER_GPU
    log.info(f"Found {len(videos)} videos")
    log.info(f"Actors: {n_actors} (1 per GPU × {args.num_gpus} GPUs)")

    wandb.init(project="talking-characters", entity="rlx-labs",
               name="detect-track", resume="allow")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    workers = [DetectTrackWorker.remote() for _ in range(n_actors)]

    futures = [
        workers[i % n_actors].process.remote(str(v), args.out_dir)
        for i, v in enumerate(videos)
    ]

    ok = failed = cached = 0
    total   = len(videos)
    pending = list(futures)
    t0      = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(20, len(pending)), timeout=60)
        for res in ray.get(done):
            s = res["status"]
            if   s == "ok":     ok     += 1
            elif s == "cached": cached += 1
            else:
                failed += 1
                log.error(f"FAILED {Path(res['path']).name}: {s}")
        completed = ok + failed + cached
        elapsed   = time.perf_counter() - t0
        log.info(f"[{completed}/{total}]  ok={ok}  cached={cached}  failed={failed}  "
                 f"elapsed={elapsed:.1f}s")
        wandb.log({"detect_track/ok": ok, "detect_track/cached": cached,
                   "detect_track/failed": failed, "detect_track/elapsed_s": elapsed})

    log.info(f"Final: ok={ok}  cached={cached}  failed={failed}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
