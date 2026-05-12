"""
syncnet_score.py
----------------
Stage 2: Score each face track for audio-visual sync using SyncNet.

For each video + its track file from detect_track.py:
  - Extract audio (16 kHz mono)
  - For each face track: extract 25-fps face crop windows + MFCC audio windows
  - Run SyncNet in a sliding window to get per-window sync confidence
  - A track passes if mean confidence > SYNC_THRESHOLD
  - Write per-video sync result: <stem>.sync.json

Sync result format:
  {
    "path":   str,
    "fps":    float,
    "tracks": {
      "<track_id>": {
        "frames":     [int, ...],
        "confidence": float,       # mean SyncNet confidence (higher = more synced)
        "offset":     int,         # best A/V offset (0 = in sync)
        "passes":     bool         # confidence > SYNC_THRESHOLD
      }
    },
    "status": "ok"
  }

SyncNet (Chung & Zisserman, ECCV 2016):
  Measures audio-visual correspondence by comparing lip-crop embeddings
  with MFCC embeddings in a shared metric space.
  confidence > 5–7 = clearly synced speaker.
  offset = 0 = no A/V delay (live speech, not dubbing).

GPU assignment:
  SyncNet is tiny (~5M params). 1 GPU actor handles the full video.
  CPU fallback works fine for short clips.

Usage:
  python syncnet_score.py \\
    --video_dir  $SCRATCH_TC/raw_videos \\
    --track_dir  $SCRATCH_TC/tracks \\
    --out_dir    $SCRATCH_TC/sync \\
    --num_gpus   1
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
import torch.nn as nn
import torch.nn.functional as F
import ray
import wandb

SYNC_THRESHOLD = 5.0     # min mean confidence to consider a track as speaking
FACE_SIZE      = 112     # face crop size fed to SyncNet visual stream
CROP_PAD       = 0.25    # padding around face bbox for SyncNet (tighter than export)
AUDIO_SR       = 16000
WINDOW_FRAMES  = 25      # SyncNet window: 25 video frames (1s at 25fps)
MFCC_PER_FRAME = 4       # MFCC frames per video frame (100 mfcc fps / 25 video fps)
N_MFCC         = 13      # MFCC coefficients
ACTORS_PER_GPU = 2

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("syncnet_score")


# ─────────────────────────────────────────────────────────────────────────────
# SyncNet model  (Chung & Zisserman ECCV 2016, "out-of-time" variant)
# Visual stream : 5×112×112 grayscale face crops → 1024-d embedding
# Audio stream  : 1×13×20 MFCC window           → 1024-d embedding
# ─────────────────────────────────────────────────────────────────────────────

class _SyncNetVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1,  96, (5,7,7), stride=(1,2,2), padding=0), nn.BatchNorm3d(96),  nn.ReLU(),
            nn.MaxPool3d((1,3,3), stride=(1,2,2)),
            nn.Conv3d(96, 256,(1,5,5), stride=(1,2,2), padding=(0,1,1)), nn.BatchNorm3d(256), nn.ReLU(),
            nn.MaxPool3d((1,3,3), stride=(1,2,2)),
            nn.Conv3d(256,256,(1,3,3), padding=(0,1,1)), nn.BatchNorm3d(256), nn.ReLU(),
            nn.Conv3d(256,256,(1,3,3), padding=(0,1,1)), nn.BatchNorm3d(256), nn.ReLU(),
            nn.Conv3d(256,256,(1,3,3), padding=(0,1,1)), nn.BatchNorm3d(256), nn.ReLU(),
            nn.MaxPool3d((1,3,3), stride=(1,2,2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):   # x: [B, 1, 5, 112, 112]
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)   # [B, 512]  (actual syncnet uses 1024 but we normalise)


class _SyncNetAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  96, (3,3), stride=(1,1), padding=(1,1)), nn.BatchNorm2d(96),  nn.ReLU(),
            nn.MaxPool2d((1,1), stride=(1,1)),
            nn.Conv2d(96, 256,(3,3), stride=(1,1), padding=(1,1)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((1,1), stride=(1,1)),
            nn.Conv2d(256,384,(3,3), padding=(1,1)), nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384,256,(3,3), padding=(1,1)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,(3,3), padding=(1,1)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((3,1), stride=(2,1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*1*20, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):   # x: [B, 1, 13, 20]
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)   # [B, 512]


class SyncNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _SyncNetVisual()
        self.audio  = _SyncNetAudio()

    def forward(self, face_seq, mfcc_win):
        v = F.normalize(self.visual(face_seq), p=2, dim=1)
        a = F.normalize(self.audio(mfcc_win),  p=2, dim=1)
        return (v * a).sum(dim=1)   # cosine similarity per window


def _load_syncnet(device: str) -> SyncNet:
    weights = os.environ.get(
        "SYNCNET_WEIGHTS",
        str(Path.home() / ".cache" / "talking_characters" / "syncnet.pth")
    )
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"SyncNet weights not found at {weights}. "
            "Run prefetch_models_talking_characters.py first."
        )
    model = SyncNet()
    state = torch.load(weights, map_location=device, weights_only=True)
    # Handle DataParallel wrapper
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_audio(video_path: str, sr: int = AUDIO_SR) -> np.ndarray | None:
    import subprocess, tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-ac", "1", "-ar", str(sr), "-f", "wav", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        wav, _ = sf.read(tmp_path, dtype="float32")
        return wav
    except Exception:
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _compute_mfcc(wav: np.ndarray, sr: int) -> np.ndarray:
    """Compute MFCC at 100 fps. Returns [T_mfcc, N_MFCC] float32."""
    import python_speech_features
    mfcc = python_speech_features.mfcc(
        wav, sr, numcep=N_MFCC, winlen=0.025, winstep=0.010,
    )
    return mfcc.astype(np.float32)   # [T_mfcc, 13]


def _crop_face(frame: np.ndarray, bbox: list, pad: float) -> np.ndarray:
    import cv2
    h, w  = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half   = max(bw, bh) / 2 * (1 + pad)
    lx, ly = max(0, int(cx - half)), max(0, int(cy - half))
    rx, ry = min(w, int(cx + half)), min(h, int(cy + half))
    crop   = frame[ly:ry, lx:rx]
    if crop.size == 0:
        return np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (FACE_SIZE, FACE_SIZE))


def _score_track(
    face_crops: list[np.ndarray],   # [N] each (112,112) uint8 grayscale
    mfcc: np.ndarray,               # [T_mfcc, 13]
    frame_indices: list[int],
    fps: float,
    model: SyncNet,
    device: str,
    debug_log=None,
) -> tuple[float, int]:
    """
    Slide a 25-frame window over the track.
    Returns (mean_confidence, best_offset).
    """
    N = len(face_crops)
    if N < WINDOW_FRAMES:
        if debug_log:
            debug_log(f"    _score_track: only {N} frames < {WINDOW_FRAMES} minimum → skip")
        return 0.0, 0

    mfcc_per_frame = 100.0 / fps   # MFCC frames per video frame
    confidences = []
    best_offset = 0
    n_windows = 0

    if debug_log:
        debug_log(
            f"    _score_track: {N} face frames  mfcc={mfcc.shape}  fps={fps:.2f}"
            f"  mfcc_per_frame={mfcc_per_frame:.2f}  device={device}"
        )

    # Slide with stride=5 frames
    for start in range(0, N - WINDOW_FRAMES + 1, max(1, WINDOW_FRAMES // 5)):
        end = start + WINDOW_FRAMES
        n_windows += 1

        # Visual: 5 evenly-spaced frames from the window
        v_indices = np.linspace(start, end - 1, 5, dtype=int)
        faces_5 = np.stack([face_crops[i] for i in v_indices])  # [5, H, W]
        face_t  = torch.from_numpy(faces_5).float().div(255.0)
        face_t  = face_t.unsqueeze(0).unsqueeze(0)  # [1, 1, 5, H, W]

        # Audio: MFCC_PER_FRAME * WINDOW_FRAMES mfcc frames centred on the window
        fi_start = frame_indices[start]
        fi_end   = frame_indices[end - 1]
        a_start  = max(0, int(fi_start * mfcc_per_frame))
        a_end    = a_start + WINDOW_FRAMES * MFCC_PER_FRAME   # 25*4=100 mfcc frames
        if a_end > len(mfcc):
            a_end   = len(mfcc)
            a_start = max(0, a_end - WINDOW_FRAMES * MFCC_PER_FRAME)

        mfcc_slice = mfcc[a_start:a_end]    # [~100, 13]
        target_cols = WINDOW_FRAMES * MFCC_PER_FRAME // 5   # 20
        if len(mfcc_slice) < target_cols:
            mfcc_slice = np.pad(mfcc_slice, ((0, target_cols - len(mfcc_slice)), (0, 0)))
        mfcc_slice = mfcc_slice[:target_cols].T    # [13, 20]
        aud_t = torch.from_numpy(mfcc_slice).float().unsqueeze(0).unsqueeze(0)  # [1,1,13,20]

        with torch.inference_mode():
            face_t = face_t.to(device)
            aud_t  = aud_t.to(device)
            conf   = model(face_t, aud_t).item()

        confidences.append(conf)

        if debug_log and n_windows <= 3:
            debug_log(
                f"    window[{n_windows}]: frames[{start}:{end}]"
                f"  a_mfcc=[{a_start}:{a_end}]"
                f"  face_t={tuple(face_t.shape)}  aud_t={tuple(aud_t.shape)}"
                f"  conf={conf:.4f}"
            )

    if not confidences:
        if debug_log:
            debug_log(f"    _score_track: 0 windows scored")
        return 0.0, 0

    mean_conf = float(np.mean(confidences))
    if debug_log:
        debug_log(
            f"    _score_track done: {n_windows} windows  "
            f"conf min={min(confidences):.3f} max={max(confidences):.3f} mean={mean_conf:.3f}"
        )
    return mean_conf, best_offset


# ─────────────────────────────────────────────────────────────────────────────
# Ray worker
# ─────────────────────────────────────────────────────────────────────────────

@ray.remote(num_gpus=1.0)
class SyncNetWorker:
    def __init__(self):
        self._log = logging.getLogger("syncnet.worker")
        self._log.info(f"Worker init pid={os.getpid()}")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = _load_syncnet(self.device)
        self._log.info("SyncNet loaded")

    def process(self, video_path: str, track_path: str, out_dir: str) -> dict:
        out_path = Path(out_dir) / (Path(video_path).stem + ".sync.json")
        if out_path.exists():
            self._log.info(f"CACHED {Path(video_path).name}")
            return {"path": video_path, "status": "cached"}

        stem = Path(video_path).name
        self._log.info(f"START {stem}")
        t0 = time.perf_counter()
        try:
            track_data = json.loads(Path(track_path).read_text())
            tracks     = track_data.get("tracks", {})
            fps        = float(track_data.get("fps", 25.0))
            self._log.info(f"  {stem}: fps={fps:.2f}  n_tracks={len(tracks)}")

            if not tracks:
                self._log.warning(f"  {stem}: no tracks found in track file — skipping")
                result = {"path": video_path, "fps": fps, "tracks": {}, "status": "ok",
                          "time_s": round(time.perf_counter() - t0, 3)}
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result))
                return {"path": video_path, "status": "ok", "n_tracks": 0}

            # Decode only needed frames
            import av
            needed = set()
            for detections in tracks.values():
                for d in detections:
                    needed.add(d["frame"])
            self._log.info(f"  {stem}: need {len(needed)} unique frames from {len(tracks)} tracks")

            frame_cache: dict[int, np.ndarray] = {}
            with av.open(video_path) as container:
                for fi, frame in enumerate(container.decode(video=0)):
                    if fi in needed:
                        frame_cache[fi] = frame.to_ndarray(format="rgb24")
                    if len(frame_cache) == len(needed):
                        break
            self._log.info(f"  {stem}: decoded {len(frame_cache)}/{len(needed)} needed frames")

            wav  = _extract_audio(video_path)
            if wav is None:
                self._log.warning(f"  {stem}: audio extraction failed — using silence")
                mfcc = np.zeros((1, N_MFCC), dtype=np.float32)
            else:
                mfcc = _compute_mfcc(wav, AUDIO_SR)
                self._log.info(
                    f"  {stem}: audio wav={wav.shape}  mfcc={mfcc.shape}"
                    f"  mfcc_dur={len(mfcc)/100:.1f}s  video_dur={len(needed)/fps:.1f}s"
                )

            sync_tracks = {}
            for track_id, detections in tracks.items():
                frame_indices = [d["frame"] for d in detections]
                bboxes        = [d["bbox"]  for d in detections]

                face_crops = []
                n_missing = 0
                for fi, bbox in zip(frame_indices, bboxes):
                    if fi in frame_cache:
                        face_crops.append(_crop_face(frame_cache[fi], bbox, CROP_PAD))
                    else:
                        face_crops.append(np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8))
                        n_missing += 1

                self._log.info(
                    f"  {stem} track={track_id}: {len(frame_indices)} frames"
                    f"  missing_frames={n_missing}"
                    f"  face_size={face_crops[0].shape if face_crops else 'n/a'}"
                )

                conf, offset = _score_track(
                    face_crops, mfcc, frame_indices, fps,
                    self._model, self.device,
                    debug_log=self._log.info,
                )

                passes = conf >= SYNC_THRESHOLD
                sync_tracks[track_id] = {
                    "frames":     frame_indices,
                    "confidence": round(conf, 4),
                    "offset":     offset,
                    "passes":     passes,
                }
                self._log.info(
                    f"  {stem} track={track_id}: conf={conf:.3f}"
                    f"  threshold={SYNC_THRESHOLD}  passes={passes}"
                )

            n_pass = sum(1 for t in sync_tracks.values() if t["passes"])
            elapsed = time.perf_counter() - t0
            self._log.info(
                f"DONE {stem}: {len(sync_tracks)} tracks  {n_pass} pass  {elapsed:.1f}s"
            )

            result = {
                "path":   video_path,
                "fps":    round(fps, 3),
                "tracks": sync_tracks,
                "status": "ok",
                "time_s": round(elapsed, 3),
            }
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result))

            return {"path": video_path, "status": "ok",
                    "n_tracks": len(sync_tracks), "n_pass": n_pass}

        except Exception as e:
            import traceback
            self._log.error(f"FAILED {stem}\n{traceback.format_exc()}")
            return {"path": video_path, "status": f"failed: {e}", "n_tracks": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global SYNC_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",      required=True)
    ap.add_argument("--track_dir",      required=True)
    ap.add_argument("--out_dir",        required=True)
    ap.add_argument("--num_gpus",       type=int, default=1)
    ap.add_argument("--actors_per_gpu", type=int, default=ACTORS_PER_GPU)
    ap.add_argument("--sync_threshold", type=float, default=SYNC_THRESHOLD,
                    help="Minimum SyncNet confidence to pass a track (default 5.0)")
    args = ap.parse_args()

    SYNC_THRESHOLD = args.sync_threshold

    track_dir = Path(args.track_dir)
    videos    = sorted(Path(args.video_dir).rglob("*.mp4"))

    runnable = []
    for v in videos:
        tp = track_dir / (v.stem + ".tracks.json")
        if tp.exists():
            runnable.append((str(v), str(tp)))
            log.info(f"  queued: {v.name}")
        else:
            log.warning(f"  SKIP {v.name}: no track file at {tp}")

    n_actors      = args.num_gpus * args.actors_per_gpu
    gpu_per_actor = 1.0 / args.actors_per_gpu
    log.info(f"Found {len(videos)} videos  runnable={len(runnable)}")
    log.info(f"track_dir={track_dir}  out_dir={args.out_dir}")
    log.info(f"Actors: {n_actors} ({args.actors_per_gpu}/GPU × {args.num_gpus} GPUs)")

    wandb.init(project="talking-characters", entity="rlx-labs",
               name="syncnet-score", resume="allow",
               config={"sync_threshold": SYNC_THRESHOLD})

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    ActorClass = SyncNetWorker.options(num_gpus=gpu_per_actor)
    workers = [ActorClass.remote() for _ in range(n_actors)]

    futures = [
        workers[i % n_actors].process.remote(vp, tp, args.out_dir)
        for i, (vp, tp) in enumerate(runnable)
    ]

    ok = failed = cached = 0
    total   = len(runnable)
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
        wandb.log({"sync/ok": ok, "sync/cached": cached,
                   "sync/failed": failed, "sync/elapsed_s": elapsed})

    log.info(f"Final: ok={ok}  cached={cached}  failed={failed}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
