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

SYNC_THRESHOLD = 1.0     # min confidence to pass (median_dist - min_dist); overridden by --sync_threshold
FACE_SIZE      = 224     # face crop size fed to SyncNet visual stream (official repo uses 224)
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
# SyncNet model — exact copy of syncnet_python/SyncNetModel.py class S
#
# Visual input : [B, 3, 5, H, W]  (RGB, 5 frames, H/W = face crop size)
# Audio input  : [B, 1, 13, 20]   (1ch MFCC, 13 coeffs, 20 time steps)
# Both streams output 1024-d embeddings.
# Sync confidence = median(pairwise_dist) - min(pairwise_dist)
# ─────────────────────────────────────────────────────────────────────────────

class SyncNet(nn.Module):
    def __init__(self, num_layers_in_fc_layers: int = 1024):
        super().__init__()

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        mid = mid.view(mid.size(0), -1)
        return self.netfclip(mid)

    def forward_aud(self, x):
        mid = self.netcnnaud(x)
        mid = mid.view(mid.size(0), -1)
        return self.netfcaud(mid)


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
    # Use the same loading method as the official syncnet_python repo:
    # copy_ each param by name rather than load_state_dict (avoids strict mismatch)
    loaded_state = torch.load(weights, map_location="cpu", weights_only=True)
    self_state   = model.state_dict()
    log.info(f"SyncNet checkpoint keys: {sorted({k.split('.')[0] for k in loaded_state})}")
    for name, param in loaded_state.items():
        if name in self_state:
            self_state[name].copy_(param)
        else:
            log.warning(f"SyncNet: unexpected key in checkpoint: {name}")
    missing = [k for k in self_state if k not in loaded_state]
    if missing:
        log.warning(f"SyncNet: {len(missing)} keys not in checkpoint: {missing[:3]}")
    else:
        log.info("SyncNet: all weights loaded")
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _extract_audio(video_path: str, sr: int = AUDIO_SR) -> np.ndarray | None:
    import subprocess, tempfile, soundfile as sf
    ffmpeg = _ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-i", video_path,
             "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        if r.returncode != 0:
            log.warning(
                f"ffmpeg audio extract failed (exit={r.returncode}) for {Path(video_path).name}:\n"
                + r.stderr.decode(errors="replace")[-500:]
            )
            return None
        if not Path(tmp_path).exists() or Path(tmp_path).stat().st_size < 100:
            log.warning(f"audio WAV missing or empty for {Path(video_path).name}")
            return None
        wav, _ = sf.read(tmp_path, dtype="float32")
        return wav
    except Exception as e:
        log.warning(f"_extract_audio exception for {Path(video_path).name}: {e}")
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _compute_mfcc(wav: np.ndarray, sr: int) -> np.ndarray:
    """Compute MFCC at 100 fps. Returns [13, T_mfcc] float32 (coefficients × time)."""
    import python_speech_features
    mfcc = python_speech_features.mfcc(
        wav, sr, numcep=N_MFCC, winlen=0.025, winstep=0.010,
    )
    return mfcc.T.astype(np.float32)   # [13, T_mfcc]


def _crop_face(frame: np.ndarray, bbox: list, pad: float) -> np.ndarray:
    """Returns RGB crop resized to FACE_SIZE×FACE_SIZE, uint8."""
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
        return np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE))  # RGB, uint8


def _score_track(
    face_crops: list[np.ndarray],   # [N] each (FACE_SIZE, FACE_SIZE, 3) uint8 RGB
    mfcc: np.ndarray,               # [13, T_mfcc]  — transposed, 13 rows = coefficients
    frame_indices: list[int],
    fps: float,
    model: SyncNet,
    device: str,
    vshift: int = 15,
    batch_size: int = 20,
    debug_log=None,
) -> tuple[float, int]:
    """
    Score a track using the official SyncNetInstance.evaluate approach:
      - Extract lip embedding per frame-window using forward_lip
      - Extract audio embedding per frame-window using forward_aud
      - Compute pairwise distance with temporal shift ±vshift
      - confidence = median(mean_dist) - min(mean_dist)
    Returns (confidence, offset_frames).
    """
    N = len(face_crops)
    if N < 5:
        if debug_log:
            debug_log(f"    _score_track: only {N} frames < 5 minimum → skip")
        return 0.0, 0

    mfcc_per_frame = 100.0 / fps  # MFCC frames per video frame (≈4 at 25fps)

    if debug_log:
        debug_log(
            f"    _score_track: {N} face frames  mfcc={mfcc.shape}  fps={fps:.2f}"
            f"  mfcc_per_frame={mfcc_per_frame:.2f}  device={device}"
        )

    # Build face tensor: [N_total, 3, 5, H, W] — one entry per sliding position
    # Official code: imtv shape is [1, T, H, W, C] → transposed to [1, C, T, H, W]
    # then batch is [B, 3, 5, H, W]
    lastframe = N - 4   # need 5 consecutive frames per window
    if lastframe <= 0:
        return 0.0, 0

    # Stack all frames as a float tensor [N, H, W, 3] → [1, N, H, W, 3]
    frames_np = np.stack(face_crops, axis=0).astype(np.float32)  # [N, H, W, 3]
    # Transpose to [1, 3, N, H, W] matching official imtv layout
    imtv = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, N, H, W]

    # MFCC tensor: official cct shape is [1, 1, 13, T_mfcc]
    cct = torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, 13, T]

    im_feat_list = []
    cc_feat_list = []

    with torch.inference_mode():
        for i in range(0, lastframe, batch_size):
            # Video batch: [B, 3, 5, H, W]
            im_batch = [imtv[:, :, vf:vf+5, :, :] for vf in range(i, min(lastframe, i+batch_size))]
            im_in    = torch.cat(im_batch, 0).to(device)
            im_out   = model.forward_lip(im_in)
            im_feat_list.append(im_out.cpu())

            # Audio batch: [B, 1, 13, 20]  — frame vf maps to mfcc col vf*mfcc_per_frame
            cc_batch = []
            for vf in range(i, min(lastframe, i+batch_size)):
                fi      = frame_indices[vf]
                a_start = int(fi * mfcc_per_frame)
                a_end   = a_start + 20
                if a_end > cct.shape[3]:
                    a_end   = cct.shape[3]
                    a_start = max(0, a_end - 20)
                cc_slice = cct[:, :, :, a_start:a_end]
                if cc_slice.shape[3] < 20:
                    cc_slice = F.pad(cc_slice, (0, 20 - cc_slice.shape[3]))
                cc_batch.append(cc_slice)
            cc_in  = torch.cat(cc_batch, 0).to(device)
            cc_out = model.forward_aud(cc_in)
            cc_feat_list.append(cc_out.cpu())

    im_feat = torch.cat(im_feat_list, 0)  # [lastframe, 1024]
    cc_feat = torch.cat(cc_feat_list, 0)  # [lastframe, 1024]

    if debug_log:
        debug_log(f"    im_feat={tuple(im_feat.shape)}  cc_feat={tuple(cc_feat.shape)}")

    # Pairwise distance with temporal shift  (official calc_pdist)
    win_size = vshift * 2 + 1
    feat2p   = F.pad(cc_feat, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(im_feat)):
        dists.append(
            F.pairwise_distance(
                im_feat[[i], :].repeat(win_size, 1),
                feat2p[i:i+win_size, :],
            )
        )

    mdist  = torch.mean(torch.stack(dists, 1), 1)   # [win_size]
    minval, minidx = torch.min(mdist, 0)
    conf   = float(torch.median(mdist) - minval)
    offset = int(vshift - minidx.item())

    if debug_log:
        debug_log(
            f"    mdist min={minval.item():.3f}  median={torch.median(mdist).item():.3f}"
            f"  conf={conf:.3f}  offset={offset}"
        )

    return conf, offset


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
                mfcc = np.zeros((N_MFCC, 1), dtype=np.float32)
            else:
                mfcc = _compute_mfcc(wav, AUDIO_SR)
                self._log.info(
                    f"  {stem}: audio wav={wav.shape}  mfcc={mfcc.shape}"
                    f"  mfcc_dur={mfcc.shape[1]/100:.1f}s  video_dur={len(needed)/fps:.1f}s"
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
                        face_crops.append(np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8))
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
