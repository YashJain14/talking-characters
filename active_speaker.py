"""
active_speaker.py
-----------------
Stage 2: Active speaker detection using LoCoNet.

For each video + its track file from detect_track.py:
  - Decode full audio at 16 kHz mono
  - For each face track: extract 112×112 face crop windows + mel spectrogram slices
  - Run LoCoNet to score each frame: 0.0 (silent) → 1.0 (speaking)
  - Write per-video ASD result: <stem>.asd.json

ASD result format:
  {
    "path":   str,
    "fps":    float,
    "tracks": {
      "<track_id>": {
        "frames":  [int, ...],       # frame indices
        "scores":  [float, ...],     # LoCoNet speaking probability per frame
      }
    },
    "status": "ok"
  }

Why LoCoNet (CVPR 2024):
  95.2% mAP on AVA-ActiveSpeaker, 68.4% on Ego4D.
  More importantly for this pipeline: +3.0% mAP over TalkNet in multi-speaker
  scenes (exactly what we have). LoCoNet uses long-term intra-speaker
  self-attention + short-term inter-speaker convolution — the two signals that
  disambiguate overlapping speakers.

  Fallback: Light-ASD (94.1% mAP, 0.84M params) if LoCoNet weights are
  unavailable. Set ASD_MODEL=light_asd in environment to force it.

Input geometry:
  - Face crop: 112×112 px (centre-cropped from bbox with crop_pad=0.2 for ASD)
  - Audio: mel spectrogram, 13 MFCCs, 0.04s hop = 25 fps aligned with video

GPU assignment:
  0.5 GPU/actor → 8 concurrent actors across 4 A100s. LoCoNet (34M params)
  fits comfortably at 0.5 GPU in fp32. Audio preprocessing on CPU.
  Decode uses PyAV (CPU) — same fractional-GPU constraint as the main pipeline.

Usage:
  python active_speaker.py \\
    --video_dir  $SCRATCH_TC/raw_videos \\
    --track_dir  $SCRATCH_TC/tracks \\
    --out_dir    $SCRATCH_TC/asd \\
    --num_gpus   4
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


ASD_MODEL      = os.environ.get("ASD_MODEL", "loconet")   # "loconet" or "light_asd"
ACTORS_PER_GPU = 2     # 0.5 GPU/actor → 8 concurrent across 4 A100s
FACE_SIZE      = 112    # ASD model input face crop size
CROP_PAD_ASD   = 0.20   # tighter pad for ASD than export (0.35)
AUDIO_SR       = 16000  # 16 kHz mono

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("active_speaker")


def _extract_audio(video_path: str, sr: int = AUDIO_SR) -> np.ndarray | None:
    """Extract mono audio at sr Hz using ffmpeg via soundfile+subprocess."""
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


def _mel_for_frames(wav: np.ndarray, sr: int, frame_indices: list[int],
                    fps: float) -> np.ndarray:
    """
    Build mel spectrogram slices aligned to video frames.
    Returns [N_frames, 13, T_mels] float32.
    """
    import python_speech_features
    mfcc = python_speech_features.mfcc(
        wav, sr, numcep=13, winlen=0.025, winstep=0.010,
    )  # shape [T_mels, 13]

    mels_per_frame = sr / (fps * 160)   # approx mel hops per video frame
    slices = []
    window = 4  # ±4 mel frames either side
    for fi in frame_indices:
        center = int(fi * mels_per_frame)
        start  = max(0, center - window)
        end    = min(len(mfcc), center + window + 1)
        chunk  = mfcc[start:end]
        # Pad to fixed length
        pad    = 2 * window + 1 - len(chunk)
        if pad > 0:
            chunk = np.pad(chunk, ((0, pad), (0, 0)))
        slices.append(chunk[:2*window+1])  # [9, 13]
    return np.array(slices, dtype=np.float32)  # [N, 9, 13]


def _crop_face(frame: np.ndarray, bbox: list, pad: float) -> np.ndarray:
    """Centre-crop a face from frame with proportional padding, resize to 112×112."""
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
    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)


def _load_loconet(device: str):
    """
    Load LoCoNet weights.  Expects weights at:
      ~/.cache/talking_characters/loconet.pth
    or LOCONET_WEIGHTS env var.
    """
    weights = os.environ.get("LOCONET_WEIGHTS",
                             str(Path.home() / ".cache" / "talking_characters" / "loconet.pth"))
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"LoCoNet weights not found at {weights}. "
            "Run prefetch_models_talking_characters.py first."
        )
    import sys
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from loconet import LoCoNet
    model = LoCoNet()
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _load_light_asd(device: str):
    """
    Load Light-ASD weights.  Expects weights at:
      ~/.cache/talking_characters/light_asd.pth
    or LIGHT_ASD_WEIGHTS env var.
    """
    weights = os.environ.get("LIGHT_ASD_WEIGHTS",
                             str(Path.home() / ".cache" / "talking_characters" / "light_asd.pth"))
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Light-ASD weights not found at {weights}. "
            "Run prefetch_models_talking_characters.py first."
        )
    from light_asd import LightASD
    model = LightASD()
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


@ray.remote(num_gpus=1.0)
class ASDWorker:
    def __init__(self, asd_model_name: str):
        self._log    = logging.getLogger("asd.worker")
        self._log.info(f"Worker init pid={os.getpid()}  model={asd_model_name}")
        self.device  = "cuda:0"
        self._model_name = asd_model_name
        if asd_model_name == "loconet":
            self._model = _load_loconet(self.device)
        else:
            self._model = _load_light_asd(self.device)
        self._log.info(f"{asd_model_name} loaded")

    def process(self, video_path: str, track_path: str, out_dir: str) -> dict:
        out_path = Path(out_dir) / (Path(video_path).stem + ".asd.json")
        if out_path.exists():
            return {"path": video_path, "status": "cached"}

        t0 = time.perf_counter()
        try:
            track_data = json.loads(Path(track_path).read_text())
            tracks     = track_data.get("tracks", {})
            fps        = float(track_data.get("fps", 25.0))
            n_frames   = int(track_data.get("n_frames", 0))

            if not tracks:
                result = {"path": video_path, "fps": fps, "tracks": {}, "status": "ok",
                          "time_s": round(time.perf_counter() - t0, 3)}
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result))
                return {"path": video_path, "status": "ok", "n_tracks": 0}

            # Collect only the frame indices needed across all tracks, then
            # decode once — avoids loading the full video into RAM.
            import av
            needed = set()
            for detections in tracks.values():
                for d in detections:
                    needed.add(d["frame"])

            frame_cache: dict[int, np.ndarray] = {}
            with av.open(video_path) as container:
                for fi, frame in enumerate(container.decode(video=0)):
                    if fi in needed:
                        frame_cache[fi] = frame.to_ndarray(format="rgb24")
                    if len(frame_cache) == len(needed):
                        break  # got everything we need

            wav = _extract_audio(video_path)

            asd_tracks = {}
            for track_id, detections in tracks.items():
                frame_indices = [d["frame"] for d in detections]
                bboxes        = [d["bbox"]  for d in detections]

                # Build face crop sequence
                face_crops = []
                for fi, bbox in zip(frame_indices, bboxes):
                    if fi in frame_cache:
                        crop = _crop_face(frame_cache[fi], bbox, CROP_PAD_ASD)
                    else:
                        crop = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
                    face_crops.append(crop)

                face_tensor = (
                    torch.from_numpy(np.stack(face_crops))          # [N, H, W, 3]
                    .permute(0, 3, 1, 2)                            # [N, 3, H, W]
                    .float().div(255.0)
                    .to(self.device)
                )

                if wav is not None:
                    mel = _mel_for_frames(wav, AUDIO_SR, frame_indices, fps)
                    mel_tensor = torch.from_numpy(mel).to(self.device)  # [N, 9, 13]
                else:
                    mel_tensor = torch.zeros(len(frame_indices), 9, 13,
                                             device=self.device)

                with torch.inference_mode():
                    scores = self._model(face_tensor, mel_tensor)  # [N] float
                    if isinstance(scores, torch.Tensor):
                        scores = torch.sigmoid(scores).cpu().numpy().tolist()
                    else:
                        scores = list(scores)

                asd_tracks[track_id] = {
                    "frames": frame_indices,
                    "scores": [round(float(s), 4) for s in scores],
                }

            result = {
                "path":   video_path,
                "fps":    round(fps, 3),
                "tracks": asd_tracks,
                "status": "ok",
                "time_s": round(time.perf_counter() - t0, 3),
            }
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result))
            return {"path": video_path, "status": "ok",
                    "n_tracks": len(asd_tracks)}

        except Exception as e:
            import traceback
            self._log.error(f"FAILED {Path(video_path).name}\n{traceback.format_exc()}")
            return {"path": video_path, "status": f"failed: {e}", "n_tracks": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",  required=True)
    ap.add_argument("--track_dir",  required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--num_gpus",       type=int, default=1)
    ap.add_argument("--actors_per_gpu", type=int, default=ACTORS_PER_GPU,
                    help="Ray actors per GPU. Default 2 (0.5 GPU each) for A100. "
                         "Use 1 on single-GPU machines with limited RAM (e.g. Colab T4).")
    ap.add_argument("--asd_model",  default=ASD_MODEL,
                    choices=["loconet", "light_asd"],
                    help="ASD model to use (loconet recommended for multi-speaker)")
    args = ap.parse_args()

    track_dir = Path(args.track_dir)
    videos    = sorted(Path(args.video_dir).rglob("*.mp4"))

    # Only process videos that have a track file
    runnable = []
    for v in videos:
        tp = track_dir / (v.stem + ".tracks.json")
        if tp.exists():
            runnable.append((str(v), str(tp)))
        else:
            log.warning(f"No track file for {v.name} — skipping")

    n_actors      = args.num_gpus * args.actors_per_gpu
    gpu_per_actor = 1.0 / args.actors_per_gpu
    log.info(f"ASD on {len(runnable)} videos  model={args.asd_model}")
    log.info(f"Actors: {n_actors} ({args.actors_per_gpu}/GPU × {args.num_gpus} GPUs, "
             f"{gpu_per_actor:.2f} GPU each)")

    wandb.init(project="talking-characters", entity="rlx-labs",
               name="active-speaker", resume="allow",
               config={"asd_model": args.asd_model})

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    ActorClass = ASDWorker.options(num_gpus=gpu_per_actor)
    workers = [ActorClass.remote(args.asd_model) for _ in range(n_actors)]

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
        wandb.log({"asd/ok": ok, "asd/cached": cached,
                   "asd/failed": failed, "asd/elapsed_s": elapsed})

    log.info(f"Final: ok={ok}  cached={cached}  failed={failed}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
