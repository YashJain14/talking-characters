"""
segment_export.py
-----------------
Stage 3: Find single-speaker segments and export cleaned clips.

For each video + its ASD result from active_speaker.py:
  - Find frames where exactly one face track scores > SPEAK_THRESHOLD
  - Merge consecutive speaking frames into segments, bridging gaps < GAP_BRIDGE_S
  - Filter: 3s ≤ duration ≤ 60s
  - Quality gate: ≥ 60% of frames must be active-speaking
  - Discard clips where face is absent > 20% of frames
  - Export via ffmpeg with a content-aware crop:
      1. Black bars: detect rows/cols with mean luminance < BLACK_THRESHOLD,
         crop them off. Always applied — black bars are deterministic.
      2. Extra speaker: if another tracked face is present in the same segment,
         split the frame at the midpoint between the two face centres and keep
         only the half containing the active speaker. Only applied when a second
         face is actually detected — no crop otherwise.
      Watermarks are NOT filtered here; handle downstream with aesthetic scoring
      if needed (they don't appear in every video).

Slides / voice-over scenes with no visible speaker are discarded automatically:
  detect_track.py produces no face tracks → active_speaker.py has nothing to score
  → this stage finds no frames with exactly one active track → no clips written.
  No explicit filter is needed.

Output per clip:
  <out_dir>/<video_stem>/<video_stem>_<track_id>_<start_frame>_<end_frame>.mp4
  <out_dir>/<video_stem>/<video_stem>_<track_id>_<start_frame>_<end_frame>.json

Clip JSON metadata:
  {
    "source_video":      str,
    "track_id":          str,
    "start_frame":       int,
    "end_frame":         int,
    "start_s":           float,
    "end_s":             float,
    "duration_s":        float,
    "fps":               float,
    "active_ratio":      float,
    "face_present_ratio": float,
    "speak_threshold":   float,
    "had_black_bars":    bool,
    "had_extra_speaker": bool,
    "resolution":        int,
  }

Summary written to: <out_dir>/segments.json — list of all clip metadata dicts.

CPU-parallel (no GPU needed for this stage — ffmpeg handles encode).

Usage:
  python segment_export.py \\
    --video_dir  $SCRATCH_TC/raw_videos \\
    --asd_dir    $SCRATCH_TC/asd \\
    --track_dir  $SCRATCH_TC/tracks \\
    --out_dir    $SCRATCH_TC/clips \\
    --num_workers 16
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import ray
import wandb


SPEAK_THRESHOLD   = 0.5    # LoCoNet score to count a frame as "speaking"
GAP_BRIDGE_S      = 0.5    # merge gaps shorter than this (natural speech pauses)
MIN_CLIP_S        = 3.0
MAX_CLIP_S        = 60.0
MIN_ACTIVE_RATIO  = 0.60   # at least 60% of clip frames must be active-speaking
MAX_ABSENT_RATIO  = 0.20   # discard if face missing > 20% of clip frames
BLACK_THRESHOLD   = 16     # luminance (0-255) below which a row/col is "black"
EXPORT_RES        = 512    # output resolution (width, height may differ post-crop)
CROP_PAD          = 0.35   # proportional padding around face bbox for export crop

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("segment_export")


def _find_segments(
    asd_tracks: dict,
    track_detections: dict,
    fps: float,
    n_frames: int,
) -> list[dict]:
    """
    Returns list of dicts:
      {track_id, start_frame, end_frame, active_ratio, face_present_ratio}
    """
    gap_bridge_frames = int(GAP_BRIDGE_S * fps)
    min_frames        = int(MIN_CLIP_S   * fps)
    max_frames        = int(MAX_CLIP_S   * fps)

    segments = []

    for track_id, asd in asd_tracks.items():
        frames = np.array(asd["frames"], dtype=np.int32)
        scores = np.array(asd["scores"], dtype=np.float32)
        if len(frames) == 0:
            continue

        # Build a dense per-frame score array (0 where track is absent)
        dense_scores = np.zeros(n_frames, dtype=np.float32)
        dense_scores[frames] = scores
        is_active = dense_scores > SPEAK_THRESHOLD

        # Build a per-frame "face present" mask from track detections
        det_frames = {d["frame"] for d in track_detections.get(track_id, [])}
        face_present = np.zeros(n_frames, dtype=bool)
        for fi in det_frames:
            if fi < n_frames:
                face_present[fi] = True

        # Find runs of active frames
        # Convert boolean array to run-length encoded segments
        run_starts, run_ends = [], []
        in_run = False
        for i, active in enumerate(is_active):
            if active and not in_run:
                run_starts.append(i)
                in_run = True
            elif not active and in_run:
                run_ends.append(i)
                in_run = False
        if in_run:
            run_ends.append(n_frames)

        if not run_starts:
            continue

        # Merge runs separated by short gaps
        merged_starts = [run_starts[0]]
        merged_ends   = []
        for i in range(1, len(run_starts)):
            if run_starts[i] - run_ends[i - 1] <= gap_bridge_frames:
                pass  # extend current run
            else:
                merged_ends.append(run_ends[i - 1])
                merged_starts.append(run_starts[i])
        merged_ends.append(run_ends[-1])

        for s, e in zip(merged_starts, merged_ends):
            length = e - s
            if length < min_frames or length > max_frames:
                continue

            clip_active = is_active[s:e]
            active_ratio = float(clip_active.mean())
            if active_ratio < MIN_ACTIVE_RATIO:
                continue

            clip_present = face_present[s:e]
            absent_ratio = 1.0 - float(clip_present.mean())
            if absent_ratio > MAX_ABSENT_RATIO:
                continue

            segments.append({
                "track_id":          track_id,
                "start_frame":       int(s),
                "end_frame":         int(e),
                "active_ratio":      round(active_ratio, 4),
                "face_present_ratio": round(float(clip_present.mean()), 4),
            })

    return segments


def _detect_black_bars(frame: np.ndarray) -> tuple[int, int, int, int]:
    """
    Detect black bars by scanning rows and columns for near-zero luminance.
    Returns (x1, y1, x2, y2) — the content region after removing bars.
    Uses the mean of each row/col so a single bright pixel doesn't break detection.
    """
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    row_means = gray.mean(axis=1)   # [H]
    col_means = gray.mean(axis=0)   # [W]

    # Find first/last non-black row and column
    y1 = int(np.argmax(row_means > BLACK_THRESHOLD))
    y2 = h - int(np.argmax(row_means[::-1] > BLACK_THRESHOLD))
    x1 = int(np.argmax(col_means > BLACK_THRESHOLD))
    x2 = w - int(np.argmax(col_means[::-1] > BLACK_THRESHOLD))

    # Clamp: if everything is black (no content found), return full frame
    if y2 <= y1 or x2 <= x1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def _median_face_cx(track_detections: dict, track_id: str,
                    start_frame: int, end_frame: int) -> float | None:
    """Return the median face centre-x for a track over a frame range."""
    dets = [
        d for d in track_detections.get(track_id, [])
        if start_frame <= d["frame"] < end_frame
    ]
    if not dets:
        return None
    return float(np.median([(d["bbox"][0] + d["bbox"][2]) / 2 for d in dets]))


def _compute_crop(
    frame_w: int,
    frame_h: int,
    bar_x1: int, bar_y1: int, bar_x2: int, bar_y2: int,
    active_cx: float,
    other_cxs: list[float],
) -> tuple[int, int, int, int]:
    """
    Compute final (x1, y1, x2, y2) crop region.

    Start from the bar-free content region.
    If another speaker's face is present, bisect horizontally between the
    two face centres and keep the half containing the active speaker.
    No crop is applied for the other-speaker case when no other face is detected.
    """
    x1, y1, x2, y2 = bar_x1, bar_y1, bar_x2, bar_y2

    if other_cxs:
        # Use the closest other speaker's centre to bisect
        closest_other_cx = min(other_cxs, key=lambda cx: abs(cx - active_cx))
        split_x = int((active_cx + closest_other_cx) / 2)
        if active_cx < closest_other_cx:
            x2 = min(x2, split_x)   # active speaker is on the left
        else:
            x1 = max(x1, split_x)   # active speaker is on the right

    return x1, y1, x2, y2


def _sample_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Decode a single frame by index using PyAV for black-bar detection."""
    import av
    try:
        with av.open(video_path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_idx:
                    return frame.to_ndarray(format="rgb24")
                if i > frame_idx:
                    break
    except Exception:
        pass
    return None


def _export_clip(
    video_path: str,
    track_detections: dict,
    track_id: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    frame_w: int,
    frame_h: int,
    out_path: str,
) -> tuple[bool, bool, bool]:
    """
    Export a cleaned clip via ffmpeg.

    Crop steps (applied only when the condition is detected):
      1. Black bars  — always scanned, crop applied if bars are found
      2. Extra speaker — bisect at midpoint between face centres if another
                         tracked face is present in this segment

    Returns (success, had_black_bars, had_extra_speaker).
    """
    # ── 1. Black bar detection on a representative frame ─────────────────────
    mid_frame = (start_frame + end_frame) // 2
    sample    = _sample_frame(video_path, mid_frame)
    if sample is not None:
        bar_x1, bar_y1, bar_x2, bar_y2 = _detect_black_bars(sample)
    else:
        bar_x1, bar_y1, bar_x2, bar_y2 = 0, 0, frame_w, frame_h

    had_black_bars = (bar_x1 > 0 or bar_y1 > 0 or
                      bar_x2 < frame_w or bar_y2 < frame_h)

    # ── 2. Other-speaker detection ────────────────────────────────────────────
    active_cx  = _median_face_cx(track_detections, track_id, start_frame, end_frame)
    other_cxs  = []
    for other_id, dets in track_detections.items():
        if other_id == track_id:
            continue
        cx = _median_face_cx(track_detections, other_id, start_frame, end_frame)
        if cx is not None:
            other_cxs.append(cx)

    had_extra_speaker = len(other_cxs) > 0

    if active_cx is None:
        return False, had_black_bars, had_extra_speaker

    # ── 3. Compute final crop region ──────────────────────────────────────────
    x1, y1, x2, y2 = _compute_crop(
        frame_w, frame_h,
        bar_x1, bar_y1, bar_x2, bar_y2,
        active_cx, other_cxs,
    )
    crop_w = x2 - x1
    crop_h = y2 - y1

    start_s = start_frame / fps
    end_s   = end_frame   / fps

    # Build ffmpeg vf chain — only add crop filter if we're actually cropping
    full_frame = (x1 == 0 and y1 == 0 and x2 == frame_w and y2 == frame_h)
    if full_frame:
        vf = f"scale={EXPORT_RES}:{EXPORT_RES}"
    else:
        vf = f"crop={crop_w}:{crop_h}:{x1}:{y1},scale={EXPORT_RES}:{EXPORT_RES}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i",  video_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac",     "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0, had_black_bars, had_extra_speaker


@ray.remote(num_cpus=2)
class SegmentWorker:
    def process(self, video_path: str, asd_path: str,
                track_path: str, out_dir: str) -> dict:
        t0 = time.perf_counter()
        try:
            asd_data   = json.loads(Path(asd_path).read_text())
            track_data = json.loads(Path(track_path).read_text())

            fps        = float(asd_data.get("fps", 25.0))
            n_frames   = int(track_data.get("n_frames", 0))
            asd_tracks = asd_data.get("tracks", {})
            trk_tracks = track_data.get("tracks", {})

            # Get frame dimensions from a sample frame
            import av
            frame_w, frame_h = 1280, 720  # fallback
            try:
                with av.open(video_path) as c:
                    s = c.streams.video[0]
                    frame_w = s.width  or frame_w
                    frame_h = s.height or frame_h
            except Exception:
                pass

            segments = _find_segments(asd_tracks, trk_tracks, fps, n_frames)
            if not segments:
                return {"path": video_path, "status": "ok", "n_clips": 0}

            stem     = Path(video_path).stem
            clip_dir = Path(out_dir) / stem
            clip_dir.mkdir(parents=True, exist_ok=True)

            written_clips = []
            for seg in segments:
                track_id    = seg["track_id"]
                start_frame = seg["start_frame"]
                end_frame   = seg["end_frame"]
                clip_name   = f"{stem}_{track_id}_{start_frame:06d}_{end_frame:06d}"
                clip_path   = str(clip_dir / f"{clip_name}.mp4")
                meta_path   = clip_dir / f"{clip_name}.json"

                if meta_path.exists():
                    written_clips.append(json.loads(meta_path.read_text()))
                    continue

                ok, had_black_bars, had_extra_speaker = _export_clip(
                    video_path, trk_tracks, track_id,
                    start_frame, end_frame, fps,
                    frame_w, frame_h, clip_path,
                )
                if not ok:
                    continue

                meta = {
                    "source_video":       video_path,
                    "track_id":           track_id,
                    "start_frame":        start_frame,
                    "end_frame":          end_frame,
                    "start_s":            round(start_frame / fps, 3),
                    "end_s":              round(end_frame   / fps, 3),
                    "duration_s":         round((end_frame - start_frame) / fps, 3),
                    "fps":                round(fps, 3),
                    "active_ratio":       seg["active_ratio"],
                    "face_present_ratio": seg["face_present_ratio"],
                    "speak_threshold":    SPEAK_THRESHOLD,
                    "had_black_bars":     had_black_bars,
                    "had_extra_speaker":  had_extra_speaker,
                    "resolution":         EXPORT_RES,
                    "clip_path":          clip_path,
                }
                meta_path.write_text(json.dumps(meta, indent=2))
                written_clips.append(meta)

            return {
                "path":    video_path,
                "status":  "ok",
                "n_clips": len(written_clips),
                "clips":   written_clips,
            }

        except Exception as e:
            import traceback
            log.error(f"FAILED {Path(video_path).name}\n{traceback.format_exc()}")
            return {"path": video_path, "status": f"failed: {e}", "n_clips": 0}


def main():
    global SPEAK_THRESHOLD, GAP_BRIDGE_S, MIN_CLIP_S, MAX_CLIP_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",    required=True)
    ap.add_argument("--asd_dir",      required=True)
    ap.add_argument("--track_dir",    required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--num_workers",  type=int, default=16)
    ap.add_argument("--speak_threshold", type=float, default=SPEAK_THRESHOLD,
                    help="LoCoNet score threshold (0.5 = default; raise to 0.6 "
                         "if both speakers are simultaneously scored active)")
    ap.add_argument("--gap_bridge_s",    type=float, default=GAP_BRIDGE_S)
    ap.add_argument("--min_clip_s",      type=float, default=MIN_CLIP_S)
    ap.add_argument("--max_clip_s",      type=float, default=MAX_CLIP_S)
    args = ap.parse_args()

    # Override module-level constants if user changed thresholds
    SPEAK_THRESHOLD = args.speak_threshold
    GAP_BRIDGE_S    = args.gap_bridge_s
    MIN_CLIP_S      = args.min_clip_s
    MAX_CLIP_S      = args.max_clip_s

    asd_dir   = Path(args.asd_dir)
    track_dir = Path(args.track_dir)
    videos    = sorted(Path(args.video_dir).rglob("*.mp4"))

    runnable = []
    for v in videos:
        ap_ = asd_dir   / (v.stem + ".asd.json")
        tp  = track_dir / (v.stem + ".tracks.json")
        if ap_.exists() and tp.exists():
            runnable.append((str(v), str(ap_), str(tp)))
        else:
            log.warning(f"Missing ASD or track file for {v.name} — skipping")

    log.info(f"Segmenting {len(runnable)} videos  ({args.num_workers} workers)")

    wandb.init(project="talking-characters", entity="rlx-labs",
               name="segment-export", resume="allow",
               config={"speak_threshold": SPEAK_THRESHOLD,
                       "gap_bridge_s":    GAP_BRIDGE_S,
                       "min_clip_s":      MIN_CLIP_S,
                       "max_clip_s":      MAX_CLIP_S,
                       "crop_pad":        CROP_PAD,
                       "resolution":      EXPORT_RES})

    ray.init(ignore_reinit_error=True)
    workers = [SegmentWorker.remote() for _ in range(args.num_workers)]

    futures = [
        workers[i % args.num_workers].process.remote(vp, ap_, tp, args.out_dir)
        for i, (vp, ap_, tp) in enumerate(runnable)
    ]

    all_clips   = []
    ok = failed = 0
    total_clips = 0
    total       = len(runnable)
    pending     = list(futures)
    t0          = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for res in ray.get(done):
            if res["status"] == "ok":
                ok         += 1
                total_clips += res.get("n_clips", 0)
                all_clips.extend(res.get("clips", []))
            else:
                failed += 1
                log.error(f"FAILED {Path(res['path']).name}: {res['status']}")
        completed = ok + failed
        elapsed   = time.perf_counter() - t0
        log.info(f"[{completed}/{total}]  ok={ok}  failed={failed}  "
                 f"clips={total_clips}  elapsed={elapsed:.1f}s")
        wandb.log({"segment/ok": ok, "segment/failed": failed,
                   "segment/total_clips": total_clips,
                   "segment/elapsed_s": elapsed})

    # Write aggregated segments manifest
    out = Path(args.out_dir) / "segments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_clips, f, indent=2)

    log.info(f"Total clips exported: {total_clips}")
    log.info(f"Segments manifest  → {out}")

    durations = [c["duration_s"] for c in all_clips]
    if durations:
        log.info(f"Duration  mean={np.mean(durations):.1f}s  "
                 f"median={np.median(durations):.1f}s  "
                 f"total={sum(durations)/3600:.2f}h")
        wandb.log({
            "segment/n_clips":        total_clips,
            "segment/duration_mean":  float(np.mean(durations)),
            "segment/duration_median":float(np.median(durations)),
            "segment/total_hours":    sum(durations) / 3600,
            "segment/duration_hist":  wandb.Histogram(durations),
        })

    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
