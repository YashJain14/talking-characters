"""
segment_export.py
-----------------
Stage 3: Export single-speaker clips from SyncNet-scored tracks.

For each video + its sync result from syncnet_score.py:
  - Keep only tracks that passed the SyncNet confidence threshold
  - For each passing track: treat the full track span as one candidate clip,
    then split on long gaps (> GAP_BRIDGE_S of absence) to produce sub-clips
  - Quality gate: clip must contain the face ≥ MIN_FACE_RATIO of frames
  - Duration gate: MIN_CLIP_S ≤ duration ≤ MAX_CLIP_S
  - Export via ffmpeg with content-aware crop:
      1. Black bars  — scan row/col luminance, crop if found (always applied)
      2. Other faces — if another face track is present in this clip's frame
                       range, bisect at the midpoint and keep the active half
  - Write per-clip JSON metadata

Output:
  <out_dir>/<video_stem>/<stem>_<track_id>_<start_frame>_<end_frame>.mp4
  <out_dir>/<video_stem>/<stem>_<track_id>_<start_frame>_<end_frame>.json
  <out_dir>/segments.json — aggregated manifest

Usage:
  python segment_export.py \\
    --video_dir  $SCRATCH_TC/raw_videos \\
    --sync_dir   $SCRATCH_TC/sync \\
    --track_dir  $SCRATCH_TC/tracks \\
    --out_dir    $SCRATCH_TC/clips
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ray
import wandb

GAP_BRIDGE_S    = 0.5    # bridge track gaps shorter than this (seconds)
MIN_CLIP_S      = 3.0
MAX_CLIP_S      = 60.0
MIN_FACE_RATIO  = 0.80   # face must be detected in ≥ 80% of clip frames
BLACK_THRESHOLD = 16     # luminance below which a row/col is considered black bar
EXPORT_RES      = 512    # output resolution (square)
MIN_FACE_PX_SOLO = 150   # clips with face < this must be solo (n_faces_in_frame == 0)

# Per-scene confidence thresholds based on how many other faces share the frame.
# Solo close-ups score 1.0–3.0; two-person shots score 0.3–0.9 (nodders included).
# Requiring 0.8 for two-person shots filters nodders (0.3–0.5) while keeping
# real speakers (1.0+). Panel shots (≥2 others) are rejected entirely.
CONF_THRESHOLD_BY_N_OTHER = {
    0: 0.3,   # solo: low bar, SyncNet is reliable on close-ups
    1: 0.8,   # two-person: high bar to filter nodders
}
# any n_other >= 2 → rejected (panel shot, face too small to crop cleanly)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("segment_export")


def _find_clips(
    sync_tracks: dict,
    trk_tracks:  dict,
    fps: float,
) -> tuple[list[dict], list[dict]]:
    """
    For each passing track, split on long gaps to produce sub-clips.
    Returns (accepted_clips, rejected_clips), each a list of dicts with reject_reason on rejected.
    """
    gap_frames  = int(GAP_BRIDGE_S * fps)
    min_frames  = int(MIN_CLIP_S   * fps)
    max_frames  = int(MAX_CLIP_S   * fps)
    clips    = []
    rejected = []

    for track_id, sync in sync_tracks.items():
        if not sync.get("passes", False):
            continue

        det_set = {d["frame"] for d in trk_tracks.get(track_id, [])}
        frames  = sorted(det_set)
        if not frames:
            continue

        # Split into runs separated by gaps > gap_frames
        runs = []
        run_start = frames[0]
        prev      = frames[0]
        for fi in frames[1:]:
            if fi - prev > gap_frames:
                runs.append((run_start, prev))
                run_start = fi
            prev = fi
        runs.append((run_start, prev))

        for s, e in runs:
            length = e - s + 1
            dur_s  = round(length / fps, 2)

            if length < min_frames:
                rejected.append({"track_id": track_id, "start_frame": s, "end_frame": e,
                                  "duration_s": dur_s, "reject_reason": "too_short",
                                  "sync_confidence": sync["confidence"]})
                continue
            if length > max_frames:
                rejected.append({"track_id": track_id, "start_frame": s, "end_frame": e,
                                  "duration_s": dur_s, "reject_reason": "too_long",
                                  "sync_confidence": sync["confidence"]})
                continue

            det_in_span = sum(1 for f in frames if s <= f <= e)
            face_ratio  = det_in_span / length
            if face_ratio < MIN_FACE_RATIO:
                rejected.append({"track_id": track_id, "start_frame": s, "end_frame": e,
                                  "duration_s": dur_s, "reject_reason": "low_face_ratio",
                                  "face_present_ratio": round(face_ratio, 4),
                                  "sync_confidence": sync["confidence"]})
                continue

            clips.append({
                "track_id":           track_id,
                "start_frame":        s,
                "end_frame":          e,
                "face_present_ratio": round(face_ratio, 4),
                "sync_confidence":    sync["confidence"],
            })

    # Winner-takes-all: for overlapping clips, keep only highest sync_confidence
    clips.sort(key=lambda c: -c["sync_confidence"])
    kept   = []
    for clip in clips:
        overlaps = any(
            clip["start_frame"] <= k["end_frame"] and clip["end_frame"] >= k["start_frame"]
            for k in kept
        )
        if overlaps:
            dur = round((clip["end_frame"] - clip["start_frame"]) / fps, 2)
            rejected.append({**clip, "duration_s": dur, "reject_reason": "overlap_lost_to_higher_conf"})
        else:
            kept.append(clip)
    clips = kept

    return clips, rejected


def _detect_black_bars(frame: np.ndarray) -> tuple[int, int, int, int]:
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    y1 = int(np.argmax(row_means > BLACK_THRESHOLD))
    y2 = h - int(np.argmax(row_means[::-1] > BLACK_THRESHOLD))
    x1 = int(np.argmax(col_means > BLACK_THRESHOLD))
    x2 = w - int(np.argmax(col_means[::-1] > BLACK_THRESHOLD))
    if y2 <= y1 or x2 <= x1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def _median_face_cx(trk_tracks: dict, track_id: str,
                    start_frame: int, end_frame: int) -> float | None:
    dets = [d for d in trk_tracks.get(track_id, [])
            if start_frame <= d["frame"] <= end_frame]
    if not dets:
        return None
    return float(np.median([(d["bbox"][0] + d["bbox"][2]) / 2 for d in dets]))


def _sample_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    import av
    try:
        with av.open(video_path) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i == frame_idx:
                    return f.to_ndarray(format="rgb24")
                if i > frame_idx:
                    break
    except Exception:
        pass
    return None


def _export_clip(
    video_path: str,
    trk_tracks: dict,
    track_id: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    frame_w: int,
    frame_h: int,
    out_path: str,
) -> tuple[bool, bool, bool]:
    """Export clip with black-bar and multi-speaker crop. Returns (ok, had_bars, had_other)."""
    mid   = (start_frame + end_frame) // 2
    frame = _sample_frame(video_path, mid)
    if frame is not None:
        bx1, by1, bx2, by2 = _detect_black_bars(frame)
    else:
        bx1, by1, bx2, by2 = 0, 0, frame_w, frame_h

    had_bars  = (bx1 > 0 or by1 > 0 or bx2 < frame_w or by2 < frame_h)
    active_cx = _median_face_cx(trk_tracks, track_id, start_frame, end_frame)

    other_cxs = []
    for oid in trk_tracks:
        if oid == track_id:
            continue
        cx = _median_face_cx(trk_tracks, oid, start_frame, end_frame)
        if cx is not None:
            other_cxs.append(cx)

    had_other = len(other_cxs) > 0
    x1, y1, x2, y2 = bx1, by1, bx2, by2

    if active_cx is not None and other_cxs:
        closest = min(other_cxs, key=lambda cx: abs(cx - active_cx))
        split   = int((active_cx + closest) / 2)
        if active_cx < closest:
            x2 = min(x2, split)
        else:
            x1 = max(x1, split)

    if active_cx is None:
        return False, had_bars, had_other

    start_s = start_frame / fps
    end_s   = end_frame   / fps
    cw, ch  = x2 - x1, y2 - y1
    full    = (x1 == 0 and y1 == 0 and x2 == frame_w and y2 == frame_h)
    vf      = f"scale={EXPORT_RES}:{EXPORT_RES}" if full else \
              f"crop={cw}:{ch}:{x1}:{y1},scale={EXPORT_RES}:{EXPORT_RES}"

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0, had_bars, had_other


@ray.remote(num_cpus=2)
class SegmentWorker:
    def process(self, video_path: str, sync_path: str,
                track_path: str, out_dir: str) -> dict:
        stem = Path(video_path).name
        done_marker = Path(out_dir) / Path(video_path).stem / ".done"
        if done_marker.exists():
            log.info(f"CACHED {stem}")
            cached = json.loads(done_marker.read_text()) if done_marker.stat().st_size else []
            return {"path": video_path, "status": "ok", "n_clips": len(cached), "clips": cached}

        log.info(f"START {stem}")
        t0 = time.perf_counter()
        try:
            sync_data  = json.loads(Path(sync_path).read_text())
            track_data = json.loads(Path(track_path).read_text())

            fps         = float(sync_data.get("fps", 25.0))
            sync_tracks = sync_data.get("tracks", {})
            trk_tracks  = track_data.get("tracks", {})

            # Log sync summary
            n_pass = sum(1 for t in sync_tracks.values() if t.get("passes", False))
            log.info(
                f"  {stem}: fps={fps:.2f}  sync_tracks={len(sync_tracks)}"
                f"  passing={n_pass}"
            )
            for tid, t in sync_tracks.items():
                log.info(
                    f"  {stem} track={tid}: conf={t.get('confidence', 'n/a'):.3f}"
                    f"  passes={t.get('passes', False)}"
                )

            import av
            frame_w, frame_h = 1280, 720
            try:
                with av.open(video_path) as c:
                    s = c.streams.video[0]
                    frame_w = s.width  or frame_w
                    frame_h = s.height or frame_h
            except Exception:
                pass
            log.info(f"  {stem}: frame_size={frame_w}x{frame_h}")

            clips, rejected_clips = _find_clips(sync_tracks, trk_tracks, fps)
            log.info(
                f"  {stem}: _find_clips → {len(clips)} accepted  {len(rejected_clips)} rejected"
                f"  (gap_bridge={GAP_BRIDGE_S}s  min={MIN_CLIP_S}s  max={MAX_CLIP_S}s)"
            )
            for rc in rejected_clips:
                dur = rc.get("duration_s") or (rc["end_frame"] - rc["start_frame"]) / fps
                log.info(
                    f"    REJECTED: track={rc['track_id']}"
                    f"  dur={dur:.1f}s"
                    f"  reason={rc['reject_reason']}"
                )
            for i, c in enumerate(clips):
                dur = (c["end_frame"] - c["start_frame"]) / fps
                log.info(
                    f"    clip[{i}]: track={c['track_id']}"
                    f"  frames=[{c['start_frame']},{c['end_frame']}]"
                    f"  dur={dur:.1f}s"
                    f"  face_ratio={c['face_present_ratio']:.2f}"
                    f"  sync_conf={c['sync_confidence']:.3f}"
                )

            if not clips:
                log.info(f"  {stem}: no clips passed all gates")
                done_marker.parent.mkdir(parents=True, exist_ok=True)
                done_marker.write_text(json.dumps({"clips": [], "rejected_clips": rejected_clips}))
                return {"path": video_path, "status": "ok", "n_clips": 0, "clips": [],
                        "rejected_clips": rejected_clips}

            stem_base = Path(video_path).stem
            clip_dir  = Path(out_dir) / stem_base
            clip_dir.mkdir(parents=True, exist_ok=True)

            written = []
            for clip in clips:
                tid   = clip["track_id"]
                sf    = clip["start_frame"]
                ef    = clip["end_frame"]
                name  = f"{stem_base}_{tid}_{sf:06d}_{ef:06d}"
                cpath = str(clip_dir / f"{name}.mp4")
                mpath = clip_dir / f"{name}.json"

                if mpath.exists():
                    log.info(f"    {name}: already exported — reusing")
                    written.append(json.loads(mpath.read_text()))
                    continue

                log.info(f"    exporting {name}  ({(ef-sf)/fps:.1f}s)")
                # Count how many other tracks overlap this clip's frame range
                n_faces_in_frame = sum(
                    1 for oid, odets in trk_tracks.items()
                    if oid != tid and any(sf <= d["frame"] <= ef for d in odets)
                )

                # Median face size for this track within clip range
                clip_dets = [d for d in trk_tracks.get(tid, []) if sf <= d["frame"] <= ef]
                if clip_dets:
                    face_sizes = [min(d["bbox"][2]-d["bbox"][0], d["bbox"][3]-d["bbox"][1])
                                  for d in clip_dets]
                    median_face_px = round(float(np.median(face_sizes)), 1)
                else:
                    median_face_px = 0.0

                # Gate: reject panel shots (≥2 other faces) entirely — too small to crop cleanly
                if n_faces_in_frame >= 2:
                    log.info(f"    {name}: SKIP panel shot  n_other={n_faces_in_frame}  face={median_face_px:.0f}px")
                    rejected_clips.append({
                        "track_id": tid, "start_frame": sf, "end_frame": ef,
                        "duration_s": round((ef-sf)/fps, 2),
                        "reject_reason": "panel_shot",
                        "median_face_px": median_face_px,
                        "n_faces_in_frame": n_faces_in_frame,
                        "sync_confidence": clip["sync_confidence"],
                    })
                    continue

                # Gate: two-person shots require higher confidence to filter nodders
                required_conf = CONF_THRESHOLD_BY_N_OTHER.get(n_faces_in_frame, 0.3)
                if clip["sync_confidence"] < required_conf:
                    log.info(
                        f"    {name}: SKIP low conf for {n_faces_in_frame}-other scene"
                        f"  conf={clip['sync_confidence']:.3f} < {required_conf}"
                    )
                    rejected_clips.append({
                        "track_id": tid, "start_frame": sf, "end_frame": ef,
                        "duration_s": round((ef-sf)/fps, 2),
                        "reject_reason": f"low_conf_for_scene_type",
                        "median_face_px": median_face_px,
                        "n_faces_in_frame": n_faces_in_frame,
                        "sync_confidence": clip["sync_confidence"],
                        "required_conf": required_conf,
                    })
                    continue

                ok, had_bars, had_other = _export_clip(
                    video_path, trk_tracks, tid, sf, ef, fps,
                    frame_w, frame_h, cpath,
                )
                if not ok:
                    log.warning(f"    {name}: ffmpeg export FAILED")
                    continue

                meta = {
                    "source_video":        video_path,
                    "track_id":            tid,
                    "start_frame":         sf,
                    "end_frame":           ef,
                    "start_s":             round(sf / fps, 3),
                    "end_s":               round(ef / fps, 3),
                    "duration_s":          round((ef - sf) / fps, 3),
                    "fps":                 round(fps, 3),
                    "sync_confidence":     clip["sync_confidence"],
                    "face_present_ratio":  clip["face_present_ratio"],
                    "had_black_bars":      had_bars,
                    "had_extra_speaker":   had_other,
                    "n_faces_in_frame":    n_faces_in_frame,
                    "median_face_px":      median_face_px,
                    "resolution":          EXPORT_RES,
                    "clip_path":           cpath,
                }
                mpath.write_text(json.dumps(meta, indent=2))
                written.append(meta)
                log.info(
                    f"    {name}: OK  had_bars={had_bars}  had_other={had_other}"
                )

            elapsed = time.perf_counter() - t0
            log.info(f"DONE {stem}: {len(written)} clips written  {elapsed:.1f}s")
            done_marker.write_text(json.dumps({"clips": written, "rejected_clips": rejected_clips}))
            return {"path": video_path, "status": "ok",
                    "n_clips": len(written), "clips": written,
                    "rejected_clips": rejected_clips}

        except Exception as e:
            import traceback
            log.error(f"FAILED {stem}\n{traceback.format_exc()}")
            return {"path": video_path, "status": f"failed: {e}", "n_clips": 0, "clips": []}


def main():
    global GAP_BRIDGE_S, MIN_CLIP_S, MAX_CLIP_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",    required=True)
    ap.add_argument("--sync_dir",     required=True)
    ap.add_argument("--track_dir",    required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--num_workers",  type=int,   default=16)
    ap.add_argument("--gap_bridge_s", type=float, default=GAP_BRIDGE_S)
    ap.add_argument("--min_clip_s",   type=float, default=MIN_CLIP_S)
    ap.add_argument("--max_clip_s",   type=float, default=MAX_CLIP_S)
    args = ap.parse_args()

    GAP_BRIDGE_S = args.gap_bridge_s
    MIN_CLIP_S   = args.min_clip_s
    MAX_CLIP_S   = args.max_clip_s

    sync_dir  = Path(args.sync_dir)
    track_dir = Path(args.track_dir)
    videos    = sorted(Path(args.video_dir).rglob("*.mp4"))

    runnable = []
    for v in videos:
        sp = sync_dir  / (v.stem + ".sync.json")
        tp = track_dir / (v.stem + ".tracks.json")
        if sp.exists() and tp.exists():
            runnable.append((str(v), str(sp), str(tp)))
            log.info(f"  queued: {v.name}")
        else:
            missing = []
            if not sp.exists(): missing.append(f"sync={sp}")
            if not tp.exists(): missing.append(f"track={tp}")
            log.warning(f"  SKIP {v.name}: missing {', '.join(missing)}")

    log.info(f"Found {len(videos)} videos  runnable={len(runnable)}  ({args.num_workers} workers)")
    log.info(f"sync_dir={sync_dir}  track_dir={track_dir}  out_dir={args.out_dir}")

    wandb.init(project="talking-characters", entity="rlx-labs",
               name="segment-export", resume="allow",
               config={"gap_bridge_s": GAP_BRIDGE_S,
                       "min_clip_s":   MIN_CLIP_S,
                       "max_clip_s":   MAX_CLIP_S,
                       "resolution":   EXPORT_RES})

    ray.init(ignore_reinit_error=True)
    workers = [SegmentWorker.remote() for _ in range(args.num_workers)]

    futures = [
        workers[i % args.num_workers].process.remote(vp, sp, tp, args.out_dir)
        for i, (vp, sp, tp) in enumerate(runnable)
    ]

    all_clips    = []
    all_rejected = []
    ok = failed  = 0
    total   = len(runnable)
    pending = list(futures)
    t0      = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for res in ray.get(done):
            if res["status"] == "ok":
                ok += 1
                all_clips.extend(res.get("clips", []))
                all_rejected.extend(res.get("rejected_clips", []))
            else:
                failed += 1
                log.error(f"FAILED {Path(res['path']).name}: {res['status']}")
        completed = ok + failed
        elapsed   = time.perf_counter() - t0
        log.info(f"[{completed}/{total}]  ok={ok}  failed={failed}  "
                 f"clips={len(all_clips)}  rejected={len(all_rejected)}  elapsed={elapsed:.1f}s")
        wandb.log({"segment/ok": ok, "segment/failed": failed,
                   "segment/total_clips": len(all_clips),
                   "segment/elapsed_s": elapsed})

    out = Path(args.out_dir) / "segments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clips": all_clips, "rejected_clips": all_rejected}, indent=2))

    durations = [c["duration_s"] for c in all_clips]
    log.info(f"Total clips: {len(all_clips)}  rejected: {len(all_rejected)}")
    if durations:
        log.info(f"Duration mean={np.mean(durations):.1f}s  "
                 f"total={sum(durations)/3600:.2f}h")
    if all_rejected:
        from collections import Counter
        reasons = Counter(r["reject_reason"] for r in all_rejected)
        log.info(f"Rejection reasons: {dict(reasons)}")
    log.info(f"Manifest → {out}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
