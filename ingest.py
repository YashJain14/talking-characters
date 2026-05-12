"""
ingest.py
---------
Download YouTube videos from a CSV file using yt-dlp.

CSV format (one video per row):
  url                                          — required, YouTube URL
  id (optional)                                — custom identifier; defaults to yt video ID
  label (optional)                             — category/speaker label for subdirectory

Examples:
  url
  https://www.youtube.com/watch?v=abc123
  https://youtu.be/xyz789

  url,label
  https://www.youtube.com/watch?v=abc123,elon_musk
  https://youtu.be/xyz789,lex_fridman

Output:
  $SCRATCH_TC/raw_videos/<label>/<video_id>.mp4
  If no label column is present, all videos go into raw_videos/unlabelled/.

Re-run safe: already-downloaded videos are skipped (.done marker files).

Usage:
  python ingest.py --csv videos.csv --out_dir $SCRATCH_TC/raw_videos
  python ingest.py --csv videos.csv --out_dir $SCRATCH_TC/raw_videos --workers 4
"""

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("ingest")


def _load_csv(csv_path: str) -> list[dict]:
    """
    Load CSV into a list of {url, label, id} dicts.
    Handles csvs with or without a header row and optional columns.
    """
    rows = []
    with open(csv_path, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.DictReader(f) if has_header else csv.reader(f)

        if has_header:
            for row in reader:
                # Normalise key names to lowercase stripped
                row = {k.strip().lower(): v.strip() for k, v in row.items()}
                url = row.get("url") or row.get("link") or row.get("youtube_url")
                if not url:
                    continue
                rows.append({
                    "url":   url,
                    "label": row.get("label", "unlabelled"),
                    "id":    row.get("id", ""),
                })
        else:
            for row in reader:
                if not row or not row[0].strip():
                    continue
                url = row[0].strip()
                if url.lower() in ("url", "link", "youtube_url"):
                    continue  # skip accidental header row
                rows.append({
                    "url":   url,
                    "label": row[1].strip() if len(row) > 1 else "unlabelled",
                    "id":    row[2].strip() if len(row) > 2 else "",
                })

    log.info(f"Loaded {len(rows)} URLs from {csv_path}")
    return rows


def _ffmpeg_exe() -> str:
    """Return path to ffmpeg: imageio_ffmpeg bundled binary, else system ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _download(entry: dict, out_dir: Path, format_str: str) -> dict:
    """
    Download one YouTube video via yt-dlp.
    Returns {"url", "status": "ok"|"cached"|"failed: ...", "path": str}
    """
    import subprocess

    url   = entry["url"]
    label = entry["label"] or "unlabelled"
    dest  = out_dir / label
    dest.mkdir(parents=True, exist_ok=True)

    # Use yt-dlp's %(id)s so the filename is the YouTube video ID
    outtmpl = str(dest / "%(id)s.%(ext)s")

    # Done-marker keyed on URL hash to avoid re-downloading
    url_hash    = str(abs(hash(url)))[:12]
    done_marker = dest / f".{url_hash}.done"
    if done_marker.exists():
        # Find the actual mp4 that was written
        existing = list(dest.glob("*.mp4"))
        path     = str(existing[0]) if existing else ""
        return {"url": url, "status": "cached", "path": path}

    t0 = time.perf_counter()
    try:
        ffmpeg = _ffmpeg_exe()
        cmd = [
            "yt-dlp",
            "--ffmpeg-location", ffmpeg,
            "--format", format_str,
            "--output", outtmpl,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--merge-output-format", "mp4",
            "--remux-video", "mp4",
            "--no-embed-subs",
            "--no-embed-thumbnail",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"yt-dlp failed for {url}:\n{result.stderr.strip()[-500:]}")
            return {"url": url, "status": f"failed: {result.stderr.strip()[:200]}",
                    "path": ""}

        # Verify the downloaded file has an audio stream
        mp4s = sorted(dest.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s:
            probe = subprocess.run([ffmpeg, "-i", str(mp4s[0])], capture_output=True, text=True)
            if "Audio:" not in probe.stderr:
                log.warning(
                    f"Downloaded file has no audio stream: {mp4s[0].name}\n"
                    f"ffmpeg info: {probe.stderr[-300:]}"
                )

        # Find the downloaded file
        mp4s = sorted(dest.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        path = str(mp4s[0]) if mp4s else ""

        done_marker.touch()
        elapsed = time.perf_counter() - t0
        return {"url": url, "status": "ok", "path": path, "time_s": elapsed}

    except Exception as e:
        return {"url": url, "status": f"failed: {e}", "path": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True,
                    help="CSV file with YouTube URLs (columns: url, label[optional])")
    ap.add_argument("--out_dir",   default=None,
                    help="Output directory. Defaults to $SCRATCH_TC/raw_videos")
    ap.add_argument("--workers",   type=int, default=4,
                    help="Parallel download threads (yt-dlp is I/O bound)")
    ap.add_argument("--format",    default="bestvideo[height<=1080]+bestaudio/best",
                    help="yt-dlp format selector")
    ap.add_argument("--limit",     type=int, default=None,
                    help="Max number of videos to download (useful for testing)")
    args = ap.parse_args()

    scratch = os.environ.get("SCRATCH_TC") or os.path.expanduser("~/scratch/talking-characters")
    out_dir = Path(args.out_dir) if args.out_dir else Path(scratch) / "raw_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_csv(args.csv)
    if args.limit:
        entries = entries[: args.limit]

    log.info(f"Downloading {len(entries)} videos → {out_dir}  (workers={args.workers})")

    wandb.init(project="talking-characters", entity="rlx-labs",
               name="ingest", resume="allow",
               config={"csv": args.csv, "workers": args.workers,
                       "limit": args.limit, "out_dir": str(out_dir)})

    t0 = time.perf_counter()
    ok = failed = cached = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_download, entry, out_dir, args.format): entry
            for entry in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if   res["status"] == "ok":     ok     += 1
            elif res["status"] == "cached": cached += 1
            else:
                failed += 1
                log.error(f"FAILED {res['url']}: {res['status']}")

            if i % 10 == 0 or i == len(entries):
                elapsed = time.perf_counter() - t0
                log.info(f"[{i}/{len(entries)}]  ok={ok}  cached={cached}  "
                         f"failed={failed}  elapsed={elapsed:.1f}s")
                wandb.log({"ingest/ok": ok, "ingest/cached": cached,
                           "ingest/failed": failed, "ingest/elapsed_s": elapsed})

    elapsed = time.perf_counter() - t0
    total_mp4s = len(list(out_dir.rglob("*.mp4")))
    log.info(f"Done in {elapsed:.1f}s — ok={ok}  cached={cached}  failed={failed}")
    log.info(f"Total MP4s on disk: {total_mp4s}  →  {out_dir}")
    wandb.log({"ingest/total_mp4s": total_mp4s, "ingest/elapsed_s": elapsed})
    wandb.finish()


if __name__ == "__main__":
    main()