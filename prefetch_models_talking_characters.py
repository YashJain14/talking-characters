"""
prefetch_models_talking_characters.py
--------------------------------------
Pre-download all model weights needed by the talking-characters pipeline.
Must be run on the LOGIN NODE — compute nodes have no internet access.

Downloads:
  - InsightFace buffalo_sc (SCRFD-10GF) — face detection
  - SyncNet weights                     — audio-visual sync scoring

Usage:
  python prefetch_models_talking_characters.py
"""

import os
import urllib.request
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "talking_characters"

# SyncNet pretrained weights (Chung & Zisserman ECCV 2016)
# Hosted on the syncnet_python repo
SYNCNET_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"


def prefetch_insightface():
    print("Prefetching InsightFace buffalo_sc (SCRFD-10GF) ...")
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_sc",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("  InsightFace buffalo_sc OK")


def prefetch_syncnet():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "syncnet.pth"
    if dest.exists():
        print(f"  SyncNet already cached at {dest}")
        return
    print(f"  Downloading SyncNet → {dest} ...")
    # Try primary URL
    try:
        urllib.request.urlretrieve(SYNCNET_URL, str(dest))
    except Exception as e:
        print(f"  Primary URL failed ({e}), trying mirror ...")
        # Mirror: syncnet_python GitHub release
        mirror = "https://github.com/joonson/syncnet_python/releases/download/v1.0/syncnet_v2.model"
        try:
            urllib.request.urlretrieve(mirror, str(dest))
        except Exception as e2:
            raise RuntimeError(
                f"SyncNet download failed from both sources.\n"
                f"Primary: {SYNCNET_URL}\n"
                f"Mirror:  {mirror}\n"
                f"Error: {e2}\n\n"
                f"Manual download:\n"
                f"  wget {SYNCNET_URL} -O {dest}"
            )

    if not dest.exists() or dest.stat().st_size < 1e5:
        raise RuntimeError(f"SyncNet download looks too small: {dest}")
    print(f"  SyncNet OK  ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    print("=== Prefetching talking-characters model weights ===")
    print(f"Cache dir: {CACHE_DIR}\n")

    prefetch_insightface()
    prefetch_syncnet()

    print("\nAll models prefetched. Safe to submit SLURM job.")


if __name__ == "__main__":
    main()
