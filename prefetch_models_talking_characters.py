"""
prefetch_models_talking_characters.py
--------------------------------------
Pre-download all model weights needed by the talking-characters pipeline.
Must be run on the LOGIN NODE — compute nodes have no internet access
(HF_HUB_OFFLINE=1 is set at job start).

Downloads:
  - InsightFace buffalo_sc (SCRFD-10GF) — face detection
  - LoCoNet weights                     — active speaker detection
  - Light-ASD weights                   — ASD fallback

Usage:
  python prefetch_models_talking_characters.py
  python prefetch_models_talking_characters.py --skip_loconet   # use Light-ASD only
"""

import argparse
import os
from pathlib import Path


CACHE_DIR = Path.home() / ".cache" / "talking_characters"

# LoCoNet pretrained weights (AVA-ActiveSpeaker)
LOCONET_URL = "https://github.com/kahnchana/loconet/releases/download/v1.0/loconet_ava.pth"

# Light-ASD pretrained weights
LIGHT_ASD_URL = "https://github.com/Junhua-Liao/Light-ASD/releases/download/v1.0/light_asd.pth"


def prefetch_insightface():
    """Download InsightFace buffalo_sc model pack (SCRFD-10GF + 2d106det)."""
    print("Prefetching InsightFace buffalo_sc (SCRFD-10GF) ...")
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_sc",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],  # CPU is fine for download
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("  InsightFace buffalo_sc OK")


def prefetch_loconet():
    """Download LoCoNet weights."""
    import urllib.request
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "loconet.pth"
    if dest.exists():
        print(f"  LoCoNet already cached at {dest}")
        return
    print(f"  Downloading LoCoNet → {dest} ...")
    urllib.request.urlretrieve(LOCONET_URL, dest)
    print(f"  LoCoNet OK  ({dest.stat().st_size / 1e6:.1f} MB)")


def prefetch_light_asd():
    """Download Light-ASD weights."""
    import urllib.request
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "light_asd.pth"
    if dest.exists():
        print(f"  Light-ASD already cached at {dest}")
        return
    print(f"  Downloading Light-ASD → {dest} ...")
    urllib.request.urlretrieve(LIGHT_ASD_URL, dest)
    print(f"  Light-ASD OK  ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_loconet",   action="store_true",
                    help="Skip LoCoNet download (use Light-ASD only)")
    ap.add_argument("--skip_light_asd", action="store_true",
                    help="Skip Light-ASD download")
    args = ap.parse_args()

    print("=== Prefetching talking-characters model weights ===")
    print(f"Cache dir: {CACHE_DIR}\n")

    prefetch_insightface()

    if not args.skip_loconet:
        prefetch_loconet()

    if not args.skip_light_asd:
        prefetch_light_asd()

    print("\nAll models prefetched. Safe to submit PBS job.")


if __name__ == "__main__":
    main()
