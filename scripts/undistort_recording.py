"""Undistort every image in a recording directory using config/calibration.yaml.

Reads K and dist_coeffs from the YAML file the rest of the pipeline already
consumes (see src/main.py:load_config), builds a cv2.remap LUT once per image
size, and writes the undistorted frames either in-place, next to the originals,
or into a sibling output directory.

Usage:
    uv run python scripts/undistort_recording.py --recording data/recordings/20260513_115203
    uv run python scripts/undistort_recording.py \
        --recording data/recordings/20260513_115203 \
        --output data/recordings/20260513_115203_undistorted
    uv run python scripts/undistort_recording.py \
        --recording data/recordings/20260513_115203 \
        --calibration config/calibration.yaml \
        --in-place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cal = yaml.safe_load(path.read_text())
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["dist_coeffs"], dtype=np.float64).ravel()
    return K, dist


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--recording", type=Path, required=True,
        help="Directory of recorded frames (img_*.png by default).",
    )
    ap.add_argument(
        "--calibration", type=Path,
        default=REPO_ROOT / "config" / "calibration.yaml",
        help="Path to calibration YAML. Default: config/calibration.yaml.",
    )
    out = ap.add_mutually_exclusive_group()
    out.add_argument(
        "--output", type=Path, default=None,
        help="Directory to write undistorted images into. Created if missing. "
             "Defaults to '<recording>_undistorted' next to the input.",
    )
    out.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the original images in the recording directory.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.recording.is_dir():
        print(f"Recording dir not found: {args.recording}", file=sys.stderr)
        return 1
    if not args.calibration.is_file():
        print(f"Calibration file not found: {args.calibration}", file=sys.stderr)
        return 1

    K, dist = load_calibration(args.calibration)

    image_paths = sorted(
        p for p in args.recording.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        print(f"No images found under {args.recording}", file=sys.stderr)
        return 1

    if args.in_place:
        out_dir = args.recording
    else:
        out_dir = args.output or args.recording.with_name(
            args.recording.name + "_undistorted"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build remap LUTs lazily on first image; recreate if size changes
    # (shouldn't happen within a recording, but cheap to guard against).
    map1: np.ndarray | None = None
    map2: np.ndarray | None = None
    map_size: tuple[int, int] | None = None

    written = 0
    failed: list[Path] = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            failed.append(path)
            continue
        size_wh = (img.shape[1], img.shape[0])
        if map_size != size_wh:
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, R=None, newCameraMatrix=K,
                size=size_wh, m1type=cv2.CV_16SC2,
            )
            map_size = size_wh
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_dir / path.name), und)
        written += 1

    print(f"Undistorted {written} / {len(image_paths)} images -> {out_dir}")
    if failed:
        print(
            f"  Failed to read: {', '.join(p.name for p in failed[:10])}"
            + (" ..." if len(failed) > 10 else ""),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
