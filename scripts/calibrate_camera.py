"""Calibrate the AI-deck (or any) camera from a directory of chessboard frames.

Detects inner corners on each PNG/JPG, runs cv2.calibrateCamera with the
rational distortion model (8 distortion coefficients — fits the HM01B0's
wide FOV better than the default 5-term Brown model), and writes the
result to config/calibration.yaml in the format the rest of the pipeline
already consumes (see src/main.py:load_config).

Default board geometry matches the EPFL 7x9-square × 21 mm board:
inner corners 6x8 (=(7-1, 9-1)), square size 0.021 m.

Usage:
    uv run python scripts/calibrate_camera.py --images data/calib_run/
    uv run python scripts/calibrate_camera.py \
        --images data/calib_run/ \
        --inner-corners 6 8 \
        --square-size 0.021 \
        --output config/calibration.yaml

Capture tips:
- ~20-30 frames is plenty. More is fine but redundant.
- Vary tilt, in-plane rotation, and distance. Get the board near the
  image edges and corners — that's what pins down the distortion terms.
- Hold the board flat (it must actually be planar). Keep it still per
  shot; motion blur kills corner localisation on the 324x244 sensor.
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


def find_corners(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    """Return refined (N, 1, 2) float32 corner array, or None on failure.

    Tries the sector-based detector first (more robust on small/blurry
    frames like the AI-deck's 324x244 output) and falls back to the
    legacy detector with sub-pixel refinement.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # cv2.findChessboardCornersSB returns corners already sub-pixel refined.
    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if found:
        return corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return corners.astype(np.float32)


def build_object_points(
    pattern_size: tuple[int, int], square_size: float
) -> np.ndarray:
    """(N, 3) grid of corner coordinates in the board frame, Z=0."""
    cols, rows = pattern_size
    grid = np.zeros((cols * rows, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--images", type=Path, required=True,
        help="Directory containing chessboard frames (PNG/JPG/BMP/TIFF).",
    )
    ap.add_argument(
        "--inner-corners", type=int, nargs=2, default=[6, 8],
        metavar=("COLS", "ROWS"),
        help="Inner-corner count (cols rows). A 7x9-square board has 6 8. "
             "Default: 6 8.",
    )
    ap.add_argument(
        "--square-size", type=float, default=0.021,
        help="Side length of one square in metres. Default: 0.021 (21 mm).",
    )
    ap.add_argument(
        "--model", choices=["rational", "standard", "fisheye"],
        default="rational",
        help="Distortion model. 'rational' (8 coeffs, default) fits wide-FOV "
             "lenses like the AI-deck; 'standard' is the classic 5-term Brown; "
             "'fisheye' uses cv2.fisheye for very wide / true fisheye optics.",
    )
    ap.add_argument(
        "--output", type=Path, default=REPO_ROOT / "config" / "calibration.yaml",
        help="Where to write the YAML. Default: config/calibration.yaml.",
    )
    ap.add_argument(
        "--debug-dir", type=Path, default=None,
        help="If set, write per-image corner overlays here for sanity checking.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pattern_size = (args.inner_corners[0], args.inner_corners[1])

    image_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        print(f"No images found under {args.images}", file=sys.stderr)
        return 1

    object_template = build_object_points(pattern_size, args.square_size)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None  # (width, height)
    accepted: list[Path] = []
    rejected: list[Path] = []

    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            rejected.append(path)
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != image_size:
            print(
                f"Skipping {path.name}: size {img.shape[1]}x{img.shape[0]} "
                f"!= first image {image_size[0]}x{image_size[1]}",
                file=sys.stderr,
            )
            rejected.append(path)
            continue

        corners = find_corners(img, pattern_size)
        if corners is None:
            rejected.append(path)
            continue

        obj_points.append(object_template.copy())
        img_points.append(corners)
        accepted.append(path)

        if args.debug_dir is not None:
            vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            cv2.imwrite(str(args.debug_dir / path.name), vis)

    print(f"Detected board in {len(accepted)} / {len(image_paths)} images.")
    if rejected:
        print(f"  Rejected: {', '.join(p.name for p in rejected[:10])}"
              + (" ..." if len(rejected) > 10 else ""))
    if len(accepted) < 8:
        print("Need at least ~8 good views to calibrate. Capture more.",
              file=sys.stderr)
        return 1
    assert image_size is not None

    if args.model == "fisheye":
        K, dist, rms = _calibrate_fisheye(obj_points, img_points, image_size)
    else:
        flags = 0
        if args.model == "rational":
            flags |= cv2.CALIB_RATIONAL_MODEL
        rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None, flags=flags,
        )
        dist = dist.ravel()

    print(f"\nRMS reprojection error: {rms:.4f} px")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"K =\n{K}")
    print(f"dist = {dist}")

    write_calibration_yaml(args.output, K, dist, image_size, rms, args.model)
    print(f"\nWrote {args.output}")
    return 0


def _calibrate_fisheye(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """cv2.fisheye expects slightly different array shapes than the
    pinhole API. Returns (K, dist (4,), rms)."""
    obj_fish = [p.reshape(-1, 1, 3).astype(np.float64) for p in obj_points]
    img_fish = [p.reshape(-1, 1, 2).astype(np.float64) for p in img_points]
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_fish]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in obj_fish]
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_CHECK_COND
    )
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        obj_fish, img_fish, image_size, K, D, rvecs, tvecs, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    return K, D.ravel(), rms


def write_calibration_yaml(
    path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    model: str,
) -> None:
    header = (
        f"# Camera intrinsics produced by scripts/calibrate_camera.py.\n"
        f"# Distortion model: {model}\n"
        f"# Image size: {image_size[0]}x{image_size[1]} (width x height)\n"
        f"# RMS reprojection error: {rms:.4f} px\n"
    )
    data = {
        "camera_matrix": [[float(v) for v in row] for row in K],
        "dist_coeffs": [float(v) for v in dist],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
