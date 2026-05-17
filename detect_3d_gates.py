"""Run the YOLO pose detector over every frame in a recording, lift the four
detected corners into 3D using the camera intrinsics and the known gate height
(48 cm inside the LED frame), and dump the result to JSON.

Gate model: planar rectangle, height fixed at 0.48 m, width unknown. For each
frame and each detected gate we run cv2.solvePnP with a parameterized model
and grid-search the width that minimises reprojection error. Corners are
returned in camera frame (x right, y down, z forward — OpenCV convention),
TL / TR / BR / BL order, metres.

Fill in CAMERA_MATRIX and DIST_COEFFS below before running.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from models.yolo_pose.detector import predict_gates

# ---------------------------------------------------------------------------
# Camera intrinsics — derived from the AI-deck HM01B0 datasheet (no per-camera
# calibration). The sensor is 324x324, pixel size 3.6um, horizontal and
# vertical FOV are both 87 deg over the full active array. We operate in the
# 324x244 window mode, which crops vertically and is assumed centred on the
# sensor.
#
#   fx = fy = (324 / 2) / tan(87/2 deg) ~= 170.7 px
#   cx = 324 / 2 = 162 px
#   cy = 244 / 2 = 122 px  (centred crop)
#
# Distortion is left at zero. The datasheet lists a 115 deg diagonal FOV
# against 87 deg H/V, which is inconsistent with a pinhole and implies real
# barrel distortion of the lens — expect noticeable depth error (~ tens of cm)
# for gates near the image edges. Replace with a calibrated K + dist for
# better accuracy.
# ---------------------------------------------------------------------------
import math

_HFOV_DEG = 87.0
_FX = (324.0 / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
CAMERA_MATRIX = np.array(
    [
        [_FX, 0.0, 162.0],
        [0.0, _FX, 122.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST_COEFFS = np.zeros(5, dtype=np.float64)  # [k1, k2, p1, p2, k3]

GATE_HEIGHT_M = 0.48  # inner height of the LED frame
WIDTH_SEARCH = np.arange(0.20, 1.21, 0.01)  # metres; covers all gates in the dataset


def gate_model(width: float) -> np.ndarray:
    """Return the 4 model points (TL, TR, BR, BL) in the gate's own frame.
    Z = 0 (gate is planar), x is width, y is height (down positive matches
    OpenCV image conventions for image-side points)."""
    h = GATE_HEIGHT_M / 2.0
    w = width / 2.0
    return np.array(
        [
            [-w, -h, 0.0],  # TL
            [+w, -h, 0.0],  # TR
            [+w, +h, 0.0],  # BR
            [-w, +h, 0.0],  # BL
        ],
        dtype=np.float64,
    )


def solve_gate_3d(
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Recover the 3D corner positions (camera frame) for one gate detection.

    image_points: (4, 2) float pixel coords in TL, TR, BR, BL order.
    Returns (corners_cam[4,3], best_width, reproj_err_px) or None on failure.
    """
    img = image_points.astype(np.float64).reshape(-1, 1, 2)

    best = None
    for w in WIDTH_SEARCH:
        model_pts = gate_model(float(w))
        try:
            ok, rvec, tvec, errs = cv2.solvePnPGeneric(
                model_pts,
                img,
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE,  # planar, exactly 4 points
            )
        except cv2.error:
            continue
        if not ok or not len(rvec):
            continue
        for r, t, e in zip(rvec, tvec, errs):
            err_px = float(np.asarray(e).reshape(-1)[0])
            if best is None or err_px < best[0]:
                best = (err_px, r, t, float(w))

    if best is None:
        return None

    err_px, rvec, tvec, width = best
    R, _ = cv2.Rodrigues(rvec)
    corners_cam = (R @ gate_model(width).T + tvec.reshape(3, 1)).T
    return corners_cam, width, err_px


def process_recording(rec_dir: Path) -> dict:
    images = sorted(rec_dir.glob("img_*.png"))
    output: dict[str, list[dict]] = {}

    for img_path in images:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip (unreadable): {img_path.name}")
            continue

        quads = predict_gates(gray)
        gates_out: list[dict] = []
        for q in quads:
            if q.shape != (4, 2):
                continue
            result = solve_gate_3d(q, CAMERA_MATRIX, DIST_COEFFS)
            if result is None:
                continue
            corners_cam, width, err = result
            gates_out.append(
                {
                    "image_points": q.tolist(),
                    "corners_cam_m": {
                        "top_left": corners_cam[0].tolist(),
                        "top_right": corners_cam[1].tolist(),
                        "bottom_right": corners_cam[2].tolist(),
                        "bottom_left": corners_cam[3].tolist(),
                    },
                    "width_m": width,
                    "height_m": GATE_HEIGHT_M,
                    "reprojection_error_px": err,
                }
            )
        output[img_path.name] = gates_out
        if gates_out:
            print(f"{img_path.name}: {len(gates_out)} gate(s)")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        type=Path,
        default=Path("recordings/20260513_115203"),
        help="Recording directory containing img_*.png frames.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <recording>/gate_predictions_3d.json.",
    )
    args = parser.parse_args()

    out_path = args.out or args.recording / "gate_predictions_3d.json"
    results = process_recording(args.recording)
    payload = {
        "recording": str(args.recording),
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "dist_coeffs": DIST_COEFFS.tolist(),
        "gate_height_m": GATE_HEIGHT_M,
        "frame_coordinate_system": "OpenCV camera frame (x right, y down, z forward), metres",
        "frames": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
