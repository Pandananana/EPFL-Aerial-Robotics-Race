"""Camera calibration helper for the AI-deck HM01B0.

STUB. The current config/calibration.yaml uses datasheet-derived
intrinsics with zero distortion, which is fine to within tens of cm at
the image centre but degrades near the edges (the lens's diagonal FOV
is inconsistent with a pinhole model).

To do a real calibration:
  1. Print a chessboard pattern (e.g. 9x6 squares, 25 mm).
  2. Capture ~30 frames covering the full image area at varied poses
     using crazyflie_fpv_example.py or scripts/live_viewer.py.
  3. Run cv2.findChessboardCorners + cv2.calibrateCamera here, then
     write the resulting K and dist coeffs to config/calibration.yaml.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError(
        "Camera calibration not implemented yet. See module docstring "
        "for the intended workflow."
    )


if __name__ == "__main__":
    raise SystemExit(main())
