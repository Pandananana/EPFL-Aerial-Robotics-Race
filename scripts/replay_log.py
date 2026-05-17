"""Replay a recording through the perception pipeline (no drone, no UDP).

Reads a recordings/<id>/ directory, plays its frames + poses back through
ReplayThread, runs them through GateDetector + PoseEstimator, and prints
each Gate3D result. Useful for sanity-checking the detector and pose
estimator together without flying.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtWidgets

# Make `src` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.replay import ReplayThread  # noqa: E402
from src.main import load_config  # noqa: E402
from src.messages import Gate3D  # noqa: E402
from src.perception.gate_detector import GateDetector  # noqa: E402
from src.perception.pose_estimator import PoseEstimator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path, help="recordings/<id>/")
    ap.add_argument("--speed", type=float, default=4.0,
                    help="Replay speed multiplier (default 4x).")
    args = ap.parse_args()

    cfg, cal = load_config()
    app = QtWidgets.QApplication(sys.argv)

    replay = ReplayThread(args.recording, speed=args.speed)
    detector = GateDetector(model_name=cfg["perception"]["detector"])
    estimator = PoseEstimator(
        camera_matrix=np.array(cal["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(cal["dist_coeffs"], dtype=np.float64),
        gate_height_m=cfg["perception"]["gate_height_m"],
        width_search=tuple(cfg["perception"]["gate_width_search_m"]),
    )

    replay.frame_ready.connect(detector.on_frame)
    detector.detection_ready.connect(estimator.on_detection)

    def _print(g: Gate3D) -> None:
        if not g.corners_cam_m:
            return
        for i, (c, w, e) in enumerate(zip(
            g.corners_cam_m, g.widths_m, g.reprojection_errors_px
        )):
            centroid = c.mean(axis=0)
            print(f"frame {g.frame_seq}, gate {i}: "
                  f"centroid_cam={centroid.round(2).tolist()} "
                  f"w={w:.2f}m err={e:.2f}px")

    estimator.gate_ready.connect(_print)
    replay.finished.connect(app.quit)
    replay.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
