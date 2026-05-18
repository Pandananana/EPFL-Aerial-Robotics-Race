"""YOLO pose gate detector — wraps a fine-tuned yolo26x-pose checkpoint.

Predicts a per-gate bbox plus four corner keypoints in TL, TR, BR, BL order.
We return the keypoints directly as the (4, 2) quad to match the same
`predict_gates(image) -> list[(4,2)]` contract as the other detectors.

The model is loaded lazily on first call so importing this module is cheap.
Override the checkpoint path with the YOLO_POSE_WEIGHTS env var; otherwise
the trainer's exported best.pt next to this file is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "pose.pt"
CONF_THRESHOLD = 0.25

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO  # imported lazily — heavy dependency

    weights = Path(os.environ.get("YOLO_POSE_WEIGHTS", DEFAULT_WEIGHTS))
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO Pose weights not found at {weights}. "
            f"Download them from here: https://www.dropbox.com/scl/fo/340ih7m6w5my5rtl6vgon/APJeP3MCXA9o0W8G7lH2KS8?rlkey=t7pp4au21ek4nqs0z0iqfxkv3&st=2b3ny5hk&dl=0"
        )
    _model = YOLO(str(weights))
    return _model


def predict_gates(image: np.ndarray) -> list[np.ndarray]:
    """Predict gate quadrilaterals in an image.

    Returns a list of (4, 2) float arrays in TL, TR, BR, BL order. Empty
    list means no gates detected.
    """
    model = _load_model()

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    results = model.predict(bgr, conf=CONF_THRESHOLD, verbose=False)
    if not results:
        return []

    keypoints = results[0].keypoints
    if keypoints is None or len(keypoints) == 0:
        return []

    xy = keypoints.xy.cpu().numpy().astype(np.float32)  # (N, 4, 2)
    return [xy[i] for i in range(xy.shape[0])]


def main():
    print("YOLO Pose detector. Train with `python -m models.yolo_pose.train`.")


if __name__ == "__main__":
    main()
