"""YOLO segmentation gate detector — wraps a fine-tuned yolo26x-seg checkpoint.

Unlike the OBB detector, segmentation predicts a freeform mask per gate. We
collapse each mask contour to a 4-corner quad so the output matches the same
`predict_gates(image) -> list[(4,2)]` contract as the other detectors.

The model is loaded lazily on first call so importing this module is cheap.
Override the checkpoint path with the YOLO_SEG_WEIGHTS env var; otherwise the
trainer's exported best.pt next to this file is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "seg.pt"
CONF_THRESHOLD = 0.25

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO  # imported lazily — heavy dependency

    weights = Path(os.environ.get("YOLO_SEG_WEIGHTS", DEFAULT_WEIGHTS))
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO Seg weights not found at {weights}. "
            f"Download them from here: https://www.dropbox.com/scl/fo/340ih7m6w5my5rtl6vgon/APJeP3MCXA9o0W8G7lH2KS8?rlkey=t7pp4au21ek4nqs0z0iqfxkv3&st=2b3ny5hk&dl=0"
        )
    _model = YOLO(str(weights))
    return _model


def _polygon_to_quad(poly: np.ndarray) -> np.ndarray:
    """Collapse a mask contour to a 4-corner quad.

    Strategy: take the convex hull, then bisect on the approxPolyDP epsilon
    until we land on exactly 4 vertices. If no such epsilon exists (rare —
    typically a degenerate / very-noisy mask) fall back to the min-area rect,
    which still matches our output contract but loses the quad-ness we want.
    """
    pts = poly.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    if len(hull) <= 4:
        # Hull itself is a triangle/quad — pad with a duplicate if needed.
        corners = hull.reshape(-1, 2)
        if len(corners) == 4:
            return corners.astype(np.float32)
        return cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)

    arc = cv2.arcLength(hull, True)
    lo, hi = 1e-4, 0.5
    for _ in range(30):
        eps = 0.5 * (lo + hi)
        approx = cv2.approxPolyDP(hull, eps * arc, True)
        n = len(approx)
        if n == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if n > 4:
            lo = eps
        else:
            hi = eps

    return cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)


def predict_gates(image: np.ndarray) -> list[np.ndarray]:
    """Predict gate quadrilaterals in an image.

    Returns a list of (4, 2) float arrays in cyclic order. Empty list means
    no gates detected.
    """
    model = _load_model()

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image

    results = model.predict(bgr, conf=CONF_THRESHOLD, verbose=False)
    if not results:
        return []

    masks = results[0].masks
    if masks is None or len(masks) == 0:
        return []

    quads: list[np.ndarray] = []
    for poly in masks.xy:  # list of (N_i, 2) float arrays in image coords
        if poly is None or len(poly) < 3:
            continue
        quads.append(_polygon_to_quad(np.asarray(poly)))
    return quads


def main():
    print("YOLO Seg detector. Train with `python -m models.yolo_seg.train`.")


if __name__ == "__main__":
    main()
