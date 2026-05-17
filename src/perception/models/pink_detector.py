"""Pink-LED gate detector for the Webots assignment world.

The aerial-robotics RacingGate proto puts an emissive pink panel inside the
inner opening (PBRAppearance.emissiveColor 1 0 1, transparency 0.2). It is
the only saturated pink object in the scene, so an HSV threshold + contour
quadrilateral extraction is enough — no learned model required.

Ported from aerial-robotics/controllers/main/assignment/my_assignment.py
(GateDetector._detect_pixels), generalised to return every gate found in the
frame instead of only the largest one, and made tolerant of frames that
touch the image border (the EPFL pipeline crops to the inner LED edge but
here we want raw pink-panel corners as the GT contract).
"""

from __future__ import annotations

import cv2
import numpy as np

LOWER_PINK = np.array([140, 42, 0], dtype=np.uint8)
UPPER_PINK = np.array([156, 255, 255], dtype=np.uint8)
MIN_CONTOUR_AREA = 50.0
APPROX_EPSILON_FRAC = 0.02


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def predict_gates(image: np.ndarray) -> list[np.ndarray]:
    """Predict gate quadrilaterals from a colour Webots frame.

    Expects BGR or BGRA uint8. Grayscale input yields no detections since
    the cue is colour. Returns (4, 2) float arrays in TL, TR, BR, BL order.
    """
    if image.ndim != 3:
        return []
    bgr = image[:, :, :3] if image.shape[2] == 4 else image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        approx = cv2.approxPolyDP(c, APPROX_EPSILON_FRAC * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        quads.append(_order_corners(approx))
    return quads


def main():
    print("Pink gate detector for Webots assignment world.")


if __name__ == "__main__":
    main()
