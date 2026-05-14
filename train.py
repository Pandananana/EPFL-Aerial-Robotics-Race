"""Gate detector.

v1: random predictor — guesses 0–3 random quads per image. Exists so we can
sanity-check the eval pipeline before plugging in a real model. Scores from
test.py should be near-zero with this predictor; if they're not, something's
wrong upstream.
"""

import random

import numpy as np


def predict_gates(image: np.ndarray) -> list[np.ndarray]:
    """Predict gate quadrilaterals in an image.

    Returns a list of (4, 2) float arrays, each one [TL, TR, BR, BL] in pixel
    coordinates. Empty list means "no gates."
    """
    h, w = image.shape[:2]
    n = random.randint(0, 3)
    preds = []
    for _ in range(n):
        cx = random.uniform(0.15 * w, 0.85 * w)
        cy = random.uniform(0.15 * h, 0.85 * h)
        side = random.uniform(0.1, 0.4) * min(h, w)
        angle = random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32) * side / 2
        quad = local @ rot.T + np.array([cx, cy])
        preds.append(quad.astype(np.float32))
    return preds


def main():
    print("train.py v1: random predictor — no training needed.")


if __name__ == "__main__":
    main()
