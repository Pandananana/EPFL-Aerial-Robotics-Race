"""Benchmark inference speed of the YOLO pose detector."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perception.models.yolo_pose.detector import predict_gates, _load_model

IMG_DIR = ROOT / "src/perception/models/yolo_pose/data/images/val"
WARMUP = 5


def main():
    images = sorted(IMG_DIR.glob("*.png"))
    if not images:
        print(f"No images found in {IMG_DIR}")
        return

    print(f"Loading model...")
    _load_model()

    print(f"Warming up ({WARMUP} iters)...")
    first = cv2.imread(str(images[0]))
    for _ in range(WARMUP):
        predict_gates(first)

    print(f"Benchmarking on {len(images)} images...")
    times_ms: list[float] = []
    for p in images:
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        predict_gates(img)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)

    n = len(times_ms)
    avg = sum(times_ms) / n
    times_sorted = sorted(times_ms)
    p50 = times_sorted[n // 2]
    p95 = times_sorted[int(n * 0.95)]
    mn, mx = times_sorted[0], times_sorted[-1]

    print()
    print(f"Images:   {n}")
    print(f"Avg:      {avg:.2f} ms ({1000.0 / avg:.1f} FPS)")
    print(f"Median:   {p50:.2f} ms")
    print(f"P95:      {p95:.2f} ms")
    print(f"Min/Max:  {mn:.2f} / {mx:.2f} ms")


if __name__ == "__main__":
    main()
