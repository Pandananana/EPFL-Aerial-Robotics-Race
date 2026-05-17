"""Run the YOLO segmentation model over every frame of every recording and
write a 3 FPS video of the original frames with each predicted segment's
outline drawn in red (interior left transparent).

Runs are concatenated in sorted-by-folder-name order. Frames within a run are
ordered by filename.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from models.yolo_seg.detector import CONF_THRESHOLD, _load_model

OUTLINE_BGR = (0, 0, 255)


def predict_polygons(model, image: np.ndarray) -> list[np.ndarray]:
    """Return a list of polygons (each an (N, 2) int32 array) — one per
    predicted gate, in original-image coordinates."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    results = model.predict(bgr, conf=CONF_THRESHOLD, verbose=False)
    if not results:
        return []
    masks = results[0].masks
    if masks is None or len(masks) == 0:
        return []
    return [np.asarray(poly, dtype=np.int32) for poly in masks.xy if poly is not None and len(poly) >= 3]


def collect_frames(recordings_dir: Path) -> list[Path]:
    runs = sorted(p for p in recordings_dir.iterdir() if p.is_dir())
    frames: list[Path] = []
    for run in runs:
        frames.extend(sorted(run.glob("img_*.png")))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--out", type=Path, default=Path("seg_inference.mp4"))
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor for the output video.")
    args = parser.parse_args()

    frames = collect_frames(args.recordings)
    if not frames:
        raise SystemExit(f"No frames found under {args.recordings}")

    sample = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    h, w = sample.shape[:2]
    out_w, out_h = w * args.scale, h * args.scale

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (out_w, out_h), isColor=True)
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.out}")

    model = _load_model()

    for i, frame_path in enumerate(frames, 1):
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        polys = predict_polygons(model, gray)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if polys:
            cv2.polylines(canvas, polys, isClosed=True, color=OUTLINE_BGR, thickness=1, lineType=cv2.LINE_AA)
        if args.scale != 1:
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        writer.write(canvas)
        if i % 50 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}  ({frame_path.parent.name}/{frame_path.name})")

    writer.release()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
