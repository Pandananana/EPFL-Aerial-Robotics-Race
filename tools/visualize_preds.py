"""Step through the test split, overlaying ground truth (green) and the
detector's predictions (red). Space = next image, q / Esc = quit."""

import json
from pathlib import Path

import cv2
import numpy as np

from models.hough_detector import predict_gates
from test import load_gt_quads


def draw_quads(img: np.ndarray, quads: list[np.ndarray], color: tuple[int, int, int]) -> None:
    for q in quads:
        cv2.polylines(img, [q.astype(np.int32)], isClosed=True, color=color, thickness=1)


def main():
    manifest = json.loads(Path("dataset/splits.json").read_text())
    items = [it for it in manifest["items"] if it["split"] == "test"]

    i = 0
    while 0 <= i < len(items):
        item = items[i]
        gray = cv2.imread(item["image"], cv2.IMREAD_GRAYSCALE)
        gts = load_gt_quads(Path(item["label"]))
        preds = predict_gates(gray)

        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        draw_quads(canvas, gts, (0, 255, 0))       # GT: green
        draw_quads(canvas, preds, (0, 0, 255))     # pred: red

        canvas = cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        label = f"{i + 1}/{len(items)}  {item['id']}  gt={len(gts)} pred={len(preds)}"
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("gate detector", canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("a") and i > 0:
            i -= 1
        else:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
