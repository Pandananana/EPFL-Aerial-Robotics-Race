"""Evaluate gate detector on the test split.

Runs predict_gates() from train.py over every test-split image in
dataset/splits.json, matches predictions to ground-truth gates by IoU, and
reports precision / recall / F1 at a configurable IoU threshold.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from train import predict_gates


def load_gt_quads(label_path: Path) -> list[np.ndarray]:
    data = json.loads(label_path.read_text())
    return [
        np.array(s["points"], dtype=np.float32)
        for s in data.get("shapes", [])
        if s.get("label") == "gate"
    ]


def quad_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [quad.astype(np.int32)], 1)
    return m


def quad_iou(q1: np.ndarray, q2: np.ndarray, shape: tuple[int, int]) -> float:
    m1 = quad_mask(q1, shape)
    m2 = quad_mask(q2, shape)
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return inter / union if union > 0 else 0.0


def match_greedy(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    shape: tuple[int, int],
    threshold: float,
) -> tuple[int, int, int, list[float]]:
    """Greedy IoU matching. Returns (tp, fp, fn, matched_ious)."""
    if not preds:
        return 0, 0, len(gts), []
    if not gts:
        return 0, len(preds), 0, []

    iou = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou[i, j] = quad_iou(p, g, shape)

    matched_p: set[int] = set()
    matched_g: set[int] = set()
    matched_ious: list[float] = []
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] < threshold:
            break
        matched_ious.append(float(iou[i, j]))
        matched_p.add(int(i))
        matched_g.add(int(j))
        iou[i, :] = -1.0
        iou[:, j] = -1.0

    tp = len(matched_p)
    fp = len(preds) - tp
    fn = len(gts) - len(matched_g)
    return tp, fp, fn, matched_ious


def evaluate(test_items: list[dict], iou_threshold: float) -> dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: list[float] = []

    for item in test_items:
        image = cv2.imread(item["image"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(item["image"])
        gts = load_gt_quads(Path(item["label"]))
        preds = predict_gates(image)
        tp, fp, fn, ious = match_greedy(preds, gts, image.shape[:2], iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        "n_images": len(test_items),
        "iou_threshold": iou_threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": mean_iou,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("dataset/splits.json"))
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a true positive")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (the random predictor uses random.*)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest = json.loads(args.manifest.read_text())
    test_items = [it for it in manifest["items"] if it["split"] == "test"]
    if not test_items:
        print("No test items in manifest.")
        return

    scores = evaluate(test_items, args.iou)

    print(f"Test images:        {scores['n_images']}")
    print(f"IoU threshold:      {scores['iou_threshold']}")
    print(f"TP / FP / FN:       {scores['tp']} / {scores['fp']} / {scores['fn']}")
    print(f"Precision:          {scores['precision']:.3f}")
    print(f"Recall:             {scores['recall']:.3f}")
    print(f"F1:                 {scores['f1']:.3f}")
    print(f"Mean IoU (matched): {scores['mean_iou_matched']:.3f}")


if __name__ == "__main__":
    main()
