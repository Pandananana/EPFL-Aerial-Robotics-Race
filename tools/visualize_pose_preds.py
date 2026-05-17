"""Step through a set of images and overlay YOLO pose predictions.

Each predicted gate has four corner keypoints (TL, TR, BR, BL). They are
drawn on a padded canvas so corners that ultralytics clamps to / near the
image border are still legible, and so future off-image extrapolations show
up outside the original frame.

  cyan dot   predicted corner
  yellow     predicted gate quadrilateral TL->TR->BR->BL->TL
  green      ground-truth quadrilateral (only when --show-gt and a labelme
             sidecar exists next to the image)
  white box  original image bounds

Sources (pick one):
  --splits dataset/splits.json [--split test|train|all]   labeled subset
  --images-dir PATH                                       any folder of PNGs
  --recording NAME                                        recordings/<NAME>

Keys: space / right = next, a / left = previous, q / Esc = quit.

Examples (run from repo root):
    uv run python tools/visualize_pose_preds.py --split test
    uv run python tools/visualize_pose_preds.py --recording 20260513_115203
    uv run python tools/visualize_pose_preds.py --images-dir to_label
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.yolo_pose import predict_gates

CORNER_NAMES = ("TL", "TR", "BR", "BL")
PAD_BG = (50, 50, 50)
COLOR_KP = (255, 255, 0)         # cyan dots for predicted corners
COLOR_EDGE = (0, 255, 255)       # yellow quad edges
COLOR_GT = (0, 255, 0)           # green GT
COLOR_IMG_BORDER = (255, 255, 255)


def collect_from_splits(splits_path: Path, split: str) -> list[dict]:
    manifest = json.loads(splits_path.read_text())
    items = manifest["items"]
    if split != "all":
        items = [it for it in items if it["split"] == split]
    return [
        {"image": Path(it["image"]), "label": Path(it["label"]), "id": it["id"]}
        for it in items
    ]


def collect_from_dir(images_dir: Path) -> list[dict]:
    paths = sorted(images_dir.glob("img_*.png"))
    if not paths:
        paths = sorted(images_dir.glob("*.png"))
    out = []
    for p in paths:
        sidecar = p.with_suffix(".json")
        out.append({
            "image": p,
            "label": sidecar if sidecar.exists() else None,
            "id": f"{p.parent.name}/{p.stem}",
        })
    return out


def load_gt_quads(label_path: Path | None) -> list[np.ndarray]:
    if label_path is None or not label_path.exists():
        return []
    data = json.loads(label_path.read_text())
    return [
        np.array(s["points"], dtype=np.float32)
        for s in data.get("shapes", [])
        if s.get("label") == "gate"
    ]


def render(entry: dict, pad: int, scale: int, show_gt: bool) -> np.ndarray | None:
    gray = cv2.imread(str(entry["image"]), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    H, W = gray.shape

    canvas = np.full((H + 2 * pad, W + 2 * pad, 3), PAD_BG, dtype=np.uint8)
    canvas[pad:pad + H, pad:pad + W] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + W, pad + H),
                  COLOR_IMG_BORDER, 1)

    def to_canvas(x, y):
        return int(round(x + pad)), int(round(y + pad))

    if show_gt:
        for gt in load_gt_quads(entry.get("label")):
            pts = np.array([to_canvas(p[0], p[1]) for p in gt], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=COLOR_GT, thickness=1)

    preds = predict_gates(gray)
    for quad in preds:
        pts_canvas = [to_canvas(c[0], c[1]) for c in quad]
        for i in range(4):
            cv2.line(canvas, pts_canvas[i], pts_canvas[(i + 1) % 4],
                     COLOR_EDGE, 1)
        for (cx, cy), name in zip(pts_canvas, CORNER_NAMES):
            cv2.circle(canvas, (cx, cy), 4, COLOR_KP, -1)
            cv2.putText(canvas, name, (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_KP, 1, cv2.LINE_AA)

    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_NEAREST)
    return canvas, len(preds)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--splits", type=Path, default=Path("dataset/splits.json"),
                     help="labeled-manifest source (used when --images-dir / --recording omitted)")
    src.add_argument("--images-dir", type=Path,
                     help="step through any directory of PNG images")
    src.add_argument("--recording", type=str,
                     help="shorthand for --images-dir recordings/<NAME>")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="when sourcing from --splits")
    parser.add_argument("--show-gt", action="store_true",
                        help="overlay ground-truth quads from labelme sidecars (green)")
    parser.add_argument("--pad", type=int, default=200)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    if args.recording:
        entries = collect_from_dir(Path("recordings") / args.recording)
        source_desc = f"recordings/{args.recording}"
    elif args.images_dir:
        entries = collect_from_dir(args.images_dir)
        source_desc = str(args.images_dir)
    else:
        entries = collect_from_splits(args.splits, args.split)
        source_desc = f"{args.splits} [{args.split}]"

    if not entries:
        print(f"No images to display under {source_desc}.")
        return

    print(f"Displaying {len(entries)} frame(s) from {source_desc}. "
          f"space=next, a=prev, q=quit.")

    i = 0
    while 0 <= i < len(entries):
        entry = entries[i]
        result = render(entry, args.pad, args.scale, args.show_gt)
        if result is None:
            print(f"missing image: {entry['image']}")
            i += 1
            continue
        canvas, n_preds = result

        label = f"{i + 1}/{len(entries)}  {entry['id']}  preds={n_preds}"
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("pose predictions", canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        elif key in (ord("a"), 81) and i > 0:
            i -= 1
        else:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
