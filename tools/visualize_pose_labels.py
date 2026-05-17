"""Step through pose labels produced by tools/convert_to_pose.py.

Each frame is rendered onto a padded canvas so off-image corners (v=0,
extrapolated) are visible outside the original image bounds.

  green dot  v=2  on-image corner
  red dot    v=0  off-image corner (extrapolated)
  yellow     gate quadrilateral TL->TR->BR->BL->TL
  white box  original image bounds

Keys: space / right = next, a / left = previous, q / Esc = quit.

Run from repo root:
    uv run python tools/visualize_pose_labels.py --split test
    uv run python tools/visualize_pose_labels.py --only-unconverted
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CORNER_NAMES = ("TL", "TR", "BR", "BL")
PAD_BG = (50, 50, 50)         # canvas color outside the image bounds
COLOR_VISIBLE = (0, 255, 0)   # BGR
COLOR_OFFIMG = (0, 0, 255)
COLOR_EDGE = (0, 255, 255)
COLOR_IMG_BORDER = (255, 255, 255)


def render(item, pose_data, pad: int, scale: int):
    image_path = Path(item["image"])
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    H, W = gray.shape

    # Pad and convert to BGR
    canvas = np.full((H + 2 * pad, W + 2 * pad, 3), PAD_BG, dtype=np.uint8)
    canvas[pad:pad + H, pad:pad + W] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + W, pad + H), COLOR_IMG_BORDER, 1)

    def to_canvas(x, y):
        return int(round(x + pad)), int(round(y + pad))

    for gate in pose_data.get("gates", []):
        corners = gate["corners"]
        pts_canvas = [to_canvas(c[0], c[1]) for c in corners]

        for i in range(4):
            cv2.line(canvas, pts_canvas[i], pts_canvas[(i + 1) % 4], COLOR_EDGE, 1)

        for (x, y, v), (cx, cy), name in zip(corners, pts_canvas, CORNER_NAMES):
            color = COLOR_VISIBLE if v == 2 else COLOR_OFFIMG
            cv2.circle(canvas, (cx, cy), 4, color, -1)
            cv2.putText(canvas, name, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)

    # Draw any unconverted polygons in magenta so the user can spot them
    for poly in pose_data.get("unconverted", []):
        magenta = (255, 0, 255)
        pts = np.array([to_canvas(p[0], p[1]) for p in poly], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=magenta, thickness=1)
        for p in pts:
            cv2.circle(canvas, tuple(p.tolist()), 3, magenta, -1)

    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_NEAREST)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits.json"))
    parser.add_argument("--pose-dir", type=Path, default=Path("dataset/labels_pose"))
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--only-unconverted", action="store_true",
                        help="show only frames that have at least one unconverted gate")
    parser.add_argument("--include-empty", action="store_true",
                        help="include frames with zero gates/unconverted (default: skip)")
    parser.add_argument("--pad", type=int, default=200)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.splits.read_text())
    items = manifest["items"]
    if args.split != "all":
        items = [it for it in items if it["split"] == args.split]

    entries = []
    for it in items:
        pose_path = args.pose_dir / it["run"] / f"img_{it['frame']:06d}.json"
        if not pose_path.exists():
            continue
        data = json.loads(pose_path.read_text())
        n_gates = len(data.get("gates", []))
        n_unconv = len(data.get("unconverted", []))
        if args.only_unconverted and n_unconv == 0:
            continue
        if not args.include_empty and n_gates == 0 and n_unconv == 0:
            continue
        entries.append((it, data, n_gates, n_unconv))

    if not entries:
        print("No frames to display (try --include-empty or drop --only-unconverted).")
        return

    print(f"Displaying {len(entries)} frame(s). space=next, a=prev, q=quit.")

    i = 0
    while 0 <= i < len(entries):
        item, pose_data, n_gates, n_unconv = entries[i]
        canvas = render(item, pose_data, args.pad, args.scale)
        if canvas is None:
            print(f"missing image: {item['image']}")
            i += 1
            continue

        label = (f"{i + 1}/{len(entries)}  {item['id']}  "
                 f"split={item['split']}  gates={n_gates}  unconverted={n_unconv}")
        cv2.putText(canvas, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("pose labels", canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        elif key in (ord("a"), 81) and i > 0:  # 81 = left arrow on some platforms
            i -= 1
        else:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
