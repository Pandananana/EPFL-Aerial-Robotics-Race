"""Build a YOLO pose dataset from data/labels/pose/.

Layout produced (under <out>/):
    data.yaml
    images/{train,val,test}/<id>.png
    labels/{train,val,test}/<id>.txt

Reads pose labels from `data/labels/pose/<run>/img_NNNNNN.json` (produced
by `tools/convert_to_pose.py`), not from the labelme seg JSONs under
data/labels/seg/.
The split partition (train/val/test) is shared with the polygon dataset
builder so the held-out test run stays consistent across model types.

Label format per line:
    `0 cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4`

All coords normalized by image dimensions. Keypoint order is TL, TR, BR, BL
(matching `tools/convert_to_pose.py`). Visibility mapping:
  source v=2 (on-image)   -> 2 (labeled, visible)
  source v=0 (off-image)  -> 1 (labeled, occluded)
The bbox is the axis-aligned bounding box of the four corners (clipped to
the image for on-image extent; off-image corners are still supervised via
keypoint loss with v=1).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .dataset import (
    DEFAULT_MANIFEST,
    REPO_ROOT,
    VAL_BUFFER,
    VAL_RUN,
    VAL_TAIL,
    _partition_train_val,
)

DEFAULT_POSE_LABELS = REPO_ROOT / "data" / "labels" / "pose"
DEFAULT_OUT = REPO_ROOT / "src" / "perception" / "models" / "yolo_pose" / "data"

# TL, TR, BR, BL — on horizontal flip, swap TL<->TR and BR<->BL.
FLIP_IDX = [1, 0, 3, 2]


def _gate_to_pose_line(corners: list[list[float]], w: int, h: int) -> str | None:
    """Encode one gate as a YOLO pose label line, or None to skip.

    `corners` is a 4-list of [x, y, v] with v in {0, 2}. Coordinates may be
    outside the image bounds (off-image corners). We clip the bbox to the
    image so YOLO's bbox loss stays well-defined, while keypoint coords are
    written unclipped so the network learns the true geometric position.
    """
    if len(corners) != 4:
        return None
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    vs_src = [int(c[2]) for c in corners]

    # AABB of the gate, clipped to image for the bbox term.
    x_min = max(0.0, min(xs))
    x_max = min(float(w), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(h), max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    cx = (x_min + x_max) / 2.0 / w
    cy = (y_min + y_max) / 2.0 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for x, y, vs in zip(xs, ys, vs_src):
        v_out = 2 if vs == 2 else 1
        parts.append(f"{x / w:.6f} {y / h:.6f} {v_out}")
    return " ".join(parts)


def _write_label(pose_json: Path, out_txt: Path) -> None:
    data = json.loads(pose_json.read_text())
    w = int(data["image_width"])
    h = int(data["image_height"])
    lines: list[str] = []
    for gate in data.get("gates", []):
        line = _gate_to_pose_line(gate["corners"], w, h)
        if line is not None:
            lines.append(line)
    out_txt.write_text("\n".join(lines))


def build(
    manifest_path: Path = DEFAULT_MANIFEST,
    pose_labels_dir: Path = DEFAULT_POSE_LABELS,
    out_dir: Path = DEFAULT_OUT,
) -> Path:
    """Build the pose YOLO dataset; return the path to data.yaml."""
    manifest = json.loads(manifest_path.read_text())
    train_pool = [it for it in manifest["items"] if it["split"] == "train"]
    test_items = [it for it in manifest["items"] if it["split"] == "test"]
    if not train_pool:
        raise RuntimeError("No train items in manifest")

    train_items, val_items = _partition_train_val(train_pool)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split_name, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        for item in items:
            img_src = REPO_ROOT / item["image"]
            lbl_src = pose_labels_dir / item["run"] / f"img_{item['frame']:06d}.json"
            if not lbl_src.exists():
                raise FileNotFoundError(f"Missing pose label: {lbl_src}")
            img_dst = out_dir / "images" / split_name / f"{item['id']}.png"
            lbl_dst = out_dir / "labels" / split_name / f"{item['id']}.txt"
            shutil.copyfile(img_src, img_dst)
            _write_label(lbl_src, lbl_dst)

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "kpt_shape: [4, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "names:\n"
        "  0: gate\n"
    )
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_text)

    print(f"Built pose YOLO dataset at {out_dir}")
    print(f"  train: {len(train_items)} images")
    print(f"  val:   {len(val_items)} images  (tail of {VAL_RUN}, buffer={VAL_BUFFER})")
    print(f"  test:  {len(test_items)} images  (held out, not used for training)")
    return yaml_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pose-labels", type=Path, default=DEFAULT_POSE_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.manifest, args.pose_labels, args.out)


if __name__ == "__main__":
    main()
