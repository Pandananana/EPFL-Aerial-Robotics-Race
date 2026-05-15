"""Build a YOLO OBB dataset from dataset/splits.json.

Layout produced (under <out>/):
    data.yaml
    images/train/<id>.png
    images/val/<id>.png
    labels/train/<id>.txt
    labels/val/<id>.txt

The "val" split here is a deterministic slice of the manifest's train items,
NOT the manifest's test items — the test split must stay held out for test.py.

Label format per line: `0 x1 y1 x2 y2 x3 y3 x4 y4` with normalized coords.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "dataset" / "splits.json"
DEFAULT_OUT = REPO_ROOT / "models" / "yolo_obb" / "data"


def _quad_to_yolo_line(points: list[list[float]], w: int, h: int) -> str:
    coords = []
    for x, y in points:
        coords.append(f"{x / w:.6f}")
        coords.append(f"{y / h:.6f}")
    return "0 " + " ".join(coords)


def _write_label(label_json: Path, out_txt: Path) -> None:
    data = json.loads(label_json.read_text())
    w = int(data["imageWidth"])
    h = int(data["imageHeight"])
    lines = []
    for shape in data.get("shapes", []):
        if shape.get("label") != "gate":
            continue
        pts = shape["points"]
        if len(pts) != 4:
            continue
        lines.append(_quad_to_yolo_line(pts, w, h))
    out_txt.write_text("\n".join(lines))


def build(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_dir: Path = DEFAULT_OUT,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Path:
    """Build the YOLO OBB dataset; return the path to data.yaml."""
    manifest = json.loads(manifest_path.read_text())
    train_items = [it for it in manifest["items"] if it["split"] == "train"]
    if not train_items:
        raise RuntimeError("No train items in manifest")

    rng = random.Random(seed)
    shuffled = train_items.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split_name, items in (("train", train_items), ("val", val_items)):
        for item in items:
            img_src = REPO_ROOT / item["image"]
            lbl_src = REPO_ROOT / item["label"]
            img_dst = out_dir / "images" / split_name / f"{item['id']}.png"
            lbl_dst = out_dir / "labels" / split_name / f"{item['id']}.txt"
            shutil.copyfile(img_src, img_dst)
            _write_label(lbl_src, lbl_dst)

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: gate\n"
    )
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_text)

    print(f"Built YOLO OBB dataset at {out_dir}")
    print(f"  train: {len(train_items)} images")
    print(f"  val:   {len(val_items)} images")
    return yaml_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.manifest, args.out, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
