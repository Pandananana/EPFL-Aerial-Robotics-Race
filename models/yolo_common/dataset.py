"""Build a shared YOLO dataset (OBB + Seg) from dataset/splits.json.

Layout produced (under <out>/):
    data.yaml
    images/{train,val,test}/<id>.png
    labels/{train,val,test}/<id>.txt

Split policy here:
  - train: manifest train items, minus the tail of run VAL_RUN (and a small
    buffer adjacent to that tail to keep near-duplicate frames out of train).
  - val:   the last VAL_TAIL labeled frames of run VAL_RUN. A contiguous
    tail-of-run slice avoids the consecutive-frame leakage that a random
    per-frame val split would cause.
  - test:  the manifest test items, exposed read-only so YOLO can report
    test-split metrics via `model.val(split='test')`. NEVER used for training.

Label format per line: `0 x1 y1 x2 y2 x3 y3 x4 y4` with normalized coords.
This is byte-identical for Ultralytics OBB (exactly 4 points) and Seg (polygon
of any vertex count), so the same on-disk dataset feeds both tasks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "dataset" / "splits.json"
DEFAULT_OUT = REPO_ROOT / "models" / "yolo_common" / "data"

VAL_RUN = "20260513_115203"
VAL_TAIL = 50
VAL_BUFFER = 5


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


def _partition_train_val(train_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Carve the val set off the tail of VAL_RUN, with a buffer dropped from train."""
    val_run_items = sorted(
        (it for it in train_items if it["run"] == VAL_RUN),
        key=lambda it: it["frame"],
    )
    needed = VAL_TAIL + VAL_BUFFER + 1
    if len(val_run_items) < needed:
        raise RuntimeError(
            f"Run {VAL_RUN} has only {len(val_run_items)} labeled frames; "
            f"need at least {needed} to carve val+buffer."
        )
    val = val_run_items[-VAL_TAIL:]
    buffer_ids = {it["id"] for it in val_run_items[-(VAL_TAIL + VAL_BUFFER):-VAL_TAIL]}
    val_ids = {it["id"] for it in val}
    train = [it for it in train_items if it["id"] not in val_ids and it["id"] not in buffer_ids]
    return train, val


def build(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_dir: Path = DEFAULT_OUT,
) -> Path:
    """Build the shared YOLO dataset; return the path to data.yaml."""
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
            lbl_src = REPO_ROOT / item["label"]
            img_dst = out_dir / "images" / split_name / f"{item['id']}.png"
            lbl_dst = out_dir / "labels" / split_name / f"{item['id']}.txt"
            shutil.copyfile(img_src, img_dst)
            _write_label(lbl_src, lbl_dst)

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: gate\n"
    )
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_text)

    print(f"Built shared YOLO dataset at {out_dir}")
    print(f"  train: {len(train_items)} images")
    print(f"  val:   {len(val_items)} images  (tail of {VAL_RUN}, buffer={VAL_BUFFER})")
    print(f"  test:  {len(test_items)} images  (held out, not used for training)")
    return yaml_path


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.manifest, args.out)


if __name__ == "__main__":
    main()
