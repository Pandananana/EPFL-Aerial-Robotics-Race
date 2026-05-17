"""Run the YOLO seg detector over recording dirs and emit labelme JSONs.

Skips any image that already has a sidecar JSON, so existing human labels are
never overwritten. Outputs match the labelme polygon schema used elsewhere in
this repo (see recordings/.../img_*.json) so the files can be opened directly
in labelme for review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.yolo_seg.detector import predict_gates


LABELME_VERSION = "6.2.0"
DEFAULT_DIRS = [
    Path("recordings/20260513_112256"),
    Path("recordings/20260513_115203"),
]


def build_payload(image_path: Path, quads, height: int, width: int) -> dict:
    shapes = []
    for quad in quads:
        shapes.append(
            {
                "label": "gate",
                "points": [[float(x), float(y)] for x, y in quad],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def process_dir(images_dir: Path, dry_run: bool) -> tuple[int, int, int]:
    pngs = sorted(images_dir.glob("*.png"))
    written = 0
    skipped = 0
    failed = 0
    for png in pngs:
        json_path = png.with_suffix(".json")
        if json_path.exists():
            skipped += 1
            continue
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  ! could not read {png}")
            failed += 1
            continue
        quads = predict_gates(gray)
        h, w = gray.shape[:2]
        payload = build_payload(png, quads, h, w)
        if dry_run:
            print(f"  would write {json_path} ({len(quads)} gates)")
        else:
            json_path.write_text(json.dumps(payload, indent=2))
            print(f"  wrote {json_path.name} ({len(quads)} gates)")
        written += 1
    return written, skipped, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dirs",
        type=Path,
        nargs="*",
        default=DEFAULT_DIRS,
        help="Recording dirs to process. Default: %(default)s",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_written = total_skipped = total_failed = 0
    for d in args.dirs:
        if not d.is_dir():
            print(f"skipping {d}: not a directory")
            continue
        print(f"processing {d}")
        w, s, f = process_dir(d, args.dry_run)
        total_written += w
        total_skipped += s
        total_failed += f
        print(f"  -> wrote {w}, skipped {s}, failed {f}")

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"\ntotal: {verb} {total_written}, skipped {total_skipped}"
        f"{f', failed {total_failed}' if total_failed else ''}"
    )


if __name__ == "__main__":
    main()
