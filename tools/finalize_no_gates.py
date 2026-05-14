"""Write empty-shapes labelme JSONs for sampled images that weren't labeled.

After a labeling batch in labelme: anything in to_label/ that doesn't have a
matching .json in the originals directory was reviewed and found to contain
no gates (or no gates passing the size/visibility thresholds). This script
materializes that as an empty labelme JSON so those decisions are recorded.

Run this AFTER you're done labeling a batch, BEFORE re-sampling.
"""

import argparse
import json
from pathlib import Path

import cv2


LABELME_VERSION = "5.5.0"


def write_empty_json(image_path: Path, json_path: Path) -> None:
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    payload = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    json_path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path, help="Directory with the original PNGs and JSONs")
    parser.add_argument("--to-label", type=Path, default=Path("to_label"), help="Symlink working dir")
    parser.add_argument(
        "--all-unlabeled",
        action="store_true",
        help="Ignore --to-label and treat every unlabeled PNG in images_dir as reviewed-no-gates.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all_unlabeled:
        candidates = sorted(args.images_dir.glob("*.png"))
        source_desc = f"{args.images_dir}/ (all-unlabeled mode)"
    else:
        candidates = sorted(args.to_label.glob("*.png"))
        source_desc = f"{args.to_label}/"

    if not candidates:
        print(f"No PNGs in {source_desc} — nothing to finalize.")
        return

    written = 0
    skipped = 0
    for png in candidates:
        original = args.images_dir / png.name
        json_path = original.with_suffix(".json")
        if json_path.exists():
            skipped += 1
            continue
        if args.dry_run:
            print(f"would write empty JSON for {png.name}")
        else:
            write_empty_json(original, json_path)
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {written} empty JSONs; {skipped} already had labels.")


if __name__ == "__main__":
    main()
