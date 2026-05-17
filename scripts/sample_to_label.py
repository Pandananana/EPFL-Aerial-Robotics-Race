"""Sample N random unlabeled images and symlink them into a working directory for labelme.

Run labelme with --output pointing at data/labels/seg/<run>/ so the .json
sidecars land in the label tree (not the symlinks dir, not the images dir).
"""

import argparse
import random
from pathlib import Path

DEFAULT_LABELS_ROOT = Path("data/labels/seg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path, help="Directory containing img_*.png (e.g. data/recordings/<run>)")
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT,
                        help="Root dir for seg labels. Run name (images_dir.name) is appended.")
    parser.add_argument("--n", type=int, default=50, help="How many to sample")
    parser.add_argument("--out", type=Path, default=Path("to_label"), help="Working dir for symlinks")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = parser.parse_args()

    labels_dir = args.labels_root / args.images_dir.name
    images = sorted(args.images_dir.glob("*.png"))
    unlabeled = [p for p in images if not (labels_dir / p.with_suffix(".json").name).exists()]
    print(f"{args.images_dir}: {len(images)} images, {len(unlabeled)} unlabeled")

    if not unlabeled:
        print("Nothing left to label.")
        return

    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(unlabeled, min(args.n, len(unlabeled)))

    args.out.mkdir(exist_ok=True)
    for stale in args.out.glob("*.png"):
        if stale.is_symlink():
            stale.unlink()

    for p in sample:
        (args.out / p.name).symlink_to(p.resolve())

    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Symlinked {len(sample)} images into {args.out}/")
    print()
    print("Next:")
    print(
        f"  uvx labelme {args.out} --output {labels_dir} "
        f"--labels gate --nodata --autosave"
    )


if __name__ == "__main__":
    main()
