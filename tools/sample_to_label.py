"""Sample N random unlabeled images and symlink them into a working directory for labelme.

Run labelme with --output pointing back at the original images dir, so the
.json sidecars land next to the real PNGs (not the symlinks).
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path, help="Directory containing img_*.png")
    parser.add_argument("--n", type=int, default=50, help="How many to sample")
    parser.add_argument("--out", type=Path, default=Path("to_label"), help="Working dir for symlinks")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = parser.parse_args()

    images = sorted(args.images_dir.glob("*.png"))
    unlabeled = [p for p in images if not p.with_suffix(".json").exists()]
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

    print(f"Symlinked {len(sample)} images into {args.out}/")
    print()
    print("Next:")
    print(
        f"  uvx labelme {args.out} --output {args.images_dir} "
        f"--labels gate --nodata --autosave"
    )


if __name__ == "__main__":
    main()
