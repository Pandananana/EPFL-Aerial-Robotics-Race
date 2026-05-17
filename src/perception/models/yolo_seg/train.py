from __future__ import annotations

import argparse
import random
import shutil
import string
from pathlib import Path

from src.perception.models.yolo_common import dataset as ds

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
BEST_DST = HERE / "best.pt"


def train(
    yaml_path: Path,
    base_model: str = "yolo26x-seg.pt",
    epochs: int = 100,
    imgsz: int = 320,
    batch: int = 32,
    name: str = "train",
) -> Path:
    """Run YOLO training; return the path to the best.pt checkpoint."""
    from ultralytics import YOLO

    model = YOLO(base_model)
    result = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        cache="ram",
        # Optimization
        cos_lr=True,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=5.0,
        patience=30,
        amp=True,
        # Regularization
        dropout=0.1,
        label_smoothing=0.05,
        weight_decay=0.0005,
        # Seg-specific: finer masks for thin LED outlines
        mask_ratio=2,
        overlap_mask=True,
        single_cls=True,
        # Shape / framing
        rect=True,
        close_mosaic=0,
        # Grayscale frames: hue/saturation jitter is wasted; keep value jitter.
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        bgr=0.0,
        # Drone rolls/pitches in flight but the camera is never inverted.
        degrees=20.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        # Composition: LED outlines are thin and the segment is the inner hole,
        # so copy_paste would paste a dark patch with no surrounding LEDs.
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )
    save_dir = Path(result.save_dir) if hasattr(result, "save_dir") else RUNS_DIR / name
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but best.pt not found at {best}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26x-seg.pt",
                        help="Base YOLO Seg checkpoint to fine-tune from.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Source frames are 324x244; rect=True keeps native AR.")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default=None,
                        help="Run name under runs/. Defaults to run_<random5>.")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="Reuse existing dataset on disk.")
    args = parser.parse_args()

    if args.name is None:
        args.name = "run_" + "".join(random.choices(string.ascii_lowercase, k=5))
        print(f"No --name given; using {args.name}")

    if args.skip_dataset:
        yaml_path = ds.DEFAULT_OUT / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"No prebuilt dataset at {yaml_path}")
    else:
        yaml_path = ds.build()

    best = train(
        yaml_path,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )
    shutil.copyfile(best, BEST_DST)
    print(f"Copied best checkpoint to {BEST_DST}")

    from ultralytics import YOLO

    print("Evaluating best checkpoint on held-out test split...")
    YOLO(best).val(
        data=str(yaml_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(RUNS_DIR),
        name=f"{args.name}_test",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
