"""Build dataset/splits.json from labeled images across all recordings.

Policy (encoded below in SPLIT_BY_RUN):
  20260513_112205       -> test  (unrelated short flight, held out)
  everything else       -> train

Re-run this whenever you label more images. The manifest is idempotent and
deterministic — it just reflects whatever .json sidecars currently exist.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SPLIT_BY_RUN = {
    "20260513_112205": "test",
    "20260513_112256": "train",
    "20260513_115203": "train",
}


def load_label(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_items(recordings_dir: Path, repo_root: Path) -> list[dict]:
    items = []
    recordings_abs = recordings_dir.resolve()
    for run_dir in sorted(recordings_abs.iterdir()):
        if not run_dir.is_dir():
            continue
        run = run_dir.name
        if run not in SPLIT_BY_RUN:
            print(f"warning: run '{run}' has no split assignment — skipping")
            continue
        split = SPLIT_BY_RUN[run]

        for label_path in sorted(run_dir.glob("img_*.json")):
            image_path = label_path.with_suffix(".png")
            if not image_path.exists():
                print(f"warning: label without image: {label_path}")
                continue

            data = load_label(label_path)
            n_gates = sum(1 for s in data.get("shapes", []) if s.get("label") == "gate")
            frame = int(image_path.stem.split("_")[1])

            items.append({
                "id": f"{run}_{frame:06d}",
                "image": str(image_path.relative_to(repo_root)),
                "label": str(label_path.relative_to(repo_root)),
                "run": run,
                "frame": frame,
                "split": split,
                "n_gates": n_gates,
            })
    return items


def summarize(items: list[dict]) -> dict:
    by_split = Counter(it["split"] for it in items)
    gates_by_split: Counter[str] = Counter()
    empty_by_split: Counter[str] = Counter()
    by_run: Counter[str] = Counter()
    for it in items:
        gates_by_split[it["split"]] += it["n_gates"]
        if it["n_gates"] == 0:
            empty_by_split[it["split"]] += 1
        by_run[it["run"]] += 1
    return {
        "images_per_split": dict(by_split),
        "gates_per_split": dict(gates_by_split),
        "empty_images_per_split": dict(empty_by_split),
        "images_per_run": dict(by_run),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--out", type=Path, default=Path("dataset/splits.json"))
    args = parser.parse_args()

    repo_root = Path.cwd()
    items = collect_items(args.recordings, repo_root)
    stats = summarize(items)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": SPLIT_BY_RUN,
        "stats": stats,
        "items": items,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {args.out} with {len(items)} items")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
