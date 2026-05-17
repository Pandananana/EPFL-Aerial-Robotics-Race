"""Convert labelme polygon labels to 4-corner pose labels.

For each gate, the polygon is reduced to four corners in TL, TR, BR, BL order
with a per-corner visibility flag:

  v=2  corner is inside the image (taken directly from a polygon vertex)
  v=0  corner is off-image; coordinate is either a geometric extrapolation
       (single-corner-missing case) or a placeholder on the image edge
       (multi-corner-missing case, where the true off-image distance is
       unknowable)

Strategy by number of interior (non-boundary) polygon vertices K:
  - n < 4              skip (not enough info; written to `unconverted`)
  - K = 4              full gate; all corners v=2
  - K = 3              one corner off-image; extrapolate it by intersecting
                       the two gate edges that exit the frame on either side
  - K <= 2             two or more corners off-image; we cannot know how far
                       off they go, so each off-image corner is placed at the
                       image-edge clip point (quadrant assignment)
  - K >= 5             accidental extra vertex; quadrant assignment picks the
                       four best corners and discards the extras

Run from repo root:
    uv run python tools/convert_to_pose.py

Reads dataset/splits.json. Writes dataset/labels_pose/<run>/img_NNNNNN.json
(one per labeled frame). Originals under recordings/ are not modified.
"""

import argparse
import json
import math
from pathlib import Path

BOUNDARY_TOL = 2.0  # pixels; a polygon vertex within this distance of any
                    # image edge is treated as an image-boundary clip point,
                    # not a real gate corner.

# Frames whose converted pose labels were manually inspected and found to
# have bad off-image corner positions. Their original polygons are routed
# to `unconverted` so they can be relabeled (or excluded from training).
SKIP_FRAMES = {
    "20260513_112256_000082",
    "20260513_112256_000297",
    "20260513_112256_000299",
    "20260513_115203_000046",
    "20260513_115203_000052",
    "20260513_115203_000053",
    "20260513_115203_000111",
    "20260513_115203_000384",
    "20260513_115203_000385",
    "20260513_115203_000386",
    "20260513_115203_000578",
}


def on_image_boundary(p: tuple[float, float], W: int, H: int, tol: float = BOUNDARY_TOL) -> bool:
    x, y = p
    return x <= tol or x >= W - 1 - tol or y <= tol or y >= H - 1 - tol


def line_intersection(p1, p2, p3, p4):
    """Intersection of infinite line p1-p2 with infinite line p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def sort_tl_tr_br_bl(corners):
    """Sort 4 (x, y, v) corners clockwise starting at the upper-left.

    Uses angle from centroid for clockwise ordering, then rotates the sequence
    so the corner with smallest (x+y) starts the list. Robust for moderately
    rotated quadrilaterals; can be ambiguous for ~45° rotations (rare here)."""
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    cw = sorted(corners, key=lambda c: math.atan2(c[1] - cy, c[0] - cx))
    tl_i = min(range(4), key=lambda i: (cw[i][0] + cw[i][1], cw[i][0]))
    return [cw[(tl_i + i) % 4] for i in range(4)]


def _convert_k3(pts, n, interior_idx):
    """1 missing corner: find it by intersecting the two gate edges that
    exit the frame on either side of the missing corner."""
    corners = []
    for k in range(3):
        a_idx = interior_idx[k]
        b_idx = interior_idx[(k + 1) % 3]
        a = pts[a_idx]
        b = pts[b_idx]
        corners.append((a[0], a[1], 2))

        if (b_idx - a_idx) % n == 1:
            continue  # gate edge a->b is fully visible; no missing corner between

        next_after_a = pts[(a_idx + 1) % n]
        prev_before_b = pts[(b_idx - 1) % n]
        m = line_intersection(a, next_after_a, prev_before_b, b)
        if m is None:
            return None
        corners.append((m[0], m[1], 0))

    if len(corners) != 4:
        return None
    return sort_tl_tr_br_bl(corners)


def _convert_quadrant_fallback(pts, on_b):
    """Assign each of the four (TL, TR, BR, BL) slots to the polygon vertex
    falling in that quadrant relative to the polygon centroid, preferring
    interior vertices over boundary ones. Used for K=0, K=1, K>=5, and the
    diagonal K=2 case. v=2 for interior chosen vertices, v=0 for boundary."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    bb_xmin = min(p[0] for p in pts)
    bb_xmax = max(p[0] for p in pts)
    bb_ymin = min(p[1] for p in pts)
    bb_ymax = max(p[1] for p in pts)
    targets = [(bb_xmin, bb_ymin), (bb_xmax, bb_ymin),
               (bb_xmax, bb_ymax), (bb_xmin, bb_ymax)]

    def quadrant(p):
        if p[0] < cx and p[1] < cy:
            return 0
        if p[0] >= cx and p[1] < cy:
            return 1
        if p[0] >= cx and p[1] >= cy:
            return 2
        return 3

    chosen = []
    for q in range(4):
        cands = [(i, p) for i, p in enumerate(pts) if quadrant(p) == q]
        if not cands:
            return None
        interior_cands = [(i, p) for i, p in cands if not on_b[i]]
        pool = interior_cands if interior_cands else cands
        tx, ty = targets[q]
        best_i, best_p = min(pool, key=lambda ip: (ip[1][0] - tx) ** 2 + (ip[1][1] - ty) ** 2)
        chosen.append((best_p[0], best_p[1], 0 if on_b[best_i] else 2))
    return chosen  # already in TL, TR, BR, BL order


def convert_polygon(points, W: int, H: int):
    """Return [TL, TR, BR, BL] as (x, y, v) tuples, or None if unconvertible."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 4:
        return None  # 3-point polygons skipped per labeling spec
    on_b = [on_image_boundary(p, W, H) for p in pts]
    interior_idx = [i for i in range(n) if not on_b[i]]
    K = len(interior_idx)

    if K == 4:
        return sort_tl_tr_br_bl([(pts[i][0], pts[i][1], 2) for i in interior_idx])

    if K == 3:
        result = _convert_k3(pts, n, interior_idx)
        if result is not None:
            return result

    return _convert_quadrant_fallback(pts, on_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("dataset/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("dataset/labels_pose"))
    args = parser.parse_args()

    manifest = json.loads(args.splits.read_text())
    items = manifest["items"]

    n_total = 0
    n_ok = 0
    n_fail = 0
    n_skipped = 0
    by_split_fail = {}
    failed_frames = []

    for item in items:
        label_path = Path(item["label"])
        data = json.loads(label_path.read_text())
        W = data["imageWidth"]
        H = data["imageHeight"]
        force_skip = item["id"] in SKIP_FRAMES

        gates_out = []
        unconverted = []
        for shape in data.get("shapes", []):
            if shape.get("label") != "gate":
                continue
            n_total += 1
            corners = None if force_skip else convert_polygon(shape["points"], W, H)
            if corners is None:
                if force_skip:
                    n_skipped += 1
                else:
                    n_fail += 1
                    by_split_fail[item["split"]] = by_split_fail.get(item["split"], 0) + 1
                unconverted.append(shape["points"])
            else:
                n_ok += 1
                gates_out.append({"corners": [[c[0], c[1], c[2]] for c in corners]})

        if unconverted:
            failed_frames.append({
                "id": item["id"],
                "split": item["split"],
                "n_unconverted": len(unconverted),
            })

        out_path = args.out / item["run"] / f"img_{item['frame']:06d}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "image": item["image"],
            "image_width": W,
            "image_height": H,
            "gates": gates_out,
            "unconverted": unconverted,
        }, indent=2))

    print(f"Total gates:   {n_total}")
    print(f"Converted:     {n_ok}")
    print(f"Skipped:       {n_skipped}  (in SKIP_FRAMES list)")
    print(f"Needs review:  {n_fail}  (by split: {by_split_fail})")
    if failed_frames:
        print(f"\nFrames with unconverted gates ({len(failed_frames)}):")
        for r in failed_frames[:30]:
            print(f"  {r['split']:5s} {r['id']}  ({r['n_unconverted']} gate(s))")
        if len(failed_frames) > 30:
            print(f"  ... and {len(failed_frames) - 30} more")
    print(f"\nWrote pose labels to {args.out}/")


if __name__ == "__main__":
    main()
