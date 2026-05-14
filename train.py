"""Gate detector -- method 3.

Bright-pixel mask -> Hough line segments -> group parallel pairs into quads.

Pipeline:
    1. White tophat + absolute threshold -> bright-pixel mask. Tophat
       isolates thin LED structures from smooth ceiling / sky bloom.
    2. Morphological close so the dotted LED rectangle becomes a continuous
       ring; dilate a copy for the perimeter-coverage check.
    3. Each gate's interior is a dark hole surrounded by bright LEDs. Find
       interior dark connected components in the closed mask -- these
       localize gates even when adjacent LED rings would merge.
    4. For each hole, run HoughLinesP on the hole's own boundary (its edges
       are the inner LED edge, which is what GT traces); if too small for
       Hough, run it on the surrounding LED ring instead.
    5. Cluster lines by orientation (dominant + perpendicular buckets) and
       in each bucket pick the line closest to the hole centroid on each
       side, weighted by line length so short noisy segments don't win.
    6. Intersect the four chosen lines to assemble a candidate quad. Also
       compute the hole's min-area-rect as an alternative candidate.
    7. Score each candidate by what fraction of its perimeter sits on LED
       pixels (dilated mask) and keep the better one above a threshold.
"""

import cv2
import numpy as np


def _hough_lines(
    edges: np.ndarray,
    min_line: int,
    threshold: int = 15,
    max_gap: int = 10,
) -> np.ndarray | None:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=threshold,
        minLineLength=min_line,
        maxLineGap=max_gap,
    )
    if lines is None:
        return None
    return lines[:, 0, :].astype(np.float32)


def _intersect(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = line_a
    x3, y3, x4, y4 = line_b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)


def _quad_from_lines(
    lines: np.ndarray,
    center: np.ndarray,
    angle_tol: float = np.pi / 9,
) -> np.ndarray | None:
    if len(lines) < 4:
        return None

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.arctan2(dy, dx) % np.pi
    lengths = np.hypot(dx, dy)

    bins = 36
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, np.pi), weights=lengths)
    wrap = np.concatenate([hist[-3:], hist, hist[:3]])
    smooth = np.convolve(wrap, np.ones(5) / 5.0, mode="same")[3:-3]
    dom_angle = (int(np.argmax(smooth)) + 0.5) / bins * np.pi
    perp_angle = (dom_angle + np.pi / 2) % np.pi

    def ang_dist(a: np.ndarray, b: float) -> np.ndarray:
        return np.abs(((a - b + np.pi / 2) % np.pi) - np.pi / 2)

    in_dom = ang_dist(angles, dom_angle) < angle_tol
    in_perp = ang_dist(angles, perp_angle) < angle_tol

    if in_dom.sum() < 2 or in_perp.sum() < 2:
        return None

    def innermost_pair(mask: np.ndarray, target_angle: float):
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return None
        normal = np.array([-np.sin(target_angle), np.cos(target_angle)])
        midpoints = np.column_stack(
            [(lines[idx, 0] + lines[idx, 2]) / 2.0,
             (lines[idx, 1] + lines[idx, 3]) / 2.0]
        )
        signed = (midpoints - center) @ normal
        pos = signed > 0
        if pos.sum() == 0 or (~pos).sum() == 0:
            order = np.argsort(signed)
            return lines[idx[order[0]]], lines[idx[order[-1]]]
        pos_idx = idx[pos]
        neg_idx = idx[~pos]
        d_pos = signed[pos]
        d_neg = signed[~pos]
        len_pos = lengths[pos_idx]
        len_neg = lengths[neg_idx]
        # Score: prefer short distance from centroid (inner edge), break ties
        # by line length so noisy short segments don't beat a clean LED edge.
        score_pos = d_pos / np.maximum(len_pos, 1.0)
        score_neg = -d_neg / np.maximum(len_neg, 1.0)
        a = pos_idx[np.argmin(score_pos)]
        b = neg_idx[np.argmin(score_neg)]
        return lines[a], lines[b]

    res_dom = innermost_pair(in_dom, dom_angle)
    res_perp = innermost_pair(in_perp, perp_angle)
    if res_dom is None or res_perp is None:
        return None
    l1, l2 = res_dom
    l3, l4 = res_perp

    corners = []
    for la in (l1, l2):
        for lb in (l3, l4):
            p = _intersect(la, lb)
            if p is None:
                return None
            corners.append(p)
    corners = np.array(corners, dtype=np.float32)

    centroid = corners.mean(0)
    order = np.argsort(
        np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    )
    return corners[order]


def _perimeter_coverage(quad: np.ndarray, mask: np.ndarray, band: int = 3) -> float:
    """Fraction of the quad's perimeter that sits on bright-mask pixels.

    Rasterizes the 4 edges as a thin band and asks: what share of those band
    pixels are inside `mask`? Real gates should have most of their perimeter
    covered by LED pixels.
    """
    H, W = mask.shape
    perim = np.zeros_like(mask)
    cv2.polylines(perim, [quad.astype(np.int32)], isClosed=True, color=255,
                   thickness=band)
    band_area = int((perim > 0).sum())
    if band_area == 0:
        return 0.0
    overlap = int(((perim > 0) & (mask > 0)).sum())
    return overlap / band_area


def predict_gates(image: np.ndarray) -> list[np.ndarray]:
    """Predict gate quadrilaterals in an image.

    Strategy: each gate's interior is a dark hole surrounded by bright LEDs.
    Locating those holes splits adjacent gates whose LED rings merge after
    morphological closing. For each hole, run Hough on the LED ring nearby
    and assemble a quad from the inner edges.

    Returns a list of (4, 2) float arrays in cyclic order. Empty list means
    no gates detected.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # White tophat extracts thin bright structures (LED dots) and rejects
    # smooth bright regions (ceiling bloom, sky). Kernel must be larger than
    # a single LED stroke (~5 px) but small enough that we don't wipe out
    # entire dim gates. 15 works well at 324x244.
    ktop = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, ktop)
    # Combine tophat (LEDs above local mean) with a high absolute threshold
    # (saturated LEDs at 255 even in bright scenes).
    mask_top = (tophat > 25).astype(np.uint8) * 255
    mask_abs = (blur > 230).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_top, mask_abs)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    dilated = cv2.dilate(closed, k5, iterations=1)

    # Interior dark holes: pixels not in the bright mask and not connected to
    # the image border via the dark background. Use the un-dilated mask so
    # narrow oblique gates don't have their interior holes filled in.
    inv = cv2.bitwise_not(closed)
    n_h, hole_lbls, hole_stats, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=8
    )

    quads: list[np.ndarray] = []
    for i in range(1, n_h):
        x = int(hole_stats[i, cv2.CC_STAT_LEFT])
        y = int(hole_stats[i, cv2.CC_STAT_TOP])
        w = int(hole_stats[i, cv2.CC_STAT_WIDTH])
        h = int(hole_stats[i, cv2.CC_STAT_HEIGHT])
        a = int(hole_stats[i, cv2.CC_STAT_AREA])

        # Drop the dark background — it touches the image border.
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if a < 60 or w < 10 or h < 10:
            continue
        aspect = w / max(h, 1)
        if aspect > 5 or aspect < 1 / 5:
            continue
        # Hole shouldn't be too sparse inside its bbox (real gate interiors
        # are mostly contiguous). Concentric gates produce annular holes
        # with a bright island, so the bar stays low.
        if a / float(w * h) < 0.30:
            continue

        # First try: the hole's own boundary IS the inner edge of the LED
        # ring (which is what GT traces). Hough lines on it are usually the
        # most accurate when the hole is big enough.
        pad_h = 4
        x0h, y0h = max(0, x - pad_h), max(0, y - pad_h)
        x1h, y1h = min(W, x + w + pad_h), min(H, y + h + pad_h)
        hole_mask = ((hole_lbls[y0h:y1h, x0h:x1h] == i).astype(np.uint8)) * 255
        h_edges = cv2.Canny(hole_mask, 50, 150)
        min_line = max(6, int(0.25 * min(w, h)))
        lines = _hough_lines(h_edges, min_line)
        if lines is not None and len(lines) >= 4:
            lines[:, [0, 2]] += x0h
            lines[:, [1, 3]] += y0h
        else:
            # Fallback: Hough on the surrounding LED ring (closed mask).
            pad_r = max(10, int(0.45 * max(w, h)))
            x0, y0 = max(0, x - pad_r), max(0, y - pad_r)
            x1, y1 = min(W, x + w + pad_r), min(H, y + h + pad_r)
            region = closed[y0:y1, x0:x1]
            edges = cv2.Canny(region, 50, 150)
            min_line = max(8, int(0.3 * min(w, h)))
            lines = _hough_lines(edges, min_line)
            if lines is None or len(lines) < 4:
                continue
            lines[:, [0, 2]] += x0
            lines[:, [1, 3]] += y0

        ys_c, xs_c = np.where(hole_lbls == i)
        center = np.array([xs_c.mean(), ys_c.mean()], dtype=np.float32)

        # Two candidates: Hough-line intersection quad and the hole's own
        # min-area-rect. Keep the one whose perimeter better hugs the LEDs.
        candidates: list[np.ndarray] = []
        hough_quad = _quad_from_lines(lines, center)
        if hough_quad is not None and cv2.contourArea(hough_quad) > 120:
            candidates.append(hough_quad)

        hole_pts = np.column_stack([xs_c, ys_c]).astype(np.float32)
        if len(hole_pts) >= 4:
            rect = cv2.minAreaRect(hole_pts)
            mar_quad = cv2.boxPoints(rect).astype(np.float32)
            if cv2.contourArea(mar_quad) > 120:
                candidates.append(mar_quad)

        if not candidates:
            continue

        scored = [
            (_perimeter_coverage(q, dilated, band=3), q) for q in candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        score, quad = scored[0]
        if score < 0.70:
            continue

        quad[:, 0] = np.clip(quad[:, 0], 0, W - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, H - 1)
        quads.append(quad)

    return quads


def main():
    print("train.py method 3: bright mask -> HoughLinesP -> quad assembly.")


if __name__ == "__main__":
    main()
