#!/usr/bin/env python3
"""
Topographical map ➜ SVG layer extractor (clean, fixed)

Features
========
• Reads JPG/PNG/TIF.
• Isolates contour lines (HSV mask or adaptive threshold).
• Traces to vector paths; groups by peak (disconnected region) and nesting depth (level).
• Exports:
  - one SVG per discovered closed loop (per‑contour),
  - per‑peak (all levels + per level),
  - global per‑level,
  - overview of all layers.
• `--single-line`: thin to 1‑px centerlines before tracing (single path per contour).
  - Select method: ximgproc Zhang/Guo, scikit‑image, or fallback morphological.
  - `--single-line-dilate N`: pre‑dilate to bridge micro gaps before thinning.
  - `--single-line-bridge P`: post‑bridge endpoints within P px after thinning.
• `--normalize-width`: optionally normalize line thickness (thickest/thinnest/average) before thinning.
• `--fill-lines`: render filled bands (even‑odd fill) instead of strokes.
• `--thin`: force stroked output to 1px.

Dependencies
------------
  pip install opencv-python numpy svgwrite
  # Optional backends for improved skeletons:
  pip install scikit-image     # enables skimage skeletonize
  # OpenCV ximgproc if available provides better thinning backends

Usage example
-------------
  python topo_to_svg.py input.png --out out --single-line --thin \
      --normalize-width average --single-line-bridge 3 --dp-eps 1.0

  # Filled bands (no inner/outer double edges in the raster)
  python topo_to_svg.py input.png --out out --fill-lines --dp-eps 1.0
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import svgwrite

# -----------------------------
# Types
# -----------------------------
Point = Tuple[float, float]

# -----------------------------
# Helpers / utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def approx_poly(points: np.ndarray, eps: float) -> np.ndarray:
    """Douglas–Peucker simplification on a closed contour (Nx1x2)."""
    peri = max(1.0, cv2.arcLength(points, True))
    epsilon = (eps / 100.0) * peri if eps > 1 else eps
    epsilon = max(0.5, float(epsilon))  # avoid epsilon=0 creating huge paths
    return cv2.approxPolyDP(points, epsilon, True)


def to_svg(paths: List[List[Point]], size: Tuple[int, int], stroke: float, fn: Path,
           thin: bool, fill_lines: bool = False) -> None:
    w, h = int(size[0]), int(size[1])
    dwg = svgwrite.Drawing(str(fn), size=(f"{w}px", f"{h}px"), profile="tiny")
    dwg.viewbox(0, 0, w, h)

    if fill_lines:
        # Combine all rings into one compound path with even-odd fill
        d_parts: List[str] = []
        for pts in paths:
            if len(pts) < 2:
                continue
            segs = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
            segs += [f"L {x:.2f} {y:.2f}" for (x, y) in pts[1:]]
            segs.append("Z")
            d_parts.append(" ".join(segs))
        if d_parts:
            path = dwg.path(" ".join(d_parts))
            path.update({"fill": "black", "stroke": "none", "fill-rule": "evenodd"})
            dwg.add(path)
        dwg.save()
        return

    stroke_width = 1.0 if thin else stroke
    grp = dwg.g(id="contours", fill="none", stroke="black", stroke_width=stroke_width)
    for pts in paths:
        if len(pts) < 2:
            continue
        segs = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        segs += [f"L {x:.2f} {y:.2f}" for (x, y) in pts[1:]]
        segs.append("Z")
        grp.add(dwg.path(" ".join(segs)))
    dwg.add(grp)
    dwg.save()


# -----------------------------
# Core image processing
# -----------------------------

def isolate_lines(img_bgr: np.ndarray,
                  hsv_low: Optional[Tuple[int, int, int]],
                  hsv_high: Optional[Tuple[int, int, int]],
                  close_k: int,
                  close_iters: int) -> np.ndarray:
    """Return binary uint8 image (0/255) with elevation lines highlighted."""
    den = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    if hsv_low is not None and hsv_high is not None:
        hsv = cv2.cvtColor(den, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low, dtype=np.uint8), np.array(hsv_high, dtype=np.uint8))
        bin_img = mask
    else:
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 5)
        bin_img = cv2.medianBlur(bin_img, 3)

    if close_k > 0 and close_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, close_k), max(1, close_k)))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=close_iters)
    return bin_img.astype(np.uint8)


def find_contour_forest(bin_img: np.ndarray,
                        min_perimeter: float,
                        dp_eps: float) -> Tuple[List[np.ndarray], np.ndarray]:
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], np.zeros((0, 4), dtype=np.int32)

    simp: List[np.ndarray] = []
    keep: List[int] = []
    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        if peri < float(min_perimeter):
            continue
        if dp_eps > 0:
            cnt = approx_poly(cnt, dp_eps)
        simp.append(cnt)
        keep.append(i)

    if not keep:
        return [], np.zeros((0, 4), dtype=np.int32)

    # Map original hierarchy to kept indices
    hierarchy = hierarchy[0]
    kept_h = hierarchy[keep]
    return simp, kept_h


def build_trees(contours: List[np.ndarray]) -> Dict[int, Dict]:
    """Group contours into trees using bbox shortlist + point-in-polygon confirmation."""
    n = len(contours)
    if n == 0:
        return {}

    bboxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]

    parent = [-1] * n
    for i in range(n):
        xi, yi, wi, hi = bboxes[i]
        bx0, by0, bx1, by1 = xi, yi, xi + wi, yi + hi
        best = -1
        best_area = float('inf')
        # shortlist by bbox containment
        for j in range(n):
            if i == j:
                continue
            xj, yj, wj, hj = bboxes[j]
            if xj <= bx0 and yj <= by0 and xj + wj >= bx1 and yj + hj >= by1:
                # confirm by point-in-polygon using a vertex of child
                pt = tuple(contours[i][0][0])  # (x,y)
                inside = cv2.pointPolygonTest(contours[j], pt, False) >= 0
                if inside and areas[j] < best_area:
                    best_area = areas[j]
                    best = j
        parent[i] = best

    children: List[List[int]] = [[] for _ in range(n)]
    roots: List[int] = []
    for i, p in enumerate(parent):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)

    trees: Dict[int, Dict] = {}
    for tidx, r in enumerate(roots):
        q = deque([(r, 0)])
        depths: Dict[int, List[int]] = defaultdict(list)
        nodes: List[int] = []
        while q:
            idx, d = q.popleft()
            nodes.append(idx)
            depths[d].append(idx)
            for ch in children[idx]:
                q.append((ch, d + 1))
        trees[tidx] = {"root": r, "nodes": nodes, "depths": depths}
    return trees


def contour_to_points(cnt: np.ndarray) -> List[Point]:
    pts = cnt.reshape(-1, 2)
    return [(float(x), float(y)) for (x, y) in pts]

# -----------------------------
# Skeletonization & width normalization
# -----------------------------

def skeletonize_ximgproc(bin_img: np.ndarray, method: str = "zhang") -> Optional[np.ndarray]:
    try:
        import cv2.ximgproc as xip
    except Exception:
        return None
    t = xip.THINNING_ZHANGSUEN if method == "zhang" else xip.THINNING_GUOHALL
    skel = xip.thinning(bin_img, thinningType=t)
    return skel.astype(np.uint8)


def skeletonize_skimage(bin_img: np.ndarray) -> Optional[np.ndarray]:
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return None
    skel = skeletonize((bin_img > 0)).astype(np.uint8) * 255
    return skel


def skeletonize_cv(bin_img: np.ndarray) -> np.ndarray:
    # Morphological skeleton (fallback)
    img = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def skeletonize_image(bin_img: np.ndarray, method: str = "auto", dilate_iters: int = 1) -> np.ndarray:
    # Pre-bridge tiny gaps
    if dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_img = cv2.dilate(bin_img, k, iterations=dilate_iters)

    # Prefer ximgproc
    if method in ("auto", "ximgproc", "zhang", "guohall"):
        m = "zhang" if method in ("auto", "ximgproc", "zhang") else "guohall"
        sk = skeletonize_ximgproc(bin_img, m)
        if sk is not None:
            return sk

    # Try scikit-image
    if method in ("auto", "skimage"):
        sk = skeletonize_skimage(bin_img)
        if sk is not None:
            return sk

    return skeletonize_cv(bin_img)


def bridge_skeleton(skel: np.ndarray, max_dist: int) -> np.ndarray:
    if max_dist <= 0:
        return skel
    img = skel.copy()
    # endpoints = pixels with exactly 1 neighbor in 8-connectivity
    kernel = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D((img > 0).astype(np.uint8), -1, kernel)
    endpoints = np.argwhere(((img > 0) & (nb == 2)))  # 1 neighbor + self
    if len(endpoints) == 0:
        return img
    used: set[int] = set()
    m2 = max_dist * max_dist
    for i, (y1, x1) in enumerate(endpoints):
        if i in used:
            continue
        best_j = -1
        best_d2 = m2
        for j in range(i + 1, len(endpoints)):
            if j in used:
                continue
            y2, x2 = endpoints[j]
            d2 = (y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1)
            if d2 <= best_d2:
                best_d2 = d2
                best_j = j
        if best_j >= 0:
            y2, x2 = endpoints[best_j]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
            used.add(i)
            used.add(best_j)
    return img


def compute_target_radius(bin_img: np.ndarray, skel: np.ndarray,
                          mode: str, target: int, percentile: float) -> int:
    mask = (bin_img > 0).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    vals = dist[skel > 0]
    if vals.size == 0:
        return max(1, target if target > 0 else 1)
    if target > 0:
        r = float(target)
    else:
        percentile = float(np.clip(percentile, 0.0, 100.0))
        if mode == 'thickest':
            r = float(np.percentile(vals, percentile))
        elif mode == 'thinnest':
            r = float(np.percentile(vals, 100.0 - percentile))
        else:  # average
            r = float(np.median(vals))
    return max(1, int(round(r)))


def normalize_width_from_skeleton(bin_img: np.ndarray, mode: str, target: int,
                                  percentile: float, skel_method: str, pre_dilate: int) -> np.ndarray:
    skel = skeletonize_image(bin_img, method=skel_method, dilate_iters=pre_dilate)
    r = compute_target_radius(bin_img, skel, mode, target, percentile)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    uniform = cv2.dilate(skel, k, iterations=1)
    return uniform

# -----------------------------
# Orchestration
# -----------------------------

def process(in_path: Path,
            out_dir: Path,
            hsv_low: Optional[Tuple[int, int, int]],
            hsv_high: Optional[Tuple[int, int, int]],
            min_perimeter: float,
            close_k: int,
            close_iters: int,
            dp_eps: float,
            stroke: float,
            thin: bool,
            single_line: bool,
            single_line_method: str,
            single_line_dilate: int,
            single_line_bridge: int,
            norm_width_mode: str,
            norm_width_target: int,
            norm_width_percentile: float,
            fill_lines: bool) -> None:

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    h, w = img.shape[:2]
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'layers')
    ensure_dir(out_dir / 'peaks')
    ensure_dir(out_dir / 'contours')

    bin_img = isolate_lines(img, hsv_low, hsv_high, close_k, close_iters)

    # Optional: normalize thickness prior to thinning
    if single_line and norm_width_mode != 'none':
        bin_img = normalize_width_from_skeleton(
            bin_img,
            mode=norm_width_mode,
            target=norm_width_target,
            percentile=norm_width_percentile,
            skel_method=single_line_method,
            pre_dilate=single_line_dilate,
        )

    # Optional: skeletonize to single-pixel centerlines
    if single_line:
        bin_img = skeletonize_image(bin_img, method=single_line_method, dilate_iters=0)
        bin_img = bridge_skeleton(bin_img, max_dist=single_line_bridge)

    contours, _ = find_contour_forest(bin_img, min_perimeter=min_perimeter, dp_eps=dp_eps)
    if len(contours) == 0:
        logging.warning("No contours found above min_perimeter. Try lowering --min-perimeter or adjusting the mask.")
        return

    trees = build_trees(contours)

    # Build mapping: contour index -> (peak_id, level_depth, local_seq)
    idx_info: Dict[int, Tuple[int, int, int]] = {}
    for t_id, tree in trees.items():
        for depth, idxs in sorted(tree['depths'].items()):
            for local_seq, idx in enumerate(idxs):
                idx_info[idx] = (t_id, depth, local_seq)

    # per-contour SVGs (filenames include peak/level/seq)
    for i, cnt in enumerate(contours):
        pts = contour_to_points(cnt)
        peak_id, level_depth, local_seq = idx_info.get(i, (-1, -1, i))
        fn = out_dir / 'contours' / f'contour{local_seq:03d}_level{level_depth:03d}_peak{peak_id:03d}.svg'
        to_svg([pts], (w, h), stroke, fn, thin, fill_lines)

    # global per-level & per-peak
    global_levels: Dict[int, List[List[Point]]] = defaultdict(list)
    for t_id, tree in trees.items():
        peak_dir = out_dir / 'peaks' / f'peak_{t_id:03d}'
        ensure_dir(peak_dir)
        level_paths: Dict[int, List[List[Point]]] = defaultdict(list)
        for depth, idxs in tree['depths'].items():
            for idx in idxs:
                level_paths[depth].append(contour_to_points(contours[idx]))
                global_levels[depth].append(contour_to_points(contours[idx]))
        # per-level for this peak
        for depth, paths in sorted(level_paths.items()):
            to_svg(paths, (w, h), stroke, peak_dir / f'level_{depth:02d}.svg', thin, fill_lines)
        # combined for this peak
        all_paths = [p for ps in level_paths.values() for p in ps]
        to_svg(all_paths, (w, h), stroke, peak_dir / 'all_levels.svg', thin, fill_lines)

    # global per-level
    for depth, paths in sorted(global_levels.items()):
        to_svg(paths, (w, h), stroke, out_dir / 'layers' / f'level_{depth:02d}.svg', thin, fill_lines)

    # overview
    overview = [p for ps in global_levels.values() for p in ps]
    to_svg(overview, (w, h), stroke, out_dir / 'overview_all_layers.svg', thin, fill_lines)

    logging.info("✅ Done. Contours: %d | Peaks: %d | Levels: %d", len(contours), len(trees), len(global_levels))
    logging.info("Output directory: %s", out_dir)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a topo map image into per-elevation-layer SVGs.")

    # Input & preprocessing
    parser.add_argument('input', type=Path, help='Input image (png/jpg/tif)')
    parser.add_argument('--out', type=Path, default=Path('out'), help='Output directory')
    parser.add_argument('--hsv-low', type=int, nargs=3, metavar=('H', 'S', 'V'), help='Lower HSV bound for contour color mask')
    parser.add_argument('--hsv-high', type=int, nargs=3, metavar=('H', 'S', 'V'), help='Upper HSV bound for contour color mask')
    parser.add_argument('--close-k', type=int, default=3, help='Structuring element size for closing (pixels)')
    parser.add_argument('--close-iters', type=int, default=1, help='Iterations for morphological closing')

    # Contour filtering / simplification
    parser.add_argument('--min-perimeter', type=float, default=100.0, help='Ignore tiny contours below this perimeter (pixels)')
    parser.add_argument('--dp-eps', type=float, default=1.5, help='Douglas–Peucker epsilon. If >1 treat as % of perimeter; if <=1 absolute pixels')

    # Output styling
    parser.add_argument('--stroke', type=float, default=1.0, help='SVG stroke width (px) for stroked output')
    parser.add_argument('--thin', action='store_true', help='Force stroked SVG contours to be exactly 1px wide')
    parser.add_argument('--fill-lines', action='store_true', help='Render filled bands with even-odd fill (no stroked outlines)')

    # Single-line skeletonization options
    parser.add_argument('--single-line', action='store_true', help='Skeletonize raster lines to 1‑pixel centerlines (one path per contour)')
    parser.add_argument('--single-line-method', choices=['auto', 'ximgproc', 'zhang', 'guohall', 'skimage', 'cv'], default='auto', help='Algorithm for single-line skeletonization')
    parser.add_argument('--single-line-dilate', type=int, default=1, help='3x3 dilation iterations before skeletonizing (bridges tiny gaps)')
    parser.add_argument('--single-line-bridge', type=int, default=0, help='Connect skeleton endpoints within this many pixels after thinning (0=off)')

    # Thickness normalization (pre-thin)
    parser.add_argument('--normalize-width', choices=['none', 'thickest', 'thinnest', 'average'], default='none', help='Normalize raster line thickness before thinning')
    parser.add_argument('--normalize-target', type=int, default=0, help='Override target half-width (px) for normalization (0=auto)')
    parser.add_argument('--normalize-percentile', type=float, default=95.0, help='Percentile for thickest/thinnest selection (e.g., 95 or 5)')

    # Logging
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (-v, -vv)')

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Logging setup
    lvl = logging.WARNING
    if args.verbose == 1:
        lvl = logging.INFO
    elif args.verbose >= 2:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='[%(levelname)s] %(message)s')

    # Validate some ranges
    if args.dp_eps <= 0:
        logging.info("Adjusting --dp-eps to 0.5 minimum")
        args.dp_eps = 0.5
    if not (0.0 <= args.normalize_percentile <= 100.0):
        raise SystemExit("--normalize-percentile must be within [0,100]")
    if args.normalize_target < 0:
        raise SystemExit("--normalize-target must be >= 0")

    # Warn on redundant/ignored combos
    if args.fill_lines and args.single_line:
        logging.info("--fill-lines requested with --single-line; output will be filled bands, single-line thinning mostly irrelevant for visual result.")

    hsv_low = tuple(args.hsv_low) if args.hsv_low else None
    hsv_high = tuple(args.hsv_high) if args.hsv_high else None

    process(
        in_path=args.input,
        out_dir=args.out,
        hsv_low=hsv_low,
        hsv_high=hsv_high,
        min_perimeter=args.min_perimeter,
        close_k=args.close_k,
        close_iters=args.close_iters,
        dp_eps=args.dp_eps,
        stroke=args.stroke,
        thin=args.thin,
        single_line=args.single_line,
        single_line_method=args.single_line_method,
        single_line_dilate=args.single_line_dilate,
        single_line_bridge=args.single_line_bridge,
        norm_width_mode=args.normalize_width,
        norm_width_target=args.normalize_target,
        norm_width_percentile=args.normalize_percentile,
        fill_lines=args.fill_lines,
    )


if __name__ == '__main__':
    main()
