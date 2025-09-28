#!/usr/bin/env python3
"""
Topographical map ➜ SVG layer extractor

Given a raster image (photo/scan) of a topographic map (JPG/PNG/TIF), this script:
  1) isolates contour (elevation) lines,
  2) traces them into vector paths,
  3) reconstructs the nesting (hierarchy) of the lines to approximate elevation steps,
  4) outputs clean SVG files grouping lines by peak (disconnected region) and by level (nesting depth),
  5) saves **each closed-loop contour** as its **own SVG file**.

New Features
-----------
• `--thin`: draw all stroked SVG contours at a fixed **1px** width, regardless of `--stroke`.
• `--single-line`: **skeletonize** the raster lines to 1‑pixel centerlines before tracing. This yields a **single path per contour** (no inner/outer pair). Use together with `--thin` for clean hairline loops.

Output structure (example):
  out/
    overview_all_layers.svg            # all traced lines in one SVG
    contours/                          # one SVG per discovered loop
      contour_00000.svg
      contour_00001.svg
      ...
    layers/
      level_00.svg                     # every line at depth 0 across all peaks
      level_01.svg                     # every line at depth 1 across all peaks
      ...
    peaks/
      peak_000/
        all_levels.svg                 # this peak, all layers
        level_00.svg                   # only depth 0 for this peak
        level_01.svg                   # only depth 1 for this peak
        ...

Dependencies
------------
  pip install opencv-python numpy svgwrite

Usage
-----
  python topo_to_svg.py input.jpg --out out \
      --min-perimeter 120 --close-k 3 --close-iters 1 \
      --dp-eps 1.5 --stroke 1.0 --thin

"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import svgwrite

Point = Tuple[float, float]

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def approx_poly(points: np.ndarray, eps: float) -> np.ndarray:
    """Douglas–Peucker polyline simplification on a contour (Nx1x2)."""
    peri = cv2.arcLength(points, True)
    epsilon = (eps / 100.0) * peri if eps > 1 else eps
    approx = cv2.approxPolyDP(points, epsilon, True)
    return approx


def contour_closed(_: np.ndarray) -> bool:
    return True


def to_svg(paths: List[List[Point]], size: Tuple[int, int], stroke: float, fn: Path, thin: bool):
    w, h = size
    dwg = svgwrite.Drawing(str(fn), size=(w, h))
    stroke_width = 1.0 if thin else stroke
    for pts in paths:
        if len(pts) < 2:
            continue
        d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        for x, y in pts[1:]:
            d.append(f"L {x:.2f} {y:.2f}")
        d.append("Z")
        dwg.add(dwg.path(" ".join(d), fill="none", stroke="black", stroke_width=stroke_width))
    dwg.save()


def to_svg_single(path_pts: List[Point], size: Tuple[int, int], stroke: float, fn: Path, thin: bool):
    to_svg([path_pts], size, stroke, fn, thin)

# Skeletonization helpers
# Prefer robust thinning (ximgproc Zhang-Suen or Guo-Hall). Fallbacks to scikit-image or
# a morphological skeleton if nothing else is available.

def skeletonize_ximgproc(bin_img: np.ndarray, method: str = "zhang") -> np.ndarray:
    try:
        import cv2.ximgproc as xip
    except Exception:
        return None  # handled by caller
    t = xip.THINNING_ZHANGSUEN if method == "zhang" else xip.THINNING_GUOHALL
    skel = xip.thinning(bin_img, thinningType=t)
    return skel


def skeletonize_skimage(bin_img: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return None
    skel = skeletonize((bin_img > 0)).astype(np.uint8) * 255
    return skel


def skeletonize_cv(bin_img: np.ndarray) -> np.ndarray:
    # Morphological skeleton (fallback). May produce gaps; kept for compatibility.
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
    # Pre-bridge tiny gaps before thinning
    if dilate_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_img = cv2.dilate(bin_img, k, iterations=dilate_iters)

    # Prefer ximgproc if available
    if method in ("auto", "ximgproc", "zhang", "guohall"):
        m = "zhang" if method in ("auto", "ximgproc", "zhang") else "guohall"
        skel = skeletonize_ximgproc(bin_img, m)
        if skel is not None:
            return skel

    # Try scikit-image
    if method in ("auto", "skimage"):
        skel = skeletonize_skimage(bin_img)
        if skel is not None:
            return skel

    # Fallback to morphological skeleton
    return skeletonize_cv(bin_img)

# -----------------------------
# Core processing
# -----------------------------

def isolate_lines(img_bgr: np.ndarray,
                  hsv_low: Tuple[int, int, int] | None,
                  hsv_high: Tuple[int, int, int] | None,
                  close_k: int,
                  close_iters: int) -> np.ndarray:
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

    return bin_img


def find_contour_forest(bin_img: np.ndarray,
                        min_perimeter: float,
                        dp_eps: float) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], np.zeros((0, 4), dtype=np.int32), []

    simp: List[np.ndarray] = []
    keep_idx: List[int] = []
    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        if peri < min_perimeter:
            continue
        if not contour_closed(cnt):
            continue
        if dp_eps > 0:
            cnt = approx_poly(cnt, dp_eps)
        simp.append(cnt)
        keep_idx.append(i)

    if not keep_idx:
        return [], np.zeros((0, 4), dtype=np.int32), []

    hierarchy = hierarchy[0][keep_idx]
    return simp, hierarchy, keep_idx


def build_trees(contours: List[np.ndarray], hierarchy: np.ndarray) -> Dict[int, Dict]:
    n = len(contours)
    bboxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]
    recomputed_parent = [-1] * n
    for i in range(n):
        x, y, w, h = bboxes[i]
        bx0, by0, bx1, by1 = x, y, x + w, y + h
        best_parent = -1
        best_area = float('inf')
        for j in range(n):
            if i == j:
                continue
            x2, y2, w2, h2 = bboxes[j]
            if x2 <= bx0 and y2 <= by0 and x2 + w2 >= bx1 and y2 + h2 >= by1:
                a = areas[j]
                if a < best_area:
                    best_area = a
                    best_parent = j
        recomputed_parent[i] = best_parent

    children = [[] for _ in range(n)]
    roots = []
    for i, p in enumerate(recomputed_parent):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)

    trees: Dict[int, Dict] = {}
    for t_id, r in enumerate(roots):
        q = deque([(r, 0)])
        nodes = []
        depths = defaultdict(list)
        while q:
            idx, d = q.popleft()
            nodes.append(idx)
            depths[d].append(idx)
            for ch in children[idx]:
                q.append((ch, d + 1))
        trees[t_id] = {
            'root': r,
            'nodes': nodes,
            'depths': depths,
            'children': children,
        }
    return trees


def contour_to_points(cnt: np.ndarray) -> List[Point]:
    pts = cnt.reshape(-1, 2)
    return [(float(x), float(y)) for (x, y) in pts]

# -----------------------------
# Orchestration
# -----------------------------

def process(in_path: Path,
            out_dir: Path,
            hsv_low: Tuple[int, int, int] | None,
            hsv_high: Tuple[int, int, int] | None,
            min_perimeter: float,
            close_k: int,
            close_iters: int,
            dp_eps: float,
            stroke: float,
            thin: bool,
            single_line: bool,
            single_line_method: str,
            single_line_dilate: int):

    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {in_path}")

    h, w = img.shape[:2]
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'layers')
    ensure_dir(out_dir / 'peaks')
    ensure_dir(out_dir / 'contours')

    bin_img = isolate_lines(img, hsv_low, hsv_high, close_k, close_iters)

    # Optional: thin to 1-pixel centerlines so each contour traces as a single path
    if single_line:
        bin_img = skeletonize_image(bin_img, method=single_line_method, dilate_iters=single_line_dilate)

    contours, hierarchy, kept = find_contour_forest(bin_img, min_perimeter=min_perimeter, dp_eps=dp_eps)
    if len(contours) == 0:
        print("No contours found above min_perimeter. Try lowering --min-perimeter or adjusting the mask.")
        return

    trees = build_trees(contours, hierarchy)

    for i, cnt in enumerate(contours):
        pts = contour_to_points(cnt)
        to_svg_single(pts, (w, h), stroke, out_dir / 'contours' / f'contour_{i:05d}.svg', thin)

    global_levels: Dict[int, List[List[Point]]] = defaultdict(list)

    for t_id, tree in trees.items():
        peak_dir = out_dir / 'peaks' / f'peak_{t_id:03d}'
        ensure_dir(peak_dir)
        level_paths: Dict[int, List[List[Point]]] = defaultdict(list)

        for depth, idxs in tree['depths'].items():
            for idx in idxs:
                pts = contour_to_points(contours[idx])
                level_paths[depth].append(pts)
                global_levels[depth].append(pts)

        for depth, paths in sorted(level_paths.items()):
            to_svg(paths, (w, h), stroke, peak_dir / f'level_{depth:02d}.svg', thin)

        all_paths = [p for ps in level_paths.values() for p in ps]
        to_svg(all_paths, (w, h), stroke, peak_dir / 'all_levels.svg', thin)

    for depth, paths in sorted(global_levels.items()):
        to_svg(paths, (w, h), stroke, out_dir / 'layers' / f'level_{depth:02d}.svg', thin)

    overview = [p for ps in global_levels.values() for p in ps]
    to_svg(overview, (w, h), stroke, out_dir / 'overview_all_layers.svg', thin)

    print(f"✅ Done. Contours saved: {len(contours)} | Peaks: {len(trees)} | Levels found: {len(global_levels)}")
    print(f"Output directory: {out_dir}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Convert a topo map image into per-elevation-layer SVGs.")
    ap.add_argument('input', type=Path, help='Input image (png/jpg/tif)')
    ap.add_argument('--out', type=Path, default=Path('out'), help='Output directory')

    ap.add_argument('--hsv-low', type=int, nargs=3, metavar=('H', 'S', 'V'), help='Lower HSV bound for contour color mask')
    ap.add_argument('--hsv-high', type=int, nargs=3, metavar=('H', 'S', 'V'), help='Upper HSV bound for contour color mask')

    ap.add_argument('--close-k', type=int, default=3, help='Structuring element size for closing (pixels)')
    ap.add_argument('--close-iters', type=int, default=1, help='Iterations for morphological closing')

    ap.add_argument('--min-perimeter', type=float, default=100.0, help='Ignore tiny contours below this perimeter (pixels)')
    ap.add_argument('--dp-eps', type=float, default=1.5, help='Douglas–Peucker epsilon. If >1, treated as % of perimeter; if <=1, absolute pixels.')

    ap.add_argument('--stroke', type=float, default=1.0, help='SVG stroke width (px)')
    ap.add_argument('--thin', action='store_true', help='Force all SVG contours to be exactly 1px wide')
    ap.add_argument('--single-line', action='store_true', help='Skeletonize raster lines to 1‑pixel centerlines (one path per contour)')
    ap.add_argument('--single-line-method', choices=['auto','ximgproc','zhang','guohall','skimage','cv'], default='auto', help='Algorithm for single-line skeletonization')
    ap.add_argument('--single-line-dilate', type=int, default=1, help='3x3 dilation iterations before skeletonizing (bridges tiny gaps)')')

    return ap.parse_args()

    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()

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
    )
