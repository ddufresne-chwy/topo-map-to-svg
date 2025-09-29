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
    """Group contours by geometric containment using bbox shortlist + point-in-polygon.
    Note: For maps like your sample, many low-elevation rings enclose the whole region,
    so peak separation by nesting can degrade to a chain. Prefer spatial clustering for peaks.
    """
    n = len(contours)
    if n == 0:
        return {}

    bboxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]

    parent = [-1] * n
    for i in range(n):
        # centroid as test point
        M = cv2.moments(contours[i])
        if M['m00'] == 0:
            cx, cy = float(contours[i][0][0][0]), float(contours[i][0][0][1])
        else:
            cx, cy = float(M['m10']/M['m00']), float(M['m01']/M['m00'])
        best = -1
        best_area = float('inf')
        xi, yi, wi, hi = bboxes[i]
        bx0, by0, bx1, by1 = xi, yi, xi + wi, yi + hi
        for j in range(n):
            if i == j:
                continue
            xj, yj, wj, hj = bboxes[j]
            # must be strictly larger and bbox contain
            if areas[j] <= areas[i]:
                continue
            if not (xj <= bx0 and yj <= by0 and xj + wj >= bx1 and yj + hj >= by1):
                continue
            # precise containment by point-in-polygon on centroid
            inside = cv2.pointPolygonTest(contours[j], (cx, cy), False) >= 0
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
        trees[tidx] = {"root": r, "nodes": nodes, "depths": depths, "children": children}
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

def spatial_cluster(centroids: List[Tuple[float, float]], eps: float) -> Tuple[List[int], int]:
    """Simple DBSCAN-like clustering without extra deps (single-link eps)."""
    pts = np.array(centroids, dtype=np.float32)
    n = len(pts)
    if n == 0:
        return [], 0
    # Build adjacency by eps
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (pts[i] - pts[j]) @ (pts[i] - pts[j]) <= eps * eps:
                adj[i].append(j)
                adj[j].append(i)
    # BFS components
    labels = [-1] * n
    cid = 0
    from collections import deque as _dq
    for i in range(n):
        if labels[i] != -1:
            continue
        q = _dq([i])
        labels[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = cid
                    q.append(v)
        cid += 1
    return labels, cid


def compute_layers_cluster(labels: List[int], areas: List[float], target_layers: int) -> Tuple[List[int], List[int]]:
    """Return (global_layer, local_seq_within_peak) per contour index using per‑peak area ranks.
       If target_layers>0, map local ranks to [0..target_layers-1] by linear scaling.
    """
    n = len(labels)
    global_layer = [0] * n
    local_seq = [0] * n
    labels_arr = np.array(labels)
    for peak_id in sorted(set(labels)):
        idxs = np.where(labels_arr == peak_id)[0].tolist()
        # Sort outer→inner by area (desc)
        idxs_sorted = sorted(idxs, key=lambda i: areas[i], reverse=True)
        max_rank = max(0, len(idxs_sorted) - 1)
        for rank, i in enumerate(idxs_sorted):
            local_seq[i] = rank
            if target_layers > 0 and max_rank > 0:
                g = int(round(rank * (target_layers - 1) / max_rank))
            else:
                g = rank
            global_layer[i] = g
    return global_layer, local_seq


def process(
        in_path=args.input,
        out_dir=args.out,
        hsv_low=tuple(args.hsv_low) if args.hsv_low else None,
        hsv_high=tuple(args.hsv_high) if args.hsv_high else None,
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
        remove_border=args.remove_border,
        max_area_frac=args.max_area_frac,
        peak_mode=args.peak_mode,
        cluster_eps=args.cluster_eps,
        target_layers=args.target_layers,
    )


if __name__ == '__main__':
    main()
