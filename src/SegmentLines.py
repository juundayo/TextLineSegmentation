#!/usr/bin/env python3
"""
A Python implementation of the paper
"A Statistical Approach to Line Segmentation in Handwritten Documents"
(SPIE 2007; Arivazhagan, Srinivasan, Srihari)

Usage
-----
python spie2007_line_segmentation.py \
    --input path/to/image.png \
    --out out_dir \
    [--dpi 200] [--debug] \
    --nlines 

Outputs
------------
- out_dir/overlay_lines.png          : original image with the final lines overlaid
- out_dir/chunks/profile_chunk_XX.png: per‑chunk projection profiles with detected peaks/valleys
- out_dir/lines/line_####.png        : extracted line images (foreground on white)
- out_dir/intermediates/*.png        : optional debug visuals (--debug)

Notes
-----
This implementation follows the paper’s logic and math:
- Binarization with Otsu; auto-fallback to adaptive thresholding when warranted
- Piece‑wise (5% width) vertical projection profiles; smoothing with a moving average (win=5)
- Candidate lines from valleys of the first 25%; then connect valleys chunk‑to‑chunk
- Recursive updates of per‑line bivariate Gaussian (mean, covariance) of accumulated line pixels
- Collision handling with obstructing components using Gaussian likelihood ratio; with distance metric
  fallback in first 25% or when evidence is weak / insufficient statistics
- Overlap detection heuristics and cut‑through at local valley; final assignment by vertical bands

Dependencies: numpy, opencv‑python, matplotlib (for profile plots), pillow (only for robust save)
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# ----------------------------------------------------------------------------#

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def clear_directory(path: str):
    """
    Removes all files and subdirectories 
    in a folder, then recreates it.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def imread_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def save_image(path: str, img: np.ndarray):
    # Ensure uint8 0..255
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1:
        return x.copy()
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(xpad, kernel, mode='valid')

def find_peaks_and_valleys(profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple peak/valley detector: peaks are local maxima, valleys local minima.
    Returns indices of peaks, valleys (sorted ascending).
    """
    x = profile
    n = len(x)
    # gradient sign changes
    d = np.sign(np.diff(x))
    # consolidate flats
    for i in range(1, len(d)):
        if d[i] == 0:
            d[i] = d[i-1]
    # turning points where derivative changes sign
    turning = np.where(np.diff(d) != 0)[0] + 1
    # classify by comparing with neighbors
    peaks = []
    valleys = []
    for idx in turning:
        left = x[max(0, idx-1)]
        center = x[idx]
        right = x[min(n-1, idx+1)]
        if center >= left and center >= right:
            peaks.append(idx)
        if center <= left and center <= right:
            valleys.append(idx)
    return np.array(sorted(set(peaks)), dtype=int), np.array(sorted(set(valleys)), dtype=int)

# ----------------------------------------------------------------------------#

def binarize(image: np.ndarray, debug_dir: Optional[str] = None) -> np.ndarray:
    # Otsu threshold.
    blur = cv2.GaussianBlur(image, (5,5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive (mean) threshold as fallback candidate.
    adapt = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Deciding between Otsu and adaptive: chooses
    # the one with higher foreground entropy balance
    def score(bin_img: np.ndarray) -> float:
        # 0 = black, 255 = white; we consider foreground = black
        p = np.mean(bin_img == 0)
        # Encouraging ~30%-70% foreground proportion to avoid all-black/all-white
        balance = 1.0 - abs(p - 0.35)
        # Edge density proxy (more text strokes).
        edges = cv2.Canny(bin_img, 100, 200)
        edge_density = np.mean(edges > 0)
        return 0.7 * balance + 0.3 * edge_density

    chosen = otsu if score(otsu) >= score(adapt) else adapt

    if debug_dir:
        save_image(os.path.join(debug_dir, '01_otsu.png'), otsu)
        save_image(os.path.join(debug_dir, '02_adaptive.png'), adapt)
        save_image(os.path.join(debug_dir, '03_chosen_bin.png'), chosen)

    # Inverting -> foreground=1, background=0 for easier math.
    bin_inv = (chosen == 0).astype(np.uint8)
    return bin_inv

# ----------------------------------------------------------------------------#

@dataclass
class ChunkProfiles:
    chunk_profiles: List[np.ndarray]  # Raw counts per y for each chunk.
    chunk_smoothed: List[np.ndarray]
    chunk_x_ranges: List[Tuple[int, int]]  # (x0,x1)
    first25_smoothed: np.ndarray
    height: int
    width: int

def compute_chunk_profiles(bin_img: np.ndarray, window: int = 5, out_dir: Optional[str] = None) -> ChunkProfiles:
    H, W = bin_img.shape
    N = 20  # 5% width chunks!
    chunk_w = W // N

    chunk_profiles = []
    chunk_smoothed = []
    chunk_x_ranges = []

    for i in range(N):
        x0 = i * chunk_w
        x1 = (i+1) * chunk_w if i < N-1 else W
        chunk = bin_img[:, x0:x1]
        profile = np.sum(chunk, axis=1).astype(np.float32)  # Foreground count per row (y).
        sm = moving_average(profile, window)
        chunk_profiles.append(profile)
        chunk_smoothed.append(sm)
        chunk_x_ranges.append((x0, x1))

    # First 25% (0..25% of width) for initial lines.
    first25_x1 = max(1, int(0.25 * W))
    first25 = bin_img[:, :first25_x1]
    first25_profile = np.sum(first25, axis=1).astype(np.float32)
    first25_smoothed = moving_average(first25_profile, window)

    # Visualization.
    if out_dir:
        cdir = os.path.join(out_dir, 'chunks')
        
        for i, (prof, sm) in enumerate(zip(chunk_profiles, chunk_smoothed)):
            peaks, valleys = find_peaks_and_valleys(sm)
            fig = plt.figure(figsize=(6, 3))
            plt.plot(sm)
            plt.scatter(peaks, sm[peaks], marker='^')
            plt.scatter(valleys, sm[valleys], marker='v')
            plt.title(f'Chunk {i+1} profile (x in [{chunk_x_ranges[i][0]}, {chunk_x_ranges[i][1]})')
            plt.xlabel('y (row)')
            plt.ylabel('foreground count')
            plt.tight_layout()
            fig.savefig(os.path.join(cdir, f'profile_chunk_{i+1:02d}.png'))
            plt.close(fig)

    return ChunkProfiles(
        chunk_profiles, chunk_smoothed, chunk_x_ranges, first25_smoothed, H, W
    )

# ----------------------------------------------------------------------------#

@dataclass
class LinePath:
    # For each x, the y position of the line (int). We store
    # sparse control points per chunk and interpolate while marching.
    y_by_x: Dict[int, int]  # x -> y


def estimate_local_stats(profs: ChunkProfiles):
    """
    Compute dynamic measures from the chunk profiles:
      - mean_profile_range : average (max-min) across chunks
      - median_valley_spacing: median distance between adjacent valleys in first25 profile
      - approx_line_height: rough expected line height (fallback)
    """
    H, W = profs.height, profs.width
    # per-chunk profile ranges
    ranges = [float(np.max(p) - np.min(p)) if p.size else 0.0 for p in profs.chunk_profiles]
    mean_profile_range = float(np.median(ranges)) if ranges else 1.0

    # estimate spacing from first25
    first = profs.first25_smoothed
    p_peaks, p_valleys = find_peaks_and_valleys(first)
    if len(p_valleys) >= 2:
        spacings = np.diff(np.sort(p_valleys)).astype(float)
        median_valley_spacing = float(np.median(spacings)) if spacings.size else max(20.0, profs.height/20.0)
    else:
        # fallback proportional to image height (guess 1/20th)
        median_valley_spacing = max(20.0, H / 20.0)

    # approximate line height (use conservative estimate)
    approx_line_height = max(8.0, 0.6 * median_valley_spacing)

    return {
        "mean_profile_range": mean_profile_range,
        "median_valley_spacing": median_valley_spacing,
        "approx_line_height": approx_line_height
    }


def select_valleys(sm: np.ndarray,
                   min_prom_rel: Optional[float] = None,
                   min_dist: Optional[int] = None,
                   window_prom: Optional[int] = None,
                   abs_prom: Optional[float] = None,
                   dynamic_stats: Optional[dict] = None) -> np.ndarray:
    """
    Adaptive valley selection:
      - min_prom_rel, min_dist, window_prom and abs_prom are computed dynamically if None
      - returns valleys sorted by increasing sm[value] (deepest first)
    """
    peaks, valleys = find_peaks_and_valleys(sm)
    if len(valleys) == 0:
        return np.array([], dtype=int)

    H = len(sm)
    # dynamic stats fallback
    rng = float(np.max(sm) - np.min(sm)) if np.max(sm) > np.min(sm) else 1.0
    if dynamic_stats is None:
        # minimal defaults
        median_spacing = max(20.0, H / 20.0)
    else:
        median_spacing = float(dynamic_stats.get("median_valley_spacing", max(20.0, H / 20.0)))

    # dynamic defaults
    if window_prom is None:
        window_prom = int(max(6, min( max(8, H // 40), median_spacing // 2 )))
    if min_dist is None:
        # require separation proportional to estimated line spacing
        min_dist = int(max(6, round(0.35 * median_spacing)))
    if abs_prom is None:
        abs_prom = max(1.0, 0.06 * rng)  # small absolute floor relative to profile range
    if min_prom_rel is None:
        min_prom_rel = max(0.10, 0.12 * (median_spacing / max(20.0, median_spacing)))  # weaker for small spacing

    candidates = []
    for v in valleys:
        l = max(0, v - window_prom)
        r = min(H, v + window_prom + 1)
        left_max = np.max(sm[l:v]) if v > l else sm[v]
        right_max = np.max(sm[v+1:r]) if v+1 < r else sm[v]
        prom = min(left_max, right_max) - sm[v]
        # dynamic threshold uses both relative to range and absolute floor
        thresh = max(abs_prom, min_prom_rel * rng)
        if prom >= thresh:
            candidates.append((int(v), float(prom)))

    if not candidates:
        return np.array([], dtype=int)

    # rank by a combined score: deeper valleys and (optionally) local contrast (prom)
    # we'll sort ascending by sm[v] (deeper) and descending prom as tie-breaker
    candidates_sorted = sorted(candidates, key=lambda t: (sm[t[0]], -t[1]))

    # enforce min_dist greedily on the sorted candidates but prefer deepest
    selected = []
    for v, p in candidates_sorted:
        if all(abs(v - s) >= min_dist for s in selected):
            selected.append(v)
    selected.sort()
    return np.array(selected, dtype=int)


def initial_candidate_lines(profs: ChunkProfiles,
                            bin_img: Optional[np.ndarray] = None,
                            debug_dir: Optional[str] = None,
                            min_prom_rel: float = 0.20,
                            min_dist_pixels: Optional[int] = None,
                            nlines_expected: Optional[int] = None) -> List[LinePath]:
    """
    Build candidate lines by:
      - selecting strong valleys per chunk,
      - seeding from the deepest valleys in first25 (top-nlines_expected if provided),
      - connecting valleys chunk-to-chunk using connect_valleys,
      - preferring valleys that do not cut through large connected components,
      - pruning low-coverage candidates and merging very-close ones,
      - finally adjusting to nlines_expected (merge or synthesize).
    """
    H, W = profs.height, profs.width
    N = len(profs.chunk_smoothed)
    if N == 0:
        return []
    
    if min_dist_pixels is None:
        min_dist_pixels = max(6, H // 130)

    # Chunk x boundaries.
    chunk_x0s = [r[0] for r in profs.chunk_x_ranges]
    chunk_x1s = [r[1] for r in profs.chunk_x_ranges]
    chunk_widths = [chunk_x1s[i] - chunk_x0s[i] for i in range(N)]
    x_seed = chunk_x1s[0] - 1

    # Precomputing valleys & scores per chunk.
    valleys_per_chunk: List[np.ndarray] = []
    valley_scores_per_chunk: List[np.ndarray] = []
    for i in range(N):
        sm = profs.chunk_smoothed[i]
        v = select_valleys(sm,
                           min_prom_rel=min_prom_rel,
                           min_dist=min_dist_pixels,
                           window_prom=max(8, H // 40),
                           abs_prom=2.0)
        valleys_per_chunk.append(np.array(v, dtype=int))
        valley_scores_per_chunk.append(np.array([sm[ii] for ii in v], dtype=float) if len(v) else np.array([], dtype=float))

    # If bin_img given, compute connected components once for blocking checks.
    labels = None; stats = None
    if bin_img is not None:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8), connectivity=8)

    # Helper: deciding whether a valley at (y) within columns around x_col is "blocked".
    def valley_blocked_by_big_component(y: int, x_col: int, horiz_span: int) -> bool:
        # Scanning a horizontal window around x_col, check components touching row y,
        # if such component has a wide bbox (likely a character/stroke spanning) mark as blocked.
        if labels is None:
            return False
        
        x0 = max(0, x_col - horiz_span)
        x1 = min(W - 1, x_col + horiz_span)

        # Checking several columns.
        cols = list(range(x0, x1 + 1, max(1, (x1 - x0) // 6))) if x1 > x0 else [x_col]
        min_comp_width = max(3, W // 120) 
        for xc in cols:
            if y < 0 or y >= H:
                continue
            cid = int(labels[y, xc])
            if cid <= 0:
                continue

            x_cc, y_cc, w_cc, h_cc, area = stats[cid]
            if w_cc >= min_comp_width and area >= 10:
                # Component is sizable horizontally — avoid cutting through it.
                return True
        return False

    # Helper: greedy connect valleys between two adjacent chunks.
    def connect_valleys(v_prev: np.ndarray, v_next: np.ndarray) -> List[Tuple[int, int]]:
        if len(v_prev) == 0 or len(v_next) == 0:
            return []
        # Cost = vertical distance.
        cost = np.abs(v_prev[:, None].astype(int) - v_next[None, :].astype(int))
        pairs = []
        used_next = set()
        # For each prev valley (in order), pick nearest next valley that is unused.
        for i_idx in range(len(v_prev)):
            order = np.argsort(cost[i_idx])
            chosen = None
            for j in order:
                if int(j) not in used_next:
                    chosen = int(j)
                    break
            if chosen is not None:
                pairs.append((int(i_idx), chosen))
                used_next.add(chosen)
        return pairs

    # Seeding lines from first 25%: picks the deepest separated valleys.
    v0 = valleys_per_chunk[0] if len(valleys_per_chunk) else np.array([], dtype=int)
    seeds: List[int] = []
    if len(v0) > 0:
        # Sorting indices of v0 by valley depth (smaller value -> deeper).
        depths = profs.first25_smoothed[v0]
        order = np.argsort(depths)  # Ascending => deepest first.
        for idx in order:
            y = int(v0[int(idx)])
            if all(abs(y - s) >= min_dist_pixels for s in seeds):
                seeds.append(y)
            if nlines_expected and len(seeds) >= nlines_expected:
                break
    else:
        # Fallback: evenly spaced seeds.
        target = nlines_expected if (nlines_expected and nlines_expected > 0) else max(4, H // 100)
        seeds = list(np.linspace(int(H * 0.12), int(H * 0.88), max(1, target), dtype=int))

    # Instantiating lines with seed at x_seed.
    lines: List[LinePath] = [LinePath(y_by_x={x_seed: int(s)}) for s in seeds]

    # Marching chunks left -> right.
    for i in range(0, N - 1):
        x_curr = chunk_x1s[i] - 1
        x_next = chunk_x0s[i + 1]
        v_prev = valleys_per_chunk[i]
        v_next = valleys_per_chunk[i + 1]
        if len(v_prev) and len(v_next):
            pairs = connect_valleys(v_prev, v_next)
            assigned_next = set()

            # For each paired valley, find the best matching existing line by y at x_curr.
            for idx_prev, idx_next in pairs:
                y_prev = int(v_prev[idx_prev])
                y_next = int(v_next[idx_next])
                best_lp = None; best_d = 1e9

                for lp in lines:
                    # Finding lp's y at x_curr (or nearest left key <= x_curr).
                    if x_curr in lp.y_by_x:
                        y_lp = lp.y_by_x[x_curr]
                    else:
                        keys = sorted(lp.y_by_x.keys())
                        # Choosing the closest key to left if exists, else nearest key.
                        left_keys = [k for k in keys if k <= x_curr]
                        if left_keys:
                            y_lp = lp.y_by_x[left_keys[-1]]
                        else:
                            y_lp = lp.y_by_x[keys[0]]
                    d = abs(int(y_lp) - int(y_prev))
                    if d < best_d:
                        best_d = d
                        best_lp = lp

                if best_lp is not None:
                    # Prefering the valley that is not blocked by any large component in the local horizontal span.
                    horiz_span = max(1, chunk_widths[i+1] // 2)
                    blocked = valley_blocked_by_big_component(y_next, x_next, horiz_span) if bin_img is not None else False
                    if blocked:
                        # Trying to pick an unblocked alternative valley near the chosen 
                        # idx_next. Looks among v_next sorted by distance to y_prev.
                        cand_order = np.argsort(np.abs(v_next - y_prev))
                        chosen_alt = None
                        for jj in cand_order:
                            if int(jj) in assigned_next:
                                continue
                            y_alt = int(v_next[int(jj)])
                            if not valley_blocked_by_big_component(y_alt, x_next, horiz_span):
                                chosen_alt = y_alt
                                assigned_idx = int(jj)
                                break
                        if chosen_alt is not None:
                            best_lp.y_by_x[x_next] = int(chosen_alt)
                            assigned_next.add(assigned_idx)
                        else:
                            # Fall back to the original valley (maybe unavoidable).
                            best_lp.y_by_x[x_next] = int(y_next)
                            assigned_next.add(int(idx_next))
                    else:
                        best_lp.y_by_x[x_next] = int(y_next)
                        assigned_next.add(int(idx_next))

            # Spawns lines for unassigned valleys in next chunk.
            existing_assigned = {lp.y_by_x.get(x_next) for lp in lines if x_next in lp.y_by_x}
            for j, yn in enumerate(v_next):
                if int(yn) not in existing_assigned:
                    # Only spawn if valley is strong enough (non-trivial)!
                    lines.append(LinePath(y_by_x={x_next: int(yn)}))
        else:
            # If either side has no valleys, simply propagate previous y forward for each line.
            for lp in lines:
                # y at end of chunk i.
                if x_curr in lp.y_by_x:
                    y_curr = lp.y_by_x[x_curr]
                else:
                    keys = sorted(lp.y_by_x.keys())
                    left_keys = [k for k in keys if k <= x_curr]
                    y_curr = lp.y_by_x[left_keys[-1]] if left_keys else lp.y_by_x[keys[0]]
                lp.y_by_x[x_next] = int(y_curr)

        # For robustness: if any line still lacks x_next entry, 
        # assign nearest valley within a window or carry forward
        for lp in lines:
            if x_next not in lp.y_by_x:
                # Finding the nearest valley within a vertical window centered at the last known y.
                keys = sorted(lp.y_by_x.keys())
                left_keys = [k for k in keys if k <= x_curr]
                y_last = lp.y_by_x[left_keys[-1]] if left_keys else lp.y_by_x[keys[0]]
                window = max(min_dist_pixels * 2, H // 12)
                candidates = [int(v) for v in valleys_per_chunk[i+1] if abs(int(v) - y_last) <= window] if len(valleys_per_chunk[i+1]) else []

                if candidates:
                    # Picks the candidate whose valley score is the
                    # deepest (lowest) and not blocked if possible.
                    best = None; best_score = 1e9
                    for c in candidates:
                        score = profs.chunk_smoothed[i+1][c]
                        blocked = valley_blocked_by_big_component(c, x_next, max(1, chunk_widths[i+1] // 2)) if bin_img is not None else False
                        if blocked:
                            score += 1000.0
                        if score < best_score:
                            best_score = score
                            best = c
                    lp.y_by_x[x_next] = int(best)
                else:
                    # Carries forward the previous y.
                    lp.y_by_x[x_next] = int(y_last)

    # Post-filter: require that a candidate covers a minimum fraction of chunks / columns.
    if bin_img is not None and lines:
        filtered = []
        min_chunk_coverage = max(3, int(0.10 * N))  # Must have y assignments in at least 10% of chunks
        for lp in lines:
            assigned_chunks = sum(1 for xk in lp.y_by_x.keys() if xk in chunk_x0s or xk in chunk_x1s)
            if assigned_chunks >= min_chunk_coverage:
                filtered.append(lp)
        lines = filtered if filtered else lines  # Keeping at least something.

    # Merging very close final candidates (median y).
    if len(lines) > 1:
        # Computing medians and sorting.
        medians = [int(np.median(list(lp.y_by_x.values()))) for lp in lines]
        idx_sorted = np.argsort(medians)
        sorted_lines = [lines[i] for i in idx_sorted]
        min_sep = max(6, H // 220)
        merged_any = True

        while merged_any:
            merged_any = False
            new_list = []
            i = 0
            while i < len(sorted_lines):
                if i < len(sorted_lines) - 1:
                    y_i = int(np.median(list(sorted_lines[i].y_by_x.values())))
                    y_j = int(np.median(list(sorted_lines[i + 1].y_by_x.values())))
                    if abs(y_i - y_j) < min_sep:
                        # Merging!
                        lp1 = sorted_lines[i]; lp2 = sorted_lines[i + 1]
                        new_yby = {}
                        keys = set(lp1.y_by_x.keys()) | set(lp2.y_by_x.keys())
                        for k in keys:
                            v1 = lp1.y_by_x.get(k)
                            v2 = lp2.y_by_x.get(k)
                            if v1 is None:
                                new_yby[k] = v2
                            elif v2 is None:
                                new_yby[k] = v1
                            else:
                                new_yby[k] = int(round((v1 + v2) / 2.0))
                        new_list.append(LinePath(y_by_x=new_yby))
                        i += 2
                        merged_any = True
                        continue
                new_list.append(sorted_lines[i])
                i += 1
            sorted_lines = new_list
        lines = sorted_lines

    # Final adjustment to expected number of lines (merge or synthesize).
    if nlines_expected is not None and nlines_expected > 0:
        # Sorting by median y.
        lines = sorted(lines, key=lambda lp: int(np.median(list(lp.y_by_x.values()))))
        if len(lines) > nlines_expected:
            while len(lines) > nlines_expected:
                # Merging the closest pair.
                gaps = [abs(int(np.median(list(lines[j].y_by_x.values()))) - int(np.median(list(lines[j+1].y_by_x.values())))) for j in range(len(lines) - 1)]
                jmin = int(np.argmin(gaps))
                lp1, lp2 = lines[jmin], lines[jmin + 1]
                new_yby = {}
                keys = set(lp1.y_by_x.keys()) | set(lp2.y_by_x.keys())
                for k in keys:
                    v1 = lp1.y_by_x.get(k)
                    v2 = lp2.y_by_x.get(k)
                    if v1 is None: new_yby[k] = v2
                    elif v2 is None: new_yby[k] = v1
                    else: new_yby[k] = int(round((v1 + v2) / 2.0))
                lines = lines[:jmin] + [LinePath(y_by_x=new_yby)] + lines[jmin+2:]
        elif len(lines) < nlines_expected:
            # Synthesizes evenly spaced lines across the vertical extent of existing lines.
            if lines:
                existing = [int(np.median(list(lp.y_by_x.values()))) for lp in lines]
                ymin, ymax = min(existing), max(existing)
            else:
                ymin, ymax = int(H*0.12), int(H*0.88)
            targets = np.linspace(ymin, ymax, nlines_expected, dtype=int)
            new_lines = []
            for t in targets:
                # If there's already a line close then skip.
                if any(abs(t - int(np.median(list(lp.y_by_x.values())))) < max(6, H//220) for lp in lines):
                    continue
                new_lines.append(LinePath(y_by_x={chunk_x0s[0]: int(t), chunk_x1s[-1]-1: int(t)}))
            lines.extend(new_lines)
            lines = sorted(lines, key=lambda lp: int(np.median(list(lp.y_by_x.values()))))

    # Visualization of candidate control points.
    if debug_dir:
        try:
            vis = np.stack([np.ones((H, W), dtype=np.uint8) * 255] * 3, axis=2)
            colors = [(0, 0, 255), (0, 128, 0), (255, 0, 0), (0, 128, 128), (128, 0, 128)]
            for idx, lp in enumerate(lines):
                color = colors[idx % len(colors)]
                for xi, yi in lp.y_by_x.items():
                    if 0 <= yi < H and 0 <= xi < W:
                        cv2.circle(vis, (int(xi), int(yi)), 1 + (idx % 3), color, -1)
            save_image(os.path.join(debug_dir, 'candidate_lines.png'), vis)
        except Exception:
            pass

    return lines

# ----------------------------------------------------------------------------#

class RunningGaussian2D:
    def __init__(self):
        self.N = 0
        self.mu = np.zeros(2, dtype=np.float64)
        self.S = np.zeros((2, 2), dtype=np.float64)  # Sum of outer products for covariance.

    def update(self, p: np.ndarray):
        # p: shape (2,), [x,y]
        self.N += 1
        if self.N == 1:
            self.mu = p.astype(np.float64)
            self.S[:] = 0
            return
        # Recursive mean.
        mu_new = ( (self.N - 1) / self.N ) * self.mu + (1.0 / self.N) * p
        # Covariance update using Welford-like; paper gives a slightly different recursive form.
        # We'll compute unbiased sample covariance at query time.
        delta = p - self.mu
        delta2 = p - mu_new
        self.S += np.outer(delta, delta2)
        self.mu = mu_new

    def covariance(self) -> np.ndarray:
        if self.N <= 1:
            return np.eye(2) * 1e-3
        return self.S / (self.N - 1)

    def logpdf(self, p: np.ndarray) -> float:
        mu = self.mu
        Sigma = self.covariance()
        
        # Check for valid input
        if np.any(np.isnan(mu)) or np.any(np.isnan(Sigma)):
            return -1e6
        
        # Ensure Sigma is positive definite with stronger regularization
        Sigma = Sigma + np.eye(2) * 1e-6  
        
        try:
            L = np.linalg.cholesky(Sigma)
            inv_L = np.linalg.inv(L)
            diff = p - mu
            z = inv_L @ diff
            mahalanobis = np.sum(z**2)
            mahalanobis = np.clip(mahalanobis, 0, 50)  # tighter clamp

            log_det = 2 * np.sum(np.log(np.diag(L) + 1e-12))
            logp = -0.5 * (2 * np.log(2 * np.pi) + log_det + mahalanobis)

            # Bound logp to avoid extreme influence
            return float(logp if logp > -1e6 else -1e6)

        except np.linalg.LinAlgError:
            var_x = max(Sigma[0, 0], 1e-6)
            var_y = max(Sigma[1, 1], 1e-6)
            diff = p - mu
            mahalanobis = (diff[0]**2 / var_x) + (diff[1]**2 / var_y)
            mahalanobis = np.clip(mahalanobis, 0, 50)
            log_det = np.log(var_x * var_y + 1e-12)
            logp = -0.5 * (2 * np.log(2 * np.pi) + log_det + mahalanobis)
            return float(logp if logp > -1e6 else -1e6)

# ----------------------------------------------------------------------------#

@dataclass
class SegmentationResult:
    line_paths: List[LinePath]
    ymap: np.ndarray  # shape (W, L): y position for each line per x
    overlay: np.ndarray  # RGB overlay image
    assignments: np.ndarray  # label image (0..L-1 for foreground; 255 background)

def draw_lines_with_collisions(image_gray: np.ndarray, bin_img: np.ndarray, lines: List[LinePath],
                               profs: ChunkProfiles, debug_dir: Optional[str] = None) -> SegmentationResult:
    H, W = bin_img.shape
    L = len(lines)
    if L == 0:
        raise RuntimeError("No candidate lines detected; check input or thresholds.")

    # Initializes ymap with interpolated candidate lines.
    xs_all = np.arange(W)
    ymap = np.zeros((W, L), dtype=np.int32)
    
    for li, lp in enumerate(lines):
        # Getting all x positions with known y values.
        known_xs = np.array(sorted(lp.y_by_x.keys()))
        known_ys = np.array([lp.y_by_x[x] for x in known_xs])
        
        # Interpolate to get y values for all x positions
        if len(known_xs) > 1:
            ymap[:, li] = np.interp(xs_all, known_xs, known_ys).astype(int)
        else:
            ymap[:, li] = known_ys[0] if len(known_ys) > 0 else H // 2
    
    # Gets connected components for obstruction detection.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=8)
    
    # Create a mapping from component ID to its bounding box and pixels
    comp_info = {}
    for comp_id in range(1, num_labels):
        x, y, w, h, area = stats[comp_id]
        comp_mask = (labels == comp_id)
        comp_pixels = np.column_stack(np.where(comp_mask))
        comp_info[comp_id] = {
            'bbox': (x, y, w, h),
            'pixels': comp_pixels,
            'min_y': np.min(comp_pixels[:, 0]),
            'max_y': np.max(comp_pixels[:, 0])
        }

    # Initializing Gaussian models for each line.
    gaussians = [RunningGaussian2D() for _ in range(L)]
    
    # Update Gaussians with initial line positions
    for x in range(W):
        for li in range(L):
            y_val = ymap[x, li]
            if 0 <= y_val < H:
                gaussians[li].update(np.array([x, y_val], dtype=np.float64))

    # Main line adjustment loop
    for x in range(W):
        for li in range(L):
            y_current = ymap[x, li]
            
            # Check if we hit a component at this position
            if 0 <= y_current < H and bin_img[y_current, x] > 0:
                comp_id = labels[y_current, x]
                if comp_id == 0:  # Background
                    continue
                    
                comp_data = comp_info[comp_id]
                x_comp, y_comp, w_comp, h_comp = comp_data['bbox']
                comp_min_y, comp_max_y = comp_data['min_y'], comp_data['max_y']
                
                # Determine if this is an overlapping component (spans multiple lines)
                is_overlapping = False
                if li > 0 and li < L-1:
                    # Check if component spans across adjacent lines
                    y_above = ymap[x, li-1]
                    y_below = ymap[x, li+1]
                    is_overlapping = (comp_min_y < (y_above + y_current)/2 and 
                                     comp_max_y > (y_current + y_below)/2)
                
                if is_overlapping:
                    # Cut through at the valley position
                    chunk_idx = min(x // (W // len(profs.chunk_smoothed)), len(profs.chunk_smoothed)-1)
                    sm = profs.chunk_smoothed[chunk_idx]
                    valleys = select_valleys(sm)
                    
                    if len(valleys) > 0:
                        # Find the deepest valley within the component's vertical range
                        valley_in_range = [v for v in valleys if comp_min_y <= v <= comp_max_y]
                        if valley_in_range:
                            cut_y = min(valley_in_range, key=lambda v: sm[v])
                            # Adjust the line to cut through at the valley
                            for xx in range(max(0, x_comp), min(W, x_comp + w_comp)):
                                ymap[xx, li] = cut_y
                                gaussians[li].update(np.array([xx, cut_y], dtype=np.float64))
                else:
                    # Regular obstructing component - decide whether to go above or below
                    use_distance = (x < int(0.25 * W)) or (gaussians[li].N < 10)
                    decision = None
                    
                    if not use_distance and li > 0 and li < L-1:
                        # Use Gaussian decision
                        logp_above_total = 0.0
                        logp_below_total = 0.0
                        pixels = comp_data['pixels']
                        
                        # Sample points for efficiency
                        sample_size = min(100, len(pixels))
                        indices = np.random.choice(len(pixels), sample_size, replace=False)
                        
                        for idx in indices:
                            p = pixels[idx]
                            p_xy = np.array([p[1], p[0]])  # (x, y)
                            logp_above_total += gaussians[li-1].logpdf(p_xy)
                            logp_below_total += gaussians[li+1].logpdf(p_xy)

                        # Calculate average log probabilities
                        avg_logp_above = logp_above_total / sample_size
                        avg_logp_below = logp_below_total / sample_size

                        # Use log probability difference for decision
                        diff = avg_logp_above - avg_logp_below
                        scale = max(abs(avg_logp_above), abs(avg_logp_below), 1e-6)
                        norm_diff = diff / scale
                        if norm_diff > 0.02:
                            decision = 'above'
                        elif norm_diff < -0.02:
                            decision = 'below'
                        else:
                            use_distance = True

                    
                    if use_distance or decision is None:
                        # Uses distance metric.
                        yh = y_current
                        yu = comp_min_y
                        yd = comp_max_y
                        
                        pu = abs(yh - yu) / (abs(yh - yu) + abs(yh - yd))
                        pd = abs(yh - yd) / (abs(yh - yu) + abs(yh - yd))
                        
                        decision = 'above' if pu < pd else 'below'
                    
                    # Adjust the line to traverse around the component
                    if decision == 'above':
                        # Go below the component
                        new_y = comp_max_y + 1
                        for xx in range(max(0, x_comp), min(W, x_comp + w_comp)):
                            if new_y < H:
                                ymap[xx, li] = new_y
                                gaussians[li].update(np.array([xx, new_y], dtype=np.float64))
                    else:
                        # Go above the component
                        new_y = comp_min_y - 1
                        for xx in range(max(0, x_comp), min(W, x_comp + w_comp)):
                            if new_y >= 0:
                                ymap[xx, li] = new_y
                                gaussians[li].update(np.array([xx, new_y], dtype=np.float64))

    # Build overlay image
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for x in range(0, W, 2):  # Draw every other column for visibility
        for li in range(L):
            y = int(np.clip(ymap[x, li], 0, H-1))
            cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    # Final assignment using vertical bands
    assignments = np.full((H, W), 255, dtype=np.uint8)
    boundary_mid_y = np.zeros((W, L-1), dtype=np.int32)
    
    for x in range(W):
        # Compute mid boundaries between adjacent lines
        for li in range(L-1):
            boundary_mid_y[x, li] = (ymap[x, li] + ymap[x, li+1]) // 2
        
        # Assign foreground pixels to lines based on vertical position
        for y in range(H):
            if bin_img[y, x] > 0:
                li = 0
                for bound_idx in range(L-1):
                    if y > boundary_mid_y[x, bound_idx]:
                        li = bound_idx + 1
                assignments[y, x] = li

    return SegmentationResult(lines, ymap, overlay, assignments)

# ------------------------------
# Save per‑line images and final visuals
# ------------------------------

def save_outputs(res: SegmentationResult, bin_img: np.ndarray, out_dir: str):
    H, W = bin_img.shape
    L = res.ymap.shape[1]

    save_image(os.path.join(out_dir, 'overlay_lines.png'), res.overlay)

    # Save label visualization
    vis = np.full((H, W, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(123)
    palette = rng.integers(0, 255, size=(L, 3), dtype=np.uint8)
    ys, xs = np.where(res.assignments != 255)
    for y, x in zip(ys, xs):
        vis[y, x] = palette[res.assignments[y, x]]
    save_image(os.path.join(out_dir, 'assignment_map.png'), vis)

    # Extract and save each line as a tight crop
    ldir = os.path.join(out_dir, 'lines')
    for li in range(L):
        mask = (res.assignments == li).astype(np.uint8) * 255
        # tight bounding box
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()+1
        y0, y1 = ys.min(), ys.max()+1
        crop = mask[y0:y1, x0:x1]
        # Put on white background
        save_image(os.path.join(ldir, f'line_{li+1:04d}.png'), 255 - crop)

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to input image (grayscale or color)')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--debug', action='store_true', help='Save intermediate debug images')
    ap.add_argument('--nlines', type=int, default=None,
                help='Expected number of text lines (notebook blocks). '
                     'If given, algorithm prunes/merges candidates to match this count')
    args = ap.parse_args()

    clear_directory(args.out)

    ensure_dir(args.out)  # top level
    ensure_dir(os.path.join(args.out, "chunks"))
    ensure_dir(os.path.join(args.out, "lines"))
    if args.debug:
        ensure_dir(os.path.join(args.out, "intermediates"))

    img_gray = imread_grayscale(args.input)
    bin_img = binarize(img_gray, os.path.join(args.out, 'intermediates') if args.debug else None)
    profs = compute_chunk_profiles(bin_img, window=5, out_dir=args.out)
    lines = initial_candidate_lines(profs, bin_img,
                                    os.path.join(args.out, 'intermediates') if args.debug else None,
                                    nlines_expected=args.nlines)
    res = draw_lines_with_collisions(img_gray, bin_img, lines, profs,
                                     os.path.join(args.out, 'intermediates') if args.debug else None)
    save_outputs(res, bin_img, args.out)
    print(f"Saved results into: {args.out}")

if __name__ == '__main__':
    main()
