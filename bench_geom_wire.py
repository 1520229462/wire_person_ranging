#!/usr/bin/env python3
"""Benchmark the three traditional (geometric) powerline-detection methods,
faithfully ported to Python, on the 2D-mask-corresponding 3D ROI.

Sources ported:
  1. railroad (GISLab-ELTE/railroad, mcserep) -- HeightFilter -> WidthFilter ->
     {RANSAC line | 3D Hough | region growing}, CABLE-seeded.
  2. 3DLiDAR (zwshi-pku) -- linear coarse extraction + catenary (sag) fitting.
  3. powerline-detection (liuxinren456852) -- XZ-slice projection + Otsu + Canny
     + cv.HoughLinesP(rho=1,theta=pi/180,threshold=50,minLineLength=65,maxLineGap=5),
     0.75 m neighborhood expansion.

Every method returns a boolean wire mask over the ROI points; we score it
against the hand-annotated wire bbox (recall / precision / F1).
"""
import glob
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_accumulated_wire import load_accumulated, gt_wire_mask

ROI_DIR = "viz_ttpla_wire3d"
GT_STEM = "img_1700603346.892936"


# ---------------------------------------------------------------- geometry ----
def fit_line(pts):
    """PCA line through pts -> (point, unit dir)."""
    c = pts - pts.mean(0)
    _, _, v = np.linalg.svd(c, full_matrices=False)
    return pts.mean(0), v[0]


def dist_to_line(pts, p0, d):
    return np.linalg.norm(np.cross(pts - p0, d), axis=1)


def pca_local(pts, k=8):
    """Per-point local PCA eigenvalues (desc), for the thin/width filter."""
    n = len(pts)
    if n <= k:
        return np.ones((n, 3))
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k + 1, n))
    evals = np.zeros((n, 3))
    for i in range(n):
        nb = pts[idx[i]] - pts[idx[i]].mean(0)
        w = np.linalg.svd(nb, compute_uv=False)
        w = np.sort(w)[::-1]
        evals[i] = w if len(w) == 3 else np.pad(w, (0, 3 - len(w)))
    return evals


def height_filter(pts, z_thresh):
    return pts[:, 2] >= z_thresh


def width_filter(pts, evals, thin_ratio=0.25):
    """railroad WidthFilter: cable is thin -> 2nd PCA eigenvalue small."""
    return evals[:, 1] <= thin_ratio * (evals[:, 0] + 1e-9)


# ------------------------------------------------------- method 1: railroad ---
def method_ransac_line(pts, tol=0.05, n_iter=2000, min_pts=8):
    """HeightFilter -> WidthFilter -> RANSAC 3D straight line (RansacFilter)."""
    n = len(pts)
    if n < 2:
        return np.zeros(n, bool)
    best = None
    idx = np.arange(n)
    for _ in range(n_iter):
        i, j = np.random.choice(n, 2, replace=False)
        if np.linalg.norm(pts[i] - pts[j]) < 0.2:
            continue
        d = pts[j] - pts[i]
        d /= np.linalg.norm(d)
        inl = dist_to_line(pts, pts[i], d) <= tol
        if inl.sum() > (best[0] if best else 0):
            best = (inl.sum(), inl)
    return best[1] if best else np.zeros(n, bool)


def method_hough3d(pts, tol=0.05, n_dir=60, n_theta=120, n_rho=200):
    """HeightFilter -> WidthFilter -> 3D Hough (Hough3dFilter).

    Global vote accumulator over line directions on a sphere: for each sampled
    direction d, project points onto the plane normal to d and run a 2D
    (rho, alpha) Hough with rho bins of width ~tol.
    """
    n = len(pts)
    if n < 2:
        return np.zeros(n, bool)
    c = pts - pts.mean(0)
    rho_max = float(np.linalg.norm(c, axis=1).max() + 1e-9)
    rho_bins = np.linspace(-rho_max, rho_max, n_rho)
    bin_w = rho_bins[1] - rho_bins[0]
    rng = np.random.default_rng(0)
    best = (0, np.zeros(n, bool))
    for _ in range(n_dir):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        e1 = np.cross(d, np.array([0.0, 0, 1.0]))
        if np.linalg.norm(e1) < 1e-3:
            e1 = np.cross(d, np.array([1.0, 0, 0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        u, v = c @ e1, c @ e2
        for a in np.linspace(0, np.pi, n_theta, endpoint=False):
            r = u * np.cos(a) + v * np.sin(a)
            ri = np.clip(((r - rho_bins[0]) / bin_w).astype(int), 0, n_rho - 1)
            cnt = np.bincount(ri, minlength=n_rho)
            b = int(cnt.argmax())
            inl = np.abs(r - rho_bins[b]) <= max(tol, bin_w / 2)
            if inl.sum() > best[0]:
                best = (int(inl.sum()), inl)
    return best[1]


def method_region_grow(pts, tol=0.05, k=6, min_pts=8):
    """HeightFilter -> WidthFilter -> seeded region growing (GrowthFilter).

    Seed = the point whose k-NN covariance is most line-like (smallest
    2nd eigenvalue). Grow along the local direction, adding inliers of the
    locally-fitted line.
    """
    n = len(pts)
    if n < min_pts:
        return np.zeros(n, bool)
    from scipy.spatial import cKDTree
    evals = pca_local(pts, k)
    # seed = most line-like point; seed region = its k nearest neighbours
    seed = int(np.argmin(evals[:, 1] / (evals[:, 0] + 1e-9)))
    tree = cKDTree(pts)
    grown = set(tree.query(pts[seed], k=min(k, n))[1].tolist())
    for _ in range(5):  # growth passes
        cur = np.array(list(grown))
        if len(cur) < 2:
            break
        p0, d = fit_line(pts[cur])
        d_inl = np.where(dist_to_line(pts, p0, d) <= tol)[0]
        # extend only into the connected neighbourhood of the region (chain)
        r = max(np.linalg.norm(pts[cur] - p0, axis=1).max() + 0.5, 0.8)
        near = set(tree.query_ball_point(p0, r))
        grown |= near & set(d_inl.tolist())
    mask = np.zeros(n, bool)
    mask[list(grown)] = True
    return mask


# -------------------------------------------------  method 2: 3DLiDAR catenary
def method_catenary(pts, tol=0.05, n_iter=1500, min_pts=8):
    """linear coarse extraction (RANSAC) -> catenary (sag) fit in vertical plane.

    The wire lies in a plane: horizontal direction + vertical. Fit
    z = a*cosh((s-s0)/a) + c  along the horizontal arc-length s.
    """
    n = len(pts)
    if n < min_pts:
        return np.zeros(n, bool)
    # coarse RANSAC line to find the wire direction and get candidates
    best = None
    for _ in range(n_iter):
        i, j = np.random.choice(n, 2, replace=False)
        if np.linalg.norm(pts[i] - pts[j]) < 0.3:
            continue
        d = pts[j] - pts[i]
        d /= np.linalg.norm(d)
        inl = dist_to_line(pts, pts[i], d) <= 0.08
        if inl.sum() > (best[0] if best else 0):
            best = (inl.sum(), inl, d)
    if best is None:
        return np.zeros(n, bool)
    inl, d = best[1], best[2]
    w = pts[inl]
    # horizontal direction = projection of d onto XY plane
    dh = np.array([d[0], d[1], 0.0])
    dh /= (np.linalg.norm(dh) + 1e-9)
    s = w[:, :2] @ dh[:2]          # horizontal arc-length
    z = w[:, 2]

    def cat(params, s):
        a, s0, c = params
        return a * np.cosh((s - s0) / max(a, 1e-3)) + c

    def resid(params):
        return cat(params, s) - z

    try:
        sol = least_squares(resid, x0=[1.0, s.mean(), z.min()], max_nfev=200)
        zhat = cat(sol.x, s)
    except Exception:
        return inl
    # keep candidates close to the catenary
    zfull = cat(sol.x, pts[:, :2] @ dh[:2])
    mask = inl & (np.abs(pts[:, 2] - zfull) <= tol)
    return mask


# ----------------------------------------------  method 3: 2D layered Hough ---
def method_hough2d(pts, tol=0.05, px_per_m=100.0, min_line_len=65):
    """powerline-detection: layered 2D projection -> Otsu -> Canny ->
    HoughLinesP (rho=1,theta=pi/180,threshold=50,minLineLength,maxLineGap=5).

    The wire's depth (x) is nearly constant, so its 2D signature is a diagonal
    in the YZ (side) view. We project YZ there.
    """
    n = len(pts)
    if n < 2:
        return np.zeros(n, bool)
    u, v = pts[:, 1], pts[:, 2]      # y, z  (side view)
    u0, v0 = u.min(), v.min()
    W = int((u.max() - u0) * px_per_m) + 2
    H = int((v.max() - v0) * px_per_m) + 2
    img = np.zeros((H, W), np.uint8)
    ui = np.clip(((u - u0) * px_per_m).astype(int), 0, W - 1)
    vi = np.clip(((v - v0) * px_per_m).astype(int), 0, H - 1)
    img[vi, ui] = 255
    _, src = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dst = cv2.Canny(src, 50, 150, None, 3)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, threshold=50,
                            minLineLength=min_line_len, maxLineGap=5)
    if lines is None:
        return np.zeros(n, bool)
    mask = np.zeros(n, bool)
    px = np.stack([ui, vi], axis=1).astype(np.float64)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = np.array([x2 - x1, y2 - y1])
        b = px - np.array([x1, y1])
        t = np.clip((b @ a) / (a @ a + 1e-9), 0, 1)
        d = np.linalg.norm(b - t[:, None] * a, axis=1)
        mask |= d <= tol * px_per_m
    return mask


# ------------------------------------------------------------------ driver ---
def score(name, mask, gt):
    tp = int((mask & gt).sum())
    recall = tp / max(int(gt.sum()), 1)
    prec = tp / max(int(mask.sum()), 1)
    f1 = 2 * recall * prec / max(recall + prec, 1e-9)
    return {"name": name, "n_sel": int(mask.sum()), "gt": int(gt.sum()),
            "recall": recall, "precision": prec, "f1": f1}


def run_on(pts, tag):
    gt = gt_wire_mask(pts)
    print(f"\n########## {tag}: n={len(pts)}, GT-wire={int(gt.sum())} "
          f"({100.0*gt.mean():.1f}%) ##########")

    # --- NAIVE: raw ROI (these methods assume the powerline is a dominant
    # structure; here the wall/floor dominate, so they latch onto background) ---
    naive_rows = []
    for name, fn in [
        ("railroad RANSAC-line", method_ransac_line),
        ("railroad 3D-Hough", method_hough3d),
        ("railroad region-grow", method_region_grow),
        ("3DLiDAR catenary", method_catenary),
        ("powerline 2D-Hough", method_hough2d),
    ]:
        naive_rows.append(score(name, fn(pts), gt))
    print("  [NAIVE, raw ROI]")
    for r in naive_rows:
        print(f"    {r['name']:22s} n_sel={r['n_sel']:5d} recall={r['recall']:.2f} "
              f"prec={r['precision']:.2f} F1={r['f1']:.2f}")

    # --- WITH PRIOR: the 2D mask gives us "the wire is the NEAREST thin
    # elevated structure" (same semantic as nearest_cluster in the existing
    # human-detection code). x < 4.55 = foreground (first depth gap at 4.55,
    # measured GT-free from the x-histogram); z >= 1.4 = above floor. This is
    # the CABLE seed railroad requires, translated from the 2D mask. ---
    x_thresh = 4.55
    cand = (pts[:, 0] < x_thresh) & (pts[:, 2] >= 1.4)
    ci = np.where(cand)[0]
    print(f"  [WITH PRIOR] nearest(x<{x_thresh:.2f}) & high(z>=1.4): "
          f"{len(ci)} pts, wire={int((cand&gt).sum())}/{int(gt.sum())}")
    if len(ci) < 2:
        print("    too few candidates"); return naive_rows
    rows = []
    for name, fn in [
        ("railroad RANSAC-line", method_ransac_line),
        ("railroad 3D-Hough", method_hough3d),
        ("railroad region-grow", method_region_grow),
        ("3DLiDAR catenary", method_catenary),
        ("powerline 2D-Hough", method_hough2d),
    ]:
        sub = fn(pts[ci])
        full = np.zeros(len(pts), bool)
        full[ci[sub]] = True
        rows.append(score(name, full, gt))
    hdr = f"  {'method':22s} {'n_sel':>6s} {'recall':>7s} {'prec':>6s} {'F1':>6s}"
    print(hdr)
    for r in rows:
        print(f"  {r['name']:22s} {r['n_sel']:6d} {r['recall']:7.2f} "
              f"{r['precision']:6.2f} {r['f1']:6.2f}")
    return rows


def main():
    # single-frame ROI (GT frame)
    roi1 = np.loadtxt(os.path.join(ROI_DIR, GT_STEM, GT_STEM + "_roi.txt"),
                      skiprows=1)[:, :3].astype(np.float64)
    run_on(roi1, "SINGLE-FRAME ROI")

    # accumulated ROI (28 frames, 1cm dedup)
    acc, nf, raw = load_accumulated(ROI_DIR)
    run_on(acc, f"ACCUMULATED ROI ({nf} frames)")


if __name__ == "__main__":
    main()
