#!/usr/bin/env python3
"""TTPLA 2D segmentation -> 3D wire detection + 2D-mask-prior completion.

Given the TTPLA "Line" segmentation mask (the big diagonal wire), recover the
wire's 3D geometry. The wire may be *curved*, so the 2D mask is treated as a
curve (its skeleton), NOT fitted to a straight line.

Per frame:
  TTPLA Line mask (highest-conf class-0 detection)
    -> skeletonize -> ordered 2D centerline polyline  (u(s), v(s), arc-length s)
    -> wide band around the skeleton (tolerates extrinsic error) -> candidate pts
    -> height pre-filter (reject the low near wall/floor)  -> candidate wire pts
    -> robust PCHIP depth-vs-arclength fit (curved)   -> d(s)
    -> unproject the skeleton at the fitted depth     -> completed 3D curve
    -> (optional) extend the off-image upper end along the local tangent

The 2D mask prior supplies the *shape* (continuous 2D path + extent); the lidar
supplies the *depth*. Output is a compact 3D polyline (not a densified cloud).

Reuses geometry/writers from open_vocab_3d_detect.py. Standalone; does not touch
the main pipeline.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import skimage.morphology as skim
from scipy import ndimage as ndi
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_vocab_3d_detect as ovd
from open_vocab_3d_detect import FX, FY, CX, CY, load_lidar, write_points_ply

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "ttpla", "yolo11s_seg_1280.pt")


# --------------------------------------------------------------------------- #
# TTPLA mask
# --------------------------------------------------------------------------- #
def run_ttpla_line(model, img_bgr, conf=0.5):
    """Run TTPLA, return (mask, conf) for the highest-conf class-0 (Line)
    detection, or (None, 0.0) if none above threshold."""
    res = model.predict(img_bgr, imgsz=1280, conf=conf, iou=0.7, verbose=False)[0]
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return None, 0.0
    cls = res.boxes.cls.cpu().numpy()
    c = res.boxes.conf.cpu().numpy()
    line_idx = np.where(cls == 0)[0]
    if len(line_idx) == 0:
        return None, 0.0
    best = line_idx[int(np.argmax(c[line_idx]))]
    mask = (res.masks.data[best].cpu().numpy() > 0.5).astype(np.uint8)
    return mask, float(c[best])


# --------------------------------------------------------------------------- #
# Skeleton -> ordered 2D centerline polyline
# --------------------------------------------------------------------------- #
def _adjacency(skel):
    """Build an adjacency list over skeleton pixels (largest 8-connected
    component only). Returns (pts (N,2) [x,y], adj list-of-lists)."""
    lab, n = ndi.label(skel > 0, structure=np.ones((3, 3), np.int32))
    if n == 0:
        return None, None
    # largest component
    sizes = ndi.sum(skel > 0, lab, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    comp = (lab == keep)
    ys, xs = np.where(comp)
    H, W = skel.shape
    idx_map = -np.ones((H, W), np.int32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        idx_map[y, x] = i
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    adj = [[] for _ in range(len(pts))]
    for i, (x, y) in enumerate(pts.astype(int)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H and idx_map[yy, xx] >= 0:
                    adj[i].append(int(idx_map[yy, xx]))
    return pts, adj


def _bfs_path(adj, start):
    """BFS from start; returns (farthest_node, path start->farthest_node)."""
    n = len(adj)
    dist = np.full(n, -1, np.int32)
    parent = np.full(n, -1, np.int32)
    dist[start] = 0
    q = [start]
    for v in q:
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                parent[w] = v
                q.append(w)
    far = int(np.argmax(dist))
    path = []
    cur = far
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return far, path


def skeleton_polyline(mask):
    """Skeletonize + longest-path ordering -> (us, vs, ss) arc-length polyline
    in pixel coords, or (None, None, None)."""
    skel = skim.skeletonize(mask > 0).astype(np.uint8)
    pts, adj = _adjacency(skel)
    if pts is None or len(pts) < 2:
        return None, None, None
    # tree diameter endpoints (robust to small side branches)
    a, _ = _bfs_path(adj, 0)
    b, path = _bfs_path(adj, a)
    poly = pts[path]                                  # (M,2) [x,y]
    # orient so s=0 is at the top of the image (smallest y)
    if poly[0, 1] > poly[-1, 1]:
        poly = poly[::-1]
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seg)])
    return poly[:, 0], poly[:, 1], ss


# --------------------------------------------------------------------------- #
# Wire inlier selection: PCA linearity + nearest coherent depth cluster
# --------------------------------------------------------------------------- #
def _local_linearity(pts, k=12):
    """Per-point local linearity from neighbour PCA.

    For each point, take its k nearest neighbours, centre them and compute the
    eigenvalues of the local 3x3 covariance. ``linearity = (e0 - e1) / e0`` is
    ~1 for points sitting on a locally 1D structure (a line) and drops for 2D
    surfaces (walls / floor) or volumetric clutter.

    The neighbourhood size matters: the lidar samples the floor/wall along scan
    lines, so a *small* k makes background look locally 1D too. A larger k
    (~12-20) lets the neighbourhood reveal the background's true 2D spread while
    the sparse wire stays 1D, so the wire's high-linearity points separate from
    the background's.
    """
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 2:
        return np.zeros(n)
    kk = min(k + 1, n)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=kk)
    lin = np.zeros(n)
    for i in range(n):
        nb = pts[idx[i, 1:]]                     # column 0 is the point itself
        if len(nb) < 2:
            continue
        c = nb - nb.mean(axis=0)
        ev = np.linalg.eigvalsh(c.T @ c / len(nb))
        ev = np.sort(ev)[::-1]
        if ev[0] > 1e-12:
            lin[i] = (ev[0] - ev[1]) / ev[0]
    return lin


def _nearest_coherent_depth(ds, gap_kmad=4.0, min_size=3):
    """Depth bounds of the nearest *coherent* cluster (gap statistics).

    The wire is a tight depth cluster (its depth is nearly constant along the
    skeleton), while the surviving 1D background (wall/pole edges) sits farther
    away and is more scattered. Sort the depths and split them into contiguous
    groups at "significant" gaps, where a gap is significant when it exceeds
    ``median(gap) + gap_kmad * 1.4826 * MAD(gap)`` -- a robust outlier threshold
    on the gap sizes, so the tolerance grows out of the data rather than a
    scene-specific metre value. Return the depth range of the first (nearest)
    group that holds at least ``min_size`` points; a nearer *singleton* (a stray
    point in front of the wire) is skipped, so it cannot hijack "nearest".

    Returns ``(lo, hi)`` depth bounds.
    """
    d = np.sort(np.asarray(ds, float))
    if len(d) < 2:
        return float(d.min()), float(d.max())
    gaps = np.diff(d)
    med = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - med)))
    eps = med + gap_kmad * 1.4826 * mad
    if eps <= 0.0:
        eps = 1e-6
    start = 0
    for i in range(len(gaps)):
        if gaps[i] > eps:
            grp = d[start:i + 1]
            if len(grp) >= min_size:
                return float(grp.min()), float(grp.max())
            start = i + 1
    grp = d[start:]
    if len(grp) >= min_size:
        return float(grp.min()), float(grp.max())
    return float(d.min()), float(d.max())


def _line_consensus(pts, ds, lin_k=12, lin_thr=0.9, gap_kmad=4.0, min_size=3):
    """Select the wire inliers: PCA linearity -> nearest coherent depth cluster.

    Two data-driven filters, no absolute depth/height:

      1. *Statistics filter*: keep only locally-1D candidates
         (``_local_linearity >= lin_thr``). This drops the 2D floor / near wall
         / far wall, whose scan-line-sampled points only look 1D under a too
         small neighbourhood.
      2. *Nearest coherent cluster*: among those 1D points, the wire is the
         nearest tight depth mode -- ``_nearest_coherent_depth`` finds it from
         the gap statistics, skipping nearer singletons.

    Returns a bool inlier mask over the inputs, or None if degenerate.
    """
    line = _local_linearity(pts, lin_k) >= lin_thr
    if int(line.sum()) < 3:
        return None
    lo, hi = _nearest_coherent_depth(ds[line], gap_kmad=gap_kmad,
                                     min_size=min_size)
    inl = line & (ds >= lo) & (ds <= hi)
    if int(inl.sum()) < 3:
        return None
    return inl


# --------------------------------------------------------------------------- #
# Off-image upper-end extension
# --------------------------------------------------------------------------- #
def _endpoint_tangent(pts, k=5):
    """Local tangent at the upper end of the inlier set (pts[0] = highest lidar-z,
    sorted by ascending arc-length s). PCA principal direction of the top-k points,
    oriented upward (increasing z). Returns a unit vector."""
    n = len(pts)
    if n < 2:
        return np.array([0.0, 0.0, 1.0])
    k = min(k, n)
    sub = pts[:k]
    c = sub.mean(axis=0)
    _, _, vh = np.linalg.svd(sub - c, full_matrices=False)
    t = vh[0].astype(float)
    if t[2] < 0:                     # the wire rises toward its top
        t = -t
    nrm = float(np.linalg.norm(t))
    return t / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])


def _extend_up(pts, tangent, extend_m=1.0, extend_to_z=3.0, step=0.05):
    """Extrapolate the curve's upper end (pts[0]) along `tangent`.

    The upper wire segment projects above the image top (v < 0), so no lidar
    point inside the band samples it and the recovered curve stops at the top
    visible inlier. Extend a short tangent beyond it, stopping when either
    `extend_m` metres are added or the lidar-z reaches `extend_to_z`.

    Returns (extended_pts, n_added). Input is returned unchanged if extension is
    disabled, the top already reaches the target height, or there are too few
    points to define a tangent."""
    if extend_m <= 0 or len(pts) < 2:
        return pts, 0
    top = pts[0]
    if extend_to_z > 0 and top[2] >= extend_to_z:
        return pts, 0
    dist = extend_m
    if extend_to_z > 0 and tangent[2] > 1e-3:
        dist = min(extend_m, (extend_to_z - top[2]) / tangent[2])
    if dist <= 0:
        return pts, 0
    n_add = max(1, int(round(dist / step)))
    # ts descends so ext[0] is the FURTHEST point (the new top) and ext[-1] sits
    # just above the original top -- the prepended block then continues the curve
    # smoothly upward instead of jumping back at the junction.
    ts = np.linspace(dist, dist / n_add, n_add)
    ext = top[None, :] + tangent[None, :] * ts[:, None]
    return np.vstack([ext, pts]), n_add


# --------------------------------------------------------------------------- #
# Library entry point: per-frame wire curve detection (writes nothing)
# --------------------------------------------------------------------------- #
def detect_wire_curve(model, stem, data_dir, conf=0.5, dilate=150,
                      lin_k=12, lin_thr=0.9, gap_kmad=4.0, min_size=3,
                      min_points=6, n_sample=100, extend_m=1.0,
                      extend_to_z=3.0, extend_top_k=5, extend_step=0.05,
                      mask_max_area=20000):
    """Detect the per-frame wire 3D curve (PCA linearity -> nearest coherent
    depth cluster), used by the ranging pipeline (`range_person_wire.py`).
    Same detection steps as `process_frame`, sharing the `_line_consensus`
    core, but writes no files.

    Thin file-loading shell over `detect_wire_curve_arrays`: reads
    images/<stem>.jpg (BGR), lidar/<stem>.bin and extrinsics/<stem>.txt from
    `data_dir`, then delegates all numerics. Third-party code with in-memory
    frames should call `detect_wire_curve_arrays` directly.

    Returns ``(curve, info)`` where ``curve`` is an ``(N,3)`` polyline in the
    lidar frame (``None`` on failure) and ``info["status"]`` is one of
    ``ok`` / ``no_line`` / ``mask_blowup`` / ``no_skeleton`` /
    ``too_few_band_points`` / ``too_few_inliers`` / ``degenerate_extent``.
    """
    img_path = os.path.join(data_dir, "images", stem + ".jpg")
    lidar_path = os.path.join(data_dir, "lidar", stem + ".bin")
    ext_path = os.path.join(data_dir, "extrinsics", stem + ".txt")
    if not all(os.path.exists(p) for p in (img_path, lidar_path, ext_path)):
        return None, {"stem": stem, "status": "missing_files"}
    img = cv2.imread(img_path)
    if img is None:
        return None, {"stem": stem, "status": "unreadable_image"}

    points = load_lidar(lidar_path)
    T_cam_to_lidar = np.loadtxt(ext_path, dtype=np.float64)      # raw file content
    T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)

    return detect_wire_curve_arrays(
        model, img, points, T_lidar_to_cam, conf=conf, dilate=dilate,
        lin_k=lin_k, lin_thr=lin_thr, gap_kmad=gap_kmad, min_size=min_size,
        min_points=min_points, n_sample=n_sample, extend_m=extend_m,
        extend_to_z=extend_to_z, extend_top_k=extend_top_k,
        extend_step=extend_step, mask_max_area=mask_max_area, stem=stem)


def detect_wire_curve_arrays(model, img_bgr, points, T_lidar_to_cam, conf=0.5,
                             dilate=150, lin_k=12, lin_thr=0.9, gap_kmad=4.0,
                             min_size=3, min_points=6, n_sample=100,
                             extend_m=1.0, extend_to_z=3.0, extend_top_k=5,
                             extend_step=0.05, mask_max_area=20000,
                             intrinsics=None, stem=None):
    """In-memory variant of `detect_wire_curve` (identical numerics, writes
    nothing): inputs are already-loaded arrays instead of file paths.

    Args:
        model:       TTPLA YOLO segmentation model (ultralytics).
        img_bgr:     (H,W,3) uint8 BGR image.
        points:      (N,3) lidar-frame points (float64).
        T_lidar_to_cam: 4x4 transform lidar -> camera frame.
        intrinsics:  optional (fx, fy, cx, cy); None uses the built-in
                     calibration of `open_vocab_3d_detect`.
        stem:        frame id recorded in the returned info dict (cosmetic).
    """
    # 1. TTPLA mask
    mask, mconf = run_ttpla_line(model, img_bgr, conf=conf)
    if mask is None:
        return None, {"stem": stem, "status": "no_line", "conf": mconf}
    area = int(mask.sum())
    if mask_max_area > 0 and area > mask_max_area:
        return None, {"stem": stem, "status": "mask_blowup", "conf": mconf,
                      "area": area}

    # 2. skeleton -> ordered 2D centerline
    us, vs, ss = skeleton_polyline(mask)
    if us is None:
        return None, {"stem": stem, "status": "no_skeleton", "conf": mconf,
                      "area": area}

    # 3. wide band -> candidate lidar points
    skel_img = np.zeros(mask.shape, np.uint8)
    skel_img[np.round(vs).astype(int), np.round(us).astype(int)] = 1
    band = cv2.dilate(skel_img, np.ones((2 * dilate + 1, 2 * dilate + 1),
                                        np.uint8))
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T_lidar_to_cam @ ph.T).T[:, :3]
    if intrinsics is None:
        u, v, z = ovd.project(p_cam)
    else:
        fx, fy, cx, cy = intrinsics
        u, v, z = ovd.project(p_cam, fx, fy, cx, cy)
    H, W = mask.shape
    valid = (z > 0.5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    sel = band[vi, ui].astype(bool)
    if sel.sum() < min_points:
        return None, {"stem": stem, "status": "too_few_band_points",
                      "conf": mconf, "n_band": int(sel.sum())}

    cand_lidar = points[valid][sel]          # lidar frame
    cand_u = u[valid][sel]
    cand_v = v[valid][sel]
    cand_d = z[valid][sel]                   # camera depth

    # arc-length of nearest skeleton pixel for each candidate
    tree = cKDTree(np.stack([us, vs], axis=1))
    _, si = tree.query(np.stack([cand_u, cand_v], axis=1))
    cand_s = ss[si]

    # 4. wire inliers: PCA linearity -> nearest coherent depth cluster
    inl = _line_consensus(cand_lidar, cand_d, lin_k=lin_k, lin_thr=lin_thr,
                          gap_kmad=gap_kmad, min_size=min_size)
    if inl is None:
        return None, {"stem": stem, "status": "too_few_inliers",
                      "conf": mconf, "n_inliers": 0, "select": "none"}
    n_inliers = int(inl.sum())
    if n_inliers < min_points:
        return None, {"stem": stem, "status": "too_few_inliers",
                      "conf": mconf, "n_inliers": n_inliers,
                      "select": "line_cluster"}

    inl_idx = np.where(inl)[0]
    if (cand_s[inl_idx].max() - cand_s[inl_idx].min()) < 1e-6:
        return None, {"stem": stem, "status": "degenerate_extent",
                      "conf": mconf, "select": "line_cluster"}

    # 5/6. fit a 3D PCHIP curve through the inlier wire points (lidar frame),
    #    parameterized by arc-length s, then sample densely.
    inl_s = cand_s[inl_idx].astype(float)
    inl_pts = cand_lidar[inl_idx]
    o = np.argsort(inl_s)
    inl_s = inl_s[o]
    inl_pts = inl_pts[o]
    us_s, inv = np.unique(inl_s, return_inverse=True)
    up = np.array([inl_pts[inv == i].mean(axis=0) for i in range(len(us_s))])
    s_lo, s_hi = float(us_s[0]), float(us_s[-1])
    s_grid = np.linspace(s_lo, s_hi, n_sample)
    if len(us_s) >= 3:
        lidar_pts = np.stack([
            PchipInterpolator(us_s, up[:, 0])(s_grid),
            PchipInterpolator(us_s, up[:, 1])(s_grid),
            PchipInterpolator(us_s, up[:, 2])(s_grid),
        ], axis=1)
    else:
        lidar_pts = np.stack(
            [np.interp(s_grid, us_s, up[:, k]) for k in range(3)], axis=1)

    # 6b. extend the upper (off-image) end along the local tangent
    tangent = _endpoint_tangent(up, k=extend_top_k)
    lidar_pts, n_ext = _extend_up(lidar_pts, tangent, extend_m=extend_m,
                                  extend_to_z=extend_to_z, step=extend_step)
    length_m = float(np.linalg.norm(np.diff(lidar_pts, axis=0), axis=1).sum())

    info = {
        "stem": stem, "status": "ok", "conf": round(mconf, 3),
        "n_inliers": n_inliers, "select": "line_cluster",
        "length_m": round(length_m, 3),
        "endpoints": [lidar_pts[0].tolist(), lidar_pts[-1].tolist()],
        "s_range": [round(s_lo, 1), round(s_hi, 1)], "n_extended": n_ext,
    }
    return lidar_pts, info


# --------------------------------------------------------------------------- #
# Per-frame processing
# --------------------------------------------------------------------------- #
def process_frame(model, stem, args):
    data_dir, out_dir = args.data_dir, args.out_dir
    img_path = os.path.join(data_dir, "images", stem + ".jpg")
    lidar_path = os.path.join(data_dir, "lidar", stem + ".bin")
    ext_path = os.path.join(data_dir, "extrinsics", stem + ".txt")
    if not all(os.path.exists(p) for p in (img_path, lidar_path, ext_path)):
        return {"stem": stem, "status": "missing_files"}

    img = cv2.imread(img_path)
    if img is None:
        return {"stem": stem, "status": "unreadable_image"}
    points = load_lidar(lidar_path)
    T_cam_to_lidar = np.loadtxt(ext_path, dtype=np.float64)      # raw file content
    T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)

    # 1. TTPLA mask
    mask, conf = run_ttpla_line(model, img, conf=args.conf)
    if mask is None:
        return {"stem": stem, "status": "no_line", "conf": conf}
    area = int(mask.sum())
    if args.mask_max_area > 0 and area > args.mask_max_area:
        return {"stem": stem, "status": "mask_blowup", "conf": conf, "area": area}

    # 2. skeleton -> ordered 2D centerline
    us, vs, ss = skeleton_polyline(mask)
    if us is None:
        return {"stem": stem, "status": "no_skeleton", "conf": conf, "area": area}

    # 3. narrow band -> candidate lidar points
    skel_img = np.zeros(mask.shape, np.uint8)
    skel_img[np.round(vs).astype(int), np.round(us).astype(int)] = 1
    band = cv2.dilate(skel_img, np.ones((2 * args.dilate + 1, 2 * args.dilate + 1),
                                        np.uint8))
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T_lidar_to_cam @ ph.T).T[:, :3]
    u, v, z = ovd.project(p_cam)
    H, W = mask.shape
    valid = (z > 0.5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    sel = band[vi, ui].astype(bool)
    if sel.sum() < args.min_points:
        return {"stem": stem, "status": "too_few_band_points",
                "conf": conf, "n_band": int(sel.sum())}

    cand_lidar = points[valid][sel]          # lidar frame
    cand_cam = p_cam[valid][sel]             # camera frame
    cand_u = u[valid][sel]
    cand_v = v[valid][sel]
    cand_d = z[valid][sel]                   # camera depth

    # arc-length of nearest skeleton pixel for each candidate
    tree = cKDTree(np.stack([us, vs], axis=1))
    _, si = tree.query(np.stack([cand_u, cand_v], axis=1))
    cand_s = ss[si]

    # 4. select the wire inliers: PCA local-linearity (the data-driven 1D
    #    template) -> nearest coherent depth cluster (gap statistics). Both
    #    filters grow their thresholds out of the data, so no scene-specific
    #    absolute depth/height constants are needed -- see _line_consensus.
    inl = _line_consensus(cand_lidar, cand_d, lin_k=args.lin_k,
                          lin_thr=args.lin_thr, gap_kmad=args.gap_kmad,
                          min_size=args.min_size)
    if inl is None:
        return {"stem": stem, "status": "too_few_inliers",
                "conf": conf, "n_inliers": 0, "select": "none"}
    select = "line_cluster"
    n_inliers = int(inl.sum())
    if n_inliers < args.min_points:
        return {"stem": stem, "status": "too_few_inliers",
                "conf": conf, "n_inliers": n_inliers, "select": select}

    # inlier indices into the band set
    inl_idx = np.where(inl)[0]

    # 5. curve extent = the lidar-sampled arclength range [s_lo, s_hi]. We only
    #    keep the measured portion (no extrapolation): the ends of the wire that
    #    lidar never sampled would otherwise drift frame-to-frame.
    if (cand_s[inl_idx].max() - cand_s[inl_idx].min()) < 1e-6:
        return {"stem": stem, "status": "degenerate_extent", "conf": conf,
                "select": select}

    # 6. fit a 3D curve through the inlier wire points (lidar frame),
    #    parameterized by arc-length s, and sample it densely. Fitting through
    #    the lidar inliers (rather than unprojecting the skeleton) recovers the
    #    wire's TRUE 3D position -- unprojecting the skeleton would inherit the
    #    extrinsic error that already shifted the projection off the skeleton.
    inl_s = cand_s[inl_idx].astype(float)
    inl_pts = cand_lidar[inl_idx]
    o = np.argsort(inl_s)
    inl_s = inl_s[o]
    inl_pts = inl_pts[o]
    us_s, inv = np.unique(inl_s, return_inverse=True)
    up = np.array([inl_pts[inv == i].mean(axis=0) for i in range(len(us_s))])
    s_lo, s_hi = float(us_s[0]), float(us_s[-1])
    s_grid = np.linspace(s_lo, s_hi, args.n_sample)
    if len(us_s) >= 3:
        lidar_pts = np.stack([
            PchipInterpolator(us_s, up[:, 0])(s_grid),
            PchipInterpolator(us_s, up[:, 1])(s_grid),
            PchipInterpolator(us_s, up[:, 2])(s_grid),
        ], axis=1)
    else:
        lidar_pts = np.stack(
            [np.interp(s_grid, us_s, up[:, k]) for k in range(3)], axis=1)

    # 6b. extend the upper end (z > ~2.6 m projects above the image top, v < 0,
    #     so lidar never samples it) along the local tangent. This is a short
    #     straight extrapolation beyond the last measured inlier, not a fit.
    tangent = _endpoint_tangent(up, k=args.extend_top_k)
    lidar_pts, n_ext = _extend_up(lidar_pts, tangent, extend_m=args.extend_m,
                                  extend_to_z=args.extend_to_z,
                                  step=args.extend_step)

    # curve arc length (m)
    length_m = float(np.linalg.norm(np.diff(lidar_pts, axis=0), axis=1).sum())

    # 7. outputs
    fdir = os.path.join(out_dir, stem)
    os.makedirs(fdir, exist_ok=True)

    with open(os.path.join(fdir, stem + "_wire_curve.txt"), "w") as f:
        f.write(f"# TTPLA wire 3D curve  stem={stem}  conf={conf:.3f}\n")
        f.write(f"# n_inliers={n_inliers}  n_curve={len(lidar_pts)}  "
                f"length_m={length_m:.3f}\n")
        f.write(f"# s_lo={s_lo:.1f}px  s_hi={s_hi:.1f}px  model=3d_pchip_thru_inliers  "
                f"select={select}  lin_thr={args.lin_thr:.2f}\n")
        f.write(f"# n_extended={n_ext}  extend_m={args.extend_m}m  "
                f"extend_to_z={args.extend_to_z}m\n")
        f.write("# x y z (lidar frame)\n")
        for p in lidar_pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    write_points_ply(
        os.path.join(fdir, stem + "_wire_points.ply"), cand_lidar[inl_idx],
        np.tile(np.array([0, 165, 255], np.uint8), (n_inliers, 1)))   # orange
    write_points_ply(
        os.path.join(fdir, stem + "_wire_curve.ply"), lidar_pts,
        np.tile(np.array([0, 255, 0], np.uint8), (len(lidar_pts), 1)))  # green

    # combined context PLY: the *full* original scan (grey) + rejected band
    # points (dim red: floor / near cable / far wall that were filtered out)
    # + wire inliers (orange) + curve (green), so the detection can be judged
    # against the whole scene in one file.
    valid_idx = np.where(valid)[0]
    band_idx = np.where(sel)[0]
    band_full_idx = valid_idx[band_idx]
    inlier_full_idx = band_full_idx[inl_idx]
    ctx_cols = np.tile(np.array([150, 150, 150], np.uint8), (len(points), 1))
    rejected = np.ones(len(band_full_idx), bool)
    rejected[inl_idx] = False
    ctx_cols[band_full_idx[rejected]] = np.array([180, 70, 70], np.uint8)
    ctx_cols[inlier_full_idx] = np.array([0, 165, 255], np.uint8)
    ctx_pts = np.vstack([points, lidar_pts])
    ctx_cols = np.vstack([
        ctx_cols,
        np.tile(np.array([0, 255, 0], np.uint8), (len(lidar_pts), 1)),
    ])
    write_points_ply(os.path.join(fdir, stem + "_context.ply"), ctx_pts, ctx_cols)

    # 2D overlay: mask contour + skeleton + inlier projections
    vis = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 200, 0), 1)
    for x, y in zip(us, vs):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)          # skeleton red
    for x, y in zip(cand_u[inl], cand_v[inl]):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 165, 255), -1)        # inliers orange
    for x, y in zip(cand_u[~inl], cand_v[~inl]):
        cv2.circle(vis, (int(x), int(y)), 4, (128, 128, 128), -1)      # rejected grey
    cv2.imwrite(os.path.join(fdir, stem + "_2d.jpg"), vis)

    return {
        "stem": stem, "status": "ok", "conf": round(conf, 3),
        "n_inliers": n_inliers, "n_band": int(sel.sum()),
        "select": select, "length_m": round(length_m, 3),
        "endpoints": [lidar_pts[0].tolist(), lidar_pts[-1].tolist()],
        "s_range": [round(s_lo, 1), round(s_hi, 1)],
        "n_extended": n_ext,
        "curve3d": lidar_pts.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=ovd.DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default="viz_ttpla_wire3d")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--frames", default="all",
                    help="'all' or comma-separated stems")
    ap.add_argument("--dilate", type=int, default=150,
                    help="band half-width (px) around the skeleton; must exceed "
                         "the extrinsic-error offset so the real wire points "
                         "become candidates")
    ap.add_argument("--n_sample", type=int, default=100, help="curve points")
    ap.add_argument("--lin_k", type=int, default=12,
                    help="neighbourhood size (k-NN) for the local-linearity wire "
                         "template. A wire is locally 1D (high linearity); the "
                         "lidar floor/wall is 2D but is sampled along scan lines, "
                         "so a SMALL k makes it look 1D too -- k must be large "
                         "enough (~12) to expose the background's 2D spread")
    ap.add_argument("--lin_thr", type=float, default=0.9,
                    help="linearity threshold for the wire template (0..1); only "
                         "locally-1D candidate points pass")
    ap.add_argument("--gap_kmad", type=float, default=4.0,
                    help="robust gap-significance factor for the nearest coherent "
                         "depth cluster: a gap is 'significant' when it exceeds "
                         "median(gap) + gap_kmad * 1.4826 * MAD(gap)")
    ap.add_argument("--min_size", type=int, default=3,
                    help="minimum cluster size (points) a depth cluster needs to "
                         "count as the wire; skips nearer singletons")
    ap.add_argument("--min_points", type=int, default=6)
    ap.add_argument("--extend_m", type=float, default=1.0,
                    help="max upper-end extension length (m) along the local "
                         "tangent; the off-image top segment (z above the image) "
                         "is extrapolated; 0 disables")
    ap.add_argument("--extend_to_z", type=float, default=3.0,
                    help="stop the upper-end extension once lidar-z reaches this "
                         "height (m); 0 disables the height stop and extends the "
                         "full --extend_m")
    ap.add_argument("--extend_top_k", type=int, default=5,
                    help="number of top inliers used to fit the extension tangent")
    ap.add_argument("--extend_step", type=float, default=0.05,
                    help="spacing (m) between extrapolated extension points")
    ap.add_argument("--mask_max_area", type=int, default=20000,
                    help="skip TTPLA mask above this area (px); 0 disables")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    from ultralytics import YOLO
    model = YOLO(args.weights).to(device)

    if args.frames == "all":
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(args.data_dir, "images", "*.jpg")))
    else:
        stems = [s.strip() for s in args.frames.split(",") if s.strip()]

    print(f"{len(stems)} frame(s) -> {args.out_dir}")
    results = {}
    for stem in stems:
        r = process_frame(model, stem, args)
        results[stem] = r
        if r["status"] == "ok":
            print(f"[ok]    {stem}: conf={r['conf']} inliers={r['n_inliers']} "
                  f"len={r['length_m']}m endpoints={[round(x,2) for x in r['endpoints'][0]]}->"
                  f"{[round(x,2) for x in r['endpoints'][1]]}")
        else:
            print(f"[skip]  {stem}: {r['status']} { {k: v for k, v in r.items() if k not in ('stem','status')} }")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    print(f"done: {n_ok}/{len(stems)} frames ok -> {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
