#!/usr/bin/env python3
"""Catenary (悬链线) wire reconstruction — a standalone processing path.

Multi-frame accumulation + 3DLiDAR-style catenary fit, ported from the
zwshi-pku/3DLiDAR ISPRS-2020 method (linear coarse extraction + catenary
modelling). Unlike ttpla_wire_3d.py's per-frame PCHIP depth curve, this path
accumulates the per-frame 2D-mask candidates across the (near-static) sequence
and fits a single physical catenary to the densified wire.

Model: the wire is straight in plan (XY) and sags as a catenary in the vertical
plane containing that direction:
    xy(s) = p0_xy + dh * s
    z(s)  = a * cosh((s - s0) / a) + c
where s is horizontal arc-length along the wire's horizontal direction dh.

Steps per frame (identical front-end to ttpla_wire_3d.process_frame):
  TTPLA Line mask -> skeleton -> wide band -> lidar candidates.
Across frames: accumulate candidates (body ~static, no registration needed).
Then: nearest & elevated prior -> RANSAC coarse 3D line -> catenary least-squares
fit -> dense 3D catenary polyline + visualization (2D overlay, fit plot, PLY).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_vocab_3d_detect as ovd
from open_vocab_3d_detect import load_lidar, write_points_ply
from ttpla_wire_3d import run_ttpla_line, skeleton_polyline, DEFAULT_WEIGHTS


# --------------------------------------------------------------------------- #
# Candidate collection (front-end, identical to ttpla_wire_3d steps 1-3)
# --------------------------------------------------------------------------- #
def collect_candidates(model, stem, args):
    """Return (frame_dict, None) with lidar-frame candidates + 2D geometry, or
    (None, status) if this frame cannot contribute."""
    data_dir, = args.data_dir,
    img_path = os.path.join(data_dir, "images", stem + ".jpg")
    lidar_path = os.path.join(data_dir, "lidar", stem + ".bin")
    ext_path = os.path.join(data_dir, "extrinsics", stem + ".txt")
    if not all(os.path.exists(p) for p in (img_path, lidar_path, ext_path)):
        return None, "missing_files"
    img = cv2.imread(img_path)
    if img is None:
        return None, "unreadable_image"

    mask, conf = run_ttpla_line(model, img, conf=args.conf)
    if mask is None:
        return None, "no_line"
    us, vs, ss = skeleton_polyline(mask)
    if us is None:
        return None, "no_skeleton"

    points = load_lidar(lidar_path)
    T_cam_to_lidar = np.loadtxt(ext_path, dtype=np.float64)
    T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)

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
    if sel.sum() < 2:
        return None, "too_few_band_points"

    cand_lidar = points[valid][sel]
    cand_cam = p_cam[valid][sel]
    cand_u = u[valid][sel]
    cand_v = v[valid][sel]
    cand_d = z[valid][sel]

    tree = cKDTree(np.stack([us, vs], axis=1))
    _, si = tree.query(np.stack([cand_u, cand_v], axis=1))
    cand_s = ss[si]

    return {
        "stem": stem, "conf": conf, "img": img, "mask": mask,
        "us": us, "vs": vs, "ss": ss, "T_lidar_to_cam": T_lidar_to_cam,
        "cand_lidar": cand_lidar, "cand_cam": cand_cam,
        "cand_u": cand_u, "cand_v": cand_v, "cand_d": cand_d, "cand_s": cand_s,
    }, "ok"


# --------------------------------------------------------------------------- #
# Nearest / elevated prior (the 2D mask's "nearest thin high structure" seed)
# --------------------------------------------------------------------------- #
def _first_depth_gap(x, lo_frac=0.1, hi_frac=0.6):
    """Threshold at the first big gap in the sorted x (lidar-forward) values:
    foreground (wire) vs background. Data-driven, GT-free."""
    xs = np.sort(x)
    if len(xs) < 4:
        return xs[-1] + 1
    gaps = np.diff(xs)
    lo, hi = int(lo_frac * len(xs)), int(hi_frac * len(xs))
    if hi <= lo + 1:
        return xs[-1] + 1
    k = lo + int(np.argmax(gaps[lo:hi]))
    return xs[k]


# --------------------------------------------------------------------------- #
# Catenary fit
# --------------------------------------------------------------------------- #
def _dist_to_line(pts, p0, d):
    return np.linalg.norm(np.cross(pts - p0, d), axis=1)


def fit_catenary(pts, tol=0.05, ransac_tol=0.08, n_iter=2000, n_curve=200):
    """RANSAC coarse line -> catenary in the vertical plane.

    Returns (inlier_mask, curve_pts (n_curve,3), info dict) or
    (None, None, {}) if there are too few points.
    """
    n = len(pts)
    if n < 8:
        return None, None, {}
    rng = np.random.default_rng(0)

    # 1. RANSAC coarse 3D line -> direction d and anchor p0
    best = (0, None, None, None)
    idx = np.arange(n)
    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(pts[i] - pts[j]) < 0.3:
            continue
        d = pts[j] - pts[i]
        d /= np.linalg.norm(d)
        inl = _dist_to_line(pts, pts[i], d) <= ransac_tol
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl, pts[i], d)
    _, inl, p0, d = best
    if inl.sum() < 8:
        return None, None, {}
    w = pts[inl]

    # 2. horizontal direction (straight in plan) + horizontal arc-length s
    dh = np.array([d[0], d[1]], float)
    nrm = np.linalg.norm(dh)
    if nrm < 1e-6:
        dh = np.array([1.0, 0.0])
    else:
        dh /= nrm
    s = (w[:, :2] - p0[:2]) @ dh
    z = w[:, 2]

    def cat(p, s):
        a, s0, c = p
        return a * np.cosh((s - s0) / max(a, 1e-3)) + c

    # 3. least-squares catenary fit (soft_l1 for robustness, a constrained > 0)
    def refit(a, s0, c, s, z):
        try:
            sol = least_squares(lambda p: cat(p, s) - z, x0=[a, s0, c],
                                max_nfev=500, loss="soft_l1",
                                bounds=([1e-3, -np.inf, -np.inf],
                                        [np.inf, np.inf, np.inf]))
            return sol.x
        except Exception:
            return np.array([a, s0, c])

    # coarse init: a ~ span (metres), s0 ~ centre, c ~ min z
    span = float(s.max() - s.min())
    a, s0, c = refit(max(span, 1.0), float(s.mean()), float(z.min()), s, z)

    # 4. robust inliers near the catenary -> refit once on those inliers
    zfull = cat([a, s0, c], (pts[:, :2] - p0[:2]) @ dh)
    mask = np.abs(pts[:, 2] - zfull) <= tol
    if mask.sum() >= 8:
        w2 = pts[mask]
        s2 = (w2[:, :2] - p0[:2]) @ dh
        z2 = w2[:, 2]
        a, s0, c = refit(a, s0, c, s2, z2)
        zfull = cat([a, s0, c], (pts[:, :2] - p0[:2]) @ dh)
        mask = np.abs(pts[:, 2] - zfull) <= tol

    # 5. dense curve over the measured horizontal extent
    s_grid = np.linspace(s.min(), s.max(), n_curve)
    xy = p0[:2] + dh[None, :] * s_grid[:, None]
    zz = cat([a, s0, c], s_grid)
    curve = np.column_stack([xy[:, 0], xy[:, 1], zz])

    # physical sag = max vertical deviation of the catenary from its straight
    # chord between the two endpoints (not the total vertical span)
    chord_z = np.interp(s_grid, [s_grid[0], s_grid[-1]], [zz[0], zz[-1]])
    sag = float(np.max(np.abs(zz - chord_z)))

    info = {"a": float(a), "s0": float(s0), "c": float(c),
            "dir": dh.tolist(), "n_inliers": int(mask.sum()),
            "n_input": n, "span_h": float(s.max() - s.min()),
            "span_z": float(zz.max() - zz.min()), "sag": sag}
    return mask, curve, info


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def write_vis(out_dir, ref, cand_lidar, mask, curve, info, x_thresh,
              data_dir, dilate):
    os.makedirs(out_dir, exist_ok=True)
    stem = ref["stem"]
    T = ref["T_lidar_to_cam"]

    # 1. ASCII curve
    with open(os.path.join(out_dir, "catenary_curve.txt"), "w") as f:
        f.write(f"# catenary wire  ref_frame={stem}\n")
        f.write(f"# a={info['a']:.3f} s0={info['s0']:.3f} c={info['c']:.3f} "
                f"dir={info['dir']}\n")
        f.write(f"# n_inliers={info['n_inliers']}/{info['n_input']} "
                f"span_h={info['span_h']:.3f}m sag={info['sag']:.3f}m\n")
        f.write("# x y z (lidar frame)\n")
        for p in curve:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    # 2. PLY: inliers (orange) + catenary curve (green)
    write_points_ply(os.path.join(out_dir, "catenary_wire_points.ply"),
                     cand_lidar[mask],
                     np.tile(np.array([0, 165, 255], np.uint8),
                             (int(mask.sum()), 1)))
    write_points_ply(os.path.join(out_dir, "catenary_curve.ply"), curve,
                     np.tile(np.array([0, 255, 0], np.uint8),
                             (len(curve), 1)))

    # 3. context PLY: full reference scan (grey) + rejected band (dim red) +
    #    wire inliers (orange) + curve (green)
    points = load_lidar(os.path.join(data_dir, "lidar", stem + ".bin"))
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T @ ph.T).T[:, :3]
    u, v, z = ovd.project(p_cam)
    H, W = ref["mask"].shape
    valid = (z > 0.5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    band = np.zeros((H, W), np.uint8)
    band[np.round(ref["vs"]).astype(int), np.round(ref["us"]).astype(int)] = 1
    band = cv2.dilate(band, np.ones((2 * dilate + 1, 2 * dilate + 1),
                                    np.uint8))
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    in_band = band[vi, ui].astype(bool)
    cols = np.tile(np.array([150, 150, 150], np.uint8), (len(points), 1))
    cols[valid][in_band] = np.array([180, 70, 70], np.uint8)
    ctx_pts = np.vstack([points, cand_lidar[mask], curve])
    ctx_cols = np.vstack([
        cols,
        np.tile(np.array([0, 165, 255], np.uint8), (int(mask.sum()), 1)),
        np.tile(np.array([0, 255, 0], np.uint8), (len(curve), 1)),
    ])
    write_points_ply(os.path.join(out_dir, "catenary_context.ply"),
                     ctx_pts, ctx_cols)

    # 4. 2D overlay on the reference frame
    img = ref["img"].copy()
    cnts, _ = cv2.findContours(ref["mask"], cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 200, 0), 1)
    for x, y in zip(ref["us"], ref["vs"]):
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
    # inliers / rejected
    for x, y in zip(ref["cand_u"][mask], ref["cand_v"][mask]):
        cv2.circle(img, (int(x), int(y)), 4, (0, 165, 255), -1)
    for x, y in zip(ref["cand_u"][~mask], ref["cand_v"][~mask]):
        cv2.circle(img, (int(x), int(y)), 4, (128, 128, 128), -1)
    # re-project the catenary curve
    ch = np.concatenate([curve, np.ones((len(curve), 1))], axis=1)
    cc = (T @ ch.T).T[:, :3]
    cu, cvv, cz = ovd.project(cc)
    ok = (cz > 0.5) & (cu >= 0) & (cu < W - 1) & (cvv >= 0) & (cvv < H - 1)
    pts2d = np.stack([cu[ok], cvv[ok]], axis=1).astype(int)
    cv2.polylines(img, [pts2d], False, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "catenary_2d.jpg"), img)

    return os.path.join(out_dir, "catenary_2d.jpg")


def write_fit_plot(out_dir, cand_lidar, mask, curve, info):
    """z vs horizontal arc-length plot showing the catenary fit (the sag)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    dh = np.array(info["dir"])
    p0 = curve[0][:2]
    s_all = (cand_lidar[:, :2] - p0) @ dh
    s_curve = (curve[:, :2] - p0) @ dh
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # (a) z vs s: inliers + fitted catenary
    ax = axes[0]
    ax.scatter(s_all[~mask], cand_lidar[~mask, 2], s=4, c="0.7", label="rejected")
    ax.scatter(s_all[mask], cand_lidar[mask, 2], s=8, c="#00a5ff", label="inliers")
    ax.plot(s_curve, curve[:, 2], "g-", lw=2, label="catenary fit")
    ax.set_xlabel("horizontal arc-length s (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("catenary z(s) = a·cosh((s-s0)/a)+c")
    ax.legend()
    # (b) 3D view
    ax = axes[1]
    ax.scatter(cand_lidar[~mask, 0], cand_lidar[~mask, 2], s=4, c="0.7")
    ax.scatter(cand_lidar[mask, 0], cand_lidar[mask, 2], s=8, c="#00a5ff")
    ax.plot(curve[:, 0], curve[:, 2], "g-", lw=2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("x-z view")
    fig.suptitle(f"catenary  a={info['a']:.2f}  sag={info['sag']:.3f}m  "
                 f"inliers={info['n_inliers']}")
    fig.tight_layout()
    p = os.path.join(out_dir, "catenary_fit.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=ovd.DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default="viz_catenary_wire3d")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--frames", default="all")
    ap.add_argument("--ref_frame", default="img_1700603346.892936",
                    help="frame used for the 2D overlay / context PLY")
    ap.add_argument("--dilate", type=int, default=150)
    ap.add_argument("--z_min", type=float, default=1.4,
                    help="elevated prior: keep wire candidates with lidar-z >= this")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="catenary inlier tolerance (m)")
    ap.add_argument("--ransac_tol", type=float, default=0.08,
                    help="coarse RANSAC line tolerance (m)")
    ap.add_argument("--ransac_iter", type=int, default=2000)
    ap.add_argument("--n_curve", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    if args.frames == "all":
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(args.data_dir, "images", "*.jpg")))
    else:
        stems = [s.strip() for s in args.frames.split(",") if s.strip()]

    frames = []
    n_skip = 0
    for stem in stems:
        f, st = collect_candidates(model, stem, args)
        if st == "ok":
            frames.append(f)
        else:
            n_skip += 1

    print(f"{len(frames)}/{len(stems)} frames collected "
          f"({n_skip} skipped) -> accumulate")
    cand = np.concatenate([f["cand_lidar"] for f in frames], axis=0)
    print(f"accumulated candidates: {len(cand)} pts (before prior)")

    # reference frame: used for the "nearest" threshold (single-frame depth gap
    # is clean) and for the 2D overlay. The wire is static, so its depth is
    # fixed across the sequence.
    ref = next(f for f in frames if f["stem"] == args.ref_frame) \
        if any(f["stem"] == args.ref_frame for f in frames) else frames[0]
    x_thresh = _first_depth_gap(ref["cand_lidar"][:, 0])
    mask = (cand[:, 0] < x_thresh) & (cand[:, 2] >= args.z_min)
    print(f"nearest(x<{x_thresh:.2f}, from ref {ref['stem']}) & "
          f"high(z>={args.z_min}): {int(mask.sum())} pts")

    inl, curve, info = fit_catenary(cand[mask], tol=args.tol,
                                    ransac_tol=args.ransac_tol,
                                    n_iter=args.ransac_iter,
                                    n_curve=args.n_curve)
    if curve is None:
        print("catenary fit failed (too few points)"); return

    # label the ref frame's candidates as wire inliers by proximity to the
    # fitted catenary curve (for the 2D overlay).
    dh = np.array(info["dir"])
    p0 = curve[0][:2]
    s_curve = (curve[:, :2] - p0) @ dh
    s_ref = (ref["cand_lidar"][:, :2] - p0) @ dh
    z_interp = interp1d(s_curve, curve[:, 2], bounds_error=False,
                        fill_value="extrapolate")
    z_ref = z_interp(s_ref)
    ref_mask = np.abs(ref["cand_lidar"][:, 2] - z_ref) <= args.tol

    info["n_frames"] = len(frames)
    info["x_thresh"] = round(float(x_thresh), 3)
    with open(os.path.join(args.out_dir, "catenary_fit.json"), "w") as f:
        json.dump({**info,
                   "endpoints": [curve[0].tolist(), curve[-1].tolist()],
                   "length_m": float(np.linalg.norm(np.diff(curve, axis=0),
                                                    axis=1).sum())}, f, indent=2)

    # viz: build a ref dict carrying the needed fields
    ref_viz = {
        "stem": ref["stem"], "img": ref["img"], "mask": ref["mask"],
        "us": ref["us"], "vs": ref["vs"], "T_lidar_to_cam": ref["T_lidar_to_cam"],
        "cand_u": ref["cand_u"], "cand_v": ref["cand_v"],
    }
    p2d = write_vis(args.out_dir, ref_viz, ref["cand_lidar"], ref_mask,
                    curve, info, x_thresh, args.data_dir, args.dilate)
    p_plot = write_fit_plot(args.out_dir, cand[mask], inl, curve, info)

    print(f"catenary: a={info['a']:.3f} s0={info['s0']:.3f} c={info['c']:.3f} "
          f"dir={[round(x,3) for x in info['dir']]}")
    print(f"  inliers={info['n_inliers']}/{int(mask.sum())}  sag={info['sag']:.3f}m "
          f"span_h={info['span_h']:.3f}m  n_frames={len(frames)}")
    print(f"  endpoints {[round(x,2) for x in curve[0]]} -> "
          f"{[round(x,2) for x in curve[-1]]}")
    print(f"  -> {args.out_dir}/  (2D={p2d}  plot={p_plot})")


if __name__ == "__main__":
    main()
