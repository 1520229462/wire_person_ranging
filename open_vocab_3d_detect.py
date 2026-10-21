#!/usr/bin/env python3
"""Zero-training open-vocabulary 3D detection + ranging ("Just Add Geometry" style).

Pipeline per frame:
  1. Open-vocabulary 2D detector (YOLO-World or OWLv2, text prompt -> boxes).
  2. (optional) SAM2 refines each box into a pixel mask for tighter point selection.
  3. Back-project the LiDAR/radar points into the image with the (rough) extrinsic
     + intrinsics, keep the points that fall inside each detection box/mask.
  4. Fit an oriented 3D box to those points via PCA -> 3D position + distance.

Calibration robustness: the extrinsic is used ONLY to select which points belong to
an object; the 3D box / distance are computed from raw point-cloud geometry and are
therefore independent of extrinsic accuracy. New categories = change the --categories
text prompt, no retraining.

Outputs per frame (in <out_dir>/<stem>/):
  <stem>.ply        -- the point cloud (green = points inside a detection),
                       open in MeshLab to see the points
  <stem>_box.ply    -- the 3D detection box(es) as a red mesh; import into the
                       same MeshLab session (File > Import Mesh) to see the box
                       on top of the points
  <stem>_2d.jpg     -- 2D image with box (+ mask) + distance label
  <stem>_box.txt    -- per-detection center / extents / distance / score
and a top-level <out_dir>/results.json summarising all frames.

(The points and box are split into two PLYs because MeshLab's default render
mode draws only the triangles of a mesh and hides the isolated point vertices.)
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

# --- Undistorted pinhole intrinsics @1280x720 (verified against Data/custom) ---
FX, FY, CX, CY = 622.6716, 694.2052, 644.2907, 522.0188
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 12 triangles (6 quad faces) of the unit box, corners indexed as in box_corners().
BOX_FACES = [
    [0, 1, 3], [0, 3, 2],   # z- (bottom)
    [4, 5, 7], [4, 7, 6],   # z+ (top)
    [0, 4, 5], [0, 5, 1],   # y-
    [2, 3, 7], [2, 7, 6],   # y+
    [0, 2, 6], [0, 6, 4],   # x-
    [1, 5, 7], [1, 7, 3],   # x+
]

# The 12 edges of the unit box (corner indices as in box_corners()), used by the
# wireframe rendering in write_box_ply().
BOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),   # z- (bottom)
    (4, 5), (4, 6), (5, 7), (6, 7),   # z+ (top)
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]

# Class colours. PLY files carry RGB; OpenCV uses BGR.
RGB_PERSON = (255, 0, 0)     # red 3D box
RGB_WIRE = (0, 0, 255)       # blue 3D box
PTS_PERSON = (0, 200, 0)     # green selected points (RGB)
PTS_WIRE = (0, 165, 255)     # orange selected points (RGB)
BGR_PERSON = (0, 255, 0)     # green 2D box
BGR_WIRE = (255, 0, 0)       # blue 2D box


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def load_lidar(path):
    """Read KITTI-style .bin -> (N,3) float64 points in the lidar frame."""
    arr = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return arr[:, :3].astype(np.float64)


def load_extrinsic(path):
    """Read extrinsics/<stem>.txt (cam_pose = T_cam_to_lidar) -> T_lidar_to_cam."""
    cam_pose = np.loadtxt(path, dtype=np.float64)
    return np.linalg.inv(cam_pose)


def project(p_cam, fx=FX, fy=FY, cx=CX, cy=CY):
    """p_cam (N,3) in camera frame (right,down,forward) -> (u, v, z)."""
    z = p_cam[:, 2]
    u = fx * p_cam[:, 0] / z + cx
    v = fy * p_cam[:, 1] / z + cy
    return u, v, z


def fit_box(p):
    """PCA oriented box on points p (N,3). Returns (center, extents, R).

    center (3,), extents (3,) = full side lengths along principal axes,
    R (3,3) = orthonormal rotation with the principal axes as columns.
    """
    mean = p.mean(axis=0)
    c = p - mean
    cov = (c.T @ c) / (len(p) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(eigvals)[::-1]               # descending variance
    R = eigvecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1                              # enforce right-handedness
    q = c @ R                                       # (N,3) in box frame
    mins, maxs = q.min(axis=0), q.max(axis=0)
    extents = maxs - mins
    center_local = (mins + maxs) / 2.0
    center = mean + R @ center_local
    return center, extents, R


def box_corners(center, extents, R):
    """8 corners of an oriented box, in the order referenced by BOX_FACES."""
    dx, dy, dz = extents / 2.0
    local = np.array([
        [-dx, -dy, -dz], [+dx, -dy, -dz], [-dx, +dy, -dz], [+dx, +dy, -dz],
        [-dx, -dy, +dz], [+dx, -dy, +dz], [-dx, +dy, +dz], [+dx, +dy, +dz],
    ])
    return center + local @ R.T                       # (8,3)


def _dbscan_labels(points, eps=1.0, min_samples=3):
    """DBSCAN clustering -> (labels, n_clusters). labels[i] == -1 marks noise.

    Falls back to (all-noise, 0 clusters) when the selection is too large
    (O(N^2) distance matrix) or too small to cluster.
    """
    n = len(points)
    labels = -np.ones(n, dtype=np.int64)
    if n <= min_samples or n > 1500:
        return labels, 0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)  # (N,N)
    neighbor = d <= eps
    core = neighbor.sum(axis=1) >= min_samples
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        labels[i] = cluster
        while stack:
            p = stack.pop()
            for q in np.nonzero(neighbor[p])[0]:
                if labels[q] == -1:
                    labels[q] = cluster
                    if core[q]:
                        stack.append(q)
        cluster += 1
    return labels, cluster


def largest_cluster(points, eps=1.0, min_samples=3):
    """Return a boolean mask of the largest DBSCAN cluster (by point count)."""
    labels, n_clusters = _dbscan_labels(points, eps, min_samples)
    if n_clusters == 0:
        return np.ones(len(points), dtype=bool)
    counts = np.bincount(labels[labels != -1])
    if counts.size == 0:
        return np.ones(len(points), dtype=bool)
    return labels == int(np.argmax(counts))


def nearest_cluster(points, depths, eps=1.0, min_samples=3):
    """Return a boolean mask of the DBSCAN cluster with the smallest mean camera
    depth (the nearest surface).

    Projection-based point selection is depth-ambiguous: every point along a
    camera ray that falls in the 2D mask is selected, so background (wall/floor
    behind the object) gets mixed in. The object is the *nearest* surface, so we
    keep the nearest 3D cluster instead of the largest one.
    """
    labels, n_clusters = _dbscan_labels(points, eps, min_samples)
    if n_clusters == 0:
        return np.ones(len(points), dtype=bool)
    best, best_depth = None, None
    for c in range(n_clusters):
        m = labels == c
        if int(m.sum()) < min_samples:
            continue
        md = float(np.mean(depths[m]))
        if best_depth is None or md < best_depth:
            best_depth, best = md, m
    if best is None:
        return np.ones(len(points), dtype=bool)
    return best


# --------------------------------------------------------------------------- #
# VGGT single-image depth prior (canonical -> metric via per-frame scale)
# --------------------------------------------------------------------------- #
def vggt_scale(radar_z, vggt_depth, ui, vi, z_lo=1.5, z_hi=12.0, lo=25, hi=75):
    """Per-frame metric scale s = median(radar_z / vggt_depth) over valid points.

    VGGT single-image depth is *canonical*: during training it is normalized by
    the scene's avg_scale (mean distance of all world points to the first
    camera), so at inference the predicted depth is metric only up to an unknown
    per-frame constant (~4.9 for this data). Radar is metric, so the ratio
    radar_z / vggt_depth recovers the scale. Robust median after trimming.
    """
    d = vggt_depth[vi, ui]
    m = np.isfinite(d) & (d > 0.15) & (radar_z > z_lo) & (radar_z < z_hi)
    if int(m.sum()) < 30:
        return None
    r = radar_z[m] / d[m]
    qlo, qhi = np.percentile(r, [lo, hi])
    r = r[(r >= qlo) & (r <= qhi)]
    return float(np.median(r)) if len(r) else None


def vggt_target_depth(vggt_depth, mask, lo=10, hi=90):
    """Robust median of the VGGT depth inside a 2D mask (canonical units).

    Trims the 10%/90% quantiles first so a few background pixels at the mask
    edge don't drag the target depth toward the wall behind the object.
    """
    d = vggt_depth[mask]
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return None
    qlo, qhi = np.percentile(d, [lo, hi])
    d = d[(d >= qlo) & (d <= qhi)]
    return float(np.median(d)) if len(d) else None


def fps(xyz, k):
    """Farthest-point sampling of (N,3) points down to min(N,k) points."""
    n = len(xyz)
    if n <= k:
        return xyz
    idx = [0]
    dist = np.linalg.norm(xyz - xyz[0], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(dist))
        idx.append(i)
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        dist = np.minimum(dist, d)
    return xyz[idx]


def chamfer_distance(a, b):
    """Bidirectional Chamfer distance between two point sets (N,3),(M,3)."""
    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a), cKDTree(b)
    d_ab, _ = ta.query(b, k=1)
    d_ba, _ = tb.query(a, k=1)
    return float(np.mean(d_ab) + np.mean(d_ba))


def shape_chamfer(a, b):
    """Scale- and translation-invariant Chamfer (both centered, unit max-norm).

    Both point clouds live in the same camera frame with the same orientation,
    so no rotation is estimated; centering + normalizing by the max radius makes
    the score compare pure SHAPE, robust to VGGT's per-object scale/depth error.
    """
    ca = a - a.mean(axis=0)
    cb = b - b.mean(axis=0)
    ca = ca / max(np.linalg.norm(ca, axis=1).max(), 1e-9)
    cb = cb / max(np.linalg.norm(cb, axis=1).max(), 1e-9)
    return chamfer_distance(ca, cb)


def _two_mode_threshold(d):
    """Robust near/far depth split -> threshold keeping the whole near mode.

    Runs 1D k-means (k=2) on the depths, seeded at min/max, and returns the
    midpoint between the two cluster centres. Unlike a median split (which cuts
    the target in half whenever the target is the dominant mode) or a histogram
    Otsu (whose percentile binning clips the sparse far tail), k-means finds the
    target centre and the background centre regardless of their size ratio, so
    the midpoint lands in the gap BETWEEN them and keeps the whole target.
    """
    c0, c1 = float(d.min()), float(d.max())
    for _ in range(24):
        lab = np.abs(d - c1) < np.abs(d - c0)       # True = nearer to c1 (far)
        if not lab.any() or lab.all():
            break
        c0, c1 = float(d[~lab].mean()), float(d[lab].mean())
    if c0 > c1:
        c0, c1 = c1, c0
    return 0.5 * (c0 + c1)


def _sor_mask(pts, k=8, m=2.0):
    """Statistical outlier removal -> bool mask over pts (True = keep).

    For each point, mean distance to its k nearest neighbours; drop points whose
    mean distance is > m std above the global mean. A depth-map "tail" is a thin
    1D streak of points (interpolated depth at a silhouette edge), so its points
    are sparser than the dense body surface and get dropped. cKDTree gives the
    neighbour distances without any external dependency.
    """
    n = len(pts)
    if n <= k:
        return np.ones(n, dtype=bool)
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(pts).query(pts, k=k + 1)
    mean_k = dist[:, 1:].mean(axis=1)               # col 0 is self (0.0)
    thr = mean_k.mean() + m * mean_k.std()
    return mean_k <= thr


def vggt_target_template(vggt_depth, mask, k=128, open_radius=2, sor_m=2.0):
    """Build the VGGT target template + intermediate stage data (camera frame).

    Three stages, each fixing a failure mode of the old "median depth mode":
      1. Foreground/background split via 1D k-means (k=2) on depth -> keeps the
         WHOLE target (the median split kept only the front/belly half; a
         histogram Otsu clips the sparse far tail through its percentile binning).
      2. 2D morphological opening of the foreground mask -> strips the thin
         depth-edge "tail" rim where the upsampled depth smears at the silhouette.
      3. Statistical outlier removal (SOR) on the unprojected cloud -> drops any
         residual sparse streaks that opening missed.
    Returns a dict, or None if too few valid pixels:
      template   -- (K,3) FPS-downsampled foreground cloud (the shape template)
      full_cloud -- (M,3) full mask cloud, canonical camera frame
      full_depth -- (M,)  depth of each full-cloud point
      near       -- (M,)  bool, True for the surviving foreground (template) pts
      px, py     -- (M,)  pixel coords of each full-cloud point (for 2D overlay)
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    d = vggt_depth[ys, xs]
    ok = np.isfinite(d) & (d > 0)
    if int(ok.sum()) < 16:
        return None
    xs, ys, d = xs[ok], ys[ok], d[ok]
    n = len(d)

    # (1) foreground / background split at the near/far depth valley
    thr = _two_mode_threshold(d)
    fg = d < thr
    # degenerate guard: no clean bimodality -> fall back to a band around the
    # mask-centroid depth (target = near coherent surface anchored at centre)
    if int(fg.sum()) < 16 or int(fg.sum()) > int(0.95 * n):
        d0 = float(np.median(d))
        sig = 1.4826 * float(np.median(np.abs(d - d0)))
        fg = (d >= d0 - 4.0 * sig) & (d <= d0 + 4.0 * sig)
    if int(fg.sum()) < 16:
        fg = np.ones(n, dtype=bool)                 # last-resort: whole mask

    # (2) 2D opening to strip the thin tail rim (depth smears at the silhouette)
    if open_radius > 0:
        from scipy import ndimage
        H, W = vggt_depth.shape
        img = np.zeros((H, W), dtype=bool)
        img[ys[fg], xs[fg]] = True
        img = ndimage.binary_opening(
            img, structure=ndimage.generate_binary_structure(2, 1),
            iterations=open_radius)
        fg = img[ys, xs]

    # (3) unproject + SOR, tracking which full-cloud points survive
    sel = np.nonzero(fg)[0]
    xc = (xs[sel] - CX) * d[sel] / FX
    yc = (ys[sel] - CY) * d[sel] / FY
    zc = d[sel]
    pts = np.stack([xc, yc, zc], axis=-1)
    keep = _sor_mask(pts, m=sor_m)
    pts = pts[keep]
    sel = sel[keep]
    if len(pts) < 16:
        pts = np.stack([(xs[fg] - CX) * d[fg] / FX,
                        (ys[fg] - CY) * d[fg] / FY, d[fg]], axis=-1)
        sel = np.nonzero(fg)[0]

    near = np.zeros(n, dtype=bool)
    near[sel] = True

    full = np.stack([(xs - CX) * d / FX, (ys - CY) * d / FY, d], axis=-1)
    return {"template": fps(pts, k), "full_cloud": full, "full_depth": d,
            "near": near, "px": xs, "py": ys}


# --------------------------------------------------------------------------- #
# 2D open-vocab detector: YOLO-World (ultralytics) or OWLv2 (transformers)
# --------------------------------------------------------------------------- #
def build_owlv2(model_dir, device):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    processor = Owlv2Processor.from_pretrained(model_dir)
    model = Owlv2ForObjectDetection.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def run_owlv2(processor, model, image_np, queries, box_threshold, device):
    """Returns (boxes (n,4) xyxy px, scores (n,), labels (n,) int index into queries)."""
    inputs = processor(images=image_np, text=[queries], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    results = processor.image_processor.post_process_object_detection(
        outputs, threshold=box_threshold,
        target_sizes=torch.tensor([image_np.shape[:2]]),
    )
    res = results[0]
    boxes = res["boxes"].cpu().numpy()       # (n,4) xyxy in pixels
    scores = res["scores"].cpu().numpy()     # (n,)
    labels = res["labels"].cpu().numpy()     # (n,) int
    return boxes, scores, labels


def build_yoloworld(weights_path, device):
    from ultralytics import YOLOWorld
    model = YOLOWorld(weights_path)
    model.to(device)
    return model


def run_yoloworld(model, image_np, box_threshold, imgsz):
    """Returns (boxes (n,4) xyxy px, scores (n,), labels (n,) int index into classes).

    model.set_classes(...) must have been called once before inference.
    ultralytics expects a BGR numpy array for numpy input.
    """
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = model.predict(image_bgr, conf=box_threshold, imgsz=imgsz, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(np.int64)
    return boxes, scores, labels


# --------------------------------------------------------------------------- #
# SAM2 mask refinement (prompted with the detector boxes)
# --------------------------------------------------------------------------- #
def resolve_sam2_config(ckpt_path, explicit=None):
    """Return the SAM2 config path from the checkpoint filename
    (sam2.1_hiera_base_plus.pt -> configs/sam2.1/sam2.1_hiera_b+.yaml).

    build_sam2() feeds this to hydra compose() with a pkg://sam2 search path, so
    it must be RELATIVE to the sam2 package, not an absolute path."""
    if explicit:
        return explicit
    base = os.path.basename(ckpt_path).lower()
    if "hiera_large" in base:
        cfg = "sam2.1_hiera_l.yaml"
    elif "hiera_base" in base or "hiera_b" in base:
        cfg = "sam2.1_hiera_b+.yaml"
    elif "hiera_small" in base:
        cfg = "sam2.1_hiera_s.yaml"
    else:
        cfg = "sam2.1_hiera_t.yaml"
    return os.path.join("configs", "sam2.1", cfg)


def build_sam2(ckpt_path, config_file, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(config_file, ckpt_path, device=device)
    return SAM2ImagePredictor(model)


@torch.no_grad()
def run_sam2(predictor, image_np, boxes_xyxy):
    """Segment each detection box -> masks (n, H, W) bool in original image coords."""
    predictor.set_image(image_np)            # RGB numpy
    n = len(boxes_xyxy)
    if n == 0:
        return np.zeros((0,) + image_np.shape[:2], dtype=bool)
    boxes_np = np.asarray(boxes_xyxy, dtype=np.float32).reshape(n, 4)
    masks, _, _ = predictor.predict(box=boxes_np, multimask_output=False)
    return np.asarray(masks, dtype=bool)


# --------------------------------------------------------------------------- #
# Wire detection (thin white lines)
# --------------------------------------------------------------------------- #
def detect_wires(image_rgb, bright_thresh=200, sat_thresh=60, min_area=200,
                 min_linearity=15.0, dilate=3):
    """Detect thin white wire/cable regions via colour threshold + 2D linearity.

    Open-vocab detectors (YOLO-World / OWLv2) miss thin wires, but the wires are
    bright white elongated structures. Threshold white pixels, keep connected
    components that form a thin straight line, and return each as
    (mask (H,W) bool, box [x1,y1,x2,y2]).

    Line-ness is measured by the PCA eigenvalue ratio of the component's pixel
    coordinates (lambda_max/lambda_min), NOT the bounding-box aspect ratio: a
    diagonal line's bbox is near-square (aspect ~1.5) and would be wrongly
    rejected, but its pixels are strongly elongated along one axis regardless of
    orientation.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    white = (gray > bright_thresh) & (hsv[:, :, 1] < sat_thresh)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    wires = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        if len(xs) < 4:
            continue
        c = np.stack([xs, ys], axis=1).astype(np.float64)
        c -= c.mean(axis=0)
        cov = c.T @ c / len(c)
        eigs = np.linalg.eigvalsh(cov)          # ascending: [lambda_min, lambda_max]
        lin = (eigs[1] + 1e-6) / (eigs[0] + 1e-6)
        if lin < min_linearity:
            continue
        mask = (labels == i)
        if dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (dilate * 2 + 1, dilate * 2 + 1))
            mask = cv2.dilate(mask.astype(np.uint8), k) > 0
        wires.append((mask, [int(x), int(y), int(x + w), int(y + h)]))
    return wires


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_points_ply(path, points, colors):
    """Point cloud ONLY (no faces). MeshLab renders a face-less PLY as points by
    default, so this is what makes the points actually visible."""
    n = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def _rod_verts_faces(a, b, thickness):
    """Square-cross-section tube from a to b -> (verts (8,3), faces (8,3) int).

    Used to render a box as a *wireframe*: each of the 12 edges becomes a thin
    solid rod instead of a solid face, so the box faces stay transparent."""
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return None
    z = d / L
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, ref)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    h = thickness / 2.0
    # 4 cross-section corners in loop order (so consecutive pairs form a side face)
    off = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64) * h
    corner = np.array([x * ou + y * ov for ou, ov in off])   # (4,3) offsets
    verts = np.vstack([a + corner, b + corner])              # 0..3 at a, 4..7 at b
    faces = []
    for k in range(4):
        k1 = (k + 1) % 4
        faces.append([k, k1, 4 + k1])
        faces.append([k, 4 + k1, 4 + k])
    return verts, np.asarray(faces, dtype=np.int64)


def write_box_ply(path, boxes, thickness=0.03):
    """Oriented box(es) as *wireframes*: 12 thin square-cross-section rods per
    box, no filled faces, coloured per class (boxes carry a 4th RGB element).
    Solid rods are used instead of per-face alpha because MeshLab renders PLY
    triangle alpha unreliably, whereas thin opaque rods are visible from every
    angle and leave the faces transparent."""
    all_verts, vert_colors = [], []
    all_faces, face_colors = [], []
    n_vert = 0
    for box in boxes:
        center, extents, R = box[:3]
        rgb = box[3] if len(box) > 3 else RGB_PERSON
        corners = box_corners(center, extents, R)
        for i, j in BOX_EDGES:
            rod = _rod_verts_faces(corners[i], corners[j], thickness)
            if rod is None:
                continue
            verts, faces = rod
            all_verts.append(verts)
            vert_colors.append(np.tile(np.asarray(rgb, np.uint8), (len(verts), 1)))
            all_faces.append(faces + n_vert)
            face_colors.append(np.tile(np.asarray(rgb, np.uint8), (len(faces), 1)))
            n_vert += len(verts)
    all_verts = np.vstack(all_verts) if all_verts else np.zeros((0, 3))
    vert_colors = np.vstack(vert_colors) if vert_colors else np.zeros((0, 3), np.uint8)
    all_faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), np.int64)
    face_colors = np.vstack(face_colors) if face_colors else np.zeros((0, 3), np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(all_verts, vert_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for face, c in zip(all_faces, face_colors):
            f.write(f"3 {face[0]} {face[1]} {face[2]} {c[0]} {c[1]} {c[2]}\n")


def write_2d(path, image_rgb, dets, masks=None):
    """Annotated 2D image: (translucent mask) + green box + 'label score d=…m' text."""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if masks is not None:
        overlay = img.copy()
        for m in masks:
            overlay[m] = (0, 200, 0)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = tuple(d.get("color", BGR_PERSON))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if "distance" in d:
            label = f"{d['label']} {d['score']:.2f} d={d['distance']:.2f}m"
        else:
            label = f"{d['label']} {d['score']:.2f} ({d.get('n_points',0)}pts)"
        cv2.putText(img, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(path, img)


def write_projection(path, image_rgb, u, v, valid, boxed_person, boxed_wire,
                     boxes, wire_boxes):
    """Projection overlay: every projected point faint gray, selected points
    bright (person green, wire orange), plus per-class 2D boxes. Shows where
    each object's points land on the image."""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    if valid.any():
        uu = np.floor(u[valid]).astype(int)
        vv = np.floor(v[valid]).astype(int)
        for x, y in zip(uu, vv):
            cv2.circle(img, (int(x), int(y)), 1, (90, 90, 90), -1)
    for bm, bgr in ((boxed_person, (0, 255, 0)), (boxed_wire, (0, 165, 255))):
        sel = valid & bm
        if sel.any():
            uu = np.floor(u[sel]).astype(int)
            vv = np.floor(v[sel]).astype(int)
            for x, y in zip(uu, vv):
                cv2.circle(img, (int(x), int(y)), 4, bgr, -1)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for b in wire_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(path, img)


_CLUSTER_PALETTE = np.array([
    [0, 114, 189], [217, 83, 25], [237, 177, 32], [126, 47, 142],
    [119, 172, 48], [162, 20, 47], [76, 190, 199], [128, 128, 128],
], dtype=np.uint8)


def _depth_jet(d, vmin, vmax):
    """Map depth in [vmin, vmax] to red(near) -> blue(far) RGB uint8 colours."""
    t = np.clip((d - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def write_match_viz(out_dir, stem, k, md, radar_cam, labels, nc, chosen,
                    scores, image_np, mask):
    """Dump the shape-matching stages for one person detection as PLYs + a 2D
    overlay + a score log. Everything camera-frame so template & radar overlay.

    Stages: (1) full VGGT mask cloud, depth-coloured; (2) near-mode template;
    (3) radar frustum clusters, coloured per cluster; (4) the chosen cluster;
    (5) per-cluster Chamfer scores; (6) 2D overlay of near-mode pixels + clusters.
    """
    full, d = md["full_cloud"], md["full_depth"]
    tpl = md["template"]
    near, px, py = md["near"], md["px"], md["py"]
    tag = f"{stem}_p{k}"

    # (1) full VGGT mask cloud, near=red -> far=blue
    write_points_ply(os.path.join(out_dir, tag + "_vggt_cloud.ply"), full,
                     _depth_jet(d, np.percentile(d, 5), np.percentile(d, 95)))
    # (2) foreground cloud after Otsu split + opening + SOR (dense, whole body,
    #     no tails) — the raw source of the template before FPS
    write_points_ply(os.path.join(out_dir, tag + "_foreground.ply"), full[near],
                     np.tile(np.array([0, 255, 255], np.uint8), (int(near.sum()), 1)))
    # (2b) FPS-downsampled template (what actually enters the Chamfer match)
    write_points_ply(os.path.join(out_dir, tag + "_template.ply"), tpl,
                     np.tile(np.array([0, 255, 255], np.uint8), (len(tpl), 1)))
    # (3) radar frustum clusters, one colour per cluster (noise points grey)
    pal = np.vstack([_CLUSTER_PALETTE, np.array([[128, 128, 128]], np.uint8)])
    lidx = np.where(labels >= 0, labels, len(_CLUSTER_PALETTE))
    write_points_ply(os.path.join(out_dir, tag + "_radar_clusters.ply"), radar_cam,
                     pal[lidx % len(pal)])
    # (4) chosen cluster (green)
    if chosen is not None:
        m = labels == chosen
        write_points_ply(os.path.join(out_dir, tag + "_selected.ply"), radar_cam[m],
                         np.tile(np.array([0, 255, 0], np.uint8), (int(m.sum()), 1)))
    # (5) score log
    with open(os.path.join(out_dir, tag + "_match.txt"), "w") as f:
        f.write(f"# VGGT shape-template matching, person detection {k}\n")
        f.write(f"template points = {len(tpl)} (nearest depth mode, FPS)\n")
        f.write(f"radar clusters = {nc}\n")
        for c in range(nc):
            m = labels == c
            if c in scores:
                f.write(f"cluster {c}: n={int(m.sum())} "
                        f"cam-z~{float(radar_cam[m][:, 2].mean()):.2f}m "
                        f"chamfer={scores[c]:.4f}{'  <-- chosen' if c == chosen else ''}\n")
            else:
                f.write(f"cluster {c}: n={int(m.sum())} skipped (< min_points)\n")
    # (6) 2D overlay: mask (faint green), near-mode pixels (yellow), radar
    #     cluster points (big, cluster-coloured) projected back onto the image
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR).copy()
    ov = img.copy()
    ov[mask] = (0, 200, 0)
    img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    for x, y in zip(px[near], py[near]):
        cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), -1)
    uu = FX * radar_cam[:, 0] / radar_cam[:, 2] + CX
    vv = FY * radar_cam[:, 1] / radar_cam[:, 2] + CY
    for x, y, cc in zip(uu, vv, pal[lidx % len(pal)]):
        cv2.circle(img, (int(x), int(y)), 4, (int(cc[0]), int(cc[1]), int(cc[2])), -1)
    cv2.imwrite(os.path.join(out_dir, tag + "_match_2d.jpg"), img)


def write_vggt_full_cloud(out_dir, stem, vggt_depth, scale, T_lidar_to_cam,
                          image_np, stride=2):
    """Full-frame VGGT point cloud, coloured by the image RGB.

    Unprojects every valid pixel of the canonical depth map into the camera
    frame, scales to metric by `scale`, then transforms camera->lidar so it
    overlays the radar `<stem>.ply` (which is in the lidar frame). Writes
    `<stem>_vggt_full.ply`. `stride` subsamples both axes (2 = every other
    pixel, 4x fewer points) to keep the ASCII PLY a manageable size.
    """
    H, W = vggt_depth.shape
    d = vggt_depth[::stride, ::stride]
    img = image_np[::stride, ::stride]
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    ok = np.isfinite(d) & (d > 0.1)
    if int(ok.sum()) < 16:
        return
    uu, vv, d = uu[ok], vv[ok], d[ok]
    xc = (uu - CX) * d / FX
    yc = (vv - CY) * d / FY
    zc = d
    if scale is not None:
        xc, yc, zc = xc * scale, yc * scale, zc * scale
    p_cam = np.stack([xc, yc, zc], axis=-1)                    # (N,3) metric camera
    ph = np.concatenate([p_cam, np.ones((len(p_cam), 1))], axis=1)
    p_lid = (np.linalg.inv(T_lidar_to_cam) @ ph.T).T[:, :3]   # (N,3) lidar frame
    colors = img[ok].reshape(-1, 3)                            # (N,3) uint8 RGB
    write_points_ply(os.path.join(out_dir, stem + "_vggt_full.ply"), p_lid, colors)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--categories", default="person")
    p.add_argument("--out_dir", default="viz")
    p.add_argument("--frames", default="all",
                   help="'all' or comma-separated stems (e.g. img_1700603346.892936)")
    p.add_argument("--detector", choices=["yoloworld", "owlv2"], default="yoloworld")
    p.add_argument("--model_dir", default=None,
                   help="dir with OWLv2 weights (default: <script>/weights/owlv2)")
    p.add_argument("--yolo_weights", default=None,
                   help="YOLO-World .pt (default: <script>/weights/yoloworld/yolov8m-worldv2.pt)")
    p.add_argument("--sam2", action=argparse.BooleanOptionalAction, default=True,
                   help="refine detections with SAM2 masks (default on)")
    p.add_argument("--sam2_ckpt", default=None,
                   help="SAM2 checkpoint (default: <script>/weights/sam2/sam2.1_hiera_base_plus.pt)")
    p.add_argument("--sam2_config", default=None,
                   help="SAM2 config yaml (default: auto from checkpoint name)")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO-World inference size")
    p.add_argument("--box_threshold", type=float, default=0.1)
    p.add_argument("--min_dist", type=float, default=0.5,
                   help="drop points closer than this (m) in camera frame")
    p.add_argument("--min_points", type=int, default=10,
                   help="skip box fit when fewer lidar points fall in a box")
    p.add_argument("--dbscan_eps", type=float, default=1.0,
                   help="DBSCAN neighbour radius (m) for tightening the box")
    p.add_argument("--dbscan_min_samples", type=int, default=3,
                   help="DBSCAN min neighbours to be a core point")
    p.add_argument("--box_thickness", type=float, default=0.03,
                   help="cross-section width (m) of the wireframe box edges")
    p.add_argument("--detect_wires", action=argparse.BooleanOptionalAction, default=True,
                   help="also detect thin white wires (default on)")
    p.add_argument("--wire_bright", type=int, default=200,
                   help="white-pixel brightness threshold for wires")
    p.add_argument("--wire_sat", type=int, default=60,
                   help="white-pixel max saturation for wires")
    p.add_argument("--wire_min_area", type=int, default=200,
                   help="min white-component area (px) for a wire")
    p.add_argument("--wire_lin_2d", type=float, default=15.0,
                   help="min 2D linearity (PCA eigenvalue ratio of white-component "
                        "pixels) to count as a wire; orientation-independent so it "
                        "catches diagonal wires, not just vertical/horizontal")
    p.add_argument("--wire_dilate", type=int, default=3,
                   help="dilate wire mask by this many px before point selection")
    p.add_argument("--wire_min_points", type=int, default=4,
                   help="min lidar points inside a wire to fit its 3D box")
    p.add_argument("--wire_min_linearity", type=float, default=2.0,
                   help="a wire must be a thin 3D line: longest extent >= this "
                        "times the 2nd-longest (rejects planar/blob false positives)")
    p.add_argument("--vggt_depth_dir", default="",
                   help="dir with cached VGGT single-image depth (<stem>.npz, "
                        "canonical units); empty string disables the depth prior. "
                        "NOTE: single-frame VGGT depth is too noisy on the "
                        "near-stationary frames to improve point selection (see "
                        "VGGTFusion/README.md) — currently neutral vs nearest_cluster.")
    p.add_argument("--depth_tol", type=float, default=0.6,
                   help="absolute tolerance (m) for the depth-consistency filter")
    p.add_argument("--vggt_match", action=argparse.BooleanOptionalAction, default=False,
                   help="use VGGT shape-template matching (per-cluster shape Chamfer) "
                        "to pick the target cluster; requires --vggt_depth_dir")
    p.add_argument("--match_k", type=int, default=128,
                   help="template size (FPS downsample) for VGGT shape matching")
    p.add_argument("--match_viz", action=argparse.BooleanOptionalAction, default=True,
                   help="dump per-stage PLYs + 2D overlay + score log for matching "
                        "(default on; only written when --vggt_match engages)")
    p.add_argument("--tmpl_open", type=int, default=2,
                   help="template morphological-opening radius (px) to strip the "
                        "depth-edge tail rim (0 = off)")
    p.add_argument("--tmpl_sor_m", type=float, default=2.0,
                   help="template SOR aggressiveness: drop points whose mean k-NN "
                        "distance exceeds m std above the global mean (lower = more)")
    p.add_argument("--vggt_stride", type=int, default=2,
                   help="subsample both axes of the full-frame VGGT point cloud "
                        "dump (<stem>_vggt_full.ply) by this stride (1 = full res)")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # --- detector ---
    if args.detector == "yoloworld":
        yolo_path = args.yolo_weights or os.path.join(
            script_dir, "weights", "yoloworld", "yolov8m-worldv2.pt")
        if not os.path.exists(yolo_path):
            raise SystemExit(f"YOLO-World weights not found at {yolo_path}; run download_weights.py")
        print(f"[model] loading YOLO-World from {yolo_path} on {device} ...", flush=True)
        det_model = build_yoloworld(yolo_path, device)
        det_model.set_classes(categories)
        run_detector = lambda img, th: run_yoloworld(det_model, img, th, args.imgsz)
    else:
        model_dir = args.model_dir or os.path.join(script_dir, "weights", "owlv2")
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            raise SystemExit(f"OWLv2 weights not found at {model_dir}; run download_weights.py")
        print(f"[model] loading OWLv2 from {model_dir} on {device} ...", flush=True)
        processor, owl_model = build_owlv2(model_dir, device)
        queries = [f"a photo of a {c}" for c in categories]
        run_detector = lambda img, th: run_owlv2(processor, owl_model, img, queries, th, device)

    # --- SAM2 (optional mask refinement) ---
    sam2_predictor = None
    if args.sam2:
        ckpt = args.sam2_ckpt or os.path.join(
            script_dir, "weights", "sam2", "sam2.1_hiera_base_plus.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"SAM2 checkpoint not found at {ckpt}; run download_weights.py")
        cfg = resolve_sam2_config(ckpt, args.sam2_config)
        print(f"[model] loading SAM2 from {ckpt} on {device} ...", flush=True)
        sam2_predictor = build_sam2(ckpt, cfg, device)

    img_dir = os.path.join(args.data_dir, "images")
    if args.frames == "all":
        stems = sorted(os.path.splitext(os.path.basename(f))[0]
                       for f in glob.glob(os.path.join(img_dir, "*.jpg")))
    else:
        stems = [s.strip() for s in args.frames.split(",") if s.strip()]

    all_results = {}
    for stem in stems:
        img_path = os.path.join(img_dir, stem + ".jpg")
        lidar_path = os.path.join(args.data_dir, "lidar", stem + ".bin")
        ext_path = os.path.join(args.data_dir, "extrinsics", stem + ".txt")
        if not (os.path.exists(img_path) and os.path.exists(lidar_path)
                and os.path.exists(ext_path)):
            print(f"[skip] {stem}: missing image/lidar/extrinsic", flush=True)
            continue

        out_dir = os.path.join(args.out_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        pil = Image.open(img_path).convert("RGB")
        image_np = np.array(pil)                     # (H,W,3) RGB uint8
        H, W = image_np.shape[:2]
        points = load_lidar(lidar_path)              # (N,3) lidar frame
        T_lidar_to_cam = load_extrinsic(ext_path)    # 4x4
        p_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        p_cam = (T_lidar_to_cam @ p_h.T).T[:, :3]    # (N,3) camera frame

        boxes, scores, labels = run_detector(image_np, args.box_threshold)

        masks = None
        if sam2_predictor is not None and len(boxes) > 0:
            masks = run_sam2(sam2_predictor, image_np, boxes)

        u, v, z = project(p_cam)
        valid = (z > args.min_dist) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        ui = np.floor(u[valid]).astype(int)
        vi = np.floor(v[valid]).astype(int)
        valid_idx = np.nonzero(valid)[0]

        # --- VGGT depth prior: canonical single-image depth -> metric scale ---
        vggt_depth = None
        scale = None
        if args.vggt_depth_dir:
            dpath = os.path.join(args.vggt_depth_dir, stem + ".npz")
            if os.path.exists(dpath):
                vggt_depth = np.load(dpath)["depth"]   # (H,W) float32, NaN where invalid
                scale = vggt_scale(z[valid], vggt_depth, ui, vi)
                print(f"[vggt] {stem}: depth prior on, scale={scale:.2f}"
                      if scale is not None else
                      f"[vggt] {stem}: depth prior loaded but scale unfit -> off")
            else:
                print(f"[vggt] {stem}: no cache at {dpath} -> depth prior off")

        # --- full-frame VGGT point cloud (coloured, metric, lidar frame) ---
        if vggt_depth is not None and scale is not None:
            write_vggt_full_cloud(out_dir, stem, vggt_depth, scale,
                                  T_lidar_to_cam, image_np, args.vggt_stride)

        dets = []
        boxes_3d = []
        boxed_person = np.zeros(len(points), dtype=bool)
        boxed_wire = np.zeros(len(points), dtype=bool)
        for k in range(len(boxes)):
            label = categories[int(labels[k])] if int(labels[k]) < len(categories) else "obj"
            score = float(scores[k])
            det = {"box": boxes[k].tolist(), "score": score, "label": label,
                   "color": BGR_PERSON}

            if masks is not None:
                in_obj = masks[k][vi, ui]            # (num_valid,) bool
            else:
                x1, y1, x2, y2 = boxes[k]
                in_obj = (ui >= x1) & (ui < x2) & (vi >= y1) & (vi < y2)

            idx = valid_idx[in_obj]
            sel = points[idx]                        # lidar-frame points in box/mask
            cam_depth = z[idx]                       # camera forward depth (nearest = object)
            ui_sel = ui[in_obj]
            vi_sel = vi[in_obj]
            det["n_points"] = int(len(idx))
            if len(idx) >= args.min_points:
                if len(sel) > args.min_points:
                    cmask = None
                    # (1) VGGT shape-template matching: build the target's
                    # nearest-depth-mode template from the mask cloud, then pick
                    # the radar cluster whose (scale-invariant) shape matches it
                    # best. More robust than a scalar depth gate because it
                    # estimates scale per object and compares whole shapes.
                    if args.vggt_match and vggt_depth is not None and masks is not None:
                        md = vggt_target_template(vggt_depth, masks[k], args.match_k,
                                                  args.tmpl_open, args.tmpl_sor_m)
                        if md is not None:
                            tpl = md["template"]
                            radar_cam = p_cam[idx]
                            labels, nc = _dbscan_labels(radar_cam, args.dbscan_eps,
                                                        args.dbscan_min_samples)
                            if nc == 0:
                                labels = np.zeros(len(idx), dtype=np.int64)
                                nc = 1
                            best_c, best_sc, scores = None, None, {}
                            for c in range(nc):
                                m = labels == c
                                if int(m.sum()) < args.min_points:
                                    continue
                                sc = shape_chamfer(tpl, radar_cam[m])
                                scores[c] = sc
                                if best_sc is None or sc < best_sc:
                                    best_sc, best_c = sc, c
                            if best_c is not None:
                                cmask = labels == best_c
                            if args.match_viz:
                                write_match_viz(out_dir, stem, k, md, radar_cam,
                                                labels, nc, best_c, scores,
                                                image_np, masks[k])
                    # (2) depth-consistency filter (fallback)
                    if cmask is None and scale is not None and vggt_depth is not None \
                            and masks is not None:
                        d_t = vggt_target_depth(vggt_depth, masks[k])
                        if d_t is not None:
                            vd = vggt_depth[vi_sel, ui_sel] * scale
                            cmask = np.abs(cam_depth - vd) <= args.depth_tol
                            if int(cmask.sum()) < args.min_points:
                                cmask = None       # too few survive -> fall back
                    # (3) nearest 3D cluster (baseline)
                    if cmask is None:
                        cmask = nearest_cluster(sel, cam_depth,
                                                args.dbscan_eps, args.dbscan_min_samples)
                    if int(cmask.sum()) >= args.min_points:
                        sel = sel[cmask]
                        idx = idx[cmask]
                center, extents, R = fit_box(sel)
                det["center"] = center.tolist()
                det["extents"] = extents.tolist()
                det["distance"] = float(np.linalg.norm(center))
                det["height_z"] = float(sel[:, 2].max() - sel[:, 2].min())
                det["n_cluster"] = int(len(sel))
                boxes_3d.append((center, extents, R, RGB_PERSON))
                boxed_person[idx] = True
            else:
                det["skipped"] = "too_few_points"
            dets.append(det)

        # --- wire detections (thin white lines; open-vocab detectors miss them) ---
        wire_boxes = []
        if args.detect_wires:
            for wmask, wbox in detect_wires(image_np, args.wire_bright, args.wire_sat,
                                            args.wire_min_area, args.wire_lin_2d,
                                            args.wire_dilate):
                in_wire = wmask[vi, ui]
                idx = valid_idx[in_wire]
                if len(idx) < args.wire_min_points:
                    continue            # false positive: no radar return on this line
                sel = points[idx]
                # Depth-consistency filter (best-effort). VGGT can't resolve a
                # sub-pixel wire, so its per-pixel depth is the wall/ceiling
                # BEHIND the wire — the filter can therefore wrongly delete a real
                # wire segment. Trust it only when it refines WITHOUT removing a
                # whole depth cluster (a vanished cluster means it ate a surface).
                if scale is not None and vggt_depth is not None:
                    d_t = vggt_target_depth(vggt_depth, wmask)
                    if d_t is not None:
                        vd = vggt_depth[vi[in_wire], ui[in_wire]] * scale
                        keep = np.abs(z[idx] - vd) <= args.depth_tol
                        if int(keep.sum()) >= args.wire_min_points:
                            sel_f = sel[keep]
                            idx_f = idx[keep]
                            lab_f, n_f = _dbscan_labels(sel_f, args.dbscan_eps,
                                                        args.dbscan_min_samples)
                            lab_u, n_u = _dbscan_labels(sel, args.dbscan_eps,
                                                        args.dbscan_min_samples)
                            if n_f >= n_u:
                                sel, idx = sel_f, idx_f
                # A 2D line projects to a whole frustum, so points at several
                # depths can fall in one wire mask. Cluster them in 3D and emit one
                # box per cluster, so each box sits ON its own depth layer instead
                # of one box spanning every depth and landing behind the true wire.
                labels, n_clusters = _dbscan_labels(sel, args.dbscan_eps,
                                                    args.dbscan_min_samples)
                if n_clusters == 0:
                    labels = np.zeros(len(sel), dtype=np.int64)
                    n_clusters = 1
                for c in range(n_clusters):
                    m = labels == c
                    if int(m.sum()) < args.wire_min_points:
                        continue
                    sc = sel[m]
                    center, extents, R = fit_box(sc)
                    # a wire is a thin 3D line (longest extent >> 2nd-longest)
                    if extents[0] < args.wire_min_linearity * extents[1]:
                        continue
                    det = {"box": wbox, "score": 1.0, "label": "wire",
                           "color": BGR_WIRE, "n_points": int(m.sum()),
                           "center": center.tolist(),
                           "extents": extents.tolist(),
                           "distance": float(np.linalg.norm(center)),
                           "height_z": float(sc[:, 2].max() - sc[:, 2].min()),
                           "n_cluster": int(m.sum())}
                    boxes_3d.append((center, extents, R, RGB_WIRE))
                    boxed_wire[idx[m]] = True
                    dets.append(det)
                    wire_boxes.append(wbox)

        colors = np.full((len(points), 3), 150, dtype=np.uint8)
        colors[boxed_person] = PTS_PERSON
        colors[boxed_wire] = PTS_WIRE

        write_points_ply(os.path.join(out_dir, stem + ".ply"), points, colors)
        if boxes_3d:
            write_box_ply(os.path.join(out_dir, stem + "_box.ply"), boxes_3d,
                          thickness=args.box_thickness)
        write_2d(os.path.join(out_dir, stem + "_2d.jpg"), image_np, dets, masks)
        write_projection(os.path.join(out_dir, stem + "_proj.jpg"),
                         image_np, u, v, valid, boxed_person, boxed_wire,
                         boxes, wire_boxes)
        with open(os.path.join(out_dir, stem + "_box.txt"), "w") as f:
            for d in dets:
                if "center" in d:
                    f.write(
                        f"{d['label']} score={d['score']:.3f} distance={d['distance']:.3f}m "
                        f"center={[round(c,3) for c in d['center']]} "
                        f"extents={[round(e,3) for e in d['extents']]} "
                        f"height_z={d['height_z']:.3f}\n")
                else:
                    f.write(f"{d['label']} score={d['score']:.3f} {d.get('skipped','')} "
                            f"n_points={d.get('n_points',0)}\n")

        summary = []
        for d in dets:
            s = {"label": d["label"], "score": round(d["score"], 4)}
            if "center" in d:
                s.update({"center": [round(c, 3) for c in d["center"]],
                          "extents": [round(e, 3) for e in d["extents"]],
                          "distance_m": round(d["distance"], 3),
                          "n_points": d.get("n_points", 0)})
            else:
                s.update({"skipped": d.get("skipped"), "n_points": d.get("n_points", 0)})
            summary.append(s)
        all_results[stem] = summary
        print(f"[{stem}] {len(boxes_3d)} 3D box(es) / {len(dets)} det(s) "
              f"(person {len(boxes)}, wire {len(wire_boxes)}; points {len(points)})",
              flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[done] wrote {len(all_results)} frames -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
