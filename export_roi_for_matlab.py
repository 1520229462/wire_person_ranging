#!/usr/bin/env python3
"""Export the 3D ROI (2D TTPLA mask -> frustum band -> cropped lidar) for MATLAB.

This is the "project the 2D mask/box into 3D space, then segment the 3D points"
front half of the user's proposed pipeline. It reproduces steps 1-3 of
ttpla_wire_3d.process_frame and dumps the cropped candidate point cloud so that
MATLAB's segmentAerialLidarPowerline (R2023b+, pretrained RandLA-Net) can run on
it WITHOUT re-implementing the projection/extrinsic logic in MATLAB.

Outputs per frame (both in the lidar frame, units = meters):
  {stem}_roi.txt   ASCII "x y z" per line -- read in MATLAB with readmatrix
  {stem}_roi.ply   same points, colored, for MeshLab inspection
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_vocab_3d_detect as ovd
from open_vocab_3d_detect import load_lidar, write_points_ply
from ttpla_wire_3d import run_ttpla_line, skeleton_polyline, DEFAULT_WEIGHTS


def export_roi(model, stem, args):
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
    T_cam_to_lidar = np.loadtxt(ext_path, dtype=np.float64)
    T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)

    # 1-2. TTPLA Line mask -> skeleton -> wide band (tolerates extrinsic error)
    mask, conf = run_ttpla_line(model, img, conf=args.conf)
    if mask is None:
        return {"stem": stem, "status": "no_line", "conf": conf}
    us, vs, ss = skeleton_polyline(mask)
    if us is None:
        return {"stem": stem, "status": "no_skeleton", "conf": conf}

    skel_img = np.zeros(mask.shape, np.uint8)
    skel_img[np.round(vs).astype(int), np.round(us).astype(int)] = 1
    band = cv2.dilate(skel_img, np.ones((2 * args.dilate + 1, 2 * args.dilate + 1),
                                        np.uint8))

    # 3. project lidar -> camera, keep points inside the band
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T_lidar_to_cam @ ph.T).T[:, :3]
    u, v, z = ovd.project(p_cam)
    H, W = mask.shape
    valid = (z > 0.5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    sel = band[vi, ui].astype(bool)

    roi = points[valid][sel]                    # (M,3) lidar frame, meters
    n = len(roi)

    fdir = os.path.join(out_dir, stem)
    os.makedirs(fdir, exist_ok=True)

    # ASCII for MATLAB readmatrix
    with open(os.path.join(fdir, stem + "_roi.txt"), "w") as f:
        f.write("# x y z (lidar frame, meters)  -- ROI = 2D mask band "
                f"(dilate={args.dilate}px) projected into 3D\n")
        for p in roi:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    # colored PLY for inspection (single cyan color = raw ROI, unsegmented)
    write_points_ply(
        os.path.join(fdir, stem + "_roi.ply"), roi,
        np.tile(np.array([255, 200, 0], np.uint8), (max(n, 1), 1)))

    return {"stem": stem, "status": "ok", "conf": conf, "n_roi": n,
            "skeleton_len_px": float(ss[-1]) if len(ss) else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=ovd.DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default="viz_ttpla_wire3d")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--frames", default="all",
                    help="'all' or comma-separated stems")
    ap.add_argument("--dilate", type=int, default=150,
                    help="band half-width (px) around the skeleton")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    if args.frames == "all":
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(args.data_dir, "images", "*.jpg")))
    else:
        stems = [s.strip() for s in args.frames.split(",") if s.strip()]

    for stem in stems:
        r = export_roi(model, stem, args)
        if r["status"] == "ok":
            print(f"[ok]   {stem}: n_roi={r['n_roi']} conf={r['conf']:.2f} "
                  f"skel={r['skeleton_len_px']:.0f}px -> {args.out_dir}/{stem}/")
        else:
            print(f"[skip] {stem}: {r['status']}")


if __name__ == "__main__":
    main()
