#!/usr/bin/env python3
"""Hand-annotated wire -> 3D detection via the person-style flow.

Takes a user-drawn red wire mask on an image, then for each connected wire
component runs the SAME flow used for the person:

    back-project radar points into the wire-mask frustum
      -> density-based clustering (DBSCAN, `_dbscan_labels`)
      -> VGGT shape-template matching to pick the wire cluster
      -> PCA oriented 3D box on the chosen cluster

This isolates the 3D stage: given a *clean* 2D wire segmentation (hand label),
can the pipeline complete 3D wire detection? (The existing `detect_wires` uses
white-line segmentation which brings its own false-positive/miss errors.)

The annotated image is 1920x1080; the pipeline image is 1280x720 undistorted,
so the red mask is resize+undistorted to match (same chain as prepare_custom_data).

Outputs per frame (mirrors viz_vggt_full's person files, with a `_w{k}_` prefix):
  <stem>.ply / _box.ply / _2d.jpg / _box.txt / results.json
  <stem>_w{k}_vggt_cloud.ply / _foreground.ply / _template.ply
         / _radar_clusters.ply / _selected.ply / _match.txt / _match_2d.jpg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_vocab_3d_detect as ovd
from open_vocab_3d_detect import (
    FX, FY, CX, CY, RGB_WIRE,
    load_lidar, load_extrinsic, project, fit_box,
    _dbscan_labels, vggt_scale, vggt_target_template, shape_chamfer,
    write_points_ply, write_box_ply, _depth_jet, _CLUSTER_PALETTE,
)

# Raw (distorted) 1280x720 intrinsics + distortion from calib_webcam_1280x720.yaml.
FX_R, FY_R, CX_R, CY_R = (655.77081890212833, 669.42189140913456,
                          623.67928178114516, 540.10827477395355)
DIST = np.array([-0.21726910255312651, 0.042236963397948302, -0.048378312076898584,
                 0.0082375715658483048, 0.026781005616867786])
K_RAW = np.array([[FX_R, 0.0, CX_R], [0.0, FY_R, CY_R], [0.0, 0.0, 1.0]])
NEW_K, _ = cv2.getOptimalNewCameraMatrix(K_RAW, DIST, (1280, 720), alpha=0.0)


def extract_wire_masks(annotated_path, min_area=20, dilate=3):
    """Red pixels in the 1920x1080 annotated image -> per-wire masks @1280x720
    (resize + undistort to match the pipeline image). Returns list of bool masks.
    """
    img = cv2.imread(annotated_path, cv2.IMREAD_COLOR)  # BGR 1920x1080
    b = img[:, :, 0].astype(int)
    g = img[:, :, 1].astype(int)
    r = img[:, :, 2].astype(int)
    red = (r > 150) & (r > 2 * g + 30) & (r > 2 * b + 30)
    red = red.astype(np.uint8)
    red = cv2.resize(red, (1280, 720), interpolation=cv2.INTER_NEAREST)
    red = cv2.undistort(red, K_RAW, DIST, None, NEW_K)
    red = red > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    masks = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        m = labels == i
        if dilate > 0:
            m = cv2.dilate(m.astype(np.uint8),
                           np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)).astype(bool)
        masks.append(m)
    return masks


def write_wire_viz(out_dir, stem, k, md, radar_cam, labels, nc, chosen, scores,
                   image_np, mask):
    """Mirror of write_match_viz for one wire detection (camera-frame PLYs)."""
    full, d = md["full_cloud"], md["full_depth"]
    tpl = md["template"]
    near, px, py = md["near"], md["px"], md["py"]
    tag = f"{stem}_w{k}"

    write_points_ply(os.path.join(out_dir, tag + "_vggt_cloud.ply"), full,
                     _depth_jet(d, np.percentile(d, 5), np.percentile(d, 95)))
    write_points_ply(os.path.join(out_dir, tag + "_foreground.ply"), full[near],
                     np.tile(np.array([0, 255, 255], np.uint8), (int(near.sum()), 1)))
    write_points_ply(os.path.join(out_dir, tag + "_template.ply"), tpl,
                     np.tile(np.array([0, 255, 255], np.uint8), (len(tpl), 1)))

    pal = np.vstack([_CLUSTER_PALETTE, np.array([[128, 128, 128]], np.uint8)])
    lidx = np.where(labels >= 0, labels, len(_CLUSTER_PALETTE))
    write_points_ply(os.path.join(out_dir, tag + "_radar_clusters.ply"), radar_cam,
                     pal[lidx % len(pal)])
    if chosen is not None:
        m = labels == chosen
        write_points_ply(os.path.join(out_dir, tag + "_selected.ply"), radar_cam[m],
                         np.tile(np.array([0, 255, 0], np.uint8), (int(m.sum()), 1)))

    with open(os.path.join(out_dir, tag + "_match.txt"), "w") as f:
        f.write(f"# VGGT shape-template matching, wire {k}\n")
        f.write(f"template points = {len(tpl)} (VGGT depth-mode, FPS)\n")
        f.write(f"radar clusters = {nc}\n")
        for c in range(nc):
            m = labels == c
            if c in scores:
                f.write(f"cluster {c}: n={int(m.sum())} "
                        f"cam-z~{float(radar_cam[m][:, 2].mean()):.2f}m "
                        f"chamfer={scores[c]:.4f}{'  <-- chosen' if c == chosen else ''}\n")
            else:
                f.write(f"cluster {c}: n={int(m.sum())} skipped (< min_points)\n")

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", required=True,
                    help="path to the *_annotateWire.jpg (red wire mask, 1920x1080)")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--data_dir", default=ovd.DEFAULT_DATA_DIR)
    ap.add_argument("--vggt_depth_dir", default="",
                    help="dir with cached VGGT depth <stem>.npz")
    ap.add_argument("--out_dir", default="viz_wire_annot")
    ap.add_argument("--eps", type=float, default=1.0, help="DBSCAN radius (m)")
    ap.add_argument("--min_samples", type=int, default=3, help="DBSCAN core points")
    ap.add_argument("--min_points", type=int, default=4, help="min radar pts in a wire")
    ap.add_argument("--dilate", type=int, default=3, help="dilate wire mask (px)")
    ap.add_argument("--match_k", type=int, default=128, help="template FPS size")
    args = ap.parse_args()

    stem = args.stem
    img_path = os.path.join(args.data_dir, "images", stem + ".jpg")
    lidar_path = os.path.join(args.data_dir, "lidar", stem + ".bin")
    ext_path = os.path.join(args.data_dir, "extrinsics", stem + ".txt")

    masks = extract_wire_masks(args.annotated, dilate=args.dilate)
    print(f"[annot] {len(masks)} wire component(s) extracted from {args.annotated}")

    image_np = np.array(Image.open(img_path).convert("RGB"))  # (720,1280,3) RGB
    points = load_lidar(lidar_path)
    T_lidar_to_cam = load_extrinsic(ext_path)
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T_lidar_to_cam @ ph.T).T[:, :3]
    u, v, z = project(p_cam)
    valid = (z > 0.5) & (u >= 0) & (u < 1279) & (v >= 0) & (v < 719)
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    valid_idx = np.nonzero(valid)[0]

    vggt_depth = None
    scale = None
    if args.vggt_depth_dir:
        dpath = os.path.join(args.vggt_depth_dir, stem + ".npz")
        if os.path.exists(dpath):
            vggt_depth = np.load(dpath)["depth"]
            scale = vggt_scale(z[valid], vggt_depth, ui, vi)
            print(f"[vggt] {stem}: depth on, scale={scale:.2f}" if scale else
                  f"[vggt] {stem}: depth loaded but scale unfit")

    out_dir = os.path.join(args.out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)

    # full radar cloud: grey, wire-selected points highlighted
    all_selected = np.zeros(len(points), dtype=bool)
    boxes_3d = []
    results = []
    dets_for_2d = []

    for k, mask in enumerate(masks):
        in_mask = mask[vi, ui]
        idx = valid_idx[in_mask]
        radar_cam = p_cam[idx]                       # camera-frame frustum points
        sel_lidar = points[idx]                      # lidar-frame frustum points
        if len(idx) < args.min_points:
            print(f"[wire {k}] only {len(idx)} pts in frustum -> skip")
            continue

        # --- density clustering (DBSCAN, person's algorithm) ---
        labels, nc = _dbscan_labels(radar_cam, args.eps, args.min_samples)
        if nc == 0:
            labels = np.zeros(len(idx), dtype=np.int64)
            nc = 1

        # --- VGGT shape-template matching to pick the wire cluster ---
        md = None
        chosen = None
        scores = {}
        if vggt_depth is not None:
            md = vggt_target_template(vggt_depth, mask, args.match_k)
        if md is not None:
            tpl = md["template"]
            best_c, best_sc = None, None
            for c in range(nc):
                m = labels == c
                if int(m.sum()) < args.min_points:
                    continue
                sc = shape_chamfer(tpl, radar_cam[m])
                scores[c] = sc
                if best_sc is None or sc < best_sc:
                    best_sc, best_c = sc, c
            chosen = best_c
            if args.out_dir:
                write_wire_viz(out_dir, stem, k, md, radar_cam, labels, nc,
                               chosen, scores, image_np, mask)

        # --- fit the box on the chosen cluster (fallback: whole frustum) ---
        if chosen is not None and int((labels == chosen).sum()) >= args.min_points:
            pick = labels == chosen
        else:
            pick = np.ones(len(idx), dtype=bool)
        sel = sel_lidar[pick]
        center, extents, R = fit_box(sel)
        dist = float(np.linalg.norm(center))
        height_z = float(sel[:, 2].max() - sel[:, 2].min())
        boxes_3d.append((center, extents, R, RGB_WIRE))
        all_selected[idx[pick]] = True
        results.append({
            "wire": k, "n_points": int(pick.sum()),
            "n_frustum": int(len(idx)),
            "chosen_cluster": chosen,
            "chosen_chamfer": scores.get(chosen) if chosen is not None else None,
            "center": center.tolist(), "extents": extents.tolist(),
            "distance_m": round(dist, 3), "height_z": round(height_z, 3),
        })
        dets_for_2d.append({"wire": k, "center": center.tolist(),
                            "distance": dist, "n_points": int(pick.sum())})
        print(f"[wire {k}] frustum={len(idx)} pts, {nc} cluster(s), "
              f"chosen={chosen} chamfer={scores.get(chosen) if chosen is not None else None}, "
              f"dist={dist:.2f}m extents={extents.round(2)}")

    # ---- frame-level outputs (mirror viz_vggt_full) ----
    colors = np.tile(np.array([128, 128, 128], np.uint8), (len(points), 1))
    colors[all_selected] = (0, 255, 255)
    write_points_ply(os.path.join(out_dir, stem + ".ply"), points, colors)
    if boxes_3d:
        write_box_ply(os.path.join(out_dir, stem + "_box.ply"), boxes_3d,
                      thickness=0.02)

    with open(os.path.join(out_dir, stem + "_box.txt"), "w") as f:
        for r in results:
            f.write(f"wire score=1.000 distance={r['distance_m']}m "
                    f"center={[round(x,3) for x in r['center']]} "
                    f"extents={[round(x,3) for x in r['extents']]} "
                    f"height_z={r['height_z']} n_points={r['n_points']} "
                    f"chamfer={r['chosen_chamfer']}\n")

    # 2D overlay: wire masks + box labels on the pipeline image
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR).copy()
    ov = img.copy()
    for m in masks:
        ov[m] = (0, 200, 0)
    img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
    for r in results:
        c = r["center"]
        # project lidar center back to image for a label anchor
        cc = T_lidar_to_cam @ np.array([c[0], c[1], c[2], 1.0])
        if cc[2] > 0.1:
            uu = FX * cc[0] / cc[2] + CX
            vv = FY * cc[1] / cc[2] + CY
            cv2.putText(img, f"w{r['wire']} {r['distance_m']}m",
                        (int(uu), int(vv)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, stem + "_2d.jpg"), img)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({stem: results}, f, indent=2)
    print(f"[done] {stem}: {len(results)} wire box(es) -> {out_dir}")


if __name__ == "__main__":
    main()
