#!/usr/bin/env python3
"""测距 (ranging): person (template matching) + wire (catenary) per frame.

Combines the two detection paths built in this project and measures the 3D
distance between them (the "next step" after detection), on every frame of the
(near-static) sequence:

  - Person: YOLO-World box -> SAM2 mask -> VGGT shape-template matching
    (vggt_target_template -> DBSCAN clusters -> per-cluster shape Chamfer ->
    nearest-shape cluster), exactly the `--vggt_match` path of
    open_vocab_3d_detect.py.
  - Wire:   two modes (--wire_mode):
      * per_frame (default): TTPLA 2D line mask -> skeleton -> band -> lidar
        candidates -> PCA local-linearity (k-NN) -> nearest coherent depth
        cluster (gap statistics) -> 3D PCHIP curve. Detected fresh on every
        frame with no absolute depth/height constants (ttpla_wire_3d.py).
      * catenary: TTPLA mask -> skeleton -> band -> lidar candidates ->
        multi-frame accumulation -> RANSAC coarse line -> catenary (sag) fit
        (ttpla_wire_3d_catenary.py). The wire is static, fitted once and
        shared across all frames.

All distances are in the lidar frame (metres):

  person_distance      = ||person_center||            (sensor -> person box centre)
  wire_distance        = min ||wire_curve||           (sensor -> nearest wire point)
  person_wire_distance = min ||wire_curve - person_center||   (person <-> wire)

Outputs:
  <out_dir>/results.json    -- complete per-frame inference results (keyed by stem)
  <out_dir>/summary.csv     -- one row per frame (headline numbers)
  <out_dir>/<stem>/range.json / range.txt / range_2d.jpg / range.ply
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import open_vocab_3d_detect as ovd
import deploy_utils
from open_vocab_3d_detect import (
    load_lidar, project, fit_box, build_yoloworld, run_yoloworld,
    build_sam2, run_sam2, resolve_sam2_config, vggt_scale,
    vggt_target_template, _dbscan_labels, shape_chamfer, nearest_cluster,
    write_points_ply, RGB_WIRE, PTS_PERSON, PTS_WIRE)
from ttpla_wire_3d import (run_ttpla_line, skeleton_polyline, detect_wire_curve,
                           detect_wire_curve_arrays, DEFAULT_WEIGHTS)
from ttpla_wire_3d_catenary import collect_candidates, fit_catenary, _first_depth_gap
from manual_wire_person_ref import (reference_wire, person_points,
                                    accumulate_frames)

DEFAULT_WIRE_CURVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed", "catenary_curve.txt")
DEFAULT_VGGT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vggt_depth")


# --------------------------------------------------------------------------- #
# Person detection (VGGT shape-template matching)
# --------------------------------------------------------------------------- #
def detect_person(image_np, points, p_cam, vggt_depth, ui, vi, valid_idx, z,
                  det_model, sam2_predictor, args):
    """Run YOLO-World + SAM2 + VGGT shape-template matching; return a list of
    person detections (dicts with center/extents/dist/score/sel/idx/box), or [].

    Mirrors open_vocab_3d_detect.py's per-detection loop for the person class,
    with the template-matching path (`--vggt_match`) always on and
    nearest_cluster as the fallback.
    """
    boxes, scores, labels = run_yoloworld(det_model, image_np, args.box_threshold,
                                          args.imgsz)
    if len(boxes) == 0:
        return []
    masks = run_sam2(sam2_predictor, image_np, boxes) if sam2_predictor is not None \
        else np.zeros((0,) + image_np.shape[:2], dtype=bool)

    out = []
    for k in range(len(boxes)):
        if masks is not None and len(masks) > k:
            in_obj = masks[k][vi, ui]
        else:
            x1, y1, x2, y2 = boxes[k]
            in_obj = (ui >= x1) & (ui < x2) & (vi >= y1) & (vi < y2)
        idx = valid_idx[in_obj]
        if len(idx) < args.min_points:
            continue

        sel = points[idx]
        cam_depth = z[idx]
        cmask = None

        # (1) VGGT shape-template matching: build the nearest-depth-mode
        #     template, then pick the radar cluster whose shape matches best.
        if vggt_depth is not None and masks is not None and len(masks) > k:
            md = vggt_target_template(vggt_depth, masks[k], args.match_k,
                                      args.tmpl_open, args.tmpl_sor_m)
            if md is not None:
                tpl = md["template"]
                radar_cam = p_cam[idx]
                lab, nc = _dbscan_labels(radar_cam, args.dbscan_eps,
                                         args.dbscan_min_samples)
                if nc == 0:
                    lab = np.zeros(len(idx), dtype=np.int64)
                    nc = 1
                best_c, best_sc = None, None
                for c in range(nc):
                    m = lab == c
                    if int(m.sum()) < args.min_points:
                        continue
                    sc = shape_chamfer(tpl, radar_cam[m])
                    if best_sc is None or sc < best_sc:
                        best_sc, best_c = sc, c
                if best_c is not None:
                    cmask = lab == best_c

        # (2) nearest 3D cluster (baseline fallback)
        if cmask is None:
            cmask = nearest_cluster(sel, cam_depth, args.dbscan_eps,
                                    args.dbscan_min_samples)
        if int(cmask.sum()) < args.min_points:
            continue

        sel = sel[cmask]
        idx = idx[cmask]
        center, extents, R = fit_box(sel)
        out.append({
            "center": center, "extents": extents, "R": R,
            "dist": float(np.linalg.norm(center)),
            "score": float(scores[k]), "sel": sel, "idx": idx,
            "box": boxes[k],
        })
    return out


# --------------------------------------------------------------------------- #
# Wire geometry (load precomputed catenary, or recompute from frames)
# --------------------------------------------------------------------------- #
def load_wire_curve(path):
    """Read a catenary_curve.txt -> (N,3) lidar-frame curve points."""
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pts.append([float(t) for t in line.split()[:3]])
    return np.asarray(pts, dtype=np.float64)


def recompute_wire_curve(args, model, stems):
    """Accumulate candidates across `stems` -> nearest/elevated prior -> catenary.
    Returns (curve (N,3), info dict)."""
    frames = []
    for stem in stems:
        f, st = collect_candidates(model, stem, args)
        if st == "ok":
            frames.append(f)
    cand = np.concatenate([f["cand_lidar"] for f in frames], axis=0)
    ref = next(f for f in frames if f["stem"] == args.ref_frame) \
        if any(f["stem"] == args.ref_frame for f in frames) else frames[0]
    x_thresh = _first_depth_gap(ref["cand_lidar"][:, 0])
    prior = (cand[:, 0] < x_thresh) & (cand[:, 2] >= args.z_min)
    inl, curve, info = fit_catenary(cand[prior], tol=args.tol,
                                    ransac_tol=args.ransac_tol,
                                    n_iter=args.ransac_iter,
                                    n_curve=args.n_curve)
    if curve is None:
        raise RuntimeError("catenary fit failed")
    info["n_frames"] = len(frames)
    return curve, info


# --------------------------------------------------------------------------- #
# Combined visualization
# --------------------------------------------------------------------------- #
def write_range_viz(out_dir, stem, image_np, person, wire_curve, p_cam, u, v, z,
                    T_lidar_to_cam, points, person_dist, wire_dist, pw_dist,
                    closest_person, closest_wire, ref_dist=None,
                    intrinsics=None):
    os.makedirs(out_dir, exist_ok=True)
    H, W = image_np.shape[:2]

    def _proj(pts_cam):
        if intrinsics is None:
            return project(pts_cam)
        fx, fy, cx, cy = intrinsics
        return project(pts_cam, fx, fy, cx, cy)

    # 2D: reproject the wire curve, the person box, and the closest-approach line
    img = image_np.copy()
    x1, y1, x2, y2 = person["box"].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ch = np.concatenate([wire_curve, np.ones((len(wire_curve), 1))], axis=1)
    cc = (T_lidar_to_cam @ ch.T).T[:, :3]
    cu, cvv, cz = _proj(cc)
    ok = (cz > 0.5) & (cu >= 0) & (cu < W - 1) & (cvv >= 0) & (cvv < H - 1)
    wire_2d = np.stack([cu[ok], cvv[ok]], axis=1).astype(int)
    cv2.polylines(img, [wire_2d], False, (255, 165, 0), 2)
    # detected person surface points (lidar frame) reprojected as green dots
    psel = person["sel"]
    if len(psel):
        psel_h = np.concatenate([psel, np.ones((len(psel), 1))], axis=1)
        psel_c = (T_lidar_to_cam @ psel_h.T).T[:, :3]
        psu, psv, psz = _proj(psel_c)
        pok = (psz > 0.5) & (psu >= 0) & (psu < W - 1) & (psv >= 0) & (psv < H - 1)
        for px, py in zip(psu[pok], psv[pok]):
            cv2.circle(img, (int(px), int(py)), 1, PTS_PERSON, -1)
    # closest-approach line: person surface point <-> wire point
    seg = np.array([closest_person, closest_wire])
    sh = np.concatenate([seg, np.ones((2, 1))], axis=1)
    scc = (T_lidar_to_cam @ sh.T).T[:, :3]
    su, sv, sz = _proj(scc)
    if (sz > 0.5).all():
        pa = (int(su[0]), int(sv[0]))
        pb = (int(su[1]), int(sv[1]))
        cv2.line(img, pa, pb, (0, 0, 255), 2)
        cv2.circle(img, pa, 6, (0, 255, 0), -1)
        cv2.circle(img, pb, 6, (0, 165, 255), -1)
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
        label = f"d={pw_dist:.2f}m" if ref_dist is None else \
            f"d={pw_dist:.2f}m (ref {ref_dist:.2f}m)"
        cv2.putText(img, label, (mid[0] - 90, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, f"person {person_dist:.2f}m", (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"wire {wire_dist:.2f}m", (10, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "range_2d.jpg"), img)

    # PLY: full raw lidar scan (grey) + person points (green) + wire curve (blue)
    # + closest-approach line (red)
    n_line = 200
    t = np.linspace(0, 1, n_line)[:, None]
    line_pts = closest_person[None, :] + t * (closest_wire - closest_person)[None, :]
    # drop the detected person points from the grey background scan: they are a
    # subset of `points`, and leaving the grey copies would overdraw the green
    # person vertices at identical coordinates in the viewer.
    bg = points
    pidx = np.asarray(person.get("idx", []))
    if len(pidx):
        keep = np.ones(len(points), dtype=bool)
        keep[pidx] = False
        bg = points[keep]
    parts = [bg, person["sel"], wire_curve, line_pts]
    cols = [np.tile(np.array([150, 150, 150], np.uint8), (len(bg), 1)),
            np.tile(np.array(PTS_PERSON, np.uint8), (len(person["sel"]), 1)),
            np.tile(np.array(RGB_WIRE, np.uint8), (len(wire_curve), 1)),
            np.tile(np.array([255, 0, 0], np.uint8), (n_line, 1))]
    write_points_ply(os.path.join(out_dir, "range.ply"), np.vstack(parts),
                     np.vstack(cols))


# --------------------------------------------------------------------------- #
# Per-frame processing
# --------------------------------------------------------------------------- #
def process_frame(stem, wire_curve, ttpla, det_model, sam2_predictor, args,
                  script_dir, ref_tree=None):
    """Detect person (template matching) on one frame and range vs the wire.
    In `--wire_mode per_frame` the wire is re-detected on this frame (the
    passed-in `wire_curve` is ignored); in `catenary` mode the shared static
    curve is used. If `ref_tree` is given, the manual reference wire->person
    distance is computed too and written alongside. Writes range.json /
    range.txt / range_2d.jpg / range.ply. Returns a result dict, or None if
    the frame has no data / no person.

    Thin file-loading shell over `process_frame_arrays`: reads
    images/<stem>.jpg, lidar/<stem>.bin, extrinsics/<stem>.txt and (if
    present) vggt_depth/<stem>.npz from `args.data_dir` / `args.vggt_depth_dir`,
    then delegates all numerics. Third-party code with in-memory frames should
    call `process_frame_arrays` directly (see `ranging_api.RangingSession`).
    """
    img_path = os.path.join(args.data_dir, "images", stem + ".jpg")
    lidar_path = os.path.join(args.data_dir, "lidar", stem + ".bin")
    ext_path = os.path.join(args.data_dir, "extrinsics", stem + ".txt")
    if not all(os.path.exists(p) for p in (img_path, lidar_path, ext_path)):
        return None

    from PIL import Image
    image_np = np.array(Image.open(img_path).convert("RGB"))
    H, W = image_np.shape[:2]
    if (H, W) != (720, 1280):
        print(f"[warn] {stem}: image is {W}x{H}, expected 1280x720 -- the "
              f"built-in intrinsics are calibrated for 1280x720; pass "
              f"--intrinsics to override", file=sys.stderr, flush=True)
    points = load_lidar(lidar_path)
    T_lidar_to_cam = ovd.load_extrinsic(ext_path)

    vggt_depth = None
    dpath = os.path.join(args.vggt_depth_dir, stem + ".npz")
    if os.path.exists(dpath):
        vggt_depth = np.load(dpath)["depth"]

    return process_frame_arrays(
        stem, image_np, points, T_lidar_to_cam, wire_curve, ttpla, det_model,
        sam2_predictor, args, script_dir, ref_tree=ref_tree,
        out_dir=os.path.join(args.out_dir, stem),
        intrinsics=getattr(args, "intrinsics", None), vggt_depth=vggt_depth)


def process_frame_arrays(stem, image_np, points, T_lidar_to_cam, wire_curve,
                         ttpla, det_model, sam2_predictor, args, script_dir,
                         ref_tree=None, out_dir=None, intrinsics=None,
                         vggt_depth=None):
    """In-memory variant of `process_frame` (identical numerics): all inputs
    are arrays already in the process, no file reads.

    Args:
        stem:             frame id (used in the result dict / written files).
        image_np:         (H,W,3) uint8 RGB image.
        points:           (N,3) lidar-frame points (float64).
        T_lidar_to_cam:   4x4 transform lidar -> camera frame.
        wire_curve:       (M,3) static wire curve (used in catenary mode).
        ttpla:            TTPLA YOLO model (per_frame wire mode only).
        det_model:        YOLO-World detector with classes already set.
        sam2_predictor:   SAM2 predictor, or None to skip mask refinement.
        args:             argparse.Namespace with all tunables (same names as
                          the CLI; see `ranging_api.RangingSession`).
        script_dir:       repo root (for `build_yoloworld`'s CLIP resolution).
        ref_tree:         optional manual-reference wire KD-tree (scene-specific).
        out_dir:          per-frame output dir; None disables range.json /
                          range.txt / range_2d.jpg / range.ply writes.
        intrinsics:       optional (fx, fy, cx, cy); None uses the built-in
                          calibration of `open_vocab_3d_detect`.
        vggt_depth:       optional (H,W) float32 depth map; None degrades the
                          person branch to the nearest-cluster fallback and
                          sets ``result["degraded"] = True``.

    Returns a result dict (adds ``error`` on failure), same schema as
    `process_frame`'s return value.
    """
    H, W = image_np.shape[:2]
    p_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T_lidar_to_cam @ p_h.T).T[:, :3]

    if intrinsics is None:
        u, v, z = project(p_cam)
    else:
        fx, fy, cx, cy = intrinsics
        u, v, z = project(p_cam, fx, fy, cx, cy)
    valid = (z > args.min_dist) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    valid_idx = np.nonzero(valid)[0]

    if vggt_depth is None:
        print(f"[warn] {stem}: no VGGT depth cache -- person branch degrades "
              f"to the nearest-cluster fallback (result['degraded']=True)",
              file=sys.stderr, flush=True)

    dets = detect_person(image_np, points, p_cam, vggt_depth, ui, vi,
                         valid_idx, z, det_model, sam2_predictor, args)
    if not dets:
        return {"frame": stem, "error": "no_person_detected"}

    # per-frame wire (PCA linearity + nearest coherent depth cluster) if
    # requested; overrides the shared curve. Detected fresh every frame with no
    # absolute depth/height constants.
    wire_mode = args.wire_mode
    if wire_mode == "per_frame":
        curve, winfo = detect_wire_curve_arrays(
            ttpla, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), points,
            T_lidar_to_cam, conf=args.conf, dilate=args.dilate,
            lin_k=args.lin_k, lin_thr=args.lin_thr, gap_kmad=args.gap_kmad,
            min_size=args.min_size, min_points=args.wire_min_points,
            n_sample=args.n_sample, extend_m=args.extend_m,
            extend_to_z=args.extend_to_z, extend_top_k=args.extend_top_k,
            extend_step=args.extend_step, mask_max_area=args.mask_max_area,
            intrinsics=intrinsics, stem=stem)
        if curve is None:
            return {"frame": stem, "error": f"wire_{winfo['status']}"}
        wire_curve = curve

    person = max(dets, key=lambda d: d["score"])
    pc = person["center"]
    person_pts = person["sel"]                 # person surface points (lidar frame)
    person_dist = float(np.linalg.norm(pc))
    wire_d = np.linalg.norm(wire_curve, axis=1)
    wire_dist = float(wire_d.min())
    # wire <-> person closest distance: for each person surface point, distance to
    # the nearest wire point; take the minimum (true closest approach).
    tree = cKDTree(wire_curve)
    d_pw, wi = tree.query(person_pts, k=1)
    j = int(np.argmin(d_pw))
    closest_person = person_pts[j]
    closest_wire = wire_curve[wi[j]]
    pw_dist = float(d_pw[j])

    # manual reference wire -> person distance (independent hand-specified
    # standard; uses the static accumulated reference wire + manual person box)
    ref_dist = None
    if ref_tree is not None:
        pp = person_points(points)
        if len(pp) >= 3:
            d_ref, _ = ref_tree.query(pp, k=1)
            ref_dist = float(d_ref.min())

    if out_dir is not None:
        write_range_viz(out_dir, stem, image_np, person, wire_curve, p_cam, u,
                        v, z, T_lidar_to_cam, points, person_dist, wire_dist,
                        pw_dist, closest_person, closest_wire, ref_dist,
                        intrinsics)

    def r3(a):
        return [round(float(x), 3) for x in a]

    result = {
        "frame": stem,
        "person": {
            "center": r3(pc), "extents": r3(person["extents"]),
            "score": round(person["score"], 4),
            "n_points": int(len(person["sel"])),
            "distance_m": round(person_dist, 3),
        },
        "all_persons": [
            {"center": r3(d["center"]), "score": round(d["score"], 4),
             "distance_m": round(d["dist"], 3), "n_points": int(len(d["sel"]))}
            for d in sorted(dets, key=lambda d: -d["score"])
        ],
        "wire": {
            "mode": wire_mode,
            "endpoints": [r3(wire_curve[0]), r3(wire_curve[-1])],
            "length_m": round(float(np.linalg.norm(
                np.diff(wire_curve, axis=0), axis=1).sum()), 3),
            "distance_m": round(wire_dist, 3),
        },
        "wire_person_min_distance_m": round(pw_dist, 3),
        "ref_wire_person_min_distance_m": round(ref_dist, 3) if ref_dist is not None else None,
        "wire_person_bias_m": round(pw_dist - ref_dist, 3) if ref_dist is not None else None,
        "closest_person_point": r3(closest_person),
        "closest_wire_point": r3(closest_wire),
    }
    if vggt_depth is None:
        result["degraded"] = True
    if out_dir is not None:
        with open(os.path.join(out_dir, "range.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(out_dir, "range.txt"), "w") as f:
            f.write(f"frame={stem}\n")
            f.write(f"person  center={r3(pc)} distance={person_dist:.3f}m "
                    f"(template matching, score={person['score']:.2f})\n")
            f.write(f"wire    endpoints={r3(wire_curve[0])} -> {r3(wire_curve[-1])} "
                    f"distance={wire_dist:.3f}m ({wire_mode})\n")
            f.write(f"wire->person min distance = {pw_dist:.3f}m "
                    f"(person surface point {r3(closest_person)} <-> "
                    f"wire point {r3(closest_wire)})\n")
            if ref_dist is not None:
                f.write(f"wire->person min distance (ref) = {ref_dist:.3f}m  "
                        f"(bias algo-ref = {pw_dist - ref_dist:+.3f}m)\n")
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=ovd.DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default="viz_range")
    ap.add_argument("--ref_frame", default="img_1700603346.892936",
                    help="frame used for the 2D overlay / context PLY")
    ap.add_argument("--frames", default="all",
                    help="frames to run person+ranging on ('all' or comma stems)")
    ap.add_argument("--write_ref", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="also write the manual reference wire->person distance "
                         "(independent hand-specified standard, valid ONLY for "
                         "the bundled demo scene) into the outputs")
    # third-party integration
    ap.add_argument("--intrinsics", default=None,
                    help="camera intrinsics 'fx,fy,cx,cy' (comma-separated); "
                         "default: built-in 1280x720 calibration "
                         "(622.6716,694.2052,644.2907,522.0188)")
    ap.add_argument("--categories", default="person",
                    help="comma-separated open-vocabulary category texts for "
                         "YOLO-World (default: 'person')")
    ap.add_argument("--strict", action="store_true",
                    help="treat soft frame errors (missing files, detection "
                         "failures) as fatal: exit code 1")
    # wire
    ap.add_argument("--wire_mode", choices=["per_frame", "catenary"],
                    default="per_frame",
                    help="per_frame (default): PCA-linearity + nearest-depth-"
                         "cluster wire detection every frame (no precomputed "
                         "curve); catenary: static multi-frame catenary fit")
    ap.add_argument("--wire_curve", default=DEFAULT_WIRE_CURVE,
                    help="[catenary mode] precomputed catenary_curve.txt; set "
                         "'recompute' to rebuild from frames")
    ap.add_argument("--ttpla_weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=150,
                    help="band half-width (px) around the skeleton")
    # per_frame wire detection (data-driven, no absolute depth/height)
    ap.add_argument("--lin_k", type=int, default=12,
                    help="[per_frame] k-NN neighbourhood for the PCA linearity "
                         "wire template")
    ap.add_argument("--lin_thr", type=float, default=0.9,
                    help="[per_frame] linearity threshold (0..1)")
    ap.add_argument("--gap_kmad", type=float, default=4.0,
                    help="[per_frame] gap-significance factor for the nearest "
                         "coherent depth cluster")
    ap.add_argument("--min_size", type=int, default=3,
                    help="[per_frame] min depth-cluster size (points)")
    ap.add_argument("--wire_min_points", type=int, default=6,
                    help="[per_frame] min wire inlier/band points")
    ap.add_argument("--n_sample", type=int, default=100,
                    help="[per_frame] wire curve points")
    ap.add_argument("--extend_m", type=float, default=1.0,
                    help="[per_frame] upper-end tangent extension length (m)")
    ap.add_argument("--extend_to_z", type=float, default=3.0,
                    help="[per_frame] stop extension once lidar-z reaches this (m)")
    ap.add_argument("--extend_top_k", type=int, default=5)
    ap.add_argument("--extend_step", type=float, default=0.05)
    ap.add_argument("--mask_max_area", type=int, default=20000,
                    help="[per_frame] skip TTPLA mask above this area (px)")
    # catenary wire detection
    ap.add_argument("--z_min", type=float, default=1.4,
                    help="[catenary] lidar-z floor for wire candidates (m)")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--ransac_tol", type=float, default=0.08)
    ap.add_argument("--ransac_iter", type=int, default=2000)
    ap.add_argument("--n_curve", type=int, default=200)
    # person
    ap.add_argument("--yolo_weights", default=None)
    ap.add_argument("--sam2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sam2_ckpt", default=None)
    ap.add_argument("--sam2_config", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--box_threshold", type=float, default=0.1)
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--dbscan_eps", type=float, default=1.0)
    ap.add_argument("--dbscan_min_samples", type=int, default=3)
    ap.add_argument("--min_dist", type=float, default=0.5)
    ap.add_argument("--match_k", type=int, default=128)
    ap.add_argument("--tmpl_open", type=int, default=2)
    ap.add_argument("--tmpl_sor_m", type=float, default=2.0)
    ap.add_argument("--vggt_depth_dir", default=DEFAULT_VGGT_DIR)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.intrinsics:
        parts = [t.strip() for t in args.intrinsics.split(",")]
        if len(parts) != 4:
            ap.error("--intrinsics must be 'fx,fy,cx,cy' (4 comma-separated "
                     "numbers)")
        try:
            args.intrinsics = tuple(float(t) for t in parts)
        except ValueError:
            ap.error("--intrinsics values must be numbers")
    else:
        args.intrinsics = None

    import torch
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # YOLO-World set_classes needs weights/clip/ViT-B-32.pt regardless of the
    # caller's cwd; point ultralytics' weights_dir at this repo's weights/.
    deploy_utils.ensure_ultralytics_weights_dir()
    print(f"[out] outputs -> {os.path.abspath(args.out_dir)}", flush=True)

    # frames
    if args.frames == "all":
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(args.data_dir,
                                                       "images", "*.jpg")))
    else:
        stems = [s.strip() for s in args.frames.split(",") if s.strip()]

    # --- manual reference wire (independent hand-specified standard) -------- #
    ref_tree = None
    if args.write_ref:
        print("[warn] --write_ref: ref 距离来自原作者为本场景手工标注的参考导线，"
              "仅对随附演示数据有效；第三方数据上请保持关闭（默认）",
              file=sys.stderr, flush=True)
        try:
            _, allpts = accumulate_frames(args.data_dir)
            wire_pts, (wL, wu) = reference_wire(allpts)
            ref_tree = cKDTree(wire_pts)
            print(f"[ref] manual reference wire: {len(wire_pts)} pts "
                  f"(segment {wL:.2f}m, dir {np.round(wu, 3)})", flush=True)
        except Exception as e:
            print(f"[warn] ref computation failed, ref/bias columns left "
                  f"empty: {e}", file=sys.stderr, flush=True)
            if args.strict:
                raise

    # --- wire geometry ------------------------------------------------------ #
    ttpla = None
    wire_curve = None
    if args.wire_mode == "per_frame" or args.wire_curve == "recompute":
        if not os.path.exists(args.ttpla_weights):
            raise SystemExit(f"TTPLA weights not found at {args.ttpla_weights} "
                             f"(per_frame wire mode needs them; see "
                             f"weights/README.md)")
        from ultralytics import YOLO
        ttpla = YOLO(args.ttpla_weights).to(device)
    if args.wire_mode == "per_frame":
        wire_info = {"mode": "per_frame"}    # curve re-detected per frame below
    elif args.wire_curve == "recompute":
        wire_curve, wire_info = recompute_wire_curve(args, ttpla, stems)
        wire_info["mode"] = "catenary"
    else:
        wire_curve = load_wire_curve(args.wire_curve)
        wire_info = {"loaded_from": args.wire_curve, "mode": "catenary"}
        # provenance guard: the bundled catenary curve was fitted on the demo
        # scene -- warn when it is applied to frames it was not built from.
        if os.path.abspath(args.wire_curve) == os.path.abspath(DEFAULT_WIRE_CURVE):
            ref = "unknown"
            with open(args.wire_curve) as f:
                for line in f:
                    m = re.search(r"ref_frame=(\S+)", line)
                    if m:
                        ref = m.group(1)
                        break
            if ref not in stems:
                print(f"[warn] catenary mode: "
                      f"{os.path.basename(args.wire_curve)} was fitted on "
                      f"frame {ref}, which is not in the current frame set -- "
                      f"the static wire curve is scene-specific; use "
                      f"--wire_curve recompute or --wire_mode per_frame on new "
                      f"data", file=sys.stderr, flush=True)

    # --- person models (once) ---------------------------------------------- #
    yolo_path = args.yolo_weights or os.path.join(
        script_dir, "weights", "yoloworld", "yolov8m-worldv2.pt")
    if not os.path.exists(yolo_path):
        raise SystemExit(f"YOLO-World weights not found at {yolo_path}")
    det_model = build_yoloworld(yolo_path, device)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        raise SystemExit("--categories must name at least one category text")
    det_model.set_classes(categories)
    print(f"[detect] open-vocab categories: {categories}", flush=True)
    sam2_predictor = None
    if args.sam2:
        ckpt = args.sam2_ckpt or os.path.join(
            script_dir, "weights", "sam2", "sam2.1_hiera_base_plus.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"SAM2 checkpoint not found at {ckpt}")
        cfg = resolve_sam2_config(ckpt, args.sam2_config)
        sam2_predictor = build_sam2(ckpt, cfg, device)

    # --- loop over frames -------------------------------------------------- #
    results = []
    n_ok = 0
    n_soft = 0
    for i, stem in enumerate(stems, 1):
        r = process_frame(stem, wire_curve, ttpla, det_model, sam2_predictor,
                          args, script_dir, ref_tree)
        if r is None:
            print(f"[{i}/{len(stems)}] {stem}: missing files", flush=True)
            n_soft += 1
            continue
        if "error" in r:
            print(f"[{i}/{len(stems)}] {stem}: {r['error']}", flush=True)
            results.append(r)
            n_soft += 1
            continue
        n_ok += 1
        results.append(r)
        p = r["person"]
        ref = r.get("ref_wire_person_min_distance_m")
        ref_s = f" ref={ref:.3f}" if ref is not None else ""
        print(f"[{i}/{len(stems)}] {stem}: person {p['center']} "
              f"d={p['distance_m']}m  wire->person={r['wire_person_min_distance_m']}m"
              f"{ref_s} (score {p['score']:.2f}, {p['n_points']} pts)", flush=True)

    # --- top-level outputs -------------------------------------------------- #
    deploy_utils.write_run_outputs(args.out_dir, results)

    pws = [r["wire_person_min_distance_m"] for r in results
           if "wire_person_min_distance_m" in r]
    pds = [r["person"]["distance_m"] for r in results
           if "person" in r]
    print(f"\n[done] {n_ok}/{len(stems)} frames with a person; "
          f"wrote {args.out_dir}/results.json + summary.csv")
    if pws:
        print(f"  wire->person: mean={np.mean(pws):.3f}m "
              f"std={np.std(pws):.3f}m min={np.min(pws):.3f} max={np.max(pws):.3f}")
        print(f"  person dist  : mean={np.mean(pds):.3f}m std={np.std(pds):.3f}m")

    # exit codes: 0 = done (>=1 frame ok), 1 = fatal, 2 = usage (argparse)
    if n_ok == 0:
        print(f"[fatal] no frame produced a result ({n_soft} soft error(s))",
              file=sys.stderr, flush=True)
        sys.exit(1)
    if args.strict and n_soft > 0:
        print(f"[fatal] --strict: {n_soft} soft frame error(s) -> exit 1",
              file=sys.stderr, flush=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
