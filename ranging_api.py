#!/usr/bin/env python3
"""Library API for the wire + person ranging pipeline.

`RangingSession` wraps the same models and numerics as the
`range_person_wire.py` CLI behind a small Python API, for embedding into
third-party pipelines (e.g. a ROS2 node):

    session = RangingSession(categories=["person"])
    result = session.process_frame_files("img_1700603346.892936")

    # or fully in-memory (no file I/O):
    result = session.process(stem, image_rgb, points, T_lidar_to_cam)

The per-frame inputs are: image + point cloud + camera intrinsics +
extrinsics + open-vocabulary category texts (`categories`). The wire is a
separate fixed branch of the pipeline, driven by the TTPLA model in
`wire_mode="per_frame"` (or a static curve in `"catenary"` mode).

`process` performs no file I/O: the caller supplies the RGB image (H,W,3
uint8), the lidar points (N,3, lidar frame) and the lidar->camera transform
(4x4). `process_frame_files` loads those from the repo's data/ layout
instead (same as the CLI) and writes per-frame outputs.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_utils
import open_vocab_3d_detect as ovd
from range_person_wire import (load_wire_curve, DEFAULT_WIRE_CURVE,
                               DEFAULT_VGGT_DIR)
from ttpla_wire_3d import DEFAULT_WEIGHTS as DEFAULT_TTPLA_WEIGHTS

DEFAULT_INTRINSICS = deploy_utils.DEFAULT_INTRINSICS


class RangingSession:
    """Model holder + per-frame entry points for the ranging pipeline.

    All model-building parameters mirror the CLI flags of the same name
    (defaults identical). Pass `intrinsics` as a 4-tuple or an
    'fx,fy,cx,cy' string; None uses the built-in 1280x720 calibration.
    """

    def __init__(self, *, categories="person", wire_mode="per_frame",
                 wire_curve=None, device=None, intrinsics=None,
                 data_dir=None, out_dir="viz_range", vggt_depth_dir=None,
                 sam2=True, sam2_ckpt=None, sam2_config=None,
                 yolo_weights=None, ttpla_weights=None,
                 conf=0.5, dilate=150, lin_k=12, lin_thr=0.9, gap_kmad=4.0,
                 min_size=3, wire_min_points=6, n_sample=100, extend_m=1.0,
                 extend_to_z=3.0, extend_top_k=5, extend_step=0.05,
                 mask_max_area=20000,
                 imgsz=640, box_threshold=0.1, min_points=10, dbscan_eps=1.0,
                 dbscan_min_samples=3, min_dist=0.5, match_k=128, tmpl_open=2,
                 tmpl_sor_m=2.0):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if device is None:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if wire_mode not in ("per_frame", "catenary"):
            raise ValueError(f"wire_mode must be 'per_frame' or 'catenary', "
                             f"got {wire_mode!r}")
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",") if c.strip()]
        if not categories:
            raise ValueError("categories must name at least one category text")

        self.script_dir = script_dir
        self.device = device
        self.data_dir = data_dir or ovd.DEFAULT_DATA_DIR
        self.out_dir = out_dir
        self.vggt_depth_dir = vggt_depth_dir or DEFAULT_VGGT_DIR
        self.intrinsics = deploy_utils.parse_intrinsics(intrinsics)
        self.categories = list(categories)

        # YOLO-World set_classes needs weights/clip/ViT-B-32.pt regardless of
        # the caller's cwd.
        deploy_utils.ensure_ultralytics_weights_dir()

        # --- wire branch (TTPLA per frame, or a static curve) ------------- #
        self.ttpla = None
        self.wire_curve = None
        if wire_mode == "per_frame":
            from ultralytics import YOLO
            path = ttpla_weights or DEFAULT_TTPLA_WEIGHTS
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TTPLA weights not found at {path} (per_frame wire mode "
                    f"needs them; see weights/README.md)")
            self.ttpla = YOLO(path).to(device)
        else:
            self.wire_curve = load_wire_curve(wire_curve or DEFAULT_WIRE_CURVE)

        # --- person branch (YOLO-World + SAM2) ---------------------------- #
        yolo_path = yolo_weights or os.path.join(
            script_dir, "weights", "yoloworld", "yolov8m-worldv2.pt")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(
                f"YOLO-World weights not found at {yolo_path}")
        self.det_model = ovd.build_yoloworld(yolo_path, device)
        self.det_model.set_classes(self.categories)
        self.sam2_predictor = None
        if sam2:
            ckpt = sam2_ckpt or os.path.join(
                script_dir, "weights", "sam2", "sam2.1_hiera_base_plus.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"SAM2 checkpoint not found at {ckpt}")
            cfg = ovd.resolve_sam2_config(ckpt, sam2_config)
            self.sam2_predictor = ovd.build_sam2(ckpt, cfg, device)

        # --- tunables shared with the CLI (identical names / defaults) ---- #
        self.args = argparse.Namespace(
            data_dir=self.data_dir, out_dir=self.out_dir,
            vggt_depth_dir=self.vggt_depth_dir, wire_mode=wire_mode,
            conf=conf, dilate=dilate, lin_k=lin_k, lin_thr=lin_thr,
            gap_kmad=gap_kmad, min_size=min_size,
            wire_min_points=wire_min_points, n_sample=n_sample,
            extend_m=extend_m, extend_to_z=extend_to_z,
            extend_top_k=extend_top_k, extend_step=extend_step,
            mask_max_area=mask_max_area,
            imgsz=imgsz, box_threshold=box_threshold, min_points=min_points,
            dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
            min_dist=min_dist, match_k=match_k, tmpl_open=tmpl_open,
            tmpl_sor_m=tmpl_sor_m, intrinsics=self.intrinsics)

    # ------------------------------------------------------------------ #
    # Per-frame entry points
    # ------------------------------------------------------------------ #
    def process(self, stem, image_rgb, points, T_lidar_to_cam, *,
                vggt_depth=None, out_dir=None, ref_tree=None):
        """Run person + wire ranging on one in-memory frame (no file I/O).

        Args:
            stem:            frame id (recorded in the result dict).
            image_rgb:       (H,W,3) uint8 RGB image.
            points:          (N,3) lidar-frame points (float64).
            T_lidar_to_cam:  4x4 transform lidar -> camera frame.
            vggt_depth:      optional (H,W) float32 depth map; None degrades
                             the person branch to the nearest-cluster
                             fallback and sets ``result["degraded"]=True``.
            out_dir:         optional per-frame output dir for range.json /
                             range.txt / range_2d.jpg / range.ply.
            ref_tree:        optional manual-reference wire KD-tree
                             (scene-specific; the CLI's --write_ref).

        Returns the result dict (adds ``error`` on failure).
        """
        H, W = image_rgb.shape[:2]
        if self.intrinsics is None and (H, W) != (720, 1280):
            print(f"[warn] {stem}: image is {W}x{H}, expected 1280x720 -- the "
                  f"built-in intrinsics are calibrated for 1280x720; pass "
                  f"intrinsics=('fx','fy','cx','cy') to RangingSession",
                  file=sys.stderr, flush=True)
        from range_person_wire import process_frame_arrays
        return process_frame_arrays(
            stem, image_rgb, points, T_lidar_to_cam, self.wire_curve,
            self.ttpla, self.det_model, self.sam2_predictor, self.args,
            self.script_dir, ref_tree=ref_tree, out_dir=out_dir,
            intrinsics=self.intrinsics, vggt_depth=vggt_depth)

    def process_frame_files(self, stem, ref_tree=None):
        """Load one frame from data/ (same layout as the CLI) and run it.

        Writes <out_dir>/<stem>/range.json|range.txt|range_2d.jpg|range.ply.
        Returns the result dict, or None if the frame's files are missing.
        `ref_tree` mirrors the CLI's --write_ref (scene-specific manual
        reference wire KD-tree).
        """
        from PIL import Image
        img_path = os.path.join(self.data_dir, "images", stem + ".jpg")
        lidar_path = os.path.join(self.data_dir, "lidar", stem + ".bin")
        ext_path = os.path.join(self.data_dir, "extrinsics", stem + ".txt")
        if not all(os.path.exists(p) for p in (img_path, lidar_path, ext_path)):
            return None
        image_np = np.array(Image.open(img_path).convert("RGB"))
        points = ovd.load_lidar(lidar_path)
        T_lidar_to_cam = ovd.load_extrinsic(ext_path)
        vggt_depth = None
        dpath = os.path.join(self.vggt_depth_dir, stem + ".npz")
        if os.path.exists(dpath):
            vggt_depth = np.load(dpath)["depth"]
        return self.process(
            stem, image_np, points, T_lidar_to_cam, vggt_depth=vggt_depth,
            out_dir=os.path.join(self.out_dir, stem), ref_tree=ref_tree)

    # ------------------------------------------------------------------ #
    # Batch run
    # ------------------------------------------------------------------ #
    def run(self, stems=None, strict=False):
        """Run over frames (default: every image in data/) and write the
        top-level results.json + summary.csv (same as the CLI). Returns
        (results, n_ok). Raises RuntimeError on fatal conditions instead of
        calling sys.exit -- exit codes are a CLI concern.
        """
        if stems is None:
            stems = sorted(os.path.splitext(os.path.basename(p))[0]
                           for p in glob.glob(
                               os.path.join(self.data_dir, "images", "*.jpg")))
        results = []
        n_ok = n_soft = 0
        for i, stem in enumerate(stems, 1):
            r = self.process_frame_files(stem)
            if r is None:
                print(f"[{i}/{len(stems)}] {stem}: missing files", flush=True)
                n_soft += 1
                continue
            if "error" in r:
                print(f"[{i}/{len(stems)}] {stem}: {r['error']}", flush=True)
                n_soft += 1
            else:
                n_ok += 1
            results.append(r)
        if n_ok == 0:
            raise RuntimeError(f"no frame produced a result ({n_soft} error(s))")
        if strict and n_soft:
            raise RuntimeError(f"strict: {n_soft} soft frame error(s)")
        deploy_utils.write_run_outputs(self.out_dir, results)
        return results, n_ok

    # ------------------------------------------------------------------ #
    def close(self):
        """Drop model references; GPU memory is freed once the caller drops
        the session."""
        self.det_model = None
        self.sam2_predictor = None
        self.ttpla = None
