#!/usr/bin/env python3
"""Generate deterministic golden snapshots of the pure-geometry primitives that
the deployment refactor (mechanical extraction in ttpla_wire_3d.py /
range_person_wire.py) touches. Model-dependent stages (TTPLA/YOLO-World/SAM2)
are covered by the end-to-end md5 parity gate (tests/compare_runs.py) instead.

Run from the repo root with the project env active:

    python tests/generate_golden.py

Writes tests/golden/*.json. Deterministic: no model inference, fixed inputs.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import open_vocab_3d_detect as ovd
from ttpla_wire_3d import (skeleton_polyline, _endpoint_tangent, _extend_up,
                           _line_consensus)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
STEM = "img_1700603346.892936"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    os.makedirs(GOLDEN, exist_ok=True)
    out = {}

    # --- 1. GT-frame geometry: lidar / extrinsics / projection -----------------
    points = ovd.load_lidar(os.path.join(BASE, "data", "lidar", STEM + ".bin"))
    T_lidar_to_cam = ovd.load_extrinsic(
        os.path.join(BASE, "data", "extrinsics", STEM + ".txt"))
    p_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    p_cam = (T_lidar_to_cam @ p_h.T).T[:, :3]
    u0, v0, z0 = ovd.project(p_cam)                      # default intrinsics
    u1, v1, z1 = ovd.project(p_cam, *ovd.DEFAULT_INTRINSICS
                             if hasattr(ovd, "DEFAULT_INTRINSICS") else
                             (ovd.FX, ovd.FY, ovd.CX, ovd.CY))
    assert np.array_equal(u0, u1) and np.array_equal(v0, v1), \
        "explicit intrinsics must reproduce default projection"
    out["project"] = {
        "u_sha": sha(u0), "v_sha": sha(v0), "z_sha": sha(z0),
        "n_points": int(len(points)),
        "T_lidar_to_cam_sha": sha(T_lidar_to_cam),
    }

    # --- 2. synthetic skeleton ------------------------------------------------
    import cv2
    skel = np.zeros((80, 80), np.uint8)
    pts = [(int(t), int(20 + 25 * np.sin(t / 70.0 * 3.0))) for t in range(5, 71)]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(skel, a, b, 1, 1)             # 8-connected, guaranteed unbroken
    us, vs, ss = skeleton_polyline(skel)
    out["skeleton_polyline"] = {
        "n": int(len(us)), "us_sha": sha(us), "vs_sha": sha(vs),
        "ss_sha": sha(ss), "s0": round(float(ss[0]), 3),
        "s_last": round(float(ss[-1]), 3),
    }

    # --- 3. synthetic tangent / extension -------------------------------------
    rng = np.random.RandomState(0)
    line_pts = np.array([[1.0 + 0.02 * i, -0.5 + 0.005 * i, 2.0 - 0.06 * i]
                         for i in range(40)]) + rng.normal(0, 0.002, (40, 3))
    tangent = _endpoint_tangent(line_pts, k=5)
    ext, n_ext = _extend_up(line_pts, tangent, extend_m=1.0, extend_to_z=3.0,
                            step=0.05)
    out["endpoint_extension"] = {
        "tangent": [round(float(x), 6) for x in tangent],
        "n_ext": int(n_ext), "ext_sha": sha(ext),
        "z_max": round(float(ext[:, 2].max()), 3),
    }

    # --- 4. synthetic line consensus ------------------------------------------
    # a 1D string of points (the wire) + a 2D sheet (the background) in camera
    # depth; k=12 exposes the sheet's 2D spread.
    cand_lidar = np.vstack([
        np.array([[4.0 + 0.01 * i, -2.0 + 0.02 * i, 2.0 - 0.04 * i]
                  for i in range(15)]),
        np.array([[3.0 + 0.1 * i, -1.0 + 0.1 * j, 1.0]
                  for i in range(6) for j in range(6)]),
    ])
    cand_d = np.r_[np.linspace(2.1, 2.2, 15), np.full(36, 1.2)]
    inl = _line_consensus(cand_lidar, cand_d, lin_k=12, lin_thr=0.9,
                          gap_kmad=4.0, min_size=3)
    out["line_consensus"] = {
        "n_inliers": int(inl.sum()), "inliers": [int(i) for i in np.where(inl)[0]],
    }

    path = os.path.join(GOLDEN, "primitives.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
