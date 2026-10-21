#!/usr/bin/env python3
"""手工参考：导线 -> 人体表面 最小 3D 距离（独立于算法的测距精度标准）。

思路：不从 `range_person_wire.py` / `ttpla_wire_3d.py` 取任何结果，只读原始
LiDAR 点云，用**手工指定**的几何把导线和人分别挑出来，再算最近距离。

  * 导线（近静止，28 帧累积）：手工指定一条 3D 线段
        P_top = [4.36, -1.94, 3.0] -> P_bot = [4.12, -2.78, 1.5]
    （这两个端点是此前手标 GT；本脚本会打印 tube 内点数/残差供人工复核）。
    取所有帧中落在该线段 r=6cm 细管内的点作为「导线真实点集」。

  * 人（逐帧）：手工指定一个 3D 包围盒
        x in [3.9, 4.6], y in [-2.0, -1.1], z in [0.8, 1.6]
    取盒内「最近（x 最小）相干簇」作为该帧的人体表面点（墙在更远 x 处）。

  * 参考距离 = min over 人体表面点 of (到导线点集最近点的距离)（cKDTree）。

输出：每帧一行 (frame, 参考距离, 人体点数)，并与算法 results.json 里的
`wire_person_min_distance_m` 对齐，给出偏差（算法 - 参考）与 MAE/RMSE。
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree

# ---- 手工指定的几何（唯一需要人眼复核的地方） ----------------------------- #
WIRE_PTOP = np.array([4.36, -1.94, 3.0])    # 导线顶端（lidar 系，手标）
WIRE_PBOT = np.array([4.12, -2.78, 1.5])    # 导线底端（lidar 系，手标）
WIRE_TUBE_R = 0.08                          # 细管半径 (m)；敏感性检验在 r>=0.08m 收敛
PERSON_BOX = (3.9, 4.6, -2.0, -1.1, 0.8, 1.6)   # x0,x1,y0,y1,z0,z1
PERSON_X_THR = 4.4                          # 人/墙分界（人在 x 更小一侧）


def accumulate_frames(data_dir):
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(data_dir, "lidar", "*.bin")))
    return stems, np.concatenate(
        [np.fromfile(os.path.join(data_dir, "lidar", s + ".bin"),
                     dtype=np.float32).reshape(-1, 4)[:, :3] for s in stems], 0)


def reference_wire(allpts):
    seg = WIRE_PTOP - WIRE_PBOT
    L = float(np.linalg.norm(seg))
    u = seg / L
    t = np.clip((allpts - WIRE_PBOT) @ u, 0.0, L)
    d = np.linalg.norm((allpts - WIRE_PBOT) - t[:, None] * u, axis=1)
    return allpts[d < WIRE_TUBE_R], (L, u)


def person_points(frame_pts):
    x0, x1, y0, y1, z0, z1 = PERSON_BOX
    m = (frame_pts[:, 0] > x0) & (frame_pts[:, 0] < x1) \
        & (frame_pts[:, 1] > y0) & (frame_pts[:, 1] < y1) \
        & (frame_pts[:, 2] > z0) & (frame_pts[:, 2] < z1)
    p = frame_pts[m]
    if len(p) == 0:
        return p
    p = p[p[:, 0] < PERSON_X_THR]     # 墙在更远 x 处
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--results_json", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "viz_range_perframe",
        "results.json"), help="算法结果（用于对齐偏差）")
    args = ap.parse_args()

    stems, allpts = accumulate_frames(args.data_dir)
    wire, (L, u) = reference_wire(allpts)
    print(f"reference wire: {len(wire)} pts (tube r={WIRE_TUBE_R}m), "
          f"segment length {L:.2f}m, direction {np.round(u, 3)}")

    tree = cKDTree(wire)

    # 读取算法结果
    algo = {}
    if os.path.exists(args.results_json):
        with open(args.results_json) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if "wire_person_min_distance_m" in v:
                algo[k] = v["wire_person_min_distance_m"]

    print(f"\n{'frame':<26}{'ref(m)':>8}{'n_person':>10}{'algo(m)':>10}{'bias(m)':>9}")
    rows = []
    for s in stems:
        fp = np.fromfile(os.path.join(args.data_dir, "lidar", s + ".bin"),
                         dtype=np.float32).reshape(-1, 4)[:, :3]
        pp = person_points(fp)
        if len(pp) < 3:
            continue
        dd, _ = tree.query(pp, k=1)
        ref = float(dd.min())
        a = algo.get(s, float("nan"))
        bias = a - ref if np.isfinite(a) else float("nan")
        rows.append((s, ref, len(pp), a, bias))
        print(f"{s:<26}{ref:>8.3f}{len(pp):>10}{a:>10.3f}{bias:>9.3f}")

    refs = np.array([r[1] for r in rows])
    biases = np.array([r[4] for r in rows if np.isfinite(r[4])])
    print(f"\nreference mean={refs.mean():.3f} std={refs.std():.3f} "
          f"min={refs.min():.3f} max={refs.max():.3f}  (n={len(refs)})")
    if len(biases):
        print(f"bias (algo-ref): mean={biases.mean():+.3f} "
              f"MAE={np.abs(biases).mean():.3f} RMSE={np.sqrt((biases**2).mean()):.3f} "
              f"(n={len(biases)})")


if __name__ == "__main__":
    main()
