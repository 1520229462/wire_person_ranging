#!/usr/bin/env python3
"""Multi-frame accumulation -> pretrained powerline segmenter, feasibility test.

Question: the two pretrained models (SegFormer UrbanFiltering aerial ALS,
LitePT-L MLS mobile-mapping) gave 0 recall on the single-frame wire because it
is only ~13 points. Does accumulating the per-frame ROI across the (near-static)
sequence densify the wire enough that either model can now find it?

Pipeline: read all {stem}_roi.txt (band-projected lidar, lidar frame), concat,
dedup @1cm, run BOTH backends, score wire recall against the hand-annotated bbox.
"""
import argparse
import glob
import os
import sys

import numpy as np

PLUGIN_DIR = "/home/xjt/experiment/ObjectDistance/third_party/AerialLidarClassifier"

SEGFORMER_CFG = os.path.join(PLUGIN_DIR, "core", "model_config.json")
SEGFORMER_W = ("/home/xjt/experiment/ObjectDistance/weights/segformer3d/"
               "urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth")
SEGFORMER_CLS = {0: "no-pred", 1: "ground", 2: "vegetation", 3: "vehicles",
                 4: "WIRING", 5: "fence", 6: "POLE", 7: "building"}

LITEPT_CARD = os.path.join(PLUGIN_DIR, "core", "litept_l_mls_5cm.json")
LITEPT_W = ("/home/xjt/experiment/ObjectDistance/weights/litept/"
            "litept_l_mls_5cm_fp16.pth")
LITEPT_CLS = {1: "ground", 2: "low_veg", 3: "high_veg", 4: "building",
              5: "pole_like", 6: "vehicle", 7: "fence_barrier",
              8: "WIRE", 9: "unknown"}


def gt_wire_mask(pts):
    return ((pts[:, 0] > 4.1) & (pts[:, 0] < 4.4)
            & (pts[:, 1] > -2.8) & (pts[:, 1] < -1.9)
            & (pts[:, 2] > 1.45) & (pts[:, 2] < 3.0))


def load_accumulated(roi_dir, dedup_cm=1):
    txts = sorted(glob.glob(os.path.join(roi_dir, "*", "*_roi.txt")))
    pts = np.concatenate(
        [np.loadtxt(t, skiprows=1)[:, :3].astype(np.float64) for t in txts],
        axis=0)
    raw = len(pts)
    if dedup_cm > 0:
        pts = np.unique(np.round(pts, 2), axis=0)  # 1 cm
    return pts, len(txts), raw


def _qgis_stub():
    import types
    _q = types.ModuleType("qgis")
    _q.PyQt = types.ModuleType("qgis.PyQt")
    _c = types.ModuleType("qgis.PyQt.QtCore")
    for _n in ("QCoreApplication", "QTranslator", "QLocale", "QSettings"):
        setattr(_c, _n, type(_n, (), {}))
    _q.PyQt.QtCore = _c
    sys.modules["qgis"] = _q
    sys.modules["qgis.PyQt"] = _q.PyQt
    sys.modules["qgis.PyQt.QtCore"] = _c
    sys.path.insert(0, os.path.dirname(PLUGIN_DIR))


def report(tag, pts, labels, wire_ids, cls_map):
    n = len(pts)
    ids, cnts = np.unique(labels, return_counts=True)
    wire = np.isin(labels, wire_ids)
    gt = gt_wire_mask(pts)
    print(f"\n=== {tag}  (n={n}) ===")
    for i, c in zip(ids, cnts):
        print(f"  {cls_map.get(int(i), i):12s}: {c:6d} ({100.0*c/n:5.1f}%)")
    tp = int((gt & wire).sum())
    gt_n = int(gt.sum())
    wire_n = int(wire.sum())
    print(f"  GT-wire pts in input: {gt_n}   predicted {cls_map.get(wire_ids[0])}: {wire_n}")
    recall = tp / max(gt_n, 1)
    prec = tp / max(wire_n, 1)
    print(f"  recall {recall:.3f}   precision {prec:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roi_dir", default="viz_ttpla_wire3d")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    acc, n_frames, raw = load_accumulated(args.roi_dir)
    gt_n = int(gt_wire_mask(acc).sum())
    print(f"accumulated ROI: {n_frames} frames, raw={raw} pts -> "
          f"{len(acc)} unique @1cm;  GT-wire pts = {gt_n} "
          f"({100.0*gt_n/max(len(acc),1):.1f}% of ROI)")

    _qgis_stub()

    from AerialLidarClassifier.core.backends.segformer import SegFormerBackend
    sfb = SegFormerBackend(SEGFORMER_CFG, SEGFORMER_W)
    sfb.load(args.device)
    print(f"\nSegFormer (UrbanFiltering, 30cm) on {sfb.device}")
    report("SegFormer", acc, sfb.predict(acc), [4], SEGFORMER_CLS)

    from AerialLidarClassifier.core.backends.litept import LitePTBackend
    lpb = LitePTBackend(LITEPT_CARD, LITEPT_W, log=lambda m: print("  [litept]", m))
    lpb.load(args.device)
    print(f"\nLitePT-L MLS (5cm) on {lpb.device}")
    report("LitePT-L MLS", acc, lpb.predict(acc), [8], LITEPT_CLS)


if __name__ == "__main__":
    main()
