#!/usr/bin/env python3
"""Run AerialLidarClassifier's LitePT-L MLS (mobile-mapping, 5 cm) standalone
on the wire ROI, and score it against the hand-annotated wire.

This is the second pretrained model bundled in AerialLidarClassifier; unlike the
SegFormer UrbanFiltering (aerial ALS), LitePT-L MLS is trained on street-level
mobile-mapping scans, the closest available domain to our close-range drone lidar.
It has a dedicated "wire" class.

Plugin class ids (network index + 1):
  1 ground, 2 low_veg, 3 high_veg, 4 building, 5 pole_like, 6 vehicle,
  7 fence_barrier, 8 WIRE, 9 unknown.
"""
import argparse
import os
import sys

import numpy as np

PLUGIN_DIR = "/home/xjt/experiment/ObjectDistance/third_party/AerialLidarClassifier"
CARD = os.path.join(PLUGIN_DIR, "core", "litept_l_mls_5cm.json")
WEIGHTS = ("/home/xjt/experiment/ObjectDistance/weights/litept/"
           "litept_l_mls_5cm_fp16.pth")

CLASS_NAMES = {1: "ground", 2: "low_veg", 3: "high_veg", 4: "building",
               5: "pole_like", 6: "vehicle", 7: "fence_barrier",
               8: "WIRE", 9: "unknown"}


def load_roi(roi_txt):
    return np.loadtxt(roi_txt, skiprows=1)[:, :3].astype(np.float64)


def gt_wire_mask(pts):
    return ((pts[:, 0] > 4.1) & (pts[:, 0] < 4.4)
            & (pts[:, 1] > -2.8) & (pts[:, 1] < -1.9)
            & (pts[:, 2] > 1.45) & (pts[:, 2] < 3.0))


def report(pts, labels):
    n = len(pts)
    ids, cnts = np.unique(labels, return_counts=True)
    wire = labels == 8
    pole = labels == 5
    gt = gt_wire_mask(pts)
    print(f"\n=== LitePT-L MLS on ROI  ({n} pts) ===")
    for i, c in zip(ids, cnts):
        print(f"  class {i} {CLASS_NAMES.get(int(i), '?'):12s}: {c:5d} "
              f"({100.0 * c / n:.1f}%)")
    print(f"  WIRE(class8)={int(wire.sum())}  pole_like(class5)={int(pole.sum())}")
    if int(gt.sum()) > 0:
        gt_lab = labels[gt]
        gv, gc = np.unique(gt_lab, return_counts=True)
        tp = int((gt & wire).sum())
        print(f"  GT wire pts in ROI: {int(gt.sum())} -> predicted as: "
              f"{ {CLASS_NAMES.get(int(k), k): int(v) for k, v in zip(gv, gc)} }")
        recall = tp / int(gt.sum())
        prec = tp / max(int(wire.sum()), 1)
        print(f"  recall (GT wire -> WIRE): {recall:.2f}  "
              f"precision (WIRE that is GT wire): {prec:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roi_txt", default=("viz_ttpla_wire3d/img_1700603346.892936/"
                                          "img_1700603346.892936_roi.txt"))
    args = ap.parse_args()

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
    from AerialLidarClassifier.core.backends.litept import LitePTBackend

    backend = LitePTBackend(CARD, WEIGHTS, log=lambda m: print("  [litept]", m))
    backend.load("cuda")
    print(f"loaded LitePT-L MLS on {backend.device}")

    roi = load_roi(args.roi_txt)
    labels = backend.predict(roi)
    report(roi, labels)


if __name__ == "__main__":
    main()
