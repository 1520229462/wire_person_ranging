#!/usr/bin/env python3
"""Run AerialLidarClassifier's 3D SegFormer (UrbanFiltering, NRCan TreeAIBox)
standalone on the wire ROI, and score it against the hand-annotated wire.

This is the Python drop-in for the user's "segmentAerialLidarPowerline" idea:
project the 2D mask into a 3D ROI (done by export_roi_for_matlab.py), then let
a pretrained powerline semantic segmenter classify the points. The SegFormer
backend (core.backends.segformer) is QGIS-free and runs on torch alone.

Class mapping (ASPRS): 1 ground, 2 vegetation, 3 vehicles, 4 WIRING (power
line), 5 fence, 6 POLE, 7 building. Class 0 = "no prediction".
"""
import argparse
import os
import sys

import numpy as np

PLUGIN_DIR = "/home/xjt/experiment/ObjectDistance/third_party/AerialLidarClassifier"
WEIGHTS = ("/home/xjt/experiment/ObjectDistance/weights/segformer3d/"
           "urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth")
CONFIG = os.path.join(PLUGIN_DIR, "core", "model_config.json")

CLASS_NAMES = {0: "no-pred", 1: "ground", 2: "vegetation", 3: "vehicles",
               4: "WIRING", 5: "fence", 6: "POLE", 7: "building"}


def load_roi(roi_txt):
    return np.loadtxt(roi_txt, skiprows=1)[:, :3].astype(np.float64)


def gt_wire_mask(pts):
    """The hand-annotated wire region (see objectdistance-wire-extrinsic-error):
    lidar x 4.1-4.4, y -2.8..-1.9, z 1.45-3.0."""
    return ((pts[:, 0] > 4.1) & (pts[:, 0] < 4.4)
            & (pts[:, 1] > -2.8) & (pts[:, 1] < -1.9)
            & (pts[:, 2] > 1.45) & (pts[:, 2] < 3.0))


def report(tag, pts, labels):
    n = len(pts)
    ids, cnts = np.unique(labels, return_counts=True)
    hist = {int(i): int(c) for i, c in zip(ids, cnts)}
    wire = labels == 4
    pole = labels == 6
    gt = gt_wire_mask(pts)
    print(f"\n=== {tag}  ({n} pts) ===")
    for k in sorted(hist):
        print(f"  class {k} {CLASS_NAMES.get(k, '?'):12s}: {hist[k]:6d} "
              f"({100.0 * hist[k] / n:.1f}%)")
    print(f"  WIRING(class4)={int(wire.sum())}  POLE(class6)={int(pole.sum())}")
    if int(gt.sum()) > 0:
        gt_labels = labels[gt]
        gt_vals, gt_cnts = np.unique(gt_labels, return_counts=True)
        gt_hist = {int(i): int(c) for i, c in zip(gt_vals, gt_cnts)}
        tp = int((gt & wire).sum())
        print(f"  GT wire pts in input: {int(gt.sum())} -> predicted as: "
              f"{ {CLASS_NAMES.get(k, k): v for k, v in gt_hist.items()} }")
        recall = tp / int(gt.sum())
        prec = tp / max(int(wire.sum()), 1)
        print(f"  recall (GT wire -> class4): {recall:.2f}  "
              f"precision (class4 that is GT wire): {prec:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roi_txt", default=("viz_ttpla_wire3d/img_1700603346.892936/"
                                          "img_1700603346.892936_roi.txt"))
    ap.add_argument("--full_bin", default=("/home/xjt/experiment/ObjectDistance/"
                                           "UniCalib/Data/custom/lidar/"
                                           "img_1700603346.892936.bin"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--full", action="store_true",
                    help="also run on the full scene (5698 pts) for context")
    args = ap.parse_args()

    # The plugin's package __init__.py imports qgis.PyQt.QtCore even though the
    # inference core never touches it. Stub those names so the package imports
    # cleanly without QGIS installed.
    import types
    _qgis = types.ModuleType("qgis")
    _qgis.PyQt = types.ModuleType("qgis.PyQt")
    _qtcore = types.ModuleType("qgis.PyQt.QtCore")
    for _n in ("QCoreApplication", "QTranslator", "QLocale", "QSettings"):
        setattr(_qtcore, _n, type(_n, (), {}))
    _qgis.PyQt.QtCore = _qtcore
    sys.modules["qgis"] = _qgis
    sys.modules["qgis.PyQt"] = _qgis.PyQt
    sys.modules["qgis.PyQt.QtCore"] = _qtcore

    # Import as a real subpackage so the plugin's relative imports (..utils,
    # ..config) resolve against AerialLidarClassifier/ rather than top-level.
    sys.path.insert(0, os.path.dirname(PLUGIN_DIR))
    from AerialLidarClassifier.core.backends.segformer import SegFormerBackend

    backend = SegFormerBackend(CONFIG, WEIGHTS)
    backend.load(args.device)
    print(f"loaded SegFormer (UrbanFiltering) on {backend.device}")

    roi = load_roi(args.roi_txt)
    labels = backend.predict(roi)
    report("ROI (2D-mask band -> 3D)", roi, labels)

    if args.full:
        full = np.fromfile(args.full_bin, dtype=np.float32).reshape(-1, 4)[:, :3]
        full = full.astype(np.float64)
        labels_f = backend.predict(full)
        report("FULL scene (5698 pts)", full, labels_f)


if __name__ == "__main__":
    main()
