#!/usr/bin/env python3
"""Tolerance-based regression gate vs the frozen baseline (benchmarks/baseline).

For machines where byte-level parity is not expected (different GPU / CUDA /
driver), check a new run dir against the benchmarked baseline with documented
tolerances:

  - error-frame set must match exactly
  - wire_person_min_distance_m: |diff| <= 0.02 m per frame,
    mean |diff| <= 0.005 m
  - person distance_m:          |diff| <= 0.05 m per frame,
    mean |diff| <= 0.01 m
  - GT frame wire->person must stay at 0.905 m (+-0.02)

Usage:
  python scripts/regression_check.py [run_dir]     # default /tmp/wpr_after

Exit codes: 0 = pass, 1 = regression.
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(SCRIPT_DIR, "benchmarks", "baseline",
                        "perframe_results.json")

GT_FRAME = "img_1700603346.892936"
GT_PW = 0.905
PW_PER_FRAME_TOL = 0.02
PW_MEAN_TOL = 0.005
PD_PER_FRAME_TOL = 0.05
PD_MEAN_TOL = 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default="/tmp/wpr_after")
    args = ap.parse_args()

    with open(BASELINE) as f:
        base = json.load(f)
    with open(os.path.join(args.run_dir, "results.json")) as f:
        new = json.load(f)

    err_base = {k for k, v in base.items() if "error" in v}
    err_new = {k for k, v in new.items() if "error" in v}
    issues = []
    if err_base != err_new:
        issues.append(f"error-frame set differs: {sorted(err_base)} vs "
                      f"{sorted(err_new)}")
    if set(base) != set(new):
        issues.append(f"frame set differs: {sorted(set(base) ^ set(new))}")

    dpw, dpd = [], []
    for stem in sorted(base):
        b, n = base[stem], new.get(stem, {})
        if "error" in b:
            continue
        dpw.append(abs(b["wire_person_min_distance_m"]
                       - n["wire_person_min_distance_m"]))
        dpd.append(abs(b["person"]["distance_m"] - n["person"]["distance_m"]))
    dpw, dpd = np.array(dpw), np.array(dpd)
    print(f"frames: {len(dpw)} ok / {len(err_new)} errors")
    print(f"wire->person: max|d|={dpw.max():.4f}m mean|d|={dpw.mean():.4f}m")
    print(f"person dist : max|d|={dpd.max():.4f}m mean|d|={dpd.mean():.4f}m")
    if dpw.max() > PW_PER_FRAME_TOL or dpw.mean() > PW_MEAN_TOL:
        issues.append(f"wire->person drifted: max {dpw.max():.4f} > "
                      f"{PW_PER_FRAME_TOL} or mean {dpw.mean():.4f} > "
                      f"{PW_MEAN_TOL}")
    if dpd.max() > PD_PER_FRAME_TOL or dpd.mean() > PD_MEAN_TOL:
        issues.append(f"person distance drifted: max {dpd.max():.4f} > "
                      f"{PD_PER_FRAME_TOL} or mean {dpd.mean():.4f} > "
                      f"{PD_MEAN_TOL}")
    gt = new.get(GT_FRAME, {})
    if "error" in gt:
        issues.append(f"GT frame errored: {gt['error']}")
    elif abs(gt["wire_person_min_distance_m"] - GT_PW) > PW_PER_FRAME_TOL:
        issues.append(f"GT frame wire->person = "
                      f"{gt['wire_person_min_distance_m']}, expected {GT_PW}")

    if issues:
        print(f"[FAIL] {len(issues)} issue(s):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print("[PASS] within tolerance of the frozen baseline")
    sys.exit(0)


if __name__ == "__main__":
    main()
