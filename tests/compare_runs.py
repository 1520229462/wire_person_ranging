#!/usr/bin/env python3
"""Compare two ranging-pipeline run directories (byte-level regression gate).

Usage:
  python tests/compare_runs.py <dir_before> <dir_after> [--strict] [--tol F]

--strict (default): byte-level equality (md5) of every written output,
pixel-equality of range_2d.jpg, identical error-frame set, plus the GT-frame
headline checks (wire->person = 0.905 m on img_1700603346.892936, dataset
mean = 0.8403 m). Exit 1 on any mismatch, 0 if identical.

--tol F: numeric tolerance mode (for cross-environment checks): JSON files
are compared field-by-field with |a-b| <= F (strings/keys must still match),
summary.csv columns with <= F, JPGs by mean abs pixel difference <= F*255.

Exit codes: 0 = pass, 1 = mismatch, 2 = usage error.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys

import cv2
import numpy as np

GT_FRAME = "img_1700603346.892936"
GT_PW = 0.905          # wire->person distance on the GT frame (m, 3dp)
MEAN_TARGET = 0.8403   # dataset mean of wire_person_min_distance_m (m, 4dp)

PER_FRAME_FILES = ("range.json", "range.txt", "range_2d.jpg", "range.ply")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_nested(a, b, tol, issues, path=""):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            issues.append(f"{path}: dict keys differ")
            return
        for k in a:
            walk_nested(a[k], b[k], tol, issues, f"{path}.{k}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            issues.append(f"{path}: list differs")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk_nested(x, y, tol, issues, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            issues.append(f"{path}: {a} != {b} (tol {tol})")
    else:
        if a != b:
            issues.append(f"{path}: {a!r} != {b!r}")


def _stems_in(dirpath, only):
    stems = [s for s in sorted(os.listdir(dirpath))
             if os.path.isdir(os.path.join(dirpath, s))]
    if only is not None:
        want = set(only)
        missing = want - set(stems)
        if missing:
            return None, sorted(missing)
        stems = [s for s in stems if s in want]
    return stems, []


def compare_strict(a, b, only=None):
    issues = []
    stems, missing = _stems_in(a, only)
    if missing:
        issues.append(f"frames missing from {a}: {missing}")
        return issues
    stems_b, missing = _stems_in(b, only)
    if missing:
        issues.append(f"frames missing from {b}: {missing}")
        return issues
    if stems != stems_b:
        issues.append(f"frame sets differ: {stems} vs {stems_b}")
        return issues
    for name in ("results.json", "summary.csv"):
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            issues.append(f"missing {name} in one dir")
            continue
        if md5(pa) != md5(pb):
            issues.append(f"{name}: md5 differs")
    for stem in stems:
        da, db = os.path.join(a, stem), os.path.join(b, stem)
        if not (os.path.isdir(da) and os.path.isdir(db)):
            issues.append(f"{stem}: missing per-frame dir")
            continue
        for fname in PER_FRAME_FILES:
            pa, pb = os.path.join(da, fname), os.path.join(db, fname)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                issues.append(f"{stem}/{fname}: missing in one dir")
                continue
            if fname == "range_2d.jpg":
                ia = cv2.imread(pa)
                ib = cv2.imread(pb)
                if ia is None or ib is None or ia.shape != ib.shape \
                        or (np.abs(ia.astype(int) - ib.astype(int))).max() != 0:
                    issues.append(f"{stem}/range_2d.jpg: pixels differ")
            elif md5(pa) != md5(pb):
                issues.append(f"{stem}/{fname}: md5 differs")
    # headline numbers (also covered by md5, but reported explicitly);
    # only meaningful when comparing full runs
    if only is None:
        with open(os.path.join(a, "results.json")) as f:
            ra = json.load(f)
        gt = ra.get(GT_FRAME, {})
        if "error" in gt:
            issues.append(f"GT frame {GT_FRAME} errored: {gt['error']}")
        else:
            pw = gt.get("wire_person_min_distance_m")
            if pw != GT_PW:
                issues.append(f"GT frame wire->person = {pw}, expected {GT_PW}")
            pws = [r["wire_person_min_distance_m"] for r in ra.values()
                   if "wire_person_min_distance_m" in r]
            if pws and round(float(np.mean(pws)), 4) != MEAN_TARGET:
                issues.append(f"mean wire->person = {np.mean(pws):.4f}, "
                              f"expected {MEAN_TARGET}")
    return issues


def compare_tol(a, b, tol, only=None):
    issues = []
    stems, missing = _stems_in(a, only)
    if missing:
        issues.append(f"frames missing from {a}: {missing}")
        return issues
    stems_b, missing = _stems_in(b, only)
    if missing:
        issues.append(f"frames missing from {b}: {missing}")
        return issues
    if stems != stems_b:
        issues.append(f"frame sets differ: {stems} vs {stems_b}")
        return issues
    with open(os.path.join(a, "results.json")) as f:
        ra = json.load(f)
    with open(os.path.join(b, "results.json")) as f:
        rb = json.load(f)
    for stem in stems:
        da, db = os.path.join(a, stem), os.path.join(b, stem)
        if os.path.isdir(da) and os.path.isdir(db):
            ja = json.load(open(os.path.join(da, "range.json")))
            jb = json.load(open(os.path.join(db, "range.json")))
            walk_nested(ja, jb, tol, issues, f"{stem}/range.json")
            ia = cv2.imread(os.path.join(da, "range_2d.jpg"))
            ib = cv2.imread(os.path.join(db, "range_2d.jpg"))
            if ia is None or ib is None or ia.shape != ib.shape:
                issues.append(f"{stem}/range_2d.jpg: missing or shape differs")
            else:
                m = np.abs(ia.astype(float) - ib.astype(float)).mean()
                if m > tol * 255:
                    issues.append(f"{stem}/range_2d.jpg: mean pixel diff "
                                  f"{m:.2f} > {tol * 255:.2f}")
    # summary.csv: numeric columns with tol, string columns exact
    with open(os.path.join(a, "summary.csv")) as f:
        rows_a = list(csv.reader(f))
    with open(os.path.join(b, "summary.csv")) as f:
        rows_b = list(csv.reader(f))
    if len(rows_a) != len(rows_b):
        issues.append("summary.csv: row count differs")
    else:
        hdr = rows_a[0]
        for i, h in enumerate(hdr):
            for ra_, rb_ in zip(rows_a[1:], rows_b[1:]):
                va, vb = ra_[i], rb_[i]
                if h == "frame":
                    if va != vb:
                        issues.append(f"summary.csv: frame {va} != {vb}")
                else:
                    try:
                        if abs(float(va) - float(vb)) > tol:
                            issues.append(f"summary.csv {h}: {va} != {vb}")
                    except ValueError:
                        if va != vb:
                            issues.append(f"summary.csv {h}: {va!r} != {vb!r}")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_before")
    ap.add_argument("dir_after")
    ap.add_argument("--strict", action="store_true",
                    help="byte-level equality (default)")
    ap.add_argument("--tol", type=float, default=None,
                    help="numeric tolerance mode with |diff| <= tol")
    ap.add_argument("--frames", default=None,
                    help="comma-separated stems to restrict the comparison to")
    args = ap.parse_args()

    only = ([s.strip() for s in args.frames.split(",") if s.strip()]
            if args.frames else None)
    if args.tol is not None:
        issues = compare_tol(args.dir_before, args.dir_after, args.tol, only)
    else:
        issues = compare_strict(args.dir_before, args.dir_after, only)
    if issues:
        print(f"[FAIL] {len(issues)} mismatch(es):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print(f"[PASS] {args.dir_before} == {args.dir_after}"
          + (f" (tol {args.tol})" if args.tol is not None else " (byte-level)"))
    sys.exit(0)


if __name__ == "__main__":
    main()
