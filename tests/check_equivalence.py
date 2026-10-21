#!/usr/bin/env python3
"""Equivalence checks between the CLI and the library API (ranging_api.py).

Checks:
  1. deploy_utils.DEFAULT_INTRINSICS == open_vocab_3d_detect built-ins.
  2. RangingSession's internal Namespace covers every args.<attr> that
     detect_person / process_frame_arrays read (no silent default drift).
  3. detect_wire_curve (file shell) == detect_wire_curve_arrays on the
     GT frame (identical curve + info).
  4. RangingSession.process_frame_files(stem) == CLI per-frame outputs
     (range.json field-exact at 3dp, range.txt / range.ply bytes, jpg
     pixels) for every stem (default 28; --stems N for a quick mode).
  5. In-memory RangingSession.process(...) == process_frame_files on the
     first 3 stems (identical result dicts, and no files written when
     out_dir=None).

Needs the full model stack (~3 GB GPU). --quick runs only the static
checks 1-2 (no models). --cli-dir points at a CLI run of the SAME
post-refactor code (default /tmp/wpr_after).

Exit codes: 0 = pass, 1 = mismatch, 2 = usage error.
"""
import argparse
import inspect
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deploy_utils
import open_vocab_3d_detect as ovd
import ranging_api
import range_person_wire as rpw
from ttpla_wire_3d import detect_wire_curve, detect_wire_curve_arrays

GT_FRAME = "img_1700603346.892936"


def args_attrs_of(func):
    """Every args.<name> attribute the function's source references."""
    src = inspect.getsource(func)
    return sorted(set(re.findall(r"args\.([A-Za-z_][A-Za-z0-9_]*)", src)))


def check_static(session):
    issues = []
    # 1. intrinsics single source of truth
    got = deploy_utils.DEFAULT_INTRINSICS
    want = (ovd.FX, ovd.FY, ovd.CX, ovd.CY)
    if got != want:
        issues.append(f"DEFAULT_INTRINSICS {got} != ovd constants {want}")
    # 2. Namespace coverage
    for func in (rpw.detect_person, rpw.process_frame_arrays):
        for a in args_attrs_of(func):
            if not hasattr(session.args, a):
                issues.append(f"RangingSession.args missing '{a}' "
                              f"(read by {func.__name__})")
    return issues


def check_wire_arrays(data_dir, session):
    """3. file shell vs in-memory arrays: identical curve + info."""
    stem = GT_FRAME
    curve1, info1 = detect_wire_curve(session.ttpla, stem, data_dir)
    img = None
    import cv2
    img = cv2.imread(os.path.join(data_dir, "images", stem + ".jpg"))
    points = ovd.load_lidar(os.path.join(data_dir, "lidar", stem + ".bin"))
    T_cam_to_lidar = np.loadtxt(os.path.join(data_dir, "extrinsics",
                                             stem + ".txt"),
                                dtype=np.float64)
    T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)
    curve2, info2 = detect_wire_curve_arrays(session.ttpla, img, points,
                                             T_lidar_to_cam, stem=stem)
    issues = []
    if info1 != info2:
        issues.append(f"detect_wire_curve info differs:\n  {info1}\n  {info2}")
    if curve1 is None or curve2 is None:
        if curve1 is not curve2:
            issues.append("detect_wire_curve curve None-ness differs")
    elif not np.array_equal(curve1, curve2):
        d = np.abs(curve1 - curve2).max()
        issues.append(f"detect_wire_curve curve differs (max {d})")
    return issues


def check_process_frame_files(session, stems, cli_dir):
    """4. RangingSession.process_frame_files == CLI per-frame outputs.

    The CLI run was produced with --write_ref, so the same manual reference
    wire KD-tree is built here and passed as ref_tree (the API's equivalent
    of the CLI flag) -- outputs then must be byte-identical.
    """
    import json
    import hashlib
    import cv2
    from manual_wire_person_ref import accumulate_frames, reference_wire
    from scipy.spatial import cKDTree

    def md5(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    _, allpts = accumulate_frames(session.data_dir)
    wire_pts, _ = reference_wire(allpts)
    ref_tree = cKDTree(wire_pts)

    issues = []
    for stem in stems:
        r = session.process_frame_files(stem, ref_tree=ref_tree)
        cli_json = os.path.join(cli_dir, stem, "range.json")
        if not os.path.exists(cli_json):
            # error frames write no per-frame dir in either path
            if r is not None and "error" not in r:
                issues.append(f"{stem}: CLI errored but API returned a result "
                              f"{r}")
            continue
        with open(cli_json) as f:
            want = json.load(f)
        if r != want:
            issues.append(f"{stem}: result dict != CLI range.json\n"
                          f"  api : {r}\n  cli : {want}")
        out = os.path.join(session.out_dir, stem)
        for name, mode in (("range.txt", "bytes"), ("range.ply", "bytes"),
                           ("range_2d.jpg", "pixels")):
            pa = os.path.join(out, name)
            pb = os.path.join(cli_dir, stem, name)
            if mode == "pixels":
                ia = cv2.imread(pa)
                ib = cv2.imread(pb)
                if ia is None or ib is None or ia.shape != ib.shape \
                        or (np.abs(ia.astype(int) - ib.astype(int))).max() != 0:
                    issues.append(f"{stem}/{name}: pixels differ")
            elif md5(pa) != md5(pb):
                issues.append(f"{stem}/{name}: bytes differ")
    # GT headline through the API
    gt = session.process_frame_files(GT_FRAME, ref_tree=ref_tree)
    if gt.get("wire_person_min_distance_m") != 0.905:
        issues.append(f"API GT frame wire->person = "
                      f"{gt.get('wire_person_min_distance_m')}")
    return issues


def check_in_memory(session, stems):
    """5. process(...) == process_frame_files (ref-free baseline), and
    out_dir=None writes nothing."""
    from PIL import Image
    import json
    import tempfile
    issues = []
    tmp = tempfile.mkdtemp(prefix="wpr_api_none_")
    for stem in stems:
        img_path = os.path.join(session.data_dir, "images", stem + ".jpg")
        lidar_path = os.path.join(session.data_dir, "lidar", stem + ".bin")
        ext_path = os.path.join(session.data_dir, "extrinsics", stem + ".txt")
        image_np = np.array(Image.open(img_path).convert("RGB"))
        points = ovd.load_lidar(lidar_path)
        T_lidar_to_cam = ovd.load_extrinsic(ext_path)
        dpath = os.path.join(session.vggt_depth_dir, stem + ".npz")
        vggt = np.load(dpath)["depth"] if os.path.exists(dpath) else None
        # in-memory, no writes
        before = set(os.listdir(tmp))
        r_mem = session.process(stem, image_np, points, T_lidar_to_cam,
                                vggt_depth=vggt, out_dir=None)
        after = set(os.listdir(tmp))
        if after != before:
            issues.append(f"{stem}: out_dir=None still wrote files "
                          f"({sorted(after - before)})")
        # ref-free file-based baseline (ref_tree=None -> ref fields null)
        r_files = session.process_frame_files(stem, ref_tree=None)
        if r_mem != r_files:
            issues.append(f"{stem}: process() != process_frame_files()\n"
                          f"  mem : {r_mem}\n  files: {r_files}")
        # explicit out_dir writes the same result dict
        sub = os.path.join(tmp, stem)
        r_mem2 = session.process(stem, image_np, points, T_lidar_to_cam,
                                 vggt_depth=vggt, out_dir=sub)
        if r_mem2 != r_mem:
            issues.append(f"{stem}: out_dir run changed the result dict")
        for name in ("range.json", "range.txt", "range_2d.jpg", "range.ply"):
            if not os.path.exists(os.path.join(sub, name)):
                issues.append(f"{stem}: out_dir={sub} did not write {name}")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli-dir", default="/tmp/wpr_after",
                    help="post-refactor CLI run dir to compare against")
    ap.add_argument("--stems", type=int, default=28,
                    help="number of stems for the per-frame equivalence "
                         "check (default 28; 3 for quick mode)")
    ap.add_argument("--api-out", default="/tmp/wpr_api",
                    help="output dir for the RangingSession run")
    ap.add_argument("--quick", action="store_true",
                    help="static checks only (no models loaded)")
    args = ap.parse_args()

    session = ranging_api.RangingSession(out_dir=args.api_out)
    issues = check_static(session)

    if args.quick:
        session.close()
        return finish(issues)

    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in __import__("glob").glob(
                       os.path.join(session.data_dir, "images", "*.jpg")))
    if args.stems > len(stems):
        args.stems = len(stems)
    issues += check_wire_arrays(session.data_dir, session)
    issues += check_process_frame_files(session, stems[:args.stems],
                                        args.cli_dir)
    issues += check_in_memory(session, stems[:3])
    session.close()
    return finish(issues)


def finish(issues):
    if issues:
        print(f"[FAIL] {len(issues)} issue(s):")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("[PASS] CLI <-> RangingSession equivalence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
