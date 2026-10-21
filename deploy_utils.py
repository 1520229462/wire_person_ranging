#!/usr/bin/env python3
"""Deployment helpers shared by the CLI entry points and the library API.

No model code lives here -- just environment setup, argument parsing and
output writing.
"""
import json
import os

# Built-in undistorted pinhole intrinsics @1280x720.
# NOTE: must equal open_vocab_3d_detect.FX/FY/CX/CY (asserted in
# tests/check_equivalence.py).
DEFAULT_INTRINSICS = (622.6716, 694.2052, 644.2907, 522.0188)


def ensure_ultralytics_weights_dir():
    """Point ultralytics' global weights_dir at this repo's weights/.

    YOLO-World's set_classes hardcodes "clip:ViT-B/32", which ultralytics
    resolves relative to its global weights_dir (default 'weights' relative
    to the process cwd). Calling this makes the pipeline work from any
    working directory. No-op if ultralytics is unavailable or already set.
    """
    try:
        from ultralytics.utils import SETTINGS
        repo_weights = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights")
        if SETTINGS.get("weights_dir") != repo_weights:
            SETTINGS.update({"weights_dir": repo_weights})
    except Exception:
        pass


def parse_intrinsics(s):
    """Parse 'fx,fy,cx,cy' (or None/'') -> (fx, fy, cx, cy) float tuple, or
    None. Raises ValueError on malformed input."""
    if not s:
        return None
    parts = [t.strip() for t in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "intrinsics must be 'fx,fy,cx,cy' (4 comma-separated numbers)")
    return tuple(float(t) for t in parts)


def write_run_outputs(out_dir, results):
    """Write results.json + summary.csv for a list of per-frame result dicts
    (byte-identical to the top-level outputs of the range_person_wire CLI)."""
    os.makedirs(out_dir, exist_ok=True)
    out = {r["frame"]: r for r in results}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(out_dir, "summary.csv"), "w") as f:
        f.write("frame,person_x,person_y,person_z,person_dist_m,wire_dist_m,"
                "wire_person_min_dist_m,ref_wire_person_min_dist_m,"
                "wire_person_bias_m,score,n_points\n")
        for r in results:
            if "error" in r:
                f.write(f"{r['frame']},,,,,,,,,,{r['error']}\n")
                continue
            p = r["person"]
            c = p["center"]
            ref = r.get("ref_wire_person_min_distance_m")
            bias = r.get("wire_person_bias_m")
            f.write(f"{r['frame']},{c[0]},{c[1]},{c[2]},{p['distance_m']},"
                    f"{r['wire']['distance_m']},{r['wire_person_min_distance_m']},"
                    f"{ref if ref is not None else ''},"
                    f"{bias if bias is not None else ''},"
                    f"{p['score']},{p['n_points']}\n")
