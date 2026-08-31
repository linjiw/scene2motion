"""EXP-015b: did EXP-015's elicited lift occur where an obstacle needed it?

EXP-015 reported peak foot height anywhere in a broad middle window and contained no obstacle.
This CPU-only post-hoc diagnostic reuses its saved qpos clips, places a virtual corridor-spanning
floor box at one fixed route coordinate, and asks the operational question: what box height did
the complete body actually clear there?  It also reports the distance between the local swing
peak and the obstacle, plus foot height and support at crossing.

This is explicitly exploratory: the obstacle placement was absent from EXP-015's predeclared
design.  It can diagnose the timing/placement failure and motivate EXP-016, but it cannot serve
as confirmatory evidence for EXP-016's text-by-scaffold comparison.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.stepover_eval import (  # noqa: E402
    BoxHeightProbe,
    motion_metrics,
    step_scene,
)


ARMS = ("A-walk-text", "B-stepover-text", "C-walk-text+pos-lift")


def entry_to_qpos(entry: dict) -> np.ndarray:
    """Invert the Scene2Motion SONIC pickle representation for its stored kinematics."""
    root = np.asarray(entry["root_trans_offset"], dtype=float)
    quat_xyzw = np.asarray(entry["root_rot"], dtype=float)
    dof = np.asarray(entry["dof"], dtype=float)
    if root.ndim != 2 or root.shape[1] != 3 or quat_xyzw.shape != (len(root), 4):
        raise ValueError("invalid root state in SONIC motion entry")
    if dof.ndim != 2 or len(dof) != len(root):
        raise ValueError("invalid DOF state in SONIC motion entry")
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([root, quat_wxyz, dof], axis=1)


def seed_from_key(key: str) -> int:
    try:
        return int(key.rsplit("__s", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"motion key does not end in __s<seed>: {key!r}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path("outputs/exp015"))
    ap.add_argument("--out", type=Path, default=Path("outputs/exp015b_spatial_reanalysis"))
    ap.add_argument("--obstacle-x", type=float, default=3.8)
    ap.add_argument("--obstacle-depth", type=float, default=0.20)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.05, 0.08, 0.12])
    ap.add_argument("--gate-height", type=float, default=0.08,
                    help="fixed obstacle used for the exact local step-topology diagnostic")
    args = ap.parse_args()
    if args.obstacle_depth <= 0 or args.gate_height <= 0 or any(h <= 0 for h in args.heights):
        raise SystemExit("obstacle dimensions must be positive")
    for arm in ARMS:
        if not (args.source / f"{arm}.pkl").exists():
            raise SystemExit(f"missing EXP-015 artifact: {args.source / f'{arm}.pkl'}")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    free = G1Body(None)
    probe = BoxHeightProbe(args.obstacle_x, args.obstacle_depth, hi=max(0.30, max(args.heights)))
    gate_body = G1Body(step_scene(args.obstacle_x, args.gate_height, args.obstacle_depth))
    fixed = {float(h): G1Body(step_scene(args.obstacle_x, float(h), args.obstacle_depth))
             for h in args.heights}
    rows = []
    for arm in ARMS:
        with open(args.source / f"{arm}.pkl", "rb") as fh:
            motions = pickle.load(fh)
        for key, entry in motions.items():
            qpos = entry_to_qpos(entry)
            # EXP-015 had no planned obstacle, so use its realized route only to satisfy the
            # generic path descriptor. The operational endpoints remain exact scene collision,
            # exact physical-foot overlap, and the full local contact topology.
            realized_route = np.stack([qpos[:, 1], qpos[:, 0]], axis=-1)
            m = motion_metrics(free, gate_body, probe, qpos, realized_route,
                               args.obstacle_x, "left", fps=25.0)
            row = {
                "arm": arm,
                "motion_key": key,
                "seed": seed_from_key(key),
                "selected_swing_side": m["selected_lead_side"],
                "max_box_height_lower_bound_m": m["max_box_height_lower_bound_m"],
                "phase_error_m": m["phase_error_m"],
                "phase_error_frames": m["phase_error_frames"],
                "swing_foot_at_crossing_m": m["swing_foot_at_crossing_m"],
                "mean_support_feet": m["mean_support_feet"],
                "bilateral_flight_fraction": m["bilateral_flight_fraction"],
                "progress_m": m["progress_m"],
                "obstacle_collision_free_at_gate_height": m["obstacle_collision_free"],
                "local_step_success": m["local_step_success"],
                "local_step": m["local_step"],
            }
            for height, body in fixed.items():
                row[f"clears_{height:.3f}m"] = bool(
                    body.trajectory_report(qpos)["collision_free"])
            rows.append(row)

    with open(args.out / "rows.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    summary = {}
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "n": len(group),
            "max_box_height_lower_bound_m": float(np.mean(
                [r["max_box_height_lower_bound_m"] for r in group])),
            "mean_abs_phase_error_m": float(np.mean([abs(r["phase_error_m"]) for r in group])),
            "swing_foot_at_crossing_m": float(np.mean(
                [r["swing_foot_at_crossing_m"] for r in group])),
            "mean_support_feet": float(np.mean([r["mean_support_feet"] for r in group])),
            "bilateral_flight_fraction": float(np.mean(
                [r["bilateral_flight_fraction"] for r in group])),
            "local_step_success_rate": float(np.mean(
                [r["local_step_success"] for r in group])),
            **{f"clear_rate_{h:.3f}m": float(np.mean(
                [r[f'clears_{h:.3f}m'] for r in group])) for h in fixed},
        }

    keyed = {(r["arm"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows if all((arm, r["seed"]) in keyed for arm in ARMS)})
    paired = {}
    for arm in ARMS[1:]:
        paired[f"{arm}_minus_{ARMS[0]}"] = {
            "n": len(seeds),
            "d_max_box_height_lower_bound_m": float(np.mean([
                keyed[(arm, seed)]["max_box_height_lower_bound_m"] -
                keyed[(ARMS[0], seed)]["max_box_height_lower_bound_m"]
                for seed in seeds
            ])),
            "d_swing_foot_at_crossing_m": float(np.mean([
                keyed[(arm, seed)]["swing_foot_at_crossing_m"] -
                keyed[(ARMS[0], seed)]["swing_foot_at_crossing_m"] for seed in seeds
            ])),
        }

    receipt = {
        "experiment": "exp015b_spatial_reanalysis",
        "status": "post_hoc_exploratory",
        "source": str(args.source),
        "source_sampler": "legacy_noise_stream_v1_reseeded_each_autoregressive_window",
        "obstacle_x_m": args.obstacle_x,
        "obstacle_depth_m": args.obstacle_depth,
        "local_step_gate_height_m": args.gate_height,
        "fixed_heights_m": list(fixed),
        "box_height_probe": probe.metadata(),
        "body_margin_is_already_in_collision_geometry": True,
        "summary": summary,
        "paired": paired,
        "interpretation_guard": (
            "The obstacle was not part of EXP-015's design; use this only to diagnose spatial "
            "misalignment and motivate held-out EXP-016. Phase is measured from exact physical "
            "foot geometry, not the pelvis."
        ),
        "wall_clock_s": round(time.time() - t0, 2),
    }
    with open(args.out / "receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
