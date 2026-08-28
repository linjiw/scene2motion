"""EXP-001b: how narrow can the frozen prior actually make the G1?

EXP-001 found that the three channels do not behave the way one would assume:

  duck   works, and scales: -53 cm of top height at a 0.50 m pelvis dip
  tuck   works but saturates: about -5 cm of half-width, then nothing more
  sidle  rotates the body to 83 deg of the requested 90 -- and buys NO half-width at all,
         because once the robot is side-on its arm SWING ARC along the travel direction
         becomes the corridor width, replacing the shoulder width it just removed

So the interesting question is whether the channels COMPOSE: does sidling plus tucking the
arms (so there is no swing arc left to replace the shoulders) finally narrow the robot?

This matters because the answer sets the `narrow` body mode in the planner, and therefore
which gap widths the whole system can ever claim to traverse. Getting it from measurement
rather than assumption is the difference between an honest feasibility boundary and a
fictional one.

Every condition is compared against a MATCHED CONTROL: the same seed, same path, same
prompt, no adaptation. EXP-001 initially compared each clip against its own opening window
and produced a spurious "ducking makes you wider" effect that vanished under matched
control -- the gait's opening arm swing is simply wider than its steady state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402

PROMPT_WALK = "A person walks forward."
PROMPT_SIDE = "A person steps sideways through a narrow gap."
DURATION, SPEED, NOMINAL_PELVIS = 8.0, 0.95, 0.78
CENTER, SIGMA = 0.50, 0.16
MEASURE = (0.42, 0.58)
SEEDS = [0, 1, 2]

# The arm chain, plus the wrists: the hand capsules (r=0.05) are the widest part of a
# free-swinging arm, so leaving them out understates what tucking can do.
ARM_JOINTS = ["left_shoulder_roll_skel", "left_elbow_skel",
              "left_wrist_roll_skel", "left_hand_roll_skel",
              "right_shoulder_roll_skel", "right_elbow_skel",
              "right_wrist_roll_skel", "right_hand_roll_skel"]

# (sidle degrees, tuck strength, prompt)
GRID = [(0, 0.0, PROMPT_WALK), (0, 0.7, PROMPT_WALK), (0, 0.85, PROMPT_WALK),
        (45, 0.0, PROMPT_WALK), (45, 0.7, PROMPT_WALK), (45, 0.85, PROMPT_WALK),
        (60, 0.0, PROMPT_WALK), (60, 0.7, PROMPT_WALK), (60, 0.85, PROMPT_WALK),
        (75, 0.0, PROMPT_WALK), (75, 0.7, PROMPT_WALK), (75, 0.85, PROMPT_WALK),
        (90, 0.0, PROMPT_WALK), (90, 0.7, PROMPT_WALK), (90, 0.85, PROMPT_WALK),
        (90, 0.0, PROMPT_SIDE), (90, 0.7, PROMPT_SIDE), (90, 0.85, PROMPT_SIDE),
        (75, 0.85, PROMPT_SIDE), (60, 0.85, PROMPT_SIDE)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp001b")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DURATION * fps)
    t = np.arange(T)
    w = np.exp(-0.5 * ((t - T * CENTER) / (T * SIGMA)) ** 2)
    path = np.stack([np.zeros(T), SPEED * t / fps], -1)
    body = G1Body(None)
    lateral = np.array([0.0, 1.0, 0.0])   # travel is +x, so a slot constrains world y
    mid = slice(int(T * MEASURE[0]), int(T * MEASURE[1]))
    arm_idx = np.array([runner.joint_names.index(j) for j in ARM_JOINTS])
    step = max(1, int(0.2 * fps))
    active = np.where(w > 0.35)[0][::step]

    controls, rows = {}, []
    for s in SEEDS:
        spec = ConstraintSpec(root_xz=path, heading=np.zeros(T),
                              root_y=np.full(T, NOMINAL_PELVIS))
        o = runner.generate([PROMPT_WALK], [spec], T, args.diffusion_steps, seed=s)[0]
        q = runner.to_qpos(o)
        controls[s] = {"sample": o,
                       "hw": max(body.half_width(x, lateral) for x in q[mid]),
                       "top": max(body.top_height(x) for x in q[mid])}

    conds = []
    for s in SEEDS:
        nom = controls[s]["sample"]
        for phi, k, prompt in GRID:
            if k == 0.0:
                spec = ConstraintSpec(root_xz=path, heading=np.deg2rad(phi) * w,
                                      root_y=np.full(T, NOMINAL_PELVIS))
            else:
                off = (nom["posed_joints"][active][:, arm_idx, :]
                       - nom["smooth_root_pos"][active][:, None, :])
                # Pull the arms in along BOTH ground axes: once the body is side-on it is
                # the fore-aft swing that sets the corridor width, so shrinking only the
                # lateral offset (as EXP-001 did) cannot help a sidling robot.
                tgt = np.stack([
                    path[active][:, None, 0] + off[:, :, 0] * (1.0 - k),
                    nom["posed_joints"][active][:, arm_idx, 1],
                    path[active][:, None, 1] + off[:, :, 2] * (1.0 - k),
                ], -1)
                spec = ConstraintSpec(root_xz=path, heading=np.deg2rad(phi) * w,
                                      root_y=np.full(T, NOMINAL_PELVIS),
                                      pos_frames=active, pos_joints=arm_idx, pos_targets=tgt)
            conds.append((phi, k, prompt, s, spec))

    from scipy.spatial.transform import Rotation
    for i in range(0, len(conds), 8):
        chunk = conds[i:i + 8]
        outs = runner.generate([c[2] for c in chunk], [c[4] for c in chunk], T,
                               args.diffusion_steps, seed=chunk[0][3])
        for (phi, k, prompt, s, spec), o in zip(chunk, outs):
            q = runner.to_qpos(o)
            hw = max(body.half_width(x, lateral) for x in q[mid])
            top = max(body.top_height(x) for x in q[mid])
            yaw = Rotation.from_quat(q[:, 3:7][:, [1, 2, 3, 0]]).as_euler("zyx")[:, 0]
            rep = body.trajectory_report(q)
            got = np.stack([q[:, 1], q[:, 0]], -1)
            rows.append({
                "sidle_deg": phi, "tuck": k, "prompt": prompt, "seed": s,
                "halfwidth_m": float(hw), "top_m": float(top),
                "d_halfwidth_cm": float(100 * (hw - controls[s]["hw"])),
                "d_top_cm": float(100 * (top - controls[s]["top"])),
                "achieved_yaw_deg": float(np.rad2deg(yaw[mid]).mean()),
                "min_slot_width_m": float(2 * hw),
                "travel_m": float(q[-1, 0] - q[0, 0]),
                "path_err_mean_m": float(np.linalg.norm(got - spec.root_xz, axis=-1).mean()),
                "foot_floor_pen_max_m": rep["max_foot_floor_penetration_m"],
            })
            np.save(out / f"qpos_phi{phi}_k{k:g}_{'side' if prompt is PROMPT_SIDE else 'walk'}_s{s}.npy",
                    q.astype(np.float32))

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ctrl_hw = float(np.mean([c["hw"] for c in controls.values()]))
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp001b_min_halfwidth", "model": runner.model_name,
                   "fps": fps, "duration_s": DURATION, "seeds": SEEDS,
                   "arm_joints": ARM_JOINTS, "grid": [list(g[:2]) + [g[2]] for g in GRID],
                   "control_halfwidth_m": ctrl_hw, "n_rows": len(rows),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"control half-width {ctrl_hw*100:.1f} cm -> slot {200*ctrl_hw:.0f} cm needed")
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
