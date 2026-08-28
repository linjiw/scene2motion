"""EXP-001c: how tall a floor obstacle can the frozen prior step over?

The third body mode the planner needs. Unlike duck (one scalar channel) and tuck (arm
targets), stepping over is PHASE-DEPENDENT: the foot that must clear the obstacle is
whichever one happens to be in swing as the robot arrives, and lifting a foot that is
currently bearing weight asks the prior for something it will refuse or stumble through.

So the swing phase is read off a matched control clip (ARDY returns `foot_contacts` as
[L_heel, L_toe, R_heel, R_toe]), the crossing frame is found from where the control's pelvis
passes the obstacle, and each leg is lifted only over the frames it is actually airborne,
across the two swings that bracket the crossing. Lifting just the nearest swing -- the
obvious first implementation, and the one this experiment originally used -- measures ~0 cm
at every requested lift height, because the trailing foot then walks straight through the
obstacle. The failure looks exactly like "the constraint channel does not work", which it
is not: the targets are followed, and in fact overshot (0.19 m requested -> 0.27 m achieved).

The metric is deliberately operational rather than kinematic: for each generated motion we
binary-search the TALLEST floor box at the crossing point that the whole body still clears,
using G1's own MuJoCo collision primitives. That number is exactly the `max_step` the
planner needs, and it cannot be inflated by measuring a toe height that the shin then
collides through.

Three conditions, to separate what the text prompt buys from what the constraint channel
buys:
  prompt   "A person steps over an obstacle." with no foot targets
  targets  neutral walking prompt, swing-foot height targets only
  both     prompt and targets
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
from scene2motion.scenes import Box, Scene  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402

PROMPT_WALK = "A person walks forward."
PROMPT_STEP = "A person steps over an obstacle."
DURATION, SPEED, NOMINAL_PELVIS = 8.0, 0.95, 0.78
OBSTACLE_X = 3.8          # where the box sits, world metres along travel
SEEDS = [0, 1, 2, 3]
LIFTS = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55]   # requested extra swing-foot height, metres

LEG_JOINTS = {"left": ["left_knee_skel", "left_ankle_pitch_skel",
                       "left_ankle_roll_skel", "left_toe_base"],
              "right": ["right_knee_skel", "right_ankle_pitch_skel",
                        "right_ankle_roll_skel", "right_toe_base"]}


def probe_box(body_free: G1Body, qpos: np.ndarray, x: float,
              hi: float = 0.60, tol: float = 0.005) -> float:
    """Tallest floor box at `x` (0.25 m deep, corridor-spanning) the motion still clears."""
    def clears(h: float) -> bool:
        if h <= 0:
            return True
        sc = Scene("probe", "probe",
                   [Box((x, 0.0, h / 2), (0.125, 1.4, h / 2), "probe_box")],
                   start=(0, 0), goal=(8, 0))
        return G1Body(sc).trajectory_report(qpos)["collision_free"]

    if not clears(tol):
        return 0.0
    if clears(hi):
        return hi
    lo = tol
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if clears(mid):
            lo = mid
        else:
            hi = mid
    return lo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp001c")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DURATION * fps)
    t = np.arange(T)
    path = np.stack([np.zeros(T), SPEED * t / fps], -1)
    body = G1Body(None)
    jn = runner.joint_names
    leg_idx = {k: np.array([jn.index(j) for j in v]) for k, v in LEG_JOINTS.items()}

    rows = []
    for s in SEEDS:
        ctrl_spec = ConstraintSpec(root_xz=path, heading=np.zeros(T),
                                   root_y=np.full(T, NOMINAL_PELVIS))
        ctrl = runner.generate([PROMPT_WALK], [ctrl_spec], T, args.diffusion_steps, seed=s)[0]
        q_ctrl = runner.to_qpos(ctrl)
        base_h = probe_box(body, q_ctrl, OBSTACLE_X)
        rows.append({"cond": "control", "lift": 0.0, "seed": s,
                     "max_box_h_m": base_h, "travel_m": float(q_ctrl[-1, 0] - q_ctrl[0, 0]),
                     "foot_pen_m": body.trajectory_report(q_ctrl)["max_foot_floor_penetration_m"]})

        # crossing frame: where the control's pelvis passes the obstacle
        cross = int(np.argmin(np.abs(q_ctrl[:, 0] - OBSTACLE_X)))
        contacts = ctrl["foot_contacts"]              # (T, 4) L_heel L_toe R_heel R_toe

        # BOTH legs must clear a corridor-spanning box, over the two swings that bracket the
        # crossing. Lifting only the nearest swing (the obvious first implementation) leaves
        # the trailing foot to walk straight through the obstacle, and measures ~0 cm no
        # matter how high the lift is requested.
        W = int(1.2 * fps)
        per_frame: dict[int, list[tuple[str, float]]] = {}
        for sd, pair in (("left", (0, 2)), ("right", (2, 4))):
            air = np.where(~contacts[:, pair[0]:pair[1]].any(-1))[0]
            for f in air[np.abs(air - cross) <= W]:
                per_frame.setdefault(int(f), []).append(
                    (sd, float(np.exp(-0.5 * ((f - cross) / (0.5 * W)) ** 2))))
        near = np.array(sorted(per_frame)) if per_frame else np.arange(
            max(0, cross - int(0.2 * fps)), min(T, cross + int(0.2 * fps)))
        both_idx = np.concatenate([leg_idx["left"], leg_idx["right"]])
        side = "both"

        conds = []
        for lift in LIFTS:
            for prompt, tag in ((PROMPT_STEP, "prompt"), (PROMPT_WALK, "targets"),
                                (PROMPT_STEP, "both")):
                if lift == 0.0 and tag != "prompt":
                    continue
                if tag == "prompt":
                    spec = ConstraintSpec(root_xz=path, heading=np.zeros(T),
                                          root_y=np.full(T, NOMINAL_PELVIS))
                else:
                    tgt = np.zeros((len(near), len(both_idx), 3))
                    for i, f in enumerate(near):
                        nom, root = ctrl["posed_joints"][f], ctrl["smooth_root_pos"][f]
                        for j, ji in enumerate(both_idx):
                            tgt[i, j, 0] = path[f, 0] + (nom[ji, 0] - root[0])
                            tgt[i, j, 2] = path[f, 1] + (nom[ji, 2] - root[2])
                            tgt[i, j, 1] = nom[ji, 1]
                        for sd, ph in per_frame.get(int(f), []):
                            for j, ji in enumerate(both_idx):
                                if ji in leg_idx[sd]:
                                    tgt[i, j, 1] = nom[ji, 1] + lift * ph
                    spec = ConstraintSpec(root_xz=path, heading=np.zeros(T),
                                          root_y=np.full(T, NOMINAL_PELVIS),
                                          pos_frames=near, pos_joints=both_idx, pos_targets=tgt)
                conds.append((tag, lift, prompt, spec))

        for i in range(0, len(conds), 8):
            chunk = conds[i:i + 8]
            outs = runner.generate([c[2] for c in chunk], [c[3] for c in chunk], T,
                                   args.diffusion_steps, seed=s)
            for (tag, lift, prompt, spec), o in zip(chunk, outs):
                q = runner.to_qpos(o)
                rep = body.trajectory_report(q)
                rows.append({"cond": tag, "lift": lift, "seed": s, "swing_side": side,
                             "max_box_h_m": probe_box(body, q, OBSTACLE_X),
                             "travel_m": float(q[-1, 0] - q[0, 0]),
                             "foot_pen_m": rep["max_foot_floor_penetration_m"],
                             "path_err_m": float(np.linalg.norm(
                                 np.stack([q[:, 1], q[:, 0]], -1) - spec.root_xz, axis=-1).mean())})
                np.save(out / f"qpos_{tag}_l{lift:g}_s{s}.npy", q.astype(np.float32))
        print(f"seed {s} ({side} swing) done, {time.time()-t0:.0f}s", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp001c_step_over", "model": runner.model_name,
                   "fps": fps, "obstacle_x": OBSTACLE_X, "lifts": LIFTS, "seeds": SEEDS,
                   "n_rows": len(rows), "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
