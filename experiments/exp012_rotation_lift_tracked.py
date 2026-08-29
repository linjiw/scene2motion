"""EXP-012: does a lift commanded through the ROTATION channel survive the tracker?

The hypothesis, which falls out of two experiments that were never compared
------------------------------------------------------------------------------
EXP-011 concluded that lift is not executable: 0 of 8 under SONIC, twice, with `accel_dist` at
28.3 against a neutral walk's 2.04.  That lift was commanded through `global_joints_positions` —
the same mechanism the planner has always used.

EXP-010 measured what that mechanism actually does to the body.  The position channel delivers
foot height at 10.64 sigma, which is excellent, but it buys up to **122 mm of that height by
raising the pelvis**.  The rotation channel reaches +271.8 mm of foot clearance with the pelvis
essentially still (-7.1 mm) and a seed spread of 6.6 mm.

A whole-body vertical translation is a HOP.  A tracking controller trained on human locomotion
has every reason to reject a hop and no particular reason to reject a leg swing.  So EXP-011 may
have measured "the position channel's way of lifting is not executable" and reported it as "lift
is not executable" — exactly the class of error this project has made four times already, where
a property of the encoding is attributed to the prior or, here, to the robot's dynamics.

If a rotation-commanded lift tracks, the executable repertoire is not one axis but two, the
`low_obstacle` refusal may be premature, and the report's central negative finding needs
rewriting.  If it does not track, the finding is confirmed through a second, independent
mechanism and is much stronger for it.

Design
------
One straight walk, no obstacle, matched per-sample seeds, four bodies:

    neutral              the control
    lift-POS             EXP-011's mechanism, re-run here so the comparison is within one run
    lift-ROT-20          leg chain rotated 20 deg off vertical during the swing
    lift-ROT-30          the same at 30 deg, EXP-010's best pelvis-still setting

Both lift mechanisms are gated on the SAME airborne frames from the control clip's own
`foot_contacts`, so the comparison is of channels and not of gating.  Each body is exported to
its own motion-library pickle and tracked separately, because SONIC's eval callback writes only
a pooled aggregate and a mixed run could not attribute an outcome to a body.

The measurement that decides it is `success` against neutral, with `d_pelvis` reported alongside
to confirm the mechanism did what EXP-010 says it does: if lift-ROT raises the pelvis as much as
lift-POS, the two arms are not actually different and nothing can be concluded.
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
from scene2motion.scenes import BUILDERS  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp011_tracked_addressability import run_sonic  # noqa: E402

PROMPT = "A person walks forward."
SPEED = 0.9
BODIES = ["neutral", "lift-POS", "lift-ROT-20", "lift-ROT-30"]


def axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, float) / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp012")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--diffusion_steps", type=int, default=5)
    ap.add_argument("--lift", type=float, default=0.35)
    ap.add_argument("--timeout_s", type=int, default=2400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(args.duration * fps)
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(1100, 1100 + args.n_seeds))
    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading = np.zeros(T)
    root_y = np.full(T, 0.78)
    base = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y)

    ctrl = runner.generate([PROMPT] * len(seeds), [base] * len(seeds), T,
                           args.diffusion_steps, seeds=seeds)
    ctrl_q = [runner.to_qpos(o) for o in ctrl]
    win = range(int(0.35 * T), int(0.65 * T))
    ctrl_pelvis = [float(q[win.start:win.stop, 2].mean()) for q in ctrl_q]

    nominal = ctrl[0]
    nom_j = np.asarray(nominal["posed_joints"])
    nom_r = np.asarray(nominal["smooth_root_pos"])
    grm = [np.asarray(o["global_rot_mats"]) for o in ctrl]
    contacts = np.asarray(nominal["foot_contacts"])
    legs = np.array([i for i, n in enumerate(runner.joint_names)
                     if any(k in n for k in ("hip", "knee", "ankle", "toe"))])
    air = {"L": ~contacts[:, 0:2].any(-1), "R": ~contacts[:, 2:4].any(-1)}
    frames = np.array(sorted({int(f) for s in air for f in np.where(air[s])[0]
                              if win.start <= f < win.stop}))
    step = max(1, int(0.12 * fps))
    frames = frames[::step] if len(frames) > step else frames
    if not len(frames):
        raise SystemExit("no airborne frames in the window")
    print(f"{len(frames)} airborne target frames, {len(legs)} leg joints", flush=True)

    def spec_for(label: str, seed_i: int) -> ConstraintSpec:
        if label == "neutral":
            return base
        if label == "lift-POS":
            off = nom_j[frames][:, legs, :] - nom_r[frames][:, None, :]
            tgt = np.stack([root_xz[frames][:, None, 0] + off[:, :, 0],
                            nom_j[frames][:, legs, 1] + args.lift,
                            root_xz[frames][:, None, 1] + off[:, :, 2]], -1)
            return ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                                  pos_frames=frames, pos_joints=legs, pos_targets=tgt)
        deg = float(label.rsplit("-", 1)[1])
        R = axis_rotation(np.array([1.0, 0.0, 0.0]), -np.deg2rad(deg))   # EXP-010's ROT- sign
        tgt = np.empty((len(frames), len(legs), 3, 3))
        for i, f in enumerate(frames):
            for k, j in enumerate(legs):
                tgt[i, k] = R @ grm[seed_i][f, j]
        return ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                              rot_frames=frames, rot_joints=legs, rot_targets=tgt)

    clips, pelvis = {}, {}
    for label in BODIES:
        outs = runner.generate([PROMPT] * len(seeds),
                               [spec_for(label, i) for i in range(len(seeds))], T,
                               args.diffusion_steps, seeds=seeds)
        d = []
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            clips[f"{label}__s{seeds[s]}"] = q
            d.append(float(q[win.start:win.stop, 2].mean()) - ctrl_pelvis[s])
        pelvis[label] = float(np.mean(d))
        write_motion_pkl({k: v for k, v in clips.items() if k.startswith(label + "__")},
                         out / f"{label}.pkl", fps=int(round(fps)), mj_model=body.model)
        print(f"  {label:12s} d_pelvis {1000*pelvis[label]:+7.1f} mm  ({time.time()-t0:.0f}s)",
              flush=True)
    del runner

    results = {}
    for label in BODIES:
        rc, log = run_sonic(out / f"{label}.pkl", out / label, args.n_seeds, args.timeout_s)
        (out / f"sonic_{label}.log").write_text(log)
        m = {"returncode": rc, "d_pelvis_mm": 1000 * pelvis[label]}
        for line in log.splitlines():
            if line.startswith("Success Rate:"):
                m["success_rate"] = float(line.split(":")[1])
            if line.startswith("Progress Rate:"):
                m["progress_rate"] = float(line.split(":")[1])
            if line.startswith("All:"):
                for t in line.split("\t"):
                    t = t.strip()
                    if t.split(":")[0] in ("mpjpe_l", "accel_dist", "mpjpe_l_legs"):
                        m[t.split(":")[0]] = float(t.split(":")[1])
        results[label] = m
        print(f"  {label:12s} rc {rc}  success {m.get('success_rate', float('nan')):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    b = results.get("neutral", {}).get("success_rate", float("nan"))
    print(f"\n{'body':14s} {'d_pelvis':>10s} {'success':>8s} {'vs neutral':>11s} "
          f"{'progress':>9s} {'accel':>8s} {'mpjpe_l_legs':>13s}")
    print("-" * 78)
    for label in BODIES:
        m = results[label]
        print(f"{label:14s} {m['d_pelvis_mm']:+10.1f} "
              f"{m.get('success_rate', float('nan')):8.3f} "
              f"{m.get('success_rate', float('nan')) - b:+11.3f} "
              f"{m.get('progress_rate', float('nan')):9.3f} "
              f"{m.get('accel_dist', float('nan')):8.2f} "
              f"{m.get('mpjpe_l_legs', float('nan')):13.1f}")
    print("\n  d_pelvis is the manipulation check: lift-POS should raise the body and lift-ROT\n"
          "  should not.  If they are the same, the two arms are not different requests and no\n"
          "  conclusion about the channel can be drawn from the success column.")
    json.dump({"experiment": "exp012_rotation_lift_tracked", "n_seeds": args.n_seeds,
               "lift_m": args.lift, "diffusion_steps": args.diffusion_steps,
               "results": results, "wall_clock_s": round(time.time() - t0, 1)},
              open(out / "receipt.json", "w"), indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
