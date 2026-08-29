"""EXP-010: the step-over lift, through the channel the decoder poses from.

Why lift and not tuck
---------------------
EXP-008 answered the channel question for the ARM chain and got a clear result -- rotation
reaches |effect|/sd = 8.68 where position maxes at 1.57 and changes sign twice.  But it also
found that the direction a gap planner wants, NARROWING, is capped at 0.58 sigma by the robot's
own geometry, so a better channel does not buy a narrower G1.

Lift is the axis where a better channel could actually change what the robot can do.  It is the
only mechanism for stepping over a floor obstacle, `low_obstacle` is the family the conformal
envelope refuses outright, and the commanded-versus-realised matrix puts a lift request at
22 % invalid alone and 36-87 % invalid in combination.  If that is the position channel rather
than the prior, the step-over capability is recoverable; if it is the prior, `low_obstacle`
stays refused and the refusal is a real one.

The comparison
--------------
One straight walk, no obstacle, three ways of asking the swing leg to come up higher, at matched
seeds and matched control:

    POS      the shipped mechanism -- raise the leg joints' world heights by `lift`
    ROT+     rotate the swing leg's chain about the lateral axis (hip flexion)
    ROT-     the same rotation, opposite sign

Both mechanisms are gated on the SAME airborne frames, taken from the control clip's own
`foot_contacts`, because `_limb_targets` only lifts a leg that is already off the ground and a
comparison that let the rotation arm fire during stance would be measuring the gating, not the
channel.  As in EXP-008 the two signs are each other's control, and `channel_usage` asserts each
request actually reaches ARDY's mask before any of it is believed.

Reported, paired against an unconstrained control at the same seed:

    d_foot_max      change in the swing foot's peak height, in mm
    |effect|/sd     addressability -- what a planner may rely on
    P(valid)        goal reached and collision-free; what the request costs
    d_pelvis        guards against buying foot height by crouching
    travel          guards against buying it by stopping
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import (ConstraintSpec, build_conditions,  # noqa: E402
                                      channel_usage)
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED, WINDOW = 0.9, (0.35, 0.65)
LIFTS = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55]      # metres, EXP-001c's ladder
DEGS = [0.0, 10.0, 20.0, 30.0, 40.0]             # hip flexion


def leg_chain(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) ARDY joint indices for everything distal to each hip."""
    def side(pre):
        return np.array([i for i, n in enumerate(joint_names)
                         if n.startswith(pre) and any(k in n for k in
                                                      ("hip", "knee", "ankle", "toe"))])
    return side("left_"), side("right_")


def axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def foot_peak(body: G1Body, qpos: np.ndarray, window: range) -> float:
    """Highest either foot gets in the window, in metres above the floor."""
    feet = [g for g in body.robot_geoms if "foot" in body.geom_name[g]]
    if not feet:
        return float("nan")
    hi = []
    for t in window:
        body.fk(qpos[t])
        hi.append(float(body.data.geom_xpos[feet][:, 2].max()))
    return float(np.max(hi)) if hi else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp010")
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(args.duration * fps)
    win = range(int(WINDOW[0] * T), int(WINDOW[1] * T))
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(800, 800 + args.n_seeds))
    left, right = leg_chain(runner.joint_names)
    print(f"leg chain: {len(left)} left + {len(right)} right joints of "
          f"{len(runner.joint_names)}", flush=True)

    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading = np.zeros(T)
    root_y = np.full(T, 0.78)
    base = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y)

    ctrl = runner.generate([PROMPT] * len(seeds), [base] * len(seeds), T,
                           args.diffusion_steps, seeds=seeds)
    ctrl_q = [runner.to_qpos(o) for o in ctrl]
    ctrl_f = [foot_peak(body, q, win) for q in ctrl_q]
    ctrl_p = [float(q[win.start:win.stop, 2].mean()) for q in ctrl_q]
    print(f"control foot peak {np.mean(ctrl_f):.4f} m (sd {np.std(ctrl_f):.4f})  "
          f"({time.time()-t0:.0f}s)", flush=True)

    nominal = ctrl[0]
    nom_j = np.asarray(nominal["posed_joints"])
    nom_r = np.asarray(nominal["smooth_root_pos"])
    grm = [np.asarray(o["global_rot_mats"]) for o in ctrl]
    contacts = np.asarray(nominal["foot_contacts"])          # (T, 4) Lh Lt Rh Rt

    # Airborne frames per leg, from the CONTROL clip, shared by both mechanisms so the
    # comparison is of channels and not of gating.
    air = {"left": ~contacts[:, 0:2].any(-1), "right": ~contacts[:, 2:4].any(-1)}
    frames = np.array(sorted({int(f) for side in air for f in np.where(air[side])[0]
                              if win.start <= f < win.stop}))
    step = max(1, int(0.12 * fps))
    frames = frames[::step] if len(frames) > step else frames
    if len(frames) == 0:
        raise SystemExit("no airborne frames in the window -- nothing to lift")
    joints = np.concatenate([left, right])
    print(f"{len(frames)} airborne target frames in the window", flush=True)

    rows = []

    def audit(label: str, spec: ConstraintSpec, want: str) -> None:
        _, mask = build_conditions(runner.model, spec, runner.device)
        u = {k: v for k, v in channel_usage(runner.model, mask).items() if v}
        print(f"    channels written by {label:8s}: {u}", flush=True)
        if not u.get(want):
            raise SystemExit(f"{label} writes no {want}; the request never reaches ARDY and a "
                             f"null result would mean nothing.")

    def run(label: str, specs: list[ConstraintSpec], amp: float) -> None:
        outs = runner.generate([PROMPT] * len(seeds), specs, T, args.diffusion_steps,
                               seeds=seeds)
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            rep = body.trajectory_report(q)
            fp = foot_peak(body, q, win)
            rows.append({"arm": label, "amp": amp, "seed": seeds[s],
                         "foot_peak_m": fp, "d_foot_m": fp - ctrl_f[s],
                         "pelvis_m": float(q[win.start:win.stop, 2].mean()),
                         "d_pelvis_m": float(q[win.start:win.stop, 2].mean()) - ctrl_p[s],
                         "travel_m": float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                         "coll_free": bool(rep["collision_free"]),
                         "foot_floor_pen_m": float(rep["mean_foot_floor_penetration_m"])})
        d = [r["d_foot_m"] for r in rows if r["arm"] == label and r["amp"] == amp]
        print(f"  {label:8s} amp {amp:5.2f}  d_foot {1000*np.mean(d):+7.1f} mm  "
              f"sd {1000*np.std(d):5.1f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- POS: the shipped mechanism -----------------------------------------------------
    for lift in LIFTS:
        off = nom_j[frames][:, joints, :] - nom_r[frames][:, None, :]
        tgt = np.stack([root_xz[frames][:, None, 0] + off[:, :, 0],
                        nom_j[frames][:, joints, 1] + lift,
                        root_xz[frames][:, None, 1] + off[:, :, 2]], -1)
        spec = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                              pos_frames=frames, pos_joints=joints, pos_targets=tgt)
        if lift == LIFTS[1]:
            audit("POS", spec, "local_joints_positions")
        run("POS", [spec] * len(seeds), lift)

    # ---- ROT: hip flexion about the lateral axis ----------------------------------------
    lat = np.array([1.0, 0.0, 0.0])          # heading is 0, so +X is lateral, +Z forward
    for sign, label in ((+1, "ROT+"), (-1, "ROT-")):
        for deg in DEGS:
            if deg == 0.0 and label == "ROT-":
                continue
            specs = []
            for s in range(len(seeds)):
                tgt = np.empty((len(frames), len(joints), 3, 3))
                R = axis_rotation(lat, sign * np.deg2rad(deg))
                for i, f in enumerate(frames):
                    for k, j in enumerate(joints):
                        tgt[i, k] = R @ grm[s][f, j]
                specs.append(ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                                            rot_frames=frames, rot_joints=joints,
                                            rot_targets=tgt))
            if deg == DEGS[1] and label == "ROT+":
                audit("ROT", specs[0], "global_rot_data")
            run(label, specs, deg)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{'arm':8s} {'amp':>6s} {'d_foot':>10s} {'sd':>7s} {'|eff|/sd':>9s} "
          f"{'d_pelvis':>9s} {'travel':>8s} {'coll-free':>10s} {'foot pen':>9s}")
    print("-" * 82)
    summary = {"experiment": "exp010_lift_channel", "n_seeds": args.n_seeds,
               "control_foot_peak_m": float(np.mean(ctrl_f)),
               "control_sd_m": float(np.std(ctrl_f)), "n_frames": int(len(frames)),
               "arms": {}}
    for label in ("POS", "ROT+", "ROT-"):
        for amp in sorted({r["amp"] for r in rows if r["arm"] == label}):
            R = [r for r in rows if r["arm"] == label and r["amp"] == amp]
            d = np.array([r["d_foot_m"] for r in R])
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
            ratio = abs(d.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
            summary["arms"].setdefault(label, {})[str(amp)] = {
                "d_foot_mm": 1000 * float(d.mean()), "sd_mm": 1000 * sd,
                "effect_over_sd": ratio,
                "d_pelvis_mm": 1000 * float(np.mean([r["d_pelvis_m"] for r in R])),
                "travel_m": float(np.mean([r["travel_m"] for r in R])),
                "coll_free": float(np.mean([r["coll_free"] for r in R])),
                "foot_pen_m": float(np.mean([r["foot_floor_pen_m"] for r in R]))}
            print(f"{label:8s} {amp:6.2f} {1000*d.mean():+10.1f} {1000*sd:7.1f} {ratio:9.2f} "
                  f"{1000*np.mean([r['d_pelvis_m'] for r in R]):+9.1f} "
                  f"{np.mean([r['travel_m'] for r in R]):8.2f} "
                  f"{np.mean([r['coll_free'] for r in R]):10.2f} "
                  f"{np.mean([r['foot_floor_pen_m'] for r in R]):9.4f}")
    print("\n  d_pelvis guards against buying foot height by crouching -- a large negative "
          "there means the\n  robot lowered its body rather than raising its foot, and the "
          "step-over gains nothing.\n  The two rotation signs are each other's control: opposite "
          "signs must move the foot in\n  opposite directions or the channel is not reaching "
          "the leg.")
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
