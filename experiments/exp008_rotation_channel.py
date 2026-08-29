"""EXP-008: we have been steering the body through the channel the decoder does not pose from.

The discovery
-------------
`ardy/motion_rep/reps/ardy_motionrep.py:373`, in `inverse`:

    if posed_joints_from == "rotations":        # <- the DEFAULT
        _, posed_joints, _ = fk(local_rot_mats, root_positions, self.skeleton)

The pose ARDY returns is FORWARD KINEMATICS FROM THE ROTATION CHANNEL.  `local_joints_positions`
is never read when decoding a pose.  Yet every tuck and every step-over this project has ever
requested was written into `global_joints_positions` and nothing else -- an INDIRECT request,
honoured only insofar as the denoiser chooses to make the rotation block agree with the position
block it was conditioned on.

That is a live alternative explanation for the project's central measurement.  "The frozen prior
has low control bandwidth for body adaptation" and "we asked through the wrong channel" predict
the same 0.68-sigma tuck and the same 44-point validity cost for a lift.  They are not the same
claim, and only one of them is about ARDY.

The test
--------
One straight-line walk, no obstacle, three ways of asking for the same thing -- bring the arms
in -- at matched amplitude ladders and matched seeds:

    POS      the shipped mechanism: shrink the arms' world positions by (1 - tuck)
    ROT-IN   rotate the arm chain's GLOBAL rotations about the forward axis, adducting
    ROT-OUT  the same rotation with the opposite sign

ROT-OUT is the control that makes ROT-IN interpretable.  If the rotation channel is honoured,
the two signs must move the half-width in OPPOSITE directions; if both do nothing, the channel
is not reaching the body; and if both narrow the robot, the effect is not the rotation but the
mere presence of a constraint, which would invalidate the comparison.

Reported per arm, all paired against an unconstrained control at the SAME seed
-----------------------------------------------------------------------------
    d_halfwidth     the effect, in mm
    sd_across_seeds the noise the effect has to beat
    |effect| / sd   ADDRESSABILITY -- the number that decides whether a planner may rely on it
    P(valid)        goal reached and no self-inconsistency; what the request costs
    top / pelvis    a check that a width request did not silently buy width by ducking

The prediction, recorded before the run: the rotation channel gives a LARGER effect per unit of
noise than the position channel.  If it does not -- if a direct request to the channel the
decoder poses from is no better than an indirect one -- then the low bandwidth really is the
prior, the capability audit stands as written, and that is a much stronger result than it is
now, because the obvious objection will have been tested and closed.
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
from scene2motion.morphology import _signed_extents, heading_normal  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED, WINDOW = 0.9, (0.42, 0.58)          # EXP-001's own measure window
TUCKS = [0.0, 0.35, 0.70, 0.85]            # position-channel amplitudes
DEGS = [0.0, 10.0, 20.0, 30.0, 40.0]       # rotation-channel amplitudes


def arm_chain(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) ARDY joint indices for everything distal to each shoulder."""
    def side(pre):
        return np.array([i for i, n in enumerate(joint_names)
                         if n.startswith(pre) and any(k in n for k in
                                                      ("shoulder", "elbow", "wrist", "hand"))])
    return side("left_"), side("right_")


def axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def halfwidth_window(body: G1Body, qpos: np.ndarray) -> float:
    """Mean of max(left, right) extent over the measure window, on the robot's own axis."""
    T = len(qpos)
    w = range(int(WINDOW[0] * T), int(WINDOW[1] * T))
    v = []
    for t in w:
        L, R = _signed_extents(body, qpos[t], heading_normal(qpos[t]))
        v.append(max(L, R))
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp008")
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
    # a beam far overhead: G1Body needs a scene, and nothing here should ever touch it
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(400, 400 + args.n_seeds))
    left, right = arm_chain(runner.joint_names)
    print(f"arm chain: {len(left)} left + {len(right)} right joints of "
          f"{len(runner.joint_names)}", flush=True)

    # straight walk down +Z, constant heading, nominal pelvis height
    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading = np.zeros(T)
    root_y = np.full(T, 0.78)
    base = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y)

    ctrl = runner.generate([PROMPT] * len(seeds), [base] * len(seeds), T,
                           args.diffusion_steps, seeds=seeds)
    ctrl_q = [runner.to_qpos(o) for o in ctrl]
    ctrl_w = [halfwidth_window(body, q) for q in ctrl_q]
    ctrl_top = [float(np.max([body.top_height(q[t]) for t in
                              range(int(WINDOW[0] * T), int(WINDOW[1] * T))])) for q in ctrl_q]
    print(f"control half-width {np.mean(ctrl_w):.4f} m "
          f"(sd across seeds {np.std(ctrl_w):.4f})  ({time.time()-t0:.0f}s)", flush=True)

    # the nominal global rotations the rotation requests are built FROM
    grm = [np.asarray(o["global_rot_mats"]) for o in ctrl]
    print(f"global_rot_mats {grm[0].shape}", flush=True)

    rows = []

    def run(label: str, specs: list[ConstraintSpec], amp: float) -> None:
        outs = runner.generate([PROMPT] * len(seeds), specs, T, args.diffusion_steps,
                               seeds=seeds)
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            rep = body.trajectory_report(q)
            hw = halfwidth_window(body, q)
            top = float(np.max([body.top_height(q[t]) for t in
                                range(int(WINDOW[0] * T), int(WINDOW[1] * T))]))
            rows.append({"arm": label, "amp": amp, "seed": seeds[s],
                         "halfwidth_m": hw, "d_halfwidth_m": hw - ctrl_w[s],
                         "top_m": top, "d_top_m": top - ctrl_top[s],
                         "pelvis_mean_m": float(q[:, 2].mean()),
                         "travel_m": float(np.linalg.norm(q[-1, :2] - q[0, :2])),
                         "coll_free": bool(rep["collision_free"]),
                         "foot_floor_pen_m": float(rep["mean_foot_floor_penetration_m"])})
        d = [r["d_halfwidth_m"] for r in rows if r["arm"] == label and r["amp"] == amp]
        print(f"  {label:10s} amp {amp:5.2f}  d_halfwidth {1000*np.mean(d):+7.1f} mm  "
              f"sd {1000*np.std(d):5.1f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- POS: the shipped mechanism -----------------------------------------------------
    nominal = ctrl[0]
    nom_j = np.asarray(nominal["posed_joints"])
    nom_r = np.asarray(nominal["smooth_root_pos"])
    step = max(1, int(0.12 * fps))
    frames = np.arange(int(WINDOW[0] * T) - 20, int(WINDOW[1] * T) + 20, step)
    frames = frames[(frames >= 0) & (frames < T)]
    joints = np.concatenate([left, right])
    for tuck in TUCKS:
        off = nom_j[frames][:, joints, :] - nom_r[frames][:, None, :]
        tgt = np.stack([root_xz[frames][:, None, 0] + off[:, :, 0] * (1 - tuck),
                        nom_j[frames][:, joints, 1],
                        root_xz[frames][:, None, 1] + off[:, :, 2] * (1 - tuck)], -1)
        spec = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                              pos_frames=frames, pos_joints=joints, pos_targets=tgt)
        run("POS", [spec] * len(seeds), tuck)

    # ---- ROT: the channel the decoder poses from ----------------------------------------
    # Heading is 0 (walking down +Z), so the forward axis is +Z and adduction is a rotation
    # about it. The whole chain distal to the shoulder gets the SAME global rotation, which is
    # a rigid swing of the arm about the shoulder rather than a per-joint re-articulation.
    fwd = np.array([0.0, 0.0, 1.0])
    for sign, label in ((+1, "ROT-IN"), (-1, "ROT-OUT")):
        for deg in DEGS:
            if deg == 0.0 and label == "ROT-OUT":
                continue                                  # identical to ROT-IN at 0
            specs = []
            for s in range(len(seeds)):
                tgt = np.empty((len(frames), len(joints), 3, 3))
                RL = axis_rotation(fwd, +sign * np.deg2rad(deg))
                RR = axis_rotation(fwd, -sign * np.deg2rad(deg))
                for i, f in enumerate(frames):
                    for k, j in enumerate(joints):
                        R = RL if j in set(left.tolist()) else RR
                        tgt[i, k] = R @ grm[s][f, j]
                specs.append(ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                                            rot_frames=frames, rot_joints=joints,
                                            rot_targets=tgt))
            run(label, specs, deg)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # ---- report -------------------------------------------------------------------------
    print(f"\n{'arm':10s} {'amp':>6s} {'d_halfwidth':>13s} {'sd':>7s} {'|eff|/sd':>9s} "
          f"{'d_top':>8s} {'travel':>8s} {'foot pen':>9s}")
    print("-" * 78)
    summary = {"experiment": "exp008_rotation_channel", "n_seeds": args.n_seeds,
               "control_halfwidth_m": float(np.mean(ctrl_w)),
               "control_sd_m": float(np.std(ctrl_w)), "arms": {}}
    for label in ("POS", "ROT-IN", "ROT-OUT"):
        for amp in sorted({r["amp"] for r in rows if r["arm"] == label}):
            R = [r for r in rows if r["arm"] == label and r["amp"] == amp]
            d = np.array([r["d_halfwidth_m"] for r in R])
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
            ratio = abs(d.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
            summary["arms"].setdefault(label, {})[str(amp)] = {
                "d_halfwidth_mm": 1000 * float(d.mean()), "sd_mm": 1000 * sd,
                "effect_over_sd": ratio,
                "travel_m": float(np.mean([r["travel_m"] for r in R])),
                "d_top_mm": 1000 * float(np.mean([r["d_top_m"] for r in R])),
                "foot_pen_m": float(np.mean([r["foot_floor_pen_m"] for r in R]))}
            print(f"{label:10s} {amp:6.2f} {1000*d.mean():+13.1f} {1000*sd:7.1f} "
                  f"{ratio:9.2f} {1000*np.mean([r['d_top_m'] for r in R]):+8.1f} "
                  f"{np.mean([r['travel_m'] for r in R]):8.2f} "
                  f"{np.mean([r['foot_floor_pen_m'] for r in R]):9.4f}")
    print("\n  |eff|/sd is the addressability number: an effect a planner may rely on has to be\n"
          "  large against the spread of the clips that deliver it, not merely against zero.\n"
          "  ROT-OUT is the sign control -- if it does not move the half-width the OTHER way,\n"
          "  the rotation channel is not reaching the body and no comparison can be drawn.\n"
          "  d_top guards against buying width by ducking; travel guards against buying it by\n"
          "  stopping.")
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
