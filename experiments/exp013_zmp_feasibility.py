"""EXP-013: is the lift reference dynamically infeasible, or merely unfamiliar to SONIC?

The caveat this exists to close
-------------------------------
EXP-011 and EXP-012 agree that a step-over lift does not track: 0/8 through the position channel
and 0/8 through the rotation channel, the latter with the pelvis held still.  But SONIC is
trained on retargeted human motion capture, so an OUT-OF-DISTRIBUTION reference and an
INFEASIBLE one fail identically from the outside.  Every statement in the report about lift
carries that ambiguity, and it is the largest unresolved caveat in the project.

A controller-independent test
-----------------------------
The zero-moment point.  For a motion on flat ground with only foot contacts, the centre of
pressure required to produce the observed centre-of-mass acceleration must lie inside the convex
hull of the contacting feet.  If it does not, the motion cannot be produced by ANY controller
using those contacts -- it would need a force the ground cannot supply, because the ground can
only push.  That is a statement about physics and the reference, with no learned policy in it.

    ZMP_x = COM_x - COM_z * a_x / (a_z + g)          (flat ground at z = 0)

computed from the whole-body COM of the exact MuJoCo mass model, with acceleration by second
difference of the 25 fps kinematics.

What the comparison controls for
--------------------------------
Finite-differenced acceleration from 25 fps kinematics is noisy, and the absolute violation rate
is therefore not meaningful on its own.  What is meaningful is the SAME computation applied to
the neutral walk, the duck (which EXP-011 shows tracks at 0.75-1.00) and the lift (0/8): the
neutral and duck rows calibrate what this measure reads on motion that is known to be
executable, and the lift row is only interpretable against them.

Reading it
----------
  lift violates far more than duck  ->  the reference is dynamically infeasible and the tracker
                                        was right to refuse it; the caveat closes
  lift violates about as much as duck -> the reference is producible and SONIC's failure is a
                                        controller/distribution matter; the caveat stands and
                                        the report must keep it
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

PROMPT, SPEED, G = "A person walks forward.", 0.9, 9.81
BODIES = [("neutral", 0.00, 0.00), ("duck", 0.35, 0.00),
          ("duck-deep", 0.50, 0.00), ("lift", 0.00, 0.35)]


def com_series(body: G1Body, qpos: np.ndarray) -> np.ndarray:
    """(T, 3) whole-body centre of mass from the exact mass model."""
    m = body.model.body_mass
    tot = float(m.sum())
    out = np.empty((len(qpos), 3))
    for t, q in enumerate(qpos):
        body.fk(q)
        out[t] = (body.data.xipos * m[:, None]).sum(0) / tot
    return out


def foot_support(body: G1Body, qpos_t: np.ndarray, thresh: float = 0.03):
    """xy points of foot geoms whose lowest surface is within `thresh` of the floor."""
    body.fk(qpos_t)
    pts = []
    for g in body.robot_geoms:
        if "foot" not in body.geom_name[g]:
            continue
        p = body.data.geom_xpos[g]
        r = float(body.model.geom_size[g, 0])
        if p[2] - r < thresh:
            pts.append(p[:2])
    return np.array(pts) if pts else np.zeros((0, 2))


def inside(pt: np.ndarray, pts: np.ndarray, pad: float = 0.02) -> bool:
    """Is `pt` inside the convex hull of `pts`, padded outward by `pad`?"""
    if len(pts) == 0:
        return False
    if len(pts) < 3:
        # one foot contact or a line: fall back to distance from the segment/point
        d = np.min(np.linalg.norm(pts - pt, axis=1))
        return bool(d <= pad + 0.10)
    from scipy.spatial import ConvexHull, Delaunay
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        return bool(np.min(np.linalg.norm(pts - pt, axis=1)) <= pad + 0.10)
    c = hull.mean(0)
    hull = c + (hull - c) * (1.0 + pad / max(np.linalg.norm(hull - c, axis=1).mean(), 1e-6))
    try:
        return bool(Delaunay(hull).find_simplex(pt) >= 0)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp013")
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--diffusion_steps", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(args.duration * fps)
    dt = 1.0 / fps
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(1200, 1200 + args.n_seeds))
    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading = np.zeros(T)
    win = range(int(0.35 * T), int(0.65 * T))

    ref = runner.generate([PROMPT],
                          [ConstraintSpec(root_xz=root_xz, heading=heading,
                                          root_y=np.full(T, 0.78))],
                          T, args.diffusion_steps, seeds=[seeds[0]])[0]
    nom_j = np.asarray(ref["posed_joints"])
    nom_r = np.asarray(ref["smooth_root_pos"])
    legs = np.array([i for i, n in enumerate(runner.joint_names)
                     if any(k in n for k in ("knee", "ankle", "toe"))])
    step = max(1, int(0.12 * fps))
    frames = np.arange(win.start, win.stop, step)

    rows = []
    for label, dip, lift in BODIES:
        root_y = np.full(T, 0.78 - dip)
        pf = pj = pt_ = None
        if lift > 0:
            off = nom_j[frames][:, legs, :] - nom_r[frames][:, None, :]
            pt_ = np.stack([root_xz[frames][:, None, 0] + off[:, :, 0],
                            nom_j[frames][:, legs, 1] + lift,
                            root_xz[frames][:, None, 1] + off[:, :, 2]], -1)
            pf, pj = frames, legs
        spec = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                              pos_frames=pf, pos_joints=pj, pos_targets=pt_)
        outs = runner.generate([PROMPT] * len(seeds), [spec] * len(seeds), T,
                               args.diffusion_steps, seeds=seeds)
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            com = com_series(body, q)
            a = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
            viol, nsup, zmp_err = [], [], []
            for t in win:
                az = a[t, 2] + G
                if az <= 1e-3:                      # free fall: no ZMP exists
                    viol.append(True)
                    nsup.append(0)
                    zmp_err.append(np.nan)
                    continue
                zmp = com[t, :2] - com[t, 2] * a[t, :2] / az
                sup = foot_support(body, q[t])
                ok = inside(zmp, sup)
                viol.append(not ok)
                nsup.append(len(sup))
                zmp_err.append(np.min(np.linalg.norm(sup - zmp, axis=1))
                               if len(sup) else np.nan)
            rows.append({"body": label, "seed": seeds[s],
                         "zmp_violation_frac": float(np.mean(viol)),
                         "mean_contacts": float(np.mean(nsup)),
                         "frac_no_contact": float(np.mean(np.array(nsup) == 0)),
                         "com_accel_rms": float(np.sqrt((a[win.start:win.stop] ** 2).sum(1).mean())),
                         "zmp_dist_p90": float(np.nanpercentile(zmp_err, 90))})
        v = [r["zmp_violation_frac"] for r in rows if r["body"] == label]
        print(f"  {label:10s} ZMP violation {np.mean(v):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n{'body':12s} {'ZMP viol':>9s} {'no contact':>11s} {'contacts':>9s} "
          f"{'COM accel rms':>14s} {'tracked (EXP-011/12)':>21s}")
    print("-" * 82)
    tracked = {"neutral": "0.875 / 1.000", "duck": "0.750", "duck-deep": "0.625",
               "lift": "0.000"}
    summ = {}
    for label, _, _ in BODIES:
        R = [r for r in rows if r["body"] == label]
        summ[label] = {k: float(np.mean([r[k] for r in R]))
                       for k in ("zmp_violation_frac", "frac_no_contact", "mean_contacts",
                                 "com_accel_rms")}
        print(f"{label:12s} {summ[label]['zmp_violation_frac']:9.3f} "
              f"{summ[label]['frac_no_contact']:11.3f} {summ[label]['mean_contacts']:9.2f} "
              f"{summ[label]['com_accel_rms']:14.2f} {tracked.get(label,''):>21s}")
    print("\n  Neutral and duck CALIBRATE this measure: both are known to track, so whatever they\n"
          "  read is what executable motion looks like under a finite-differenced ZMP test.\n"
          "  Lift is only interpretable against them.")
    json.dump({"experiment": "exp013_zmp_feasibility", "n_seeds": args.n_seeds,
               "bodies": summ, "wall_clock_s": round(time.time() - t0, 1)},
              open(out / "receipt.json", "w"), indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
