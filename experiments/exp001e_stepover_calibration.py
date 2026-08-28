"""EXP-001e: recalibrate step-over with a real coverage bound.

Why this specific axis, now
---------------------------
`low_obstacle` is the worst family in the suite: 4/24 feasible, and EXP-005c found **0 of 20**
refusals recoverable even against a deliberately relaxed envelope. That reads as physics — the
prior simply cannot step over anything.

But look at where the number came from. `body_modes.json` gives `step_over.max_step = 0.028 m`,
and that is the **worst of four seeds** from EXP-001c, whose *mean* over the same condition was
0.158 m and whose best sample cleared 0.228 m. A worst-of-4 statistic on a quantity with that
much spread is not a bound, it is a draw, and it is the single number that declares the whole
family impossible.

EXP-001d showed exactly this pattern on the other two axes and changed both conclusions once
n was raised — including overturning a retraction. So the same treatment applies here before
`low_obstacle` is written off: sweep the requested swing-foot lift at **20 independent samples
per level**, and report a split conformal upper... no, a *lower* bound, since for step-over the
planner needs a height it can be confident of EXCEEDING. The valid one-sided bound is the
floor((n+1)*alpha)-th order statistic, so with n = 20 and alpha = 0.1 that is the 2nd smallest.

If the 90 % lower bound is materially above 2.8 cm, the family was written off on a sampling
artefact and the certified envelope should carry the calibrated value. If it is not, the
refusal is real and `low_obstacle` is an honest limitation of the frozen prior — which is
itself worth stating precisely rather than by implication.
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
from scene2motion.scenes import Box, Scene  # noqa: E402

PROMPT_WALK = "A person walks forward."
PROMPT_STEP = "A person steps over an obstacle."
DURATION, SPEED, NOMINAL_PELVIS = 8.0, 0.95, 0.78
OBSTACLE_X = 3.8
N_SAMPLES = 20
LIFTS = [0.0, 0.25, 0.40, 0.55]
LEG = {s: [f"{s}_knee_skel", f"{s}_ankle_pitch_skel", f"{s}_ankle_roll_skel",
           f"{s}_toe_base"] for s in ("left", "right")}


def conformal_lower(x, alpha: float = 0.1) -> float:
    """Split-conformal LOWER bound: the floor((n+1)*alpha)-th order statistic.

    For a clearance the planner must be confident of EXCEEDING, the useful one-sided bound is
    from below, not above. Returns 0.0 when n is too small for the level rather than silently
    returning the minimum.
    """
    x = np.sort(np.asarray(x))
    k = int(np.floor((len(x) + 1) * alpha))
    return float(x[k - 1]) if k >= 1 else 0.0


def probe_box(qpos: np.ndarray, x: float, hi: float = 0.60, tol: float = 0.005) -> float:
    """Tallest corridor-spanning floor box at `x` this motion still clears."""
    def clears(h: float) -> bool:
        if h <= 0:
            return True
        sc = Scene("probe", "probe", [Box((x, 0.0, h / 2), (0.125, 1.4, h / 2), "probe")],
                   start=(0, 0), goal=(8, 0))
        return G1Body(sc).trajectory_report(qpos)["collision_free"]
    if not clears(tol):
        return 0.0
    if clears(hi):
        return hi
    lo = tol
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if clears(mid) else (lo, mid)
    return lo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp001e")
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
    leg_idx = {k: np.array([jn.index(j) for j in v]) for k, v in LEG.items()}
    both = np.concatenate([leg_idx["left"], leg_idx["right"]])

    base = ConstraintSpec(root_xz=path, heading=np.zeros(T),
                          root_y=np.full(T, NOMINAL_PELVIS), first_heading=0.0)
    ctrls = []
    for i in range(0, N_SAMPLES, 8):
        k = min(8, N_SAMPLES - i)
        ctrls += runner.generate([PROMPT_WALK] * k, [base] * k, T, args.diffusion_steps,
                                 seeds=list(range(i, i + k)))

    rows = []
    for lift in LIFTS:
        specs = []
        for s in range(N_SAMPLES):
            ctrl = ctrls[s]
            if lift == 0.0:
                specs.append(base)
                continue
            q0 = runner.to_qpos(ctrl)
            cross = int(np.argmin(np.abs(q0[:, 0] - OBSTACLE_X)))
            contacts = ctrl["foot_contacts"]
            W = int(1.2 * fps)
            per_frame: dict[int, list] = {}
            for sd, pair in (("left", (0, 2)), ("right", (2, 4))):
                air = np.where(~contacts[:, pair[0]:pair[1]].any(-1))[0]
                for f in air[np.abs(air - cross) <= W]:
                    per_frame.setdefault(int(f), []).append(
                        (sd, float(np.exp(-0.5 * ((f - cross) / (0.5 * W)) ** 2))))
            near = np.array(sorted(per_frame)) if per_frame else np.arange(
                max(0, cross - 5), min(T, cross + 5))
            tgt = np.zeros((len(near), len(both), 3))
            for i2, f in enumerate(near):
                nom, root = ctrl["posed_joints"][f], ctrl["smooth_root_pos"][f]
                for j, ji in enumerate(both):
                    tgt[i2, j, 0] = path[f, 0] + (nom[ji, 0] - root[0])
                    tgt[i2, j, 2] = path[f, 1] + (nom[ji, 2] - root[2])
                    tgt[i2, j, 1] = nom[ji, 1]
                for sd, ph in per_frame.get(int(f), []):
                    for j, ji in enumerate(both):
                        if ji in leg_idx[sd]:
                            tgt[i2, j, 1] = nom[ji, 1] + lift * ph
            specs.append(ConstraintSpec(root_xz=path, heading=np.zeros(T),
                                        root_y=np.full(T, NOMINAL_PELVIS),
                                        pos_frames=near, pos_joints=both, pos_targets=tgt,
                                        first_heading=0.0))
        outs = []
        for i in range(0, N_SAMPLES, 8):
            k = specs[i:i + 8]
            outs += runner.generate([PROMPT_STEP if lift > 0 else PROMPT_WALK] * len(k), k,
                                    T, args.diffusion_steps, seeds=list(range(i, i + len(k))))
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            rows.append({"lift": lift, "sample": s,
                         "max_box_h_m": probe_box(q, OBSTACLE_X),
                         "travel_m": float(q[-1, 0] - q[0, 0]),
                         "foot_pen_m": body.trajectory_report(q)[
                             "max_foot_floor_penetration_m"]})
        print(f"  lift {lift:.2f} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    env = []
    for lift in LIFTS:
        g = [r["max_box_h_m"] for r in rows if r["lift"] == lift]
        env.append({"lift": lift, "n": len(g), "mean": float(np.mean(g)),
                    "worst4": float(np.min(g[:4])), "p90_conformal_lower":
                    conformal_lower(g), "max": float(np.max(g)),
                    "std": float(np.std(g))})
    summary = {"experiment": "exp001e_stepover_calibration", "alpha": 0.1,
               "n_samples": N_SAMPLES, "lifts": LIFTS,
               "old_body_modes_max_step": 0.028, "envelope": env,
               "best_p90_lower": max(e["p90_conformal_lower"] for e in env),
               "wall_clock_s": round(time.time() - t0, 1)}
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
