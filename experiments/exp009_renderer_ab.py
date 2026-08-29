"""EXP-009: how much of the composed-program collapse was the renderer, not the prior?

The observation
---------------
In the EXP-005g ledger, candidates whose program commands a duck AND a tuck-or-lift come back
valid on 0.19 of held-out seeds; candidates that command neither come back valid on 0.97.
43.5 % of all candidates are in the first group.  The three CPU enumerators put 10-14 of their
16 candidates there; the two arms that search WITH ARDY feedback put almost none there, because
they measured the failures and stopped proposing them.

And a bug found by the adversarial audit predicts exactly that: `_limb_targets` read limb HEIGHT
targets out of the un-ducked nominal clip, so a duck-and-tuck program asked ARDY to drop the
pelvis 40 cm and hold the hands at standing height in the same request -- verified at dip = 0.40
to be exactly 40 cm of contradiction.

The confound this experiment exists to break
--------------------------------------------
"Affected" above is defined by the CONTENT of the request, not by the bug.  Composed requests ask
for more, and asking for more may simply be harder -- for the prior, or because composed programs
appear in harder scenes.  A correlation between "commands two axes" and "fails" is therefore not
evidence that the renderer caused the failure, and reporting it as such would be the same
mistake as blaming the prior in the first place, with the sign flipped.

So this is a controlled A/B on the RENDERER ALONE.  The same programs, the same scenes, the same
seeds, generated twice:

    OLD   limb height targets pinned to the un-ducked nominal   (the shipped behaviour)
    NEW   limb height targets shifted by the commanded pelvis displacement

Nothing else differs -- not the request, not the amplitude, not the scene, not the noise.  Any
difference in validity is caused by the renderer and by nothing else.

A pure-tuck / pure-lift row is carried as the null: those programs command no duck, so the two
renderers emit byte-identical requests and the arm MUST show zero difference.  If it does not,
the harness is not doing what this docstring says and no other row can be believed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion import planner as P  # noqa: E402
from scene2motion.planner import plan, plan_to_path_spec  # noqa: E402
from scene2motion.program import ConstraintProgram, decode, encode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL, MAX_DUR = 0.9, 0.5, 14.0
SCENES = [("overhead_beam", 1.10), ("overhead_beam", 1.00), ("partial_beam", 1.05),
          ("beam_and_gap", 1.10), ("pillar", 0.30), ("narrow_gap", 0.70)]
# (label, dip, tuck, lift).  The first two command NO duck and are the null rows.
REQUESTS = [("tuck only  [null]", 0.00, 0.60, 0.00),
            ("lift only  [null]", 0.00, 0.00, 0.35),
            ("duck+tuck", 0.35, 0.60, 0.00),
            ("duck+lift", 0.35, 0.00, 0.35),
            ("duck+tuck+lift", 0.35, 0.60, 0.35),
            ("deepduck+tuck", 0.50, 0.60, 0.00)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp009")
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    seeds = list(range(700, 700 + args.n_seeds))
    real = P._limb_targets
    rows = []

    def old_limb_targets(root_xz, tuck, lift, nominal, joint_names, fps_, root_y=None):
        """The shipped behaviour: drop root_y, so heights stay pinned to the nominal."""
        return real(root_xz, tuck, lift, nominal, joint_names, fps_, root_y=None)

    for fam, val in SCENES:
        sc = BUILDERS[fam](val, 0)
        p = plan(sc, "adaptive")
        if not p.feasible:
            print(f"  {fam} {val}: route infeasible", flush=True)
            continue
        body = G1Body(sc)
        T = min(int(MAX_DUR * fps), max(int(2 * fps), int(round(p.length / SPEED * fps))))
        ref = runner.generate([PROMPT],
                              [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                              args.diffusion_steps, seeds=[seeds[0]])[0]
        route = encode(p, sc, fps, SPEED)

        for label, dip, tuck, lift in REQUESTS:
            prog = ConstraintProgram(lat=route.lat.copy(),
                                     slot=np.zeros_like(route.slot), speed=route.speed)
            prog.slot[0] = [0.5, 0.12, dip, tuck, lift]
            for arm, fn in (("OLD", old_limb_targets), ("NEW", real)):
                P._limb_targets = fn
                try:
                    spec = decode(prog, sc, fps, ref, runner.joint_names, duration=T / fps)
                finally:
                    P._limb_targets = real
                outs = runner.generate([PROMPT] * len(seeds), [spec] * len(seeds), T,
                                       args.diffusion_steps, seeds=seeds)
                for s, o in enumerate(outs):
                    q = runner.to_qpos(o)
                    rep = body.trajectory_report(q)
                    goal = bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL)
                    rows.append({"scene": sc.scene_id, "family": fam, "request": label,
                                 "dip": dip, "tuck": tuck, "lift": lift, "arm": arm,
                                 "seed": seeds[s],
                                 "valid": bool(goal and rep["collision_free"]),
                                 "goal_ok": bool(goal),
                                 "coll_free": bool(rep["collision_free"]),
                                 "min_clear_m": float(rep["min_clearance_m"]),
                                 "pelvis_mean_m": float(q[:, 2].mean()),
                                 "foot_pen_m": float(rep["mean_foot_floor_penetration_m"])})
        print(f"  {sc.scene_id[:30]:30s} ({time.time()-t0:.0f}s)", flush=True)
        P._limb_targets = real

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{len(rows)} clips ({time.time()-t0:.0f}s)\n")
    print(f"{'request':20s} {'OLD valid':>10s} {'NEW valid':>10s} {'delta':>8s} "
          f"{'OLD goal':>9s} {'NEW goal':>9s} {'OLD coll-free':>14s} {'NEW coll-free':>14s}")
    print("-" * 100)
    summary = {"experiment": "exp009_renderer_ab", "n_seeds": args.n_seeds,
               "n_rows": len(rows), "requests": {}}
    for label, dip, tuck, lift in REQUESTS:
        R = {a: [r for r in rows if r["request"] == label and r["arm"] == a]
             for a in ("OLD", "NEW")}
        if not R["OLD"] or not R["NEW"]:
            continue
        v = {a: float(np.mean([r["valid"] for r in R[a]])) for a in R}
        g = {a: float(np.mean([r["goal_ok"] for r in R[a]])) for a in R}
        c = {a: float(np.mean([r["coll_free"] for r in R[a]])) for a in R}
        summary["requests"][label] = {"old_valid": v["OLD"], "new_valid": v["NEW"],
                                      "delta": v["NEW"] - v["OLD"],
                                      "old_goal": g["OLD"], "new_goal": g["NEW"],
                                      "old_coll_free": c["OLD"], "new_coll_free": c["NEW"],
                                      "n_per_arm": len(R["OLD"])}
        print(f"{label:20s} {v['OLD']:10.3f} {v['NEW']:10.3f} {v['NEW']-v['OLD']:+8.3f} "
              f"{g['OLD']:9.3f} {g['NEW']:9.3f} {c['OLD']:14.3f} {c['NEW']:14.3f}")
    nulls = [k for k in summary["requests"] if "null" in k]
    bad = [k for k in nulls if abs(summary["requests"][k]["delta"]) > 1e-9]
    print(f"\n  NULL ROWS: {', '.join(nulls)} command no duck, so both renderers emit the SAME\n"
          f"  request and the delta must be exactly 0.  "
          + ("PASS." if not bad else f"*** FAILED on {bad} -- the harness is not isolating the\n"
                                     f"  renderer and no other row in this table can be believed."))
    print("  goal vs coll-free splits WHY a clip is invalid: a contradictory request should\n"
          "  break locomotion (goal) rather than push the body into the scene (collision).")
    summary["null_rows_pass"] = not bad
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
