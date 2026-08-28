"""EXP-005c: when the planner refuses a scene, is that conservatism or physics?

Why this is now the most important experiment
---------------------------------------------
After the EXP-002 refresh the oracle does not collide any more — it *refuses*. Adaptive is
collision-free on 88/88 feasible plans, and end-to-end success (68.8 %) equals plan
feasibility exactly. Every remaining failure is `plan(...).feasible == False`.

So the question "how much better can any proposer be?" is entirely the question "how many
refusals are recoverable?" If most are real physics, then end-to-end kinematic success cannot
be the headline for a learned model, because there is nothing to win. That is a claim about
the ceiling, and it has to be measured before any training code is written.

The certified envelope is a worst-case-over-seeds bound, so a scene it refuses may still be
traversable most of the time. `relaxed_modes(top-0.08, width-0.05)` plans against a
deliberately UNCERTIFIED envelope; if the resulting motion then verifies against the robot's
own collision geometry, the refusal was conservatism. If it does not verify at any seed, it
was physics.

Reported per family and per seed, because "recoverable at 1 of 8 seeds" and "recoverable at
8 of 8" are very different claims and only the second justifies relaxing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.planner import plan, plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import build_suite, is_heldout, random_scene  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL, MAX_DURATION = 0.9, 0.5, 14.0
RELAX = (0.08, 0.05)
N_SEEDS = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp005c")
    ap.add_argument("--seeds_per_rung", type=int, default=4)
    ap.add_argument("--n_random", type=int, default=120)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    suite = build_suite(args.seeds_per_rung)
    rnd, s = [], 5_000_000
    while len(rnd) < args.n_random:
        sc = random_scene(s)
        s += 1
        if not is_heldout(sc) and not plan(sc, "adaptive").feasible:
            rnd.append(sc)

    refused = [sc for sc in suite if not plan(sc, "adaptive").feasible] + rnd
    print(f"{len(refused)} refused scenes "
          f"({len(refused)-len(rnd)} from the eval suite, {len(rnd)} random)", flush=True)

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    rows = []

    for sc in refused:
        p = plan(sc, "adaptive", relax=RELAX)
        rec = {"scene_id": sc.scene_id, "family": sc.family,
               "param_value": sc.param_value,
               "relaxed_plan_feasible": bool(p.feasible),
               "n_ok": 0, "n_seeds": 0}
        if p.feasible:
            T = min(int(MAX_DURATION * fps),
                    max(int(2 * fps), int(round(p.length / SPEED * fps))))
            base = int(hashlib.sha1(sc.scene_id.encode()).hexdigest()[:8], 16) % (2 ** 20)
            seeds = [base + i for i in range(N_SEEDS)]
            body = G1Body(sc)
            ctrl = runner.generate([PROMPT],
                                   [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                                   args.diffusion_steps, seeds=[seeds[0]])[0]
            spec = plan_to_spec(p, fps, ctrl, runner.joint_names, SPEED, duration=T / fps)
            outs = []
            for i in range(0, N_SEEDS, 8):
                k = seeds[i:i + 8]
                outs += runner.generate([PROMPT] * len(k), [spec] * len(k), T,
                                        args.diffusion_steps, seeds=k)
            oks, pens = [], []
            for o in outs:
                q = runner.to_qpos(o)
                r = body.trajectory_report(q)
                goal = bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL)
                oks.append(bool(goal and r["collision_free"]))
                pens.append(r["max_penetration_m"])
            rec.update(n_ok=int(sum(oks)), n_seeds=len(oks),
                       any_ok=bool(any(oks)), all_ok=bool(all(oks)),
                       mean_max_pen_m=float(np.mean(pens)),
                       plan_length_m=p.length)
        rows.append(rec)
    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    fam = defaultdict(lambda: {"n": 0, "relaxed_feasible": 0, "any_ok": 0, "all_ok": 0})
    for r in rows:
        f = fam[r["family"]]
        f["n"] += 1
        f["relaxed_feasible"] += int(r["relaxed_plan_feasible"])
        f["any_ok"] += int(r.get("any_ok", False))
        f["all_ok"] += int(r.get("all_ok", False))
    summary = {"experiment": "exp005c_refusal_headroom", "relax": list(RELAX),
               "n_seeds": N_SEEDS, "n_refused": len(rows),
               "recoverable_any": sum(int(r.get("any_ok", False)) for r in rows),
               "recoverable_all": sum(int(r.get("all_ok", False)) for r in rows),
               "recoverable_fraction_any": float(np.mean(
                   [r.get("any_ok", False) for r in rows])),
               "per_family": {k: dict(v) for k, v in sorted(fam.items())},
               "wall_clock_s": round(time.time() - t0, 1)}
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
