"""EXP-003: does the frozen prior support a DISTRIBUTION over traversal strategies?

Why this experiment and not another clearance table
---------------------------------------------------
"A G1 ducks under a beam" is done, on hardware, by at least two 2025-26 systems (Gallant
2511.14625; HumanoidPF 2601.16035), and "diffusion generator + RL tracker for terrain-aware
G1" is done by 2604.17335. What none of them can do is give more than ONE answer: each is a
single deterministic policy mapping geometry to one behaviour. A large pretrained prior can
in principle carry several topologically distinct ways through the same aperture.

So the question here is not "can it duck" (EXP-001 settled that) but:

  For one scene that genuinely admits several strategies, how many does the prior
  actually realise as collision-free, goal-reaching motion -- and at what relative cost?

Method
------
Strategies are ENUMERATED by constrained re-planning, not by hoping sampling finds them:

  under     forbid the lateral band containing the bypass  -> must duck under the obstacle
  around    forbid every non-standing body mode            -> must walk around it
  left/right (pillar) forbid one side of the obstacle
  free      unconstrained A*, i.e. what a cost-minimising planner would have picked

Each strategy is then rendered into constraint channels and generated with several seeds,
and scored exactly as in EXP-002. A strategy counts as REALISED if some seed reaches the
goal collision-free; the per-seed rate is reported separately so a strategy that works one
time in four is not passed off as a capability.

The headline number is the count of simultaneously-realised strategies per scene. A
deterministic controller scores 1 by construction.
"""

from __future__ import annotations

import argparse
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
from scene2motion.scenes import LADDERS, build_partial_beam, build_pillar  # noqa: E402

PROMPT = "A person walks forward."
SPEED = 0.9
GOAL_TOL = 0.5
SEEDS = [0, 1, 2, 3]
STAND_ONLY = ("stand",)


def strategies_for(sc):
    """(name, plan) for every strategy this scene admits, by constrained re-planning."""
    out = [("free", plan(sc, "adaptive"))]
    if sc.family == "partial_beam":
        edge = sc.meta["beam_edge_y"]
        ch = sc.meta["corridor_half"]
        # Forbidding the open side leaves only the route under the beam.
        out.append(("under", plan(sc, "adaptive", forbid_y=(-ch - 0.4, edge - 0.05))))
        # Forbidding every adaptation leaves only the route around it.
        out.append(("around", plan(sc, "adaptive", allow_modes=STAND_ONLY)))
    elif sc.family == "pillar":
        py = sc.meta.get("pillar_x") and sc.param_value
        r = sc.meta["radius"]
        ch = sc.meta["corridor_half"]
        out.append(("left", plan(sc, "adaptive", forbid_y=(-ch - 0.4, py - r - 0.05))))
        out.append(("right", plan(sc, "adaptive", forbid_y=(py + r + 0.05, ch + 0.4))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp003")
    ap.add_argument("--seeds_per_rung", type=int, default=3)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scenes = []
    for seed in range(args.seeds_per_rung):
        for v in LADDERS["partial_beam"]:
            scenes.append(build_partial_beam(v, seed))
        for v in LADDERS["pillar"][3:]:      # only the rungs where the pillar is in the way
            scenes.append(build_pillar(v, seed))
    print(f"{len(scenes)} ambiguous scenes", flush=True)

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    free_body = G1Body(None)
    rows = []

    jobs = []
    for sc in scenes:
        for name, p in strategies_for(sc):
            if not p.feasible:
                rows.append({"scene_id": sc.scene_id, "family": sc.family,
                             "param_value": sc.param_value, "strategy": name,
                             "plan_feasible": False, "seed": None})
                continue
            T = min(int(14 * fps), max(int(2 * fps), int(round(p.length / SPEED * fps))))
            for s in SEEDS:
                jobs.append((sc, name, p, T, s))
    print(f"{len(jobs)} (scene, strategy, seed) clips to generate", flush=True)

    by = defaultdict(list)
    for j in jobs:
        by[(j[3], j[4])].append(j)      # group by (length, seed): both must match to batch

    for (T, s), group in sorted(by.items()):
        for i in range(0, len(group), 8):
            chunk = group[i:i + 8]
            cspecs = [plan_to_path_spec(p, fps, SPEED, duration=T / fps)
                      for _, _, p, _, _ in chunk]
            ctrls = runner.generate([PROMPT] * len(chunk), cspecs, T, args.diffusion_steps, seed=s)
            aspecs = [plan_to_spec(p, fps, c, runner.joint_names, SPEED, duration=T / fps)
                      for (_, _, p, _, _), c in zip(chunk, ctrls)]
            adapts = runner.generate([PROMPT] * len(chunk), aspecs, T, args.diffusion_steps, seed=s)
            for (sc, name, p, _, _), o in zip(chunk, adapts):
                q = runner.to_qpos(o)
                rep = G1Body(sc).trajectory_report(q)
                end = q[-1, :2]
                rows.append({
                    "scene_id": sc.scene_id, "family": sc.family,
                    "param_value": sc.param_value, "strategy": name, "seed": s,
                    "plan_feasible": True, "plan_length_m": p.length,
                    "goal_reached": bool(np.linalg.norm(end - np.array(sc.goal)) < GOAL_TOL),
                    "collision_free": rep["collision_free"],
                    "max_penetration_m": rep["max_penetration_m"],
                    "min_clearance_m": rep["min_clearance_m"],
                    "min_pelvis_z_m": float(q[:, 2].min()),
                    "max_abs_y_m": float(np.abs(q[:, 1]).max()),
                    "foot_floor_pen_m": rep["max_foot_floor_penetration_m"],
                })
                np.save(out / f"qpos_{sc.scene_id}_{name}_s{s}.npy", q.astype(np.float32))
        print(f"  T={T} seed={s} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp003_multimodality", "model": runner.model_name,
                   "fps": fps, "seeds": SEEDS, "n_scenes": len(scenes), "n_rows": len(rows),
                   "goal_tol_m": GOAL_TOL, "speed_mps": SPEED,
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
