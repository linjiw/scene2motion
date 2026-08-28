"""EXP-003b: how few numbers does a whole-body traversal strategy actually take?

Why this comes before EXP-005
------------------------------
EXP-003 showed the frozen prior instantiates whichever of two strategies it is handed. It did
NOT show that the prior *finds* them: the strategies came from constrained re-planning, and the
request was a DENSE per-frame root path. Two questions follow, and both must be answered before
a generative model is designed, because between them they fix its output dimensionality:

 1. **Sparsity.** ARDY accepts constraints at arbitrary frame subsets, and treats unconstrained
    frames as free. If a strategy survives being specified by 4 waypoints instead of 200 frames,
    then p(C|S,s,g) diffuses ~20 numbers rather than ~1000 — a different problem entirely.

 2. **Commitment.** ARDY is autoregressive: it generates chunk after chunk. The worry is that
    once it has walked toward one side of an obstacle, a later constraint cannot pull it back —
    the early chunks have already committed to a homotopy class. If true, sparse long-horizon
    goals are useless and the strategy must be specified densely and early.

Design
------
On the `partial_beam` scenes (which admit both "duck under" and "walk around"), each enumerated
strategy is rendered at several sparsity levels and generated with several seeds:

    dense      every frame constrained (the EXP-002/003 setting)
    wp32/16/8/4/2  root_2d + heading + root_y at that many evenly spaced frames
    goal_only  the last ~0.4 s only — a pure long-horizon goal with no route

Scored on: does it reach the goal, does it stay in the intended homotopy class (the thing that
makes it *that* strategy rather than the other one), does it still duck when the strategy called
for ducking, and is it collision-free.

The homotopy test is the one that matters. A sparse request that reaches the goal by drifting
into the other strategy's corridor has not preserved the strategy — it has silently substituted
a different one, which is exactly the failure a success-rate table would hide.
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
from scene2motion.planner import plan, plan_to_spec, plan_to_path_spec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import LADDERS, build_partial_beam  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL = 0.9, 0.5
SEEDS = [0, 1, 2, 3]
# (label, number of constrained frames or None for dense, tail-only flag)
# Levels bracket the knee found in the first run: the ROUTE survives down to ~4 waypoints,
# but the pelvis-height profile that makes the duck work is lost between 8 and 4, so the
# interesting region is sampled finely.
LEVELS = [("dense", None, False), ("wp32", 32, False), ("wp16", 16, False),
          ("wp12", 12, False), ("wp10", 10, False), ("wp8", 8, False), ("wp6", 6, False),
          ("wp5", 5, False), ("wp4", 4, False), ("wp2", 2, False), ("goal_only", 10, True)]


def sparsify(spec: ConstraintSpec, n: int | None, tail: bool) -> ConstraintSpec:
    """Subsample a dense ConstraintSpec down to `n` constrained frames."""
    T = spec.T
    if n is None:
        return spec
    idx = (np.arange(T - n, T) if tail
           else np.unique(np.linspace(0, T - 1, n).round().astype(int)))
    return ConstraintSpec(
        root_xz=spec.root_xz[idx],
        heading=None if spec.heading is None else spec.heading[idx],
        root_y=None if spec.root_y is None else spec.root_y[idx],
        root_frames=idx, n_frames=T,
        # Carried from the dense program: the start heading is a property of the plan, not
        # of how densely we chose to sample it.
        first_heading=spec.first_heading,
        # Joint targets are dropped: they are already sparse, and keeping them would confound
        # "how sparse can the ROOT program be" with "how much do limb targets carry".
    )


def homotopy_side(qpos: np.ndarray, obstacle_x: float) -> float:
    """Signed lateral offset (world y) as the robot passes the obstacle.

    Which side of the beam edge the body went is what distinguishes the two strategies; the
    sign of this number IS the strategy label.
    """
    i = int(np.argmin(np.abs(qpos[:, 0] - obstacle_x)))
    lo, hi = max(0, i - 12), min(len(qpos), i + 13)
    return float(np.mean(qpos[lo:hi, 1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp003b")
    ap.add_argument("--seeds_per_rung", type=int, default=2)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    rows = []

    scenes = [build_partial_beam(v, s)
              for s in range(args.seeds_per_rung) for v in LADDERS["partial_beam"]]
    print(f"{len(scenes)} scenes x 2 strategies x {len(LEVELS)} sparsity levels "
          f"x {len(SEEDS)} seeds", flush=True)

    for sc in scenes:
        ch, edge = sc.meta["corridor_half"], sc.meta["beam_edge_y"]
        bx = sc.meta["beam_x"]
        strategies = {
            "under": plan(sc, "adaptive", forbid_y=(-ch - 0.4, edge - 0.05)),
            "around": plan(sc, "adaptive", allow_modes=("stand",)),
        }
        body = G1Body(sc)
        for name, p in strategies.items():
            if not p.feasible:
                continue
            T = min(int(14 * fps), max(int(2 * fps), int(round(p.length / SPEED * fps))))
            # The dense adapted spec is the reference every sparsification is derived from.
            ctrl = runner.generate([PROMPT], [plan_to_path_spec(p, fps, SPEED, duration=T / fps)],
                                   T, args.diffusion_steps, seed=0)[0]
            dense = plan_to_spec(p, fps, ctrl, runner.joint_names, SPEED, duration=T / fps)
            intended_side = float(np.mean(dense.root_xz[
                np.argsort(np.abs(dense.root_xz[:, 1] - bx))[:25], 0]))

            for seed in SEEDS:
                specs, labels = [], []
                for label, n, tail in LEVELS:
                    specs.append(sparsify(dense, n, tail))
                    labels.append(label)
                outs = runner.generate([PROMPT] * len(specs), specs, T,
                                       args.diffusion_steps, seed=seed)
                for label, o, spec in zip(labels, outs, specs):
                    q = runner.to_qpos(o)
                    rep = body.trajectory_report(q)
                    end = q[-1, :2]
                    side = homotopy_side(q, bx)
                    n_numbers = (len(spec.root_xz) *
                                 (2 + (spec.heading is not None) + (spec.root_y is not None)))
                    rows.append({
                        "scene_id": sc.scene_id, "param_value": sc.param_value,
                        "strategy": name, "level": label, "seed": seed,
                        "n_constrained_frames": int(len(spec.root_xz)),
                        "n_numbers": int(n_numbers), "n_frames": int(T),
                        "goal_reached": bool(np.linalg.norm(end - np.array(sc.goal)) < GOAL_TOL),
                        "goal_dist_m": float(np.linalg.norm(end - np.array(sc.goal))),
                        "collision_free": rep["collision_free"],
                        "max_penetration_m": rep["max_penetration_m"],
                        "min_pelvis_z_m": float(q[:, 2].min()),
                        "side_at_obstacle_m": side,
                        "intended_side_m": intended_side,
                        # Same sign and not a near-zero straddle => the strategy survived.
                        "homotopy_kept": bool(np.sign(side) == np.sign(intended_side)
                                              or abs(intended_side) < 0.05),
                        "max_abs_y_m": float(np.abs(q[:, 1]).max()),
                    })
        print(f"  {sc.scene_id} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp003b_program_sparsity", "model": runner.model_name,
                   "fps": fps, "levels": [list(map(str, l)) for l in LEVELS], "seeds": SEEDS,
                   "n_scenes": len(scenes), "n_rows": len(rows), "goal_tol_m": GOAL_TOL,
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
