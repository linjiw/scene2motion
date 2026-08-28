"""EXP-005e: is the strategy enumerator incomplete on harder scenes?

The premise under exit (b)
--------------------------
EXP-005d's gate fired: on the evaluation suite, `ORACLE-ENUMERATE` reaches 1.93 of the >= 2
strategies available at K=2 and `ORACLE-RELAXED@8` hits the 75.0 % traversal ceiling exactly,
so a learned `p_phi` has nothing to win. Of the two pre-committed exits, (b) keeps the learned
model alive by moving to scenes "where enumeration is expensive or incomplete".

That is a PREMISE, not a finding. On the current suite the enumerator is near-complete, which
is precisely why the gate fired. Adopting exit (b) without testing whether harder scenes
actually break the enumerator would be assuming the thing that makes the model necessary --
the same mistake the gate exists to prevent.

Method
------
Build harder scenes (three to five obstacles, tighter corridors, mixed types) and compare:

  HEURISTIC   `enumerate_strategies` as shipped: two mode sets x iterated exclusion of the
              corridor each solution used, at most `max_k` rounds.
  REFERENCE   a deliberately expensive search over the same plan space: many restarts with
              RANDOM forbidden boxes, random mode subsets, and random via-point detours, so it
              can reach homotopy classes the heuristic's exclusion rule never proposes.

Both are scored by the same (homotopy, morphology) signature. The number that matters is the
fraction of reference signatures the heuristic MISSES, and whether the misses concentrate on
scenes with more obstacles — a heuristic that degrades with scene complexity is what exit (b)
needs, and a heuristic that stays complete kills exit (b) too.

This is CPU-only: no generation, no verification. It asks whether the strategies EXIST to be
found, which is the precondition; whether the prior can execute them is a separate question
and only worth asking if this one comes back positive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.planner import MODES, plan  # noqa: E402
from scene2motion.scenes import (OBSTACLE_KINDS, Scene, WALL_T, _rand_obstacle,  # noqa: E402
                                 _side_walls, is_heldout)
from scene2motion.strategies import signature_of  # noqa: E402
from scene2motion.strategies import enumerate_strategies  # noqa: E402

N_RESTARTS = 40


def hard_scene(seed: int, n_obstacles: int) -> Scene:
    """A deliberately harder corridor: more obstacles, tighter, mixed types."""
    rng = np.random.default_rng(seed)
    corridor_half = float(rng.uniform(0.9, 1.5))
    goal_x = float(rng.uniform(8.0, 11.0))
    lo_x, hi_x = 2.0, goal_x - 1.2
    k = max(1, min(n_obstacles, int((hi_x - lo_x) // 1.4) + 1))
    xs = np.linspace(lo_x, hi_x, k) + rng.uniform(-0.3, 0.3, size=k)
    kinds = [OBSTACLE_KINDS[int(i)] for i in rng.integers(0, len(OBSTACLE_KINDS), size=k)]
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    for x, kind in zip(xs, kinds):
        boxes += _rand_obstacle(rng, kind, float(x), corridor_half)
    return Scene(f"hard_{seed:06d}_k{k}", "hard", boxes,
                 start=(0.0, float(rng.uniform(-0.2, 0.2))),
                 goal=(goal_x, float(rng.uniform(-0.3, 0.3))),
                 bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
                 meta={"corridor_half": corridor_half, "kinds": kinds,
                       "obstacle_xs": [float(x) for x in xs]})


def reference_signatures(sc: Scene, rng: np.random.Generator,
                         n_restarts: int = N_RESTARTS) -> dict:
    """Expensive search: random exclusions, random mode subsets, random detours.

    Deliberately not the heuristic's rule. The point is to reach homotopy classes the
    heuristic's "exclude the corridor you just used" step would never propose.
    """
    sigs: dict = {}
    names = [m.name for m in MODES]
    y_lo, y_hi = sc.bounds[2], sc.bounds[3]
    obs_x = [float(b.center[0]) for b in sc.boxes if not b.label.startswith("wall_")]
    for _ in range(n_restarts):
        k = int(rng.integers(1, len(names) + 1))
        allow = tuple(rng.choice(names, size=k, replace=False).tolist())
        if "stand" not in allow and rng.random() < 0.5:
            allow = allow + ("stand",)
        boxes = []
        for x in obs_x:
            if rng.random() < 0.7:
                y = float(rng.uniform(y_lo, y_hi))
                h = float(rng.uniform(0.15, 0.6))
                boxes.append((x - rng.uniform(0.4, 1.4), x + rng.uniform(0.4, 1.4),
                              y - h, y + h))
        p = plan(sc, "adaptive", allow_modes=allow, forbid_boxes=boxes or None)
        if p.feasible:
            sig = signature_of(p, sc)
            # Keep the CHEAPEST plan found for each signature. A signature reached only by a
            # wildly contrived detour is not a strategy anyone would use, and counting it
            # would inflate the enumerator's apparent incompleteness.
            if sig not in sigs or p.length < sigs[sig]:
                sigs[sig] = p.length
    return sigs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp005e")
    ap.add_argument("--n_per_k", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = []
    seed = 7_000_000
    for k in (1, 2, 3, 4, 5):
        made = 0
        while made < args.n_per_k:
            sc = hard_scene(seed, k)
            seed += 1
            if is_heldout(sc):
                continue
            made += 1
            hs = enumerate_strategies(sc, max_k=4)
            heur = {s.signature: s.plan.length for s in hs}
            ref = reference_signatures(sc, np.random.default_rng(seed))
            missed = {k: v for k, v in ref.items() if k not in heur}
            best = min(list(heur.values()) + list(ref.values()), default=float("nan"))
            # A missed strategy only counts as a real alternative if its cost is within
            # `bloat` of the best route in the scene. Beyond that it is a detour nobody
            # would choose, and calling it a missed capability would be self-serving.
            useful = {k: v for k, v in missed.items() if v <= best * 1.35}
            rows.append({"scene_id": sc.scene_id, "n_obstacles": len(sc.meta["kinds"]),
                         "kinds": sc.meta["kinds"],
                         "n_heuristic": len(heur), "n_reference": len(ref),
                         "n_missed": len(missed), "n_missed_useful": len(useful),
                         "best_length_m": float(best),
                         "missed_lengths": [round(v, 2) for v in sorted(missed.values())],
                         "heuristic_lengths": [round(v, 2) for v in sorted(heur.values())],
                         "feasible": bool(ref or heur)})
        print(f"  k={k}: {made} scenes ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    by_k = {}
    for k in sorted({r["n_obstacles"] for r in rows}):
        g = [r for r in rows if r["n_obstacles"] == k and r["feasible"]]
        if not g:
            continue
        by_k[str(k)] = {
            "n_scenes": len(g),
            "mean_heuristic": float(np.mean([r["n_heuristic"] for r in g])),
            "mean_reference": float(np.mean([r["n_reference"] for r in g])),
            "mean_missed": float(np.mean([r["n_missed"] for r in g])),
            "mean_missed_useful": float(np.mean([r["n_missed_useful"] for r in g])),
            "frac_scenes_with_a_useful_miss": float(np.mean(
                [r["n_missed_useful"] > 0 for r in g])),
            "recall": float(np.mean([
                (r["n_reference"] - r["n_missed"]) / max(r["n_reference"], 1) for r in g])),
            "recall_useful": float(np.mean([
                1.0 - r["n_missed_useful"] / max(r["n_heuristic"] + r["n_missed_useful"], 1)
                for r in g])),
        }
    summary = {"experiment": "exp005e_enumerator_completeness",
               "n_restarts_reference": N_RESTARTS, "n_scenes": len(rows),
               "by_n_obstacles": by_k,
               "overall_recall": float(np.mean([
                   (r["n_reference"] - r["n_missed"]) / max(r["n_reference"], 1)
                   for r in rows if r["feasible"]])),
               "overall_recall_useful": float(np.mean([
                   1.0 - r["n_missed_useful"] / max(r["n_heuristic"] + r["n_missed_useful"], 1)
                   for r in rows if r["feasible"]])),
               "useful_bloat_threshold": 1.35,
               "wall_clock_s": round(time.time() - t0, 1)}
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
