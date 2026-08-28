"""EXP-005d: ORACLE@K — the baseline a learned proposer has to beat.

The gate
--------
`p_phi(C | S, s, g)` is only a contribution if it beats what the enumerator already does when
given the same sampling budget. The enumerator runs in 0.13 s of CPU, produces a SET of
strategies, and every candidate can be generated and verified against the robot's own
collision geometry — so the honest baseline is not "the oracle's single best plan", it is
**propose, sample K, verify, select**. A reviewer will construct this in their head; better to
construct it here.

Two proposers, identical downstream:

  ORACLE-ENUMERATE   the certified (worst-of-seeds) envelope
  ORACLE-RELAXED     `relaxed_modes(top-0.08, width-0.05)`, deliberately uncertified

Two numbers per proposer:

  K4   traversal success@K -- any of the K generated clips reaches the goal collision-free.
       This is the end-to-end number, and §0 of the spec argues it is capped near 75 % for
       everyone on this suite.
  B1   distinct verified strategy signatures@K on the ambiguous scenes -- how many genuinely
       different ways through the model actually delivers. This is the number the project's
       surviving claim rests on, and it is the one a single deterministic planner cannot move.

The kill criterion is explicit: if ORACLE-ENUMERATE@8 already reaches the learned model's
plausible K4 *and* its strategy coverage, then p_phi would be distillation of a 0.13 s planner
into a network, and it is not a contribution. Deciding that here costs 20 minutes of GPU
instead of three hours of dataset plus a training run.
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

from scene2motion.planner import plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import build_suite  # noqa: E402
from scene2motion.strategies import enumerate_strategies  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL, MAX_DURATION = 0.9, 0.5, 14.0
KS = [1, 2, 4, 8]
RELAX = (0.08, 0.05)


def realised_signature(qpos: np.ndarray, scene) -> tuple:
    """What the GENERATED motion actually did, read off the motion rather than the label.

    Using the commanded strategy name would make coverage a tautology — it would count what we
    asked for, not what the prior delivered. The signature is (which side of each choice
    obstacle the body passed, did it duck under anything), both measured from qpos.
    """
    from scene2motion.strategies import _choice_obstacles
    homotopy = []
    for x, oy in _choice_obstacles(scene):
        i = int(np.argmin(np.abs(qpos[:, 0] - x)))
        lo, hi = max(0, i - 12), min(len(qpos), i + 13)
        d = float(np.mean(qpos[lo:hi, 1])) - oy
        homotopy.append(0 if abs(d) < 0.12 else int(np.sign(d)))
    ducked = int(qpos[:, 2].min() < 0.68)      # pelvis well below its standing height
    return (tuple(homotopy), ducked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp005d")
    ap.add_argument("--seeds_per_rung", type=int, default=4)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scenes = build_suite(args.seeds_per_rung)
    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    Kmax = max(KS)
    rows = []

    for arm, relax in (("enumerate", None), ("relaxed", RELAX)):
        for sc in scenes:
            strats = enumerate_strategies(sc, max_k=4, relax=relax)
            base = int(hashlib.sha1(f"{arm}|{sc.scene_id}".encode()).hexdigest()[:8], 16)
            rec = {"arm": arm, "scene_id": sc.scene_id, "family": sc.family,
                   "param_value": sc.param_value, "n_strategies": len(strats),
                   "samples": []}
            if strats:
                body = G1Body(sc)
                # Spread the budget over the strategies: K=1 takes the cheapest plan, K=8 with
                # two strategies takes four seeds of each. A budget spent entirely on the
                # cheapest plan would measure the sampler, not the proposer.
                assign = [(strats[i % len(strats)], i) for i in range(Kmax)]
                for st, i in assign:
                    T = min(int(MAX_DURATION * fps),
                            max(int(2 * fps), int(round(st.plan.length / SPEED * fps))))
                    seed = (base + i) % (2 ** 20)
                    ctrl = runner.generate(
                        [PROMPT], [plan_to_path_spec(st.plan, fps, SPEED, duration=T / fps)],
                        T, args.diffusion_steps, seeds=[seed])[0]
                    spec = plan_to_spec(st.plan, fps, ctrl, runner.joint_names, SPEED,
                                        duration=T / fps)
                    o = runner.generate([PROMPT], [spec], T, args.diffusion_steps,
                                        seeds=[seed])[0]
                    q = runner.to_qpos(o)
                    r = body.trajectory_report(q)
                    goal = bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL)
                    rec["samples"].append({
                        "strategy": st.name, "ok": bool(goal and r["collision_free"]),
                        "signature": list(map(list, [realised_signature(q, sc)[0]]))[0]
                                     + [realised_signature(q, sc)[1]],
                        "max_pen_m": r["max_penetration_m"]})
            rows.append(rec)
        print(f"  arm {arm} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    summary = {"experiment": "exp005d_oracle_at_k", "n_scenes": len(scenes),
               "Ks": KS, "relax": list(RELAX), "arms": {}}
    for arm in ("enumerate", "relaxed"):
        a = [r for r in rows if r["arm"] == arm]
        amb = [r for r in a if r["n_strategies"] >= 2]
        per_k = {}
        for K in KS:
            succ = [any(s["ok"] for s in r["samples"][:K]) for r in a]
            sigs = [len({tuple(s["signature"]) for s in r["samples"][:K] if s["ok"]})
                    for r in amb]
            per_k[str(K)] = {
                "K4_traversal_success": float(np.mean(succ)),
                "B1_distinct_verified_signatures": float(np.mean(sigs)) if sigs else 0.0,
            }
        summary["arms"][arm] = {"per_k": per_k,
                                "n_ambiguous": len(amb),
                                "mean_strategies": float(np.mean([r["n_strategies"]
                                                                  for r in a]))}
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
