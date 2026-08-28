"""EXP-005b: does the AVERAGE of two valid strategies collide?

The architecture question this settles
--------------------------------------
Flow matching and diffusion are the obvious choices for `p_phi(C | S, s, g)`, and a review
pass argued against them with a specific empirical claim: for a scene admitting "duck under"
and "walk around", the arithmetic mean of the two constraint programs is *in collision*, so
any objective whose conditional optimum is a mean will emit an infeasible program. At a few
thousand training pairs a learned velocity field is a smoothed estimator, and the smoothing
puts terminal mass exactly in the infeasible gap between the modes.

That is a strong claim and it decides the model, so it gets measured on this repo's own data
rather than taken on faith.

Method
------
From the built corpus, take every scene with two or more VALIDATED programs — each one already
generated through the frozen prior and confirmed to reach the goal without collision. For each
such scene:

    mean of the valid program vectors  ->  decode  ->  ARDY  ->  collision check

If the individual programs succeed by construction and their mean fails, mean-seeking is
disqualified for this output space and the model must be a set predictor or a genuinely
multimodal sampler. If the mean also succeeds, the argument does not hold here and flow
matching stays on the table — which would be the cheaper model to build.

The interpolation midpoint is also swept at a few mixing weights, because a mean that fails
tells us less than knowing *where* along the interpolation feasibility is lost: a narrow
infeasible band near 0.5 is a much weaker objection than one covering most of the segment.
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

from scene2motion.planner import plan_to_path_spec  # noqa: E402
from scene2motion.program import ConstraintProgram, decode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import Scene  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL = 0.9, 0.5
MIXES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="outputs/dataset")
    ap.add_argument("--out", default="outputs/exp005b")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scenes = {}
    for line in (data / "scenes.jsonl").open():
        sc = Scene.from_dict(json.loads(line))
        scenes[sc.scene_id] = sc
    labels = [json.loads(l) for l in (data / "labels.jsonl").open()]
    vecs = np.load(data / "programs.npy")

    by_scene = defaultdict(list)
    for r in labels:
        if r["valid"]:
            by_scene[r["scene_id"]].append(r)
    multi = {k: v for k, v in by_scene.items() if len(v) >= 2}
    print(f"{len(multi)} scenes with >=2 validated programs "
          f"(of {len(by_scene)} with any)", flush=True)
    if not multi:
        print("nothing to test")
        return

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    rows = []

    for sid, recs in multi.items():
        sc = scenes[sid]
        body = G1Body(sc)
        a, b = recs[0], recs[1]          # the two cheapest distinct strategies
        va, vb = vecs[a["vec_row"]], vecs[b["vec_row"]]
        T = int(a["n_frames"])
        seed = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) % (2 ** 31)

        # A path-only control at this length supplies the nominal limb reference the decoder
        # needs; using one control for every mix keeps the comparison clean.
        from scene2motion.planner import plan
        p = plan(sc, "adaptive")
        if not p.feasible:
            continue
        ctrl = runner.generate([PROMPT],
                               [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                               args.diffusion_steps, seeds=[seed])[0]

        specs = [decode(ConstraintProgram.from_vec((1 - m) * va + m * vb), sc, fps, ctrl,
                        runner.joint_names, duration=T / fps) for m in MIXES]
        outs = runner.generate([PROMPT] * len(specs), specs, T, args.diffusion_steps,
                               seeds=[seed] * len(specs))
        for m, o in zip(MIXES, outs):
            q = runner.to_qpos(o)
            rep = body.trajectory_report(q)
            rows.append({
                "scene_id": sid, "mix": m,
                "strategy_a": a["strategy"], "strategy_b": b["strategy"],
                "homotopy_a": a["homotopy"], "homotopy_b": b["homotopy"],
                "morph_a": a["morphology"], "morph_b": b["morphology"],
                "goal_reached": bool(
                    np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL),
                "collision_free": rep["collision_free"],
                "max_penetration_m": rep["max_penetration_m"],
                "min_clearance_m": rep["min_clearance_m"],
                "min_pelvis_m": float(q[:, 2].min()),
            })
    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    ok = defaultdict(list)
    for r in rows:
        ok[r["mix"]].append(r["goal_reached"] and r["collision_free"])
    summary = {"experiment": "exp005b_mean_collides", "n_scenes": len(multi),
               "success_by_mix": {str(m): float(np.mean(v)) for m, v in sorted(ok.items())},
               "endpoints_ok": float(np.mean(ok[0.0] + ok[1.0])),
               "midpoint_ok": float(np.mean(ok[0.5])),
               "wall_clock_s": round(time.time() - t0, 1)}
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
