"""Build the EXP-005 training corpus: (scene, SET of validated constraint programs).

The one thing this must not get wrong
-------------------------------------
`p_phi(C | S, s, g)` is a distribution, so each scene needs a SET of labels. A* is a
deterministic function of the scene, so a corpus of {(S, encode(A*(S)))} has exactly one
target per scene and the conditional it defines is a point mass — a model trained on it
cannot be multimodal no matter how it is parameterised, and the whole project reduces to
regression. Labels therefore come from `enumerate_strategies`, which returns every distinct
(homotopy, morphology) class it can find, not from a single plan.

A candidate is not a label until it has been generated through the frozen prior and passed
the geometry check. EXP-002 measured roughly a fifth of A*-accepted plans still colliding once
rendered, and training on those would teach the generator to propose exactly them.

Scenes that admit NO strategy are kept too, with an empty label set. About 38 % of random
scenes are infeasible, and they are the only training signal for "decline" — a planner that
cannot say no is the one that walks into things.

Layout
------
    scenes.jsonl    one scene per line (scene2motion.scenes.Scene.to_dict)
    labels.jsonl    one record per (scene, strategy): the 39-number program plus its
                    validation metrics and its strategy signature
    programs.npy    (N, 39) float32 of every VALID program, row-aligned to labels.jsonl
                    entries with valid=true
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.planner import plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.program import ConstraintProgram, encode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import sample_train_scenes  # noqa: E402
from scene2motion.strategies import enumerate_strategies  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL, MAX_DURATION = 0.9, 0.5, 14.0
BATCH = 8


def seed_for(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16) % (2 ** 31)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/dataset")
    ap.add_argument("--n_scenes", type=int, default=600)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--enrich", type=float, nargs=3, default=None,
                    metavar=("MULTI", "UNI", "NONE"),
                    help="target mix of multimodal / unimodal / infeasible scenes, e.g. "
                         "0.4 0.4 0.2. Enumeration is CPU-cheap and validation is the GPU "
                         "cost, so scenes are drawn and enumerated until the mix is met.")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps

    # 1. Sample and enumerate on CPU. Enumeration is ~0.13 s/scene against ~1 s/scene of GPU
    # validation, so it is worth over-drawing here to control the mix.
    #
    # Only ~18 % of random scenes admit two or more strategies, so an unenriched corpus gives
    # p_phi mostly point targets and almost no pressure to be multimodal -- the one thing it
    # exists to be. Enriching changes p(S), which is harmless for a CONDITIONAL model and is
    # noted in the receipt: evaluation stays on the unenriched ladders.
    enriched: list = []
    kept = Counter()
    if args.enrich:
        want = {k: int(round(f * args.n_scenes))
                for k, f in zip(("multi", "uni", "none"), args.enrich)}
        seed, tried = args.seed0, 0
        while sum(kept.values()) < sum(want.values()) and tried < 200 * args.n_scenes:
            batch = sample_train_scenes(64, seed)
            seed += 1000
            tried += 64
            for sc in batch:
                strats = enumerate_strategies(sc, max_k=4)
                bucket = "none" if not strats else ("uni" if len(strats) == 1 else "multi")
                if kept[bucket] < want[bucket]:
                    kept[bucket] += 1
                    enriched.append((sc, strats))
        scenes = [sc for sc, _ in enriched]
        print(f"enriched draw: {dict(kept)} from {tried} sampled ({time.time()-t0:.0f}s)",
              flush=True)
    else:
        scenes = sample_train_scenes(args.n_scenes, args.seed0)
        enriched = [(sc, enumerate_strategies(sc, max_k=4)) for sc in scenes]
    print(f"{len(scenes)} scenes ({time.time()-t0:.0f}s)", flush=True)

    jobs = []
    per_scene = Counter()
    for sc, strats in enriched:
        per_scene[len(strats)] += 1
        for st in strats:
            T = min(int(MAX_DURATION * fps),
                    max(int(2 * fps), int(round(st.plan.length / SPEED * fps))))
            jobs.append((sc, st, T))
    print(f"{len(jobs)} candidate strategies; strategies/scene {dict(sorted(per_scene.items()))} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 2. Validate on GPU, batched by clip length so a batch shares num_frames.
    by_T: dict[int, list] = {}
    for j in jobs:
        by_T.setdefault(j[2], []).append(j)

    labels, vectors = [], []
    done = 0
    for T, group in sorted(by_T.items()):
        for i in range(0, len(group), BATCH):
            chunk = group[i:i + BATCH]
            seeds = [seed_for(sc.scene_id, st.name) for sc, st, _ in chunk]
            ctrls = runner.generate(
                [PROMPT] * len(chunk),
                [plan_to_path_spec(st.plan, fps, SPEED, duration=T / fps)
                 for _, st, _ in chunk],
                T, args.diffusion_steps, seeds=seeds)
            specs = [plan_to_spec(st.plan, fps, c, runner.joint_names, SPEED,
                                  duration=T / fps)
                     for (_, st, _), c in zip(chunk, ctrls)]
            outs = runner.generate([PROMPT] * len(chunk), specs, T,
                                   args.diffusion_steps, seeds=seeds)
            for (sc, st, _), o in zip(chunk, outs):
                q = runner.to_qpos(o)
                rep = G1Body(sc).trajectory_report(q)
                goal_ok = bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL)
                valid = bool(goal_ok and rep["collision_free"])
                prog = encode(st.plan, sc, fps, SPEED)
                vec = prog.to_vec()
                rec = {"scene_id": sc.scene_id, "strategy": st.name,
                       "homotopy": list(st.homotopy), "morphology": list(st.morphology),
                       "detour_m": st.detour_m, "n_frames": int(T),
                       "n_active_slots": len(prog.active_slots),
                       "goal_reached": goal_ok,
                       "collision_free": rep["collision_free"],
                       "max_penetration_m": rep["max_penetration_m"],
                       "min_clearance_m": rep["min_clearance_m"],
                       "valid": valid,
                       "vec_row": len(vectors) if valid else -1}
                labels.append(rec)
                if valid:
                    vectors.append(vec)
            done += len(chunk)
            if done % 200 < BATCH:
                print(f"  validated {done}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "scenes.jsonl", "w") as fh:
        for sc in scenes:
            fh.write(json.dumps(sc.to_dict()) + "\n")
    with open(out / "labels.jsonl", "w") as fh:
        for r in labels:
            fh.write(json.dumps(r) + "\n")
    np.save(out / "programs.npy", np.asarray(vectors, dtype=np.float32))

    valid_per_scene = Counter()
    for sc in scenes:
        valid_per_scene[sum(1 for r in labels
                            if r["scene_id"] == sc.scene_id and r["valid"])] += 1
    summary = {
        "n_scenes": len(scenes), "n_candidates": len(jobs), "n_valid": len(vectors),
        "validation_rate": len(vectors) / max(len(jobs), 1),
        "candidates_per_scene": dict(sorted(per_scene.items())),
        "valid_per_scene": dict(sorted(valid_per_scene.items())),
        "multimodal_scenes": sum(v for k, v in valid_per_scene.items() if k >= 2),
        "dim_c": int(np.asarray(vectors).shape[1]) if vectors else 0,
        "model": runner.model_name, "fps": fps, "seed0": args.seed0,
        "enrich": args.enrich, "enriched_mix": dict(kept) if args.enrich else None,
        "note": ("scene marginal is enriched for multimodality; the conditional p(C|S) is "
                 "what is learned and evaluation stays on the unenriched ladders"),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
