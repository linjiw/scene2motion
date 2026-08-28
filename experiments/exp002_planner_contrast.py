"""EXP-002: does planning in the body's configuration space actually change anything?

The headline V0 experiment. On a counterfactual scene suite, three planners that differ
ONLY in what body volume they treat as blocking are each taken all the way to a generated
G1 motion and scored with the robot's own collision geometry:

  PELVIS     only the pelvis band blocks -- what (x, y, theta) humanoid navigation reduces
             to. Expected: high plan-success, high whole-body COLLISION.
  STANDING   the whole standing volume blocks. Expected: zero collision, and a large number
             of scenes declared INFEASIBLE that are in fact traversable.
  ADAPTIVE   A* over (x, y, body mode), modes calibrated in EXP-001*. Expected: high
             plan-success AND low collision -- if the prior actually delivers the envelopes
             it was calibrated for.

That last "if" is the real question. The planner assumes measured envelopes hold; nothing
guarantees the prior reproduces them once the request is a curved path with a mode schedule
rather than the straight-line calibration setting. This experiment is where that assumption
gets tested rather than asserted.

Per scene and planner we generate TWO clips:
  path-only   the plan's route with no body adaptation -- the matched control
  adapted     the same route with the mode schedule rendered into constraint channels
Differencing them isolates the effect of the adaptation from the effect of the route, and
gives EXP-003's locality metric its baseline for free.

Reported per clip: plan feasibility, goal success, whole-body collision-free, minimum
clearance, maximum penetration, foot-ground penetration and foot slip. Collision and the
physical-consistency signals stay in separate columns: foot slip is a proxy for tracker
failure, not a measurement of it, and merging them would overstate what has been shown.
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
from scene2motion.scenes import build_suite, save_scenes  # noqa: E402

PLANNERS = ["pelvis", "standing", "adaptive"]
PROMPT = "A person walks forward."
SPEED = 0.9
GOAL_TOL = 0.5     # m, how close the pelvis must end to the goal to count as reached
MAX_DURATION = 14.0


def foot_slip(body: G1Body, qpos: np.ndarray, fps: float, thresh: float = 0.03) -> float:
    foot = [g for g in body.robot_geoms if "foot" in body.geom_name[g]]
    if not foot:
        return float("nan")
    pos = []
    for q in qpos:
        body.fk(q)
        pos.append(body.data.geom_xpos[foot].copy())
    pos = np.asarray(pos)
    vel = np.linalg.norm(np.diff(pos[:, :, :2], axis=0), axis=-1) * fps
    contact = (pos[:-1, :, 2] - body.model.geom_size[foot, 0][None, :]) < thresh
    return float(vel[contact].mean()) if contact.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp002")
    ap.add_argument("--seeds_per_rung", type=int, default=4)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scenes = build_suite(args.seeds_per_rung, args.families)
    save_scenes(out / "scenes.jsonl", scenes)
    print(f"{len(scenes)} scenes x {len(PLANNERS)} planners", flush=True)

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    free_body = G1Body(None)
    rows = []

    # Plan everything first: planning is CPU-only and cheap, and knowing which
    # (scene, planner) pairs are feasible lets the GPU work be batched by clip length.
    jobs = []
    for sc in scenes:
        for pl in PLANNERS:
            p = plan(sc, pl)
            base = {"scene_id": sc.scene_id, "family": sc.family, "planner": pl,
                    "param_name": sc.param_name, "param_value": sc.param_value,
                    "required_adaptation": sc.required_adaptation,
                    "plan_feasible": p.feasible, "plan_length_m": p.length}
            if not p.feasible:
                rows.append({**base, "variant": "n/a", "goal_reached": False,
                             "collision_free": None})
                continue
            T = min(int(MAX_DURATION * fps),
                    max(int(2 * fps), int(round(p.length / SPEED * fps))))
            jobs.append((sc, p, base, T))
    print(f"{len(jobs)} feasible (scene, planner) pairs to generate "
          f"({time.time()-t0:.0f}s planning)", flush=True)

    by_T = defaultdict(list)
    for j in jobs:
        by_T[j[3]].append(j)

    done = 0
    for T, group in sorted(by_T.items()):
        for i in range(0, len(group), 8):
            chunk = group[i:i + 8]
            # 1. matched control: the route, no body adaptation
            ctrl_specs = [plan_to_path_spec(p, fps, SPEED, duration=T / fps)
                          for _, p, _, _ in chunk]
            # Per-sample seeds derived from the scene id: a clip's noise no longer depends
            # on which other clips happened to share its batch, so the table is reproducible
            # regardless of how the suite is chunked.
            sseeds = [int(hashlib.sha1(sc.scene_id.encode()).hexdigest()[:8], 16) % (2**31)
                      for sc, _, _, _ in chunk]
            ctrls = runner.generate([PROMPT] * len(chunk), ctrl_specs, T,
                                    args.diffusion_steps, seeds=sseeds)
            # 2. adapted: the same route plus the mode schedule
            adapt_specs = [plan_to_spec(p, fps, c, runner.joint_names, SPEED,
                                        duration=T / fps)
                           for (_, p, _, _), c in zip(chunk, ctrls)]
            adapts = runner.generate([PROMPT] * len(chunk), adapt_specs, T,
                                     args.diffusion_steps, seeds=sseeds)

            for (sc, p, base, _), ctrl, adapt, cspec in zip(chunk, ctrls, adapts, ctrl_specs):
                body = G1Body(sc)
                for variant, sample in (("path_only", ctrl), ("adapted", adapt)):
                    q = runner.to_qpos(sample)
                    rep = body.trajectory_report(q)
                    end = q[-1, :2]
                    rows.append({
                        **base, "variant": variant,
                        "goal_reached": bool(np.linalg.norm(end - np.array(sc.goal)) < GOAL_TOL),
                        "goal_dist_m": float(np.linalg.norm(end - np.array(sc.goal))),
                        "collision_free": rep["collision_free"],
                        "penetration_frames": rep["penetration_frames"],
                        "penetration_fraction": rep["penetration_fraction"],
                        "max_penetration_m": rep["max_penetration_m"],
                        "min_clearance_m": rep["min_clearance_m"],
                        "culprit_geoms": rep["culprit_geoms"],
                        "worst_scene_geom": rep["worst"]["scene_geom"],
                        "foot_floor_pen_m": rep["max_foot_floor_penetration_m"],
                        "foot_slip_mps": foot_slip(free_body, q, fps),
                        "path_err_mean_m": float(np.linalg.norm(
                            np.stack([q[:, 1], q[:, 0]], -1) - cspec.root_xz, axis=-1).mean()),
                        "n_frames": int(T),
                    })
                    if sc.scene_id.endswith("_0") or done < 40:
                        np.save(out / f"qpos_{sc.scene_id}_{base['planner']}_{variant}.npy",
                                q.astype(np.float32))
                done += 1
            if done % 80 < 8:
                print(f"  {done}/{len(jobs)} pairs ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp002_planner_contrast", "model": runner.model_name,
                   "fps": fps, "planners": PLANNERS, "prompt": PROMPT, "speed_mps": SPEED,
                   "goal_tol_m": GOAL_TOL, "seeds_per_rung": args.seeds_per_rung,
                   "n_scenes": len(scenes), "n_rows": len(rows),
                   "diffusion_steps": args.diffusion_steps,
                   "body_modes": json.loads(
                       (Path(__file__).resolve().parents[1] / "outputs/body_modes.json").read_text()),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
