"""EXP-005a: is the 39-number constraint program expressive enough to be worth learning?

The precondition for EXP-005
----------------------------
`p_phi(C | S, s, g)` can only ever be as good as C. If compressing an oracle plan into 39
numbers loses enough fidelity that the decoded program no longer traverses the scene, then a
perfect generator over that space is still useless, and no amount of model capacity fixes it.
So this is the cheapest experiment that could falsify the whole EXP-005 design, and it runs
before any training code is written.

Method
------
For every evaluation scene and every enumerated strategy:

    oracle plan --> plan_to_spec --> ARDY --> collision check     (the reference)
    oracle plan --> encode --> 39 numbers --> decode --> ARDY --> collision check

Both paths share `_limb_targets`, `_dilate_channel` and `LEAD_S`, so any difference is
attributable to the COMPRESSION and not to two code paths disagreeing. Identical per-sample
noise seeds mean the two clips differ only in their request.

The number that matters is not the L-infinity error between the two requests — it is whether
the compressed one still reaches the goal without hitting anything. A 13 cm route error is
alarming in the abstract and irrelevant if the corridor is 1.4 m wide; it is fatal if the gap
is 0.5 m. Only generation and collision checking can tell which.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.planner import plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.program import ConstraintProgram, decode, encode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS, LADDERS  # noqa: E402
from scene2motion.strategies import enumerate_strategies  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL = 0.9, 0.5
FAMILIES = ["overhead_beam", "partial_beam", "beam_and_gap", "pillar", "narrow_gap",
            "low_obstacle"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp005a")
    ap.add_argument("--seeds_per_rung", type=int, default=3)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    rows = []

    for fam in FAMILIES:
        for sseed in range(args.seeds_per_rung):
            for v in LADDERS[fam]:
                sc = BUILDERS[fam](v, sseed)
                body = G1Body(sc)
                for st in enumerate_strategies(sc):
                    p = st.plan
                    T = min(int(14 * fps),
                            max(int(2 * fps), int(round(p.length / SPEED * fps))))
                    seed = int(hashlib.sha1(
                        f"{sc.scene_id}|{st.name}".encode()).hexdigest()[:8], 16) % (2 ** 31)
                    ctrl = runner.generate(
                        [PROMPT], [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                        args.diffusion_steps, seeds=[seed])[0]

                    ref_spec = plan_to_spec(p, fps, ctrl, runner.joint_names, SPEED,
                                            duration=T / fps)
                    prog = encode(p, sc, fps, SPEED)
                    vec = prog.to_vec()
                    prog_spec = decode(ConstraintProgram.from_vec(vec), sc, fps, ctrl,
                                       runner.joint_names, duration=T / fps)

                    outs = runner.generate([PROMPT] * 2, [ref_spec, prog_spec], T,
                                           args.diffusion_steps, seeds=[seed, seed])
                    rec = {"scene_id": sc.scene_id, "family": fam, "ladder_seed": sseed,
                           "param_value": v, "strategy": st.name,
                           "n_active_slots": len(prog.active_slots),
                           "vec_min": float(vec.min()), "vec_max": float(vec.max())}
                    for tag, o, spec in (("oracle", outs[0], ref_spec),
                                         ("program", outs[1], prog_spec)):
                        q = runner.to_qpos(o)
                        rep = body.trajectory_report(q)
                        end = q[-1, :2]
                        rec[f"{tag}_goal"] = bool(
                            np.linalg.norm(end - np.array(sc.goal)) < GOAL_TOL)
                        rec[f"{tag}_collision_free"] = rep["collision_free"]
                        rec[f"{tag}_max_pen_m"] = rep["max_penetration_m"]
                        rec[f"{tag}_min_clear_m"] = rep["min_clearance_m"]
                        rec[f"{tag}_min_pelvis_m"] = float(q[:, 2].min())
                    n = min(len(ref_spec.root_y), len(prog_spec.root_y))
                    rec["req_root_y_err_m"] = float(
                        np.abs(ref_spec.root_y[:n] - prog_spec.root_y[:n]).max())
                    rec["req_route_err_m"] = float(np.linalg.norm(
                        ref_spec.root_xz[:n] - prog_spec.root_xz[:n], axis=-1).max())
                    rows.append(rec)
            print(f"  {fam} seed {sseed} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp005a_program_expressivity", "model": runner.model_name,
                   "fps": fps, "families": FAMILIES, "dim_c": len(vec),
                   "seeds_per_rung": args.seeds_per_rung, "n_rows": len(rows),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
