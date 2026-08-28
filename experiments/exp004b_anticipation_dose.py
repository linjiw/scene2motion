"""EXP-004b: how much lead time does a whole-body adaptation actually need?

Why the onset metric is not enough
----------------------------------
EXP-004 measures WHEN the adaptation starts, and gets 1.31 s of lead for the adaptive
renderer. That number is not evidence about the prior: `plan_to_spec` dilates the mode
schedule by a `lead_s` constant, ARDY hard-infills the root slice, and the achieved onset
therefore tracks the commanded onset to within a frame. Onset measures the renderer's own
constant. Worse, the `global` strawman — which ducks for the entire clip — scores a LARGER
lead (3.75 s), so a bigger onset number is not even better.

The causal question is different and answerable: **sweep the lead time and watch collisions.**
If anticipation is doing real work, penetration falls as lead grows and saturates once the
body has enough time to change envelope. The saturation point is the required lead, measured
rather than assumed, and the shape of the curve says whether the +/-0.8 s constant that EXP-002
adopted was lucky or right.

Everything except `lead_s` is held fixed, including the noise draw: `seeds=` gives each sample
its own generator, so two clips differing only in lead differ only in lead. Without that, the
whole curve sits inside the sampling noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.planner import plan, plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS, LADDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED = 0.9
DURATION = 10.0
LEADS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0]
SAMPLE_SEEDS = [11, 12, 13]
FAMILIES = ["overhead_beam", "beam_and_gap"]
LATERAL = np.array([0.0, 1.0, 0.0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp004b")
    ap.add_argument("--seeds_per_rung", type=int, default=3)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DURATION * fps)
    rows = []

    for fam in FAMILIES:
        for sseed in range(args.seeds_per_rung):
            for v in LADDERS[fam]:
                sc = BUILDERS[fam](v, sseed)
                p = plan(sc, "adaptive")
                if not p.feasible:
                    continue
                body = G1Body(sc)
                ctrl = runner.generate(
                    [PROMPT], [plan_to_path_spec(p, fps, SPEED, duration=DURATION)], T,
                    args.diffusion_steps, seeds=[SAMPLE_SEEDS[0]])[0]
                for sd in SAMPLE_SEEDS:
                    specs = [plan_to_spec(p, fps, ctrl, runner.joint_names, SPEED,
                                          duration=DURATION, lead_s=L) for L in LEADS]
                    outs = []
                    for i in range(0, len(specs), 8):
                        chunk = specs[i:i + 8]
                        outs += runner.generate([PROMPT] * len(chunk), chunk, T,
                                                args.diffusion_steps,
                                                seeds=[sd] * len(chunk))
                    for L, o, spec in zip(LEADS, outs, specs):
                        q = runner.to_qpos(o)
                        rep = body.trajectory_report(q)
                        tops = np.array([body.top_height(x) for x in q])
                        rows.append({
                            "family": fam, "ladder_seed": sseed, "rung": v,
                            "lead_s": L, "sample_seed": sd,
                            "collision_free": rep["collision_free"],
                            "max_penetration_m": rep["max_penetration_m"],
                            "penetration_frames": rep["penetration_frames"],
                            "min_clearance_m": rep["min_clearance_m"],
                            "min_top_m": float(tops.min()),
                            "requested_min_pelvis_m": float(np.min(spec.root_y)),
                            "achieved_min_pelvis_m": float(q[:, 2].min()),
                            "foot_floor_pen_m": rep["max_foot_floor_penetration_m"],
                        })
            print(f"  {fam} seed {sseed} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp004b_anticipation_dose", "model": runner.model_name,
                   "fps": fps, "duration_s": DURATION, "leads_s": LEADS,
                   "sample_seeds": SAMPLE_SEEDS, "families": FAMILIES,
                   "per_sample_noise": True, "n_rows": len(rows),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
