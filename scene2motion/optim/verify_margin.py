"""How much margin does the optimiser actually need? Measured against ARDY, not the surrogate.

The optimiser plans against a fitted response whose holdout error is ~43 mm and whose per-seed
scatter is 30-74 mm. A margin smaller than that error is not a safety factor, it is a rounding
error, and the demo showed the consequence immediately: a minimal-crouch schedule at the default
0.12 m margin produced a clip with -0.02 m of clearance -- an actual collision.

So this sweeps the margin and measures the collision-free rate through the real prior. The
answer is the margin at which minimal-crouch stops trading clearance for effort, and it is a
property of the surrogate's error, not of the optimiser.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.scene_builder import BeamParams, build
from ..demo.strategy_planner import evaluate
from ..learn.predictor import spec_from_dip
from ..learn.route_profile import N_SAMPLES, profile
from ..robot import G1Body
from .response import DIP_MAX, DuckResponse
from .scheduler import dt_for, solve

PROMPT, SPEED, STEPS = "A person walks forward.", 0.9, 5
SCENES = [(1.20, 1.75, 1, 3.0), (1.10, 1.75, 1, 3.0), (1.00, 1.75, 1, 3.0),
          (1.05, 1.75, 2, 2.0), (1.05, 1.75, 2, 5.0)]
MARGINS = (0.12, 0.18, 0.24, 0.30)
SEEDS = (100, 101)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/duck_model_v3_m018")
    args = ap.parse_args()
    from ..runner import ArdyRunner

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    resp = DuckResponse.load()
    rows, t0 = [], time.time()

    for margin in MARGINS:
        for (h, w, nb, gap) in SCENES:
            sc = build(BeamParams(h, w, n_beams=nb, gap=gap))
            st = evaluate(sc, "shortest")
            if not st.feasible:
                continue
            prof = profile(sc, st.plan.xy, speed=SPEED)
            L = float(np.linalg.norm(np.diff(np.asarray(st.plan.xy), axis=0), axis=1).sum())
            sol = solve(prof[:, 0], resp, dt_for(L, N_SAMPLES, SPEED), margin_m=margin)
            if not sol.feasible:
                continue
            body = G1Body(sc)
            for seed in SEEDS:
                spec = spec_from_dip(sc, st.plan.xy, sol.q * DIP_MAX, fps, speed=SPEED)
                out = runner.generate([PROMPT], [spec], spec.T, STEPS, seeds=[seed])[0]
                q = runner.to_qpos(out)
                rep = body.trajectory_report(q)
                rows.append({"margin_m": margin, "beam_h": h, "n_beams": nb, "gap": gap,
                             "seed": seed, "peak_dip_m": float(sol.q.max() * DIP_MAX),
                             "collision_free": bool(rep["collision_free"]),
                             "min_clearance_m": float(rep["min_clearance_m"]),
                             "goal_err_m": float(np.linalg.norm(q[-1, :2]
                                                                - np.asarray(sc.goal)))})
        R = [r for r in rows if r["margin_m"] == margin]
        if R:
            print(f"  margin {margin:.2f} m: collision-free "
                  f"{np.mean([r['collision_free'] for r in R]):.3f}  "
                  f"mean peak dip {np.mean([r['peak_dip_m'] for r in R])*100:.1f} cm  "
                  f"min clearance {np.min([r['min_clearance_m'] for r in R]):+.3f} m  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    summ = {}
    for m in MARGINS:
        R = [r for r in rows if r["margin_m"] == m]
        if R:
            summ[str(m)] = {
                "n": len(R),
                "collision_free_rate": float(np.mean([r["collision_free"] for r in R])),
                "mean_peak_dip_m": float(np.mean([r["peak_dip_m"] for r in R])),
                "worst_clearance_m": float(np.min([r["min_clearance_m"] for r in R])),
                "goal_reached_rate": float(np.mean([r["goal_err_m"] < 0.5 for r in R])),
            }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "margin_sweep.json").write_text(json.dumps(
        {"scenes": SCENES, "margins": list(MARGINS), "seeds": list(SEEDS),
         "rows": rows, "summary": summ,
         "surrogate_holdout_mae_m": resp.fit.get("static_direct", {}).get("holdout_mae_m"),
         "wall_clock_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\n{len(rows)} clips in {time.time()-t0:.1f}s -> {out}/margin_sweep.json")


if __name__ == "__main__":
    main()
