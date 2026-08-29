"""Experiment A: minimal crouch on a single beam, all four body planners through real ARDY.

Same scene, same route, same seeds, same frame count. The only thing that differs is who fills
the duck channel. Excess crouch is measured against what the beam actually requires, so a
planner that ducks deeper than necessary is charged for it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.scene_builder import BeamParams, build
from ..demo.schedules import all_schedules, dip_for_layer
from ..demo.strategy_planner import evaluate
from ..learn.predictor import spec_from_dip
from ..robot import G1Body
from .response import DuckResponse

PROMPT, SPEED, STEPS = "A person walks forward.", 0.9, 5
HEIGHTS = (1.25, 1.15, 1.05, 0.95)
WIDTH = 1.75
SEEDS = (100, 101)
METHODS = ("heuristic", "learned", "optimizer", "optimized")
LABEL = {"heuristic": "heuristic", "learned": "Phase-2 CNN",
         "optimizer": "QP teacher", "optimized": "Phase-3 TCN"}


def smooth1(v):
    return float(np.abs(np.diff(v)).mean()) if len(v) > 1 else 0.0


def smooth2(v):
    return float(np.abs(np.diff(v, n=2)).mean()) if len(v) > 2 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/duck_model_v3_m018")
    args = ap.parse_args()
    from ..runner import ArdyRunner

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    resp = DuckResponse.load()
    stand_top = float(resp.g(0.0))
    rows, t0 = [], time.time()

    for h in HEIGHTS:
        sc = build(BeamParams(h, WIDTH))
        st = evaluate(sc, "shortest")
        if not st.feasible:
            continue
        sched = all_schedules(sc, st.plan, speed=SPEED)
        s_m = np.asarray(sched["s_m"])
        body = G1Body(sc)
        needed = max(0.0, stand_top - h)          # minimum dip that clears this beam
        for meth in METHODS:
            dip = (np.asarray(sched["schedules"][meth], float)
                   if meth in sched["schedules"] else None)
            if dip is None:
                continue
            on = np.where(dip > 0.03)[0]
            onset_m = float(s_m[on[0]]) if len(on) else None
            for seed in SEEDS:
                spec = spec_from_dip(sc, st.plan.xy, dip, fps, speed=SPEED)
                out = runner.generate([PROMPT], [spec], spec.T, STEPS, seeds=[seed])[0]
                q = runner.to_qpos(out)
                rep = body.trajectory_report(q)
                rows.append({
                    "beam_h": h, "method": meth, "seed": seed,
                    "collision_free": bool(rep["collision_free"]),
                    "min_clearance_m": float(rep["min_clearance_m"]),
                    "goal_reached": bool(np.linalg.norm(q[-1, :2]
                                                        - np.asarray(sc.goal)) < 0.5),
                    "peak_dip_m": float(dip.max()),
                    "excess_crouch_m": float(dip.max() - needed),
                    "duck_integral_m2": float(np.trapz(dip, s_m)),
                    "d1": smooth1(dip), "d2": smooth2(dip),
                    "onset_m": onset_m, "beam_x": 4.0,
                })
        print(f"  beam {h:.2f} m done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{len(rows)} clips in {time.time()-t0:.1f}s")
    hdr = (f"{'method':13s} {'coll-free':>10s} {'goal':>6s} {'peak dip':>9s} "
           f"{'excess':>8s} {'∫duck':>8s} {'d1':>8s} {'d2':>8s} {'onset':>7s}")
    print(hdr)
    print("-" * len(hdr))
    summ = {}
    for meth in METHODS:
        R = [r for r in rows if r["method"] == meth]
        if not R:
            continue
        summ[meth] = {
            "n": len(R),
            "collision_free_rate": float(np.mean([r["collision_free"] for r in R])),
            "goal_reached_rate": float(np.mean([r["goal_reached"] for r in R])),
            "mean_peak_dip_m": float(np.mean([r["peak_dip_m"] for r in R])),
            "mean_excess_crouch_m": float(np.mean([r["excess_crouch_m"] for r in R])),
            "mean_duck_integral_m2": float(np.mean([r["duck_integral_m2"] for r in R])),
            "mean_d1": float(np.mean([r["d1"] for r in R])),
            "mean_d2": float(np.mean([r["d2"] for r in R])),
            "mean_onset_m": float(np.mean([r["onset_m"] for r in R if r["onset_m"] is not None]))
            if any(r["onset_m"] is not None for r in R) else None,
            "worst_clearance_m": float(np.min([r["min_clearance_m"] for r in R])),
        }
        s = summ[meth]
        print(f"{LABEL[meth]:13s} {s['collision_free_rate']:10.3f} "
              f"{s['goal_reached_rate']:6.2f} {s['mean_peak_dip_m']*100:8.1f}cm "
              f"{s['mean_excess_crouch_m']*100:7.1f}cm {s['mean_duck_integral_m2']:8.3f} "
              f"{s['mean_d1']:8.4f} {s['mean_d2']:8.4f} "
              f"{(s['mean_onset_m'] or float('nan')):6.2f}m")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_a.json").write_text(json.dumps(
        {"heights": list(HEIGHTS), "seeds": list(SEEDS), "stand_top_m": stand_top,
         "rows": rows, "summary": summ, "wall_clock_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\nwrote {out}/experiment_a.json")


if __name__ == "__main__":
    main()
