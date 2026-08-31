"""Calibrate the repair's anticipation lead against the real prior.

The first Phase 4A run raised a specific suspicion. Two repairs took the m018 TCN to 100%
collision-free, but the repaired schedules crouched MORE than the heuristic (13.7 cm of excess
vs 8.3 cm) while satisfying the 0.18 m target LESS often (0.417 vs 0.583). The heuristic's
schedule has a lower peak and a larger duck integral -- it commits earlier and holds longer.

That is the signature of a lead that is too short. The repair was adding depth (mean magnitude
131 mm) and almost no width (onset shifted 61 mm earlier, duration 61 mm longer), and against
a first-order body a narrow deep command lands shallower than a broad moderate one.

`3*tau` is the textbook settling time of the lag. This asks the prior directly whether it is
the right number, holding everything else fixed and varying only the lead.

    python -m scene2motion.verify.experiment_lead --leads 3 5 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.cache import ClipCache
from ..demo.scene_builder import BeamParams, build
from ..demo.schedules import SPEED, all_schedules, response
from ..demo.strategy_planner import evaluate
from ..optim.response import DIP_MAX
from ..optim.scheduler import MARGIN_M
from .loop import run
from .repair import LEAD_TAUS

OUT = Path("outputs/phase4a_lead")
CACHE = Path("scene2motion/demo_outputs/clips")
STAND_TOP = 1.35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=float, nargs="+", default=[1.5, 3.0, 5.0, 8.0])
    ap.add_argument("--n-beams", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--heights", type=float, nargs="+", default=[0.95, 1.05, 1.20])
    ap.add_argument("--gaps", type=float, nargs="+", default=[1.5, 2.5, 3.5])
    ap.add_argument("--width", type=float, default=2.25)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    resp = response()
    cache = ClipCache(CACHE)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    for n in a.n_beams:
        for h in a.heights:
            for g in a.gaps:
                bp = BeamParams(beam_height=h, beam_width=a.width, n_beams=n, gap=g).clamped()
                sc = build(bp)
                st = evaluate(sc, "shortest")
                if not st.feasible:
                    continue
                sched = all_schedules(sc, st.plan, speed=SPEED)
                dip = sched["schedules"].get("optimized")
                if dip is None:
                    continue
                s_m = np.asarray(sched["s_m"], float)
                q0 = np.clip(np.asarray(dip, float) / DIP_MAX, 0.0, 1.0)
                needed = max(0.0, STAND_TOP - h)
                for lead in a.leads:
                    r = run(sc, st.plan, q0, resp, cache, preference="shortest",
                            max_repairs=a.max_repairs, lead_taus=lead)
                    tr = r.final.trace
                    dip_f = np.asarray(r.provenance["dip_final_m"], float)
                    rows.append({"scene_id": sc.scene_id, "n_beams": n, "beam_h": h, "gap": g,
                                 "lead_taus": lead, "outcome": r.outcome,
                                 "collision_free": tr.collision_free,
                                 "meets_target": not tr.below_margin(MARGIN_M),
                                 "min_overhead_m": round(tr.min_overhead_m, 5),
                                 "peak_dip_m": round(float(dip_f.max()), 5),
                                 "excess_crouch_m": round(float(dip_f.max() - needed), 5),
                                 "duck_integral_m2": round(float(np.trapz(dip_f, s_m)), 5),
                                 "n_repairs": len(r.repairs),
                                 "onset_shift_m": round(float(np.mean(
                                     [p.onset_shift_m for p in r.repairs])), 4) if r.repairs else 0.0,
                                 "ardy_calls": len(r.attempts),
                                 "legacy_ardy_calls": 2 * len(r.attempts)})
                print(f"n={n} h={h:.2f} g={g:.1f} done ({time.time()-t0:.0f}s)", flush=True)

    out = {"generated_at": time.time(), "elapsed_s": round(time.time() - t0, 1),
           "default_lead_taus": LEAD_TAUS, "target_m": MARGIN_M, "by_lead": {}, "rows": rows}
    print(f"\n{'lead':>6s} {'n':>4s} {'coll-free':>10s} {'meets .18':>10s} {'min over':>9s} "
          f"{'peak dip':>9s} {'excess':>8s} {'∫duck':>8s} {'calls':>6s}")
    for lead in a.leads:
        R = [r for r in rows if r["lead_taus"] == lead]
        if not R:
            continue
        s = {"n": len(R),
             "collision_free_rate": round(float(np.mean([r["collision_free"] for r in R])), 4),
             "margin_satisfaction_rate": round(float(np.mean([r["meets_target"] for r in R])), 4),
             "mean_min_overhead_m": round(float(np.mean([r["min_overhead_m"] for r in R])), 4),
             "mean_peak_dip_m": round(float(np.mean([r["peak_dip_m"] for r in R])), 4),
             "mean_excess_crouch_m": round(float(np.mean([r["excess_crouch_m"] for r in R])), 4),
             "mean_duck_integral_m2": round(float(np.mean([r["duck_integral_m2"] for r in R])), 4),
             "mean_ardy_calls": round(float(np.mean([r["ardy_calls"] for r in R])), 2)}
        out["by_lead"][str(lead)] = s
        print(f"{lead:6.1f} {s['n']:4d} {s['collision_free_rate']:10.3f} "
              f"{s['margin_satisfaction_rate']:10.3f} {s['mean_min_overhead_m']*100:8.1f}cm "
              f"{s['mean_peak_dip_m']*100:8.1f}cm {s['mean_excess_crouch_m']*100:7.1f}cm "
              f"{s['mean_duck_integral_m2']:8.3f} {s['mean_ardy_calls']:6.1f}")
    (a.out / "lead_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"-> {a.out / 'lead_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
