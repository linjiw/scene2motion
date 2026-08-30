"""Search the FINAL m018 system for a genuine, reproducible verification failure.

The demo's repair story is only honest if the failure it repairs is real. The provisional
0.12-margin model collided readily; that model is quarantined and its failures say nothing
about the shipped 0.18 system. So this sweeps the final m018 TCN over scenes harder than
anything it was trained on -- 3 to 6 beams, where the training distribution stopped at 2 --
and over boundary-clearance beams whose headroom sits within a few centimetres of the target,
and reports what actually happens.

A null result is a result. If the sampled final system does not fail, this says so, and the
demo says so too.

    python -m scene2motion.verify.search_failures --n-beams 3 4 5 6 --limit 24
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.cache import ClipCache
from ..demo.scene_builder import BeamParams, build
from ..demo.schedules import SPEED, all_schedules, dip_for_layer, response
from ..demo.strategy_planner import evaluate
from ..optim.response import DIP_MAX
from ..optim.scheduler import MARGIN_M
from .loop import run
from .trace import schedule_hash

OUT = Path("outputs/phase4_failure_search")
CACHE = Path("scene2motion/demo_outputs/clips")

# Heights chosen around the point where a standing G1 stops fitting: the demo's stand mode
# needs ~1.35 m, so 1.30 and below force a duck, and 0.85 is near the deepest the response
# can reach. The tight end is where a surrogate error is most likely to matter.
HEIGHTS = (0.85, 0.95, 1.05, 1.20)
GAPS = (1.5, 2.5, 3.5)
# Wide, because a beam narrow enough to walk past is a ROUTE question, not a body question.
# 2.25 m is the widest the builder allows and leaves a 0.15 m bypass -- far narrower than the
# 0.38 m footprint radius -- so the route must pass underneath and the duck schedule is what
# is actually on trial. 1.45 m is the demo default, still unwalkable-around at this corridor.
WIDTHS = (1.45, 2.25)


def scenes(counts, limit: int | None = None):
    out = []
    for n in counts:
        for h in HEIGHTS:
            for gap in GAPS:
                for w in WIDTHS:
                    out.append(BeamParams(beam_height=h, beam_width=w, n_beams=n, gap=gap).clamped())
    # Interleave by beam count so a truncated run still covers every count.
    out.sort(key=lambda b: (HEIGHTS.index(b.beam_height) if b.beam_height in HEIGHTS else 9,
                            b.n_beams))
    return out[:limit] if limit else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-beams", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--preference", default="clearance")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-generate", action="store_true")
    a = ap.parse_args()

    resp = response()
    if resp is None:
        raise SystemExit("no fitted duck response at outputs/duck_response/response.json")
    cache = ClipCache(CACHE)
    a.out.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for bp in scenes(a.n_beams, a.limit):
        sc = build(bp)
        st = evaluate(sc, a.preference)
        if not st.feasible:
            rows.append({"beam": bp.__dict__, "skipped": "route infeasible"})
            continue
        sched = all_schedules(sc, st.plan, speed=SPEED)
        dip = dip_for_layer(sched, "optimized")
        if dip is None:
            rows.append({"beam": bp.__dict__, "skipped": "no m018 schedule"})
            continue
        q0 = np.clip(dip / DIP_MAX, 0.0, 1.0)
        for seed in a.seeds:
            # max_repairs=0: this pass only asks whether the one-shot system fails. Repair is
            # measured separately, on whatever this finds.
            r = run(sc, st.plan, q0, resp, cache, preference=a.preference, seed=seed,
                    max_repairs=0, allow_generate=not a.no_generate,
                    provenance={"beam": bp.__dict__, "tcn_dataset_hash": sched.get("tcn_dataset_hash"),
                                "tcn_margin_m": sched.get("tcn_margin_m"),
                                "route_len_m": sched["route_len_m"],
                                "schedule_hash": schedule_hash(q0)})
            d = r.to_dict()
            d["beam"] = bp.__dict__
            d["seed"] = seed
            rows.append(d)
            f = d["attempts"][-1] if d["attempts"] else {}
            print(f"n={bp.n_beams} h={bp.beam_height:.2f} w={bp.beam_width:.2f} gap={bp.gap:.1f} "
                  f"seed={seed}: {d['outcome']:16s} over={f.get('min_overhead_m', float('nan')):+.4f} "
                  f"tot={f.get('min_clearance_m', float('nan')):+.4f} "
                  f"deficit={f.get('max_deficit_m', 0):.4f} lat={f.get('max_lateral_deficit_m', 0):.3f} "
                  f"peak_dip={f.get('peak_dip_m', 0):.3f} "
                  f"goal_err={f.get('goal_error_m', float('nan')):.2f} [{f.get('source','-')}]",
                  flush=True)

    done = [r for r in rows if "outcome" in r]
    coll = [r for r in done if not r["attempts"][-1]["collision_free"]]
    short = [r for r in done if r["attempts"][-1]["collision_free"]
             and not r["attempts"][-1]["meets_target"]]
    summary = {"generated_at": time.time(), "elapsed_s": round(time.time() - t0, 1),
               "target_m": MARGIN_M, "n_scenes": len(done), "n_skipped": len(rows) - len(done),
               "n_collision": len(coll), "n_below_margin": len(short),
               "n_clean": len(done) - len(coll) - len(short),
               "collision_free_rate": round(1 - len(coll) / max(len(done), 1), 4),
               "margin_satisfaction_rate": round(1 - (len(coll) + len(short)) / max(len(done), 1), 4),
               "worst_min_clearance_m": round(min((r["attempts"][-1]["min_clearance_m"]
                                                   for r in done), default=float("nan")), 5),
               "worst_min_overhead_m": round(min((r["attempts"][-1]["min_overhead_m"]
                                                  for r in done), default=float("nan")), 5),
               "n_lateral_tight": sum(1 for r in done
                                      if r["attempts"][-1]["max_lateral_deficit_m"] > 1e-9),
               "rows": rows}
    (a.out / "search.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(done)} scenes: {len(coll)} collision, {len(short)} below the {MARGIN_M} m "
          f"OVERHEAD target, {summary['n_clean']} clean. worst overhead "
          f"{summary['worst_min_overhead_m']:+.4f} m, worst total "
          f"{summary['worst_min_clearance_m']:+.4f} m, {summary['n_lateral_tight']} laterally tight")
    print(f"-> {a.out / 'search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
