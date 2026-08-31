"""Phase 4B experiment: does scoring the body change which route gets chosen, and is it right?

Four selectors over the same candidate list per scene. Two questions, kept apart:

    agreement / regret   a pure selection question, answered on the common QP scale with no
                         generation at all. Does a learned body cost pick the same route the
                         convex reference picks, and what does it cost when it does not?
    collision / margin   an execution question, answered by generating the chosen route
                         through the frozen prior and verifying it. Only tcn_verify is
                         allowed to change its mind at this stage.

Separating them matters because a selector can pick the right route and still produce a clip
that collides -- that is a scheduling failure, not a selection failure, and Phase 4A already
measured it.

    python -m scene2motion.verify.experiment_select --latency
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..demo.cache import ClipCache
from ..demo.scene_builder import BeamParams, build
from ..demo.schedules import SPEED, response
from ..optim.response import DIP_MAX
from ..optim.scheduler import MARGIN_M
from .loop import run as run_loop
from .routes import candidates
from .select import batched_latency, load_tcn, pick, ranking, regret, score_all

OUT = Path("outputs/phase4b_select")
CACHE = Path("scene2motion/demo_outputs/clips")
RULES = ("oracle_qp", "heuristic", "tcn_body", "tcn_verify")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-beams", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--heights", type=float, nargs="+", default=[0.95, 1.05, 1.20])
    ap.add_argument("--widths", type=float, nargs="+", default=[1.45, 2.25])
    ap.add_argument("--gaps", type=float, nargs="+", default=[2.5])
    ap.add_argument("--k", type=int, default=8, help="candidate routes per scene")
    ap.add_argument("--preference", default="balanced")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--latency-k", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    resp, model = response(), load_tcn()
    if resp is None or model is None:
        raise SystemExit("need both outputs/duck_response and the m018 checkpoint")
    cache = ClipCache(CACHE)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    for n in a.n_beams:
        for h in a.heights:
            for w in a.widths:
                for g in a.gaps:
                    bp = BeamParams(beam_height=h, beam_width=w, n_beams=n, gap=g).clamped()
                    sc = build(bp)
                    routes = candidates(sc, k=a.k)
                    if len(routes) < 2:
                        rows.append({"scene_id": sc.scene_id, "skipped": "fewer than 2 routes",
                                     "n_routes": len(routes)})
                        continue
                    S = score_all(sc, routes, resp, model, a.preference, SPEED)
                    row = {"scene_id": sc.scene_id, "n_beams": n, "beam_h": h, "beam_w": w,
                           "gap": g, "n_routes": len(routes),
                           "routes": [{"label": s.route.label, "len_m": round(float(s.s_m[-1]), 3),
                                       "c_pred_m": round(s.c_pred_m, 4),
                                       "peak_qp_m": (None if s.u_qp is None
                                                     else round(float(s.u_qp.max() * DIP_MAX), 4)),
                                       "peak_tcn_m": round(float(s.u_tcn.max() * DIP_MAX), 4),
                                       "j_qp": None if s.j_qp is None else round(s.j_qp, 3),
                                       "j_tcn": round(s.j_tcn, 3)} for s in S],
                           "picks": {}}
                    oracle = pick(S, "oracle_qp")
                    for rule in ("oracle_qp", "heuristic", "tcn_body"):
                        i = pick(S, rule)
                        row["picks"][rule] = {"index": i, "label": S[i].route.label,
                                              "agrees_with_oracle": i == oracle,
                                              "regret": round(regret(S, i), 4)}

                    # tcn_verify: walk the TCN's ranking, generating and verifying until one
                    # is accepted. A route that still collides after two repairs is abandoned,
                    # which is the "select around this route" fallback.
                    order, tried, chosen, ver = ranking(S, "tcn_body"), [], None, None
                    for i in order:
                        s = S[i]
                        r = run_loop(sc, s.route.plan, s.u_tcn, resp, cache,
                                     preference="shortest", max_repairs=2,
                                     provenance={"scene": sc.scene_id, "route": s.route.label})
                        tried.append({"index": int(i), "label": s.route.label,
                                      "outcome": r.outcome, "repaired": r.repaired,
                                      "ardy_calls": len(r.attempts),
                                      "legacy_ardy_calls": 2 * len(r.attempts),
                                      "min_overhead_m": round(r.final.trace.min_overhead_m, 4)})
                        if r.outcome != "rejected":
                            chosen, ver = int(i), r
                            break
                    if chosen is None and order:
                        chosen, ver = int(order[0]), r
                    row["picks"]["tcn_verify"] = {
                        "index": chosen, "label": S[chosen].route.label,
                        "agrees_with_oracle": chosen == oracle,
                        "regret": round(regret(S, chosen), 4),
                        "fell_back": len(tried) > 1, "tried": tried,
                        "ardy_calls": sum(t["ardy_calls"] for t in tried)}
                    row["verified"] = {
                        "outcome": ver.outcome, "repaired": ver.repaired,
                        "collision_free": ver.final.trace.collision_free,
                        "meets_target": not ver.final.trace.below_margin(MARGIN_M),
                        "min_overhead_m": round(ver.final.trace.min_overhead_m, 4),
                        "goal_error_m": round(ver.final.trace.goal_error_m, 4)}
                    rows.append(row)
                    p = row["picks"]
                    print(f"n={n} h={h:.2f} w={w:.2f}: {len(routes)} routes | "
                          f"oracle {S[oracle].route.label:16s} | heur {p['heuristic']['label']:16s} "
                          f"(reg {p['heuristic']['regret']:7.2f}) | tcn {p['tcn_body']['label']:16s} "
                          f"(reg {p['tcn_body']['regret']:6.2f}) | verify {p['tcn_verify']['label']:16s} "
                          f"{'FALLBACK ' if p['tcn_verify']['fell_back'] else ''}"
                          f"{row['verified']['outcome']}", flush=True)

    done = [r for r in rows if "picks" in r]
    summary = {}
    for rule in RULES:
        P = [r["picks"][rule] for r in done]
        s = {"n": len(P),
             "agreement_with_oracle": round(float(np.mean([p["agrees_with_oracle"] for p in P])), 4),
             "mean_regret": round(float(np.mean([p["regret"] for p in P])), 4),
             "max_regret": round(float(np.max([p["regret"] for p in P])), 4),
             "median_regret": round(float(np.median([p["regret"] for p in P])), 4)}
        if rule == "tcn_verify":
            s["fallback_rate"] = round(float(np.mean([p["fell_back"] for p in P])), 4)
            s["mean_ardy_calls"] = round(float(np.mean([p["ardy_calls"] for p in P])), 2)
            V = [r["verified"] for r in done]
            s["collision_free_rate"] = round(float(np.mean([v["collision_free"] for v in V])), 4)
            s["margin_satisfaction_rate"] = round(float(np.mean([v["meets_target"] for v in V])), 4)
            s["repaired_rate"] = round(float(np.mean([v["repaired"] for v in V])), 4)
        summary[rule] = s

    lat = []
    if a.latency:
        sc = build(BeamParams(1.05, 2.25, 5, 2.5).clamped())
        routes = candidates(sc, k=a.k)
        for k in a.latency_k:
            lat.append(batched_latency(sc, routes, resp, model, k, SPEED))
            print(f"  latency k={k}: TCN {lat[-1]['tcn_batched_ms']} ms batched vs QP "
                  f"{lat[-1]['qp_sequential_ms']} ms sequential ({lat[-1]['speedup']}x)", flush=True)

    payload = {"generated_at": time.time(), "elapsed_s": round(time.time() - t0, 1),
               "preference": a.preference, "k_candidates": a.k, "target_m": MARGIN_M,
               "summary": summary, "latency": lat, "n_scenes": len(done),
               "n_skipped": len(rows) - len(done), "rows": rows}
    (a.out / "select.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{len(done)} scenes, {len(rows)-len(done)} skipped, {time.time()-t0:.0f}s\n")
    print(f"{'selector':12s} {'agree':>7s} {'mean reg':>9s} {'med reg':>8s} {'max reg':>8s}")
    for rule in RULES:
        s = summary[rule]
        print(f"{rule:12s} {s['agreement_with_oracle']:7.3f} {s['mean_regret']:9.2f} "
              f"{s['median_regret']:8.2f} {s['max_regret']:8.2f}")
    v = summary["tcn_verify"]
    print(f"\ntcn_verify: fallback {v['fallback_rate']:.3f} · collision-free "
          f"{v['collision_free_rate']:.3f} · meets target {v['margin_satisfaction_rate']:.3f} "
          f"· repaired {v['repaired_rate']:.3f} · {v['mean_ardy_calls']:.1f} ARDY calls")
    print(f"-> {a.out / 'select.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
