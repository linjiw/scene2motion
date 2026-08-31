"""Phase 4A experiment: does verification-guided repair pay for itself on hard scenes?

Five methods on 3-6 beam corridor-blocking scenes, all through the real frozen prior:

    heuristic       the calibrated mode lattice, one shot
    qp              the Phase 3 convex QP teacher, one shot
    tcn             the m018 residual TCN, one shot -- the shipped Phase 3 system
    tcn+1           the same, with at most one verification-guided repair
    tcn+2           the same, with at most two

3-6 beams is out of distribution by construction: the m018 dataset stopped at 2. Beams are
wide enough to block the corridor, so the route must pass underneath and the duck schedule is
what is on trial rather than the router.

Collision-free rate and target-margin satisfaction are reported SEPARATELY. They are different
claims: a motion with 4 cm of headroom did not hit anything, and also did not leave the 18 cm
the scheduler was asked for. Conflating them would both overstate the failure rate of the
one-shot system and let repair take credit for margin it never needed to buy.

    python -m scene2motion.verify.experiment_repair --heights 0.95 1.05 1.20
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
from ..demo.schedules import SPEED, all_schedules, response
from ..demo.strategy_planner import evaluate
from ..optim.response import DIP_MAX
from ..optim.scheduler import MARGIN_M
from .loop import run
from .trace import schedule_hash

OUT = Path("outputs/phase4a_repair")
CACHE = Path("scene2motion/demo_outputs/clips")
# The mode lattice's standing top-of-body height: what a beam has to be cleared FROM.
STAND_TOP = 1.35
METHODS = ("heuristic", "qp", "tcn", "tcn+1", "tcn+2")
SCHED_KEY = {"heuristic": "heuristic", "qp": "optimizer", "tcn": "optimized",
             "tcn+1": "optimized", "tcn+2": "optimized"}
REPAIRS = {"heuristic": 0, "qp": 0, "tcn": 0, "tcn+1": 1, "tcn+2": 2}


def _row(sc, bp, method, res, needed, s_m, seed) -> dict:
    # Resampling controls keep attempts chronological and may select an earlier sample.
    # Always charge outcome and motion-quality metrics to the explicitly selected clip.
    a = res.final if res.attempts else None
    d = res.to_dict()
    dip0 = np.asarray(d["provenance"].get("dip_initial_m", []), float)
    row = {"scene_id": sc.scene_id, "n_beams": bp.n_beams, "beam_h": bp.beam_height,
           "beam_w": bp.beam_width, "gap": bp.gap, "method": method, "seed": seed,
           "outcome": d["outcome"], "reason": d["reason"], "ardy_calls": d["ardy_calls"], "n_attempts": d["n_attempts"],
           "legacy_ardy_calls": d.get("legacy_ardy_calls", 2 * d["n_attempts"]),
           "ardy_calls_executed": d["ardy_calls_executed"], "cache_hits": d["cache_hits"],
           "necessary_adapted_generations": d["necessary_adapted_generations"],
           "adapted_generations_executed": d["adapted_generations_executed"],
           "selected_attempt": d["selected_attempt"],
           "n_repairs": d["n_repairs"], "repaired": d["repaired"],
           "initial_schedule_hash": d["provenance"]["initial_schedule_hash"],
           "final_schedule_hash": d["provenance"].get("final_schedule_hash"),
           "repairs": d["repairs"],
           # Compact candidate ledger. Full 64-sample traces remain in LoopResult, while the
           # experiment artifact keeps exactly enough evidence to audit resample selection.
           "attempt_ledger": [{k: attempt.get(k) for k in
                               ("iteration", "seed", "schedule_hash", "source", "clip_key",
                                "collision_free", "meets_target", "min_overhead_m",
                                "min_clearance_m", "goal_error_m")}
                              for attempt in d["attempts"]]}
    if a is None:
        return row
    dip_f = np.asarray(d["provenance"]["dip_final_m"], float)
    row.update({
        "collision_free": a.trace.collision_free,
        "meets_target": not a.trace.below_margin(MARGIN_M),
        "min_overhead_m": round(a.trace.min_overhead_m, 5),
        "min_clearance_m": round(a.trace.min_clearance_m, 5),
        "max_deficit_m": round(float(a.trace.deficit(MARGIN_M).max()), 5),
        "max_lateral_deficit_m": round(float(a.trace.lateral_deficit(MARGIN_M).max()), 5),
        "goal_reached": bool(a.trace.goal_error_m < 0.5),
        "goal_error_m": round(a.trace.goal_error_m, 4),
        # Crouch is charged against the schedule that was actually EXECUTED, so a repaired
        # clip pays for the extra dip the repair added.
        "peak_dip_m": round(float(dip_f.max()), 5),
        "excess_crouch_m": round(float(dip_f.max() - needed), 5),
        "duck_integral_m2": round(float(np.trapz(dip_f, s_m)), 5),
        "peak_dip_initial_m": round(float(dip0.max()), 5) if len(dip0) else None,
        "selected_seed": a.seed,
        "clip_key": a.key,
    })
    return row


def _aggregate(rows: list[dict]) -> dict:
    """Common Phase-4 aggregate, including both legacy and necessary-generation costs."""
    def adapted_executed(row: dict) -> float:
        if "adapted_generations_executed" in row:
            return float(row["adapted_generations_executed"])
        # A row with the additive legacy field was emitted after the unused reference was
        # removed, so its executed call count is already candidate-only. Older committed
        # rows have no such field and counted reference+candidate pairs.
        divisor = 1.0 if "legacy_ardy_calls" in row else 2.0
        return float(row["ardy_calls_executed"]) / divisor

    return {"n": len(rows),
            "collision_free_rate": round(float(np.mean([r["collision_free"] for r in rows])), 4),
            "margin_satisfaction_rate": round(float(np.mean([r["meets_target"] for r in rows])), 4),
            "goal_reached_rate": round(float(np.mean([r["goal_reached"] for r in rows])), 4),
            # Current implementation: one candidate-producing generation per attempt.
            "mean_ardy_calls": round(float(np.mean([r["ardy_calls"] for r in rows])), 2),
            "mean_legacy_ardy_calls": round(float(np.mean(
                [r.get("legacy_ardy_calls", 2 * r["n_attempts"]) for r in rows])), 2),
            # Architecture comparison axis: only generations that instantiate a candidate.
            "mean_necessary_adapted_generations": round(float(np.mean(
                [r.get("necessary_adapted_generations", r["n_attempts"]) for r in rows])), 2),
            "mean_adapted_generations_executed": round(float(np.mean(
                [adapted_executed(r) for r in rows])), 2),
            "mean_attempts": round(float(np.mean([r["n_attempts"] for r in rows])), 2),
            "mean_peak_dip_m": round(float(np.mean([r["peak_dip_m"] for r in rows])), 4),
            "mean_excess_crouch_m": round(float(np.mean([r["excess_crouch_m"] for r in rows])), 4),
            "mean_duck_integral_m2": round(float(np.mean([r["duck_integral_m2"] for r in rows])), 4),
            "mean_min_overhead_m": round(float(np.mean([r["min_overhead_m"] for r in rows])), 4),
            "repaired_rate": round(float(np.mean([r["repaired"] for r in rows])), 4),
            "rejected_rate": round(float(np.mean([r["outcome"] == "rejected" for r in rows])), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-beams", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--heights", type=float, nargs="+", default=[0.95, 1.05, 1.20])
    ap.add_argument("--gaps", type=float, nargs="+", default=[1.5, 2.5, 3.5])
    ap.add_argument("--width", type=float, default=2.25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100])
    ap.add_argument("--preference", default="shortest")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tcn-dir", type=Path, default=None,
                    help="checkpoint to use for the 'tcn' methods; defaults to m018")
    a = ap.parse_args()

    resp = response()
    if resp is None:
        raise SystemExit("no fitted duck response")
    cache = ClipCache(CACHE)
    a.out.mkdir(parents=True, exist_ok=True)

    rows, skipped, t0 = [], [], time.time()
    combos = [(n, h, g) for h in a.heights for n in a.n_beams for g in a.gaps]
    for n, h, g in combos:
        bp = BeamParams(beam_height=h, beam_width=a.width, n_beams=n, gap=g).clamped()
        sc = build(bp)
        st = evaluate(sc, a.preference)
        if not st.feasible:
            skipped.append({"beam": bp.__dict__, "why": "route infeasible"})
            continue
        sched = all_schedules(sc, st.plan, speed=SPEED, tcn_dir=a.tcn_dir)
        s_m = np.asarray(sched["s_m"], float)
        needed = max(0.0, STAND_TOP - h)
        prov_base = {"beam": bp.__dict__, "tcn_dataset_hash": sched.get("tcn_dataset_hash"),
                     "tcn_margin_m": sched.get("tcn_margin_m"),
                     "route_len_m": sched["route_len_m"], "needed_dip_m": needed,
                     "tcn_dir": sched.get("tcn_dir")}

        for method in METHODS:
            dip = sched["schedules"].get(SCHED_KEY[method])
            if dip is None:
                skipped.append({"beam": bp.__dict__, "why": f"no {method} schedule"})
                continue
            q0 = np.clip(np.asarray(dip, float) / DIP_MAX, 0.0, 1.0)
            for seed in a.seeds:
                r = run(sc, st.plan, q0, resp, cache, preference=a.preference, seed=seed,
                        max_repairs=REPAIRS[method],
                        provenance={**prov_base, "method": method,
                                    "schedule_hash": schedule_hash(q0)})
                rows.append(_row(sc, bp, method, r, needed, s_m, seed))
                f = rows[-1]
                print(f"n={n} h={h:.2f} g={g:.1f} {method:9s} seed={seed}: "
                      f"{f['outcome']:16s} over={f.get('min_overhead_m', float('nan')):+.4f} "
                      f"free={str(f.get('collision_free')):5s} target={str(f.get('meets_target')):5s} "
                      f"dip={f.get('peak_dip_m', 0):.3f} calls={f['ardy_calls']}", flush=True)

    # -- aggregate ----------------------------------------------------------------------
    by = defaultdict(list)
    for r in rows:
        if "collision_free" in r:
            by[r["method"]].append(r)

    summary = {m: _aggregate(by[m]) for m in METHODS if by[m]}
    by_count = {m: {str(n): _aggregate([r for r in by[m] if r["n_beams"] == n])
                    for n in a.n_beams if [r for r in by[m] if r["n_beams"] == n]}
                for m in METHODS if by[m]}

    reps = [p for r in rows for p in r.get("repairs", [])]
    repair_stats = {"n_repair_steps": len(reps)}
    if reps:
        repair_stats.update({
            "mean_magnitude_m": round(float(np.mean([p["repair_magnitude_m"] for p in reps])), 4),
            "max_magnitude_m": round(float(np.max([p["repair_magnitude_m"] for p in reps])), 4),
            "mean_onset_shift_m": round(float(np.mean([p["onset_shift_m"] for p in reps])), 4),
            "mean_duration_change_m": round(float(np.mean([p["duration_change_m"] for p in reps])), 4),
            "slope_floor_bound_steps": int(sum(1 for p in reps if p["slope_floor_bound"] > 0))})

    payload = {"generated_at": time.time(), "elapsed_s": round(time.time() - t0, 1),
               "target_m": MARGIN_M, "preference": a.preference, "width_m": a.width,
               "n_beams": a.n_beams, "heights": a.heights, "gaps": a.gaps, "seeds": a.seeds,
               "tcn_dir": str(a.tcn_dir) if a.tcn_dir else "outputs/duck_model_v3_m018",
               "cost_accounting": {
                   "primary": "necessary_adapted_generations",
                   "necessary_adapted_generations":
                       "one candidate-producing ARDY generation per attempted clip",
                   "ardy_calls": "one candidate-producing ARDY invocation per attempt",
                   "legacy_ardy_calls":
                       "historical v1 implementation: candidate plus unused path reference",
                   "executed_fields": "cache-dependent diagnostics, not method complexity"},
               "stand_top_m": STAND_TOP, "summary": summary, "by_beam_count": by_count,
               "repair_stats": repair_stats, "n_skipped": len(skipped), "skipped": skipped,
               "rows": rows}
    (a.out / "experiment.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{len(rows)} runs, {len(skipped)} skipped, {time.time()-t0:.0f}s\n")
    print(f"{'method':10s} {'n':>4s} {'coll-free':>10s} {'meets .18':>10s} {'goal':>6s} "
          f"{'calls':>6s} {'peak dip':>9s} {'excess':>8s} {'∫duck':>8s} {'reject':>7s}")
    for m in METHODS:
        if m not in summary:
            continue
        s = summary[m]
        print(f"{m:10s} {s['n']:4d} {s['collision_free_rate']:10.3f} "
              f"{s['margin_satisfaction_rate']:10.3f} {s['goal_reached_rate']:6.2f} "
              f"{s['mean_ardy_calls']:6.1f} {s['mean_peak_dip_m']*100:8.1f}cm "
              f"{s['mean_excess_crouch_m']*100:7.1f}cm {s['mean_duck_integral_m2']:8.3f} "
              f"{s['rejected_rate']:7.3f}")
    print(f"\nrepair steps: {repair_stats}")
    print(f"-> {a.out / 'experiment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
