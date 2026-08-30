"""Phase 4C: do the measured rate bounds and a properly dimensioned objective change anything?

Three questions, each with a number rather than an argument:

  1. Does the measured command-rate bound BIND at the resolution the scheduler plans at?
     r_down and r_up are ~1.3 command units per second. At 64 samples over a 14 m route at
     0.9 m/s each sample spans ~0.25 s, so a single step may move ~0.32 units. If the
     smoothness terms already keep every step well inside that, the constraint is decoration.

  2. Does the dimensioned objective avoid the 1/dt^3 domination that sank Phase 3's
     `time_weighted` flag? That version forced a merged crouch at every gap and speed tested.
     The dimensioned form drops the jerk term entirely, so there is nothing left to blow up.

  3. Does the merge/split boundary now move with SPEED? Experiment B found it fixed at 4.0 m
     for 0.6, 0.9 and 1.2 m/s -- distance, not time -- which is the correct consequence of a
     per-sample objective over a distance grid. A genuine time integral should make walking
     faster cheaper to stand up between beams, and move the boundary. If it does not, the
     Experiment B negative is stronger than a discretisation artefact.

    python -m scene2motion.optim.experiment_rates
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.scene_builder import BeamParams, beam_footprints, build
from ..demo.strategy_planner import evaluate
from ..learn.route_profile import profile
from .rates import load as load_rates
from .response import DIP_MAX, DuckResponse
from .scheduler import MARGIN_M, dt_for, solve

OUT = Path("outputs/phase4c_rates")
N = 64
VARIANTS = ("default", "dimensioned", "dimensioned+rates", "default+rates")


def _solve(clear, resp, dt, variant, rb, **extra):
    kw = dict(extra)
    if "dimensioned" in variant:
        kw["dimensioned"] = True
    if "rates" in variant:
        kw["rate_bounds"] = rb
    return solve(clear, resp, dt, **kw)


def _between_beams(scene, xy, n=N):
    """Index window strictly BETWEEN the two beams, on the schedule's route-distance grid.

    The middle third of the route is not the gap: the route runs several metres past the last
    beam to the goal, so a fixed fraction lands in the run-out and reports every schedule as
    split. Merge has to be read where the robot would actually stand back up.
    """
    seg = np.linalg.norm(np.diff(np.asarray(xy, float), axis=0), axis=1)
    total = float(seg.sum()) or 1.0
    s = np.linspace(0.0, total, n)
    f = beam_footprints(scene)
    if len(f) < 2:
        return slice(0, 0)
    lo, hi = float(f[0]["x_hi"]), float(f[1]["x_lo"])
    i0, i1 = int(np.searchsorted(s, lo)), int(np.searchsorted(s, hi))
    return slice(min(i0 + 1, n - 1), max(i1, i0 + 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.6, 0.9, 1.2])
    ap.add_argument("--gaps", type=float, nargs="+",
                    default=[2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    ap.add_argument("--height", type=float, default=1.00)
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.1, 0.2, 0.375, 0.75, 1.5, 6.0])
    # beta*(alpha*dt^2)^-1 equals the per-sample form's beta/alpha at dt = 0.25 s, i.e. at the
    # 0.9 m/s reference speed. Calibrating there means any speed dependence that shows up is a
    # property of the objective's form, not of a weight that happens to differ.
    ap.add_argument("--calibrated-beta", type=float, default=0.375)
    ap.add_argument("--fine-gaps", type=float, nargs="+",
                    default=[round(1.2 + 0.2 * i, 1) for i in range(23)])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    rates = load_rates()
    if rates is None:
        raise SystemExit("run `python -m scene2motion.optim.rates` first")
    rb = (rates["r_down"], rates["r_up"])
    resp = DuckResponse.load(Path("outputs/duck_response/response.json"))
    a.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    for gap in a.gaps:
        sc = build(BeamParams(a.height, 2.25, 2, gap).clamped())
        st = evaluate(sc, "shortest")
        if not st.feasible:
            continue
        for speed in a.speeds:
            prof = profile(sc, st.plan.xy, speed=speed)
            route_len = float(np.linalg.norm(np.diff(st.plan.xy, axis=0), axis=1).sum())
            dt = dt_for(route_len, N, speed)
            win = _between_beams(sc, st.plan.xy)
            for v in VARIANTS:
                s = _solve(prof[:, 0], resp, dt, v, rb)
                if not s.feasible:
                    rows.append({"gap": gap, "speed": speed, "variant": v, "feasible": False})
                    continue
                d = np.diff(s.q)
                cap_down, cap_up = rb[0] * dt, rb[1] * dt
                # "Merged" = the command never returns near zero between the two beams.
                mid = s.q[win]
                if not len(mid):
                    mid = s.q[len(s.q) // 3: 2 * len(s.q) // 3]
                rows.append({
                    "gap": gap, "speed": speed, "variant": v, "feasible": True,
                    "dt_s": round(dt, 4), "peak_dip_m": round(float(s.q.max() * DIP_MAX), 4),
                    "merged": bool(mid.min() > 0.15),
                    "min_between_m": round(float(mid.min() * DIP_MAX), 4),
                    "max_step": round(float(d.max()), 5), "min_step": round(float(d.min()), 5),
                    "cap_down": round(cap_down, 5), "cap_up": round(cap_up, 5),
                    "headroom_down": round(float(cap_down - d.max()), 5),
                    "headroom_up": round(float(cap_up + d.min()), 5),
                    "n_binding": int(((d > cap_down - 1e-6) | (d < -cap_up + 1e-6)).sum()),
                    "objective": round(float(s.objective), 5)})

    ok = [r for r in rows if r.get("feasible")]
    summary = {}
    for v in VARIANTS:
        R = [r for r in ok if r["variant"] == v]
        if not R:
            continue
        merged = [r for r in R if r["merged"]]
        summary[v] = {
            "n": len(R), "merged_rate": round(len(merged) / len(R), 4),
            "mean_peak_dip_m": round(float(np.mean([r["peak_dip_m"] for r in R])), 4),
            "worst_max_step": round(float(np.max([r["max_step"] for r in R])), 5),
            "min_headroom_down": round(float(np.min([r["headroom_down"] for r in R])), 5),
            "min_headroom_up": round(float(np.min([r["headroom_up"] for r in R])), 5),
            "n_binding_total": int(sum(r["n_binding"] for r in R)),
            # The Experiment B question: does the merge boundary depend on speed?
            "merge_gap_by_speed": {
                str(sp): (min([r["gap"] for r in R if r["speed"] == sp and r["merged"]],
                              default=None))
                for sp in a.speeds},
        }

    print(f"r_down {rb[0]} / r_up {rb[1]} command units per second "
          f"(from {rates['n_clips']} clips)\n")
    print(f"{'variant':20s} {'n':>3s} {'merged':>7s} {'peak dip':>9s} {'worst step':>11s} "
          f"{'slack down':>11s} {'slack up':>9s} {'binding':>8s}")
    for v, s in summary.items():
        print(f"{v:20s} {s['n']:3d} {s['merged_rate']:7.3f} {s['mean_peak_dip_m']*100:8.1f}cm "
              f"{s['worst_max_step']:11.4f} {s['min_headroom_down']:11.4f} "
              f"{s['min_headroom_up']:9.4f} {s['n_binding_total']:8d}")
    # -- the merge/split boundary, per variant and per speed ---------------------------
    # Reported as the LARGEST gap that still merges: that is where the boundary sits, and
    # whether it moves with speed is the whole question.
    def boundary(rows_, want_variant=None, want_beta=None):
        out = {}
        for sp in a.speeds:
            R = [r for r in rows_ if r.get("feasible") and r["speed"] == sp
                 and (want_variant is None or r["variant"] == want_variant)
                 and (want_beta is None or r.get("w_d1") == want_beta)]
            m = [r["gap"] for r in R if r["merged"]]
            out[str(sp)] = max(m) if m else None
        return out

    print("\nlargest gap that still merges (the boundary), by speed:")
    for v in summary:
        print(f"  {v:20s} {boundary(rows, v)}")

    # -- beta recalibration -------------------------------------------------------------
    # The dimensioned objective merges at every gap tested because the Phase 3 weights do not
    # transfer: its rate/effort ratio is beta/(alpha*dt^2), which at dt ~ 0.25 s is 16x the
    # per-sample form's beta/alpha. Sweeping beta finds the value that reproduces the observed
    # boundary; whether the boundary then MOVES with speed is what a genuine time integral
    # predicts and the per-sample objective cannot produce.
    beta_rows = []
    for beta in a.betas:
        for gap in a.gaps:
            sc = build(BeamParams(a.height, 2.25, 2, gap).clamped())
            st = evaluate(sc, "shortest")
            if not st.feasible:
                continue
            win = _between_beams(sc, st.plan.xy)
            route_len = float(np.linalg.norm(np.diff(st.plan.xy, axis=0), axis=1).sum())
            for speed in a.speeds:
                prof = profile(sc, st.plan.xy, speed=speed)
                dt = dt_for(route_len, N, speed)
                sol = solve(prof[:, 0], resp, dt, dimensioned=True, w_d1=beta)
                if not sol.feasible:
                    continue
                mid = sol.q[win]
                beta_rows.append({"w_d1": beta, "gap": gap, "speed": speed, "feasible": True,
                                  "variant": "dimensioned", "merged": bool(len(mid) and mid.min() > 0.15),
                                  "peak_dip_m": round(float(sol.q.max() * DIP_MAX), 4)})
    print("\nbeta recalibration for the dimensioned objective "
          "(largest merging gap, by speed):")
    beta_boundary = {}
    for beta in a.betas:
        b = boundary(beta_rows, want_beta=beta)
        beta_boundary[str(beta)] = b
        moves = len({v for v in b.values() if v is not None}) > 1
        print(f"  beta={beta:<7g} {b}   {'boundary MOVES with speed' if moves else 'speed-independent'}")
    # -- the boundary on a fine gap grid, in BOTH units --------------------------------
    # The coarse sweep above resolves the boundary only to the nearest whole metre, which is
    # not enough to say whether it is fixed in distance or in time. This locates it to 0.2 m
    # and reports its coefficient of variation across speeds in each unit. A boundary that is
    # a property of the objective's discretisation is constant in distance; one that reflects
    # a real time cost of standing up is constant in time.
    fine = {}
    for label, beta in (("dimensioned_beta_calibrated", a.calibrated_beta),
                        ("per_sample_default", None)):
        per_speed = {}
        for sp in a.speeds:
            merged = []
            for gap in a.fine_gaps:
                sc = build(BeamParams(a.height, 2.25, 2, gap).clamped())
                st = evaluate(sc, "shortest")
                if not st.feasible:
                    continue
                win = _between_beams(sc, st.plan.xy)
                route_len = float(np.linalg.norm(np.diff(st.plan.xy, axis=0), axis=1).sum())
                sol = (solve(profile(sc, st.plan.xy, speed=sp)[:, 0], resp,
                             dt_for(route_len, N, sp), dimensioned=True, w_d1=beta)
                       if beta is not None else
                       solve(profile(sc, st.plan.xy, speed=sp)[:, 0], resp,
                             dt_for(route_len, N, sp)))
                if sol.feasible and len(sol.q[win]) and sol.q[win].min() > 0.15:
                    merged.append(gap)
            per_speed[str(sp)] = max(merged) if merged else None
        d = [v for v in per_speed.values() if v is not None]
        t = [v / sp for v, sp in zip(per_speed.values(), a.speeds) if v is not None]
        cv = lambda x: round(float(np.std(x) / np.mean(x)), 4) if len(x) > 1 else None
        fine[label] = {"boundary_m_by_speed": per_speed,
                       "boundary_s_by_speed": {k: (None if v is None else round(v / float(k), 3))
                                               for k, v in per_speed.items()},
                       "cv_in_distance": cv(d), "cv_in_time": cv(t),
                       "fixed_in": ("distance" if (cv(d) or 1) < (cv(t) or 1) else "time")}
    print("\nfine-grid boundary (0.2 m resolution), in both units:")
    for k, v in fine.items():
        print(f"  {k:28s} {v['boundary_m_by_speed']} m -> {v['boundary_s_by_speed']} s")
        print(f"  {'':28s} CV {v['cv_in_distance']:.3f} in distance vs {v['cv_in_time']:.3f} "
              f"in time -> fixed in {v['fixed_in']}")

    payload = {"generated_at": time.time(), "fine_boundary": fine, "elapsed_s": round(time.time() - t0, 1),
               "rates": rates, "target_m": MARGIN_M, "n_samples": N,
               "summary": summary, "rows": rows,
               "beta_rows": beta_rows, "beta_boundary": beta_boundary,
               "boundary_by_variant": {v: boundary(rows, v) for v in summary}}
    (a.out / "rates_experiment.json").write_text(json.dumps(payload, indent=2))
    print(f"-> {a.out / 'rates_experiment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
