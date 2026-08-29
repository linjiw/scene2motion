"""Bounded step calibration: the one thing cached clips cannot tell us.

Fitting tau from the cached demo clips failed and failed informatively -- the bracket came out
0.05-0.44 s and the lag improved the fit by 1 %. That is not a weak signal, it is an absent
one: `plan_to_spec` dilates every adaptation by LEAD_S and smooths it over 0.6 s, so the
commanded schedules contain no transition sharp enough for a first-order lag to be visible.
A response time cannot be recovered from a signal that was pre-filtered by the same time scale.

So this commands a genuine STEP -- q jumps at a fixed time, holds, and drops back -- writing
root_y directly and bypassing the renderer's smoothing entirely. Rise and fall are fitted
SEPARATELY because the optimiser's merge/split behaviour depends on whether recovery is slower
than descent, and assuming symmetry would decide that question by fiat.

18 clips. Small enough to run beside LUCID.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..constraints import ConstraintSpec
from ..planner import MODE_BY_NAME
from ..robot import G1Body
from ..scenes import BUILDERS
from .response import DIP_MAX, DuckResponse, fit_static

PROMPT = "A person walks forward."
SPEED, STEPS, DUR = 0.9, 5, 8.0
T_UP, T_DOWN = 3.0, 5.5          # s, when the commanded step rises and falls
AMPS = (0.4, 0.7, 1.0)
# Static sweep amplitudes, held for the whole clip. EXP-001d also swept dip, but it rendered
# each one through `plan_to_spec` -- dilated by LEAD_S and smoothed over 0.6 s -- and its
# reported top height disagrees with a directly commanded hold by up to 147 mm at the same q.
# The optimiser commands root_y directly, so g has to be fitted on that pathway or every
# clearance constraint it writes is wrong by more than the margin it is trying to protect.
STATIC_QS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00)
NOMINAL_PELVIS = MODE_BY_NAME["stand"].pelvis_y
TAU_GRID = np.arange(0.02, 1.01, 0.01)


def step_spec(T: int, fps: float, amp: float) -> tuple[ConstraintSpec, np.ndarray]:
    t = np.arange(T) / fps
    q = np.where((t >= T_UP) & (t < T_DOWN), amp, 0.0)
    z = np.linspace(0.0, SPEED * DUR, T)
    return ConstraintSpec(root_xz=np.stack([np.zeros(T), z], -1),
                          heading=np.zeros(T),
                          root_y=NOMINAL_PELVIS - q * DIP_MAX), q


def hold_spec(T: int, fps: float, q: float) -> ConstraintSpec:
    """A constant commanded duck for the whole clip -- the static-gain pathway."""
    z = np.linspace(0.0, SPEED * DUR, T)
    return ConstraintSpec(root_xz=np.stack([np.zeros(T), z], -1), heading=np.zeros(T),
                          root_y=np.full(T, NOMINAL_PELVIS - q * DIP_MAX))


def fit_static_direct(runner, body, seeds, fps, T) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit g(q) from directly commanded holds, with a held-out seed split."""
    from .response import _isotonic_decreasing
    win = slice(int(0.45 * T), int(0.75 * T))       # settled, away from start-up and the end
    qs, hs, per = [], [], {}
    for q in STATIC_QS:
        outs = runner.generate([PROMPT] * len(seeds), [hold_spec(T, fps, q)] * len(seeds),
                               T, STEPS, seeds=seeds)
        tops = []
        for o in outs:
            qp = runner.to_qpos(o)
            tops.append(float(np.mean([body.top_height(qp[t])
                                       for t in range(win.start, win.stop)])))
        qs += [q] * len(tops)
        hs += tops
        per[float(q)] = {"mean_m": float(np.mean(tops)), "sd_m": float(np.std(tops))}
        print(f"  hold q={q:.2f}  top {np.mean(tops):.3f} m  (sd {np.std(tops):.3f})",
              flush=True)
    qs, hs = np.array(qs), np.array(hs)
    tr = np.arange(len(qs)) % len(seeds) != 0        # hold out one seed per level
    ho = ~tr
    levels = np.unique(qs[tr])
    means = np.array([hs[tr][qs[tr] == v].mean() for v in levels])
    qk, hk = _isotonic_decreasing(levels.copy(), means.copy())
    resp = DuckResponse(qk, hk)
    res_ho = resp.g(qs[ho]) - hs[ho]
    info = {"source": "step_calib.fit_static_direct", "pathway": "direct root_y hold",
            "n": int(len(qs)), "n_holdout": int(ho.sum()),
            "holdout_mae_m": float(np.abs(res_ho).mean()),
            "holdout_rmse_m": float(np.sqrt((res_ho ** 2).mean())),
            "per_level": per}
    return qk, hk, info


def fit_segment(traces, resp, lo_s, hi_s, fps) -> dict:
    """Fit tau on one segment (rise or fall) across all traces."""
    lo, hi = int(lo_s * fps), int(hi_s * fps)
    curve = []
    for tau in TAU_GRID:
        errs = []
        for tr in traces:
            pred = resp.top_from_command(tr["q"], 1.0 / fps, tau)
            errs.append(np.abs(pred[lo:hi] - tr["top"][lo:hi]).mean())
        curve.append(float(np.mean(errs)))
    curve = np.array(curve)
    b = int(np.argmin(curve))
    within = np.where(curve <= curve[b] * 1.05)[0]
    return {"tau_s": float(TAU_GRID[b]), "tau_lo": float(TAU_GRID[within[0]]),
            "tau_hi": float(TAU_GRID[within[-1]]),
            "mae_m": float(curve[b]), "mae_no_lag_m": float(curve[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/duck_response")
    ap.add_argument("--n_seeds", type=int, default=6)
    args = ap.parse_args()
    from ..runner import ArdyRunner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DUR * fps)
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(2000, 2000 + args.n_seeds))
    print("static gain, directly commanded holds:")
    qk, hk, info = fit_static_direct(runner, body, seeds, fps, T)
    resp = DuckResponse(qk, hk, fit={"static_direct": info,
                                     "static_exp001d": fit_static()[2]})
    print(f"  g holdout MAE {info['holdout_mae_m']*1000:.1f} mm "
          f"(n={info['n_holdout']} held-out clips)\n")

    print("step transitions, for the lag:")
    traces = []
    for amp in AMPS:
        spec, q = step_spec(T, fps, amp)
        outs = runner.generate([PROMPT] * len(seeds), [spec] * len(seeds), T, STEPS,
                               seeds=seeds)
        for s, o in enumerate(outs):
            qp = runner.to_qpos(o)
            top = np.array([body.top_height(qp[t]) for t in range(T)])
            traces.append({"amp": amp, "seed": seeds[s], "q": q, "top": top})
        settled = np.mean([tr["top"][int(5.0 * fps):int(5.4 * fps)].mean()
                           for tr in traces[-len(seeds):]])
        print(f"  amp {amp:.1f}  settled top {settled:.3f} m  "
              f"(g predicts {float(resp.g(amp)):.3f} m)  ({time.time()-t0:.0f}s)", flush=True)

    rise = fit_segment(traces, resp, T_UP, T_DOWN, fps)
    fall = fit_segment(traces, resp, T_DOWN, DUR, fps)
    both = fit_segment(traces, resp, T_UP, DUR, fps)
    print(f"\ntau rise {rise['tau_s']:.2f} s [{rise['tau_lo']:.2f}, {rise['tau_hi']:.2f}]  "
          f"MAE {rise['mae_m']*1000:.0f} mm (no lag {rise['mae_no_lag_m']*1000:.0f} mm)")
    print(f"tau fall {fall['tau_s']:.2f} s [{fall['tau_lo']:.2f}, {fall['tau_hi']:.2f}]  "
          f"MAE {fall['mae_m']*1000:.0f} mm (no lag {fall['mae_no_lag_m']*1000:.0f} mm)")
    print(f"tau both {both['tau_s']:.2f} s [{both['tau_lo']:.2f}, {both['tau_hi']:.2f}]  "
          f"MAE {both['mae_m']*1000:.0f} mm")
    # Compare the BRACKETS, not the point estimates. An earlier run of this reported
    # "recovery slower than descent" from 0.71 s against 0.06 s -- but that was fitted against
    # the EXP-001d gain, which is wrong for this command pathway, and the lag had absorbed the
    # gain error. With g refitted on directly commanded holds the two collapse to 0.22 s and
    # 0.19 s with heavily overlapping brackets. Gain and lag trade off, so an asymmetry claim
    # is only safe once the gain is right.
    disjoint = rise["tau_hi"] < fall["tau_lo"]
    verdict = "SLOWER (brackets disjoint)" if disjoint else "NOT DISTINGUISHABLE (overlap)"
    print(f"recovery vs descent: {verdict} -- "
          f"rise [{rise['tau_lo']:.2f}, {rise['tau_hi']:.2f}] "
          f"fall [{fall['tau_lo']:.2f}, {fall['tau_hi']:.2f}]")

    resp.tau_s, resp.tau_lo, resp.tau_hi = both["tau_s"], both["tau_lo"], both["tau_hi"]
    resp.fit["tau_step"] = {"rise": rise, "fall": fall, "both": both,
                            "amps": list(AMPS), "n_seeds": args.n_seeds,
                            "t_up": T_UP, "t_down": T_DOWN, "source": "step_calib"}
    resp.save(out / "response.json")
    np.savez_compressed(out / "step_traces.npz",
                        q=np.stack([t["q"] for t in traces]),
                        top=np.stack([t["top"] for t in traces]),
                        amp=np.array([t["amp"] for t in traces]),
                        seed=np.array([t["seed"] for t in traces]), fps=fps)
    print(f"\n{len(traces)} clips in {time.time()-t0:.1f}s -> {out}/response.json")


if __name__ == "__main__":
    main()
