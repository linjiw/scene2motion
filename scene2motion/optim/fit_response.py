"""Fit g(q) and tau, then check the surrogate predicts the things the optimiser relies on.

tau is fitted from clips already on disk. Every cached demo clip has a known commanded duck
schedule (rebuildable from its scene and preference) and a measured top-height trajectory, so
the lag can be estimated without generating anything new -- which matters while LUCID owns the
GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.ardy_runner import PROMPT, SPEED, n_frames
from ..demo.cache import ClipCache
from ..demo.scene_builder import BeamParams, build
from ..demo.strategy_planner import evaluate
from ..planner import LEAD_S, MODE_BY_NAME, plan_to_spec
from ..robot import G1Body
from .response import DIP_MAX, DuckResponse, fit_static

NOMINAL_PELVIS = MODE_BY_NAME["stand"].pelvis_y
TAU_GRID = np.arange(0.05, 1.21, 0.01)


def _commanded_q(scene, strat, fps: float, T: int, nominal) -> np.ndarray:
    spec = plan_to_spec(strat.plan, fps, nominal, JOINTS, speed=SPEED,
                        duration=T / fps, lead_s=LEAD_S)
    return np.clip((NOMINAL_PELVIS - np.asarray(spec.root_y, float)) / DIP_MAX, 0.0, 1.0)


JOINTS: list[str] = []          # filled at run time from the cached clip metadata


def collect(cache: ClipCache, resp: DuckResponse) -> list[dict]:
    """Rebuild (commanded q, measured top height) for every cached heuristic duck clip."""
    global JOINTS
    from ..runner import ArdyRunner
    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    JOINTS = runner.joint_names
    fps = runner.fps
    out = []
    for meta_path in sorted(cache.root.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("body_layer", "heuristic") != "heuristic":
            continue
        sid = meta["scene_id"]
        if not sid.startswith("demo_partial_beam_h"):
            continue
        h = float(sid.split("_h")[1].split("_w")[0])
        w = float(sid.split("_w")[1])
        scene = build(BeamParams(h, w))
        strat = evaluate(scene, meta["preference"])
        if not strat.feasible:
            continue
        hit = cache.get(meta["key"])
        if hit is None:
            continue
        qpos, _ = hit
        T = len(qpos)
        ref = runner.generate([PROMPT], [None], T, 5, seeds=[meta["seed"]])[0]
        q = _commanded_q(scene, strat, fps, T, ref)
        if q.max() < 0.05:                      # no duck: carries no information about tau
            continue
        body = G1Body(scene)
        top = np.array([body.top_height(qpos[t]) for t in range(T)])
        out.append({"key": meta["key"], "beam_h": h, "q": q, "top": top,
                    "dt": 1.0 / fps, "preference": meta["preference"]})
    return out


def fit_tau(traces: list[dict], resp: DuckResponse) -> dict:
    """Grid-search tau; report the whole curve, not just the argmin."""
    curve = []
    for tau in TAU_GRID:
        errs = [np.abs(resp.top_from_command(t["q"], t["dt"], tau) - t["top"]).mean()
                for t in traces]
        curve.append(float(np.mean(errs)))
    curve = np.array(curve)
    best = int(np.argmin(curve))
    # Sensitivity bracket: every tau whose error is within 5 % of the minimum is not
    # distinguishable from it by this data, and quoting a single value would overstate it.
    within = np.where(curve <= curve[best] * 1.05)[0]
    return {"tau_s": float(TAU_GRID[best]),
            "tau_lo": float(TAU_GRID[within[0]]), "tau_hi": float(TAU_GRID[within[-1]]),
            "mae_at_best_m": float(curve[best]),
            "mae_at_zero_lag_m": float(curve[0]),
            "n_traces": len(traces),
            "curve": {float(round(t, 3)): float(round(c, 5))
                      for t, c in zip(TAU_GRID[::10], curve[::10])}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/duck_response")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    qk, hk, info = fit_static()
    resp = DuckResponse(qk, hk, fit=info)
    print(f"static g(q): {len(qk)} knots, holdout MAE {info['holdout_mae_m']*1000:.1f} mm "
          f"(per-seed sd {np.mean(list(info['per_level_sd_m'].values()))*1000:.1f} mm)")

    traces = collect(ClipCache("scene2motion/demo_outputs/clips"), resp)
    print(f"{len(traces)} cached duck traces for the lag fit")
    tau = fit_tau(traces, resp)
    resp.tau_s, resp.tau_lo, resp.tau_hi = tau["tau_s"], tau["tau_lo"], tau["tau_hi"]
    resp.fit["tau"] = tau
    print(f"tau = {tau['tau_s']:.2f} s  (within 5 % of best: "
          f"{tau['tau_lo']:.2f}-{tau['tau_hi']:.2f} s)")
    print(f"  surrogate MAE {tau['mae_at_best_m']*1000:.1f} mm vs "
          f"{tau['mae_at_zero_lag_m']*1000:.1f} mm with no lag "
          f"({100*(1-tau['mae_at_best_m']/tau['mae_at_zero_lag_m']):.0f} % better)")
    resp.save(out / "response.json")
    np.savez_compressed(out / "traces.npz",
                        **{f"q_{i}": t["q"] for i, t in enumerate(traces)},
                        **{f"top_{i}": t["top"] for i, t in enumerate(traces)})
    print(f"wrote {out}/response.json in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
