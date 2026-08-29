"""Heuristic vs learned body layer, end to end through the frozen prior.

Same scene, same route, same seed, same frame count, same prompt -- the ONLY difference is
which schedule filled the duck channel. Deliberately small: LUCID owns the GPU and this needs
a couple of dozen clips, not a batch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..demo.strategy_planner import SHORTEST_MODES
from ..planner import plan
from ..robot import G1Body
from .dataset import build_scene, duck_label
from .predictor import predict_dip, spec_from_dip
from .probes import smoothness

PROMPT = "A person walks forward."
SPEED, STEPS, SEED = 0.9, 5, 100


def scenes_for(split: str) -> list:
    """Unseen geometry: the test split's held-out beam heights and positions."""
    if split == "test":
        return [build_scene(h, x, w) for h in (0.95, 1.05) for x in (4.5,)
                for w in (1.45, 1.75, 2.00)]
    return [build_scene(h, x, w) for h in (0.90, 1.00, 1.10) for x in (4.0,)
            for w in (1.45, 1.75)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default="outputs/duck_model")
    args = ap.parse_args()
    from ..runner import ArdyRunner

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    rows, t0 = [], time.time()

    for sc in scenes_for(args.split):
        p = plan(sc, "adaptive", modes_override=SHORTEST_MODES)
        if not p.feasible:
            continue
        dips = {"heuristic": duck_label(p), "learned": predict_dip(sc, p.xy)}
        body = G1Body(sc)
        for name, dip in dips.items():
            spec = spec_from_dip(sc, p.xy, dip, fps, SPEED)
            out = runner.generate([PROMPT], [spec], spec.T, STEPS, seeds=[SEED])[0]
            q = runner.to_qpos(out)
            rep = body.trajectory_report(q)
            rows.append({
                "scene": sc.scene_id, "beam_h": sc.meta["beam_h"],
                "beam_w": sc.meta["beam_w"], "planner": name,
                "collision_free": bool(rep["collision_free"]),
                "min_clearance_m": float(rep["min_clearance_m"]),
                "goal_error_m": float(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal))),
                "goal_reached": bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < 0.5),
                "schedule_smoothness": smoothness(dip),
                "peak_dip_m": float(dip.max()),
            })
        print(f"  {sc.meta['beam_h']:.2f}m/{sc.meta['beam_w']:.2f}m  "
              + "  ".join(f"{r['planner']}: {'clear' if r['collision_free'] else 'COLLIDE'}"
                          f" goal={r['goal_error_m']:.2f}"
                          for r in rows[-2:]), flush=True)

    print(f"\n{len(rows)} clips in {time.time()-t0:.1f}s ({args.split} geometry)")
    print(f"{'planner':10s} {'collision-free':>15s} {'goal reached':>13s} "
          f"{'min clr (m)':>12s} {'smoothness':>11s} {'peak dip':>9s}")
    print("-" * 76)
    summ = {}
    for name in ("heuristic", "learned"):
        R = [r for r in rows if r["planner"] == name]
        if not R:
            continue
        summ[name] = {
            "n": len(R),
            "collision_free_rate": float(np.mean([r["collision_free"] for r in R])),
            "goal_reached_rate": float(np.mean([r["goal_reached"] for r in R])),
            "mean_min_clearance_m": float(np.mean([r["min_clearance_m"] for r in R])),
            "mean_smoothness": float(np.mean([r["schedule_smoothness"] for r in R])),
            "mean_peak_dip_m": float(np.mean([r["peak_dip_m"] for r in R])),
        }
        s = summ[name]
        print(f"{name:10s} {s['collision_free_rate']:15.3f} {s['goal_reached_rate']:13.3f} "
              f"{s['mean_min_clearance_m']:12.3f} {s['mean_smoothness']:11.5f} "
              f"{s['mean_peak_dip_m']:9.3f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{args.split}.json").write_text(
        json.dumps({"split": args.split, "rows": rows, "summary": summ,
                    "wall_clock_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\nwrote {out / ('compare_' + args.split + '.json')}")


if __name__ == "__main__":
    main()
