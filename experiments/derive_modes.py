"""Derive the planner's body-mode table from measured envelopes.

The planner may only claim envelopes the prior actually reaches. Aggregation is the WORST
case over seeds, not the mean: a planner that assumes the average clearance will route the
robot through gaps it clears only half the time, and the resulting collisions would be
attributed to the method rather than to the calibration.

Reads outputs/exp001 (duck), exp001b (tuck/sidle) and exp001c (step-over) and writes
outputs/body_modes.json, which planner.py loads.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def worst(rows: list[dict], key: str, agg=max) -> float:
    v = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return float(agg(v)) if v else float("nan")


def main() -> None:
    duck = load(ROOT / "outputs/exp001/rows.jsonl")
    lat = load(ROOT / "outputs/exp001b/rows.jsonl")
    step = load(ROOT / "outputs/exp001c/rows.jsonl")

    by_dip = defaultdict(list)
    for r in duck:
        if r["channel"] == "duck":
            by_dip[r["value"]].append(r)

    modes = []
    # Standing baseline, worst case over every unadapted clip we generated.
    stand_rows = by_dip.get(0.0, [])
    stand_top = worst(stand_rows, "top_adapted_m")
    stand_hw = worst(stand_rows, "halfwidth_adapted_m")
    modes.append({"name": "stand", "half_width": round(stand_hw, 3), "top": round(stand_top, 3),
                  "pelvis_y": 0.78, "cost": 1.0, "tuck": 0.0, "sidle_deg": 0.0,
                  "max_step": 0.0, "source": "exp001 duck=0.0, worst of 3 seeds"})

    # Duck rungs: pick dips that give distinct, reliably-reached ceilings.
    for dip, name, cost in [(0.15, "duck_light", 1.20), (0.25, "duck", 1.45),
                            (0.35, "duck_deep", 1.80), (0.50, "duck_max", 2.40)]:
        rows = by_dip.get(dip, [])
        if not rows:
            continue
        modes.append({"name": name,
                      "half_width": round(worst(rows, "halfwidth_adapted_m"), 3),
                      "top": round(worst(rows, "top_adapted_m"), 3),
                      "pelvis_y": round(0.78 - dip, 3), "cost": cost, "tuck": 0.0,
                      "sidle_deg": 0.0, "max_step": 0.0,
                      "source": f"exp001 duck={dip}, worst of 3 seeds"})

    # Lateral: the single best-measured narrowing condition, worst case over seeds.
    if lat:
        by_cond = defaultdict(list)
        for r in lat:
            by_cond[(r["sidle_deg"], r["tuck"], "side" if "sideways" in r["prompt"] else "walk")].append(r)
        # Rank candidate conditions by their WORST-case half-width, and keep the best one.
        ranked = sorted(by_cond.items(), key=lambda kv: worst(kv[1], "halfwidth_m"))
        (phi, k, prompt), rows = ranked[0]
        modes.append({"name": "narrow",
                      "half_width": round(worst(rows, "halfwidth_m"), 3),
                      "top": round(worst(rows, "top_m"), 3),
                      "pelvis_y": 0.78, "cost": 1.6, "tuck": k, "sidle_deg": float(phi),
                      "max_step": 0.0,
                      "source": f"exp001b sidle={phi} tuck={k} prompt={prompt}, worst of 3 seeds"})

    # Step-over: the tallest floor box cleared in the WORST seed of the best condition.
    if step:
        by_cond = defaultdict(list)
        for r in step:
            by_cond[(r["cond"], r["lift"])].append(r)
        ranked = sorted(by_cond.items(),
                        key=lambda kv: -worst(kv[1], "max_box_h_m", agg=min))
        (cond, lift), rows = ranked[0]
        h = worst(rows, "max_box_h_m", agg=min)
        base = by_cond.get(("control", 0.0), [])
        modes.append({"name": "step_over",
                      "half_width": round(stand_hw, 3), "top": round(stand_top, 3),
                      "pelvis_y": 0.78, "cost": 1.7, "tuck": 0.0, "sidle_deg": 0.0,
                      "lift": lift, "max_step": round(h, 3),
                      "control_max_step": round(worst(base, "max_box_h_m", agg=min), 3)
                      if base else None,
                      "source": f"exp001c cond={cond} lift={lift}, worst seed"})

    out = {"modes": modes,
           "note": "half_width and top are WORST-CASE over seeds; a planner must not "
                   "assume better than the prior reliably delivers."}
    (ROOT / "outputs/body_modes.json").write_text(json.dumps(out, indent=2))
    w = max(len(m["name"]) for m in modes)
    print(f"{'mode':<{w}} {'half_width':>10} {'top':>7} {'pelvis_y':>9} {'max_step':>9}  source")
    for m in modes:
        print(f"{m['name']:<{w}} {m['half_width']:10.3f} {m['top']:7.3f} "
              f"{m['pelvis_y']:9.3f} {m['max_step']:9.3f}  {m['source']}")


if __name__ == "__main__":
    sys.exit(main())
