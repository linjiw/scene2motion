"""Rebuild the planner's mode table from the CALIBRATED envelope (EXP-001d).

Why the old table has to go
---------------------------
`outputs/body_modes.json` was aggregated as the worst of three seeds per mode. Compared
against the n=20 conformal envelope, it is not uniformly conservative — it is wrong in *both*
directions, and the dangerous direction dominates the deep ducks:

    mode         body_modes top / half-width      EXP-001d 90 % bound      error
    duck_light        1.216 / 0.360                 1.220 / 0.402        hw  -0.042  OPTIMISTIC
    duck              1.127 / 0.375                 1.120 / 0.420        hw  -0.045  OPTIMISTIC
    duck_deep         1.017 / 0.497                 1.034 / 0.420        hw  +0.077  conservative
    duck_max          0.822 / 0.344                 0.879 / 0.510        top -0.057  OPTIMISTIC
                                                                         hw  -0.166  OPTIMISTIC

So the planner has been certifying `duck_max` against a 0.344 m half-width when the 90 %
bound is 0.510 m, and a 0.822 m top when the bound is 0.879 m. It has not produced collisions
— EXP-002 is 100 % collision-free on feasible plans — but that is the *mean* behaviour, and a
certificate is supposed to bound the tail. A guarantee that holds on average is not a
guarantee.

Rebuilding from the envelope will almost certainly REDUCE feasibility, because the honest
deep-duck envelope is taller and much wider than the one currently claimed. That is the
correct trade and it should be reported as such: the previous 68.8 % rested in part on an
optimistic certificate, and the number after this is the one that comes with a coverage
statement.

`narrow` and `step_over` keep their own calibrations (EXP-001b, EXP-001c) since the dip sweep
does not cover the tuck and lift axes jointly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.envelope import get_envelope  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOMINAL_PELVIS = 0.78

# (name, pelvis height, traversal cost). The dip ladder is unchanged so the comparison against
# the previous table is like-for-like; only the certified envelope moves.
DUCKS = [("stand", 0.78, 1.00), ("duck_light", 0.63, 1.20), ("duck", 0.53, 1.45),
         ("duck_deep", 0.43, 1.80), ("duck_max", 0.28, 2.40)]


def main() -> None:
    env = get_envelope()
    old = json.loads((ROOT / "outputs/body_modes.json").read_text())["modes"]
    old_by = {m["name"]: m for m in old}

    modes = []
    for name, pelvis, cost in DUCKS:
        dip = NOMINAL_PELVIS - pelvis
        modes.append({
            "name": name,
            "half_width": round(float(env.half_width(dip)), 3),
            "top": round(float(env.top(dip)), 3),
            "pelvis_y": pelvis, "cost": cost, "tuck": 0.0, "sidle_deg": 0.0,
            "max_step": 0.0,
            "source": f"exp001d dip={dip:.2f}, 90% split-conformal bound, n=20",
        })
    # narrow: the tuck axis, calibrated at nominal pelvis height.
    modes.append({
        "name": "narrow",
        "half_width": round(float(env.half_width(0.0, 0.85)), 3),
        "top": round(float(env.top(0.0)), 3),
        "pelvis_y": 0.78, "cost": 1.6, "tuck": 0.85, "sidle_deg": 0.0, "max_step": 0.0,
        "source": "exp001d tuck=0.85, 90% split-conformal bound, n=20",
    })
    # step_over keeps its EXP-001c calibration until EXP-001e re-measures it at n=20.
    so = dict(old_by["step_over"])
    so["source"] += " (NOT yet conformally calibrated; see exp001e)"
    modes.append(so)

    out = {"modes": modes,
           "note": ("half_width and top are 90% split-conformal upper bounds at n=20 "
                    "(EXP-001d), replacing worst-of-3-seeds. The bound is on the ENVELOPE, "
                    "not on collision-freeness; the per-instance guarantee is the MuJoCo "
                    "check."),
           "previous": old}
    (ROOT / "outputs/body_modes.json").write_text(json.dumps(out, indent=2))

    w = max(len(m["name"]) for m in modes)
    print(f"{'mode':<{w}} {'half_width':>18} {'top':>18}")
    for m in modes:
        o = old_by.get(m["name"], {})
        dh = m["half_width"] - o.get("half_width", m["half_width"])
        dt = m["top"] - o.get("top", m["top"])
        print(f"{m['name']:<{w}} {o.get('half_width', float('nan')):8.3f} ->"
              f"{m['half_width']:7.3f} ({dh:+.3f}) {o.get('top', float('nan')):8.3f} ->"
              f"{m['top']:7.3f} ({dt:+.3f})")


if __name__ == "__main__":
    main()
