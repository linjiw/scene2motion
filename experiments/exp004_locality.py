"""EXP-004: is the adaptation LOCAL in space and ANTICIPATORY in time?

The question success rate cannot ask
------------------------------------
Two systems can both reach 83% on the beam family and be doing completely different things.
One raises the beam by 10 cm and adjusts the crouch over the half-second it spends underneath.
The other re-solves from scratch and produces an unrelated trajectory that also happens to
work. Only the first has understood *which part of its behaviour the geometry required it to
change*.

So: take matched scenes differing in exactly one clearance parameter (the families in
scene2motion/scenes.py are ladders — same family, same seed, one swept dimension) and measure
where the motion difference goes.

Measuring the right thing
-------------------------
A first version of this experiment differenced raw collision-primitive positions and produced
SNR < 1: the sample-to-sample noise floor (18 cm) exceeded the effect of a 10 cm beam-height
change (16 cm), and anticipation read 0.00 s everywhere. The cause was not that adaptation is
absent — it is that two clips of the same walk drift out of gait phase, and a swinging wrist
moves tens of centimetres between phases. Limb position is dominated by a nuisance variable.

What the geometry actually acts on is the **clearance envelope**: how tall the body is, how
wide it is across the corridor, where the pelvis sits. Those are near-invariant to gait phase
and are exactly what an obstacle constrains. So the per-frame difference is

    d(t) = |top_A(t) - top_B(t)| + |halfwidth_A(t) - halfwidth_B(t)| + |pelvis_A(t) - pelvis_B(t)|

The control that makes it mean anything
---------------------------------------
Differencing two rungs conflates "the geometry demanded a change" with "any scene edit moves
the sample around". So each adaptation-forcing change is compared against a NULL PERTURBATION:
the obstacle nudged 12 cm sideways, which changes the scene without changing what the body must
do. Both are generated at the same batch position, so the noise draw is shared.

    d_geom  adjacent rungs   — the clearance parameter changed
    d_null  obstacle nudged  — the scene changed, the requirement did not

A locality or anticipation number is only meaningful where d_geom clears d_null.

Strawman baseline
-----------------
`global` holds the deepest mode the scene requires over the WHOLE clip. It should reach the
goal collision-free just as often, and score badly on locality. If the metric does not
separate `global` from `adaptive`, the metric is broken.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.planner import plan, plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS, LADDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED = 0.9
DURATION = 10.0            # fixed across a ladder so frame t means the same thing in each rung
FAMILIES = ["overhead_beam", "partial_beam"]
INTERACTION_HALF_WIDTH = 0.7   # m either side of the obstacle, along travel
NULL_SHIFT = 0.12              # m of lateral obstacle shift for the null perturbation
LATERAL = np.array([0.0, 1.0, 0.0])


def envelope(body: G1Body, qpos: np.ndarray) -> np.ndarray:
    """(T, 3) clearance envelope: top height, lateral half-width, pelvis height.

    Phase-robust by construction: these are extrema over the whole body, so a swinging limb
    changes them far less than it changes its own position.
    """
    out = np.empty((len(qpos), 3))
    for i, q in enumerate(qpos):
        out[i] = (body.top_height(q), body.half_width(q, LATERAL), q[2])
    return out


def diff_series(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    return np.abs(a[:n] - b[:n]).sum(axis=1)


def obstacle_x(sc) -> float:
    for k in ("beam_x", "wall_x", "box_x", "pillar_x"):
        if k in sc.meta:
            return float(sc.meta[k])
    return float(0.5 * (sc.start[0] + sc.goal[0]))


def shifted(sc, dy: float):
    """The same scene with its non-wall obstacles nudged laterally: the null perturbation."""
    out = deepcopy(sc)
    for b in out.boxes:
        if not b.label.startswith("wall_"):
            b.center = (b.center[0], b.center[1] + dy, b.center[2])
    out.scene_id = sc.scene_id + f"_null{dy:+.2f}"
    return out


def strategy_plan(sc, family: str):
    """The plan whose adaptation the ladder is supposed to modulate.

    For `partial_beam` the cheapest plan walks AROUND the beam at every rung, so differencing
    free plans compares two identical motions and measures exactly nothing — the first run of
    this experiment reported d = 0.00 for that family for precisely this reason. The locality
    question there is about the ducking strategy, so it is requested explicitly.
    """
    if family == "partial_beam":
        ch, edge = sc.meta["corridor_half"], sc.meta["beam_edge_y"]
        return plan(sc, "adaptive", forbid_y=(-ch - 0.4, edge - 0.05))
    return plan(sc, "adaptive")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp004")
    ap.add_argument("--seeds_per_rung", type=int, default=3)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DURATION * fps)
    rows = []

    def render(sc, variant: str):
        """(qpos, envelope, report, min requested pelvis) for one scene under one variant."""
        p = strategy_plan(sc, sc.family)
        if not p.feasible:
            return None
        ctrl = runner.generate([PROMPT], [plan_to_path_spec(p, fps, SPEED, duration=DURATION)],
                               T, args.diffusion_steps, seed=0)[0]
        spec = plan_to_spec(p, fps, ctrl, runner.joint_names, SPEED, duration=DURATION)
        if variant == "global":
            deepest = float(np.min(spec.root_y))
            spec = ConstraintSpec(root_xz=spec.root_xz.copy(), heading=spec.heading.copy(),
                                  root_y=np.full_like(spec.root_y, deepest),
                                  first_heading=spec.first_heading)
        o = runner.generate([PROMPT], [spec], T, args.diffusion_steps, seed=0)[0]
        q = runner.to_qpos(o)
        body = G1Body(sc)
        return q, envelope(body, q), body.trajectory_report(q), float(np.min(spec.root_y))

    for fam in FAMILIES:
        for sseed in range(args.seeds_per_rung):
            ladder, nulls = {}, {}
            for v in LADDERS[fam]:
                sc = BUILDERS[fam](v, sseed)
                r = render(sc, "adaptive")
                if r is None:
                    continue
                ladder[v] = {"scene": sc, "adaptive": r, "ox": obstacle_x(sc)}
                g = render(sc, "global")
                if g is not None:
                    ladder[v]["global"] = g
                n = render(shifted(sc, NULL_SHIFT), "adaptive")
                if n is not None:
                    nulls[v] = n
            if len(ladder) < 2:
                continue

            vals = sorted(ladder)
            for lo, hi in zip(vals[:-1], vals[1:]):
                A, B = ladder[lo], ladder[hi]
                ox = A["ox"]
                # Null floor: the same rung with the obstacle nudged sideways. Shared by both
                # arms so their locality and anticipation numbers stay comparable.
                d_null = (diff_series(A["adaptive"][1], nulls[lo][1])
                          if lo in nulls else np.zeros(T))
                floor = float(np.percentile(d_null, 90))

                for variant in ("adaptive", "global"):
                    if variant not in A or variant not in B:
                        continue
                    qa, ea, repa, ry_a = A[variant]
                    qb, eb, repb, ry_b = B[variant]
                    d = diff_series(ea, eb)
                    n = len(d)
                    inside = np.abs(qa[:n, 0] - ox) <= INTERACTION_HALF_WIDTH
                    if inside.sum() < 3 or (~inside).sum() < 3:
                        continue
                    locality = float(d[inside].mean() / max(d[~inside].mean(), 1e-6))
                    cross = int(np.argmin(np.abs(qa[:n, 0] - ox)))
                    thresh = max(floor * 1.5, 0.01)      # 1 cm of envelope change, minimum
                    onset = cross
                    for t in range(cross, -1, -1):
                        if d[t] > thresh:
                            onset = t
                        else:
                            break
                    rows.append({
                        "family": fam, "ladder_seed": sseed, "variant": variant,
                        "rung_lo": lo, "rung_hi": hi,
                        "locality_ratio": locality,
                        "inside_share": float(d[inside].sum() / max(d.sum(), 1e-9)),
                        "inside_fraction_of_clip": float(inside.mean()),
                        "anticipation_s": float((cross - onset) / fps),
                        "d_geom_mean_m": float(d.mean()), "d_geom_peak_m": float(d.max()),
                        "d_null_peak_m": float(d_null.max()), "null_floor_m": floor,
                        "snr": float(d.max() / max(floor, 1e-6)),
                        "requested_pelvis_lo": ry_a, "requested_pelvis_hi": ry_b,
                        "collision_free_lo": repa["collision_free"],
                        "collision_free_hi": repb["collision_free"],
                    })
            print(f"  {fam} seed {sseed}: {len(ladder)} rungs ({time.time()-t0:.0f}s)",
                  flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp004_locality", "model": runner.model_name, "fps": fps,
                   "duration_s": DURATION, "families": FAMILIES,
                   "interaction_half_width_m": INTERACTION_HALF_WIDTH,
                   "null_shift_m": NULL_SHIFT,
                   "metric": "clearance envelope (top, halfwidth, pelvis)",
                   "n_rows": len(rows), "wall_clock_s": round(time.time() - t0, 1)}, fh,
                  indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
