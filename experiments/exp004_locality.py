"""EXP-004: is the adaptation LOCAL in space and ANTICIPATORY in time?

The question success rate cannot ask
------------------------------------
Two systems can both reach 83% on the beam family and be doing completely different things.
One raises the beam by 10 cm and adjusts the crouch by a few centimetres over the half-second
it spends underneath. The other re-solves from scratch and produces an unrelated trajectory
that also happens to work. Only the first has understood *which part of its behaviour the
geometry required it to change*.

So: take matched scenes that differ in exactly one clearance parameter (the families in
scene2motion/scenes.py are built as ladders — same family, same seed, one swept dimension) and
measure where the motion difference goes.

    LOCALITY      is the difference concentrated in the interaction interval, or smeared?
    ANTICIPATION  how long before the encounter does the difference become detectable?

Anticipation is not a nicety here. EXP-002 found that rendering the adaptation only where A*
labelled it — physically under the obstacle — collided on 4 of 6 beam heights, and dilating it
+/-0.8 s took penetration from 18.9 cm to 0.0 cm. The lead time is doing real work, and this
experiment measures it rather than assuming the dilation constant was right.

The control that makes it mean anything
---------------------------------------
A motion difference between two rungs has two sources: the geometry changed, and the sample
changed. Without separating them the metric measures nothing. So every geometric difference is
compared against a NOISE FLOOR measured on the *same* scene across seeds:

    d_geom(t) = || body(t; h+delta) - body(t; h) ||     same seed, adjacent rungs
    d_noise(t) = || body(t; h, seed_i) - body(t; h, seed_j) ||   same rung, different seeds

Anticipation onset is the first frame where d_geom exceeds the noise floor by a margin, not the
first frame where d_geom is merely nonzero.

Strawman baseline
-----------------
`global` applies the deepest mode the scene requires over the WHOLE clip instead of only where
needed. It should reach the goal collision-free just as often — and score badly on locality.
If the locality metric does not separate `global` from `adaptive`, the metric is broken.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.planner import (plan, plan_to_path_spec,  # noqa: E402
                                  plan_to_spec)
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import LADDERS, BUILDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED = 0.9
# These index BATCH POSITIONS, not torch seeds. ARDY seeds per generation call, so several
# copies of one spec in a single batch are genuinely independent samples -- which is exactly
# what a noise floor needs. Calling them "seeds" would misdescribe how they were drawn.
N_SAMPLES = 4
# Fixed duration across a whole ladder so frame t means the same thing in every rung.
# Without this, matched motions drift out of phase and every difference looks like adaptation.
DURATION = 10.0
FAMILIES = ["overhead_beam", "partial_beam", "narrow_gap"]
INTERACTION_HALF_WIDTH = 0.7   # m either side of the obstacle, along travel


def body_track(body: G1Body, qpos: np.ndarray) -> np.ndarray:
    """(T, G, 3) world positions of every collision primitive."""
    out = []
    for q in qpos:
        body.fk(q)
        out.append(body.data.geom_xpos[body.robot_geoms].copy())
    return np.asarray(out)


def diff_series(a: np.ndarray, b: np.ndarray, root_relative: bool = False) -> np.ndarray:
    """Per-frame mean body displacement between two aligned motions, metres."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if root_relative:
        # Subtracting the pelvis separates "held itself differently" from "went elsewhere".
        a = a - a[:, :1]
        b = b - b[:, :1]
    return np.linalg.norm(a - b, axis=-1).mean(axis=1)


def obstacle_x(sc) -> float:
    for k in ("beam_x", "wall_x", "box_x", "pillar_x"):
        if k in sc.meta:
            return float(sc.meta[k])
    return float(0.5 * (sc.start[0] + sc.goal[0]))


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
    rows, pairs = [], []

    for fam in FAMILIES:
        for sseed in range(args.seeds_per_rung):
            # One ladder: the same nuisance parameters, one swept clearance dimension.
            ladder = {}
            for v in LADDERS[fam]:
                sc = BUILDERS[fam](v, sseed)
                p = plan(sc, "adaptive")
                if not p.feasible:
                    continue
                body = G1Body(sc)
                ctrl = runner.generate(
                    [PROMPT], [plan_to_path_spec(p, fps, SPEED, duration=DURATION)], T,
                    args.diffusion_steps, seed=0)[0]
                dense = plan_to_spec(p, fps, ctrl, runner.joint_names, SPEED, duration=DURATION)

                # Strawman: hold the deepest requested pelvis height for the ENTIRE clip.
                # Limb targets are dropped rather than kept at their nominal heights, so the
                # arm is not left in a pose that contradicts a permanently lowered pelvis --
                # the contrast we want is purely "adapt where needed" vs "adapt throughout".
                deepest = float(np.min(dense.root_y))
                glob = ConstraintSpec(
                    root_xz=dense.root_xz.copy(), heading=dense.heading.copy(),
                    root_y=np.full_like(dense.root_y, deepest),
                    first_heading=dense.first_heading)

                # One call: N independent samples of the adapted spec plus the strawman.
                specs = [dense] * N_SAMPLES + [glob]
                outs = runner.generate([PROMPT] * len(specs), specs, T,
                                       args.diffusion_steps, seed=0)
                tracks = {}
                for j in range(N_SAMPLES):
                    q = runner.to_qpos(outs[j])
                    tracks[("adaptive", j)] = (q, body_track(body, q),
                                               body.trajectory_report(q))
                qg = runner.to_qpos(outs[N_SAMPLES])
                tracks[("global", 0)] = (qg, body_track(body, qg),
                                         body.trajectory_report(qg))
                ladder[v] = {"scene": sc, "tracks": tracks, "ox": obstacle_x(sc),
                             "min_root_y": deepest}
            if len(ladder) < 2:
                continue

            vals = sorted(ladder)
            for lo, hi in zip(vals[:-1], vals[1:]):
                A, B = ladder[lo], ladder[hi]
                # ONE noise floor per rung pair, measured on the adaptive samples and shared
                # by both arms. Giving each arm its own threshold would make the locality and
                # anticipation numbers incomparable -- the strawman would be scored against a
                # floor of zero and look infinitely anticipatory.
                noise = [diff_series(A["tracks"][("adaptive", i)][1],
                                     A["tracks"][("adaptive", j)][1])
                         for i in range(N_SAMPLES) for j in range(i + 1, N_SAMPLES)]
                d_noise = np.mean(noise, axis=0)
                floor = float(np.percentile(d_noise, 90))

                for variant in ("adaptive", "global"):
                    ka = (variant, 0)
                    if ka not in A["tracks"] or ka not in B["tracks"]:
                        continue
                    qa, ta, repa = A["tracks"][ka]
                    qb, tb, repb = B["tracks"][ka]
                    d_geom = diff_series(ta, tb)
                    d_geom_rel = diff_series(ta, tb, root_relative=True)
                    ox = A["ox"]
                    n = len(d_geom)
                    inside = np.abs(qa[:n, 0] - ox) <= INTERACTION_HALF_WIDTH
                    if inside.sum() < 3 or (~inside).sum() < 3:
                        continue
                    # Locality: how much more the body differs inside the interaction window
                    # than outside it. 1.0 = uniformly smeared, high = surgical.
                    locality = float(d_geom[inside].mean() / max(d_geom[~inside].mean(), 1e-6))
                    inside_share = float(d_geom[inside].sum() / max(d_geom.sum(), 1e-9))

                    # Anticipation: first frame before the encounter where the geometric
                    # difference clears the sampling noise floor and stays clear.
                    cross = int(np.argmin(np.abs(qa[:n, 0] - ox)))
                    above = d_geom > (floor * 1.5 + 1e-4)
                    onset = cross
                    for t in range(cross, -1, -1):
                        if above[t]:
                            onset = t
                        else:
                            break
                    rows.append({
                        "family": fam, "ladder_seed": sseed, "variant": variant,
                        "n_samples_for_floor": N_SAMPLES,
                        "rung_lo": lo, "rung_hi": hi,
                        "locality_ratio": locality,
                        "inside_share": inside_share,
                        "inside_fraction_of_clip": float(inside.mean()),
                        "anticipation_s": float((cross - onset) / fps),
                        "d_geom_mean_m": float(d_geom.mean()),
                        "d_geom_peak_m": float(d_geom.max()),
                        "d_geom_rel_peak_m": float(d_geom_rel.max()),
                        "noise_floor_m": floor,
                        "snr": float(d_geom.max() / max(floor, 1e-6)),
                        "collision_free_lo": repa["collision_free"],
                        "collision_free_hi": repb["collision_free"],
                        "min_root_y_lo": A["min_root_y"], "min_root_y_hi": B["min_root_y"],
                    })
            pairs.append((fam, sseed, len(ladder)))
            print(f"  {fam} seed {sseed}: {len(ladder)} rungs ({time.time()-t0:.0f}s)",
                  flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp004_locality", "model": runner.model_name, "fps": fps,
                   "duration_s": DURATION, "families": FAMILIES, "n_samples": N_SAMPLES,
                   "interaction_half_width_m": INTERACTION_HALF_WIDTH,
                   "ladders": pairs, "n_rows": len(rows),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
