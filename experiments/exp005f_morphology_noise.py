"""EXP-005f: are the missed body variants stable MODES, or one continuum sampled densely?

The question this must not beg
------------------------------
EXP-005e found the enumerator misses body adaptations at equal cost. It is tempting to read
that as "there are strategies it cannot find". But a low recall over continuous dip/tuck/lift
samples is *not yet* evidence of missing strategies — it may be evidence that one continuous
feasible manifold was sampled densely, in which case counting samples measures the sampler.

So before any coverage number is computed, three things have to be separated:

  1. genuinely different body strategies        -> different discrete ACTIVE SET
  2. continuous variation within one strategy   -> same active set, different amplitude
  3. seed-induced variation                     -> the same program, twice

and (3) has to be measured first, because it sets the unit for (2). This experiment measures
the ARDY seed-noise floor, then reports how far apart body programs actually are in units of
that noise.

Design
------
For a fixed scene and a fixed ROUTE, generate:

  * a neutral-body control `b_0` (route only, standing) at N seeds -- the matched control
  * a ladder of body programs spanning the reachable box (dip x tuck x lift), each at N seeds

Every descriptor is a PAIRED DELTA against the control at the SAME seed, so gait phase, the
opening arm swing, and seed style all cancel. That pairing is not optional: EXP-001 once
concluded that ducking makes the robot wider from an unpaired comparison, and the effect
vanished under matched control.

Then report:

  Sigma_seed   pooled within-program covariance -> the whitening for d_morph
  d_morph      between-program distances in units of seed sigma
  Stability    fraction of seeds landing on a program's modal active set
  active sets  how many DISCRETE strategies the reachable box actually contains
  eps-net size how many distinguishable points remain after deduplication, vs eps

If the eps-net size collapses to a handful while the raw sample count is large, the missed
variants are a continuum and a set-valued generative model is the wrong tool -- a
preference-conditioned deterministic map tracing the frontier would be cleaner. If instead
several stable, well-separated active sets survive, the set-valued model is justified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.morphology import (CHANNELS, Interaction, active_set,  # noqa: E402
                                     d_morph, envelope_series, epsilon_net,
                                     matched_delta, raw_descriptor, seed_statistics,
                                     stability, whitener)
from scene2motion.planner import plan, plan_to_path_spec, plan_to_spec  # noqa: E402
from scene2motion.program import ConstraintProgram, encode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS, LADDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL = 0.9, 0.5
N_SEEDS = 6
# A ladder spanning the measured reachable box. Deliberately dense in dip, because that is the
# axis EXP-001 showed is strong and therefore the one most likely to be a continuum.
DIPS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
TUCKS = [0.0, 0.4, 0.85]
LIFTS = [0.0, 0.35]
SCENES = [("overhead_beam", 1.10), ("overhead_beam", 0.90), ("partial_beam", 1.05),
          ("narrow_gap", 0.70), ("beam_and_gap", 1.10), ("pillar", 0.30)]


def body_program(route_prog: ConstraintProgram, dip: float, tuck: float,
                 lift: float) -> ConstraintProgram:
    """The route's program with ONE adaptation slot set to (dip, tuck, lift) at mid-route."""
    p = ConstraintProgram(lat=route_prog.lat.copy(), slot=np.zeros_like(route_prog.slot),
                          speed=route_prog.speed)
    p.slot[0] = [0.5, 0.12, dip, tuck, lift]
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp005f")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    from scene2motion.program import decode

    rows, all_covs, per_scene = [], [], {}
    for fam, val in SCENES:
        sc = BUILDERS[fam](val, 0)
        p = plan(sc, "adaptive")
        if not p.feasible:
            print(f"  {fam} {val}: route infeasible, skipped", flush=True)
            continue
        body = G1Body(sc)
        ox = float(sc.meta.get("beam_x") or sc.meta.get("wall_x")
                   or sc.meta.get("pillar_x") or sc.meta.get("box_x") or 4.0)
        inter = Interaction(ox)
        T = min(int(14 * fps), max(int(2 * fps), int(round(p.length / SPEED * fps))))
        seeds = list(range(100, 100 + N_SEEDS))

        # matched control: this route, neutral body, one clip per seed
        ctrl_ref = runner.generate([PROMPT],
                                   [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                                   args.diffusion_steps, seeds=[seeds[0]])[0]
        route_prog = encode(p, sc, fps, SPEED)
        neutral = body_program(route_prog, 0.0, 0.0, 0.0)
        ctrl_spec = decode(neutral, sc, fps, ctrl_ref, runner.joint_names, duration=T / fps)
        ctrls = runner.generate([PROMPT] * N_SEEDS, [ctrl_spec] * N_SEEDS, T,
                                args.diffusion_steps, seeds=seeds)
        ctrl_q = [runner.to_qpos(o) for o in ctrls]
        ctrl_d = [raw_descriptor(body, q, inter, fps) for q in ctrl_q]
        ctrl_e = [envelope_series(body, q) for q in ctrl_q]

        # the ladder
        combos = [(d, t, l) for d in DIPS for t in TUCKS for l in LIFTS]
        prog_rows = []
        for (dip, tuck, lift) in combos:
            prog = body_program(route_prog, dip, tuck, lift)
            spec = decode(prog, sc, fps, ctrl_ref, runner.joint_names, duration=T / fps)
            outs = []
            for i in range(0, N_SEEDS, 8):
                k = seeds[i:i + 8]
                outs += runner.generate([PROMPT] * len(k), [spec] * len(k), T,
                                        args.diffusion_steps, seeds=k)
            deltas, sigs, oks = [], [], []
            for s, o in enumerate(outs):
                q = runner.to_qpos(o)
                rep = body.trajectory_report(q)
                goal = bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL)
                dsc = raw_descriptor(body, q, inter, fps)
                dm = matched_delta(dsc, ctrl_d[s], q, ctrl_q[s], inter, fps, body=body,
                                   env_adapted=envelope_series(body, q),
                                   env_control=ctrl_e[s])
                deltas.append(dm)
                oks.append(bool(goal and rep["collision_free"]))
            deltas = np.array(deltas)
            mu, cov = seed_statistics(deltas)
            all_covs.append(cov)
            prog_rows.append({"dip": dip, "tuck": tuck, "lift": lift,
                              "mu": mu.tolist(), "deltas": deltas.tolist(),
                              "valid_frac": float(np.mean(oks))})
        per_scene[sc.scene_id] = {"rows": prog_rows, "family": fam, "param": val}
        print(f"  {fam} {val}: {len(combos)} programs x {N_SEEDS} seeds "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- pooled seed noise, then everything measured in units of it --------------------
    W = whitener(all_covs)
    noise_q99 = np.zeros(len(CHANNELS))
    resid = []
    for sid, sd in per_scene.items():
        for r in sd["rows"]:
            d = np.array(r["deltas"])
            resid.append(np.abs(d - d.mean(axis=0)))
    if resid:
        noise_q99 = np.percentile(np.concatenate(resid), 99, axis=0)

    summary = {"experiment": "exp005f_morphology_noise", "n_seeds": N_SEEDS,
               "channels": list(CHANNELS),
               "seed_noise_q99": noise_q99.tolist(),
               "scenes": {}}
    for sid, sd in per_scene.items():
        rows_ = [r for r in sd["rows"] if r["valid_frac"] > 0.5]
        if not rows_:
            continue
        mus = np.array([r["mu"] for r in rows_])
        sigs = []
        stab = []
        for r in rows_:
            per_seed = [active_set(np.array(d), noise_q99) for d in r["deltas"]]
            stab.append(stability(per_seed))
            sigs.append(Counter(per_seed).most_common(1)[0][0])
        nets = {str(e): len(epsilon_net(mus, W, e)) for e in (1.0, 2.0, 3.0, 5.0)}
        # nearest-neighbour distance between distinct programs, in seed sigma
        nn = []
        for i in range(len(mus)):
            o = [d_morph(mus[i], mus[j], W) for j in range(len(mus)) if j != i]
            if o:
                nn.append(min(o))
        summary["scenes"][sid] = {
            "family": sd["family"], "param": sd["param"],
            "n_valid_programs": len(rows_),
            "n_distinct_active_sets": len(set(sigs)),
            "active_sets": [list(map(str, s)) for s in sorted(set(sigs))],
            "mean_stability": float(np.mean(stab)),
            "frac_stable_ge_0.8": float(np.mean([s >= 0.8 for s in stab])),
            "epsilon_net_sizes": nets,
            "median_nn_distance_sigma": float(np.median(nn)) if nn else None,
        }
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "per_scene.json", "w") as fh:
        json.dump(per_scene, fh)
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}, indent=2))
    for sid, v in summary["scenes"].items():
        print(f"\n{sid[:34]:34s} valid {v['n_valid_programs']:3d}  "
              f"distinct active sets {v['n_distinct_active_sets']:2d}  "
              f"stability {v['mean_stability']:.2f}  "
              f"eps-net {v['epsilon_net_sizes']}  nn {v['median_nn_distance_sigma']}")


if __name__ == "__main__":
    main()
