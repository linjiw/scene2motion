"""EXP-006: is the low control bandwidth the PRIOR, or the inference setting?

The question
------------
Every capability number this project has produced -- 8 nominal channels, 2 of them dead, ~3.5
stable strategies per scene, a seed scatter wide enough that eps had to be calibrated at 3.99
units on a null arm -- was measured at ONE inference configuration: ARDY-G1 Horizon52, 10
denoising steps, cfg_weight (text 2.0, constraint 2.0).  Before any of it is written down as a
property of the trained prior, it has to be shown not to be a property of that configuration.

ARDY exposes exactly the knobs that matter.  `ardy/model/cfg.py:149` implements SEPARATED
classifier-free guidance,

    out = out_uncond + w_text * (out_text - out_uncond) + w_cstr * (out_cstr - out_uncond)

so the constraint term can be strengthened without touching the text term, and the registry
ships both G1 horizons (`g152`, `g18`).

This is emphatically NOT a tuning run
-------------------------------------
Tuning the inference setting on the gate's scenes and then re-reporting the gate would be
choosing a configuration by its effect on a headline number.  So this suite uses FAMILIES THE
GATE NEVER SAW (`low_obstacle`, `pillar`, `beam_and_gap` -- the gate's 30 scenes were
partial_beam / overhead_beam / narrow_gap), and its output is a frontier, not a winner.

Three outcomes, all publishable, declared before the run
--------------------------------------------------------
  1. scatter stays large at every setting            -> non-addressability is prior-level, and
                                                        the capability audit stands as measured
  2. stronger constraint guidance sharpens control
     without damaging motion quality                 -> adopt it, and re-measure the audit once
  3. stronger guidance buys control and costs quality -> there is a controllability-naturalness
                                                        frontier, which is itself a result

Measured per configuration
--------------------------
    q99 seed scatter per channel     the width of p(m | c) -- the thing eps has to sit above
    P(channel fires | requested)     did the request reach the body at all
    P(valid)                         goal reached and collision-free
    roughness / contact / jerk       what stronger guidance costs
    latency                          what it costs in seconds

`P(channel fires)` is judged against the FIXED EXP-005f noise quantiles for every
configuration, so all settings are read on one ruler; each configuration's own measured scatter
is reported separately, as the quantity being compared rather than the yardstick.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.morphology import (CHANNELS, Interaction, active_set,  # noqa: E402
                                     envelope_series, matched_delta, raw_descriptor)
from scene2motion.planner import plan, plan_to_path_spec  # noqa: E402
from scene2motion.program import ConstraintProgram, decode, encode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL = 0.9, 0.5

# Families the gate never scored, so nothing here can be tuning on the test set.
SCENES = [("low_obstacle", 0.20), ("low_obstacle", 0.30), ("pillar", 0.30),
          ("pillar", 0.45), ("beam_and_gap", 1.10), ("beam_and_gap", 1.20)]

# (label, dip, tuck, lift) -- one per capability claim, plus the neutral control.
REQUESTS = [("neutral", 0.00, 0.00, 0.00),
            ("duck", 0.35, 0.00, 0.00),
            ("duck-deep", 0.50, 0.00, 0.00),
            ("lift", 0.00, 0.00, 0.35),
            ("tuck", 0.00, 0.60, 0.00),
            ("duck+lift", 0.35, 0.00, 0.35)]

# (label, model, denoising steps, (w_text, w_cstr))
#
# MORE denoising steps is NOT a knob this checkpoint exposes, and the guidance's suggestion to
# test it cannot be followed.  `diffusion.space_timesteps` builds the schedule as
# `arange(num_base_steps) * frac_stride`, so it always returns exactly `num_base_steps` entries
# -- 10 for ARDY-G1-RP-25FPS-Horizon52 -- while the sampler loops `num_denoising_steps` times
# indexing those buffers.  Asking for 15 runs off the end and dies in an IndexKernel with
# `index out of bounds`, a device-side assert rather than a checked error.  Verified: 10 works,
# 15 fails.  So the schedule can only be made COARSER, and the ladder goes down from 10 rather
# than up.  Constraint guidance and the horizon are still free.
CONFIGS = [("base  s10 cfg2/2", "g1", 10, (2.0, 2.0)),
           ("fewer s5  cfg2/2", "g1", 5, (2.0, 2.0)),
           ("cstr  s10 cfg2/4", "g1", 10, (2.0, 4.0)),
           ("cstr  s10 cfg2/6", "g1", 10, (2.0, 6.0)),
           ("cstr  s10 cfg2/9", "g1", 10, (2.0, 9.0)),
           ("horiz s10 cfg2/2 h8", "g18", 10, (2.0, 2.0))]

# Which descriptor channel each request should move, if the channel works at all.
FIRES = {"duck": [0], "duck-deep": [0], "lift": [3, 4], "tuck": [1, 2],
         "duck+lift": [0, 3, 4], "neutral": []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp006")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--configs", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    q99 = np.array(json.load(open("outputs/exp005f/receipt.json"))["seed_noise_q99"])
    configs = [c for c in CONFIGS if args.configs is None or c[0].split()[0] in args.configs]
    seeds = list(range(200, 200 + args.n_seeds))
    runners: dict[str, ArdyRunner] = {}
    rows = []

    for label, model, steps, cfg in configs:
        if model not in runners:
            try:
                runners[model] = ArdyRunner(model_name=model,
                                            cache_path="outputs/text_cache.npz")
            except Exception as e:
                # Horizon8 is a different checkpoint and may not roll out to this clip length.
                # One configuration failing must not cost the other four.
                print(f"  {label}: cannot load {model} ({type(e).__name__}: {e}); skipped",
                      flush=True)
                continue
        runner = runners[model]
        fps = runner.fps
        for fam, val in SCENES:
            sc = BUILDERS[fam](val, 0)
            p = plan(sc, "adaptive")
            if not p.feasible:
                continue
            if any(r["config"] == label and r["scene"] == sc.scene_id for r in rows):
                continue
            body = G1Body(sc)
            ox = float(sc.meta.get("beam_x") or sc.meta.get("wall_x")
                       or sc.meta.get("pillar_x") or sc.meta.get("box_x") or 4.0)
            inter = Interaction(ox)
            T = min(int(14 * fps), max(int(2 * fps), int(round(p.length / SPEED * fps))))
            route = encode(p, sc, fps, SPEED)

            def prog_for(dip, tuck, lift):
                pr = ConstraintProgram(lat=route.lat.copy(),
                                       slot=np.zeros_like(route.slot), speed=route.speed)
                pr.slot[0] = [0.5, 0.12, dip, tuck, lift]
                return pr

            ref = runner.generate([PROMPT],
                                  [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                                  steps, cfg_weight=cfg, seeds=[seeds[0]])[0]
            # matched control: same route, neutral body, SAME config, SAME seeds
            cspec = decode(prog_for(0, 0, 0), sc, fps, ref, runner.joint_names,
                           duration=T / fps)
            cq, cd, ce = [], [], []
            for i in range(0, len(seeds), 8):
                blk = seeds[i:i + 8]
                for o in runner.generate([PROMPT] * len(blk), [cspec] * len(blk), T, steps,
                                         cfg_weight=cfg, seeds=blk):
                    q = runner.to_qpos(o)
                    cq.append(q)
                    cd.append(raw_descriptor(body, q, inter, fps))
                    ce.append(envelope_series(body, q))

            for name, dip, tuck, lift in REQUESTS:
                spec = decode(prog_for(dip, tuck, lift), sc, fps, ref, runner.joint_names,
                              duration=T / fps)
                g0 = time.time()
                outs = []
                for i in range(0, len(seeds), 8):
                    blk = seeds[i:i + 8]
                    outs += runner.generate([PROMPT] * len(blk), [spec] * len(blk), T, steps,
                                            cfg_weight=cfg, seeds=blk)
                lat = (time.time() - g0) / max(len(outs), 1)
                for s, o in enumerate(outs):
                    q = runner.to_qpos(o)
                    rep = body.trajectory_report(q)
                    d = matched_delta(raw_descriptor(body, q, inter, fps), cd[s], q, cq[s],
                                      inter, fps, body=body,
                                      env_adapted=envelope_series(body, q), env_control=ce[s])
                    env = envelope_series(body, q)
                    rows.append({
                        "config": label, "model": model, "steps": steps,
                        "w_text": cfg[0], "w_cstr": cfg[1],
                        "scene": sc.scene_id, "family": fam, "request": name, "seed": seeds[s],
                        "delta": np.round(d, 5).tolist(),
                        "sig": [bool(x) if not isinstance(x, int) else int(x)
                                for x in active_set(d, q99)],
                        "valid": bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL
                                      and rep["collision_free"]),
                        "goal_ok": bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal))
                                        < GOAL_TOL),
                        "coll_free": bool(rep["collision_free"]),
                        "rough": float(np.abs(np.diff(env, n=2, axis=0)).mean())
                                 if len(env) > 2 else 0.0,
                        "jerk": float(np.abs(np.diff(q[:, 7:], n=2, axis=0)).mean()
                                      * fps * fps),
                        "contact": float(rep["mean_foot_floor_penetration_m"]),
                        "latency_s": lat})
            print(f"  {label:22s} {sc.scene_id[:26]:26s} ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # ---- report -----------------------------------------------------------------------
    print(f"\n{len(rows)} clips over {len(configs)} configurations, "
          f"{args.n_seeds} seeds each ({time.time()-t0:.0f}s)\n")
    summary = {"experiment": "exp006_inference_sensitivity", "n_rows": len(rows),
               "n_seeds": args.n_seeds, "configs": {}}
    hdr = (f"{'configuration':22s} {'P(valid)':>9s} {'duck':>7s} {'lift':>7s} {'tuck':>7s} "
           f"{'scatter':>8s} {'rough':>8s} {'jerk':>8s} {'contact':>8s} {'s/clip':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for label, model, steps, cfg in configs:
        R = [r for r in rows if r["config"] == label]
        if not R:
            continue
        # per-config seed scatter: q99 of |delta - mean(delta)| within (scene, request)
        resid = []
        for sid in {r["scene"] for r in R}:
            for req in {r["request"] for r in R}:
                D = np.array([r["delta"] for r in R
                              if r["scene"] == sid and r["request"] == req])
                if len(D) >= 2:
                    resid.append(np.abs(D - D.mean(0)))
        sc_q99 = (np.percentile(np.concatenate(resid), 99, axis=0) if resid
                  else np.zeros(len(CHANNELS)))

        def fired(req, idxs):
            rr = [r for r in R if r["request"] == req]
            if not rr:
                return float("nan")
            # sig is (duck, tuck, liftL, liftR, order, yaw); map channel -> sig position
            pos = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
            return float(np.mean([any(bool(r["sig"][pos[i]]) for i in idxs) for r in rr]))

        row = {"P_valid": float(np.mean([r["valid"] for r in R])),
               "duck_fires": fired("duck", [0]), "lift_fires": fired("lift", [3, 4]),
               "tuck_fires": fired("tuck", [1, 2]),
               "scatter_mean_ratio": float(np.mean(sc_q99[:5] / np.maximum(q99[:5], 1e-9))),
               "scatter_q99": sc_q99.tolist(),
               "rough": float(np.mean([r["rough"] for r in R])),
               "jerk": float(np.mean([r["jerk"] for r in R])),
               "contact": float(np.mean([r["contact"] for r in R])),
               "latency_s": float(np.mean([r["latency_s"] for r in R]))}
        summary["configs"][label] = row
        print(f"{label:22s} {row['P_valid']:9.3f} {row['duck_fires']:7.3f} "
              f"{row['lift_fires']:7.3f} {row['tuck_fires']:7.3f} "
              f"{row['scatter_mean_ratio']:8.3f} {row['rough']:8.4f} {row['jerk']:8.2f} "
              f"{row['contact']:8.4f} {row['latency_s']:7.3f}")
    print("\n  duck/lift/tuck = P(the requested channel actually fired), judged against the "
          "FIXED\n  EXP-005f noise quantiles so every configuration is read on one ruler.\n"
          "  scatter = this configuration's own q99 seed scatter over the five body channels, "
          "as a\n  RATIO to the base measurement: < 1 means stronger guidance genuinely "
          "narrowed p(m | c).\n  rough / jerk / contact are what that control costs in motion "
          "quality.")
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
