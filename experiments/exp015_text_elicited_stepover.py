"""EXP-015: ask for the behaviour with TEXT and address only the ROOT.

The threat this addresses
-------------------------
The project's conclusion — one executable body axis — is a well-measured statement about the
43-dimensional CONSTRAINT INTERFACE.  It is not yet a statement about ARDY.  Every step-over this
project ever requested went through `global_joints_positions`, a channel the pose decoder does
not read (design §24), which over-responds by ~2.5x, and which produces a reference with no foot
on the ground (design §34) — so all three refutations of lift tracked a levitation.

But ARDY is a TEXT-conditioned model, and the root channels are the ones that demonstrably work
(root_2d tracks the pelvis to ~1.4 cm; root_y_pos is the one clean capability).  So there is an
obvious request this project has never made: **name the behaviour in the prompt, constrain only
the root, and never touch a body slice at all.**  If the frozen prior contains a supported,
contact-consistent step-over, that is how to elicit it, and its absence from our interface would
be a fact about our encoding rather than about the model.

This is the strongest remaining threat to the report's central claim, and it costs nothing:
`'A person steps over an obstacle.'` is already in the embedding cache, so no text encoder has
to be loaded.

Design
------
Two arms, IDENTICAL root constraints, identical per-sample seeds, differing only in the prompt:

    A  "A person walks forward."            root path only
    B  "A person steps over an obstacle."   root path only  <- same constraints, different words

and, as a reference point rather than a competitor, the interface's own attempt:

    C  "A person walks forward." + the shipped gated lift through the position channel

Measured: peak foot height in the window, mean ground contacts, and pelvis displacement, all
paired against arm A at the same seed.  Then each arm is tracked by SONIC separately.

What would overturn the report
------------------------------
Arm B raising the swing foot materially above arm A **while keeping a foot on the ground** and
tracking near neutral.  That would mean the prior can step over and our constraint interface
could not ask for it.

What would confirm it
---------------------
Arm B looking like arm A.  Then the behaviour is not elicitable by text either, the interface is
not the limitation, and "step-over is unavailable on this system" holds through a fourth
independent mechanism.
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
from scene2motion.planner import _limb_targets  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp011_tracked_addressability import run_sonic  # noqa: E402
from exp013_zmp_feasibility import foot_support  # noqa: E402

WALK = "A person walks forward."
STEP = "A person steps over an obstacle."          # already in outputs/text_cache.npz
SPEED = 0.9
ARMS = ["A-walk-text", "B-stepover-text", "C-walk-text+pos-lift"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp015")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--diffusion_steps", type=int, default=5)
    ap.add_argument("--lift", type=float, default=0.20)
    ap.add_argument("--timeout_s", type=int, default=2400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(args.duration * fps)
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(1500, 1500 + args.n_seeds))
    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading, root_y = np.zeros(T), np.full(T, 0.78)
    win = range(int(0.35 * T), int(0.65 * T))
    base = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y)
    feet = [g for g in body.robot_geoms if "foot" in body.geom_name[g]]

    def describe(q: np.ndarray) -> tuple[float, float, float]:
        hi, ns = [], []
        for t in win:
            body.fk(q[t])
            hi.append(float(body.data.geom_xpos[feet][:, 2].max()))
            ns.append(len(foot_support(body, q[t])))
        return max(hi), float(np.mean(ns)), float(q[win.start:win.stop, 2].mean())

    # arm A first: it is the matched control for the paired deltas
    outs_a = runner.generate([WALK] * len(seeds), [base] * len(seeds), T,
                             args.diffusion_steps, seeds=seeds)
    a_desc = [describe(runner.to_qpos(o)) for o in outs_a]

    lift_ch = np.zeros(T)
    lift_ch[win.start:win.stop] = args.lift
    gf, gj, gt = _limb_targets(root_xz, np.zeros(T), lift_ch, outs_a[0], runner.joint_names,
                               fps, root_y=root_y)
    spec_c = base if gf is None else ConstraintSpec(
        root_xz=root_xz, heading=heading, root_y=root_y,
        pos_frames=gf, pos_joints=gj, pos_targets=gt)

    plan = {"A-walk-text": (WALK, base), "B-stepover-text": (STEP, base),
            "C-walk-text+pos-lift": (WALK, spec_c)}
    kin = {}
    for label in ARMS:
        prompt, spec = plan[label]
        outs = outs_a if label == "A-walk-text" else runner.generate(
            [prompt] * len(seeds), [spec] * len(seeds), T, args.diffusion_steps, seeds=seeds)
        clips, d = {}, []
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            clips[f"{label}__s{seeds[s]}"] = q
            fp, nc, pel = describe(q)
            d.append((fp - a_desc[s][0], nc, pel - a_desc[s][2]))
        d = np.array(d)
        kin[label] = {"d_foot_mm": 1000 * float(d[:, 0].mean()),
                      "d_foot_sd_mm": 1000 * float(d[:, 0].std(ddof=1)),
                      "contacts": float(d[:, 1].mean()),
                      "d_pelvis_mm": 1000 * float(d[:, 2].mean())}
        write_motion_pkl(clips, out / f"{label}.pkl", fps=int(round(fps)), mj_model=body.model)
        print(f"  {label:22s} d_foot {kin[label]['d_foot_mm']:+7.1f} mm  "
              f"contacts {kin[label]['contacts']:5.2f}  "
              f"d_pelvis {kin[label]['d_pelvis_mm']:+7.1f} mm  ({time.time()-t0:.0f}s)",
              flush=True)
    del runner

    res = {}
    for label in ARMS:
        rc, log = run_sonic(out / f"{label}.pkl", out / label, args.n_seeds, args.timeout_s)
        (out / f"sonic_{label}.log").write_text(log)
        m = {"returncode": rc, **kin[label]}
        for line in log.splitlines():
            if line.startswith("Success Rate:"):
                m["success_rate"] = float(line.split(":")[1])
            if line.startswith("Progress Rate:"):
                m["progress_rate"] = float(line.split(":")[1])
            if line.startswith("All:"):
                for t in line.split("\t"):
                    t = t.strip()
                    if t.split(":")[0] in ("mpjpe_l", "accel_dist"):
                        m[t.split(":")[0]] = float(t.split(":")[1])
        res[label] = m
        print(f"  {label:22s} rc {rc}  success {m.get('success_rate', float('nan')):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'arm':24s} {'d_foot':>9s} {'sd':>7s} {'contacts':>9s} {'d_pelvis':>9s} "
          f"{'success':>8s} {'accel':>7s}")
    print("-" * 78)
    for label in ARMS:
        m = res[label]
        print(f"{label:24s} {m['d_foot_mm']:+9.1f} {m['d_foot_sd_mm']:7.1f} "
              f"{m['contacts']:9.2f} {m['d_pelvis_mm']:+9.1f} "
              f"{m.get('success_rate', float('nan')):8.3f} "
              f"{m.get('accel_dist', float('nan')):7.2f}")
    b = res["B-stepover-text"]
    print(f"\n  Arm B differs from arm A ONLY in the prompt -- identical root constraints, "
          f"identical seeds.\n  It raises the swing foot by {b['d_foot_mm']:+.0f} mm "
          f"(sd {b['d_foot_sd_mm']:.0f}) while holding {b['contacts']:.2f} ground contacts.\n"
          f"  A material rise WITH contact preserved and tracking near neutral would mean the "
          f"prior can\n  step over and our constraint interface could not ask for it.")
    json.dump({"experiment": "exp015_text_elicited_stepover", "n_seeds": args.n_seeds,
               "prompts": {"A": WALK, "B": STEP}, "lift_for_C": args.lift,
               "results": res, "wall_clock_s": round(time.time() - t0, 1)},
              open(out / "receipt.json", "w"), indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
