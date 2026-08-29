"""EXP-014: EXP-011 and EXP-012 tested a JUMP. This tests the step-over the planner ships.

The error
---------
EXP-013 computed foot contacts on the lift reference and found `mean_contacts = 0.00` over the
whole interaction window: **no foot on the ground at any frame.**  A motion with no contacts is
ballistic, and no controller can track it, so "lift is 0/8 executable" was never a statement
about stepping over anything.

The cause is in my experiment code, not in the planner.  `planner._limb_targets` gates the lift
PER SIDE on that leg already being airborne:

    for side in (left, right):
        air = ~contacts[:, side_pair].any(-1)
        for f in frames where lift > 0.02:
            if air[f]:  raise THAT side's knee/ankle/toe

EXP-011 and EXP-013 raised **both** legs at every frame in the window; EXP-012 raised both legs
at frames where *either* leg was airborne.  All three requested a two-legged raise.  The shipped
step-over has never been tracked.

The test
--------
Three bodies, matched per-sample seeds, tracked separately:

    neutral      control
    lift-BOTH    the ungated two-leg raise EXP-011/012 actually tested, kept so the comparison
                 is within one run rather than across experiments
    lift-GATED   `planner._limb_targets` itself -- the mechanism the planner uses

`mean_contacts` is reported alongside tracking success as the manipulation check: lift-GATED must
keep a foot on the ground and lift-BOTH must not, or the two arms are not different requests.

This is the fourth time in this project that a capability was declared absent and the cause
turned out to be how it was asked for.  The prior claim is not retracted until this run reports;
it is suspended.
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

PROMPT, SPEED = "A person walks forward.", 0.9
BODIES = ["neutral", "lift-BOTH", "lift-GATED"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp014")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--diffusion_steps", type=int, default=5)
    ap.add_argument("--lift", type=float, default=0.35)
    ap.add_argument("--timeout_s", type=int, default=2400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(args.duration * fps)
    body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
    seeds = list(range(1300, 1300 + args.n_seeds))
    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading, root_y = np.zeros(T), np.full(T, 0.78)
    win = range(int(0.35 * T), int(0.65 * T))
    base = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y)

    ctrl = runner.generate([PROMPT] * len(seeds), [base] * len(seeds), T,
                           args.diffusion_steps, seeds=seeds)
    nominal = ctrl[0]

    # the lift channel the planner would render: `lift` inside the window, 0 outside
    lift_ch = np.zeros(T)
    lift_ch[win.start:win.stop] = args.lift
    tuck_ch = np.zeros(T)

    gf, gj, gt = _limb_targets(root_xz, tuck_ch, lift_ch, nominal, runner.joint_names, fps,
                               root_y=root_y)
    if gf is None:
        raise SystemExit("_limb_targets produced no targets — no airborne frames to lift")
    print(f"lift-GATED: {len(gf)} frames x {len(gj)} joints from the shipped renderer",
          flush=True)

    nom_j = np.asarray(nominal["posed_joints"])
    nom_r = np.asarray(nominal["smooth_root_pos"])
    both = np.array([i for i, n in enumerate(runner.joint_names)
                     if any(k in n for k in ("knee", "ankle", "toe"))])
    bf = np.arange(win.start, win.stop, max(1, int(0.12 * fps)))
    off = nom_j[bf][:, both, :] - nom_r[bf][:, None, :]
    bt = np.stack([root_xz[bf][:, None, 0] + off[:, :, 0],
                   nom_j[bf][:, both, 1] + args.lift,
                   root_xz[bf][:, None, 1] + off[:, :, 2]], -1)
    print(f"lift-BOTH : {len(bf)} frames x {len(both)} joints, ungated (what EXP-011 tested)",
          flush=True)

    specs = {"neutral": base,
             "lift-BOTH": ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                                         pos_frames=bf, pos_joints=both, pos_targets=bt),
             "lift-GATED": ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                                          pos_frames=gf, pos_joints=gj, pos_targets=gt)}

    contacts = {}
    for label in BODIES:
        outs = runner.generate([PROMPT] * len(seeds), [specs[label]] * len(seeds), T,
                               args.diffusion_steps, seeds=seeds)
        clips, nsup = {}, []
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            clips[f"{label}__s{seeds[s]}"] = q
            nsup.append(np.mean([len(foot_support(body, q[t])) for t in win]))
        contacts[label] = float(np.mean(nsup))
        write_motion_pkl(clips, out / f"{label}.pkl", fps=int(round(fps)), mj_model=body.model)
        print(f"  {label:11s} mean foot contacts {contacts[label]:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    del runner

    results = {}
    for label in BODIES:
        rc, log = run_sonic(out / f"{label}.pkl", out / label, args.n_seeds, args.timeout_s)
        (out / f"sonic_{label}.log").write_text(log)
        m = {"returncode": rc, "mean_contacts": contacts[label]}
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
        results[label] = m
        print(f"  {label:11s} rc {rc}  success {m.get('success_rate', float('nan')):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    b = results.get("neutral", {}).get("success_rate", float("nan"))
    print(f"\n{'body':12s} {'contacts':>9s} {'success':>8s} {'vs neutral':>11s} "
          f"{'progress':>9s} {'accel':>8s}")
    print("-" * 62)
    for label in BODIES:
        m = results[label]
        print(f"{label:12s} {m['mean_contacts']:9.2f} "
              f"{m.get('success_rate', float('nan')):8.3f} "
              f"{m.get('success_rate', float('nan')) - b:+11.3f} "
              f"{m.get('progress_rate', float('nan')):9.3f} "
              f"{m.get('accel_dist', float('nan')):8.2f}")
    print("\n  contacts is the manipulation check: lift-GATED must keep a foot down and\n"
          "  lift-BOTH must not.  If they match, the arms are not different requests.")
    json.dump({"experiment": "exp014_gated_lift_tracked", "n_seeds": args.n_seeds,
               "lift_m": args.lift, "results": results,
               "wall_clock_s": round(time.time() - t0, 1)},
              open(out / "receipt.json", "w"), indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
