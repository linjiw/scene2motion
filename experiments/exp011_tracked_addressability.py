"""EXP-011: do the kinematically addressable body modes survive a physics tracker?

Why this is the experiment the project has been missing
------------------------------------------------------
Every capability number here -- the funnel, the envelope, the commanded-versus-realised matrix,
EXP-008's 8.68 sigma and EXP-010's 10.64 -- is computed from ARDY's exported qpos through MuJoCo
forward kinematics. Nothing has ever been executed. Two clips can be distinct, collision-free
and perfectly addressable in that sense and still, once a controller has to realise them,
converge to the same posture, lose the clearance the plan was built on, or fall over.

    TrackedAddressability(z) = P[ v_dynamic = 1  and  sigma_tracked = z ]

This stage can only ever REMOVE capability, which is why it belongs at the end and why a null
result here is a genuine kill for the morphology-set claim rather than a disappointment.

What makes it runnable
----------------------
SONIC is installed and working on this machine (its own eval reports success rate 1.0 and
mpjpe_g 92.9 mm over 64 envs), and `scene2motion.sonic_export` converts ARDY qpos into its
motion-library format. The one fact that makes the conversion safe is that ARDY's qpos[7:36] is
in the SAME joint order as SONIC's 29-DOF G1 -- verified name by name -- and `check_joint_order`
re-asserts it at export time, because a silent reordering would track a plausible but different
motion and every number below would be wrong in a way no assertion catches.

Design
------
One straight walk, no obstacle, five requested bodies plus a neutral control, N per-sample seeds
each. Every clip is exported into one motion-library pickle, SONIC evaluates it, and the
per-motion metrics come back keyed by the clip name so a tracking outcome can be attributed to
the body that was requested.

The comparison that matters is NOT the absolute tracking error -- it is whether an ADAPTED body
tracks worse than the neutral one generated from the same seed. A duck that tracks as well as a
neutral walk is a capability the robot actually has; one that only tracks at twice the error is
a kinematic fiction the planner should not be spending its budget on.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import BUILDERS  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    sonic_state_hydra_overrides,
    sonic_state_subprocess_env,
)

PROMPT = "A person walks forward."
SPEED = 0.9
# The editable install (`__editable__.gear_sonic-0.1.0.pth`) points at the `lucid` checkout, so
# `gear_sonic` resolves there whatever directory we launch from.  Running out of a DIFFERENT
# checkout mixes the two and produces a cross-tree ImportError.
SONIC = Path("/home/linjiw/lucid/GR00T-WholeBodyControl")
SONIC_PY = Path("/home/linjiw/isaaclab-install/env_isaaclab/bin/python")
CKPT = SONIC / "sonic_release" / "last.pt"

# (label, dip, lift).  Tuck is absent: EXP-006 measured it firing on 0.000 of clips at every
# inference setting and EXP-008 capped arm narrowing at 0.58 sigma, so spending tracker budget
# on it would be measuring a channel already known to carry nothing.
REQUESTS = [("neutral", 0.00, 0.00),
            ("duck-shallow", 0.20, 0.00),
            ("duck", 0.35, 0.00),
            ("duck-deep", 0.50, 0.00),
            ("lift", 0.00, 0.35),
            ("duck+lift", 0.35, 0.35)]


def build_clips(runner, body, args) -> dict[str, np.ndarray]:
    fps = runner.fps
    T = int(args.duration * fps)
    z = np.linspace(0.0, SPEED * args.duration, T)
    root_xz = np.stack([np.zeros(T), z], -1)
    heading = np.zeros(T)
    seeds = list(range(900, 900 + args.n_seeds))
    nominal_pelvis = 0.78
    clips: dict[str, np.ndarray] = {}

    ref = runner.generate([PROMPT], [ConstraintSpec(root_xz=root_xz, heading=heading,
                                                    root_y=np.full(T, nominal_pelvis))],
                          T, args.diffusion_steps, seeds=[seeds[0]])[0]
    nom_j = np.asarray(ref["posed_joints"])
    nom_r = np.asarray(ref["smooth_root_pos"])
    legs = [i for i, n in enumerate(runner.joint_names)
            if any(k in n for k in ("knee", "ankle", "toe"))]
    step = max(1, int(0.12 * fps))
    frames = np.arange(int(0.35 * T), int(0.65 * T), step)

    for label, dip, lift in REQUESTS:
        root_y = np.full(T, nominal_pelvis - dip)
        pos_frames = pos_joints = pos_targets = None
        if lift > 0.0:
            joints = np.array(legs)
            off = nom_j[frames][:, joints, :] - nom_r[frames][:, None, :]
            pos_targets = np.stack([root_xz[frames][:, None, 0] + off[:, :, 0],
                                    nom_j[frames][:, joints, 1] + lift,
                                    root_xz[frames][:, None, 1] + off[:, :, 2]], -1)
            pos_frames, pos_joints = frames, joints
        spec = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                              pos_frames=pos_frames, pos_joints=pos_joints,
                              pos_targets=pos_targets)
        outs = runner.generate([PROMPT] * len(seeds), [spec] * len(seeds), T,
                               args.diffusion_steps, seeds=seeds)
        for s, o in enumerate(outs):
            clips[f"{label}__s{seeds[s]}"] = runner.to_qpos(o)
        print(f"  {label:14s} {len(outs)} clips", flush=True)
    return clips


def sonic_env() -> dict:
    """The environment `isaaclab-install/env.sh` documents, propagated to the subprocess.

    Omniverse Kit prompts `Do you accept the EULA? (Yes/No):` on stdin and a captured
    subprocess has none, so it dies with `EOF when reading a line` six seconds in.  The machine
    owner accepted the EULA on 2026-08-28 and recorded it as OMNI_KIT_ACCEPT_EULA in env.sh;
    this reuses that acceptance rather than answering a licence prompt on their behalf.
    """
    e = dict(os.environ)
    e["OMNI_KIT_ACCEPT_EULA"] = "YES"
    e["ISAACLAB_PATH"] = "/home/linjiw/isaaclab-install/IsaacLab"
    e.setdefault("PYTHONUNBUFFERED", "1")
    # SONIC runs with its own checkout as cwd.  Put this repository on PYTHONPATH so Hydra can
    # instantiate Scene2Motion's achieved-state callback without modifying the SONIC checkout.
    return sonic_state_subprocess_env(e)


def run_sonic(pkl: Path, out_dir: Path, num_envs: int, timeout_s: int) -> tuple[int, str]:
    # ABSOLUTE paths: the subprocess runs with cwd=SONIC, so a relative motion_file resolves
    # against the wrong directory and `load_data` falls through its isfile() branch into the
    # directory assertion, which reports the path as if the FORMAT were wrong rather than the
    # location.
    pkl = pkl.resolve()
    out_dir = out_dir.resolve()
    # `-m`, not a script path.  Invoking `gear_sonic/eval_agent_trl.py` directly puts
    # `gear_sonic/` itself on sys.path[0], where its local `trl` subpackage shadows the
    # installed trl 0.28.0 -- and `gear_sonic/trl/trainer/ppo_trainer.py` does
    # `from trl import models`, which then resolves to itself and fails.  Running as a module
    # puts the REPO ROOT on the path instead, so both packages resolve as intended.
    cmd = [str(SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl",
           f"+checkpoint={CKPT}", "+headless=True", "++eval_callbacks=im_eval",
           "++run_eval_loop=False", f"++num_envs={num_envs}",
           f"++eval_output_dir={out_dir}",
           "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
           "+manager_env/terminations=tracking/eval",
           f"+manager_env.commands.motion.motion_lib_cfg.motion_file={pkl}",
           f"+log_keys={pkl.stem}",
           *sonic_state_hydra_overrides()]
    print("  " + " ".join(cmd[:4]) + " ...", flush=True)
    p = subprocess.run(cmd, cwd=SONIC, capture_output=True, text=True, timeout=timeout_s,
                       env=sonic_env(), stdin=subprocess.DEVNULL)
    return p.returncode, (p.stdout + "\n" + p.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp011")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--diffusion_steps", type=int, default=5)   # EXP-006: 5 beats 10
    ap.add_argument("--num_envs", type=int, default=24)
    ap.add_argument("--timeout_s", type=int, default=2400)
    ap.add_argument("--skip_generate", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "ardy_bodies.pkl"
    t0 = time.time()

    if not CKPT.exists():
        raise SystemExit(f"SONIC checkpoint not found at {CKPT}")

    if not args.skip_generate or not pkl.exists():
        runner = ArdyRunner(cache_path="outputs/text_cache.npz")
        body = G1Body(BUILDERS["overhead_beam"](3.0, 0))
        clips = build_clips(runner, body, args)
        # check_joint_order runs inside; a reordering raises rather than tracking the wrong
        # motion silently.
        # ONE PICKLE PER REQUESTED BODY.  The headline metric summary remains AGGREGATE (the
        # Scene2Motion callback additionally writes each achieved qpos), so a single mixed run
        # would obscure whether a ducked clip tracked worse than a neutral one.  Evaluating each
        # body separately makes each run's aggregate that body's number.
        for label, _, _ in REQUESTS:
            sub = {k: v for k, v in clips.items() if k.startswith(label + "__")}
            write_motion_pkl(sub, out / f"{label}.pkl", fps=int(round(runner.fps)),
                             mj_model=body.model)
        print(f"exported {len(clips)} clips into {len(REQUESTS)} per-body pickles "
              f"({time.time()-t0:.0f}s)", flush=True)
        del runner

    results = {}
    for label, dip, lift in REQUESTS:
        f = out / f"{label}.pkl"
        if not f.exists():
            print(f"  {label}: no pickle, skipped", flush=True)
            continue
        rc, log = run_sonic(f, out / label, args.n_seeds, args.timeout_s)
        (out / f"sonic_{label}.log").write_text(log)
        m = {}
        for line in log.splitlines():
            for key, name in (("Success Rate:", "success_rate"),
                              ("Progress Rate:", "progress_rate")):
                if line.startswith(key):
                    m[name] = float(line.split(":")[1])
            if line.startswith("All:"):
                for fld in ("mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist"):
                    hit = [t for t in line.split("\t") if t.strip().startswith(fld + ":")]
                    if hit:
                        m[fld] = float(hit[0].split(":")[1])
        m["returncode"] = rc
        results[label] = m
        print(f"  {label:14s} rc {rc}  success {m.get('success_rate', float('nan')):.3f}  "
              f"mpjpe_l {m.get('mpjpe_l', float('nan')):7.1f} mm  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'requested body':16s} {'success':>8s} {'progress':>9s} {'mpjpe_l':>9s} "
          f"{'mpjpe_pa':>9s} {'accel':>8s}   vs neutral")
    print("-" * 78)
    base = results.get("neutral", {})
    for label, _, _ in REQUESTS:
        m = results.get(label)
        if not m:
            continue
        ds = (m.get("success_rate", float("nan")) - base.get("success_rate", float("nan")))
        print(f"{label:16s} {m.get('success_rate', float('nan')):8.3f} "
              f"{m.get('progress_rate', float('nan')):9.3f} "
              f"{m.get('mpjpe_l', float('nan')):9.1f} {m.get('mpjpe_pa', float('nan')):9.1f} "
              f"{m.get('accel_dist', float('nan')):8.2f}   {ds:+.3f}")
    print("\n  The comparison that matters is the last column, not the absolute rate: an "
          "adapted body\n  that tracks as well as a neutral walk from the same seeds is a "
          "capability the robot has;\n  one that only tracks at half the rate is a kinematic "
          "fiction the planner should not spend\n  its budget on.  mpjpe_l is posture error; "
          "mpjpe_g is dominated by accumulated root drift\n  over a 7 m walk and is not the "
          "discriminating quantity here.")
    json.dump({"experiment": "exp011_tracked_addressability",
               "n_seeds": args.n_seeds, "diffusion_steps": args.diffusion_steps,
               "results": results, "wall_clock_s": round(time.time() - t0, 1)},
              open(out / "receipt.json", "w"), indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
