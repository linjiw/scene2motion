"""EXP-001: the capability envelope of the frozen ARDY-G1 motion prior.

Question
--------
Before asking whether a *learned* scene-conditioned planner is possible, ask what the
frozen prior can already be TOLD to do. Concretely: across ARDY's native conditioning
channels, what (overhead clearance, lateral clearance) envelopes can the G1 body reach,
how far can each channel be pushed before the motion falls apart, and what does each cost?

This decides the whole programme. If the prior's reachable envelope is narrow, a
scene->constraint model (V1) has nothing to aim at and we would have to retrain the prior.
If it is wide, V1 is the right first model and the prior stays frozen.

Design
------
Three channels are swept independently against a fixed straight 8 s path, so any change in
the body envelope is attributable to the channel alone:

  duck   root_y_pos dipped by d in [0, 0.5] m over a Gaussian window at mid-clip
  tuck   sparse joint-position targets pulling the arm chain's lateral offset to (1-s) of
         nominal, s in [0, 0.8]
  sidle  global_root_heading offset from the path tangent by phi in [0, 90] deg

Envelopes are measured with the robot's OWN MuJoCo collision primitives after exporting to
qpos, not from ARDY's 34-joint rig: G1's head reaches ~1.30 m while the highest ARDY joint
is ~1.07 m, so a joint-derived envelope is 23 cm too optimistic and would certify
head-first collisions as clear.

Reported per condition: achieved top height, half-width perpendicular to travel, path
tracking error, forward progress, foot-ground penetration and foot slip (the two cheap
physical-consistency signals available without a simulator).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402

PROMPT = "A person walks forward."
DURATION = 8.0
SPEED = 0.95
NOMINAL_PELVIS = 0.78
# The window over which the adaptation is requested, as a fraction of the clip.
CENTER, SIGMA = 0.50, 0.16
MEASURE = (0.42, 0.58)  # sub-window the envelope is measured over

ARM_JOINTS = ["left_shoulder_roll_skel", "left_elbow_skel", "left_hand_roll_skel",
              "right_shoulder_roll_skel", "right_elbow_skel", "right_hand_roll_skel"]

DUCKS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TUCKS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
SIDLES = [0, 15, 30, 45, 60, 75, 90]
SEEDS = [0, 1, 2]


def gaussian_window(T: int) -> np.ndarray:
    t = np.arange(T)
    return np.exp(-0.5 * ((t - T * CENTER) / (T * SIGMA)) ** 2)


def foot_slip(body: G1Body, qpos: np.ndarray, fps: float, thresh: float = 0.03) -> float:
    """Mean horizontal speed of foot pads that are within `thresh` of the ground.

    A planted foot that slides is the signature of a kinematically-authored motion that a
    physics tracker will not be able to follow. Cheap, and it correlates with the thing we
    ultimately care about.
    """
    m = body.model
    foot = [g for g in body.robot_geoms
            if "foot" in (body.geom_name[g] or "") or "ankle" in (body.geom_name[g] or "")]
    if not foot:
        return float("nan")
    pos = []
    for q in qpos:
        body.fk(q)
        pos.append(body.data.geom_xpos[foot].copy())
    pos = np.asarray(pos)                                   # (T, F, 3)
    vel = np.linalg.norm(np.diff(pos[:, :, :2], axis=0), axis=-1) * fps
    contact = (pos[:-1, :, 2] - m.geom_size[foot, 0][None, :]) < thresh
    return float(vel[contact].mean()) if contact.any() else float("nan")


def measure(body: G1Body, qpos: np.ndarray, spec: ConstraintSpec, fps: float,
            travel_normal: np.ndarray) -> dict:
    T = len(qpos)
    mid = slice(int(T * MEASURE[0]), int(T * MEASURE[1]))
    nom = slice(0, int(T * 0.15))
    tops = np.array([body.top_height(q) for q in qpos])
    hw = np.array([body.half_width(q, travel_normal) for q in qpos])
    rep = body.trajectory_report(qpos)
    # qpos is Z-up world: (x forward, y lateral). The spec is ARDY-frame (x lateral, z fwd).
    got = np.stack([qpos[:, 1], qpos[:, 0]], -1)
    err = np.linalg.norm(got - spec.root_xz, axis=-1)
    return {
        "top_nominal_m": float(tops[nom].max()),
        "top_adapted_m": float(tops[mid].max()),
        "overhead_gain_m": float(tops[nom].max() - tops[mid].max()),
        "halfwidth_nominal_m": float(hw[nom].max()),
        "halfwidth_adapted_m": float(hw[mid].max()),
        "lateral_gain_m": float(hw[nom].max() - hw[mid].max()),
        "pelvis_min_m": float(qpos[:, 2].min()),
        "path_err_mean_m": float(err.mean()),
        "path_err_max_m": float(err.max()),
        "travel_m": float(qpos[-1, 0] - qpos[0, 0]),
        "foot_floor_pen_max_m": rep["max_foot_floor_penetration_m"],
        "foot_slip_mps": foot_slip(body, qpos, fps),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp001")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DURATION * fps)
    w = gaussian_window(T)
    t = np.arange(T)
    straight_xz = np.stack([np.zeros(T), SPEED * t / fps], -1)
    body = G1Body(None)
    lateral = np.array([0.0, 1.0, 0.0])  # travel is +x, so the gap direction is world y

    # A nominal clip per seed supplies the reference arm positions the tuck condition
    # shrinks. Taking them from the model itself (rather than a hand-authored pose) keeps
    # the tuck request on the prior's own manifold.
    nominal = {}
    for s in SEEDS:
        spec = ConstraintSpec(root_xz=straight_xz, heading=np.zeros(T),
                              root_y=np.full(T, NOMINAL_PELVIS))
        nominal[s] = runner.generate([PROMPT], [spec], T, args.diffusion_steps, seed=s)[0]

    arm_idx = np.array([runner.joint_names.index(j) for j in ARM_JOINTS])
    rows = []

    def run_batch(conds: list[tuple[str, float, int, ConstraintSpec]]) -> None:
        for i in range(0, len(conds), 8):
            chunk = conds[i:i + 8]
            outs = runner.generate([PROMPT] * len(chunk), [c[3] for c in chunk], T,
                                   args.diffusion_steps, seed=chunk[0][2])
            for (channel, value, seed, spec), o in zip(chunk, outs):
                q = runner.to_qpos(o)
                r = measure(body, q, spec, fps, lateral)
                r.update(channel=channel, value=value, seed=seed)
                rows.append(r)
                np.save(out / f"qpos_{channel}_{value:g}_s{seed}.npy", q.astype(np.float32))

    # -- duck ---------------------------------------------------------------------------
    conds = []
    for s in SEEDS:
        for d in DUCKS:
            spec = ConstraintSpec(root_xz=straight_xz, heading=np.zeros(T),
                                  root_y=NOMINAL_PELVIS - d * w)
            conds.append(("duck", d, s, spec))
    run_batch(conds)
    print(f"duck: {len(conds)} conditions done ({time.time()-t_start:.0f}s)", flush=True)

    # -- tuck ---------------------------------------------------------------------------
    conds = []
    step = max(1, int(0.2 * fps))
    active = np.where(w > 0.35)[0][::step]
    for s in SEEDS:
        nom = nominal[s]
        for k in TUCKS:
            if k == 0.0:
                spec = ConstraintSpec(root_xz=straight_xz, heading=np.zeros(T),
                                      root_y=np.full(T, NOMINAL_PELVIS))
            else:
                off = nom["posed_joints"][active][:, arm_idx, :] - nom["smooth_root_pos"][active][:, None, :]
                tgt = np.stack([
                    straight_xz[active][:, None, 0] + off[:, :, 0] * (1.0 - k),
                    nom["posed_joints"][active][:, arm_idx, 1],
                    straight_xz[active][:, None, 1] + off[:, :, 2],
                ], -1)
                spec = ConstraintSpec(root_xz=straight_xz, heading=np.zeros(T),
                                      root_y=np.full(T, NOMINAL_PELVIS),
                                      pos_frames=active, pos_joints=arm_idx, pos_targets=tgt)
            conds.append(("tuck", k, s, spec))
    run_batch(conds)
    print(f"tuck: {len(conds)} conditions done ({time.time()-t_start:.0f}s)", flush=True)

    # -- sidle --------------------------------------------------------------------------
    # Heading is offset from the path tangent, so the robot keeps travelling +x while
    # facing progressively sideways. At 90 deg it is pure lateral stepping.
    conds = []
    for s in SEEDS:
        for phi in SIDLES:
            spec = ConstraintSpec(root_xz=straight_xz,
                                  heading=np.deg2rad(phi) * w,
                                  root_y=np.full(T, NOMINAL_PELVIS))
            conds.append(("sidle", float(phi), s, spec))
    run_batch(conds)
    print(f"sidle: {len(conds)} conditions done ({time.time()-t_start:.0f}s)", flush=True)

    # -- joint duck+tuck, to test whether the two channels compose --------------------
    conds = []
    for s in SEEDS:
        nom = nominal[s]
        for d, k in [(0.15, 0.5), (0.30, 0.5), (0.30, 0.7), (0.40, 0.7)]:
            off = nom["posed_joints"][active][:, arm_idx, :] - nom["smooth_root_pos"][active][:, None, :]
            ry = NOMINAL_PELVIS - d * w
            tgt = np.stack([
                straight_xz[active][:, None, 0] + off[:, :, 0] * (1.0 - k),
                nom["posed_joints"][active][:, arm_idx, 1] - d * w[active][:, None],
                straight_xz[active][:, None, 1] + off[:, :, 2],
            ], -1)
            spec = ConstraintSpec(root_xz=straight_xz, heading=np.zeros(T), root_y=ry,
                                  pos_frames=active, pos_joints=arm_idx, pos_targets=tgt)
            conds.append(("duck+tuck", d * 100 + k, s, spec))
    run_batch(conds)
    print(f"duck+tuck: {len(conds)} conditions done ({time.time()-t_start:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    receipt = {
        "experiment": "exp001_capability_envelope",
        "model": runner.model_name,
        "fps": fps, "duration_s": DURATION, "diffusion_steps": args.diffusion_steps,
        "prompt": PROMPT, "seeds": SEEDS, "n_conditions": len(rows),
        "ducks": DUCKS, "tucks": TUCKS, "sidles": SIDLES,
        "arm_joints": ARM_JOINTS,
        "measure_window": MEASURE,
        "git_ardy": subprocess.run(["git", "-C", "/home/linjiw/ardy", "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    with open(out / "receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    print(f"\nwrote {len(rows)} rows to {out}/rows.jsonl in {receipt['wall_clock_s']}s")


if __name__ == "__main__":
    main()
