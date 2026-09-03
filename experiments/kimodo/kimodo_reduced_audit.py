#!/home/linjiw/kimodo/.venv/bin/python
"""Kimodo-G1 reduced capability audit: the same clips, counted four ways, on a second prior.

One paper-table row replicating scene2motion's naive-vs-calibrated counting result
(experiments/audit_delta.py) on Kimodo-G1-RP-v1 instead of ARDY. The audit counts BODY
modes, not routes, so a single straight 6 s route at 0.9 m/s carries the whole battery.

Run (KIMODO venv, once the GPU is free; DO NOT run while the GPU is busy):

    cd /tmp/claude-1000/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c/scratchpad/kimodo
    /home/linjiw/kimodo/.venv/bin/python kimodo_reduced_audit.py --steps 100

Cheap checks (no GPU, no model load):

    /home/linjiw/kimodo/.venv/bin/python -c "import kimodo_reduced_audit; print('import ok')"
    /home/linjiw/kimodo/.venv/bin/python kimodo_reduced_audit.py --selftest

Budget: 14 six-clip generate batches = 12 programs x 6 seeds + 6 matched controls
+ 6 null-arm controls = 84 clips, generated in B=6 batches (T=180 frames, 100 steps).
At the indoor_nav reference rate (~3.1 s/clip at 100 steps) expect ~5-9 min of GPU time
plus ~30 s model load. Well under the ~200-clip / 15-min budget; --steps scales it.

Design (mirror of scene2motion's audit, scoped down)
----------------------------------------------------
* Battery of 12 programs, each a ConstraintSpec DELTA against a neutral control on the
  same route (straight +Z, heading 0, root_y 0.78 = MODE_BY_NAME["stand"].pelvis_y):
    duck ladder     root_y dips 0.10 / 0.20 / 0.30 / 0.40 m over a mid-route window
    lateral tuck    arm global_joints_positions offsets shrunk by 0.3 / 0.6, both arms
                    (the 8 arm joints and the (1 - tuck) shrink of
                     scene2motion/planner.py:501 _limb_targets)
    leg lift        left-leg position targets +0.08 / +0.20 m on airborne frames
                    (airborne gate from the reference clip's foot_contacts, as
                     _limb_targets does; per-side gating deliberately omitted)
    rotation        arm-chain global rotations rolled +/-30 deg about the forward axis,
                    authored per seed from that seed's OWN control clip's
                    global_rot_mats (experiments/exp008_rotation_channel.py:211-229)
    combos          duck0.20+tuck0.60 and duck0.30+lift0.20, with limb heights shifted
                    by the commanded pelvis displacement (the _limb_targets coherence
                    fix, planner.py:556-571)
* Seeds: 6 paired seeds (100..105) shared by every program AND its matched control --
  matched pairs via KimodoRunner's _per_sample_noise -- plus a NULL arm: the neutral
  program on 6 MORE disjoint seeds (200..205) to calibrate the per-channel q99 noise
  threshold (null clip i is paired against control clip i).
* Descriptors: scene2motion.morphology is reused wherever it is mujoco-free
  (CHANNELS, PHYSICAL_FLOOR, Interaction, matched_delta, active_set, seed_statistics,
  stability, whitener, d_morph). raw_descriptor/envelope_series COULD NOT be reused:
  they require G1Body (scene2motion/robot.py:133), which needs mujoco -- not installed
  in the kimodo venv -- and ARDY's g1.xml under /home/linjiw/ardy. The minimal
  descriptor math is therefore VENDORED here (_descriptor/_envelope_series below) on
  Kimodo's native posed_joints (joint POINTS, y-up) instead of MuJoCo collision
  geometry. Constant geometry offsets (head capsule, primitive radii) cancel in the
  paired deltas the audit counts, but absolute tops/widths are NOT comparable to
  scene2motion's MuJoCo numbers -- only deltas are.
* Validity is likewise vendored (no collision scene here): finite qpos, mean root-path
  tracking error < 0.20 m, net forward progress within 1.0 m of the request, pelvis
  height within [0.30, 1.10] m. scene2motion's rule was goal-reached AND
  collision-free (exp005f_morphology_noise.py:151-156); this is the closest
  scene-free analogue and is recorded in the receipt.
* A neutral rung is recorded as literal zero deltas (exp005f's (0,0,0) ladder combo
  regenerates the control at the same seeds; under v2 per-sample noise that clip is
  bit-identical to the control), so the "no adaptation" signature participates in every
  counting row as it does in the ARDY audit, at zero clip cost.
* Counting: receipt.json carries the four rows computed with EXACTLY the rules of
  experiments/audit_delta.py (bits() copied verbatim from audit_delta.py:31-33):
    1 seed, any change > 1 mm      th = 0.001, first seed only
    1 seed, round 1 cm threshold   th = 0.01, first seed only
    1 seed, 1 cm, drop never-valid th = 0.01, rows with valid_frac > 0 only
    6-seed paired, q99, stable     valid_frac > 0.5, per-seed active bits against the
                                   null-calibrated q99, modal signature kept when
                                   stability >= 0.8
  plus per-program per-channel effect (mean delta) and effect/sigma.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SCENE2MOTION_ROOT = os.environ.get("SCENE2MOTION_ROOT", "/home/linjiw/scene2motion")
sys.path.insert(0, SCENE2MOTION_ROOT)

# Pure-numpy pieces of the scene2motion audit -- verified to import in the kimodo venv
# without mujoco (morphology.py's module level needs only numpy; scipy is used inside
# raw_descriptor, which we do not call).
from scene2motion.morphology import (CHANNELS, Interaction, active_set,  # noqa: E402
                                     matched_delta, seed_statistics, stability,
                                     whitener)

PROMPT = "A person walks forward at a steady pace."   # verified in kimodo's 300-prompt cache
SPEED_DEFAULT = 0.9
SECONDS_DEFAULT = 6.0
STAND_PELVIS = 0.78            # MODE_BY_NAME["stand"].pelvis_y (scene2motion/planner.py:64)
WINDOW = (0.35, 0.65)          # mid-route adaptation window, fraction of the clip
RAMP_S = 0.4                   # ease dense channels in/out of the window
TARGET_STEP_S = 0.12           # ~8 sparse targets/s, as _limb_targets (planner.py:551)
SEEDS_PAIRED = list(range(100, 106))
SEEDS_NULL = list(range(200, 206))
KIMODO_CACHE = "/home/linjiw/kimodo/data/indoor_nav_1k/text_cache.npz"

ARM_TUCK_JOINTS = [f"{s}_{j}" for s in ("left", "right")
                   for j in ("shoulder_roll_skel", "elbow_skel", "wrist_roll_skel",
                             "hand_roll_skel")]            # planner.py:539-542
LEFT_LEG_JOINTS = [f"left_{j}" for j in ("knee_skel", "ankle_pitch_skel",
                                          "ankle_roll_skel", "toe_base")]  # planner.py:532


def bits(d: np.ndarray, th: np.ndarray) -> tuple:
    """Verbatim from scene2motion/experiments/audit_delta.py:31-33."""
    on = lambda i: bool(d[i] > th[i])                       # noqa: E731
    return (on(0), on(1) or on(2), on(3), on(4), on(7))


def arm_chain(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) indices distal to each shoulder (exp008_rotation_channel.py:80-87)."""
    def side(pre):
        return np.array([i for i, n in enumerate(joint_names)
                         if n.startswith(pre) and any(k in n for k in
                                                      ("shoulder", "elbow", "wrist", "hand"))])
    return side("left_"), side("right_")


def axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix (exp008_rotation_channel.py:89-94)."""
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def window_ramp(T: int, fps: float) -> np.ndarray:
    """0->1->0 profile: 1 inside WINDOW, cosine ramps of RAMP_S seconds at the edges."""
    lo, hi = int(WINDOW[0] * T), int(WINDOW[1] * T)
    r = max(1, int(RAMP_S * fps))
    w = np.zeros(T)
    w[lo:hi] = 1.0
    for k in range(r):
        c = 0.5 * (1 - np.cos(np.pi * (k + 1) / (r + 1)))
        if lo - r + k >= 0:
            w[lo - r + k] = max(w[lo - r + k], c)
        if hi + r - 1 - k < T:
            w[hi + r - 1 - k] = max(w[hi + r - 1 - k], c)
    return w


def window_frames(T: int, fps: float) -> np.ndarray:
    step = max(1, int(TARGET_STEP_S * fps))
    f = np.arange(int(WINDOW[0] * T), int(WINDOW[1] * T), step)
    return f[(f >= 0) & (f < T)]


# ---------------------------------------------------------------------------------------
# Vendored descriptor math (mujoco-free; see module docstring for why).
# Kimodo native frame: y up, route along +z. Heading angle theta has 0 == facing +Z
# (kimodo/motion_rep/feature_utils.py:112), so the hip line r-l runs along (-cos t, 0, sin t)
# and the body's own lateral axis is that direction; extents are measured along it.
# ---------------------------------------------------------------------------------------

def _lateral_axis(theta: np.ndarray) -> np.ndarray:
    return np.stack([-np.cos(theta), np.zeros_like(theta), np.sin(theta)], -1)


def _heading_angle(sample: dict) -> np.ndarray:
    h = np.asarray(sample["global_root_heading"])          # (T, 2) = (cos, sin)
    return np.arctan2(h[:, 1], h[:, 0])


def _envelope_series(sample: dict) -> np.ndarray:
    """(T, 3) per-frame (top, left extent, right extent) from joint points.

    Same shape/meaning as morphology.envelope_series so matched_delta's timing channels
    work unchanged; measured on the robot's own lateral axis per frame (the audit's
    "heading" mode), from posed joints rather than collision primitives.
    """
    pj = np.asarray(sample["posed_joints"])                # (T, J, 3), y-up
    root = np.asarray(sample["root_positions"])            # (T, 3)
    n = _lateral_axis(_heading_angle(sample))              # (T, 3)
    rel = pj - root[:, None, :]
    proj = np.einsum("tjk,tk->tj", rel, n)
    top = pj[..., 1].max(axis=1)
    return np.stack([top, proj.max(axis=1), -proj.min(axis=1)], -1)


def _descriptor(sample: dict, joint_names: list[str], mask: np.ndarray) -> dict:
    """Vendored analogue of morphology.raw_descriptor (morphology.py:114) on joint points."""
    env = _envelope_series(sample)
    pj = np.asarray(sample["posed_joints"])
    if mask.sum() < 3:
        mask = np.ones(len(pj), bool)
    idx = {n: i for i, n in enumerate(joint_names)}
    feet = {s: [idx[f"{s}_ankle_roll_skel"], idx[f"{s}_toe_base"]] for s in ("left", "right")}
    m = np.flatnonzero(mask)
    theta = np.unwrap(_heading_angle(sample))
    return {"top": float(env[m, 0].max()),
            "w_left": float(env[m, 1].max()),
            "w_right": float(env[m, 2].max()),
            "foot_left": float(pj[m][:, feet["left"], 1].max()),
            "foot_right": float(pj[m][:, feet["right"], 1].max()),
            "yaw": float(theta[m].mean())}


def _travel_col(sample: dict) -> np.ndarray:
    """(T, 1) pseudo-qpos whose column 0 is the travel coordinate (native z).

    morphology.Interaction.mask and matched_delta only ever read qpos[:, 0]
    (morphology.py:54-55, :169), so this is all they need.
    """
    return np.asarray(sample["root_positions"])[:, [2]]


def _validity(qpos: np.ndarray, root: np.ndarray, spec_xz: np.ndarray,
              target_travel: float) -> dict:
    """Vendored, scene-free validity (see module docstring for the deviation)."""
    finite = bool(np.isfinite(qpos).all())
    err = np.linalg.norm(root[:, [0, 2]] - spec_xz, axis=1)
    tracked = bool(err.mean() < 0.20)
    travel = float(root[-1, 2] - root[0, 2])
    progressed = bool(abs(travel - target_travel) < 1.0)
    h = root[:, 1]
    upright = bool(0.30 < h.min() and h.max() < 1.10)
    return {"valid": finite and tracked and progressed and upright,
            "finite": finite, "track_err_mean_m": float(err.mean()),
            "travel_m": travel, "pelvis_min_m": float(h.min()),
            "pelvis_max_m": float(h.max())}


# ---------------------------------------------------------------------------------------
# Program battery: 12 ConstraintSpec builders, each a delta against the neutral control.
# ---------------------------------------------------------------------------------------

def build_battery(ConstraintSpec, T: int, fps: float, speed: float,
                  joint_names: list[str], ctrl_ref: dict, ctrl_outputs: list[dict]):
    """[(name, [spec per seed])] plus the neutral spec. Specs per seed differ only for the
    rotation programs, which are authored from each seed's own control clip (exp008)."""
    t = np.arange(T) / fps
    root_xz = np.stack([np.zeros(T), speed * t], -1)
    heading = np.zeros(T)
    root_y0 = np.full(T, STAND_PELVIS)
    ramp = window_ramp(T, fps)
    frames = window_frames(T, fps)
    idx = {n: i for i, n in enumerate(joint_names)}

    neutral = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y0)

    nom_j = np.asarray(ctrl_ref["posed_joints"])           # (T, J, 3)
    nom_r = np.asarray(ctrl_ref["smooth_root_pos"])        # (T, 3)
    contacts = np.asarray(ctrl_ref["foot_contacts"])       # (T, 4) Lheel Ltoe Rheel Rtoe

    def limb_spec(tuck: float, lift: float, dip: float) -> "ConstraintSpec":
        """_limb_targets (scene2motion/planner.py:501) re-authored for this battery."""
        root_y = root_y0 - dip * ramp
        joints, per_frame_dy = [], {}
        if tuck > 0:
            joints += [idx[n] for n in ARM_TUCK_JOINTS]
        if lift > 0:
            legs = [idx[n] for n in LEFT_LEG_JOINTS]
            air = ~contacts[:, 0:2].any(-1)                # left foot airborne
            joints += legs
            for f in frames:
                if air[min(f, len(air) - 1)]:
                    per_frame_dy[int(f)] = lift
        joints = np.array(sorted(set(joints)))
        off = nom_j[frames][:, joints, :] - nom_r[frames][:, None, :]
        shrink = np.ones((len(frames), len(joints)))
        arm_set = {idx[n] for n in ARM_TUCK_JOINTS}
        arm_cols = np.array([i for i, j in enumerate(joints) if j in arm_set], int)
        if tuck > 0 and len(arm_cols):
            shrink[:, arm_cols] = (1.0 - tuck * ramp[frames])[:, None]
        height = nom_j[frames][:, joints, 1].copy()
        # coherence fix (planner.py:556-571): limbs ride down with the COMMANDED pelvis
        height += (root_y[frames] - STAND_PELVIS)[:, None]
        leg_set = {idx[n] for n in LEFT_LEG_JOINTS}
        for i, f in enumerate(frames):
            if int(f) in per_frame_dy:
                for c, j in enumerate(joints):
                    if j in leg_set:
                        height[i, c] += per_frame_dy[int(f)]
        targets = np.stack([root_xz[frames][:, None, 0] + off[:, :, 0] * shrink,
                            height,
                            root_xz[frames][:, None, 1] + off[:, :, 2] * shrink], -1)
        return ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                              pos_frames=frames, pos_joints=joints, pos_targets=targets)

    def rot_specs(sign: float, deg: float) -> list:
        """exp008_rotation_channel.py:211-229: rigid arm-chain roll about forward (+Z)."""
        left, right = arm_chain(joint_names)
        joints = np.concatenate([left, right])
        fwd = np.array([0.0, 0.0, 1.0])
        RL = axis_rotation(fwd, +sign * np.deg2rad(deg))
        RR = axis_rotation(fwd, -sign * np.deg2rad(deg))
        specs = []
        for o in ctrl_outputs:
            grm = np.asarray(o["global_rot_mats"])         # (T, J, 3, 3)
            tgt = np.empty((len(frames), len(joints), 3, 3))
            lset = set(left.tolist())
            for i, f in enumerate(frames):
                for k, j in enumerate(joints):
                    R = RL if j in lset else RR
                    tgt[i, k] = R @ grm[f, j]
            specs.append(ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y0,
                                        rot_frames=frames, rot_joints=joints,
                                        rot_targets=tgt))
        return specs

    n = len(ctrl_outputs)
    battery = []
    for dip in (0.10, 0.20, 0.30, 0.40):
        s = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y0 - dip * ramp)
        battery.append((f"duck_{dip:.2f}", [s] * n))
    for tk in (0.30, 0.60):
        battery.append((f"tuck_{tk:.2f}", [limb_spec(tk, 0.0, 0.0)] * n))
    for lf in (0.08, 0.20):
        battery.append((f"lift_{lf:.2f}", [limb_spec(0.0, lf, 0.0)] * n))
    battery.append(("rot_+30", rot_specs(+1.0, 30.0)))
    battery.append(("rot_-30", rot_specs(-1.0, 30.0)))
    battery.append(("duck_0.20+tuck_0.60", [limb_spec(0.60, 0.0, 0.20)] * n))
    battery.append(("duck_0.30+lift_0.20", [limb_spec(0.0, 0.20, 0.30)] * n))
    return neutral, battery, root_xz


# ---------------------------------------------------------------------------------------
# Counting: the four rows of experiments/audit_delta.py:47-63, single-scene version.
# ---------------------------------------------------------------------------------------

def count_rows(rows: list[dict], q99: np.ndarray, min_stability: float) -> dict:
    def count(mode: str) -> float:
        if mode == "calibrated":
            sigs = set()
            for r in rows:
                if r["valid_frac"] <= 0.5:
                    continue
                p = [bits(np.array(x), q99) for x in r["deltas"]]
                c = Counter(p).most_common(1)[0]
                if c[1] / len(p) >= min_stability:
                    sigs.add(c[0])
        else:
            th = np.full(len(CHANNELS), 0.001 if mode == "any" else 0.01)
            R = rows if mode != "valid" else [r for r in rows if r["valid_frac"] > 0]
            sigs = {bits(np.array(r["deltas"][0]), th) for r in R}
        return float(len(sigs))

    return {"1 seed, any change > 1 mm": count("any"),
            "1 seed, round 1 cm threshold": count("cm"),
            "1 seed, 1 cm, drop clips that never validate": count("valid"),
            f"6 seeds, paired, q99-calibrated, stability >= {min_stability}":
                count("calibrated")}


# ---------------------------------------------------------------------------------------

def selftest() -> int:
    """Exercise descriptors + counting on synthetic data. No kimodo, no GPU, no model."""
    rng = np.random.default_rng(0)
    T, J = 180, 34
    names = [f"j{i}" for i in range(J)]
    names[0] = "pelvis_skel"
    for i, n in enumerate(("left_ankle_roll_skel", "left_toe_base",
                           "right_ankle_roll_skel", "right_toe_base")):
        names[1 + i] = n

    def fake(dip=0.0, tuck=0.0):
        z = np.linspace(0, 5.4, T)
        pj = rng.normal(0, 0.02, (T, J, 3))
        pj[..., 1] += 0.8 - dip
        pj[..., 2] += z[:, None]
        pj[:, J // 2:, 0] += 0.25 * (1 - tuck)             # "arms" out to one side
        pj[:, 1:5, 1] = 0.05                                # feet near the floor
        root = pj[:, 0].copy()
        return {"posed_joints": pj, "root_positions": root,
                "global_root_heading": np.stack([np.ones(T), np.zeros(T)], -1),
                "smooth_root_pos": root}

    inter = Interaction(2.7)
    mask = inter.mask(_travel_col(fake()))
    assert mask.sum() > 3
    ctrl = fake()
    dc = _descriptor(ctrl, names, mask)
    da = _descriptor(fake(dip=0.3), names, mask)
    delta = matched_delta(da, dc, _travel_col(fake(dip=0.3)), _travel_col(ctrl), inter, 30.0,
                          env_adapted=_envelope_series(fake(dip=0.3)),
                          env_control=_envelope_series(ctrl))
    assert delta.shape == (len(CHANNELS),)
    assert delta[0] > 0.2, f"duck delta should register on dh_top, got {delta[0]}"

    # Counting rules on a synthetic ledger: a neutral rung (exact zeros, as the real one
    # is under v2 per-sample noise), one real duck, one pure-noise program. The calibrated
    # row must count exactly TWO modes -- "no adaptation" and "duck" -- because the noise
    # program collapses onto the neutral signature instead of minting a new one; the naive
    # 1-seed rows credit the noise with a mode of its own.
    q99 = np.full(len(CHANNELS), 0.05)
    duck = np.tile([0.3, 0, 0, 0, 0, 0, 0, 0], (6, 1)) + rng.normal(0, 0.01, (6, 8))
    noise = rng.normal(0, 0.01, (6, 8))
    rows = [{"name": "neutral", "deltas": np.zeros((6, 8)).tolist(), "valid_frac": 1.0},
            {"name": "duck", "deltas": duck.tolist(), "valid_frac": 1.0},
            {"name": "noise", "deltas": noise.tolist(), "valid_frac": 1.0}]
    noise_sigs = [bits(d, q99) for d in noise]
    assert Counter(noise_sigs).most_common(1)[0][0] == (False,) * 5, \
        "q99 threshold failed to silence a pure-noise program"
    duck_sigs = [bits(d, q99) for d in duck]
    assert Counter(duck_sigs).most_common(1)[0][0] == (True, False, False, False, False)
    c = count_rows(rows, q99, 0.8)
    assert c["1 seed, any change > 1 mm"] >= 3.0            # noise counted as its own mode
    assert c[[k for k in c if k.startswith("6 seeds")][0]] == 2.0  # neutral + duck only
    print("selftest OK:", json.dumps(c, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100,
                    help="DDIM denoising steps (indoor_nav reference used 100)")
    ap.add_argument("--seconds", type=float, default=SECONDS_DEFAULT)
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT)
    ap.add_argument("--min_stability", type=float, default=0.8)
    ap.add_argument("--cache", default=KIMODO_CACHE)
    ap.add_argument("--out", default=str(HERE / "audit_out"))
    ap.add_argument("--selftest", action="store_true",
                    help="run descriptor+counting checks on synthetic data and exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    os.environ.setdefault("LOCAL_CACHE", "true")
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from kimodo_runner import (KimodoConstraintSet, KimodoRunner, channel_usage,
                               load_constraint_spec_class)
    import torch
    ConstraintSpec = load_constraint_spec_class()

    runner = KimodoRunner("Kimodo-G1-RP-v1", device=args.device,
                          cache_path=args.cache, text_encoder=False)
    fps = runner.fps
    T = int(round(args.seconds * fps))
    target_travel = args.speed * args.seconds
    inter = Interaction(0.5 * target_travel)               # mid-route window, metres
    print(f"model={runner.model_name} fps={fps} T={T} steps={args.steps} "
          f"clips={12 * len(SEEDS_PAIRED) + len(SEEDS_PAIRED) + len(SEEDS_NULL)}")

    def gen(specs, seeds):
        return runner.generate([PROMPT] * len(seeds), specs, T, args.steps, seeds=seeds)

    # -- matched controls (paired seeds) and the null arm (disjoint seeds) ---------------
    t_ar = np.arange(T) / fps
    neutral0 = ConstraintSpec(root_xz=np.stack([np.zeros(T), args.speed * t_ar], -1),
                              heading=np.zeros(T), root_y=np.full(T, STAND_PELVIS))
    ctrls = gen([neutral0] * len(SEEDS_PAIRED), SEEDS_PAIRED)
    nulls = gen([neutral0] * len(SEEDS_NULL), SEEDS_NULL)
    print(f"controls + null generated ({time.time() - t0:.0f}s)", flush=True)

    neutral, battery, spec_xz = build_battery(
        ConstraintSpec, T, fps, args.speed, runner.joint_names, ctrls[0], ctrls)

    # channel-usage guard (exp008's audit_spec): a request that never reaches the mask
    # would make every row below a free generation.
    for probe_name in ("tuck_0.60", "rot_+30"):
        spec = dict(battery)[probe_name][0]
        _, mask = runner.model.motion_rep.create_conditions_from_constraints_batched(
            [[KimodoConstraintSet(spec, runner.skeleton.root_idx, args.device)]],
            torch.tensor([T], device=args.device), to_normalize=True, device=args.device)
        u = {k: v for k, v in channel_usage(runner.model, mask).items() if v}
        print(f"  channels written by {probe_name}: {u}", flush=True)
        need = "local_joints_positions" if probe_name.startswith("tuck") else "global_rot_data"
        if not u.get(need):
            raise SystemExit(f"{probe_name} writes no {need}; constraint wiring is broken.")

    ctrl_d = [_descriptor(o, runner.joint_names, inter.mask(_travel_col(o)))
              for o in ctrls]
    ctrl_e = [_envelope_series(o) for o in ctrls]
    ctrl_q = [runner.to_qpos(o) for o in ctrls]

    def delta_against_control(o, s):
        d = _descriptor(o, runner.joint_names, inter.mask(_travel_col(o)))
        return matched_delta(d, ctrl_d[s], _travel_col(o), _travel_col(ctrls[s]),
                             inter, fps, env_adapted=_envelope_series(o),
                             env_control=ctrl_e[s])

    # -- q99 from the NULL arm ----------------------------------------------------------
    null_deltas = np.array([delta_against_control(o, s) for s, o in enumerate(nulls)])
    noise_q99 = np.percentile(np.abs(null_deltas), 99, axis=0)
    print("null q99 per channel:",
          {c: round(float(v), 4) for c, v in zip(CHANNELS, noise_q99)}, flush=True)

    # -- the battery ---------------------------------------------------------------------
    # Neutral rung first, at zero clip cost: exp005f's ladder includes the (0,0,0) combo,
    # which regenerates the control spec at the SAME seeds -- under v2 per-sample noise
    # streams that is bit-identical to the control, so its paired deltas are exactly zero.
    # Recording it as literal zeros keeps the "no adaptation" signature in every counting
    # row, as the ARDY audit has it, without spending 6 clips to generate the same motion.
    ctrl_checks = [_validity(ctrl_q[s], np.asarray(ctrls[s]["root_positions"]),
                             spec_xz, target_travel) for s in range(len(SEEDS_PAIRED))]
    Z = np.zeros((len(SEEDS_PAIRED), len(CHANNELS)))
    rows = [{"name": "neutral", "deltas": Z.tolist(), "mu": Z[0].tolist(),
             "sigma": Z[0].tolist(), "effect_over_sigma": Z[0].tolist(),
             "valid_frac": float(np.mean([c["valid"] for c in ctrl_checks])),
             "validity": ctrl_checks,
             "note": "identical clips to the matched control by construction"}]
    all_covs = []
    for name, specs in battery:
        outs = gen(specs, SEEDS_PAIRED)
        deltas, checks = [], []
        for s, o in enumerate(outs):
            deltas.append(delta_against_control(o, s))
            checks.append(_validity(runner.to_qpos(o), np.asarray(o["root_positions"]),
                                    spec_xz, target_travel))
        deltas = np.array(deltas)
        mu, cov = seed_statistics(deltas)
        all_covs.append(cov)
        sigma = deltas.std(axis=0, ddof=1)
        rows.append({"name": name, "deltas": deltas.tolist(), "mu": mu.tolist(),
                     "sigma": sigma.tolist(),
                     "effect_over_sigma": (np.abs(mu) / np.maximum(sigma, 1e-9)).tolist(),
                     "valid_frac": float(np.mean([c["valid"] for c in checks])),
                     "validity": checks})
        per_seed = [active_set(d, noise_q99) for d in deltas]
        print(f"  {name:22s} valid {rows[-1]['valid_frac']:.2f}  "
              f"stability {stability(per_seed):.2f}  "
              f"mu[dh,dwl,dwr,dzl,dzr,psi]="
              f"{[round(float(mu[i]), 3) for i in (0, 1, 2, 3, 4, 7)]}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # exp005f-style residual q99, reported alongside for comparability with the ARDY run
    resid = np.concatenate([np.abs(np.array(r["deltas"]) -
                                   np.array(r["deltas"]).mean(axis=0)) for r in rows])
    resid_q99 = np.percentile(resid, 99, axis=0)

    counts = count_rows(rows, noise_q99, args.min_stability)
    W = whitener(all_covs)

    receipt = {
        "experiment": "kimodo_reduced_audit",
        "model": runner.model_name, "fps": fps, "prompt": PROMPT,
        "route": {"seconds": args.seconds, "speed": args.speed, "T": T,
                  "root_y": STAND_PELVIS, "window": WINDOW},
        "diffusion_steps": args.steps,
        "seeds_paired": SEEDS_PAIRED, "seeds_null": SEEDS_NULL,
        "n_clips": len(SEEDS_PAIRED) * (len(battery) + 1) + len(SEEDS_NULL),
        "noise_stream_version": runner.noise_stream_version,
        "channels": list(CHANNELS),
        "seed_noise_q99": noise_q99.tolist(),                 # NULL-arm calibrated
        "seed_noise_q99_residual_style": resid_q99.tolist(),  # exp005f-style, for comparison
        "validity_rule": "finite qpos; mean root-xz tracking err < 0.20 m; net travel "
                         "within 1.0 m of request; pelvis in [0.30, 1.10] m "
                         "(vendored, scene-free; ARDY audit used goal+collision-free)",
        "descriptor_note": "vendored joint-point descriptors (no MuJoCo geometry); "
                           "paired deltas comparable, absolute extents are not",
        "counts": counts,
        "overstatement_kinematic":
            max(v for k, v in counts.items() if not k.startswith("6 seeds")) /
            max(next(v for k, v in counts.items() if k.startswith("6 seeds")), 1e-9),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(out / "receipt.json", "w") as fh:
        json.dump(receipt, fh, indent=2)
    with open(out / "per_program.json", "w") as fh:
        json.dump({"rows": rows}, fh)

    print(f"\ndistinct body modes, same clips, four countings ({runner.model_name}):")
    for k, v in counts.items():
        print(f"   {k:52s} {v:5.2f}")
    print(f"\noverstatement factor: {receipt['overstatement_kinematic']:.1f}x  "
          f"-> {out / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
