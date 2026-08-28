"""EXP-001d: the body envelope as a calibrated FUNCTION of the adaptation, with coverage.

Why this replaces the mode table
--------------------------------
`outputs/body_modes.json` holds seven discrete modes whose envelopes are the WORST of three
seeds. Two things are wrong with that, and both were surfaced by review rather than by the
numbers looking odd:

 1. **It is not a bound.** Worst-of-3 is a 36.8 %-content tolerance interval at 95 %
    confidence — it says almost nothing about a new sample. The same nominal condition gave
    worst-of-3 half-widths of 0.281 m in EXP-001b and 0.380 m in EXP-001, a 10 cm spread
    attributable to nothing but which three draws were taken.

 2. **It is non-monotone in a way that blocks the constraint program.** The tabulated
    half-widths run 0.380 / 0.360 / 0.375 / 0.497 / 0.344 across increasing dip, so
    `duck_deep` is *wider* than standing. A* certifies a cell using the mode's half-width, so
    a program that emits a continuous dip lands between measured points and inherits no
    certificate at all. Either the program stays on the lattice, or the envelope becomes a
    function — and a 0.497 m entry that is 12 cm off its neighbours looks like one sample
    flinging its arms out, not like physics.

So: sweep the dip finely, take enough independent samples per level to compute a **split
conformal upper bound** rather than a worst-of-k, and report whether the non-monotonicity
survives. With n samples per level, the ceil((n+1)(1-alpha))-th order statistic is a valid
(1-alpha) upper bound for a fresh sample under exchangeability — so n = 20 gives a genuine
90 % bound (the 19th of 20), where n = 3 cannot give one at all.

The claim this licenses is narrow and worth stating exactly: *for a request drawn from this
distribution, the body's envelope exceeds the calibrated bound at most 10 % of the time.* It
is a statement about the ENVELOPE, not about collision-freeness, and the per-instance
guarantee remains the MuJoCo check.
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
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402

PROMPT = "A person walks forward."
DURATION, SPEED, NOMINAL_PELVIS = 8.0, 0.95, 0.78
CENTER, SIGMA = 0.50, 0.16
MEASURE = (0.42, 0.58)
N_SAMPLES = 20                      # -> a valid 90% conformal bound (19th order statistic)
DIPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
TUCKS = [0.0, 0.2, 0.4, 0.6, 0.85]
LATERAL = np.array([0.0, 1.0, 0.0])

ARM_JOINTS = ["left_shoulder_roll_skel", "left_elbow_skel", "left_wrist_roll_skel",
              "left_hand_roll_skel", "right_shoulder_roll_skel", "right_elbow_skel",
              "right_wrist_roll_skel", "right_hand_roll_skel"]


def conformal_upper(x: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal upper bound: the ceil((n+1)(1-alpha))-th order statistic.

    Valid for a fresh exchangeable sample. Returns +inf when n is too small for the level,
    which is the honest answer rather than silently degrading to the maximum.
    """
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(x[k - 1]) if k <= n else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp001d")
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    T = int(DURATION * fps)
    t = np.arange(T)
    w = np.exp(-0.5 * ((t - T * CENTER) / (T * SIGMA)) ** 2)
    path = np.stack([np.zeros(T), SPEED * t / fps], -1)
    body = G1Body(None)
    mid = slice(int(T * MEASURE[0]), int(T * MEASURE[1]))
    nom_slice = slice(0, int(T * 0.15))
    rows = []

    def measure(q):
        tops = np.array([body.top_height(x) for x in q])
        hws = np.array([body.half_width(x, LATERAL) for x in q])
        return tops, hws

    # A nominal clip per sample index supplies the arm positions the tuck condition shrinks.
    arm_idx = np.array([runner.joint_names.index(j) for j in ARM_JOINTS])
    step = max(1, int(0.2 * fps))
    active = np.where(w > 0.35)[0][::step]
    base_spec = ConstraintSpec(root_xz=path, heading=np.zeros(T),
                               root_y=np.full(T, NOMINAL_PELVIS), first_heading=0.0)
    noms = []
    for i in range(0, N_SAMPLES, 8):
        k = min(8, N_SAMPLES - i)
        noms += runner.generate([PROMPT] * k, [base_spec] * k, T, args.diffusion_steps,
                                seeds=list(range(i, i + k)))

    def run(channel, value, specs):
        outs = []
        for i in range(0, len(specs), 8):
            k = specs[i:i + 8]
            outs += runner.generate([PROMPT] * len(k), k, T, args.diffusion_steps,
                                    seeds=list(range(i, i + len(k))))
        for s, o in enumerate(outs):
            q = runner.to_qpos(o)
            tops, hws = measure(q)
            rows.append({"channel": channel, "value": value, "sample": s,
                         "top_nominal_m": float(tops[nom_slice].max()),
                         "top_adapted_m": float(tops[mid].max()),
                         "halfwidth_adapted_m": float(hws[mid].max()),
                         "pelvis_min_m": float(q[:, 2].min()),
                         "travel_m": float(q[-1, 0] - q[0, 0])})

    for d in DIPS:
        run("dip", d, [ConstraintSpec(root_xz=path, heading=np.zeros(T),
                                      root_y=NOMINAL_PELVIS - d * w, first_heading=0.0)
                       ] * N_SAMPLES)
        print(f"  dip {d:.2f} done ({time.time()-t0:.0f}s)", flush=True)

    for kk in TUCKS:
        specs = []
        for s in range(N_SAMPLES):
            if kk == 0.0:
                specs.append(base_spec)
                continue
            nom = noms[s]
            off = (nom["posed_joints"][active][:, arm_idx, :]
                   - nom["smooth_root_pos"][active][:, None, :])
            tgt = np.stack([path[active][:, None, 0] + off[:, :, 0] * (1 - kk),
                            nom["posed_joints"][active][:, arm_idx, 1],
                            path[active][:, None, 1] + off[:, :, 2] * (1 - kk)], -1)
            specs.append(ConstraintSpec(root_xz=path, heading=np.zeros(T),
                                        root_y=np.full(T, NOMINAL_PELVIS),
                                        pos_frames=active, pos_joints=arm_idx,
                                        pos_targets=tgt, first_heading=0.0))
        run("tuck", kk, specs)
        print(f"  tuck {kk:.2f} done ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # Calibrated envelope: per level, the 90% conformal upper bound on top and half-width.
    env = {}
    for ch, vals in (("dip", DIPS), ("tuck", TUCKS)):
        env[ch] = []
        for v in vals:
            g = [r for r in rows if r["channel"] == ch and r["value"] == v]
            env[ch].append({
                "value": v, "n": len(g),
                "top_mean": float(np.mean([r["top_adapted_m"] for r in g])),
                "top_worst3": float(np.max([r["top_adapted_m"] for r in g[:3]])),
                "top_p90_conformal": conformal_upper([r["top_adapted_m"] for r in g]),
                "top_max": float(np.max([r["top_adapted_m"] for r in g])),
                "hw_mean": float(np.mean([r["halfwidth_adapted_m"] for r in g])),
                "hw_worst3": float(np.max([r["halfwidth_adapted_m"] for r in g[:3]])),
                "hw_p90_conformal": conformal_upper([r["halfwidth_adapted_m"] for r in g]),
                "hw_max": float(np.max([r["halfwidth_adapted_m"] for r in g])),
                "hw_std": float(np.std([r["halfwidth_adapted_m"] for r in g])),
            })
    with open(out / "envelope.json", "w") as fh:
        json.dump({"alpha": 0.1, "n_samples": N_SAMPLES, "envelope": env}, fh, indent=2)
    with open(out / "receipt.json", "w") as fh:
        json.dump({"experiment": "exp001d_envelope_function", "model": runner.model_name,
                   "fps": fps, "duration_s": DURATION, "n_samples": N_SAMPLES,
                   "dips": DIPS, "tucks": TUCKS, "conformal_alpha": 0.1,
                   "n_rows": len(rows), "wall_clock_s": round(time.time() - t0, 1)}, fh,
                  indent=2)
    print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
