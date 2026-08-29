"""The capability funnel, computed identically across configurations so they can be compared.

Section 23 rebuilt the guidance's headline figure so that its stages are genuinely NESTED --
each a subset of the one above -- after finding that the published version mixed a continuous
eps-net count into a discrete-mode chain, which let stage 4 exceed stage 3. This script is that
computation, factored out of a heredoc so the same code runs on every configuration and the
numbers are comparable by construction rather than by my care in retyping them.

Stages, one filter chain:

    requested         the dip x tuck x lift ladder as submitted
    valid             goal reached and collision-free on > half the seeds
    valid & stable    one modal active set on >= `--min_stability` of the seeds
    modes             distinct active sets among those
    modes, pruned     the same count with the channels that cannot be commanded removed

Each run's OWN measured seed-noise quantile is used as the firing threshold, never a hardcoded
one -- EXP-005g shipped with `SEED_SIGMA_PRIOR * 2.33`, which measured 26-42 % below the q99
sitting on disk and inflated every mode count that rested on it.

The continuous eps-net is deliberately NOT a stage here. It is a different kind of quantity and
it is scale-dependent -- 6.2 members per scene in pooled-covariance sigma against 2.5 in q99
units at the same eps -- so it belongs beside the funnel with its unit named, not inside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.morphology import CHANNELS  # noqa: E402


def bits(delta: np.ndarray, q99: np.ndarray, drop: set[str]) -> tuple:
    """Realised active set, with named channels suppressed rather than merely ignored."""
    on = lambda i: bool(delta[i] > q99[i])                       # noqa: E731
    duck = on(CHANNELS.index("dh_top"))
    tuck = on(CHANNELS.index("dw_left")) or on(CHANNELS.index("dw_right"))
    liftL = on(CHANNELS.index("dz_foot_left"))
    liftR = on(CHANNELS.index("dz_foot_right"))
    yaw = on(CHANNELS.index("dpsi"))
    out = [duck, liftL, liftR]
    if "tuck" not in drop:
        out.append(tuck)
    if "yaw" not in drop:
        out.append(yaw)
    return tuple(out)


def funnel(per_scene: dict, q99: np.ndarray, min_stability: float, drop: set[str]) -> dict:
    rows = []
    for sid, sd in per_scene.items():
        R = sd["rows"]
        valid = [r for r in R if r["valid_frac"] > 0.5]
        stable, sigs = [], []
        for r in valid:
            per = [bits(np.array(d), q99, drop) for d in r["deltas"]]
            c = Counter(per).most_common(1)[0]
            if c[1] / len(per) >= min_stability:
                stable.append(r)
                sigs.append(c[0])
        rows.append({"scene": sid, "requested": len(R), "valid": len(valid),
                     "stable": len(stable), "modes": len(set(sigs))})
    m = {k: float(np.mean([r[k] for r in rows]))
         for k in ("requested", "valid", "stable", "modes")}
    return {"per_scene": rows, "mean": m}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="one or more outputs/<run> directories holding per_scene.json")
    ap.add_argument("--min_stability", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = [("full alphabet", set()), ("no tuck", {"tuck"}),
                ("no tuck, no yaw", {"tuck", "yaw"})]
    report = {}
    print(f"{'run':26s} {'alphabet':18s} {'req':>6s} {'valid':>7s} {'stable':>7s} {'modes':>7s}")
    print("-" * 76)
    for run in args.runs:
        d = Path(run)
        per = json.load(open(d / "per_scene.json"))
        q99 = np.array(json.load(open(d / "receipt.json"))["seed_noise_q99"], float)
        report[run] = {"seed_noise_q99": q99.tolist(), "variants": {}}
        for label, drop in variants:
            f = funnel(per, q99, args.min_stability, drop)
            report[run]["variants"][label] = f
            m = f["mean"]
            print(f"{d.name[:26]:26s} {label:18s} {m['requested']:6.0f} {m['valid']:7.2f} "
                  f"{m['stable']:7.2f} {m['modes']:7.2f}")
        print()
    print("  Every stage is a subset of the one above it.  The threshold is each run's OWN")
    print("  measured q99, so a configuration that is quieter is not credited with more modes")
    print("  merely for being compared against a stale noise scale.")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
