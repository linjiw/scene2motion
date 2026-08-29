"""What does the audit actually buy?  The same clips, counted four ways.

This is the project's contribution stated as a number.  Every row below uses the SAME 36 body
programs and the SAME generated clips from `outputs/exp005f_fixed_s5`.  Only the counting
methodology differs, and each of the naive rows is a choice a careful practitioner could make
without doing anything obviously wrong:

  1 seed, any change > 1 mm      -- "the descriptor moved, so the channel did something"
  1 seed, round 1 cm threshold   -- a plausible round number instead of a measured one
  1 seed, 1 cm, drop dead clips  -- the same, after discarding clips that never validate
  6 seeds, paired, q99, stable   -- the audit: matched controls, a threshold measured from the
                                    prior's own seed scatter, and a stability requirement

The gap between the first three and the last is the false capability that a single-seed,
round-threshold reading manufactures.  The final line adds the physics tracker (EXP-011), which
removes what the kinematic audit still allows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bits(d: np.ndarray, th: np.ndarray) -> tuple:
    on = lambda i: bool(d[i] > th[i])                       # noqa: E731
    return (on(0), on(1) or on(2), on(3), on(4), on(7))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/exp005f_fixed_s5")
    ap.add_argument("--min_stability", type=float, default=0.8)
    args = ap.parse_args()
    d = Path(args.run)
    per = json.load(open(d / "per_scene.json"))
    q99 = np.array(json.load(open(d / "receipt.json"))["seed_noise_q99"], float)

    def count(mode: str) -> float:
        out = []
        for _sid, sd in per.items():
            R = sd["rows"]
            if mode == "calibrated":
                sigs = set()
                for r in R:
                    if r["valid_frac"] <= 0.5:
                        continue
                    p = [bits(np.array(x), q99) for x in r["deltas"]]
                    c = Counter(p).most_common(1)[0]
                    if c[1] / len(p) >= args.min_stability:
                        sigs.add(c[0])
            else:
                th = np.full(8, 0.001 if mode == "any" else 0.01)
                rows = R if mode != "valid" else [r for r in R if r["valid_frac"] > 0]
                sigs = {bits(np.array(r["deltas"][0]), th) for r in rows}
            out.append(len(sigs))
        return float(np.mean(out))

    print(f"distinct body modes per scene — same 36 programs, same clips ({d.name})\n")
    rows = [("1 seed, any change > 1 mm", "any"),
            ("1 seed, round 1 cm threshold", "cm"),
            ("1 seed, 1 cm, drop clips that never validate", "valid"),
            ("6 seeds, paired, q99-calibrated, stability >= 0.8", "calibrated")]
    res = {}
    for label, mode in rows:
        res[label] = count(mode)
        print(f"   {label:52s} {res[label]:5.2f}")
    cal = res[rows[-1][0]]
    worst = max(res[r[0]] for r in rows[:-1])
    print(f"\n   ...and after the physics tracker (EXP-011: only duck executes)      ~1")
    print(f"\n  The audit is the difference between reporting {worst:.0f} capabilities and "
          f"{cal:.1f};\n  the tracker takes it to about 1.  Overstatement factor without the "
          f"audit: {worst / max(cal, 1e-9):.1f}x kinematic, ~{worst:.0f}x after physics.")
    json.dump({"experiment": "audit_delta", "run": str(d), "counts": res,
               "overstatement_kinematic": worst / max(cal, 1e-9)},
              open(d.parent / "audit_delta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
