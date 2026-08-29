"""EXP-005h: are the body alternatives DECISION-RELEVANT, or merely different?

What changed, and why
---------------------
The first version of this experiment scored decision-relevance by Pareto non-dominance over
SEVEN objectives.  That is the wrong instrument, for a reason that has nothing to do with the
code being wrong: as the objective count grows, dominance becomes sparse, and with seven
objectives and 8-14 candidates the likely failure is not that the front collapses to one point
-- it is that ALMOST EVERYTHING is nondominated and the metric certifies diversity it never
measured.  Hypervolume inherits the same problem plus a sensitivity to normalisation and
reference choice.  (The first version also had a moving-box hypervolume bug, caught by a unit
check before any GPU time: with `ideal` taken per arm, a single DOMINATED point scored 1.0.)

So the hierarchy is now:

  1.  HARD CONSTRAINTS, not objectives:  goal reached, collision-free, stability >= 0.8.
  2.  THREE interpretable objective groups, for a 3-D hypervolume that means something.
  3.  PREFERENCE REGRET as the primary evidence -- does the set contain a good choice for
      someone who actually wants something?  A candidate earns its place by WINNING under some
      declared preference, not by sitting more than eps away from its neighbours.

And it costs no GPU: every quantity is read from EXP-005g's per-candidate ledger, scored on the
HELD-OUT seed block that no proposer could select on.

The objectives, all lower-is-better, normalised per scene against the shared feasible pool
--------------------------------------------------------------------------------------------
    safety          -clearance_q10 + LAMBDA_INVALID * P(invalid across held-out seeds)
    burden          pelvis deviation, envelope roughness, joint jerk, contact inconsistency
    preference      commanded dip + lift + tuck integral, and adaptation duration

`foot_floor_pen` is KINEMATIC CONTACT INCONSISTENCY -- feet intersecting the ground plane in
the exported kinematics.  It is not physical slip and is not called that until a tracker runs.

Pre-committed kill condition (stated before the numbers are read)
-----------------------------------------------------------------
The morphology-diversity claim weakens substantially if EITHER

  * fewer than 25 % of feasible scenes contain at least two stable modes that are optimal under
    DIFFERENT declared preferences, or
  * the K=8 set does not materially reduce normalised preference regret against the single best
    neutral program,

with a scene-level paired bootstrap interval, not a bare difference of means.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_STAB, MIN_FEAS = 0.8, 0.75
LAMBDA_INVALID = 0.5      # metres of clearance a coin-flip program is worth giving up
K_SET = 8

# The raw components, lower-is-better, and the group each belongs to.
COMPONENTS = (
    ("neg_clear_q10", "safety"),      # -10th-percentile clearance along the clip
    ("p_invalid", "safety"),
    ("pelvis_dev", "burden"),         # how far from standing
    ("env_rough", "burden"),          # second difference of the clearance envelope
    ("joint_jerk", "burden"),
    ("contact_incons", "burden"),     # feet through the floor, NOT physical slip
    ("dip_cmd", "preference"),
    ("lift_cmd", "preference"),
    ("tuck_cmd", "preference"),
    ("adapt_span", "preference"),
)
NAMES = [c for c, _ in COMPONENTS]
GROUPS = ("safety", "burden", "preference")

# Predeclared preferences.  Each is a weight vector over NAMES.  These are fixed before the
# numbers are read; "minimise heading change" is deliberately absent because no channel can
# request a heading change (program.py:31), so it is not a preference anyone could express.
PREFS = {
    "max clearance":      {"neg_clear_q10": 1.0},
    "stay upright":       {"pelvis_dev": 1.0, "dip_cmd": 0.5},
    "minimise lifting":   {"lift_cmd": 1.0, "contact_incons": 0.5},
    "smoothest":          {"env_rough": 1.0, "joint_jerk": 1.0},
    "least adaptation":   {"dip_cmd": 1.0, "lift_cmd": 1.0, "tuck_cmd": 1.0, "adapt_span": 1.0},
    "most reliable":      {"p_invalid": 1.0},
    "balanced":           {n: 1.0 for n in NAMES},
}


def sig_key(sig) -> tuple:
    out = []
    for s in sig:
        duck, tuck, liftL, liftR, order, yaw = (list(s) + [0] * 6)[:6]
        out.append((bool(duck), bool(liftL), bool(liftR), bool(yaw)))
    return tuple(out)


def addressable(ev) -> bool:
    return bool(ev and ev["feasible_rate"] >= MIN_FEAS and ev["stability"] >= MIN_STAB)


def raw_objectives(row: dict) -> np.ndarray | None:
    """The ten lower-is-better components for one candidate, from held-out seeds only."""
    ev = row.get("heldout")
    if not ev or not ev.get("features"):
        return None
    keep = [f for f, ok in zip(ev["features"], ev["feasible"]) if ok]
    if not keep:
        return None
    m = lambda k: float(np.mean([f[k] for f in keep]))
    c = row["commanded"]
    return np.array([
        -m("clear_q10"), 1.0 - float(ev["feasible_rate"]),
        m("pelvis_dev"), m("env_rough"), m("joint_jerk"), m("foot_floor_pen"),
        c["dip"], c["lift"], c["tuck"], c["span"],
    ], float)


def normalise(F: np.ndarray) -> np.ndarray:
    """Per-scene min-max against the SHARED feasible pool, so no arm defines its own scale."""
    lo, hi = F.min(0), F.max(0)
    return (F - lo) / np.maximum(hi - lo, 1e-9)


def pareto_front(F: np.ndarray) -> np.ndarray:
    keep = []
    for i in range(len(F)):
        if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                   for j in range(len(F)) if j != i):
            keep.append(i)
    return np.array(keep, dtype=int)


def hypervolume(F: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    """Fraction of the SHARED quasi-random points in the fixed unit box dominated by `F`.

    `pts` is passed in so every arm in a scene is scored on the SAME points -- common random
    numbers, so arm-to-arm differences are not estimator noise.  Returns (value, MC s.e.).
    """
    if not len(F):
        return 0.0, 0.0
    dom = np.zeros(len(pts), bool)
    for f in F:
        dom |= np.all(f <= pts, axis=1)
    p = float(dom.mean())
    return p, float(np.sqrt(max(p * (1 - p), 0.0) / len(pts)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="outputs/exp005g/candidates.jsonl")
    ap.add_argument("--out", default="outputs/exp005h")
    ap.add_argument("--n_mc", type=int, default=32768)   # power of 2: Sobol balance
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.ledger)]
    scenes = defaultdict(list)
    for r in rows:
        scenes[r["scene_id"]].append(r)
    rng = np.random.default_rng(0)
    from scipy.stats import qmc
    Wp = np.array([[PREFS[p].get(n, 0.0) for n in NAMES] for p in PREFS], float)
    Wp = Wp / np.maximum(Wp.sum(1, keepdims=True), 1e-9)
    gidx = {g: [i for i, (_, gg) in enumerate(COMPONENTS) if gg == g] for g in GROUPS}

    recs = []
    for sid, g in scenes.items():
        # Deduplicate the POOL by program -- two arms proposing the same program is one
        # candidate -- but remember EVERY arm that proposed it and the best rank each gave it.
        # Collapsing onto a single arm label would silently take a candidate away from every
        # arm but one, and the arms most penalised would be exactly the ones that agree with
        # the others, which is backwards.
        owners: dict[str, dict[str, int]] = {}
        uniq: dict[str, dict] = {}
        for r in g:
            if r["arm"] == "NULL-SEED":
                continue
            uniq.setdefault(r["prog_key"], r)
            o_ = owners.setdefault(r["prog_key"], {})
            o_[r["arm"]] = min(o_.get(r["arm"], 10 ** 9), r["rank"])
        keep, F, own = [], [], []
        for k, c in uniq.items():
            if not addressable(c.get("heldout")):
                continue                       # hard constraints first, not objectives
            v = raw_objectives(c)
            if v is not None:
                keep.append(c)
                F.append(v)
                own.append(owners[k])
        if len(keep) < 2:
            continue
        F = normalise(np.array(F))
        G = np.stack([F[:, gidx[g_]].mean(1) for g_ in GROUPS], 1)
        front = pareto_front(G)
        pts = qmc.Sobol(d=len(GROUPS), scramble=True,
                        seed=abs(hash(sid)) % (2 ** 31)).random(args.n_mc)

        # who wins under each declared preference, and is it always the same body?
        scores = F @ Wp.T                                    # (n_cand, n_pref)
        winners = scores.argmin(0)
        win_modes = [sig_key(keep[i]["heldout"]["signature"]) for i in winners]
        n_distinct = len(set(win_modes))

        # the neutral reference: the least-adapted addressable candidate
        neutral = int(F[:, [NAMES.index(n) for n in
                            ("dip_cmd", "lift_cmd", "tuck_cmd", "adapt_span")]].sum(1).argmin())

        def regret(idx: list[int]) -> float:
            """Mean over preferences of (best in the set - best available), normalised.

            NOTE the trap this function sets, which the first version of this experiment walked
            straight into: `regret(all candidates)` is IDENTICALLY 0, because best_set and
            best_all are then the same minimum. Comparing the neutral program against the full
            pool therefore always shows a "significant reduction", and the pre-committed kill
            condition -- "the K=8 set does not materially reduce regret over the best single
            neutral program" -- could never fire. The comparison has to be against a SET OF
            SIZE K, not against the pool that defines the optimum.
            """
            if not len(idx):
                return 1.0
            best_set = scores[idx].min(0)
            best_all = scores.min(0)
            span = np.maximum(scores.max(0) - best_all, 1e-9)
            return float(np.mean((best_set - best_all) / span))

        def best_k(K: int) -> list[int]:
            """Greedy size-K subset minimising mean preference regret (an upper bound)."""
            chosen: list[int] = []
            for _ in range(min(K, len(keep))):
                pick = min((i for i in range(len(keep)) if i not in chosen),
                           key=lambda i: regret(chosen + [i]), default=None)
                if pick is None:
                    break
                chosen.append(pick)
            return chosen

        hv_all, se_all = hypervolume(G[front], pts)
        rec = {"scene_id": sid, "family": g[0]["family"], "n_addressable": len(keep),
               "front_size": int(len(front)), "hv_reference": hv_all, "hv_se": se_all,
               "n_distinct_pref_winners": n_distinct,
               "pref_winners": {p: "".join("DLRY"[i] for i, b in enumerate(m[0]) if b) or "none"
                                for p, m in zip(PREFS, win_modes)},
               "regret_neutral": regret([neutral]),
               # regret_full is retained ONLY as a visible zero: it is what the first version
               # of this experiment reported as the headline, and it is 0 by construction.
               "regret_full": regret(list(range(len(keep)))),
               "regret_bestK": {str(K): regret(best_k(K)) for K in (1, 2, 4, K_SET)},
               "arms": {}}
        for arm in sorted({r["arm"] for r in g if r["arm"] != "NULL-SEED"}):
            # an arm's K=8 set, ordered by the rank THAT arm gave each program
            mine = sorted([i for i, o_ in enumerate(own) if arm in o_],
                          key=lambda i: own[i][arm])[:K_SET]
            hv, se = hypervolume(G[mine], pts) if mine else (0.0, 0.0)
            rec["arms"][arm] = {"n": len(mine), "regret": regret(mine), "hv": hv, "hv_se": se,
                                "hv_ratio": hv / hv_all if hv_all > 0 else 0.0,
                                "front_recall": (len(set(mine) & set(front.tolist()))
                                                 / max(len(front), 1))}
        recs.append(rec)

    def boot(v):
        v = np.asarray(v, float)
        idx = rng.integers(0, len(v), (args.boot, len(v)))
        return tuple(np.percentile(v[idx].mean(1), [2.5, 97.5]))

    n = len(recs)
    print(f"EXP-005h over {n} scenes with >= 2 addressable candidates "
          f"(mean {np.mean([r['n_addressable'] for r in recs]):.1f} candidates)\n")
    print(f"mean 3-D Pareto front size          {np.mean([r['front_size'] for r in recs]):.2f}"
          f"   (of {np.mean([r['n_addressable'] for r in recs]):.1f} addressable)")
    frac2 = float(np.mean([r["n_distinct_pref_winners"] >= 2 for r in recs]))
    lo, hi = boot([r["n_distinct_pref_winners"] >= 2 for r in recs])
    print(f"scenes where >= 2 DIFFERENT modes win different preferences   "
          f"{frac2:.1%}  [{lo:.1%}, {hi:.1%}]")
    print(f"   pre-committed kill threshold is 25 %  ->  "
          f"{'PASSES' if frac2 >= 0.25 else 'FAILS -- the diversity has no decision value'}")

    print(f"\nnormalised preference regret, best size-K set (greedy, hindsight upper bound):")
    print(f"   {'neutral single':>18s} {np.mean([r['regret_neutral'] for r in recs]):.3f}")
    for K in (1, 2, 4, K_SET):
        v = np.mean([r["regret_bestK"][str(K)] for r in recs])
        print(f"   {'best K=' + str(K):>18s} {v:.3f}")
    print(f"   {'whole pool':>18s} "
          f"{np.mean([r['regret_full'] for r in recs]):.3f}   <- 0 BY CONSTRUCTION: the pool "
          f"defines the optimum.\n{'':22s}The first version of this experiment used it as the "
          f"headline, which made the\n{'':22s}pre-committed kill condition unable to fire.")
    dr = (np.array([r["regret_neutral"] for r in recs])
          - np.array([r["regret_bestK"][str(K_SET)] for r in recs]))
    lo, hi = boot(dr)
    print(f"\n   K={K_SET} vs neutral, paired reduction {dr.mean():.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"{'(significant)' if lo > 0 else '(NOT significant)'}")
    d21 = (np.array([r["regret_bestK"]["1"] for r in recs])
           - np.array([r["regret_bestK"]["2"] for r in recs]))
    lo2, hi2 = boot(d21)
    print(f"   the SECOND body alone buys {d21.mean():.3f}  [{lo2:.3f}, {hi2:.3f}] -- if this "
          f"is ~0 there is\n   no trade-off to serve and one well-chosen body is the whole "
          f"answer.")

    arms = sorted({a for r in recs for a in r["arms"]})
    print(f"\n{'arm':16s} {'n':>4s} {'regret':>8s} {'vs neutral':>18s} {'hv':>7s} "
          f"{'hv/ref':>7s} {'front recall':>13s}")
    for a in arms:
        v = [r["arms"][a] for r in recs if a in r["arms"]]
        if not v:
            continue
        d = np.array([r["regret_neutral"] - r["arms"][a]["regret"]
                      for r in recs if a in r["arms"]])
        lo, hi = boot(d)
        print(f"{a:16s} {np.mean([x['n'] for x in v]):4.1f} "
              f"{np.mean([x['regret'] for x in v]):8.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {np.mean([x['hv'] for x in v]):7.3f} "
              f"{np.mean([x['hv_ratio'] for x in v]):7.3f} "
              f"{np.mean([x['front_recall'] for x in v]):13.3f}")
    print(f"  MC s.e. on hypervolume {np.mean([r['hv_se'] for r in recs]):.4f} -- differences "
          f"smaller than this are estimator noise.\n  All arms share the same Sobol points "
          f"within a scene (common random numbers).")

    # which preference is served by which body -- and does yaw ever win?
    print("\nwinning body per declared preference (fraction of scenes):")
    for p in PREFS:
        c = defaultdict(int)
        for r in recs:
            c[r["pref_winners"][p]] += 1
        top = sorted(c.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {p:18s} " + "  ".join(f"{k or 'none':>6s} {v / n:.2f}" for k, v in top))
    yaw_wins = np.mean([any("Y" in v for v in r["pref_winners"].values()) for r in recs])
    print(f"\nyaw appears in a preference-winning body on {yaw_wins:.1%} of scenes -- the "
          f"guidance's test of\nwhether yaw earns its place as a capability rather than a "
          f"style variation.")

    json.dump({"experiment": "exp005h_pareto_morph", "n_scenes": n,
               "components": NAMES, "groups": list(GROUPS), "prefs": list(PREFS),
               "lambda_invalid": LAMBDA_INVALID,
               "frac_scenes_two_pref_winners": frac2,
               "regret_neutral": float(np.mean([r["regret_neutral"] for r in recs])),
               "regret_full_is_zero_by_construction": float(
                   np.mean([r["regret_full"] for r in recs])),
               "regret_bestK": {str(K): float(np.mean([r["regret_bestK"][str(K)] for r in recs]))
                                for K in (1, 2, 4, K_SET)},
               "mean_front_size": float(np.mean([r["front_size"] for r in recs])),
               "yaw_wins_frac": float(yaw_wins), "scenes": recs},
              open(out / "receipt.json", "w"), indent=2, default=float)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
