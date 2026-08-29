"""EXP-007: certify addressability, instead of estimating it.

The gap this closes
-------------------
Every "addressable" count this project has reported is a POINT ESTIMATE.  The guidance defines
addressability with a lower confidence bound,

    A_tau(x, z) = 1[ exists c :  P_lower(v = 1, sigma = z | x, c) >= tau ],

and with the gate's four held-out seeds no such bound can reach tau = 0.8.  The Clopper-Pearson
lower bound after k = n successes is alpha**(1/n), which at n = 4 is 0.47 and at n = 8 is 0.69;
certifying 0.8 needs

    alpha**(1/n) >= tau   ->   n >= ln(0.05) / ln(0.8) = 13.4   ->   n >= 14

unanimous seeds.  So "3.5 stable addressable strategies per scene" is, as measured, a statement
about four coin flips.  This experiment re-runs the eps-net representatives -- the distinct
bodies that actually define the reference set -- at enough seeds to say the sentence properly.

What it certifies
-----------------
For each representative program c and its modal mode z:

    p_hat  = fraction of seeds that are BOTH valid and realise z
    p_low  = Clopper-Pearson lower bound at 95 %
    certified  =  p_low >= tau

and per scene, the number of DISTINCT modes with at least one certified program.  That count is
the honest version of the project's headline number, and it can only be smaller than the point
estimate -- which is the direction a claim should move when it is checked.

The seeds are disjoint from BOTH of the gate's blocks (selection 100+, held-out 500+), so
nothing here is scored on a seed that chose it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion import body_enumerate as be  # noqa: E402
from scene2motion.morphology import d_morph, epsilon_net  # noqa: E402
from scene2motion.planner import plan  # noqa: E402
from scene2motion.program import ConstraintProgram  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import build_suite  # noqa: E402

TAU, ALPHA = 0.8, 0.05
SEED0 = 3000                      # disjoint from selection (100+) and held-out (500+)


def cp_lower(k: int, n: int, alpha: float = ALPHA) -> float:
    """Clopper-Pearson lower confidence bound on a binomial rate."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(alpha, k, n - k + 1))


def sig_key(sig) -> tuple:
    out = []
    for s in sig:
        duck, tuck, liftL, liftR, order, yaw = (list(s) + [0] * 6)[:6]
        out.append((bool(duck), bool(liftL), bool(liftR), bool(yaw)))
    return tuple(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="outputs/exp005g/candidates.jsonl")
    ap.add_argument("--out", default="outputs/exp007")
    ap.add_argument("--n_seeds", type=int, default=24)
    ap.add_argument("--per_scene", type=int, default=6, help="eps-net representatives to certify")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds_per_rung", type=int, default=2)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    n_need = int(np.ceil(np.log(ALPHA) / np.log(TAU)))
    print(f"certifying P >= {TAU} at {1-ALPHA:.0%} needs n >= {n_need} unanimous seeds; "
          f"running n = {args.n_seeds}")
    print(f"  Clopper-Pearson lower bound at {args.n_seeds}/{args.n_seeds}: "
          f"{cp_lower(args.n_seeds, args.n_seeds):.3f}")

    rows = [json.loads(l) for l in open(args.ledger)]
    by_scene = defaultdict(list)
    for r in rows:
        if r["arm"] != "NULL-SEED" and r["heldout"]:
            by_scene[r["scene_id"]].append(r)

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    scenes = {s.scene_id: s for s in build_suite(seeds_per_rung=args.seeds_per_rung)}
    seeds = list(range(SEED0, SEED0 + args.n_seeds))
    results, order = [], list(by_scene)
    if args.limit:
        order = order[:args.limit]

    for sid in order:
        sc = scenes.get(sid)
        if sc is None:
            continue
        p = plan(sc, "adaptive")
        if not p.feasible:
            continue
        ctx = be.route_context(sc, p, fps=runner.fps)
        if not ctx.feasible:
            continue
        ev = be.BodyEvaluator(runner, ctx, seeds=seeds, batch=args.batch,
                              diffusion_steps=args.diffusion_steps)

        # the representatives: eps-net over the addressable held-out candidates
        cand = list({r["prog_key"]: r for r in by_scene[sid]}.values())
        addr = [c for c in cand
                if c["heldout"]["feasible_rate"] >= 0.75 and c["heldout"]["stability"] >= TAU]
        if len(addr) < 2:
            continue
        mus = np.array([c["heldout"]["mu"] for c in addr], float)
        W = np.eye(mus.shape[1])
        try:
            eps = json.load(open(Path(args.ledger).parent / "receipt.json"))[
                "eps_calibration"]["eps_used"]
            covs = []
            for c in addr:
                D = np.asarray(c["heldout"]["deltas"], float)
                if len(D) >= 2:
                    covs.append(np.cov(D.T))
            S = np.mean(covs, axis=0) + 1e-6 * np.eye(mus.shape[1])
            w, V = np.linalg.eigh(S)
            W = V @ np.diag(np.maximum(w, 1e-9) ** -0.5) @ V.T
        except Exception:
            eps = 4.0
        reps = [addr[i] for i in epsilon_net(mus, W, eps)][:args.per_scene]

        progs = [ConstraintProgram(lat=ctx.base.lat.copy(),
                                   slot=np.array(c["slot"], float), speed=ctx.base.speed)
                 for c in reps]
        evs = ev.evaluate(progs, ["CERTIFY"] * len(progs))
        rec = {"scene_id": sid, "family": by_scene[sid][0]["family"],
               "n_addressable_pointest": len(addr),
               "n_modes_pointest": len({sig_key(c["heldout"]["signature"]) for c in addr}),
               "reps": []}
        for c, e in zip(reps, evs):
            z = sig_key(e.signature)
            k = int(sum(1 for f, s in zip(e.feasible, e.signatures)
                        if f and sig_key(s) == z))
            low = cp_lower(k, len(e.feasible))
            rec["reps"].append({
                "prog_key": c["prog_key"], "mode": ["".join("DLRY"[i] for i, b in enumerate(t)
                                                            if b) or "none" for t in z],
                "k": k, "n": len(e.feasible), "p_hat": k / max(len(e.feasible), 1),
                "p_lower": low, "certified": bool(low >= TAU),
                "p_hat_heldout_4seed": c["heldout"]["feasible_rate"] * c["heldout"]["stability"]})
        certified = {tuple(r["mode"]) for r in rec["reps"] if r["certified"]}
        rec["n_modes_certified"] = len(certified)
        results.append(rec)
        print(f"  {sid[:30]:30s} reps {len(reps)}  modes point-est "
              f"{rec['n_modes_pointest']} -> certified {rec['n_modes_certified']}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    n = max(len(results), 1)
    pe = float(np.mean([r["n_modes_pointest"] for r in results])) if results else 0.0
    ce = float(np.mean([r["n_modes_certified"] for r in results])) if results else 0.0
    allr = [x for r in results for x in r["reps"]]
    print(f"\n{len(results)} scenes, {len(allr)} representative programs at "
          f"{args.n_seeds} seeds ({time.time()-t0:.0f}s)")
    print(f"\nmodes per scene:  point estimate {pe:.2f}  ->  CERTIFIED at tau={TAU} {ce:.2f}")
    if allr:
        print(f"representative programs certified: "
              f"{np.mean([r['certified'] for r in allr]):.1%}")
        print(f"mean p_hat {np.mean([r['p_hat'] for r in allr]):.3f}   "
              f"mean p_lower {np.mean([r['p_lower'] for r in allr]):.3f}")
        print(f"\n4-seed estimate vs {args.n_seeds}-seed truth: "
              f"correlation {np.corrcoef([r['p_hat_heldout_4seed'] for r in allr], [r['p_hat'] for r in allr])[0,1]:+.3f}, "
              f"mean 4-seed {np.mean([r['p_hat_heldout_4seed'] for r in allr]):.3f} vs "
              f"{np.mean([r['p_hat'] for r in allr]):.3f}")
        print("  A 4-seed estimate that sits ABOVE the many-seed truth is the optimism a small "
              "sample buys\n  by rounding 3/4 up to 0.75 and 4/4 up to 1.00; the gap is what "
              "the gate's addressability\n  column was worth.")
    json.dump({"experiment": "exp007_certify_addressability", "tau": TAU, "alpha": ALPHA,
               "n_seeds": args.n_seeds, "n_needed": n_need,
               "modes_per_scene_pointest": pe, "modes_per_scene_certified": ce,
               "scenes": results}, open(out / "receipt.json", "w"), indent=2)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
