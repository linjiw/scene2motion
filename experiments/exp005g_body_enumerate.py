"""EXP-005g: BODY-ENUMERATE@K -- can a classical same-route body enumerator cover F_B(S, r)?

The question, and why it is THE gate
------------------------------------
EXP-005e measured that the shipped strategy enumerator is near-complete over ROUTES (2.9 % of
its misses involve a new route) and structurally blind to a different BODY along the same route
(93.6 %), and that the misses are cost-competitive (median 1.005x the best route, 73.5 % within
1.02x).  That indicts the CORRIDOR-EXCLUSION RULE in `strategies.py:143-148`, not classical
search.  Before any learned proposer is trained a reviewer will ask the obvious question:

    "why not just enumerate different mode assignments at a fixed route?"

This experiment answers it with the strongest classical enumerator we can write
(`scene2motion.body_enumerate`), and it is arranged so that a NEGATIVE result for the learned
model is the easy outcome: if the classical baseline covers F_B at K=8, we say so and do not
train.

For every scene of the 128-scene suite with a feasible route, each baseline is run at
K in {1, 2, 4, 8}; every candidate is generated through the FROZEN prior with per-sample seeds,
verified with `G1Body.trajectory_report`, and described by a PAIRED DELTA against a matched
neutral-body control on the SAME ROUTE and the SAME SEED (`morphology.matched_delta`).  Without
that pairing the descriptor rediscovers the confounds EXP-001 already corrected.

Reported per baseline:
    discrete morphology-mode recall   over `morphology.active_set` signatures
    continuous MorphRecall@K          over an eps-net reference (`morphology.epsilon_net`,
                                      `morphology.morph_recall_at_k`), MACRO-AVERAGED PER SCENE
    AREA UNDER K                      over the whole {1,2,4,8} grid, not just K=8
    ARDY calls charged                the currency the comparison is made in
    planner CPU                       separately, as the guidance requires

How the reference set is built, and why that is not circular
------------------------------------------------------------
For each (scene, route) the reference POOL is

    UNION over all baselines of their evaluated candidates
  + REF-RANDOM, an expensive random-restart BODY search sharing no rule with any baseline
    (random whole-route posture floors, off-lattice continuous amplitudes above the geometric
    requirement, random within-interaction splits, random gratuitous adaptations, random
    onset/offset) -- the body analogue of `exp005e.reference_signatures`,

filtered to members that are ADDRESSABLE (feasible on >= `min_feasible_rate` of seeds AND
`Stability >= 0.8`, the guidance's pre-committed bar), then deduplicated with
`morphology.epsilon_net` under a whitener estimated from THIS RUN's own replicate seeds.

Three reasons that is not circular, and the first is the important one:

 1. THE BIAS RUNS AGAINST THE LEARNED MODEL.  Every baseline is scored against a reference that
    CONTAINS its own contributions, so each baseline automatically covers everything it found.
    That inflates classical coverage.  If a classical baseline still fails to reach 90 % under
    a reference biased in its favour, the failure is not an artefact of a hostile reference --
    which is the direction of conservatism a gate needs.
 2. The LEAVE-ONE-OUT columns remove that inflation: each baseline is re-scored against a
    reference rebuilt WITHOUT its own contributions.  Both are reported; the gap is exactly how
    much of a baseline's score was self-supplied.
 3. REF-RANDOM is produced by a procedure no baseline uses, and `reference_contributors` reports
    how much of the final net only it reached.  If that share is ~0, the reference really is
    just the union of the baselines and the experiment cannot detect a body they all miss --
    that is a stated failure condition of this design, not a hidden one.

No learned model is fitted here and the reference is not a training set.  The only claim the
eps-net supports is "these are the distinguishable, addressable body realisations we could find
at this route".

THE CONTROL THAT DECIDES WHETHER ANY OF THIS MEANS ANYTHING
-----------------------------------------------------------
NULL-SEED submits the SAME program (the shipped planner's own answer) K times, each on a
DISJOINT seed block.  It proposes nothing.  Two numbers come out of it:

  * `internal_eps_net` -- how many "distinct bodies" ONE program appears to contribute.  It
    must be 1.  If it is K, the coverage metric is measuring ARDY's sampler.
  * the null distance distribution -- which is what `eps` is CALIBRATED ON.  The guidance's
    `d_morph > 2..3` is stated per channel, but `d_morph` is a chi-distance over 8 channels x
    n_windows, in which two SEED MEANS of the same program sit ~sqrt(2 D / n_seeds) apart --
    ~2.3 at D=8, n_seeds=3.  Measured here at n_seeds=2 the mean same-program distance was
    3.79 sigma with 95 % of pairs above 2.0, i.e. the pre-committed constant sits INSIDE the
    null distribution and one program regenerated eight times would score as eight distinct
    bodies.  So the primary table uses `eps = q95(same-program distance)` -- "different" means
    "further apart than 95 % of pairs that are the same body" -- and the eps=2.0 table is
    printed underneath as the pre-committed ablation.  Calibrating on the null arm is not
    circular: it contains no proposer information, only the frozen prior's scatter.

Cost
----
Charged ARDY calls per scene at defaults (n_seeds=3, K=8, per-K equal-call budgets for the
searching baselines, 24 reference programs) are ~520, of which the clip cache generates ~80 %;
at ~0.20 s of generation plus ~0.07 s of MuJoCo checking per clip that is ~2.5 min of GPU per
scene and ~3.5 h for the 84 feasible scenes of the 128-scene suite.  Run `--smoke` first
(6 scenes, 2 seeds, ~5 min) and read the NULL-SEED line before anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion import body_enumerate as be                        # noqa: E402
from scene2motion.morphology import (CHANNELS, N_CHANNELS, d_morph,   # noqa: E402
                                     epsilon_net, morph_recall_at_k, seed_statistics,
                                     whitener)
from scene2motion.planner import plan                                 # noqa: E402
from scene2motion.runner import ArdyRunner                            # noqa: E402
from scene2motion.scenes import BUILDERS, build_suite                 # noqa: E402

KS = (1, 2, 4, 8)
CPU_BASELINES = ("A-KBEST", "B-NOGOOD", "C-WSWEEP")
SEARCH_BASELINES = ("D-REFINE", "COMPOSITE")
SMOKE_SCENES = [("overhead_beam", 1.10), ("overhead_beam", 0.90), ("beam_and_gap", 1.10),
                ("narrow_gap", 0.60), ("partial_beam", 1.05), ("pillar", 0.30)]


# =======================================================================================
# metrics
# =======================================================================================

def area_under_k(curve: dict) -> float:
    """Mean recall over K in {1,2,4,8}.

    Reported instead of recall@8 alone because a baseline that needs eight calls to reach what
    another reaches at two is worse at every budget anyone would actually spend.  The simple
    mean over the pre-committed grid is used rather than a trapezoid in K so that each budget
    carries equal weight instead of K=8 carrying 4/7 of it.
    """
    v = [curve[k] for k in KS if k in curve and curve[k] == curve[k]]
    return float(np.mean(v)) if v else float("nan")


def discrete_recall(ref_sigs: set, cand_sigs) -> float:
    if not ref_sigs:
        return float("nan")
    return float(len(ref_sigs & set(cand_sigs)) / len(ref_sigs))


def pooled_whitener(evals) -> np.ndarray:
    """(8, 8) whitener from THIS RUN's replicate seeds.

    Per-window blocks are pooled as independent draws of the same 8-channel noise, which is what
    makes `d_morph` "how many ARDY-noise standard deviations apart" and comparable between
    scenes (morphology.py:199-207).  Estimating it in-run rather than trusting a stored constant
    is deliberate: a sigma taken from one strong program per scene under-reads the timing
    channels by ~10x, and under a 10x-optimistic sigma 96 % of same-program seed pairs read as
    ">2 sigma apart".
    """
    covs = []
    for e in evals:
        if len(e.deltas) < 2 or not len(e.mu):
            continue
        for w in range(max(1, len(e.mu) // N_CHANNELS)):
            _, cov = seed_statistics(e.deltas[:, w * N_CHANNELS:(w + 1) * N_CHANNELS])
            if np.all(np.isfinite(cov)) and cov.shape == (N_CHANNELS, N_CHANNELS):
                covs.append(cov)
    return whitener(covs) if covs else np.eye(N_CHANNELS)


def block_whitener_from(W8: np.ndarray, n_win: int, channels) -> np.ndarray:
    """Block-diagonal lift of the 8x8 whitener, RMS-normalised over interaction windows.

    The 1/sqrt(n_win) stops a three-obstacle scene from looking more diverse than a
    one-obstacle scene purely because it has more blocks: without it the same threshold means
    "2 sigma" on one scene and "1.15 sigma per window" on the other.  Channels outside
    `channels` are zeroed, so the reported coverage is over exactly the channels named.
    """
    keep = np.zeros((N_CHANNELS, N_CHANNELS))
    for c in channels:
        keep[c, c] = 1.0
    B = keep @ W8 @ keep
    out = np.zeros((n_win * N_CHANNELS, n_win * N_CHANNELS))
    for w in range(n_win):
        s = slice(w * N_CHANNELS, (w + 1) * N_CHANNELS)
        out[s, s] = B
    return out / np.sqrt(max(n_win, 1))


# =======================================================================================
# one scene
# =======================================================================================

def run_scene(runner, sc, args) -> dict:
    t0 = time.time()
    p = plan(sc, "adaptive")
    t_plan = time.time() - t0
    if not p.feasible:
        return {"scene_id": sc.scene_id, "family": sc.family, "feasible_route": False,
                "refusal": p.refusal, "planner_cpu_s": {"plan": t_plan}}
    t0 = time.time()
    ctx = be.route_context(sc, p, fps=runner.fps, speed=args.speed)
    t_ctx = time.time() - t0
    if not ctx.feasible:
        return {"scene_id": sc.scene_id, "family": sc.family, "feasible_route": True,
                "context_feasible": False, "note": ctx.note,
                "planner_cpu_s": {"plan": t_plan, "context": t_ctx}}

    seeds = [100 + i for i in range(args.n_seeds)]
    ev = be.BodyEvaluator(runner, ctx, seeds=seeds, batch=args.batch,
                          diffusion_steps=args.diffusion_steps)
    control = ev.n_ardy                      # 1 nominal + n_seeds matched controls
    cpu = {"plan": t_plan, "context": t_ctx}
    charged, evals, proposals, search_arms = {}, {}, {}, {}

    # -- CPU baselines: propose for free, pay K * n_seeds to verify ----------------------
    for name in CPU_BASELINES:
        t0 = time.time()
        progs = be.BASELINES[name](sc, p, args.K, None, ctx=ctx)
        cpu[name] = time.time() - t0
        before = ev.n_ardy
        evals[name] = ev.evaluate(progs, [name] * len(progs))
        charged[name] = ev.n_ardy - before + control
        proposals[name] = progs

    # -- NULL-SEED: the SAME program K times, on DISJOINT seed blocks --------------------
    t0 = time.time()
    nulls = be.null_seed_baseline(sc, p, args.K, None, ctx=ctx)
    cpu["NULL-SEED"] = time.time() - t0
    nseeds = [[900 + k * args.n_seeds + i for i in range(args.n_seeds)]
              for k in range(len(nulls))]
    before = ev.n_ardy
    evals["NULL-SEED"] = ev.evaluate(nulls, ["NULL-SEED"] * len(nulls), seeds=nseeds)
    charged["NULL-SEED"] = ev.n_ardy - before + control
    proposals["NULL-SEED"] = nulls

    # -- searching baselines, at EQUAL ARDY CALLS PER K ----------------------------------
    # A CPU baseline at budget K costs (1 nominal + n_seeds controls) + K * n_seeds clips.  D
    # and the composite are given exactly that, SEPARATELY FOR EACH K, so the comparison is at
    # equal calls and their search budget is what a K-set proposer would be allowed to spend.
    # Giving them the K=8 budget and reading off prefixes would silently hand them 4x the calls
    # at K=2.  The `x{mult}` arms answer the guidance's fourth row -- "covers everything but
    # needs far more ARDY calls" -- which cannot be read off an equal-call table.
    for name in SEARCH_BASELINES:
        fn = be.BASELINES[name]
        for mult in args.budget_mults:
            arm = name if mult == 1 else f"{name}(x{mult})"
            per_k, spent = {}, 0
            t0 = time.time()
            for K in [k for k in KS if k <= args.K]:
                room = int(mult * (control + K * args.n_seeds)) - control
                before = ev.n_ardy
                res = fn(sc, p, K, runner, ctx=ctx, budget=ev.n_ardy + max(room, args.n_seeds),
                         n_seeds=args.n_seeds, batch=args.batch, evaluator=ev,
                         sigma=be.SEED_SIGMA_PRIOR, eps=2.0, channels=args.channels,
                         seed=args.seed, return_evaluations=True)
                per_k[K] = res[0] if isinstance(res, tuple) else res
                spent = max(spent, ev.n_ardy - before + control)
            cpu[arm] = time.time() - t0
            search_arms[arm] = per_k
            evals[arm] = per_k.get(args.K, [])
            proposals[arm] = [e.program for e in evals[arm]]
            charged[arm] = int(mult * (control + args.K * args.n_seeds))

    # -- the independent reference arm ---------------------------------------------------
    t0 = time.time()
    refs = be.reference_random_restart(sc, p, args.n_reference, ctx=ctx, seed=args.seed)
    cpu["REF-RANDOM"] = time.time() - t0
    before = ev.n_ardy
    evals["REF-RANDOM"] = ev.evaluate(refs, ["REF-RANDOM"] * len(refs))
    charged["REF-RANDOM"] = ev.n_ardy - before + control

    return {"scene_id": sc.scene_id, "family": sc.family, "feasible_route": True,
            "context_feasible": True, "n_windows": ctx.n_windows,
            "n_interactions": len(ev.interactions), "n_free_segments": len(ctx.free_segments),
            "route_len_m": ctx.tube.length, "T": ctx.T, "planner_cpu_s": cpu,
            "ardy_charged": charged, "ardy_unique_clips": ev.n_unique,
            "gpu_s": round(ev.gpu_s, 2), "control_calls": control,
            "evals": evals, "search_arms": search_arms}


# =======================================================================================
# scoring
# =======================================================================================

def null_distances(rec: dict, W8: np.ndarray, args) -> tuple[list, list]:
    """Distances between realisations of ONE program: what `eps` must sit above."""
    nulls = rec.get("evals", {}).get("NULL-SEED", [])
    if len(nulls) < 2:
        return [], []
    W = block_whitener_from(W8, rec["n_interactions"], be.CHANNEL_SETS[args.channels])
    mm = [d_morph(nulls[i].mu, nulls[j].mu, W)
          for i in range(len(nulls)) for j in range(i + 1, len(nulls))]
    ss = [d_morph(e.deltas[i], e.deltas[j], W) for e in nulls
          for i in range(len(e.deltas)) for j in range(i + 1, len(e.deltas))]
    return mm, ss


def score_scene(rec: dict, W8: np.ndarray, args, eps: float) -> dict:
    evals = rec["evals"]
    W = block_whitener_from(W8, rec["n_interactions"], be.CHANNEL_SETS[args.channels])

    def addressable(e):
        return (len(e.mu) and e.feasible_rate >= args.min_feasible_rate
                and e.stability >= args.min_stability)

    # NULL-SEED is excluded from the reference pool by construction: it proposes nothing, and
    # admitting K copies of one program would let the prior's own scatter manufacture reference
    # members for every baseline to chase.
    pool = [(a, e) for a in evals if a != "NULL-SEED" for e in evals[a] if addressable(e)]
    out = {"scene_id": rec["scene_id"], "family": rec["family"],
           "n_interactions": rec["n_interactions"]}
    if not pool:
        out.update({"scorable": False, "n_reference": 0})
        return out
    mus = np.array([e.mu for _, e in pool])
    net = epsilon_net(mus, W, eps)
    ref_mus, ref_sigs = mus[net], {pool[i][1].signature for i in net}
    out.update({"scorable": True, "n_pool": len(pool), "n_reference": len(net),
                "n_reference_signatures": len(ref_sigs),
                "reference_contributors": dict(Counter(pool[i][0] for i in net)),
                "baselines": {}})

    for arm, elist in evals.items():
        loo_pool = [(a, e) for a, e in pool if a != arm]
        loo = np.array([e.mu for _, e in loo_pool]) if loo_pool else np.zeros((0, mus.shape[1]))
        loo_net = epsilon_net(loo, W, eps) if len(loo) else []
        loo_mus = loo[loo_net] if len(loo_net) else np.zeros((0, mus.shape[1]))
        loo_sigs = {loo_pool[i][1].signature for i in loo_net}
        cont, disc, cont_loo, disc_loo = {}, {}, {}, {}
        for K in [k for k in KS if k <= args.K]:
            src = rec.get("search_arms", {}).get(arm, {}).get(K, elist)
            head = [e for e in src[:K] if addressable(e)]
            hm = np.array([e.mu for e in head]) if head else np.zeros((0, mus.shape[1]))
            cont[K] = morph_recall_at_k(ref_mus, hm, W, eps, K) if len(ref_mus) else float("nan")
            disc[K] = discrete_recall(ref_sigs, [e.signature for e in head])
            cont_loo[K] = (morph_recall_at_k(loo_mus, hm, W, eps, K) if len(loo_mus)
                           else float("nan"))
            disc_loo[K] = discrete_recall(loo_sigs, [e.signature for e in head])
        out["baselines"][arm] = {
            "n_proposed": len(elist),
            "n_feasible": int(sum(e.feasible_rate >= args.min_feasible_rate for e in elist)),
            "n_addressable": int(sum(addressable(e) for e in elist)),
            "mean_stability": float(np.mean([e.stability for e in elist])) if elist else 0.0,
            "mean_feasible_rate": float(np.mean([e.feasible_rate for e in elist])) if elist else 0.0,
            "max_cost_ratio": float(max([e.cost_ratio for e in elist], default=float("nan"))),
            # Diagnostic, not a filter: `decode` dilates each slot by LEAD_S per side, so two
            # slots on different axes within ~1.6 s re-merge into a simultaneous duck-and-tuck
            # request that only `Envelope.half_width(dip, tuck)` -- envelope.py:79, "the
            # weakest link in the envelope" -- certifies.  If the infeasible candidates
            # concentrate here, the enumerator is not wrong, the cross-axis envelope is.
            "frac_overlap": (float(np.mean([e.overlap for e in elist])) if elist else 0.0),
            "frac_overlap_given_infeasible": (
                float(np.mean([e.overlap for e in elist
                               if e.feasible_rate < args.min_feasible_rate]))
                if any(e.feasible_rate < args.min_feasible_rate for e in elist) else None),
            "continuous": cont, "discrete": disc,
            "continuous_loo": cont_loo, "discrete_loo": disc_loo,
            "auk_discrete": area_under_k(disc), "auk_continuous": area_under_k(cont),
            "auk_discrete_loo": area_under_k(disc_loo),
            "auk_continuous_loo": area_under_k(cont_loo),
            "ardy_charged": rec["ardy_charged"].get(arm),
            "planner_cpu_s": rec["planner_cpu_s"].get(arm)}

    nulls = evals.get("NULL-SEED", [])
    if len(nulls) >= 2:
        nm = np.array([e.mu for e in nulls])
        dd = [d_morph(nm[i], nm[j], W) for i in range(len(nm)) for j in range(i + 1, len(nm))]
        ss = [d_morph(e.deltas[i], e.deltas[j], W) for e in nulls
              for i in range(len(e.deltas)) for j in range(i + 1, len(e.deltas))]
        out["null_seed"] = {
            "pairwise_mean_d_morph": float(np.mean(dd)),
            "pairwise_frac_above_eps": float(np.mean([d > eps for d in dd])),
            "single_seed_pair_mean": float(np.mean(ss)) if ss else None,
            "single_seed_frac_above_eps": float(np.mean([d > eps for d in ss])) if ss else None,
            # How many "distinct bodies" ONE program would appear to contribute if it were
            # admitted to the reference.  1 is correct.  K means the metric is measuring
            # ARDY's sampler and every coverage number in the table is void.
            "internal_eps_net": int(len(epsilon_net(nm, W, eps))),
            "n_null": int(len(nulls))}
    return out


def summarise(records, scored, sigma_hat, args, eps) -> dict:
    ok = [s for s in scored if s.get("scorable")]
    # Macro-averages are taken over scenes whose reference net has >= 2 members.  On a scene
    # where the net is a single point every arm scores 1.000 by returning the shipped planner's
    # own answer, and averaging those in would move the table without measuring anything.
    nz = [s for s in ok if s["n_reference"] >= 2]
    arms = sorted({a for s in ok for a in s["baselines"]})
    summary = {
        "experiment": "exp005g_body_enumerate", "eps_sigma": eps,
        "n_scenes": len(records),
        "n_route_feasible": sum(1 for r in records if r.get("feasible_route")),
        "n_scored": len(ok), "n_scored_nontrivial": len(nz),
        "n_seeds": args.n_seeds, "K": args.K, "channels": args.channels,
        "min_stability": args.min_stability, "min_feasible_rate": args.min_feasible_rate,
        "seed_sigma_measured_in_run": dict(zip(CHANNELS, np.round(sigma_hat, 4).tolist())),
        "seed_sigma_prior_exp005f": dict(zip(CHANNELS,
                                             np.round(be.SEED_SIGMA_PRIOR, 4).tolist())),
        "mean_reference_net_size": float(np.mean([s["n_reference"] for s in ok])) if ok else 0.0,
        "mean_reference_signatures": (float(np.mean([s["n_reference_signatures"] for s in ok]))
                                      if ok else 0.0),
        "reference_contributors": dict(Counter(
            {a: sum(s["reference_contributors"].get(a, 0) for s in ok) for a in arms})),
        "baselines": {}}
    for arm in arms:
        rows = [s["baselines"][arm] for s in nz if arm in s["baselines"]]
        allr = [s["baselines"][arm] for s in ok if arm in s["baselines"]]
        if not rows:
            continue

        def m(key, K=None, src=rows):
            v = [(r[key][K] if K is not None else r[key]) for r in src if r.get(key) is not None]
            v = [x for x in v if x is not None and x == x]
            return float(np.mean(v)) if v else float("nan")

        summary["baselines"][arm] = {
            "discrete_recall": {K: m("discrete", K) for K in KS if K <= args.K},
            "continuous_recall": {K: m("continuous", K) for K in KS if K <= args.K},
            "discrete_recall_loo": {K: m("discrete_loo", K) for K in KS if K <= args.K},
            "continuous_recall_loo": {K: m("continuous_loo", K) for K in KS if K <= args.K},
            "auk_discrete": m("auk_discrete"), "auk_continuous": m("auk_continuous"),
            "auk_discrete_loo": m("auk_discrete_loo"),
            "auk_continuous_loo": m("auk_continuous_loo"),
            "ardy_calls_at_K": m("ardy_charged", src=allr),
            "planner_cpu_s": m("planner_cpu_s", src=allr),
            "mean_n_proposed": m("n_proposed", src=allr),
            "mean_n_addressable": m("n_addressable", src=allr),
            "mean_stability": m("mean_stability", src=allr),
            "mean_feasible_rate": m("mean_feasible_rate", src=allr),
            "frac_overlap": m("frac_overlap", src=allr),
            "frac_overlap_given_infeasible": m("frac_overlap_given_infeasible", src=allr),
            "max_cost_ratio_p90": float(np.nanpercentile(
                [r["max_cost_ratio"] for r in allr], 90)) if allr else float("nan")}
    nulls = [s["null_seed"] for s in ok if "null_seed" in s]
    if nulls:
        summary["null_seed_control"] = {
            k: float(np.mean([n[k] for n in nulls if n.get(k) is not None]))
            for k in ("pairwise_mean_d_morph", "pairwise_frac_above_eps",
                      "single_seed_pair_mean", "single_seed_frac_above_eps",
                      "internal_eps_net")}
    return summary


def print_table(summary: dict, args, label: str) -> None:
    ks = [k for k in KS if k <= args.K]
    hdr = (f"{'baseline':22s} " + " ".join(f"disc@{k}" for k in ks) + "   "
           + " ".join(f"cont@{k}" for k in ks)
           + "    AUKd   AUKc  AUKc_LOO   calls   cpu_s  addr  feas")
    print(f"\n=== {label}: eps = {summary['eps_sigma']:.2f} sigma, "
          f"{summary['n_scored_nontrivial']} scenes with >= 2 reference members "
          f"(mean net {summary['mean_reference_net_size']:.1f}) ===")
    print(hdr)
    print("-" * len(hdr))
    for arm, b in summary["baselines"].items():
        print(f"{arm:22s} "
              + " ".join(f"{b['discrete_recall'][k]:6.3f}" for k in ks) + "   "
              + " ".join(f"{b['continuous_recall'][k]:6.3f}" for k in ks)
              + f"  {b['auk_discrete']:6.3f} {b['auk_continuous']:6.3f} "
                f"{b['auk_continuous_loo']:8.3f} {b['ardy_calls_at_K']:7.0f} "
                f"{b['planner_cpu_s']:7.3f} {b['mean_n_addressable']:5.1f} "
                f"{b['mean_feasible_rate']:5.2f}")
    print("  disc/cont = discrete-mode and continuous MorphRecall@K, macro-averaged per scene."
          "  AUK* = area under the K grid.  LOO = scored against a reference rebuilt WITHOUT\n"
          "  this arm's own contributions.  calls = ARDY clips charged at K=8 (equal across "
          "arms by construction).  addr = mean candidates of 8 that are\n"
          "  addressable (feasible on >= min_feasible_rate of seeds AND Stability >= 0.8) -- a "
          "baseline whose candidates ARDY refuses cannot cover anything,\n"
          "  which is why an arm may score below NULL-SEED on cont while proposing far more "
          "diversity.  NULL-SEED's call count is higher because its\n"
          "  disjoint seeds need their own matched controls; it is a control arm, not a "
          "competitor, and its cont column is a floor, not a score.")
    n = summary.get("null_seed_control")
    if n:
        print(f"\nNULL-SEED control (the SAME program, {args.K} times, disjoint seeds): "
              f"mean pairwise d_morph between SEED MEANS {n['pairwise_mean_d_morph']:.2f} sigma, "
              f"{n['pairwise_frac_above_eps']*100:.0f} % above eps; between SINGLE seeds "
              f"{n['single_seed_pair_mean']:.2f} sigma. "
              f"ONE program deduplicates to {n['internal_eps_net']:.2f} eps-net members "
              f"(1.00 is correct).")
        if n["internal_eps_net"] > 1.25:
            print("  *** WARNING: eps is inside ARDY's own scatter.  One program looks like "
                  f"{n['internal_eps_net']:.1f} distinct bodies, so the continuous columns are "
                  "measuring the sampler.  Raise eps, raise n_seeds, or do not report them.")


# =======================================================================================

def _status(rec: dict) -> str:
    if "error" in rec:
        return "ERROR " + rec["error"][:70]
    if not rec.get("feasible_route"):
        return f"route refused ({(rec.get('refusal') or {}).get('functional', '?')})"
    if not rec.get("context_feasible"):
        return f"context infeasible: {rec.get('note', '')}"
    return f"clips {rec['ardy_unique_clips']:4d}  gpu {rec['gpu_s']:6.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/exp005g")
    ap.add_argument("--seeds_per_rung", type=int, default=4)       # 4 -> the 128-scene suite
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_reference", type=int, default=24)
    ap.add_argument("--budget_mults", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--eps", default="auto",
                    help="'auto' calibrates eps on the NULL-SEED arm; or a float in sigma")
    ap.add_argument("--eps_quantile", type=float, default=0.99)
    ap.add_argument("--min_stability", type=float, default=0.8)    # guidance, pre-committed
    ap.add_argument("--min_feasible_rate", type=float, default=0.75)
    ap.add_argument("--channels", default="all", choices=sorted(be.CHANNEL_SETS))
    ap.add_argument("--speed", type=float, default=be.SPEED)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="6 scenes, 2 seeds, 8 reference programs, no x4 arms")
    args = ap.parse_args()
    if args.smoke:                      # smoke sets only what the caller did not
        if "--n_seeds" not in sys.argv:
            args.n_seeds = 2
        if "--n_reference" not in sys.argv:
            args.n_reference = 8
        if "--budget_mults" not in sys.argv:
            args.budget_mults = [1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scenes = ([BUILDERS[f](v, 0) for f, v in SMOKE_SCENES] if args.smoke
              else build_suite(seeds_per_rung=args.seeds_per_rung, families=args.families))
    if args.limit:
        scenes = scenes[:args.limit]

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    records = []
    for n, sc in enumerate(scenes):
        try:
            rec = run_scene(runner, sc, args)
        except Exception as exc:                     # one bad scene must not lose the run
            rec = {"scene_id": sc.scene_id, "family": sc.family, "error": repr(exc)}
        records.append(rec)
        print(f"[{n+1}/{len(scenes)}] {sc.scene_id[:38]:38s} {_status(rec)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    live = [r for r in records if r.get("context_feasible")]
    all_evals = [e for r in live for lst in r["evals"].values() for e in lst]
    W8 = pooled_whitener(all_evals)
    sigma_hat = 1.0 / np.sqrt(np.diag(W8 @ W8))

    null_mm, null_ss = [], []
    for r in live:
        mm, ss = null_distances(r, W8, args)
        null_mm += mm
        null_ss += ss
    eps_auto = float(np.percentile(null_mm, args.eps_quantile * 100)) if null_mm else 2.0
    eps_primary = eps_auto if args.eps == "auto" else float(args.eps)

    tables = [("PRIMARY (eps calibrated on the null arm)", eps_primary)]
    if abs(eps_primary - 2.0) > 1e-6:
        tables.append(("ABLATION (pre-committed eps = 2 sigma)", 2.0))

    results = {}
    for label, eps in tables:
        scored = [score_scene(r, W8, args, eps) for r in live]
        results[label] = (eps, scored, summarise(records, scored, sigma_hat, args, eps))

    label0, (eps0, scored0, summary) = tables[0][0], results[tables[0][0]]
    summary["eps_calibration"] = {
        "mode": args.eps, "quantile": args.eps_quantile, "eps_used": eps0,
        "eps_precommitted": 2.0,
        "null_seed_mean_pairs": {"n": len(null_mm),
                                 "mean": float(np.mean(null_mm)) if null_mm else None,
                                 "q50": float(np.percentile(null_mm, 50)) if null_mm else None,
                                 "q95": float(np.percentile(null_mm, 95)) if null_mm else None},
        "null_single_seed_pairs": {"n": len(null_ss),
                                   "mean": float(np.mean(null_ss)) if null_ss else None,
                                   "q95": (float(np.percentile(null_ss, 95)) if null_ss
                                           else None)}}
    if len(tables) > 1:
        summary["at_precommitted_eps_2sigma"] = {
            k: results[tables[1][0]][2][k]
            for k in ("baselines", "mean_reference_net_size", "null_seed_control")
            if k in results[tables[1][0]][2]}
    summary["ardy_calls_total_charged"] = int(sum(sum(r["ardy_charged"].values())
                                                  for r in live))
    summary["ardy_clips_generated"] = int(sum(r["ardy_unique_clips"] for r in live))
    summary["gpu_s_total"] = float(sum(r["gpu_s"] for r in live))
    summary["planner_cpu_total_s"] = float(sum(sum(r["planner_cpu_s"].values())
                                               for r in records if r.get("planner_cpu_s")))
    summary["wall_clock_s"] = round(time.time() - t0, 1)

    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / "per_scene.json", "w") as fh:
        json.dump(scored0, fh, indent=2)

    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k not in ("baselines", "at_precommitted_eps_2sigma")},
                            indent=2))
    for label, eps in tables:
        print_table(results[label][2], args, label)


if __name__ == "__main__":
    main()
