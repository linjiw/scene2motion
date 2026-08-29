"""EXP-005i: which bottleneck is it -- candidate SUPPORT, candidate SELECTION, or ARDY?

Why this exists
---------------
EXP-005g answered its question ("can a classical enumerator cover the addressable body set at
K=8?") with a clear no: 0.59 discrete / 0.38 continuous at equal calls against a 0.36 / 0.21
resampling floor, nowhere near the pre-committed 0.90.  But a coverage number of 0.59 is
consistent with three completely different worlds, and they call for three different next
methods:

    SUPPORT     the union of every proposer's candidates does not contain programs that
                realise the missing modes            -> learn a mode-conditioned inverse program
    SELECTION   it does contain them, and the heuristics rank the wrong eight
                                                     -> learn a reranker / addressability model
    ARDY        no program addresses those modes reliably at all
                                                     -> adapt the prior, or drop the claim

Every number here is a re-analysis of `candidates.jsonl`, the per-candidate ledger EXP-005g now
emits.  No ARDY calls.  That file exists because the gate's first full run computed all of this
evidence in memory and wrote only per-arm averages, which put five separate analyses behind a
second 61-minute GPU pass.

Selection and evaluation seeds are DISJOINT throughout
-----------------------------------------------------
D-REFINE and COMPOSITE choose their candidates by looking at ARDY outcomes.  Scoring them on
those same outcomes would report post-selection luck as reliability.  So everything below is
scored on the held-out block (seeds 500+), which no arm could select on, and the
selection-versus-held-out shrinkage is reported explicitly in section A as a check on whether
the gate's headline numbers were inflated.

Sections
--------
    A  held-out transfer            did feasibility/stability survive disjoint seeds?
    B  addressability               and how many seeds it would take to CERTIFY it
    C  POOL-ORACLE@K                support vs selection, the decision this experiment exists for
    D  ValidDiversityYield@B        useful bodies per ARDY call, rejects included
    E  program-vs-seed allocation   8x1 vs 4x2 vs 2x4 vs 1x8 at a fixed budget
    F  commanded vs realised        what the frozen prior can actually be asked to do
    G  yaw ablation                 an uncommandable channel in the counted alphabet
    H  failure decomposition        why each missed mode was missed
    I  distance sensitivity         diagonal q99 scaling vs full-covariance Mahalanobis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.morphology import (N_CHANNELS, active_set, d_morph,  # noqa: E402
                                     epsilon_net, stability)

MIN_STAB, MIN_FEAS = 0.8, 0.75            # pre-committed in EXP-005g, unchanged here
KS = (1, 2, 4, 8)
# The independent random-restart arm. It DEFINES the reference in section C and is therefore
# not scored there: an arm cannot be graded against a target it alone wrote.
REF_ARM = "REF-RANDOM"
ARMS_SCORED = ("A-KBEST", "B-NOGOOD", "C-WSWEEP", "D-REFINE", "COMPOSITE",
               "D-REFINE(x4)", "COMPOSITE(x4)", "REF-RANDOM")


# =======================================================================================
# signatures
# =======================================================================================

def sig_key(sig, keep_yaw: bool = True) -> tuple:
    """The realised active set as a hashable mode label.

    The per-interaction tuple is `(duck, tuck, liftL, liftR, order, yaw)`.  `tuck` and `order`
    are already retired -- EXP-005f measured tuck's effect below the prior's own width scatter
    and showed a bare sign comparison assigns an order to every clip -- so they are dropped
    from the label here; they are audited as negative controls in section G, where they must
    stay at zero.  `yaw` is kept in the PRIMARY label because the gate was pre-committed with
    it, and dropped in the section G ablation because no program can request it.
    """
    out = []
    for s in sig:
        duck, tuck, liftL, liftR, order, yaw = (list(s) + [0] * 6)[:6]
        out.append((bool(duck), bool(liftL), bool(liftR), bool(yaw)) if keep_yaw
                   else (bool(duck), bool(liftL), bool(liftR)))
    return tuple(out)


def block(r: dict, which: str) -> dict | None:
    return r.get(which)


def addressable(ev: dict | None) -> bool:
    return bool(ev and ev["feasible_rate"] >= MIN_FEAS and ev["stability"] >= MIN_STAB)


def retheshold(rows: list[dict], q99: np.ndarray) -> int:
    """Recompute every realised active set from the stored deltas at the MEASURED q99.

    `BodyEvaluator.evaluate` defaults to `noise_q99 = SEED_SIGMA_PRIOR * 2.33` and EXP-005g
    never overrides it, so the gate labelled a channel "active" against a hardcoded Gaussian
    approximation rather than against the quantile actually measured in EXP-005f.  It sits
    26-42 % BELOW the measured q99 on every channel, which makes channels fire too easily and
    inflates both the mode count and the reference size -- the third time a hardcoded noise
    scale has appeared in this project, and the third time it favoured more apparent diversity.

    This is repairable without a single ARDY call only because the ledger stores the raw
    per-seed deltas rather than just the labels derived from them.  That is the whole argument
    for recording evidence instead of conclusions.
    """
    n = 0
    for r in rows:
        for which in ("selection", "heldout"):
            ev = r.get(which)
            if not ev:
                continue
            D = np.asarray(ev["deltas"], float)
            if D.ndim != 2:
                continue
            k = r["n_interactions"]
            sigs = [tuple(active_set(d[j * N_CHANNELS:(j + 1) * N_CHANNELS], q99)
                          for j in range(k)) for d in D]
            ev["signatures"] = [[[bool(x) if not isinstance(x, int) else int(x) for x in per]
                                 for per in sig] for sig in sigs]
            modal = Counter(sigs).most_common(1)[0][0] if sigs else ()
            ev["signature"] = [[bool(x) if not isinstance(x, int) else int(x) for x in per]
                               for per in modal]
            ev["stability"] = float(stability(sigs))
            n += 1
    return n


def certifying_n(tau: float, alpha: float = 0.05) -> int:
    """Seeds needed before a UNANIMOUS run certifies P(success) >= tau at level alpha.

    The guidance defines addressability with a LOWER confidence bound, `P_lower >= tau`.  With
    k successes in n trials the Clopper-Pearson lower bound at k = n is `alpha**(1/n)`, so
    certifying tau = 0.8 needs `alpha**(1/n) >= tau`, i.e. n >= ln(alpha)/ln(tau) = 14.  At the
    gate's n = 4 the best attainable lower bound is 0.05**(1/4) = 0.47: four seeds cannot
    certify the pre-committed threshold even in principle, only estimate it.  Every
    "addressable" count below is therefore a POINT ESTIMATE at tau, and is labelled as one.
    """
    return int(np.ceil(np.log(alpha) / np.log(tau)))


# =======================================================================================
# per-scene assembly
# =======================================================================================

def scene_groups(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r["scene_id"]].append(r)
    return dict(g)


def pooled_whitener(rows: list[dict], n_int: int, ridge: float = 1e-6) -> np.ndarray:
    """Whitener from same-program, different-seed residuals, over BOTH seed blocks."""
    covs = []
    for r in rows:
        for which in ("selection", "heldout"):
            ev = block(r, which)
            if not ev:
                continue
            D = np.asarray(ev["deltas"], float)
            if D.ndim != 2 or len(D) < 2 or D.shape[1] != n_int * N_CHANNELS:
                continue
            for w in range(n_int):
                covs.append(np.cov(D[:, w * N_CHANNELS:(w + 1) * N_CHANNELS].T))
    S = np.mean([c for c in covs if np.all(np.isfinite(c))], axis=0) if covs \
        else np.eye(N_CHANNELS)
    S = S + ridge * np.eye(N_CHANNELS) * max(1.0, float(np.trace(S)) / N_CHANNELS)
    w, V = np.linalg.eigh(S)
    W1 = V @ np.diag(np.maximum(w, ridge) ** -0.5) @ V.T
    # The 1/sqrt(n_int) RMS normalisation is NOT decoration: exp005g's `block_whitener_from`
    # applies it, and `eps` is imported from that run's calibration.  Dropping it here would
    # inflate every distance by sqrt(n_int) against a threshold calibrated without it, so a
    # three-obstacle scene would read as more diverse purely for having more blocks.  It is a
    # no-op on the current suite (every scored scene has n_interactions == 1) and would have
    # been a silent unit mismatch the moment a multi-obstacle family was scored.
    return np.kron(np.eye(n_int), W1) / np.sqrt(max(n_int, 1))


def diag_whitener(rows: list[dict], n_int: int) -> np.ndarray:
    """The DIAGONAL q99 scaling -- what the gate used -- for the section I comparison."""
    resid = []
    for r in rows:
        for which in ("selection", "heldout"):
            ev = block(r, which)
            if not ev:
                continue
            D = np.asarray(ev["deltas"], float)
            if D.ndim == 2 and len(D) >= 2 and D.shape[1] == n_int * N_CHANNELS:
                resid.append(np.abs(D - D.mean(axis=0)))
    if not resid:
        return np.eye(n_int * N_CHANNELS)
    q = np.percentile(np.concatenate(resid), 99, axis=0)
    return np.diag(1.0 / np.maximum(q, 1e-6)) / np.sqrt(max(n_int, 1))


# =======================================================================================
# coverage
# =======================================================================================

def coverage(ref_modes: set, chosen: list[dict], which: str, keep_yaw=True) -> float:
    """Fraction of reference modes ADDRESSABLY hit by `chosen`, scored on `which` seeds."""
    if not ref_modes:
        return float("nan")
    hit = {sig_key(block(c, which)["signature"], keep_yaw) for c in chosen
           if addressable(block(c, which))}
    return len(ref_modes & hit) / len(ref_modes)


def cont_coverage(ref_mu: np.ndarray, chosen: list[dict], which: str,
                  W: np.ndarray, eps: float) -> float:
    if not len(ref_mu):
        return float("nan")
    P = [np.asarray(block(c, which)["mu"], float) for c in chosen
         if addressable(block(c, which))]
    if not P:
        return 0.0
    return float(np.mean([min(d_morph(r, q, W) for q in P) <= eps for r in ref_mu]))


def greedy_oracle(pool: list[dict], ref_modes: set, K: int, pick_on: str,
                  score_on: str, keep_yaw=True) -> float:
    """Best size-K subset by greedy coverage. Submodular, so greedy is a LOWER bound.

    That direction matters: if even this understated oracle clears 0.90, the support is
    genuinely there and the bottleneck is selection.
    """
    if not ref_modes:
        return float("nan")
    chosen, got = [], set()
    avail = list(pool)
    for _ in range(K):
        best, best_gain = None, -1
        for c in avail:
            ev = block(c, pick_on)
            if not addressable(ev):
                continue
            gain = len({sig_key(ev["signature"], keep_yaw)} - got)
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None:
            break
        avail.remove(best)
        chosen.append(best)
        got |= {sig_key(block(best, pick_on)["signature"], keep_yaw)}
    return coverage(ref_modes, chosen, score_on, keep_yaw)


# =======================================================================================
# main
# =======================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="outputs/exp005g/candidates.jsonl")
    ap.add_argument("--receipt", default="outputs/exp005g/receipt.json")
    ap.add_argument("--out", default="outputs/exp005i")
    ap.add_argument("--eps", type=float, default=None,
                    help="default: the eps the gate calibrated on its null arm")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--noise_q99", default="outputs/exp005f/receipt.json",
                    help="re-threshold active sets at the MEASURED q99; 'none' keeps the "
                         "gate's hardcoded 2.33*sigma labels")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.ledger)]
    eps = args.eps
    if eps is None:
        eps = json.load(open(args.receipt))["eps_calibration"]["eps_used"]
    if args.noise_q99 != "none":
        q99 = np.array(json.load(open(args.noise_q99))["seed_noise_q99"], float)
        before = [sig_key(r["heldout"]["signature"]) for r in rows if r.get("heldout")]
        n = retheshold(rows, q99)
        after = [sig_key(r["heldout"]["signature"]) for r in rows if r.get("heldout")]
        changed = sum(a != b for a, b in zip(before, after))
        print(f"re-thresholded {n} evaluations at the MEASURED q99 "
              f"(the gate used a hardcoded 2.33*sigma, 26-42 % lower): "
              f"{changed}/{len(before)} held-out modal signatures changed, "
              f"distinct modes {len(set(before))} -> {len(set(after))}")
    groups = scene_groups(rows)
    rng = np.random.default_rng(0)
    print(f"{len(rows)} ledger rows over {len(groups)} scenes; eps = {eps:.2f} calibrated units")
    print(f"selection seeds: {sorted({s for r in rows for s in r['selection']['seeds']})[:6]}..."
          f"  held-out: {sorted({s for r in rows if r['heldout'] for s in r['heldout']['seeds']})}")

    report: dict = {"experiment": "exp005i_ledger_analysis", "eps": eps,
                    "n_rows": len(rows), "n_scenes": len(groups)}

    # -- A. held-out transfer -----------------------------------------------------------
    print("\n=== A. Selection -> held-out transfer (post-selection shrinkage) ===")
    print(f"{'arm':16s} {'n':>4s} {'feas sel':>9s} {'feas held':>10s} {'stab sel':>9s} "
          f"{'stab held':>10s} {'addr sel':>9s} {'addr held':>10s}  {'sig agree':>9s}")
    A = {}
    for arm in ARMS_SCORED + ("NULL-SEED",):
        a = [r for r in rows if r["arm"] == arm and r["heldout"]]
        if not a:
            continue
        fs = np.mean([r["selection"]["feasible_rate"] for r in a])
        fh = np.mean([r["heldout"]["feasible_rate"] for r in a])
        ss = np.mean([r["selection"]["stability"] for r in a])
        sh = np.mean([r["heldout"]["stability"] for r in a])
        ads = np.mean([addressable(r["selection"]) for r in a])
        adh = np.mean([addressable(r["heldout"]) for r in a])
        agree = np.mean([sig_key(r["selection"]["signature"])
                         == sig_key(r["heldout"]["signature"]) for r in a])
        A[arm] = {"n": len(a), "feas_sel": fs, "feas_held": fh, "stab_sel": ss,
                  "stab_held": sh, "addr_sel": ads, "addr_held": adh, "sig_agree": agree}
        print(f"{arm:16s} {len(a):4d} {fs:9.3f} {fh:10.3f} {ss:9.3f} {sh:10.3f} "
              f"{ads:9.3f} {adh:10.3f}  {agree:9.3f}")
    report["A_transfer"] = A
    print("  sig agree = the modal active set is the SAME on both disjoint seed blocks.  This "
          "is the\n  ceiling on how well ANY method can command a mode: a program whose own "
          "modal outcome\n  changes between seed blocks cannot be a reliable instruction.")

    # -- B. addressability, and what it would take to certify it ------------------------
    n_cert = certifying_n(MIN_STAB)
    n_have = len(rows[0]["heldout"]["seeds"]) if rows[0]["heldout"] else 0
    print(f"\n=== B. Addressability at tau = {MIN_STAB} ===")
    print(f"held-out seeds available: {n_have}.  Clopper-Pearson lower bound at {n_have}/"
          f"{n_have} successes = {0.05 ** (1 / max(n_have, 1)):.2f}.")
    print(f"Certifying P >= {MIN_STAB} at 95 % needs n >= {n_cert} unanimous seeds, so every "
          f"'addressable' count\nbelow is a POINT ESTIMATE at tau, never a certificate.  "
          f"That is a limit of the budget, not of the definition.")
    report["B_addressability"] = {"tau": MIN_STAB, "n_heldout": n_have,
                                 "cp_lower_at_unanimous": 0.05 ** (1 / max(n_have, 1)),
                                 "n_needed_to_certify": n_cert}

    # -- C. POOL-ORACLE -----------------------------------------------------------------
    def run_C(keep_yaw: bool, ref_source: str = "union"):
        """Coverage and oracles, once with the pre-committed signature and once without yaw.

        Running it both ways is not a robustness flourish -- it decides the gate's meaning.  If
        no program can REQUEST a yaw (it cannot: program.py:31, and every candidate here shares
        one route byte-for-byte), then reference modes that differ only in their yaw bit are
        modes no proposer could ever be built to hit, and a coverage number that counts them
        charges every arm for an impossibility.  If classical coverage jumps once the bit is
        dropped, most of the gate's shortfall was an artefact of the alphabet, not a missing
        capability.
        """
        per_scene = []
        for sid, g in groups.items():
            n_int = g[0]["n_interactions"]
            W = pooled_whitener(g, n_int)
            pool = [r for r in g if r["arm"] != "NULL-SEED" and r["heldout"]]
            uniq = {r["prog_key"]: r for r in pool}
            cand = list(uniq.values())
            # THE REFERENCE MUST NOT BE THE ORACLE'S OWN POOL.
            # The first version of this built ref_modes as the image of `cand` under sig_key and
            # then let the oracle pick from `cand`, which makes hindsight@K identically
            # min(K, |ref_modes|)/|ref_modes| -- a restatement of the denominator with no
            # dependence on the candidates, the arms, or ARDY.  Every scene here has at most 5
            # reference modes, so it would have printed 1.000 everywhere and "the programs
            # exist, the bottleneck is selection" would have been published as a measurement of
            # a division identity.  The pre-registered falsification (hindsight@8 < 0.75) was
            # unreachable by construction.
            # REF-RANDOM is an independent random-restart body search that shares no rule with
            # any deployable arm, so it can carry the reference alone; the oracle then picks
            # only from the deployable arms, and hindsight@K falls below 1 exactly when the
            # deployable pool has no addressable program realising a mode REF-RANDOM found.
            # That is the support question.  It also gives every scored arm leave-one-out free.
            if ref_source == "refrandom":
                ref = [c for c in cand
                       if c["arm"] == REF_ARM and addressable(block(c, "heldout"))]
                oracle_pool = [c for c in cand if c["arm"] != REF_ARM]
            else:
                ref = [c for c in cand if addressable(block(c, "heldout"))]
                oracle_pool = cand
            ref_modes = {sig_key(block(c, "heldout")["signature"], keep_yaw) for c in ref}
            if len(ref_modes) < 2:
                continue
            mus = np.array([block(c, "heldout")["mu"] for c in ref], float)
            ref_mu = mus[epsilon_net(mus, W, eps)]
            rec = {"scene_id": sid, "family": g[0]["family"], "n_cand": len(cand),
                   "n_ref_modes": len(ref_modes), "n_ref_net": len(ref_mu),
                   "arms": {}, "oracle": {}}
            for K in KS:
                rec["oracle"][f"hindsight@{K}"] = greedy_oracle(oracle_pool, ref_modes, K,
                                                                "heldout", "heldout", keep_yaw)
                rec["oracle"][f"crossfit@{K}"] = greedy_oracle(oracle_pool, ref_modes, K,
                                                               "selection", "heldout", keep_yaw)
                # Printed so the identity the first version accidentally measured stays
                # VISIBLE rather than being quietly deleted: this is what an oracle scored
                # against its own pool would have returned, at every K, on every scene.
                rec["oracle"][f"saturation@{K}"] = min(K, len(ref_modes)) / len(ref_modes)
            rec["ref_source"] = ref_source
            for arm in ARMS_SCORED:
                if arm == REF_ARM and ref_source == "refrandom":
                    continue                      # it defines the reference; it cannot score
                if not any(r["arm"] == arm for r in rows):
                    continue
                mine = sorted([r for r in g if r["arm"] == arm and r["heldout"]],
                              key=lambda r: r["rank"])
                rec["arms"][arm] = {f"@{K}": coverage(ref_modes, mine[:K], "heldout", keep_yaw)
                                    for K in KS}
                rec["arms"][arm]["cont@8"] = cont_coverage(ref_mu, mine[:8], "heldout", W, eps)
            per_scene.append(rec)
        return per_scene

    def boot_ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return (float("nan"), float("nan"))
        idx = rng.integers(0, len(v), (args.boot, len(v)))
        return tuple(np.percentile(v[idx].mean(1), [2.5, 97.5]))

    def print_C(per_scene, title):
        print(f"\n=== C. POOL-ORACLE@K -- {title} ===")
        print(f"{len(per_scene)} scenes with >= 2 addressable reference modes "
              f"(mean {np.mean([r['n_ref_modes'] for r in per_scene]):.1f} modes, "
              f"{np.mean([r['n_cand'] for r in per_scene]):.0f} distinct candidates)")
        print(f"\n{'':22s} {'@1':>7s} {'@2':>7s} {'@4':>7s} {'@8':>7s}   {'95 % CI @8':>16s}")
        for lbl, key in (("POOL-ORACLE hindsight", "hindsight"),
                         ("POOL-ORACLE cross-fit", "crossfit"),
                         ("(saturation identity)", "saturation")):
            v = [[r["oracle"][f"{key}@{K}"] for r in per_scene] for K in KS]
            lo, hi = boot_ci(v[-1])
            print(f"{lbl:22s} " + " ".join(f"{np.nanmean(x):7.3f}" for x in v)
                  + f"   [{lo:.3f}, {hi:.3f}]")
        print("  " + "-" * 66)
        for arm in ARMS_SCORED:
            if not any(arm in r["arms"] for r in per_scene):
                continue
            v = [[r["arms"][arm][f"@{K}"] for r in per_scene] for K in KS]
            lo, hi = boot_ci(v[-1])
            print(f"{arm:22s} " + " ".join(f"{np.nanmean(x):7.3f}" for x in v)
                  + f"   [{lo:.3f}, {hi:.3f}]")

    # TWO references, because one cannot be both clean and well-powered.
    #  union     -- every addressable candidate defines the reference.  Well-powered and the
    #               same convention the gate used, but an oracle that both picks from and is
    #               scored on this pool with the SAME seeds is the min(K,M)/M identity, so the
    #               hindsight row is replaced by that identity, printed under its real name.
    #               `crossfit` stays valid here: it picks on SELECTION-seed signatures and is
    #               scored on HELD-OUT ones, and those differ exactly by the stochasticity
    #               being measured, so it is not self-referential.
    #  refrandom -- REF-RANDOM's addressable modes define the reference and the oracle picks
    #               only from the deployable arms.  This is the actual SUPPORT question and
    #               hindsight is meaningful, at the cost of a much smaller denominator; the
    #               scene count is printed because scenes with <2 reference modes drop out.
    per_scene = run_C(True, "union")
    print_C(per_scene, "PRIMARY, union reference, pre-committed signature (yaw counted)")
    print("  cross-fit = best K chosen from SELECTION-seed outcomes, scored on held-out -> "
          "could a reranker\n  trained on noisy probes reach it?  It is valid against this "
          "reference.\n  (saturation identity) = min(K, n_modes)/n_modes.  An oracle that "
          "picks from the pool that\n  DEFINES the reference, on the seeds it is scored on, "
          "returns exactly this whatever the data\n  says -- the first version of this "
          "experiment reported it as 'POOL-ORACLE hindsight' and\n  pre-registered a "
          "prediction for it.  It is printed to show what this table is NOT.")
    per_scene_rr = run_C(True, "refrandom")
    print_C(per_scene_rr, "SUPPORT, REF-RANDOM reference, deployable arms only")
    print("  Here hindsight is a real measurement: it falls below 1 exactly when the deployable "
          "pool holds\n  no addressable program for a mode an independent random search found. "
          " Fewer scenes qualify.")
    per_scene_ny = run_C(False, "union")
    print_C(per_scene_ny, "SECONDARY, union reference, yaw dropped")
    report["C_pool_oracle"] = per_scene
    report["C_pool_oracle_support"] = per_scene_rr
    report["C_pool_oracle_no_yaw"] = per_scene_ny

    # -- D. valid diversity yield --------------------------------------------------------
    print("\n=== D. ValidDiversityYield@B -- stable modes per ARDY clip, rejects included ===")
    receipt = json.load(open(args.receipt))
    spent = {}
    print(f"{'arm':16s} {'charged':>8s} {'modes/scene':>12s} {'yield x1000':>12s}")
    D = {}
    for arm in ARMS_SCORED:
        if not any(r["arm"] == arm for r in rows):
            continue
        ch = receipt.get("baselines", {}).get(arm, {}).get("ardy_calls_at_K")
        modes = []
        for sid, g in groups.items():
            mine = [r for r in g if r["arm"] == arm and r["heldout"]][:8]
            modes.append(len({sig_key(block(c, "heldout")["signature"]) for c in mine
                              if addressable(block(c, "heldout"))}))
        m = float(np.mean(modes))
        D[arm] = {"charged": ch, "modes_per_scene": m,
                  "yield": (m / ch * 1000) if ch else None}
        print(f"{arm:16s} {str(ch):>8s} {m:12.2f} "
              f"{(m / ch * 1000) if ch else float('nan'):12.2f}")
    print("  yield = addressable modes returned per 1000 charged clips.  Charged counts every "
          "clip an arm\n  caused to be generated, rejects included, which is the guidance's "
          "budget: an arm that returns\n  four bodies after generating thirty clips is not "
          "operating at K=8.")
    report["D_yield"] = D

    # -- E. program vs seed allocation ---------------------------------------------------
    print("\n=== E. Eight ARDY calls: eight programs once, or one program eight times? ===")
    print("Every row spends the SAME 8 clips.  Programs are taken IN THE ARM'S OWN RANK ORDER,\n"
          "not sampled from the pool: the question is how a deployable proposer should spend a\n"
          "budget, and no proposer draws uniformly from a union it cannot see.  Seeds come from\n"
          "the 8 available per program (4 selection + 4 held-out, all independent draws).")
    alloc = {}
    ALLOC_ARMS = [a for a in ("B-NOGOOD", "A-KBEST", "COMPOSITE") if any(r["arm"] == a
                                                                        for r in rows)]
    for arm in ALLOC_ARMS:
        print(f"\n  {arm}")
        print(f"  {'allocation':>12s} {'P(>=1 valid)':>13s} {'distinct sigs':>14s} "
              f"{'stable modes':>13s}")
        alloc[arm] = {}
        for n_prog, n_seed in ((8, 1), (4, 2), (2, 4), (1, 8)):
            pv, ds, sm = [], [], []
            for sid, g in groups.items():
                ranked = sorted([r for r in g if r["arm"] == arm and r["heldout"]],
                                key=lambda r: r["rank"])
                if len(ranked) < n_prog:
                    continue
                pick = ranked[:n_prog]
                for _ in range(400):
                    sigs, clips = [], []
                    for c in pick:
                        avail = [(f, sg) for blk in ("selection", "heldout")
                                 for f, sg in zip(c[blk]["feasible"], c[blk]["signatures"])]
                        take = rng.choice(len(avail), size=min(n_seed, len(avail)),
                                          replace=False)
                        for t in take:
                            f, sg = avail[t]
                            clips.append(bool(f))
                            if f:
                                sigs.append(sig_key(sg))
                    pv.append(any(clips))
                    ds.append(len(set(sigs)))
                    # "stable" is ground truth from all 8 seeds, never from the 1-8 clips this
                    # allocation actually bought -- otherwise a single-seed draw would score
                    # stability 1.0 by definition and 1x8 would win by construction.
                    stable = {sig_key(block(c, "heldout")["signature"]) for c in pick
                              if addressable(block(c, "heldout"))}
                    sm.append(len(stable & set(sigs)))
            if not pv:
                continue
            alloc[arm][f"{n_prog}x{n_seed}"] = {"p_valid": float(np.mean(pv)),
                                                "distinct_sigs": float(np.mean(ds)),
                                                "stable_modes": float(np.mean(sm))}
            print(f"  {f'{n_prog}x{n_seed}':>12s} {np.mean(pv):13.3f} {np.mean(ds):14.2f} "
                  f"{np.mean(sm):13.2f}")
    print("\n  P(>=1 valid) isolates the value of pure RETRY; stable modes isolates the value "
          "of program\n  DIVERSITY.  If retry wins the first column and diversity wins the "
          "third, the honest\n  recommendation is a mixed allocation, and a learned proposer "
          "has to beat that mixture --\n  not just eight classical programs.")
    report["E_allocation"] = alloc

    # -- F. commanded vs realised --------------------------------------------------------
    print("\n=== F. Commanded -> realised, on held-out seeds ===")
    print("Rows: what the program asked for, at program.py's ACTIVE floors.  There is no yaw "
          "row --\nno channel can request it (program.py:31, heading follows the path "
          "tangent), so realised\nyaw is a side effect and appears only in the columns.")
    mat = defaultdict(Counter)
    for r in rows:
        if r["arm"] == "NULL-SEED" or not r["heldout"] or r["n_interactions"] != 1:
            continue
        req = tuple(bool(x) for x in r["commanded"]["sym"])       # (dip, tuck, lift)
        for f, sg in zip(r["heldout"]["feasible"], r["heldout"]["signatures"]):
            mat[req][sig_key([sg])[0] if f else "INVALID"] += 1
    cols = sorted({c for v in mat.values() for c in v if c != "INVALID"}, key=str)
    hdr = "  ".join(f"{''.join('DLRY'[i] for i, b in enumerate(c) if b) or 'none':>6s}"
                    for c in cols)
    print(f"\n{'requested':>16s} {'n':>5s}  {hdr}  {'INVALID':>8s}")
    Fm = {}
    for req in sorted(mat, key=lambda x: (sum(x), x)):
        tot = sum(mat[req].values())
        name = "+".join(n for n, b in zip(("dip", "tuck", "lift"), req) if b) or "neutral"
        cells = "  ".join(f"{mat[req][c] / tot:6.2f}" for c in cols)
        print(f"{name:>16s} {tot:5d}  {cells}  {mat[req]['INVALID'] / tot:8.2f}")
        Fm[name] = {"n": tot, **{str(c): mat[req][c] / tot for c in cols},
                    "INVALID": mat[req]["INVALID"] / tot}
    print("  columns: D=duck  L=liftL  R=liftR  Y=yaw (realised active set); rows sum to 1.")
    print("  A high INVALID cell is not the enumerator's failure -- it is the prior refusing a "
          "request.\n  `decode` only writes the global_joints_positions channel when tuck or "
          "lift clear their ACTIVE\n  floors (program.py:234), so the limb-target path is the "
          "one place a request can turn a clip\n  invalid, and the tuck/lift rows are where "
          "that shows up.")
    report["F_commanded_realised"] = Fm

    # -- G. yaw ablation and the negative controls ---------------------------------------
    print("\n=== G. Yaw ablation, and the retired channels as negative controls ===")
    tuck_fire = order_fire = n_sig = 0
    for r in rows:
        for blk in ("selection", "heldout"):
            ev = block(r, blk)
            if not ev:
                continue
            for s in ev["signatures"]:
                for per in ([s] if not isinstance(s[0], (list, tuple)) else s):
                    n_sig += 1
                    tuck_fire += bool(per[1])
                    order_fire += bool(per[4])
    print(f"negative controls over {n_sig} realised active sets: "
          f"tuck fires {100 * tuck_fire / max(n_sig, 1):.2f} %, "
          f"order fires {100 * order_fire / max(n_sig, 1):.2f} %  (both must be ~0)")
    with_yaw, without_yaw = [], []
    for sid, g in groups.items():
        cand = list({r["prog_key"]: r for r in g
                     if r["arm"] != "NULL-SEED" and r["heldout"]}.values())
        ref = [c for c in cand if addressable(block(c, "heldout"))]
        if not ref:
            continue
        with_yaw.append(len({sig_key(block(c, "heldout")["signature"], True) for c in ref}))
        without_yaw.append(len({sig_key(block(c, "heldout")["signature"], False) for c in ref}))
    ystab = [np.mean([block(r, "heldout")["signature"][0][5] ==
                      block(r, "selection")["signature"][0][5]
                      for r in rows if r["heldout"] and r["n_interactions"] == 1])]
    print(f"reference modes per scene: {np.mean(with_yaw):.2f} with yaw -> "
          f"{np.mean(without_yaw):.2f} without "
          f"({100 * (1 - np.mean(without_yaw) / max(np.mean(with_yaw), 1e-9)):.0f} % of the "
          f"counted alphabet is a channel nothing can request)")
    print(f"yaw bit agrees across disjoint seed blocks on {100 * ystab[0]:.0f} % of programs")
    report["G_yaw"] = {"modes_with_yaw": float(np.mean(with_yaw)),
                       "modes_without_yaw": float(np.mean(without_yaw)),
                       "yaw_bit_seed_agreement": float(ystab[0]),
                       "tuck_fire_rate": tuck_fire / max(n_sig, 1),
                       "order_fire_rate": order_fire / max(n_sig, 1)}

    # -- H. failure decomposition --------------------------------------------------------
    print("\n=== H. Why each missed reference mode was missed (best equal-call arm) ===")
    causes = Counter()
    per_fam = defaultdict(Counter)
    for sid, g in groups.items():
        cand = list({r["prog_key"]: r for r in g
                     if r["arm"] != "NULL-SEED" and r["heldout"]}.values())
        ref = {sig_key(block(c, "heldout")["signature"]) for c in cand
               if addressable(block(c, "heldout"))}
        if len(ref) < 2:
            continue
        arm = "B-NOGOOD"
        mine = sorted([r for r in g if r["arm"] == arm and r["heldout"]],
                      key=lambda r: r["rank"])[:8]
        got = {sig_key(block(c, "heldout")["signature"]) for c in mine
               if addressable(block(c, "heldout"))}
        for z in ref - got:
            duck, liftL, liftR, yaw = z[0]
            want_lift = liftL or liftR
            # did the arm even ASK for the symbol?
            asked = any((c["commanded"]["sym"][0] == duck)
                        and (c["commanded"]["sym"][2] == want_lift) for c in mine)
            realised_any = any(sig_key([s]) == (z[0],) for c in mine
                               for s in block(c, "heldout")["signatures"])
            if not asked:
                cause = "symbol never requested"
            elif not realised_any:
                cause = "requested, prior realised something else"
            else:
                hits = [c for c in mine if any(sig_key([s]) == (z[0],)
                                               for s in block(c, "heldout")["signatures"])]
                if any(block(c, "heldout")["feasible_rate"] < MIN_FEAS for c in hits):
                    cause = "mode realised but collides"
                else:
                    cause = "mode realised but unstable"
            causes[cause] += 1
            per_fam[g[0]["family"]][cause] += 1
    tot = max(sum(causes.values()), 1)
    for c, n in causes.most_common():
        print(f"  {c:42s} {n:4d}  {n / tot:6.1%}")
    print(f"\n{'family':16s} " + "  ".join(f"{c[:20]:>22s}" for c in causes))
    for f, cc in sorted(per_fam.items()):
        t = max(sum(cc.values()), 1)
        print(f"{f:16s} " + "  ".join(f"{cc[c] / t:22.2f}" for c in causes))
    report["H_failure"] = {"overall": dict(causes),
                           "per_family": {k: dict(v) for k, v in per_fam.items()}}

    # -- I. distance sensitivity ---------------------------------------------------------
    print("\n=== I. Diagonal q99 scaling vs full-covariance Mahalanobis ===")
    # The point of this section is whether CORRELATION STRUCTURE changes the answer, not
    # whether the two metrics happen to be on the same scale.  The first version rescaled eps by
    # median(diag(Wd)/diag(Wc)), which is not a global scale factor at all -- the covariance
    # whitener's diagonal already mixes off-diagonal terms, so that ratio partly divides out
    # the very correlation being tested.  Here the two metrics are matched on their MEDIAN
    # PAIRWISE DISTANCE, which is a clean global scale match, and any residual difference in
    # net size is then attributable to shape rather than units.
    dn, dd, ratios = [], [], []
    for sid, g in groups.items():
        n_int = g[0]["n_interactions"]
        Wc, Wd = pooled_whitener(g, n_int), diag_whitener(g, n_int)
        cand = [r for r in g if r["arm"] != "NULL-SEED" and r["heldout"]]
        mus = np.array([block(c, "heldout")["mu"] for c in cand], float)
        if len(mus) < 3:
            continue
        pc = [d_morph(mus[i], mus[j], Wc) for i in range(len(mus)) for j in range(i + 1, len(mus))]
        pd_ = [d_morph(mus[i], mus[j], Wd) for i in range(len(mus)) for j in range(i + 1, len(mus))]
        f = float(np.median(pd_) / max(np.median(pc), 1e-12))
        ratios.append(f)
        dn.append(len(epsilon_net(mus, Wc, eps)))
        dd.append(len(epsilon_net(mus, Wd, eps * f)))
    print(f"eps-net size per scene: covariance {np.mean(dn):.2f}  diagonal {np.mean(dd):.2f}  "
          f"(scales matched on median pairwise distance, factor {np.mean(ratios):.2f})")
    print("  Matched on scale, a REMAINING gap is the correlation structure doing work: "
          "EXP-005f measured\n  corr(dpsi, lift) = +0.52 and corr(dpsi, dip) = +0.28, so one "
          "duck moves three channels at once\n  and a diagonal metric can read that single "
          "event as three.  If the two columns agree, the\n  diagonal metric the gate used is "
          "adequate and the primary result does not rest on it.")
    report["I_distance"] = {"cov_net": float(np.mean(dn)), "diag_net": float(np.mean(dd)),
                            "scale_factor": float(np.mean(ratios))}

    with open(out / "receipt.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nwrote {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
