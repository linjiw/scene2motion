"""EXP-005h: are the missing body variants DECISION-RELEVANT, or merely different?

Why this gate exists
--------------------
EXP-005g measures how much of the feasible body set a proposer covers. Coverage alone is not a
contribution: duck 31 cm, duck 32 cm and duck 33 cm are three distinct points and, unless they
differ in something a user or a controller would trade off, generating all three is scientific
noise dressed as diversity.

An earlier version of this project already made the opposite error in the other direction --
a 1.35x scalar-cost filter so loose that every candidate passed it, which is not a quality
filter at all. So the criterion here is multi-objective and explicit: a body candidate is
decision-relevant if it is **Pareto-nondominated** among the feasible candidates for the same
route. That is a statement no single scalar cost can make, and it is the one the guidance asks
for.

The objectives, all oriented so LOWER IS BETTER
-----------------------------------------------
Route length is deliberately absent: every candidate here shares one route by construction, so
it is a constant and including it would only dilute the front.

    neg_clearance   -min_clearance_m           more margin from the scene is better
    posture         integral |pelvis - 0.78|   standing upright is the neutral preference
    roughness       integral |d2/dt2 envelope| smooth body changes are easier to track
    arm_restriction integral tuck              keeping the arms free is a real preference
    head_reduction  integral dip               not crouching is a real preference
    lift_effort     integral foot excess       not lifting the feet is a real preference
    foot_slip       mean planted-foot speed    the only tracking proxy available pre-SONIC

`posture`, `head_reduction` and `arm_restriction` are correlated by construction -- a duck
raises two of them together -- and that is fine: Pareto dominance handles correlated
objectives, it just means the front is thinner than the dimension count suggests. What matters
is that they can DISAGREE, which is exactly the duck-versus-tuck trade EXP-001d measured
(ducking makes G1 wider, so buying headroom costs lateral margin).

Reported
--------
    ParetoRecall@K   fraction of the reference Pareto front a proposer's first K cover
    hypervolume@K    convergence and spread together, against a fixed nadir
    front_size       how many candidates are nondominated at all -- if this is ~1, there is
                     no trade-off to make and the whole morphology-set idea is unnecessary

A pre-committed reading: if the front is essentially a single point on most scenes, the
guidance's Outcome 3 applies and the contribution is the calibration and benchmark, not a
generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.body_enumerate import (enumerate_composite, enumerate_kbest,  # noqa: E402
                                         enumerate_nogood, enumerate_weight_sweep,
                                         refine_continuous)
from scene2motion.morphology import envelope_series  # noqa: E402
from scene2motion.planner import plan, plan_to_path_spec  # noqa: E402
from scene2motion.program import decode, encode  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.scenes import build_suite  # noqa: E402

PROMPT = "A person walks forward."
SPEED, GOAL_TOL, MAX_DURATION = 0.9, 0.5, 14.0
NOMINAL_PELVIS = 0.78
OBJECTIVES = ("neg_clearance", "posture", "roughness", "arm_restriction",
              "head_reduction", "lift_effort", "foot_slip")
BASELINES = {"A-KBEST": enumerate_kbest, "B-NOGOOD": enumerate_nogood,
             "C-WSWEEP": enumerate_weight_sweep, "D-REFINE": refine_continuous,
             "COMPOSITE": enumerate_composite}


def objectives(body: G1Body, qpos: np.ndarray, prog, fps: float,
               report: dict) -> np.ndarray:
    """(len(OBJECTIVES),) vector, all lower-is-better."""
    env = envelope_series(body, qpos)
    d2 = np.diff(env, n=2, axis=0) if len(env) > 2 else np.zeros((1, 3))
    slot = np.atleast_2d(prog.slot)
    foot = [g for g in body.robot_geoms if "foot" in body.geom_name[g]]
    slip = np.nan
    if foot:
        pos = []
        for q in qpos:
            body.fk(q)
            pos.append(body.data.geom_xpos[foot].copy())
        pos = np.asarray(pos)
        v = np.linalg.norm(np.diff(pos[:, :, :2], axis=0), axis=-1) * fps
        planted = (pos[:-1, :, 2] - body.model.geom_size[foot, 0][None, :]) < 0.03
        slip = float(v[planted].mean()) if planted.any() else 0.0
    return np.array([
        -float(report["min_clearance_m"]),
        float(np.abs(qpos[:, 2] - NOMINAL_PELVIS).mean()),
        float(np.abs(d2).mean()),
        float(slot[:, 3].sum()),
        float(slot[:, 2].sum()),
        float(slot[:, 4].sum()),
        0.0 if not np.isfinite(slip) else slip,
    ])


def pareto_front(F: np.ndarray) -> np.ndarray:
    """Indices of the nondominated rows of `F` (lower is better on every column)."""
    keep = []
    for i in range(len(F)):
        dominated = any(
            np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(F)) if j != i)
        if not dominated:
            keep.append(i)
    return np.array(keep, dtype=int)


def hypervolume(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, n_mc: int = 20000,
                rng: np.random.Generator | None = None) -> float:
    """Monte-Carlo hypervolume dominated by `F` in the FIXED box [ideal, nadir].

    Exact hypervolume is exponential in the objective count; at 7 objectives a Monte-Carlo
    estimate is the honest choice, and the same sample budget and the same box are used for
    every arm so the comparison is unaffected by the estimator's variance.

    The box must be FIXED and passed in. Deriving `ideal` from each arm's own points -- the
    first version of this function -- makes the sampling region move with the arm, and a unit
    check caught the consequence at once: a single DOMINATED point scored 1.0, and one good
    point outscored the entire front. Under a moving box every arm dominates its own corner
    completely, which is precisely what a comparison must not allow.
    """
    rng = rng or np.random.default_rng(0)
    F = np.atleast_2d(F)
    if F.size == 0:
        return 0.0
    ideal = np.asarray(ideal, float)
    span = np.maximum(np.asarray(nadir, float) - ideal, 1e-9)
    pts = ideal + rng.random((n_mc, len(ideal))) * span
    dominated = np.zeros(n_mc, bool)
    for f in F:
        dominated |= np.all(f <= pts, axis=1)
    return float(dominated.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp005h")
    ap.add_argument("--seeds_per_rung", type=int, default=2)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--diffusion_steps", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    runner = ArdyRunner(cache_path="outputs/text_cache.npz")
    fps = runner.fps
    scenes = [s for s in build_suite(args.seeds_per_rung) if plan(s, "adaptive").feasible]
    scenes = scenes[:args.limit]
    print(f"{len(scenes)} scenes with a feasible route", flush=True)

    rows = []
    for sc in scenes:
        p = plan(sc, "adaptive")
        body = G1Body(sc)
        T = min(int(MAX_DURATION * fps),
                max(int(2 * fps), int(round(p.length / SPEED * fps))))
        seed = int(hashlib.sha1(sc.scene_id.encode()).hexdigest()[:8], 16) % (2 ** 20)
        ctrl = runner.generate([PROMPT],
                               [plan_to_path_spec(p, fps, SPEED, duration=T / fps)], T,
                               args.diffusion_steps, seeds=[seed])[0]

        # Every arm's candidates go into one pool; the reference front is the front of the
        # POOL, so no arm is scored against a reference it alone defined.
        pool, owner = [], []
        for name, fn in BASELINES.items():
            try:
                cands = fn(sc, p, K=args.K, runner=runner, fps=fps)
            except Exception as e:  # a baseline that cannot run scores nothing, not a crash
                print(f"    {name} failed on {sc.scene_id}: {type(e).__name__}", flush=True)
                cands = []
            for c in cands[:args.K]:
                pool.append(c)
                owner.append(name)
        if len(pool) < 2:
            continue

        specs = [decode(c, sc, fps, ctrl, runner.joint_names, duration=T / fps) for c in pool]
        outs = []
        for i in range(0, len(specs), 8):
            k = specs[i:i + 8]
            outs += runner.generate([PROMPT] * len(k), k, T, args.diffusion_steps,
                                    seeds=[seed] * len(k))
        F, ok = [], []
        for c, o in zip(pool, outs):
            q = runner.to_qpos(o)
            rep = body.trajectory_report(q)
            goal = bool(np.linalg.norm(q[-1, :2] - np.asarray(sc.goal)) < GOAL_TOL)
            ok.append(bool(goal and rep["collision_free"]))
            F.append(objectives(body, q, c, fps, rep))
        F = np.array(F)
        feas = np.flatnonzero(ok)
        if len(feas) < 2:
            continue
        front = feas[pareto_front(F[feas])]
        # One box for the whole scene, from the feasible pool, shared by every arm.
        ideal = F[feas].min(axis=0)
        nadir = F[feas].max(axis=0) + 1e-6
        hv_all = hypervolume(F[front], ideal, nadir)

        rec = {"scene_id": sc.scene_id, "family": sc.family,
               "n_pool": len(pool), "n_feasible": int(len(feas)),
               "front_size": int(len(front)),
               "front_owners": sorted({owner[i] for i in front}),
               "hv_reference": hv_all, "arms": {}}
        for name in BASELINES:
            mine = [i for i in range(len(pool)) if owner[i] == name]
            mine_ok = [i for i in mine if ok[i]]
            covered = len(set(mine_ok) & set(front.tolist()))
            rec["arms"][name] = {
                "n": len(mine), "n_feasible": len(mine_ok),
                "pareto_recall": covered / max(len(front), 1),
                "hv": hypervolume(F[mine_ok], ideal, nadir) if mine_ok else 0.0,
                "hv_ratio": ((hypervolume(F[mine_ok], ideal, nadir) / hv_all)
                             if (mine_ok and hv_all > 0) else 0.0),
            }
        rows.append(rec)
        print(f"  {sc.scene_id[:30]:30s} pool {len(pool):3d} feasible {len(feas):3d} "
              f"front {len(front):2d} ({time.time()-t0:.0f}s)", flush=True)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"experiment": "exp005h_pareto_morph", "objectives": list(OBJECTIVES),
               "n_scenes": len(rows), "K": args.K,
               "mean_front_size": float(np.mean([r["front_size"] for r in rows])) if rows else 0,
               "frac_scenes_front_ge2": float(
                   np.mean([r["front_size"] >= 2 for r in rows])) if rows else 0,
               "arms": {}}
    for name in BASELINES:
        v = [r["arms"][name] for r in rows]
        summary["arms"][name] = {
            "pareto_recall": float(np.mean([x["pareto_recall"] for x in v])) if v else 0.0,
            "hv_ratio": float(np.mean([x["hv_ratio"] for x in v])) if v else 0.0,
            "feasible_frac": float(np.mean([x["n_feasible"] / max(x["n"], 1) for x in v])) if v else 0.0,
        }
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    with open(out / "receipt.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
