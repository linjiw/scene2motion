"""Phase 4B: choose the route by what the BODY will have to do on it.

Four selectors over the same candidate list, so the comparison isolates the selection rule:

    oracle_qp     score every route with the Phase 3 QP's schedule, take argmin J. The
                  reference the others are measured against -- it pays a full convex solve
                  per candidate, which is why it is a reference and not a proposal.
    heuristic     argmin route length. The Phase 1-3 behaviour: the router never sees the
                  body, so a route that is 30 cm shorter and needs a deep sustained crouch
                  looks strictly better than one that needs none.
    tcn_body      score every route with the m018 TCN's schedule, take argmin J. One batched
                  forward pass for all candidates.
    tcn_verify    tcn_body's ranking, then generate and verify the top choice. If it is
                  rejected, fall back down the ranking. This is the only selector whose
                  answer depends on what the prior actually did.

Regret is measured on a COMMON scale. Every selector's chosen route is rescored with the QP
schedule for that route, and regret is that score minus the best such score over all
candidates. Scoring each selector with its own schedule would confuse "picked a worse route"
with "wrote a worse schedule for the same route", and only the first is a selection error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..learn.route_profile import normalise, profile
from ..optim.response import DIP_MAX, DuckResponse
from ..optim.scheduler import MARGIN_M, dt_for, solve
from ..scenes import Scene
from .cost import evaluate as score_cost
from .routes import Route

N_SAMPLES = 64


def _arc(xy: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    seg = np.linalg.norm(np.diff(np.asarray(xy, float), axis=0), axis=1)
    return np.linspace(0.0, float(seg.sum()), n)


@dataclass
class Scored:
    route: Route
    s_m: np.ndarray
    u_qp: np.ndarray | None
    u_tcn: np.ndarray | None
    c_pred_m: float              # predicted worst headroom from the route profile
    j_qp: float | None = None
    j_tcn: float | None = None
    extra: dict = field(default_factory=dict)


def qp_schedules(scene: Scene, routes: list[Route], resp: DuckResponse, speed: float = 0.9
                 ) -> list[np.ndarray | None]:
    """One convex solve per route. Sequential by nature -- this is the expensive reference."""
    out = []
    for r in routes:
        prof = profile(scene, r.xy, speed=speed)
        s = _arc(r.xy)
        sol = solve(prof[:, 0], resp, dt_for(float(s[-1]), N_SAMPLES, speed))
        out.append(np.clip(sol.q, 0.0, 1.0) if sol.feasible else None)
    return out


def tcn_schedules(scene: Scene, routes: list[Route], resp: DuckResponse, model,
                  speed: float = 0.9) -> list[np.ndarray]:
    """One batched forward pass for every candidate.

    Batching is the whole point of a learned body layer in a selector: the QP has to solve
    each route separately, while the TCN sees all of them at once for roughly the cost of one.
    """
    import torch

    profs = [profile(scene, r.xy, speed=speed) for r in routes]
    x = np.stack([normalise(p) for p in profs])
    q_req = np.stack([resp.g_inv(p[:, 0] - MARGIN_M) for p in profs])
    with torch.no_grad():
        q = model(torch.from_numpy(x).float(), torch.from_numpy(q_req).float()).numpy()
    return [np.clip(row, 0.0, 1.0) for row in q]


def score_all(scene: Scene, routes: list[Route], resp: DuckResponse, model=None,
              preference: str = "balanced", speed: float = 0.9) -> list[Scored]:
    """Every candidate, scored under both schedules, on the same cost."""
    qps = qp_schedules(scene, routes, resp, speed)
    tcns = tcn_schedules(scene, routes, resp, model, speed) if model is not None else [None] * len(routes)
    out = []
    for r, uq, ut in zip(routes, qps, tcns):
        prof = profile(scene, r.xy, speed=speed)
        s = _arc(r.xy)
        # Predicted worst headroom once the schedule is applied: overhead clearance above the
        # route minus where the fitted gain says the top of the body will be. This is the same
        # quantity the Phase 3 QP constrains, so the selector and the scheduler agree on what
        # they are trading. It is a PREDICTION -- Phase 4A measured it to be optimistic out of
        # distribution, which is why tcn_verify exists.
        def c_pred(u):
            u = np.zeros(N_SAMPLES) if u is None else u
            return float(np.min(prof[:, 0] - resp.g(u)))
        sc = Scored(route=r, s_m=s, u_qp=uq, u_tcn=ut, c_pred_m=c_pred(uq))
        if uq is not None:
            sc.j_qp = score_cost(uq, s, c_pred(uq), preference).total
        if ut is not None:
            sc.j_tcn = score_cost(ut, s, c_pred(ut), preference).total
        out.append(sc)
    return out


def pick(scored: list[Scored], rule: str) -> int:
    """Index of the chosen candidate under one selection rule."""
    if rule == "heuristic":
        return int(np.argmin([s.s_m[-1] for s in scored]))
    key = "j_qp" if rule == "oracle_qp" else "j_tcn"
    vals = [getattr(s, key) for s in scored]
    ok = [i for i, v in enumerate(vals) if v is not None]
    if not ok:
        return 0
    return min(ok, key=lambda i: vals[i])


def ranking(scored: list[Scored], rule: str) -> list[int]:
    """Full preference order, so a rejected first choice has somewhere to fall back to."""
    if rule == "heuristic":
        return list(np.argsort([s.s_m[-1] for s in scored]))
    key = "j_qp" if rule == "oracle_qp" else "j_tcn"
    vals = [(getattr(s, key) if getattr(s, key) is not None else float("inf")) for s in scored]
    return list(np.argsort(vals))


def regret(scored: list[Scored], chosen: int) -> float:
    """Joint-objective regret of a route choice, on the common QP scale.

    Infinite-cost candidates (no feasible QP schedule) are excluded from the best, but a
    selector that CHOOSES one is charged the worst finite regret rather than infinity, so one
    pathological pick cannot dominate a mean.
    """
    vals = [s.j_qp for s in scored if s.j_qp is not None]
    if not vals:
        return 0.0
    best = min(vals)
    got = scored[chosen].j_qp
    return (max(vals) - best) if got is None else (got - best)


def batched_latency(scene: Scene, routes: list[Route], resp: DuckResponse, model,
                    k: int, speed: float = 0.9, repeats: int = 3) -> dict:
    """Wall-clock to score `k` route instances, TCN batched against QP sequential.

    When the scene offers fewer than `k` distinct traversals the list is cycled to length k.
    That is honest for this measurement: the cost is per route instance, and the question is
    how the two schedulers scale in the number of candidates a selector has to consider.
    """
    reps = [routes[i % len(routes)] for i in range(k)]
    t_tcn, t_qp = [], []
    for _ in range(repeats):
        t0 = time.perf_counter(); tcn_schedules(scene, reps, resp, model, speed)
        t_tcn.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); qp_schedules(scene, reps, resp, speed)
        t_qp.append(time.perf_counter() - t0)
    return {"k": k, "n_distinct_routes": len(routes),
            "tcn_batched_ms": round(float(np.median(t_tcn)) * 1000, 2),
            "qp_sequential_ms": round(float(np.median(t_qp)) * 1000, 2),
            "tcn_ms_per_route": round(float(np.median(t_tcn)) * 1000 / k, 3),
            "qp_ms_per_route": round(float(np.median(t_qp)) * 1000 / k, 3),
            "speedup": round(float(np.median(t_qp)) / max(float(np.median(t_tcn)), 1e-9), 1)}


def load_tcn():
    """The m018 checkpoint, refusing any margin other than the one in use."""
    import json

    import torch

    from ..optim.model_v3 import DuckTCN
    from ..demo.schedules import TCN_CKPT, TCN_DIR
    if not TCN_CKPT.exists():
        return None
    mj = json.loads((TCN_DIR / "tcn.json").read_text())
    if abs(float(mj.get("margin_m", -1)) - MARGIN_M) > 1e-9:
        raise ValueError(f"checkpoint margin {mj.get('margin_m')} != {MARGIN_M}")
    m = DuckTCN()
    m.load_state_dict(torch.load(TCN_CKPT, map_location="cpu"))
    m.eval()
    return m
