"""Route and body costs on one scale, so a route can be chosen for what the BODY must do.

The Phase 1-3 planner chose a route from floor geometry alone and then asked the body layer to
cope. That is backwards whenever a slightly longer route is much easier to hold: the router
cannot know, because route length says nothing about how deep or how long the robot has to
crouch. Scoring the body's own schedule puts the two on comparable footing.

    J_body(r) = integral over s of [ u^2 + l1 (du/ds)^2 + l2 (d2u/ds2)^2 ] ds
    J(r)      = w_L * L(r) + w_B * J_body(r) - w_C * C_min(r)

`u` is the duck command in [0, 1], so J_body is dimensionless-per-metre and the three terms
are effort, rate and smoothness -- the same decomposition the Phase 3 QP minimises, evaluated
rather than solved.

Preferences are WEIGHT CONFIGURATIONS, not hard-coded route labels. "Stay upright" is not a
rule that forbids ducking; it is a large w_B, and it will still choose to duck when every
alternative is far longer. That distinction is the point: the selector should be able to
explain a choice as a trade, and be overruled by a scene that makes the trade go the other way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Matched to the Phase 3 scheduler's objective so the evaluated cost and the solved cost
# describe the same preference; see optim/scheduler.py W_EFFORT / W_D1 / W_D2.
LAMBDA_1 = 6.0
LAMBDA_2 = 25.0

# w_L (per metre of route), w_B (per unit of body cost), w_C (per metre of worst clearance).
WEIGHTS = {
    "balanced":  {"w_L": 1.0, "w_B": 1.0, "w_C": 1.0},
    "shortest":  {"w_L": 4.0, "w_B": 0.25, "w_C": 0.5},
    "upright":   {"w_L": 0.5, "w_B": 6.0, "w_C": 0.5},
    "clearance": {"w_L": 0.5, "w_B": 1.0, "w_C": 8.0},
}


@dataclass
class CostBreakdown:
    route_len_m: float
    j_effort: float
    j_rate: float
    j_smooth: float
    j_body: float
    c_min_m: float
    weights: dict
    total: float

    def to_dict(self) -> dict:
        return {"route_len_m": round(self.route_len_m, 3),
                "j_effort": round(self.j_effort, 4), "j_rate": round(self.j_rate, 4),
                "j_smooth": round(self.j_smooth, 4), "j_body": round(self.j_body, 4),
                "c_min_m": round(self.c_min_m, 4), "weights": self.weights,
                "w_route_term": round(self.weights["w_L"] * self.route_len_m, 4),
                "w_body_term": round(self.weights["w_B"] * self.j_body, 4),
                "w_clear_term": round(-self.weights["w_C"] * self.c_min_m, 4),
                "total": round(self.total, 4)}


def body_cost(u: np.ndarray, s_m: np.ndarray) -> tuple[float, float, float]:
    """(effort, rate, smoothness) integrals of a command schedule over route distance.

    Derivatives are taken with respect to DISTANCE, not sample index, so the same physical
    manoeuvre costs the same whether the route it sits on is short or long. Taking them per
    sample would make every schedule on a long route look artificially smooth.
    """
    u = np.asarray(u, float)
    s = np.asarray(s_m, float)
    if len(u) < 3:
        return float(np.trapz(u ** 2, s) if len(u) > 1 else 0.0), 0.0, 0.0
    d1 = np.gradient(u, s)
    d2 = np.gradient(d1, s)
    return (float(np.trapz(u ** 2, s)), float(np.trapz(d1 ** 2, s)), float(np.trapz(d2 ** 2, s)))


def evaluate(u: np.ndarray, s_m: np.ndarray, c_min_m: float, preference: str = "balanced",
             l1: float = LAMBDA_1, l2: float = LAMBDA_2) -> CostBreakdown:
    """Score one (route, schedule) pair under a named weight configuration."""
    w = WEIGHTS.get(preference, WEIGHTS["balanced"])
    e, r, sm = body_cost(u, s_m)
    j_body = e + l1 * r + l2 * sm
    length = float(np.asarray(s_m, float)[-1]) if len(s_m) else 0.0
    total = w["w_L"] * length + w["w_B"] * j_body - w["w_C"] * float(c_min_m)
    return CostBreakdown(route_len_m=length, j_effort=e, j_rate=r, j_smooth=sm,
                         j_body=j_body, c_min_m=float(c_min_m), weights=w, total=total)
