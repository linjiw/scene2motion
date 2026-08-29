"""The optimisation teacher: minimum-effort, minimum-jerk duck schedules.

    minimise   we*sum(q^2) + w1*sum((dq)^2) + w2*sum((ddq)^2)
    subject to g(z_i) <= clearance_i - margin,   0 <= q_i <= 1,
               z = lag(q)  (first-order, time constant tau)

The problem is a CONVEX QP, and the reason is worth stating because it is what makes the
teacher exact and reproducible rather than a heuristic search:

  * `g` is monotone non-increasing, so the clearance constraint `g(z_i) <= c_i - margin`
    inverts exactly to `z_i >= g_inv(c_i - margin)`. A nonlinear constraint becomes a bound on
    the state.
  * the lag is linear: z = L q with L[i,j] = a(1-a)^(i-j) for j <= i. So `z >= z_req` is a
    LINEAR inequality in q.
  * the objective is a positive-semidefinite quadratic form in q.

Convex QP with box and linear inequality constraints, 64 variables. SLSQP solves it to
optimality, deterministically, in milliseconds -- no random restarts, no schedule to tune.

What the optimiser is for: it decides minimal crouch depth, when to start (anticipation falls
out of the lag -- reaching z_req at the beam requires commanding q before it), when to recover,
and whether two nearby beams share one crouch or get two. None of that is encoded; it is what
minimising effort and jerk against the constraint produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .response import DuckResponse, alpha

# Defaults, recorded in every result so a schedule can be reproduced from its artifact.
W_EFFORT, W_D1, W_D2 = 1.0, 6.0, 25.0
MARGIN_M = 0.12      # clearance headroom; must exceed the surrogate's own error (~70 mm)


@dataclass
class Schedule:
    q: np.ndarray
    z: np.ndarray
    top: np.ndarray
    feasible: bool
    status: str
    objective: float
    max_violation_m: float
    n_iter: int
    weights: dict = field(default_factory=dict)
    margin_m: float = MARGIN_M
    tau_s: float = 0.0

    def to_dict(self) -> dict:
        return {"q": np.round(self.q, 5).tolist(), "feasible": self.feasible,
                "status": self.status, "objective": round(float(self.objective), 6),
                "max_violation_m": round(float(self.max_violation_m), 6),
                "n_iter": int(self.n_iter), "weights": self.weights,
                "margin_m": self.margin_m, "tau_s": self.tau_s}


def lag_matrix(n: int, dt: float, tau: float) -> np.ndarray:
    """L with z = L q for z[i] = (1-a) z[i-1] + a q[i], a = 1 - exp(-dt/tau), z[-1] = 0."""
    a = alpha(dt, tau)
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    k = np.maximum(i - j, 0)
    return np.where(j <= i, a * np.power(1.0 - a, k), 0.0)


def _diff_matrices(n: int) -> tuple[np.ndarray, np.ndarray]:
    D1 = np.eye(n) - np.eye(n, k=-1)
    D2 = np.eye(n) - 2 * np.eye(n, k=-1) + np.eye(n, k=-2)
    return D1[1:], D2[2:]


def solve(clearance: np.ndarray, resp: DuckResponse, dt: float,
          margin_m: float = MARGIN_M, w_effort: float = W_EFFORT,
          w_d1: float = W_D1, w_d2: float = W_D2,
          tau: float | None = None) -> Schedule:
    """Optimal commanded duck schedule for a clearance profile."""
    c = np.asarray(clearance, float)
    n = len(c)
    tau = resp.tau_s if tau is None else tau
    weights = {"w_effort": w_effort, "w_d1": w_d1, "w_d2": w_d2}

    need = c - margin_m
    # Refuse rather than saturate: a beam below the deepest reachable crouch is not a scene
    # the body layer can solve, and returning q=1 would silently hand the planner a schedule
    # that does not clear.
    if not bool(np.all(resp.clears(need))):
        z = np.zeros(n)
        return Schedule(z, z, resp.g(z), False, "infeasible: clearance below reachable crouch",
                        float("inf"), float(np.max(resp.g(np.ones(n)) - need)), 0,
                        weights, margin_m, tau)

    z_req = resp.g_inv(need)
    L = lag_matrix(n, dt, tau)
    D1, D2 = _diff_matrices(n)
    H = (w_effort * np.eye(n) + w_d1 * D1.T @ D1 + w_d2 * D2.T @ D2)

    def f(q):
        return float(q @ H @ q)

    def fp(q):
        return 2.0 * (H @ q)

    cons = [{"type": "ineq", "fun": lambda q: L @ q - z_req, "jac": lambda q: L}]
    q0 = np.clip(z_req, 0.0, 1.0)
    r = minimize(f, q0, jac=fp, bounds=[(0.0, 1.0)] * n, constraints=cons,
                 method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
    q = np.clip(r.x, 0.0, 1.0)
    z = L @ q
    top = resp.g(z)
    viol = float(np.max(np.maximum(top - need, 0.0)))
    return Schedule(q, z, top, bool(r.success and viol <= 1e-6),
                    str(r.message), f(q), viol, int(r.nit), weights, margin_m, tau)


def clearance_from_profile(profile: np.ndarray) -> np.ndarray:
    """Overhead-clearance channel of a route profile (channel 0)."""
    return np.asarray(profile, float)[:, 0]


def dt_for(route_len_m: float, n: int, speed: float) -> float:
    """Seconds per route sample -- the lag is in TIME, the profile is in DISTANCE."""
    return float(route_len_m / max(n - 1, 1) / max(speed, 1e-6))
