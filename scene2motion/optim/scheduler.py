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
# Clearance headroom. NOT a round number and not a guess: `verify_margin.py` sweeps it against
# the real prior over 40 clips, and 0.12 m -- which looked generous beside a 43 mm surrogate
# holdout error -- produced actual collisions (80 % collision-free, worst clearance -18 mm).
# The margin has to cover the surrogate error AND ARDY's own 30-74 mm per-seed scatter, and
# 0.18 m is the smallest swept value that restores 100 % collision-free (worst clearance
# +17 mm) at 24.3 cm of mean peak crouch. Raising it further only buys crouch depth.
MARGIN_M = 0.18


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
          tau: float | None = None, time_weighted: bool = False,
          rate_bounds: tuple[float, float] | None = None,
          dimensioned: bool = False) -> Schedule:
    """Optimal commanded duck schedule for a clearance profile.

    `time_weighted` makes the objective a genuine time integral rather than a per-sample sum,
    and it is the difference between a merge/split boundary that depends on walking speed and
    one that does not. Minimising

        integral( we q^2 + w1 (dq/dt)^2 + w2 (d2q/dt2)^2 ) dt

    discretises to weights we*dt, w1/dt and w2/dt^3, because each derivative carries a 1/dt.
    Summing raw differences instead -- the default here -- makes the cost of standing up
    between two beams depend only on how many SAMPLES separate them, and the samples are laid
    out in distance. Speed then cancels entirely, which is exactly what Experiment B measures:
    an identical boundary at 4.0 m for 0.6, 0.9 and 1.2 m/s.

    It defaults to OFF despite being the more principled form, and the reason is measured
    rather than aesthetic: at dt ~ 0.2 s the jerk term picks up a 1/dt^3 factor of ~125, which
    swamps the effort and first-difference terms and forces a merged crouch at every gap and
    speed tested -- the boundary disappears entirely rather than moving. The weights W_D1 and
    W_D2 were chosen for the per-sample form and would have to be re-tuned for the integral
    form, and the dataset, the trained model and Experiment A were all produced under the
    per-sample objective. Switching the default without re-deriving those would leave the
    artifacts describing a teacher that no longer exists, which is the failure this phase has
    already had once. Re-tuning is future work and the flag is here so it can be done.
    """
    c = np.asarray(clearance, float)
    n = len(c)
    tau = resp.tau_s if tau is None else tau
    weights = {"w_effort": w_effort, "w_d1": w_d1, "w_d2": w_d2,
               "time_weighted": time_weighted, "dimensioned": dimensioned,
               "rate_bounds": rate_bounds, "dt": dt}

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
    if dimensioned:
        # alpha*q^2*dt + beta*((dq)/dt)^2*dt = alpha*dt*q^2 + beta/dt * (dq)^2. No jerk term.
        we, k1, k2 = w_effort * dt, w_d1 / dt, 0.0
    elif time_weighted:
        we, k1, k2 = w_effort * dt, w_d1 / dt, w_d2 / (dt ** 3)
    else:
        we, k1, k2 = w_effort, w_d1, w_d2
    H = (we * np.eye(n) + k1 * D1.T @ D1 + k2 * D2.T @ D2)

    def f(q):
        return float(q @ H @ q)

    def fp(q):
        return 2.0 * (H @ q)

    cons = [{"type": "ineq", "fun": lambda q: L @ q - z_req, "jac": lambda q: L}]
    if rate_bounds is not None:
        r_down, r_up = float(rate_bounds[0]), float(rate_bounds[1])
        # D1 @ q is exactly q[i+1]-q[i], one row per adjacent pair. Descent is bounded above
        # by r_down*dt and recovery below by -r_up*dt, both per sample of duration dt.
        Dr = D1
        cons += [{"type": "ineq", "fun": lambda q: r_down * dt - Dr @ q,
                  "jac": lambda q: -Dr},
                 {"type": "ineq", "fun": lambda q: Dr @ q + r_up * dt,
                  "jac": lambda q: Dr}]
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
