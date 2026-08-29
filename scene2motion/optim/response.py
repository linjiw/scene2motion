"""How the frozen prior actually responds to a duck command.

Two pieces, both deliberately small:

    g(q)                static gain -- commanded duck q in [0,1] -> achieved top-of-head height
    z[t+1] = z[t] + dt/tau * (q[t] - z[t])    first-order lag -- the body cannot change
                                              envelope instantly, which is WHY anticipation is
                                              necessary and why two close beams share a crouch

`g` is fitted from EXP-001d, which already swept commanded dip against measured top height over
20 samples per rung with the robot's own collision geometry. Isotonic regression is used rather
than a polynomial because the only structural fact worth imposing is monotonicity -- asking for
more duck must not raise the head -- and a monotone step function makes the inverse well defined
without inventing curvature the 6 rungs cannot support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DIP_MAX = 0.50          # m, the program's duck axis full scale; q = dip / DIP_MAX
EXP001D = Path("outputs/exp001d/rows.jsonl")


def alpha(dt: float, tau: float) -> float:
    """Zero-order-hold discretisation of a first-order lag: a = 1 - exp(-dt/tau).

    The naive `dt/tau` is only valid for dt << tau and has to be clipped at 1 otherwise, which
    silently turns the lag off exactly when the route sampling is coarse -- 64 samples over a
    12 m route gives dt = 0.21 s against tau = 0.21 s, so the naive form clips and the model
    claims the body responds instantly. The exact form approaches 1 smoothly and is correct for
    any dt, so schedule length can no longer change the physics.
    """
    return float(1.0 - np.exp(-max(dt, 0.0) / max(tau, 1e-6)))


@dataclass
class DuckResponse:
    """Piecewise-linear monotone gain plus a first-order lag."""

    q_knots: np.ndarray          # ascending commanded duck, [0, 1]
    h_knots: np.ndarray          # non-increasing achieved top height, m
    tau_s: float = 0.35          # response time constant
    tau_lo: float = 0.25         # sensitivity bracket, not a confidence interval
    tau_hi: float = 0.50
    fit: dict = field(default_factory=dict)

    # ---- static gain ----------------------------------------------------------------
    def g(self, q) -> np.ndarray:
        """Commanded duck -> achieved top height (m)."""
        return np.interp(np.clip(q, 0.0, 1.0), self.q_knots, self.h_knots)

    def g_inv(self, h) -> np.ndarray:
        """Smallest commanded duck whose achieved top height is at or below `h`.

        `h_knots` DESCENDS with q, so inverting it needs the reversed arrays: np.interp
        requires an ascending `xp`, and `h_knots[::-1]` is ascending while `q_knots[::-1]` is
        the matching descending `fp`. Negating a descending array instead -- the first version
        of this -- silently violates that precondition and np.interp returns the left endpoint
        for every query, i.e. "no duck is ever needed".

        Clearances above the standing head need no duck (0); below the deepest measured crouch
        the demand saturates at 1 and the caller must treat the scene as infeasible rather than
        assume q=1 clears it.
        """
        h = np.asarray(h, float)
        q = np.interp(h, self.h_knots[::-1], self.q_knots[::-1])
        return np.clip(np.where(h >= self.h_knots[0], 0.0, q), 0.0, 1.0)

    def clears(self, h) -> np.ndarray:
        """Is a clearance of `h` reachable at all? False means refuse, not "duck harder"."""
        return np.asarray(h, float) >= self.h_knots[-1]

    # ---- dynamics -------------------------------------------------------------------
    def roll(self, q: np.ndarray, dt: float, tau: float | None = None,
             z0: float = 0.0) -> np.ndarray:
        """Integrate the lag: commanded schedule -> realised duck state."""
        tau = self.tau_s if tau is None else tau
        a = alpha(dt, tau)
        z = np.empty(len(q))
        cur = z0
        for i, u in enumerate(q):
            cur = cur + a * (u - cur)
            z[i] = cur
        return z

    def top_from_command(self, q: np.ndarray, dt: float,
                         tau: float | None = None) -> np.ndarray:
        return self.g(self.roll(np.asarray(q, float), dt, tau))

    # ---- io -------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"q_knots": self.q_knots.tolist(), "h_knots": self.h_knots.tolist(),
                "tau_s": self.tau_s, "tau_lo": self.tau_lo, "tau_hi": self.tau_hi,
                "dip_max": DIP_MAX, "fit": self.fit}

    @staticmethod
    def from_dict(d: dict) -> "DuckResponse":
        return DuckResponse(np.asarray(d["q_knots"], float), np.asarray(d["h_knots"], float),
                            d["tau_s"], d.get("tau_lo", 0.25), d.get("tau_hi", 0.50),
                            d.get("fit", {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @staticmethod
    def load(path: str | Path = "outputs/duck_response/response.json") -> "DuckResponse":
        return DuckResponse.from_dict(json.loads(Path(path).read_text()))


def _isotonic_decreasing(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a NON-INCREASING fit, on x-sorted data."""
    y = y.astype(float).copy()
    w = np.ones_like(y)
    i = 0
    while i < len(y) - 1:
        if y[i] < y[i + 1]:                       # violates non-increasing
            tot = w[i] + w[i + 1]
            y[i] = (w[i] * y[i] + w[i + 1] * y[i + 1]) / tot
            w[i] = tot
            y = np.delete(y, i + 1)
            w = np.delete(w, i + 1)
            x = np.delete(x, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    return x, y


def fit_static(path: Path = EXP001D, holdout_frac: float = 0.3,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit g from EXP-001d's dip sweep, holding out samples for an honest error."""
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("channel") == "dip" and r.get("top_adapted_m")]
    q = np.array([r["value"] for r in rows]) / DIP_MAX
    h = np.array([r["top_adapted_m"] for r in rows])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(q))
    n_ho = int(holdout_frac * len(q))
    ho, tr = idx[:n_ho], idx[n_ho:]

    def fit_on(sel):
        levels = np.unique(q[sel])
        means = np.array([h[sel][q[sel] == v].mean() for v in levels])
        return _isotonic_decreasing(levels.copy(), means.copy())

    qk, hk = fit_on(tr)
    resp = DuckResponse(qk, hk)
    res_tr = resp.g(q[tr]) - h[tr]
    res_ho = resp.g(q[ho]) - h[ho]
    info = {
        "source": str(path), "n_rows": len(q), "n_train": len(tr), "n_holdout": len(ho),
        "levels": np.unique(q).tolist(),
        "train_mae_m": float(np.abs(res_tr).mean()),
        "train_rmse_m": float(np.sqrt((res_tr ** 2).mean())),
        "holdout_mae_m": float(np.abs(res_ho).mean()),
        "holdout_rmse_m": float(np.sqrt((res_ho ** 2).mean())),
        "residual_sd_m": float(res_tr.std()),
        "per_level_sd_m": {float(v): float(h[q == v].std()) for v in np.unique(q)},
    }
    return qk, hk, info
