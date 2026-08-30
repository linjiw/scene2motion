"""Local repair of a duck schedule from a measured clearance deficit.

The scheduler plans against `g`, a fitted monotone map from duck command to top-of-body
height, whose holdout error is ~43 mm. When the real motion comes back short, the honest
response is not to re-solve the whole schedule against the same surrogate that just missed --
it is to correct the surrogate locally using the measurement that exposed it.

    e(s)      = max(0, target - measured(s))      metres of missing headroom
    dq(s)     = e(s) / |g'(q(s))|                  command needed to buy it back
    dq        = anticipate(dq)                     shifted early, because the body lags
    q'        = smooth(clip(q + dq, 0, 1))

`|g'|` is floored: near saturation the fitted gain goes flat, and dividing by a flat slope
turns a 1 cm deficit into a demand for infinite crouch. Where the floor binds, the repair is
deliberately partial -- it moves as far as the response actually supports and the next
verification says whether that was enough. Two iterations, then the route is rejected.

Repair is bounded on purpose. An unbounded loop against a noisy prior converges to whatever
the seed happened to do; two attempts is enough to fix a surrogate error and not enough to
launder a route that was never feasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..optim.response import DIP_MAX, DuckResponse

MAX_REPAIRS = 2
# Below this the fitted gain is flat enough that inversion is meaningless. 0.15 m of top
# height per unit command is ~30% of the full-scale gain -- past that the response curve is
# saturating and more command buys almost nothing.
SLOPE_FLOOR = 0.15
SLOPE_WINDOW_Q = 0.12       # secant half-width for the local slope, in command units
SMOOTH_WIN = 5              # samples; the schedule grid is 64 over the whole route


def local_slope(resp: DuckResponse, q: np.ndarray, h: float = SLOPE_WINDOW_Q) -> np.ndarray:
    """|dg/dq| by FORWARD secant over [q, q+h] -- the direction the repair actually moves.

    `g` is a PAVA fit: piecewise constant in places, so its pointwise derivative is zero on
    whole segments that are not actually flat at the scale we care about. A secant over a
    finite command window measures the slope the repair can actually use.

    Forward, not centred, because `g` is convex over most of its range: a centred secant
    borrows the steep gain BEHIND the current command, which is not available to a correction
    that increases it. Measured against the exact inverse for a 5 cm deficit, the forward
    secant lands within 4 mm of command across q in [0, 0.7] where the centred one undershoots
    by up to 27% -- and a systematic undershoot is expensive when the loop stops after two
    iterations.
    """
    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    lo = q
    hi = np.clip(q + h, 0.0, 1.0)
    # At the very top of the range the forward window collapses; fall back to looking behind,
    # which is the only slope information left there.
    lo = np.where(hi - lo < h / 2, np.clip(hi - h, 0.0, 1.0), lo)
    span = np.maximum(hi - lo, 1e-6)
    return np.abs(resp.g(hi) - resp.g(lo)) / span


def anticipation_kernel(n: int, ds_m: float, speed: float, tau_s: float,
                        trail_s: float = 0.25) -> tuple[int, int]:
    """How many samples early (and late) a correction must start, in route samples.

    The body is a first-order lag with time constant tau. Reaching ~95% of a commanded step
    takes 3*tau, so a correction that begins where the deficit begins arrives after the beam.
    The lead is derived from the measured tau rather than reusing the planner's hand-set
    LEAD_S, so it tracks the fit.
    """
    ds_m = max(ds_m, 1e-6)
    lead_m = 3.0 * max(tau_s, 1e-3) * max(speed, 1e-3)
    return (min(int(np.ceil(lead_m / ds_m)), n),
            min(int(np.ceil(trail_s * max(speed, 1e-3) / ds_m)), n))


def _dilate(x: np.ndarray, lead: int, trail: int) -> np.ndarray:
    """Running maximum over [i-trail, i+lead]: hold the correction early and release late."""
    n = len(x)
    out = np.array(x, float)
    for k in range(1, max(lead, trail) + 1):
        if k <= lead:
            out[: n - k] = np.maximum(out[: n - k], x[k:])      # pull future demand earlier
        if k <= trail:
            out[k:] = np.maximum(out[k:], x[: n - k])           # hold it a little longer
    return out


def _smooth(x: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    if win <= 1:
        return np.array(x, float)
    k = np.ones(win) / win
    return np.convolve(np.pad(x, win // 2, mode="edge"), k, mode="valid")[: len(x)]


@dataclass
class RepairStep:
    """One correction, with everything needed to say what it changed and why."""
    iteration: int
    max_deficit_m: float
    deficit_support: int
    max_delta_q: float
    repair_magnitude_m: float      # peak added dip
    onset_shift_m: float           # route distance the duck now starts earlier (negative = earlier)
    duration_change_m: float       # route distance the duck is now held longer
    slope_floor_bound: int         # samples where the gain floor limited the correction
    q_before_hash: str
    q_after_hash: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: (round(v, 5) if isinstance(v, float) else v)
             for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def _support_span(dip: np.ndarray, s_m: np.ndarray, thresh: float = 0.02) -> tuple[float, float]:
    on = np.flatnonzero(dip > thresh)
    if not len(on):
        return (float("nan"), float("nan"))
    return float(s_m[on[0]]), float(s_m[on[-1]])


def repair(q: np.ndarray, deficit_m: np.ndarray, resp: DuckResponse, s_m: np.ndarray,
           speed: float, iteration: int, tau_s: float | None = None) -> tuple[np.ndarray, RepairStep]:
    """One bounded local correction. Returns the new command schedule and its record."""
    from .trace import schedule_hash

    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    e = np.maximum(np.asarray(deficit_m, float), 0.0)
    tau = float(tau_s if tau_s is not None else resp.tau_s)
    ds = float(np.mean(np.diff(s_m))) if len(s_m) > 1 else 0.1

    slope = local_slope(resp, q)
    bound = int(((slope < SLOPE_FLOOR) & (e > 1e-9)).sum())
    dq = e / np.maximum(slope, SLOPE_FLOOR)

    lead, trail = anticipation_kernel(len(q), ds, speed, tau)
    dq = _smooth(_dilate(dq, lead, trail))
    q_new = np.clip(q + dq, 0.0, 1.0)

    d0, d1 = q * DIP_MAX, q_new * DIP_MAX
    a0, b0 = _support_span(d0, s_m)
    a1, b1 = _support_span(d1, s_m)
    onset = 0.0 if np.isnan(a0) or np.isnan(a1) else a1 - a0
    dur = 0.0 if np.isnan(a0) or np.isnan(a1) else (b1 - a1) - (b0 - a0)

    step = RepairStep(
        iteration=iteration,
        max_deficit_m=float(e.max()) if len(e) else 0.0,
        deficit_support=int((e > 1e-9).sum()),
        max_delta_q=float((q_new - q).max()),
        repair_magnitude_m=float((d1 - d0).max()),
        onset_shift_m=float(onset),
        duration_change_m=float(dur),
        slope_floor_bound=bound,
        q_before_hash=schedule_hash(q),
        q_after_hash=schedule_hash(q_new),
        extra={"lead_samples": lead, "trail_samples": trail, "tau_s": round(tau, 4)},
    )
    return q_new, step
