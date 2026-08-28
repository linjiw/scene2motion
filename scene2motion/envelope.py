# Scene2Motion-G1: the calibrated capability envelope.
#
# What "capability-calibrated planning" actually means: the planner may only propose body
# envelopes the frozen prior has been MEASURED to reach, and the measurement carries a stated
# coverage guarantee rather than being a worst-of-a-few lookup.
#
# This replaces the seven-mode table in outputs/body_modes.json for continuous requests. That
# table was the worst of three seeds per mode, which is a 36.8 %-content tolerance interval at
# 95 % confidence -- it says almost nothing about a fresh sample, and the same nominal
# condition produced worst-of-3 half-widths of 0.281 m and 0.380 m in two different
# experiments purely from which draws were taken.
#
# EXP-001d re-measures both axes at 20 independent samples per level and reports a split
# conformal upper bound: the ceil((n+1)(1-alpha))-th order statistic is a valid (1-alpha)
# bound for a fresh exchangeable sample, so n=20 gives a genuine 90 % bound where n=3 could
# not give one at all.
#
# THE CLAIM THIS LICENSES, exactly: for a request drawn from the calibration distribution, the
# body's envelope exceeds the bound at most 10 % of the time. It is a statement about the
# ENVELOPE, not about collision-freeness; the per-instance guarantee remains the MuJoCo check,
# which is why the metric tables keep them in separate columns.
#
# What the re-measurement changed
# -------------------------------
# `top(dip)` is monotone decreasing and well behaved: 1.336 m -> 0.879 m over dip 0 -> 0.50.
#
# `half_width(dip)` is monotone INCREASING: 0.378 m -> 0.510 m at the 90 % bound. Ducking
# genuinely makes the robot wider — the arms come out for balance — by ~9 cm in the mean and
# ~13 cm at the bound. An earlier three-seed analysis concluded the two axes were decoupled
# and that reading was wrong: the per-level standard deviation is 4-7 cm, so a 9 cm effect is
# invisible at n=3. The coupling is real, it is modest, and the planner must respect it: a
# deep duck needs a 1.02 m corridor, not the 0.76 m a standing envelope would suggest.
#
# `half_width(tuck)` is monotone decreasing AND variance-reducing: 0.378 -> 0.272 m at the
# bound, with the standard deviation falling 0.044 -> 0.012. Tucking makes the robot both
# narrower and more repeatable, which is why it is the axis worth spending on when a gap is
# tight.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ENVELOPE_JSON = Path(__file__).resolve().parents[1] / "outputs" / "exp001d" / "envelope.json"


class Envelope:
    """Calibrated (dip, tuck) -> (top height, lateral half-width), with coverage."""

    def __init__(self, path: Path = _ENVELOPE_JSON, bound: str = "p90_conformal"):
        raw = json.loads(Path(path).read_text())
        self.alpha = raw["alpha"]
        self.n_samples = raw["n_samples"]
        self.bound = bound
        d = raw["envelope"]["dip"]
        t = raw["envelope"]["tuck"]
        self.dips = np.array([e["value"] for e in d])
        self.top_dip = np.array([e[f"top_{bound}"] for e in d])
        self.hw_dip = np.array([e[f"hw_{bound}"] for e in d])
        self.tucks = np.array([e["value"] for e in t])
        self.hw_tuck = np.array([e[f"hw_{bound}"] for e in t])
        # Enforce the monotonicity the measurement supports, so interpolation cannot invert
        # the physical relationship on a noisy level. top falls with dip; half-width rises
        # with dip and falls with tuck.
        self.top_dip = np.minimum.accumulate(self.top_dip)
        self.hw_dip = np.maximum.accumulate(self.hw_dip)
        self.hw_tuck = np.minimum.accumulate(self.hw_tuck)

    def top(self, dip: float | np.ndarray) -> np.ndarray:
        """Top-of-robot height at this pelvis dip, as an upper bound."""
        return np.interp(np.clip(dip, self.dips[0], self.dips[-1]), self.dips, self.top_dip)

    def half_width(self, dip: float | np.ndarray = 0.0,
                   tuck: float | np.ndarray = 0.0) -> np.ndarray:
        """Lateral half-extent, as an upper bound.

        The two axes were calibrated separately, so their combination is approximated by
        applying the tuck REDUCTION measured at nominal pelvis height to the dip curve. That
        is an assumption, not a measurement: EXP-001's duck+tuck cells hint the axes are not
        perfectly separable, and a joint 2-D sweep is the honest way to settle it. The
        approximation is deliberately conservative -- it never returns less than the dip curve
        alone would minus the full tuck credit -- but it is the weakest link in the envelope
        and should not be leaned on for tight gaps without the joint sweep.
        """
        base = np.interp(np.clip(dip, self.dips[0], self.dips[-1]), self.dips, self.hw_dip)
        credit = self.hw_tuck[0] - np.interp(np.clip(tuck, self.tucks[0], self.tucks[-1]),
                                             self.tucks, self.hw_tuck)
        return np.maximum(base - credit, self.hw_tuck[-1])

    def max_dip_for_ceiling(self, free_height: float, margin: float = 0.0) -> float | None:
        """Smallest dip whose bounded top clears `free_height`, or None if none does."""
        ok = np.where(self.top_dip + margin <= free_height)[0]
        return float(self.dips[ok[0]]) if len(ok) else None

    def __repr__(self) -> str:
        return (f"Envelope(n={self.n_samples}, {int((1-self.alpha)*100)}% conformal, "
                f"top {self.top_dip[0]:.3f}->{self.top_dip[-1]:.3f} m, "
                f"half-width {self.hw_dip[0]:.3f}->{self.hw_dip[-1]:.3f} m over dip)")


_CACHED: Envelope | None = None


def get_envelope() -> Envelope:
    global _CACHED
    if _CACHED is None:
        _CACHED = Envelope()
    return _CACHED
