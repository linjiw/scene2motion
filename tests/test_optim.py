"""Tests for the Phase 3 response surrogate and schedule optimiser. CPU only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.optim.response import DIP_MAX, DuckResponse, alpha
from scene2motion.optim.scheduler import dt_for, lag_matrix, solve

RESP = Path("outputs/duck_response/response.json")
needs_resp = pytest.mark.skipif(not RESP.exists(), reason="response not fitted")


def beam_profile(n=64, lo=26, hi=31, h=1.05, ceiling=2.6):
    c = np.full(n, ceiling)
    c[lo:hi] = h
    return c


# ---- discretisation ------------------------------------------------------------------

def test_alpha_is_bounded_for_any_step():
    """The naive dt/tau EXCEEDS 1 for coarse sampling and had to be clipped, which silently
    turned the lag off exactly when route sampling was coarse. The exact form must never
    exceed 1 and must increase with dt -- it may reach 1 for dt >> tau, which is correct
    rather than degenerate: a step far longer than the time constant really does settle."""
    prev = 0.0
    for dt in (0.001, 0.1, 0.21, 1.0, 100.0):
        a = alpha(dt, 0.19)
        assert 0.0 < a <= 1.0, (dt, a)
        assert a >= prev, "alpha must increase with dt"
        prev = a
    assert alpha(0.21, 0.19) < 1.0, "at the sampling that broke the naive form, a lag remains"
    assert alpha(0.19, 0.19) == pytest.approx(1 - np.exp(-1), abs=1e-9)


def test_lag_matrix_rows_sum_to_a_settling_response():
    L = lag_matrix(64, 0.14, 0.19)
    assert np.allclose(np.triu(L, 1), 0.0), "the lag must be causal"
    assert L.sum(axis=1)[-1] == pytest.approx(1.0, abs=1e-3), "a held command must settle at 1"


# ---- static gain ---------------------------------------------------------------------

@needs_resp
def test_gain_is_monotone_non_increasing():
    r = DuckResponse.load(RESP)
    q = np.linspace(0, 1, 101)
    h = r.g(q)
    assert np.all(np.diff(h) <= 1e-9), "more duck must never raise the head"


@needs_resp
def test_inverse_round_trips_exactly():
    r = DuckResponse.load(RESP)
    q = np.linspace(0, 1, 21)
    assert np.abs(r.g_inv(r.g(q)) - q).max() < 1e-6


@needs_resp
def test_unreachable_clearance_is_refused_not_saturated():
    r = DuckResponse.load(RESP)
    deepest = float(r.g(1.0))
    assert not bool(r.clears(deepest - 0.05))
    assert bool(r.clears(deepest + 0.05))


# ---- optimiser -----------------------------------------------------------------------

@needs_resp
def test_no_beam_gives_exactly_no_duck():
    r = DuckResponse.load(RESP)
    s = solve(np.full(64, 2.6), r, dt_for(8.0, 64, 0.9))
    assert s.feasible and s.q.max() == pytest.approx(0.0, abs=1e-6)
    assert s.objective == pytest.approx(0.0, abs=1e-9)


@needs_resp
def test_solution_satisfies_the_clearance_constraint():
    r = DuckResponse.load(RESP)
    s = solve(beam_profile(), r, dt_for(8.0, 64, 0.9))
    assert s.feasible and s.max_violation_m < 1e-6
    assert s.q.min() >= -1e-9 and s.q.max() <= 1 + 1e-9


@needs_resp
def test_lower_beam_needs_more_duck():
    r = DuckResponse.load(RESP)
    dt = dt_for(8.0, 64, 0.9)
    peaks = [solve(beam_profile(h=h), r, dt).q.max() for h in (1.25, 1.15, 1.05, 0.95)]
    assert all(a <= b + 1e-9 for a, b in zip(peaks, peaks[1:])), peaks


@needs_resp
def test_anticipation_emerges_without_being_encoded():
    """Nothing in the objective mentions lead time; it follows from the lag."""
    r = DuckResponse.load(RESP)
    s = solve(beam_profile(lo=26, hi=31), r, dt_for(8.0, 64, 0.9))
    on = np.where(s.q > 0.03)[0]
    assert len(on) and on[0] < 26, f"command starts at {on[0]}, beam at 26"


@needs_resp
def test_impossible_beam_is_reported_infeasible():
    r = DuckResponse.load(RESP)
    s = solve(beam_profile(h=0.55), r, dt_for(8.0, 64, 0.9))
    assert not s.feasible and "infeasible" in s.status


@needs_resp
def test_close_beams_merge_and_distant_beams_split():
    """The merge/split decision is the optimiser's, not a rule's."""
    r = DuckResponse.load(RESP)
    n, L, speed = 64, 12.0, 0.9
    dt = dt_for(L, n, speed)

    def between_min(gap_m):
        c = np.full(n, 2.6)
        i0, w = 20, 2
        i1 = int(i0 + gap_m / L * n)
        c[i0:i0 + w] = 1.05
        c[i1:i1 + w] = 1.05
        s = solve(c, r, dt)
        assert s.feasible
        return s.q[i0 + w:i1].min()

    close, far = between_min(1.0), between_min(5.0)
    assert close > 0.2, f"close beams should stay crouched, got {close:.3f}"
    assert far < 0.05, f"distant beams should recover, got {far:.3f}"
    assert close > far


@needs_resp
def test_optimiser_is_deterministic():
    r = DuckResponse.load(RESP)
    dt = dt_for(8.0, 64, 0.9)
    a = solve(beam_profile(), r, dt)
    b = solve(beam_profile(), r, dt)
    assert np.array_equal(a.q, b.q) and a.objective == b.objective
