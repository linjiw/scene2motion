"""Tests for the E1a analysis helpers, which now carry load-bearing numbers.

The Wilson bound sizes best-of-N, the box-height profile decides where a lift is said to
be, and the selection curve is what the E2 protocol will be preregistered against.
"""

import numpy as np
import pytest

from experiments.analyze_e1a_placement import (
    lift_location,
    wilson_interval,
)
from experiments.exp021_elicited_lift_distribution import (
    best_of_n_curve,
    clears,
)


def test_wilson_interval_zero_successes_gives_the_bound_we_quote():
    low, high = wilson_interval(0, 16)
    assert low == 0.0
    # 0/16 is what bounds the packet's per-sample hit rate in the write-up.
    assert high == pytest.approx(0.19, abs=0.01)
    low8, high8 = wilson_interval(0, 8)
    assert high8 > high, "a smaller denominator must give a weaker bound"


def test_wilson_interval_edges_and_monotonicity():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(8, 8)
    assert high == 1.0 and 0.0 < low < 1.0
    mid_low, mid_high = wilson_interval(3, 8)
    assert mid_low < 3 / 8 < mid_high


def test_lift_location_reports_no_lift_on_a_flat_zero_profile():
    xs = np.linspace(0.0, 7.0, 50)
    result = lift_location(xs, np.zeros_like(xs))
    assert result["lift_x_m"] is None
    assert result["lift_height_m"] == 0.0
    assert result["n_lift_regions"] == 0


def test_lift_location_finds_the_peak_and_counts_separate_regions():
    xs = np.linspace(0.0, 10.0, 101)
    heights = np.zeros_like(xs)
    heights[20:25] = 0.05
    heights[60:63] = 0.12          # the taller, later lift
    result = lift_location(xs, heights)
    assert result["lift_height_m"] == pytest.approx(0.12)
    assert result["lift_x_m"] == pytest.approx(xs[60], abs=0.2)
    assert result["n_lift_regions"] == 2
    assert result["lift_support_m"] > 0.0


def test_clears_respects_both_the_radius_and_the_height():
    xs = np.linspace(0.0, 10.0, 101)
    heights = np.zeros_like(xs)
    heights[50] = 0.10             # a single 10 cm lift at x = 5.0
    assert clears(xs, heights, 5.0, 0.25, 0.05)
    assert not clears(xs, heights, 5.0, 0.25, 0.20), "height must bind"
    assert not clears(xs, heights, 8.0, 0.25, 0.05), "radius must bind"
    assert clears(xs, heights, 8.0, 3.10, 0.05), "a wide radius reaches it"


def test_best_of_n_curve_is_the_independent_resampling_law():
    xs = np.linspace(0.0, 10.0, 101)
    hit = np.zeros_like(xs)
    hit[50] = 0.10
    miss = np.zeros_like(xs)
    # Two clips, one of which clears the single target: per-clip rate 0.5.
    curve = best_of_n_curve(
        [(xs, hit), (xs, miss)], np.asarray([5.0]), radius_m=0.25, height_m=0.05)
    assert curve["per_clip_rate"] == pytest.approx(0.5)
    assert curve["N=1"] == pytest.approx(0.5)
    assert curve["N=2"] == pytest.approx(0.75)
    assert curve["N=4"] == pytest.approx(0.9375)
    assert curve["N=32"] > curve["N=16"] > curve["N=8"]


def test_best_of_n_curve_saturates_and_floors_cleanly():
    xs = np.linspace(0.0, 10.0, 101)
    hit = np.zeros_like(xs)
    hit[50] = 0.10
    targets = np.asarray([5.0])
    always = best_of_n_curve([(xs, hit)], targets, radius_m=0.25, height_m=0.05)
    assert always["per_clip_rate"] == 1.0
    assert all(always[f"N={n}"] == 1.0 for n in (1, 2, 32))
    never = best_of_n_curve(
        [(xs, np.zeros_like(xs))], targets, radius_m=0.25, height_m=0.05)
    assert never["per_clip_rate"] == 0.0
    assert all(never[f"N={n}"] == 0.0 for n in (1, 32))
