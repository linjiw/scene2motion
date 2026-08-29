"""Focused tests for the demo layer. CPU only -- no ARDY, no GPU.

Everything here is about the parts a user can break with a slider: does the scene respond to
its two parameters, do the three preferences actually differ, is the cache key sensitive to
what it must be sensitive to, and does the panel refuse to overclaim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.demo.cache import ClipCache, key_for
from scene2motion.demo.scene_builder import (BeamParams, beam_footprint, build,
                                             save_default)
from scene2motion.demo.strategy_planner import (PREFERENCES, UPRIGHT_MODES, evaluate,
                                                evaluate_all)

H_DUCKABLE = 1.00        # low enough to need a duck, high enough that duck_max clears it
W_DEFAULT = 1.45


# ---- scene ---------------------------------------------------------------------------

def test_beam_height_and_width_are_the_only_things_that_move():
    a = build(BeamParams(1.00, W_DEFAULT))
    b = build(BeamParams(1.20, W_DEFAULT))
    assert a.start == b.start and a.goal == b.goal
    fa, fb = beam_footprint(a), beam_footprint(b)
    assert fa["x_lo"] == fb["x_lo"] and fa["y_lo"] == fb["y_lo"]      # only height changed
    assert fb["z_lo"] > fa["z_lo"]


def test_beam_width_extends_from_the_left_wall_and_leaves_a_bypass():
    narrow = build(BeamParams(H_DUCKABLE, 0.60))
    wide = build(BeamParams(H_DUCKABLE, 2.10))
    assert beam_footprint(wide)["y_lo"] < beam_footprint(narrow)["y_lo"]
    for s in (narrow, wide):
        assert s.meta["bypass_width"] > 0.0, "a bypass must always exist"


def test_params_are_clamped_into_a_feasible_range():
    p = BeamParams(beam_height=99.0, beam_width=99.0).clamped()
    assert 0.60 <= p.beam_height <= 1.60
    assert p.beam_width <= 2 * 1.2 - 0.15


def test_default_scene_file_round_trips(tmp_path):
    out = save_default(tmp_path / "partial_beam.json")
    assert out.exists() and out.stat().st_size > 0


# ---- preferences ---------------------------------------------------------------------

def test_three_preferences_give_three_distinct_plans():
    S = evaluate_all(build(BeamParams(H_DUCKABLE, W_DEFAULT)))
    assert all(S[k].feasible for k in PREFERENCES)
    assert S["shortest"].goes_under_beam is True
    assert S["upright"].goes_under_beam is False
    assert S["clearance"].goes_under_beam is False
    # Shortest is shortest; clearance pays for its wider berth.
    assert S["shortest"].path_length_m < S["upright"].path_length_m
    assert S["clearance"].path_length_m > S["upright"].path_length_m


def test_stay_upright_never_ducks():
    for h in (1.30, 1.10, 1.00):
        s = evaluate(build(BeamParams(h, W_DEFAULT)), "upright")
        assert s.feasible and not s.duck_required
        assert all(m.name in UPRIGHT_MODES for m in s.plan.modes)


def test_maximum_clearance_keeps_further_from_the_beam_than_stay_upright():
    sc = build(BeamParams(H_DUCKABLE, W_DEFAULT))
    up, cl = evaluate(sc, "upright"), evaluate(sc, "clearance")
    assert float(np.min(cl.plan.xy[:, 1])) < float(np.min(up.plan.xy[:, 1]))


def test_lower_beam_forces_a_deeper_duck():
    tops = []
    for h in (1.30, 1.10, 1.00):
        s = evaluate(build(BeamParams(h, W_DEFAULT)), "shortest")
        assert s.duck_required, f"h={h} should still duck"
        tops.append(min(m.top for m in s.plan.modes))
    assert tops[0] > tops[1] >= tops[2], f"deeper beam must not need a shallower duck: {tops}"


def test_a_beam_too_low_for_any_duck_is_routed_around_not_through():
    s = evaluate(build(BeamParams(0.80, W_DEFAULT)), "shortest")
    assert s.feasible, "the bypass must still be found"
    assert not s.goes_under_beam


def test_a_high_beam_needs_no_duck_at_all():
    s = evaluate(build(BeamParams(1.60, W_DEFAULT)), "shortest")
    assert s.feasible and not s.duck_required


def test_duck_window_is_reported_honestly():
    s = evaluate(build(BeamParams(H_DUCKABLE, W_DEFAULT)), "shortest")
    assert s.duck_start_s is not None and s.duck_end_s is not None
    assert s.duck_end_s >= s.duck_start_s
    assert not s.duck_held_throughout, "at this height the duck should be localised"
    # It must start BEFORE the beam, which is the whole point of the anticipation lead.
    beam_x = beam_footprint(build(BeamParams(H_DUCKABLE, W_DEFAULT)))["x_lo"]
    assert s.duck_start_s * 0.9 < beam_x


def test_panel_dict_never_claims_physics_validation():
    d = evaluate(build(BeamParams(H_DUCKABLE, W_DEFAULT)), "shortest").to_dict()
    assert "sonic" not in json_lower(d) and "physics" not in json_lower(d)


def json_lower(d) -> str:
    import json
    return json.dumps(d).lower()


# ---- cache ---------------------------------------------------------------------------

def test_cache_key_depends_on_every_input_that_changes_the_clip():
    base = dict(scene_id="s", preference="shortest", program_bytes=b"p", model="m",
                seed=1, steps=5, fps=25.0, n_frames=200)
    k0 = key_for(**base)
    for field, value in (("scene_id", "other"), ("preference", "upright"),
                         ("program_bytes", b"q"), ("model", "m2"), ("seed", 2),
                         ("steps", 10), ("fps", 30.0), ("n_frames", 201)):
        assert key_for(**{**base, field: value}) != k0, f"{field} must change the key"


def test_cache_round_trips_and_reports_stats(tmp_path):
    c = ClipCache(tmp_path)
    assert c.get("nope") is None
    q = np.zeros((10, 36), np.float32)
    c.put("abc", q, {"collision_free": True})
    got = c.get("abc")
    assert got is not None and got[0].shape == (10, 36)
    assert got[1]["collision_free"] is True and got[1]["key"] == "abc"
    assert c.stats()["n_entries"] == 1
