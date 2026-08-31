"""Phase 4A: the verify -> repair contract.

The claim these guard is narrow and load-bearing: a clip called "repaired" was regenerated
from the repaired schedule and independently reverified. Everything here exists to make that
claim falsifiable rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scene2motion.demo.cache import ClipCache
from scene2motion.demo.scene_builder import BeamParams, build
from scene2motion.demo.strategy_planner import evaluate
from scene2motion.optim.response import DIP_MAX, DuckResponse
from scene2motion.robot import G1Body
from scene2motion.verify import loop as loopmod
from scene2motion.verify.repair import MAX_REPAIRS, SLOPE_FLOOR, local_slope, repair
from scene2motion.verify.trace import clearance_trace, schedule_hash

RESP = Path("outputs/duck_response/response.json")
N = 64


@pytest.fixture(scope="module")
def resp():
    if not RESP.exists():
        pytest.skip("no fitted duck response")
    return DuckResponse.load(RESP)


# -- schedule identity ------------------------------------------------------------------

def test_schedule_hash_distinguishes_schedules_and_ignores_float_noise():
    q = np.linspace(0, 1, N)
    assert schedule_hash(q) == schedule_hash(q + 1e-9)      # below the rounding grid
    assert schedule_hash(q) != schedule_hash(q + 1e-3)      # above it


def test_cache_key_binds_schedule_and_repair_iteration(tmp_path, monkeypatch):
    """The bug this prevents: serving a pre-repair clip for a post-repair request.

    Route, scene, seed, model and preference are held identical; only the schedule and the
    iteration move. Both must fork the key on their own.
    """
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    p = evaluate(sc, "shortest").plan

    class FakeRunner:
        fps, model_name = 25.0, "fake"
    monkeypatch.setattr(loopmod, "get_runner", lambda: FakeRunner())
    cache = ClipCache(tmp_path)

    def key(q, it):
        return loopmod.generate_from_schedule(sc, p, q, cache, iteration=it,
                                              preference="shortest", allow_generate=False)["key"]

    base = np.full(N, 0.4)
    assert key(base, 0) != key(base + 0.05, 0), "different schedule must not share a key"
    assert key(base, 0) != key(base, 1), "different repair iteration must not share a key"
    assert key(base, 0) == key(base.copy(), 0), "same request must be reproducible"


# -- the clearance trace ----------------------------------------------------------------

def _walk(scene, height_z: float, T: int = 60) -> np.ndarray:
    """A synthetic straight walk at a fixed pelvis height, in ARDY-exported qpos layout."""
    qpos = np.zeros((T, 36))
    qpos[:, 0] = np.linspace(0.0, float(scene.goal[0]), T)
    qpos[:, 2] = height_z
    qpos[:, 3] = 1.0                                        # identity root quaternion
    return qpos


def test_trace_is_aligned_to_the_beam_it_passes_under():
    """A deficit must land at the beam's route position, not somewhere else on the route."""
    bp = BeamParams(0.95, 2.25, 1, 3.0).clamped()
    sc = build(bp)
    p = evaluate(sc, "shortest").plan
    tr = clearance_trace(sc, _walk(sc, 0.79), p.xy)

    beam_x = sc.meta["beam_xs"][0]
    worst_s = float(tr.s_m[int(np.argmin(tr.overhead))])
    # The route here runs straight down the corridor, so route distance tracks x.
    assert abs(worst_s - beam_x) < 0.6, f"worst overhead at s={worst_s}, beam at x={beam_x}"
    assert tr.overhead.min() < tr.overhead.max(), "trace must vary along the route"


def test_trace_min_agrees_with_the_whole_clip_report():
    sc = build(BeamParams(1.05, 1.45, 2, 3.0).clamped())
    p = evaluate(sc, "shortest").plan
    qpos = _walk(sc, 0.85)
    tr = clearance_trace(sc, qpos, p.xy)
    assert tr.clearance.min() == pytest.approx(tr.min_clearance_m, abs=1e-6)


def test_collision_and_below_margin_are_separate_outcomes():
    """Below the target margin is not a collision. The trace must never merge the two.

    Tested on the type directly: this is a property of how an outcome is classified, not of
    MuJoCo, and the numbers below are the shape of a real m018 near-miss (a few centimetres
    of headroom against an 0.18 m target).
    """
    from scene2motion.verify.trace import ClearanceTrace
    s = np.linspace(0, 8, N)

    tight = ClearanceTrace(s_m=s, clearance=np.full(N, 0.04), overhead=np.full(N, 0.04),
                           min_clearance_m=0.04, min_overhead_m=0.04,
                           collision_free=True, goal_error_m=0.05)
    assert tight.collision_free, "0.04 m of headroom is a near miss, not a collision"
    assert tight.below_margin(0.18)
    assert tight.deficit(0.18).max() == pytest.approx(0.14)

    hit = ClearanceTrace(s_m=s, clearance=np.full(N, -0.03), overhead=np.full(N, -0.03),
                         min_clearance_m=-0.03, min_overhead_m=-0.03,
                         collision_free=False, goal_error_m=0.05)
    assert not hit.collision_free and hit.below_margin(0.18)

    clean = ClearanceTrace(s_m=s, clearance=np.full(N, 0.25), overhead=np.full(N, 0.25),
                           min_clearance_m=0.25, min_overhead_m=0.25,
                           collision_free=True, goal_error_m=0.05)
    assert clean.collision_free and not clean.below_margin(0.18)
    assert clean.deficit(0.18).max() == 0.0


def test_overhead_is_separated_from_lateral_clearance():
    """The measured finding this encodes: a wall squeeze looks exactly like a clearance
    deficit under a whole-scene minimum, and no amount of ducking fixes it.

    On the real m018 system a 3-beam scene reported 76 mm total clearance against a 180 mm
    target while its overhead clearance was 341 mm -- the robot was hugging a wall. A repair
    driven by the undifferentiated minimum would have answered that by crouching.
    """
    sc = build(BeamParams(1.60, 0.30, 1, 3.0).clamped())   # beam too high and narrow to matter
    p = evaluate(sc, "shortest").plan
    qpos = _walk(sc, 0.9)
    qpos[:, 1] = -0.85                                      # hug the right wall, away from the beam
    tr = clearance_trace(sc, qpos, p.xy)

    assert tr.collision_free, "tight against the wall, but not touching it"
    assert tr.min_clearance_m < 0.18, "the squeeze is genuinely inside the target margin"
    assert tr.min_overhead_m > 0.5, "yet there is nothing overhead"
    assert tr.deficit(0.18).max() == 0.0, "so a duck repair has nothing to act on"
    assert tr.lateral_deficit(0.18).max() > 0.0, "and the tightness is still reported"


# -- the repair operator ----------------------------------------------------------------

def test_zero_deficit_is_exactly_identity(resp):
    q = np.full(N, 0.3)
    out, step = repair(q, np.zeros(N), resp, np.linspace(0, 8, N), 0.9, 1)
    assert np.allclose(out, q)
    assert step.max_delta_q == pytest.approx(0.0, abs=1e-12)
    assert step.q_before_hash == step.q_after_hash


def test_repair_is_local(resp):
    """A deficit in one place must not raise the command at the far end of the route."""
    e = np.zeros(N); e[30:34] = 0.05
    out, _ = repair(np.zeros(N), e, resp, np.linspace(0, 8, N), 0.9, 1)
    assert out[:12].max() == pytest.approx(0.0, abs=1e-9)
    assert out[55:].max() == pytest.approx(0.0, abs=1e-9)
    assert out[30:34].min() > 0.0


def test_repair_is_monotone_in_the_deficit(resp):
    s = np.linspace(0, 8, N)
    e = np.zeros(N); e[30:34] = 1.0
    peaks = [repair(np.full(N, 0.2), e * m, resp, s, 0.9, 1)[0].max() for m in (0.02, 0.05, 0.10)]
    assert peaks[0] < peaks[1] < peaks[2]


def test_repair_anticipates_the_deficit(resp):
    """The body is a first-order lag, so the correction must start before the deficit does."""
    e = np.zeros(N); e[40:44] = 0.06
    out, step = repair(np.zeros(N), e, resp, np.linspace(0, 8, N), 0.9, 1)
    first = int(np.flatnonzero(out > 1e-6)[0])
    assert first < 40, f"correction starts at {first}, deficit starts at 40"
    assert step.extra["lead_samples"] > 0


def test_repair_never_leaves_the_command_range(resp):
    e = np.full(N, 5.0)                                     # absurd, unsatisfiable deficit
    out, _ = repair(np.full(N, 0.9), e, resp, np.linspace(0, 8, N), 0.9, 1)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_slope_floor_bounds_the_correction(resp):
    """Near saturation the fitted gain goes flat; inversion must not demand infinite crouch."""
    q = np.linspace(0, 1, 41)
    assert local_slope(resp, q).min() > 0.0, "secant slope must never be exactly zero"
    e = np.full(N, 0.30)
    out, step = repair(np.full(N, 0.95), e, resp, np.linspace(0, 8, N), 0.9, 1)
    # 0.30 m of deficit divided by the floor is 2.0 in command units -- it must clip, not blow up.
    assert np.isfinite(out).all() and out.max() <= 1.0
    assert step.max_delta_q <= 0.30 / SLOPE_FLOOR + 1e-9


def test_forward_secant_tracks_the_exact_inverse(resp):
    """The local slope is a robust stand-in for g_inv, so check it against the real thing."""
    for q0 in (0.0, 0.15, 0.30, 0.50):
        need = float(resp.g_inv(np.array([resp.g(np.array([q0]))[0] - 0.05]))[0]) - q0
        got = 0.05 / max(float(local_slope(resp, np.array([q0]))[0]), SLOPE_FLOOR)
        assert abs(got - need) < 0.008, f"q={q0}: secant {got:.4f} vs exact {need:.4f}"


# -- the loop's guarantees --------------------------------------------------------------

class _Stub:
    """A generator whose measured clearance improves with commanded dip, by a known law."""

    def __init__(self, gain=0.9, offset=-0.05, cap=1.0):
        self.gain, self.offset, self.cap, self.calls, self.hashes = gain, offset, cap, 0, []

    def __call__(self, scene, p, q, cache, *, iteration, preference, seed=100, speed=0.9,
                 allow_generate=True):
        self.calls += 1
        self.hashes.append(schedule_hash(q))
        return {"qpos": q, "source": "generated", "key": f"k{iteration}", "gen_s": 0.0,
                "schedule_hash": schedule_hash(q)}


def _patch(monkeypatch, stub, clear_of):
    monkeypatch.setattr(loopmod, "generate_from_schedule", stub)

    def fake_trace(scene, qpos, xy, **kw):
        from scene2motion.verify.trace import ClearanceTrace
        c = np.full(N, clear_of(np.asarray(qpos, float)))
        return ClearanceTrace(s_m=np.linspace(0, 8, N), clearance=c, overhead=c,
                              min_clearance_m=float(c.min()), min_overhead_m=float(c.min()),
                              collision_free=bool(c.min() >= 0), goal_error_m=0.05)
    monkeypatch.setattr(loopmod, "clearance_trace", fake_trace)


def test_loop_stops_at_two_repairs(monkeypatch, resp):
    """An unfixable scene must cost 3 generations, never more."""
    stub = _Stub()
    _patch(monkeypatch, stub, lambda q: -0.10)              # nothing ever helps
    r = loopmod.run(build(BeamParams(0.95, 2.25, 3, 3.0).clamped()),
                    evaluate(build(BeamParams(0.95, 2.25, 3, 3.0).clamped()), "shortest").plan,
                    np.full(N, 0.2), resp, ClipCache("/tmp/nope-unused"), max_repairs=MAX_REPAIRS)
    assert stub.calls == MAX_REPAIRS + 1 == 3
    assert len(r.repairs) == MAX_REPAIRS
    assert r.outcome == "rejected"
    assert "select around" in r.reason


def test_loop_accepts_without_repairing_when_the_first_clip_is_clean(monkeypatch, resp):
    stub = _Stub()
    _patch(monkeypatch, stub, lambda q: 0.30)
    sc = build(BeamParams(1.20, 1.45, 1, 3.0).clamped())
    r = loopmod.run(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2), resp,
                    ClipCache("/tmp/nope-unused"))
    assert stub.calls == 1 and r.repairs == [] and r.outcome == "accepted"
    assert r.repaired is False, "a clip that was never repaired must not claim to be"
    assert r.ardy_calls == 1
    d = r.to_dict()
    assert d["ardy_calls"] == 1                         # one candidate-producing call
    assert d["legacy_ardy_calls"] == 2                  # historical unused reference + candidate
    assert d["necessary_adapted_generations"] == 1
    assert d["adapted_generations_executed"] == 1


def test_repaired_flag_requires_the_final_clip_to_come_from_the_repaired_schedule(monkeypatch, resp):
    """The core honesty guard: `repaired` is a statement about which bytes were generated."""
    stub = _Stub()
    # Clean only once the commanded dip exceeds 0.35, so exactly one repair is needed.
    _patch(monkeypatch, stub, lambda q: 0.30 if float(np.max(q)) > 0.35 else -0.02)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    r = loopmod.run(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2), resp,
                    ClipCache("/tmp/nope-unused"))
    assert r.outcome == "accepted" and len(r.repairs) == 1 and r.repaired is True
    # The clip that was accepted is the one generated from the post-repair schedule.
    assert r.final.schedule_hash == r.repairs[-1].q_after_hash
    assert r.final.schedule_hash != r.provenance["initial_schedule_hash"]
    assert stub.hashes == [r.provenance["initial_schedule_hash"], r.final.schedule_hash]

    # And the pre-repair schedule was really generated separately, not reused.
    assert stub.calls == 2 and len(set(stub.hashes)) == 2


def test_provenance_records_both_schedule_hashes(monkeypatch, resp):
    stub = _Stub()
    _patch(monkeypatch, stub, lambda q: 0.30 if float(np.max(q)) > 0.35 else -0.02)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    r = loopmod.run(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2), resp,
                    ClipCache("/tmp/nope-unused"), provenance={"scene": sc.scene_id})
    d = r.to_dict()
    for f in ("initial_schedule_hash", "final_schedule_hash", "seed", "steps", "target_m"):
        assert f in d["provenance"], f
    assert d["ardy_calls"] == 2
    assert d["legacy_ardy_calls"] == 4
    assert d["necessary_adapted_generations"] == 2
    assert d["adapted_generations_executed"] == 2
    assert [a["iteration"] for a in d["attempts"]] == [0, 1]


def test_unverified_when_generation_is_disabled(monkeypatch, resp, tmp_path):
    """A cache miss with generation off must say so, not silently pass."""
    def gpu_load_is_a_bug():
        pytest.fail("cache-only verification must not construct the ARDY runner")
    monkeypatch.setattr(loopmod, "get_runner", gpu_load_is_a_bug)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    r = loopmod.run(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2), resp,
                    ClipCache(tmp_path / "empty-cache"), allow_generate=False)
    assert r.outcome == "unverified" and r.attempts == []


# -- unchanged-proposal independent-resampling control --------------------------------

class _SampleStub:
    def __init__(self, clearances, source="generated"):
        self.clearances = list(clearances)
        self.sources = ([source] * len(self.clearances) if isinstance(source, str)
                        else list(source))
        self.seeds, self.hashes = [], []

    def __call__(self, scene, p, q, cache, *, iteration, preference, seed=100, speed=0.9,
                 allow_generate=True):
        self.seeds.append(seed)
        self.hashes.append(schedule_hash(q))
        return {"qpos": np.full(N, self.clearances[iteration]),
                "source": self.sources[iteration],
                "key": f"sample-{iteration}", "gen_s": 0.0,
                "schedule_hash": schedule_hash(q)}


def _patch_samples(monkeypatch, stub):
    monkeypatch.setattr(loopmod, "generate_from_schedule", stub)

    def fake_trace(scene, qpos, xy, **kw):
        from scene2motion.verify.trace import ClearanceTrace
        c = float(np.asarray(qpos)[0])
        v = np.full(N, c)
        return ClearanceTrace(s_m=np.linspace(0, 8, N), clearance=v, overhead=v,
                              min_clearance_m=c, min_overhead_m=c,
                              collision_free=c >= 0, goal_error_m=0.05)
    monkeypatch.setattr(loopmod, "clearance_trace", fake_trace)


def test_resample_seed_ladders_are_paired_but_do_not_overlap_adjacent_runs():
    a = loopmod.resample_seeds(100, 3)
    b = loopmod.resample_seeds(101, 3)
    assert a == [100, 10_100, 20_100]
    assert b == [101, 10_101, 20_101]
    assert set(a).isdisjoint(b)
    with pytest.raises(ValueError):
        loopmod.resample_seeds(100, 0)


def test_resample_keeps_proposal_fixed_and_selects_an_earlier_better_clip(
        monkeypatch):
    stub = _SampleStub([0.01, 0.08, -0.02])
    _patch_samples(monkeypatch, stub)
    bp = BeamParams(0.95, 2.25, 3, 3.0).clamped()
    sc = build(bp)
    q0 = np.full(N, 0.2)
    r = loopmod.run_resample(sc, evaluate(sc, "shortest").plan, q0,
                             ClipCache("/tmp/nope-unused"), seed=100, max_samples=3)

    assert r.outcome == "accepted_margin"
    assert [a.iteration for a in r.attempts] == [0, 1, 2], "ledger stays chronological"
    assert r.selected_attempt == 1 and r.final.key == "sample-1"
    assert len(set(stub.seeds)) == 3, "samples must use independent generator noise"
    assert len(set(stub.hashes)) == 1 == len({schedule_hash(q0)}), \
        "the proposal must remain byte-identical"
    assert not r.repairs and not r.repaired
    d = r.to_dict()
    assert d["necessary_adapted_generations"] == 3
    assert d["adapted_generations_executed"] == 3
    assert d["ardy_calls"] == d["ardy_calls_executed"] == 3
    assert d["legacy_ardy_calls"] == 6
    assert d["selected_attempt"] == 1

    # The Phase-4 row and all downstream aggregates must describe the selected clip, not
    # the chronologically last (colliding) sample.
    from scene2motion.verify.experiment_repair import _row
    row = _row(sc, bp, "heuristic-resample3", r, needed=0.4,
               s_m=np.linspace(0, 8, N), seed=100)
    assert row["collision_free"] is True
    assert row["min_overhead_m"] == pytest.approx(0.08)
    assert row["clip_key"] == "sample-1" and row["selected_seed"] == 10_100
    assert [x["seed"] for x in row["attempt_ledger"]] == [100, 10_100, 20_100]
    assert row["attempt_ledger"][1]["clip_key"] == row["clip_key"]


def test_resample_uses_the_same_early_accept_rule_as_repair(monkeypatch):
    stub = _SampleStub([-0.02, 0.20, 0.30])
    _patch_samples(monkeypatch, stub)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    r = loopmod.run_resample(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2),
                             ClipCache("/tmp/nope-unused"), seed=100, max_samples=3)
    assert r.outcome == "accepted" and len(r.attempts) == 2
    assert r.selected_attempt == 1 and r.final.seed == 10_100
    assert r.provenance["attempt_seeds_planned"] == [100, 10_100, 20_100]
    assert r.provenance["attempt_seeds_executed"] == [100, 10_100]


def test_resample_logical_cost_is_independent_of_cache_state(monkeypatch):
    stub = _SampleStub([0.01, 0.02], source="cache")
    _patch_samples(monkeypatch, stub)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    r = loopmod.run_resample(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2),
                             ClipCache("/tmp/nope-unused"), max_samples=2)
    d = r.to_dict()
    assert d["necessary_adapted_generations"] == 2 and d["ardy_calls"] == 2
    assert d["legacy_ardy_calls"] == 4
    assert d["adapted_generations_executed"] == 0 and d["ardy_calls_executed"] == 0
    assert d["cache_hits"] == 2


def test_cache_only_partial_feedback_is_unverified_but_keeps_last_verified_clip(
        monkeypatch, resp):
    calls = 0

    def partial(scene, p, q, cache, *, iteration, preference, seed=100, speed=0.9,
                allow_generate=True):
        nonlocal calls
        calls += 1
        if iteration:
            return {"qpos": None, "source": "miss", "key": "missing", "gen_s": 0.0,
                    "schedule_hash": schedule_hash(q)}
        return {"qpos": q, "source": "cache", "key": "cached-0", "gen_s": 0.0,
                "schedule_hash": schedule_hash(q)}

    monkeypatch.setattr(loopmod, "generate_from_schedule", partial)

    def trace(scene, qpos, xy, **kw):
        from scene2motion.verify.trace import ClearanceTrace
        v = np.full(N, 0.10)
        return ClearanceTrace(s_m=np.linspace(0, 8, N), clearance=v, overhead=v,
                              min_clearance_m=0.10, min_overhead_m=0.10,
                              collision_free=True, goal_error_m=0.05)
    monkeypatch.setattr(loopmod, "clearance_trace", trace)

    bp = BeamParams(0.95, 2.25, 3, 3.0).clamped()
    sc, q0 = build(bp), np.full(N, 0.2)
    r = loopmod.run(sc, evaluate(sc, "shortest").plan, q0, resp,
                    ClipCache("/tmp/nope-unused"), max_repairs=1,
                    allow_generate=False)
    assert calls == 2 and r.outcome == "unverified" and len(r.attempts) == 1
    assert r.final.key == "cached-0" and r.provenance["final_schedule_hash"] == schedule_hash(q0)
    from scene2motion.verify.experiment_repair import _row
    row = _row(sc, bp, "heuristic+1", r, needed=0.4,
               s_m=np.linspace(0, 8, N), seed=100)
    assert row["outcome"] == "unverified" and row["clip_key"] == "cached-0"


def test_generation_accounting_invariants_with_mixed_cache(monkeypatch):
    stub = _SampleStub([0.01, 0.02], source=["cache", "generated"])
    _patch_samples(monkeypatch, stub)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    r = loopmod.run_resample(sc, evaluate(sc, "shortest").plan, np.full(N, 0.2),
                             ClipCache("/tmp/nope-unused"), max_samples=2)
    d = r.to_dict()
    assert d["n_attempts"] == d["necessary_adapted_generations"] == 2
    assert d["ardy_calls"] == d["necessary_adapted_generations"] == 2
    assert d["legacy_ardy_calls"] == 4
    assert d["cache_hits"] == 1 and d["adapted_generations_executed"] == 1
    assert d["ardy_calls_executed"] == d["adapted_generations_executed"] == 1


def test_repair_and_resample_change_only_the_declared_axis(monkeypatch, resp):
    calls = []

    def generate(scene, p, q, cache, *, iteration, preference, seed=100, speed=0.9,
                 allow_generate=True):
        calls.append((schedule_hash(q), seed))
        return {"qpos": q, "source": "generated", "key": f"k-{iteration}-{seed}",
                "gen_s": 0.0, "schedule_hash": schedule_hash(q)}

    def trace(scene, qpos, xy, **kw):
        from scene2motion.verify.trace import ClearanceTrace
        v = np.full(N, 0.10)  # collision-free, always 8 cm short of target
        return ClearanceTrace(s_m=np.linspace(0, 8, N), clearance=v, overhead=v,
                              min_clearance_m=0.10, min_overhead_m=0.10,
                              collision_free=True, goal_error_m=0.05)

    monkeypatch.setattr(loopmod, "generate_from_schedule", generate)
    monkeypatch.setattr(loopmod, "clearance_trace", trace)
    sc = build(BeamParams(0.95, 2.25, 3, 3.0).clamped())
    plan, q0 = evaluate(sc, "shortest").plan, np.full(N, 0.2)

    loopmod.run(sc, plan, q0, resp, ClipCache("/tmp/nope-unused"), seed=100,
                max_repairs=1)
    repair_calls = list(calls)
    calls.clear()
    loopmod.run_resample(sc, plan, q0, ClipCache("/tmp/nope-unused"), seed=100,
                         max_samples=2)
    sample_calls = list(calls)

    assert repair_calls[0] == (schedule_hash(q0), 100)
    assert repair_calls[1][0] != repair_calls[0][0]
    assert [seed for _, seed in repair_calls] == [100, 100]
    assert [h for h, _ in sample_calls] == [schedule_hash(q0)] * 2
    assert [seed for _, seed in sample_calls] == [100, 10_100]


def test_architecture_matrix_is_complete_and_preserves_legacy_names():
    from scene2motion.verify.experiment_architecture import build_method_specs
    specs = build_method_specs()
    expected = []
    for proposer, key in (("heuristic", "heuristic"), ("qp", "optimizer"),
                          ("tcn", "optimized")):
        expected.extend([
            (proposer, proposer, "none", 1, key, 0),
            (f"{proposer}+1", proposer, "repair", 2, key, 1),
            (f"{proposer}+2", proposer, "repair", 3, key, 2),
            (f"{proposer}-resample2", proposer, "resample_select", 2, key, 0),
            (f"{proposer}-resample3", proposer, "resample_select", 3, key, 0),
        ])
    actual = [(s.name, s.proposer, s.feedback, s.max_adapted_generations,
               s.schedule_key, s.max_repairs) for s in specs]
    assert actual == expected


def test_architecture_dispatches_exact_feedback_budget(monkeypatch):
    from scene2motion.verify import experiment_architecture as arch
    calls = []

    def repair_spy(*args, **kwargs):
        calls.append(("repair", kwargs))
        return object()

    def sample_spy(*args, **kwargs):
        calls.append(("sample", kwargs))
        return object()

    monkeypatch.setattr(arch, "run", repair_spy)
    monkeypatch.setattr(arch, "run_resample", sample_spy)
    specs = {s.name: s for s in arch.build_method_specs()}
    common = dict(scene=None, plan=None, q0=np.zeros(N), resp=None, cache=None,
                  preference="shortest", seed=100, seed_stride=777, provenance={},
                  allow_generate=False)
    arch._run_method(specs["heuristic"], **common)
    arch._run_method(specs["qp+2"], **common)
    arch._run_method(specs["tcn-resample3"], **common)
    assert [(kind, kw.get("max_repairs"), kw.get("max_samples"), kw.get("seed_stride"),
             kw.get("allow_generate"))
            for kind, kw in calls] == [
                ("repair", 0, None, None, False), ("repair", 2, None, None, False),
                ("sample", None, 3, 777, False)]


def test_aggregate_separates_logical_legacy_and_executed_costs():
    from scene2motion.verify.experiment_repair import _aggregate
    common = {"collision_free": True, "meets_target": False, "goal_reached": True,
              "peak_dip_m": 0.2, "excess_crouch_m": 0.1, "duck_integral_m2": 1.0,
              "min_overhead_m": 0.1, "repaired": False, "outcome": "accepted_margin"}
    rows = [{**common, "ardy_calls": 2, "legacy_ardy_calls": 4,
             "ardy_calls_executed": 1, "n_attempts": 2,
             "necessary_adapted_generations": 2, "adapted_generations_executed": 1},
            {**common, "ardy_calls": 1, "legacy_ardy_calls": 2,
             "ardy_calls_executed": 0, "n_attempts": 1,
             "necessary_adapted_generations": 1, "adapted_generations_executed": 0}]
    a = _aggregate(rows)
    assert a["mean_ardy_calls"] == 1.5
    assert a["mean_legacy_ardy_calls"] == 3.0
    assert a["mean_necessary_adapted_generations"] == 1.5
    assert a["mean_adapted_generations_executed"] == 0.5

    # A committed pre-extension Phase-4 row still aggregates under the additive schema.
    old = _aggregate([{**common, "ardy_calls": 4, "ardy_calls_executed": 2,
                       "n_attempts": 2}])
    assert old["mean_necessary_adapted_generations"] == 2.0
    assert old["mean_adapted_generations_executed"] == 1.0


def test_frozen_provenance_hashes_checkpoint_metadata_and_response(tmp_path):
    import hashlib
    import json
    from scene2motion.verify.experiment_architecture import frozen_provenance

    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "tcn.pt").write_bytes(b"fixed checkpoint")
    (ckpt / "tcn.json").write_text(json.dumps(
        {"margin_m": 0.18, "dataset": "d", "dataset_hash": "abc", "seed": 7}))
    response_path = tmp_path / "response.json"
    response_path.write_text("{}")
    p = frozen_provenance(ckpt, response_path=response_path, repo=tmp_path)
    assert p["tcn"]["checkpoint_sha256"] == hashlib.sha256(b"fixed checkpoint").hexdigest()
    assert p["tcn"]["dataset_hash"] == "abc" and p["tcn"]["training_seed"] == 7
    assert p["duck_response"]["sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_dip_max_is_the_ceiling_the_loop_reports(resp):
    """Saturation is a real outcome, and must be visible as a peak dip at DIP_MAX."""
    out, step = repair(np.full(N, 0.9), np.full(N, 1.0), resp, np.linspace(0, 8, N), 0.9, 1)
    assert float(out.max() * DIP_MAX) == pytest.approx(DIP_MAX)


def test_beam_builder_supports_the_phase4_range():
    for n in (1, 2, 3, 4, 5, 6):
        assert BeamParams(0.95, 2.25, n, 3.0).clamped().n_beams == n
    assert BeamParams(0.95, 2.25, 9, 3.0).clamped().n_beams == 6
    sc = build(BeamParams(0.95, 2.25, 5, 3.0).clamped())
    assert len(sc.meta["beam_xs"]) == 5
    assert sc.goal[0] > sc.meta["beam_xs"][-1] + 3.0, "goal must stay clear of the last beam"


def test_g1body_reports_the_overhead_channel():
    sc = build(BeamParams(0.95, 2.25, 1, 3.0).clamped())
    rep = G1Body(sc).trajectory_report(_walk(sc, 0.85))
    assert len(rep["per_frame_overhead_clearance"]) == len(rep["per_frame_min_clearance"])
    o = np.asarray(rep["per_frame_overhead_clearance"])
    t = np.asarray(rep["per_frame_min_clearance"])
    assert (o >= t - 1e-9).all(), "overhead is a subset minimum, so never below the total"


# -- route / body cost decomposition ----------------------------------------------------

def test_body_cost_terms_separate_effort_from_shape():
    from scene2motion.verify.cost import body_cost
    s = np.linspace(0, 8, N)
    flat = np.full(N, 0.4)
    e, r, sm = body_cost(flat, s)
    assert e == pytest.approx(0.16 * 8, rel=1e-6), "a held command is pure effort"
    assert r == pytest.approx(0.0, abs=1e-9) and sm == pytest.approx(0.0, abs=1e-9)

    wiggly = 0.4 + 0.05 * np.sin(np.linspace(0, 20 * np.pi, N))
    e2, r2, sm2 = body_cost(wiggly, s)
    assert r2 > r and sm2 > sm, "the same mean effort, spent worse, must cost more"


def test_body_cost_is_per_distance_not_per_sample():
    """The same manoeuvre must not get cheaper by sitting on a longer route."""
    from scene2motion.verify.cost import body_cost
    u = np.zeros(N); u[28:36] = 0.4
    short = body_cost(u, np.linspace(0, 8, N))
    long_ = body_cost(u, np.linspace(0, 8, N) * 1.0)
    assert short == long_
    # Stretching the SAME shape over twice the distance makes it genuinely gentler.
    stretched = body_cost(u, np.linspace(0, 16, N))
    assert stretched[1] < short[1] and stretched[2] < short[2]


def test_preferences_are_weight_configurations_not_route_rules():
    """'Stay upright' must be a price on body effort, not a ban on ducking."""
    from scene2motion.verify.cost import WEIGHTS, evaluate
    s = np.linspace(0, 8, N)
    duck = np.zeros(N); duck[28:36] = 0.5
    assert WEIGHTS["upright"]["w_B"] > WEIGHTS["shortest"]["w_B"]
    assert WEIGHTS["clearance"]["w_C"] > WEIGHTS["shortest"]["w_C"]
    up = evaluate(duck, s, 0.2, "upright")
    sh = evaluate(duck, s, 0.2, "shortest")
    # Same route, same schedule: upright charges more for the crouch, shortest for the metres.
    assert up.total != sh.total
    assert up.to_dict()["w_body_term"] > sh.to_dict()["w_body_term"]
    assert sh.to_dict()["w_route_term"] > up.to_dict()["w_route_term"]


def test_cost_total_is_the_sum_of_its_reported_terms():
    from scene2motion.verify.cost import evaluate
    d = evaluate(np.linspace(0, 0.5, N), np.linspace(0, 8, N), 0.19, "balanced").to_dict()
    assert (d["w_route_term"] + d["w_body_term"] + d["w_clear_term"]
            == pytest.approx(d["total"], abs=1e-6))


# -- candidate routes and coupled selection ---------------------------------------------

def test_candidates_are_distinct_and_deterministic():
    from scene2motion.verify.routes import candidates
    sc = build(BeamParams(0.95, 1.45, 3, 3.0).clamped())
    a1 = candidates(sc, k=8)
    a2 = candidates(sc, k=8)
    assert len(a1) >= 2, "a scene with a bypass must offer more than one traversal"
    assert [r.label for r in a1] == [r.label for r in a2], "must be reproducible"
    paths = {np.round(r.xy, 3).tobytes() for r in a1}
    assert len(paths) == len(a1), "duplicates must be dropped, not counted"
    assert all(r.plan.feasible for r in a1)


def test_slot_constraint_actually_moves_the_route():
    """A narrow forbidden band left A* free to route around it; a slot must not."""
    from scene2motion.verify.routes import SLOT_HALF, candidates
    sc = build(BeamParams(0.95, 1.45, 3, 3.0).clamped())
    rs = candidates(sc, k=8)
    slotted = [r for r in rs if r.lateral_y is not None]
    assert slotted, "slot candidates must survive"
    for r in slotted:
        mid = r.xy[len(r.xy) // 2, 1]
        assert abs(mid - r.lateral_y) <= SLOT_HALF + 0.2, \
            f"{r.label} crosses at y={mid:.2f}, outside its slot at {r.lateral_y:+.2f}"


def test_route_length_alone_disagrees_with_the_body_aware_choice(resp):
    """The Phase 4B premise: a shorter route can be much more expensive to hold."""
    from scene2motion.verify.routes import candidates
    from scene2motion.verify.select import load_tcn, pick, regret, score_all
    model = load_tcn()
    if model is None:
        pytest.skip("no m018 checkpoint")
    sc = build(BeamParams(0.95, 1.45, 3, 3.0).clamped())
    S = score_all(sc, candidates(sc, k=8), resp, model, "balanced")
    i_len, i_oracle = pick(S, "heuristic"), pick(S, "oracle_qp")
    assert i_len != i_oracle, "this scene is chosen because the two rules differ on it"
    assert S[i_len].s_m[-1] < S[i_oracle].s_m[-1], "the length rule picked the shorter route"
    assert S[i_len].u_qp.max() > S[i_oracle].u_qp.max(), "and it costs a deeper crouch"
    assert regret(S, i_oracle) == pytest.approx(0.0) and regret(S, i_len) > 0


def test_regret_is_zero_exactly_for_the_oracles_own_pick(resp):
    from scene2motion.verify.routes import candidates
    from scene2motion.verify.select import load_tcn, pick, ranking, regret, score_all
    model = load_tcn()
    if model is None:
        pytest.skip("no m018 checkpoint")
    sc = build(BeamParams(1.05, 2.25, 4, 2.5).clamped())
    S = score_all(sc, candidates(sc, k=8), resp, model, "balanced")
    assert regret(S, pick(S, "oracle_qp")) == pytest.approx(0.0, abs=1e-9)
    assert all(regret(S, i) >= -1e-9 for i in range(len(S))), "regret is never negative"
    order = ranking(S, "oracle_qp")
    assert order[0] == pick(S, "oracle_qp"), "the ranking must head with the pick"
    assert len(set(order)) == len(S), "a fallback order must cover every candidate once"


def test_batched_tcn_matches_one_route_at_a_time(resp):
    """Batching is only a latency claim if it changes nothing about the answer."""
    from scene2motion.verify.routes import candidates
    from scene2motion.verify.select import load_tcn, tcn_schedules
    model = load_tcn()
    if model is None:
        pytest.skip("no m018 checkpoint")
    sc = build(BeamParams(1.05, 2.25, 4, 2.5).clamped())
    rs = candidates(sc, k=8)
    batched = tcn_schedules(sc, rs, resp, model)
    singly = [tcn_schedules(sc, [r], resp, model)[0] for r in rs]
    for b, s in zip(batched, singly):
        assert np.allclose(b, s, atol=1e-5)
