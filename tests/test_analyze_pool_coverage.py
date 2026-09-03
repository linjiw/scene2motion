"""CPU tests for the pool-coverage / selection decomposition analyser.

The numbers this analyser produces are quoted in the paper (12 clear, 11 complete, 0 both), so
the coverage arithmetic and the disjointness logic are pinned here on synthetic rows.
"""

from __future__ import annotations

import json
from math import comb

import pytest

from experiments import analyze_pool_coverage as pc


# ---------------------------------------------------------------- coverage arithmetic

def test_coverage_at_budget_matches_hypergeometric_definition():
    n, m, k = 64, 12, 5
    assert pc.coverage_at_budget(n, m, k) == pytest.approx(1.0 - comb(n - m, k) / comb(n, k))


def test_coverage_is_zero_without_successes_and_one_when_all_succeed():
    assert pc.coverage_at_budget(64, 0, 64) == 0.0
    assert pc.coverage_at_budget(64, 12, 0) == 0.0
    assert pc.coverage_at_budget(10, 10, 1) == pytest.approx(1.0)


def test_coverage_is_certain_once_the_budget_exceeds_the_failures():
    # 64 candidates, 12 successes: any 53 of them must include a success.
    assert pc.coverage_at_budget(64, 12, 53) == pytest.approx(1.0)


def test_coverage_is_monotone_in_the_budget():
    values = [pc.coverage_at_budget(64, 12, k) for k in range(1, 65)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_budget_exceeding_the_pool_is_rejected():
    with pytest.raises(ValueError):
        pc.coverage_at_budget(8, 2, 9)


# ---------------------------------------------------------------- the two N90 conventions

def test_subsampling_and_fresh_draw_conventions_are_both_reported_and_differ():
    """The ledger's N90 = 12 at 5 cm is the fresh-draw convention; sub-sampling gives 11."""
    assert pc.smallest_fresh_draws_for(64, 12, 0.9) == 12
    assert pc.smallest_budget_for(64, 12, 0.9) == 11
    # 8 cm: 11 of 64 clear.
    assert pc.smallest_fresh_draws_for(64, 11, 0.9) == 13
    assert pc.smallest_budget_for(64, 11, 0.9) == 12


def test_sampling_without_replacement_is_never_less_efficient():
    for m in (1, 2, 6, 12, 30):
        sub = pc.smallest_budget_for(64, m, 0.9)
        fresh = pc.smallest_fresh_draws_for(64, m, 0.9)
        assert sub is not None and fresh is not None
        assert sub <= fresh


def test_no_budget_reaches_coverage_without_a_single_success():
    assert pc.smallest_budget_for(64, 0, 0.9) is None
    assert pc.smallest_fresh_draws_for(64, 0, 0.9) is None


# ---------------------------------------------------------------- the decomposition

def _rows(cases, obstacle_x_m=1.2, label="staged"):
    """cases: (seed, reference_clears_5cm, terminated, passed, in_corridor, finished)."""
    heights = pc.HEIGHTS
    reference, achieved = [], []
    for seed, ref_clear, terminated, passed, corridor, finished in cases:
        reference.append({
            "seed": seed, "obstacle_label": label, "obstacle_x_m": obstacle_x_m,
            "exact_clears": {h: (h == "0.05" and ref_clear) for h in heights},
            "achieved_replay_clear_after_passing": {h: False for h in heights},
            "tracker_terminated": False, "passed_obstacle": True,
            "passed_within_lateral_corridor": True, "finished_beyond_obstacle": True,
        })
        achieved.append({
            "seed": seed, "obstacle_label": label, "obstacle_x_m": obstacle_x_m,
            "exact_clears": {h: False for h in heights},
            # the guarded endpoint: only credited when it passed, stayed in the corridor,
            # finished beyond and was not cut off
            "achieved_replay_clear_after_passing": {
                h: bool(passed and corridor and finished and not terminated and h == "0.05"
                        and ref_clear) for h in heights},
            "tracker_terminated": terminated, "passed_obstacle": passed,
            "passed_within_lateral_corridor": corridor, "finished_beyond_obstacle": finished,
        })
    return reference, achieved


def test_disjoint_sets_give_zero_joint_coverage_at_every_budget():
    """The shape of the real result: some clear, some complete, none do both."""
    cases = (
        [(4400 + i, True, True, False, False, False) for i in range(12)]      # clear, cut off
        + [(4500 + i, False, False, True, True, True) for i in range(11)]     # completes, no clear
        + [(4600 + i, False, True, False, False, False) for i in range(41)]   # neither
    )
    reference, achieved = _rows(cases)
    entry = pc.analyse("staged", reference, achieved)
    five = entry["by_height"]["0.05"]

    assert entry["n_candidates"] == 64
    assert five["n_reference_clears"] == 12
    assert entry["n_completes_tracking"] == 11
    assert five["n_reference_clears_and_completes_tracking"] == 0
    assert five["n_traversal_endpoint"] == 0
    # no budget can find what the pool does not contain
    assert five["coverage_curve_joint"]["coverage_at_full_pool"] == 0.0
    assert five["coverage_curve_joint"]["n90_budget_subsampling_this_pool"] is None
    # but reference-level coverage does respond to sampling
    assert five["coverage_curve_reference"]["n90_budget_fresh_draws"] == 12


def test_overlap_is_counted_when_a_candidate_does_both():
    cases = [(4400, True, False, True, True, True)] + [
        (4401 + i, False, True, False, False, False) for i in range(7)]
    reference, achieved = _rows(cases)
    five = pc.analyse("staged", reference, achieved)["by_height"]["0.05"]
    assert five["n_reference_clears_and_completes_tracking"] == 1
    assert five["n_traversal_endpoint"] == 1


def test_passing_the_corridor_is_required_for_the_traversal_endpoint():
    """A candidate that clears and is not cut off but never arrives is not a traversal."""
    cases = [(4400, True, False, False, False, False)]
    reference, achieved = _rows(cases)
    entry = pc.analyse("staged", reference, achieved)
    assert entry["n_completes_tracking"] == 1
    assert entry["by_height"]["0.05"]["n_reference_clears_and_completes_tracking"] == 1
    assert entry["by_height"]["0.05"]["n_traversal_endpoint"] == 0
    assert entry["n_never_reached_obstacle"] == 1


def test_passage_counts_require_corridor_and_finish_together():
    cases = [
        (4400, False, False, True, True, True),    # full passage
        (4401, False, False, True, False, True),   # outside the corridor
        (4402, False, False, True, True, False),   # did not finish beyond
    ]
    reference, achieved = _rows(cases)
    entry = pc.analyse("staged", reference, achieved)
    assert entry["n_passed_corridor_and_finished_beyond"] == 1
    assert entry["n_never_reached_obstacle"] == 2


def test_summary_is_written_with_source_hashes(tmp_path, monkeypatch):
    cases = [(4400, True, True, False, False, False), (4401, False, False, True, True, True)]
    reference, achieved = _rows(cases)
    ref_path, ach_path = tmp_path / "reference_rows.jsonl", tmp_path / "achieved_rows.jsonl"
    ref_path.write_text("".join(json.dumps(r) + "\n" for r in reference))
    ach_path.write_text("".join(json.dumps(r) + "\n" for r in achieved))
    out = tmp_path / "summary.json"
    monkeypatch.setattr(pc, "REFERENCE_ROWS", ref_path)
    monkeypatch.setattr(pc, "ACHIEVED_ROWS", ach_path)
    monkeypatch.setattr(pc, "OUT", out)
    monkeypatch.setattr(pc, "ROOT", tmp_path)

    pc.main()

    summary = json.loads(out.read_text())
    assert summary["descriptive_only"] is True
    assert len(summary["sources"]["reference_rows"]["sha256"]) == 64
    assert len(summary["sources"]["achieved_rows"]["sha256"]) == 64
    assert summary["results"][0]["by_height"]["0.05"][
        "n_reference_clears_and_completes_tracking"] == 0
