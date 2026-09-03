"""CPU tests for the paired scene-clustered repair-vs-resampling analyser.

The differences this produces (+21.9, +36.1, -8.0 pp) are quoted in the paper, so the truth
coercion, the pairing and the bootstrap are pinned here on synthetic rows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import analyze_repair_paired_bootstrap as rb


# ---------------------------------------------------------------- truth coercion

@pytest.mark.parametrize("value", [True, "True", "true"])
def test_truthy_accepts_bool_and_its_string_forms(value):
    assert rb._truthy(value) == 1.0


@pytest.mark.parametrize("value", [False, "False", "false", None, 0, ""])
def test_truthy_rejects_everything_else(value):
    """Guards the bug where JSON booleans were compared against the string 'True'."""
    assert rb._truthy(value) == 0.0


# ---------------------------------------------------------------- pairing

def _rows(per_method):
    """per_method: {method: {scene: [outcome per seed]}} -> flat rows."""
    out = []
    for method, scenes in per_method.items():
        for scene, seeds in scenes.items():
            for seed, value in enumerate(seeds):
                out.append({"method": method, "scene_id": scene, "seed": seed,
                            "collision_free": value})
    return out


def test_matrix_is_scene_by_seed_and_ordered_by_seed():
    rows = _rows({"a": {"s1": [True, False], "s0": [False, True]}})
    matrix = rb.scene_seed_matrix(rows, "a", "collision_free", ["s0", "s1"])
    assert matrix.shape == (2, 2)
    assert matrix.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_matrix_selects_only_the_named_method():
    rows = _rows({"a": {"s0": [True, True]}, "b": {"s0": [False, False]}})
    assert rb.scene_seed_matrix(rows, "a", "collision_free", ["s0"]).mean() == 1.0
    assert rb.scene_seed_matrix(rows, "b", "collision_free", ["s0"]).mean() == 0.0


# ---------------------------------------------------------------- end to end

def _paired_rows(a_success, b_success, n_scenes=8, n_seeds=4):
    """Build `x+2` and `x-resample3` arms with the given per-scene success counts."""
    rows = []
    for scene in range(n_scenes):
        for seed in range(n_seeds):
            rows.append({"method": "x+2", "scene_id": f"s{scene}", "seed": seed,
                         "collision_free": seed < a_success, "meets_target": seed < a_success})
            rows.append({"method": "x-resample3", "scene_id": f"s{scene}", "seed": seed,
                         "collision_free": seed < b_success, "meets_target": seed < b_success})
    return rows


def _run(tmp_path, monkeypatch, rows, comparisons):
    source = tmp_path / "experiment.json"
    source.write_text(json.dumps({"rows": rows}))
    out = tmp_path / "summary.json"
    monkeypatch.setattr(rb, "SOURCE", source)
    monkeypatch.setattr(rb, "OUT", out)
    monkeypatch.setattr(rb, "ROOT", tmp_path)
    monkeypatch.setattr(rb, "COMPARISONS", comparisons)
    monkeypatch.setattr(rb, "N_BOOT", 2000)
    rb.main()
    return json.loads(out.read_text())


def test_a_uniform_advantage_gives_the_exact_difference_and_no_spread(tmp_path, monkeypatch):
    """Every scene identical: the point estimate is exact and the interval collapses onto it."""
    rows = _paired_rows(a_success=3, b_success=1)          # 75 % vs 25 %
    summary = _run(tmp_path, monkeypatch, rows,
                   [("x", "collision_free", "uniform advantage")])
    result = summary["results"][0]
    assert result["rate_a"] == pytest.approx(0.75)
    assert result["rate_b"] == pytest.approx(0.25)
    assert result["paired_difference_pp"] == pytest.approx(50.0)
    lo, hi = result["bootstrap_95_pp"]
    assert lo == pytest.approx(50.0) and hi == pytest.approx(50.0)


def test_a_uniform_disadvantage_is_reported_as_a_negative_difference(tmp_path, monkeypatch):
    """The losing comparison must survive with its sign, not be reported as an improvement."""
    rows = _paired_rows(a_success=1, b_success=3)
    summary = _run(tmp_path, monkeypatch, rows,
                   [("x", "meets_target", "uniform disadvantage")])
    result = summary["results"][0]
    assert result["paired_difference_pp"] == pytest.approx(-50.0)
    assert result["discordant_a_only"] == 0
    assert result["discordant_b_only"] == 2 * 8      # two seeds per scene, eight scenes


def test_discordant_counts_are_per_scene_seed_pair(tmp_path, monkeypatch):
    rows = _paired_rows(a_success=3, b_success=1, n_scenes=8, n_seeds=4)
    summary = _run(tmp_path, monkeypatch, rows,
                   [("x", "collision_free", "discordance")])
    result = summary["results"][0]
    assert result["discordant_a_only"] == 2 * 8
    assert result["discordant_b_only"] == 0


def test_bootstrap_interval_brackets_the_point_estimate_when_scenes_differ(tmp_path, monkeypatch):
    rows = []
    for scene in range(12):
        advantage = scene % 3          # heterogeneous across scenes
        for seed in range(4):
            rows.append({"method": "x+2", "scene_id": f"s{scene}", "seed": seed,
                         "collision_free": seed < advantage})
            rows.append({"method": "x-resample3", "scene_id": f"s{scene}", "seed": seed,
                         "collision_free": False})
    summary = _run(tmp_path, monkeypatch, rows, [("x", "collision_free", "heterogeneous")])
    result = summary["results"][0]
    lo, hi = result["bootstrap_95_pp"]
    assert lo < result["paired_difference_pp"] < hi
    assert lo > 0.0                                  # every scene weakly favours the correction


def test_summary_records_provenance_and_marks_itself_descriptive(tmp_path, monkeypatch):
    rows = _paired_rows(a_success=2, b_success=2, n_scenes=4)
    summary = _run(tmp_path, monkeypatch, rows, [("x", "collision_free", "no difference")])
    assert summary["descriptive_only"] is True
    assert summary["n_scenes"] == 4
    assert summary["n_boot"] == 2000
    assert len(summary["source"]["sha256"]) == 64
    assert summary["results"][0]["paired_difference_pp"] == pytest.approx(0.0)


def test_bootstrap_is_deterministic_for_a_fixed_seed(tmp_path, monkeypatch):
    rows = []
    for scene in range(10):
        for seed in range(4):
            rows.append({"method": "x+2", "scene_id": f"s{scene}", "seed": seed,
                         "collision_free": (scene + seed) % 2 == 0})
            rows.append({"method": "x-resample3", "scene_id": f"s{scene}", "seed": seed,
                         "collision_free": seed == 0})
    first = _run(tmp_path / "a", monkeypatch, rows, [("x", "collision_free", "determinism")])
    second = _run(tmp_path / "b", monkeypatch, rows, [("x", "collision_free", "determinism")])
    assert (first["results"][0]["bootstrap_95_pp"]
            == second["results"][0]["bootstrap_95_pp"])


@pytest.fixture(autouse=True)
def _make_tmp_subdirs(tmp_path):
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
    return tmp_path
