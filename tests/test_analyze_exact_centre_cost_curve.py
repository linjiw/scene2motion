"""CPU tests for the exact obstacle-centred cost-curve analyser (A0c)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import analyze_exact_centre_cost_curve as cc
from experiments import analyze_exp021_exact_addressability as e21


def _fake_archive(tmp_path, lifts):
    """Each clip encodes (lift height, lift position) in qpos[0, :2]."""
    archive = tmp_path / "exp021"
    archive.mkdir()
    seeds = list(range(4400, 4400 + len(lifts)))
    arrays = {
        f"s{seed}": np.asarray([[height, x_m], [0.0, 0.0]], dtype=np.float32)
        for seed, (height, x_m) in zip(seeds, lifts)
    }
    np.savez(archive / "qpos.npz", **arrays)
    receipt = {
        "schema": "exp021-elicited-lift-distribution-v1",
        "experiment": "exp021_elicited_lift_distribution",
        "status": "complete", "complete": True, "blocked": False,
        "actual_ardy_samples": len(arrays),
        "design": {"pool_seeds": seeds},
        "summary": {"n_clips": len(arrays)},
        "evidence_anchors": {"qpos": {"path": "qpos.npz", "n_arrays": len(arrays),
                                      "content_sha256": e21.array_content_sha256(arrays)}},
    }
    (archive / "receipt.json").write_text(json.dumps(receipt))
    return archive, seeds, arrays


class _WindowProbe:
    """Clears when the encoded lift is within 0.15 m of the centre and high enough."""

    def __init__(self, x_m, depth_m):
        self.x, self.depth = float(x_m), float(depth_m)

    def clears(self, qpos, height):
        return bool(abs(float(qpos[0, 1]) - self.x) <= 0.15 + 1e-9
                    and float(qpos[0, 0]) >= float(height))


def _fake_exp022(tmp_path, seeds, arrays, centres, heights, *, corrupt_seed=None):
    d = tmp_path / "exp022"
    d.mkdir()
    rows = []
    retention = {}
    for label, x in centres.items():
        probe = _WindowProbe(x, 0.20)
        ref_clear = {str(h): 0 for h in heights}
        for seed in seeds:
            clears = {str(h): probe.clears(arrays[f"s{seed}"], h) for h in heights}
            if corrupt_seed == seed:
                clears = {k: (not v) for k, v in clears.items()}
            for h in heights:
                ref_clear[str(h)] += int(clears[str(h)])
            rows.append({"obstacle_label": label, "obstacle_x_m": x, "seed": seed,
                         "exact_clears": clears})
        retention[label] = {str(h): {"reference_clear": ref_clear[str(h)],
                                     "achieved_guarded_clear": 0, "n_paired": len(seeds)}
                            for h in heights}
    (d / "reference_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "summary.json").write_text(json.dumps({
        "status": "complete", "interpretation_guard": "no box in Isaac",
        "paired_reference_to_achieved_retention": retention}))
    (d / "receipt.json").write_text("{}")
    return d


def test_summarize_curve_rates_intervals_and_budgets():
    hits = np.zeros((2, 1, 8), dtype=bool)
    hits[1, 0, :2] = True
    rows = cc.summarize_curve(hits, [1.0, 1.2], [0.05], best_of_n=(1, 8))
    assert rows[0]["exact_hit_rate"] == 0.0 and rows[0]["independent_plugin_n90"] is None
    assert rows[1]["successes"] == 2 and rows[1]["exact_hit_rate"] == pytest.approx(0.25)
    low, high = rows[1]["wilson95"]
    assert low < 0.25 < high
    assert rows[1]["independent_plugin_best_of_n"]["N=8"] == pytest.approx(1 - 0.75 ** 8)
    assert rows[1]["nominal_arrival_frame"] == pytest.approx(1.2 / 0.9045 * 25)
    with pytest.raises(ValueError, match="centre, height, clip"):
        cc.summarize_curve(np.zeros((2, 2, 3), dtype=bool), [1.0, 1.2], [0.05])


def test_curve_extrema_reports_ties_as_post_hoc_selection():
    hits = np.zeros((3, 1, 4), dtype=bool)
    hits[0, 0, :2] = True
    hits[2, 0, 1:3] = True
    rows = cc.summarize_curve(hits, [1.0, 1.1, 1.2], [0.05])
    extrema = cc.curve_extrema(rows, [0.05])["h=0.05"]
    assert extrema["max_successes"] == 2 and extrema["tied_centres_m"] == [1.0, 1.2]
    assert "post hoc" in extrema["label"]


def test_run_end_to_end_binds_overlay_and_reproduces_exp022a_hits(tmp_path):
    lifts = [(0.06, 1.2), (0.10, 1.2), (0.02, 1.2), (0.09, 3.6), (0.0, 0.0), (0.30, 2.0)]
    archive, seeds, arrays = _fake_archive(tmp_path, lifts)
    heights = (0.03, 0.05, 0.08)
    grid = [1.0, 1.2, 2.0, 3.6]
    exp022 = _fake_exp022(tmp_path, seeds, arrays, {"staged": 1.2, "control": 3.6}, heights)
    out = tmp_path / "out"
    result = cc.run(archive=archive, exp022_dir=exp022, out=out, centres_m=grid,
                    heights_m=heights, probe_factory=_WindowProbe, locked_manifest=None)
    assert result["status"] == "complete" and result["post_hoc"] is True
    assert result["n_exact_clears_calls"] == len(grid) * len(heights) * len(seeds)
    staged = result["at_exp022a_centres"]["staged"]
    assert staged["reference_exact_hits"] == {0.03: 2, 0.05: 2, 0.08: 1}
    assert staged["paired_guarded_retention"]["0.05"]["achieved_guarded_clear"] == 0
    assert result["consistency_with_exp022a_reference_rows"]["control"]["per_height"]["0.08"] == {
        "successes": 1, "agrees_with_exp022a_rows": True}
    assert result["extrema_post_hoc"]["h=0.05"]["tied_centres_m"] == [1.2]
    curve = [json.loads(line) for line in (out / "curve.jsonl").read_text().splitlines()]
    assert len(curve) == len(grid) * len(heights)
    with np.load(out / "exact_hits.npz") as payload:
        assert payload["hits"].shape == (len(grid), len(heights), len(seeds))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["inputs"]["exp022a"]["file_sha256"]["summary.json"]
    assert receipt["scoring_identity"] == {"injected_probe": True}
    assert receipt["evidence_anchors"]["curve"]["n_rows"] == len(curve)


def test_run_refuses_when_exp022a_rows_disagree(tmp_path):
    lifts = [(0.06, 1.2), (0.10, 1.2), (0.0, 0.0)]
    archive, seeds, arrays = _fake_archive(tmp_path, lifts)
    heights = (0.05,)
    exp022 = _fake_exp022(tmp_path, seeds, arrays, {"staged": 1.2}, heights, corrupt_seed=4401)
    with pytest.raises(ValueError, match="disagree with EXP-022A for seeds \\[4401\\]"):
        cc.run(archive=archive, exp022_dir=exp022, out=tmp_path / "out", centres_m=[1.2],
               heights_m=heights, probe_factory=_WindowProbe, locked_manifest=None)


def test_run_refuses_an_exp022a_centre_off_the_grid(tmp_path):
    archive, seeds, arrays = _fake_archive(tmp_path, [(0.06, 1.2)])
    exp022 = _fake_exp022(tmp_path, seeds, arrays, {"staged": 1.25}, (0.05,))
    with pytest.raises(ValueError, match="not on the scanned grid"):
        cc.run(archive=archive, exp022_dir=exp022, out=tmp_path / "out", centres_m=[1.2],
               heights_m=(0.05,), probe_factory=_WindowProbe, locked_manifest=None)


def test_default_grid_contains_both_exp022a_centres():
    assert any(abs(x - 1.2) < 1e-9 for x in cc.CENTRE_GRID_M)
    assert any(abs(x - 3.6) < 1e-9 for x in cc.CENTRE_GRID_M)
    assert cc.CENTRE_GRID_M[0] == pytest.approx(0.6) and cc.CENTRE_GRID_M[-1] == pytest.approx(6.6)
