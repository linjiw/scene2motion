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


# --- tolerant +-r union (addressability, never a fixed-obstacle success probability) --------

def _union_hits():
    """(5 centres, 1 height, 4 clips) on the grid 1.10 .. 1.30 around the 1.2 m centre.

    clip 0: clears at 1.15 and 1.20 -- two centres inside every window, one reference;
    clip 1: clears only at 1.20 -- the exact centre;
    clip 2: clears only at 1.30 -- inside +-0.10 but not at the centre;
    clip 3: never clears.
    """
    hits = np.zeros((5, 1, 4), dtype=bool)
    hits[[1, 2], 0, 0] = True
    hits[2, 0, 1] = True
    hits[4, 0, 2] = True
    return hits, [1.10, 1.15, 1.20, 1.25, 1.30], [0.05]


def test_tolerant_union_counts_a_reference_once_however_many_centres_it_clears():
    hits, centres, heights = _union_hits()
    block = cc.tolerant_union(hits, centres, heights, centre_m=1.20, radii_m=(0.05, 0.10))
    tight = block["windows"]["r=0.05"]["by_height"]["h=0.05"]
    # clips 0 and 1 clear inside [1.15, 1.25]; clip 0 clears at two of those centres.
    assert int(hits[[1, 2], 0, 0].sum()) == 2
    assert tight["union_successes"] == 2 and tight["total"] == 4
    assert tight["union_rate"] == pytest.approx(0.5)
    assert tight["exact_at_centre_successes"] == 2
    assert tight["extra_references_bought_by_the_window"] == 0
    low, high = tight["wilson95_descriptive_only"]
    assert low < 0.5 < high
    assert "wilson95" not in tight


def test_tolerant_union_is_monotone_in_the_radius():
    hits, centres, heights = _union_hits()
    block = cc.tolerant_union(hits, centres, heights, centre_m=1.20,
                              radii_m=(0.10, 0.0, 0.05))
    counts = {key: window["by_height"]["h=0.05"]["union_successes"]
              for key, window in block["windows"].items()}
    assert counts == {"r=0.00": 2, "r=0.05": 2, "r=0.10": 3}
    radii = [block["windows"][key]["radius_m"] for key in block["windows"]]
    assert radii == sorted(radii)
    ordered = [counts[key] for key in block["windows"]]
    assert all(a <= b for a, b in zip(ordered, ordered[1:]))
    # The union never drops below the exact count at the centre it is built around.
    for window in block["windows"].values():
        row = window["by_height"]["h=0.05"]
        assert row["union_successes"] >= row["exact_at_centre_successes"]
        assert (row["extra_references_bought_by_the_window"]
                == row["union_successes"] - row["exact_at_centre_successes"])


def test_tolerant_union_reports_the_window_membership_it_used():
    hits, centres, heights = _union_hits()
    block = cc.tolerant_union(hits, centres, heights, centre_m=1.20, radii_m=(0.10,))
    window = block["windows"]["r=0.10"]
    assert window["scanned_centres_m"] == pytest.approx([1.10, 1.15, 1.20, 1.25, 1.30])
    assert window["n_scanned_centres"] == 5
    assert window["window_m"] == pytest.approx([1.10, 1.30])
    assert block["centre_m"] == pytest.approx(1.20) and block["total"] == 4


def test_tolerant_union_carries_its_own_no_fixed_obstacle_labelling():
    hits, centres, heights = _union_hits()
    block = cc.tolerant_union(hits, centres, heights, centre_m=1.20, radii_m=(0.10,))
    label = block["label"]
    assert "never a fixed-obstacle success probability" in label
    assert "ANY scanned centre" in label
    assert "exact_at_centre_successes" in label
    assert "counts once" in block["definition"]
    assert "post hoc maximum" in block["centre_provenance"]
    # The interval is descriptive: the key name and the label both have to say so, because a
    # bare "wilson95" is the shape that gets lifted into a table as an inferential rate.
    assert "wilson95_descriptive_only carries no inferential claim" in label
    row = block["windows"]["r=0.10"]["by_height"]["h=0.05"]
    assert "wilson95_descriptive_only" in row and "wilson95" not in row


def test_tolerant_union_disambiguates_the_voided_historical_rates():
    hits, centres, heights = _union_hits()
    block = cc.tolerant_union(hits, centres, heights, centre_m=1.20, radii_m=(0.05, 0.10))
    dis = block["historical_rates_disambiguation"]
    # The historical triple is a lower-bound+tolerance calculation, not an exact-probe union.
    assert dis["historical"]["counts_of_64"] == {"h=0.03": 22, "h=0.05": 20, "h=0.08": 17}
    assert dis["historical"]["rates_of_64"]["h=0.05"] == 0.312
    assert "NOT a union of exact" in dis["historical"]["method"]
    # This run's own counts travel beside them, computed, never transcribed.
    assert dis["union_counts_by_window"] == {
        "r=0.05": {"h=0.05": 2}, "r=0.10": {"h=0.05": 3}}
    assert dis["exact_counts_at_centre"] == {"h=0.05": 2}


def test_tolerant_union_refuses_a_centre_off_the_grid_or_a_bad_matrix():
    hits, centres, heights = _union_hits()
    with pytest.raises(ValueError, match="not on the scanned grid"):
        cc.tolerant_union(hits, centres, heights, centre_m=1.22, radii_m=(0.10,))
    with pytest.raises(ValueError, match="centre, height, clip"):
        cc.tolerant_union(np.zeros((2, 2, 3), dtype=bool), centres, heights, centre_m=1.20)
    with pytest.raises(ValueError, match="no clips"):
        cc.tolerant_union(np.zeros((5, 1, 0), dtype=bool), centres, heights, centre_m=1.20)


def test_tolerant_union_handles_multiple_heights_independently():
    hits = np.zeros((3, 2, 4), dtype=bool)
    hits[0, 0, 0] = True          # 3 cm: one clip, off the centre
    hits[1, 0, 1] = True          # 3 cm: one clip, at the centre
    hits[1, 1, 1] = True          # 5 cm: the same clip, at the centre
    block = cc.tolerant_union(hits, [1.15, 1.20, 1.25], [0.03, 0.05],
                              centre_m=1.20, radii_m=(0.05,))
    rows = block["windows"]["r=0.05"]["by_height"]
    assert rows["h=0.03"]["union_successes"] == 2
    assert rows["h=0.03"]["exact_at_centre_successes"] == 1
    assert rows["h=0.05"]["union_successes"] == 1
    assert rows["h=0.05"]["exact_at_centre_successes"] == 1


def test_run_writes_the_tolerant_union_block_into_the_receipt(tmp_path):
    lifts = [(0.06, 1.2), (0.10, 1.2), (0.02, 1.2), (0.09, 3.6), (0.0, 0.0), (0.30, 2.0)]
    archive, seeds, arrays = _fake_archive(tmp_path, lifts)
    heights = (0.03, 0.05, 0.08)
    grid = [1.0, 1.2, 2.0, 3.6]
    exp022 = _fake_exp022(tmp_path, seeds, arrays, {"staged": 1.2, "control": 3.6}, heights)
    out = tmp_path / "out"
    result = cc.run(archive=archive, exp022_dir=exp022, out=out, centres_m=grid,
                    heights_m=heights, union_centre_m=1.2, union_radii_m=(0.1, 0.25),
                    probe_factory=_WindowProbe, locked_manifest=None)
    receipt = json.loads((out / "receipt.json").read_text())
    block = receipt["tolerant_union"]
    assert block == result["tolerant_union"]
    assert "never a fixed-obstacle success probability" in block["label"]
    assert sorted(block["windows"]) == ["r=0.10", "r=0.25"]
    # The coarse grid puts only 1.2 m inside +-0.10 m and 1.0/1.2 m inside +-0.25 m, and no
    # clip clears at 1.0 m, so both unions equal the exact count: the block cannot inflate a
    # fixed-obstacle number out of centres that were never scanned.
    assert block["windows"]["r=0.10"]["scanned_centres_m"] == pytest.approx([1.2])
    assert block["windows"]["r=0.25"]["scanned_centres_m"] == pytest.approx([1.0, 1.2])
    assert block["windows"]["r=0.10"]["n_scanned_centres"] == 1
    assert block["windows"]["r=0.25"]["n_scanned_centres"] == 2
    for key in ("r=0.10", "r=0.25"):
        row = block["windows"][key]["by_height"]["h=0.05"]
        assert row["union_successes"] == row["exact_at_centre_successes"] == 2
    assert receipt["schema"] == "exact-centre-cost-curve-v2"
    dis = block["historical_rates_disambiguation"]
    assert dis["union_counts_by_window"]["r=0.10"] == {"h=0.03": 2, "h=0.05": 2, "h=0.08": 1}
    assert dis["exact_counts_at_centre"] == {"h=0.03": 2, "h=0.05": 2, "h=0.08": 1}


def test_porcelain_paths_keeps_the_whole_path():
    # "git status --porcelain" pads the status to two columns, so an unstaged modification
    # starts with a space. Stripping the output before splitting eats the path's first
    # character -- that is a real bug this parser exists to avoid.
    text = (" M experiments/analyze_exact_centre_cost_curve.py\n"
            "?? outputs/new_dir/\n"
            "A  scene2motion/robot.py\n"
            "R  docs/old.md -> docs/new.md\n")
    assert cc.porcelain_paths(text) == [
        "docs/new.md", "experiments/analyze_exact_centre_cost_curve.py",
        "outputs/new_dir/", "scene2motion/robot.py"]
    assert cc.porcelain_paths("") == []


def test_receipt_provenance_names_the_tree_the_run_happened_in(tmp_path):
    lifts = [(0.06, 1.2), (0.10, 1.2), (0.0, 0.0)]
    archive, seeds, arrays = _fake_archive(tmp_path, lifts)
    exp022 = _fake_exp022(tmp_path, seeds, arrays, {"staged": 1.2}, (0.05,))
    out = tmp_path / "out"
    cc.run(archive=archive, exp022_dir=exp022, out=out, centres_m=[1.2], heights_m=(0.05,),
           probe_factory=_WindowProbe, locked_manifest=None)
    prov = json.loads((out / "receipt.json").read_text())["provenance"]
    # A bare {commit, dirty} cannot distinguish "run in the main tree with uncommitted work"
    # from "run in a clean linked worktree at that commit"; these fields can.
    assert set(prov["code"]) == {"commit", "dirty", "dirty_paths", "tree_path",
                                 "is_linked_worktree"}
    assert isinstance(prov["code"]["dirty_paths"], list)
    assert prov["code"]["dirty"] == bool(prov["code"]["dirty_paths"])
    assert "linked worktree" in prov["run_tree_note"]
    assert "scoring_identity.source_sha256" in prov["run_tree_note"]
