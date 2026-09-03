"""CPU tests for the EXP-026 duck-family contract analyser."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import analyze_duck_contract as dc


# --------------------------------------------------------------------------- fixtures


def write_clip(cache, key, *, scene_id="demo_partial_beam_h0.950_w2.250_n3_g1.50",
               peak_dip_m=0.35, n_frames=8, seed=100, **meta_overrides):
    cache.mkdir(parents=True, exist_ok=True)
    qpos = np.zeros((n_frames, 36), dtype=float)
    qpos[:, 2] = 0.78
    np.save(cache / f"{key}.npy", qpos)
    meta = {"cache_version": 2, "noise_stream_version": 2, "fps": 25.0, "scene_id": scene_id,
            "peak_dip_m": peak_dip_m, "n_frames": n_frames, "seed": seed, "repair_iteration": 0}
    meta.update(meta_overrides)
    (cache / f"{key}.json").write_text(json.dumps(meta))
    return qpos, meta


def make_row(key, *, scene_id="demo_partial_beam_h0.950_w2.250_n3_g1.50", peak_dip_m=0.35,
             terminated=False, **overrides):
    row = {"clip_key": key, "scene_id": scene_id, "peak_dip_m": peak_dip_m,
           "ref_min_overhead_m": 0.17, "route_len_m": 17.5, "method": "heuristic",
           "terminated": terminated, "progress": 0.6}
    row.update(overrides)
    return row


def fake_features(_body, qpos, _sup_h, _sup_v, _fps):
    """Deterministic stand-in: the feature values are encoded in the first row of qpos."""
    return {"max_unsupported_run_s": float(qpos[0, 0]),
            "max_root_planar_speed": float(qpos[0, 1]),
            "mean_root_planar_speed": float(qpos[0, 1]) / 2.0,
            "bilateral_flight_frac": 0.1, "mean_support_feet": 1.5, "root_z_min": 0.6}


def encode(cache, key, *, run_s, speed, **kwargs):
    qpos, meta = write_clip(cache, key, **kwargs)
    qpos[0, 0], qpos[0, 1] = run_s, speed
    np.save(cache / f"{key}.npy", qpos)
    return meta


# ----------------------------------------------------------------------------- inputs


def test_first_rollout_per_clip_keeps_file_order_and_dedupes():
    rows = [make_row("a", terminated=True), make_row("b"), make_row("a", terminated=False),
            make_row("c"), make_row("b", terminated=True)]
    got = dc.first_rollout_per_clip(rows)
    assert [r["clip_key"] for r in got] == ["a", "b", "c"]
    assert got[0]["terminated"] is True     # the *first* rollout of "a", not the later one
    assert got[1]["terminated"] is False


def test_load_clip_returns_qpos_and_meta(tmp_path):
    cache = tmp_path / "clips"
    write_clip(cache, "k1")
    qpos, meta = dc.load_clip(cache, "k1", make_row("k1"))
    assert qpos.shape == (8, 36) and meta["seed"] == 100


@pytest.mark.parametrize("kwargs,row_kwargs,match", [
    ({"cache_version": 1}, {}, "cache_version"),
    ({"noise_stream_version": 1}, {}, "noise-stream-v2"),
    ({"fps": 30.0}, {}, "fps"),
    ({"scene_id": "other_scene"}, {}, "sidecar scene"),
    ({"peak_dip_m": 0.5}, {}, "peak_dip_m"),
])
def test_load_clip_refuses_every_provenance_mismatch(tmp_path, kwargs, row_kwargs, match):
    cache = tmp_path / "clips"
    write_clip(cache, "k1", **kwargs)
    with pytest.raises(dc.DuckContractRefusal, match=match):
        dc.load_clip(cache, "k1", make_row("k1", **row_kwargs))


def test_load_clip_refuses_a_sidecar_frame_count_that_disagrees(tmp_path):
    cache = tmp_path / "clips"
    _, meta = write_clip(cache, "k1", n_frames=8)
    (cache / "k1.json").write_text(json.dumps({**meta, "n_frames": 999}))
    with pytest.raises(dc.DuckContractRefusal, match="n_frames disagrees"):
        dc.load_clip(cache, "k1", make_row("k1"))


def test_load_clip_refuses_a_missing_or_malformed_clip(tmp_path):
    cache = tmp_path / "clips"
    cache.mkdir()
    with pytest.raises(dc.DuckContractRefusal, match="missing from the cache"):
        dc.load_clip(cache, "gone", make_row("gone"))
    write_clip(cache, "bad")
    np.save(cache / "bad.npy", np.zeros((8, 12)))
    with pytest.raises(dc.DuckContractRefusal, match="shape"):
        dc.load_clip(cache, "bad", make_row("bad"))


def test_beam_count_parses_the_scene_id():
    assert dc.beam_count("demo_partial_beam_h0.950_w2.250_n3_g1.50") == 3
    assert dc.beam_count("demo_partial_beam_h1.050_w2.250_n6_g2.50") == 6
    assert dc.beam_count("scene_without_a_count") is None


# ------------------------------------------------------------------------- statistics


def test_pooled_auc_is_directional_and_handles_ties():
    # terminated clips score higher: a perfect separation in the hypothesised direction.
    assert dc.pooled_auc([3.0, 4.0, 1.0, 2.0], [True, True, False, False]) == 1.0
    # reversed: the AUC goes below 0.5 rather than being flipped.
    assert dc.pooled_auc([1.0, 2.0, 3.0, 4.0], [True, True, False, False]) == 0.0
    assert dc.pooled_auc([1.0, 1.0], [True, False]) == 0.5
    with pytest.raises(ValueError, match="align"):
        dc.pooled_auc([1.0, 2.0], [True])


def test_cluster_bootstrap_auc_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    scenes = [f"s{i // 20}" for i in range(120)]
    y = [bool(i % 2) for i in range(120)]
    values = [float(rng.normal(1.0 if t else 0.0)) for t in y]
    first = dc.cluster_bootstrap_auc(values, y, scenes, n_boot=200)
    second = dc.cluster_bootstrap_auc(values, y, scenes, n_boot=200)
    assert first == second
    low, high = first["ci95"]
    assert low < dc.pooled_auc(values, y) < high
    assert first["n_scenes"] == 6 and first["n_resamples_used"] > 0


def test_within_scene_auc_applies_the_evaluability_rule_and_weights_by_pairs():
    # scene A: 6 vs 6 with perfect separation; scene B: 6 vs 6 reversed; scene C: too few.
    scenes = ["A"] * 12 + ["B"] * 12 + ["C"] * 4
    y = ([True] * 6 + [False] * 6) * 2 + [True, True, False, False]
    values = ([2.0] * 6 + [1.0] * 6) + ([1.0] * 6 + [2.0] * 6) + [9.0, 9.0, 0.0, 0.0]
    got = dc.within_scene_auc(values, y, scenes)
    per = {r["scene_id"]: r for r in got["per_scene"]}
    assert per["A"]["auc"] == 1.0 and per["B"]["auc"] == 0.0
    assert per["C"]["evaluable"] is False and per["C"]["auc"] is None
    assert got["n_evaluable_scenes"] == 2 and got["n_scenes"] == 3
    # Equal pair counts, so both means are the midpoint; C never contributes.
    assert got["weighted_mean_auc"] == pytest.approx(0.5)
    assert got["unweighted_mean_auc"] == pytest.approx(0.5)
    assert got["total_pairs"] == 72


def test_within_scene_auc_reports_none_when_no_scene_is_evaluable():
    got = dc.within_scene_auc([1.0, 2.0], [True, False], ["A", "A"])
    assert got["weighted_mean_auc"] is None and got["n_evaluable_scenes"] == 0


def test_within_scene_auc_can_reverse_the_pooled_ranking():
    """The confound-controlled measure must be able to disagree with the pooled AUC."""
    # Scene A is the high-value scene and mostly terminates; scene B is the low-value scene and
    # mostly survives, so pooling makes the feature look predictive.  Inside each scene the
    # ranking is reversed.
    scenes = ["A"] * 20 + ["B"] * 20
    y = ([True] * 15 + [False] * 5) + ([True] * 5 + [False] * 15)
    values = ([10.0] * 15 + [10.1] * 5) + ([1.0] * 5 + [1.1] * 15)
    assert dc.pooled_auc(values, y) > 0.5           # pooled: terminated look larger
    assert dc.within_scene_auc(values, y, scenes)["weighted_mean_auc"] == 0.0


def test_cluster_bootstrap_differences_are_deterministic():
    scenes = [f"s{i // 12}" for i in range(48)]
    y = [bool(i % 2) for i in range(48)]
    a = [3.0 if t else 1.0 for t in y]
    b = [1.0 if t else 3.0 for t in y]
    pooled = dc.cluster_bootstrap_difference(a, b, y, scenes, n_boot=100)
    assert pooled == dc.cluster_bootstrap_difference(a, b, y, scenes, n_boot=100)
    assert pooled["ci95"][0] > 0            # a separates, b anti-separates
    within = dc.cluster_bootstrap_within_scene_difference(a, b, y, scenes, n_boot=100)
    assert within == dc.cluster_bootstrap_within_scene_difference(a, b, y, scenes, n_boot=100)
    assert within["ci95"][0] > 0


def test_screen_table_arithmetic_and_intervals():
    runs = [0.5, 0.3, 0.1, 0.05]
    y = [True, True, False, False]
    got = dc.screen_table(runs, y, 0.20)
    assert got["flagged_terminated"] == 2 and got["terminated"] == 2
    assert got["flagged_survivors"] == 0 and got["survivors"] == 2
    assert got["sensitivity"] == 1.0 and got["specificity"] == 1.0
    low, high = got["sensitivity_wilson95"]
    assert 0.0 < low < 1.0 and high == pytest.approx(1.0)
    # strictly greater than the threshold, matching the step-family gate convention
    assert dc.screen_table([0.20], [True], 0.20)["flagged_terminated"] == 0


def test_strata_table_marks_small_strata_not_evaluable():
    records = []
    for i in range(24):
        records.append({"terminated": bool(i % 2), "beam_count": 3,
                        "features": {"max_root_planar_speed": float(i), "peak_dip_m": 0.3,
                                     "max_unsupported_run_s": float(i)}})
    for i in range(4):
        records.append({"terminated": bool(i % 2), "beam_count": 6,
                        "features": {"max_root_planar_speed": 1.0, "peak_dip_m": 0.3,
                                     "max_unsupported_run_s": 1.0}})
    got = dc.strata_table(records, lambda r: r["beam_count"], "beam_count")
    by = {s["stratum"]: s for s in got["strata"]}
    assert by[3]["evaluable"] is True and set(by[3]["auc"]) == set(dc.GROUPS)
    assert by[6]["evaluable"] is False and by[6]["auc"] is None


def test_decide_covers_every_branch():
    def pooled(contact, speed):
        return {"contact": {"auc": contact}, "speed": {"auc": speed}}

    def within(contact, speed):
        return {"contact": {"weighted_mean_auc": contact},
                "speed": {"weighted_mean_auc": speed}}

    transfer = dc.decide(pooled(0.9, 0.6), within(0.8, 0.5))
    assert transfer["verdict"] == "contract_transfers_to_the_duck_family"
    assert transfer["pooled_and_within_scene_agree"] is True

    speed_limited = dc.decide(pooled(0.5, 0.8), within(0.4, 0.7))
    assert speed_limited["verdict"] == "speed_limited_and_confounded_by_the_14s_clip_cap"

    split = dc.decide(pooled(0.9, 0.6), within(0.4, 0.7))
    assert split["verdict"] == "pooled_and_within_scene_disagree_claim_limited_to_what_they_share"
    assert split["pooled_and_within_scene_agree"] is False

    none = dc.decide(pooled(0.9, 0.6), within(None, None))
    assert none["verdict"] == "pooled_only_no_evaluable_scene"
    assert none["within_scene"]["transfer"] is None


# --------------------------------------------------------------------------------- run


def _campaign(tmp_path, *, n_per_scene=12, contact_beats_speed=True):
    cache = tmp_path / "clips"
    rows = []
    for scene in ("A", "B"):
        scene_id = f"demo_partial_beam_h0.950_w2.250_n3_g1.50_{scene}"
        for i in range(n_per_scene):
            key = f"{scene}{i:02d}"
            terminated = i < n_per_scene // 2
            run_s = (0.5 if terminated else 0.05) if contact_beats_speed else 0.1
            speed = 0.5 if contact_beats_speed else (2.0 if terminated else 0.5)
            encode(cache, key, run_s=run_s, speed=speed, scene_id=scene_id, peak_dip_m=0.35)
            rows.append(make_row(key, scene_id=scene_id, peak_dip_m=0.35, terminated=terminated))
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return cache, rows_path, len(rows)


def _run_kwargs(tmp_path, cache, rows_path, n):
    return {"out": tmp_path / "out", "rows_path": rows_path, "cache": cache, "n_boot": 50,
            "body": object(), "feature_fn": fake_features, "expected_n_clips": n,
            "thresholds": {"support_height_m": 0.0465, "support_speed_mps": 1.175}}


def test_run_end_to_end_writes_rows_and_a_receipt(tmp_path):
    cache, rows_path, n = _campaign(tmp_path)
    receipt = dc.run(**_run_kwargs(tmp_path, cache, rows_path, n))
    assert receipt["status"] == "complete" and receipt["post_hoc"] is True
    summary = receipt["summary"]
    assert summary["n_clips"] == n and summary["n_terminated"] == n // 2
    assert summary["n_scenes"] == 2
    assert summary["pooled_auc_by_group"]["contact"]["auc"] == 1.0
    assert summary["pooled_auc_by_group"]["speed"]["auc"] == 0.5
    assert summary["within_scene_auc_by_group"]["contact"]["weighted_mean_auc"] == 1.0
    assert summary["decision"]["verdict"] == "contract_transfers_to_the_duck_family"
    assert summary["screen"]["calibrated_0p20s"]["sensitivity"] == 1.0
    assert summary["float_fraction"]["n_with_run_over_0p20s"] == n // 2
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == n and rows[0]["beam_count"] == 3
    assert rows[0]["features"]["peak_dip_m"] == 0.35     # committed row field joined in
    assert receipt["evidence_anchors"]["rows"]["n_rows"] == n
    assert receipt["protocol"]["sha256"] and receipt["inputs"]["rows"]["sha256"]


def test_run_reports_the_speed_limited_verdict_when_speed_wins(tmp_path):
    cache, rows_path, n = _campaign(tmp_path, contact_beats_speed=False)
    receipt = dc.run(**_run_kwargs(tmp_path, cache, rows_path, n))
    decision = receipt["summary"]["decision"]
    assert decision["verdict"] == "speed_limited_and_confounded_by_the_14s_clip_cap"
    assert decision["pooled"]["speed_auc"] > decision["pooled"]["contact_auc"]


def test_run_refuses_a_wrong_denominator_or_a_nonempty_output(tmp_path):
    cache, rows_path, n = _campaign(tmp_path)
    kwargs = _run_kwargs(tmp_path, cache, rows_path, n)
    with pytest.raises(dc.DuckContractRefusal, match="planned denominator"):
        dc.run(**{**kwargs, "expected_n_clips": n + 1})
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "stale.txt").write_text("x")
    with pytest.raises(dc.DuckContractRefusal, match="non-empty output"):
        dc.run(**{**kwargs, "out": out})


def test_group_primaries_and_thresholds_match_the_protocol():
    assert dc.GROUPS["speed"][0] == "max_root_planar_speed"
    assert dc.GROUPS["crouch"][0] == "peak_dip_m"
    assert dc.GROUPS["contact"][0] == "max_unsupported_run_s"
    assert dc.SCREEN_S == 0.20 and dc.POSTHOC_S == 0.28
    assert dc.N_CLIPS == 526 and dc.MIN_PER_OUTCOME == 5 and dc.MIN_STRATUM_N == 20
