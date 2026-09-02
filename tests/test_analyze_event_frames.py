"""Focused CPU tests for the A0 event-frame analyser (no MuJoCo, no archives)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import analyze_event_frames as ef
from experiments import analyze_exp021_exact_addressability as exact
from experiments import calibrate_ramp_route_phase as cal


# ---------------------------------------------------------------------------------------
# Synthetic world: root x advances linearly; qpos[0, 7] encodes the profile peak index + 1
# (0 = no lift); qpos[0, 8] encodes the frame of the synthetic foot-clearance bump.
# ---------------------------------------------------------------------------------------

def _scan_grid():
    route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)
    return np.linspace(route[0, 1] + 0.3, route[-1, 1] - 0.3, ef.PROFILE_POINTS)


def _fake_profile(qpos, route, depth_m, *, n_points):
    assert depth_m == ef.OBSTACLE_DEPTH_M and n_points == ef.PROFILE_POINTS
    xs = np.linspace(route[0, 1] + 0.3, route[-1, 1] - 0.3, n_points)
    heights = np.zeros(n_points)
    index = int(round(float(qpos[0, 7]))) - 1
    if index >= 0:
        heights[index - 1:index + 2] = [0.05, 0.10, 0.05]
    return xs, heights


def _fake_support(body, qpos, sup_h, sup_v):
    T = len(qpos)
    root_x = np.asarray(qpos[:, 0], dtype=float)
    frames = np.arange(T)
    bump_frame = int(round(float(qpos[0, 8])))
    lifting = int(round(float(qpos[0, 7]))) > 0
    left_bottom = 0.20 * np.exp(-((frames - bump_frame) / 3.0) ** 2) if lifting else np.zeros(T)
    feet = {
        "left": {"forward_representative_m": root_x, "bottom_clearance_m": left_bottom,
                 "planar_speed_mps": np.zeros(T)},
        "right": {"forward_representative_m": root_x - 0.30, "bottom_clearance_m": np.zeros(T),
                  "planar_speed_mps": np.zeros(T)},
    }
    sup_left = left_bottom <= sup_h
    sup_right = np.ones(T, dtype=bool)
    if lifting:
        sup_right[bump_frame - 2:bump_frame + 3] = False
    return feet, sup_left, sup_right


def _fake_side(qpos, feet, lift_x, depth_m):
    return "left"


def _clip(speed_mps, peak_index, bump_frame, frames=200):
    q = np.zeros((frames, 36), dtype=np.float32)
    q[:, 0] = np.arange(frames) * speed_mps / ef.FPS
    q[:, 2] = 0.75
    q[:, 3] = 1.0
    q[0, 7] = (peak_index + 1) if peak_index is not None else 0
    q[0, 8] = bump_frame if bump_frame is not None else 0
    return q


def _exp021_archive(tmp_path, clips):
    archive = tmp_path / "exp021"
    archive.mkdir()
    xs = _scan_grid()
    step = float(xs[1] - xs[0])
    seeds = list(range(4400, 4400 + len(clips)))
    arrays = {f"s{seed}": clip for seed, (clip, _peak) in zip(seeds, clips)}
    np.savez(archive / "qpos.npz", **arrays)
    rows = []
    for seed, (clip, peak) in zip(seeds, clips):
        row = {"seed": seed, "prompt": "A person steps over an obstacle.",
               "progress_ratio": 1.0}
        if peak is None:
            row.update(lift_x_m=None, lift_height_m=0.0, n_lift_regions=0,
                       lift_support_m=0.0)
        else:
            row.update(lift_x_m=float(xs[peak]), lift_height_m=0.10, n_lift_regions=1,
                       lift_support_m=3 * step, lift_side="left")
        row["clears_height"] = {"0.03": row["lift_height_m"] >= 0.03}
        rows.append(row)
    (archive / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {
        "schema": "exp021-elicited-lift-distribution-v1",
        "experiment": "exp021_elicited_lift_distribution",
        "status": "complete", "complete": True, "blocked": False,
        "actual_ardy_samples": len(arrays),
        "design": {"pool_seeds": seeds},
        "summary": {"n_clips": len(arrays)},
        "provenance": {"code": {"commit": "fake"}},
        "evidence_anchors": {"qpos": {"path": "qpos.npz", "n_arrays": len(arrays),
                                      "content_sha256": exact.array_content_sha256(arrays)}},
    }
    (archive / "receipt.json").write_text(json.dumps(receipt))
    return archive, rows


def _exp023_archive(tmp_path, clips_by_key):
    """``clips_by_key``: {"s4500_step_0": (clip, event_frame or None), ...}."""
    archive = tmp_path / "exp023"
    archive.mkdir()
    arrays = {key: clip for key, (clip, _event) in clips_by_key.items()}
    np.savez(archive / "qpos.npz", **arrays)
    rows = []
    for key, (clip, event_frame) in clips_by_key.items():
        seed, arm = key.split("_", 1)
        row = {"archive_key": key, "seed": int(seed[1:]), "arm": arm,
               "onset_frame": ef.EXP023_ONSETS[arm], "archived_frames": len(clip),
               "scored_frames": 200, "prompt_schedule": ["x"] * 4,
               "supporting_motion": {"progress_ratio": 1.0}}
        if arm == "all_walk":
            row["event"] = None
            row["control_events"] = {
                str(onset): {"present": False, "max_profile_height_m": 0.0}
                for onset in (0, 52, 104)}
        elif event_frame is None:
            row["event"] = {"present": False, "frame": None, "latency_frames": None,
                            "missing_reason": "whole_body_clearance_below_3cm",
                            "max_profile_height_m": 0.0}
        else:
            onset = ef.EXP023_ONSETS[arm]
            row["event"] = {"present": True, "frame": event_frame,
                            "latency_frames": event_frame - onset,
                            "latency_s": (event_frame - onset) / 25.0, "side": "left",
                            "profile_x_m": float(clip[event_frame, 0]),
                            "foot_x_m": float(clip[event_frame, 0]),
                            "whole_body_clearance_m": 0.10, "missing_reason": None,
                            "max_profile_height_m": 0.10,
                            "analysis_window_start_frame": onset,
                            "analysis_window_end_frame": onset + 95}
        rows.append(row)
    (archive / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {
        "schema": "exp023-prompt-handoff-v1", "experiment": "exp023_prompt_handoff",
        "status": "complete", "complete": True, "blocked": False,
        "actual_ardy_samples": len(arrays), "sample_count_exact": True,
        "provenance": {"code": {"commit": "fake"}},
        "evidence_anchors": {"qpos": {"path": "qpos.npz", "n_arrays": len(arrays),
                                      "content_sha256": exact.array_content_sha256(arrays)}},
    }
    (archive / "receipt.json").write_text(json.dumps(receipt))
    return archive, rows


def _fake_git(_repo):
    return {"commit": "test", "dirty": False, "status": []}


def _run(tmp_path, clips021, clips023, **overrides):
    archive021, _ = _exp021_archive(tmp_path, clips021)
    archive023, _ = _exp023_archive(tmp_path, clips023)
    kwargs = dict(exp021_archive=archive021, exp023_archive=archive023, locked=False,
                  profile_fn=_fake_profile, support_fn=_fake_support, lift_side_fn=_fake_side,
                  code_state_fn=_fake_git)
    kwargs.update(overrides)
    return ef.run_analysis(tmp_path / "out", **kwargs)


# ---------------------------------------------------------------------------------------
# Pure definitions
# ---------------------------------------------------------------------------------------

def test_root_crossing_frame_is_first_frame_reaching_the_target():
    root_x = np.arange(200) * 0.04  # 0.04 m per frame
    assert ef.root_crossing_frame(root_x, 1.2) == 30
    assert ef.root_crossing_frame(root_x, 1.21) == 31
    assert ef.root_crossing_frame(root_x, 0.0) == 0
    assert ef.root_crossing_frame(root_x, 50.0) is None
    with pytest.raises(ValueError):
        ef.root_crossing_frame(np.array([np.nan]), 1.0)


def test_nominal_frame_is_the_prescribed_route_arrival_frame():
    route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)
    for frame in (0, 34, 60, 199):
        assert ef.nominal_frame(route[frame, 1]) == pytest.approx(frame, abs=1e-9)
    assert ef.nominal_frame(1.2) == pytest.approx(1.2 / 0.9045226130653267 * 25.0)
    assert ef.nominal_frame(1.2, speed_mps=1.0, fps=30.0) == pytest.approx(36.0)


def test_lift_region_and_foot_lift_frame_on_a_synthetic_lift():
    xs = _scan_grid()
    heights = np.zeros_like(xs)
    heights[15:19] = [0.02, 0.08, 0.05, 0.01]  # one region; peak at 16
    heights[40:42] = 0.03  # a second, separate region
    region = ef.lift_region_extent(xs, heights, float(xs[16]))
    assert (region["lo_index"], region["hi_index"], region["peak_index"]) == (15, 18, 16)
    assert region["n_points"] == 4
    with pytest.raises(ValueError, match="positive profile point"):
        ef.lift_region_extent(xs, heights, float(xs[30]))
    with pytest.raises(ValueError, match="profile grid"):
        ef.lift_region_extent(xs, heights, float(xs[16]) + 0.01)

    frames = np.arange(200)
    forward = frames * 0.036  # crosses xs[15]-0.14 .. xs[18]+0.14 over a known band
    bump = 0.25 * np.exp(-((frames - 36) / 2.0) ** 2)
    bump[100] = 0.9  # a taller peak far outside the region must be ignored
    foot = {"forward_representative_m": forward, "bottom_clearance_m": bump}
    hit = ef.foot_lift_frame(foot, region["lo_x_m"], region["hi_x_m"])
    assert hit["frame"] == 36
    assert hit["clearance_m"] == pytest.approx(0.25)
    lo = int(np.ceil((region["lo_x_m"] - ef.HALF_SLAB_M) / 0.036))
    hi = int(np.floor((region["hi_x_m"] + ef.HALF_SLAB_M) / 0.036))
    assert (hit["first_frame_inside"], hit["last_frame_inside"]) == (lo, hi)
    assert ef.foot_lift_frame(foot, 50.0, 50.0) is None
    assert ef.containing_run([(10, 20), (30, 40)], 35) == (30, 40)
    assert ef.containing_run([(10, 20), (30, 40)], 20) is None


def test_summarize_frames_uses_planned_denominators_and_10_frame_bins():
    values = [10, 20, 34, 45, 49, 50, 61, None]
    block = ef.summarize_frames(values, planned_n=8)
    assert block["n_present"] == 7 and block["n_missing"] == 1
    assert block["frames"]["median"] == 45
    assert block["seconds"]["median"] == pytest.approx(45 / 25)
    first = block["inside_first_50_frames"]
    assert first["count"] == 5 and first["planned_denominator"] == 8
    assert first["fraction_of_planned"] == pytest.approx(5 / 8)
    assert first["fraction_of_present"] == pytest.approx(5 / 7)
    assert first["wilson95_of_planned"][0] < 5 / 8 < first["wilson95_of_planned"][1]
    late = block["after_frame_60"]
    assert late["count"] == 1  # 61 only; 50 and 49 are not "after 60"
    hist = block["histogram_10_frame_bins"]
    assert hist["edges_frames"][:3] == [0, 10, 20] and hist["edges_frames"][-1] == 200
    assert hist["counts"][1] == 1 and hist["counts"][4] == 2 and hist["counts"][5] == 1
    assert sum(hist["counts"]) == 7
    with pytest.raises(ValueError):
        ef.summarize_frames([1, 2, 3], planned_n=2)
    empty = ef.summarize_frames([None, None], planned_n=2)
    assert empty["frames"] is None and empty["inside_first_50_frames"]["count"] == 0


def test_pairwise_agreement_is_descriptive_and_skips_missing_pairs():
    item = ef.pairwise_agreement([30, 40, None, 70], [28, 40, 10, 80])
    assert item["n_pairs"] == 3
    assert item["diff_frames"]["median"] == 0
    assert item["diff_frames"]["min"] == -10 and item["diff_frames"]["max"] == 2
    assert item["n_abs_diff_le_2_frames"] == 2
    assert item["n_abs_diff_le_5_frames_0p2s"] == 2
    assert item["diff_s"]["max_abs"] == pytest.approx(10 / 25)


# ---------------------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------------------

def test_run_analysis_derives_known_frames_and_writes_the_bundle(tmp_path):
    xs = _scan_grid()
    v = cal.REFERENCE_SPEED_MPS
    # Seed 4400 walks 5 % faster than the route, so A (root) precedes B (nominal); its foot
    # bump sits at frame 32 inside the region; seed 4401 never lifts; seed 4402 walks at the
    # nominal speed with a late lift (peak index 60 -> x = 3.63 m -> frame ~100).
    clips021 = [
        (_clip(1.05 * v, 16, 32), 16),
        (_clip(v, None, None), None),
        (_clip(v, 60, 102), 60),
        (_clip(0.95 * v, 16, 36), 16),
    ]
    clips023 = {
        "s4500_all_walk": (_clip(v, None, None, frames=208), None),
        "s4500_step_0": (_clip(v, 16, 34, frames=208), 33),
        "s4500_step_52": (_clip(v, None, None, frames=208), None),
        "s4500_step_104": (_clip(v, 10, 20, frames=208), None),
    }
    receipt = _run(tmp_path, clips021, clips023)
    out = tmp_path / "out"
    assert receipt["status"] == "complete" and receipt["blocked"] is False
    assert receipt["schema"] == ef.SCHEMA_VERSION
    assert receipt["identity"]["injected"] is True
    assert receipt["inputs"]["threshold_receipt"]["max_unsupported_run_s"] == 0.2
    assert set(receipt["definitions"]) >= {
        "A_root_crossing_frame", "B_nominal_frame", "C_foot_lift_frame", "C_nosupport_run",
        "forward_axis", "planned_denominators"}
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and receipt["n_rows"] == 8
    assert json.loads((out / "receipt.json").read_text())["status"] == "complete"

    first = rows[0]
    assert first["family"] == "exp021" and first["key"] == "s4400" and first["lifting"] is True
    assert first["archived_agreement"]["agrees"] is True
    events = first["events"]
    expected_a = int(np.ceil(xs[16] / (1.05 * v / 25.0)))
    assert events["A_root_crossing_frame"] == expected_a
    assert events["B_nominal_frame"] == pytest.approx(xs[16] / v * 25.0)
    assert events["A_minus_B_s"] == pytest.approx((expected_a - xs[16] / v * 25.0) / 25.0)
    assert events["C_foot_lift_frame"] == 32 and events["C_side"] == "left"
    assert events["C0_window_frame"] == 32
    assert events["C_inside_nosupport_run"] is True
    assert events["C_nosupport_run"]["start_frame"] == 30
    assert events["C_nosupport_run"]["duration_s"] == pytest.approx(5 / 25)
    assert events["C_region"]["lo_index"] == 15 and events["C_region"]["hi_index"] == 17
    assert first["no_support"]["first_gate_length_run_onset_frame"] == 30

    assert rows[1]["lifting"] is False and rows[1]["events"] is None
    assert rows[1]["max_clearance_any_foot"]["frame"] == 0
    assert rows[2]["events"]["C_foot_lift_frame"] == 102
    assert rows[2]["events"]["A_root_crossing_frame"] > 60

    e21 = receipt["summary"]["exp021"]
    assert e21["planned_clips"] == 4 and e21["lifting_clips_any_positive_profile"] == 3
    assert e21["non_lifting_clips"] == 1
    block = e21["lifting_all"]
    assert block["planned_denominator"] == 3
    assert block["A_root_crossing"]["n_present"] == 3
    assert block["A_root_crossing"]["after_frame_60"]["count"] == 1
    assert block["C_foot_lift"]["inside_first_50_frames"]["count"] == 2
    assert block["pairwise_agreement"]["A_minus_C"]["n_pairs"] == 3
    assert block["C_inside_bilateral_nosupport_run"]["count"] == 3
    assert e21["archived_row_agreement"] == {"n_agreeing": 4, "of": 4}
    assert e21["provisional_numbers_in_plan"]["computed_A_root_crossing"]["after_frame_60"] == "1/3"
    assert e21["non_lifting_max_clearance_frame"]["n_present"] == 1

    e23 = receipt["summary"]["exp023"]["by_arm"]
    assert e23["step_0"]["planned_clips"] == 1
    assert e23["step_0"]["archived_event"]["present"] == 1
    comparison = e23["step_0"]["archived_event"]["comparison_against_full_clip_definitions"]
    assert comparison["per_clip"][0]["archived_frame"] == 33
    assert comparison["per_clip"][0]["C_foot_lift_frame"] == 34
    assert comparison["per_clip"][0]["C_minus_archived_frames"] == 1
    assert comparison["n_side_agrees"] == 1
    assert e23["step_104"]["full_clip_lift_before_prompt_onset"]["count"] == 1
    assert e23["step_104"]["archived_event"]["present"] == 0
    assert e23["all_walk"]["archived_control_events"]["n_seeds_with_any_window_event"] == 0
    step0_row = next(row for row in rows if row["key"] == "s4500_step_0")
    assert step0_row["archived_frames"] == 208 and step0_row["scored_frames"] == 200


def test_run_analysis_blocks_when_the_archived_lift_does_not_reproduce(tmp_path):
    v = cal.REFERENCE_SPEED_MPS
    archive021, rows = _exp021_archive(tmp_path, [(_clip(v, 16, 32), 16)])
    rows[0]["lift_x_m"] = rows[0]["lift_x_m"] + 0.5
    (archive021 / "rows.jsonl").write_text(json.dumps(rows[0]) + "\n")
    archive023, _ = _exp023_archive(
        tmp_path, {"s4500_all_walk": (_clip(v, None, None, frames=208), None)})
    with pytest.raises(ValueError, match="do not reproduce their archived lift"):
        ef.run_analysis(tmp_path / "out", exp021_archive=archive021,
                        exp023_archive=archive023, locked=False,
                        profile_fn=_fake_profile, support_fn=_fake_support,
                        lift_side_fn=_fake_side, code_state_fn=_fake_git)
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["status"] == "blocked" and receipt["blocked"] is True
    assert receipt["mismatches"][0]["key"] == "s4400"
    assert receipt["mismatches"][0]["fields"]["lift_x_m"] is False


def test_locked_manifests_refuse_lookalike_archives(tmp_path):
    v = cal.REFERENCE_SPEED_MPS
    archive021, _ = _exp021_archive(tmp_path, [(_clip(v, 16, 32), 16)])
    archive023, _ = _exp023_archive(
        tmp_path, {"s4500_all_walk": (_clip(v, None, None, frames=208), None)})
    with pytest.raises(ValueError, match="locked EXP-021 artifact hash mismatch"):
        ef.run_analysis(tmp_path / "out", exp021_archive=archive021,
                        exp023_archive=archive023, locked=True,
                        profile_fn=_fake_profile, support_fn=_fake_support,
                        lift_side_fn=_fake_side, code_state_fn=_fake_git)
    with pytest.raises(ValueError, match="locked EXP-023 artifact hash mismatch"):
        ef._load_validated_exp023(archive023, ef.LOCKED_EXP023_MANIFEST)


def test_exp023_loader_refuses_content_and_schema_drift(tmp_path):
    v = cal.REFERENCE_SPEED_MPS
    archive, rows = _exp023_archive(
        tmp_path, {"s4500_all_walk": (_clip(v, None, None, frames=208), None),
                   "s4500_step_0": (_clip(v, 16, 34, frames=208), 33)})
    receipt, loaded_rows, arrays, hashes = ef._load_validated_exp023(archive, None)
    assert len(loaded_rows) == 2 and set(arrays) == {"s4500_all_walk", "s4500_step_0"}
    assert hashes == {}
    arrays["s4500_step_0"][5, 0] += 0.01
    np.savez(archive / "qpos.npz", **arrays)
    with pytest.raises(ValueError, match="qpos content hash mismatch"):
        ef._load_validated_exp023(archive, None)
    np.savez(archive / "qpos.npz", **{k: v_ for k, v_ in arrays.items()})
    receipt["evidence_anchors"]["qpos"]["content_sha256"] = exact.array_content_sha256(arrays)
    (archive / "receipt.json").write_text(json.dumps(receipt))
    rows[1]["onset_frame"] = 52
    (archive / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="onset disagrees"):
        ef._load_validated_exp023(archive, None)


def test_run_analysis_refuses_a_non_empty_output_directory(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("x")
    with pytest.raises(ValueError, match="refusing non-empty"):
        ef.run_analysis(out, locked=False, profile_fn=_fake_profile,
                        support_fn=_fake_support, lift_side_fn=_fake_side,
                        code_state_fn=_fake_git)


def test_locked_function_sources_match_the_imported_definitions():
    from experiments.analyze_trackability_contract import runs_of, support_masks
    assert ef.function_source_sha256(support_masks) == (
        ef.LOCKED_FUNCTION_SOURCES["analyze_trackability_contract.support_masks"])
    assert ef.function_source_sha256(runs_of) == (
        ef.LOCKED_FUNCTION_SOURCES["analyze_trackability_contract.runs_of"])
