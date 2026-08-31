import json

import numpy as np
import pytest
from types import SimpleNamespace

from experiments import exp016_semantic_geometric_stepover as exp016
from experiments.exp016_semantic_geometric_stepover import (
    _mcnemar,
    _resample_plan,
    aggregate,
)
from scene2motion.stepover_eval import KinematicStepEvent


def _row(prompt, scaffold, frame, seed, success, clearance):
    arm = f"{prompt}__{scaffold}"
    return {
        "arm": arm,
        "prompt": prompt,
        "scaffold": scaffold,
        "target_frame": frame,
        "seed": seed,
        "obstacle_collision_free": success,
        "kinematic_traversal_success": success,
        "local_step_structure": success,
        "kinematic_step_success": success,
        "max_box_height_lower_bound_m": clearance,
        "obstacle_min_clearance_m": clearance - 0.08,
        "progress_ratio": 1.0,
        "mean_support_feet": 1.0,
        "bilateral_flight_fraction": 0.0,
        "swing_foot_at_crossing_m": clearance,
        "phase_error_m": 0.0,
        "path_error_m": 0.0,
        "max_foot_floor_penetration_m": 0.0,
    }


def test_aggregate_pairs_seed_and_position_without_overwriting_cells():
    rows = []
    for frame in (56, 78):
        for seed in (1, 2):
            for prompt in ("walk", "step"):
                for mode in ("none", "leg_pos", "fullbody_pos", "fullbody_posrot"):
                    gain = (prompt == "step") + (mode != "none")
                    success = bool(prompt == "step" and mode == "fullbody_posrot")
                    rows.append(_row(prompt, mode, frame, seed, success,
                                     0.05 + 0.02 * gain))

    arms, contrasts = aggregate(rows)

    assert arms["step__fullbody_posrot"]["n_seed_position_rows"] == 4
    assert set(arms["step__fullbody_posrot"]["by_target_frame"]) == {"56", "78"}
    assert arms["step__fullbody_posrot"]["worst_position"][
        "kinematic_step_success"] == 1.0
    text = contrasts["text_effect__fullbody_posrot"]
    assert text["max_box_height_lower_bound_m"]["n_seed_position_pairs"] == 4
    assert text["max_box_height_lower_bound_m"]["n_seed_clusters"] == 2
    assert text["max_box_height_lower_bound_m"]["mean_difference"] == pytest.approx(0.02)
    for frame in (56, 78):
        table = text["kinematic_step_success"]["by_target_frame"][str(frame)]["mcnemar"]
        assert table["baseline_fail_new_pass"] == 2
        assert table["baseline_pass_new_fail"] == 0
    assert "coherence_effect__step__fullbody_minus_leg" in contrasts
    assert "rotation_effect__step__posrot_minus_pos" in contrasts
    assert "kinematic_step_success" in contrasts["interaction__fullbody_posrot"]


def test_exact_mcnemar_and_route_resampling():
    assert _mcnemar([False] * 4, [True] * 4) == {
        "baseline_fail_new_pass": 4,
        "baseline_pass_new_fail": 0,
        "discordant": 4,
        "exact_two_sided_p": 0.125,
    }
    plan = np.asarray([[0.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(_resample_plan(plan, 3), [[0, 0], [0, 1], [0, 2]])


def test_terminated_rollout_gets_the_route_prefix_not_the_compressed_route():
    # An early fall keeps valid_frames of a ref_len-frame rollout; its plan must be the
    # matching prefix of the full-rate route, not the whole route squeezed into the
    # surviving frames (review defect 5).
    plan = np.asarray([[0.0, 0.0], [0.0, 3.0]])
    full = _resample_plan(plan, 7)
    np.testing.assert_allclose(_resample_plan(plan, 7, 3), full[:3])
    np.testing.assert_allclose(_resample_plan(plan, 7, 3)[-1], [0.0, 1.0])
    np.testing.assert_allclose(_resample_plan(plan, 2, 2), plan)
    with pytest.raises(ValueError, match="valid_frames"):
        _resample_plan(plan, 4, 0)
    with pytest.raises(ValueError, match="valid_frames"):
        _resample_plan(plan, 4, 5)


def _run_cpu_smoke(monkeypatch, out, extra_argv=()):
    names = ["pelvis", "left_knee", "left_ankle", "left_toe",
             "right_knee", "right_ankle", "right_toe", "torso"]

    class FakeRunner:
        fps = 25.0
        model_name = "FAKE-G1"
        noise_stream_version = 2
        device = "cpu"
        joint_names = names
        skeleton = SimpleNamespace(root_idx=0)
        model = SimpleNamespace(gen_horizon_len=52)

        def __init__(self, **_kwargs):
            pass

        def generate(self, prompts, specs, num_frames, *_args, **_kwargs):
            assert len(prompts) == len(specs)
            T = num_frames
            root = np.stack([np.zeros(T), np.full(T, 0.78),
                             np.linspace(0, 7.2, T)], axis=-1)
            posed = np.repeat(root[:, None, :], len(names), axis=1)
            posed[:, :, 1] += np.linspace(0.0, 0.7, len(names))[None]
            rotations = np.broadcast_to(
                np.eye(3), (T, len(names), 3, 3)).copy()
            qpos = np.zeros((T, 36))
            qpos[:, 0] = root[:, 2]
            qpos[:, 2] = 0.78
            qpos[:, 3] = 1.0
            sample = {
                "smooth_root_pos": root,
                "posed_joints": posed,
                "global_rot_mats": rotations,
                "global_root_heading": np.tile([1.0, 0.0], (T, 1)),
                "qpos": qpos,
            }
            return [{k: np.array(v, copy=True) for k, v in sample.items()}
                    for _ in prompts]

        @staticmethod
        def to_qpos(sample):
            return sample["qpos"]

    class FakeBody:
        model = object()

        def __init__(self, *_args, **_kwargs):
            pass

    class FakeProbe:
        def __init__(self, _x, depth):
            self.depth = depth

        def body(self, _height):
            return object()

        def metadata(self):
            return {"quantity": "fake-lower-bound"}

    def fake_event(_body, _q, _fps, *_args, **_kwargs):
        return KinematicStepEvent(50, "left", 0.12, 0.12, 0.0, 1.0,
                                  0.1, 0.0, 47, 53)

    def fake_feet(_body, q, _fps):
        x = np.asarray(q)[:, 0] + 0.1
        values = {"forward_representative_m": x}
        return {"left": values, "right": values}

    gates = {name: True for name in (
        "whole_body_collision_free", "root_traversal", "lateral_corridor",
        *exp016.LOCAL_STRUCTURE_GATES)}

    def fake_metrics(*_args, **_kwargs):
        return {
            "obstacle_collision_free": True,
            "kinematic_traversal_success": True,
            "local_step_structure": True,
            "kinematic_step_success": True,
            "max_box_height_lower_bound_m": 0.10,
            "obstacle_min_clearance_m": 0.02,
            "progress_ratio": 1.0,
            "mean_support_feet": 1.0,
            "bilateral_flight_fraction": 0.0,
            "swing_foot_at_crossing_m": 0.10,
            "phase_error_m": 0.0,
            "path_error_m": 0.0,
            "max_foot_floor_penetration_m": 0.0,
            "local_step_success": True,
            "local_step": {"gates": dict(gates)},
        }

    monkeypatch.setattr(exp016, "ArdyRunner", FakeRunner)
    monkeypatch.setattr(exp016, "G1Body", FakeBody)
    monkeypatch.setattr(exp016, "BoxHeightProbe", FakeProbe)
    monkeypatch.setattr(exp016, "select_kinematic_step_event", fake_event)
    monkeypatch.setattr(exp016, "foot_kinematics_series", fake_feet)
    monkeypatch.setattr(exp016, "motion_metrics", fake_metrics)
    monkeypatch.setattr(exp016, "build_conditions", lambda *_args: (None, object()))
    monkeypatch.setattr(exp016, "channel_usage", lambda *_args: {"root_2d": 1})
    monkeypatch.setattr(
        exp016, "write_motion_pkl",
        lambda _clips, path, **_kwargs: path.write_bytes(b"fake"))
    monkeypatch.setattr(exp016, "_sha256", lambda _path: "fake-sha")
    monkeypatch.setattr(exp016, "_git_state", lambda _path: {"commit": "fake"})
    monkeypatch.setattr(
        exp016.sys, "argv",
        ["exp016", "--out", str(out), "--n_seeds", "1", "--n_donors", "1",
         "--target_frames", "56", "--skip_sonic", *extra_argv])

    exp016.main()

    return json.loads((out / "receipt.json").read_text())


def test_exp016_cpu_orchestration_smoke(monkeypatch, tmp_path):
    out = tmp_path / "exp016"
    receipt = _run_cpu_smoke(monkeypatch, out)
    assert receipt["status"] == "pilot_kinematics_complete_sonic_skipped"
    assert receipt["noise_stream_version"] == 2
    assert receipt["query_accounting"]["held_out_evaluation_ardy_samples"] == 8
    assert receipt["query_accounting"]["held_out_seed_position_rows"] == 8
    assert receipt["success_gate"]["threshold_source"] == "cli_pilot_defaults"
    assert receipt["success_gate"]["thresholds"]["max_unsupported_run_s"] == 0.08
    assert receipt["success_gate"]["thresholds"]["landing_dwell_s"] == 0.12
    assert len((out / "rows.jsonl").read_text().splitlines()) == 8


def test_exp016_consumes_a_calibration_receipt_in_the_tool_schema(
        monkeypatch, tmp_path):
    from scene2motion.stepover_eval import StepOverThresholds

    calibrated = StepOverThresholds(
        support_height_m=0.0465, support_speed_mps=1.175,
        max_unsupported_run_s=0.2, landing_dwell_s=0.12)
    calibration_path = tmp_path / "calibration_receipt.json"
    calibration_path.write_text(json.dumps(
        {"experiment": "stepover_threshold_calibration",
         "stepover_thresholds": exp016.asdict(calibrated)}))

    receipt = _run_cpu_smoke(
        monkeypatch, tmp_path / "exp016",
        extra_argv=["--threshold_calibration_receipt", str(calibration_path)])

    gate = receipt["success_gate"]
    assert gate["threshold_source"] == "calibration_receipt"
    assert gate["thresholds"] == exp016.asdict(calibrated)
    assert gate["threshold_calibration_receipt"] == str(calibration_path)


def test_exp016_rejects_a_frame_based_calibration_receipt(monkeypatch, tmp_path):
    stale = tmp_path / "stale_receipt.json"
    stale.write_text(json.dumps(
        {"stepover_thresholds": {"max_unsupported_run_frames": 2,
                                 "landing_dwell_frames": 3}}))
    monkeypatch.setattr(
        exp016.sys, "argv",
        ["exp016", "--out", str(tmp_path / "out"), "--skip_sonic",
         "--threshold_calibration_receipt", str(stale)])
    with pytest.raises(SystemExit, match="invalid threshold_calibration_receipt"):
        exp016.main()
