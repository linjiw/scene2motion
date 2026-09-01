import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments import exp017_ramp_residual_stepover as exp017
from scene2motion.ramp.step_phase import enumerate_step_phase_cycles
from scene2motion.stepover_eval import StepOverThresholds


NAMES = ["pelvis", "hip", "knee"]
PARENTS = np.array([-1, 0, 1], dtype=int)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.value, copy=True)


def _cycle(*, length=31, takeoff=6, apex=9, landing=13, side="left"):
    kinematics = {
        foot: {
            "bottom_clearance_m": np.zeros(length, dtype=float),
            "planar_speed_mps": np.zeros(length, dtype=float),
        }
        for foot in ("left", "right")
    }
    before = np.linspace(0.04, 0.16, apex - takeoff + 1)
    after = np.linspace(0.16, 0.04, landing - apex)[1:]
    heights = np.concatenate((before, after))
    kinematics[side]["bottom_clearance_m"][takeoff:landing] = heights
    kinematics[side]["planar_speed_mps"][takeoff:landing] = 1.0
    return enumerate_step_phase_cycles(
        kinematics,
        fps=10.0,
        swing_side=side,
        support_window_s=0.2,
        min_stance_support_fraction=0.9,
        min_relative_lift_m=0.04,
    )[0]


SOURCE_ADAPTED = _cycle(takeoff=6, apex=9, landing=13)
SOURCE_NEUTRAL = _cycle(takeoff=5, apex=9, landing=14)
TARGET_A = _cycle(takeoff=6, apex=9, landing=13)
TARGET_B = _cycle(takeoff=17, apex=20, landing=24)


def _compose(local):
    result = np.empty_like(local)
    result[:, 0] = local[:, 0]
    result[:, 1] = result[:, 0] @ local[:, 1]
    result[:, 2] = result[:, 1] @ local[:, 2]
    return result


def _sample(frames, *, role, seed):
    local = np.broadcast_to(np.eye(3), (frames, len(NAMES), 3, 3)).copy()
    height = 0.80
    marker = 3.0
    if role == "adapted":
        local[:, 1] = Rotation.from_euler("x", 25, degrees=True).as_matrix()
        height = 0.73
        marker = 1.0
    elif role == "neutral":
        marker = 2.0
    root = np.stack([
        np.zeros(frames),
        np.full(frames, height),
        np.linspace(0.0, 3.0, frames),
    ], axis=-1)
    qpos = np.zeros((frames, 36), dtype=float)
    qpos[:, 0] = np.linspace(0.0, 3.0, frames)
    qpos[:, 2] = height
    qpos[:, 3] = 1.0
    qpos[:, 34] = seed
    qpos[:, 35] = marker
    return {
        "smooth_root_pos": root,
        "global_rot_mats": _compose(local),
        "global_root_heading": np.tile([1.0, 0.0], (frames, 1)),
        "qpos": qpos,
        "test_seed": np.asarray([seed], dtype=np.int64),
    }


class FakeRunner:
    fps = 10.0
    model_name = "fake-g1"
    noise_stream_version = 2
    device = "cpu"
    joint_names = NAMES
    skeleton = SimpleNamespace(
        root_idx=0,
        joint_parents=FakeTensor(PARENTS),
    )
    model = object()

    def __init__(self, out):
        self.out = out
        self.calls = []

    def generate(self, prompts, specs, num_frames, diffusion_steps, *, cfg_weight, seeds):
        self.calls.append({
            "prompts": list(prompts),
            "specs": list(specs),
            "num_frames": num_frames,
            "diffusion_steps": diffusion_steps,
            "cfg_weight": tuple(cfg_weight),
            "seeds": list(seeds),
            "manifest_exists": (self.out / "manifest.json").exists(),
        })
        call = len(self.calls)
        samples = []
        for index, (prompt, seed) in enumerate(zip(prompts, seeds)):
            if call == 1:
                role = "adapted" if index % 2 == 0 else "neutral"
            else:
                role = "nominal"
            sample = _sample(num_frames, role=role, seed=seed)
            if call >= 3:
                sample["qpos"][:, 33] = index
                sample["test_arm_index"] = np.asarray([index], dtype=np.int64)
            samples.append(sample)
        return samples

    @staticmethod
    def to_qpos(sample):
        return np.asarray(sample["qpos"], dtype=float)


def _args(
    tmp_path, *, donors=2, seeds=2, candidates=None, obstacles=(1.0, 2.0)
):
    calibration = tmp_path / "thresholds.json"
    calibration.write_text(json.dumps({
        "experiment": "stepover_threshold_calibration",
        "status": "calibrated",
        "stepover_thresholds": vars(StepOverThresholds()),
        "calibration": {
            "n_clips": 10,
            "n_accepted": 9,
            "outlier_clip_indices": [9],
            "clips": [{"label": f"clip-{index}"} for index in range(10)],
        },
        "corpus": {
            "exp1c_dir": "outputs/exp1c_stepover",
            "arms": ["ctrl-l05"],
            "sources": "both",
            "reference_fps": 25.0,
            "selection": "test tracker-successful corpus",
            "reference": [{"path": "fake.npy", "sha256": "1" * 64}],
            "achieved": [{"archive": "fake.npz", "sha256": "2" * 64}],
            "excluded_achieved": [],
        },
        "provenance": {
            "code": {
                "commit": "calibration-commit",
                "dirty": False,
                "tracked_diff_sha256": "3" * 64,
            },
            "source_sha256": {
                "experiments/calibrate_stepover_thresholds.py": "4" * 64,
                "scene2motion/stepover_eval.py": "5" * 64,
                "scene2motion/sonic_state_export.py": "6" * 64,
            },
        },
    }))
    return Namespace(
        out=str(tmp_path / "exp017"),
        n_donors=donors,
        n_seeds=seeds,
        n_nominal_candidates=seeds if candidates is None else candidates,
        donor_seed_start=100,
        seed_start=200,
        duration=3.1,
        speed=1.0,
        obstacle_x=list(obstacles),
        obstacle_height=0.08,
        obstacle_depth=0.20,
        diffusion_steps=5,
        cfg_weight=(2.0, 2.0),
        half_window_frames=2,
        max_center_shift_frames=3,
        support_window_s=0.2,
        min_relative_lift_m=0.04,
        min_source_progress_ratio=0.75,
        min_nominal_progress_ratio=0.75,
        threshold_calibration_receipt=calibration,
    )


def _patch_cpu_dependencies(
    monkeypatch, *, fail_nominal=False, fail_source=False,
    ineligible_nominal_seeds=(),
):
    monkeypatch.setattr(exp017, "_checkpoint_identity", lambda _runner: {
        "generator_id": "fake-generator@revision",
        "model_name": "fake-g1",
        "hf_revision": "revision",
        "checkpoint_path": "/fake/denoiser.safetensors",
        "checkpoint_sha256": "c" * 64,
    })
    monkeypatch.setattr(exp017, "_git_state", lambda _root: {
        "commit": "test-commit", "dirty": False, "status": [],
        "tracked_diff_sha256": "d" * 64,
    })

    def phase_cycles(_body, qpos, _fps, _thresholds, **_kwargs):
        marker = int(round(float(np.asarray(qpos)[0, 35])))
        seed = int(round(float(np.asarray(qpos)[0, 34])))
        if fail_source and marker in (1, 2):
            raise ValueError("synthetic source has no supported phase cycle")
        if marker == 1:
            return (SOURCE_ADAPTED,)
        if marker == 2:
            return (SOURCE_NEUTRAL,)
        if fail_nominal or seed in set(ineligible_nominal_seeds):
            raise ValueError("synthetic nominal has no target cycle")
        return (TARGET_A, TARGET_B)

    monkeypatch.setattr(exp017, "_phase_cycles", phase_cycles)

    def feet(_body, qpos, _fps):
        forward = np.asarray(qpos)[:, 0] + 0.1
        values = {
            "forward_representative_m": forward,
            "bottom_clearance_m": np.zeros(len(forward), dtype=float),
            "planar_speed_mps": np.zeros(len(forward), dtype=float),
        }
        return {"left": values, "right": values}

    monkeypatch.setattr(exp017, "foot_kinematics_series", feet)
    monkeypatch.setattr(exp017, "_actual_channel_usage", lambda _runner, _spec: {
        "root_2d": 62,
        "root_y_pos": 62,
        "global_joints_rots": 15,
    })
    monkeypatch.setattr(exp017, "BoxHeightProbe", lambda x, depth: SimpleNamespace(
        x=x, depth=depth))

    def score(_body, _probe, qpos, _root_xz, obstacle_x, swing_side,
              _fps, _thresholds, _height):
        return {
            "obstacle_collision_free": True,
            "obstacle_min_clearance_m": 0.02,
            "max_box_height_lower_bound_m": 0.10,
            "progress_ratio": 1.0,
            "mean_support_feet": 1.0,
            "bilateral_flight_fraction": 0.0,
            "max_foot_floor_penetration_m": 0.0,
            "phase_error_m": 0.0,
            "selected_lead_side": swing_side,
            "kinematic_traversal_success": True,
            "kinematic_step_success": True,
        }

    monkeypatch.setattr(exp017, "_score", score)


def test_cpu_orchestration_freezes_manifest_and_spends_exact_budget(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path)
    runner = FakeRunner(tmp_path / "exp017")

    receipt = exp017.run_experiment(args, runner=runner, body=object())

    # K=N=2 here: 2D + K + 2NP = 4 + 2 + 8 = 14 samples.
    assert receipt["actual_ardy_samples"] == 14
    assert receipt["planned_ardy_samples"] == 14
    assert receipt["query_accounting"] == {
        "donor_source_samples_launched": 4,
        "donor_source_samples_returned": 4,
        "nominal_samples_launched": 2,
        "nominal_samples_returned": 2,
        "paired_evaluation_samples_launched": 8,
        "paired_evaluation_samples_returned": 8,
        "generate_invocations": 6,
    }
    donor_anchor = receipt["evidence_anchors"]["donor_qpos"]
    assert donor_anchor["n_candidates"] == 2
    assert donor_anchor["n_qpos_arrays"] == 4
    assert runner.calls[0]["prompts"] == [
        exp017.STEP, exp017.WALK, exp017.STEP, exp017.WALK]
    assert runner.calls[0]["seeds"] == [100, 100, 101, 101]
    assert runner.calls[1]["prompts"] == [exp017.WALK, exp017.WALK]
    assert runner.calls[1]["seeds"] == [200, 201]
    for call in runner.calls[2:]:
        assert call["prompts"] == [exp017.STEP, exp017.STEP]
        assert call["seeds"][0] == call["seeds"][1]
        assert call["manifest_exists"] is True

    out = tmp_path / "exp017"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["contains_final_outcomes"] is False
    assert manifest["execution_mode"] == "single-shot_non_resumable_pilot"
    assert manifest["resume_supported"] is False
    assert manifest["budget_formula"] == "2D+K+2NP"
    assert manifest["K"] == 2
    assert manifest["nominal_candidate_seeds"] == [200, 201]
    assert manifest["donor_seeds"] == [100, 101]
    assert manifest["evaluation_seeds"] == [200, 201]
    assert manifest["nominal_selection"]["policy"] == exp017.NOMINAL_SELECTOR
    assert manifest["nominal_selection"]["selected_evaluation_seeds"] == [200, 201]
    assert manifest["nominal_selection"]["n_eligible"] == 2
    assert manifest["nominal_selection"]["eligibility_fraction"] == 1.0
    assert manifest["nominal_selection"]["n_selected"] == 2
    assert manifest["nominal_selection"]["selected_fraction_of_eligible"] == 1.0
    assert manifest["nominal_selection"]["n_unselected_eligible"] == 0
    assert manifest["nominal_selection"]["planned_arm_denominator_np_per_arm"] == 4
    assert [scene["obstacle_x_m"] for scene in manifest["fixed_scenes"]] == [1.0, 2.0]
    assert len(manifest["programs"]) == 8
    assert len((out / "rows.jsonl").read_text().splitlines()) == 8
    experiment_identity = manifest["experiment_identity"]
    assert experiment_identity["sha256"] == exp017._json_hash({
        "schema": experiment_identity["schema"],
        "fields": experiment_identity["fields"],
    })
    assert "manifest" not in json.dumps(experiment_identity).lower()
    assert manifest["threshold_calibration"]["status"] == "calibrated"
    assert manifest["threshold_calibration"]["n_accepted"] == 9
    assert manifest["qpos_evidence"]["all_donor_candidates"] == donor_anchor
    nominal_anchor = receipt["evidence_anchors"]["nominal_qpos"]
    assert manifest["qpos_evidence"]["all_nominal_candidates"] == nominal_anchor
    assert nominal_anchor["n_attempted_seeds"] == 2
    assert nominal_anchor["n_qpos_arrays"] == 2
    assert (experiment_identity["fields"]["all_nominal_qpos_content_sha256"]
            == nominal_anchor["content_sha256"])
    assert donor_anchor["archive_sha256"] == exp017._sha256(out / "donor_qpos.npz")
    assert (out / "selected_source_qpos.npz").exists()
    assert (out / "nominal_qpos.npz").exists()
    with np.load(out / "nominal_qpos.npz") as stored:
        nominal_arrays = {key: np.array(stored[key], copy=True) for key in stored.files}
    assert nominal_anchor["content_sha256"] == exp017._array_hash(nominal_arrays)
    assert nominal_anchor["archive_sha256"] == exp017._sha256(
        out / "nominal_qpos.npz")
    substrate_anchor = receipt["evidence_anchors"]["nominal_substrate"]
    with np.load(out / "nominal_substrate.npz") as stored:
        substrate_arrays = {
            key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(substrate_arrays) == {
        "s200__global_rot_mats", "s200__smooth_root_pos",
        "s201__global_rot_mats", "s201__smooth_root_pos",
    }
    assert substrate_anchor["n_candidates"] == 2
    assert substrate_anchor["n_arrays"] == 4
    assert substrate_anchor["content_sha256"] == exp017._array_hash(
        substrate_arrays)
    assert substrate_anchor["archive_sha256"] == exp017._sha256(
        out / "nominal_substrate.npz")
    assert manifest["qpos_evidence"]["all_nominal_substrates"] == substrate_anchor
    assert (experiment_identity["fields"]
            ["all_nominal_substrate_content_sha256"]
            == substrate_anchor["content_sha256"])
    nominal_rows = [json.loads(line) for line in
                    (out / "nominal_rows.jsonl").read_text().splitlines()]
    assert len(nominal_rows) == 2
    assert all(row["processing_status"] == "selected_programs_rendered"
               for row in nominal_rows)
    assert all(row["assignment_diagnostics"]["assignment_status"] == "assigned"
               for row in nominal_rows)
    for row in nominal_rows:
        identity = row["candidate_eligibility_identity"]
        assert identity["sha256"] == exp017._json_hash({
            "schema": identity["schema"], "fields": identity["fields"]})
        assert row["candidate_eligibility_identity_sha256"] == identity["sha256"]
        for field in ("global_rot_mats", "smooth_root_pos"):
            key = row[f"{field}_archive_key"]
            assert row[f"{field}_content_sha256"] == exp017._array_hash(
                {key: substrate_arrays[key]})
        assert (row["nominal_substrate_archive_content_sha256"]
                == substrate_anchor["content_sha256"])
    nominal_rows_anchor = receipt["evidence_anchors"]["nominal_rows"]
    assert manifest["evidence_anchors"]["nominal_rows"] == nominal_rows_anchor
    assert nominal_rows_anchor["n_rows"] == len(nominal_rows)
    assert nominal_rows_anchor["logical_sha256"] == exp017._json_hash(nominal_rows)
    assert nominal_rows_anchor["file_sha256"] == exp017._sha256(
        out / "nominal_rows.jsonl")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["inference_unit"].startswith("descriptive only")
    assert summary["n_fixed_placements"] == 2
    assert "No confidence interval or p-value" in summary["interpretation"]
    for metric in summary["metrics"].values():
        assert "placement_effects_after_collapsing_seeds" in metric
        assert "scene_cluster_bootstrap_95" not in metric

    grouped = {}
    for program in manifest["programs"]:
        grouped.setdefault((program["seed"], program["scene_id"]), []).append(program)
    for pair in grouped.values():
        assert {row["arm"] for row in pair} == {"absolute", "residual"}
        assert len({row["support_hash"] for row in pair}) == 1
        assert len({row["target_phase_match_hash"] for row in pair}) == 1
        assert len({json.dumps(row["channel_usage"], sort_keys=True) for row in pair}) == 1
        assert len({row["prompt"] for row in pair}) == 1
        assert all("deformation_vs_nominal" in row for row in pair)
        for row in pair:
            identity = row["program_identity"]
            assert row["experiment_identity_sha256"] == experiment_identity["sha256"]
            assert row["program_identity_sha256"] == identity["sha256"]
            assert identity["sha256"] == exp017._json_hash({
                "schema": identity["schema"], "fields": identity["fields"]})
            assert "manifest" not in json.dumps(identity).lower()

    final_rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    for row in final_rows:
        identity = row["output_identity"]
        assert row["experiment_identity_sha256"] == experiment_identity["sha256"]
        assert row["output_identity_sha256"] == identity["sha256"]
        assert identity["sha256"] == exp017._json_hash({
            "schema": identity["schema"], "fields": identity["fields"]})
        assert "manifest" not in json.dumps(identity).lower()
        metric_payload = {
            key: row[key]
            for key in exp017._score(
                object(), object(), np.zeros((1, 36)), np.zeros((1, 2)),
                0.0, "left", 10.0, StepOverThresholds(), 0.08,
            )
        }
        assert row["metrics_sha256"] == exp017._json_hash(metric_payload)

    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "exp017-success-receipt-v4"
    assert receipt["planned_arm_denominator_np_per_arm"] == 4
    assert receipt["terminal_counts_per_arm"] == {"absolute": 4, "residual": 4}
    assert receipt["status_counts_per_arm"] == {
        "absolute": {"completed": 4}, "residual": {"completed": 4}}
    assert receipt["actual_attempted_seeds"] == [200, 201]
    assert receipt["actual_evaluated_seeds"] == [200, 201]
    assert receipt["actual_attempted_arm_counts_per_arm"] == {
        "absolute": 4, "residual": 4}
    assert receipt["actual_evaluated_arm_counts_per_arm"] == {
        "absolute": 4, "residual": 4}
    assert receipt["rows_sha256"] == exp017._sha256(out / "rows.jsonl")
    assert receipt["qpos_archive_sha256"] == exp017._sha256(out / "qpos.npz")
    with np.load(out / "qpos.npz") as archive:
        qpos_arrays = {name: archive[name] for name in archive.files}
    assert receipt["qpos_content_sha256"] == exp017._array_hash(qpos_arrays)
    assert receipt["output_identity_set_sha256"] == exp017._json_hash([
        row["output_identity_sha256"] for row in final_rows])
    attempt_plan = json.loads((out / "attempt_plan.json").read_text())
    assert attempt_plan["sha256"] == exp017._json_hash({
        "schema": attempt_plan["schema"], "fields": attempt_plan["fields"]})
    assert manifest["attempt_plan"] == attempt_plan
    attempts = [json.loads(line) for line in
                (out / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 8
    assert all(row["status"] == "completed" for row in attempts)
    assert receipt["evidence_anchors"]["attempts"]["logical_sha256"] == (
        exp017._json_hash(attempts))


def test_overlap_fails_before_any_generation_and_writes_receipt(monkeypatch, tmp_path):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path)
    args.seed_start = 101
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="must be disjoint"):
        exp017.run_experiment(args, runner=runner, body=object())

    assert runner.calls == []
    receipt = json.loads((tmp_path / "exp017" / "receipt.json").read_text())
    assert receipt["status"] == "failed_closed"
    assert receipt["failed_stage"] == "validation"
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 0


def test_frozen_k_pool_selects_first_eligible_and_spends_2d_k_2np(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(
        monkeypatch, ineligible_nominal_seeds={200, 202})
    args = _args(
        tmp_path, donors=4, candidates=8, seeds=2, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    receipt = exp017.run_experiment(args, runner=runner, body=object())

    # 2D + K + 2NP = 8 + 8 + 4 = 20; one donor batch, one K batch,
    # then one paired call per selected seed and fixed scene.
    assert receipt["actual_ardy_samples"] == 20
    assert receipt["planned_ardy_samples"] == 20
    assert receipt["budget_formula"] == "2D+K+2NP"
    assert (receipt["D"], receipt["K"], receipt["N"], receipt["P"]) == (
        4, 8, 2, 1)
    assert receipt["query_accounting"] == {
        "donor_source_samples_launched": 8,
        "donor_source_samples_returned": 8,
        "nominal_samples_launched": 8,
        "nominal_samples_returned": 8,
        "paired_evaluation_samples_launched": 4,
        "paired_evaluation_samples_returned": 4,
        "generate_invocations": 4,
    }
    assert runner.calls[1]["seeds"] == list(range(200, 208))
    assert [call["seeds"] for call in runner.calls[2:]] == [[201, 201], [203, 203]]
    assert all(call["manifest_exists"] for call in runner.calls[2:])

    out = tmp_path / "exp017"
    selection = json.loads((out / "nominal_selection.json").read_text())
    assert selection["status"] == "selected"
    assert selection["policy"] == exp017.NOMINAL_SELECTOR
    assert selection["nominal_candidate_seeds"] == list(range(200, 208))
    assert selection["eligible_seeds_in_pool_order"] == [201, 203, 204, 205, 206, 207]
    assert selection["selected_evaluation_seeds"] == [201, 203]
    assert selection["attrition_counts"] == {"phase_cycle": 2, "eligible": 6}
    assert selection["n_eligible"] == 6
    assert selection["eligibility_fraction"] == 0.75
    assert selection["n_selected"] == 2
    assert selection["selected_fraction_of_eligible"] == pytest.approx(1 / 3)
    assert selection["n_unselected_eligible"] == 4
    assert selection["planned_arm_denominator_np_per_arm"] == 2
    selection_identity = selection["selection_identity"]
    assert selection["selection_identity_sha256"] == selection_identity["sha256"]
    assert selection_identity["sha256"] == exp017._json_hash({
        "schema": selection_identity["schema"],
        "fields": selection_identity["fields"],
    })
    for field in (
        "n_eligible", "eligibility_fraction", "n_selected",
        "selected_fraction_of_eligible", "n_unselected_eligible",
        "planned_arm_denominator_np_per_arm",
    ):
        assert selection_identity["fields"][field] == selection[field]

    nominal_rows = [json.loads(line) for line in
                    (out / "nominal_rows.jsonl").read_text().splitlines()]
    assert len(nominal_rows) == 8
    assert [row["seed"] for row in nominal_rows if row["selected"]] == [201, 203]
    assert [row["selection_rank"] for row in nominal_rows if row["selected"]] == [0, 1]
    assert {row["seed"]: row["attrition_stage"] for row in nominal_rows
            if row["eligibility_status"] == "ineligible"} == {
                200: "phase_cycle", 202: "phase_cycle"}
    candidate_cores = [
        row["candidate_eligibility_identity"]["fields"] for row in nominal_rows]
    assert selection["candidate_core_sha256"] == exp017._json_hash(candidate_cores)
    assert selection["candidate_eligibility_identity_sha256"] == [
        row["candidate_eligibility_identity_sha256"] for row in nominal_rows]

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["nominal_candidate_seeds"] == list(range(200, 208))
    assert manifest["evaluation_seeds"] == [201, 203]
    assert manifest["nominal_selection"] == selection
    fields = manifest["experiment_identity"]["fields"]
    assert fields["nominal_candidate_seeds"] == list(range(200, 208))
    assert fields["evaluation_seeds"] == [201, 203]
    assert fields["nominal_candidate_core_sha256"] == selection[
        "candidate_core_sha256"]
    assert fields["nominal_selection_identity_sha256"] == selection[
        "selection_identity_sha256"]
    assert fields["nominal_selection_logical_sha256"] == exp017._json_hash(selection)
    assert fields["nominal_attrition_counts"] == selection["attrition_counts"]
    assert fields["n_eligible"] == 6
    assert fields["eligibility_fraction"] == 0.75
    assert fields["n_selected"] == 2
    assert fields["selected_fraction_of_eligible"] == pytest.approx(1 / 3)
    assert fields["n_unselected_eligible"] == 4
    assert fields["planned_arm_denominator_np_per_arm"] == 2
    selection_anchor = receipt["evidence_anchors"]["nominal_selection"]
    assert selection_anchor["logical_sha256"] == exp017._json_hash(selection)
    assert selection_anchor["file_sha256"] == exp017._sha256(
        out / "nominal_selection.json")


def test_k_smaller_than_n_aborts_before_generation(monkeypatch, tmp_path):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, candidates=1, seeds=2, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="K must be at least"):
        exp017.run_experiment(args, runner=runner, body=object())

    assert runner.calls == []
    receipt = json.loads((tmp_path / "exp017" / "receipt.json").read_text())
    assert receipt["failed_stage"] == "validation"
    assert receipt["query_accounting"]["nominal_samples_returned"] == 0


def test_pool_exhaustion_does_not_partially_select_or_imply_evaluation(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch, ineligible_nominal_seeds={201})
    args = _args(
        tmp_path, donors=1, candidates=2, seeds=2, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="has 1 eligible seeds"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    selection = json.loads((out / "nominal_selection.json").read_text())
    assert selection["status"] == "insufficient_eligible_candidates"
    assert selection["eligible_seeds_in_pool_order"] == [200]
    assert selection["eligible_prefix_seeds"] == [200]
    assert selection["selected_evaluation_seeds"] == []
    assert selection["n_eligible"] == 1
    assert selection["n_selected"] == 0
    assert selection["selected_fraction_of_eligible"] == 0.0
    assert selection["n_unselected_eligible"] == 1
    rows = [json.loads(line) for line in
            (out / "nominal_rows.jsonl").read_text().splitlines()]
    assert not any(row["selected"] for row in rows)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["actual_attempted_seeds"] == []
    assert receipt["actual_evaluated_seeds"] == []
    assert receipt["actual_attempted_arm_counts_per_arm"] == {
        "absolute": 0, "residual": 0}
    assert receipt["actual_evaluated_arm_counts_per_arm"] == {
        "absolute": 0, "residual": 0}
    assert not (out / "manifest.json").exists()


def test_late_nominal_conversion_failure_preserves_incremental_candidate_evidence(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(
        tmp_path, donors=1, candidates=3, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")
    original_to_qpos = runner.to_qpos

    def fail_last_nominal(sample):
        seed = int(np.asarray(sample["test_seed"])[0])
        if seed == 202 and "test_arm_index" not in sample:
            raise ValueError("synthetic late-K nominal conversion failure")
        return original_to_qpos(sample)

    runner.to_qpos = fail_last_nominal
    with pytest.raises(exp017.ExperimentAbort,
                       match="late-K nominal conversion failure"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["query_accounting"]["nominal_samples_returned"] == 3
    with np.load(out / "nominal_qpos.npz") as stored:
        qpos_arrays = {key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(qpos_arrays) == {"s200", "s201"}
    with np.load(out / "nominal_substrate.npz") as stored:
        substrate_arrays = {
            key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(substrate_arrays) == {
        "s200__global_rot_mats", "s200__smooth_root_pos",
        "s201__global_rot_mats", "s201__smooth_root_pos",
    }
    qpos_anchor = receipt["evidence_anchors"]["nominal_qpos"]
    substrate_anchor = receipt["evidence_anchors"]["nominal_substrate"]
    assert qpos_anchor["n_predeclared_candidates"] == 3
    assert qpos_anchor["n_attempted_seeds"] == 3
    assert qpos_anchor["n_qpos_arrays"] == 2
    assert qpos_anchor["content_sha256"] == exp017._array_hash(qpos_arrays)
    assert qpos_anchor["archive_sha256"] == exp017._sha256(
        out / "nominal_qpos.npz")
    assert substrate_anchor["n_complete_candidates"] == 2
    assert substrate_anchor["content_sha256"] == exp017._array_hash(
        substrate_arrays)
    assert substrate_anchor["archive_sha256"] == exp017._sha256(
        out / "nominal_substrate.npz")
    rows = [json.loads(line) for line in
            (out / "nominal_rows.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[-1]["processing_status"] == "rejected_nominal_evidence"
    assert rows[-1]["attrition_stage"] == "nominal_evidence"
    assert "late-K nominal conversion failure" in rows[-1]["assignment_reason"]
    rows_anchor = receipt["evidence_anchors"]["nominal_rows"]
    assert rows_anchor["logical_sha256"] == exp017._json_hash(rows)
    assert rows_anchor["file_sha256"] == exp017._sha256(
        out / "nominal_rows.jsonl")


def test_assignment_shift_equal_to_locked_bound_remains_eligible(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(
        tmp_path, donors=1, candidates=1, seeds=1, obstacles=(1.1,))
    args.max_center_shift_frames = 1
    runner = FakeRunner(tmp_path / "exp017")

    exp017.run_experiment(args, runner=runner, body=object())

    nominal_row = json.loads(
        (tmp_path / "exp017" / "nominal_rows.jsonl").read_text())
    assert nominal_row["eligibility_status"] == "eligible"
    assert nominal_row["selected"] is True
    assert nominal_row["assignments"][0]["controls"]["center_shift_frames"] == 1
    selected_check = nominal_row["assignment_diagnostics"]["selected"][0]
    assert selected_check["center_shift_frames"] == 1


def test_selected_program_failure_does_not_replace_with_later_eligible_seed(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(
        tmp_path, donors=1, candidates=3, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    def fail_render(*_args, **_kwargs):
        raise ValueError("synthetic selected render failure")

    monkeypatch.setattr(exp017, "render_packet", fail_render)
    with pytest.raises(exp017.ExperimentAbort,
                       match="synthetic selected render failure"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    assert not (out / "manifest.json").exists()
    assert len(runner.calls) == 2
    selection = json.loads((out / "nominal_selection.json").read_text())
    assert selection["selected_evaluation_seeds"] == [200]
    rows = [json.loads(line) for line in
            (out / "nominal_rows.jsonl").read_text().splitlines()]
    assert rows[0]["processing_status"] == "selected_program_failure"
    assert rows[0]["selected"] is True
    assert all(row["processing_status"] == "eligible_not_selected"
               and row["selected"] is False for row in rows[1:])
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["query_accounting"]["nominal_samples_returned"] == 3
    assert receipt["query_accounting"]["paired_evaluation_samples_returned"] == 0


def test_post_manifest_generation_failure_preserves_full_attempt_denominator(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(
        tmp_path, donors=1, candidates=3, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")
    original_generate = runner.generate

    def fail_final_generation(*call_args, **call_kwargs):
        outputs = original_generate(*call_args, **call_kwargs)
        if len(runner.calls) >= 3:
            raise RuntimeError("synthetic paired generation failure")
        return outputs

    runner.generate = fail_final_generation
    with pytest.raises(exp017.ExperimentAbort,
                       match="synthetic paired generation failure"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    receipt = json.loads((out / "receipt.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert receipt["schema"] == "exp017-failure-receipt-v4"
    assert receipt["manifest_sha256"] == exp017._sha256(out / "manifest.json")
    assert receipt["evidence_anchors"]["manifest"]["sha256"] == receipt[
        "manifest_sha256"]
    assert manifest["evaluation_seeds"] == [200]
    assert manifest["nominal_selection"]["n_unselected_eligible"] == 2
    assert receipt["planned_arm_denominator_np_per_arm"] == 1
    assert receipt["actual_ardy_samples"] is None
    assert receipt["sample_count_exact"] is False
    assert receipt["returned_ardy_samples_lower_bound"] == 5
    assert receipt["conservative_charged_ardy_samples"] == 7
    assert receipt["query_accounting"]["paired_evaluation_samples_launched"] == 2
    assert receipt["query_accounting"]["paired_evaluation_samples_returned"] == 0
    assert receipt["status_counts_per_arm"] == {
        "absolute": {"generation_failed": 1},
        "residual": {"generation_failed": 1},
    }
    assert receipt["terminal_counts_per_arm"] == {"absolute": 1, "residual": 1}
    assert receipt["actual_attempted_seeds"] == [200]
    assert receipt["actual_evaluated_seeds"] == []
    assert receipt["actual_attempted_arm_counts_per_arm"] == {
        "absolute": 1, "residual": 1}
    assert receipt["actual_evaluated_arm_counts_per_arm"] == {
        "absolute": 0, "residual": 0}

    attempts = [json.loads(line) for line in
                (out / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 2
    assert all(row["status"] == "generation_failed" for row in attempts)
    attempts_anchor = receipt["evidence_anchors"]["attempts"]
    assert attempts_anchor["logical_sha256"] == exp017._json_hash(attempts)
    assert attempts_anchor["file_sha256"] == exp017._sha256(out / "attempts.jsonl")
    assert attempts_anchor["attempt_plan_identity_sha256"] == manifest[
        "attempt_plan"]["sha256"]
    assert (out / "rows.jsonl").read_text() == ""
    with np.load(out / "qpos.npz") as stored:
        assert stored.files == []
    assert receipt["evidence_anchors"]["paired_rows"]["n_rows"] == 0
    assert receipt["evidence_anchors"]["paired_qpos"]["n_arrays"] == 0


@pytest.mark.parametrize(
    "failure_kind,failed_status,expected_qpos_arrays",
    [("conversion", "conversion_failed", 1),
     ("scoring", "scoring_failed", 2)],
)
def test_one_arm_failure_keeps_other_arm_evidence_and_never_substitutes(
    monkeypatch, tmp_path, failure_kind, failed_status, expected_qpos_arrays
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(
        tmp_path, donors=1, candidates=3, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")
    if failure_kind == "conversion":
        original_to_qpos = runner.to_qpos

        def fail_absolute_conversion(sample):
            if int(np.asarray(sample.get("test_arm_index", [-1]))[0]) == 0:
                raise ValueError("synthetic absolute conversion failure")
            return original_to_qpos(sample)

        runner.to_qpos = fail_absolute_conversion
    else:
        original_score = exp017._score

        def fail_absolute_score(body, probe, qpos, *score_args, **score_kwargs):
            if int(round(float(np.asarray(qpos)[0, 33]))) == 0:
                raise ValueError("synthetic absolute scoring failure")
            return original_score(body, probe, qpos, *score_args, **score_kwargs)

        monkeypatch.setattr(exp017, "_score", fail_absolute_score)

    with pytest.raises(exp017.ExperimentAbort, match=failure_kind):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    receipt = json.loads((out / "receipt.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["evaluation_seeds"] == [200]
    assert manifest["nominal_selection"]["eligible_seeds_in_pool_order"] == [
        200, 201, 202]
    assert receipt["planned_arm_denominator_np_per_arm"] == 1
    assert receipt["actual_ardy_samples"] == 7  # 2D + K + both returned arms.
    assert receipt["sample_count_exact"] is True
    assert receipt["returned_ardy_samples_lower_bound"] == 7
    assert receipt["conservative_charged_ardy_samples"] == 7
    assert receipt["query_accounting"]["paired_evaluation_samples_launched"] == 2
    assert receipt["query_accounting"]["paired_evaluation_samples_returned"] == 2
    assert receipt["status_counts_per_arm"] == {
        "absolute": {failed_status: 1}, "residual": {"completed": 1}}
    assert receipt["terminal_counts_per_arm"] == {"absolute": 1, "residual": 1}
    assert receipt["actual_attempted_seeds"] == [200]
    assert receipt["actual_evaluated_seeds"] == []
    assert receipt["actual_attempted_arm_counts_per_arm"] == {
        "absolute": 1, "residual": 1}
    assert receipt["actual_evaluated_arm_counts_per_arm"] == {
        "absolute": 0, "residual": 1}

    attempts = [json.loads(line) for line in
                (out / "attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == [failed_status, "completed"]
    assert "sample_sha256" in attempts[0]
    assert receipt["evidence_anchors"]["attempts"]["logical_sha256"] == (
        exp017._json_hash(attempts))
    rows = [json.loads(line) for line in
            (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["arm"] == "residual"
    assert receipt["evidence_anchors"]["paired_rows"]["logical_sha256"] == (
        exp017._json_hash(rows))
    with np.load(out / "qpos.npz") as stored:
        assert len(stored.files) == expected_qpos_arrays
    assert receipt["evidence_anchors"]["paired_qpos"]["n_arrays"] == (
        expected_qpos_arrays)
    # Candidate 200 remains the sole selected seed despite two later eligible seeds.
    selection = json.loads((out / "nominal_selection.json").read_text())
    assert selection["selected_evaluation_seeds"] == [200]


def test_dirty_worktree_fails_before_source_generation(monkeypatch, tmp_path):
    _patch_cpu_dependencies(monkeypatch)
    monkeypatch.setattr(exp017, "_git_state", lambda _root: {
        "commit": "test-commit", "dirty": True,
        "status": ["?? experiments/exp017_ramp_residual_stepover.py"],
        "tracked_diff_sha256": "d" * 64,
    })
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="exactly clean git worktree"):
        exp017.run_experiment(args, runner=runner, body=object())

    assert runner.calls == []
    receipt = json.loads((tmp_path / "exp017" / "receipt.json").read_text())
    assert receipt["failed_stage"] == "validation"
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 0


def test_noise_stream_v1_fails_before_source_generation(monkeypatch, tmp_path):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")
    runner.noise_stream_version = 1

    with pytest.raises(exp017.ExperimentAbort, match="noise_stream_version == 2"):
        exp017.run_experiment(args, runner=runner, body=object())

    assert runner.calls == []
    receipt = json.loads((tmp_path / "exp017" / "receipt.json").read_text())
    assert receipt["failed_stage"] == "validation"
    assert receipt["resume_supported"] is False


def test_source_generation_exception_reports_launched_charge_and_uncertainty(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, candidates=2, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")
    original_generate = runner.generate

    def fail_source(*call_args, **call_kwargs):
        original_generate(*call_args, **call_kwargs)
        raise RuntimeError("synthetic source generation exception")

    runner.generate = fail_source
    with pytest.raises(exp017.ExperimentAbort,
                       match="source generation exception"):
        exp017.run_experiment(args, runner=runner, body=object())

    receipt = json.loads((tmp_path / "exp017" / "receipt.json").read_text())
    assert receipt["schema"] == "exp017-failure-receipt-v4"
    assert receipt["failed_stage"] == "source_discovery"
    assert receipt["sample_count_exact"] is False
    assert receipt["actual_ardy_samples"] is None
    assert receipt["returned_ardy_samples_lower_bound"] == 0
    assert receipt["conservative_charged_ardy_samples"] == 2
    assert receipt["query_accounting"] == {
        "donor_source_samples_launched": 2,
        "donor_source_samples_returned": 0,
        "nominal_samples_launched": 0,
        "nominal_samples_returned": 0,
        "paired_evaluation_samples_launched": 0,
        "paired_evaluation_samples_returned": 0,
        "generate_invocations": 1,
    }


def test_nominal_generation_exception_preserves_source_and_charges_k(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, candidates=3, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")
    original_generate = runner.generate

    def fail_nominal(*call_args, **call_kwargs):
        outputs = original_generate(*call_args, **call_kwargs)
        if len(runner.calls) == 2:
            raise RuntimeError("synthetic nominal generation exception")
        return outputs

    runner.generate = fail_nominal
    with pytest.raises(exp017.ExperimentAbort,
                       match="nominal generation exception"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["failed_stage"] == "nominal_discovery"
    assert receipt["sample_count_exact"] is False
    assert receipt["actual_ardy_samples"] is None
    assert receipt["returned_ardy_samples_lower_bound"] == 2
    assert receipt["conservative_charged_ardy_samples"] == 5
    assert receipt["query_accounting"] == {
        "donor_source_samples_launched": 2,
        "donor_source_samples_returned": 2,
        "nominal_samples_launched": 3,
        "nominal_samples_returned": 0,
        "paired_evaluation_samples_launched": 0,
        "paired_evaluation_samples_returned": 0,
        "generate_invocations": 2,
    }
    donor_anchor = receipt["evidence_anchors"]["donor_qpos"]
    assert donor_anchor["archive_sha256"] == exp017._sha256(out / "donor_qpos.npz")
    with np.load(out / "donor_qpos.npz") as stored:
        donor_arrays = {key: np.array(stored[key], copy=True) for key in stored.files}
    assert donor_anchor["content_sha256"] == exp017._array_hash(donor_arrays)
    assert not (out / "nominal_qpos.npz").exists()


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda value: value.update(experiment="not-the-calibrator"),
         "experiment identity"),
        (lambda value: value.update(status="draft"), "status"),
        (lambda value: value["calibration"].update(n_accepted=0), "nonzero"),
        (lambda value: value["corpus"].pop("achieved"), "achieved"),
        (lambda value: value["provenance"]["source_sha256"].pop(
            "scene2motion/stepover_eval.py"), "source hashes"),
    ],
)
def test_threshold_receipt_schema_fails_before_runner_use(
    monkeypatch, tmp_path, mutate, message
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    value = json.loads(args.threshold_calibration_receipt.read_text())
    mutate(value)
    args.threshold_calibration_receipt.write_text(json.dumps(value))
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match=message):
        exp017.run_experiment(args, runner=runner, body=object())

    assert runner.calls == []
    receipt = json.loads((tmp_path / "exp017" / "receipt.json").read_text())
    assert receipt["failed_stage"] == "validation"


def test_missing_target_cycle_aborts_before_manifest_or_paired_budget(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch, fail_nominal=True)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="0 eligible seeds"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    assert not (out / "manifest.json").exists()
    assert len(runner.calls) == 2
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["failed_stage"] == "nominal_discovery"
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 2
    assert receipt["query_accounting"]["nominal_samples_returned"] == 1
    assert receipt["query_accounting"]["paired_evaluation_samples_returned"] == 0
    archive = out / "nominal_qpos.npz"
    assert archive.exists()
    with np.load(archive) as stored:
        arrays = {key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(arrays) == {"s200"}
    anchor = receipt["evidence_anchors"]["nominal_qpos"]
    assert anchor["content_sha256"] == exp017._array_hash(arrays)
    assert anchor["archive_sha256"] == exp017._sha256(archive)
    substrate_anchor = receipt["evidence_anchors"]["nominal_substrate"]
    with np.load(out / "nominal_substrate.npz") as stored:
        substrate_arrays = {
            key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(substrate_arrays) == {
        "s200__global_rot_mats", "s200__smooth_root_pos"}
    assert substrate_anchor["content_sha256"] == exp017._array_hash(
        substrate_arrays)
    assert substrate_anchor["archive_sha256"] == exp017._sha256(
        out / "nominal_substrate.npz")
    nominal_row = json.loads((out / "nominal_rows.jsonl").read_text())
    assert nominal_row["processing_status"] == "rejected_phase_cycle"
    assert nominal_row["clip_sha256"]
    assert nominal_row["qpos_content_sha256"] == exp017._array_hash(
        {"s200": arrays["s200"]})
    assert nominal_row["final_progress_ratio_vs_prescribed_route"] == pytest.approx(
        3 / 3.1)
    assert nominal_row["support_diagnostics"]["interpretation"].startswith(
        "descriptive only")
    assert "no target cycle" in nominal_row["phase_cycle_error"]
    assert "no target cycle" in nominal_row["assignment_reason"]
    assert (receipt["experiment_identity"]["fields"]
            ["all_nominal_qpos_content_sha256"] == anchor["content_sha256"])
    assert receipt["experiment_identity_sha256"] == receipt[
        "evidence_anchors"]["experiment_identity"]["sha256"]
    row_anchor = receipt["evidence_anchors"]["nominal_rows"]
    assert row_anchor["n_rows"] == 1
    assert row_anchor["logical_sha256"] == exp017._json_hash([nominal_row])
    assert row_anchor["file_sha256"] == exp017._sha256(
        out / "nominal_rows.jsonl")


def test_bounded_assignment_failure_preserves_all_nominals_and_rejection_details(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, seeds=2, obstacles=(1.5,))
    args.max_center_shift_frames = 0
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="0 eligible seeds"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    assert not (out / "manifest.json").exists()
    assert len(runner.calls) == 2
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["failed_stage"] == "nominal_discovery"
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 2
    assert receipt["query_accounting"]["nominal_samples_returned"] == 2
    assert receipt["query_accounting"]["paired_evaluation_samples_returned"] == 0
    assert sum(receipt["query_accounting"][key] for key in (
        "donor_source_samples_returned", "nominal_samples_returned",
        "paired_evaluation_samples_returned")) == 4
    selection = json.loads((out / "nominal_selection.json").read_text())
    assert selection["status"] == "insufficient_eligible_candidates"
    assert selection["nominal_candidate_seeds"] == [200, 201]
    assert selection["eligible_seeds_in_pool_order"] == []
    assert selection["selected_evaluation_seeds"] == []
    assert selection["attrition_counts"] == {"target_assignment": 2}
    assert selection["n_eligible"] == 0
    assert selection["eligibility_fraction"] == 0.0
    assert selection["n_selected"] == 0
    assert selection["selected_fraction_of_eligible"] is None
    assert selection["n_unselected_eligible"] == 0
    assert selection["planned_arm_denominator_np_per_arm"] == 2
    selection_anchor = receipt["evidence_anchors"]["nominal_selection"]
    assert selection_anchor["logical_sha256"] == exp017._json_hash(selection)
    assert selection_anchor["file_sha256"] == exp017._sha256(
        out / "nominal_selection.json")
    assert (receipt["experiment_identity"]["fields"]
            ["nominal_selection_identity_sha256"]
            == selection["selection_identity_sha256"])
    for field in (
        "n_eligible", "eligibility_fraction", "n_selected",
        "selected_fraction_of_eligible", "n_unselected_eligible",
        "planned_arm_denominator_np_per_arm",
    ):
        assert receipt[field] == selection[field]

    with np.load(out / "nominal_qpos.npz") as stored:
        arrays = {key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(arrays) == {"s200", "s201"}
    anchor = receipt["evidence_anchors"]["nominal_qpos"]
    assert anchor["n_attempted_seeds"] == 2
    assert anchor["n_qpos_arrays"] == 2
    assert anchor["content_sha256"] == exp017._array_hash(arrays)
    assert anchor["archive_sha256"] == exp017._sha256(out / "nominal_qpos.npz")

    nominal_rows = [json.loads(line) for line in
                    (out / "nominal_rows.jsonl").read_text().splitlines()]
    assert len(nominal_rows) == 2
    row_anchor = receipt["evidence_anchors"]["nominal_rows"]
    assert row_anchor["n_rows"] == 2
    assert row_anchor["logical_sha256"] == exp017._json_hash(nominal_rows)
    assert row_anchor["file_sha256"] == exp017._sha256(
        out / "nominal_rows.jsonl")
    failed, second_failed = nominal_rows
    assert failed["processing_status"] == "rejected_target_assignment"
    assert "no bounded one-to-one" in failed["assignment_reason"]
    assert len(failed["cycles"]) == 2
    assert failed["support_diagnostics"]["n_frames"] == 31
    checks = failed["assignment_diagnostics"]["cycle_scene_checks"]
    assert len(checks) == 2
    assert all(check["alignment_error"] is None for check in checks)
    assert all(check["maximum_alignment_phase_error"] >= 0 for check in checks)
    assert all(check["rejection_stage"] == "center_shift_bound"
               for check in checks)
    assert all(check["shift_within_bound"] is False for check in checks)
    assert all(abs(check["required_center_shift_frames"]) > 0 for check in checks)
    assert all("render_first_frame" in check and "render_last_frame" in check
               and "window_within_clip" in check for check in checks)

    assert second_failed["processing_status"] == "rejected_target_assignment"
    assert second_failed["eligibility_status"] == "ineligible"
    assert second_failed["attrition_stage"] == "target_assignment"
    # Every pool member is independently classified after earlier attrition.
    assert len(second_failed["cycles"]) == 2
    assert second_failed["support_diagnostics"]["n_frames"] == 31
    assert second_failed["final_progress_ratio_vs_prescribed_route"] == pytest.approx(
        3 / 3.1)
    for row in nominal_rows:
        key = row["qpos_archive_key"]
        assert row["qpos_content_sha256"] == exp017._array_hash(
            {key: arrays[key]})
        assert row["nominal_qpos_archive_sha256"] == anchor["archive_sha256"]
        assert (row["nominal_qpos_archive_content_sha256"]
                == anchor["content_sha256"])


def test_prescribed_route_progress_gates_stalled_sources_before_selection(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    # Fake clips advance 3.0 m against a 3.1 m prescription; this catches the difference
    # between a real route ratio and normalizing a stalled clip by its own final distance.
    args.min_source_progress_ratio = 1.0
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="no donor seed"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    donor = json.loads((out / "donor_candidates.jsonl").read_text())
    assert donor["adapted_final_progress_ratio_vs_prescribed_route"] == pytest.approx(3 / 3.1)
    assert "progress below locked" in donor["ineligible_reason"]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 2
    assert receipt["query_accounting"]["nominal_samples_returned"] == 0


def test_source_selection_failure_archives_every_donor_qpos_with_receipt_hashes(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch, fail_source=True)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="no donor seed"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    archive = out / "donor_qpos.npz"
    assert archive.exists()
    with np.load(archive) as stored:
        arrays = {key: np.array(stored[key], copy=True) for key in stored.files}
    assert set(arrays) == {"s100__adapted", "s100__neutral"}

    receipt = json.loads((out / "receipt.json").read_text())
    anchor = receipt["evidence_anchors"]["donor_qpos"]
    assert anchor["n_candidates"] == 1
    assert anchor["n_qpos_arrays"] == 2
    assert anchor["content_sha256"] == exp017._array_hash(arrays)
    assert anchor["archive_sha256"] == exp017._sha256(archive)
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 2
    assert receipt["query_accounting"]["nominal_samples_returned"] == 0
    provenance = receipt["run_provenance"]
    assert provenance["prepared_through"] == "source_generation_ready"
    assert provenance["complete_before_source_generation"] is True
    assert provenance["code"]["commit"] == "test-commit"
    assert provenance["code"]["tracked_diff_sha256"] == "d" * 64
    assert provenance["checkpoint"]["generator_id"] == "fake-generator@revision"
    assert provenance["checkpoint"]["checkpoint_sha256"] == "c" * 64
    assert provenance["threshold_calibration"]["status"] == "calibrated"
    assert provenance["prompts"]["adapted_source"] == exp017.STEP
    assert provenance["prompts"]["neutral_source"] == exp017.WALK
    assert provenance["generation_settings"]["noise_stream_version"] == 2
    assert (provenance["D"], provenance["K"], provenance["N"], provenance["P"]) == (
        1, 1, 1, 1)
    assert provenance["planned_ardy_samples"] == 5
    assert provenance["donor_seeds"] == [100]
    assert provenance["nominal_candidate_seeds"] == [200]
    assert provenance["evaluation_seeds"] is None
    assert len(provenance["fixed_scenes"]) == 1
    assert len(provenance["route_content_sha256"]) == 64
    assert "manifest" not in json.dumps(provenance).lower()
    assert receipt["run_provenance_sha256"] == exp017._json_hash(provenance)

    donor = json.loads((out / "donor_candidates.jsonl").read_text())
    assert donor["donor_qpos_archive_sha256"] == anchor["archive_sha256"]
    assert donor["donor_qpos_archive_content_sha256"] == anchor["content_sha256"]
    for arm in ("adapted", "neutral"):
        key = donor[f"{arm}_qpos_archive_key"]
        assert donor[f"{arm}_qpos_content_sha256"] == exp017._array_hash(
            {key: arrays[key]})
        diagnostic = donor[f"{arm}_support_diagnostics"]
        assert diagnostic["interpretation"].startswith("descriptive only")
        assert diagnostic["sides"]["left"]["support_fraction"] == 1.0
        assert diagnostic["longest_bilateral_unsupported_run_frames"] == 0
    assert "synthetic source has no supported phase cycle" in donor["ineligible_reason"]


def test_nominal_progress_gate_stops_before_manifest_and_final_sampling(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    args.min_nominal_progress_ratio = 1.0
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="0 eligible seeds"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    assert not (out / "manifest.json").exists()
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["query_accounting"]["donor_source_samples_returned"] == 2
    assert receipt["query_accounting"]["nominal_samples_returned"] == 1
    assert receipt["query_accounting"]["paired_evaluation_samples_returned"] == 0


def test_content_hash_is_key_order_independent_and_shape_sensitive():
    left = {
        "b": np.asarray([[1.0, 2.0]]),
        "a": np.asarray([3], dtype=np.int64),
    }
    right = {"a": left["a"].copy(), "b": left["b"].copy()}
    assert exp017._array_hash(left) == exp017._array_hash(right)
    assert exp017._array_hash(left) != exp017._array_hash({
        "a": left["a"], "b": left["b"].reshape(-1),
    })
