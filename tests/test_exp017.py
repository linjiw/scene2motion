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
            samples.append(_sample(num_frames, role=role, seed=seed))
        return samples

    @staticmethod
    def to_qpos(sample):
        return np.asarray(sample["qpos"], dtype=float)


def _args(tmp_path, *, donors=2, seeds=2, obstacles=(1.0, 2.0)):
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


def _patch_cpu_dependencies(monkeypatch, *, fail_nominal=False):
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
        if marker == 1:
            return (SOURCE_ADAPTED,)
        if marker == 2:
            return (SOURCE_NEUTRAL,)
        if fail_nominal:
            raise ValueError("synthetic nominal has no target cycle")
        return (TARGET_A, TARGET_B)

    monkeypatch.setattr(exp017, "_phase_cycles", phase_cycles)

    def feet(_body, qpos, _fps):
        forward = np.asarray(qpos)[:, 0] + 0.1
        values = {"forward_representative_m": forward}
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

    # 2D + N + 2NP = 4 + 2 + 8 = 14 samples. Python invocation count is separate.
    assert receipt["actual_ardy_samples"] == 14
    assert receipt["planned_ardy_samples"] == 14
    assert receipt["query_accounting"] == {
        "donor_source_samples": 4,
        "nominal_samples": 2,
        "paired_evaluation_samples": 8,
        "generate_invocations": 6,
    }
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
    assert manifest["budget_formula"] == "2D+N+2NP"
    assert manifest["donor_seeds"] == [100, 101]
    assert manifest["evaluation_seeds"] == [200, 201]
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
    assert (out / "selected_source_qpos.npz").exists()
    assert (out / "nominal_qpos.npz").exists()
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
    assert receipt["rows_sha256"] == exp017._sha256(out / "rows.jsonl")
    assert receipt["qpos_archive_sha256"] == exp017._sha256(out / "qpos.npz")
    with np.load(out / "qpos.npz") as archive:
        qpos_arrays = {name: archive[name] for name in archive.files}
    assert receipt["qpos_content_sha256"] == exp017._array_hash(qpos_arrays)
    assert receipt["output_identity_set_sha256"] == exp017._json_hash([
        row["output_identity_sha256"] for row in final_rows])


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
    assert receipt["query_accounting"]["donor_source_samples"] == 0


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
    assert receipt["query_accounting"]["donor_source_samples"] == 0


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

    with pytest.raises(exp017.ExperimentAbort, match="no target cycle"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    assert not (out / "manifest.json").exists()
    assert len(runner.calls) == 2
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["failed_stage"] == "nominal_discovery"
    assert receipt["query_accounting"]["donor_source_samples"] == 2
    assert receipt["query_accounting"]["nominal_samples"] == 1
    assert receipt["query_accounting"]["paired_evaluation_samples"] == 0


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
    assert receipt["query_accounting"]["donor_source_samples"] == 2
    assert receipt["query_accounting"]["nominal_samples"] == 0


def test_nominal_progress_gate_stops_before_manifest_and_final_sampling(
    monkeypatch, tmp_path
):
    _patch_cpu_dependencies(monkeypatch)
    args = _args(tmp_path, donors=1, seeds=1, obstacles=(1.0,))
    args.min_nominal_progress_ratio = 1.0
    runner = FakeRunner(tmp_path / "exp017")

    with pytest.raises(exp017.ExperimentAbort, match="nominal seed 200 progress"):
        exp017.run_experiment(args, runner=runner, body=object())

    out = tmp_path / "exp017"
    assert not (out / "manifest.json").exists()
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["query_accounting"]["donor_source_samples"] == 2
    assert receipt["query_accounting"]["nominal_samples"] == 1
    assert receipt["query_accounting"]["paired_evaluation_samples"] == 0


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
