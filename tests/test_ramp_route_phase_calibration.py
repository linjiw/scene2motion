import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as calibration
from scene2motion.ramp.route_phase import RouteTimingBounds
from scene2motion.stepover_eval import StepOverThresholds


def _threshold_receipt(tmp_path):
    path = tmp_path / "physical_thresholds.json"
    source = (
        Path(calibration.__file__).resolve().parents[1]
        / "outputs"
        / "exp016_threshold_calibration"
        / "receipt.json"
    )
    path.write_bytes(source.read_bytes())
    return path


def _clean_code(_repo):
    return {
        "commit": "test-commit",
        "dirty": False,
        "status": [],
        "tracked_diff_sha256": "b" * 64,
    }


def _generator_identity(_runner):
    snapshot_files = {"config.yaml": "f" * 64}
    return {
        "checkpoint": {
            "generator_id": "fake-g1@test-revision",
            "checkpoint_sha256": "c" * 64,
            "snapshot_manifest_sha256": calibration._json_hash(snapshot_files),
            "snapshot_file_sha256": snapshot_files,
            "required_model_files": ["config.yaml"],
        },
        "text_cache": {
            "path": "/fake/cache.npz",
            "sha256": "d" * 64,
            "walk_prompt": calibration.WALK_PROMPT,
            "walk_prompt_cache_key_sha1": hashlib.sha1(
                calibration.WALK_PROMPT.encode()
            ).hexdigest(),
            "walk_prompt_embedding_content_sha256": "e" * 64,
            "runner_memory_embedding_content_sha256": "e" * 64,
            "runner_memory_byte_matches_file": True,
            "walk_prompt_embedding_shape": [4, 8],
        },
        "runner_class": "FakeRunner",
        "runner_fps": calibration.FPS,
        "noise_stream_version": 2,
    }


class FakeRunner:
    fps = calibration.FPS
    noise_stream_version = 2
    model_name = "fake-g1"
    cache_path = None
    model = SimpleNamespace(gen_horizon_len=64)

    def __init__(self, *, wrong_return_call=None, extra_return_call=None, raise_call=None):
        self.calls = []
        self.wrong_return_call = wrong_return_call
        self.extra_return_call = extra_return_call
        self.raise_call = raise_call

    def generate(self, prompts, specs, num_frames, diffusion_steps, *, cfg_weight, seeds):
        call_index = len(self.calls)
        self.calls.append(
            {
                "prompts": list(prompts),
                "specs": list(specs),
                "num_frames": num_frames,
                "diffusion_steps": diffusion_steps,
                "cfg_weight": tuple(cfg_weight),
                "seeds": list(seeds),
            }
        )
        if self.raise_call == call_index:
            raise RuntimeError("synthetic generation failure")
        outputs = []
        for seed, spec in zip(seeds, specs):
            qpos = np.zeros((num_frames, 8), dtype=float)
            qpos[:, 0] = np.asarray(spec.root_xz)[:, 1]
            qpos[:, 2] = 0.8
            qpos[:, 3] = 1.0
            outputs.append(
                {
                    "qpos": qpos,
                    "seed": np.asarray([seed], dtype=np.int64),
                    "route": np.asarray(spec.root_xz, dtype=float),
                }
            )
        if self.wrong_return_call == call_index:
            return outputs[:-1]
        if self.extra_return_call == call_index:
            extra = {name: np.array(value, copy=True) for name, value in outputs[-1].items()}
            extra["seed"] = np.asarray([999999], dtype=np.int64)
            return [*outputs, extra]
        return outputs

    @staticmethod
    def to_qpos(sample):
        return np.array(sample["qpos"], copy=True)


def _background_identity(side, start, end):
    return {
        "baseline_support_side": side,
        "window_start_frame": int(start),
        "window_end_frame": int(end),
        "method": calibration.PHASE_BACKGROUND_METHOD,
    }


def _cycle(
    *,
    split="calibration",
    seed=3200,
    speed_label="reference",
    requested_speed_mps=calibration.REFERENCE_SPEED_MPS,
    side="left",
    apex=99,
    prominence=0.06,
    backgrounds=(0.006, 0.008),
    offset=0.0,
    identity_shift=0,
    digest=None,
):
    takeoff = apex - 9
    landing = apex + 9
    identities = (
        _background_identity(side, takeoff - 4 + identity_shift, takeoff - 1 + identity_shift),
        _background_identity(side, landing + identity_shift, landing + 3 + identity_shift),
    )
    digest = digest or calibration._json_hash(
        {
            "split": split,
            "seed": seed,
            "speed": speed_label,
            "side": side,
            "apex": apex,
            "identity_shift": identity_shift,
        }
    )
    return calibration.ObservedCycle(
        split=split,
        seed=seed,
        speed_label=speed_label,
        requested_speed_mps=requested_speed_mps,
        swing_side=side,
        takeoff_frame=takeoff,
        apex_frame=apex,
        landing_frame=landing,
        prominence_m=prominence,
        background_contrasts_m=tuple(backgrounds),
        background_window_identities=identities,
        nominal_foot_forward_offset_m=offset,
        evidence_digest=digest,
        phase_evidence={"receipt_digest": digest},
    )


def _clip(*, split, seed, speed_label, cycles):
    speed = dict(calibration.SPEEDS)[speed_label]
    key = f"{split}__{speed_label}__seed{seed}"
    return calibration.AnalyzedClip(
        split=split,
        seed=seed,
        speed_label=speed_label,
        requested_speed_mps=speed,
        sample_sha256="1" * 64,
        qpos_content_sha256="2" * 64,
        qpos_archive_key=key,
        cycles=tuple(cycles),
    )


def _fake_analyzer(**kwargs):
    split = kwargs["split"]
    seed = kwargs["seed"]
    speed_label = kwargs["speed_label"]
    speed = kwargs["requested_speed_mps"]
    sample = kwargs["sample"]
    qpos = np.asarray(kwargs["qpos"])
    key = kwargs["qpos_archive_key"]
    cycles = (
        _cycle(
            split=split,
            seed=seed,
            speed_label=speed_label,
            requested_speed_mps=speed,
            side="left",
            apex=99,
        ),
        _cycle(
            split=split,
            seed=seed,
            speed_label=speed_label,
            requested_speed_mps=speed,
            side="right",
            apex=100,
        ),
    )
    return calibration.AnalyzedClip(
        split=split,
        seed=seed,
        speed_label=speed_label,
        requested_speed_mps=speed,
        sample_sha256=calibration._sample_hash(sample),
        qpos_content_sha256=calibration._array_hash({key: qpos}),
        qpos_archive_key=key,
        cycles=cycles,
    )


def _run(tmp_path, runner=None, **kwargs):
    runner = runner or FakeRunner()
    receipt = calibration.run_campaign(
        out=tmp_path / "out",
        threshold_receipt=_threshold_receipt(tmp_path),
        runner=runner,
        body=object(),
        code_state_fn=kwargs.pop("code_state_fn", _clean_code),
        generator_identity_fn=_generator_identity,
        analyze_clip_fn=kwargs.pop("analyze_clip_fn", _fake_analyzer),
        **kwargs,
    )
    return runner, receipt


def test_locked_plan_has_exact_nine_batches_and_72_paired_samples():
    plan = calibration.locked_batch_plan()
    assert len(plan) == 9
    assert sum(len(batch.seeds) for batch in plan) == 72
    assert [batch.index for batch in plan] == list(range(9))
    assert [(batch.split, batch.seed_block, batch.speed_label) for batch in plan] == [
        ("calibration", 0, "slow"),
        ("calibration", 0, "reference"),
        ("calibration", 0, "fast"),
        ("calibration", 1, "slow"),
        ("calibration", 1, "reference"),
        ("calibration", 1, "fast"),
        ("validation", 0, "slow"),
        ("validation", 0, "reference"),
        ("validation", 0, "fast"),
    ]
    for block in (plan[:3], plan[3:6], plan[6:9]):
        assert block[0].seeds == block[1].seeds == block[2].seeds


def test_quantile_and_outward_rounding_boundaries():
    assert calibration.nearest_rank(range(1, 17), 0.95) == 16
    assert calibration.nearest_rank(range(1, 21), 0.95) == 19
    assert calibration.nearest_rank([3.0, 1.0, 2.0], 0.25) == 1.0
    assert calibration.ceil_outward(0.010, 0.001) == 0.010
    assert calibration.ceil_outward(0.0100001, 0.001) == 0.011
    assert calibration.ceil_outward(0.0, 0.1, positive_floor=True) == 0.1
    with pytest.raises(ValueError):
        calibration.nearest_rank([], 0.95)


def test_locked_text_cache_requires_runner_memory_to_byte_match_file(tmp_path):
    key = hashlib.sha1(calibration.WALK_PROMPT.encode()).hexdigest()
    embedding = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache = tmp_path / "cache.npz"
    np.savez(cache, **{key: embedding})
    runner = SimpleNamespace(_text_cache={key: np.array(embedding, copy=True)})
    identity = calibration._locked_text_cache_identity(runner, cache)
    assert identity["runner_memory_byte_matches_file"] is True
    assert identity["runner_memory_embedding_content_sha256"] == identity[
        "walk_prompt_embedding_content_sha256"
    ]
    runner._text_cache[key][0, 0] += 1.0
    with pytest.raises(ValueError, match="does not byte-match"):
        calibration._locked_text_cache_identity(runner, cache)


def test_atomic_evidence_write_preserves_prior_file_when_replace_fails(
    tmp_path, monkeypatch
):
    target = tmp_path / "receipt.json"
    target.write_bytes(b"prior evidence\n")

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace interruption")

    monkeypatch.setattr(calibration.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace interruption"):
        calibration._write_json(target, {"new": "evidence"})
    assert target.read_bytes() == b"prior evidence\n"
    assert not (tmp_path / ".receipt.json.tmp").exists()


def test_known_cap_quantile_headroom_and_outward_rounding_values():
    acceleration = calibration.calibrated_upper_bound(
        [0.011] * 15 + [0.019], quantum=0.01, positive_floor=True
    )
    jerk = calibration.calibrated_upper_bound(
        [0.11] * 15 + [0.21], quantum=0.1, positive_floor=True
    )
    endpoint = calibration.calibrated_upper_bound(
        [0.002] * 15 + [0.004], quantum=0.01
    )
    assert acceleration["nearest_rank_value"] == pytest.approx(0.019)
    assert acceleration["expanded_value"] == pytest.approx(0.02375)
    assert acceleration["value"] == pytest.approx(0.03)
    assert jerk["nearest_rank_value"] == pytest.approx(0.21)
    assert jerk["expanded_value"] == pytest.approx(0.2625)
    assert jerk["value"] == pytest.approx(0.3)
    assert endpoint["nearest_rank_value"] == pytest.approx(0.004)
    assert endpoint["expanded_value"] == pytest.approx(0.005)
    assert endpoint["value"] == pytest.approx(0.01)


def test_foot_offset_sign_matches_obstacle_minus_nominal_foot_offset():
    ahead = _cycle(offset=0.2)
    behind = _cycle(offset=-0.2)
    assert ahead.event_root_progress_m == pytest.approx(3.4)
    assert behind.event_root_progress_m == pytest.approx(3.8)


def test_real_analyzer_wires_exact_kinematics_to_phase_v3(monkeypatch):
    qpos = np.zeros((calibration.N_FRAMES, 8), dtype=float)
    qpos[:, 0] = np.linspace(0.0, calibration.PILOT_ROUTE_LENGTH_M, calibration.N_FRAMES)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    kinematics = {
        side: {
            "bottom_clearance_m": np.zeros(calibration.N_FRAMES, dtype=float),
            "planar_speed_mps": np.zeros(calibration.N_FRAMES, dtype=float),
            "forward_representative_m": qpos[:, 0] + (0.15 if side == "left" else -0.1),
        }
        for side in ("left", "right")
    }
    takeoff, apex, landing = 90, 99, 109
    rising = np.linspace(0.021, 0.06, apex - takeoff + 1)
    falling = np.linspace(0.06, 0.021, landing - apex)[1:]
    kinematics["left"]["bottom_clearance_m"][takeoff:landing] = np.concatenate(
        [rising, falling]
    )
    kinematics["left"]["planar_speed_mps"][takeoff:landing] = 1.0
    monkeypatch.setattr(
        calibration,
        "foot_kinematics_series",
        lambda body, exact_qpos, fps: kinematics,
    )
    sample = {"qpos": qpos, "marker": np.asarray([7], dtype=np.int64)}
    analyzed = calibration.analyze_generated_clip(
        body=object(),
        qpos=qpos,
        sample=sample,
        split="calibration",
        seed=3200,
        speed_label="reference",
        requested_speed_mps=calibration.REFERENCE_SPEED_MPS,
        thresholds=StepOverThresholds(),
        qpos_archive_key="calibration__reference__seed3200",
    )
    assert analyzed.measurement_rejection is None
    assert len(analyzed.cycles) == 1
    cycle = analyzed.cycles[0]
    assert cycle.swing_side == "left"
    assert cycle.apex_frame == apex
    assert cycle.prominence_m == pytest.approx(0.06)
    assert len(cycle.background_contrasts_m) == 2
    assert cycle.nominal_foot_forward_offset_m == pytest.approx(0.15)
    assert cycle.event_root_progress_m == pytest.approx(3.45)
    assert cycle.phase_evidence["schema_version"] == (
        calibration.PHASE_OBSERVABILITY_SCHEMA_VERSION
    )


def test_program_selection_is_minimum_deformation_then_receipt_digest():
    far = _cycle(apex=95, digest="f" * 64)
    near_b = _cycle(apex=99, digest="b" * 64, identity_shift=1)
    near_a = _cycle(apex=99, digest="a" * 64, identity_shift=2)
    clip = _clip(
        split="calibration",
        seed=3200,
        speed_label="reference",
        cycles=(far, near_b, near_a),
    )
    _, selected = calibration.build_and_select_programs(
        [clip], target_min_prominence_m=0.01
    )
    assert len(selected) == 1
    assert selected[0].cycle.apex_frame == 99
    assert selected[0].cycle.evidence_digest == "a" * 64
    _, reversed_selected = calibration.build_and_select_programs(
        [
            _clip(
                split="calibration",
                seed=3200,
                speed_label="reference",
                cycles=(near_a, near_b, far),
            )
        ],
        target_min_prominence_m=0.01,
    )
    assert reversed_selected[0].cycle.evidence_digest == "a" * 64


def test_separation_uses_selected_side_cycle_not_unused_high_prominence_cycle():
    clips = []
    for seed in calibration.VALIDATION_SEEDS:
        cycles = (
            _cycle(
                split="validation",
                seed=seed,
                side="left",
                apex=99,
                prominence=0.045,
                backgrounds=(0.035, 0.04),
            ),
            _cycle(
                split="validation",
                seed=seed,
                side="left",
                apex=90,
                prominence=0.20,
                backgrounds=(0.035, 0.04),
            ),
            _cycle(
                split="validation",
                seed=seed,
                side="right",
                apex=100,
                prominence=0.08,
                backgrounds=(0.008, 0.01),
            ),
        )
        clips.append(
            _clip(
                split="validation",
                seed=seed,
                speed_label="reference",
                cycles=cycles,
            )
        )
    _, selected = calibration.build_and_select_programs(
        clips, target_min_prominence_m=0.01
    )
    separation = calibration.validation_separation(
        clips,
        selected,
        speed_label="reference",
        target_min_prominence_m=0.01,
    )
    left = separation["side_results"]["left"]
    assert set(left["selected_signal_by_seed_m"].values()) == {0.045}
    assert left["required_signal_floor_m"] == pytest.approx(0.05)
    assert left["passed"] is False
    assert separation["passed"] is False
    assert separation["unused_high_prominence_cycles_cannot_satisfy_separation"] is True


def test_background_nulls_deduplicate_by_physical_window_not_prominence():
    first = _cycle(digest="a" * 64)
    second = _cycle(digest="b" * 64)
    clip = _clip(
        split="calibration",
        seed=3200,
        speed_label="reference",
        cycles=(first, second),
    )
    unique = calibration.deduplicated_clip_backgrounds(clip)
    assert len(unique) == 2
    receipt = calibration.background_deduplication_receipt([clip])
    assert receipt["raw_count"] == 4
    assert receipt["unique_count"] == 2
    assert receipt["duplicate_count"] == 2
    assert receipt["prominence_receipt_digest_is_not_part_of_null_identity"] is True


def test_zero_null_calibration_blocks_instead_of_accepting_flat_phase():
    clips = []
    for seed in calibration.CALIBRATION_SEEDS:
        clips.append(
            _clip(
                split="calibration",
                seed=seed,
                speed_label="reference",
                cycles=(
                    _cycle(
                        seed=seed,
                        backgrounds=(0.0, 0.0),
                        prominence=0.0,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="rounded Pmin is zero"):
        calibration.freeze_target_prominence(clips)


def test_full_cpu_orchestration_spends_exact_72_in_locked_order(tmp_path):
    runner, receipt = _run(tmp_path)
    assert receipt["status"] == "complete"
    assert receipt["complete"] is True
    assert receipt["actual_ardy_samples"] == 72
    assert len(runner.calls) == 9
    plan = calibration.locked_batch_plan()
    for call, batch in zip(runner.calls, plan):
        assert call["seeds"] == list(batch.seeds)
        assert call["prompts"] == [calibration.WALK_PROMPT] * 8
        assert call["num_frames"] == 200
        assert call["diffusion_steps"] == 5
        assert call["cfg_weight"] == (2.0, 2.0)
        assert all(
            spec.heading is None and spec.root_y is None for spec in call["specs"]
        )
        endpoint = batch.requested_speed_mps * ((200 - 1) / 25)
        assert all(spec.root_xz[-1, 1] == pytest.approx(endpoint) for spec in call["specs"])
    assert receipt["query_accounting"] == {
        "generate_invocations_planned": 9,
        "generate_invocations_started": 9,
        "generate_invocations_completed": 9,
        "samples_planned": 72,
        "samples_launched": 72,
        "samples_returned": 72,
        "samples_converted_to_qpos": 72,
        "samples_analyzed": 72,
    }
    assert len((tmp_path / "out" / "rows.jsonl").read_text().splitlines()) == 72
    with np.load(tmp_path / "out" / "qpos.npz") as archive:
        assert len(archive.files) == 72
    target = receipt["target_prominence_receipt"]
    donor = receipt["donor_step_quality_dependency"]
    assert target["sha256"] != donor["sha256"]
    assert target["fields"]["target_min_prominence_m"] == pytest.approx(0.01)
    assert donor["fields"]["min_relative_lift_m"] == pytest.approx(0.04)
    assert receipt["physical_threshold_dependency"]["fields"][
        "evidentiary_role"
    ] == "fixed-common-support-dependency-only"


def _validation_clips(*, prominence=0.06, backgrounds=(0.006, 0.008), sides=("left", "right"), apex=99):
    clips = []
    for speed_label, speed in calibration.SPEEDS:
        for seed in calibration.VALIDATION_SEEDS:
            cycles = tuple(
                _cycle(
                    split="validation",
                    seed=seed,
                    speed_label=speed_label,
                    requested_speed_mps=speed,
                    side=side,
                    apex=apex + (side == "right"),
                    prominence=prominence,
                    backgrounds=backgrounds,
                )
                for side in sides
            )
            clips.append(
                _clip(
                    split="validation",
                    seed=seed,
                    speed_label=speed_label,
                    cycles=cycles,
                )
            )
    return clips


def test_validation_cannot_widen_frozen_bounds_and_triggers_kill_rule():
    clips = _validation_clips(apex=90)
    _, selected = calibration.build_and_select_programs(
        clips, target_min_prominence_m=0.01
    )
    frozen = RouteTimingBounds(
        fps=25.0,
        min_discrete_route_progress_speed_mps=0.6,
        max_discrete_route_progress_speed_mps=1.2,
        max_abs_route_progress_acceleration_mps2=0.01,
        max_abs_discrete_route_progress_jerk_mps3=0.1,
        reference_route_progress_speed_mps=calibration.REFERENCE_SPEED_MPS,
        max_endpoint_route_progress_speed_deviation_mps=0.01,
    )
    before = frozen.as_dict()
    validation = calibration.validate_frozen_calibration(
        clips,
        selected,
        target_min_prominence_m=0.01,
        frozen_bounds=frozen,
    )
    assert validation["passed"] is False
    assert frozen.as_dict() == before
    assert validation["frozen_route_timing_bounds"] == before
    assert validation["validation_cannot_widen_calibration"] is True
    assert any("reference dynamics coverage" in reason for reason in validation["kill_reasons"])
    assert any("endpoint full-frozen coverage" in reason for reason in validation["kill_reasons"])
    assert not any(
        reason.startswith("slow: reference dynamics")
        or reason.startswith("fast: reference dynamics")
        or reason.startswith("reference: endpoint full-frozen")
        for reason in validation["kill_reasons"]
    )


def test_validation_background_and_side_kill_rules_are_fail_closed():
    clips = _validation_clips(sides=("left",))
    # Two validation seeds at every speed exceed the frozen target background gate.
    modified = []
    for clip in clips:
        if clip.seed in (3300, 3301):
            cycle = clip.cycles[0]
            cycle = calibration.ObservedCycle(
                **{
                    **cycle.__dict__,
                    "background_contrasts_m": (0.02, 0.021),
                }
            )
            clip = _clip(
                split=clip.split,
                seed=clip.seed,
                speed_label=clip.speed_label,
                cycles=(cycle,),
            )
        modified.append(clip)
    _, selected = calibration.build_and_select_programs(
        modified, target_min_prominence_m=0.01
    )
    permissive = RouteTimingBounds(
        fps=25.0,
        min_discrete_route_progress_speed_mps=0.6,
        max_discrete_route_progress_speed_mps=1.2,
        max_abs_route_progress_acceleration_mps2=10.0,
        max_abs_discrete_route_progress_jerk_mps3=10.0,
        reference_route_progress_speed_mps=calibration.REFERENCE_SPEED_MPS,
        max_endpoint_route_progress_speed_deviation_mps=1.0,
    )
    validation = calibration.validate_frozen_calibration(
        modified,
        selected,
        target_min_prominence_m=0.01,
        frozen_bounds=permissive,
    )
    assert validation["passed"] is False
    assert any("background exceedance" in reason for reason in validation["kill_reasons"])
    assert any("/right: side coverage" in reason for reason in validation["kill_reasons"])


def test_wrong_batch_return_preserves_partial_evidence_and_charged_budget(tmp_path):
    runner = FakeRunner(wrong_return_call=2)
    with pytest.raises(calibration.CalibrationAbort, match="seven|7 samples"):
        _run(tmp_path, runner=runner)
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["failed_stage"] == "generation"
    assert receipt["sample_count_exact"] is False
    assert receipt["actual_ardy_samples"] is None
    assert receipt["returned_ardy_samples_lower_bound"] == 23
    assert receipt["conservative_charged_ardy_samples"] == 24
    assert receipt["query_accounting"]["samples_converted_to_qpos"] == 23
    assert len((tmp_path / "out" / "rows.jsonl").read_text().splitlines()) == 23
    with np.load(tmp_path / "out" / "qpos.npz") as archive:
        assert len(archive.files) == 23


def test_over_return_charge_covers_observed_lower_bound_and_hashes_extra(tmp_path):
    runner = FakeRunner(extra_return_call=0)
    with pytest.raises(calibration.CalibrationAbort, match="9 samples"):
        _run(tmp_path, runner=runner)
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["sample_count_exact"] is False
    assert receipt["returned_ardy_samples_lower_bound"] == 9
    assert receipt["conservative_charged_ardy_samples"] == 9
    assert receipt["query_accounting"]["samples_launched"] == 8
    assert receipt["evidence_anchors"]["unexpected_returned_samples"]["count"] == 1
    (digest,) = receipt["evidence_anchors"]["unexpected_returned_samples"][
        "sample_sha256"
    ]
    assert calibration._is_sha256(digest)


def test_mid_batch_analysis_failure_prehashes_all_returned_samples(tmp_path):
    def fail_fourth(**kwargs):
        if kwargs["seed"] == 3203:
            raise ValueError("synthetic exact-foot analysis failure")
        return _fake_analyzer(**kwargs)

    runner = FakeRunner()
    with pytest.raises(calibration.CalibrationAbort, match="exact-foot analysis failure"):
        _run(tmp_path, runner=runner, analyze_clip_fn=fail_fourth)
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["returned_ardy_samples_lower_bound"] == 8
    assert receipt["conservative_charged_ardy_samples"] == 8
    assert receipt["query_accounting"]["samples_converted_to_qpos"] == 4
    assert receipt["query_accounting"]["samples_analyzed"] == 3
    attempts = [
        json.loads(line)
        for line in (tmp_path / "out" / "attempts.jsonl").read_text().splitlines()
    ]
    first_batch = [item for item in attempts if item["batch_index"] == 0]
    assert all(calibration._is_sha256(item["sample_sha256"]) for item in first_batch)
    assert [item["status"] for item in first_batch] == [
        "analyzed",
        "analyzed",
        "analyzed",
        "qpos_archived",
        "returned_unprocessed",
        "returned_unprocessed",
        "returned_unprocessed",
        "returned_unprocessed",
    ]


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("fps", 20.0, "fps == 25"),
        ("noise_stream_version", 1, "noise_stream_version == 2"),
        ("noise_stream_version", 2.5, "noise_stream_version == 2"),
    ],
)
def test_wrong_runner_fps_or_noise_refuses_before_generation(
    tmp_path, attribute, value, message
):
    runner = FakeRunner()
    setattr(runner, attribute, value)
    with pytest.raises(calibration.CalibrationAbort, match=message):
        _run(tmp_path, runner=runner)
    assert runner.calls == []
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["conservative_charged_ardy_samples"] == 0


def test_dirty_worktree_refuses_before_generation_and_writes_receipt(tmp_path):
    runner = FakeRunner()

    def dirty(_repo):
        return {
            "commit": "test-commit",
            "dirty": True,
            "status": ["?? experiments/calibrate_ramp_route_phase.py"],
            "tracked_diff_sha256": "e" * 64,
        }

    with pytest.raises(calibration.CalibrationAbort, match="exactly clean"):
        _run(tmp_path, runner=runner, code_state_fn=dirty)
    assert runner.calls == []
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["query_accounting"]["samples_launched"] == 0


def test_launch_git_state_is_captured_before_campaign_creates_its_output(tmp_path):
    out = tmp_path / "out"
    calls = []

    def output_aware_code_state(_repo):
        output_exists = out.exists()
        calls.append(output_exists)
        return {
            "commit": "test-commit",
            "dirty": False,
            "status": [],
            "tracked_diff_sha256": "b" * 64,
        }

    runner, receipt = _run(
        tmp_path,
        code_state_fn=output_aware_code_state,
    )
    assert len(runner.calls) == 9
    assert receipt["status"] == "complete"
    assert calls[0] is False
    assert all(calls[index] is True for index in range(1, len(calls)))


def test_completion_git_check_allows_only_the_campaign_output(tmp_path):
    repo = tmp_path / "repo"
    output = repo / "outputs" / "calibration"
    repo.mkdir()
    initial = {
        "commit": "test-commit",
        "dirty": False,
        "status": [],
        "tracked_diff_sha256": "b" * 64,
    }
    current = {
        "commit": "test-commit",
        "dirty": True,
        "status": ["?? outputs/calibration/"],
        "tracked_diff_sha256": "b" * 64,
    }
    check = calibration._verify_completion_git_state(
        initial,
        current,
        repo=repo,
        output=output,
    )
    assert check["allowed_output_status"] == ["?? outputs/calibration/"]
    assert check["unexpected_status"] == []


def test_nonempty_output_refusal_does_not_overwrite_existing_evidence(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_text("preserve me")
    runner = FakeRunner()
    with pytest.raises(calibration.CalibrationAbort, match="nonempty"):
        calibration.run_campaign(
            out=out,
            threshold_receipt=_threshold_receipt(tmp_path),
            runner=runner,
            body=object(),
            code_state_fn=_clean_code,
            generator_identity_fn=_generator_identity,
            analyze_clip_fn=_fake_analyzer,
        )
    assert runner.calls == []
    assert sentinel.read_text() == "preserve me"
    assert sorted(path.name for path in out.iterdir()) == ["sentinel.txt"]
