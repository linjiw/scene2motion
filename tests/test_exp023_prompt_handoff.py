import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import exp023_prompt_handoff as exp


class FakeRunner:
    fps = exp.FPS
    noise_stream_version = exp.NOISE_STREAM_VERSION
    history_frames = 196

    def __init__(self, output: Path):
        self.output = output
        self.calls = 0
        # Factory-based DI preserves the production ordering: evidence predates model setup.
        assert (output / "receipt.json").exists()
        assert (output / "rows.jsonl").exists()
        self.model = SimpleNamespace(
            gen_horizon_len=exp.HORIZON,
            num_frames_per_token=4,
        )

    def generate_prompt_schedule(
        self, schedules, specs, num_frames, diffusion_steps, cfg_weight, *, seeds,
        history_frames
    ):
        # The campaign must make its empty evidence durable before this GPU-equivalent call.
        assert json.loads((self.output / "receipt.json").read_text())["status"] == "running"
        assert (self.output / "rows.jsonl").read_text() == ""
        assert (self.output / "qpos.npz").exists()
        assert (self.output / "features.npz").exists()
        if self.calls == 0:
            assert json.loads((self.output / "noise_audit.json").read_text()) == []
        assert num_frames == exp.N_FRAMES
        assert diffusion_steps == exp.DIFFUSION_STEPS
        assert tuple(cfg_weight) == exp.CFG_WEIGHT
        assert history_frames == 4
        assert len(schedules) == exp.CHUNK_ROWS
        assert len(specs) == len(schedules) == len(seeds)
        expected_seeds = exp.SEEDS[
            self.calls * exp.CHUNK_SEED_COUNT:(self.calls + 1) * exp.CHUNK_SEED_COUNT
        ]
        assert tuple(dict.fromkeys(seeds)) == expected_seeds
        self.calls += 1

        features = np.zeros((len(schedules), exp.PADDED_FRAMES, 4), dtype=np.float32)
        audit = []
        for window in range(exp.N_WINDOWS):
            hashes = []
            for row, (schedule, seed) in enumerate(zip(schedules, seeds)):
                prompt_code = 10.0 if schedule[window] == exp.STEP else 0.0
                start = window * exp.HORIZON
                end = start + exp.HORIZON
                # Seed term and advancing-window term mimic shared corresponding-window noise.
                features[row, start:end, 0] = prompt_code
                features[row, start:end, 1] = float(seed - exp.SEEDS[0])
                features[row, start:end, 2] = float(window + 1) / 10.0
                value = f"{seed}:{window}".encode()
                hashes.append(hashlib.sha256(value).hexdigest())
            accepted_before = exp.EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
            history_start = exp.EXPECTED_GLOBAL_HISTORY_START_FRAMES[window]
            transcript_frames = exp.EXPECTED_TRANSCRIPT_FRAMES[window]
            input_hashes = (
                [] if window == 0 else [
                    exp._raw_row_sha256(features[row, history_start:accepted_before])
                    for row in range(len(schedules))
                ]
            )
            audit.append({
                "shape": [len(schedules), 13, 4],
                "row_sha256": hashes,
                "window_index": window,
                "global_history_start_frame": history_start,
                "accepted_transcript_frames_before": accepted_before,
                "input_history_frames": exp.EXPECTED_INPUT_HISTORY_FRAMES[window],
                "model_num_frames": exp.EXPECTED_MODEL_NUM_FRAMES[window],
                "transcript_frames": transcript_frames,
                "input_history_row_sha256": input_hashes,
                "returned_input_history_row_sha256": list(input_hashes),
                "returned_history_reconstruction_exact": [True] * len(schedules),
                "returned_history_reconstruction_max_abs": [0.0] * len(schedules),
                "stable_transcript_row_sha256": [
                    exp._raw_row_sha256(features[row, :transcript_frames])
                    for row in range(len(schedules))
                ],
            })
        return features, audit

    def decode_features(self, features):
        samples = []
        for feature in np.asarray(features):
            qpos = np.zeros((exp.PADDED_FRAMES, 36), dtype=np.float32)
            qpos[:, :4] = feature
            samples.append({"qpos": qpos})
        return samples

    @staticmethod
    def to_qpos(sample):
        return sample["qpos"]


class ReplayedWindowRunner(FakeRunner):
    def generate_prompt_schedule(self, *args, **kwargs):
        features, audit = super().generate_prompt_schedule(*args, **kwargs)
        audit[1]["row_sha256"] = list(audit[0]["row_sha256"])
        return features, audit


class CorruptHistoryAuditRunner(FakeRunner):
    def generate_prompt_schedule(self, *args, **kwargs):
        features, audit = super().generate_prompt_schedule(*args, **kwargs)
        audit[1]["input_history_row_sha256"][0] = "f" * 64
        return features, audit


class SecondChunkFailureRunner(FakeRunner):
    def generate_prompt_schedule(self, *args, **kwargs):
        if self.calls == 1:
            self.calls += 1
            raise RuntimeError("synthetic second-chunk failure")
        return super().generate_prompt_schedule(*args, **kwargs)


class Float64PrefixMismatchRunner(FakeRunner):
    def decode_features(self, features):
        samples = super().decode_features(features)
        for sample in samples:
            sample["qpos"] = np.asarray(sample["qpos"], dtype=np.float64)
        # Row two is seed 4500 / step_52; this would collapse to zero if downcast to float32.
        samples[2]["qpos"][0, 0] = 1e-50
        return samples


def clean_code_state(_repo):
    return {
        "commit": "a" * 40,
        "dirty": False,
        "status": [],
        "tracked_diff_sha256": "b" * 64,
    }


def fake_event(_body, qpos, route, onset):
    present = bool(np.asarray(qpos)[onset, 0] >= 5.0)
    if not present:
        return {
            "present": False,
            "missing_reason": "fake_no_event",
            "max_profile_height_m": 0.0,
        }
    return {
        "present": True,
        "missing_reason": None,
        "frame": int(onset + 12),
        "latency_frames": 12,
        "latency_s": 12 / exp.FPS,
        "side": "left",
        "profile_x_m": float(route[onset, 1] + 0.9),
        "foot_x_m": float(route[onset, 1] + 0.9),
        "whole_body_clearance_m": 0.08,
        "foot_bottom_clearance_m": 0.10,
        "max_profile_height_m": 0.08,
    }


def missing_delayed_event(body, qpos, route, onset):
    if onset > 0:
        return {
            "present": False,
            "missing_reason": "forced_delayed_missing",
            "max_profile_height_m": 0.0,
        }
    return fake_event(body, qpos, route, onset)


def missing_event(_body, _qpos, _route, _onset):
    return {
        "present": False,
        "missing_reason": "forced_missing",
        "max_profile_height_m": 0.0,
    }


def event_for_every_prompt(_body, _qpos, route, onset):
    return {
        "present": True,
        "missing_reason": None,
        "frame": int(onset + 12),
        "latency_frames": 12,
        "latency_s": 12 / exp.FPS,
        "side": "left",
        "profile_x_m": float(route[onset, 1] + 0.9),
        "foot_x_m": float(route[onset, 1] + 0.9),
        "whole_body_clearance_m": 0.08,
        "foot_bottom_clearance_m": 0.10,
        "max_profile_height_m": 0.08,
    }


def out_of_window_event(_body, _qpos, route, onset):
    return {
        "present": True,
        "missing_reason": None,
        "frame": int(onset + exp.POST_ONSET_FRAMES),
        "latency_frames": exp.POST_ONSET_FRAMES,
        "latency_s": exp.POST_ONSET_FRAMES / exp.FPS,
        "side": "left",
        "profile_x_m": float(route[onset, 1] + 0.9),
        "foot_x_m": float(route[onset, 1] + 0.9),
        "whole_body_clearance_m": 0.08,
        "foot_bottom_clearance_m": 0.10,
        "max_profile_height_m": 0.08,
    }


def fake_exact_box(_body, _qpos, x):
    collision_free = {
        f"{height:g}": bool(height <= 0.08)
        for height in exp.GRADED_HEIGHTS_M
    }
    return {
        "obstacle_x_m": float(x),
        "obstacle_depth_m": exp.OBSTACLE_DEPTH_M,
        "traversal": {
            "traversed": True,
            "criterion": "both physical feet cross before-over-after",
            "per_side": {
                side: {"crossed_before_over_after": True}
                for side in ("left", "right")
            },
        },
        "max_box_height_collision_free_lower_bound_m": 0.08,
        "max_box_height_lower_bound_m": 0.08,
        "collision_free": collision_free,
        "clears": dict(collision_free),
    }


def fake_metrics(_body, _qpos, _route):
    return {
        "progress_ratio": 1.0,
        "route_path_mae_m": 0.0,
        "max_foot_floor_penetration_m": 0.0,
    }


def valid_audit(features, plan):
    exact = np.asarray(features)
    audit = []
    for window in range(exp.N_WINDOWS):
        before = exp.EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
        start = exp.EXPECTED_GLOBAL_HISTORY_START_FRAMES[window]
        after = exp.EXPECTED_TRANSCRIPT_FRAMES[window]
        input_hashes = [] if window == 0 else [
            exp._raw_row_sha256(exact[row, start:before]) for row in range(len(plan))
        ]
        audit.append({
            "shape": [len(plan), 13, 3],
            "row_sha256": [
                hashlib.sha256(f"{item['seed']}:{window}".encode()).hexdigest()
                for item in plan
            ],
            "window_index": window,
            "global_history_start_frame": start,
            "accepted_transcript_frames_before": before,
            "input_history_frames": exp.EXPECTED_INPUT_HISTORY_FRAMES[window],
            "model_num_frames": exp.EXPECTED_MODEL_NUM_FRAMES[window],
            "transcript_frames": after,
            "input_history_row_sha256": input_hashes,
            "returned_input_history_row_sha256": list(input_hashes),
            "returned_history_reconstruction_exact": [True] * len(plan),
            "returned_history_reconstruction_max_abs": [0.0] * len(plan),
            "stable_transcript_row_sha256": [
                exp._raw_row_sha256(exact[row, :after]) for row in range(len(plan))
            ],
        })
    return audit


def campaign_kwargs(output: Path, event_fn=fake_event):
    return {
        "out": output,
        "runner_factory": lambda: FakeRunner(output),
        "body": object(),
        "code_state_fn": clean_code_state,
        "source_hashes_fn": lambda _repo: {
            exp.PROTOCOL_PATH: "d" * 64,
            "experiments/exp023_prompt_handoff.py": "c" * 64,
        },
        "generator_identity_fn": lambda _runner: {"generator": "fake"},
        "generator_identity_validator_fn": lambda value: dict(value),
        "runtime_identity_fn": lambda: {"runtime": "fake"},
        "physical_identity_fn": lambda: {"physical": "fake"},
        "prompt_identity_fn": lambda _runner, _path: {"prompts": "fake"},
        "pin_validator_fn": lambda _generator, _runtime, _physical: None,
        "channel_usage_fn": lambda _runner, _spec: {"root_pos": 400},
        "event_detector_fn": event_fn,
        "exact_box_fn": fake_exact_box,
        "motion_metrics_fn": fake_metrics,
    }


def test_locked_plan_and_complete_campaign_accounting(tmp_path):
    output = tmp_path / "exp023"
    receipt = exp.run_campaign(**campaign_kwargs(output))

    assert receipt["status"] == "complete"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["execution_mode"]["scientific_evidence_eligible"] is False
    assert receipt["execution_mode"]["pre_model_construction_evidence_guaranteed"] is True
    assert receipt["actual_ardy_samples"] == 32
    assert receipt["provenance"]["protocol"] == {
        "path": exp.PROTOCOL_PATH,
        "sha256": "d" * 64,
    }
    assert [chunk["seeds"] for chunk in exp.locked_chunk_plan()] == [
        [4500, 4501], [4502, 4503], [4504, 4505], [4506, 4507]
    ]
    assert receipt["query_accounting"] == {
        "schedule_invocations_planned": 4,
        "schedule_invocations_started": 4,
        "schedule_invocations_completed": 4,
        "autoregressive_window_calls_planned": 16,
        "autoregressive_window_calls_completed": 16,
        "trajectories_planned": 32,
        "trajectories_launched": 32,
        "trajectories_returned": 32,
        "trajectories_converted_to_qpos": 32,
        "trajectories_analyzed": 32,
    }
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    assert {(row["seed"], row["arm"]) for row in rows} == {
        (seed, arm) for seed in exp.SEEDS for arm in exp.ARMS
    }
    assert all(row["archived_frames"] == 208 for row in rows)
    assert all(row["scored_frames"] == 200 for row in rows)
    assert receipt["causal_pairing_audit"]["noise_fresh_across_windows"] is True
    assert receipt["causal_pairing_audit"][
        "history_inputs_match_visible_transcript_suffixes"
    ] is True
    assert receipt["causal_pairing_audit"]["qpos_prefixes_exact"] is True
    assert all(
        chunk["status"] == "complete"
        for chunk in receipt["generation_chunks"].values()
    )
    assert receipt["summary"]["event_rates_missing_retained"]["step_104"]["present"] == 8
    assert receipt["summary"]["all_walk_specificity"]["seeds_with_any_event"] == 0
    assert receipt["summary"]["pooled_present_event_slopes_descriptive"][
        "event_frame_on_onset_frame"
    ] == pytest.approx(1.0)
    assert receipt["summary"]["timing_interpretation"][
        "binary_timed_prompting_verdict"
    ] is None
    assert receipt["summary"]["fixed_box_rates"]["step_52"]["0.05"] == {
        "clears": 8,
        "planned": 8,
        "rate": 1.0,
    }
    assert receipt["summary"]["fixed_box_rates"]["all_walk_matched_windows"][
        "104"
    ]["0.12"]["rate"] == 0.0
    with np.load(output / "qpos.npz") as archive:
        assert len(archive.files) == 32
        assert archive["s4500_step_104"].shape == (208, 36)
    with np.load(output / "features.npz") as archive:
        assert len(archive.files) == 32
        assert archive["s4500_step_104"].shape == (208, 4)
    noise = json.loads((output / "noise_audit.json").read_text())
    assert len(noise) == 4
    assert all(len(window["rows"]) == 32 for window in noise)
    assert [window["input_history_frames"] for window in noise] == [0, 4, 4, 4]
    assert [window["global_history_start_frame"] for window in noise] == [0, 48, 100, 152]
    assert all(
        "stable_transcript_sha256" in row
        and "returned_history_reconstruction_max_abs" in row
        for window in noise for row in window["rows"]
    )


def test_substrate_refusal_keeps_all_rows_and_spends_seeds(tmp_path):
    output = tmp_path / "refused"
    with pytest.raises(exp.PromptHandoffAbort, match="substrate gate failed"):
        exp.run_campaign(**campaign_kwargs(output, event_fn=missing_event))

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "refused"
    assert receipt["refusal_reason"] == "step0_substrate_gate_failed"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["actual_ardy_samples"] == 32
    assert len((output / "rows.jsonl").read_text().splitlines()) == 32
    with np.load(output / "qpos.npz") as archive:
        assert len(archive.files) == 32


def test_specificity_refusal_is_not_relabelled_as_complete(tmp_path):
    output = tmp_path / "nonspecific"
    with pytest.raises(exp.PromptHandoffAbort, match="specificity gate failed"):
        exp.run_campaign(**campaign_kwargs(output, event_fn=event_for_every_prompt))

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "refused"
    assert receipt["refusal_reason"] == "all_walk_specificity_gate_failed"
    assert receipt["measurement_gates"]["step0_substrate"]["pass"] is True
    assert receipt["measurement_gates"]["all_walk_specificity"]["pass"] is False


def test_delayed_missingness_stays_in_denominator_and_cannot_become_support(tmp_path):
    output = tmp_path / "delayed_missing"
    receipt = exp.run_campaign(
        **campaign_kwargs(output, event_fn=missing_delayed_event))

    summary = receipt["summary"]
    assert summary["event_rates_missing_retained"]["step_52"] == {
        "present": 0,
        "planned": 8,
        "rate": 0.0,
        "missing": 8,
    }
    assert summary["event_rates_missing_retained"]["step_104"]["planned"] == 8
    assert summary["pooled_present_event_slopes_descriptive"][
        "event_frame_on_onset_frame"
    ] is None
    assert summary["timing_interpretation"]["binary_timed_prompting_verdict"] is None
    assert summary["timing_interpretation"]["complete_case_seed_count"] == 0


def test_injected_detector_cannot_put_event_outside_locked_window(tmp_path):
    output = tmp_path / "bad_event"
    with pytest.raises(exp.PromptHandoffAbort, match="outside the locked 96-frame"):
        exp.run_campaign(**campaign_kwargs(output, event_fn=out_of_window_event))

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["failed_stage"] == "analysis"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True


def test_noise_audit_rejects_replayed_window():
    plan = exp.locked_row_plan()
    features = np.zeros((32, 208, 3), dtype=np.float32)
    audit = valid_audit(features, plan)
    # Make every row's second-window latent a replay of its first-window latent.
    audit[1]["row_sha256"] = list(audit[0]["row_sha256"])
    with pytest.raises(ValueError, match="latent replay detected"):
        exp._validate_noise_and_feature_forks(features, audit, plan)


def test_history_audit_must_match_visible_four_frame_suffix():
    plan = exp.locked_row_plan()
    features = np.zeros((32, 208, 3), dtype=np.float32)
    audit = valid_audit(features, plan)
    audit[2]["input_history_row_sha256"][7] = "f" * 64
    with pytest.raises(ValueError, match="visible transcript suffix"):
        exp._validate_noise_and_feature_forks(features, audit, plan)


def test_stable_transcript_hash_must_match_archived_features():
    plan = exp.locked_row_plan()
    features = np.zeros((32, 208, 3), dtype=np.float32)
    audit = valid_audit(features, plan)
    audit[3]["stable_transcript_row_sha256"][0] = "f" * 64
    with pytest.raises(ValueError, match="archived features"):
        exp._validate_noise_and_feature_forks(features, audit, plan)


def test_causal_refusal_archives_raw_return_before_validation(tmp_path):
    output = tmp_path / "replayed"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: ReplayedWindowRunner(output)

    with pytest.raises(exp.PromptHandoffAbort, match="latent replay detected"):
        exp.run_campaign(**kwargs)

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["failed_stage"] == "generation_chunk00_seeds4500_4501"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["query_accounting"]["trajectories_returned"] == 8
    assert receipt["query_accounting"]["schedule_invocations_completed"] == 1
    assert receipt["spent_seeds"] == [4500, 4501]
    assert receipt["unlaunched_locked_seeds"] == [4502, 4503, 4504, 4505, 4506, 4507]
    with np.load(output / "features.npz") as archive:
        assert len(archive.files) == 8
        assert archive["s4500_step_104"].shape == (208, 4)
    raw_noise = json.loads((output / "noise_audit.json").read_text())
    assert raw_noise[0]["kind"] == "raw_chunk_runner_audit_before_validation"
    assert raw_noise[0]["chunk"] == "chunk00_seeds4500_4501"
    assert len(raw_noise[0]["audit"]) == 4


def test_corrupt_visible_history_audit_blocks_after_durable_chunk_return(tmp_path):
    output = tmp_path / "bad_history"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: CorruptHistoryAuditRunner(output)

    with pytest.raises(exp.PromptHandoffAbort, match="visible transcript suffix"):
        exp.run_campaign(**kwargs)

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["query_accounting"]["trajectories_returned"] == 8
    assert receipt["spent_seeds"] == [4500, 4501]
    with np.load(output / "features.npz") as archive:
        assert len(archive.files) == 8


def test_second_chunk_exception_preserves_first_chunk_and_spent_seed_ledger(tmp_path):
    output = tmp_path / "second_chunk_failure"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: SecondChunkFailureRunner(output)

    with pytest.raises(exp.PromptHandoffAbort, match="synthetic second-chunk failure"):
        exp.run_campaign(**kwargs)

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["sample_count_exact"] is False
    assert receipt["query_accounting"] == {
        "schedule_invocations_planned": 4,
        "schedule_invocations_started": 2,
        "schedule_invocations_completed": 1,
        "autoregressive_window_calls_planned": 16,
        "autoregressive_window_calls_completed": 4,
        "trajectories_planned": 32,
        "trajectories_launched": 16,
        "trajectories_returned": 8,
        "trajectories_converted_to_qpos": 0,
        "trajectories_analyzed": 0,
    }
    assert receipt["spent_seeds"] == [4500, 4501, 4502, 4503]
    assert receipt["unlaunched_locked_seeds"] == [4504, 4505, 4506, 4507]
    chunks = receipt["generation_chunks"]
    assert chunks["chunk00_seeds4500_4501"]["status"] == "complete"
    assert chunks["chunk01_seeds4502_4503"]["status"] == (
        "generation_exception_window_count_unknown"
    )
    with np.load(output / "features.npz") as archive:
        assert len(archive.files) == 8
        assert set(archive.files) == {
            f"s{seed}_{arm}" for seed in (4500, 4501) for arm in exp.ARMS
        }


def test_native_qpos_prefix_mismatch_cannot_hide_in_float32_archive(tmp_path):
    output = tmp_path / "qpos_mismatch"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: Float64PrefixMismatchRunner(output)

    with pytest.raises(exp.PromptHandoffAbort, match="decoded qpos prefix differs"):
        exp.run_campaign(**kwargs)

    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    with np.load(output / "qpos.npz") as archive:
        assert archive["s4500_step_52"].dtype == np.float64
        assert archive["s4500_step_52"][0, 0] == 1e-50


def _synthetic_feet(length=96):
    representative = np.linspace(0.0, 3.0, length)
    feet = {}
    for side, bonus in (("left", 0.15), ("right", 0.08)):
        clearance = 0.01 + bonus * np.exp(-((representative - 2.0) / 0.12) ** 2)
        feet[side] = {
            "forward_min_m": representative - 0.04,
            "forward_max_m": representative + 0.04,
            "forward_representative_m": representative,
            "bottom_clearance_m": clearance,
        }
    return feet


def test_fixed_box_endpoint_requires_both_feet_to_traverse():
    feet = _synthetic_feet(length=200)
    traversal = exp.fixed_box_traversal(feet, obstacle_x_m=2.0)
    assert traversal["traversed"] is True
    assert all(
        item["crossed_before_over_after"]
        for item in traversal["per_side"].values()
    )

    stalled = _synthetic_feet(length=200)
    stalled["right"] = {
        **stalled["right"],
        "forward_min_m": np.zeros(200),
        "forward_max_m": np.full(200, 0.05),
    }
    traversal = exp.fixed_box_traversal(stalled, obstacle_x_m=2.0)
    assert traversal["traversed"] is False
    assert traversal["per_side"]["right"]["crossed_before_over_after"] is False


def test_collision_free_nonarrival_is_not_a_fixed_box_success():
    class AlwaysClearProbe:
        def __init__(self, _x, _depth):
            pass

        @staticmethod
        def probe(_qpos):
            return 0.4

        @staticmethod
        def clears(_qpos, _height):
            return True

    stalled = _synthetic_feet(length=200)
    stalled["right"] = {
        **stalled["right"],
        "forward_min_m": np.zeros(200),
        "forward_max_m": np.full(200, 0.05),
    }
    record = exp.score_exact_boxes(
        object(), np.zeros((exp.N_FRAMES, 36)), 2.0,
        probe_factory=AlwaysClearProbe,
        foot_series_fn=lambda _body, _qpos, _fps: stalled,
    )
    assert record["traversal"]["traversed"] is False
    assert record["max_box_height_collision_free_lower_bound_m"] == 0.4
    assert record["max_box_height_lower_bound_m"] == 0.0
    assert all(record["collision_free"].values())
    assert not any(record["clears"].values())
    exp._validated_exact_box_record(record, 2.0)

    invalid = {**record, "clears": {key: True for key in record["clears"]}}
    with pytest.raises(ValueError, match="traversal-gated"):
        exp._validated_exact_box_record(invalid, 2.0)


def test_event_selector_requires_threshold_and_ordered_two_foot_crossing():
    xs = np.asarray([1.0, 2.0])
    feet = _synthetic_feet()
    event = exp.select_event_from_profile(
        xs, np.asarray([0.04, 0.08]), feet, onset_frame=52)
    assert event["present"] is True
    assert event["side"] == "left"
    assert event["profile_x_m"] == 2.0
    assert event["frame"] == 52 + event["latency_frames"]

    missing = exp.select_event_from_profile(
        xs, np.asarray([0.01, 0.025]), feet, onset_frame=52)
    assert missing["present"] is False
    assert missing["missing_reason"] == "whole_body_clearance_below_3cm"
    assert missing["max_profile_height_m"] == 0.025

    not_traversed = _synthetic_feet()
    not_traversed["right"] = {
        **not_traversed["right"],
        "forward_min_m": np.zeros(96),
        "forward_max_m": np.full(96, 0.05),
        "forward_representative_m": np.full(96, 0.025),
    }
    event = exp.select_event_from_profile(
        xs, np.asarray([0.04, 0.08]), not_traversed, onset_frame=52)
    assert event["present"] is False
    assert "no_two_foot" in event["missing_reason"]


def test_event_selector_rejects_untraversed_high_centre_before_threshold():
    xs = np.asarray([0.05, 2.0])
    feet = _synthetic_feet()
    event = exp.select_event_from_profile(
        xs, np.asarray([0.40, 0.02]), feet, onset_frame=0)
    assert event["present"] is False
    assert event["missing_reason"] == "whole_body_clearance_below_3cm"
    assert event["max_profile_height_m"] == pytest.approx(0.02)
    assert event["global_unfiltered_max_profile_height_m"] == pytest.approx(0.40)
