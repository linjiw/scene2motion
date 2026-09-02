"""CPU tests for the EXP-023b WALK->SQUEEZE prompt-switch positive-control driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import exp023_prompt_handoff as e23
from experiments import exp023b_prompt_switch_control as x
from experiments.analyze_exp021_exact_addressability import wilson_interval
from scene2motion import host_gate as hg
from scene2motion.robot import G1Body

REPO = Path(__file__).resolve().parents[1]
ARCHIVED_EXP023_QPOS = REPO / "outputs" / "exp023_prompt_handoff" / "qpos.npz"
LOCKED_ROW_PLAN_SHA256 = "3aff84b1effd53c194b7a1e67aba5737ed57caef315db218efba3e5852d963f1"
PROMPT_CODES = {x.WALK: 0.0, x.STEP: 10.0, x.SQUEEZE: 20.0}
JOINT_SCALE = 10.0


# --- fakes -----------------------------------------------------------------------------


class FakeRunner:
    fps = x.FPS
    noise_stream_version = x.NOISE_STREAM_VERSION

    def __init__(self, output: Path):
        self.output = output
        self.calls = 0
        # Factory-based DI preserves the production ordering: evidence predates model setup.
        assert (output / "receipt.json").exists()
        assert (output / "rows.jsonl").exists()
        self.model = SimpleNamespace(gen_horizon_len=x.HORIZON, num_frames_per_token=4)

    def generate_prompt_schedule(
        self, schedules, specs, num_frames, diffusion_steps, cfg_weight, *, seeds,
        history_frames
    ):
        receipt = json.loads((self.output / "receipt.json").read_text())
        assert receipt["status"] == "running"
        assert receipt["host_resource_gate"]["pass"] is True
        assert (self.output / "rows.jsonl").read_text() == ""
        assert (self.output / "qpos.npz").exists()
        assert (self.output / "features.npz").exists()
        assert num_frames == x.N_FRAMES
        assert diffusion_steps == x.DIFFUSION_STEPS
        assert tuple(cfg_weight) == x.CFG_WEIGHT
        assert history_frames == 4
        assert len(schedules) == x.CHUNK_ROWS
        assert len(specs) == len(schedules) == len(seeds)
        expected_seeds = x.SEEDS[
            self.calls * x.CHUNK_SEED_COUNT:(self.calls + 1) * x.CHUNK_SEED_COUNT
        ]
        assert tuple(dict.fromkeys(seeds)) == expected_seeds
        self.calls += 1

        features = np.zeros((len(schedules), x.PADDED_FRAMES, 4), dtype=np.float32)
        audit = []
        for window in range(x.N_WINDOWS):
            hashes = []
            for row, (schedule, seed) in enumerate(zip(schedules, seeds)):
                start = window * x.HORIZON
                end = start + x.HORIZON
                # Seed term and advancing-window term mimic shared corresponding-window noise.
                features[row, start:end, 0] = PROMPT_CODES[schedule[window]]
                features[row, start:end, 1] = float(seed - x.SEEDS[0])
                features[row, start:end, 2] = float(window + 1) / 10.0
                hashes.append(hashlib.sha256(f"{seed}:{window}".encode()).hexdigest())
            audit.append(window_audit(window, features, hashes))
        return self.mutate(features, audit)

    @staticmethod
    def mutate(features, audit):
        return features, audit

    def decode_features(self, features):
        samples = []
        for feature in np.asarray(features):
            qpos = np.zeros((x.PADDED_FRAMES, 36), dtype=np.float32)
            qpos[:, :4] = feature
            # One joint carries the prompt code so the transmission statistic is non-trivial.
            qpos[:, 7] = feature[:, 0] / JOINT_SCALE
            samples.append({"qpos": qpos})
        return samples

    @staticmethod
    def to_qpos(sample):
        return sample["qpos"]


def window_audit(window, features, hashes):
    exact = np.asarray(features)
    accepted_before = e23.EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]
    history_start = e23.EXPECTED_GLOBAL_HISTORY_START_FRAMES[window]
    transcript_frames = e23.EXPECTED_TRANSCRIPT_FRAMES[window]
    input_hashes = [] if window == 0 else [
        x._raw_row_sha256(exact[row, history_start:accepted_before])
        for row in range(len(exact))
    ]
    return {
        "shape": [len(exact), 13, 4],
        "row_sha256": list(hashes),
        "window_index": window,
        "global_history_start_frame": history_start,
        "accepted_transcript_frames_before": accepted_before,
        "input_history_frames": e23.EXPECTED_INPUT_HISTORY_FRAMES[window],
        "model_num_frames": e23.EXPECTED_MODEL_NUM_FRAMES[window],
        "transcript_frames": transcript_frames,
        "input_history_row_sha256": input_hashes,
        "returned_input_history_row_sha256": list(input_hashes),
        "returned_history_reconstruction_exact": [True] * len(exact),
        "returned_history_reconstruction_max_abs": [0.0] * len(exact),
        "stable_transcript_row_sha256": [
            x._raw_row_sha256(exact[row, :transcript_frames]) for row in range(len(exact))
        ],
    }


def valid_audit(features, plan):
    return [
        window_audit(
            window, features,
            [hashlib.sha256(f"{item['seed']}:{window}".encode()).hexdigest() for item in plan],
        )
        for window in range(x.N_WINDOWS)
    ]


class SqueezePrefixMismatchRunner(FakeRunner):
    @staticmethod
    def mutate(features, audit):
        # Local row 2 is the first seed's squeeze_52; break its WALK prefix inside window 0.
        features[2, 10, 3] = 1.0
        return features, valid_audit_from(features, audit)


class ReplayedWindowRunner(FakeRunner):
    @staticmethod
    def mutate(features, audit):
        audit[1]["row_sha256"] = list(audit[0]["row_sha256"])
        return features, audit


class SecondChunkFailureRunner(FakeRunner):
    def generate_prompt_schedule(self, *args, **kwargs):
        if self.calls == 1:
            self.calls += 1
            raise RuntimeError("synthetic second-chunk failure")
        return super().generate_prompt_schedule(*args, **kwargs)


class QposPrefixMismatchRunner(FakeRunner):
    def decode_features(self, features):
        samples = super().decode_features(features)
        for sample in samples:
            sample["qpos"] = np.asarray(sample["qpos"], dtype=np.float64)
        # Local row 3 is the first seed's step_52; a float32 downcast would erase this.
        samples[3]["qpos"][0, 5] = 1e-50
        return samples


def valid_audit_from(features, audit):
    return [window_audit(window, features, item["row_sha256"])
            for window, item in enumerate(audit)]


def clean_code_state(_repo):
    return {
        "commit": "a" * 40,
        "dirty": False,
        "status": [],
        "tracked_diff_sha256": "b" * 64,
    }


def passing_host_report():
    return hg.host_resource_report(
        vram_fn=lambda: {"free_mib": 15000, "total_mib": 16303, "used_mib": 1303, "error": None},
        ram_fn=lambda: {"available_mib": 24000, "total_mib": 31000, "error": None},
        isaac_fn=lambda: [],
        require_no_isaac=False,
    )


def failing_host_report():
    return hg.host_resource_report(
        vram_fn=lambda: {"free_mib": 4277, "total_mib": 16303, "used_mib": 11532, "error": None},
        ram_fn=lambda: {"available_mib": 7216, "total_mib": 31000, "error": None},
        isaac_fn=lambda: [{"pid": 7, "args": "env_isaaclab"}],
        require_no_isaac=False,
    )


def raising_host_gate():
    raise x.HostGateRefusal("synthetic host gate failure", failing_host_report())


def step_event(route, onset):
    return {
        "present": True,
        "missing_reason": None,
        "frame": int(onset + 12),
        "latency_frames": 12,
        "latency_s": 12 / x.FPS,
        "side": "left",
        "profile_x_m": float(route[onset, 1] + 0.9),
        "foot_x_m": float(route[onset, 1] + 0.9),
        "whole_body_clearance_m": 0.08,
        "foot_bottom_clearance_m": 0.10,
        "max_profile_height_m": 0.08,
    }


def missing_step_event():
    return {"present": False, "missing_reason": "fake_no_event", "max_profile_height_m": 0.0}


def step_event_for_step_code(_body, qpos, route, onset):
    code = float(np.asarray(qpos)[onset, 0])
    return step_event(route, onset) if 5.0 <= code < 15.0 else missing_step_event()


def no_step_event(_body, _qpos, _route, _onset):
    return missing_step_event()


def step_event_everywhere(_body, _qpos, route, onset):
    return step_event(route, onset)


SIGNATURE_LOCKS = {
    "heading_sidle": (x.SUSTAINED_MIN_FRAMES, "threshold_deg", x.HEADING_SIDLE_MIN_DEVIATION_DEG),
    "lateral_excursion": (x.SUSTAINED_MIN_FRAMES, "threshold_m", x.LATERAL_EXCURSION_MIN_M),
    "foot_crossing": (x.FOOT_CROSSING_MIN_FRAMES, "threshold_m", x.FOOT_CROSSING_MIN_M),
}


def sidestep_record(onset, present=(), latency=12):
    names = [name for name in x.SIDESTEP_SIGNATURES if name in present]
    signatures = {}
    for name in x.SIDESTEP_SIGNATURES:
        min_run, threshold_key, threshold = SIGNATURE_LOCKS[name]
        flag = name in names
        signatures[name] = {
            "present": flag,
            "min_run_frames": min_run,
            "longest_run_frames": min_run + 2 if flag else 0,
            "qualifying_run_count": int(flag),
            "first_frame": int(onset + latency) if flag else None,
            "first_latency_frames": int(latency) if flag else None,
            "flagged_frame_count": min_run + 2 if flag else 0,
            threshold_key: threshold,
        }
    return {
        "present": bool(names),
        "composite_rule": "any",
        "present_signatures": names,
        "first_signature_frame": int(onset + latency) if names else None,
        "first_signature_latency_frames": int(latency) if names else None,
        "signatures": signatures,
    }


def sidestep_for_squeeze_code(_body, qpos, _route, onset):
    present = ("heading_sidle",) if float(np.asarray(qpos)[onset, 0]) >= 15.0 else ()
    return sidestep_record(onset, present)


def sidestep_only_from_start(_body, qpos, _route, onset):
    code = float(np.asarray(qpos)[onset, 0])
    present = ("lateral_excursion", "foot_crossing") if code >= 15.0 and onset == 0 else ()
    return sidestep_record(onset, present)


def sidestep_two_delayed_seeds(_body, qpos, _route, onset):
    code = float(np.asarray(qpos)[onset, 0])
    seed_offset = float(np.asarray(qpos)[onset, 1])
    if code < 15.0:
        return sidestep_record(onset, ())
    if onset == 0 or seed_offset in (0.0, 1.0):
        return sidestep_record(onset, ("heading_sidle",))
    return sidestep_record(onset, ())


def no_sidestep(_body, _qpos, _route, onset):
    return sidestep_record(onset, ())


def sidestep_everywhere(_body, _qpos, _route, onset):
    return sidestep_record(onset, ("foot_crossing",))


def out_of_window_sidestep(_body, _qpos, _route, onset):
    return sidestep_record(onset, ("heading_sidle",), latency=x.POST_ONSET_FRAMES)


def fake_exact_box(_body, _qpos, obstacle_x):
    collision_free = {f"{height:g}": bool(height <= 0.08) for height in x.GRADED_HEIGHTS_M}
    return {
        "obstacle_x_m": float(obstacle_x),
        "obstacle_depth_m": e23.OBSTACLE_DEPTH_M,
        "traversal": {
            "traversed": True,
            "criterion": "both physical feet cross before-over-after",
            "per_side": {
                side: {"crossed_before_over_after": True} for side in ("left", "right")
            },
        },
        "max_box_height_collision_free_lower_bound_m": 0.08,
        "max_box_height_lower_bound_m": 0.08,
        "collision_free": collision_free,
        "clears": dict(collision_free),
    }


def fake_metrics(_body, _qpos, _route):
    return {"progress_ratio": 1.0, "route_path_mae_m": 0.0, "max_foot_floor_penetration_m": 0.0}


def campaign_kwargs(
    output: Path,
    *,
    sidestep_fn=sidestep_for_squeeze_code,
    event_fn=no_step_event,
    host_gate_fn=passing_host_report,
):
    return {
        "out": output,
        "runner_factory": lambda: FakeRunner(output),
        "body": object(),
        "host_gate_fn": host_gate_fn,
        "code_state_fn": clean_code_state,
        "source_hashes_fn": lambda _repo: {
            x.PROTOCOL_PATH: "d" * 64,
            "experiments/exp023b_prompt_switch_control.py": "c" * 64,
            "experiments/exp023_prompt_handoff.py": "e" * 64,
        },
        "generator_identity_fn": lambda _runner: {"generator": "fake"},
        "generator_identity_validator_fn": lambda value: dict(value),
        "runtime_identity_fn": lambda: {"runtime": "fake"},
        "physical_identity_fn": lambda: {"physical": "fake"},
        "prompt_identity_fn": lambda _runner, _path: {"prompts": "fake"},
        "pin_validator_fn": lambda _generator, _runtime, _physical: None,
        "channel_usage_fn": lambda _runner, _spec: {"root_pos": 400},
        "event_detector_fn": event_fn,
        "sidestep_detector_fn": sidestep_fn,
        "exact_box_fn": fake_exact_box,
        "motion_metrics_fn": fake_metrics,
    }


# --- locked plan -----------------------------------------------------------------------


def test_locked_plan_is_seed_paired_and_hash_stable():
    plan = x.locked_row_plan()
    chunks = x.locked_chunk_plan(plan)
    assert len(plan) == 32
    assert x.SEEDS == tuple(range(4640, 4648))
    assert x.ARMS == ("all_walk", "squeeze_0", "squeeze_52", "step_52")
    assert x.SQUEEZE == "A person steps sideways through a narrow gap."
    assert {(row["seed"], row["arm"]) for row in plan} == {
        (seed, arm) for seed in x.SEEDS for arm in x.ARMS
    }
    assert [row["row_index"] for row in plan] == list(range(32))
    assert [chunk["seeds"] for chunk in chunks] == [
        [4640, 4641], [4642, 4643], [4644, 4645], [4646, 4647]
    ]
    for chunk in chunks:
        # Every same-seed comparison (all four arms) sits inside one B=8 call.
        for seed in chunk["seeds"]:
            arms = {row["arm"] for row in chunk["rows"] if row["seed"] == seed}
            assert arms == set(x.ARMS)
        assert len(chunk["row_indices"]) == 8
    schedules = {row["arm"]: tuple(row["prompt_schedule"]) for row in plan[:4]}
    assert schedules == {
        "all_walk": (x.WALK,) * 4,
        "squeeze_0": (x.SQUEEZE,) * 4,
        "squeeze_52": (x.WALK, x.SQUEEZE, x.SQUEEZE, x.SQUEEZE),
        "step_52": (x.WALK, x.STEP, x.STEP, x.STEP),
    }
    assert {row["arm"]: row["onset_frame"] for row in plan[:4]} == {
        "all_walk": None, "squeeze_0": 0, "squeeze_52": 52, "step_52": 52
    }
    assert e23.cal._json_hash(plan) == LOCKED_ROW_PLAN_SHA256
    assert e23.cal._json_hash(x.locked_row_plan()) == LOCKED_ROW_PLAN_SHA256


def test_locked_thresholds_match_protocol_text():
    assert x.SUSTAINED_MIN_FRAMES == 13 and x.SUSTAINED_MIN_FRAMES / x.FPS >= 0.5
    assert (x.SUSTAINED_MIN_FRAMES - 1) / x.FPS < 0.5
    assert x.HEADING_SIDLE_MIN_DEVIATION_DEG == 45.0
    assert x.LATERAL_EXCURSION_MIN_M == 0.15
    assert x.FOOT_CROSSING_MIN_M == 0.10 and x.FOOT_CROSSING_MIN_FRAMES == 3
    assert x.MIN_SQUEEZE0_SIDESTEPS == 4
    assert x.MAX_ALL_WALK_SEEDS_WITH_ANY_SIDESTEP == 1
    assert x.MAX_ALL_WALK_SEEDS_WITH_ANY_STEP_EVENT == 1
    assert x.PROTOCOL_PATH in x.SOURCE_FILES
    assert "experiments/exp023_prompt_handoff.py" in x.SOURCE_FILES
    assert "scene2motion/host_gate.py" in x.SOURCE_FILES


# --- sidestep detector -----------------------------------------------------------------


def synthetic_walk_qpos(route, *, yaw_rad=None, lateral_m=None, height_m=0.79):
    qpos = np.zeros((x.N_FRAMES, 36), dtype=float)
    qpos[:, 0] = route[:, 1]
    qpos[:, 1] = route[:, 0] if lateral_m is None else route[:, 0] + lateral_m
    qpos[:, 2] = height_m
    yaw = np.zeros(x.N_FRAMES) if yaw_rad is None else np.asarray(yaw_rad, dtype=float)
    qpos[:, 3] = np.cos(yaw / 2.0)
    qpos[:, 6] = np.sin(yaw / 2.0)
    return qpos


@pytest.fixture(scope="module")
def body():
    return G1Body(None)


@pytest.fixture(scope="module")
def route():
    return x.route_xz()


def test_axis_convention_forward_x_lateral_y_yaw_about_z(route):
    # Route column 1 is forward (0 -> 7.2 m) and column 0 lateral (all zeros).
    assert route[0, 1] == 0.0 and route[-1, 1] == pytest.approx(7.2)
    assert np.all(route[:, 0] == 0.0)
    assert np.allclose(x.route_travel_heading(route), 0.0)
    quat = np.array([[np.cos(0.25), 0.0, 0.0, np.sin(0.25)]])
    assert x.yaw_from_quaternion(quat) == pytest.approx([0.5])
    assert x._wrap_angle(np.array([np.pi + 0.1])) == pytest.approx([-np.pi + 0.1])
    with pytest.raises(ValueError, match="stationary"):
        x.route_travel_heading(np.zeros((10, 2)))


def test_pure_forward_walk_has_no_signature_and_left_foot_is_at_plus_y(body, route):
    qpos = synthetic_walk_qpos(route)
    for onset in x.CONTROL_ONSETS:
        record = x.detect_sidestep(body, qpos, route, onset)
        x._validated_sidestep_record(record, onset)
        assert record["present"] is False
        assert record["present_signatures"] == []
        assert record["first_signature_frame"] is None
        assert record["analysis_window_start_frame"] == onset
        assert record["analysis_window_end_frame"] == onset + 95
        signatures = record["signatures"]
        assert signatures["heading_sidle"]["max_abs_deviation_deg"] == pytest.approx(0.0)
        assert signatures["lateral_excursion"]["max_abs_excursion_m"] == pytest.approx(0.0)
        assert signatures["foot_crossing"]["separation_at_onset_m"] > 0.15
        assert signatures["foot_crossing"]["min_separation_m"] > 0.15


def test_sustained_ninety_degree_heading_run_fires_only_the_heading_signature(body, route):
    yaw = np.zeros(x.N_FRAMES)
    yaw[60:101] = np.pi / 2.0  # 41 frames >= 13
    qpos = synthetic_walk_qpos(route, yaw_rad=yaw)
    record = x.detect_sidestep(body, qpos, route, 52)
    x._validated_sidestep_record(record, 52)
    assert record["present"] is True
    assert record["present_signatures"] == ["heading_sidle"]
    heading = record["signatures"]["heading_sidle"]
    assert heading["first_frame"] == 60
    assert heading["first_latency_frames"] == 8
    assert heading["longest_run_frames"] == 41
    assert heading["max_abs_deviation_deg"] == pytest.approx(90.0)
    assert record["first_signature_frame"] == 60
    # The foot-crossing signature is evaluated in the pelvis frame, so a turned body whose
    # feet stay in nominal stance is not a crossing even though the world-Y separation vanishes.
    crossing = record["signatures"]["foot_crossing"]
    assert crossing["present"] is False
    assert crossing["min_separation_m"] > 0.15
    assert abs(crossing["min_world_lateral_separation_m"]) < 0.05
    assert record["signatures"]["lateral_excursion"]["present"] is False
    # Also visible from onset 0 (frames 60..100 lie inside 0..95).
    assert x.detect_sidestep(body, qpos, route, 0)["present_signatures"] == ["heading_sidle"]


def test_short_heading_run_below_half_second_does_not_count(body, route):
    yaw = np.zeros(x.N_FRAMES)
    yaw[60:72] = np.pi / 2.0  # 12 frames = 0.48 s < 0.5 s
    record = x.detect_sidestep(body, synthetic_walk_qpos(route, yaw_rad=yaw), route, 52)
    assert record["present"] is False
    assert record["signatures"]["heading_sidle"]["longest_run_frames"] == 12
    assert record["signatures"]["heading_sidle"]["flagged_frame_count"] == 12


def test_sustained_lateral_drift_fires_only_the_excursion_signature(body, route):
    lateral = np.zeros(x.N_FRAMES)
    lateral[70:] = 0.30
    record = x.detect_sidestep(body, synthetic_walk_qpos(route, lateral_m=lateral), route, 52)
    x._validated_sidestep_record(record, 52)
    assert record["present_signatures"] == ["lateral_excursion"]
    excursion = record["signatures"]["lateral_excursion"]
    assert excursion["first_frame"] == 70
    assert excursion["max_abs_excursion_m"] == pytest.approx(0.30)
    assert excursion["signed_excursion_at_peak_m"] == pytest.approx(0.30)
    assert excursion["route_lateral_reference_m"] == 0.0
    assert excursion["root_lateral_at_onset_m"] == 0.0
    assert record["signatures"]["heading_sidle"]["present"] is False
    assert record["signatures"]["foot_crossing"]["present"] is False

    brief = np.zeros(x.N_FRAMES)
    brief[70:80] = 0.30  # 10 frames < 13
    assert x.detect_sidestep(
        body, synthetic_walk_qpos(route, lateral_m=brief), route, 52)["present"] is False
    small = np.full(x.N_FRAMES, 0.14)  # below the 0.15 m threshold everywhere
    assert x.detect_sidestep(
        body, synthetic_walk_qpos(route, lateral_m=small), route, 52)["present"] is False


def crossing_feet_series(crossed_frames, *, depth_m=0.15):
    def series(_body, segment, _fps):
        length = len(segment)
        left = np.full(length, 0.12)
        right = np.full(length, -0.12)
        left[crossed_frames] = -0.12 - depth_m
        right[crossed_frames] = -0.12
        forward = np.asarray(segment)[:, 0]
        return {
            "left": {"forward_representative_m": forward, "lateral_representative_m": left},
            "right": {"forward_representative_m": forward, "lateral_representative_m": right},
        }
    return series


def test_crossing_feet_fire_only_the_crossing_signature(route):
    qpos = synthetic_walk_qpos(route)
    record = x.detect_sidestep(
        object(), qpos, route, 52, foot_series_fn=crossing_feet_series(slice(20, 26)))
    x._validated_sidestep_record(record, 52)
    assert record["present_signatures"] == ["foot_crossing"]
    crossing = record["signatures"]["foot_crossing"]
    assert crossing["first_frame"] == 72
    assert crossing["longest_run_frames"] == 6
    assert crossing["min_separation_m"] == pytest.approx(-0.15)
    assert crossing["separation_at_onset_m"] == pytest.approx(0.24)
    assert record["first_signature_frame"] == 72

    brief = x.detect_sidestep(
        object(), qpos, route, 52, foot_series_fn=crossing_feet_series(slice(20, 22)))
    assert brief["present"] is False
    assert brief["signatures"]["foot_crossing"]["longest_run_frames"] == 2
    shallow = x.detect_sidestep(
        object(), qpos, route, 52,
        foot_series_fn=crossing_feet_series(slice(20, 26), depth_m=0.09))
    assert shallow["present"] is False


def test_detector_rejects_invalid_inputs(body, route):
    qpos = synthetic_walk_qpos(route)
    with pytest.raises(ValueError, match="invalid clip"):
        x.detect_sidestep(body, qpos, route, 104)  # not a locked control onset
    with pytest.raises(ValueError, match="invalid clip"):
        x.detect_sidestep(body, qpos[:150], route, 0)
    with pytest.raises(ValueError, match="invalid clip"):
        x.detect_sidestep(body, qpos, route[:100], 0)

    def broken_feet(_body, segment, _fps):
        return {"left": {"forward_representative_m": np.zeros(3),
                         "lateral_representative_m": np.zeros(3)}}
    with pytest.raises(ValueError, match="lacks right|invalid for left"):
        x.detect_sidestep(object(), qpos, route, 0, foot_series_fn=broken_feet)


@pytest.mark.skipif(not ARCHIVED_EXP023_QPOS.is_file(), reason="EXP-023 archive not present")
def test_archived_exp023_walk_clip_has_no_sidestep_signature(body, route):
    with np.load(ARCHIVED_EXP023_QPOS) as archive:
        walk = np.asarray(archive["s4500_all_walk"], dtype=float)[:x.N_FRAMES]
        step = np.asarray(archive["s4500_step_0"], dtype=float)[:x.N_FRAMES]
    assert walk[-1, 0] > 7.0 and np.abs(walk[:, 1]).max() < 0.05  # forward +X, lateral +Y
    for onset in x.CONTROL_ONSETS:
        record = x.detect_sidestep(body, walk, route, onset)
        x._validated_sidestep_record(record, onset)
        assert record["present"] is False
        signatures = record["signatures"]
        assert signatures["heading_sidle"]["max_abs_deviation_deg"] < 15.0
        assert signatures["lateral_excursion"]["max_abs_excursion_m"] < 0.03
        assert signatures["foot_crossing"]["min_separation_m"] > 0.0
        assert x.detect_sidestep(body, step, route, onset)["present"] is False


def test_sidestep_validator_rejects_inconsistent_records():
    good = sidestep_record(52, ("heading_sidle",))
    x._validated_sidestep_record(good, 52)

    wrong_rule = {**good, "composite_rule": "all"}
    with pytest.raises(ValueError, match="composite rule"):
        x._validated_sidestep_record(wrong_rule, 52)

    disagreeing = {**good, "present": False}
    with pytest.raises(ValueError, match="composite disagrees"):
        x._validated_sidestep_record(disagreeing, 52)

    outside = sidestep_record(52, ("lateral_excursion",), latency=x.POST_ONSET_FRAMES)
    with pytest.raises(ValueError, match="outside the locked 96-frame"):
        x._validated_sidestep_record(outside, 52)

    unsustained = json.loads(json.dumps(good))
    unsustained["signatures"]["heading_sidle"]["longest_run_frames"] = 5
    with pytest.raises(ValueError, match="lacks a sustained run"):
        x._validated_sidestep_record(unsustained, 52)

    loosened = json.loads(json.dumps(good))
    loosened["signatures"]["lateral_excursion"]["threshold_m"] = 0.05
    with pytest.raises(ValueError, match="locked threshold"):
        x._validated_sidestep_record(loosened, 52)

    absent_with_run = json.loads(json.dumps(sidestep_record(52, ())))
    absent_with_run["signatures"]["foot_crossing"]["longest_run_frames"] = 4
    with pytest.raises(ValueError, match="reports a sustained run"):
        x._validated_sidestep_record(absent_with_run, 52)


# --- handoff transmission and rates ----------------------------------------------------


def test_handoff_transmission_is_rms_over_post_switch_joints():
    base = np.zeros((208, 36))
    clips = {f"s4640_{arm}": base.copy() for arm in x.ARMS}
    clips["s4640_squeeze_52"][52:, 7] = 2.0
    clips["s4640_step_52"][52:, 7] = 1.0
    clips["s4640_squeeze_0"][:, 7] = 2.0
    record = x.handoff_transmission(clips, 4640)
    assert record["frames"] == [52, 147]
    assert record["joint_dofs"] == 29
    rms = record["joint_rms_rad"]
    assert rms["squeeze_52_vs_all_walk"] == pytest.approx(np.sqrt(4.0 / 29.0))
    assert rms["step_52_vs_all_walk"] == pytest.approx(np.sqrt(1.0 / 29.0))
    assert rms["squeeze_0_vs_all_walk"] == pytest.approx(np.sqrt(4.0 / 29.0))
    assert rms["step_52_vs_squeeze_52"] == pytest.approx(np.sqrt(1.0 / 29.0))
    assert record["prefix_joint_rms_rad_frames_0_51"] == {
        "squeeze_52_vs_all_walk": 0.0, "step_52_vs_all_walk": 0.0
    }
    clips["s4640_step_52"] = np.zeros((208, 20))
    with pytest.raises(ValueError, match="different joint widths"):
        x.handoff_transmission(clips, 4640)


def test_rate_record_uses_wilson_interval():
    record = x._rate_record(4, 8)
    assert record["rate"] == 0.5 and record["missing"] == 4
    assert record["wilson95"] == pytest.approx(list(wilson_interval(4, 8)))
    assert x._rate_record(0, 8)["wilson95"][0] == 0.0
    assert x._two_by_two([True, True, False, False], [True, False, True, False]) == {
        "both": 1, "first_only": 1, "second_only": 1, "neither": 1, "planned": 4
    }


# --- fork validator --------------------------------------------------------------------


def test_fork_validator_accepts_prefix_sharing_features_and_records_divergence():
    plan = x.locked_row_plan()
    features = np.zeros((32, 208, 3), dtype=np.float32)
    for row in plan:
        onset = row["onset_frame"]
        if row["arm"] == "squeeze_0":
            features[row["row_index"], :, 0] = 2.0
        elif onset is not None:
            features[row["row_index"], onset:, 0] = 1.0
    audit = valid_audit(features, plan)
    result = x.validate_noise_and_feature_forks(features, audit, plan)
    assert result["feature_prefixes_exact"] is True
    assert result["distinct_seeds_differ"] is True
    assert result["fork_frames"] == {"squeeze_52": 52, "step_52": 52}
    first = result["feature_forks"][0]
    assert first["squeeze_52"]["first_divergence_frame_from_all_walk"] == 52
    assert first["step_52"]["first_divergence_frame_from_all_walk"] == 52
    assert first["squeeze_0"]["first_divergence_frame_from_all_walk"] == 0
    assert len(result["paired_noise"]) == 32
    assert all(item["all_arms_equal"] for item in result["paired_noise"])


def test_fork_validator_rejects_broken_prefixes_replays_and_seed_collisions():
    plan = x.locked_row_plan()
    features = np.zeros((32, 208, 3), dtype=np.float32)
    broken = features.copy()
    broken[2, 10, 0] = 1.0  # row 2 = seed 4640 squeeze_52 inside the WALK prefix
    with pytest.raises(ValueError, match="squeeze_52 feature prefix differs"):
        x.validate_noise_and_feature_forks(broken, valid_audit(broken, plan), plan)
    broken = features.copy()
    broken[3, 51, 0] = 1.0  # row 3 = seed 4640 step_52, last prefix frame
    with pytest.raises(ValueError, match="step_52 feature prefix differs"):
        x.validate_noise_and_feature_forks(broken, valid_audit(broken, plan), plan)

    audit = valid_audit(features, plan)
    audit[1]["row_sha256"] = list(audit[0]["row_sha256"])
    with pytest.raises(ValueError, match="latent replay detected"):
        x.validate_noise_and_feature_forks(features, audit, plan)

    audit = valid_audit(features, plan)
    audit[2]["row_sha256"][4:8] = audit[2]["row_sha256"][0:4]  # seed 4641 copies 4640
    with pytest.raises(ValueError, match="distinct seeds collided"):
        x.validate_noise_and_feature_forks(features, audit, plan)

    audit = valid_audit(features, plan)
    audit[1]["row_sha256"][1] = audit[1]["row_sha256"][5]
    with pytest.raises(ValueError, match="unequal noise"):
        x.validate_noise_and_feature_forks(features, audit, plan)

    audit = valid_audit(features, plan)
    audit[2]["input_history_row_sha256"][7] = "f" * 64
    with pytest.raises(ValueError, match="visible transcript suffix"):
        x.validate_noise_and_feature_forks(features, audit, plan)

    wrong_arm_plan = [dict(row, arm="step_104") if row["arm"] == "step_52" else row
                      for row in plan]
    with pytest.raises(ValueError, match="complete local seed/arm product"):
        x.validate_noise_and_feature_forks(features, valid_audit(features, plan), wrong_arm_plan)


def test_qpos_fork_validator_requires_exact_prefixes():
    clips = {f"s{seed}_{arm}": np.zeros((208, 36)) for seed in x.SEEDS for arm in x.ARMS}
    rows = x._validate_qpos_forks(clips)
    assert len(rows) == 8
    assert rows[0]["squeeze_52"]["first_divergence_frame_from_all_walk"] is None
    clips["s4643_step_52"][51, 9] = 1e-9
    with pytest.raises(ValueError, match="step_52 decoded qpos prefix differs for seed 4643"):
        x._validate_qpos_forks(clips)


# --- host gate -------------------------------------------------------------------------


def test_evaluate_host_gate_measures_once_and_binds_isaac_observation():
    calls = {"vram": 0, "ram": 0, "isaac": 0}

    def vram():
        calls["vram"] += 1
        return {"free_mib": 15000, "total_mib": 16303, "used_mib": 1303, "error": None}

    def ram():
        calls["ram"] += 1
        return {"available_mib": 24000, "total_mib": 31000, "error": None}

    def isaac():
        calls["isaac"] += 1
        return [{"pid": 9, "args": "env_isaaclab train"}]

    report = x.evaluate_host_gate(vram_fn=vram, ram_fn=ram, isaac_fn=isaac)
    assert report["pass"] is True
    assert report["thresholds"]["require_no_concurrent_isaac"] is False
    assert report["thresholds"]["min_free_vram_mib"] == 4 * 1024
    assert report["concurrent_isaac_processes_informational"] == [
        {"pid": 9, "args": "env_isaaclab train"}
    ]
    assert calls == {"vram": 1, "ram": 1, "isaac": 1}

    with pytest.raises(x.HostGateRefusal, match="vram, ram") as info:
        x.evaluate_host_gate(
            vram_fn=lambda: {"free_mib": 3277, "total_mib": 16303, "used_mib": 12532,
                             "error": None},
            ram_fn=lambda: {"available_mib": 7216, "total_mib": 31000, "error": None},
            isaac_fn=isaac,
        )
    assert info.value.report["pass"] is False
    assert info.value.report["vram"]["free_mib"] == 3277
    assert info.value.report["checks"] == {"vram": False, "ram": False, "no_isaac": True}


def test_host_gate_failure_leaves_output_untouched(tmp_path):
    output = tmp_path / "gated"
    with pytest.raises(x.HostGateRefusal, match="synthetic host gate failure"):
        x.run_campaign(**campaign_kwargs(output, host_gate_fn=raising_host_gate))
    assert not output.exists()

    with pytest.raises(x.HostGateRefusal, match="does not pass"):
        x.run_campaign(**campaign_kwargs(output, host_gate_fn=failing_host_report))
    assert not output.exists()

    nested = tmp_path / "existing_empty"
    nested.mkdir()
    with pytest.raises(x.HostGateRefusal):
        x.run_campaign(**campaign_kwargs(nested, host_gate_fn=raising_host_gate))
    assert list(nested.iterdir()) == []


def test_nonempty_output_is_refused_before_the_host_gate(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "stale.txt").write_text("x")
    gate_calls = []

    def gate():
        gate_calls.append(1)
        return passing_host_report()

    with pytest.raises(x.PromptSwitchAbort, match="nonempty"):
        x.run_campaign(**campaign_kwargs(output, host_gate_fn=gate))
    assert gate_calls == []
    assert [path.name for path in output.iterdir()] == ["stale.txt"]


def test_main_exits_nonzero_and_prints_report_on_host_gate_failure(tmp_path, capsys):
    output = tmp_path / "main_gated"
    rc = x.main(["--out", str(output)], host_gate_fn=raising_host_gate)
    assert rc == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "host_gate_failed"
    assert printed["output_untouched"] is True
    assert printed["host_resource_gate"]["pass"] is False
    assert printed["host_resource_gate"]["ram"]["available_mib"] == 7216
    assert not output.exists()


def test_dry_run_prints_plan_and_gate_without_writing(tmp_path, capsys):
    output = tmp_path / "dry"
    rc = x.main(["--out", str(output), "--dry-run"], host_gate_fn=raising_host_gate)
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dry_run"] is True and printed["nothing_written"] is True
    assert printed["row_plan_sha256"] == LOCKED_ROW_PLAN_SHA256
    assert len(printed["row_plan"]) == 32
    assert [chunk["seeds"] for chunk in printed["chunk_plan"]] == [
        [4640, 4641], [4642, 4643], [4644, 4645], [4646, 4647]
    ]
    assert all("rows" not in chunk for chunk in printed["chunk_plan"])
    assert printed["host_gate_pass"] is False
    assert printed["host_resource_gate"]["vram"]["free_mib"] == 4277
    assert printed["protocol"]["path"] == x.PROTOCOL_PATH
    assert not output.exists()

    rc = x.main(["--dry-run"], host_gate_fn=passing_host_report)
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["host_gate_pass"] is True


# --- campaign --------------------------------------------------------------------------


def test_complete_campaign_accounting_gates_and_transmits_verdict(tmp_path):
    output = tmp_path / "exp023b"
    receipt = x.run_campaign(**campaign_kwargs(output))

    assert receipt["status"] == "complete"
    assert receipt["schema"] == x.SCHEMA_VERSION
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["spent_seeds"] == list(x.SEEDS)
    assert receipt["unlaunched_locked_seeds"] == []
    assert receipt["execution_mode"]["scientific_evidence_eligible"] is False
    assert "host_gate_fn" in receipt["execution_mode"]["dependency_injections"]
    assert receipt["execution_mode"]["pre_model_construction_evidence_guaranteed"] is True
    assert receipt["actual_ardy_samples"] == 32
    assert receipt["host_resource_gate"]["pass"] is True
    assert receipt["host_resource_gate"]["vram"]["free_mib"] == 15000
    assert receipt["campaign_design"]["host_gate"] == {
        "preset": "scene2motion.host_gate.ARDY_GENERATION_GATE",
        "min_free_vram_mib": 4 * 1024,
        "min_available_ram_mib": 8 * 1024,
        "require_no_concurrent_isaac": False,
    }
    assert receipt["campaign_design"]["row_plan_sha256"] == LOCKED_ROW_PLAN_SHA256
    assert receipt["campaign_design"]["predicted_box_centres_m"].keys() == {"0", "52"}
    assert receipt["provenance"]["protocol"] == {"path": x.PROTOCOL_PATH, "sha256": "d" * 64}
    assert receipt["provenance"]["source_sha256"]["experiments/exp023_prompt_handoff.py"] == (
        "e" * 64)
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
    assert all(chunk["status"] == "complete" for chunk in receipt["generation_chunks"].values())

    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 32
    assert {(row["seed"], row["arm"]) for row in rows} == {
        (seed, arm) for seed in x.SEEDS for arm in x.ARMS
    }
    walk_rows = [row for row in rows if row["arm"] == "all_walk"]
    assert all(row["sidestep"] is None and row["event"] is None for row in walk_rows)
    assert all(set(row["control_sidesteps"]) == {"0", "52"} for row in walk_rows)
    assert all(set(row["control_events"]) == {"0", "52"} for row in walk_rows)
    prompt_rows = [row for row in rows if row["arm"] != "all_walk"]
    assert all(row["control_sidesteps"] is None for row in prompt_rows)
    assert all(row["sidestep"]["composite_rule"] == "any" for row in prompt_rows)

    audit = receipt["causal_pairing_audit"]
    assert audit["feature_prefixes_exact"] is True
    assert audit["qpos_prefixes_exact"] is True
    assert audit["noise_fresh_across_windows"] is True
    assert audit["feature_forks"][0]["squeeze_52"]["first_divergence_frame_from_all_walk"] == 52
    assert audit["feature_forks"][0]["step_52"]["first_divergence_frame_from_all_walk"] == 52
    assert audit["feature_forks"][0]["squeeze_0"]["first_divergence_frame_from_all_walk"] == 0
    assert audit["qpos_forks"][0]["squeeze_52"]["first_divergence_frame_from_all_walk"] == 52

    summary = receipt["summary"]
    assert summary["sidestep_rates_missing_retained"]["squeeze_0"]["composite"] == {
        "present": 8, "planned": 8, "missing": 0, "rate": 1.0,
        "wilson95": pytest.approx(list(wilson_interval(8, 8))),
    }
    assert summary["sidestep_rates_missing_retained"]["squeeze_52"]["composite"]["present"] == 8
    assert summary["sidestep_rates_missing_retained"]["step_52"]["composite"]["present"] == 0
    assert summary["sidestep_rates_missing_retained"]["squeeze_52"]["signatures"][
        "heading_sidle"]["present"] == 8
    assert summary["step_event_rates_missing_retained"]["step_52"]["present"] == 0
    assert summary["all_walk_sidestep_specificity"]["seeds_with_any_signature_in_any_window"] == 0
    assert summary["all_walk_sidestep_specificity"]["planned_windows"] == 16
    assert summary["all_walk_step_specificity"]["seeds_with_any_event"] == 0
    assert summary["paired_counts"]["squeeze_0_sidestep_vs_squeeze_52_sidestep"]["both"] == 8
    assert len(summary["paired_seed_table"]) == 8
    assert summary["paired_seed_table"][0]["squeeze_52"]["sidestep_signatures"] == [
        "heading_sidle"]
    assert summary["prompt_relative_latency"]["squeeze_52"][
        "sidestep_first_signature_latency_frames"]["median"] == 12.0
    assert summary["fixed_box_rates"]["step_52"]["0.05"]["present"] == 8
    assert summary["fixed_box_rates"]["all_walk_matched_windows"]["52"]["0.12"]["rate"] == 0.0
    assert summary["route_fidelity_reported_not_enforced"]["squeeze_0"]["progress_ratio"][
        "median"] == 1.0
    transmission = summary["handoff_transmission"]
    assert len(transmission["per_seed"]) == 8
    assert transmission["joint_rms_rad"]["squeeze_52_vs_all_walk"]["median"] == pytest.approx(
        np.sqrt(4.0 / 29.0))
    assert transmission["joint_rms_rad"]["step_52_vs_all_walk"]["median"] == pytest.approx(
        np.sqrt(1.0 / 29.0))
    assert transmission["joint_rms_rad"]["squeeze_0_vs_all_walk"]["median"] == pytest.approx(
        np.sqrt(4.0 / 29.0))
    assert transmission["joint_rms_rad"]["step_52_vs_squeeze_52"]["median"] == pytest.approx(
        np.sqrt(1.0 / 29.0))
    assert transmission["prefix_joint_rms_rad_frames_0_51"]["squeeze_52_vs_all_walk"][
        "max"] == 0.0

    gates = receipt["measurement_gates"]
    assert gates["squeeze0_substrate"] == {
        "required_min_present": 4, "observed_present": 8, "planned": 8, "pass": True}
    assert gates["all_walk_sidestep_specificity"]["pass"] is True
    assert gates["all_walk_step_specificity"]["pass"] is True
    assert gates["all_pass"] is True
    decision = receipt["decision_rule"]
    assert decision["verdict"] == x.VERDICT_TRANSMITS
    assert decision["verdict_valid"] is True
    assert decision["inputs"] == {
        "squeeze_52_sidestep_present": 8, "squeeze_0_sidestep_present": 8,
        "step_52_step_event_present": 0, "planned": 8,
    }
    assert decision["step_52_replicates_exp023_zero_of_eight"] is True

    with np.load(output / "qpos.npz") as archive:
        assert len(archive.files) == 32
        assert archive["s4640_squeeze_52"].shape == (208, 36)
    with np.load(output / "features.npz") as archive:
        assert len(archive.files) == 32
    noise = json.loads((output / "noise_audit.json").read_text())
    assert len(noise) == 4 and all(len(window["rows"]) == 32 for window in noise)
    assert [window["input_history_frames"] for window in noise] == [0, 4, 4, 4]


def test_no_switch_verdict_when_only_from_start_squeeze_sidesteps(tmp_path):
    receipt = x.run_campaign(
        **campaign_kwargs(tmp_path / "no_switch", sidestep_fn=sidestep_only_from_start))
    summary = receipt["summary"]
    assert summary["sidestep_rates_missing_retained"]["squeeze_0"]["composite"]["present"] == 8
    assert summary["sidestep_rates_missing_retained"]["squeeze_52"]["composite"]["present"] == 0
    assert summary["sidestep_rates_missing_retained"]["squeeze_0"]["signatures"][
        "foot_crossing"]["present"] == 8
    assert summary["paired_counts"]["squeeze_0_sidestep_vs_squeeze_52_sidestep"] == {
        "both": 0, "first_only": 8, "second_only": 0, "neither": 0, "planned": 8}
    assert receipt["decision_rule"]["verdict"] == x.VERDICT_NO_SWITCH
    assert receipt["decision_rule"]["verdict_valid"] is True
    assert receipt["status"] == "complete"


def test_indeterminate_verdict_between_rules_and_step_replication_flag(tmp_path):
    receipt = x.run_campaign(**campaign_kwargs(
        tmp_path / "indeterminate",
        sidestep_fn=sidestep_two_delayed_seeds,
        event_fn=step_event_for_step_code,
    ))
    summary = receipt["summary"]
    assert summary["sidestep_rates_missing_retained"]["squeeze_52"]["composite"]["present"] == 2
    assert summary["step_event_rates_missing_retained"]["step_52"]["present"] == 8
    decision = receipt["decision_rule"]
    # step_52 = 8/8 on fresh seeds: EXP-023's 0/8 did not replicate, which the protocol ranks
    # above the transmission reading; squeeze_52 = 2/8 alone would have been indeterminate.
    assert decision["verdict"] == x.VERDICT_STEP_REPLICATION_FAILED
    assert decision["step_52_replicates_exp023_zero_of_eight"] is False
    assert [rule["satisfied"] for rule in decision["rules"]] == [True, False, False]
    assert decision["pooled_step_52_with_exp023"]["pooled_present_of_16"] == 8
    assert receipt["measurement_gates"]["all_pass"] is True


def test_substrate_refusal_keeps_all_rows_and_spends_seeds(tmp_path):
    output = tmp_path / "refused"
    with pytest.raises(x.PromptSwitchAbort, match="substrate gate failed"):
        x.run_campaign(**campaign_kwargs(output, sidestep_fn=no_sidestep))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "refused"
    assert receipt["schema"] == x.FAILURE_SCHEMA_VERSION
    assert receipt["refusal_reason"] == "squeeze0_substrate_gate_failed"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["actual_ardy_samples"] == 32
    assert receipt["decision_rule"]["verdict_valid"] is False
    assert len((output / "rows.jsonl").read_text().splitlines()) == 32
    with np.load(output / "qpos.npz") as archive:
        assert len(archive.files) == 32


def test_sidestep_specificity_refusal_is_not_relabelled_as_complete(tmp_path):
    output = tmp_path / "nonspecific"
    with pytest.raises(x.PromptSwitchAbort, match="sidestep-specificity gate failed"):
        x.run_campaign(**campaign_kwargs(output, sidestep_fn=sidestep_everywhere))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "refused"
    assert receipt["refusal_reason"] == "all_walk_sidestep_specificity_gate_failed"
    assert receipt["measurement_gates"]["squeeze0_substrate"]["pass"] is True
    assert receipt["measurement_gates"]["all_walk_sidestep_specificity"] == {
        "allowed_max_seeds_with_any_signature": 1,
        "observed_seeds_with_any_signature": 8,
        "planned": 8,
        "pass": False,
    }
    assert receipt["summary"]["all_walk_sidestep_specificity"]["seeds_with_signature"] == {
        "heading_sidle": 0, "lateral_excursion": 0, "foot_crossing": 8}


def test_step_specificity_refusal_carries_over_from_exp023(tmp_path):
    output = tmp_path / "step_nonspecific"
    with pytest.raises(x.PromptSwitchAbort, match="step-specificity gate failed"):
        x.run_campaign(**campaign_kwargs(output, event_fn=step_event_everywhere))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["refusal_reason"] == "all_walk_step_specificity_gate_failed"
    assert receipt["measurement_gates"]["all_walk_sidestep_specificity"]["pass"] is True
    assert receipt["summary"]["all_walk_step_specificity"]["window_events"] == 16


def test_injected_sidestep_outside_window_blocks_in_analysis(tmp_path):
    output = tmp_path / "bad_sidestep"
    with pytest.raises(x.PromptSwitchAbort, match="outside the locked 96-frame"):
        x.run_campaign(**campaign_kwargs(output, sidestep_fn=out_of_window_sidestep))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["failed_stage"] == "analysis"
    assert receipt["seeds_spent_and_must_not_be_reused"] is True


def test_broken_squeeze_prefix_blocks_after_durable_chunk_archive(tmp_path):
    output = tmp_path / "prefix"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: SqueezePrefixMismatchRunner(output)
    with pytest.raises(x.PromptSwitchAbort, match="squeeze_52 feature prefix differs"):
        x.run_campaign(**kwargs)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["failed_stage"] == "generation_chunk00_seeds4640_4641"
    assert receipt["spent_seeds"] == [4640, 4641]
    assert receipt["unlaunched_locked_seeds"] == [4642, 4643, 4644, 4645, 4646, 4647]
    assert receipt["query_accounting"]["trajectories_returned"] == 8
    with np.load(output / "features.npz") as archive:
        assert set(archive.files) == {
            f"s{seed}_{arm}" for seed in (4640, 4641) for arm in x.ARMS}
    raw_noise = json.loads((output / "noise_audit.json").read_text())
    assert raw_noise[0]["kind"] == "raw_chunk_runner_audit_before_validation"


def test_replayed_window_and_second_chunk_failure_keep_exact_ledgers(tmp_path):
    output = tmp_path / "replayed"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: ReplayedWindowRunner(output)
    with pytest.raises(x.PromptSwitchAbort, match="latent replay detected"):
        x.run_campaign(**kwargs)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["query_accounting"]["schedule_invocations_completed"] == 1

    output = tmp_path / "second_chunk"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: SecondChunkFailureRunner(output)
    with pytest.raises(x.PromptSwitchAbort, match="synthetic second-chunk failure"):
        x.run_campaign(**kwargs)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["sample_count_exact"] is False
    assert receipt["query_accounting"]["trajectories_launched"] == 16
    assert receipt["query_accounting"]["trajectories_returned"] == 8
    assert receipt["spent_seeds"] == [4640, 4641, 4642, 4643]
    assert receipt["generation_chunks"]["chunk01_seeds4642_4643"]["status"] == (
        "generation_exception_window_count_unknown")


def test_native_qpos_prefix_mismatch_cannot_hide_in_float32_archive(tmp_path):
    output = tmp_path / "qpos_mismatch"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: QposPrefixMismatchRunner(output)
    with pytest.raises(x.PromptSwitchAbort, match="step_52 decoded qpos prefix differs"):
        x.run_campaign(**kwargs)
    with np.load(output / "qpos.npz") as archive:
        assert archive["s4640_step_52"].dtype == np.float64
        assert archive["s4640_step_52"][0, 5] == 1e-50


def test_dirty_worktree_and_wrong_runner_contract_block_before_generation(tmp_path):
    output = tmp_path / "dirty"
    kwargs = campaign_kwargs(output)
    kwargs["code_state_fn"] = lambda _repo: {**clean_code_state(_repo), "dirty": True}
    with pytest.raises(x.PromptSwitchAbort, match="clean git worktree"):
        x.run_campaign(**kwargs)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked" and receipt["spent_seeds"] == []
    assert receipt["query_accounting"]["trajectories_launched"] == 0

    class V1Runner(FakeRunner):
        noise_stream_version = 1

    output = tmp_path / "v1"
    kwargs = campaign_kwargs(output)
    kwargs["runner_factory"] = lambda: V1Runner(output)
    with pytest.raises(x.PromptSwitchAbort, match="noise_stream_version == 2"):
        x.run_campaign(**kwargs)
    assert json.loads((output / "receipt.json").read_text())["spent_seeds"] == []


def test_decision_rule_step_replication_failure_precedes_transmission_reading():
    def _summary(squeeze0, squeeze52, step52):
        return {
            "sidestep_rates_missing_retained": {
                "squeeze_0": {"composite": {"present": squeeze0}},
                "squeeze_52": {"composite": {"present": squeeze52}},
            },
            "step_event_rates_missing_retained": {"step_52": {"present": step52}},
        }
    gates = {"all_pass": True}
    failed = x.evaluate_decision_rule(_summary(6, 5, 2), gates)
    assert failed["verdict"] == x.VERDICT_STEP_REPLICATION_FAILED
    assert failed["step_52_replicates_exp023_zero_of_eight"] is False
    assert failed["pooled_step_52_with_exp023"]["pooled_present_of_16"] == 2
    low, high = failed["pooled_step_52_with_exp023"]["wilson95"]
    assert low < 2 / 16 < high
    ok = x.evaluate_decision_rule(_summary(6, 5, 1), gates)
    assert ok["verdict"] == x.VERDICT_TRANSMITS
    assert ok["rules"][0]["verdict"] == x.VERDICT_STEP_REPLICATION_FAILED
    assert ok["rules"][0]["satisfied"] is False
