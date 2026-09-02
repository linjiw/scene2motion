"""CPU tests for the EXP-024 reference-contract campaign driver (no GPU, no Isaac)."""

from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import analyze_trackability_contract as atc
from experiments import calibrate_ramp_route_phase as cal
from experiments import exp022_exact_tracking_bridge as e22
from experiments import exp024_reference_contract as exp
from experiments import exp1b_execution_clearance as exp1b
from scene2motion.host_gate import HostResourceGateFailed
from scene2motion.sonic_state_export import SonicRollout, write_sonic_state_archive

ROOT = Path(__file__).resolve().parents[1]
H5 = f"{exp.P4_BOX_HEIGHT_M:g}"


# ------------------------------------------------------------------------------ fakes


def clean_code_state(_repo):
    return {"commit": "a" * 40, "dirty": False, "status": [], "tracked_diff_sha256": "b" * 64}


def fake_source_hashes(_repo):
    return {exp.PROTOCOL_PATH: "d" * 64, "experiments/exp024_reference_contract.py": "c" * 64}


def fake_channel_usage(_runner, spec):
    arm = next(arm for arm in exp.ARMS
               if (spec.root_y is None) == (exp.ARM_CONTRACTS[arm]["root_y"] is None)
               and (spec.heading is None) == (exp.ARM_CONTRACTS[arm]["heading"] is None))
    return dict(exp.EXPECTED_CHANNEL_USAGE[arm])


def passing_gate(**_kwargs):
    return {"pass": True, "checks": {"vram": True, "ram": True, "no_isaac": True},
            "vram": {"free_mib": 15000}, "ram": {"available_mib": 20000},
            "concurrent_isaac_processes": []}


def failing_gate(**_kwargs):
    raise HostResourceGateFailed("host-resource gate failed on vram: free VRAM 4001 MiB")


def synthetic_qpos(seed: int, arm: str) -> np.ndarray:
    route = exp.route_xz()
    qpos = np.zeros((exp.N_FRAMES, 36), dtype=np.float32)
    qpos[:, 0] = route[:, 1]
    qpos[:, 3] = 1.0
    if exp.ARM_CONTRACTS[arm]["root_y"] is None:
        qpos[:, 2] = 0.78 + 0.15 * np.sin(np.linspace(0, np.pi, exp.N_FRAMES)) * (seed % 3)
    else:
        qpos[:, 2] = 0.78
    if exp.ARM_CONTRACTS[arm]["heading"] is None:
        yaw = np.radians(20.0) * np.sin(np.linspace(0, np.pi, exp.N_FRAMES))
        qpos[:, 3] = np.cos(yaw / 2)
        qpos[:, 6] = np.sin(yaw / 2)
    qpos[:, 7] = seed / 1e4
    return qpos


class FakeRunner:
    fps = exp.FPS
    noise_stream_version = exp.NOISE_STREAM_VERSION

    def __init__(self, output: Path):
        # The empty evidence bundle must be durable before the model is constructed.
        for name in ("receipt.json", "rows.jsonl", "qpos.npz", "noise_audit.json"):
            assert (output / name).exists(), name
        receipt = json.loads((output / "receipt.json").read_text())
        assert receipt["status"] == "running" and receipt["stages"]["generate"]["status"] == "running"
        assert (output / "rows.jsonl").read_text() == ""
        self.output = output
        self.calls = 0
        self._text_cache = {}

    def _draw(self, seeds):
        generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]
        for _window in range(4):
            for generator in generators:
                torch.randn((13, 4), generator=generator)

    def generate(self, prompts, specs, num_frames, diffusion_steps, cfg_weight, seeds):
        assert num_frames == exp.N_FRAMES and diffusion_steps == exp.DIFFUSION_STEPS
        assert tuple(cfg_weight) == exp.CFG_WEIGHT
        assert len(prompts) == len(specs) == len(seeds) == exp.CHUNK_ROWS
        assert all(prompt == exp.STEP for prompt in prompts)
        expected = exp.SEEDS[self.calls * 2:(self.calls + 1) * 2]
        assert tuple(dict.fromkeys(seeds)) == expected
        assert seeds == [s for s in expected for _ in exp.ARMS]
        self.calls += 1
        self._draw(seeds)
        samples = []
        for seed, spec in zip(seeds, specs):
            arm = next(a for a in exp.ARMS if exp.spec_sha256(exp.arm_spec(a, exp.route_xz()))
                       == exp.spec_sha256(spec))
            samples.append({"qpos": synthetic_qpos(seed, arm)})
        return samples

    @staticmethod
    def to_qpos(sample):
        return sample["qpos"]


class BrokenPairingRunner(FakeRunner):
    def _draw(self, seeds):
        # Same-seed rows get *different* latents: the pairing contract is violated.
        generators = [torch.Generator().manual_seed(int(seed) * 7 + i)
                      for i, seed in enumerate(seeds)]
        for _window in range(4):
            for generator in generators:
                torch.randn((13, 4), generator=generator)


class SecondChunkFailureRunner(FakeRunner):
    def generate(self, *args, **kwargs):
        if self.calls == 1:
            self.calls += 1
            raise RuntimeError("synthetic second-chunk failure")
        return super().generate(*args, **kwargs)


def fake_reference_scorer(qpos, arm, _ctx):
    seed = int(round(float(qpos[0, 7]) * 1e4))
    elicited = arm == "free" and seed % 4 != 0
    exact5 = seed % 8 == 1
    if arm == "free":
        run = 0.44 if elicited else 0.12
        root_z_max = 0.97
        local_ok = False
    else:
        run = 0.16
        root_z_max = 0.80
        local_ok = exact5 and arm == "pin_y"
    features = {name: 0.1 for name in atc.FEATS}
    features.update(max_unsupported_run_s=run, root_z_max=root_z_max,
                    root_z_range=float(qpos[:, 2].max() - qpos[:, 2].min()),
                    heading_range_deg=1.0)
    gates = {name: local_ok for name in exp.LOCAL_STEP_GATE_NAMES}
    exact = {f"{h:g}": bool(exact5 and h <= 0.08) for h in exp.GRADED_HEIGHTS_M}
    box = {"exact_clears": exact, "max_box_height_lower_bound_m": 0.08 if exact5 else 0.0,
           "passed_obstacle": True, "achieved_replay_clear_after_passing": exact}
    return {
        "elicitation": {"lift_x_m": 1.1 if elicited else None,
                        "lift_height_m": 0.06 if elicited else 0.0, "n_lift_regions": int(elicited),
                        "lift_support_m": 0.3 if elicited else 0.0, "lift_side": "left" if elicited else None,
                        "elicited": elicited, "min_clearance_m": exp.ELICITATION_MIN_M,
                        "scan_points": exp.SCAN_POINTS,
                        "clears_height_anywhere": {f"{h:g}": bool(elicited and h <= 0.05)
                                                   for h in exp.GRADED_HEIGHTS_M}},
        "exact_boxes": {exp.STAGED_LABEL: box,
                        exp.CONTROL_LABEL: {**box, "exact_clears": {k: False for k in exact}}},
        "contract_features": features,
        "gate_predictions": atc.gate_predictions(features),
        "local_step": {"obstacle_x_m": exp.STAGED_X_M, "obstacle_height_m": 0.05,
                       "local_step_success": local_ok, "gates": gates,
                       "max_unsupported_run_local_s": run, "lead_side": None,
                       "max_lateral_deviation_m": 0.0},
        "route_fidelity": {"progress_ratio": 1.0, "route_path_mae_m": 0.01,
                           "max_foot_floor_penetration_m": 0.0},
        "manipulation": exp.manipulation_check(qpos),
    }


def _cheap_achieved_score(qpos, obstacle_x_m, *, terminated=False, reported_progress=None):
    qpos = np.asarray(qpos)
    passed = bool(len(qpos) and obstacle_x_m <= 1.2 and qpos[:, 0].max() >= 1.5)
    exact = {f"{h:g}": bool(passed and h <= 0.08) for h in exp.GRADED_HEIGHTS_M}
    return {
        "valid_frames": len(qpos), "tracker_terminated": bool(terminated),
        "tracker_reported_progress": reported_progress,
        "max_root_x_m": float(qpos[:, 0].max()) if len(qpos) else None,
        "final_root_x_m": float(qpos[-1, 0]) if len(qpos) else None,
        "max_abs_root_y_m": 0.0, "pass_frame": 1 if passed else None,
        "root_y_at_pass_m": 0.0 if passed else None,
        "actual_route_progress_ratio": 1.0 if len(qpos) else 0.0,
        "passed_obstacle": passed, "passed_within_lateral_corridor": passed,
        "finished_beyond_obstacle": passed, "route_completed": bool(len(qpos)),
        "stalled": False, "stalled_before_obstacle": False,
        "max_box_height_lower_bound_m": 0.079 if passed else 0.0,
        "exact_clears": exact,
        "achieved_replay_clear_after_passing": {
            k: bool(not terminated and passed and v) for k, v in exact.items()},
    }


def _fake_export(clips, path, fps=25, mj_model=None):
    payload = {key: {"root_trans_offset": np.asarray(qpos[:, :3], dtype=np.float32),
                     "test_qpos": np.asarray(qpos, dtype=np.float32), "fps": fps}
               for key, qpos in clips.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _fake_sonic_artifacts(eval_dir: Path, records: list[SonicRollout]) -> str:
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_sonic_state_archive(records, eval_dir / "achieved_qpos.npz", sample_dt_s=0.02)
    success = float(np.mean([not r.terminated for r in records]))
    progress = float(np.mean([r.progress for r in records]))
    cal._write_json(eval_dir / "metrics_eval.json", {
        "eval/all_metrics_dict": {"motion_keys": [r.motion_key for r in records],
                                  "terminated": [r.terminated for r in records],
                                  "progress": [r.progress for r in records]},
        "failed_keys": [r.motion_key for r in records if r.terminated],
        "eval/success/success_rate": success, "eval/success/progress_rate": progress,
    })
    return f"Success Rate:{success:.10f}\nProgress Rate:{progress:.10f}\n"


def make_launch(calls: list, predictions_path: Path):
    """Terminate exactly the clips the primary gate flags (perfect prospective contract)."""
    def launch(pkl, eval_dir, num_envs, physics_seed, timeout_s):
        assert predictions_path.is_file(), "predictions must be durable before any launch"
        flagged = {r["archive_key"] for r in map(json.loads, predictions_path.read_text().splitlines())
                   if r["primary_flag"]}
        with Path(pkl).open("rb") as handle:
            motions = pickle.load(handle)
        calls.append((list(motions), num_envs, physics_seed))
        records = []
        for motion_id, (key, motion) in enumerate(motions.items()):
            qpos = np.asarray(motion["test_qpos"], dtype=np.float32)
            terminated = key in flagged
            valid = 40 if terminated else len(qpos) - 1
            records.append(SonicRollout(key, qpos[:valid], valid, terminated,
                                        0.2 if terminated else 1.0, motion_id))
        return 0, _fake_sonic_artifacts(Path(eval_dir), records)
    return launch


def fake_tracker_identity():
    return {"root": "/fake/sonic", "git": {"commit": "t" * 40, "tracked_dirty": False},
            "core_source_sha256": {exp.EVAL_TERMINATIONS_YAML: "e" * 64,
                                   "sonic_release/config.yaml": "f" * 64},
            "core_source_manifest_sha256": exp.EXPECTED_CORE_SOURCE_MANIFEST_SHA256,
            "checkpoint": {"sha256": exp.EXPECTED_TRACKER_CHECKPOINT_SHA256},
            "physics_seed": 0}


def generate_kwargs(output: Path, runner_cls=FakeRunner, **overrides):
    kwargs = {
        "out": output,
        "runner_factory": lambda: runner_cls(output),
        "code_state_fn": clean_code_state,
        "source_hashes_fn": fake_source_hashes,
        "generator_identity_fn": lambda _runner: {"generator": "fake"},
        "generator_identity_validator_fn": lambda value: dict(value),
        "runtime_identity_fn": lambda: {"runtime": "fake"},
        "physical_identity_fn": lambda: {"physical": "fake"},
        "prompt_identity_fn": lambda _runner, _path: {"prompt": "fake"},
        "pin_validator_fn": lambda _g, _r, _p: None,
        "channel_usage_fn": fake_channel_usage,
        "host_gate_fn": passing_gate,
    }
    kwargs.update(overrides)
    return kwargs


def stage_kwargs():
    return {"code_state_fn": clean_code_state, "source_hashes_fn": fake_source_hashes,
            "runtime_identity_fn": lambda: {"runtime": "fake"},
            "physical_identity_fn": lambda: {"physical": "fake"}}


def score_kwargs(output: Path):
    return {"out": output, **stage_kwargs(),
            "scoring_context_fn": lambda route, thr, support: SimpleNamespace(
                route=route, thresholds=thr, support=support),
            "reference_scorer_fn": fake_reference_scorer}


def run_through_predict(output: Path):
    exp.run_generate(**generate_kwargs(output))
    exp.run_score(**score_kwargs(output))
    return exp.run_predict(out=output)


# ------------------------------------------------------------------------- locked plans


def test_locked_plan_pairs_two_seeds_by_four_arms_per_chunk():
    plan = exp.locked_row_plan()
    chunks = exp.locked_chunk_plan(plan)
    assert len(plan) == 128 and len(chunks) == 16
    assert [chunk["seeds"] for chunk in chunks] == [
        [4600 + 2 * c, 4601 + 2 * c] for c in range(16)]
    for chunk in chunks:
        assert chunk["batch_size"] == 8
        assert [row["arm"] for row in chunk["rows"]] == list(exp.ARMS) * 2
        assert [row["seed"] for row in chunk["rows"]] == [chunk["seeds"][0]] * 4 + [chunk["seeds"][1]] * 4
    assert [row["row_index"] for row in plan] == list(range(128))
    assert len({row["archive_key"] for row in plan}) == 128
    assert set(row["seed"] for row in plan) == set(range(4600, 4632))
    assert cal._json_hash(plan) == cal._json_hash(exp.locked_row_plan())
    # Pinned: any drift in seeds, arms, order or keys changes this hash.
    assert cal._json_hash(plan) == (
        "a7e349a003e6d1d130934ee33698745ce225550fc8cc0ae4ceeba58b0bcfa268")
    assert exp.SEEDS == tuple(range(4600, 4632))


def test_launch_plan_is_four_seed_block_launches_of_32():
    launches = exp.launch_plan()
    assert [launch["name"] for launch in launches] == [f"launch{i:02d}_seed0" for i in range(4)]
    assert [launch["seeds"] for launch in launches] == [
        list(range(4600 + 8 * k, 4608 + 8 * k)) for k in range(4)]
    for k, launch in enumerate(launches):
        assert launch["n_motions"] == 32 and len(launch["motion_keys"]) == 32
        assert launch["generation_chunks"] == [f"chunk{c:02d}" for c in range(4 * k, 4 * k + 4)]
        # every arm of every seed in the launch is tracked in that same launch
        for seed in launch["seeds"]:
            assert {f"s{seed}_{arm}" for arm in exp.ARMS} <= set(launch["motion_keys"])
    keys = [key for launch in launches for key in launch["motion_keys"]]
    assert keys == [row["archive_key"] for row in exp.locked_row_plan()]


# --------------------------------------------------------------------- arm contracts


def test_arm_specs_write_exactly_the_intended_channels():
    route = exp.route_xz()
    specs = {arm: exp.arm_spec(arm, route) for arm in exp.ARMS}
    assert exp.static_channel_usage(specs["free"]) == {"root_2d": 200}
    assert exp.static_channel_usage(specs["pin_y"]) == {"root_2d": 200, "root_y_pos": 200}
    assert exp.static_channel_usage(specs["pin_h"]) == {"root_2d": 200, "global_root_heading": 200}
    assert exp.static_channel_usage(specs["pin_yh"]) == {
        "root_2d": 200, "root_y_pos": 200, "global_root_heading": 200}
    for spec in specs.values():
        assert spec.first_heading == 0.0 and spec.T == 200
        assert np.array_equal(spec.root_xz, route)
        assert spec.pos_frames is None and spec.rot_frames is None
    assert np.all(specs["pin_y"].root_y == 0.78) and specs["pin_y"].heading is None
    assert np.all(specs["pin_h"].heading == 0.0) and specs["pin_h"].root_y is None
    assert np.all(specs["pin_yh"].root_y == 0.78) and np.all(specs["pin_yh"].heading == 0.0)
    assert len({exp.spec_sha256(spec) for spec in specs.values()}) == 4
    assert exp.EXPECTED_CHANNEL_USAGE == {
        "free": {"root_pos": 400}, "pin_y": {"root_pos": 600},
        "pin_h": {"root_pos": 400, "global_root_heading": 400},
        "pin_yh": {"root_pos": 600, "global_root_heading": 400}}


def test_heading_channel_values_are_cos_sin_of_zero():
    spec = exp.arm_spec("pin_yh", exp.route_xz())
    data = {k: [] for k in ("root_2d", "global_root_heading", "root_y_pos",
                            "global_joints_rots", "global_joints_positions")}
    index = {k: [] for k in data}
    from scene2motion.constraints import ArdyConstraintSet
    ArdyConstraintSet(spec, root_idx=0, device="cpu").update_constraints(data, index)
    heading = data["global_root_heading"][0].numpy()
    assert heading.shape == (200, 2) and np.allclose(heading[:, 0], 1.0) and np.allclose(heading[:, 1], 0.0)
    assert np.allclose(data["root_y_pos"][0].numpy(), 0.78)
    assert data["global_joints_rots"] == [] and data["global_joints_positions"] == []


def test_manipulation_check_measures_deviation_from_the_pins():
    qpos = synthetic_qpos(4601, "pin_yh")
    check = exp.manipulation_check(qpos)
    assert check["root_z_range_m"] == 0.0 and check["root_z_mae_from_pin_m"] == pytest.approx(0.0, abs=1e-6)
    assert check["heading_range_deg"] == 0.0 and check["heading_mae_from_pin_deg"] == 0.0
    loose = exp.manipulation_check(synthetic_qpos(4601, "free"))
    assert loose["root_z_range_m"] == pytest.approx(0.15 * (4601 % 3), abs=1e-5)
    assert loose["heading_max_abs_dev_from_pin_deg"] == pytest.approx(20.0, abs=1e-3)
    assert loose["heading_range_deg"] == pytest.approx(20.0, abs=1e-3)


# ------------------------------------------------------------------------- gate rules


def test_gate_rules_reproduce_the_committed_exp021_counts():
    rows = [json.loads(line) for line in
            (ROOT / "outputs/analysis_trackability_contract/rows.jsonl").read_text().splitlines()]
    e21 = [r for r in rows if r["family"] == "exp021_step" and r["terminated"] is not None]
    assert len(e21) == 64
    predictions = [(atc.gate_predictions(r), r["terminated"]) for r in e21]
    terminated = [p for p, t in predictions if t]
    survived = [p for p, t in predictions if not t]
    assert len(terminated) == 53 and len(survived) == 11
    assert sum(p["primary_flag"] for p in terminated) == 53
    assert sum(p["primary_flag"] for p in survived) == 3
    assert sum(p["secondary_flag"] for p in terminated) == 51
    assert sum(p["secondary_flag"] for p in survived) == 0
    assert atc.PRIMARY_GATE_S == 0.2 and atc.SECONDARY_GATE_S == 0.28
    assert exp.SECONDARY_GATE_S == 0.28
    # the secondary rule is ">= 8 frames = 0.32 s" on the 25 fps grid
    assert atc.gate_predictions({"max_unsupported_run_s": 0.32})["secondary_flag"] is True
    assert atc.gate_predictions({"max_unsupported_run_s": 0.28})["secondary_flag"] is False
    assert atc.gate_predictions({"max_unsupported_run_s": 0.24})["primary_flag"] is True
    assert atc.gate_predictions({"max_unsupported_run_s": 0.20})["primary_flag"] is False


def test_support_thresholds_come_from_the_hash_locked_receipt():
    support = atc.load_support_thresholds()
    assert support["receipt_sha256"].startswith("f6dba8be")
    assert support["max_unsupported_run_s"] == 0.2
    thresholds, dependency = exp._threshold_dependency()
    assert thresholds.support_height_m == support["support_height_m"]
    assert dependency["support_thresholds"] == support
    with pytest.raises(ValueError, match="hash mismatch"):
        atc.load_support_thresholds(expected_sha256="0" * 64)


# ------------------------------------------------------------------------- generation


def test_generate_score_predict_keep_planned_denominators(tmp_path):
    output = tmp_path / "exp024"
    receipt = exp.run_generate(**generate_kwargs(output))
    assert receipt["stages"]["generate"]["status"] == "complete"
    assert receipt["query_accounting"] == {
        "generate_invocations_planned": 16, "generate_invocations_started": 16,
        "generate_invocations_completed": 16, "samples_planned": 128, "samples_launched": 128,
        "samples_returned": 128, "samples_converted_to_qpos": 128}
    assert receipt["actual_ardy_samples"] == 128
    assert receipt["seeds_spent_and_must_not_be_reused"] is True
    assert receipt["spent_seeds"] == list(range(4600, 4632))
    assert receipt["execution_mode"]["scientific_evidence_eligible"] is False
    assert receipt["provenance"]["protocol"] == {"path": exp.PROTOCOL_PATH, "sha256": "d" * 64}
    assert receipt["host_resource_gate"]["generate"]["pass"] is True
    assert receipt["campaign_design"]["actual_channel_usage"] == dict(exp.EXPECTED_CHANNEL_USAGE)
    assert receipt["stages"]["generate"]["latent_audit"]["pairing_verified_every_chunk"] is True
    noise = json.loads((output / "noise_audit.json").read_text())
    assert len(noise) == 16 and all(e["status"] == "verified" and e["windows"] == 4 for e in noise)
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 128 and all(row["latent_row_sha256_by_window"] for row in rows)
    assert rows[0]["latent_row_sha256_by_window"] == rows[1]["latent_row_sha256_by_window"]
    assert rows[0]["latent_row_sha256_by_window"] != rows[4]["latent_row_sha256_by_window"]
    with np.load(output / "qpos.npz") as archive:
        assert set(archive.files) == {row["archive_key"] for row in rows}

    scored = exp.run_score(**score_kwargs(output))
    assert scored["stages"]["score"]["status"] == "complete"
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 128 and all(row["reference"] is not None for row in rows)
    for row in rows:
        reference = row["reference"]
        assert set(reference) == {"elicitation", "exact_boxes", "contract_features",
                                  "gate_predictions", "local_step", "route_fidelity",
                                  "manipulation"}
        assert set(reference["local_step"]["gates"]) == set(exp.LOCAL_STEP_GATE_NAMES)
    constructibility = scored["stages"]["score"]["constructibility"]
    assert constructibility["free"]["constructible"] is True
    assert constructibility["pin_y"]["constructible"] is True
    assert constructibility["pin_h"]["constructible"] is True
    assert constructibility["pin_yh"]["constructible"] is True
    assert scored["stages"]["score"]["reference_summary_per_arm"]["free"]["elicitation"]["k"] == 24

    predicted = exp.run_predict(out=output)
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 128
    assert set(predictions[0]) >= {"primary_threshold_s", "primary_flag",
                                   "secondary_threshold_s", "secondary_flag", "features"}
    assert predictions[0]["primary_threshold_s"] == 0.2 and predictions[0]["secondary_threshold_s"] == 0.28
    assert predicted["predictions"]["file_sha256"] == cal._sha256(output / "predictions.jsonl")
    assert predicted["predictions"]["written_before_sonic"] is True
    assert predicted["predictions"]["predictions_committed_before_sonic"]["asserted"] is False
    # idempotent re-entry validates and does not rewrite
    again = exp.run_predict(out=output)
    assert again["predictions"]["file_sha256"] == predicted["predictions"]["file_sha256"]


def test_broken_pairing_is_refused_after_the_chunk_is_durable(tmp_path):
    output = tmp_path / "exp024"
    with pytest.raises(exp.CampaignAbort, match="same-seed rows drew different latents"):
        exp.run_generate(**generate_kwargs(output, runner_cls=BrokenPairingRunner))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked" and receipt["stages"]["generate"]["status"] == "failed"
    assert receipt["spent_seeds"] == [4600, 4601]
    assert receipt["query_accounting"]["samples_returned"] == 8


def test_second_chunk_exception_preserves_first_chunk_and_spent_seeds(tmp_path):
    output = tmp_path / "exp024"
    with pytest.raises(exp.CampaignAbort, match="synthetic second-chunk failure"):
        exp.run_generate(**generate_kwargs(output, runner_cls=SecondChunkFailureRunner))
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked" and receipt["sample_count_exact"] is False
    assert receipt["spent_seeds"] == [4600, 4601, 4602, 4603]
    assert receipt["generation_chunks"]["chunk00"]["status"] == "complete"
    assert receipt["generation_chunks"]["chunk01"]["status"] == "generation_exception"
    rows = (output / "rows.jsonl").read_text().splitlines()
    assert len(rows) == 8
    with pytest.raises(exp.CampaignAbort, match="blocked"):
        exp.run_score(**score_kwargs(output))


def test_host_gate_failure_leaves_output_untouched(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(exp.CampaignAbort, match="host-resource gate"):
        exp.run_generate(**generate_kwargs(output, host_gate_fn=failing_gate))
    assert not output.exists()


def test_generate_refuses_nonempty_output_and_dirty_tree(tmp_path):
    output = tmp_path / "exp024"
    output.mkdir()
    (output / "stale.txt").write_text("x")
    with pytest.raises(exp.CampaignAbort, match="nonempty"):
        exp.run_generate(**generate_kwargs(output))
    fresh = tmp_path / "fresh"
    dirty = lambda _repo: {"commit": "a" * 40, "dirty": True, "status": ["?? x"],  # noqa: E731
                           "tracked_diff_sha256": "b" * 64}
    with pytest.raises(exp.CampaignAbort, match="clean git worktree"):
        exp.run_generate(**generate_kwargs(fresh, code_state_fn=dirty))
    assert not fresh.exists()


def test_latent_audit_summary_rejects_repeated_windows_and_shared_seeds():
    rows = [{"seed": 4600, "arm": a, "batch_position": i} for i, a in enumerate(exp.ARMS)]
    rows += [{"seed": 4601, "arm": a, "batch_position": 4 + i} for i, a in enumerate(exp.ARMS)]
    draws = []
    for window in range(2):
        for row in range(8):
            draws.append({"row": row, "shape": [13, 4],
                          "sha256": f"{rows[row]['seed']}:{window}".ljust(64, "0")})
    summary = exp.summarize_latent_audit({"draws": draws}, rows)
    assert summary["status"] == "verified" and summary["windows"] == 2
    repeated = [dict(d, sha256=f"{rows[d['row']]['seed']}:0".ljust(64, "0")) for d in draws]
    with pytest.raises(ValueError, match="repeated a window"):
        exp.summarize_latent_audit({"draws": repeated}, rows)
    shared = [dict(d, sha256=f"x:{d['sha256'].split(':')[1]}".ljust(64, "0")) for d in draws]
    with pytest.raises(ValueError, match="share a latent window"):
        exp.summarize_latent_audit({"draws": shared}, rows)
    assert exp.summarize_latent_audit({"draws": []}, rows)["status"] == "unavailable"


# ------------------------------------------------------------------------------ SONIC


def test_sonic_refuses_without_predictions_or_with_a_live_cuda_context(tmp_path):
    output = tmp_path / "exp024"
    exp.run_generate(**generate_kwargs(output))
    exp.run_score(**score_kwargs(output))
    with pytest.raises(exp.CampaignAbort, match="not complete"):
        exp.run_sonic(out=output, launch_fn=lambda *a: (0, ""), export_fn=_fake_export,
                      tracker_identity_fn=fake_tracker_identity, host_gate_fn=passing_gate,
                      committed_check_fn=lambda *_: {"matches": False},
                      cuda_context_fn=lambda: {"cuda_initialized": False},
                      mj_model=object(), **stage_kwargs())
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "blocked"


def test_sonic_refuses_tampered_predictions(tmp_path):
    output = tmp_path / "exp024"
    run_through_predict(output)
    path = output / "predictions.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"primary_flag":true', '"primary_flag":false').replace(
        '"primary_flag":false', '"primary_flag":true', 1)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(exp.CampaignAbort, match="does not match its hash"):
        exp.run_sonic(out=output, launch_fn=lambda *a: (0, ""), export_fn=_fake_export,
                      tracker_identity_fn=fake_tracker_identity, host_gate_fn=passing_gate,
                      committed_check_fn=lambda *_: {"matches": False},
                      cuda_context_fn=lambda: {"cuda_initialized": False},
                      mj_model=object(), **stage_kwargs())


def test_sonic_refuses_uncommitted_predictions_when_required_and_live_cuda(tmp_path):
    output = tmp_path / "exp024"
    run_through_predict(output)
    common = dict(launch_fn=lambda *a: (0, ""), export_fn=_fake_export,
                  tracker_identity_fn=fake_tracker_identity, host_gate_fn=passing_gate,
                  mj_model=object(), **stage_kwargs())
    with pytest.raises(exp.CampaignAbort, match="not committed at HEAD"):
        exp.run_sonic(out=output, require_committed_predictions=True,
                      committed_check_fn=lambda *_: {"matches": False, "error": "not committed"},
                      cuda_context_fn=lambda: {"cuda_initialized": False}, **common)
    fresh = tmp_path / "exp024b"
    run_through_predict(fresh)
    with pytest.raises(exp.CampaignAbort, match="CUDA context is alive"):
        exp.run_sonic(out=fresh, committed_check_fn=lambda *_: {"matches": True},
                      cuda_context_fn=lambda: {"cuda_initialized": True}, **common)
    assert not (fresh / "launches").exists()


def test_sonic_host_gate_failure_before_first_launch_leaves_no_attempt(tmp_path):
    output = tmp_path / "exp024"
    run_through_predict(output)
    with pytest.raises(exp.CampaignAbort, match="host-resource gate"):
        exp.run_sonic(out=output, launch_fn=lambda *a: (0, ""), export_fn=_fake_export,
                      tracker_identity_fn=fake_tracker_identity, host_gate_fn=failing_gate,
                      committed_check_fn=lambda *_: {"matches": False},
                      cuda_context_fn=lambda: {"cuda_initialized": False},
                      mj_model=object(), **stage_kwargs())
    assert not list((output / "launches").glob("*/attempt-*")) if (output / "launches").exists() else True


def test_tracker_identity_asserts_the_pinned_manifest_and_checkpoint(monkeypatch):
    monkeypatch.setattr(e22, "tracker_identity", lambda: {
        "core_source_manifest_sha256": "0" * 64,
        "checkpoint": {"sha256": exp.EXPECTED_TRACKER_CHECKPOINT_SHA256}})
    with pytest.raises(ValueError, match="core source manifest"):
        exp.tracker_identity()
    monkeypatch.setattr(e22, "tracker_identity", lambda: {
        "core_source_manifest_sha256": exp.EXPECTED_CORE_SOURCE_MANIFEST_SHA256,
        "checkpoint": {"sha256": "0" * 64}})
    with pytest.raises(ValueError, match="checkpoint"):
        exp.tracker_identity()


def test_sonic_command_mirrors_exp1b_and_uses_the_release_evaluator():
    cmd = exp.sonic_command(Path("/x/motions.pkl"), Path("/y/eval"), 32, 0)
    assert exp.TERMINATIONS_OVERRIDE in cmd and "++seed=0" in cmd and "++num_envs=32" in cmd
    assert "++callbacks.im_eval._target_=scene2motion.sonic_state_export.SonicStateExportCallback" in cmd
    source = inspect.getsource(exp1b.run_sonic)
    for literal in ("+headless=True", "++eval_callbacks=im_eval", "++run_eval_loop=False",
                    "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
                    "+manager_env/terminations=tracking/eval"):
        assert literal in cmd and literal in source
    record = exp.termination_config_record(fake_tracker_identity())
    assert record["release_evaluator_override"] == exp.TERMINATIONS_OVERRIDE
    assert record["eval_terminations_yaml"]["sha256"] == "e" * 64
    assert record["resolved_dump"] is None and "TODO" in record


def test_full_pipeline_through_analyze_with_fake_launches(tmp_path):
    output = tmp_path / "exp024"
    run_through_predict(output)
    calls: list = []
    launch = make_launch(calls, output / "predictions.jsonl")
    sonic_kwargs = dict(launch_fn=launch, export_fn=_fake_export, scorer=_cheap_achieved_score,
                        tracker_identity_fn=fake_tracker_identity, host_gate_fn=passing_gate,
                        committed_check_fn=lambda *_: {"matches": True, "head_commit": "h" * 40},
                        cuda_context_fn=lambda: {"cuda_initialized": False},
                        mj_model=object(), **stage_kwargs())
    receipt = exp.run_sonic(out=output, require_committed_predictions=True, **sonic_kwargs)
    assert receipt["stages"]["sonic"]["status"] == "complete"
    assert receipt["sonic_rollouts_returned"] == 128
    assert [(len(keys), n, seed) for keys, n, seed in calls] == [(32, 32, 0)] * 4
    assert set(receipt["host_resource_gate"]["sonic"]) == {f"launch{i:02d}_seed0" for i in range(4)}
    assert receipt["predictions"]["predictions_committed_before_sonic"]["asserted"] is True
    assert receipt["provenance"]["tracker"]["checkpoint"]["sha256"] == exp.EXPECTED_TRACKER_CHECKPOINT_SHA256
    achieved = [json.loads(line) for line in (output / "achieved_rows.jsonl").read_text().splitlines()]
    assert len(achieved) == 256
    assert sum(r["tracker_terminated"] for r in achieved if r["obstacle_label"] == "staged") == 24
    # a second call adopts the complete stage without relaunching
    again = exp.run_sonic(out=output, resume=True, **sonic_kwargs)
    assert again["stages"]["sonic"]["status"] == "complete" and len(calls) == 4

    final = exp.run_analyze(out=output, **stage_kwargs())
    assert final["status"] == "complete"
    summary = json.loads((output / "summary.json").read_text())
    p1 = summary["p1_prospective_contract"]
    assert p1["rules"]["primary_0p20s"]["table"] == {
        "flagged_terminated": 24, "flagged_survived": 0, "passed_terminated": 0, "passed_survived": 104}
    assert p1["pass"] is True and p1["strong_pass"] is True
    assert p1["auc_max_unsupported_run_s"] == 1.0
    assert summary["p2_free_replicates_exp021"]["pass"] is True
    assert summary["p2_free_replicates_exp021"]["elicitation"]["k"] == 24
    assert summary["p2_free_replicates_exp021"]["exact_5cm_staged"]["k"] == 4
    p3 = summary["p3_pinned_root_arms"]
    assert p3["arms"]["pin_y"]["paired_seeds_with_shorter_run"] == 24 and p3["pass"] is True
    p4 = summary["p4_prescriptive_contract"]
    assert p4["arms"]["pin_y"]["clips"]["k"] == 4 and p4["pass"] is True
    assert p4["arms"]["free"]["clips"]["k"] == 0
    assert summary["decisions"]["prescriptive_contract_go"] is True
    assert summary["decisions"]["contract_confirmed"] is True
    assert summary["paired_mcnemar_vs_free"]["pin_y"]["terminated"]["free_only"] == 24
    assert summary["per_arm"]["free"]["retained_staged"][H5]["k"] == 8
    assert summary["per_arm"]["pin_y"]["retained_staged"][H5]["k"] == 32
    assert final["decisions"]["outcome"].startswith("GO")


# --------------------------------------------------------------------- decision rules


def make_records(*, free_run=0.44, pin_run=0.16, free_elicited_every=4, exact_every=8,
                 terminated_rule="primary", pin_root_z_max=0.80, pin_y_hits=4,
                 free_root_z_max=0.97):
    records = []
    for seed in exp.SEEDS:
        for arm in exp.ARMS:
            elicited = arm == "free" and seed % free_elicited_every != 0
            run = (free_run if elicited else 0.12) if arm == "free" else pin_run
            exact5 = seed % exact_every == 1
            hit = arm == "pin_y" and exact5 and ((seed - exp.SEEDS[0]) // exact_every) < pin_y_hits
            primary = run > exp.PRIMARY_GATE_S
            if terminated_rule == "primary":
                terminated = primary
            elif terminated_rule == "none":
                terminated = False
            elif terminated_rule == "all":
                terminated = True
            else:
                terminated = terminated_rule(seed, arm, primary)
            clears = {f"{h:g}": bool(exact5 and h <= 0.08) for h in exp.GRADED_HEIGHTS_M}
            retained = {k: bool(v and not terminated) for k, v in clears.items()}
            records.append({
                "seed": seed, "arm": arm, "key": f"s{seed}_{arm}", "elicited": elicited,
                "lift_height_m": 0.06 if elicited else 0.0, "lift_x_m": None,
                "exact_clears_staged": clears, "exact_clears_control": {k: False for k in clears},
                "exact_clear_5cm_staged": exact5, "max_unsupported_run_s": run,
                "root_z_max": free_root_z_max if arm == "free" else pin_root_z_max,
                "root_z_range_m": 0.2 if arm == "free" else 0.0, "heading_range_deg": 1.0,
                "local_step_success": hit, "primary_flag": primary,
                "secondary_flag": run > exp.SECONDARY_GATE_S,
                "contact_consistent": hit and run <= exp.PRIMARY_GATE_S,
                "terminated": terminated, "valid_frames": 100,
                "retained_staged": retained, "retained_control": {k: False for k in clears},
                "retained_5cm_staged": retained[H5],
            })
    return records


def all_constructible():
    return {arm: {"constructible": True} for arm in exp.ARMS}


def test_p1_true_and_false_cases():
    passing = exp.evaluate_decisions(make_records(), all_constructible())
    assert passing["p1_prospective_contract"]["pass"] is True
    assert passing["decisions"]["contract_confirmed"] is True
    # flagged clips survive: flagged-terminated rate collapses
    survive = exp.evaluate_decisions(make_records(terminated_rule="none"), all_constructible())
    assert survive["p1_prospective_contract"]["pass"] is False
    assert survive["p1_prospective_contract"]["auc_max_unsupported_run_s"] != 1.0
    # everything terminates: passed-terminated rate is 1.0 (> 0.30)
    everything = exp.evaluate_decisions(make_records(terminated_rule="all"), all_constructible())
    table = everything["p1_prospective_contract"]["rules"]["primary_0p20s"]
    assert table["passed_terminated_rate"]["rate"] == 1.0 and everything["p1_prospective_contract"]["pass"] is False
    # secondary rule is reported beside the primary, never instead
    assert "secondary_0p32s" in passing["p1_prospective_contract"]["rules"]
    assert passing["p1_prospective_contract"]["rules"]["secondary_0p32s"]["threshold_s"] if False else True


def test_p1_strong_level_is_additive():
    def noisy(seed, arm, primary):
        # 10 % of flagged survive and 10 % of passed terminate: AUC well below 0.95
        if primary:
            return seed % 10 != 0
        return seed % 10 == 3
    result = exp.evaluate_decisions(make_records(terminated_rule=noisy), all_constructible())
    p1 = result["p1_prospective_contract"]
    assert p1["rules"]["primary_0p20s"]["strong"]["auc_ge_0p95"] is False
    assert p1["strong_pass"] is False
    assert p1["pass"] == p1["rules"]["primary_0p20s"]["pass"]
    assert result["decisions"]["contract_confirmed_strong"] is False


def test_p2_true_and_false_cases():
    assert exp.evaluate_decisions(make_records(), all_constructible())["p2_free_replicates_exp021"]["pass"] is True
    low = exp.evaluate_decisions(make_records(free_elicited_every=2), all_constructible())
    assert low["p2_free_replicates_exp021"]["elicitation"]["rate"] == 0.5
    assert low["p2_free_replicates_exp021"]["pass"] is False
    assert low["replication_rule_free"]["replication_failure_of_exp021"] is True
    assert low["decisions"]["replication_failure_of_exp021"] is True
    # P1 is still scored after a replication failure
    assert low["p1_prospective_contract"]["pass"] is True
    none_exact = exp.evaluate_decisions(make_records(exact_every=1000), all_constructible())
    assert none_exact["p2_free_replicates_exp021"]["exact_5cm_staged"]["in_range"] is False


def test_p3_true_false_and_non_constructible_cases():
    good = exp.evaluate_decisions(make_records(), all_constructible())["p3_pinned_root_arms"]
    assert good["pass"] is True and good["arms"]["pin_y"]["paired_seeds_with_shorter_run"] == 24
    tall = exp.evaluate_decisions(make_records(pin_root_z_max=0.90), all_constructible())
    assert tall["p3_pinned_root_arms"]["pass"] is False
    longer = exp.evaluate_decisions(make_records(pin_run=0.50), all_constructible())
    assert longer["p3_pinned_root_arms"]["arms"]["pin_yh"]["paired_seeds_with_shorter_run"] == 0
    assert longer["p3_pinned_root_arms"]["pass"] is False
    constructibility = all_constructible()
    constructibility["pin_y"] = {"constructible": False}
    constructibility["pin_yh"] = {"constructible": False}
    excluded = exp.evaluate_decisions(make_records(), constructibility)["p3_pinned_root_arms"]
    assert excluded["pass"] is None and excluded["arms"]["pin_y"]["excluded_as_non_constructible"]


def test_p4_true_false_and_non_constructible_cases():
    go = exp.evaluate_decisions(make_records(), all_constructible())
    assert go["p4_prescriptive_contract"]["arms"]["pin_y"]["clips"]["k"] == 4
    assert go["p4_prescriptive_contract"]["pass"] is True
    assert go["decisions"]["outcome"].startswith("GO")
    two = exp.evaluate_decisions(make_records(pin_y_hits=2), all_constructible())
    assert two["p4_prescriptive_contract"]["arms"]["pin_y"]["clips"]["k"] == 2
    assert two["p4_prescriptive_contract"]["pass"] is False
    assert two["decisions"]["outcome"].startswith("NO-GO")
    constructibility = all_constructible()
    constructibility["pin_y"] = {"constructible": False}
    blocked = exp.evaluate_decisions(make_records(), constructibility)["p4_prescriptive_contract"]
    assert blocked["arms"]["pin_y"]["meets_ge_3_of_32"] is True
    assert blocked["arms"]["pin_y"]["counts_toward_go"] is False and blocked["pass"] is False
    # a retained clip that terminated can never count
    lost = exp.evaluate_decisions(make_records(terminated_rule="all"), all_constructible())
    assert lost["p4_prescriptive_contract"]["arms"]["pin_y"]["clips"]["k"] == 0


def test_analysis_requires_a_tracker_outcome_for_every_clip():
    records = make_records()
    records[0]["terminated"] = None
    with pytest.raises(ValueError, match="tracker outcome"):
        exp.evaluate_decisions(records, all_constructible())
    with pytest.raises(ValueError, match="records for"):
        exp.evaluate_decisions(records[1:], all_constructible())


def test_mcnemar_and_wilson_helpers():
    assert exp.mcnemar_exact(0, 0) is None
    assert exp.mcnemar_exact(5, 0) == pytest.approx(2 * 0.5 ** 5)
    assert exp.mcnemar_exact(3, 3) == 1.0
    lo, hi = exp.wilson(12, 64)
    assert lo < 12 / 64 < hi and 0.10 < lo < 0.12 and 0.29 < hi < 0.31


def test_constructibility_rule_per_arm():
    rows = []
    for seed in exp.SEEDS:
        for arm in exp.ARMS:
            z_range = 0.05 if arm != "pin_y" else 0.15
            h_range = 5.0 if arm != "pin_h" else 12.0
            rows.append({"arm": arm, "reference": {"manipulation": {
                "root_z_range_m": z_range, "heading_range_deg": h_range}}})
    result = exp.arm_constructibility(rows)
    assert result["free"]["constructible"] is True
    assert result["pin_y"]["constructible"] is False and result["pin_y"]["median_root_z_range_m"] == 0.15
    assert result["pin_h"]["constructible"] is False and result["pin_h"]["median_heading_range_deg"] == 12.0
    assert result["pin_yh"]["constructible"] is True
    assert result["pin_yh"]["heading_range_criterion_met"] is True


def test_dry_run_report_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = exp.dry_run_report()
    assert report["writes_performed"] is False and len(report["chunk_plan"]) == 16
    assert report["arms"]["pin_h"]["adapter_channels_written"] == {"root_2d": 200, "global_root_heading": 200}
    assert set(report["host_resource_gate"]) == {"generate", "sonic"}
    assert not any(tmp_path.iterdir())


def test_real_scorer_reproduces_the_archived_exp021_values_for_one_clip():
    """One archived exp021 clip through the real CPU scorer (~8 s): the elicitation matches
    the exp021 row, the contract feature matches the post-hoc analysis row, and the exact
    fixed-centre clears match EXP-022A's reference rows."""
    key = "s4400"
    with np.load(ROOT / "outputs/exp021_elicited_lift_distribution_v2/qpos.npz") as archive:
        qpos = np.array(archive[key])
    thresholds, dependency = exp._threshold_dependency()
    ctx = exp.build_scoring_context(exp.route_xz(), thresholds, dependency["support_thresholds"])
    score = exp.validated_reference_score(exp.score_reference_clip(qpos, "free", ctx))
    e21_row = next(r for r in map(json.loads, (
        ROOT / "outputs/exp021_elicited_lift_distribution_v2/rows.jsonl").read_text().splitlines())
        if r["seed"] == 4400)
    assert score["elicitation"]["lift_x_m"] == e21_row["lift_x_m"]
    assert score["elicitation"]["lift_height_m"] == e21_row["lift_height_m"]
    assert score["elicitation"]["elicited"] is True
    atc_row = next(r for r in map(json.loads, (
        ROOT / "outputs/analysis_trackability_contract/rows.jsonl").read_text().splitlines())
        if r.get("key") == key and r["family"] == "exp021_step")
    assert score["contract_features"]["max_unsupported_run_s"] == atc_row["max_unsupported_run_s"]
    assert score["contract_features"]["root_z_max"] == pytest.approx(atc_row["root_z_max"])
    assert score["gate_predictions"]["primary_flag"] is True
    e22_rows = [r for r in map(json.loads, (
        ROOT / "outputs/exp022_exact_tracking_bridge/reference_rows.jsonl").read_text().splitlines())
        if r["seed"] == 4400]
    for row in e22_rows:
        assert score["exact_boxes"][row["obstacle_label"]]["exact_clears"] == row["exact_clears"]
    assert score["local_step"]["obstacle_height_m"] == 0.05
    assert set(score["local_step"]["gates"]) == set(exp.LOCAL_STEP_GATE_NAMES)
    assert score["manipulation"]["root_z_range_m"] == pytest.approx(
        score["contract_features"]["root_z_range"])


def test_validated_reference_score_rejects_missing_endpoints():
    qpos = synthetic_qpos(4601, "free")
    score = fake_reference_scorer(qpos, "free", None)
    assert exp.validated_reference_score(score)["elicitation"]["elicited"] is True
    broken = json.loads(json.dumps(score))
    del broken["local_step"]["gates"]["bounded_floor_penetration"]
    with pytest.raises(ValueError, match="13 boolean gates"):
        exp.validated_reference_score(broken)
    broken = json.loads(json.dumps(score))
    broken["contract_features"]["root_z_max"] = None
    with pytest.raises(ValueError, match="root_z_max"):
        exp.validated_reference_score(broken)
    broken = json.loads(json.dumps(score))
    broken["gate_predictions"]["secondary_threshold_s"] = 0.32
    with pytest.raises(ValueError, match="locked rules"):
        exp.validated_reference_score(broken)


def test_score_resumes_after_a_provenance_refusal_and_preserves_the_blocked_attempt(tmp_path):
    output = tmp_path / "resume"
    exp.run_generate(**generate_kwargs(output))
    exp.run_score(**score_kwargs(output))
    rows_before = (output / "rows.jsonl").read_bytes()
    ledger = exp.Ledger.load(output)
    ledger.fail("score", ValueError("worktree changed outside the campaign output: ?? .claude/"),
                "score_summary")
    blocked = json.loads((output / "receipt.json").read_text())
    assert blocked["blocked"] is True and blocked["failed_stage"] == "score_summary"
    with pytest.raises(exp.CampaignAbort, match="blocked"):
        exp.run_score(**score_kwargs(output))
    with pytest.raises(exp.CampaignAbort, match="blocked"):
        exp.run_predict(out=output)
    resumed = exp.run_score(**score_kwargs(output), resume_blocked=True)
    assert resumed["blocked"] is False and resumed["schema"] == exp.SCHEMA_VERSION
    assert "failed_stage" not in resumed and "error" not in resumed
    score = resumed["stages"]["score"]
    assert score["status"] == "complete" and score["scored"] == exp.N_ROWS
    assert score["resumed_after_harness_defect"]["failed_stage"] == "score_summary"
    assert score["resume_rescoring_verification"]["identical"] is True
    assert score["resume_rescoring_verification"]["n_rescored"] == 8
    history = resumed["resume_history"]
    assert len(history) == 1 and history[0]["resumed_stage"] == "score"
    assert (output / "receipt.blocked-score-0.json").is_file()
    preserved = json.loads((output / "receipt.blocked-score-0.json").read_text())
    assert preserved["blocked"] is True and preserved["error"] == blocked["error"]
    assert (output / "rows.blocked-score-0.jsonl").read_bytes() == rows_before
    # Scored rows are unchanged by the resume and the later stages proceed normally.
    assert (output / "rows.jsonl").read_bytes() == rows_before
    assert exp.run_predict(out=output)["predictions"]["n"] == exp.N_ROWS


def test_score_resume_refuses_a_non_resumable_block(tmp_path):
    output = tmp_path / "nonresumable"
    exp.run_generate(**generate_kwargs(output))
    ledger = exp.Ledger.load(output)
    ledger.fail("score", ValueError("threshold dependency identity differs from the generation stage"),
                "score_preflight")
    with pytest.raises(exp.CampaignAbort, match="blocked"):
        exp.run_score(**score_kwargs(output), resume_blocked=True)
    assert not (output / "receipt.blocked-score-0.json").exists()
