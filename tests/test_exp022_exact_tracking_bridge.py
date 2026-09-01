from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp022_exact_tracking_bridge as exp022
from scene2motion.sonic_state_export import SonicRollout, write_sonic_state_archive
from scene2motion.stepover_eval import BoxHeightProbe


def _clean_code_state(_repo: Path) -> dict:
    return {
        "commit": "test-commit",
        "dirty": False,
        "status": [],
        "tracked_diff_sha256": "0" * 64,
    }


def _tracker_identity() -> dict:
    return {
        "root": "/fake/sonic",
        "git": {"commit": "tracker-test", "tracked_dirty": False, "status": []},
        "core_source_sha256": {"eval.py": "a" * 64},
        "checkpoint": {"sha256": "b" * 64},
        "physics_seed": 0,
    }


def _cheap_score(qpos, obstacle_x_m, *, terminated=False, reported_progress=None):
    qpos = np.asarray(qpos)
    passed = bool(len(qpos) and obstacle_x_m <= 1.2)
    exact = {f"{height:g}": bool(passed and height <= 0.08)
             for height in exp022.GRADED_HEIGHTS_M}
    return {
        "valid_frames": len(qpos),
        "tracker_terminated": bool(terminated),
        "tracker_reported_progress": reported_progress,
        "max_root_x_m": float(qpos[:, 0].max()) if len(qpos) else None,
        "final_root_x_m": float(qpos[-1, 0]) if len(qpos) else None,
        "max_abs_root_y_m": float(np.abs(qpos[:, 1]).max()) if len(qpos) else None,
        "pass_frame": 1 if passed else None,
        "root_y_at_pass_m": 0.0 if passed else None,
        "actual_route_progress_ratio": 1.0 if len(qpos) else 0.0,
        "passed_obstacle": passed,
        "passed_within_lateral_corridor": passed,
        "finished_beyond_obstacle": passed,
        "route_completed": bool(len(qpos)),
        "stalled": False,
        "stalled_before_obstacle": False,
        "max_box_height_lower_bound_m": 0.079,
        "exact_clears": exact,
        "achieved_replay_clear_after_passing": {
            key: bool(not terminated and passed and value) for key, value in exact.items()
        },
    }


def _fake_export(clips, path, fps=25, mj_model=None):
    payload = {
        key: {
            "root_trans_offset": np.asarray(qpos[:, :3], dtype=np.float32),
            "test_qpos": np.asarray(qpos, dtype=np.float32),
            "fps": fps,
        }
        for key, qpos in clips.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _fake_sonic_artifacts(eval_dir: Path, records: list[SonicRollout]) -> str:
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_sonic_state_archive(records, eval_dir / "achieved_qpos.npz", sample_dt_s=0.02)
    success_rate = float(np.mean([not record.terminated for record in records]))
    progress_rate = float(np.mean([record.progress for record in records]))
    failed_keys = [record.motion_key for record in records if record.terminated]
    cal._write_json(eval_dir / "metrics_eval.json", {
        "eval/all_metrics_dict": {
            "motion_keys": [record.motion_key for record in records],
            "terminated": [record.terminated for record in records],
            "progress": [record.progress for record in records],
        },
        "failed_keys": failed_keys,
        "eval/success/success_rate": success_rate,
        "eval/success/progress_rate": progress_rate,
    })
    return f"Success Rate:{success_rate:.10f}\nProgress Rate:{progress_rate:.10f}\n"


def _write_process_result(attempt: Path, chunk: dict, pkl: Path, returncode: int = 0):
    log = attempt / "sonic.log"
    return exp022._write_process_result(
        attempt, returncode=returncode, log_path=log, chunk=chunk,
        motion_pkl_sha256=str(exp022._sha256(pkl)),
    )


def test_locked_exp021_source_validates_down_to_per_array_hashes():
    source = exp022.load_source_bundle()
    assert source["identity"]["noise_stream_version"] == 2
    assert source["identity"]["n_qpos"] == 64
    assert set(source["clips"]) == {f"s{seed}" for seed in exp022.POOL_SEEDS}


def test_exact_fixed_center_counts_are_not_the_probe_lower_bound_counts():
    source = exp022.load_source_bundle()
    probe = BoxHeightProbe(1.2, exp022.OBSTACLE_DEPTH_M)
    exact = {height: 0 for height in (0.05, 0.08, 0.20)}
    lower = {height: 0 for height in (0.05, 0.08, 0.20)}
    for qpos in source["clips"].values():
        for height in exact:
            exact[height] += int(probe.clears(qpos, height))
        lower_bound = probe.probe(qpos)
        for height in lower:
            lower[height] += int(lower_bound >= height)
    assert exact == {0.05: 12, 0.08: 11, 0.20: 6}
    assert lower == {0.05: 12, 0.08: 10, 0.20: 5}


def test_dry_run_allows_dirty_tree_and_writes_nothing(tmp_path):
    out = tmp_path / "must-not-exist"
    dirty = lambda _repo: {  # noqa: E731
        "commit": "dirty-test", "dirty": True, "status": ["?? note"],
        "tracked_diff_sha256": "0" * 64,
    }
    result = exp022.run_bridge(
        out=out, dry_run=True, code_state_fn=dirty,
        tracker_identity_fn=_tracker_identity)
    assert result["status"] == "dry_run"
    assert result["project_dirty_observed"] is True
    assert result["writes_performed"] is False
    assert not out.exists()


def test_project_recheck_allows_only_the_campaign_output():
    initial = {
        "git": {"commit": "c", "dirty": False, "status": [],
                "tracked_diff_sha256": "0" * 64},
        "source_sha256": {"driver": "a" * 64},
        "physical_model": {"sha256": "b" * 64},
    }
    current = {
        **initial,
        "git": {"commit": "c", "dirty": True,
                "status": ["?? outputs/exp022_exact_tracking_bridge/"],
                "tracked_diff_sha256": "0" * 64},
    }
    check = exp022.validate_project_recheck(
        initial, current, exp022.ROOT / "outputs/exp022_exact_tracking_bridge")
    assert check["unexpected_status"] == []
    current["git"]["status"].append(" M scene2motion/robot.py")
    with pytest.raises(ValueError, match="outside"):
        exp022.validate_project_recheck(
            initial, current, exp022.ROOT / "outputs/exp022_exact_tracking_bridge")


def test_full_injected_bridge_is_two_32_motion_launches_and_idempotent(tmp_path):
    out = tmp_path / "bridge"
    calls = []

    def launch(pkl, eval_dir, num_envs, physics_seed, timeout_s):
        # Reference evidence must be durable before the first external process starts.
        assert (out / "reference_rows.jsonl").is_file()
        with Path(pkl).open("rb") as handle:
            motions = pickle.load(handle)
        calls.append((list(motions), num_envs, physics_seed, timeout_s))
        records = []
        for motion_id, (key, motion) in enumerate(motions.items()):
            qpos = np.asarray(motion["test_qpos"], dtype=np.float32)
            records.append(SonicRollout(key, qpos, len(qpos), False, 1.0, motion_id))
        return 0, _fake_sonic_artifacts(Path(eval_dir), records)

    receipt = exp022.run_bridge(
        out=out, launch_fn=launch, export_fn=_fake_export, scorer=_cheap_score,
        code_state_fn=_clean_code_state, tracker_identity_fn=_tracker_identity,
        mj_model=object())
    assert receipt["status"] == "complete"
    assert receipt["actual_ardy_samples"] == 0
    assert receipt["sonic_rollouts_returned"] == 64
    assert [(len(keys), n, seed) for keys, n, seed, _ in calls] == [
        (32, 32, 0), (32, 32, 0)]
    assert len((out / "reference_rows.jsonl").read_text().splitlines()) == 128
    assert len((out / "achieved_rows.jsonl").read_text().splitlines()) == 128
    summary = json.loads((out / "summary.json").read_text())
    assert summary["reference"]["staged"]["exact_clears"]["0.08"] == 64
    assert summary["reference"]["unstaged"]["exact_clears"]["0.08"] == 0
    paired = summary["paired_reference_to_achieved_retention"]["staged"]["0.08"]
    assert paired["reference_clear"] == 64
    assert paired["retained_reference_clear"] == 64
    assert paired["lost_reference_clear"] == 0
    assert "not contact-rich" in summary["interpretation_guard"].lower()

    again = exp022.run_bridge(
        out=out, launch_fn=launch, export_fn=_fake_export, scorer=_cheap_score,
        code_state_fn=_clean_code_state, tracker_identity_fn=_tracker_identity,
        mj_model=object())
    assert again["status"] == "complete"
    assert len(calls) == 2, "a complete campaign must never relaunch"

    first_attempt = Path(receipt["launches"]["chunk00_seed0"]["attempt"]) / "receipt.json"
    altered = json.loads(first_attempt.read_text())
    altered["manual_change"] = True
    cal._write_json(first_attempt, altered)
    with pytest.raises(exp022.BridgeAbort, match="artifacts changed"):
        exp022.run_bridge(
            out=out, launch_fn=launch, export_fn=_fake_export, scorer=_cheap_score,
            code_state_fn=_clean_code_state, tracker_identity_fn=_tracker_identity,
            mj_model=object())


def test_valid_interrupted_attempt_is_adopted_without_relaunch(tmp_path):
    chunk = exp022.chunk_plan()[0]
    chunk_dir = tmp_path / "launches" / chunk["name"]
    attempt = chunk_dir / "attempt-000"
    eval_dir = attempt / "eval"
    eval_dir.mkdir(parents=True)
    records = []
    for motion_id, key in enumerate(chunk["motion_keys"]):
        qpos = np.zeros((4, 36), dtype=np.float32)
        qpos[:, 2] = 0.78
        qpos[:, 3] = 1.0
        records.append(SonicRollout(key, qpos, len(qpos), False, 1.0, motion_id))
    (attempt / "sonic.log").write_text(_fake_sonic_artifacts(eval_dir, records))
    pkl = chunk_dir / "motions.pkl"
    _fake_export({key: np.zeros((200, 36)) for key in chunk["motion_keys"]}, pkl)
    cal._write_json(attempt / "receipt.json", {
        "status": "running",
        "chunk": chunk["name"],
        "physics_seed": 0,
        "motion_keys": chunk["motion_keys"],
        "motion_pkl_sha256": exp022._sha256(pkl),
    })
    _write_process_result(attempt, chunk, pkl)

    def forbidden_launch(*_args, **_kwargs):
        raise AssertionError("valid interrupted artifacts should be adopted")

    record, rollouts = exp022.run_or_resume_launch(
        chunk, pkl, tmp_path, launch_fn=forbidden_launch, timeout_s=1)
    assert record["recovered_or_resumed"] is True
    assert len(rollouts) == 32


def test_resume_refuses_to_relabel_attempt_input_provenance(tmp_path):
    chunk = exp022.chunk_plan()[0]
    chunk_dir = tmp_path / "launches" / chunk["name"]
    attempt = chunk_dir / "attempt-000"
    attempt.mkdir(parents=True)
    pkl = chunk_dir / "motions.pkl"
    _fake_export({key: np.zeros((200, 36)) for key in chunk["motion_keys"]}, pkl)
    cal._write_json(attempt / "receipt.json", {
        "status": "running", "chunk": chunk["name"], "physics_seed": 9,
        "motion_keys": chunk["motion_keys"],
        "motion_pkl_sha256": exp022._sha256(pkl),
    })
    with pytest.raises(exp022.BridgeAbort, match="physics_seed"):
        exp022.run_or_resume_launch(
            chunk, pkl, tmp_path, launch_fn=lambda *_args: (0, ""), timeout_s=1)


def test_wrong_archive_keys_fail_validation(tmp_path):
    attempt = tmp_path / "attempt"
    eval_dir = attempt / "eval"
    eval_dir.mkdir(parents=True)
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    records = [SonicRollout("wrong", qpos, 2, False, 1.0, 0)]
    (attempt / "sonic.log").write_text(_fake_sonic_artifacts(eval_dir, records))
    with pytest.raises(ValueError, match="keys"):
        exp022.validate_attempt(attempt, ["expected"])


def test_motion_id_to_key_swap_fails_even_when_key_set_and_metrics_agree(tmp_path):
    attempt = tmp_path / "attempt"
    eval_dir = attempt / "eval"
    eval_dir.mkdir(parents=True)
    qpos = np.zeros((2, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    # Both expected keys exist, and metrics are generated from this same swapped archive.
    # Only the independent pickle-order mapping can detect the relabel.
    records = [
        SonicRollout("s1", qpos, 2, False, 1.0, 0),
        SonicRollout("s0", qpos, 2, False, 1.0, 1),
    ]
    (attempt / "sonic.log").write_text(_fake_sonic_artifacts(eval_dir, records))
    with pytest.raises(ValueError, match="motion-id/key"):
        exp022.validate_attempt(attempt, ["s0", "s1"])


def test_valid_artifacts_without_durable_return_code_are_not_adopted(tmp_path):
    chunk = exp022.chunk_plan()[0]
    chunk_dir = tmp_path / "launches" / chunk["name"]
    attempt = chunk_dir / "attempt-000"
    eval_dir = attempt / "eval"
    eval_dir.mkdir(parents=True)
    records = []
    for motion_id, key in enumerate(chunk["motion_keys"]):
        qpos = np.zeros((4, 36), dtype=np.float32)
        qpos[:, 3] = 1.0
        records.append(SonicRollout(key, qpos, len(qpos), False, 1.0, motion_id))
    (attempt / "sonic.log").write_text(_fake_sonic_artifacts(eval_dir, records))
    pkl = chunk_dir / "motions.pkl"
    _fake_export({key: np.zeros((200, 36)) for key in chunk["motion_keys"]}, pkl)
    cal._write_json(attempt / "receipt.json", {
        "status": "running", "chunk": chunk["name"], "physics_seed": 0,
        "motion_keys": chunk["motion_keys"],
        "motion_pkl_sha256": exp022._sha256(pkl),
    })
    with pytest.raises(exp022.BridgeAbort, match="return-code evidence"):
        exp022.run_or_resume_launch(
            chunk, pkl, tmp_path, launch_fn=lambda *_args: (0, ""), timeout_s=1)


def test_later_failed_attempt_blocks_an_earlier_valid_attempt(tmp_path):
    chunk = exp022.chunk_plan()[0]
    chunk_dir = tmp_path / "launches" / chunk["name"]
    first = chunk_dir / "attempt-000"
    eval_dir = first / "eval"
    eval_dir.mkdir(parents=True)
    records = []
    for motion_id, key in enumerate(chunk["motion_keys"]):
        qpos = np.zeros((4, 36), dtype=np.float32)
        qpos[:, 3] = 1.0
        records.append(SonicRollout(key, qpos, len(qpos), False, 1.0, motion_id))
    (first / "sonic.log").write_text(_fake_sonic_artifacts(eval_dir, records))
    pkl = chunk_dir / "motions.pkl"
    _fake_export({key: np.zeros((200, 36)) for key in chunk["motion_keys"]}, pkl)
    cal._write_json(first / "receipt.json", {
        "status": "running", "chunk": chunk["name"], "physics_seed": 0,
        "motion_keys": chunk["motion_keys"],
        "motion_pkl_sha256": exp022._sha256(pkl),
    })
    _write_process_result(first, chunk, pkl)
    second = chunk_dir / "attempt-001"
    second.mkdir()
    cal._write_json(second / "receipt.json", {
        "status": "failed", "chunk": chunk["name"], "physics_seed": 0,
        "motion_keys": chunk["motion_keys"],
        "motion_pkl_sha256": exp022._sha256(pkl),
    })
    with pytest.raises(exp022.BridgeAbort, match="recorded failed"):
        exp022.run_or_resume_launch(
            chunk, pkl, tmp_path, launch_fn=lambda *_args: (0, ""), timeout_s=1)


def test_lateral_bypass_and_reversal_cannot_count_as_retained(monkeypatch):
    class AlwaysClear:
        def __init__(self, *_args, **_kwargs):
            pass

        def clears(self, _qpos, _height):
            return True

        def probe(self, _qpos):
            return 0.40

    monkeypatch.setattr(exp022, "BoxHeightProbe", AlwaysClear)
    qpos = np.zeros((4, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 0] = [0.0, 1.0, 1.4, 2.0]
    qpos[:, 1] = [0.0, 0.0, exp022.OBSTACLE_HALF_WIDTH_M + 0.01, 0.0]
    bypass = exp022.score_trajectory(qpos, 1.2)
    assert bypass["passed_obstacle"] is True
    assert bypass["passed_within_lateral_corridor"] is False
    assert not any(bypass["achieved_replay_clear_after_passing"].values())

    qpos[:, 1] = 0.0
    qpos[-1, 0] = 0.5
    reversal = exp022.score_trajectory(qpos, 1.2)
    assert reversal["passed_within_lateral_corridor"] is True
    assert reversal["finished_beyond_obstacle"] is False
    assert not any(reversal["achieved_replay_clear_after_passing"].values())


def test_project_recheck_rejects_numerical_runtime_change():
    initial = {
        "git": {"commit": "c", "dirty": False, "status": [],
                "tracked_diff_sha256": "0" * 64},
        "source_sha256": {"driver": "a" * 64},
        "runtime": {"sha256": "1" * 64},
        "physical_model": {"sha256": "b" * 64},
    }
    current = {**initial, "runtime": {"sha256": "2" * 64}}
    with pytest.raises(ValueError, match="numerical runtime"):
        exp022.validate_project_recheck(
            initial, current, exp022.ROOT / "outputs/exp022_exact_tracking_bridge")


def test_paired_summary_does_not_call_marginal_turnover_retention():
    reference_rows = []
    achieved_rows = []
    for seed in exp022.POOL_SEEDS:
        for label, x in exp022.OBSTACLES:
            reference_exact = {f"{height:g}": False for height in exp022.GRADED_HEIGHTS_M}
            achieved_exact = {f"{height:g}": False for height in exp022.GRADED_HEIGHTS_M}
            achieved_guarded = {
                f"{height:g}": False for height in exp022.GRADED_HEIGHTS_M
            }
            if label == "staged" and seed == exp022.POOL_SEEDS[0]:
                reference_exact["0.05"] = True
            if label == "staged" and seed == exp022.POOL_SEEDS[1]:
                achieved_exact["0.05"] = True
                achieved_guarded["0.05"] = True
            common = {
                "seed": seed, "obstacle_label": label, "obstacle_x_m": x,
                "passed_obstacle": True, "passed_within_lateral_corridor": True,
                "finished_beyond_obstacle": True, "route_completed": True,
                "tracker_terminated": False, "stalled": False,
                "stalled_before_obstacle": False,
            }
            reference_rows.append({
                **common, "exact_clears": reference_exact,
                "achieved_replay_clear_after_passing": reference_exact,
            })
            achieved_rows.append({
                **common, "exact_clears": achieved_exact,
                "achieved_replay_clear_after_passing": achieved_guarded,
            })
    summary = exp022.summarize(reference_rows, achieved_rows)
    paired = summary["paired_reference_to_achieved_retention"]["staged"]["0.05"]
    assert paired["reference_clear"] == 1
    assert paired["achieved_guarded_clear"] == 1
    assert paired["retained_reference_clear"] == 0
    assert paired["lost_reference_clear"] == 1
    assert paired["achieved_only_gain"] == 1
