"""CPU-only contract tests for the EXP-031 paired SONIC execution driver."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp030_obstacle_present as e30
from experiments import exp031_constructive_step_repair as x
from scene2motion.sonic_state_export import SonicRollout, write_sonic_state_archive


def _qpos(n: int = 397, x_end: float = 7.2, *, repaired: bool = False) -> np.ndarray:
    qpos = np.zeros((n, 36), dtype=np.float32)
    qpos[:, 0] = np.linspace(0.0, x_end, n)
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    qpos[:, 10] = float(repaired)
    return qpos


def _clip_sets(**_kwargs):
    return {
        "raw": {key: _qpos(e30.N_FRAMES, repaired=False) for key in x.CANDIDATE_KEYS},
        "repaired": {key: _qpos(e30.N_FRAMES, repaired=True) for key in x.CANDIDATE_KEYS},
    }


SOURCE = {"directory": "/fake/raw", "qpos": {"sha256": "a" * 64},
          "rows": {"sha256": "b" * 64}}
PROTOCOL = {"path": "/fake/protocol", "sha256": "c" * 64, "status": "preregistered"}


def _prepared(_directory=None):
    return {
        "directory": "/fake/prepared",
        "receipt": {"path": "/fake/prepared/receipt.json", "sha256": "d" * 64},
        "rows": {"path": "/fake/prepared/rows.jsonl", "sha256": "e" * 64, "n": 64},
        "qpos": {"path": "/fake/prepared/qpos.npz", "sha256": "f" * 64,
                 "keys": list(x.CANDIDATE_KEYS), "content_hashes": {}},
        "protocol": dict(PROTOCOL),
        "project": {"git": {"commit": "test-commit", "dirty": False}},
        "source": dict(SOURCE),
        "historical_outcome_disclosure": {
            "role": "disclosure_only",
            "source": {"path": "/fake/history", "sha256": "9" * 64},
            "candidates": {},
        },
        "support_rule": {}, "config": {},
        "summary": {
            "n_assigned_trials": 64, "n_input_support_pass": 8,
            "n_refused_input_support": 56, "n_rejected_after_projection": 6,
            "n_accepted_for_execution": 2, "accepted_keys": list(x.CANDIDATE_KEYS),
        },
    }


def _source(_directory=None):
    return dict(SOURCE)


def _project_state(_repo: Path):
    return {"commit": "test-commit", "dirty": False, "status": [],
            "tracked_diff_sha256": "0" * 64}


def _protocol():
    return dict(PROTOCOL)


def _tracker(sonic_root=None):
    return {
        "root": str(Path(sonic_root).resolve()), "branch": e30.SONIC_EXP029_BRANCH,
        "git": {"commit": e30.ADD_TABLE_FIX_COMMIT, "dirty": False, "status": []},
        "checkpoint": {"path": "/fake/release/last.pt", "sha256": "1" * 64},
        "core_source_manifest_sha256": "2" * 64,
        "evaluator_source_sha256": {"terminations.py": "3" * 64},
        "add_table_fix": {"fix_present": True, "sha256": "4" * 64, "problems": []},
        "python_runtime": {"python": "fake"},
        "isaaclab": {"git": {"commit": "isaac-test"}},
        "guarded_dirty_paths": [],
    }


def _gate(**_kwargs):
    return {"pass": True, "checks": {"vram": True, "ram": True, "no_isaac": True}}


def _isaac(**_kwargs):
    return []


def _export(clips, path, fps=25, mj_model=None):
    payload = {
        key: {"root_trans_offset": np.asarray(qpos[:, :3], dtype=np.float32),
              "test_qpos": np.asarray(qpos, dtype=np.float32), "fps": fps}
        for key, qpos in clips.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _termination_log(success_rate: float, progress_rate: float) -> str:
    flags = {name: name == "time_out" for name in e30.RELEASE_TERMINATION_TERMS}
    rows = "\n".join(
        f"|   {index}   | {name:<18} |  {str(flag):<6}  |"
        for index, (name, flag) in enumerate(flags.items())
    )
    return (
        f"[INFO] Termination Manager: <TerminationManager> contains {len(flags)} active terms.\n"
        "+---------------------------------------+\n"
        "|       Active Termination Terms        |\n"
        "+-------+--------------------+----------+\n"
        "| Index | Name               | Time Out |\n"
        "+-------+--------------------+----------+\n"
        f"{rows}\n"
        "+-------+--------------------+----------+\n"
        f"Success Rate:{success_rate:.10f}\nProgress Rate:{progress_rate:.10f}\n"
    )


def _launcher(calls: list[dict]):
    def launch(pkl, eval_dir, num_envs, physics_seed, timeout_s, extra_overrides):
        with Path(pkl).open("rb") as handle:
            motions = pickle.load(handle)
        repaired = bool(next(iter(motions.values()))["test_qpos"][0, 10] > 0.5)
        present = any("add_table=true" in value for value in extra_overrides)
        calls.append({"repaired": repaired, "present": present, "keys": list(motions)})
        rollouts = []
        for motion_id, key in enumerate(motions):
            completes = not present or (repaired and key == x.CANDIDATE_KEYS[0])
            valid = 397 if completes else 60
            achieved = _qpos(valid, 7.2 if completes else 1.0, repaired=repaired)
            rollouts.append(SonicRollout(
                key, achieved, valid, not completes, 1.0 if completes else 0.2, motion_id,
            ))
        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_sonic_state_archive(
            rollouts, eval_dir / "achieved_qpos.npz", sample_dt_s=e30.SAMPLE_DT_S,
        )
        success_rate = float(np.mean([not item.terminated for item in rollouts]))
        progress_rate = float(np.mean([item.progress for item in rollouts]))
        cal._write_json(eval_dir / "metrics_eval.json", {
            "eval/all_metrics_dict": {
                "motion_keys": [item.motion_key for item in rollouts],
                "terminated": [item.terminated for item in rollouts],
                "progress": [item.progress for item in rollouts],
            },
            "failed_keys": [item.motion_key for item in rollouts if item.terminated],
            "eval/success/success_rate": success_rate,
            "eval/success/progress_rate": progress_rate,
        })
        return 0, _termination_log(success_rate, progress_rate)
    return launch


def _collision(scene, qpos):
    return {
        "collision_free": True, "penetration_frames": 0, "max_penetration_m": 0.0,
        "min_clearance_m": 0.01, "worst": {"frame": -1, "depth_m": 0.0},
        "first": {"frame": -1, "depth_m": 0.0},
    }


def _kwargs(out: Path, calls: list[dict]):
    sonic_root = out.parent / "patched-sonic"
    sonic_root.mkdir(parents=True, exist_ok=True)
    return {
        "out": out, "prepared_dir": "/fake/prepared", "raw_source": "/fake/raw",
        "sonic_root": sonic_root, "launch_fn": _launcher(calls), "export_fn": _export,
        "host_gate_fn": _gate, "host_report_fn": _gate, "isaac_fn": _isaac,
        "code_state_fn": _project_state, "tracker_identity_fn": _tracker,
        "protocol_identity_fn": _protocol, "prepared_identity_fn": _prepared,
        "source_identity_fn": _source, "clip_sets_fn": _clip_sets,
        "collision_fn": _collision, "mj_model": object(),
    }


def test_launch_plan_is_four_paired_arms_over_the_same_two_candidates():
    plan = x.launch_plan()
    assert [item["arm"] for item in plan] == list(x.ARM_NAMES)
    assert all(tuple(item["motion_keys"]) == x.CANDIDATE_KEYS for item in plan)
    assert all(item["n_motions"] == 2 for item in plan)
    assert [item["obstacle_in_physics"] for item in plan] == [False, False, True, True]
    assert plan[2]["table"]["size_xyz"][2] == 0.05


def test_dry_run_writes_nothing_and_exposes_all_four_commands(tmp_path):
    out = tmp_path / "pilot"
    report = x.run_campaign(dry_run=True, **_kwargs(out, []))
    assert report["writes_performed"] is False
    assert len(report["launch_plan"]) == 4
    assert set(report["commands"]) == {item["name"] for item in x.launch_plan()}
    assert not out.exists()


def test_mocked_campaign_closes_one_trial_and_keeps_both_denominators(tmp_path):
    out = tmp_path / "pilot"
    calls: list[dict] = []
    receipt = x.run_campaign(**_kwargs(out, calls))
    summary = receipt["summary"]
    assert len(calls) == 4
    assert receipt["sonic_rollouts_requested"] == 8
    assert receipt["sonic_rollouts_returned"] == 8
    assert summary["P1_first_closure"]["held"] is True
    assert summary["P1_first_closure"]["completed"] == 1
    assert summary["denominators"]["candidate_conditional"]["n"] == 2
    assert summary["denominators"]["candidate_conditional"]["rate"] == 0.5
    assert summary["denominators"]["source_pool"]["n"] == 64
    assert summary["denominators"]["source_pool"]["rate"] == 1 / 64
    assert summary["denominators"]["source_pool"][
        "n_not_executed_after_preexecution_disposition"
    ] == 62
    assert len((out / "rows.jsonl").read_text().splitlines()) == 8
    assert json.loads((out / "summary.json").read_text()) == summary
    x.validate_completed_output(out, receipt)
