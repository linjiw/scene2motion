"""CPU tests for the EXP-028 termination-free rollout / physics-seed re-roll driver."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp022_exact_tracking_bridge as exp022
from experiments import exp028_termination_free_rollouts as x
from scene2motion import host_gate as hg
from scene2motion.sonic_state_export import SonicRollout, write_sonic_state_archive

FULL_LEN = 397  # EXP-022A survivors: ref_len - 1 archived samples


# ------------------------------------------------------------------------------ fakes

def _clean_code_state(_repo: Path) -> dict:
    return {"commit": "test-commit", "dirty": False, "status": [], "tracked_diff_sha256": "0" * 64}


def _dirty_code_state(_repo: Path) -> dict:
    return {"commit": "dirty-test", "dirty": True, "status": ["?? note"],
            "tracked_diff_sha256": "0" * 64}


def _tracker_identity() -> dict:
    return {
        "root": "/fake/sonic",
        "git": {"commit": "tracker-test", "dirty": True, "tracked_dirty": True,
                "status": [" M gear_sonic/research/practice_utility/x.py"]},
        "dirty_paths": ["gear_sonic/research/practice_utility/x.py"],
        "core_source_sha256": {"eval.py": "a" * 64},
        "core_source_manifest_sha256": x.EXPECTED_CORE_MANIFEST_SHA256,
        "evaluator_source_sha256": {"terminations.py": "c" * 64},
        "checkpoint": {"sha256": x.EXPECTED_CHECKPOINT_SHA256},
        "python_runtime": {"packages": {"torch": "test"}},
        "isaaclab": {"git": {"commit": "isaac-test"}},
    }


def _protocol(status: str = "preregistered"):
    def identity() -> dict:
        return {"path": "/fake/protocol.md", "sha256": "d" * 64, "status": status}
    return identity


def _gate_pass(**_kwargs) -> dict:
    return {"pass": True, "checks": {"vram": True, "ram": True, "no_isaac": True},
            "vram": {"free_mib": 15000}, "ram": {"available_mib": 24000},
            "concurrent_isaac_processes": []}


def _gate_fail(**_kwargs) -> dict:
    raise hg.HostResourceGateFailed("host-resource gate failed on vram, no_isaac: free VRAM "
                                    "4436 MiB, available RAM 8000 MiB, 2 Isaac process(es)")


def _term(func: str, params: dict) -> dict:
    return {"_target_": "isaaclab.managers.TerminationTermCfg", "func": func,
            "params": {"command_name": "motion", **params}}


def _release_terms() -> dict:
    return {
        "_target_": "gear_sonic.envs.manager_env.mdp.terminations.TerminationsCfg",
        "anchor_pos": _term("gear_sonic.envs.manager_env.mdp:exceeded_anchor_height",
                            {"threshold": 0.25, "threshold_adaptive": False,
                             "down_threshold": 0.25, "root_height_threshold": 0.5}),
        "anchor_ori_full": _term("gear_sonic.envs.manager_env.mdp:exceeded_anchor_ori",
                                 {"asset_cfg": {"name": "robot"}, "threshold": 1.0}),
        "ee_body_pos": _term("gear_sonic.envs.manager_env.mdp:exceeded_body_height",
                             {"threshold": 0.25, "threshold_adaptive": False,
                              "down_threshold": 0.25, "root_height_threshold": 0.5,
                              "body_names": list(x.EE_HEIGHT_BODIES)}),
        "foot_pos_xyz": _term("gear_sonic.envs.manager_env.mdp:exceeded_body_pos",
                              {"threshold": 0.2, "body_names": list(x.FOOT_BODIES)}),
        "time_out": {"_target_": "isaaclab.managers.TerminationTermCfg",
                     "func": "gear_sonic.envs.manager_env.mdp:tracking_time_out",
                     "time_out": True, "params": {"command_name": "motion"}},
    }


def _termfree_terms() -> dict:
    terms = _release_terms()
    for name in x.EXPECTED_TRACKING_TERMS:
        params = terms[name]["params"]
        params["threshold"] = 1000000.0
        if "down_threshold" in params:
            params["down_threshold"] = 1000000.0
        if "threshold_adaptive" in params:
            params["threshold_adaptive"] = False
    return terms


def _fake_compose(overrides) -> dict:
    overrides = list(overrides)
    termfree = any(item.endswith("=1e6") for item in overrides)
    return {"terminations": _termfree_terms() if termfree else _release_terms(),
            "overrides": overrides, "hydra_version": "test", "omegaconf_version": "test",
            "method": "fake"}


def _fake_export(clips, path, fps=25, mj_model=None):
    payload = {key: {"root_trans_offset": np.asarray(q[:, :3], dtype=np.float32),
                     "test_qpos": np.asarray(q, dtype=np.float32), "fps": fps}
               for key, q in clips.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _termination_table(names_time_out: dict[str, bool]) -> str:
    rows = "\n".join(f"|   {i}   | {name:<15} |  {str(flag):<6}  |"
                     for i, (name, flag) in enumerate(names_time_out.items()))
    return (f"[INFO] Termination Manager:  <TerminationManager> contains {len(names_time_out)} "
            "active terms.\n+------------------------------------+\n"
            "|      Active Termination Terms      |\n+-------+-----------------+----------+\n"
            "| Index | Name            | Time Out |\n+-------+-----------------+----------+\n"
            f"{rows}\n+-------+-----------------+----------+\n")


def _fake_sonic_artifacts(eval_dir: Path, records: list[SonicRollout]) -> str:
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_sonic_state_archive(records, eval_dir / "achieved_qpos.npz", sample_dt_s=0.02)
    success_rate = float(np.mean([not r.terminated for r in records]))
    progress_rate = float(np.mean([r.progress for r in records]))
    cal._write_json(eval_dir / "metrics_eval.json", {
        "eval/all_metrics_dict": {"motion_keys": [r.motion_key for r in records],
                                  "terminated": [r.terminated for r in records],
                                  "progress": [r.progress for r in records]},
        "failed_keys": [r.motion_key for r in records if r.terminated],
        "eval/success/success_rate": success_rate,
        "eval/success/progress_rate": progress_rate,
    })
    table = _termination_table({"time_out": True, "anchor_pos": False, "anchor_ori_full": False,
                                "ee_body_pos": False, "foot_pos_xyz": False})
    return f"{table}\nSuccess Rate:{success_rate:.10f}\nProgress Rate:{progress_rate:.10f}\n"


def _walk_qpos(n: int, x_end: float = 2.0) -> np.ndarray:
    q = np.zeros((n, 36), dtype=np.float32)
    q[:, 0] = np.linspace(0.0, x_end, n)
    q[:, 2] = 0.78
    q[:, 3] = 1.0
    return q


def _make_launcher(calls: list, *, terminate_every: int = 3, terminate_termfree: bool = False):
    def launch(pkl, eval_dir, num_envs, physics_seed, timeout_s, extra_overrides):
        with Path(pkl).open("rb") as handle:
            motions = pickle.load(handle)
        calls.append({"keys": list(motions), "num_envs": num_envs, "physics_seed": physics_seed,
                      "extra_overrides": list(extra_overrides), "eval_dir": str(eval_dir)})
        termfree = bool(extra_overrides)
        records = []
        for motion_id, key in enumerate(motions):
            terminated = (not termfree and motion_id % terminate_every == 0) or (
                termfree and terminate_termfree and motion_id == 0)
            valid = 50 if terminated else FULL_LEN
            progress = valid / (FULL_LEN + 1) if terminated else 1.0
            records.append(SonicRollout(key, _walk_qpos(valid, 2.0 if not terminated else 0.9),
                                        valid, terminated, progress, motion_id))
        return 0, _fake_sonic_artifacts(Path(eval_dir), records)
    return launch


def _cheap_score(qpos, obstacle_x_m, *, terminated=False, reported_progress=None):
    qpos = np.asarray(qpos)
    passed = bool(len(qpos) and float(qpos[:, 0].max()) >= 1.34)
    exact = {f"{h:g}": bool(passed and h <= 0.08) for h in x.GRADED_HEIGHTS_M}
    return {
        "valid_frames": len(qpos), "tracker_terminated": bool(terminated),
        "tracker_reported_progress": reported_progress,
        "max_root_x_m": float(qpos[:, 0].max()) if len(qpos) else None,
        "final_root_x_m": float(qpos[-1, 0]) if len(qpos) else None,
        "max_abs_root_y_m": float(np.abs(qpos[:, 1]).max()) if len(qpos) else None,
        "pass_frame": 1 if passed else None, "root_y_at_pass_m": 0.0 if passed else None,
        "actual_route_progress_ratio": 1.0 if passed else 0.1,
        "passed_obstacle": passed, "passed_within_lateral_corridor": passed,
        "finished_beyond_obstacle": passed, "route_completed": passed,
        "stalled": bool(not terminated and not passed), "stalled_before_obstacle": False,
        "max_box_height_lower_bound_m": 0.079, "exact_clears": exact,
        "achieved_replay_clear_after_passing": {
            k: bool(not terminated and passed and v) for k, v in exact.items()},
    }


def _cheap_contract(qpos, fps):
    return {"max_unsupported_run_s": 0.0, "fps": fps, "n": int(len(qpos))}


def _cheap_evaluator(reference, achieved):
    n = int(len(achieved))
    per_term = {name: {"threshold": thr, "max_value": 0.0, "max_ratio_to_threshold": 0.1,
                       "n_samples_exceeding": 0, "first_exceed_sample": None,
                       "first_exceed_time_s": None}
                for name, thr in x.EVALUATOR_TERM_THRESHOLDS.items()}
    return {"values": {name: np.zeros(n, dtype=np.float32) for name in per_term},
            "exceeded": {}, "per_term": per_term, "first_firing_sample": None,
            "first_firing_time_s": None, "first_firing_reference_step": None,
            "first_firing_terms": [], "any_term_exceeded": False,
            "closest_term_at_last_sample": "foot_pos_xyz",
            "ratio_to_threshold_at_last_sample": {name: 0.1 for name in per_term},
            "n_samples": n}


def _campaign_kwargs(out: Path, calls: list, **overrides):
    kwargs = dict(
        out=out, launch_fn=_make_launcher(calls), export_fn=_fake_export,
        compose_fn=_fake_compose, host_gate_fn=_gate_pass, host_report_fn=_gate_pass,
        code_state_fn=_clean_code_state, tracker_identity_fn=_tracker_identity,
        protocol_identity_fn=_protocol(), scorer=_cheap_score, contract_fn=_cheap_contract,
        evaluator_fn=_cheap_evaluator, mj_model=object(),
    )
    kwargs.update(overrides)
    return kwargs


# ------------------------------------------------------------------------------ overrides

def test_override_list_covers_every_active_term_and_leaves_time_out():
    coverage = x.override_coverage(_release_terms())
    assert coverage["tracking_terms"] == sorted(x.EXPECTED_TRACKING_TERMS)
    assert coverage["time_out_term"] == "time_out"
    assert all(item.startswith("++manager_env.terminations.") for item in x.TERMINATION_FREE_OVERRIDES)
    assert not any("time_out" in item for item in x.TERMINATION_FREE_OVERRIDES)

    extra = _release_terms()
    extra["anchor_pos_xy"] = _term("gear_sonic.envs.manager_env.mdp:exceeded_anchor_pos_xy",
                                   {"threshold": 0.5})
    with pytest.raises(ValueError, match="reachable"):
        x.override_coverage(extra)
    with pytest.raises(ValueError, match="motion_time_out"):
        x.override_coverage(_release_terms(),
                            overrides=(*x.TERMINATION_FREE_OVERRIDES,
                                       "++manager_env.terminations.time_out.params.threshold=1e6"))
    partial = [item for item in x.TERMINATION_FREE_OVERRIDES if "foot_pos_xyz" not in item]
    with pytest.raises(ValueError, match="foot_pos_xyz.threshold"):
        x.override_coverage(_release_terms(), overrides=partial)


def test_audit_distinguishes_termination_free_from_release():
    free = x.audit_terminations(_termfree_terms(), "termination_free")
    assert free["pass"] and free["active_terms"] == sorted([*x.EXPECTED_TRACKING_TERMS, "time_out"])
    assert free["tracking_terms"]["anchor_pos"]["threshold"] >= 1e6
    assert free["tracking_terms"]["anchor_pos"]["threshold_adaptive"] is False
    release = x.audit_terminations(_release_terms(), "release")
    assert release["tracking_terms"]["foot_pos_xyz"]["threshold"] == 0.2
    assert release["tracking_terms"]["anchor_ori_full"]["threshold"] == 1.0
    with pytest.raises(ValueError, match="reachable"):
        x.audit_terminations(_release_terms(), "termination_free")
    with pytest.raises(ValueError, match="release evaluator"):
        x.audit_terminations(_termfree_terms(), "release")
    no_timeout = _termfree_terms()
    del no_timeout["time_out"]
    with pytest.raises(ValueError, match="time-out"):
        x.audit_terminations(no_timeout, "termination_free")
    still_adaptive = _termfree_terms()
    still_adaptive["ee_body_pos"]["params"]["threshold_adaptive"] = True
    with pytest.raises(ValueError, match="threshold_adaptive"):
        x.audit_terminations(still_adaptive, "termination_free")


def test_parse_log_termination_terms_reads_sonic_table():
    log = "noise\n" + _termination_table({"time_out": True, "anchor_pos": False,
                                           "anchor_ori_full": False, "ee_body_pos": False,
                                           "foot_pos_xyz": False}) + "\nSuccess Rate:1.0\n"
    assert x.parse_log_termination_terms(log) == {
        "time_out": True, "anchor_pos": False, "anchor_ori_full": False,
        "ee_body_pos": False, "foot_pos_xyz": False}
    with pytest.raises(ValueError, match="termination tables"):
        x.parse_log_termination_terms("no table here")


# ------------------------------------------------------------------------------ plan

def test_launch_plan_matches_exp022a_chunks_and_seeds():
    plan = x.campaign_launch_plan()
    chunks = exp022.chunk_plan()
    assert [spec["name"] for spec in plan] == [
        "partA_chunk00_seed0_termfree", "partA_chunk01_seed0_termfree",
        "partB_chunk00_seed1", "partB_chunk01_seed1",
        "partB_chunk00_seed2", "partB_chunk01_seed2"]
    assert [spec["physics_seed"] for spec in plan] == [0, 0, 1, 1, 2, 2]
    for spec in plan:
        chunk = chunks[spec["chunk"]]
        assert spec["motion_keys"] == chunk["motion_keys"]
        assert spec["seeds"] == chunk["seeds"]
        assert spec["n_motions"] == 32
        if spec["part"] == "part_a":
            assert spec["extra_overrides"] == list(x.TERMINATION_FREE_OVERRIDES)
            assert spec["config_expectation"] == "termination_free"
        else:
            assert spec["extra_overrides"] == [] and spec["config_expectation"] == "release"
    smoke = x.smoke_launch_plan()
    assert [s["part"] for s in smoke] == ["smoke_a", "smoke_b"]
    assert all(s["motion_keys"] == list(x.SMOKE_MOTION_KEYS) and s["n_motions"] == 2 for s in smoke)
    assert smoke[0]["physics_seed"] == 0 and smoke[1]["physics_seed"] == 1


def test_command_extends_exp022a_launcher_with_overrides():
    from experiments import exp1b_execution_clearance as exp1b
    from scene2motion.sonic_state_export import sonic_state_hydra_overrides
    pkl, eval_dir = Path("/tmp/m.pkl"), Path("/tmp/eval")
    release = x.build_sonic_command(pkl, eval_dir, 32, 2)
    assert release == [str(exp1b.SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl",
                       f"+checkpoint={exp1b.CKPT}", "+headless=True", "++eval_callbacks=im_eval",
                       "++run_eval_loop=False", "++num_envs=32", f"++eval_output_dir={eval_dir}",
                       "++seed=2", "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
                       "+manager_env/terminations=tracking/eval",
                       f"+manager_env.commands.motion.motion_lib_cfg.motion_file={pkl}",
                       "+log_keys=m", *sonic_state_hydra_overrides()]
    free = x.build_sonic_command(pkl, eval_dir, 32, 2, x.TERMINATION_FREE_OVERRIDES)
    assert free[:len(release)] == release
    assert free[len(release):] == list(x.TERMINATION_FREE_OVERRIDES)
    assert "++seed=0" in x.build_sonic_command(pkl, eval_dir, 32, 0)
    assert x.hydra_overrides_of(free)[0] == f"+checkpoint={exp1b.CKPT}"


# ------------------------------------------------------------------------------ outcomes

class _AlwaysClear:
    def __init__(self, *_a, **_k):
        pass

    def clears(self, _q, _h):
        return True

    def probe(self, _q):
        return 0.40


class _NeverClear(_AlwaysClear):
    def clears(self, _q, _h):
        return False

    def probe(self, _q):
        return 0.0


def test_outcome_classifier_one_trajectory_per_class(monkeypatch):
    monkeypatch.setattr(exp022, "BoxHeightProbe", _AlwaysClear)
    cleared = x.physical_outcome(_walk_qpos(40, 2.0))
    assert cleared["physical_state"] == "reached_box"
    assert set(cleared["outcome_class"].values()) == {"cleared"}

    stalled = x.physical_outcome(_walk_qpos(40, 0.9))
    assert stalled["physical_state"] == "stalled"
    assert set(stalled["outcome_class"].values()) == {"stalled"}
    assert stalled["reached_box_front_edge"] is False

    fell = _walk_qpos(40, 2.0)
    fell[20:, 2] = 0.40
    fallen = x.physical_outcome(fell)
    assert fallen["physical_state"] == "fell" and fallen["fell_by_pelvis_height"]
    assert fallen["fell_first_sample"] == 20 and fallen["fell_first_time_s"] == pytest.approx(0.42)
    assert set(fallen["outcome_class"].values()) == {"fell"}

    tilted = _walk_qpos(40, 2.0)
    tilted[30:, 3:7] = [np.cos(0.5), np.sin(0.5), 0.0, 0.0]  # 1.0 rad roll: up_z = cos(1) < 0.7
    assert x.physical_outcome(tilted)["fell_by_tilt"] is True

    stopped = x.physical_outcome(_walk_qpos(40, 1.2))
    assert stopped["reached_box_front_edge"] and set(stopped["outcome_class"].values()) == {
        "residual_stopped_at_box"}

    bypass = _walk_qpos(40, 2.0)
    bypass[:, 1] = exp022.OBSTACLE_HALF_WIDTH_M + 0.05
    assert set(x.physical_outcome(bypass)["outcome_class"].values()) == {
        "residual_bypassed_or_reversed"}

    monkeypatch.setattr(exp022, "BoxHeightProbe", _NeverClear)
    walked = x.physical_outcome(_walk_qpos(40, 2.0))
    assert set(walked["outcome_class"].values()) == {"walked_through"}
    assert all(walked["collides_at_height"].values())


def test_classify_outcome_applies_the_preregistered_order():
    everything = dict(reached_front_edge=True, passed_obstacle=True, passed_within_corridor=True,
                      finished_beyond=True, exact_clear=True)
    assert x.classify_outcome(fell=True, **everything) == "fell"
    assert x.classify_outcome(fell=False, **everything) == "cleared"
    assert x.classify_outcome(fell=False, **{**everything, "reached_front_edge": False}) == "stalled"
    assert x.classify_outcome(fell=False, **{**everything, "exact_clear": False}) == "walked_through"
    assert x.classify_outcome(fell=False, **{**everything, "passed_obstacle": False}) == "residual_stopped_at_box"
    assert x.classify_outcome(fell=False, **{**everything, "finished_beyond": False}) == "residual_bypassed_or_reversed"


# ------------------------------------------------------------------------------ evaluator terms

def _yaw_quat(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def test_resample_reference_50hz_grid():
    with np.load(exp022.SOURCE_OUT / "qpos.npz") as archive:
        q25 = np.asarray(archive["s4400"], dtype=float)[:12]
    q50 = x.resample_reference_50hz(q25)
    assert q50.shape == (22, 36)
    assert np.allclose(q50[0::2], q25[:-1])
    assert np.allclose(q50[1, :3], 0.5 * (q25[0, :3] + q25[1, :3]))
    assert np.allclose(q50[1, 7:], 0.5 * (q25[0, 7:] + q25[1, 7:]))
    assert np.isclose(np.linalg.norm(q50[1, 3:7]), 1.0)


def test_evaluator_terms_known_answers():
    with np.load(exp022.SOURCE_OUT / "qpos.npz") as archive:
        q25 = np.asarray(archive["s4400"], dtype=float)[:12]
    geometry = x.EvaluatorGeometry()
    ref50 = x.resample_reference_50hz(q25)
    achieved = ref50[:20].copy()
    exact = x.evaluator_terms(geometry, ref50, achieved)
    assert exact["any_term_exceeded"] is False and exact["first_firing_terms"] == []
    assert exact["n_samples"] == 20
    assert exact["per_term"]["anchor_pos"]["max_value"] == pytest.approx(0.0, abs=1e-9)
    assert exact["per_term"]["anchor_ori_full"]["max_value"] == pytest.approx(0.0, abs=1e-9)
    assert exact["per_term"]["ee_body_pos"]["max_value"] == pytest.approx(0.0, abs=1e-9)
    # foot_pos_xyz re-anchors to the previous sample's pelvis: only the per-step pelvis motion
    # of a walking reference remains, far below 0.2 m.
    assert exact["per_term"]["foot_pos_xyz"]["max_value"] < 0.05

    raised = achieved.copy()
    raised[8:, 2] += 0.30
    terms = x.evaluator_terms(geometry, ref50, raised)
    assert terms["first_firing_sample"] == 8 and terms["first_firing_time_s"] == pytest.approx(0.18)
    assert terms["first_firing_terms"] == ["anchor_pos", "ee_body_pos", "foot_pos_xyz"]
    assert terms["per_term"]["anchor_pos"]["max_value"] == pytest.approx(0.30)
    assert terms["per_term"]["anchor_ori_full"]["first_exceed_sample"] is None

    yawed = achieved.copy()
    for i in range(5, len(yawed)):
        yawed[i, 3:7] = x._quat_mul(_yaw_quat(1.2), yawed[i, 3:7])
    terms = x.evaluator_terms(geometry, ref50, yawed)
    assert terms["per_term"]["anchor_ori_full"]["first_exceed_sample"] == 5
    assert terms["per_term"]["anchor_ori_full"]["max_value"] == pytest.approx(1.44, abs=1e-6)
    assert terms["per_term"]["anchor_pos"]["first_exceed_sample"] is None
    assert terms["first_firing_sample"] == 5
    assert "anchor_ori_full" in terms["first_firing_terms"]

    assert x.quat_error_angle(_yaw_quat(0.3), _yaw_quat(-0.2)) == pytest.approx(0.5)
    assert np.allclose(x.heading_quat(x._quat_mul(_yaw_quat(0.4), [np.cos(0.1), np.sin(0.1), 0, 0])),
                       _yaw_quat(0.4), atol=1e-6)
    with pytest.raises(ValueError, match="outruns"):
        x.evaluator_terms(geometry, ref50[:5], achieved)


# ------------------------------------------------------------------------------ statistics

def test_agreement_matrix_counts_pairwise_and_unanimous():
    labels = {0: {"a": True, "b": False, "c": True, "d": False},
              1: {"a": True, "b": True, "c": True, "d": False},
              2: {"a": False, "b": False, "c": True, "d": False}}
    result = x.agreement_matrix(labels)
    assert result["n_clips"] == 4
    assert result["pairwise"]["0-1"]["agree"] == 3
    assert result["pairwise"]["0-2"]["agree"] == 3
    assert result["pairwise"]["1-2"]["agree"] == 2
    assert result["unanimous"]["agree"] == 2 and result["unanimous"]["fraction"] == 0.5
    assert result["matrix"][0][0] == 1.0 and result["matrix"][1][2] == 0.5
    lo, hi = result["unanimous"]["wilson95"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert result["terminated_per_seed"]["1"]["terminated"] == 3


def _synthetic_part_rows():
    part_a = []
    for i, seed in enumerate(x.POOL_SEEDS):
        clear_ref = i < 12
        cls = "fell" if i < 9 else ("cleared" if i < 12 else "walked_through")
        part_a.append({
            "seed": seed, "motion_key": f"s{seed}", "tracker_terminated": False,
            "physical_state": "fell" if cls == "fell" else "reached_box",
            "reference_exact_clears": {f"{h:g}": clear_ref for h in x.GRADED_HEIGHTS_M},
            "outcome_class": {f"{h:g}": cls for h in x.GRADED_HEIGHTS_M},
            "evaluator_terms": {"first_firing_terms": ["foot_pos_xyz"], "first_firing_time_s": 0.5,
                                "any_term_exceeded": True, "n_samples": 397,
                                "per_term": {"foot_pos_xyz": {"n_samples_exceeding": 3}},
                                "closest_term_at_last_sample": "foot_pos_xyz"},
        })
    part_b = []
    for physics_seed in (0, 1, 2):
        for i, seed in enumerate(x.POOL_SEEDS):
            terminated = (i % 2 == 0) if physics_seed < 2 else (i % 4 == 0)
            part_b.append({
                "seed": seed, "motion_key": f"s{seed}", "physics_seed": physics_seed,
                "tracker_terminated": terminated, "valid_frames": 50 if terminated else 397,
                "reference_exact_clears": {f"{h:g}": i < 12 for h in x.GRADED_HEIGHTS_M},
                "achieved_replay_clear_after_passing": {
                    f"{h:g}": (i < 12 and not terminated and physics_seed == 2)
                    for h in x.GRADED_HEIGHTS_M},
                "evaluator_terms": {"any_term_exceeded": False, "n_samples": 50,
                                    "per_term": {"foot_pos_xyz": {"n_samples_exceeding": 0}},
                                    "closest_term_at_last_sample": "ee_body_pos"},
            })
    return part_a, part_b


def test_summaries_and_decision_rules():
    part_a, part_b = _synthetic_part_rows()
    summary_a = x.summarize_part_a(part_a)
    assert summary_a["outcome_class_by_height"]["0.05"]["fell"] == 9
    assert summary_a["reference_exact_clear_subset_by_height"]["0.05"] == {
        **summary_a["reference_exact_clear_subset_by_height"]["0.05"],
        "n_reference_exact_clear": 12, "fell": 9, "cleared": 3, "fell_or_walked_through": 9}
    assert summary_a["evaluator_first_firing_terms"] == {"foot_pos_xyz": 64}
    summary_b = x.summarize_part_b(part_b)
    assert summary_b["terminated_per_seed"] == {"0": 32, "1": 32, "2": 16}
    assert summary_b["agreement"]["pairwise"]["0-1"]["fraction"] == 1.0
    assert summary_b["agreement"]["unanimous"]["fraction"] == 0.75
    retention = summary_b["retention_over_all_seeds"]["0.05"]
    assert retention["n_rollouts"] == 36 and retention["retained_rollouts"] == 9
    decisions = x.evaluate_decision_rules(summary_a, summary_b)
    assert decisions["not_tracked_wording"]["allowed"] is True
    assert decisions["test_retest"]["single_seed_zeros_are_lower_bounds"] is True
    consistency = x.offline_term_consistency([r for r in part_b if r["physics_seed"] == 0])
    assert consistency["consistent"] and consistency["n_alive_samples"] == 64 * 50

    part_a[0]["outcome_class"] = {f"{h:g}": "cleared" for h in x.GRADED_HEIGHTS_M}
    assert x.evaluate_decision_rules(x.summarize_part_a(part_a), summary_b)[
        "not_tracked_wording"]["allowed"] is False


# ------------------------------------------------------------------------------ campaign

def test_dry_run_allows_dirty_tree_and_writes_nothing(tmp_path):
    out = tmp_path / "must-not-exist"
    calls: list = []
    result = x.run_campaign(stage="all", dry_run=True, **_campaign_kwargs(
        out, calls, code_state_fn=_dirty_code_state, protocol_identity_fn=_protocol("draft")))
    assert result["status"] == "dry_run" and result["writes_performed"] is False
    assert result["project_dirty_observed"] is True
    assert len(result["launch_plan"]) == 8 and len(result["commands"]) == 8
    assert result["commands"]["partA_chunk00_seed0_termfree"][-1] == x.TERMINATION_FREE_OVERRIDES[-1]
    assert "++seed=2" in result["commands"]["partB_chunk01_seed2"]
    assert result["host_resource_gate"]["pass"] is True
    assert result["full_reference_valid_length"] == FULL_LEN
    assert not out.exists() and calls == []


def test_production_refuses_draft_protocol_and_dirty_tree(tmp_path):
    calls: list = []
    with pytest.raises(x.BridgeAbort, match="preregistered"):
        x.run_campaign(stage="smoke", **_campaign_kwargs(
            tmp_path / "a", calls, protocol_identity_fn=_protocol("draft")))
    with pytest.raises(x.BridgeAbort, match="clean"):
        x.run_campaign(stage="smoke", **_campaign_kwargs(
            tmp_path / "b", calls, code_state_fn=_dirty_code_state))
    assert calls == [] and not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_part_a_refuses_without_a_passing_smoke_receipt(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    with pytest.raises(x.BridgeAbort, match="smoke_a"):
        x.run_campaign(stage="part_a", **_campaign_kwargs(out, calls))
    assert calls == []
    assert (out / "receipt.json").is_file()  # ledger persisted before any launch
    with pytest.raises(x.BridgeAbort, match="smoke_b"):
        x.run_campaign(stage="part_b", resume=True, **_campaign_kwargs(out, calls))


def test_smoke_fails_closed_when_termination_free_rollout_terminates(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    kwargs = _campaign_kwargs(out, calls,
                              launch_fn=_make_launcher(calls, terminate_termfree=True))
    with pytest.raises(x.BridgeAbort, match="still terminated"):
        x.run_campaign(stage="smoke", **kwargs)
    smoke = json.loads((out / "smoke_receipt.json").read_text())
    assert smoke["smoke_a"]["pass"] is False
    assert x.smoke_passed(out, "smoke_a") is False
    with pytest.raises(x.BridgeAbort, match="blocked|smoke_a"):
        x.run_campaign(stage="part_a", resume=True, **_campaign_kwargs(out, calls))


def test_full_injected_campaign_is_eight_launches_and_idempotent(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    receipt = x.run_campaign(stage="all", **_campaign_kwargs(out, calls))
    assert receipt["status"] == "complete" and receipt["actual_ardy_samples"] == 0
    assert [c["num_envs"] for c in calls] == [2, 2, 32, 32, 32, 32, 32, 32]
    assert [c["physics_seed"] for c in calls] == [0, 1, 0, 0, 1, 1, 2, 2]
    assert [bool(c["extra_overrides"]) for c in calls] == [True, False, True, True, False, False,
                                                            False, False]
    assert calls[2]["keys"] == exp022.chunk_plan()[0]["motion_keys"]
    assert calls[7]["keys"] == exp022.chunk_plan()[1]["motion_keys"]
    smoke = json.loads((out / "smoke_receipt.json").read_text())
    assert smoke["smoke_a"]["pass"] and smoke["smoke_b"]["pass"]
    assert smoke["smoke_a"]["audit"]["expectation"] == "termination_free"
    assert smoke["smoke_b"]["audit"]["tracking_terms"]["foot_pos_xyz"]["threshold"] == 0.2
    for name, record in receipt["launches"].items():
        assert record["status"] == "complete"
        assert record["host_resource_gate"]["pass"] is True
        assert record["resolved_terminations"]["audit"]["pass"] is True
        assert (Path(record["attempt"]) / "resolved_terminations.json").is_file()
        assert sorted(record["log_termination_terms"]) == record["resolved_terminations"]["audit"]["active_terms"]
        if name.startswith("partA"):
            assert record["rollout_expectation_check"]["terminated_keys"] == []
    assert receipt["stages_complete"] == {"smoke_a": True, "smoke_b": True, "part_a": True,
                                          "part_b": True, "analysis": True}
    part_a = x._read_jsonl(out / "part_a_rows.jsonl")
    part_b = x._read_jsonl(out / "part_b_rows.jsonl")
    assert len(part_a) == 64 and len(part_b) == 192
    assert {row["rollout_source"] for row in part_b if row["physics_seed"] == 0} == {"exp022a_archive"}
    assert sum(row["tracker_terminated"] for row in part_b if row["physics_seed"] == 0) == 53
    assert all(row["valid_frames"] == FULL_LEN and not row["tracker_terminated"] for row in part_a)
    assert part_a[0]["achieved_contract"]["fps"] == 50.0 and part_a[0]["reference_contract"]["fps"] == 25.0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["part_b"]["agreement"]["seeds"] == [0, 1, 2]
    retention = summary["part_b"]["retention_over_all_seeds"]["0.05"]
    # The cheap scorer calls every archived reference an exact clear, so the restated
    # denominator is 3 seeds x 64 clips here; the real scorer gives 3 x 12.
    assert retention["n_reference_exact_clear"] == 64 and retention["n_rollouts"] == 192
    assert retention["per_seed"]["0"]["retained"] == 11  # EXP-022A survivors under the cheap scorer
    assert "not_tracked_wording" in summary["decision_rules"]
    assert summary["seed0_restatement_matches_exp022a"] is None
    assert (out / "part_a_evaluator_terms.npz").is_file()

    again = x.run_campaign(stage="all", resume=True, **_campaign_kwargs(out, calls))
    assert again["status"] == "complete" and len(calls) == 8, "a complete campaign must never relaunch"
    with pytest.raises(x.BridgeAbort, match="non-empty"):
        x.run_campaign(stage="all", **_campaign_kwargs(out, calls))


def test_resume_skips_completed_launches_and_reruns_interrupted_ones(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    x.run_campaign(stage="smoke", **_campaign_kwargs(out, calls))
    x.run_campaign(stage="part_a", resume=True, **_campaign_kwargs(out, calls))
    assert len(calls) == 4
    # An interrupted attempt (pre-launch receipt, no artifacts) must be superseded, not adopted.
    spec = x.launches_for_part("part_b")[0]
    interrupted = out / "launches" / spec["name"] / "attempt-000"
    interrupted.mkdir(parents=True)
    pkl = out / "launches" / spec["name"] / "motions.pkl"
    clips = exp022.load_source_bundle()["clips"]
    _fake_export({key: clips[key] for key in spec["motion_keys"]}, pkl)
    cal._write_json(interrupted / "receipt.json", {
        "status": "running", **x._attempt_expectations(spec, x._sha256(pkl))})
    x.run_campaign(stage="part_b", resume=True, **_campaign_kwargs(out, calls))
    assert len(calls) == 8
    assert calls[4]["eval_dir"].endswith(f"{spec['name']}/attempt-001/eval")
    receipt = x.run_campaign(stage="analyze", resume=True, **_campaign_kwargs(out, calls))
    assert receipt["status"] == "complete" and len(calls) == 8
    # A completed launch whose artifacts changed must block instead of being adopted silently.
    record = receipt["launches"]["partB_chunk01_seed2"]
    altered = json.loads((Path(record["attempt"]) / "receipt.json").read_text())
    altered["manual_change"] = True
    cal._write_json(Path(record["attempt"]) / "receipt.json", altered)
    with pytest.raises(x.BridgeAbort, match="artifacts changed"):
        x.run_campaign(stage="analyze", resume=True, **_campaign_kwargs(out, calls))


def test_host_gate_failure_paths(tmp_path):
    calls: list = []
    out = tmp_path / "campaign"
    with pytest.raises(x.BridgeAbort, match="host-resource gate failed before"):
        x.run_campaign(stage="smoke", **_campaign_kwargs(out, calls, host_gate_fn=_gate_fail))
    assert not out.exists() and calls == []

    x.run_campaign(stage="smoke", **_campaign_kwargs(out, calls))
    assert len(calls) == 2
    with pytest.raises(x.BridgeAbort, match="blocked_host_gate"):
        x.run_campaign(stage="part_a", resume=True,
                       **_campaign_kwargs(out, calls, host_gate_fn=_gate_fail))
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "running" and receipt["blocked"] is False
    assert receipt["host_gate_blocks"][0]["note"] == "blocked_host_gate"
    assert receipt["host_gate_blocks"][0]["launch"] == "partA_chunk00_seed0_termfree"
    assert not (out / "launches" / "partA_chunk00_seed0_termfree" / "attempt-000").exists()
    assert len(calls) == 2
    x.run_campaign(stage="part_a", resume=True, **_campaign_kwargs(out, calls))
    assert len(calls) == 4
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["stages_complete"]["part_a"] is True


def test_locked_exp022a_identity_and_smoke_keys_validate():
    identity = x.exp022a_identity()
    assert identity["receipt_sha256"] == x.EXP022A_RECEIPT_SHA256
    rollouts = x.load_exp022a_rollouts(identity)
    assert len(rollouts) == 64
    assert x.full_reference_valid_length(rollouts) == FULL_LEN
    assert x.validate_smoke_keys(rollouts) == list(x.SMOKE_MOTION_KEYS)
    assert sum(r.terminated for r in rollouts.values()) == 53
    thresholds = x.threshold_identity()
    assert thresholds["max_unsupported_run_s"] == 0.2
    protocol = x.protocol_identity()
    assert protocol["status"] in {"draft", "preregistered"} and len(protocol["sha256"]) == 64
