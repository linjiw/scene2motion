"""The pure parts of the reconstructed exp1b driver.

SONIC itself cannot run here; what can be pinned is everything whose silent drift would
poison a resumed campaign: selection determinism against the ledger, launch chunking, the
reboot-damage repairs, the teleport-frame handling for both archive schemas, and the
conformal gate's scene-level split.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import exp1b_execution_clearance as exp1b
from scene2motion.demo.cache import ClipCache
from scene2motion.demo.scene_builder import BeamParams, build
from scene2motion.robot import G1Body
from scene2motion.sonic_state_export import (
    QPOS_WIDTH,
    SonicRollout,
    load_sonic_state_rollouts,
    sonic_state_archive_schema,
    write_sonic_state_archive,
)
from tests.test_sonic_state_export import write_v1_archive


def _ledger_row(method: str, seed: int, outcome: str = "accepted",
                clip_key: str | None = None) -> dict:
    return {
        "scene_id": "demo_partial_beam_h0.950_w2.250_n3_g1.50",
        "method": method, "seed": seed, "outcome": outcome,
        "clip_key": clip_key or f"clip_{method}_{seed}",
        "beam_h": 0.95, "beam_w": 2.25, "n_beams": 3, "gap": 1.5,
        "min_overhead_m": 0.25, "min_clearance_m": 0.25, "goal_error_m": 0.05,
        "peak_dip_m": 0.5, "peak_dip_initial_m": 0.5, "n_repairs": 0,
        "initial_schedule_hash": "abcd", "final_schedule_hash": "abcd",
    }


def _put_clip(cache: ClipCache, row: dict, n_frames: int = 12) -> None:
    cache.put(row["clip_key"], np.zeros((n_frames, 36), np.float32),
              {"scene_id": row["scene_id"], "preference": "shortest", "seed": row["seed"],
               "schedule_hash": row["final_schedule_hash"], "n_frames": n_frames,
               "fps": 25.0, "noise_stream_version": 2})


def _ledger(rows: list[dict]) -> dict:
    return {"preference": "shortest", "target_m": 0.18, "rows": rows}


def test_selection_is_deterministic_grouped_by_method_and_excludes_rejected(tmp_path):
    cache = ClipCache(tmp_path)
    rows = [_ledger_row("heuristic+1", 101), _ledger_row("heuristic", 100),
            _ledger_row("heuristic", 101, outcome="accepted_margin"),
            _ledger_row("heuristic", 102, outcome="rejected"),
            _ledger_row("other-method", 100)]
    for r in rows:
        _put_clip(cache, r)
    first = exp1b.derive_selection(_ledger(rows), cache)
    second = exp1b.derive_selection(_ledger(rows), cache)
    assert first == second
    selected, skipped = first
    assert skipped == []
    # Method groups in METHODS order, ledger order inside a group; rejected and unlisted
    # methods never selected.
    assert [(r["method"], r["seed"]) for r in selected] == [
        ("heuristic", 100), ("heuristic", 101), ("heuristic+1", 101)]
    assert selected[0]["motion_key"] == "demo_partial_beam_h0.950_w2.250_n3_g1.50__heuristic__s100"
    assert selected[0]["route_len_m_rebuilt"] == 10.5  # goal_x = 4.0 + 2*1.5 + 3.5
    assert selected[1]["outcome"] == "accepted_margin"


def test_selection_skips_unverifiable_clips(tmp_path):
    cache = ClipCache(tmp_path)
    ok, missing, stale = (_ledger_row("heuristic", 100), _ledger_row("heuristic", 101),
                          _ledger_row("heuristic", 102))
    _put_clip(cache, ok)
    cache.put(stale["clip_key"], np.zeros((12, 36), np.float32),
              {"scene_id": stale["scene_id"], "preference": "shortest",
               "schedule_hash": stale["final_schedule_hash"], "n_frames": 12,
               "fps": 25.0, "noise_stream_version": 1})  # v1 noise: not a verified artifact
    selected, skipped = exp1b.derive_selection(_ledger([ok, missing, stale]), cache)
    assert [r["seed"] for r in selected] == [100]
    reasons = {s["motion_key"].rsplit("s", 1)[1]: s["reason"] for s in skipped}
    assert "missing" in reasons["101"]
    assert "noise_stream_version" in reasons["102"]


def test_plan_launches_chunks_like_the_receipt():
    rows = [{"method": "heuristic", "motion_key": f"k{i}"} for i in range(83)]
    plan = exp1b.plan_launches(rows, chunk_size=36)
    assert [(s.name, len(s.rows)) for s in plan] == [
        ("heuristic_00_seed0", 36), ("heuristic_01_seed0", 36), ("heuristic_02_seed0", 11)]
    assert plan[0].motion_keys == [f"k{i}" for i in range(36)]
    assert plan[2].motion_keys[-1] == "k82"


def test_repair_launches_jsonl_truncates_nul_tail(tmp_path):
    path = tmp_path / "launches.jsonl"
    good = b'{"launch": "a", "rc": 0}\n{"launch": "b", "rc": 0}\n'
    path.write_bytes(good + b"\x00" * 169)
    summary = exp1b.repair_launches_jsonl(path)
    assert (summary["kept"], summary["dropped_bytes"]) == (2, 169)
    assert path.read_bytes() == good
    again = exp1b.repair_launches_jsonl(path)  # idempotent
    assert (again["kept"], again["dropped_bytes"]) == (2, 0)


def test_repair_run_state_moves_zeroed_artifacts_aside(tmp_path):
    spec = exp1b.LaunchSpec("heuristic", 0, 0, [])
    d = tmp_path / "launches" / spec.name
    (d / "eval").mkdir(parents=True)
    (d / f"{spec.name}.pkl").write_bytes(b"")
    (d / "sonic.log").write_bytes(b"")
    (d / "eval" / "achieved_qpos.npz").write_bytes(b"")
    (tmp_path / "rows.jsonl").write_text('{"suspect": true}\n')

    actions = exp1b.repair_run_state(tmp_path, [spec])
    assert not (d / f"{spec.name}.pkl").exists()
    assert not (d / "sonic.log").exists()
    assert (d / "zeroed" / "sonic.log").exists()
    assert (d / "zeroed" / "achieved_qpos.npz").exists()
    assert not (tmp_path / "rows.jsonl").exists()
    assert (tmp_path / "rows.v1-suspect.jsonl").read_text() == '{"suspect": true}\n'
    assert len(actions) == 4
    assert exp1b.repair_run_state(tmp_path, [spec]) == []  # idempotent


def _walk_qpos(xs: np.ndarray) -> np.ndarray:
    qpos = np.zeros((len(xs), QPOS_WIDTH), np.float32)
    qpos[:, 0] = xs
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    return qpos


@pytest.fixture(scope="module")
def tall_beam_scene():
    # Underside at 1.6 m: an upright zero-pose G1 (head ~1.3 m) passes under with real
    # positive clearance, so the walk below is a genuine traversal, not a collision case.
    scene = build(BeamParams(beam_height=1.6, beam_width=2.25, n_beams=1, gap=3.0))
    return scene, G1Body(scene)


def _join_row(scene) -> dict:
    return {"motion_key": "m", "scene_id": scene.scene_id, "method": "heuristic",
            "seed": 100, "clip_key": "c", "outcome": "accepted",
            "ref_min_overhead_m": 0.30, "peak_dip_m": 0.0, "route_len_m_rebuilt": 8.0}


def test_join_drops_v1_teleport_frame_and_records_schema(tmp_path, tall_beam_scene):
    scene, body = tall_beam_scene
    walk = _walk_qpos(np.linspace(0.0, 7.9, 40))
    with_teleport = np.concatenate([walk, _walk_qpos(np.asarray([0.0]))])  # the reset pose
    write_v1_archive(tmp_path / "achieved_qpos.npz",
                     [SonicRollout("m", with_teleport, 41, False, 1.0, 0)])
    schema = sonic_state_archive_schema(tmp_path / "achieved_qpos.npz")
    (rollout,) = load_sonic_state_rollouts(tmp_path / "achieved_qpos.npz")
    row = exp1b.join_rollout(_join_row(scene), rollout, scene, body, launch="L",
                             physics_seed=0, schema_version=schema, sample_dt_s=0.02)
    assert row["archive_schema_version"] == 1
    assert row["teleport_frame_handling"] == "dropped_at_load"
    assert row["valid_frames"] == 40
    # Goal error from the real final pose (x=7.9, goal at 8.0) — not the teleport at x=0,
    # which would read ~8 m.  This is the defect that inflated the v1-suspect rows.
    assert row["exec_goal_error_m"] < 0.2
    assert row["passed_last_obstacle"] and row["reached_first_obstacle"]
    assert not row["execution_failure"]
    assert not row["exec_collision_any"]
    assert row["executed_success"] and row["loss_valid"]
    assert row["loss_m"] == pytest.approx(0.30 - row["exec_min_overhead_m"], abs=1e-6)
    assert 0.0 < row["exec_min_overhead_m"] < 0.6  # the beam was actually interacted with


def test_join_handles_v2_archives_without_double_trimming(tmp_path, tall_beam_scene):
    scene, body = tall_beam_scene
    walk = _walk_qpos(np.linspace(0.0, 7.9, 40))  # post-fix: teleport already excluded
    write_sonic_state_archive([SonicRollout("m", walk, 40, False, 1.0, 0)],
                              tmp_path / "achieved_qpos.npz", sample_dt_s=0.02)
    schema = sonic_state_archive_schema(tmp_path / "achieved_qpos.npz")
    (rollout,) = load_sonic_state_rollouts(tmp_path / "achieved_qpos.npz")
    row = exp1b.join_rollout(_join_row(scene), rollout, scene, body, launch="L",
                             physics_seed=0, schema_version=schema, sample_dt_s=0.02)
    assert row["archive_schema_version"] == 2
    assert row["teleport_frame_handling"] == "dropped_at_write"
    assert row["valid_frames"] == 40
    assert row["exec_goal_error_m"] < 0.2


def test_join_counts_a_stalled_survivor_as_no_success(tmp_path, tall_beam_scene):
    # The diagnosis case: progress 1.0 (SONIC survival) but the robot stalled at x=2 of an
    # 8 m route.  Collision-free is not success, and its loss is not a gate observation.
    scene, body = tall_beam_scene
    stall = _walk_qpos(np.concatenate([np.linspace(0.0, 2.0, 20), np.full(20, 2.0)]))
    write_sonic_state_archive([SonicRollout("m", stall, 40, False, 1.0, 0)],
                              tmp_path / "achieved_qpos.npz", sample_dt_s=0.02)
    (rollout,) = load_sonic_state_rollouts(tmp_path / "achieved_qpos.npz")
    row = exp1b.join_rollout(_join_row(scene), rollout, scene, body, launch="L",
                             physics_seed=0, schema_version=2, sample_dt_s=0.02)
    assert not row["execution_failure"]  # SONIC did not terminate it
    assert not row["passed_last_obstacle"]
    assert not row["executed_success"]
    assert not row["loss_valid"]
    assert row["exec_goal_error_m"] == pytest.approx(6.0, abs=0.05)


def _gate_row(scene_id: str, dip: float, loss: float) -> dict:
    return {"scene_id": scene_id, "peak_dip_m": dip, "loss_m": loss, "loss_valid": True,
            "ref_min_overhead_m": 0.5, "execution_failure": False,
            "exec_collision_any": False, "passed_last_obstacle": True}


def test_gate_fit_is_scene_split_and_monotone():
    rng = np.random.default_rng(0)
    rows = []
    for scene_id in [f"scene_{i}" for i in range(6)]:
        for dip in (0.2, 0.35, 0.5):
            for _ in range(4):
                rows.append(_gate_row(scene_id, dip,
                                      0.05 + 0.2 * dip + rng.normal(0.0, 0.01)))
    gate = exp1b.fit_execution_gate(rows, alpha=0.1)
    assert gate["status"] == "fitted"
    assert set(gate["calibration_scenes"]).isdisjoint(gate["holdout_scenes"])
    assert set(gate["calibration_scenes"]) | set(gate["holdout_scenes"]) == {
        f"scene_{i}" for i in range(6)}
    assert gate["beta1"] >= 0.0
    taus = [gate["tau_at_dip"][k] for k in sorted(gate["tau_at_dip"])]
    assert taus == sorted(taus)  # non-decreasing in dip
    assert taus[-1] >= 0.05 + 0.2 * 0.5  # the upper bound covers the mean loss
    hold = gate["holdout"]
    assert 0.0 <= hold["coverage_collision_free_gate"] <= 1.0
    # ref 0.5 m clears tau (~0.17) but not 0.18 + tau for the deepest dips only if tau is
    # large; with these numbers the collision-free gate passes everything and no row is
    # false-safe (all synthetic rows executed cleanly).
    assert hold["false_safe_collision_free_gate"] == 0.0


def test_gate_fit_declines_to_fit_from_too_little_data():
    rows = [_gate_row("only_scene", 0.35, 0.1)] * 3
    gate = exp1b.fit_execution_gate(rows)
    assert gate["status"] == "insufficient_data"
    assert gate["n_calibration_obs"] <= 3


def test_split_is_deterministic():
    ids = {f"demo_partial_beam_h0.950_w2.250_n{n}_g{g:.2f}" for n in (3, 4, 5, 6)
           for g in (1.5, 2.5, 3.5)}
    assert exp1b.scene_split(ids) == exp1b.scene_split(ids)
    calib, holdout = exp1b.scene_split(ids)
    assert calib and holdout  # both sides populated on the real scene family
