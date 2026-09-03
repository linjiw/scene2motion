"""CPU tests for the scene-level traversal-outcome analyser."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import analyze_traversal_outcomes as ato
from scene2motion import traversal_eval as te


def _archive(tmp_path, clips):
    """clips: list of (motion_key, qpos, valid_length, terminated) -> a launch archive."""
    path = tmp_path / "launches/chunk00_seed0/attempt-000/eval"
    path.mkdir(parents=True)
    width = max(len(q) for _, q, _, _ in clips)
    padded = np.zeros((len(clips), width, 36), dtype=np.float32)
    for i, (_, q, _, _) in enumerate(clips):
        padded[i, : len(q)] = q
    np.savez(
        path / "achieved_qpos.npz",
        schema_version=np.int16(2),
        qpos=padded,
        valid_lengths=np.asarray([n for _, _, n, _ in clips], dtype=np.int32),
        terminated=np.asarray([t for _, _, _, t in clips], dtype=bool),
        motion_keys=np.asarray([k for k, _, _, _ in clips]),
    )
    (tmp_path / "achieved_rows.jsonl").write_text(
        "".join(json.dumps({"motion_key": k}) + "\n" for k, _, _, _ in clips))
    return tmp_path


def _walk(x_end, n=60, z=0.75):
    q = np.zeros((n, 36), dtype=np.float32)
    q[:, 0] = np.linspace(0.0, x_end, n)
    q[:, 2] = z
    q[:, 3] = 1.0
    return q


def test_scene_carries_the_start_the_goal_and_the_obstacle():
    scene = ato.scene_for(0.05)
    assert scene.start == (0.0, 0.0)
    assert scene.goal == (ato.ROUTE_LENGTH_M, 0.0)
    obstacles = te.obstacle_boxes(scene)
    assert len(obstacles) == 1
    assert obstacles[0].center[0] == pytest.approx(ato.OBSTACLE_X_M)
    assert obstacles[0].center[2] == pytest.approx(0.025)      # half the box height


def test_taller_obstacles_build_taller_boxes():
    low, high = ato.scene_for(0.05), ato.scene_for(0.30)
    assert te.obstacle_boxes(high)[0].half[2] > te.obstacle_boxes(low)[0].half[2]


def test_load_achieved_truncates_to_the_valid_length_and_reads_every_chunk(tmp_path):
    bridge = _archive(tmp_path, [("s1", _walk(7.2), 40, False), ("s2", _walk(2.0), 12, True)])
    clips = ato.load_achieved(bridge)
    assert set(clips) == {"s1", "s2"}
    assert clips["s1"]["qpos"].shape == (40, 36)
    assert clips["s2"]["qpos"].shape == (12, 36)
    assert clips["s2"]["terminated"] is True


def test_evaluate_pool_returns_one_record_per_clip_with_its_key(tmp_path):
    bridge = _archive(tmp_path, [("s1", _walk(7.2), 60, False), ("s2", _walk(1.0), 60, True)])
    records = ato.evaluate_pool(ato.load_achieved(bridge), 0.05)
    assert [r["motion_key"] for r in records] == ["s1", "s2"]
    assert all(r["outcome"] in te.OUTCOMES for r in records)


def test_a_cut_off_rollout_that_never_arrives_is_a_cutoff_not_a_collision(tmp_path):
    bridge = _archive(tmp_path, [("s1", _walk(0.5), 60, True)])
    record = ato.evaluate_pool(ato.load_achieved(bridge), 0.05)[0]
    assert record["outcome"] == "cutoff"
    assert record["passed_obstacle"] is False
    assert record["collided_obstacle"] is False


def test_the_summary_is_written_with_scope_scene_and_source_hashes(tmp_path, monkeypatch):
    bridge = _archive(tmp_path, [("s1", _walk(7.2), 60, False)])
    out = tmp_path / "summary.json"
    monkeypatch.setattr(ato, "BRIDGE", bridge)
    monkeypatch.setattr(ato, "ACHIEVED_ROWS", bridge / "achieved_rows.jsonl")
    monkeypatch.setattr(ato, "OUT", out)
    monkeypatch.setattr(ato, "ROOT", tmp_path)

    ato.main()

    summary = json.loads(out.read_text())
    assert summary["descriptive_only"] is True
    assert "obstacle ABSENT" in summary["scope"]
    assert summary["scene"]["goal_m"] == [ato.ROUTE_LENGTH_M, 0.0]
    assert len(summary["sources"]["achieved_rows"]["sha256"]) == 64
    assert set(summary["by_height"]) == {f"{h:g}" for h in ato.HEIGHTS_M}
    for entry in summary["by_height"].values():
        assert entry["n_assigned_trials"] == 1
        assert sum(entry["outcomes"].values()) == 1


def test_every_clip_is_counted_at_every_height(tmp_path, monkeypatch):
    """No trial may be dropped: the denominator is all assigned trials."""
    clips = [(f"s{i}", _walk(7.2 if i % 2 else 0.5), 60, bool(i % 3)) for i in range(6)]
    bridge = _archive(tmp_path, clips)
    out = tmp_path / "summary.json"
    monkeypatch.setattr(ato, "BRIDGE", bridge)
    monkeypatch.setattr(ato, "ACHIEVED_ROWS", bridge / "achieved_rows.jsonl")
    monkeypatch.setattr(ato, "OUT", out)
    monkeypatch.setattr(ato, "ROOT", tmp_path)

    ato.main()

    summary = json.loads(out.read_text())
    for entry in summary["by_height"].values():
        assert entry["n_assigned_trials"] == 6
        assert sum(entry["outcomes"].values()) == 6
        assert len(entry["per_clip"]) == 6


def test_a_missing_archive_is_an_error_not_an_empty_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(ato, "BRIDGE", tmp_path / "nothing")
    with pytest.raises(SystemExit):
        ato.main()
