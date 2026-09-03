"""Fail-closed and accounting tests for the EXP-031 CPU preparation stage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import exp031_prepare_step_repair as exp


def preregistered():
    return {"path": "protocol", "sha256": "frozen", "status": "preregistered"}


def clean():
    return {"commit": "abc", "dirty": False, "status": []}


def test_dry_run_is_read_only_and_reports_the_locked_candidate_set(tmp_path):
    report = exp.dry_run_report(git_state_fn=lambda: {**clean(), "dirty": True})
    assert report["writes_performed"] is False
    assert report["project_dirty_observed"] is True
    assert report["summary"]["n_assigned_trials"] == 64
    assert report["summary"]["n_input_support_pass"] == 8
    assert report["summary"]["n_accepted_for_execution"] == 2
    assert report["summary"]["accepted_keys"] == list(exp.EXPECTED_ACCEPTED_KEYS)
    assert report["candidate_array_hashes"] == exp.EXPECTED_CANDIDATE_ARRAY_SHA256
    history = report["historical_outcome_disclosure"]
    assert history["source"]["sha256"] == exp.HISTORICAL_EXECUTION_ROWS_SHA256
    assert history["candidates"]["s4408"]["absent"]["outcome"] == "stalled"
    assert history["candidates"]["s4434"]["absent"]["outcome"] == "completed"
    assert all(
        history["candidates"][key]["present_05"]["outcome"] == "collided_obstacle"
        for key in exp.EXPECTED_ACCEPTED_KEYS
    )
    assert not any(tmp_path.iterdir())


def test_prepare_refuses_a_draft_protocol_before_creating_output(tmp_path):
    out = tmp_path / "campaign"
    with pytest.raises(exp.PreparationRefused, match="preregister"):
        exp.prepare(
            out=out, git_state_fn=clean,
            protocol_identity_fn=lambda: {"status": "draft"},
        )
    assert not out.exists()


def test_prepare_refuses_a_dirty_tree_before_creating_output(tmp_path):
    out = tmp_path / "campaign"
    with pytest.raises(exp.PreparationRefused, match="clean worktree"):
        exp.prepare(
            out=out, git_state_fn=lambda: {**clean(), "dirty": True},
            protocol_identity_fn=preregistered,
        )
    assert not out.exists()


def test_prepare_writes_all_dispositions_and_only_two_candidate_arrays(tmp_path):
    out = tmp_path / "campaign"
    receipt = exp.prepare(
        out=out, git_state_fn=clean, protocol_identity_fn=preregistered,
    )
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    with np.load(out / "qpos.npz") as archive:
        assert tuple(archive.files) == exp.EXPECTED_ACCEPTED_KEYS
        assert all(archive[key].dtype == np.float32 for key in archive.files)
        assert {key: exp.qpos_sha256(archive[key]) for key in archive.files} \
            == exp.EXPECTED_CANDIDATE_ARRAY_SHA256
    assert len(rows) == 64
    assert sum(row["status"] == "refused" for row in rows) == 56
    assert sum(row["status"] == "rejected" for row in rows) == 6
    assert sum(row["status"] == "accepted" for row in rows) == 2
    assert receipt["status"] == "prepared"
    assert receipt["sonic_rollouts_requested"] == 0
    assert receipt["summary"]["accepted_keys"] == list(exp.EXPECTED_ACCEPTED_KEYS)
    assert receipt["historical_outcome_disclosure"] == exp.historical_outcome_disclosure()
    assert receipt["artifacts"]["rows"]["sha256"] == exp.sha256_file(out / "rows.jsonl")
    assert receipt["artifacts"]["qpos"]["sha256"] == exp.sha256_file(out / "qpos.npz")


def test_nonempty_output_is_never_overwritten(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "foreign.txt").write_text("keep")
    with pytest.raises(exp.PreparationRefused, match="non-empty"):
        exp.prepare(out=out, git_state_fn=clean, protocol_identity_fn=preregistered)
    assert (out / "foreign.txt").read_text() == "keep"
