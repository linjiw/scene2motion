"""Focused CPU tests for the exact fixed-obstacle EXP-021 correction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments import analyze_exp021_exact_addressability as exact


def _fake_archive(tmp_path, clearances):
    archive = tmp_path / "exp021"
    archive.mkdir()
    seeds = list(range(4400, 4400 + len(clearances)))
    arrays = {
        f"s{seed}": np.asarray([[clearance, 0.0], [0.0, 0.0]], dtype=np.float32)
        for seed, clearance in zip(seeds, clearances)
    }
    np.savez(archive / "qpos.npz", **arrays)
    receipt = {
        "schema": "exp021-elicited-lift-distribution-v1",
        "experiment": "exp021_elicited_lift_distribution",
        "status": "complete",
        "complete": True,
        "blocked": False,
        "actual_ardy_samples": len(arrays),
        "design": {"pool_seeds": seeds},
        "summary": {"n_clips": len(arrays)},
        "evidence_anchors": {
            "qpos": {
                "path": "qpos.npz",
                "n_arrays": len(arrays),
                "content_sha256": exact.array_content_sha256(arrays),
            }
        },
    }
    (archive / "receipt.json").write_text(json.dumps(receipt))
    return archive, receipt


class _FirstValueProbe:
    def probe(self, qpos):
        return float(qpos[0, 0])

    def clears(self, qpos, height):
        return bool(float(qpos[0, 0]) >= float(height))


def test_independent_curve_and_required_call_counts():
    assert exact.independent_best_of_n(0.5, 1) == pytest.approx(0.5)
    assert exact.independent_best_of_n(0.5, 2) == pytest.approx(0.75)
    assert exact.calls_for_target(0.5, 0.90) == 4
    assert exact.calls_for_target(0.0, 0.90) is None
    assert exact.calls_for_target(1.0, 0.95) == 1


def test_wilson_interval_contains_observed_rate_and_validates_counts():
    low, high = exact.wilson_interval(12, 64)
    assert low < 12 / 64 < high
    with pytest.raises(ValueError, match="positive total"):
        exact.wilson_interval(0, 0)
    with pytest.raises(ValueError, match=r"\[0, total\]"):
        exact.wilson_interval(2, 1)


def test_summary_keeps_disjoint_sequential_n8_blocks_in_seed_order():
    clearances = [0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 8
    seeds = list(range(16))
    row = exact.summarize_clearances(clearances, seeds, heights_m=(0.05,))["h=0.05"]
    assert row["successes"] == 1
    assert row["per_clip_rate"] == pytest.approx(1 / 16)
    assert row["independent_plugin_best_of_n"]["N=8"] == pytest.approx(
        1 - (15 / 16) ** 8)
    sequential = row["sequential_n8"]
    assert sequential["successful_blocks"] == 1
    assert sequential["empirical_success_rate"] == pytest.approx(0.5)
    assert sequential["blocks"][0]["seeds"] == list(range(8))
    assert sequential["blocks"][0]["first_success_call"] == 2
    assert sequential["blocks"][1]["success"] is False
    assert sequential["blocks"][1]["calls_spent_stop_on_success"] == 8


def test_exact_binary_hits_are_not_replaced_by_probe_lower_bound_thresholds():
    # This is the boundary that changed the archived 8 cm count from 10 to 11: probe()
    # reports a conservative grid lower bound, while clears(qpos, h) asks the exact binary
    # collision question at h.
    row = exact.summarize_clearances(
        [0.075] * 8,
        list(range(8)),
        heights_m=(0.08,),
        exact_hits_by_height={0.08: [True] * 8},
    )["h=0.08"]
    assert row["successes"] == 8
    assert row["binary_outcome_source"] == "BoxHeightProbe.clears_at_exact_height"


def test_analyze_archive_validates_and_uses_one_exact_probe_location(tmp_path):
    clearances = [0.10, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0]
    archive, _ = _fake_archive(tmp_path, clearances)
    calls = []

    def factory(x_m, depth_m):
        calls.append((x_m, depth_m))
        return _FirstValueProbe()

    result = exact.analyze_archive(
        archive, probe_factory=factory, locked_manifest=None)
    assert calls == [(exact.LOCKED_OBSTACLE_X_M, exact.LOCKED_OBSTACLE_DEPTH_M)]
    assert result["source"]["qpos_arrays"] == 8
    assert result["summary"]["h=0.05"]["successes"] == 2
    assert result["summary"]["h=0.05"]["binary_outcome_source"] == (
        "BoxHeightProbe.clears_at_exact_height")
    assert "POST HOC" in result["evidentiary_scope"]["target_status"]
    assert "no spatial tolerance" in result["evidentiary_scope"]["endpoint"]


def test_analyze_archive_refuses_key_or_content_drift(tmp_path):
    archive, receipt = _fake_archive(tmp_path, [0.0] * 8)
    with np.load(archive / "qpos.npz", allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    arrays["unexpected"] = arrays.pop("s4407")
    np.savez(archive / "qpos.npz", **arrays)
    receipt["evidence_anchors"]["qpos"]["content_sha256"] = exact.array_content_sha256(arrays)
    (archive / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="qpos keys do not match"):
        exact.analyze_archive(
            archive, probe_factory=lambda _x, _d: _FirstValueProbe(),
            locked_manifest=None)


def test_analyze_archive_refuses_content_hash_drift(tmp_path):
    archive, _ = _fake_archive(tmp_path, [0.0] * 8)
    with np.load(archive / "qpos.npz", allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    arrays["s4400"][0, 0] = 0.10
    np.savez(archive / "qpos.npz", **arrays)
    with pytest.raises(ValueError, match="qpos content hash mismatch"):
        exact.analyze_archive(
            archive, probe_factory=lambda _x, _d: _FirstValueProbe(),
            locked_manifest=None)


def test_default_locked_manifest_refuses_a_lookalike_archive(tmp_path):
    archive, _ = _fake_archive(tmp_path, [0.0] * 8)
    with pytest.raises(ValueError, match="locked EXP-021 artifact hash mismatch"):
        exact.analyze_archive(
            archive, probe_factory=lambda _x, _d: _FirstValueProbe())


def test_main_is_print_only_unless_write_or_out_is_explicit(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "archive"
    archive.mkdir()
    payload = {"schema": exact.SCHEMA_VERSION, "status": "test"}
    monkeypatch.setattr(exact, "analyze_archive", lambda _archive: payload)

    exact.main(["--archive", str(archive)])
    assert json.loads(capsys.readouterr().out) == payload
    assert not (archive / exact.DEFAULT_OUTPUT_NAME).exists()

    destination = tmp_path / "explicit.json"
    exact.main(["--archive", str(archive), "--out", str(destination)])
    assert json.loads(destination.read_text()) == payload
