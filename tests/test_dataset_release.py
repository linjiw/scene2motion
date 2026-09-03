"""Tests for the Scene2Motion-DB release-preview builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scene2motion.dataset_release import (
    DatasetValidationError,
    SCHEMA_VERSION,
    build_preview,
    validate_pilot,
    validate_preview,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/corpus_pilot_v2"


def test_the_committed_pilot_is_complete_and_semantically_consistent():
    receipt, rows, clips = validate_pilot(PILOT)
    assert receipt["n_scenes"] == len(rows) == 300
    assert receipt["counts"] == {
        "accepted": 192,
        "accepted_margin": 76,
        "rejected": 6,
        "refused": 26,
    }
    assert len(clips) == 268


def test_preview_preserves_tiers_and_gives_every_rare_outcome_each_split(tmp_path):
    output = tmp_path / "preview"
    receipt = build_preview(PILOT, output)
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]

    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["release_status"] == "internal_preview_redistribution_review_required"
    assert receipt["controller_execution_counts"] == {"not_measured": 300}
    assert receipt["n_motion_payloads"] == 268
    assert receipt["motion_payloads_included"] is False
    assert len(rows) == 300
    assert len({row["identity"]["scene_id"] for row in rows}) == 300
    for split in ("train", "validation", "test"):
        assert set(receipt["split_outcome_counts"][split]) == {
            "accepted", "accepted_margin", "rejected", "refused"
        }

    refused = next(row for row in rows if row["label"]["outcome"] == "refused")
    assert refused["label"]["evidence_tier"] == "scene_feasibility"
    assert refused["label"]["controller_execution"] == "not_measured"
    assert refused["label"]["signed_overhead_target_residual_m"] < 0
    assert refused["artifact"] is None

    accepted = next(row for row in rows if row["label"]["outcome"] == "accepted")
    assert accepted["label"]["evidence_tier"] == "reference_geometry"
    assert accepted["label"]["signed_overhead_target_residual_m"] >= 0
    assert accepted["artifact"]["included"] is False
    assert len(accepted["artifact"]["sha256"]) == 64
    assert not (output / "clips").exists()
    assert validate_preview(output) == receipt


def test_clip_copy_is_opt_in_and_hash_preserving(tmp_path):
    output = tmp_path / "preview"
    build_preview(PILOT, output, include_clips=True)
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    record = next(row for row in rows if row["artifact"] is not None)
    artifact = record["artifact"]
    copied = output / artifact["path"]
    assert artifact["included"] is True
    assert copied.is_file()
    import hashlib
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == artifact["sha256"]


def test_builder_refuses_to_overwrite_any_existing_output(tmp_path):
    output = tmp_path / "preview"
    output.mkdir()
    with pytest.raises(DatasetValidationError, match="output already exists"):
        build_preview(PILOT, output)


def test_validator_rejects_a_receipt_count_mismatch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    receipt = json.loads((PILOT / "receipt.json").read_text())
    receipt["n_scenes"] = 299
    (source / "receipt.json").write_text(json.dumps(receipt))
    (source / "manifest.jsonl").write_text((PILOT / "manifest.jsonl").read_text())
    with pytest.raises(DatasetValidationError, match="n_scenes"):
        validate_pilot(source)


def test_validator_rejects_nonfinite_payload_data(tmp_path):
    # Exercise the payload guard on a complete copied pilot while changing one array only.
    source = tmp_path / "source"
    source.mkdir()
    (source / "clips").mkdir()
    (source / "receipt.json").write_bytes((PILOT / "receipt.json").read_bytes())
    (source / "manifest.jsonl").write_bytes((PILOT / "manifest.jsonl").read_bytes())
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    for row in rows:
        if "clip" in row:
            target = source / row["clip"]
            target.write_bytes((PILOT / row["clip"]).read_bytes())
    first = next(row for row in rows if "clip" in row)
    path = source / first["clip"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["qpos"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(DatasetValidationError, match="qpos is not finite"):
        validate_pilot(source)


def test_standalone_preview_validator_catches_a_changed_record(tmp_path):
    output = tmp_path / "preview"
    build_preview(PILOT, output)
    records = output / "records.jsonl"
    records.write_text(records.read_text().replace(
        '"controller_execution":"not_measured"',
        '"controller_execution":"measured"',
        1,
    ))
    with pytest.raises(DatasetValidationError, match="payload hash mismatch: records.jsonl"):
        validate_preview(output)
