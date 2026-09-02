import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp023_prompt_handoff as exp
from experiments import exp023_prompt_handoff_resume_analysis as resume
from tests.test_exp023_prompt_handoff import (
    campaign_kwargs,
    fake_event,
    fake_exact_box,
    fake_metrics,
    missing_delayed_event,
)


FAKE_SOURCES = {
    exp.PROTOCOL_PATH: "d" * 64,
    "experiments/exp023_prompt_handoff.py": "c" * 64,
}


def _complete_campaign(tmp_path: Path, event_fn=fake_event) -> tuple[Path, dict]:
    output = tmp_path / "complete"
    receipt = exp.run_campaign(**campaign_kwargs(output, event_fn=event_fn))
    # Production receipts record no injections; the interrupted fixture must look evidentiary.
    receipt["execution_mode"]["dependency_injections"] = []
    receipt["execution_mode"]["scientific_evidence_eligible"] = True
    return output, receipt


def _interrupt(complete: Path, receipt: dict, tmp_path: Path, analyzed: int = 5) -> Path:
    """Rewrite a completed bundle into the state a mid-analysis kill leaves behind."""
    bundle = tmp_path / "interrupted"
    shutil.copytree(complete, bundle)
    rows = [json.loads(line) for line in (bundle / "rows.jsonl").read_text().splitlines()]
    partial = rows[:analyzed]
    cal._write_jsonl(bundle / "rows.jsonl", partial)
    interrupted = json.loads(json.dumps(receipt))
    for key in ("summary", "measurement_gates"):
        interrupted.pop(key, None)
    interrupted["provenance"].pop("post_analysis_identity_revalidation", None)
    interrupted["provenance"].pop("completion_identity_revalidation", None)
    interrupted.update({
        "status": "running",
        "stage": "analysis",
        "complete": False,
        "blocked": False,
    })
    interrupted["query_accounting"]["trajectories_analyzed"] = analyzed
    interrupted["evidence_anchors"]["rows"] = {
        "path": "rows.jsonl",
        "n_rows": analyzed,
        "logical_sha256": cal._json_hash(partial),
        "file_sha256": cal._sha256(bundle / "rows.jsonl"),
    }
    cal._write_json(bundle / "receipt.json", interrupted)
    return bundle


def resume_clean_code_state(_repo):
    # A real clean worktree has an empty ``git diff --binary HEAD``; the resume script
    # requires exactly that, unlike the campaign fixture's placeholder diff hash.
    return {
        "commit": "f" * 40,
        "dirty": False,
        "status": [],
        "tracked_diff_sha256": resume.EMPTY_DIFF_SHA256,
    }


def resume_kwargs(bundle: Path, **overrides):
    kwargs = {
        "out": bundle,
        "reason": "test interruption",
        "body": object(),
        "code_state_fn": resume_clean_code_state,
        "source_hashes_fn": lambda _repo: dict(FAKE_SOURCES),
        "runtime_identity_fn": lambda: {"runtime": "fake"},
        "physical_identity_fn": lambda: {"physical": "fake"},
        "prompt_cache_verifier_fn": lambda _identity, _repo: {"file_unchanged": True},
        "event_detector_fn": fake_event,
        "exact_box_fn": fake_exact_box,
        "motion_metrics_fn": fake_metrics,
    }
    kwargs.update(overrides)
    return kwargs


def test_resume_reproduces_the_uninterrupted_rows_and_summary(tmp_path):
    complete, receipt = _complete_campaign(tmp_path)
    bundle = _interrupt(complete, receipt, tmp_path)
    resumed = resume.resume_analysis(**resume_kwargs(bundle))

    assert resumed["status"] == "complete" and resumed["complete"] is True
    assert resumed["query_accounting"]["trajectories_analyzed"] == 32
    assert resumed["actual_ardy_samples"] == 32
    assert (bundle / "rows.jsonl").read_text() == (complete / "rows.jsonl").read_text()
    assert resumed["summary"] == receipt["summary"]
    assert resumed["measurement_gates"] == receipt["measurement_gates"]
    # Interrupted evidence is preserved beside the completed bundle and anchored.
    assert (bundle / resume.INTERRUPTED_RECEIPT).exists()
    assert (bundle / resume.INTERRUPTED_ROWS).exists()
    anchors = resumed["evidence_anchors"]
    assert anchors["interrupted_receipt"]["file_sha256"] == cal._sha256(
        bundle / resume.INTERRUPTED_RECEIPT)
    assert anchors["rows"]["file_sha256"] == cal._sha256(bundle / "rows.jsonl")
    assert anchors["qpos"] == receipt["evidence_anchors"]["qpos"]
    block = resumed["analysis_resume"]
    assert block["regenerated_trajectories"] == 0 and block["new_ardy_samples"] == 0
    assert block["interrupted_receipt"]["trajectories_analyzed_before_interruption"] == 5
    assert len(block["archived_partial_rows_recomputed_identically"]) == 5
    assert all(item["recomputed_equals_archived"]
               for item in block["archived_partial_rows_recomputed_identically"])
    assert block["scientific_evidence_eligible"] is False  # injected test doubles
    on_disk = json.loads((bundle / "receipt.json").read_text())
    assert on_disk["status"] == "complete"


def test_resume_refuses_a_completed_bundle(tmp_path):
    complete, _ = _complete_campaign(tmp_path)
    with pytest.raises(resume.ResumeRefusal, match="interrupted during analysis"):
        resume.resume_analysis(**resume_kwargs(complete))
    assert not (complete / resume.INTERRUPTED_RECEIPT).exists()


def test_resume_refuses_changed_frozen_sources(tmp_path):
    complete, receipt = _complete_campaign(tmp_path)
    bundle = _interrupt(complete, receipt, tmp_path)
    changed = dict(FAKE_SOURCES)
    changed["experiments/exp023_prompt_handoff.py"] = "e" * 64
    with pytest.raises(resume.ResumeRefusal, match="frozen EXP-023 sources changed"):
        resume.resume_analysis(
            **resume_kwargs(bundle, source_hashes_fn=lambda _repo: changed))
    assert not (bundle / resume.INTERRUPTED_RECEIPT).exists()
    assert json.loads((bundle / "receipt.json").read_text())["status"] == "running"


def test_resume_refuses_a_tampered_archive(tmp_path):
    complete, receipt = _complete_campaign(tmp_path)
    bundle = _interrupt(complete, receipt, tmp_path)
    arrays = dict(np.load(bundle / "qpos.npz"))
    arrays["s4500_all_walk"] = np.array(arrays["s4500_all_walk"], copy=True)
    arrays["s4500_all_walk"][0, 0] += 1.0
    cal._persist_qpos(bundle / "qpos.npz", arrays)
    with pytest.raises(resume.ResumeRefusal, match="qpos.npz no longer matches"):
        resume.resume_analysis(**resume_kwargs(bundle))


def test_resume_refuses_a_dirty_worktree_outside_the_bundle(tmp_path):
    complete, receipt = _complete_campaign(tmp_path)
    bundle = _interrupt(complete, receipt, tmp_path)

    def dirty(_repo):
        state = resume_clean_code_state(_repo)
        return {**state, "dirty": True, "status": [" M scene2motion/robot.py"]}

    with pytest.raises(resume.ResumeRefusal, match="not clean outside"):
        resume.resume_analysis(**resume_kwargs(bundle, code_state_fn=dirty))


def test_resume_fails_closed_when_recomputed_rows_differ_from_archived(tmp_path):
    complete, receipt = _complete_campaign(tmp_path)
    bundle = _interrupt(complete, receipt, tmp_path)

    def shifted_event(body, qpos, route, onset):
        record = dict(fake_event(body, qpos, route, onset))
        if record["present"]:
            record["frame"] = int(record["frame"]) + 1
            record["latency_frames"] = int(record["latency_frames"]) + 1
            record["latency_s"] = record["latency_frames"] / exp.FPS
        return record

    with pytest.raises(resume.ResumeRefusal, match="differs from the archived partial row"):
        resume.resume_analysis(**resume_kwargs(bundle, event_detector_fn=shifted_event))
    # The interrupted copies exist (they are written before scoring), but the primary
    # rows file and receipt were not rewritten.
    on_disk = json.loads((bundle / "receipt.json").read_text())
    assert on_disk["status"] == "running" and "summary" not in on_disk


def test_resume_applies_the_preregistered_gates_unchanged(tmp_path):
    complete, receipt = _complete_campaign(tmp_path, event_fn=missing_delayed_event)
    bundle = _interrupt(complete, receipt, tmp_path)
    resumed = resume.resume_analysis(
        **resume_kwargs(bundle, event_detector_fn=missing_delayed_event))
    rates = resumed["summary"]["event_rates_missing_retained"]
    assert rates["step_0"]["present"] == 8
    assert rates["step_52"]["present"] == 0 and rates["step_52"]["planned"] == 8
    assert resumed["measurement_gates"]["step0_substrate"]["pass"] is True
    assert resumed["status"] == "complete"


def test_resume_refuses_second_resume(tmp_path):
    complete, receipt = _complete_campaign(tmp_path)
    bundle = _interrupt(complete, receipt, tmp_path)
    resume.resume_analysis(**resume_kwargs(bundle))
    with pytest.raises(resume.ResumeRefusal, match="resumed before"):
        resume.resume_analysis(**resume_kwargs(bundle))
