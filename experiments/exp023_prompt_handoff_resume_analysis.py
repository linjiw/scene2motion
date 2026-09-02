"""EXP-023 analysis resume: re-score the archived 32 trajectories without regeneration.

The EXP-023 production run archived every generated trajectory, its normalized features and
the latent/history audits durably, but the host process was terminated during the CPU
analysis stage after 5 of 32 rows.  This script finishes exactly that deterministic analysis:

* it refuses anything but an interrupted ``exp023_prompt_handoff`` bundle whose generation,
  causal-pairing and identity audits are complete and whose evidence anchors still match the
  files byte for byte;
* it requires every frozen source file (protocol, harness, scorers) to hash identically to
  the generation-time receipt, so the scoring code is the preregistered code;
* it re-runs the archived partial rows first and fails closed if any recomputed row differs
  from the archived one (scoring determinism check), then scores the remaining rows;
* it applies the preregistered substrate/specificity gates unchanged and preserves the
  interrupted receipt and partial rows beside the completed evidence.

It never constructs the generator, never touches a seed, and never rewrites the archives.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp023_prompt_handoff as exp  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402


RESUME_SCHEMA_VERSION = "exp023-analysis-resume-v1"
INTERRUPTED_RECEIPT = "receipt.interrupted-analysis.json"
INTERRUPTED_ROWS = "rows.interrupted-analysis.jsonl"
EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class ResumeRefusal(RuntimeError):
    """The archived bundle is not an interrupted-analysis EXP-023 run, or it changed."""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _canonical(value: Any) -> str:
    return cal._canonical_json(json.loads(json.dumps(value, allow_nan=False)))


def _verify_interrupted_receipt(receipt: Mapping[str, Any]) -> None:
    plan = exp.locked_row_plan()
    chunks = exp.locked_chunk_plan(plan)
    if receipt.get("experiment") != "exp023_prompt_handoff":
        raise ResumeRefusal("bundle is not an EXP-023 receipt")
    if receipt.get("schema") != exp.SCHEMA_VERSION:
        raise ResumeRefusal("bundle receipt schema is not the EXP-023 campaign schema")
    if receipt.get("status") != "running" or receipt.get("stage") != "analysis":
        raise ResumeRefusal(
            "resume applies only to a run interrupted during analysis "
            f"(status={receipt.get('status')!r}, stage={receipt.get('stage')!r})"
        )
    if receipt.get("complete") or receipt.get("blocked") or "summary" in receipt:
        raise ResumeRefusal("bundle already carries a completion or refusal")
    if receipt.get("execution_mode", {}).get("dependency_injections"):
        raise ResumeRefusal("bundle was produced with injected components; not evidentiary")
    design = receipt.get("campaign_design", {})
    if design.get("row_plan_sha256") != cal._json_hash(plan):
        raise ResumeRefusal("archived row plan differs from the locked EXP-023 plan")
    generation = receipt.get("generation_chunks", {})
    if set(generation) != {str(chunk["name"]) for chunk in chunks}:
        raise ResumeRefusal("archived chunk set differs from the locked chunk plan")
    if any(item.get("status") != "complete" for item in generation.values()):
        raise ResumeRefusal("not every generation chunk completed; nothing to resume")
    counters = receipt.get("query_accounting", {})
    expected = {
        "schedule_invocations_planned": len(chunks),
        "schedule_invocations_started": len(chunks),
        "schedule_invocations_completed": len(chunks),
        "autoregressive_window_calls_planned": len(chunks) * exp.N_WINDOWS,
        "autoregressive_window_calls_completed": len(chunks) * exp.N_WINDOWS,
        "trajectories_planned": len(plan),
        "trajectories_launched": len(plan),
        "trajectories_returned": len(plan),
        "trajectories_converted_to_qpos": len(plan),
    }
    for key, value in expected.items():
        if int(counters.get(key, -1)) != value:
            raise ResumeRefusal(f"archived accounting {key}={counters.get(key)} != {value}")
    analyzed = int(counters.get("trajectories_analyzed", -1))
    if analyzed < 0 or analyzed >= len(plan):
        raise ResumeRefusal("archived analysis count is not a strict partial count")
    if int(receipt.get("actual_ardy_samples", -1)) != len(plan):
        raise ResumeRefusal("archived actual_ardy_samples is not the locked 32")
    if list(receipt.get("spent_seeds", [])) != list(exp.SEEDS):
        raise ResumeRefusal("archived spent seeds differ from the locked seed block")
    if receipt.get("unlaunched_locked_seeds"):
        raise ResumeRefusal("bundle reports unlaunched locked seeds")
    audit = receipt.get("causal_pairing_audit", {})
    for key in ("feature_prefixes_exact", "qpos_prefixes_exact",
                "corresponding_window_noise_equal", "noise_fresh_across_windows"):
        if audit.get(key) is not True:
            raise ResumeRefusal(f"archived causal audit does not assert {key}")
    if not isinstance(audit.get("qpos_forks"), list) or len(audit["qpos_forks"]) != len(exp.SEEDS):
        raise ResumeRefusal("archived causal audit lacks the per-seed qpos fork records")
    provenance = receipt.get("provenance", {})
    for key in ("code", "source_sha256", "protocol", "generator", "runtime",
                "physical_model", "walk_step_prompt_cache",
                "post_generation_identity_revalidation"):
        if key not in provenance:
            raise ResumeRefusal(f"archived provenance lacks {key}")
    if provenance["source_sha256"].get(exp.PROTOCOL_PATH) != provenance["protocol"].get("sha256"):
        raise ResumeRefusal("archived protocol hash disagrees with the source manifest")
    if provenance["code"].get("dirty") is not False:
        raise ResumeRefusal("generation-time worktree was not clean")


def _verify_anchors(receipt: Mapping[str, Any], output: Path) -> None:
    anchors = receipt.get("evidence_anchors", {})
    for name in ("rows", "qpos", "features", "noise_audit"):
        anchor = anchors.get(name)
        if not isinstance(anchor, Mapping):
            raise ResumeRefusal(f"archived evidence anchor {name} is missing")
        actual = cal._sha256(output / str(anchor["path"]))
        if actual != anchor.get("file_sha256"):
            raise ResumeRefusal(f"{anchor['path']} no longer matches its archived sha256")
    if int(anchors["qpos"]["n_arrays"]) != len(exp.SEEDS) * len(exp.ARMS):
        raise ResumeRefusal("archived qpos anchor does not count 32 arrays")
    if int(anchors["features"]["n_arrays"]) != len(exp.SEEDS) * len(exp.ARMS):
        raise ResumeRefusal("archived features anchor does not count 32 arrays")


def _analysis_git_check(
    generation_code: Mapping[str, Any],
    current_code: Mapping[str, Any],
    *,
    repo: Path,
    output: Path,
) -> dict[str, Any]:
    """Require a clean worktree apart from this bundle; the commit may legitimately move."""
    status = current_code.get("status")
    if not isinstance(status, list) or any(not isinstance(line, str) for line in status):
        raise ResumeRefusal("analysis-time git status is invalid")
    if not isinstance(current_code.get("commit"), str) or not current_code["commit"].strip():
        raise ResumeRefusal("analysis-time git commit is missing")
    try:
        relative_output = output.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        relative_output = None
    allowed: list[str] = []
    unexpected: list[str] = []
    for line in status:
        path = line[3:] if len(line) >= 4 else ""
        if relative_output is not None and (
            path == relative_output or path.startswith(relative_output + "/")
        ):
            allowed.append(line)
        else:
            unexpected.append(line)
    if unexpected:
        raise ResumeRefusal(
            "worktree is not clean outside the EXP-023 bundle: " + "; ".join(unexpected)
        )
    if current_code.get("tracked_diff_sha256") != EMPTY_DIFF_SHA256:
        raise ResumeRefusal("analysis-time worktree carries a tracked diff")
    return {
        "generation_commit": generation_code.get("commit"),
        "analysis_commit": current_code["commit"],
        "commit_unchanged": current_code["commit"] == generation_code.get("commit"),
        "tracked_diff_empty": True,
        "allowed_output_status": allowed,
        "unexpected_status": unexpected,
    }


def _default_prompt_cache_verifier(identity: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    fields = identity.get("fields") if isinstance(identity, Mapping) else None
    if not isinstance(fields, Mapping):
        raise ResumeRefusal("archived prompt-cache identity has no fields")
    path = Path(str(fields.get("path")))
    if not path.is_absolute():
        path = repo / path
    actual = cal._sha256(path)
    if actual is None or actual != fields.get("file_sha256"):
        raise ResumeRefusal("prompt cache file no longer matches its archived sha256")
    return {"path": str(path), "file_sha256": actual, "file_unchanged": True}


def resume_analysis(
    *,
    out: str | Path,
    reason: str,
    body: Any | None = None,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = exp._source_hashes,
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
    prompt_cache_verifier_fn: Callable[[Mapping[str, Any], Path], Mapping[str, Any]] = (
        _default_prompt_cache_verifier
    ),
    event_detector_fn: Callable[[Any, np.ndarray, np.ndarray, int], Mapping[str, Any]] = (
        exp.detect_prompt_event
    ),
    exact_box_fn: Callable[[Any, np.ndarray, float], Mapping[str, Any]] = exp.score_exact_boxes,
    motion_metrics_fn: Callable[[Any, np.ndarray, np.ndarray], Mapping[str, Any]] = (
        exp.supporting_motion_metrics
    ),
) -> dict[str, Any]:
    """Finish the interrupted EXP-023 analysis from its durable archives, or refuse."""
    output = Path(out)
    repo = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    injected = [
        name for name, value, default in (
            ("body", body, None),
            ("code_state_fn", code_state_fn, cal._git_state),
            ("source_hashes_fn", source_hashes_fn, exp._source_hashes),
            ("runtime_identity_fn", runtime_identity_fn, cal._runtime_identity),
            ("physical_identity_fn", physical_identity_fn, cal._physical_model_identity),
            ("prompt_cache_verifier_fn", prompt_cache_verifier_fn,
             _default_prompt_cache_verifier),
            ("event_detector_fn", event_detector_fn, exp.detect_prompt_event),
            ("exact_box_fn", exact_box_fn, exp.score_exact_boxes),
            ("motion_metrics_fn", motion_metrics_fn, exp.supporting_motion_metrics),
        )
        if value is not default
    ]

    for name in ("receipt.json", "rows.jsonl", "qpos.npz", "features.npz", "noise_audit.json"):
        if not (output / name).is_file():
            raise ResumeRefusal(f"EXP-023 bundle lacks {name}")
    # A resume attempt that was itself killed during scoring leaves the interrupted copies
    # behind but rewrites nothing (rows and receipt are written atomically at the end).  Such
    # copies are byte-identical to the live files and may be reused; any other pre-existing
    # copy means the bundle was already completed or refused and must not be resumed again.
    prior_attempt = False
    for name, live in ((INTERRUPTED_RECEIPT, "receipt.json"), (INTERRUPTED_ROWS, "rows.jsonl")):
        if (output / name).exists():
            if cal._sha256(output / name) != cal._sha256(output / live):
                raise ResumeRefusal(f"{name} already exists and differs; the bundle was resumed before")
            prior_attempt = True

    receipt: dict[str, Any] = _load_json(output / "receipt.json")
    _verify_interrupted_receipt(receipt)
    _verify_anchors(receipt, output)
    interrupted_receipt_sha256 = cal._sha256(output / "receipt.json")
    interrupted_rows_sha256 = cal._sha256(output / "rows.jsonl")

    # Frozen code and identities must be the generation-time ones.
    source_hashes = dict(source_hashes_fn(repo))
    if source_hashes != dict(receipt["provenance"]["source_sha256"]):
        changed = sorted(
            key for key in set(source_hashes) | set(receipt["provenance"]["source_sha256"])
            if source_hashes.get(key) != receipt["provenance"]["source_sha256"].get(key)
        )
        raise ResumeRefusal(f"frozen EXP-023 sources changed since generation: {changed}")
    physical_identity = dict(physical_identity_fn())
    if physical_identity != dict(receipt["provenance"]["physical_model"]):
        raise ResumeRefusal("G1 physical model identity changed since generation")
    runtime_identity = dict(runtime_identity_fn())
    if runtime_identity != dict(receipt["provenance"]["runtime"]):
        raise ResumeRefusal("ARDY/numerical runtime identity changed since generation")
    prompt_cache_check = dict(prompt_cache_verifier_fn(
        receipt["provenance"]["walk_step_prompt_cache"], repo))
    current_code = dict(code_state_fn(repo))
    git_check = _analysis_git_check(
        receipt["provenance"]["code"], current_code, repo=repo, output=output)

    # Archives.
    plan = exp.locked_row_plan()
    keys = [f"s{item['seed']}_{item['arm']}" for item in plan]
    qpos_archive = _load_npz(output / "qpos.npz")
    feature_archive = _load_npz(output / "features.npz")
    noise_evidence = _load_json(output / "noise_audit.json")
    anchors = receipt["evidence_anchors"]
    if set(qpos_archive) != set(keys) or set(feature_archive) != set(keys):
        raise ResumeRefusal("archived array keys do not match the locked row plan")
    if cal._array_hash(qpos_archive) != anchors["qpos"]["content_sha256"]:
        raise ResumeRefusal("qpos archive content hash differs from its anchor")
    if cal._array_hash(feature_archive) != anchors["features"]["content_sha256"]:
        raise ResumeRefusal("features archive content hash differs from its anchor")
    if cal._json_hash(noise_evidence) != anchors["noise_audit"]["logical_sha256"]:
        raise ResumeRefusal("noise audit logical hash differs from its anchor")
    if (
        not isinstance(noise_evidence, list)
        or len(noise_evidence) != exp.N_WINDOWS
        or any(len(window.get("rows", [])) != len(plan) for window in noise_evidence)
    ):
        raise ResumeRefusal("noise audit does not hold four windows of 32 rows")
    for key in keys:
        qpos = np.asarray(qpos_archive[key])
        if qpos.ndim != 2 or qpos.shape[0] != exp.PADDED_FRAMES or not np.isfinite(qpos).all():
            raise ResumeRefusal(f"archived qpos for {key} is malformed")
    qpos_forks = exp._validate_qpos_forks(qpos_archive)
    if _canonical(qpos_forks) != _canonical(receipt["causal_pairing_audit"]["qpos_forks"]):
        raise ResumeRefusal("recomputed qpos fork audit differs from the archived audit")

    archived_rows = _load_jsonl(output / "rows.jsonl")
    archived_by_key = {str(row["archive_key"]): row for row in archived_rows}
    if len(archived_by_key) != len(archived_rows):
        raise ResumeRefusal("archived partial rows are not unique")
    if len(archived_rows) != int(receipt["query_accounting"]["trajectories_analyzed"]):
        raise ResumeRefusal("archived partial rows disagree with the analysis counter")
    if [row["archive_key"] for row in archived_rows] != keys[:len(archived_rows)]:
        raise ResumeRefusal("archived partial rows are not the leading plan prefix")

    route = exp.route_xz()
    if exp._array_sha256(route, "route_xz") != receipt["campaign_design"]["route_sha256"]:
        raise ResumeRefusal("locked route differs from the archived route hash")
    predicted_centres = {
        onset: float(route[onset + exp.FROZEN_LATENCY_FRAMES, 1])
        for onset in exp.CONTROL_ONSETS
    }
    archived_centres = receipt["campaign_design"].get("predicted_box_centres_m", {})
    if {str(k): v for k, v in predicted_centres.items()} != {
        str(k): float(v) for k, v in archived_centres.items()
    }:
        raise ResumeRefusal("predicted box centres differ from the archived design")
    noise_hash_by_row = {
        int(item["row_index"]): [
            str(noise_evidence[window]["rows"][int(item["row_index"])]["initial_noise_sha256"])
            for window in range(exp.N_WINDOWS)
        ]
        for item in plan
    }
    # Every refusal above leaves the bundle untouched.  Preserve the interrupted state now,
    # before any row or receipt is rewritten (idempotent after a killed resume attempt).
    if not prior_attempt:
        shutil.copyfile(output / "receipt.json", output / INTERRUPTED_RECEIPT)
        shutil.copyfile(output / "rows.jsonl", output / INTERRUPTED_ROWS)

    body = body or G1Body(None)

    rows: list[dict[str, Any]] = []
    determinism: list[dict[str, Any]] = []
    for item in plan:
        seed = int(item["seed"])
        arm = str(item["arm"])
        index = int(item["row_index"])
        key = f"s{seed}_{arm}"
        full_qpos = np.asarray(qpos_archive[key], dtype=float)
        scored_qpos = full_qpos[:exp.N_FRAMES]
        row: dict[str, Any] = {
            **dict(item),
            "archive_key": key,
            "archived_frames": exp.PADDED_FRAMES,
            "scored_frames": exp.N_FRAMES,
            "noise_sha256_by_window": noise_hash_by_row[index],
            "features_sha256": exp._array_sha256(feature_archive[key], "features"),
            "qpos_sha256": exp._array_sha256(qpos_archive[key], "qpos"),
            "supporting_motion": exp._validated_motion_metrics(
                motion_metrics_fn(body, scored_qpos, route)),
        }
        if arm == "all_walk":
            row["event"] = None
            row["fixed_box"] = None
            row["control_events"] = {
                str(onset): exp._validated_event_record(
                    event_detector_fn(body, scored_qpos, route, onset), onset)
                for onset in exp.CONTROL_ONSETS
            }
            row["control_fixed_boxes"] = {
                str(onset): exp._validated_exact_box_record(
                    exact_box_fn(body, scored_qpos, predicted_centres[onset]),
                    predicted_centres[onset])
                for onset in exp.CONTROL_ONSETS
            }
        else:
            onset = int(exp.ONSETS[arm])
            row["event"] = exp._validated_event_record(
                event_detector_fn(body, scored_qpos, route, onset), onset)
            row["fixed_box"] = exp._validated_exact_box_record(
                exact_box_fn(body, scored_qpos, predicted_centres[onset]),
                predicted_centres[onset])
            row["control_events"] = None
            row["control_fixed_boxes"] = None
        if key in archived_by_key:
            equal = _canonical(row) == _canonical(archived_by_key[key])
            determinism.append({"archive_key": key, "recomputed_equals_archived": equal})
            if not equal:
                raise ResumeRefusal(
                    f"recomputed row {key} differs from the archived partial row; "
                    "scoring is not deterministic or the scorers changed"
                )
        rows.append(row)

    if len(rows) != len(plan):
        raise ResumeRefusal("resumed analysis did not preserve the planned denominator")
    summary = exp.summarize_rows(rows, route)
    step0_present = int(summary["event_rates_missing_retained"]["step_0"]["present"])
    walk_seed_hits = int(summary["all_walk_specificity"]["seeds_with_any_event"])
    gates = {
        "step0_substrate": {
            "required_min_present": exp.MIN_STEP0_EVENTS,
            "observed_present": step0_present,
            "planned": len(exp.SEEDS),
            "pass": step0_present >= exp.MIN_STEP0_EVENTS,
        },
        "all_walk_specificity": {
            "allowed_max_seeds_with_any_event": exp.MAX_ALL_WALK_SEEDS_WITH_ANY_EVENT,
            "observed_seeds_with_any_event": walk_seed_hits,
            "planned": len(exp.SEEDS),
            "pass": walk_seed_hits <= exp.MAX_ALL_WALK_SEEDS_WITH_ANY_EVENT,
        },
        "delayed_arm_absence_is_an_outcome_not_a_gate": True,
    }

    counters = dict(receipt["query_accounting"])
    counters["trajectories_analyzed"] = len(rows)
    receipt["query_accounting"] = counters
    receipt["summary"] = summary
    receipt["measurement_gates"] = gates
    receipt["provenance"]["post_analysis_identity_revalidation"] = {
        "git": git_check,
        "sources_unchanged": True,
        "checkpoint_unchanged": "not reloaded; CPU analysis uses no generator",
        "runtime_unchanged": True,
        "physical_model_unchanged": True,
        "prompt_cache_file_unchanged": True,
        "prompt_cache": prompt_cache_check,
    }
    receipt["analysis_resume"] = {
        "schema": RESUME_SCHEMA_VERSION,
        "reason": str(reason),
        "resume_script": "experiments/exp023_prompt_handoff_resume_analysis.py",
        "resume_script_sha256": cal._sha256(Path(__file__).resolve()),
        "interrupted_receipt": {
            "path": INTERRUPTED_RECEIPT,
            "sha256": interrupted_receipt_sha256,
            "trajectories_analyzed_before_interruption": len(archived_rows),
        },
        "interrupted_rows": {"path": INTERRUPTED_ROWS, "sha256": interrupted_rows_sha256},
        "earlier_resume_attempt_killed_before_writing": prior_attempt,
        "regenerated_trajectories": 0,
        "new_ardy_samples": 0,
        "frozen_sources_byte_identical_to_generation": True,
        "archived_partial_rows_recomputed_identically": determinism,
        "dependency_injections": injected,
        "scientific_evidence_eligible": not injected,
        "wall_clock_s": None,
    }

    def persist(status: str, stage: str, *, complete: bool, blocked: bool,
                extra: Mapping[str, Any] | None = None) -> None:
        cal._write_jsonl(output / "rows.jsonl", rows)
        receipt.update({
            "status": status,
            "stage": stage,
            "complete": complete,
            "blocked": blocked,
        })
        if extra:
            receipt.update(dict(extra))
        receipt["evidence_anchors"] = {
            **receipt["evidence_anchors"],
            "rows": {
                "path": "rows.jsonl",
                "n_rows": len(rows),
                "logical_sha256": cal._json_hash(rows),
                "file_sha256": cal._sha256(output / "rows.jsonl"),
            },
            "interrupted_receipt": {
                "path": INTERRUPTED_RECEIPT,
                "file_sha256": interrupted_receipt_sha256,
            },
            "interrupted_rows": {
                "path": INTERRUPTED_ROWS,
                "file_sha256": interrupted_rows_sha256,
            },
        }
        receipt["analysis_resume"]["wall_clock_s"] = float(time.monotonic() - started)
        cal._write_json(output / "receipt.json", receipt)

    if not gates["step0_substrate"]["pass"]:
        persist("refused", "analysis", complete=False, blocked=True, extra={
            "schema": exp.FAILURE_SCHEMA_VERSION,
            "refusal_reason": "step0_substrate_gate_failed",
            "seeds_spent_and_must_not_be_reused": True,
        })
        raise exp.PromptHandoffAbort("STEP-from-start substrate gate failed")
    if not gates["all_walk_specificity"]["pass"]:
        persist("refused", "analysis", complete=False, blocked=True, extra={
            "schema": exp.FAILURE_SCHEMA_VERSION,
            "refusal_reason": "all_walk_specificity_gate_failed",
            "seeds_spent_and_must_not_be_reused": True,
        })
        raise exp.PromptHandoffAbort("all-WALK detector-specificity gate failed")

    receipt["provenance"]["completion_identity_revalidation"] = dict(
        receipt["provenance"]["post_analysis_identity_revalidation"])
    if any((
        counters["schedule_invocations_completed"] != len(exp.locked_chunk_plan(plan)),
        counters["autoregressive_window_calls_completed"]
        != len(exp.locked_chunk_plan(plan)) * exp.N_WINDOWS,
        counters["trajectories_returned"] != len(plan),
        counters["trajectories_converted_to_qpos"] != len(plan),
        counters["trajectories_analyzed"] != len(plan),
    )):
        raise ResumeRefusal("resumed completion accounting is not exact")
    persist("complete", "complete", complete=True, blocked=False,
            extra={"actual_ardy_samples": len(plan)})
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/exp023_prompt_handoff")
    parser.add_argument(
        "--reason",
        default=(
            "host process terminated during the analysis stage after archiving all 32 "
            "trajectories; deterministic CPU re-scoring from the durable archives"
        ),
    )
    args = parser.parse_args(argv)
    receipt = resume_analysis(out=args.out, reason=args.reason)
    summary = receipt["summary"]
    print(json.dumps({
        "status": receipt["status"],
        "actual_ardy_samples": receipt["actual_ardy_samples"],
        "event_rates": summary["event_rates_missing_retained"],
        "slopes": summary["pooled_present_event_slopes_descriptive"],
        "gates": receipt["measurement_gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
