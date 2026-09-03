"""Run the preregistered EXP-031 paired obstacle-present engineering pilot.

The CPU preparation stage freezes two repaired references before this driver can launch SONIC.
This stage then executes the same two motion identities in four paired arms: raw/repaired and
obstacle absent/present.  The result can establish whether the repaired reference-to-controller
chain closes at least once.  It cannot estimate a fresh-pool success rate or attribute a gain to
one part of the repair; both limits are encoded in the summary.

The launch/archival mechanics deliberately reuse EXP-030's obstacle-present harness, including
its patched tracker checkout, per-motion table pose, host gate, immutable attempt receipts and
achieved-state callback.  All four arms therefore differ only in the declared reference variant
and obstacle presence.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp030_obstacle_present as e30  # noqa: E402
from experiments import exp031_prepare_step_repair as prep  # noqa: E402
from scene2motion import host_gate  # noqa: E402
from scene2motion import traversal_eval as te  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402


SCHEMA_VERSION = "exp031-constructive-step-repair-execution-v1"
FAILURE_SCHEMA_VERSION = "exp031-constructive-step-repair-execution-failure-v1"
PROTOCOL = prep.PROTOCOL
RAW_SOURCE = prep.SOURCE
PREPARED = prep.DEFAULT_OUT
DEFAULT_OUT = ROOT / "outputs/exp031_constructive_step_repair_execution"
STAGES = ("launch", "analyze", "all")

PHYSICS_SEED = 0
CANDIDATE_KEYS = prep.EXPECTED_ACCEPTED_KEYS
CANDIDATE_SEEDS = tuple(int(key[1:]) for key in CANDIDATE_KEYS)
SOURCE_POOL_N = len(prep.POOL_SEEDS)

ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "raw_absent", "variant": "raw", "box_height_m": None,
     "question": "does the paired launch reproduce the executable raw substrate?"},
    {"arm": "repaired_absent", "variant": "repaired", "box_height_m": None,
     "question": "did the reference repair itself destroy route completion?"},
    {"arm": "raw_present_05", "variant": "raw", "box_height_m": prep.OBSTACLE_HEIGHT_M,
     "question": "paired obstacle-present baseline on the selected substrate"},
    {"arm": "repaired_present_05", "variant": "repaired",
     "box_height_m": prep.OBSTACLE_HEIGHT_M,
     "question": "primary constructive local-traversal endpoint"},
)
ARM_NAMES = tuple(item["arm"] for item in ARMS)
PRIMARY_ARM = "repaired_present_05"

SOURCE_FILES = (
    "env.sh",
    "experiments/exp030_obstacle_present.py",
    "experiments/exp031_constructive_step_repair.py",
    "experiments/exp031_prepare_step_repair.py",
    "scene2motion/host_gate.py",
    "scene2motion/robot.py",
    "scene2motion/sonic_export.py",
    "scene2motion/sonic_state_export.py",
    "scene2motion/step_repair.py",
    "scene2motion/stepover_eval.py",
    "scene2motion/traversal_eval.py",
)


class PilotAbort(RuntimeError):
    """Fail-closed campaign stop after preserving any evidence already written."""


class PilotPaused(PilotAbort):
    """A resumable host-gate or stage-order pause, not a scientific result."""


def _sha256(path: str | Path) -> str:
    digest = e30._sha256(Path(path))
    if digest is None:
        raise PilotAbort(f"required artifact is missing: {path}")
    return str(digest)


def _jsonable(value: Any) -> Any:
    return e30._jsonable(value)


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def protocol_identity(path: str | Path = PROTOCOL) -> dict[str, Any]:
    return prep.protocol_identity(path)


def prepared_identity(directory: str | Path = PREPARED) -> dict[str, Any]:
    """Validate the immutable pre-execution ledger and candidate arrays by content."""

    directory = Path(directory)
    receipt_path = directory / "receipt.json"
    rows_path = directory / "rows.jsonl"
    qpos_path = directory / "qpos.npz"
    try:
        receipt = json.loads(receipt_path.read_text())
        rows = _read_rows(rows_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PilotAbort(f"prepared EXP-031 artifacts are unreadable: {exc}") from exc
    if receipt.get("schema") != "exp031-step-repair-prepare-v1":
        raise PilotAbort("prepared receipt has the wrong schema")
    if receipt.get("status") != "prepared":
        raise PilotAbort(f"prepared receipt status is {receipt.get('status')!r}, expected 'prepared'")
    if len(rows) != SOURCE_POOL_N:
        raise PilotAbort(f"prepared rows cover {len(rows)} references, expected {SOURCE_POOL_N}")
    by_key = {str(row.get("motion_key")): row for row in rows}
    expected_pool = {f"s{seed}" for seed in prep.POOL_SEEDS}
    if len(by_key) != len(rows) or set(by_key) != expected_pool:
        raise PilotAbort("prepared rows do not contain each source-pool key exactly once")
    accepted = tuple(row["motion_key"] for row in rows if row.get("accepted"))
    if accepted != CANDIDATE_KEYS:
        raise PilotAbort(f"prepared accepted keys {accepted} differ from {CANDIDATE_KEYS}")

    artifacts = receipt.get("artifacts", {})
    if _sha256(rows_path) != artifacts.get("rows", {}).get("sha256"):
        raise PilotAbort("prepared rows no longer match their receipt hash")
    if _sha256(qpos_path) != artifacts.get("qpos", {}).get("sha256"):
        raise PilotAbort("prepared qpos no longer matches its receipt hash")
    expected_content = artifacts.get("qpos", {}).get("content_hashes", {})
    with np.load(qpos_path, allow_pickle=False) as archive:
        if tuple(archive.files) != CANDIDATE_KEYS:
            raise PilotAbort(f"prepared qpos keys {tuple(archive.files)} differ from {CANDIDATE_KEYS}")
        observed_content = {
            key: prep.qpos_sha256(np.asarray(archive[key])) for key in archive.files
        }
    if observed_content != expected_content:
        raise PilotAbort("prepared qpos arrays no longer match their content hashes")
    if receipt.get("summary", {}).get("n_assigned_trials") != SOURCE_POOL_N:
        raise PilotAbort("prepared receipt lost the all-source-pool denominator")
    if receipt.get("summary", {}).get("accepted_keys") != list(CANDIDATE_KEYS):
        raise PilotAbort("prepared receipt accepted-key summary changed")

    return {
        "directory": str(directory.resolve()),
        "receipt": {"path": str(receipt_path.resolve()), "sha256": _sha256(receipt_path)},
        "rows": {"path": str(rows_path.resolve()), "sha256": _sha256(rows_path),
                 "n": len(rows)},
        "qpos": {"path": str(qpos_path.resolve()), "sha256": _sha256(qpos_path),
                 "keys": list(CANDIDATE_KEYS), "content_hashes": observed_content},
        "protocol": dict(receipt.get("protocol", {})),
        "project": dict(receipt.get("project", {})),
        "source": dict(receipt.get("source", {})),
        "support_rule": dict(receipt.get("support_rule", {})),
        "config": dict(receipt.get("config", {})),
        "summary": dict(receipt.get("summary", {})),
    }


def load_clip_sets(*, prepared_dir: str | Path = PREPARED,
                   raw_source: str | Path = RAW_SOURCE) -> dict[str, dict[str, np.ndarray]]:
    """Load exactly the raw and frozen repaired arrays the four arms will consume."""

    identity = prepared_identity(prepared_dir)
    source = prep.source_identity(raw_source)
    if identity.get("source") != source:
        raise PilotAbort("prepared receipt does not bind the current raw source identity")
    raw: dict[str, np.ndarray] = {}
    repaired: dict[str, np.ndarray] = {}
    with np.load(Path(raw_source) / "qpos.npz", allow_pickle=False) as archive:
        for key in CANDIDATE_KEYS:
            raw[key] = np.asarray(archive[key], dtype=np.float32)
    with np.load(Path(prepared_dir) / "qpos.npz", allow_pickle=False) as archive:
        for key in CANDIDATE_KEYS:
            repaired[key] = np.asarray(archive[key], dtype=np.float32)
    return {"raw": raw, "repaired": repaired}


def project_identity(*, code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
                     ) -> dict[str, Any]:
    """Bind the project commit plus every local source that can change this pilot."""

    base = e30.project_identity(code_state_fn=code_state_fn)
    base["source_sha256"] = e30._file_hashes(ROOT, SOURCE_FILES)
    base["source_manifest_sha256"] = cal._json_hash(base["source_sha256"])
    return base


def launch_plan() -> list[dict[str, Any]]:
    plan = []
    for item in ARMS:
        height = item["box_height_m"]
        plan.append({
            "name": f"{item['arm']}_seed{PHYSICS_SEED}",
            "arm": item["arm"],
            "variant": item["variant"],
            "physics_seed": PHYSICS_SEED,
            "seeds": list(CANDIDATE_SEEDS),
            "motion_keys": list(CANDIDATE_KEYS),
            "n_motions": len(CANDIDATE_KEYS),
            "box_height_m": height,
            "obstacle_in_physics": height is not None,
            "table": e30.table_spec(height),
            "extra_overrides": e30.arm_overrides(height),
        })
    if len(plan) != 4:
        raise RuntimeError("EXP-031 requires exactly four paired launches")
    return plan


def _campaign_identity(project: Mapping[str, Any], tracker: Mapping[str, Any],
                       prepared: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    return cal._json_hash({
        "schema": SCHEMA_VERSION,
        "project_commit": project.get("git", {}).get("commit"),
        "project_source_manifest_sha256": project.get("source_manifest_sha256"),
        "tracker": e30.bound_tracker_identity(tracker),
        "prepared_receipt_sha256": prepared.get("receipt", {}).get("sha256"),
        "prepared_qpos_sha256": prepared.get("qpos", {}).get("sha256"),
        "protocol_sha256": protocol.get("sha256"),
        "evaluator_version": e30.evaluator_version(),
        "plan": launch_plan(),
    })


def _persist(output: Path, receipt: dict[str, Any], *, started: float,
             rows: Sequence[Mapping[str, Any]] | None = None,
             summary: Mapping[str, Any] | None = None) -> None:
    anchors = receipt.setdefault("evidence_anchors", {})
    if rows is not None:
        payload = [_jsonable(row) for row in rows]
        cal._write_jsonl(output / "rows.jsonl", payload)
        anchors["rows"] = {
            "n_rows": len(payload), "logical_sha256": cal._json_hash(payload),
            "file_sha256": _sha256(output / "rows.jsonl"),
        }
    if summary is not None:
        payload = _jsonable(summary)
        cal._write_json(output / "summary.json", payload)
        anchors["summary"] = {
            "logical_sha256": cal._json_hash(payload),
            "file_sha256": _sha256(output / "summary.json"),
        }
    receipt["wall_clock_s"] = float(time.monotonic() - started)
    cal._write_json(output / "receipt.json", _jsonable(receipt))


def build_rows(rollouts_by_arm: Mapping[str, Mapping[str, Any]],
               clip_sets: Mapping[str, Mapping[str, np.ndarray]], *,
               collision_fn: Callable[..., Mapping[str, Any]] | None = None,
               ) -> list[dict[str, Any]]:
    """One execution row per declared arm and candidate, without dropping failures."""

    specs = {item["arm"]: item for item in ARMS}
    scenes = {name: e30.scene_for(specs[name]["box_height_m"]) for name in ARM_NAMES}
    scorer = collision_fn if collision_fn is not None else e30.CollisionCache()
    rows: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        spec = specs[arm]
        for key in CANDIDATE_KEYS:
            rollout = rollouts_by_arm[arm][key]
            qpos = np.asarray(rollout.qpos, dtype=float)
            traversal = e30.score_rollout(rollout, scenes[arm], collision_fn=scorer)
            rows.append({
                "arm": arm,
                "variant": spec["variant"],
                "motion_key": key,
                "seed": int(key[1:]),
                "physics_seed": PHYSICS_SEED,
                "motion_id": int(rollout.motion_id),
                "reference_array_sha256": prep.qpos_sha256(clip_sets[spec["variant"]][key]),
                "obstacle_in_physics": spec["box_height_m"] is not None,
                "box_height_m": spec["box_height_m"],
                "valid_frames": int(rollout.valid_length),
                "valid_time_s": float(rollout.valid_length * e30.SAMPLE_DT_S),
                "tracker_terminated": bool(rollout.terminated),
                "tracker_reported_progress": float(rollout.progress),
                "max_root_x_m": float(qpos[:, 0].max()) if len(qpos) else None,
                "final_root_x_m": float(qpos[-1, 0]) if len(qpos) else None,
                "outcome": traversal["outcome"],
                "traversal": traversal,
            })
    return rows


def _arm_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    records = [row["traversal"] for row in selected]
    summary = te.summarise(records)
    completed = [row["motion_key"] for row in selected if row["outcome"] == "completed"]
    obstacle_present = bool(selected[0]["obstacle_in_physics"])
    summary.update({
        "arm": arm,
        "variant": selected[0]["variant"],
        "box_height_m": selected[0]["box_height_m"],
        "obstacle_in_physics": obstacle_present,
        "route_completion": {
            "completed": len(completed), "n_executed_candidates": len(selected),
            "rate": len(completed) / len(selected), "wilson95": e30.wilson(len(completed), len(selected)),
            "completing_motion_keys": completed,
        },
        "local_traversal_completion": ({
            "assessed": True, "completed": len(completed),
            "n_executed_candidates": len(selected), "rate": len(completed) / len(selected),
            "wilson95": e30.wilson(len(completed), len(selected)),
            "completing_motion_keys": completed,
            "definition": (
                "passed the 5 cm obstacle inside the corridor, reached within 0.5 m of the "
                "7.2 m goal after passing, collision-free and upright; walking around fails"
            ),
        } if obstacle_present else {
            "assessed": False, "completed": None, "n_executed_candidates": len(selected),
            "rate": None, "wilson95": None, "completing_motion_keys": [],
            "reason": "no obstacle was present; this arm measures route/tracking completion",
        }),
    })
    return summary


def _paired(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> list[dict[str, Any]]:
    by_arm = {
        arm: {row["motion_key"]: row for row in rows if row["arm"] == arm}
        for arm in (a, b)
    }
    return [{
        "motion_key": key,
        a: {"outcome": by_arm[a][key]["outcome"],
            "max_root_x_m": by_arm[a][key]["max_root_x_m"],
            "tracker_terminated": by_arm[a][key]["tracker_terminated"]},
        b: {"outcome": by_arm[b][key]["outcome"],
            "max_root_x_m": by_arm[b][key]["max_root_x_m"],
            "tracker_terminated": by_arm[b][key]["tracker_terminated"]},
        "b_minus_a_progress_m": (
            None if by_arm[a][key]["max_root_x_m"] is None
            or by_arm[b][key]["max_root_x_m"] is None
            else float(by_arm[b][key]["max_root_x_m"] - by_arm[a][key]["max_root_x_m"])
        ),
    } for key in CANDIDATE_KEYS]


def summarise(rows: Sequence[Mapping[str, Any]], prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Report both the conditional two-candidate endpoint and the all-64 pool endpoint."""

    arms = {arm: _arm_summary(rows, arm) for arm in ARM_NAMES}
    primary_k = int(arms[PRIMARY_ARM]["local_traversal_completion"]["completed"])
    accepted_n = int(prepared["summary"]["n_accepted_for_execution"])
    if accepted_n != len(CANDIDATE_KEYS):
        raise ValueError("prepared accepted denominator changed before summary")

    raw_absent = {row["motion_key"]: row for row in rows if row["arm"] == "raw_absent"}
    repaired_absent = {
        row["motion_key"]: row for row in rows if row["arm"] == "repaired_absent"
    }
    eligible = [key for key in CANDIDATE_KEYS if raw_absent[key]["outcome"] == "completed"]
    survived = [key for key in eligible if repaired_absent[key]["outcome"] == "completed"]
    lost = [key for key in eligible if key not in survived]

    repair_pair = _paired(rows, "raw_present_05", "repaired_present_05")
    obstacle_pair = _paired(rows, "repaired_absent", "repaired_present_05")
    progress_values = [item["b_minus_a_progress_m"] for item in obstacle_pair
                       if item["b_minus_a_progress_m"] is not None]

    return {
        "status": "complete",
        "evaluator_version": e30.evaluator_version(),
        "arms": arms,
        "P1_first_closure": {
            "rule": "at least one repaired_present_05 local traversal completion among 2 candidates",
            "completed": primary_k,
            "n_executed_candidates": accepted_n,
            "held": bool(primary_k >= 1),
            "completing_motion_keys": arms[PRIMARY_ARM]["local_traversal_completion"]
            ["completing_motion_keys"],
            "claim_if_held": (
                "existence only: the frozen prior + repair + frozen SONIC chain completed this "
                "one scene in at least one preregistered engineering trial"
            ),
        },
        "P2_surgery_survival": {
            "definition": "repaired_absent completes among candidates whose raw_absent arm completes",
            "eligible_raw_absent_completions": eligible,
            "n_eligible": len(eligible),
            "survived": survived,
            "n_survived": len(survived),
            "lost": lost,
        },
        "P3_obstacle_effect": {
            "definition": "repaired_present_05 minus repaired_absent maximum root x, per candidate",
            "pairs": obstacle_pair,
            "median_m": float(np.median(progress_values)) if progress_values else None,
            "interpretation": "descriptive paired engineering values; n=2 is not a scene-population estimate",
        },
        "paired_method_comparison": {
            "definition": "raw_present_05 versus repaired_present_05 on the same two identities",
            "pairs": repair_pair,
            "scope": (
                "the arms isolate the reference edit for these two selected identities, but the "
                "development pool is not held out and component-level mechanism attribution is untested"
            ),
        },
        "denominators": {
            "candidate_conditional": {
                "completed": primary_k, "n": accepted_n,
                "rate": primary_k / accepted_n,
                "wilson95": e30.wilson(primary_k, accepted_n),
                "meaning": "completion among the two pre-execution-admitted candidates",
            },
            "source_pool": {
                "completed": primary_k, "n": SOURCE_POOL_N,
                "rate": primary_k / SOURCE_POOL_N,
                "wilson95": e30.wilson(primary_k, SOURCE_POOL_N),
                "n_not_executed_after_preexecution_disposition": SOURCE_POOL_N - accepted_n,
                "meaning": (
                    "method endpoint over all 64 assigned source references; refusals and "
                    "projection rejections count as non-completions"
                ),
            },
        },
        "source_pool_accounting": dict(prepared["summary"]),
        "interpretation_guard": (
            "One archived development pool, one scene, one obstacle position, physics seed 0, "
            "one rollout per candidate and arm. Historical tracker outcomes were known during "
            "development. This pilot supports only a bounded existence result, not a fresh-pool "
            "rate, generalization claim, safety guarantee, or component-level mechanism claim."
        ),
    }


def validate_completed_output(output: Path, receipt: Mapping[str, Any]) -> None:
    rows = _read_rows(output / "rows.jsonl")
    expected_rows = len(ARMS) * len(CANDIDATE_KEYS)
    anchors = receipt.get("evidence_anchors", {})
    row_anchor = anchors.get("rows", {})
    if (len(rows) != expected_rows or row_anchor.get("n_rows") != expected_rows
            or cal._json_hash(rows) != row_anchor.get("logical_sha256")
            or _sha256(output / "rows.jsonl") != row_anchor.get("file_sha256")):
        raise PilotAbort("completed execution rows no longer match their evidence anchor")
    summary = json.loads((output / "summary.json").read_text())
    summary_anchor = anchors.get("summary", {})
    if (cal._json_hash(summary) != summary_anchor.get("logical_sha256")
            or _sha256(output / "summary.json") != summary_anchor.get("file_sha256")
            or receipt.get("summary") != summary):
        raise PilotAbort("completed execution summary no longer matches its evidence anchor")
    for spec in launch_plan():
        record = receipt.get("launches", {}).get(spec["name"])
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise PilotAbort(f"completed receipt lacks launch {spec['name']}")
        e30.load_completed_launch(spec, output)


def run_campaign(
    *, stage: str = "all", out: str | Path = DEFAULT_OUT, prepared_dir: str | Path = PREPARED,
    raw_source: str | Path = RAW_SOURCE, resume: bool = False, dry_run: bool = False,
    timeout_s: int = 2400, sonic_root: str | Path = e30.SONIC_EXP029_ROOT,
    launch_fn: Callable[..., tuple[int, str]] | None = None,
    export_fn: Callable[..., Path] = write_motion_pkl,
    host_gate_fn: Callable[..., Mapping[str, Any]] = host_gate.require_host_resources,
    host_report_fn: Callable[..., Mapping[str, Any]] = host_gate.host_resource_report,
    isaac_fn: Callable[..., Sequence[Mapping[str, Any]]] = host_gate.concurrent_isaac_processes,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    tracker_identity_fn: Callable[..., Mapping[str, Any]] = e30.tracker_identity,
    protocol_identity_fn: Callable[[], Mapping[str, Any]] = protocol_identity,
    prepared_identity_fn: Callable[[Any], Mapping[str, Any]] = prepared_identity,
    source_identity_fn: Callable[[Any], Mapping[str, Any]] = prep.source_identity,
    clip_sets_fn: Callable[..., Mapping[str, Mapping[str, np.ndarray]]] = load_clip_sets,
    collision_fn: Callable[..., Mapping[str, Any]] | None = None,
    mj_model: Any = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    output = Path(out)
    protocol = dict(protocol_identity_fn())
    prepared = dict(prepared_identity_fn(prepared_dir))
    source = dict(source_identity_fn(raw_source))
    project_now = project_identity(code_state_fn=code_state_fn)
    tracker = dict(tracker_identity_fn(sonic_root))
    checkpoint = tracker.get("checkpoint", {}).get("path") or str(
        Path(sonic_root) / e30.RELEASE_CHECKPOINT_RELATIVE
    )
    plan = launch_plan()
    if launch_fn is None:
        launch_fn = e30.sonic_launcher(sonic_root, checkpoint)

    if dry_run:
        return {
            "schema": SCHEMA_VERSION,
            "status": "dry_run", "writes_performed": False,
            "project_dirty_observed": project_now.get("git", {}).get("dirty"),
            "protocol": protocol, "prepared": prepared, "raw_source": source,
            "evaluator_version": e30.evaluator_version(),
            "tracker": e30.bound_tracker_identity(tracker),
            "tracker_guarded_dirty_paths": tracker.get("guarded_dirty_paths"),
            "launch_plan": plan,
            "commands": {spec["name"]: e30.build_sonic_command(
                output / "launches" / spec["name"] / "motions.pkl",
                output / "launches" / spec["name"] / "attempt-000/eval",
                spec["n_motions"], spec["physics_seed"], spec["extra_overrides"],
                checkpoint=checkpoint,
            ) for spec in plan},
            "host_resource_gate": dict(host_report_fn(**host_gate.SONIC_LAUNCH_GATE)),
            "concurrent_isaac_processes": [dict(item) for item in isaac_fn()],
            "campaign_identity_sha256": _campaign_identity(
                project_now, tracker, prepared, protocol,
            ),
        }

    if e30.evaluator_version() != 2:
        raise PilotAbort(f"EXP-031 requires traversal evaluator v2, observed v{e30.evaluator_version()}")
    e30.require_execution_root(sonic_root)
    e30.require_tracker_fix(tracker.get("add_table_fix", {}))
    if tracker.get("guarded_dirty_paths"):
        raise PilotAbort(f"guarded tracker sources are dirty: {tracker['guarded_dirty_paths']}")
    if protocol.get("status") != "preregistered":
        raise PilotAbort(f"EXP-031 protocol status is {protocol.get('status')!r}; preregister first")
    if prepared.get("protocol", {}).get("sha256") != protocol.get("sha256"):
        raise PilotAbort("prepared candidates were frozen under a different protocol hash")
    if prepared.get("source") != source:
        raise PilotAbort("prepared candidates were frozen from a different raw source identity")

    existing = output / "receipt.json"
    old: dict[str, Any] | None = None
    if resume:
        if not existing.is_file():
            raise PilotAbort(f"--resume requires an existing receipt in {output}")
        old = json.loads(existing.read_text())
        if old.get("schema") != SCHEMA_VERSION or old.get("status") == "blocked":
            raise PilotAbort("existing output is not a resumable EXP-031 execution campaign")
        pinned = old.get("provenance", {}).get("project")
        if not isinstance(pinned, dict) or pinned.get("git", {}).get("dirty") is not False:
            raise PilotAbort("existing receipt lacks a clean pinned project identity")
        try:
            e30.validate_project_recheck(pinned, project_now, output)
        except ValueError as exc:
            raise PilotAbort(str(exc)) from exc
        project = dict(pinned)
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise PilotAbort(f"refusing non-empty fresh output: {output}")
        if project_now.get("git", {}).get("dirty") is not False:
            raise PilotAbort("EXP-031 execution requires an exactly clean project worktree")
        project = project_now
        prepared_commit = prepared.get("project", {}).get("git", {}).get("commit")
        if prepared_commit != project.get("git", {}).get("commit"):
            raise PilotAbort(
                f"prepared candidates bind project commit {prepared_commit}, current is "
                f"{project.get('git', {}).get('commit')}"
            )

    campaign_id = _campaign_identity(project, tracker, prepared, protocol)
    if old is not None and old.get("campaign_identity_sha256") != campaign_id:
        raise PilotAbort("existing output has a different campaign identity")
    if old is not None and old.get("status") == "complete":
        validate_completed_output(output, old)
        return old

    started = time.monotonic()
    if old is None:
        try:
            initial_gate = dict(host_gate_fn(**host_gate.SONIC_LAUNCH_GATE))
        except host_gate.HostResourceGateFailed as exc:
            raise PilotAbort(f"host gate failed before campaign creation: {exc}") from exc
        output.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, Any] = {
            "schema": SCHEMA_VERSION, "experiment": "exp031_constructive_step_repair",
            "status": "running", "complete": False, "blocked": False,
            "stage": "preflight", "resume_supported": True,
            "campaign_identity_sha256": campaign_id,
            "claim_scope": (
                "paired engineering existence pilot on two predeclared references from one "
                "development pool; one scene, physics seed 0, one rollout per candidate and arm"
            ),
            "actual_ardy_samples": 0, "n_reused_archived_references": SOURCE_POOL_N,
            "design": {
                "launch_plan": plan, "arms": list(ARMS), "candidate_keys": list(CANDIDATE_KEYS),
                "source_pool_n": SOURCE_POOL_N, "evaluator_version": e30.evaluator_version(),
                "nested_launch_harness": (
                    "EXP-030 run_or_resume_launch is reused unchanged; its nested process-result "
                    "schema name remains exp030 and does not relabel this campaign"
                ),
            },
            "provenance": {
                "project": project, "protocol": protocol, "prepared": prepared,
                "raw_source": source, "tracker": tracker,
                "initial_host_resource_gate": initial_gate,
            },
            "stages_complete": {"launch": False, "analysis": False},
            "launches": {}, "post_launch_revalidation": {}, "host_gate_blocks": [],
        }
        _persist(output, receipt, started=started)
    else:
        receipt = old

    clip_sets = clip_sets_fn(prepared_dir=prepared_dir, raw_source=raw_source)
    model_cache: dict[str, Any] = {}

    def model_for_export() -> Any:
        if mj_model is not None:
            return mj_model
        if "model" not in model_cache:
            model_cache["model"] = e30._default_mj_model()
        return model_cache["model"]

    def revalidate(name: str) -> None:
        if dict(prepared_identity_fn(prepared_dir)) != prepared:
            raise ValueError("prepared candidates changed during SONIC execution")
        if dict(source_identity_fn(raw_source)) != source:
            raise ValueError("raw source artifacts changed during SONIC execution")
        current_tracker = dict(tracker_identity_fn(sonic_root))
        if e30.bound_tracker_identity(current_tracker) != e30.bound_tracker_identity(tracker):
            raise ValueError("tracker checkout/checkpoint changed during SONIC execution")
        project_check = e30.validate_project_recheck(
            project, project_identity(code_state_fn=code_state_fn), output,
        )
        receipt["post_launch_revalidation"][name] = {
            "prepared_unchanged": True, "raw_source_unchanged": True,
            "tracker_unchanged": True, "project": project_check,
        }

    def run_launches() -> None:
        for spec in plan:
            receipt["stage"] = f"launching_{spec['name']}"
            _persist(output, receipt, started=started)
            clips = clip_sets[spec["variant"]]
            pkl = e30.ensure_motion_pkl(
                spec, clips, output, export_fn=export_fn, mj_model=model_for_export(),
            )
            try:
                record, _ = e30.run_or_resume_launch(
                    spec, pkl, output, launch_fn=launch_fn, timeout_s=timeout_s,
                    sonic_root=sonic_root, checkpoint=checkpoint,
                    host_gate_fn=host_gate_fn, isaac_fn=isaac_fn,
                )
            except host_gate.HostResourceGateFailed as exc:
                receipt["host_gate_blocks"].append({
                    "launch": spec["name"], "error": str(exc), "at_unix_s": time.time(),
                })
                receipt["stage"] = f"blocked_host_gate_{spec['name']}"
                _persist(output, receipt, started=started)
                raise PilotPaused(f"blocked_host_gate: {exc}") from exc
            receipt["launches"][spec["name"]] = record
            if not record.get("recovered_or_resumed"):
                revalidate(spec["name"])
            _persist(output, receipt, started=started)

    try:
        stages = [stage] if stage != "all" else ["launch", "analyze"]
        for current in stages:
            if current == "launch":
                run_launches()
                receipt["stages_complete"]["launch"] = True
                _persist(output, receipt, started=started)
            elif current == "analyze":
                if not receipt["stages_complete"]["launch"]:
                    raise PilotPaused("analysis requires all four completed launches")
                receipt["stage"] = "analysis"
                _persist(output, receipt, started=started)
                rollouts_by_arm: dict[str, dict[str, Any]] = {}
                for spec in plan:
                    _, rollouts = e30.load_completed_launch(spec, output)
                    rollouts_by_arm[spec["arm"]] = {item.motion_key: item for item in rollouts}
                expected = set(CANDIDATE_KEYS)
                if set(rollouts_by_arm) != set(ARM_NAMES) or any(
                    set(items) != expected for items in rollouts_by_arm.values()
                ):
                    raise ValueError("launch archives do not cover both candidates in every arm")
                rows = build_rows(rollouts_by_arm, clip_sets, collision_fn=collision_fn)
                _persist(output, receipt, started=started, rows=rows)
                summary = summarise(rows, prepared)
                receipt["stages_complete"]["analysis"] = True
                receipt.update({
                    "status": "complete", "complete": True, "stage": "complete",
                    "sonic_rollouts_requested": len(ARMS) * len(CANDIDATE_KEYS),
                    "sonic_rollouts_returned": sum(
                        int(item.get("n_rollouts", 0)) for item in receipt["launches"].values()
                    ),
                    "summary": summary,
                })
                _persist(output, receipt, started=started, rows=rows, summary=summary)
        if receipt["status"] != "complete":
            receipt["stage"] = f"{stage}_complete"
        _persist(output, receipt, started=started)
        return receipt
    except Exception as exc:
        if isinstance(exc, PilotPaused):
            raise
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
            "blocked": True, "failed_stage": receipt.get("stage"),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        _persist(output, receipt, started=started)
        if isinstance(exc, PilotAbort):
            raise
        raise PilotAbort(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--prepared", default=str(PREPARED))
    parser.add_argument("--raw-source", default=str(RAW_SOURCE))
    parser.add_argument("--sonic-root", default=str(e30.SONIC_EXP029_ROOT))
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_campaign(
            stage=args.stage, out=args.out, prepared_dir=args.prepared,
            raw_source=args.raw_source, resume=args.resume, dry_run=args.dry_run,
            timeout_s=args.timeout_s, sonic_root=args.sonic_root,
        )
    except (PilotAbort, e30.CampaignAbort, host_gate.HostResourceGateFailed) as exc:
        print(json.dumps({"status": "aborted", "error_type": type(exc).__name__,
                          "error": str(exc)}, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps({
            "status": result["status"], "writes_performed": False,
            "project_dirty_observed": result["project_dirty_observed"],
            "protocol": result["protocol"], "prepared": result["prepared"],
            "evaluator_version": result["evaluator_version"],
            "tracker": result["tracker"],
            "tracker_guarded_dirty_paths": result["tracker_guarded_dirty_paths"],
            "host_resource_gate": result["host_resource_gate"],
            "concurrent_isaac_processes": len(result["concurrent_isaac_processes"]),
            "launch_plan": result["launch_plan"],
            "commands": {key: shlex.join(value) for key, value in result["commands"].items()},
        }, indent=2))
    else:
        print(json.dumps({
            "status": result["status"], "stage": result.get("stage"),
            "stages_complete": result.get("stages_complete"),
            "launches": len(result.get("launches", {})),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
