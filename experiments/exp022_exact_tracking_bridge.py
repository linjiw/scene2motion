"""EXP-022A: exact-obstacle SONIC replay of the archived EXP-021 STEP pool.

This is a deliberately post-hoc execution bridge, not a fresh-seed selection experiment.
It sends every one of EXP-021-v2's 64 prompt-elicited references through SONIC, then replays
the achieved states against fixed Scene2Motion boxes centred at 1.2 m and 3.6 m.  Isaac does
not contain either box, so every reported outcome is named ``achieved_replay`` and must not
be described as contact-rich or obstacle-in-the-loop execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp1b_execution_clearance as exp1b  # noqa: E402
from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN, G1Body  # noqa: E402
from scene2motion.sonic_export import SONIC_ROOT as EXPORT_SONIC_ROOT  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    ARCHIVE_SCHEMA_VERSION,
    QPOS_WIDTH,
    load_sonic_state_rollouts,
    sonic_state_archive_schema,
    sonic_state_sample_dt,
)
from scene2motion.stepover_eval import BoxHeightProbe, step_scene  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUT = ROOT / "outputs/exp021_elicited_lift_distribution_v2"
DEFAULT_OUT = ROOT / "outputs/exp022_exact_tracking_bridge"

SCHEMA_VERSION = "exp022a-exact-tracking-bridge-v1"
FAILURE_SCHEMA_VERSION = "exp022a-exact-tracking-bridge-failure-v1"
SOURCE_SCHEMA_VERSION = "exp021-elicited-lift-distribution-v1"
SOURCE_COMMIT = "cde0246fc89d45cb71f23fa6b8b3d8729ec00e6b"
SOURCE_EXP021_SHA256 = "c68dd1c50f5e8d60eb12f17f0909c9b08cbdf4a11157f17d21c0b6224c603258"
SOURCE_CALIBRATION_SHA256 = "ccbf5a13a78578c31579121650ab9378468c5b73e8c61c758a7c9cc9b503c26a"
SOURCE_RUNNER_SHA256 = "29139739ebce4891ebbe694010ff367e281a8f12cb3440a4db5410220b372aa0"
SOURCE_RECEIPT_SHA256 = "0c53d8c5dc2bdfa587f8c0b35d069fcd677f1cdc30221b5ce1afa70d1a5ccf7e"
SOURCE_ROWS_SHA256 = "1d8cc57df2494bd7179940bfe57325ac922f3f41e2581fcc7cb789b5e0c28f71"
SOURCE_QPOS_SHA256 = "2a4b34479aa24894b854301d91bafe1ad870dc530b70eed5b6703eb02c284687"

STEP_PROMPT = "A person steps over an obstacle."
POOL_SEEDS = tuple(range(4400, 4464))
N_FRAMES = 200
FPS = 25
OBSTACLE_DEPTH_M = 0.20
OBSTACLES = (("staged", 1.2), ("unstaged", 3.6))
GRADED_HEIGHTS_M = (0.03, 0.05, 0.08, 0.12, 0.20, 0.30)
CHUNK_SIZE = 32
PHYSICS_SEED = 0
EXPECTED_SAMPLE_DT_S = 0.02
ROUTE_PROGRESS_OK = 0.80
SELECTION_BUDGET = 8
PROCESS_RESULT_SCHEMA = "exp022a-sonic-process-result-v1"
ISAACLAB_ROOT = Path("/home/linjiw/isaaclab-install/IsaacLab")
# ``step_scene`` deliberately makes this box corridor-spanning.  Record the source value in
# the campaign design and require the achieved root to remain inside it when passing, so a
# lateral bypass cannot be relabelled as retained step-over clearance.
OBSTACLE_HALF_WIDTH_M = float(step_scene(1.2, 0.05, OBSTACLE_DEPTH_M).boxes[0].half[1])

CORE_SONIC_FILES = (
    "gear_sonic/eval_agent_trl.py",
    "gear_sonic/trl/callbacks/im_eval_callback.py",
    "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py",
    "gear_sonic/config/callbacks/im_eval.yaml",
    "gear_sonic/config/manager_env/terminations/tracking/eval.yaml",
    "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py",
    "sonic_release/config.yaml",
)
SOURCE_FILES = (
    "env.sh",
    "experiments/calibrate_ramp_route_phase.py",
    "experiments/exp022_exact_tracking_bridge.py",
    "experiments/exp1b_execution_clearance.py",
    "scene2motion/scenes.py",
    "scene2motion/sonic_export.py",
    "scene2motion/sonic_state_export.py",
    "scene2motion/stepover_eval.py",
    "scene2motion/robot.py",
)


class BridgeAbort(RuntimeError):
    """Fail-closed stop after any available evidence has been made durable."""


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_at_commit(repo: Path, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{relative_path}"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"the locked EXP-021 provenance file is unavailable: {relative_path}"
        ) from exc


def _historical_source_identity(repo: Path) -> dict[str, str]:
    locked = {
        "experiments/exp021_elicited_lift_distribution.py": SOURCE_EXP021_SHA256,
        "experiments/calibrate_ramp_route_phase.py": SOURCE_CALIBRATION_SHA256,
        "scene2motion/runner.py": SOURCE_RUNNER_SHA256,
    }
    payloads = {name: _git_blob_at_commit(repo, name) for name in locked}
    hashes = {name: hashlib.sha256(payload).hexdigest()
              for name, payload in payloads.items()}
    if hashes != locked:
        raise ValueError("EXP-021 historical source chain does not match its locked hashes")
    for name in ("experiments/calibrate_ramp_route_phase.py", "scene2motion/runner.py"):
        if b"NOISE_STREAM_VERSION = 2" not in payloads[name]:
            raise ValueError(f"EXP-021 historical {name} is not noise-stream v2")
    exp021_source = payloads["experiments/exp021_elicited_lift_distribution.py"]
    if (b"runner.noise_stream_version" not in exp021_source
            or b"cal.NOISE_STREAM_VERSION" not in exp021_source):
        raise ValueError("EXP-021 source lacks its runner/calibration v2 equality gate")
    return hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSONL artifact {path}: {exc}") from exc


def load_source_bundle(source_dir: str | Path = SOURCE_OUT, *, repo: Path = ROOT) -> dict[str, Any]:
    """Load and content-validate the one archived pool this bridge is allowed to consume."""
    source = Path(source_dir)
    receipt_path = source / "receipt.json"
    rows_path = source / "rows.jsonl"
    qpos_path = source / "qpos.npz"
    locked_files = {
        "receipt.json": SOURCE_RECEIPT_SHA256,
        "rows.jsonl": SOURCE_ROWS_SHA256,
        "qpos.npz": SOURCE_QPOS_SHA256,
    }
    for name, expected in locked_files.items():
        got = _sha256(source / name)
        if got != expected:
            raise ValueError(
                f"locked EXP-021 artifact {name} has sha256 {got}, expected {expected}")
    historical_sources = _historical_source_identity(repo)

    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid EXP-021 receipt: {exc}") from exc
    if (
        receipt.get("schema") != SOURCE_SCHEMA_VERSION
        or receipt.get("experiment") != "exp021_elicited_lift_distribution"
        or receipt.get("status") != "complete"
        or receipt.get("complete") is not True
    ):
        raise ValueError("EXP-021 receipt is not the locked completed campaign")
    if receipt.get("actual_ardy_samples") != len(POOL_SEEDS):
        raise ValueError("EXP-021 sample total is not exactly 64")
    accounting = receipt.get("query_accounting", {})
    if accounting.get("samples_launched") != 64 or accounting.get("samples_returned") != 64:
        raise ValueError("EXP-021 launched/returned accounting is not exactly 64/64")
    design = receipt.get("design", {})
    if design.get("pool_seeds") != list(POOL_SEEDS) or design.get("prompt") != STEP_PROMPT:
        raise ValueError("EXP-021 pool seeds or STEP prompt drifted")
    if design.get("graded_heights_m") != list(GRADED_HEIGHTS_M):
        raise ValueError("EXP-021 graded heights drifted")
    code = receipt.get("provenance", {}).get("code", {})
    if code.get("commit") != SOURCE_COMMIT or code.get("dirty") is not False:
        raise ValueError("EXP-021 code provenance is not the locked clean v2-sampler run")
    if cal.NOISE_STREAM_VERSION != 2:
        raise ValueError("current RAMP identity no longer defines noise stream version 2")

    rows = _read_jsonl(rows_path)
    anchors = receipt.get("evidence_anchors", {})
    if len(rows) != 64 or anchors.get("rows", {}).get("n_rows") != 64:
        raise ValueError("EXP-021 rows do not have the locked denominator 64")
    if cal._json_hash(rows) != anchors["rows"].get("logical_sha256"):
        raise ValueError("EXP-021 logical row hash does not match its receipt")
    if _sha256(rows_path) != anchors["rows"].get("file_sha256"):
        raise ValueError("EXP-021 rows file hash does not match its receipt")
    row_by_seed = {int(row["seed"]): row for row in rows}
    if len(row_by_seed) != 64 or set(row_by_seed) != set(POOL_SEEDS):
        raise ValueError("EXP-021 rows are missing or duplicating locked seeds")
    if any(row.get("prompt") != STEP_PROMPT for row in rows):
        raise ValueError("EXP-021 contains a non-STEP row")

    try:
        with np.load(qpos_path, allow_pickle=False) as archive:
            clips = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid EXP-021 qpos archive: {exc}") from exc
    expected_keys = {f"s{seed}" for seed in POOL_SEEDS}
    if set(clips) != expected_keys or anchors.get("qpos", {}).get("n_arrays") != 64:
        raise ValueError("EXP-021 qpos keys do not match the locked 64-seed pool")
    for seed in POOL_SEEDS:
        key = f"s{seed}"
        qpos = np.asarray(clips[key])
        if qpos.shape != (N_FRAMES, QPOS_WIDTH) or not np.isfinite(qpos).all():
            raise ValueError(f"EXP-021 {key} qpos must be finite ({N_FRAMES}, {QPOS_WIDTH})")
        if cal._array_hash({key: qpos}) != row_by_seed[seed].get("qpos_content_sha256"):
            raise ValueError(f"EXP-021 {key} does not match its per-row qpos hash")
    content_hash = cal._array_hash(clips)
    if content_hash != anchors["qpos"].get("content_sha256"):
        raise ValueError("EXP-021 qpos content hash does not match its receipt")

    identity = {
        "path": str(source.resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "rows_file_sha256": _sha256(rows_path),
        "rows_logical_sha256": cal._json_hash(rows),
        "qpos_archive_sha256": _sha256(qpos_path),
        "qpos_content_sha256": content_hash,
        "n_rows": len(rows),
        "n_qpos": len(clips),
        "pool_seeds": list(POOL_SEEDS),
        "source_commit": SOURCE_COMMIT,
        "source_exp021_sha256_at_commit": SOURCE_EXP021_SHA256,
        "historical_source_sha256": historical_sources,
        "noise_stream_version": 2,
        "noise_stream_evidence": (
            "the locked clean EXP-021 source asserts runner.noise_stream_version == "
            "cal.NOISE_STREAM_VERSION before generation; the locked constant is 2"
        ),
    }
    return {"receipt": receipt, "rows": rows, "clips": clips, "identity": identity}


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).splitlines()
        tracked_status = subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).splitlines()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"could not bind git state for {root}") from exc
    return {
        "commit": commit,
        "dirty": bool(status),
        "tracked_dirty": bool(tracked_status),
        "status": status,
        "tracked_status": tracked_status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _file_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    result = {}
    for name in names:
        digest = _sha256(root / name)
        if digest is None:
            raise ValueError(f"required source artifact is missing: {root / name}")
        result[name] = digest
    return result


def project_identity(
    repo: Path = ROOT,
    *,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
) -> dict[str, Any]:
    model_path = Path(ARDY_G1_XML)
    model_hash = _sha256(model_path)
    if model_hash is None:
        raise ValueError(f"MuJoCo physical model is missing: {model_path}")
    return {
        "git": dict(code_state_fn(repo)),
        "source_sha256": _file_hashes(repo, SOURCE_FILES),
        "runtime": cal._runtime_identity(),
        "physical_model": {
            "path": str(model_path.resolve()),
            "sha256": model_hash,
            "body_margin_m": BODY_MARGIN,
        },
    }


def validate_project_recheck(
    initial: Mapping[str, Any], current: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
    """Require unchanged code/model while allowing only this campaign's output files."""
    if current.get("source_sha256") != initial.get("source_sha256"):
        raise ValueError("Scene2Motion core source hashes changed during execution")
    if current.get("runtime") != initial.get("runtime"):
        raise ValueError("Scene2Motion numerical runtime changed during execution")
    if current.get("physical_model") != initial.get("physical_model"):
        raise ValueError("Scene2Motion physical model changed during execution")
    return cal._verify_completion_git_state(
        initial.get("git", {}), current.get("git", {}),
        repo=ROOT, output=output,
    )


def _sonic_python_runtime(python: Path) -> dict[str, Any]:
    """Bind the numerical packages used by the separate SONIC interpreter."""
    script = r'''
import importlib.metadata as metadata
import json
import sys
import torch

names = ("numpy", "torch", "isaaclab", "isaacsim", "warp-lang")
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "packages": versions,
    "torch_cuda_version": torch.version.cuda,
    "torch_cudnn_version": torch.backends.cudnn.version(),
}, sort_keys=True))
'''
    try:
        completed = subprocess.run(
            [str(python), "-c", script], capture_output=True, text=True,
            check=True, timeout=60, stdin=subprocess.DEVNULL,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not bind SONIC Python runtime: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("packages"), dict):
        raise ValueError("SONIC Python runtime identity is malformed")
    required = {"numpy", "torch", "isaaclab", "isaacsim", "warp-lang"}
    if set(payload["packages"]) != required or any(
            payload["packages"][name] is None for name in required):
        raise ValueError("SONIC Python runtime is missing a required package identity")
    resolved = python.resolve()
    executable_sha = _sha256(resolved)
    if executable_sha is None:
        raise ValueError(f"SONIC Python executable disappeared: {resolved}")
    return {
        **payload,
        "requested_executable": str(python.resolve()),
        "executable_sha256": executable_sha,
    }


def tracker_identity(sonic_root: str | Path = exp1b.SONIC) -> dict[str, Any]:
    root = Path(sonic_root).resolve()
    if root != Path(EXPORT_SONIC_ROOT).resolve():
        raise ValueError(
            f"SONIC export root {Path(EXPORT_SONIC_ROOT).resolve()} differs from "
            f"execution root {root}")
    git = _git_state(root)
    if git["tracked_dirty"]:
        raise ValueError("SONIC has tracked modifications; refusing an unpinned tracker runtime")
    checkpoint = root / "sonic_release/last.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError(f"SONIC checkpoint is missing or empty: {checkpoint}")
    if not Path(exp1b.SONIC_PY).is_file():
        raise ValueError(f"SONIC Python is missing: {exp1b.SONIC_PY}")
    if not ISAACLAB_ROOT.is_dir():
        raise ValueError(f"Isaac Lab checkout is missing: {ISAACLAB_ROOT}")
    isaac_git = _git_state(ISAACLAB_ROOT)
    if isaac_git["dirty"]:
        raise ValueError("Isaac Lab checkout must be exactly clean for an execution campaign")
    core = _file_hashes(root, CORE_SONIC_FILES)
    return {
        "root": str(root),
        "git": git,
        "core_source_sha256": core,
        "core_source_manifest_sha256": cal._json_hash(core),
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "sha256": _sha256(checkpoint),
        },
        "python": str(exp1b.SONIC_PY.resolve()),
        "python_runtime": _sonic_python_runtime(Path(exp1b.SONIC_PY)),
        "isaaclab": {"root": str(ISAACLAB_ROOT.resolve()), "git": isaac_git},
        "physics_seed": PHYSICS_SEED,
        "expected_achieved_sample_dt_s": EXPECTED_SAMPLE_DT_S,
        "callback_schema_version": ARCHIVE_SCHEMA_VERSION,
    }


def chunk_plan() -> list[dict[str, Any]]:
    plan = []
    for index, start in enumerate(range(0, len(POOL_SEEDS), CHUNK_SIZE)):
        seeds = POOL_SEEDS[start:start + CHUNK_SIZE]
        if len(seeds) != CHUNK_SIZE:
            raise RuntimeError("locked pool no longer forms exactly two 32-motion chunks")
        plan.append({
            "chunk": index,
            "name": f"chunk{index:02d}_seed{PHYSICS_SEED}",
            "physics_seed": PHYSICS_SEED,
            "seeds": list(seeds),
            "motion_keys": [f"s{seed}" for seed in seeds],
            "n_motions": len(seeds),
        })
    if len(plan) != 2:
        raise RuntimeError("locked bridge must contain exactly two launches")
    return plan


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def score_trajectory(
    qpos: np.ndarray,
    obstacle_x_m: float,
    *,
    terminated: bool = False,
    reported_progress: float | None = None,
) -> dict[str, Any]:
    """Exact fixed-centre geometry and explicit progress state for one trajectory."""
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1:] != (QPOS_WIDTH,) or not np.isfinite(qpos).all():
        raise ValueError(f"qpos must be finite (T, {QPOS_WIDTH}), got {qpos.shape}")
    if len(qpos) == 0:
        return {
            "valid_frames": 0,
            "tracker_terminated": bool(terminated),
            "tracker_reported_progress": _finite_or_none(reported_progress),
            "max_root_x_m": None,
            "final_root_x_m": None,
            "max_abs_root_y_m": None,
            "pass_frame": None,
            "root_y_at_pass_m": None,
            "actual_route_progress_ratio": 0.0,
            "passed_obstacle": False,
            "passed_within_lateral_corridor": False,
            "finished_beyond_obstacle": False,
            "route_completed": False,
            "stalled": bool(not terminated),
            "stalled_before_obstacle": bool(not terminated),
            "max_box_height_lower_bound_m": 0.0,
            "exact_clears": {f"{height:g}": False for height in GRADED_HEIGHTS_M},
            "achieved_replay_clear_after_passing": {
                f"{height:g}": False for height in GRADED_HEIGHTS_M},
        }
    max_x = float(np.max(qpos[:, 0]))
    final_x = float(qpos[-1, 0])
    max_abs_y = float(np.max(np.abs(qpos[:, 1])))
    planned_distance = float(cal.PILOT_ROUTE_LENGTH_M)
    progress = max(0.0, (max_x - float(qpos[0, 0])) / planned_distance)
    expanded_hi = float(obstacle_x_m + OBSTACLE_DEPTH_M / 2 + BODY_MARGIN)
    passed = bool(max_x >= expanded_hi)
    pass_indices = np.flatnonzero(qpos[:, 0] >= expanded_hi)
    pass_frame = int(pass_indices[0]) if len(pass_indices) else None
    root_y_at_pass = None if pass_frame is None else float(qpos[pass_frame, 1])
    passed_in_corridor = bool(
        passed and root_y_at_pass is not None
        and abs(root_y_at_pass) <= OBSTACLE_HALF_WIDTH_M
    )
    finished_beyond = bool(final_x >= expanded_hi)
    route_completed = bool(progress >= ROUTE_PROGRESS_OK)
    stalled = bool(not terminated and not route_completed)

    probe = BoxHeightProbe(float(obstacle_x_m), OBSTACLE_DEPTH_M)
    exact = {
        f"{height:g}": bool(probe.clears(qpos, height))
        for height in GRADED_HEIGHTS_M
    }
    lower_bound = float(probe.probe(qpos))
    retained = {
        key: bool(
            not terminated and passed and passed_in_corridor
            and finished_beyond and value
        )
        for key, value in exact.items()
    }
    return {
        "valid_frames": len(qpos),
        "tracker_terminated": bool(terminated),
        "tracker_reported_progress": _finite_or_none(reported_progress),
        "max_root_x_m": max_x,
        "final_root_x_m": final_x,
        "max_abs_root_y_m": max_abs_y,
        "pass_frame": pass_frame,
        "root_y_at_pass_m": root_y_at_pass,
        "actual_route_progress_ratio": progress,
        "passed_obstacle": passed,
        "passed_within_lateral_corridor": passed_in_corridor,
        "finished_beyond_obstacle": finished_beyond,
        "route_completed": route_completed,
        "stalled": stalled,
        "stalled_before_obstacle": bool(stalled and not passed),
        "max_box_height_lower_bound_m": lower_bound,
        "exact_clears": exact,
        "achieved_replay_clear_after_passing": retained,
    }


def build_rows(
    trajectories: Mapping[str, Any],
    *,
    source_rows: Mapping[int, Mapping[str, Any]],
    tier: str,
    scorer: Callable[..., Mapping[str, Any]] = score_trajectory,
    archive_schema: int | None = None,
    sample_dt_s: float | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for seed in POOL_SEEDS:
        key = f"s{seed}"
        value = trajectories[key]
        if tier == "reference":
            qpos = np.asarray(value)
            terminated = False
            reported_progress = 1.0
            motion_id = None
        else:
            qpos = np.asarray(value.qpos)
            terminated = bool(value.terminated)
            reported_progress = float(value.progress)
            motion_id = int(value.motion_id)
        for obstacle_label, obstacle_x in OBSTACLES:
            metrics = dict(scorer(
                qpos, obstacle_x, terminated=terminated,
                reported_progress=reported_progress))
            rows.append({
                "tier": tier,
                "seed": seed,
                "motion_key": key,
                "motion_id": motion_id,
                "obstacle_label": obstacle_label,
                "obstacle_x_m": obstacle_x,
                "obstacle_depth_m": OBSTACLE_DEPTH_M,
                "source_qpos_content_sha256": source_rows[seed]["qpos_content_sha256"],
                "archive_schema_version": archive_schema,
                "sample_dt_s": sample_dt_s,
                **metrics,
            })
    return rows


def summarize(reference_rows: Sequence[Mapping[str, Any]],
              achieved_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def tier_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = {}
        for label, _ in OBSTACLES:
            group = [row for row in rows if row["obstacle_label"] == label]
            entry = {
                "n": len(group),
                "passed_obstacle": sum(bool(row["passed_obstacle"]) for row in group),
                "passed_within_lateral_corridor": sum(
                    bool(row["passed_within_lateral_corridor"]) for row in group),
                "finished_beyond_obstacle": sum(
                    bool(row["finished_beyond_obstacle"]) for row in group),
                "route_completed": sum(bool(row["route_completed"]) for row in group),
                "tracker_terminated": sum(bool(row["tracker_terminated"]) for row in group),
                "stalled": sum(bool(row["stalled"]) for row in group),
                "stalled_before_obstacle": sum(
                    bool(row["stalled_before_obstacle"]) for row in group),
                "exact_clears": {},
                "achieved_replay_clear_after_passing": {},
            }
            for height in GRADED_HEIGHTS_M:
                key = f"{height:g}"
                entry["exact_clears"][key] = sum(bool(row["exact_clears"][key]) for row in group)
                entry["achieved_replay_clear_after_passing"][key] = sum(
                    bool(row["achieved_replay_clear_after_passing"][key]) for row in group)
            result[label] = entry
        return result

    reference_by_key = {
        (int(row["seed"]), str(row["obstacle_label"])): row for row in reference_rows
    }
    achieved_by_key = {
        (int(row["seed"]), str(row["obstacle_label"])): row for row in achieved_rows
    }
    if len(reference_by_key) != len(reference_rows):
        raise ValueError("reference summary rows duplicate a seed/obstacle pair")
    if len(achieved_by_key) != len(achieved_rows):
        raise ValueError("achieved summary rows duplicate a seed/obstacle pair")

    paired_retention: dict[str, Any] = {}
    if achieved_rows:
        if set(achieved_by_key) != set(reference_by_key):
            raise ValueError("paired retention requires one achieved row per reference row")
        for label, _ in OBSTACLES:
            paired_retention[label] = {}
            for height in GRADED_HEIGHTS_M:
                height_key = f"{height:g}"
                transitions = {
                    "reference_clear": 0,
                    "achieved_guarded_clear": 0,
                    "retained_reference_clear": 0,
                    "lost_reference_clear": 0,
                    "achieved_only_gain": 0,
                    "neither_clear": 0,
                    "retained_and_route_complete": 0,
                }
                for seed in POOL_SEEDS:
                    reference = reference_by_key[(seed, label)]
                    achieved = achieved_by_key[(seed, label)]
                    reference_clear = bool(reference["exact_clears"][height_key])
                    achieved_clear = bool(
                        achieved["achieved_replay_clear_after_passing"][height_key])
                    transitions["reference_clear"] += int(reference_clear)
                    transitions["achieved_guarded_clear"] += int(achieved_clear)
                    transitions["retained_reference_clear"] += int(
                        reference_clear and achieved_clear)
                    transitions["lost_reference_clear"] += int(
                        reference_clear and not achieved_clear)
                    transitions["achieved_only_gain"] += int(
                        not reference_clear and achieved_clear)
                    transitions["neither_clear"] += int(
                        not reference_clear and not achieved_clear)
                    transitions["retained_and_route_complete"] += int(
                        reference_clear and achieved_clear and achieved["route_completed"])
                denominator = transitions["reference_clear"]
                paired_retention[label][height_key] = {
                    "n_paired": len(POOL_SEEDS),
                    **transitions,
                    "retention_fraction_of_reference_clear": (
                        transitions["retained_reference_clear"] / denominator
                        if denominator else None
                    ),
                    "endpoint_guard": (
                        "achieved replay must be non-terminated, clear the exact box, pass "
                        "inside the lateral corridor, and finish beyond the obstacle"
                    ),
                }

    selection = {}
    for label, _ in OBSTACLES:
        selection[label] = {}
        for height in GRADED_HEIGHTS_M:
            height_key = f"{height:g}"
            block_rows = []
            for block_index, start in enumerate(range(0, len(POOL_SEEDS), SELECTION_BUDGET)):
                seeds = POOL_SEEDS[start:start + SELECTION_BUDGET]
                references = [row for row in reference_rows
                              if row["obstacle_label"] == label and row["seed"] in seeds]
                references.sort(key=lambda row: seeds.index(int(row["seed"])))
                selected = next(
                    (row for row in references if row["exact_clears"][height_key]), None)
                selected_seed = None if selected is None else int(selected["seed"])
                achieved = (None if selected_seed is None
                            else achieved_by_key.get((selected_seed, label)))
                retained = bool(
                    achieved is not None
                    and achieved["achieved_replay_clear_after_passing"][height_key])
                block_rows.append({
                    "block": block_index,
                    "seeds": list(seeds),
                    "selected_seed": selected_seed,
                    "calls_spent": (len(seeds) if selected is None
                                    else seeds.index(selected_seed) + 1),
                    "reference_selected": selected is not None,
                    "achieved_replay_retained": retained,
                })
            selection[label][height_key] = {
                "n_descriptive_seed_blocks": len(block_rows),
                "blocks_with_reference_selection": sum(
                    row["reference_selected"] for row in block_rows),
                "blocks_with_achieved_replay_retention": sum(
                    row["achieved_replay_retained"] for row in block_rows),
                "blocks": block_rows,
                "inference_guard": (
                    "post-hoc deterministic partitions of one archived scene; descriptive only"
                ),
            }
    return {
        "reference": tier_summary(reference_rows),
        "achieved": tier_summary(achieved_rows),
        "paired_reference_to_achieved_retention": paired_retention,
        "selection_n8_descriptive": selection,
        "interpretation_guard": (
            "SONIC tracked references without an Isaac obstacle. Achieved states were replayed "
            "against Scene2Motion geometry; these are not contact-rich execution outcomes."
        ),
    }


def _atomic_text(path: Path, text: str) -> None:
    payload = text.encode()
    cal._atomic_write(path, lambda handle: handle.write(payload))


def _parse_sonic_log(log: str) -> dict[str, float]:
    parsed = {}
    for line in log.splitlines():
        for prefix, name in (("Success Rate:", "success_rate"),
                             ("Progress Rate:", "progress_rate")):
            if line.startswith(prefix):
                value = float(line.split(":", 1)[1])
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"SONIC {name} must be finite and in [0, 1]")
                parsed[name] = value
    return parsed


def _validate_motion_pkl(path: Path, expected_keys: Sequence[str]) -> None:
    try:
        with path.open("rb") as handle:
            motions = pickle.load(handle)
    except (OSError, ValueError, pickle.PickleError, EOFError) as exc:
        raise ValueError(f"unreadable SONIC motion pickle {path}: {exc}") from exc
    if set(motions) != set(expected_keys):
        raise ValueError("SONIC motion pickle keys do not match its locked chunk")
    for key in expected_keys:
        if len(motions[key].get("root_trans_offset", ())) != N_FRAMES:
            raise ValueError(f"SONIC motion {key} does not have {N_FRAMES} frames")


def ensure_motion_pkl(
    chunk: Mapping[str, Any], clips: Mapping[str, np.ndarray], output: Path,
    *, export_fn: Callable[..., Path] = write_motion_pkl, mj_model: Any = None,
) -> Path:
    chunk_dir = output / "launches" / str(chunk["name"])
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / "motions.pkl"
    temporary = chunk_dir / "motions.expected.tmp.pkl"
    if temporary.exists():
        raise BridgeAbort(f"stale temporary motion pickle requires inspection: {temporary}")
    selected = {key: clips[key] for key in chunk["motion_keys"]}
    export_fn(selected, temporary, fps=FPS, mj_model=mj_model)
    _validate_motion_pkl(temporary, chunk["motion_keys"])
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if path.exists():
        _validate_motion_pkl(path, chunk["motion_keys"])
        if _sha256(path) != _sha256(temporary):
            raise BridgeAbort(f"existing motion pickle differs from deterministic export: {path}")
        temporary.unlink()
        return path
    os.replace(temporary, path)
    directory_fd = os.open(chunk_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def validate_attempt(
    attempt: Path, expected_keys: Sequence[str],
) -> tuple[dict[str, Any], list[Any]]:
    log_path = attempt / "sonic.log"
    metrics_path = attempt / "eval/metrics_eval.json"
    if not log_path.is_file() or not metrics_path.is_file():
        raise ValueError("SONIC log or metrics_eval.json is missing")
    log = log_path.read_text()
    parsed = _parse_sonic_log(log)
    if set(parsed) != {"success_rate", "progress_rate"}:
        raise ValueError("SONIC log lacks success/progress summaries")
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid metrics_eval.json: {exc}") from exc
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics_eval.json must be a nonempty object")
    schema = sonic_state_archive_schema(attempt / "eval")
    if schema != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(f"new bridge launch wrote archive schema {schema}, expected 2")
    sample_dt = sonic_state_sample_dt(attempt / "eval")
    if not np.isclose(sample_dt, EXPECTED_SAMPLE_DT_S, rtol=0.0, atol=1e-12):
        raise ValueError(f"SONIC sample_dt_s={sample_dt}, expected {EXPECTED_SAMPLE_DT_S}")
    rollouts = load_sonic_state_rollouts(attempt / "eval")
    if {rollout.motion_key for rollout in rollouts} != set(expected_keys):
        raise ValueError("achieved archive keys do not match the locked chunk")
    if len(rollouts) != len(expected_keys):
        raise ValueError("achieved archive denominator does not match the locked chunk")
    expected_ids = set(range(len(expected_keys)))
    if {int(rollout.motion_id) for rollout in rollouts} != expected_ids:
        raise ValueError("achieved archive motion ids do not cover the locked chunk order")
    for rollout in rollouts:
        if str(expected_keys[int(rollout.motion_id)]) != rollout.motion_key:
            raise ValueError(
                f"achieved archive motion-id/key mapping disagrees for id {rollout.motion_id}"
            )
    by_key = {rollout.motion_key: rollout for rollout in rollouts}
    all_metrics = metrics.get("eval/all_metrics_dict")
    if not isinstance(all_metrics, dict):
        raise ValueError("metrics_eval.json lacks eval/all_metrics_dict")
    metric_keys = all_metrics.get("motion_keys")
    metric_terminated = all_metrics.get("terminated")
    metric_progress = all_metrics.get("progress")
    if not all(isinstance(values, list)
               for values in (metric_keys, metric_terminated, metric_progress)):
        raise ValueError("metrics_eval.json lacks per-motion keys/termination/progress")
    if (len(metric_keys) != len(expected_keys)
            or set(map(str, metric_keys)) != set(expected_keys)
            or len(metric_terminated) != len(expected_keys)
            or len(metric_progress) != len(expected_keys)):
        raise ValueError("metrics_eval.json motion-key denominator differs from the chunk")
    if any(not isinstance(value, bool) for value in metric_terminated):
        raise ValueError("metrics_eval.json termination values must be booleans")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in metric_progress):
        raise ValueError("metrics_eval.json progress values must be numeric")
    metric_by_key = {
        str(key): (bool(terminated), float(progress))
        for key, terminated, progress in zip(metric_keys, metric_terminated, metric_progress)
    }
    if any(not np.isfinite(progress) or not 0.0 <= progress <= 1.0
           for _, progress in metric_by_key.values()):
        raise ValueError("metrics_eval.json has invalid per-motion progress")
    for key, rollout in by_key.items():
        metric_termination, metric_motion_progress = metric_by_key[key]
        if metric_termination != rollout.terminated or not np.isclose(
                metric_motion_progress, rollout.progress, rtol=0.0, atol=1e-6):
            raise ValueError(f"metrics/archive state disagrees for {key}")
    failed_keys = metrics.get("failed_keys")
    expected_failed = {key for key, rollout in by_key.items() if rollout.terminated}
    if not isinstance(failed_keys, list) or set(map(str, failed_keys)) != expected_failed:
        raise ValueError("metrics failed_keys disagree with the achieved archive")
    archive_success = float(np.mean([not rollout.terminated for rollout in rollouts]))
    archive_progress = float(np.mean([rollout.progress for rollout in rollouts]))
    metric_success = metrics.get("eval/success/success_rate")
    metric_progress_rate = metrics.get("eval/success/progress_rate")
    for label, observed, metric_value, log_value in (
        ("success", archive_success, metric_success, parsed["success_rate"]),
        ("progress", archive_progress, metric_progress_rate, parsed["progress_rate"]),
    ):
        try:
            metric_number = float(metric_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metrics_eval.json lacks numeric {label} rate") from exc
        if (not np.isfinite(metric_number)
                or not np.isclose(observed, metric_number, rtol=0.0, atol=1e-6)
                or not np.isclose(observed, log_value, rtol=0.0, atol=1e-6)):
            raise ValueError(f"SONIC log/metrics/archive disagree on {label} rate")
    canonical = attempt / "eval/achieved_qpos.npz"
    shards = sorted((attempt / "eval").glob("achieved_qpos.rank*.npz"))
    if canonical.is_file() and shards:
        raise ValueError("ambiguous achieved-state output contains canonical and rank archives")
    archive_paths = [canonical] if canonical.is_file() else shards
    return {
        "attempt": str(attempt.resolve()),
        "n_rollouts": len(rollouts),
        "archive_schema_version": schema,
        "sample_dt_s": sample_dt,
        "sonic_log": parsed,
        "metrics_key_set": sorted(metrics),
        "archive_aggregate": {
            "success_rate": archive_success,
            "progress_rate": archive_progress,
        },
        "motion_id_key_map_sha256": cal._json_hash([
            {"motion_id": int(rollout.motion_id), "motion_key": rollout.motion_key}
            for rollout in sorted(rollouts, key=lambda item: item.motion_id)
        ]),
        "artifacts": {
            "log_sha256": _sha256(log_path),
            "metrics_sha256": _sha256(metrics_path),
            "archive_sha256": {path.name: _sha256(path) for path in archive_paths},
        },
    }, rollouts


def _write_process_result(
    attempt: Path, *, returncode: int, log_path: Path,
    chunk: Mapping[str, Any], motion_pkl_sha256: str,
) -> dict[str, Any]:
    """Persist the subprocess outcome before any scientific artifact is adopted."""
    payload = {
        "schema": PROCESS_RESULT_SCHEMA,
        "returncode": int(returncode),
        "returncode_observed": True,
        "sonic_log_sha256": _sha256(log_path),
        "chunk": str(chunk["name"]),
        "physics_seed": PHYSICS_SEED,
        "motion_keys": list(chunk["motion_keys"]),
        "motion_pkl_sha256": motion_pkl_sha256,
    }
    cal._write_json(attempt / "process_result.json", payload)
    return {**payload, "file_sha256": _sha256(attempt / "process_result.json")}


def _validate_process_result(
    attempt: Path, *, chunk: Mapping[str, Any], motion_pkl_sha256: str,
) -> dict[str, Any]:
    path = attempt / "process_result.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeAbort(
            f"valid SONIC artifacts lack durable return-code evidence: {attempt}"
        ) from exc
    if not isinstance(payload, dict):
        raise BridgeAbort(f"SONIC process-result receipt is not an object: {attempt}")
    expected = {
        "schema": PROCESS_RESULT_SCHEMA,
        "returncode_observed": True,
        "sonic_log_sha256": _sha256(attempt / "sonic.log"),
        "chunk": str(chunk["name"]),
        "physics_seed": PHYSICS_SEED,
        "motion_keys": list(chunk["motion_keys"]),
        "motion_pkl_sha256": motion_pkl_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise BridgeAbort(
                f"SONIC process-result receipt has mismatched {field}: {attempt}"
            )
    returncode = payload.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise BridgeAbort(f"SONIC process-result return code is invalid: {attempt}")
    if returncode != 0:
        raise BridgeAbort(f"SONIC process returned {returncode}: {attempt}")
    return {**payload, "file_sha256": _sha256(path)}


def run_or_resume_launch(
    chunk: Mapping[str, Any], pkl: Path, output: Path, *,
    launch_fn: Callable[..., tuple[int, str]], timeout_s: int,
) -> tuple[dict[str, Any], list[Any]]:
    chunk_dir = output / "launches" / str(chunk["name"])
    attempts = sorted(path for path in chunk_dir.glob("attempt-*") if path.is_dir())
    pkl_sha256 = _sha256(pkl)
    if pkl_sha256 is None:
        raise BridgeAbort(f"SONIC motion pickle disappeared: {pkl}")

    # Audit every attempt receipt before considering any artifact for adoption.  In
    # particular, a valid earlier attempt must never hide a later recorded failure.
    attempt_infos: list[tuple[Path, dict[str, Any] | None]] = []
    for attempt in attempts:
        attempt_receipt_path = attempt / "receipt.json"
        attempt_receipt = None
        if attempt_receipt_path.is_file():
            try:
                attempt_receipt = json.loads(attempt_receipt_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise BridgeAbort(
                    f"unreadable attempt receipt requires inspection: {attempt}") from exc
            if not isinstance(attempt_receipt, dict):
                raise BridgeAbort(f"attempt receipt is not an object: {attempt}")
        elif any(attempt.iterdir()):
            raise BridgeAbort(
                f"SONIC attempt has artifacts but no pre-launch receipt: {attempt}")
        if attempt_receipt and attempt_receipt.get("status") == "failed":
            raise BridgeAbort(
                f"{chunk['name']} has a recorded failed SONIC attempt; use a fresh campaign output")
        if attempt_receipt is not None:
            if attempt_receipt.get("status") not in {"running", "complete"}:
                raise BridgeAbort(f"attempt has an unknown status: {attempt}")
            expected_receipt_fields = {
                "chunk": chunk["name"],
                "physics_seed": PHYSICS_SEED,
                "motion_keys": list(chunk["motion_keys"]),
                "motion_pkl_sha256": pkl_sha256,
            }
            for field, expected in expected_receipt_fields.items():
                if attempt_receipt.get(field) != expected:
                    raise BridgeAbort(
                        f"attempt {attempt} has mismatched {field}; refusing provenance relabel")
        attempt_infos.append((attempt, attempt_receipt))

    valid_candidates: list[tuple[Path, dict[str, Any], dict[str, Any], list[Any]]] = []
    for attempt, attempt_receipt in attempt_infos:
        if attempt_receipt is None:
            continue
        process_result = None
        if (attempt / "process_result.json").is_file():
            process_result = _validate_process_result(
                attempt, chunk=chunk, motion_pkl_sha256=pkl_sha256)
        try:
            record, rollouts = validate_attempt(attempt, chunk["motion_keys"])
        except (OSError, ValueError) as exc:
            if attempt_receipt.get("status") == "complete" or process_result is not None:
                raise BridgeAbort(f"completed attempt is now invalid: {attempt}")
            continue
        if process_result is None:
            raise BridgeAbort(
                f"valid SONIC artifacts lack durable return-code evidence: {attempt}")
        if attempt_receipt.get("status") == "complete":
            if (attempt_receipt.get("returncode_observed") is not True
                    or attempt_receipt.get("returncode") != 0):
                raise BridgeAbort(f"completed attempt has invalid return-code evidence: {attempt}")
            if attempt_receipt.get("process_result") != process_result:
                raise BridgeAbort(f"completed attempt changed its process result: {attempt}")
            for field in (
                "artifacts", "archive_schema_version", "sample_dt_s",
                "motion_id_key_map_sha256", "n_rollouts",
            ):
                if attempt_receipt.get(field) != record.get(field):
                    raise BridgeAbort(f"completed attempt changed its {field}: {attempt}")
        valid_candidates.append((attempt, record, process_result, rollouts))

    if len(valid_candidates) > 1:
        raise BridgeAbort(
            f"{chunk['name']} has multiple complete SONIC attempts; refusing ambiguous evidence")
    if valid_candidates:
        attempt, record, process_result, rollouts = valid_candidates[0]
        record["recovered_or_resumed"] = True
        record.update({
            "status": "complete", "chunk": chunk["name"],
            "physics_seed": PHYSICS_SEED, "motion_keys": list(chunk["motion_keys"]),
            "motion_pkl_sha256": pkl_sha256,
            "returncode": 0,
            "returncode_observed": True,
            "process_result": process_result,
        })
        cal._write_json(attempt / "receipt.json", record)
        record["attempt_receipt_sha256"] = _sha256(attempt / "receipt.json")
        return record, rollouts

    attempt_index = len(attempts)
    attempt = chunk_dir / f"attempt-{attempt_index:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cal._write_json(attempt / "receipt.json", {
        "status": "running", "chunk": chunk["name"], "physics_seed": PHYSICS_SEED,
        "motion_keys": list(chunk["motion_keys"]), "motion_pkl_sha256": pkl_sha256,
    })
    try:
        rc, log = launch_fn(
            pkl, attempt / "eval", int(chunk["n_motions"]), PHYSICS_SEED, timeout_s)
        _atomic_text(attempt / "sonic.log", log)
        process_result = _write_process_result(
            attempt, returncode=rc, log_path=attempt / "sonic.log",
            chunk=chunk, motion_pkl_sha256=pkl_sha256,
        )
        if rc != 0:
            raise RuntimeError(f"SONIC returned {rc}")
        record, rollouts = validate_attempt(attempt, chunk["motion_keys"])
        record.update({
            "status": "complete", "returncode": rc,
            "elapsed_s": float(time.monotonic() - started),
            "motion_pkl_sha256": pkl_sha256, "recovered_or_resumed": False,
            "returncode_observed": True,
            "process_result": process_result,
            "chunk": chunk["name"], "physics_seed": PHYSICS_SEED,
            "motion_keys": list(chunk["motion_keys"]),
        })
        cal._write_json(attempt / "receipt.json", record)
        record["attempt_receipt_sha256"] = _sha256(attempt / "receipt.json")
        return record, rollouts
    except Exception as exc:
        cal._write_json(attempt / "receipt.json", {
            "status": "failed", "chunk": chunk["name"],
            "physics_seed": PHYSICS_SEED, "motion_keys": list(chunk["motion_keys"]),
            "motion_pkl_sha256": pkl_sha256, "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_s": float(time.monotonic() - started),
        })
        raise BridgeAbort(f"SONIC launch {chunk['name']} failed: {exc}") from exc


def _persist(
    output: Path, receipt: dict[str, Any], reference_rows: Sequence[Mapping[str, Any]],
    achieved_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], started: float,
) -> None:
    cal._write_jsonl(output / "reference_rows.jsonl", reference_rows)
    cal._write_jsonl(output / "achieved_rows.jsonl", achieved_rows)
    cal._write_json(output / "summary.json", summary)
    receipt["wall_clock_s"] = float(time.monotonic() - started)
    receipt["evidence_anchors"] = {
        "reference_rows": {
            "n_rows": len(reference_rows),
            "logical_sha256": cal._json_hash(reference_rows),
            "file_sha256": _sha256(output / "reference_rows.jsonl"),
        },
        "achieved_rows": {
            "n_rows": len(achieved_rows),
            "logical_sha256": cal._json_hash(achieved_rows),
            "file_sha256": _sha256(output / "achieved_rows.jsonl"),
        },
        "summary": {
            "logical_sha256": cal._json_hash(summary),
            "file_sha256": _sha256(output / "summary.json"),
        },
    }
    cal._write_json(output / "receipt.json", receipt)


def validate_completed_output(
    output: Path, receipt: Mapping[str, Any], plan: Sequence[Mapping[str, Any]],
) -> None:
    """Revalidate a completed campaign before an idempotent resume returns it."""
    anchors = receipt.get("evidence_anchors", {})
    reference_rows = _read_jsonl(output / "reference_rows.jsonl")
    achieved_rows = _read_jsonl(output / "achieved_rows.jsonl")
    try:
        summary = json.loads((output / "summary.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeAbort(f"completed summary is unreadable: {exc}") from exc
    checks = (
        ("reference_rows", reference_rows, output / "reference_rows.jsonl", 128),
        ("achieved_rows", achieved_rows, output / "achieved_rows.jsonl", 128),
    )
    for name, rows, path, expected_n in checks:
        anchor = anchors.get(name, {})
        if (
            len(rows) != expected_n
            or anchor.get("n_rows") != expected_n
            or cal._json_hash(rows) != anchor.get("logical_sha256")
            or _sha256(path) != anchor.get("file_sha256")
        ):
            raise BridgeAbort(f"completed {name} no longer matches its evidence anchor")
    summary_anchor = anchors.get("summary", {})
    if (
        cal._json_hash(summary) != summary_anchor.get("logical_sha256")
        or _sha256(output / "summary.json") != summary_anchor.get("file_sha256")
        or summary.get("status") != "complete"
        or receipt.get("summary") != summary
    ):
        raise BridgeAbort("completed summary no longer matches its evidence anchor")
    launches = receipt.get("launches", {})
    for chunk in plan:
        chunk_dir = output / "launches" / str(chunk["name"])
        record = launches.get(chunk["name"])
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise BridgeAbort(f"completed receipt lacks launch {chunk['name']}")
        pkl = chunk_dir / "motions.pkl"
        _validate_motion_pkl(pkl, chunk["motion_keys"])
        pkl_sha256 = _sha256(pkl)
        if pkl_sha256 != record.get("motion_pkl_sha256"):
            raise BridgeAbort(f"completed motion pickle changed for {chunk['name']}")
        attempt = Path(record.get("attempt", "")).resolve()
        attempt_dirs = sorted(path.resolve() for path in chunk_dir.glob("attempt-*")
                              if path.is_dir())
        for candidate in attempt_dirs:
            candidate_receipt_path = candidate / "receipt.json"
            if not candidate_receipt_path.is_file():
                if any(candidate.iterdir()):
                    raise BridgeAbort(
                        f"completed campaign has an unreceipted attempt: {candidate}")
                continue
            try:
                candidate_receipt = json.loads(candidate_receipt_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise BridgeAbort(
                    f"completed campaign has an unreadable attempt receipt: {candidate}"
                ) from exc
            if not isinstance(candidate_receipt, dict):
                raise BridgeAbort(
                    f"completed campaign has a malformed attempt receipt: {candidate}")
            if candidate_receipt.get("status") == "failed":
                raise BridgeAbort(
                    f"completed campaign contains a recorded failed attempt: {candidate}")
            if candidate_receipt.get("status") not in {"running", "complete"}:
                raise BridgeAbort(
                    f"completed campaign contains an unknown attempt status: {candidate}")
            if candidate != attempt and (
                    candidate_receipt.get("status") == "complete"
                    or (candidate / "process_result.json").is_file()):
                raise BridgeAbort(
                    f"completed campaign contains an additional finished attempt: {candidate}")
            candidate_expected = {
                "chunk": chunk["name"],
                "physics_seed": PHYSICS_SEED,
                "motion_keys": list(chunk["motion_keys"]),
                "motion_pkl_sha256": pkl_sha256,
            }
            for field, expected in candidate_expected.items():
                if candidate_receipt.get(field) != expected:
                    raise BridgeAbort(
                        f"completed campaign attempt has mismatched {field}: {candidate}")
        if attempt not in attempt_dirs:
            raise BridgeAbort(f"completed launch points outside its attempt set: {chunk['name']}")
        expected_record_fields = {
            "chunk": chunk["name"],
            "physics_seed": PHYSICS_SEED,
            "motion_keys": list(chunk["motion_keys"]),
            "motion_pkl_sha256": pkl_sha256,
            "returncode": 0,
            "returncode_observed": True,
        }
        for field, expected in expected_record_fields.items():
            if record.get(field) != expected:
                raise BridgeAbort(
                    f"completed launch has mismatched {field}: {chunk['name']}")
        process_result = _validate_process_result(
            attempt, chunk=chunk, motion_pkl_sha256=str(pkl_sha256))
        if record.get("process_result") != process_result:
            raise BridgeAbort(f"completed process result changed for {chunk['name']}")
        validated, _ = validate_attempt(attempt, chunk["motion_keys"])
        if (
            validated["artifacts"] != record.get("artifacts")
            or validated["archive_schema_version"] != ARCHIVE_SCHEMA_VERSION
            or validated["sample_dt_s"] != EXPECTED_SAMPLE_DT_S
            or validated["motion_id_key_map_sha256"]
            != record.get("motion_id_key_map_sha256")
            or _sha256(attempt / "receipt.json")
            != record.get("attempt_receipt_sha256")
        ):
            raise BridgeAbort(f"completed SONIC artifacts changed for {chunk['name']}")


def run_bridge(
    *,
    out: str | Path = DEFAULT_OUT,
    source_dir: str | Path = SOURCE_OUT,
    dry_run: bool = False,
    timeout_s: int = 2400,
    launch_fn: Callable[..., tuple[int, str]] = exp1b.run_sonic,
    export_fn: Callable[..., Path] = write_motion_pkl,
    scorer: Callable[..., Mapping[str, Any]] = score_trajectory,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    tracker_identity_fn: Callable[[], Mapping[str, Any]] = tracker_identity,
    mj_model: Any = None,
) -> dict[str, Any]:
    source = load_source_bundle(source_dir)
    current_project = project_identity(code_state_fn=code_state_fn)
    tracker = dict(tracker_identity_fn())
    plan = chunk_plan()
    output = Path(out)
    if dry_run:
        campaign_identity = cal._json_hash({
            "schema": SCHEMA_VERSION, "project": current_project,
            "source": source["identity"], "tracker": tracker, "plan": plan,
            "obstacles": list(OBSTACLES), "heights": list(GRADED_HEIGHTS_M),
            "obstacle_half_width_m": OBSTACLE_HALF_WIDTH_M,
        })
        return {
            "schema": SCHEMA_VERSION,
            "experiment": "exp022_exact_tracking_bridge",
            "status": "dry_run",
            "writes_performed": False,
            "project_dirty_observed": current_project["git"].get("dirty"),
            "source": source["identity"],
            "tracker": tracker,
            "launch_plan": plan,
            "campaign_identity_sha256": campaign_identity,
        }
    if output.exists() and not output.is_dir():
        raise BridgeAbort(f"campaign output exists but is not a directory: {output}")
    existing_receipt = output / "receipt.json"
    if output.exists() and any(output.iterdir()) and not existing_receipt.is_file():
        raise BridgeAbort(f"refusing nonempty output without a resumable receipt: {output}")
    old: dict[str, Any] | None = None
    resume_project_check: dict[str, Any] | None = None
    if existing_receipt.is_file():
        try:
            old = json.loads(existing_receipt.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise BridgeAbort(f"existing campaign receipt is unreadable: {exc}") from exc
        if not isinstance(old, dict):
            raise BridgeAbort("existing campaign receipt is not an object")
        if old.get("status") == "blocked":
            raise BridgeAbort(
                "existing EXP-022A campaign is blocked; preserve it and use fresh output")
        if old.get("schema") != SCHEMA_VERSION or old.get("status") not in {"running", "complete"}:
            raise BridgeAbort("existing output is not a resumable EXP-022A campaign")
        pinned_project = old.get("provenance", {}).get("project")
        if (not isinstance(pinned_project, dict)
                or pinned_project.get("git", {}).get("dirty") is not False):
            raise BridgeAbort("existing receipt lacks a clean pinned Scene2Motion identity")
        try:
            resume_project_check = validate_project_recheck(
                pinned_project, current_project, output)
        except ValueError as exc:
            raise BridgeAbort(str(exc)) from exc
        project = dict(pinned_project)
    else:
        if current_project["git"].get("dirty") is not False:
            raise BridgeAbort("EXP-022A requires an exactly clean Scene2Motion worktree")
        project = current_project

    campaign_identity = cal._json_hash({
        "schema": SCHEMA_VERSION, "project": project,
        "source": source["identity"], "tracker": tracker, "plan": plan,
        "obstacles": list(OBSTACLES), "heights": list(GRADED_HEIGHTS_M),
        "obstacle_half_width_m": OBSTACLE_HALF_WIDTH_M,
    })
    if old is not None:
        if old.get("campaign_identity_sha256") != campaign_identity:
            raise BridgeAbort("existing EXP-022A output has a different campaign identity")
        if old.get("status") == "complete":
            validate_completed_output(output, old, plan)
            return old
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reference_rows: list[dict[str, Any]] = []
    achieved_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "status": "reference_pending",
        "interpretation_guard": (
            "No obstacle is present in Isaac; no contact-rich execution claim is licensed."
        ),
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp022_exact_tracking_bridge",
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": "preflight",
        "campaign_identity_sha256": campaign_identity,
        "resume_supported": True,
        "claim_scope": "post-hoc achieved-state bridge over the complete archived EXP-021 pool",
        "not_fresh_seed_evidence": True,
        "not_contact_rich_execution": True,
        "actual_ardy_samples": 0,
        "reused_archived_ardy_samples": 64,
        "design": {
            "obstacles": [{"label": label, "x_m": x} for label, x in OBSTACLES],
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "obstacle_half_width_m": OBSTACLE_HALF_WIDTH_M,
            "graded_heights_m": list(GRADED_HEIGHTS_M),
            "endpoint": (
                "exact fixed-centre BoxHeightProbe, never +/- radius; achieved retention "
                "requires non-termination, in-corridor passage, and finishing beyond the box"
            ),
            "physics_seed": PHYSICS_SEED,
            "launch_plan": plan,
            "all_archived_references_tracked": True,
        },
        "provenance": {"project": project, "source_exp021": source["identity"],
                       "tracker": tracker},
        "resume_project_check": resume_project_check,
        "post_launch_revalidation": {},
        "launches": {},
    }

    try:
        source_rows = {int(row["seed"]): row for row in source["rows"]}
        reference_rows = build_rows(
            source["clips"], source_rows=source_rows, tier="reference", scorer=scorer)
        if len(reference_rows) != 128:
            raise ValueError("reference scoring did not produce 64 x 2 exact-centre rows")
        summary = {
            "status": "reference_complete",
            "reference": summarize(reference_rows, {})["reference"],
            "interpretation_guard": receipt["claim_scope"],
        }
        receipt["stage"] = "reference_scored_durable"
        _persist(output, receipt, reference_rows, achieved_rows, summary, started)

        model = mj_model if mj_model is not None else G1Body(None).model
        achieved_by_key: dict[str, Any] = {}
        schema_by_key: dict[str, int] = {}
        dt_by_key: dict[str, float] = {}
        for chunk in plan:
            receipt["stage"] = f"launching_{chunk['name']}"
            _persist(output, receipt, reference_rows, achieved_rows, summary, started)
            pkl = ensure_motion_pkl(
                chunk, source["clips"], output, export_fn=export_fn, mj_model=model)
            launch_record, rollouts = run_or_resume_launch(
                chunk, pkl, output, launch_fn=launch_fn, timeout_s=timeout_s)
            receipt["launches"][chunk["name"]] = launch_record
            for rollout in rollouts:
                achieved_by_key[rollout.motion_key] = rollout
                schema_by_key[rollout.motion_key] = int(launch_record["archive_schema_version"])
                dt_by_key[rollout.motion_key] = float(launch_record["sample_dt_s"])

            # Revalidate every bound input after external execution before accepting its output.
            if load_source_bundle(source_dir)["identity"] != source["identity"]:
                raise ValueError("EXP-021 source artifacts changed during SONIC execution")
            if dict(tracker_identity_fn()) != tracker:
                raise ValueError("SONIC checkout/checkpoint/config changed during execution")
            current_project = project_identity(code_state_fn=code_state_fn)
            git_check = validate_project_recheck(project, current_project, output)
            receipt["post_launch_revalidation"][chunk["name"]] = {
                "source_exp021_unchanged": True,
                "tracker_identity_unchanged": True,
                "project": git_check,
            }

            # Persist all complete chunks; an interruption never discards prior achieved states.
            subset = {}
            for key, rollout in achieved_by_key.items():
                subset[key] = rollout
            partial_rows = []
            for seed in POOL_SEEDS:
                key = f"s{seed}"
                if key not in subset:
                    continue
                for label, x in OBSTACLES:
                    rollout = subset[key]
                    metrics = dict(scorer(
                        rollout.qpos, x, terminated=rollout.terminated,
                        reported_progress=rollout.progress))
                    partial_rows.append({
                        "tier": "achieved", "seed": seed, "motion_key": key,
                        "motion_id": int(rollout.motion_id), "obstacle_label": label,
                        "obstacle_x_m": x, "obstacle_depth_m": OBSTACLE_DEPTH_M,
                        "source_qpos_content_sha256": source_rows[seed]["qpos_content_sha256"],
                        "archive_schema_version": schema_by_key[key],
                        "sample_dt_s": dt_by_key[key], **metrics,
                    })
            achieved_rows = partial_rows
            summary = {
                "status": "sonic_partial",
                "n_achieved_motions": len(achieved_by_key),
                "interpretation_guard": receipt["claim_scope"],
            }
            _persist(output, receipt, reference_rows, achieved_rows, summary, started)

        if set(achieved_by_key) != {f"s{seed}" for seed in POOL_SEEDS}:
            raise ValueError("completed launches do not cover all 64 archived motions")
        # Rebuild in one locked order and summarize only after both chunks are complete.
        achieved_rows = []
        for seed in POOL_SEEDS:
            key = f"s{seed}"
            rollout = achieved_by_key[key]
            for label, x in OBSTACLES:
                metrics = dict(scorer(
                    rollout.qpos, x, terminated=rollout.terminated,
                    reported_progress=rollout.progress))
                achieved_rows.append({
                    "tier": "achieved", "seed": seed, "motion_key": key,
                    "motion_id": int(rollout.motion_id), "obstacle_label": label,
                    "obstacle_x_m": x, "obstacle_depth_m": OBSTACLE_DEPTH_M,
                    "source_qpos_content_sha256": source_rows[seed]["qpos_content_sha256"],
                    "archive_schema_version": schema_by_key[key],
                    "sample_dt_s": dt_by_key[key], **metrics,
                })
        summary = summarize(reference_rows, achieved_rows)
        summary["status"] = "complete"
        receipt.update({
            "status": "complete", "complete": True, "blocked": False, "stage": "complete",
            "sonic_rollouts_requested": 64, "sonic_rollouts_returned": len(achieved_by_key),
            "summary": summary,
        })
        _persist(output, receipt, reference_rows, achieved_rows, summary, started)
        return receipt
    except Exception as exc:
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
            "blocked": True, "failed_stage": receipt.get("stage"),
            "error_type": type(exc).__name__, "error": str(exc),
            "sonic_rollouts_returned_lower_bound": sum(
                int(record.get("n_rollouts", 0))
                for record in receipt.get("launches", {}).values()
                if record.get("status") == "complete"),
        })
        _persist(output, receipt, reference_rows, achieved_rows, summary, started)
        if isinstance(exc, BridgeAbort):
            raise
        raise BridgeAbort(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--source", default=str(SOURCE_OUT))
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    receipt = run_bridge(
        out=args.out, source_dir=args.source, dry_run=args.dry_run,
        timeout_s=args.timeout_s)
    print(json.dumps({
        "status": receipt["status"],
        "writes_performed": receipt.get("writes_performed", not args.dry_run),
        "launches": len(receipt.get(
            "launch_plan", receipt.get("design", {}).get("launch_plan", []))),
    }, indent=2))


if __name__ == "__main__":
    main()
