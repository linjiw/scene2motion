"""EXP-024: reference-contract ablation of the prompt-elicited step, with a prospective gate test.

Protocol: ``docs/ramp-exp024-reference-contract-protocol.md`` (its sha256 is bound into the
receipt before the first sample).  Four native root contracts of the STEP prompt on the fixed
7.2 m reference-speed route -- ``free`` (route ``root_xz`` only; the exp021/exp023 contract),
``pin_y`` (root height pinned at 0.78 m), ``pin_h`` (heading pinned at 0) and ``pin_yh`` (both)
-- on 32 fresh seeds (4600-4631), generated in sixteen B=8 calls of 2 seeds x 4 arms so every
same-seed comparison shares one call and identical per-sample noise.  Every clip is scored on
CPU with planned denominators (elicitation, exact fixed-centre clearance at 1.2 m and 3.6 m,
the trackability-contract features and the calibrated 0.20 s gate prediction, the 13-gate local
step, route fidelity and the pinned-arm manipulation check), the per-clip gate predictions are
written and hashed *before* any tracker launch, all 128 clips are tracked by SONIC under the
release evaluator (four self-contained launches of 32), and the preregistered predictions P1-P4
are evaluated mechanically.

Stages (each resumable, each a separate process so ARDY is never resident beside Isaac)::

    --stage generate   GPU: 16 B=8 calls; persists the empty ledger before the runner exists
    --stage score      CPU: reference endpoints for all 128 clips (rows written before SONIC)
    --stage predict    CPU: predictions.jsonl (0.20 s primary, 0.32 s secondary) + its hash
    --stage sonic      Isaac: 4 launches of 32 through the EXP-022A bridge harness
    --stage analyze    CPU: guarded retention, prospective 2x2, P1-P4, decision rules
    --stage all        the five in order; the SONIC stage is re-invoked as a subprocess when a
                       CUDA context is alive in this process

The obstacle is absent from Isaac in every SONIC path: "retained" means the achieved states
replayed against the collision model with EXP-022A's guarded endpoint, never contact-rich
execution.  "Terminated" is the evaluator's cutoff, not a measured fall.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analyze_trackability_contract as atc  # noqa: E402
from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp017_ramp_residual_stepover as e17  # noqa: E402
from experiments import exp019_ramp_gait_matched_stepover as e19  # noqa: E402
from experiments import exp021_elicited_lift_distribution as e21  # noqa: E402
from experiments import exp022_exact_tracking_bridge as e22  # noqa: E402
from experiments import exp023_prompt_handoff as e23  # noqa: E402
from experiments import exp1b_execution_clearance as exp1b  # noqa: E402
from experiments.analyze_e1a_placement import (  # noqa: E402
    box_height_profile,
    lift_location,
    lift_side,
)
from scene2motion.constraints import ArdyConstraintSet, ConstraintSpec  # noqa: E402
from scene2motion.host_gate import (  # noqa: E402
    HostResourceGateFailed,
    host_resource_report,
    require_host_resources,
)
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import QPOS_WIDTH  # noqa: E402
from scene2motion.stepover_eval import (  # noqa: E402
    StepOverThresholds,
    evaluate_local_step,
    foot_kinematics_series,
    step_scene,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/exp024_reference_contract"
SCHEMA_VERSION = "exp024-reference-contract-v1"
FAILURE_SCHEMA_VERSION = "exp024-reference-contract-failure-v1"
EXPERIMENT = "exp024_reference_contract"
PROTOCOL_PATH = "docs/ramp-exp024-reference-contract-protocol.md"

STEP = e23.STEP
FPS = float(cal.FPS)
N_FRAMES = int(cal.N_FRAMES)
DIFFUSION_STEPS = int(cal.DIFFUSION_STEPS)
CFG_WEIGHT = tuple(float(w) for w in cal.CFG_WEIGHT)
NOISE_STREAM_VERSION = int(cal.NOISE_STREAM_VERSION)
FIRST_HEADING = 0.0

SEEDS = tuple(range(4600, 4632))
ARMS = ("free", "pin_y", "pin_h", "pin_yh")
PIN_ROOT_Y_M = 0.78
PIN_HEADING_RAD = 0.0
ARM_CONTRACTS: Mapping[str, Mapping[str, float | None]] = {
    "free": {"root_y": None, "heading": None},
    "pin_y": {"root_y": PIN_ROOT_Y_M, "heading": None},
    "pin_h": {"root_y": None, "heading": PIN_HEADING_RAD},
    "pin_yh": {"root_y": PIN_ROOT_Y_M, "heading": PIN_HEADING_RAD},
}
# ARDY's ``slice_dict`` names: ``root_pos`` is (x, y, z) per frame -- root_2d writes x and z,
# root_y_pos writes y -- and ``global_root_heading`` is (cos, sin) per frame.
EXPECTED_CHANNEL_USAGE: Mapping[str, Mapping[str, int]] = {
    "free": {"root_pos": 2 * N_FRAMES},
    "pin_y": {"root_pos": 3 * N_FRAMES},
    "pin_h": {"root_pos": 2 * N_FRAMES, "global_root_heading": 2 * N_FRAMES},
    "pin_yh": {"root_pos": 3 * N_FRAMES, "global_root_heading": 2 * N_FRAMES},
}
CHUNK_SEED_COUNT = 2
CHUNK_ROWS = CHUNK_SEED_COUNT * len(ARMS)
N_CHUNKS = len(SEEDS) // CHUNK_SEED_COUNT
N_ROWS = len(SEEDS) * len(ARMS)

OBSTACLE_DEPTH_M = float(e22.OBSTACLE_DEPTH_M)
OBSTACLES = tuple(e22.OBSTACLES)                      # (("staged", 1.2), ("unstaged", 3.6))
STAGED_LABEL, STAGED_X_M = OBSTACLES[0]
CONTROL_LABEL, CONTROL_X_M = OBSTACLES[1]
GRADED_HEIGHTS_M = tuple(e22.GRADED_HEIGHTS_M)
SCAN_POINTS = int(e21.SCAN_POINTS)
ELICITATION_MIN_M = 0.03
P4_BOX_HEIGHT_M = 0.05
LOCAL_STEP_OBSTACLE_HEIGHT_M = P4_BOX_HEIGHT_M
PRIMARY_GATE_S = float(atc.PRIMARY_GATE_S)
SECONDARY_GATE_S = float(atc.SECONDARY_GATE_S)
CONSTRUCTIBLE_ROOT_Z_RANGE_M = 0.10
CONSTRUCTIBLE_HEADING_RANGE_DEG = 10.0
# The 13 gates of ``stepover_eval.evaluate_local_step``, in its order.
LOCAL_STEP_GATE_NAMES = (
    "whole_body_collision_free", "root_traversal", "lateral_corridor",
    "both_feet_cross_before_over_after", "lead_trail_order",
    "lead_overlap_has_trailing_support", "trail_overlap_has_lead_support",
    "bounded_unsupported_run", "lead_landing_dwell", "trail_landing_dwell",
    "lead_lands_before_or_during_trailing_overlap", "both_feet_finish_beyond",
    "bounded_floor_penetration",
)

P1_MIN_FLAGGED_TERMINATED_RATE = 0.90
P1_MAX_PASSED_TERMINATED_RATE = 0.30
P1_MIN_AUC = 0.90
P2_ELICITATION_RANGE = (0.55, 0.95)
P2_EXACT_5CM_RANGE = (0.06, 0.35)
P3_MAX_MEDIAN_ROOT_Z_MAX_M = 0.85
P3_MIN_SHORTER_PAIRED_SEEDS = 20
P4_MIN_CLIPS = 3
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_RESAMPLES = 2000
EXP1B_TEST_RETEST_DISAGREEMENT = (51, 179)

PHYSICS_SEED = int(e22.PHYSICS_SEED)
SONIC_CHUNK_SIZE = int(e22.CHUNK_SIZE)
SEEDS_PER_LAUNCH = SONIC_CHUNK_SIZE // len(ARMS)
N_LAUNCHES = N_ROWS // SONIC_CHUNK_SIZE
EXPECTED_CORE_SOURCE_MANIFEST_SHA256 = (
    "44e98c45f840ed32cd54d0dbc322e4ed1ef1743625e70ca94c7af01eb70efe0a"
)
EXPECTED_TRACKER_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
TERMINATIONS_OVERRIDE = "+manager_env/terminations=tracking/eval"
EVAL_TERMINATIONS_YAML = "gear_sonic/config/manager_env/terminations/tracking/eval.yaml"
# Release evaluator: ``tracking/eval`` merged over ``sonic_release/config.yaml`` (plan of record
# section 6; the checkpoint contributes ``foot_pos_xyz``).  The resolved dump must show these.
RELEASE_EVALUATOR_THRESHOLDS: Mapping[str, Mapping[str, Any]] = {
    "anchor_pos": {"threshold": 0.25, "threshold_adaptive": False, "down_threshold": 0.25},
    "anchor_ori_full": {"threshold": 1.0},
    "ee_body_pos": {"threshold": 0.25, "threshold_adaptive": False, "down_threshold": 0.25},
    "foot_pos_xyz": {"threshold": 0.2},
}
# The time-out term is defined in ``terminations/terms/motion_time_out.yaml`` but its config key
# (and hence the resolved-config key) is ``time_out``; EXP-028 dumps it under that name.
TIME_OUT_TERM = "time_out"
THRESHOLD_RECEIPT_PATH = ROOT / "outputs/exp016_threshold_calibration/receipt.json"
THRESHOLD_RECEIPT_SHA256 = cal.PHYSICAL_THRESHOLD_RECEIPT_FILE_SHA256

STAGES = ("generate", "score", "predict", "sonic", "analyze")
SOURCE_FILES = (
    PROTOCOL_PATH,
    "env.sh",
    "experiments/exp024_reference_contract.py",
    "experiments/analyze_trackability_contract.py",
    "experiments/analyze_e1a_placement.py",
    "experiments/calibrate_ramp_route_phase.py",
    "experiments/exp017_ramp_residual_stepover.py",
    "experiments/exp019_ramp_gait_matched_stepover.py",
    "experiments/exp021_elicited_lift_distribution.py",
    "experiments/exp022_exact_tracking_bridge.py",
    "experiments/exp023_prompt_handoff.py",
    "experiments/exp1b_execution_clearance.py",
    "scene2motion/constraints.py",
    "scene2motion/host_gate.py",
    "scene2motion/robot.py",
    "scene2motion/runner.py",
    "scene2motion/scenes.py",
    "scene2motion/sonic_export.py",
    "scene2motion/sonic_state_export.py",
    "scene2motion/stepover_eval.py",
)

if OBSTACLE_DEPTH_M != float(e19.OBSTACLE_DEPTH_M):
    raise RuntimeError("EXP-024 obstacle depth no longer matches the exp021 scan depth")
if (STAGED_LABEL, STAGED_X_M) != ("staged", 1.2) or CONTROL_X_M != 3.6:
    raise RuntimeError("EXP-022A obstacle centres drifted from the EXP-024 protocol")
# Both rules flag ``run > threshold``: primary > 0.20 s (>= 6 frames = 0.24 s at 25 fps),
# secondary > 0.28 s (>= 8 frames = 0.32 s at 25 fps); see analyze_trackability_contract.
if PRIMARY_GATE_S != 0.2 or SECONDARY_GATE_S != 0.28:
    raise RuntimeError("EXP-024 gate thresholds drifted from the protocol")
P1_STRONG_MIN_AUC = 0.95
if THRESHOLD_RECEIPT_SHA256 != atc.THRESHOLD_RECEIPT_SHA256:
    raise RuntimeError("the two locked threshold-receipt hashes disagree")


class CampaignAbort(RuntimeError):
    """Fail-closed stop after every available piece of evidence has been made durable."""


# --------------------------------------------------------------------------- locked plans


def locked_row_plan() -> list[dict[str, Any]]:
    """Chunk-major 128-row plan: chunk c holds seeds (4600+2c, 4601+2c) x the four arms."""
    rows: list[dict[str, Any]] = []
    for chunk in range(N_CHUNKS):
        seeds = SEEDS[chunk * CHUNK_SEED_COUNT:(chunk + 1) * CHUNK_SEED_COUNT]
        for local, (seed, arm) in enumerate((s, a) for s in seeds for a in ARMS):
            rows.append({
                "row_index": len(rows),
                "chunk": chunk,
                "chunk_name": f"chunk{chunk:02d}",
                "batch_position": local,
                "seed": int(seed),
                "arm": arm,
                "archive_key": f"s{seed}_{arm}",
                "prompt": STEP,
            })
    if len(rows) != N_ROWS:
        raise RuntimeError("locked row plan does not hold exactly 128 rows")
    return rows


def locked_chunk_plan(plan: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Sixteen B=8 generation calls; no same-seed comparison crosses a call."""
    rows = list(plan if plan is not None else locked_row_plan())
    chunks: list[dict[str, Any]] = []
    for chunk in range(N_CHUNKS):
        members = [row for row in rows if int(row["chunk"]) == chunk]
        seeds = list(dict.fromkeys(int(row["seed"]) for row in members))
        if len(members) != CHUNK_ROWS or len(seeds) != CHUNK_SEED_COUNT:
            raise RuntimeError(f"chunk {chunk} is not 2 seeds x 4 arms")
        if [row["arm"] for row in members] != list(ARMS) * CHUNK_SEED_COUNT:
            raise RuntimeError(f"chunk {chunk} arm order drifted")
        chunks.append({
            "chunk": chunk,
            "name": f"chunk{chunk:02d}",
            "seeds": seeds,
            "row_indices": [int(row["row_index"]) for row in members],
            "archive_keys": [str(row["archive_key"]) for row in members],
            "batch_size": len(members),
            "rows": members,
        })
    return chunks


def launch_plan(plan: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Four self-contained SONIC launches of 32, seed-block-major.

    Launch k tracks seeds 4600+8k .. 4607+8k under all four arms (generation chunks 4k..4k+3
    in row-plan order), so every within-seed arm comparison and every paired McNemar count
    sits inside one Isaac launch (one terrain instance, physics seed 0).
    """
    rows = list(plan if plan is not None else locked_row_plan())
    launches: list[dict[str, Any]] = []
    for index in range(N_LAUNCHES):
        seeds = list(SEEDS[index * SEEDS_PER_LAUNCH:(index + 1) * SEEDS_PER_LAUNCH])
        members = [row for row in rows if int(row["seed"]) in seeds]
        if len(members) != SONIC_CHUNK_SIZE:
            raise RuntimeError(f"launch {index} does not hold exactly 32 motions")
        launches.append({
            "launch": index,
            "name": f"launch{index:02d}_seed{PHYSICS_SEED}",
            "physics_seed": PHYSICS_SEED,
            "seeds": seeds,
            "generation_chunks": sorted({str(row["chunk_name"]) for row in members}),
            "motion_keys": [str(row["archive_key"]) for row in members],
            "n_motions": len(members),
        })
    return launches


def route_xz() -> np.ndarray:
    return cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)


def arm_spec(arm: str, route: np.ndarray) -> ConstraintSpec:
    """The four root contracts differ only in the dense root-height / heading channels."""
    contract = ARM_CONTRACTS[arm]
    n = len(route)
    return ConstraintSpec(
        root_xz=np.asarray(route, dtype=float),
        heading=(None if contract["heading"] is None
                 else np.full(n, float(contract["heading"]), dtype=float)),
        root_y=(None if contract["root_y"] is None
                else np.full(n, float(contract["root_y"]), dtype=float)),
        first_heading=FIRST_HEADING,
    )


def spec_sha256(spec: ConstraintSpec) -> str:
    fields = {
        "root_xz": cal._array_hash({"root_xz": np.asarray(spec.root_xz)}),
        "heading": (None if spec.heading is None
                    else cal._array_hash({"heading": np.asarray(spec.heading)})),
        "root_y": (None if spec.root_y is None
                   else cal._array_hash({"root_y": np.asarray(spec.root_y)})),
        "first_heading": spec.first_heading,
        "n_frames": int(spec.T),
    }
    return cal._json_hash(fields)


def static_channel_usage(spec: ConstraintSpec) -> dict[str, int]:
    """Which ARDY filler keys the adapter writes, counted without a model (CPU, no runner)."""
    data: dict[str, list] = {key: [] for key in (
        "root_2d", "global_root_heading", "root_y_pos",
        "global_joints_rots", "global_joints_positions")}
    index: dict[str, list] = {key: [] for key in data}
    ArdyConstraintSet(spec, root_idx=0, device="cpu").update_constraints(data, index)
    return {key: int(sum(len(entry) for entry in index[key]))
            for key in data if data[key]}


def _actual_channel_usage(runner: Any, spec: ConstraintSpec) -> dict[str, int]:
    return e17._actual_channel_usage(runner, spec)


# ------------------------------------------------------------------ identities and hashes


def _source_hashes(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        digest = cal._sha256(repo / relative)
        if digest is None:
            raise ValueError(f"required EXP-024 source is missing: {relative}")
        hashes[relative] = digest
    return hashes


def _step_prompt_cache_identity(runner: Any, cache_path: Path) -> dict[str, Any]:
    """Bind the exact cached STEP embedding in memory and on disk."""
    if not cache_path.is_file():
        raise ValueError("EXP-024 prompt cache is missing")
    key = hashlib.sha1(STEP.encode()).hexdigest()
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if key not in cache.files:
                raise ValueError(f"cached embedding is missing for {STEP!r}")
            file_value = np.array(cache[key], copy=True)
        memory_value = np.array(runner._text_cache[key], copy=True)
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid EXP-024 prompt cache: {exc}") from exc
    if (file_value.size == 0 or not np.isfinite(file_value).all()
            or not np.array_equal(file_value, memory_value)):
        raise ValueError("in-memory STEP embedding does not byte-match the cache")
    return cal._identity("exp024-step-prompt-cache-v1", {
        "path": str(cache_path),
        "file_sha256": cal._sha256(cache_path),
        "prompt": STEP,
        "cache_key_sha1": key,
        "content_sha256": cal._array_hash({key: file_value}),
        "shape": list(file_value.shape),
        "dtype": str(file_value.dtype),
    })


def _validate_pins(generator: Mapping[str, Any], runtime: Mapping[str, Any],
                   physical_model: Mapping[str, Any]) -> None:
    checkpoint = generator.get("checkpoint", {})
    if checkpoint.get("hf_revision") != e23.PINNED_HF_REVISION:
        raise ValueError("EXP-024 loaded the wrong ARDY checkpoint revision")
    if checkpoint.get("checkpoint_sha256") != e23.PINNED_DENOISER_SHA256:
        raise ValueError("EXP-024 loaded the wrong ARDY denoiser bytes")
    if checkpoint.get("model_name") != "ARDY-G1-RP-25FPS-Horizon52":
        raise ValueError("EXP-024 requires ARDY-G1-RP-25FPS-Horizon52")
    runtime_fields = runtime.get("fields", {})
    if runtime_fields.get("ardy_git_commit") != e23.PINNED_ARDY_COMMIT:
        raise ValueError("EXP-024 loaded the wrong ARDY runtime commit")
    if runtime_fields.get("ardy_tracked_status") != []:
        raise ValueError("EXP-024 requires a clean tracked ARDY checkout")
    if physical_model.get("fields", {}).get("sha256") != e23.PINNED_G1_XML_SHA256:
        raise ValueError("EXP-024 loaded the wrong released G1 XML")


def _threshold_dependency() -> tuple[StepOverThresholds, dict[str, Any]]:
    """The calibrated exp016 thresholds through the locked dependency loader, cross-checked
    against the contract analyser's own reading of the same receipt."""
    thresholds, dependency = cal._load_physical_threshold_dependency(THRESHOLD_RECEIPT_PATH)
    support = atc.load_support_thresholds(THRESHOLD_RECEIPT_PATH, THRESHOLD_RECEIPT_SHA256)
    if (support["support_height_m"] != thresholds.support_height_m
            or support["support_speed_mps"] != thresholds.support_speed_mps
            or support["max_unsupported_run_s"] != thresholds.max_unsupported_run_s):
        raise ValueError("contract-analyser and dependency-loader thresholds disagree")
    if thresholds.max_unsupported_run_s != PRIMARY_GATE_S:
        raise ValueError("the calibrated gate is no longer the 0.20 s primary rule")
    dependency = dict(dependency)
    dependency["support_thresholds"] = support
    return thresholds, dependency


def _verify_stage_git_state(pinned: Mapping[str, Any], current: Mapping[str, Any],
                            *, repo: Path, output: Path) -> dict[str, Any]:
    """Later stages may run after new commits (the predictions file is committed before
    SONIC), so require an unchanged tracked diff, no worktree change outside the campaign
    output, and record any commit drift; source content is bound separately by hash."""
    if current.get("tracked_diff_sha256") != pinned.get("tracked_diff_sha256"):
        raise ValueError("tracked git diff changed since the campaign was pinned")
    try:
        relative_output = output.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        relative_output = None
    unexpected: list[str] = []
    allowed: list[str] = []
    for line in current.get("status", []):
        path = line[3:] if len(line) >= 4 else ""
        if relative_output is not None and (
                path == relative_output or path.startswith(relative_output + "/")):
            allowed.append(line)
        else:
            unexpected.append(line)
    if unexpected:
        raise ValueError("worktree changed outside the campaign output: " + "; ".join(unexpected))
    drift: list[str] = []
    if current.get("commit") != pinned.get("commit"):
        try:
            drift = subprocess.check_output(
                ["git", "diff", "--name-only", f"{pinned['commit']}..{current['commit']}"],
                cwd=repo, text=True, stderr=subprocess.DEVNULL).splitlines()
        except (OSError, subprocess.CalledProcessError, KeyError):
            drift = ["<unavailable>"]
    return {
        "pinned_commit": pinned.get("commit"),
        "current_commit": current.get("commit"),
        "commit_changed": current.get("commit") != pinned.get("commit"),
        "paths_changed_between_commits": drift,
        "tracked_diff_unchanged": True,
        "allowed_output_status": allowed,
        "unexpected_status": unexpected,
    }


def committed_blob_check(repo: Path, relative_path: str) -> dict[str, Any]:
    """Does the committed HEAD blob of ``relative_path`` equal the working file, byte for byte?"""
    working = repo / relative_path
    working_sha = cal._sha256(working)
    result: dict[str, Any] = {
        "path": relative_path, "working_file_sha256": working_sha,
        "head_commit": None, "committed_blob_sha256": None, "worktree_status": None,
        "matches": False, "error": None,
    }
    try:
        result["head_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL).strip()
        blob = subprocess.check_output(
            ["git", "show", f"HEAD:{relative_path}"], cwd=repo, stderr=subprocess.DEVNULL)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", relative_path], cwd=repo, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["error"] = f"not committed at HEAD: {exc}"
        return result
    result["committed_blob_sha256"] = hashlib.sha256(blob).hexdigest()
    result["worktree_status"] = status
    result["matches"] = bool(
        working_sha is not None
        and result["committed_blob_sha256"] == working_sha
        and status == "")
    return result


def cuda_context_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_importable": False, "cuda_initialized": False}
    return {"torch_importable": True, "cuda_initialized": bool(torch.cuda.is_initialized())}


def _release_runner(runner: Any) -> dict[str, Any]:
    """Drop the ARDY runner and return its GPU memory before Isaac could ever be launched."""
    del runner
    gc.collect()
    report: dict[str, Any] = {"gc_collected": True, "cuda_cache_emptied": False}
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            report["cuda_cache_emptied"] = True
            report["cuda_memory_allocated_bytes"] = int(torch.cuda.memory_allocated())
    except Exception as exc:  # pragma: no cover - only reachable on a broken CUDA runtime
        report["error"] = str(exc)
    return report


# ---------------------------------------------------------------------------- noise audit


@contextmanager
def latent_row_audit() -> Iterator[dict[str, Any]]:
    """Observe the per-sample latent rows the runner draws during one ``generate`` call.

    ``ArdyRunner.generate`` does not return the latent audit that ``generate_prompt_schedule``
    exposes.  Its per-sample noise patch draws every batch row separately through
    ``torch.randn(shape[1:], generator=<per-sample generator>)``, so wrapping ``torch.randn``
    *around* the call records one hash per (row, window) without touching the runner.  Rows are
    identified by their generator object in order of first appearance, which is the batch
    order the runner draws in.
    """
    import torch

    real = torch.randn
    draws: list[dict[str, Any]] = []
    generator_order: dict[int, int] = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        generator = kwargs.get("generator")
        if generator is not None:
            identity = id(generator)
            if identity not in generator_order:
                generator_order[identity] = len(generator_order)
            draws.append({
                "row": generator_order[identity],
                "shape": [int(dim) for dim in result.shape],
                "sha256": hashlib.sha256(
                    result.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
            })
        return result

    audit: dict[str, Any] = {"draws": draws}
    torch.randn = wrapper
    try:
        yield audit
    finally:
        torch.randn = real


def summarize_latent_audit(audit: Mapping[str, Any],
                           chunk_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Turn the observed draws into the pairing evidence and enforce the pairing contract.

    Raises when rows that share a seed received different latents in any window (the
    protocol's pairing guarantee is broken) or when rows with different seeds coincide.
    An absent audit (no generator-tagged draws observed) is recorded as unavailable, not
    as a pass.
    """
    draws = list(audit.get("draws", []))
    n_rows = len(chunk_rows)
    if not draws:
        return {"status": "unavailable", "n_rows": n_rows, "windows": 0, "rows": [],
                "pairing_verified": False,
                "note": "no generator-tagged torch.randn draws were observed"}
    per_row: dict[int, list[dict[str, Any]]] = {}
    for draw in draws:
        per_row.setdefault(int(draw["row"]), []).append(draw)
    observed_rows = sorted(per_row)
    if observed_rows[:n_rows] != list(range(n_rows)):
        raise ValueError(
            f"latent audit observed rows {observed_rows}, expected the first {n_rows}")
    extra = [row for row in observed_rows if row >= n_rows]
    windows = {row: len(per_row[row]) for row in range(n_rows)}
    if len(set(windows.values())) != 1 or next(iter(windows.values())) < 1:
        raise ValueError(f"latent audit rows drew unequal window counts: {windows}")
    n_windows = next(iter(windows.values()))
    rows_out = []
    for local, item in enumerate(chunk_rows):
        hashes = [draw["sha256"] for draw in per_row[local]]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"latent stream repeated a window for row {local}")
        rows_out.append({
            "batch_position": local, "seed": int(item["seed"]), "arm": str(item["arm"]),
            "row_sha256_by_window": hashes,
            "latent_shape": per_row[local][0]["shape"],
        })
    by_seed: dict[int, list[list[str]]] = {}
    for entry in rows_out:
        by_seed.setdefault(entry["seed"], []).append(entry["row_sha256_by_window"])
    for seed, lists in by_seed.items():
        if any(item != lists[0] for item in lists[1:]):
            raise ValueError(f"same-seed rows drew different latents for seed {seed}")
    seeds = list(by_seed)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            if any(x == y for x, y in zip(by_seed[a][0], by_seed[b][0])):
                raise ValueError(f"different seeds {a} and {b} share a latent window")
    return {
        "status": "verified", "n_rows": n_rows, "windows": n_windows,
        "pairing_verified": True,
        "same_seed_rows_identical_every_window": True,
        "different_seed_rows_differ_every_window": True,
        "stream_advances_between_windows": True,
        "extra_generator_draw_rows": extra,
        "rows": rows_out,
    }


# ------------------------------------------------------------------------ SONIC command


def sonic_command(pkl: Path, out_dir: Path, num_envs: int, physics_seed: int) -> list[str]:
    """The exp1b/EXP-022A launch command, reproduced here so the receipt can bind it."""
    from scene2motion.sonic_state_export import sonic_state_hydra_overrides

    pkl, out_dir = Path(pkl).resolve(), Path(out_dir).resolve()
    return [str(exp1b.SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl",
            f"+checkpoint={exp1b.CKPT}", "+headless=True", "++eval_callbacks=im_eval",
            "++run_eval_loop=False", f"++num_envs={int(num_envs)}",
            f"++eval_output_dir={out_dir}",
            f"++seed={int(physics_seed)}",
            "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
            TERMINATIONS_OVERRIDE,
            f"+manager_env.commands.motion.motion_lib_cfg.motion_file={pkl}",
            f"+log_keys={pkl.stem}",
            *sonic_state_hydra_overrides()]


def launch_sonic(pkl: Path, out_dir: Path, num_envs: int, physics_seed: int,
                 timeout_s: int) -> tuple[int, str]:
    cmd = sonic_command(pkl, out_dir, num_envs, physics_seed)
    print("  " + " ".join(cmd[:4]) + " ...", flush=True)
    proc = subprocess.run(cmd, cwd=exp1b.SONIC, capture_output=True, text=True,
                          timeout=timeout_s, env=exp1b.sonic_env(), stdin=subprocess.DEVNULL)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr)


def termination_config_record(tracker: Mapping[str, Any]) -> dict[str, Any]:
    """What we can bind about the resolved termination configuration without a launch.

    SONIC merges the checkpoint's ``config.yaml`` under the CLI overrides inside
    ``eval_agent_trl.main`` and never writes the merged result; Hydra's ``.hydra`` dump holds
    only the CLI-side composition.  Until a callback-side dump exists (EXP-028 owns that
    mechanism), the receipt binds the CLI override, the ``tracking/eval`` YAML text and hash,
    and the checkpoint config hash, and adopts ``resolved_terminations.json`` from an attempt's
    eval directory whenever a launch writes one.
    """
    core = dict(tracker.get("core_source_sha256", {}))
    eval_yaml = Path(str(tracker.get("root", ""))) / EVAL_TERMINATIONS_YAML
    try:
        eval_text = eval_yaml.read_text()
    except OSError:
        eval_text = None
    return {
        "mechanism": "cli_override_and_config_hashes",
        "release_evaluator_override": TERMINATIONS_OVERRIDE,
        "eval_terminations_yaml": {
            "path": EVAL_TERMINATIONS_YAML,
            "sha256": core.get(EVAL_TERMINATIONS_YAML),
            "text": eval_text,
        },
        "checkpoint_config_sha256": core.get("sonic_release/config.yaml"),
        "active_terms_per_plan_of_record": (
            "anchor_pos 0.25 m, anchor_ori_full 1.0 rad, ee_body_pos 0.25 m (height) / "
            "foot_pos_xyz 0.2 m (checkpoint config), motion_time_out; firing term not logged"
        ),
        "resolved_dump": None,
        "resolved_dump_source": None,
        "TODO": (
            "dump the merged termination config from inside the SONIC process (EXP-028 "
            "mechanism); this receipt binds the CLI override and config hashes only"
        ),
    }


def hydra_overrides(command: Sequence[str]) -> list[str]:
    """The Hydra override list of a launch command (everything after the module name)."""
    command = list(command)
    return command[command.index("gear_sonic.eval_agent_trl") + 1:]


def default_compose_terminations(overrides: Sequence[str]) -> dict[str, Any]:
    """Resolve the termination config offline through EXP-028's mechanism when available.

    ``exp028_termination_free_rollouts.compose_resolved_terminations`` replays
    ``eval_agent_trl``'s composition (Hydra ``base_eval`` + these overrides, then the
    checkpoint config underneath) inside the SONIC interpreter without importing Isaac.
    """
    try:
        from experiments import exp028_termination_free_rollouts as e28
    except ImportError as exc:  # pragma: no cover - only when the EXP-028 driver is absent
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    resolved = dict(e28.compose_resolved_terminations(list(overrides)))
    return {"status": "composed", **resolved}


def audit_release_terminations(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Is the resolved termination config the release evaluator?  Missing or mismatched
    thresholds are problems (fail closed); extra active terms are recorded, not fatal."""
    terms = resolved.get("terminations")
    if not isinstance(terms, Mapping):
        return {"is_release_evaluator": False, "problems": ["no terminations block"],
                "active_terms": [], "unexpected_active_terms": []}
    active = {name: term for name, term in terms.items()
              if isinstance(term, Mapping) and "func" in term}
    problems: list[str] = []
    for name, expected in RELEASE_EVALUATOR_THRESHOLDS.items():
        term = active.get(name)
        if term is None:
            problems.append(f"{name} is not an active termination")
            continue
        params = term.get("params", {}) if isinstance(term.get("params"), Mapping) else {}
        for key, value in expected.items():
            if params.get(key) != value:
                problems.append(f"{name}.params.{key}={params.get(key)!r} != {value!r}")
    if TIME_OUT_TERM not in active:
        problems.append(f"{TIME_OUT_TERM} is not an active termination")
    extra = sorted(set(active) - set(RELEASE_EVALUATOR_THRESHOLDS) - {TIME_OUT_TERM})
    return {"is_release_evaluator": not problems, "problems": problems,
            "active_terms": sorted(active), "unexpected_active_terms": extra,
            "expected_thresholds": {k: dict(v) for k, v in RELEASE_EVALUATOR_THRESHOLDS.items()}}


# ------------------------------------------------------------------ reference scoring


@dataclass
class ScoringContext:
    route: np.ndarray
    thresholds: StepOverThresholds
    support: Mapping[str, float]
    free_body: Any
    local_step_body: Any


def build_scoring_context(route: np.ndarray, thresholds: StepOverThresholds,
                          support: Mapping[str, float]) -> ScoringContext:
    return ScoringContext(
        route=np.asarray(route, dtype=float), thresholds=thresholds, support=dict(support),
        free_body=G1Body(None),
        local_step_body=G1Body(step_scene(STAGED_X_M, LOCAL_STEP_OBSTACLE_HEIGHT_M,
                                          OBSTACLE_DEPTH_M)),
    )


def _yaw_from_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    return atc._heading(np.asarray(quat, dtype=float))


def manipulation_check(qpos: np.ndarray) -> dict[str, Any]:
    """Compliance of the root with the pinned values, computed for every arm."""
    exact = np.asarray(qpos, dtype=float)
    z = exact[:, 2]
    yaw = _yaw_from_quaternion_wxyz(exact[:, 3:7])
    deviation = np.arctan2(np.sin(yaw - PIN_HEADING_RAD), np.cos(yaw - PIN_HEADING_RAD))
    unwrapped = np.unwrap(yaw)
    return {
        "pin_root_y_m": PIN_ROOT_Y_M,
        "pin_heading_deg": float(np.degrees(PIN_HEADING_RAD)),
        "root_z_mae_from_pin_m": float(np.abs(z - PIN_ROOT_Y_M).mean()),
        "root_z_max_abs_dev_from_pin_m": float(np.abs(z - PIN_ROOT_Y_M).max()),
        "root_z_min_m": float(z.min()),
        "root_z_max_m": float(z.max()),
        "root_z_range_m": float(z.max() - z.min()),
        "heading_mae_from_pin_deg": float(np.degrees(np.abs(deviation).mean())),
        "heading_max_abs_dev_from_pin_deg": float(np.degrees(np.abs(deviation).max())),
        "heading_range_deg": float(np.degrees(unwrapped.max() - unwrapped.min())),
    }


def score_reference_clip(qpos: np.ndarray, arm: str, ctx: ScoringContext) -> dict[str, Any]:
    """Protocol endpoints 1-6 for one 200-frame reference clip, in the protocol's order."""
    exact = np.asarray(qpos, dtype=float)
    if exact.shape != (N_FRAMES, QPOS_WIDTH) or not np.isfinite(exact).all():
        raise ValueError(f"reference scoring requires a finite ({N_FRAMES}, {QPOS_WIDTH}) clip")
    route = ctx.route
    # 1. elicitation (exp021 definition: 120-point whole-body box-height profile)
    xs, heights = box_height_profile(exact, route, OBSTACLE_DEPTH_M, n_points=SCAN_POINTS)
    lift = lift_location(xs, heights)
    side = None
    if lift["lift_x_m"] is not None:
        side = lift_side(exact, foot_kinematics_series(ctx.free_body, exact, FPS),
                         float(lift["lift_x_m"]), OBSTACLE_DEPTH_M)
    elicitation = {
        **lift,
        "lift_side": side,
        "elicited": bool(float(lift["lift_height_m"]) >= ELICITATION_MIN_M),
        "min_clearance_m": ELICITATION_MIN_M,
        "scan_points": SCAN_POINTS,
        "clears_height_anywhere": {
            f"{h:g}": bool(float(lift["lift_height_m"]) >= h) for h in GRADED_HEIGHTS_M},
    }
    # 2. exact fixed-centre clearance (EXP-022A's reference-tier scorer, never +/- r)
    exact_boxes = {label: dict(e22.score_trajectory(exact, x)) for label, x in OBSTACLES}
    # 3. contract features + gate predictions (calibrated support thresholds)
    features = atc.features(ctx.free_body, exact, ctx.support["support_height_m"],
                            ctx.support["support_speed_mps"], FPS)
    predictions = atc.gate_predictions(features, PRIMARY_GATE_S, SECONDARY_GATE_S)
    # 4. 13-gate local step at the staged centre against the 5 cm box
    local = evaluate_local_step(
        ctx.free_body, ctx.local_step_body, exact, STAGED_X_M, OBSTACLE_DEPTH_M, FPS,
        thresholds=ctx.thresholds)
    local_step = {
        "obstacle_x_m": STAGED_X_M,
        "obstacle_height_m": LOCAL_STEP_OBSTACLE_HEIGHT_M,
        "local_step_success": bool(local["local_step_success"]),
        "gates": {name: bool(value) for name, value in local["gates"].items()},
        "max_unsupported_run_local_s": float(local["support"]["max_unsupported_run_s"]),
        "lead_side": local["crossing"]["lead_side"],
        "max_lateral_deviation_m": local["root"]["max_lateral_deviation_m"],
    }
    # 5. route fidelity (exp023's supporting metrics)
    fidelity = dict(e23.supporting_motion_metrics(ctx.free_body, exact, route))
    # 6. manipulation check
    manipulation = manipulation_check(exact)
    return {
        "elicitation": elicitation,
        "exact_boxes": exact_boxes,
        "contract_features": features,
        "gate_predictions": predictions,
        "local_step": local_step,
        "route_fidelity": fidelity,
        "manipulation": manipulation,
    }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def validated_reference_score(value: Mapping[str, Any]) -> dict[str, Any]:
    """Planned-denominator guard: every endpoint must be present and well typed."""
    try:
        score = json.loads(cal._canonical_json(e23._json_safe(dict(value))))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reference score is not JSON-serialisable: {exc}") from exc
    elicitation = score["elicitation"]
    _finite_number(elicitation["lift_height_m"], "lift_height_m")
    if elicitation["lift_x_m"] is not None:
        _finite_number(elicitation["lift_x_m"], "lift_x_m")
    if not isinstance(elicitation["elicited"], bool):
        raise ValueError("elicited must be boolean")
    for label, _ in OBSTACLES:
        box = score["exact_boxes"][label]
        clears = box["exact_clears"]
        if set(clears) != {f"{h:g}" for h in GRADED_HEIGHTS_M}:
            raise ValueError(f"exact clears at {label} do not cover the graded heights")
        if any(not isinstance(flag, bool) for flag in clears.values()):
            raise ValueError(f"exact clears at {label} must be boolean")
        _finite_number(box["max_box_height_lower_bound_m"], "max_box_height_lower_bound_m")
    features = score["contract_features"]
    for name in atc.FEATS:
        _finite_number(features[name], name)
    predictions = score["gate_predictions"]
    if (predictions["primary_threshold_s"] != PRIMARY_GATE_S
            or predictions["secondary_threshold_s"] != SECONDARY_GATE_S
            or not isinstance(predictions["primary_flag"], bool)
            or not isinstance(predictions["secondary_flag"], bool)):
        raise ValueError("gate predictions do not carry the locked rules")
    if predictions["max_unsupported_run_s"] != features["max_unsupported_run_s"]:
        raise ValueError("gate prediction feature disagrees with the contract features")
    local = score["local_step"]
    if (not isinstance(local["local_step_success"], bool)
            or set(local["gates"]) != set(LOCAL_STEP_GATE_NAMES)
            or len(local["gates"]) != len(LOCAL_STEP_GATE_NAMES)
            or any(not isinstance(flag, bool) for flag in local["gates"].values())
            or local["obstacle_x_m"] != STAGED_X_M):
        raise ValueError("local-step record does not carry the 13 boolean gates at x=1.2")
    if local["local_step_success"] != all(local["gates"].values()):
        raise ValueError("local_step_success disagrees with its gates")
    fidelity = score["route_fidelity"]
    for name in ("progress_ratio", "route_path_mae_m", "max_foot_floor_penetration_m"):
        _finite_number(fidelity[name], name)
    manipulation = score["manipulation"]
    for name in ("root_z_mae_from_pin_m", "root_z_max_abs_dev_from_pin_m", "root_z_range_m",
                 "root_z_max_m", "heading_mae_from_pin_deg",
                 "heading_max_abs_dev_from_pin_deg", "heading_range_deg"):
        _finite_number(manipulation[name], name)
    return score


def arm_constructibility(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Protocol rule: pin_y / pin_yh by median root_z_range <= 0.10 m; pin_h by median heading
    range <= 10 deg; ``free`` requests no manipulation and is constructible by definition.
    Both medians are reported for every arm."""
    result: dict[str, Any] = {}
    for arm in ARMS:
        members = [row for row in rows if row["arm"] == arm]
        if len(members) != len(SEEDS):
            raise ValueError(f"constructibility needs {len(SEEDS)} scored clips for {arm}")
        z_range = float(np.median([
            row["reference"]["manipulation"]["root_z_range_m"] for row in members]))
        h_range = float(np.median([
            row["reference"]["manipulation"]["heading_range_deg"] for row in members]))
        root_ok = z_range <= CONSTRUCTIBLE_ROOT_Z_RANGE_M
        heading_ok = h_range <= CONSTRUCTIBLE_HEADING_RANGE_DEG
        if arm == "free":
            constructible, criterion = True, "no manipulation requested"
        elif arm in ("pin_y", "pin_yh"):
            constructible, criterion = root_ok, "median root_z_range_m <= 0.10"
        else:
            constructible, criterion = heading_ok, "median heading_range_deg <= 10"
        result[arm] = {
            "n": len(members),
            "median_root_z_range_m": z_range,
            "median_heading_range_deg": h_range,
            "root_z_range_criterion_met": bool(root_ok),
            "heading_range_criterion_met": bool(heading_ok),
            "decisive_criterion": criterion,
            "constructible": bool(constructible),
        }
    return result


# ---------------------------------------------------------------------------- statistics


def wilson(k: int, n: int) -> list[float]:
    return atc.wilson(int(k), int(n))


def rate(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": (k / n if n else None),
            "wilson95": wilson(k, n) if n else [None, None]}


def _bootstrap_auc_ci(values: np.ndarray, labels: np.ndarray, *, n_boot: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, v = labels[idx], values[idx]
        if 0 < y.sum() < n:
            samples.append(atc.auc(v[y == 1], v[y == 0]))
    if not samples:
        return [float("nan"), float("nan")]
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from the discordant counts (auxiliary only)."""
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return float(min(1.0, 2.0 * tail))


def contract_table(records: Sequence[Mapping[str, Any]], flag: str) -> dict[str, Any]:
    tracked = [r for r in records if r["terminated"] is not None]
    flagged_terminated = sum(1 for r in tracked if r[flag] and r["terminated"])
    flagged_survived = sum(1 for r in tracked if r[flag] and not r["terminated"])
    passed_terminated = sum(1 for r in tracked if not r[flag] and r["terminated"])
    passed_survived = sum(1 for r in tracked if not r[flag] and not r["terminated"])
    flagged = flagged_terminated + flagged_survived
    passed = passed_terminated + passed_survived
    terminated = flagged_terminated + passed_terminated
    survived = flagged_survived + passed_survived
    return {
        "n_tracked": len(tracked),
        "table": {
            "flagged_terminated": flagged_terminated, "flagged_survived": flagged_survived,
            "passed_terminated": passed_terminated, "passed_survived": passed_survived,
        },
        "flagged_terminated_rate": rate(flagged_terminated, flagged),
        "passed_terminated_rate": rate(passed_terminated, passed),
        "sensitivity": rate(flagged_terminated, terminated),
        "specificity": rate(passed_survived, survived),
    }


# ------------------------------------------------------------------------------ analysis


def clip_records(rows: Sequence[Mapping[str, Any]],
                 achieved_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the ledger into one analysis record per clip (planned denominator 128)."""
    achieved = {(str(r["motion_key"]), str(r["obstacle_label"])): r for r in achieved_rows}
    h5 = f"{P4_BOX_HEIGHT_M:g}"
    records = []
    for row in rows:
        reference = row["reference"]
        key = str(row["archive_key"])
        staged = achieved.get((key, STAGED_LABEL))
        control = achieved.get((key, CONTROL_LABEL))
        run = float(reference["contract_features"]["max_unsupported_run_s"])
        records.append({
            "seed": int(row["seed"]), "arm": str(row["arm"]), "key": key,
            "elicited": bool(reference["elicitation"]["elicited"]),
            "lift_height_m": float(reference["elicitation"]["lift_height_m"]),
            "lift_x_m": reference["elicitation"]["lift_x_m"],
            "exact_clears_staged": dict(reference["exact_boxes"][STAGED_LABEL]["exact_clears"]),
            "exact_clears_control": dict(reference["exact_boxes"][CONTROL_LABEL]["exact_clears"]),
            "exact_clear_5cm_staged": bool(
                reference["exact_boxes"][STAGED_LABEL]["exact_clears"][h5]),
            "max_unsupported_run_s": run,
            "root_z_max": float(reference["contract_features"]["root_z_max"]),
            "root_z_range_m": float(reference["manipulation"]["root_z_range_m"]),
            "heading_range_deg": float(reference["manipulation"]["heading_range_deg"]),
            "local_step_success": bool(reference["local_step"]["local_step_success"]),
            "primary_flag": bool(reference["gate_predictions"]["primary_flag"]),
            "secondary_flag": bool(reference["gate_predictions"]["secondary_flag"]),
            "contact_consistent": bool(reference["local_step"]["local_step_success"]
                                       and run <= PRIMARY_GATE_S),
            "terminated": (None if staged is None else bool(staged["tracker_terminated"])),
            "valid_frames": (None if staged is None else int(staged["valid_frames"])),
            "retained_staged": (None if staged is None else {
                k: bool(v) for k, v in staged["achieved_replay_clear_after_passing"].items()}),
            "retained_control": (None if control is None else {
                k: bool(v) for k, v in control["achieved_replay_clear_after_passing"].items()}),
            "retained_5cm_staged": (None if staged is None else bool(
                staged["achieved_replay_clear_after_passing"][h5])),
        })
    return records


def evaluate_decisions(records: Sequence[Mapping[str, Any]],
                       constructibility: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """P1-P4, the replication rule and the decision rules, evaluated mechanically."""
    by_arm = {arm: [r for r in records if r["arm"] == arm] for arm in ARMS}
    for arm, members in by_arm.items():
        if len(members) != len(SEEDS):
            raise ValueError(f"analysis needs {len(SEEDS)} records for {arm}, got {len(members)}")
    tracked = [r for r in records if r["terminated"] is not None]
    if len(tracked) != len(records):
        raise ValueError("analysis requires a tracker outcome for every clip")
    h5 = f"{P4_BOX_HEIGHT_M:g}"

    per_arm: dict[str, Any] = {}
    for arm, members in by_arm.items():
        n = len(members)
        per_arm[arm] = {
            "n": n,
            "constructible": bool(constructibility[arm]["constructible"]),
            "elicitation": rate(sum(r["elicited"] for r in members), n),
            "exact_clears_staged": {
                h: rate(sum(r["exact_clears_staged"][h] for r in members), n)
                for h in (f"{v:g}" for v in GRADED_HEIGHTS_M)},
            "exact_clears_control": {
                h: rate(sum(r["exact_clears_control"][h] for r in members), n)
                for h in (f"{v:g}" for v in GRADED_HEIGHTS_M)},
            "terminated": rate(sum(bool(r["terminated"]) for r in members), n),
            "retained_staged": {
                h: rate(sum(bool(r["retained_staged"][h]) for r in members), n)
                for h in (f"{v:g}" for v in GRADED_HEIGHTS_M)},
            "retained_control": {
                h: rate(sum(bool(r["retained_control"][h]) for r in members), n)
                for h in (f"{v:g}" for v in GRADED_HEIGHTS_M)},
            "primary_flagged": rate(sum(r["primary_flag"] for r in members), n),
            "secondary_flagged": rate(sum(r["secondary_flag"] for r in members), n),
            "contact_consistent": rate(sum(r["contact_consistent"] for r in members), n),
            "median_root_z_max_m": float(np.median([r["root_z_max"] for r in members])),
            "median_max_unsupported_run_s": float(np.median(
                [r["max_unsupported_run_s"] for r in members])),
            "contract_2x2_primary": contract_table(members, "primary_flag"),
        }

    # P1: prospective contract test over all 128 clips
    run = np.asarray([r["max_unsupported_run_s"] for r in tracked], dtype=float)
    y = np.asarray([1 if r["terminated"] else 0 for r in tracked], dtype=int)
    auc = atc.auc(run[y == 1], run[y == 0]) if 0 < y.sum() < len(y) else float("nan")
    auc_ci = (_bootstrap_auc_ci(run, y) if 0 < y.sum() < len(y)
              else [float("nan"), float("nan")])
    contract: dict[str, Any] = {}
    for label, flag in (("primary_0p20s", "primary_flag"), ("secondary_0p32s", "secondary_flag")):
        table = contract_table(tracked, flag)
        flagged_rate = table["flagged_terminated_rate"]["rate"]
        passed_rate = table["passed_terminated_rate"]["rate"]
        table["criteria"] = {
            "flagged_terminated_rate_ge_0p90": (
                flagged_rate is not None and flagged_rate >= P1_MIN_FLAGGED_TERMINATED_RATE),
            "passed_terminated_rate_le_0p30": (
                passed_rate is not None and passed_rate <= P1_MAX_PASSED_TERMINATED_RATE),
            "auc_ge_0p90": bool(np.isfinite(auc) and auc >= P1_MIN_AUC),
        }
        table["pass"] = bool(all(table["criteria"].values()))
        table["strong"] = {
            "definition": "P1-strong: confirmation bar met and single-feature AUC >= 0.95 "
                          "(bootstrap point estimate); additive, never replaces P1",
            "auc_ge_0p95": bool(np.isfinite(auc) and auc >= P1_STRONG_MIN_AUC),
            "pass": bool(table["pass"] and np.isfinite(auc) and auc >= P1_STRONG_MIN_AUC),
        }
        contract[label] = table
    p1 = {
        "definition": ("among all 128 clips: >= 90 % of clips flagged by the calibrated "
                       "0.20 s gate terminate, <= 30 % of passed clips terminate, and the "
                       "single-feature AUC of max_unsupported_run_s >= 0.90; P1-strong adds "
                       "AUC >= 0.95 as a predeclared second level"),
        "gate_rules": {"primary": "run > 0.20 s (>= 6 frames = 0.24 s at 25 fps; calibrated)",
                       "secondary": "run > 0.28 s (>= 8 frames = 0.32 s at 25 fps; post hoc)"},
        "auc_max_unsupported_run_s": float(auc),
        "auc_ci95_bootstrap": [float(v) for v in auc_ci],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "n_terminated": int(y.sum()), "n_survived": int((y == 0).sum()),
        "rules": contract,
        "pass": bool(contract["primary_0p20s"]["pass"]),
        "strong_pass": bool(contract["primary_0p20s"]["strong"]["pass"]),
        "secondary_rule_pass_reported_beside_not_instead": bool(
            contract["secondary_0p32s"]["pass"]),
        "secondary_rule_strong_pass": bool(contract["secondary_0p32s"]["strong"]["pass"]),
    }

    # P2: free replicates exp021
    free = per_arm["free"]
    elic = free["elicitation"]["rate"]
    exact5 = free["exact_clears_staged"][h5]["rate"]
    p2 = {
        "elicitation": {
            **free["elicitation"], "range": list(P2_ELICITATION_RANGE),
            "in_range": bool(P2_ELICITATION_RANGE[0] <= elic <= P2_ELICITATION_RANGE[1])},
        "exact_5cm_staged": {
            **free["exact_clears_staged"][h5], "range": list(P2_EXACT_5CM_RANGE),
            "in_range": bool(P2_EXACT_5CM_RANGE[0] <= exact5 <= P2_EXACT_5CM_RANGE[1])},
    }
    p2["pass"] = bool(p2["elicitation"]["in_range"] and p2["exact_5cm_staged"]["in_range"])

    # P3: pinned root arms
    free_run = {r["seed"]: r["max_unsupported_run_s"] for r in by_arm["free"]}
    p3_arms: dict[str, Any] = {}
    for arm in ("pin_y", "pin_yh"):
        members = by_arm[arm]
        shorter = sum(1 for r in members if r["max_unsupported_run_s"] < free_run[r["seed"]])
        ties = sum(1 for r in members if r["max_unsupported_run_s"] == free_run[r["seed"]])
        median_z = per_arm[arm]["median_root_z_max_m"]
        constructible = bool(constructibility[arm]["constructible"])
        criteria = {
            "median_root_z_max_le_0p85": bool(median_z <= P3_MAX_MEDIAN_ROOT_Z_MAX_M),
            "shorter_run_than_free_in_ge_20_of_32_paired_seeds": bool(
                shorter >= P3_MIN_SHORTER_PAIRED_SEEDS),
        }
        p3_arms[arm] = {
            "constructible": constructible,
            "median_root_z_max_m": median_z,
            "paired_seeds_with_shorter_run": shorter,
            "paired_seeds_tied": ties,
            "paired_seeds": len(members),
            "median_run_difference_vs_free_s": float(np.median(
                [r["max_unsupported_run_s"] - free_run[r["seed"]] for r in members])),
            "criteria": criteria,
            "pass": (bool(all(criteria.values())) if constructible else None),
            "excluded_as_non_constructible": not constructible,
        }
    evaluable = [a for a in p3_arms.values() if a["pass"] is not None]
    p3 = {"arms": p3_arms,
          "pass": (bool(all(a["pass"] for a in evaluable)) if evaluable else None),
          "note": "None means no pinned-root arm was constructible, so P3 is not evaluable"}

    # P4: any constructible arm with >= 3/32 contact-consistent, exact-clearing, retained clips
    p4_arms: dict[str, Any] = {}
    for arm, members in by_arm.items():
        hits = [r["key"] for r in members if r["contact_consistent"] and r["exact_clear_5cm_staged"]
                and bool(r["retained_5cm_staged"])]
        constructible = bool(constructibility[arm]["constructible"])
        p4_arms[arm] = {
            "constructible": constructible,
            "clips": rate(len(hits), len(members)),
            "clip_keys": hits,
            "meets_ge_3_of_32": bool(len(hits) >= P4_MIN_CLIPS),
            "counts_toward_go": bool(constructible and len(hits) >= P4_MIN_CLIPS),
        }
    p4 = {
        "arms": p4_arms,
        "pass": bool(any(a["counts_toward_go"] for a in p4_arms.values())),
        "multiplicity_note": (
            "four-arm multiplicity of the >= 3/32 rule: at a true rate of 0.02 a single arm "
            "reaches 3/32 with probability ~0.03, ~0.11 over four arms; the rate and its "
            "interval, not the pass/fail, are the reported quantities"),
    }

    # Replication rule for the free arm
    replication = {
        "elicitation_in_range": bool(p2["elicitation"]["in_range"]),
        "replication_failure_of_exp021": bool(not p2["elicitation"]["in_range"]),
        "first_suspect_if_failed": (
            "batch-shape sensitivity of end-to-end GPU byte identity (CLAUDE.md)"),
        "exact_5cm_staged_free": free["exact_clears_staged"][h5],
        "exp021_reference": {"exact_5cm_x1p2": "12/64", "expected_free": "~6/32"},
    }

    # paired McNemar counts, each pinned arm against free
    endpoints = ("elicited", "exact_clear_5cm_staged", "terminated", "retained_5cm_staged",
                 "primary_flag", "contact_consistent")
    free_by_seed = {r["seed"]: r for r in by_arm["free"]}
    mcnemar: dict[str, Any] = {}
    for arm in ("pin_y", "pin_h", "pin_yh"):
        mcnemar[arm] = {}
        for endpoint in endpoints:
            n11 = n10 = n01 = n00 = 0
            for r in by_arm[arm]:
                f = bool(free_by_seed[r["seed"]][endpoint])
                a = bool(r[endpoint])
                n11 += f and a
                n10 += f and not a
                n01 += (not f) and a
                n00 += (not f) and (not a)
            mcnemar[arm][endpoint] = {
                "both": n11, "free_only": n10, "arm_only": n01, "neither": n00,
                "exact_mcnemar_p_two_sided": mcnemar_exact(n10, n01),
            }

    decisions = {
        "contract_confirmed": bool(p1["pass"]),
        "contract_confirmed_strong": bool(p1["strong_pass"]),
        "prescriptive_contract_go": bool(p4["pass"]),
        "replication_failure_of_exp021": bool(replication["replication_failure_of_exp021"]),
        "outcome": ("GO: prescriptive contract" if p4["pass"] else
                    "NO-GO: diagnostic only (float under every native contract tried)"),
    }
    return {
        "n_clips": len(records),
        "per_arm": per_arm,
        "p1_prospective_contract": p1,
        "p2_free_replicates_exp021": p2,
        "p3_pinned_root_arms": p3,
        "p4_prescriptive_contract": p4,
        "replication_rule_free": replication,
        "paired_mcnemar_vs_free": mcnemar,
        "constructibility": {arm: dict(constructibility[arm]) for arm in ARMS},
        "test_retest_ceiling": {
            "source": "exp1b physics-seed re-roll",
            "disagreeing_rollouts": EXP1B_TEST_RETEST_DISAGREEMENT[0],
            "n": EXP1B_TEST_RETEST_DISAGREEMENT[1],
            "note": "single physics seed 0; every tracker outcome carries this ceiling",
        },
        "decisions": decisions,
        "interpretation_guard": (
            "obstacle absent from Isaac; retained = achieved-state replay against the collision "
            "model with EXP-022A's guarded endpoint; terminated = evaluator cutoff, not a fall; "
            "one scene, Wilson intervals over seeds, no scene-level inference"),
    }


# ------------------------------------------------------------------------------- ledger


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise CampaignAbort(f"invalid JSONL artifact {path}: {exc}") from exc


def _read_receipt(output: Path) -> dict[str, Any]:
    path = output / "receipt.json"
    if not path.is_file():
        raise CampaignAbort(f"no EXP-024 receipt at {path}; run --stage generate first")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CampaignAbort(f"unreadable EXP-024 receipt: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("experiment") != EXPERIMENT:
        raise CampaignAbort("existing output is not an EXP-024 campaign")
    if receipt.get("blocked") is True:
        raise CampaignAbort(
            "existing EXP-024 campaign is blocked; preserve it and use fresh output")
    return receipt


def _load_qpos(output: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(output / "qpos.npz", allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise CampaignAbort(f"invalid qpos archive: {exc}") from exc


class Ledger:
    """Receipt + rows + evidence anchors for one campaign directory."""

    def __init__(self, output: Path, receipt: dict[str, Any], rows: list[dict[str, Any]],
                 started: float | None = None):
        self.output = output
        self.receipt = receipt
        self.rows = rows
        self.started = time.monotonic() if started is None else started
        self.file_hashes: dict[str, str | None] = {}

    @classmethod
    def load(cls, output: Path) -> "Ledger":
        receipt = _read_receipt(output)
        rows = _read_jsonl(output / "rows.jsonl")
        anchors = receipt.get("evidence_anchors", {}).get("rows", {})
        if (anchors.get("n_rows") != len(rows)
                or anchors.get("logical_sha256") != cal._json_hash(rows)):
            raise CampaignAbort("rows.jsonl no longer matches its evidence anchor")
        if anchors.get("file_sha256") != cal._sha256(output / "rows.jsonl"):
            raise CampaignAbort("rows.jsonl file hash no longer matches its evidence anchor")
        ledger = cls(output, receipt, rows, started=time.monotonic() - float(
            receipt.get("wall_clock_s", 0.0)))
        return ledger

    def stage(self, name: str) -> dict[str, Any]:
        return self.receipt.setdefault("stages", {}).setdefault(name, {"status": "planned"})

    def require_stage_complete(self, name: str) -> None:
        if self.stage(name).get("status") != "complete":
            raise CampaignAbort(f"EXP-024 stage {name!r} is not complete in {self.output}")

    def anchor_file(self, name: str, path: Path, **extra: Any) -> None:
        self.receipt.setdefault("evidence_anchors", {})[name] = {
            "path": path.name, "file_sha256": cal._sha256(path), **extra}

    def persist(self, *, stage_label: str | None = None) -> None:
        cal._write_jsonl(self.output / "rows.jsonl", self.rows)
        self.receipt.setdefault("evidence_anchors", {})["rows"] = {
            "path": "rows.jsonl", "n_rows": len(self.rows),
            "logical_sha256": cal._json_hash(self.rows),
            "file_sha256": cal._sha256(self.output / "rows.jsonl"),
        }
        if stage_label is not None:
            self.receipt["stage"] = stage_label
        self.receipt["wall_clock_s"] = float(time.monotonic() - self.started)
        cal._write_json(self.output / "receipt.json", self.receipt)

    def fail(self, stage_name: str, exc: BaseException, stage_label: str) -> None:
        self.stage(stage_name).update({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        self.receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
            "blocked": True, "failed_stage": stage_label,
            "error_type": type(exc).__name__, "error": str(exc),
        })
        self.persist(stage_label=stage_label)


def _validate_generation_archive(ledger: Ledger) -> dict[str, np.ndarray]:
    """Revalidate the generated archive against the receipt before any later stage uses it."""
    ledger.require_stage_complete("generate")
    plan = locked_row_plan()
    if len(ledger.rows) != N_ROWS:
        raise CampaignAbort("ledger does not hold the planned 128 rows")
    clips = _load_qpos(ledger.output)
    anchors = ledger.receipt.get("evidence_anchors", {}).get("qpos", {})
    if set(clips) != {row["archive_key"] for row in plan} or anchors.get("n_arrays") != N_ROWS:
        raise CampaignAbort("qpos archive keys do not match the locked row plan")
    if cal._array_hash(clips) != anchors.get("content_sha256"):
        raise CampaignAbort("qpos archive content hash no longer matches its evidence anchor")
    for row, item in zip(ledger.rows, plan):
        for field in ("row_index", "seed", "arm", "archive_key", "chunk"):
            if row.get(field) != item[field]:
                raise CampaignAbort(f"ledger row {row.get('row_index')} drifted from the plan")
        qpos = clips[row["archive_key"]]
        if qpos.shape != (N_FRAMES, QPOS_WIDTH) or not np.isfinite(qpos).all():
            raise CampaignAbort(f"archived {row['archive_key']} is not a finite 200x36 clip")
        if cal._array_hash({row["archive_key"]: qpos}) != row.get("qpos_content_sha256"):
            raise CampaignAbort(f"archived {row['archive_key']} does not match its row hash")
    return clips


def _stage_provenance_check(ledger: Ledger, *, code_state_fn, source_hashes_fn,
                            runtime_identity_fn, physical_identity_fn) -> dict[str, Any]:
    provenance = ledger.receipt.get("provenance", {})
    current_code = dict(code_state_fn(ROOT))
    git = _verify_stage_git_state(provenance.get("code", {}), current_code,
                                  repo=ROOT, output=ledger.output)
    if dict(source_hashes_fn(ROOT)) != provenance.get("source_sha256"):
        raise ValueError("EXP-024 source content changed since generation")
    if dict(runtime_identity_fn()) != provenance.get("runtime"):
        raise ValueError("EXP-024 numerical runtime identity changed since generation")
    if dict(physical_identity_fn()) != provenance.get("physical_model"):
        raise ValueError("EXP-024 physical model identity changed since generation")
    return {"git": git, "sources_unchanged": True, "runtime_unchanged": True,
            "physical_model_unchanged": True, "current_code": current_code}


# ------------------------------------------------------------------------------ generate


def run_generate(
    *,
    out: str | Path,
    runner: Any | None = None,
    runner_factory: Callable[[], Any] | None = None,
    cache_path: str | Path = "outputs/text_cache.npz",
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    generator_identity_fn: Callable[[Any], Mapping[str, Any]] = cal._generator_identity,
    generator_identity_validator_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        cal._validated_generator_identity),
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
    prompt_identity_fn: Callable[[Any, Path], Mapping[str, Any]] = _step_prompt_cache_identity,
    threshold_dependency_fn: Callable[[], tuple[Any, Mapping[str, Any]]] = _threshold_dependency,
    pin_validator_fn: Callable[..., None] = _validate_pins,
    channel_usage_fn: Callable[[Any, ConstraintSpec], Mapping[str, int]] = _actual_channel_usage,
    host_gate_fn: Callable[..., Mapping[str, Any]] = require_host_resources,
    latent_audit_factory: Callable[[], Any] = latent_row_audit,
) -> dict[str, Any]:
    """Stage 1: sixteen locked B=8 calls into an empty directory; the ledger exists first."""
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise CampaignAbort(f"refusing nonempty EXP-024 output directory: {output}")
    if runner is not None and runner_factory is not None:
        raise CampaignAbort("provide either runner or runner_factory, not both")
    # Host gate before anything is created or any seed is spent; failure leaves --out untouched.
    try:
        gate_report = dict(host_gate_fn(require_no_isaac=False))
    except HostResourceGateFailed as exc:
        raise CampaignAbort(f"generation refused by the host-resource gate: {exc}") from exc
    code = dict(code_state_fn(ROOT))
    if code.get("dirty") is not False:
        raise CampaignAbort("EXP-024 requires an exactly clean git worktree")
    if not isinstance(code.get("commit"), str) or not code["commit"].strip():
        raise CampaignAbort("EXP-024 requires a concrete git commit")
    if os.environ.get("CHECKPOINTS_DIR"):
        raise CampaignAbort("EXP-024 forbids ambient CHECKPOINTS_DIR")
    source_hashes = dict(source_hashes_fn(ROOT))
    protocol_sha = source_hashes.get(PROTOCOL_PATH)
    if not cal._is_sha256(protocol_sha):
        raise CampaignAbort("EXP-024 protocol content hash is missing or invalid")

    injected = [name for name, value, default in (
        ("runner_instance", runner, None), ("runner_factory", runner_factory, None),
        ("code_state_fn", code_state_fn, cal._git_state),
        ("source_hashes_fn", source_hashes_fn, _source_hashes),
        ("generator_identity_fn", generator_identity_fn, cal._generator_identity),
        ("generator_identity_validator_fn", generator_identity_validator_fn,
         cal._validated_generator_identity),
        ("runtime_identity_fn", runtime_identity_fn, cal._runtime_identity),
        ("physical_identity_fn", physical_identity_fn, cal._physical_model_identity),
        ("prompt_identity_fn", prompt_identity_fn, _step_prompt_cache_identity),
        ("threshold_dependency_fn", threshold_dependency_fn, _threshold_dependency),
        ("pin_validator_fn", pin_validator_fn, _validate_pins),
        ("channel_usage_fn", channel_usage_fn, _actual_channel_usage),
        ("host_gate_fn", host_gate_fn, require_host_resources),
        ("latent_audit_factory", latent_audit_factory, latent_row_audit),
    ) if value is not default]

    plan = locked_row_plan()
    chunks = locked_chunk_plan(plan)
    launches = launch_plan(plan)
    route = route_xz()
    specs = {arm: arm_spec(arm, route) for arm in ARMS}
    spec_hashes = {arm: spec_sha256(spec) for arm, spec in specs.items()}
    for row in plan:
        row["spec_sha256"] = spec_hashes[row["arm"]]
    counters = {
        "generate_invocations_planned": len(chunks),
        "generate_invocations_started": 0,
        "generate_invocations_completed": 0,
        "samples_planned": N_ROWS,
        "samples_launched": 0,
        "samples_returned": 0,
        "samples_converted_to_qpos": 0,
    }
    spent_seeds: list[int] = []
    qpos_archive: dict[str, np.ndarray] = {}
    noise_evidence: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": "preflight",
        "stages": {name: {"status": "planned"} for name in STAGES},
        "sample_count_exact": True,
        "actual_ardy_samples": 0,
        "campaign_design": {
            "prompt": STEP,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "arm_contracts": {arm: dict(ARM_CONTRACTS[arm]) for arm in ARMS},
            "arm_spec_sha256": spec_hashes,
            "expected_channel_usage": {arm: dict(EXPECTED_CHANNEL_USAGE[arm]) for arm in ARMS},
            "static_channel_usage": {
                arm: static_channel_usage(spec) for arm, spec in specs.items()},
            "row_plan": plan,
            "row_plan_sha256": cal._json_hash(plan),
            "chunk_plan": [{k: v for k, v in chunk.items() if k != "rows"} for chunk in chunks],
            "chunk_seed_count": CHUNK_SEED_COUNT,
            "chunk_batch_rows": CHUNK_ROWS,
            "launch_plan": launches,
            "launch_assignment": (
                "seed-block-major: launch k tracks seeds 4600+8k..4607+8k under all four arms "
                "(generation chunks 4k..4k+3), so every within-seed comparison sits in one launch"),
            "fps": FPS, "n_frames": N_FRAMES, "diffusion_steps": DIFFUSION_STEPS,
            "cfg_weight": list(CFG_WEIGHT), "first_heading": FIRST_HEADING,
            "noise_stream_version": NOISE_STREAM_VERSION,
            "route": {"speed_mps": float(cal.REFERENCE_SPEED_MPS),
                      "length_m": float(cal.PILOT_ROUTE_LENGTH_M),
                      "sha256": cal._array_hash({"route_xz": route})},
            "obstacles": [{"label": label, "x_m": x} for label, x in OBSTACLES],
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "graded_heights_m": list(GRADED_HEIGHTS_M),
            "elicitation": {"scan_points": SCAN_POINTS, "min_clearance_m": ELICITATION_MIN_M,
                            "definition": "exp021 box_height_profile / lift_location"},
            "gates": {"primary_s": PRIMARY_GATE_S, "secondary_s": SECONDARY_GATE_S,
                      "flag_rule": "max_unsupported_run_s > threshold",
                      "primary_note": "> 0.20 s = >= 6 frames (0.24 s) at 25 fps; calibrated",
                      "secondary_note": "> 0.28 s = >= 8 frames (0.32 s) at 25 fps; post hoc",
                      "p1_strong_min_auc": P1_STRONG_MIN_AUC},
            "local_step": {"obstacle_x_m": STAGED_X_M,
                           "obstacle_height_m": LOCAL_STEP_OBSTACLE_HEIGHT_M},
            "constructibility": {"root_z_range_max_m": CONSTRUCTIBLE_ROOT_Z_RANGE_M,
                                 "heading_range_max_deg": CONSTRUCTIBLE_HEADING_RANGE_DEG},
            "predictions": {
                "P1": "0.20 s gate prospective: >= 90 % of flagged terminate, <= 30 % of passed "
                      "terminate, AUC >= 0.90 (secondary >= 0.32 s = > 0.28 s reported beside); "
                      "P1-strong: primary single-feature AUC >= 0.95 (additive level)",
                "P2": "free elicitation in [0.55, 0.95]; exact 5 cm at 1.2 m in [0.06, 0.35]",
                "P3": "pin_y/pin_yh: median root_z_max <= 0.85 m and shorter max run than free "
                      "in >= 20/32 paired seeds",
                "P4": "any constructible arm: >= 3/32 clips contact-consistent and exact-clear "
                      "5 cm and retained (no prediction either way)",
            },
            "physics_seed": PHYSICS_SEED,
            "tracker_pins": {"core_source_manifest_sha256": EXPECTED_CORE_SOURCE_MANIFEST_SHA256,
                             "checkpoint_sha256": EXPECTED_TRACKER_CHECKPOINT_SHA256},
        },
        "query_accounting": dict(counters),
        "generation_chunks": {chunk["name"]: {"status": "planned", "seeds": list(chunk["seeds"]),
                                              "row_indices": list(chunk["row_indices"])}
                              for chunk in chunks},
        "host_resource_gate": {"generate": gate_report},
        "provenance": {
            "code": code,
            "source_sha256": source_hashes,
            "protocol": {"path": PROTOCOL_PATH, "sha256": protocol_sha},
        },
        "execution_mode": {
            "dependency_injections": injected,
            "scientific_evidence_eligible": not injected,
            "pre_model_construction_evidence_guaranteed": runner is None,
            "note": ("Dependency injection exists for CPU tests; any injected run is "
                     "non-evidentiary. Production constructs the ARDY runner only after the "
                     "empty evidence bundle is durable."),
        },
        "spent_seeds": [],
        "seeds_spent_and_must_not_be_reused": False,
        "predictions": {"status": "not_written"},
        "launches": {},
    }
    ledger = Ledger(output, receipt, [])
    stage_record = ledger.stage("generate")
    stage_record["status"] = "running"

    qpos_dirty = True
    noise_dirty = True
    qpos_content: str | None = None

    def persist(stage_label: str) -> None:
        nonlocal qpos_dirty, noise_dirty, qpos_content
        if qpos_dirty:
            cal._persist_qpos(output / "qpos.npz", qpos_archive)
            qpos_content = cal._array_hash(qpos_archive) if qpos_archive else None
            qpos_dirty = False
        if noise_dirty:
            cal._write_json(output / "noise_audit.json", noise_evidence)
            noise_dirty = False
        receipt["query_accounting"] = dict(counters)
        receipt["actual_ardy_samples"] = int(counters["samples_returned"])
        receipt["spent_seeds"] = list(spent_seeds)
        receipt["unlaunched_locked_seeds"] = [s for s in SEEDS if s not in spent_seeds]
        receipt["seeds_spent_and_must_not_be_reused"] = bool(spent_seeds)
        anchors = receipt.setdefault("evidence_anchors", {})
        anchors["qpos"] = {"path": "qpos.npz", "n_arrays": len(qpos_archive),
                           "content_sha256": qpos_content,
                           "file_sha256": cal._sha256(output / "qpos.npz")}
        anchors["noise_audit"] = {"path": "noise_audit.json", "n_records": len(noise_evidence),
                                  "logical_sha256": cal._json_hash(noise_evidence),
                                  "file_sha256": cal._sha256(output / "noise_audit.json")}
        ledger.persist(stage_label=stage_label)

    # The empty ledger is durable before the runner exists.
    persist("preflight")
    stage_label = "preflight"
    try:
        if runner is None:
            if runner_factory is not None:
                runner = runner_factory()
            else:
                from scene2motion.runner import ArdyRunner
                runner = ArdyRunner(cache_path=cache_path)
        if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
            raise ValueError(f"EXP-024 requires runner fps == {FPS:g}")
        if int(runner.noise_stream_version) != NOISE_STREAM_VERSION:
            raise ValueError("EXP-024 requires noise_stream_version == 2")

        generator_identity = dict(generator_identity_validator_fn(generator_identity_fn(runner)))
        runtime_identity = dict(runtime_identity_fn())
        physical_identity = dict(physical_identity_fn())
        prompt_identity = dict(prompt_identity_fn(runner, Path(cache_path)))
        pin_validator_fn(generator_identity, runtime_identity, physical_identity)
        thresholds, threshold_dependency = threshold_dependency_fn()
        receipt["provenance"].update({
            "generator": generator_identity,
            "runtime": runtime_identity,
            "physical_model": physical_identity,
            "step_prompt_cache": prompt_identity,
            "physical_threshold_dependency": dict(threshold_dependency),
        })
        usage = {}
        for arm, spec in specs.items():
            observed = {str(k): int(v) for k, v in channel_usage_fn(runner, spec).items()}
            if observed != dict(EXPECTED_CHANNEL_USAGE[arm]):
                raise ValueError(
                    f"arm {arm} conditions the wrong channels: {observed} != "
                    f"{dict(EXPECTED_CHANNEL_USAGE[arm])}")
            usage[arm] = observed
        receipt["campaign_design"]["actual_channel_usage"] = usage
        persist("identities_bound")

        def revalidate() -> dict[str, Any]:
            current_code = dict(code_state_fn(ROOT))
            git_check = cal._verify_completion_git_state(code, current_code, repo=ROOT,
                                                         output=output)
            if dict(source_hashes_fn(ROOT)) != source_hashes:
                raise ValueError("EXP-024 source content changed during generation")
            current = dict(generator_identity_validator_fn(generator_identity_fn(runner)))
            if current != generator_identity:
                raise ValueError("EXP-024 checkpoint identity changed during generation")
            if dict(runtime_identity_fn()) != runtime_identity:
                raise ValueError("EXP-024 ARDY/numerical runtime identity changed")
            if dict(physical_identity_fn()) != physical_identity:
                raise ValueError("EXP-024 G1 physical model identity changed")
            if dict(prompt_identity_fn(runner, Path(cache_path))) != prompt_identity:
                raise ValueError("EXP-024 cached STEP prompt identity changed")
            pin_validator_fn(current, runtime_identity, physical_identity)
            if (float(runner.fps) != FPS
                    or int(runner.noise_stream_version) != NOISE_STREAM_VERSION):
                raise ValueError("EXP-024 runner contract changed")
            return {"git": git_check, "sources_unchanged": True, "checkpoint_unchanged": True,
                    "runtime_unchanged": True, "physical_model_unchanged": True,
                    "prompt_cache_unchanged": True, "runner_contract_unchanged": True}

        for chunk in chunks:
            name = str(chunk["name"])
            stage_label = f"generation_{name}"
            chunk_rows = list(chunk["rows"])
            counters["generate_invocations_started"] += 1
            counters["samples_launched"] += len(chunk_rows)
            spent_seeds.extend(s for s in chunk["seeds"] if s not in spent_seeds)
            receipt["generation_chunks"][name]["status"] = "running"
            persist(stage_label)
            prompts = [STEP] * len(chunk_rows)
            batch_specs = [specs[str(row["arm"])] for row in chunk_rows]
            seeds = [int(row["seed"]) for row in chunk_rows]
            try:
                with latent_audit_factory() as audit:
                    returned = runner.generate(
                        prompts, batch_specs, N_FRAMES, DIFFUSION_STEPS,
                        cfg_weight=CFG_WEIGHT, seeds=seeds)
            except Exception:
                receipt["sample_count_exact"] = False
                receipt["generation_chunks"][name]["status"] = "generation_exception"
                persist(stage_label)
                raise
            returned = list(returned)
            counters["generate_invocations_completed"] += 1
            counters["samples_returned"] += len(returned)
            receipt["generation_chunks"][name].update({
                "status": "returned_unvalidated", "samples_returned": len(returned)})
            audit_summary = summarize_latent_audit(audit, chunk_rows)
            noise_evidence.append({"chunk": name, "seeds": list(chunk["seeds"]),
                                   "batch_seed_order": seeds, **audit_summary})
            noise_dirty = True
            if len(returned) != len(chunk_rows):
                persist(stage_label)
                raise ValueError(
                    f"{name} returned {len(returned)} samples, expected {len(chunk_rows)}")
            for row, sample in zip(chunk_rows, returned):
                key = str(row["archive_key"])
                qpos = np.asarray(runner.to_qpos(sample))
                if qpos.shape != (N_FRAMES, QPOS_WIDTH) or not np.isfinite(qpos).all():
                    qpos_archive[f"{key}_invalid_return"] = np.array(qpos, copy=True)
                    qpos_dirty = True
                    persist(stage_label)
                    raise ValueError(f"{key} decoded to an invalid qpos of shape {qpos.shape}")
                qpos_archive[key] = np.array(qpos, copy=True)
                qpos_dirty = True
                counters["samples_converted_to_qpos"] += 1
                ledger.rows.append({
                    **{k: v for k, v in row.items()},
                    "sample_sha256": e17._sample_hash(sample),
                    "qpos_content_sha256": cal._array_hash({key: qpos_archive[key]}),
                    "qpos_dtype": str(qpos_archive[key].dtype),
                    "latent_row_sha256_by_window": (
                        audit_summary["rows"][int(row["batch_position"])]["row_sha256_by_window"]
                        if audit_summary["status"] == "verified" else None),
                    "generated": True,
                })
            receipt["generation_chunks"][name].update({
                "status": "complete", "latent_audit_status": audit_summary["status"]})
            persist(stage_label)

        if len(ledger.rows) != N_ROWS or counters["samples_returned"] != N_ROWS:
            raise ValueError("EXP-024 generation accounting is not exactly 128/128")
        if [row["archive_key"] for row in ledger.rows] != [row["archive_key"] for row in plan]:
            raise ValueError("EXP-024 rows are not in locked plan order")
        stage_label = "post_generation_revalidation"
        receipt["provenance"]["post_generation_identity_revalidation"] = revalidate()
        stage_record.update({
            "status": "complete",
            "samples": N_ROWS,
            "latent_audit": {
                "verified_chunks": sum(
                    1 for e in noise_evidence if e["status"] == "verified"),
                "unavailable_chunks": sum(
                    1 for e in noise_evidence if e["status"] == "unavailable"),
                "pairing_verified_every_chunk": all(
                    e["status"] == "verified" for e in noise_evidence),
            },
        })
        receipt["actual_ardy_samples"] = N_ROWS
        receipt["stage"] = "generated"
        persist("generated")
        stage_record["runner_release"] = _release_runner(runner)
        runner = None
        persist("generated")
        return receipt
    except Exception as exc:
        receipt["stage"] = stage_label
        ledger.fail("generate", exc, stage_label)
        try:
            persist(stage_label)
        except Exception:  # pragma: no cover - the failure receipt above is already durable
            pass
        if runner is not None:
            _release_runner(runner)
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# --------------------------------------------------------------------------------- score


def run_score(
    *,
    out: str | Path,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
    threshold_dependency_fn: Callable[[], tuple[Any, Mapping[str, Any]]] = _threshold_dependency,
    scoring_context_fn: Callable[..., Any] = build_scoring_context,
    reference_scorer_fn: Callable[[np.ndarray, str, Any], Mapping[str, Any]] = score_reference_clip,
) -> dict[str, Any]:
    """Stage 2: CPU reference endpoints for every clip, written before any tracker stage."""
    output = Path(out)
    ledger = Ledger.load(output)
    if ledger.stage("score").get("status") == "complete":
        _validate_generation_archive(ledger)
        return ledger.receipt
    stage_label = "score_preflight"
    try:
        clips = _validate_generation_archive(ledger)
        check = _stage_provenance_check(
            ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
            runtime_identity_fn=runtime_identity_fn, physical_identity_fn=physical_identity_fn)
        thresholds, dependency = threshold_dependency_fn()
        pinned = ledger.receipt["provenance"].get("physical_threshold_dependency", {})
        if dict(dependency).get("sha256") != pinned.get("sha256"):
            raise ValueError("threshold dependency identity differs from the generation stage")
        stage = ledger.stage("score")
        stage.update({"status": "running", "provenance_check": check,
                      "scored": 0, "planned": N_ROWS})
        ledger.persist(stage_label="scoring")
        route = route_xz()
        pinned_route = ledger.receipt["campaign_design"]["route"]["sha256"]
        if cal._array_hash({"route_xz": route}) != pinned_route:
            raise ValueError("route drifted from the generation stage")
        ctx = scoring_context_fn(route, thresholds, dependency["support_thresholds"])
        for index, row in enumerate(ledger.rows):
            stage_label = f"score_{row['archive_key']}"
            if row.get("reference") is not None:
                continue
            started = time.monotonic()
            score = validated_reference_score(
                reference_scorer_fn(clips[row["archive_key"]], str(row["arm"]), ctx))
            row["reference"] = score
            row["reference_scoring_wall_clock_s"] = float(time.monotonic() - started)
            stage["scored"] = index + 1
            if (index + 1) % 8 == 0 or index + 1 == N_ROWS:
                ledger.persist(stage_label="scoring")
        if sum(1 for row in ledger.rows if row.get("reference") is not None) != N_ROWS:
            raise ValueError("reference scoring did not preserve the planned denominator")
        stage_label = "score_summary"
        constructibility = arm_constructibility(ledger.rows)
        records = clip_records(ledger.rows, [])
        per_arm = {}
        for arm in ARMS:
            members = [r for r in records if r["arm"] == arm]
            per_arm[arm] = {
                "n": len(members),
                "elicitation": rate(sum(r["elicited"] for r in members), len(members)),
                "exact_clear_5cm_staged": rate(
                    sum(r["exact_clear_5cm_staged"] for r in members), len(members)),
                "primary_flagged": rate(sum(r["primary_flag"] for r in members), len(members)),
                "secondary_flagged": rate(sum(r["secondary_flag"] for r in members), len(members)),
                "contact_consistent": rate(
                    sum(r["contact_consistent"] for r in members), len(members)),
                "median_root_z_max_m": float(np.median([r["root_z_max"] for r in members])),
                "median_max_unsupported_run_s": float(np.median(
                    [r["max_unsupported_run_s"] for r in members])),
            }
        stage.update({
            "status": "complete", "scored": N_ROWS,
            "constructibility": constructibility,
            "reference_summary_per_arm": per_arm,
            "post_score_provenance_check": _stage_provenance_check(
                ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
                runtime_identity_fn=runtime_identity_fn,
                physical_identity_fn=physical_identity_fn),
        })
        ledger.receipt["stage"] = "scored"
        ledger.persist(stage_label="scored")
        return ledger.receipt
    except Exception as exc:
        ledger.fail("score", exc, stage_label)
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# ------------------------------------------------------------------------------- predict


def prediction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        reference = row["reference"]
        out.append({
            "row_index": int(row["row_index"]), "seed": int(row["seed"]),
            "arm": str(row["arm"]), "archive_key": str(row["archive_key"]),
            "qpos_content_sha256": str(row["qpos_content_sha256"]),
            **{k: reference["gate_predictions"][k] for k in (
                "max_unsupported_run_s", "primary_threshold_s", "primary_flag",
                "secondary_threshold_s", "secondary_flag")},
            "features": {name: reference["contract_features"][name] for name in atc.FEATS},
        })
    return out


def run_predict(*, out: str | Path) -> dict[str, Any]:
    """Stage 3: the per-clip gate predictions, hashed into the receipt before any launch."""
    output = Path(out)
    ledger = Ledger.load(output)
    path = output / "predictions.jsonl"
    stage = ledger.stage("predict")
    try:
        ledger.require_stage_complete("score")
        expected = prediction_rows(ledger.rows)
        if len(expected) != N_ROWS:
            raise ValueError("predictions do not cover the planned 128 clips")
        if stage.get("status") == "complete":
            if _read_jsonl(path) != expected or cal._sha256(path) != stage.get("file_sha256"):
                raise ValueError("predictions.jsonl no longer matches its hashed content")
            return ledger.receipt
        cal._write_jsonl(path, expected)
        stage.update({
            "status": "complete", "path": path.name, "n": len(expected),
            "file_sha256": cal._sha256(path), "logical_sha256": cal._json_hash(expected),
            "primary_flagged": int(sum(r["primary_flag"] for r in expected)),
            "secondary_flagged": int(sum(r["secondary_flag"] for r in expected)),
            "written_before_sonic": ledger.stage("sonic").get("status") == "planned",
        })
        ledger.receipt["predictions"] = {
            "status": "written", "path": path.name, "n": len(expected),
            "file_sha256": stage["file_sha256"], "logical_sha256": stage["logical_sha256"],
            "written_before_sonic": stage["written_before_sonic"],
            "predictions_committed_before_sonic": {
                "asserted": False,
                "note": ("to be asserted at the SONIC stage by comparing the committed HEAD "
                         "blob of predictions.jsonl with the working file "
                         "(--require-committed-predictions)"),
            },
        }
        ledger.anchor_file("predictions", path, n_rows=len(expected),
                           logical_sha256=stage["logical_sha256"])
        ledger.receipt["stage"] = "predicted"
        ledger.persist(stage_label="predicted")
        return ledger.receipt
    except Exception as exc:
        ledger.fail("predict", exc, "predict")
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


def _verify_predictions(ledger: Ledger) -> dict[str, Any]:
    ledger.require_stage_complete("predict")
    path = ledger.output / "predictions.jsonl"
    stage = ledger.stage("predict")
    if not path.is_file():
        raise ValueError("predictions.jsonl is missing; refusing to launch SONIC")
    rows = _read_jsonl(path)
    if (cal._sha256(path) != stage.get("file_sha256")
            or cal._json_hash(rows) != stage.get("logical_sha256")
            or rows != prediction_rows(ledger.rows)):
        raise ValueError("predictions.jsonl does not match its hash; refusing to launch SONIC")
    return {"file_sha256": stage["file_sha256"], "logical_sha256": stage["logical_sha256"],
            "n": len(rows)}


# --------------------------------------------------------------------------------- sonic


def tracker_identity() -> dict[str, Any]:
    identity = dict(e22.tracker_identity())
    if identity.get("core_source_manifest_sha256") != EXPECTED_CORE_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "SONIC core source manifest differs from EXP-022A's "
            f"({identity.get('core_source_manifest_sha256')} != "
            f"{EXPECTED_CORE_SOURCE_MANIFEST_SHA256})")
    if identity.get("checkpoint", {}).get("sha256") != EXPECTED_TRACKER_CHECKPOINT_SHA256:
        raise ValueError("SONIC checkpoint sha256 differs from the pinned release checkpoint")
    return identity


def achieved_rows_for(rollouts: Mapping[str, Any], plan: Sequence[Mapping[str, Any]],
                      launch_by_key: Mapping[str, str], schema_by_key: Mapping[str, int],
                      dt_by_key: Mapping[str, float], rows_by_key: Mapping[str, Mapping[str, Any]],
                      *, scorer: Callable[..., Mapping[str, Any]] = e22.score_trajectory,
                      ) -> list[dict[str, Any]]:
    achieved = []
    for item in plan:
        key = str(item["archive_key"])
        if key not in rollouts:
            continue
        rollout = rollouts[key]
        for label, x in OBSTACLES:
            metrics = dict(scorer(rollout.qpos, x, terminated=rollout.terminated,
                                  reported_progress=rollout.progress))
            achieved.append({
                "tier": "achieved", "seed": int(item["seed"]), "arm": str(item["arm"]),
                "motion_key": key, "motion_id": int(rollout.motion_id),
                "launch": launch_by_key[key], "obstacle_label": label, "obstacle_x_m": x,
                "obstacle_depth_m": OBSTACLE_DEPTH_M,
                "source_qpos_content_sha256": rows_by_key[key]["qpos_content_sha256"],
                "archive_schema_version": schema_by_key[key], "sample_dt_s": dt_by_key[key],
                **metrics,
            })
    return achieved


def _resolve_launch_terminations(
    ledger: Ledger, launch: Mapping[str, Any], pkl: Path, *,
    compose_terminations_fn: Callable[[Sequence[str]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compose, audit and persist the resolved termination config of one launch.

    The dump is written beside the launch's motion pickle (``resolved_terminations.json``); a
    resumed launch must reproduce it byte for byte.  A composed config that is not the release
    evaluator stops the campaign; an unavailable mechanism leaves the receipt's TODO in place.
    """
    name = str(launch["name"])
    launch_dir = ledger.output / "launches" / name
    overrides = hydra_overrides(sonic_command(
        pkl, launch_dir / "attempt-NNN" / "eval", int(launch["n_motions"]), PHYSICS_SEED))
    resolved = dict(compose_terminations_fn(overrides))
    config = ledger.receipt["termination_config"]
    if resolved.get("status") != "composed":
        config.setdefault("compose_attempts", {})[name] = resolved
        return {"status": resolved.get("status", "unavailable"), "audit": None}
    audit = audit_release_terminations(resolved)
    payload = {"launch": name, "overrides_note": "eval_output_dir differs per attempt",
               "resolved": resolved, "release_evaluator_audit": audit}
    path = launch_dir / "resolved_terminations.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable resolved termination dump {path}: {exc}") from exc
        if existing != json.loads(cal._canonical_json(payload)):
            raise ValueError(f"{name}: resolved termination config differs from its earlier dump")
    else:
        cal._write_json(path, payload)
    if not audit["is_release_evaluator"]:
        raise ValueError(f"{name}: resolved termination config is not the release evaluator: "
                         + "; ".join(audit["problems"]))
    digest = cal._sha256(path)
    if config.get("resolved_dump") is None:
        config.update({"resolved_dump": resolved, "release_evaluator_audit": audit,
                       "resolved_dump_source": ("exp028_termination_free_rollouts."
                                                "compose_resolved_terminations (offline "
                                                "hydra.compose in the SONIC interpreter)"),
                       "TODO": None})
    elif config["resolved_dump"].get("terminations") != resolved.get("terminations"):
        raise ValueError(f"{name}: resolved terminations differ from the first launch's")
    return {"status": "composed", "path": str(path), "file_sha256": digest, "audit": audit}


def run_sonic(
    *,
    out: str | Path,
    timeout_s: int = 2400,
    resume: bool = False,
    require_committed_predictions: bool = False,
    launch_fn: Callable[..., tuple[int, str]] = launch_sonic,
    export_fn: Callable[..., Path] = write_motion_pkl,
    scorer: Callable[..., Mapping[str, Any]] = e22.score_trajectory,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
    tracker_identity_fn: Callable[[], Mapping[str, Any]] = tracker_identity,
    host_gate_fn: Callable[..., Mapping[str, Any]] = require_host_resources,
    committed_check_fn: Callable[[Path, str], Mapping[str, Any]] = committed_blob_check,
    cuda_context_fn: Callable[[], Mapping[str, Any]] = cuda_context_report,
    compose_terminations_fn: Callable[[Sequence[str]], Mapping[str, Any]] = (
        default_compose_terminations),
    mj_model: Any = None,
) -> dict[str, Any]:
    """Stage 4: four launches of 32 through the EXP-022A bridge harness (release evaluator)."""
    output = Path(out)
    ledger = Ledger.load(output)
    stage = ledger.stage("sonic")
    if stage.get("status") == "complete":
        return ledger.receipt
    if (output / "launches").exists() and not resume:
        raise CampaignAbort(f"{output}/launches exists; pass --resume to continue the SONIC stage")
    stage_label = "sonic_preflight"
    achieved_rows: list[dict[str, Any]] = _read_jsonl(output / "achieved_rows.jsonl")
    try:
        clips = _validate_generation_archive(ledger)
        ledger.require_stage_complete("score")
        predictions = _verify_predictions(ledger)
        cuda = dict(cuda_context_fn())
        if cuda.get("cuda_initialized"):
            raise ValueError("a torch CUDA context is alive in the SONIC stage process; "
                             "run --stage sonic in a fresh process")
        predictions_path = (output / "predictions.jsonl").resolve()
        try:
            relative_path = predictions_path.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            relative_path = str(predictions_path)  # outside the repo: git show cannot match
        committed = dict(committed_check_fn(ROOT, relative_path))
        if require_committed_predictions and not committed.get("matches"):
            raise ValueError("predictions.jsonl is not committed at HEAD byte-for-byte: "
                             f"{committed}")
        ledger.receipt["predictions"]["predictions_committed_before_sonic"] = {
            "asserted": bool(require_committed_predictions), **committed}
        check = _stage_provenance_check(
            ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
            runtime_identity_fn=runtime_identity_fn, physical_identity_fn=physical_identity_fn)
        tracker = dict(tracker_identity_fn())
        plan = locked_row_plan()
        launches = launch_plan(plan)
        if [l["motion_keys"] for l in launches] != [
                l["motion_keys"] for l in ledger.receipt["campaign_design"]["launch_plan"]]:
            raise ValueError("launch plan drifted from the generation-stage receipt")
        stage.update({
            "status": "running", "provenance_check": check, "cuda_context": cuda,
            "predictions_verified": predictions, "resumed": bool(resume),
            "timeout_s": int(timeout_s),
        })
        ledger.receipt["provenance"]["tracker"] = tracker
        ledger.receipt["termination_config"] = termination_config_record(tracker)
        ledger.receipt["termination_config"]["launch_command_template"] = sonic_command(
            Path("<launch>/motions.pkl"), Path("<attempt>/eval"), SONIC_CHUNK_SIZE, PHYSICS_SEED)
        ledger.receipt.setdefault("host_resource_gate", {}).setdefault("sonic", {})
        ledger.persist(stage_label="sonic_running")

        rows_by_key = {str(row["archive_key"]): row for row in ledger.rows}
        model = mj_model if mj_model is not None else G1Body(None).model
        rollouts: dict[str, Any] = {}
        launch_by_key: dict[str, str] = {}
        schema_by_key: dict[str, int] = {}
        dt_by_key: dict[str, float] = {}
        for launch in launches:
            name = str(launch["name"])
            stage_label = f"sonic_{name}"
            existing = ledger.receipt["launches"].get(name)
            if not (isinstance(existing, dict) and existing.get("status") == "complete"):
                try:
                    gate_report = dict(host_gate_fn(require_no_isaac=True))
                except HostResourceGateFailed as exc:
                    raise CampaignAbort(f"{name} refused by the host-resource gate: {exc}") from exc
                ledger.receipt["host_resource_gate"]["sonic"][name] = gate_report
                ledger.persist(stage_label=stage_label)
            pkl = e22.ensure_motion_pkl(launch, clips, output, export_fn=export_fn, mj_model=model)
            resolved = _resolve_launch_terminations(
                ledger, launch, pkl, compose_terminations_fn=compose_terminations_fn)
            record, launch_rollouts = e22.run_or_resume_launch(
                launch, pkl, output, launch_fn=launch_fn, timeout_s=timeout_s)
            record["resolved_terminations"] = resolved
            ledger.receipt["launches"][name] = record
            for rollout in launch_rollouts:
                rollouts[rollout.motion_key] = rollout
                launch_by_key[rollout.motion_key] = name
                schema_by_key[rollout.motion_key] = int(record["archive_schema_version"])
                dt_by_key[rollout.motion_key] = float(record["sample_dt_s"])
            if dict(tracker_identity_fn()) != tracker:
                raise ValueError("SONIC checkout/checkpoint/config changed during execution")
            stage.setdefault("post_launch_revalidation", {})[name] = {
                "tracker_identity_unchanged": True,
                "project": _stage_provenance_check(
                    ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
                    runtime_identity_fn=runtime_identity_fn,
                    physical_identity_fn=physical_identity_fn)["git"],
            }
            achieved_rows = achieved_rows_for(
                rollouts, plan, launch_by_key, schema_by_key, dt_by_key, rows_by_key,
                scorer=scorer)
            cal._write_jsonl(output / "achieved_rows.jsonl", achieved_rows)
            ledger.anchor_file("achieved_rows", output / "achieved_rows.jsonl",
                               n_rows=len(achieved_rows),
                               logical_sha256=cal._json_hash(achieved_rows))
            stage["rollouts_returned"] = len(rollouts)
            ledger.persist(stage_label=stage_label)

        if set(rollouts) != {row["archive_key"] for row in plan}:
            raise ValueError("completed launches do not cover all 128 motions")
        if len(achieved_rows) != N_ROWS * len(OBSTACLES):
            raise ValueError("achieved rows do not cover 128 clips x 2 obstacles")
        stage.update({"status": "complete", "rollouts_requested": N_ROWS,
                      "rollouts_returned": len(rollouts),
                      "terminated": int(sum(r.terminated for r in rollouts.values()))})
        ledger.receipt["sonic_rollouts_requested"] = N_ROWS
        ledger.receipt["sonic_rollouts_returned"] = len(rollouts)
        ledger.receipt["stage"] = "tracked"
        ledger.persist(stage_label="tracked")
        return ledger.receipt
    except Exception as exc:
        ledger.fail("sonic", exc, stage_label)
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# ------------------------------------------------------------------------------- analyze


def run_analyze(
    *,
    out: str | Path,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
) -> dict[str, Any]:
    """Stage 5: guarded retention, the prospective 2x2, P1-P4 and the decision rules."""
    output = Path(out)
    ledger = Ledger.load(output)
    stage = ledger.stage("analyze")
    if stage.get("status") == "complete":
        return ledger.receipt
    try:
        _validate_generation_archive(ledger)
        ledger.require_stage_complete("score")
        ledger.require_stage_complete("predict")
        ledger.require_stage_complete("sonic")
        _verify_predictions(ledger)
        achieved_rows = _read_jsonl(output / "achieved_rows.jsonl")
        anchor = ledger.receipt.get("evidence_anchors", {}).get("achieved_rows", {})
        if (len(achieved_rows) != N_ROWS * len(OBSTACLES)
                or anchor.get("logical_sha256") != cal._json_hash(achieved_rows)
                or anchor.get("file_sha256") != cal._sha256(output / "achieved_rows.jsonl")):
            raise ValueError("achieved_rows.jsonl does not match its evidence anchor")
        check = _stage_provenance_check(
            ledger, code_state_fn=code_state_fn, source_hashes_fn=source_hashes_fn,
            runtime_identity_fn=runtime_identity_fn, physical_identity_fn=physical_identity_fn)
        constructibility = arm_constructibility(ledger.rows)
        if constructibility != ledger.stage("score").get("constructibility"):
            raise ValueError("constructibility recomputed from rows differs from the score stage")
        records = clip_records(ledger.rows, achieved_rows)
        summary = evaluate_decisions(records, constructibility)
        summary["status"] = "complete"
        summary["predictions_file_sha256"] = ledger.stage("predict").get("file_sha256")
        summary["predictions_committed_before_sonic"] = ledger.receipt.get(
            "predictions", {}).get("predictions_committed_before_sonic")
        cal._write_json(output / "summary.json", summary)
        cal._write_jsonl(output / "clip_records.jsonl", records)
        ledger.anchor_file("summary", output / "summary.json",
                           logical_sha256=cal._json_hash(summary))
        ledger.anchor_file("clip_records", output / "clip_records.jsonl", n_rows=len(records),
                           logical_sha256=cal._json_hash(records))
        stage.update({"status": "complete", "provenance_check": check})
        ledger.receipt.update({"summary": summary, "decisions": summary["decisions"],
                               "status": "complete", "complete": True, "blocked": False,
                               "stage": "complete"})
        ledger.persist(stage_label="complete")
        return ledger.receipt
    except Exception as exc:
        ledger.fail("analyze", exc, "analyze")
        if isinstance(exc, CampaignAbort):
            raise
        raise CampaignAbort(str(exc)) from exc


# ------------------------------------------------------------------------------ dry run


def dry_run_report() -> dict[str, Any]:
    """Plan, arm channel usage and the live host gate, without touching disk or the GPU."""
    route = route_xz()
    specs = {arm: arm_spec(arm, route) for arm in ARMS}
    return {
        "schema": SCHEMA_VERSION, "experiment": EXPERIMENT, "status": "dry_run",
        "writes_performed": False,
        "protocol": {"path": PROTOCOL_PATH, "sha256": cal._sha256(ROOT / PROTOCOL_PATH)},
        "chunk_plan": [{k: v for k, v in chunk.items() if k != "rows"}
                       for chunk in locked_chunk_plan()],
        "launch_plan": launch_plan(),
        "arms": {arm: {
            "contract": dict(ARM_CONTRACTS[arm]),
            "spec_sha256": spec_sha256(specs[arm]),
            "adapter_channels_written": static_channel_usage(specs[arm]),
            "expected_model_channel_usage": dict(EXPECTED_CHANNEL_USAGE[arm]),
        } for arm in ARMS},
        "host_resource_gate": {
            "generate": host_resource_report(require_no_isaac=False),
            "sonic": host_resource_report(require_no_isaac=True),
        },
        "sonic_launch_command_template": sonic_command(
            Path("<launch>/motions.pkl"), Path("<attempt>/eval"), SONIC_CHUNK_SIZE, PHYSICS_SEED),
    }


# ------------------------------------------------------------------------------------ CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--resume", action="store_true",
                        help="continue the SONIC stage of an existing directory")
    parser.add_argument("--require-committed-predictions", action="store_true",
                        help="refuse SONIC unless predictions.jsonl is committed at HEAD")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sonic_in_subprocess(args: argparse.Namespace) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "sonic",
           "--out", str(args.out), "--timeout-s", str(args.timeout_s)]
    if args.resume:
        cmd.append("--resume")
    if args.require_committed_predictions:
        cmd.append("--require-committed-predictions")
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise CampaignAbort(f"SONIC stage subprocess returned {completed.returncode}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_report(), indent=2, sort_keys=True))
        return
    stages = list(STAGES) if args.stage == "all" else [args.stage]

    def show(payload: Mapping[str, Any]) -> None:
        print(json.dumps(payload, indent=2, sort_keys=True))

    try:
        for stage in stages:
            if stage == "generate":
                if args.stage == "all" and (Path(args.out) / "receipt.json").is_file():
                    if not args.resume:
                        raise CampaignAbort(
                            f"{args.out} already holds a campaign; pass --resume to continue "
                            "its later stages or choose a fresh --out")
                    continue
                receipt = run_generate(out=args.out, cache_path=args.cache_path)
                show({"stage": "generate", "status": receipt["stages"]["generate"]["status"],
                      "actual_ardy_samples": receipt["actual_ardy_samples"]})
            elif stage == "score":
                score_stage = run_score(out=args.out)["stages"]["score"]
                show({"stage": "score", "constructibility": score_stage["constructibility"],
                      "per_arm": score_stage["reference_summary_per_arm"]})
            elif stage == "predict":
                receipt = run_predict(out=args.out)
                show({"stage": "predict", **receipt["predictions"]})
            elif stage == "sonic":
                if args.stage == "all" and cuda_context_report()["cuda_initialized"]:
                    _sonic_in_subprocess(args)
                else:
                    receipt = run_sonic(
                        out=args.out, timeout_s=args.timeout_s, resume=args.resume,
                        require_committed_predictions=args.require_committed_predictions)
                    show({"stage": "sonic",
                          "rollouts_returned": receipt.get("sonic_rollouts_returned"),
                          "terminated": receipt["stages"]["sonic"].get("terminated")})
            elif stage == "analyze":
                summary = run_analyze(out=args.out)["summary"]
                show({"stage": "analyze", "decisions": summary["decisions"],
                      "p1": summary["p1_prospective_contract"]["rules"]["primary_0p20s"]["table"],
                      "p2": summary["p2_free_replicates_exp021"]["pass"],
                      "p3": summary["p3_pinned_root_arms"]["pass"],
                      "p4": {arm: v["clips"]["k"] for arm, v in
                             summary["p4_prescriptive_contract"]["arms"].items()}})
    except (CampaignAbort, HostResourceGateFailed) as exc:
        raise SystemExit(f"EXP-024 {args.stage}: {exc}")


if __name__ == "__main__":
    main()
