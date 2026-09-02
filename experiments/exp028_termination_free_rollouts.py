"""EXP-028: termination-free SONIC rollouts and physics-seed re-roll of the EXP-021 pool.

Part A replays the 64 archived EXP-021 STEP references through SONIC with every tracking-error
termination raised to an unreachable threshold, so the achieved-state archive spans the whole
reference and each clip receives a *physical* outcome class (fell / stalled / walked_through /
cleared) instead of an evaluator cutoff.  Part B re-rolls the same 64 references at physics
seeds 1 and 2 under the unchanged release evaluator to measure the step family's own
test-retest ceiling.  No new ARDY samples are generated.

Everything here inherits EXP-022A's launch construction, chunk plan, attempt/resume discipline
and exact fixed-centre geometry (``experiments/exp022_exact_tracking_bridge.py``); the landed
EXP-022A receipt binds those sources byte-for-byte, so this module imports them and copies only
the functions that hard-code EXP-022A specifics (one physics seed, one override set).

Stages (``--stage``): ``smoke`` (two-motion launches of both configurations that dump the
resolved termination config), ``part_a``, ``part_b``, ``analyze``, ``all``.  Every stage is
resumable; ``part_a``/``part_b`` refuse to launch until their smoke receipt passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analyze_trackability_contract as contract  # noqa: E402
from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp1b_execution_clearance as exp1b  # noqa: E402
from experiments import exp022_exact_tracking_bridge as exp022  # noqa: E402
from scene2motion import host_gate  # noqa: E402
from scene2motion.robot import ARDY_G1_XML, BODY_MARGIN, G1Body  # noqa: E402
from scene2motion.sonic_export import SONIC_ROOT as EXPORT_SONIC_ROOT  # noqa: E402
from scene2motion.sonic_export import write_motion_pkl  # noqa: E402
from scene2motion.sonic_state_export import (  # noqa: E402
    ARCHIVE_SCHEMA_VERSION,
    QPOS_WIDTH,
    load_sonic_state_rollouts,
    sonic_state_hydra_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUT = exp022.SOURCE_OUT
EXP022A_OUT = ROOT / "outputs/exp022_exact_tracking_bridge"
DEFAULT_OUT = ROOT / "outputs/exp028_termination_free_rollouts"
PROTOCOL_PATH = ROOT / "docs/ramp-exp028-termination-free-rollouts-protocol.md"

SCHEMA_VERSION = "exp028-termination-free-rollouts-v1"
FAILURE_SCHEMA_VERSION = "exp028-termination-free-rollouts-failure-v1"
PROCESS_RESULT_SCHEMA = "exp028-sonic-process-result-v1"
SMOKE_RECEIPT_SCHEMA = "exp028-smoke-receipt-v1"
STAGES = ("smoke", "part_a", "part_b", "analyze", "all")

# Locked external identities (see CLAUDE.md "External pins" and the EXP-022A receipt).
EXPECTED_CORE_MANIFEST_SHA256 = "44e98c45f840ed32cd54d0dbc322e4ed1ef1743625e70ca94c7af01eb70efe0a"
EXPECTED_CHECKPOINT_SHA256 = "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
EXPECTED_G1_XML_SHA256 = "5d76cf92f00dd49d6eb9fae38d7d38e46886848b602ac691051e886c3bcccfb1"
EXP022A_RECEIPT_SHA256 = "af686316883d3fe8ebc64deaca6d76ceeb5bf534ff0f2c7282fc306adc3680f2"
EXP022A_ARCHIVE_SHA256 = {
    "chunk00_seed0": "5c23897ecc8b02aeafc89895aaf88956955c14f2c5f9efe5fe0717a89117f6a6",
    "chunk01_seed0": "73ac1b5277c5c5badda8e31e014ae0301d020053f00c9957ea9627825de5d323",
}
EXP022A_MOTION_PKL_SHA256 = {
    "chunk00_seed0": "d81707e8c2ec2dda92f42773c811070a22727558d4acac124b0cba432291489b",
    "chunk01_seed0": "7ed177e2559539cfeacd8eaaed59c09bb2cf8f52b704e2ab373e8ff771b552d6",
}
THRESHOLD_RECEIPT_PATH = contract.THRESHOLD_RECEIPT
THRESHOLD_RECEIPT_SHA256 = contract.THRESHOLD_RECEIPT_SHA256

# Design constants (protocol "Locked design" / "Endpoints").
POOL_SEEDS = exp022.POOL_SEEDS
N_FRAMES = exp022.N_FRAMES
FPS = exp022.FPS
STAGED_OBSTACLE_X_M = 1.2
OBSTACLE_DEPTH_M = exp022.OBSTACLE_DEPTH_M
GRADED_HEIGHTS_M = exp022.GRADED_HEIGHTS_M
BOX_FRONT_EDGE_X_M = STAGED_OBSTACLE_X_M - OBSTACLE_DEPTH_M / 2  # 1.10 m
FELL_PELVIS_Z_M = 0.50
FELL_UP_Z = 0.70
PART_A_PHYSICS_SEED = 0
PART_B_PHYSICS_SEEDS = (1, 2)
SMOKE_NUM_ENVS = 2
# The two EXP-022A rollouts cut off earliest under the release evaluator (6 and 10 samples):
# if Part A's overrides take effect they must now run the whole reference.  Revalidated
# against the archive at run time.
SMOKE_MOTION_KEYS = ("s4420", "s4406")
SAMPLE_DT_S = exp022.EXPECTED_SAMPLE_DT_S
ACHIEVED_FPS = 1.0 / SAMPLE_DT_S
UNREACHABLE_THRESHOLD = 1.0e6
TIME_OUT_TERM = "time_out"
EXPECTED_TRACKING_TERMS = ("anchor_pos", "anchor_ori_full", "ee_body_pos", "foot_pos_xyz")
# Release evaluator: ``tracking/eval`` merged over ``sonic_release/config.yaml`` (the checkpoint
# contributes ``foot_pos_xyz`` and the ``root_height_threshold`` values).
RELEASE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "anchor_pos": {"threshold": 0.25, "threshold_adaptive": False, "down_threshold": 0.25},
    "anchor_ori_full": {"threshold": 1.0},
    "ee_body_pos": {"threshold": 0.25, "threshold_adaptive": False, "down_threshold": 0.25},
    "foot_pos_xyz": {"threshold": 0.2},
}
TERMINATION_FREE_OVERRIDES: tuple[str, ...] = (
    "++manager_env.terminations.anchor_pos.params.threshold=1e6",
    "++manager_env.terminations.anchor_pos.params.down_threshold=1e6",
    "++manager_env.terminations.anchor_pos.params.threshold_adaptive=false",
    "++manager_env.terminations.anchor_ori_full.params.threshold=1e6",
    "++manager_env.terminations.ee_body_pos.params.threshold=1e6",
    "++manager_env.terminations.ee_body_pos.params.down_threshold=1e6",
    "++manager_env.terminations.ee_body_pos.params.threshold_adaptive=false",
    "++manager_env.terminations.foot_pos_xyz.params.threshold=1e6",
)
ANCHOR_BODY = "pelvis"
EE_HEIGHT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
                    "left_wrist_yaw_link", "right_wrist_yaw_link")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
EVALUATOR_TERM_THRESHOLDS = {
    "anchor_pos": RELEASE_THRESHOLDS["anchor_pos"]["threshold"],
    "anchor_ori_full": RELEASE_THRESHOLDS["anchor_ori_full"]["threshold"],
    "ee_body_pos": RELEASE_THRESHOLDS["ee_body_pos"]["threshold"],
    "foot_pos_xyz": RELEASE_THRESHOLDS["foot_pos_xyz"]["threshold"],
}
OUTCOME_CLASSES = ("fell", "stalled", "walked_through", "cleared")
RESIDUAL_CLASSES = ("residual_stopped_at_box", "residual_bypassed_or_reversed")
ISAACLAB_ROOT = exp022.ISAACLAB_ROOT
CORE_SONIC_FILES = exp022.CORE_SONIC_FILES
EVALUATOR_SONIC_FILES = (
    "gear_sonic/envs/manager_env/mdp/terminations.py",
    "gear_sonic/envs/manager_env/mdp/commands.py",
    "gear_sonic/utils/motion_lib/motion_lib_base.py",
    "gear_sonic/utils/motion_lib/torch_humanoid_batch.py",
    "gear_sonic/trl/utils/torch_transform.py",
    "gear_sonic/config/base_eval.yaml",
    "gear_sonic/config/manager_env/terminations/tracking/base.yaml",
    "gear_sonic/config/manager_env/terminations/terms/anchor_pos.yaml",
    "gear_sonic/config/manager_env/terminations/terms/anchor_ori_full.yaml",
    "gear_sonic/config/manager_env/terminations/terms/ee_body_pos.yaml",
    "gear_sonic/config/manager_env/terminations/terms/foot_pos_xyz.yaml",
    "gear_sonic/config/manager_env/terminations/terms/motion_time_out.yaml",
    "gear_sonic/utils/config_utils.py",
)
SOURCE_FILES = (
    "env.sh",
    "experiments/analyze_trackability_contract.py",
    "experiments/calibrate_ramp_route_phase.py",
    "experiments/exp022_exact_tracking_bridge.py",
    "experiments/exp028_termination_free_rollouts.py",
    "experiments/exp1b_execution_clearance.py",
    "scene2motion/host_gate.py",
    "scene2motion/scenes.py",
    "scene2motion/sonic_export.py",
    "scene2motion/sonic_state_export.py",
    "scene2motion/stepover_eval.py",
    "scene2motion/robot.py",
)

EVALUATOR_FORMULAS = {
    "alignment": (
        "achieved sample i is the state after physics step i+1 (schema v2); SONIC evaluates "
        "terminations after the step and before the command update, so sample i is compared "
        "with 50 Hz reference step i (t = i * 0.02 s), one control step behind the state"
    ),
    "reference_50hz": (
        "SONIC resamples the 25 fps reference to 50 Hz over duration (T-1)/25 s: step k maps to "
        "frame k/2 for even k and to the 0.5 blend of frames (k-1)/2 and (k+1)/2 for odd k "
        "(lerp of root translation and joint angles, slerp of the root quaternion)"
    ),
    "anchor_pos": (
        "exceeded_anchor_height: |ref_pelvis_z(step i) - achieved_pelvis_z(sample i)| > 0.25 m "
        "(threshold_adaptive=false under tracking/eval, so down_threshold is inert)"
    ),
    "anchor_ori_full": (
        "exceeded_anchor_ori: quat_error_magnitude(ref_pelvis_quat(step i), achieved_pelvis_quat"
        "(sample i))^2 > 1.0, i.e. the rotation angle of q_ref * conj(q_achieved) exceeds 1.0 rad"
    ),
    "ee_body_pos": (
        "exceeded_body_height over left/right_ankle_roll_link and left/right_wrist_yaw_link: "
        "|body_pos_relative_w.z - achieved_body_z| > 0.25 m; body_pos_relative_w.z equals the "
        "reference body z because the re-anchoring rotation is yaw-only"
    ),
    "foot_pos_xyz": (
        "exceeded_body_pos over left/right_ankle_roll_link: || delta_pos + R_yaw(delta_ori) "
        "(ref_foot(step i) - ref_pelvis(step i)) - achieved_foot(sample i) || > 0.2 m with "
        "delta_pos = (achieved_pelvis_xy(sample i-1), ref_pelvis_z(step i)) and delta_ori = "
        "heading_q(achieved_pelvis_quat(sample i-1) * conj(ref_pelvis_quat(step i))); for "
        "sample 0 the previous anchor is the reset pose, which equals reference frame 0"
    ),
    "time_out": "tracking_time_out fires when time_steps + 1 >= 2*(T-1); it is not a failure",
    "approximations": [
        "reference and achieved body positions come from MuJoCo forward kinematics through "
        "ARDY's g1.xml (body offsets checked identical to SONIC's g1_29dof_rev_1_0.xml for the "
        "tracked chain) instead of SONIC's motion-library skeleton FK and Isaac's rigid-body "
        "state; both sides share the same FK so systematic offsets cancel",
        "odd 50 Hz reference steps are reconstructed by lerp of joint angles; SONIC slerps "
        "per-body local quaternions, identical for single-axis hinges up to floating point",
        "terminated archives are trimmed to the last alive sample, so the firing sample itself "
        "is absent under the release evaluator; only Part A (no cutoff) contains it",
        "running_ref_root_height (EMA for adaptive thresholds) is not reproduced because "
        "threshold_adaptive is false under tracking/eval",
    ],
    "validation": (
        "on the EXP-022A archives (release evaluator, seed 0) the reproduction must show zero "
        "samples above any threshold, because every archived sample was counted alive; the "
        "analysis stage records this consistency check"
    ),
}


class BridgeAbort(exp022.BridgeAbort):
    """Fail-closed stop after any available evidence has been made durable."""


class CampaignPaused(BridgeAbort):
    """A refusal that leaves the campaign resumable: host gate or stage lock, never evidence."""


_sha256 = exp022._sha256
_read_jsonl = exp022._read_jsonl
_git_state = exp022._git_state
_file_hashes = exp022._file_hashes
validate_project_recheck = exp022.validate_project_recheck
validate_attempt = exp022.validate_attempt
ensure_motion_pkl = exp022.ensure_motion_pkl


# ------------------------------------------------------------------------------ identities

def protocol_identity(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    digest = _sha256(path)
    if digest is None:
        raise ValueError(f"EXP-028 protocol is missing: {path}")
    text = path.read_text()
    match = re.search(r"^\*\*Status:\*\*\s*(\w+)", text, flags=re.MULTILINE)
    status = match.group(1).lower() if match else None
    return {"path": str(path.resolve()), "sha256": digest, "status": status}


def project_identity(
    repo: Path = ROOT, *,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
) -> dict[str, Any]:
    model_path = Path(ARDY_G1_XML)
    model_hash = _sha256(model_path)
    if model_hash != EXPECTED_G1_XML_SHA256:
        raise ValueError(f"g1.xml sha256 {model_hash} differs from the pinned "
                         f"{EXPECTED_G1_XML_SHA256}")
    return {
        "git": dict(code_state_fn(repo)),
        "source_sha256": _file_hashes(repo, SOURCE_FILES),
        "runtime": cal._runtime_identity(),
        "physical_model": {
            "path": str(model_path.resolve()), "sha256": model_hash,
            "body_margin_m": BODY_MARGIN,
        },
    }


def threshold_identity(path: Path = THRESHOLD_RECEIPT_PATH) -> dict[str, Any]:
    digest = _sha256(path)
    if digest != THRESHOLD_RECEIPT_SHA256:
        raise ValueError(f"threshold receipt sha256 {digest} differs from the pinned "
                         f"{THRESHOLD_RECEIPT_SHA256}")
    receipt = json.loads(path.read_text())
    thresholds = receipt["stepover_thresholds"]
    return {
        "path": str(path.resolve()), "sha256": digest,
        "support_height_m": float(thresholds["support_height_m"]),
        "support_speed_mps": float(thresholds["support_speed_mps"]),
        "max_unsupported_run_s": float(thresholds["max_unsupported_run_s"]),
    }


def tracker_identity(sonic_root: str | Path = exp1b.SONIC) -> dict[str, Any]:
    """Bind the tracker like EXP-022A, but tolerate unrelated dirty files.

    The SONIC checkout hosts other research; the receipt records its commit and every dirty
    path, refuses any dirty *core or evaluator* file, and asserts the core source manifest and
    checkpoint hashes equal the EXP-022A values.
    """
    root = Path(sonic_root).resolve()
    if root != Path(EXPORT_SONIC_ROOT).resolve():
        raise ValueError(f"SONIC export root {Path(EXPORT_SONIC_ROOT).resolve()} differs from "
                         f"execution root {root}")
    git = _git_state(root)
    dirty_paths = [line[3:] for line in git["status"] if len(line) >= 4]
    guarded = set(CORE_SONIC_FILES) | set(EVALUATOR_SONIC_FILES)
    dirty_guarded = sorted(path for path in dirty_paths if path in guarded)
    if dirty_guarded:
        raise ValueError(f"SONIC core/evaluator sources are dirty: {dirty_guarded}")
    checkpoint = root / "sonic_release/last.pt"
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"SONIC checkpoint sha256 {checkpoint_sha} differs from the pinned "
                         f"{EXPECTED_CHECKPOINT_SHA256}")
    if not Path(exp1b.SONIC_PY).is_file():
        raise ValueError(f"SONIC Python is missing: {exp1b.SONIC_PY}")
    if not ISAACLAB_ROOT.is_dir():
        raise ValueError(f"Isaac Lab checkout is missing: {ISAACLAB_ROOT}")
    isaac_git = _git_state(ISAACLAB_ROOT)
    if isaac_git["dirty"]:
        raise ValueError("Isaac Lab checkout must be exactly clean for an execution campaign")
    core = _file_hashes(root, CORE_SONIC_FILES)
    manifest = cal._json_hash(core)
    if manifest != EXPECTED_CORE_MANIFEST_SHA256:
        raise ValueError(f"SONIC core source manifest {manifest} differs from EXP-022A's "
                         f"{EXPECTED_CORE_MANIFEST_SHA256}")
    return {
        "root": str(root),
        "git": git,
        "dirty_paths": dirty_paths,
        "core_source_sha256": core,
        "core_source_manifest_sha256": manifest,
        "core_source_manifest_matches_exp022a": True,
        "evaluator_source_sha256": _file_hashes(root, EVALUATOR_SONIC_FILES),
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
                       "sha256": checkpoint_sha},
        "python": str(Path(exp1b.SONIC_PY).resolve()),
        "python_runtime": exp022._sonic_python_runtime(Path(exp1b.SONIC_PY)),
        "isaaclab": {"root": str(ISAACLAB_ROOT.resolve()), "git": isaac_git},
        "expected_achieved_sample_dt_s": SAMPLE_DT_S,
        "callback_schema_version": ARCHIVE_SCHEMA_VERSION,
    }


def bound_tracker_identity(tracker: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of the tracker identity that must not change during the campaign."""
    return {
        "root": tracker["root"],
        "commit": tracker["git"]["commit"],
        "core_source_manifest_sha256": tracker["core_source_manifest_sha256"],
        "evaluator_source_sha256": tracker.get("evaluator_source_sha256"),
        "checkpoint_sha256": tracker["checkpoint"]["sha256"],
        "python_runtime": tracker.get("python_runtime"),
        "isaaclab_commit": tracker.get("isaaclab", {}).get("git", {}).get("commit"),
    }


def exp022a_identity(exp022a_dir: str | Path = EXP022A_OUT) -> dict[str, Any]:
    """Validate the landed EXP-022A campaign that supplies Part B's physics-seed-0 rollouts."""
    directory = Path(exp022a_dir)
    receipt_path = directory / "receipt.json"
    digest = _sha256(receipt_path)
    if digest != EXP022A_RECEIPT_SHA256:
        raise ValueError(f"EXP-022A receipt sha256 {digest} differs from the pinned "
                         f"{EXP022A_RECEIPT_SHA256}")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != exp022.SCHEMA_VERSION or receipt.get("status") != "complete"
            or receipt.get("sonic_rollouts_returned") != 64):
        raise ValueError("EXP-022A receipt is not the complete 64-rollout campaign")
    anchors = receipt.get("evidence_anchors", {})
    for name in ("reference_rows", "achieved_rows"):
        path = directory / f"{name}.jsonl"
        if _sha256(path) != anchors.get(name, {}).get("file_sha256"):
            raise ValueError(f"EXP-022A {name} no longer matches its receipt anchor")
    archives: dict[str, str] = {}
    pkls: dict[str, str] = {}
    for chunk in exp022.chunk_plan():
        name = chunk["name"]
        record = receipt["launches"][name]
        attempt = directory / "launches" / name / Path(record["attempt"]).name
        archive = attempt / "eval/achieved_qpos.npz"
        archive_sha = _sha256(archive)
        recorded = record["artifacts"]["archive_sha256"].get("achieved_qpos.npz")
        if archive_sha != recorded or archive_sha != EXP022A_ARCHIVE_SHA256[name]:
            raise ValueError(f"EXP-022A archive for {name} has sha256 {archive_sha}, expected "
                             f"{EXP022A_ARCHIVE_SHA256[name]} (receipt {recorded})")
        pkl_sha = _sha256(directory / "launches" / name / "motions.pkl")
        if pkl_sha != record["motion_pkl_sha256"] or pkl_sha != EXP022A_MOTION_PKL_SHA256[name]:
            raise ValueError(f"EXP-022A motion pickle for {name} has sha256 {pkl_sha}")
        if record.get("physics_seed") != 0 or record.get("motion_keys") != chunk["motion_keys"]:
            raise ValueError(f"EXP-022A launch {name} is not the locked seed-0 chunk")
        archives[name] = str(archive.resolve())
        pkls[name] = pkl_sha
    return {
        "path": str(directory.resolve()),
        "receipt_sha256": digest,
        "project_commit": receipt.get("provenance", {}).get("project", {}).get("git", {}).get("commit"),
        "tracker_commit": receipt.get("provenance", {}).get("tracker", {}).get("git", {}).get("commit"),
        "archive_paths": archives,
        "archive_sha256": dict(EXP022A_ARCHIVE_SHA256),
        "motion_pkl_sha256": pkls,
        "physics_seed": 0,
    }


def load_exp022a_rollouts(identity: Mapping[str, Any]) -> dict[str, Any]:
    rollouts: dict[str, Any] = {}
    for name, path in identity["archive_paths"].items():
        archive = Path(path)
        if _sha256(archive) != EXP022A_ARCHIVE_SHA256[name]:
            raise ValueError(f"EXP-022A archive {name} changed on disk")
        for rollout in load_sonic_state_rollouts(archive):
            if rollout.motion_key in rollouts:
                raise ValueError(f"EXP-022A archives duplicate {rollout.motion_key}")
            rollouts[rollout.motion_key] = rollout
    if set(rollouts) != {f"s{seed}" for seed in POOL_SEEDS}:
        raise ValueError("EXP-022A archives do not cover the 64 locked motions")
    return rollouts


def full_reference_valid_length(exp022a_rollouts: Mapping[str, Any]) -> int:
    """Valid sample count of a rollout that ran the whole reference (survivors of EXP-022A)."""
    lengths = {int(r.valid_length) for r in exp022a_rollouts.values() if not r.terminated}
    if len(lengths) != 1:
        raise ValueError(f"EXP-022A survivors disagree on the full valid length: {sorted(lengths)}")
    return lengths.pop()


def validate_smoke_keys(exp022a_rollouts: Mapping[str, Any]) -> list[str]:
    """SMOKE_MOTION_KEYS must be the two earliest release-evaluator cutoffs in EXP-022A."""
    terminated = sorted(
        (int(r.valid_length), key) for key, r in exp022a_rollouts.items() if r.terminated)
    earliest = [key for _, key in terminated[:len(SMOKE_MOTION_KEYS)]]
    if earliest != list(SMOKE_MOTION_KEYS):
        raise ValueError(f"smoke keys {SMOKE_MOTION_KEYS} are not the earliest EXP-022A "
                         f"cutoffs {earliest}")
    return earliest


# ------------------------------------------------------------------------------ launch plan

def _launch(name: str, part: str, physics_seed: int, motion_keys: Sequence[str],
            expectation: str, chunk: int | None = None) -> dict[str, Any]:
    extra = list(TERMINATION_FREE_OVERRIDES) if expectation == "termination_free" else []
    return {
        "name": name, "part": part, "chunk": chunk, "physics_seed": int(physics_seed),
        "seeds": [int(key[1:]) for key in motion_keys], "motion_keys": list(motion_keys),
        "n_motions": len(motion_keys), "config_expectation": expectation,
        "extra_overrides": extra,
    }


def smoke_launch_plan() -> list[dict[str, Any]]:
    return [
        _launch(f"smoke_a_seed{PART_A_PHYSICS_SEED}_termfree", "smoke_a", PART_A_PHYSICS_SEED,
                SMOKE_MOTION_KEYS, "termination_free"),
        _launch(f"smoke_b_seed{PART_B_PHYSICS_SEEDS[0]}", "smoke_b", PART_B_PHYSICS_SEEDS[0],
                SMOKE_MOTION_KEYS, "release"),
    ]


def campaign_launch_plan() -> list[dict[str, Any]]:
    """Six campaign launches with EXP-022A's exact clip-to-chunk assignment and key order."""
    chunks = exp022.chunk_plan()
    plan = []
    for chunk in chunks:
        plan.append(_launch(
            f"partA_chunk{chunk['chunk']:02d}_seed{PART_A_PHYSICS_SEED}_termfree", "part_a",
            PART_A_PHYSICS_SEED, chunk["motion_keys"], "termination_free", chunk["chunk"]))
    for physics_seed in PART_B_PHYSICS_SEEDS:
        for chunk in chunks:
            plan.append(_launch(
                f"partB_chunk{chunk['chunk']:02d}_seed{physics_seed}", "part_b", physics_seed,
                chunk["motion_keys"], "release", chunk["chunk"]))
    if len(plan) != 6:
        raise RuntimeError("EXP-028 must contain exactly six campaign launches")
    return plan


def launches_for_part(part: str) -> list[dict[str, Any]]:
    plan = smoke_launch_plan() + campaign_launch_plan()
    return [spec for spec in plan if spec["part"] == part]


# ------------------------------------------------------------------------------ SONIC launch

def build_sonic_command(pkl: Path, eval_dir: Path, num_envs: int, physics_seed: int,
                        extra_overrides: Sequence[str] = ()) -> list[str]:
    """EXP-022A's launcher (``exp1b_execution_clearance.run_sonic``) plus optional overrides."""
    pkl, eval_dir = Path(pkl).resolve(), Path(eval_dir).resolve()
    return [str(exp1b.SONIC_PY), "-u", "-m", "gear_sonic.eval_agent_trl",
            f"+checkpoint={exp1b.CKPT}", "+headless=True", "++eval_callbacks=im_eval",
            "++run_eval_loop=False", f"++num_envs={int(num_envs)}",
            f"++eval_output_dir={eval_dir}",
            f"++seed={int(physics_seed)}",
            "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
            "+manager_env/terminations=tracking/eval",
            f"+manager_env.commands.motion.motion_lib_cfg.motion_file={pkl}",
            f"+log_keys={pkl.stem}",
            *sonic_state_hydra_overrides(),
            *[str(item) for item in extra_overrides]]


def hydra_overrides_of(command: Sequence[str]) -> list[str]:
    index = list(command).index("gear_sonic.eval_agent_trl")
    return list(command[index + 1:])


def launch_sonic(pkl: Path, eval_dir: Path, num_envs: int, physics_seed: int, timeout_s: int,
                 extra_overrides: Sequence[str] = ()) -> tuple[int, str]:
    cmd = build_sonic_command(pkl, eval_dir, num_envs, physics_seed, extra_overrides)
    print("  " + " ".join(cmd[:4]) + " ...", flush=True)
    proc = subprocess.run(cmd, cwd=exp1b.SONIC, capture_output=True, text=True,
                          timeout=timeout_s, env=exp1b.sonic_env(), stdin=subprocess.DEVNULL)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr)


_COMPOSE_MARKER = "EXP028_RESOLVED_TERMINATIONS_JSON="
# Runs inside the SONIC interpreter (CPU only; Isaac is never imported).  It replays
# ``gear_sonic/eval_agent_trl.py``'s composition: Hydra ``base_eval`` + the identical CLI
# overrides, then the checkpoint's ``config.yaml`` (with the release path rewrites and its
# ``eval_overrides``) underneath, then ``train_only_terminations`` removed.
_COMPOSE_SCRIPT = r'''
import io, json, sys
from pathlib import Path
import importlib.metadata as md
import omegaconf
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

payload = json.loads(sys.argv[1])
try:
    from gear_sonic.utils import config_utils
    config_utils.register_rl_resolvers()
    resolvers = "gear_sonic.utils.config_utils.register_rl_resolvers"
except Exception as exc:  # noqa: BLE001
    resolvers = f"unavailable: {type(exc).__name__}: {exc}"
with initialize_config_dir(config_dir=payload["config_dir"], version_base=None):
    override_config = compose(config_name="base_eval", overrides=payload["overrides"])
checkpoint = Path(override_config.checkpoint)
config_path = checkpoint.parent / "config.yaml"
if not config_path.exists():
    config_path = checkpoint.parent.parent / "config.yaml"
raw = config_path.read_text()
raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
raw = raw.replace("groot/rl/data/", "gear_sonic/data/")
raw = raw.replace("assets/bm/unitree_description/", "assets/robot_description/")
raw = raw.replace("1215_bones_seed_filtered", "bones_seed_smpl")
train_config = OmegaConf.load(io.StringIO(raw))
if train_config.eval_overrides is not None:
    train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
config = OmegaConf.merge(train_config, override_config)
removed = []
with omegaconf.open_dict(config):
    for termination in config.manager_env.config.get("train_only_terminations", []):
        if termination in config.manager_env.terminations:
            config.manager_env.terminations.pop(termination)
            removed.append(str(termination))
terms = OmegaConf.to_container(config.manager_env.terminations, resolve=True)
print(MARKER + json.dumps({
    "terminations": terms,
    "checkpoint_config_path": str(config_path),
    "resolvers": resolvers,
    "hydra_version": md.version("hydra-core"),
    "omegaconf_version": md.version("omegaconf"),
    "train_only_terminations_removed": removed,
    "seed": config.get("seed", None),
    "num_envs": config.get("num_envs", None),
}, sort_keys=True))
'''.replace("MARKER", repr(_COMPOSE_MARKER))


def compose_resolved_terminations(
    overrides: Sequence[str], *, sonic_root: Path = exp1b.SONIC,
    python: Path = exp1b.SONIC_PY, timeout_s: int = 600,
) -> dict[str, Any]:
    """Resolve ``manager_env.terminations`` offline through SONIC's own config sources."""
    payload = json.dumps({
        "config_dir": str(Path(sonic_root) / "gear_sonic/config"),
        "overrides": [str(item) for item in overrides],
    })
    try:
        completed = subprocess.run(
            [str(python), "-c", _COMPOSE_SCRIPT, payload], cwd=sonic_root,
            capture_output=True, text=True, timeout=timeout_s, env=exp1b.sonic_env(),
            stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"could not compose the SONIC termination config: {exc}") from exc
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_COMPOSE_MARKER)]
    if completed.returncode != 0 or not lines:
        raise ValueError("SONIC termination-config composition failed: "
                         + completed.stderr.strip()[-2000:])
    try:
        resolved = json.loads(lines[-1][len(_COMPOSE_MARKER):])
    except json.JSONDecodeError as exc:
        raise ValueError("SONIC termination-config composition printed invalid JSON") from exc
    resolved["overrides"] = [str(item) for item in overrides]
    resolved["method"] = ("offline hydra.compose(base_eval, overrides) merged over the "
                          "checkpoint config exactly as gear_sonic/eval_agent_trl.py does")
    return resolved


def active_terms(terminations: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: dict(term) for name, term in terminations.items()
            if isinstance(term, Mapping) and "func" in term}


def override_coverage(terminations: Mapping[str, Any],
                      overrides: Sequence[str] = TERMINATION_FREE_OVERRIDES) -> dict[str, Any]:
    """Every active tracking term's threshold family must be pinned by the override list."""
    pattern = re.compile(r"^\+\+manager_env\.terminations\.(\w+)\.params\.(\w+)=(.+)$")
    pinned: dict[str, dict[str, str]] = {}
    for item in overrides:
        match = pattern.match(item)
        if not match:
            raise ValueError(f"unrecognised termination override: {item}")
        pinned.setdefault(match.group(1), {})[match.group(2)] = match.group(3)
    terms = active_terms(terminations)
    tracking = {name: term for name, term in terms.items() if term.get("time_out") is not True}
    if TIME_OUT_TERM not in terms or terms[TIME_OUT_TERM].get("time_out") is not True:
        raise ValueError("motion_time_out is not an active time-out term")
    if TIME_OUT_TERM in pinned:
        raise ValueError("the override list must leave motion_time_out untouched")
    uncovered = []
    for name, term in tracking.items():
        params = term.get("params", {})
        keys = pinned.get(name, {})
        if float(keys.get("threshold", "0")) < UNREACHABLE_THRESHOLD:
            uncovered.append(f"{name}.threshold")
        if "down_threshold" in params and float(keys.get("down_threshold", "0")) < UNREACHABLE_THRESHOLD:
            uncovered.append(f"{name}.down_threshold")
        if "threshold_adaptive" in params and keys.get("threshold_adaptive", "").lower() != "false":
            uncovered.append(f"{name}.threshold_adaptive")
    if uncovered:
        raise ValueError(f"termination-free override list leaves active terms reachable: {uncovered}")
    unknown = sorted(set(pinned) - set(tracking))
    if unknown:
        raise ValueError(f"override list pins terms that are not active: {unknown}")
    return {"tracking_terms": sorted(tracking), "time_out_term": TIME_OUT_TERM,
            "pinned": pinned, "covered": True}


def audit_terminations(terminations: Mapping[str, Any], expectation: str) -> dict[str, Any]:
    """Assert the resolved config is the termination-free or the release configuration."""
    terms = active_terms(terminations)
    tracking = {name: term for name, term in terms.items() if term.get("time_out") is not True}
    time_outs = sorted(name for name, term in terms.items() if term.get("time_out") is True)
    if time_outs != [TIME_OUT_TERM]:
        raise ValueError(f"resolved config time-out terms {time_outs} != ['{TIME_OUT_TERM}']")
    if set(tracking) != set(EXPECTED_TRACKING_TERMS):
        raise ValueError(f"resolved tracking terms {sorted(tracking)} != "
                         f"{sorted(EXPECTED_TRACKING_TERMS)}")
    observed: dict[str, dict[str, Any]] = {}
    for name, term in tracking.items():
        params = dict(term.get("params", {}))
        observed[name] = {key: params[key] for key in
                          ("threshold", "threshold_adaptive", "down_threshold") if key in params}
        observed[name]["body_names"] = params.get("body_names")
        observed[name]["func"] = term.get("func")
    if expectation == "termination_free":
        override_coverage(terminations)
        for name, params in observed.items():
            if float(params["threshold"]) < UNREACHABLE_THRESHOLD:
                raise ValueError(f"{name}.threshold={params['threshold']} is reachable")
            if "down_threshold" in params and float(params["down_threshold"]) < UNREACHABLE_THRESHOLD:
                raise ValueError(f"{name}.down_threshold={params['down_threshold']} is reachable")
            if params.get("threshold_adaptive") is True:
                raise ValueError(f"{name}.threshold_adaptive is still true")
    elif expectation == "release":
        for name, expected in RELEASE_THRESHOLDS.items():
            got = {key: observed[name].get(key) for key in expected}
            if got != expected:
                raise ValueError(f"release evaluator {name} params {got} != {expected}")
    else:
        raise ValueError(f"unknown termination expectation {expectation!r}")
    expected_bodies = {"ee_body_pos": list(EE_HEIGHT_BODIES), "foot_pos_xyz": list(FOOT_BODIES)}
    for name, bodies in expected_bodies.items():
        if observed[name]["body_names"] != bodies:
            raise ValueError(f"{name} tracks {observed[name]['body_names']}, expected {bodies}")
    return {"expectation": expectation, "active_terms": sorted(terms),
            "tracking_terms": observed, "time_out_term": TIME_OUT_TERM, "pass": True}


_TERM_TABLE_ROW = re.compile(r"^\|\s*\d+\s*\|\s*(\w+)\s*\|\s*(True|False)\s*\|\s*$")


def parse_log_termination_terms(log: str) -> dict[str, bool]:
    """Active termination terms (name -> time_out flag) from SONIC's TerminationManager table."""
    lines = log.splitlines()
    starts = [i for i, line in enumerate(lines) if "Active Termination Terms" in line]
    if len(starts) != 1:
        raise ValueError(f"SONIC log contains {len(starts)} termination tables, expected 1")
    found: dict[str, bool] = {}
    for line in lines[starts[0] + 1:]:
        stripped = line.strip()
        if stripped.startswith("+") or stripped.startswith("| Index"):
            if found and stripped.startswith("+"):
                break
            continue
        match = _TERM_TABLE_ROW.match(stripped)
        if not match:
            break
        found[match.group(1)] = match.group(2) == "True"
    if not found:
        raise ValueError("SONIC log termination table is empty")
    return found


def _overrides_sha256(overrides: Sequence[str]) -> str:
    return cal._json_hash([str(item) for item in overrides])


# ------------------------------------------------------------------------------ attempts

def _write_process_result(attempt: Path, *, returncode: int, log_path: Path,
                          spec: Mapping[str, Any], motion_pkl_sha256: str) -> dict[str, Any]:
    payload = {
        "schema": PROCESS_RESULT_SCHEMA,
        "returncode": int(returncode),
        "returncode_observed": True,
        "sonic_log_sha256": _sha256(log_path),
        "launch": str(spec["name"]),
        "physics_seed": int(spec["physics_seed"]),
        "motion_keys": list(spec["motion_keys"]),
        "motion_pkl_sha256": motion_pkl_sha256,
        "extra_overrides": list(spec["extra_overrides"]),
        "extra_overrides_sha256": _overrides_sha256(spec["extra_overrides"]),
    }
    cal._write_json(attempt / "process_result.json", payload)
    return {**payload, "file_sha256": _sha256(attempt / "process_result.json")}


def _validate_process_result(attempt: Path, *, spec: Mapping[str, Any],
                             motion_pkl_sha256: str) -> dict[str, Any]:
    path = attempt / "process_result.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeAbort(f"valid SONIC artifacts lack durable return-code evidence: {attempt}") from exc
    if not isinstance(payload, dict):
        raise BridgeAbort(f"SONIC process-result receipt is not an object: {attempt}")
    expected = {
        "schema": PROCESS_RESULT_SCHEMA,
        "returncode_observed": True,
        "sonic_log_sha256": _sha256(attempt / "sonic.log"),
        "launch": str(spec["name"]),
        "physics_seed": int(spec["physics_seed"]),
        "motion_keys": list(spec["motion_keys"]),
        "motion_pkl_sha256": motion_pkl_sha256,
        "extra_overrides": list(spec["extra_overrides"]),
        "extra_overrides_sha256": _overrides_sha256(spec["extra_overrides"]),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise BridgeAbort(f"SONIC process-result receipt has mismatched {field}: {attempt}")
    returncode = payload.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise BridgeAbort(f"SONIC process-result return code is invalid: {attempt}")
    if returncode != 0:
        raise BridgeAbort(f"SONIC process returned {returncode}: {attempt}")
    return {**payload, "file_sha256": _sha256(path)}


def _attempt_expectations(spec: Mapping[str, Any], pkl_sha256: str) -> dict[str, Any]:
    return {
        "launch": spec["name"], "physics_seed": int(spec["physics_seed"]),
        "motion_keys": list(spec["motion_keys"]), "motion_pkl_sha256": pkl_sha256,
        "extra_overrides": list(spec["extra_overrides"]),
        "config_expectation": spec["config_expectation"],
    }


def _validate_resolved_terminations(attempt: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = attempt / "resolved_terminations.json"
    try:
        resolved = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeAbort(f"attempt lacks a readable resolved termination config: {attempt}") from exc
    expected_overrides = hydra_overrides_of(build_sonic_command(
        attempt.parent / "motions.pkl", attempt / "eval", spec["n_motions"],
        spec["physics_seed"], spec["extra_overrides"]))
    if resolved.get("overrides") != expected_overrides:
        raise BridgeAbort(f"resolved termination config was composed from different overrides: "
                          f"{attempt}")
    try:
        audit = audit_terminations(resolved["terminations"], spec["config_expectation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BridgeAbort(f"{spec['name']}: resolved termination config fails its "
                          f"{spec['config_expectation']} audit: {exc}") from exc
    return {"file_sha256": _sha256(path), "audit": audit,
            "hydra_version": resolved.get("hydra_version"),
            "omegaconf_version": resolved.get("omegaconf_version"),
            "method": resolved.get("method")}


def _check_rollouts_against_expectation(spec: Mapping[str, Any], rollouts: Sequence[Any],
                                        full_valid_length: int | None) -> dict[str, Any]:
    """Termination-free launches must return every rollout alive for the whole reference."""
    terminated = [r.motion_key for r in rollouts if r.terminated]
    lengths = {r.motion_key: int(r.valid_length) for r in rollouts}
    if spec["config_expectation"] == "termination_free":
        if terminated:
            raise BridgeAbort(f"{spec['name']}: SONIC still terminated {terminated} although "
                              "every tracking-error term was raised to an unreachable threshold")
        if full_valid_length is not None:
            short = {key: n for key, n in lengths.items() if n != full_valid_length}
            if short:
                raise BridgeAbort(f"{spec['name']}: rollouts did not span the whole reference "
                                  f"({full_valid_length} samples): {short}")
    return {"terminated_keys": terminated, "valid_lengths": lengths,
            "full_valid_length_expected": full_valid_length}


def run_or_resume_launch(
    spec: Mapping[str, Any], pkl: Path, output: Path, *,
    launch_fn: Callable[..., tuple[int, str]], timeout_s: int,
    compose_fn: Callable[[Sequence[str]], Mapping[str, Any]] = compose_resolved_terminations,
    host_gate_fn: Callable[..., Mapping[str, Any]] = host_gate.require_host_resources,
    full_valid_length: int | None = None,
    rewrite_receipt: bool = True,
) -> tuple[dict[str, Any], list[Any]]:
    """EXP-022A's attempt discipline with per-launch seed, overrides, config dump and gate.

    ``rewrite_receipt=False`` adopts a completed attempt read-only (the analysis stage must
    never touch launch evidence); the default mirrors EXP-022A's resume, which stamps the
    adopted attempt receipt with ``recovered_or_resumed``.
    """
    launch_dir = output / "launches" / str(spec["name"])
    attempts = sorted(path for path in launch_dir.glob("attempt-*") if path.is_dir())
    pkl_sha256 = _sha256(pkl)
    if pkl_sha256 is None:
        raise BridgeAbort(f"SONIC motion pickle disappeared: {pkl}")
    expectations = _attempt_expectations(spec, pkl_sha256)

    attempt_infos: list[tuple[Path, dict[str, Any] | None]] = []
    for attempt in attempts:
        receipt_path = attempt / "receipt.json"
        attempt_receipt = None
        if receipt_path.is_file():
            try:
                attempt_receipt = json.loads(receipt_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise BridgeAbort(f"unreadable attempt receipt requires inspection: {attempt}") from exc
            if not isinstance(attempt_receipt, dict):
                raise BridgeAbort(f"attempt receipt is not an object: {attempt}")
        elif any(attempt.iterdir()):
            raise BridgeAbort(f"SONIC attempt has artifacts but no pre-launch receipt: {attempt}")
        if attempt_receipt and attempt_receipt.get("status") == "failed":
            raise BridgeAbort(f"{spec['name']} has a recorded failed SONIC attempt; "
                              "use a fresh campaign output")
        if attempt_receipt is not None:
            if attempt_receipt.get("status") not in {"running", "complete"}:
                raise BridgeAbort(f"attempt has an unknown status: {attempt}")
            for field, expected in expectations.items():
                if attempt_receipt.get(field) != expected:
                    raise BridgeAbort(f"attempt {attempt} has mismatched {field}; "
                                      "refusing provenance relabel")
        attempt_infos.append((attempt, attempt_receipt))

    valid: list[tuple[Path, dict[str, Any], dict[str, Any], list[Any]]] = []
    for attempt, attempt_receipt in attempt_infos:
        if attempt_receipt is None:
            continue
        process_result = None
        if (attempt / "process_result.json").is_file():
            process_result = _validate_process_result(attempt, spec=spec, motion_pkl_sha256=pkl_sha256)
        try:
            record, rollouts = validate_attempt(attempt, spec["motion_keys"])
        except (OSError, ValueError):
            if attempt_receipt.get("status") == "complete" or process_result is not None:
                raise BridgeAbort(f"completed attempt is now invalid: {attempt}")
            continue
        if process_result is None:
            raise BridgeAbort(f"valid SONIC artifacts lack durable return-code evidence: {attempt}")
        if attempt_receipt.get("status") == "complete":
            if (attempt_receipt.get("returncode_observed") is not True
                    or attempt_receipt.get("returncode") != 0):
                raise BridgeAbort(f"completed attempt has invalid return-code evidence: {attempt}")
            if attempt_receipt.get("process_result") != process_result:
                raise BridgeAbort(f"completed attempt changed its process result: {attempt}")
            for field in ("artifacts", "archive_schema_version", "sample_dt_s",
                          "motion_id_key_map_sha256", "n_rollouts"):
                if attempt_receipt.get(field) != record.get(field):
                    raise BridgeAbort(f"completed attempt changed its {field}: {attempt}")
        valid.append((attempt, record, process_result, rollouts))

    if len(valid) > 1:
        raise BridgeAbort(f"{spec['name']} has multiple complete SONIC attempts; "
                          "refusing ambiguous evidence")
    if valid:
        attempt, record, process_result, rollouts = valid[0]
        resolved = _validate_resolved_terminations(attempt, spec)
        log_terms = parse_log_termination_terms((attempt / "sonic.log").read_text())
        if sorted(log_terms) != resolved["audit"]["active_terms"]:
            raise BridgeAbort(f"{spec['name']}: SONIC log terms {sorted(log_terms)} differ from "
                              f"the resolved config {resolved['audit']['active_terms']}")
        rollout_check = _check_rollouts_against_expectation(spec, rollouts, full_valid_length)
        gate_path = attempt / "host_resource_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
        record["recovered_or_resumed"] = True
        record.update({
            "status": "complete", **expectations,
            "returncode": 0, "returncode_observed": True, "process_result": process_result,
            "resolved_terminations": resolved, "log_termination_terms": log_terms,
            "rollout_expectation_check": rollout_check, "host_resource_gate": gate,
        })
        if rewrite_receipt:
            cal._write_json(attempt / "receipt.json", record)
        record["attempt_receipt_sha256"] = _sha256(attempt / "receipt.json")
        return record, rollouts

    # The host gate and the offline config composition run before the attempt directory
    # exists: a failed gate or an unusable composition leaves no attempt behind, so the same
    # campaign directory can be resumed once the host is free.
    attempt = launch_dir / f"attempt-{len(attempts):03d}"
    gate = dict(host_gate_fn(require_no_isaac=True))
    command = build_sonic_command(pkl, attempt / "eval", spec["n_motions"],
                                  spec["physics_seed"], spec["extra_overrides"])
    resolved_raw = dict(compose_fn(hydra_overrides_of(command)))
    try:
        audit_terminations(resolved_raw["terminations"], spec["config_expectation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BridgeAbort(f"{spec['name']}: resolved termination config fails its "
                          f"{spec['config_expectation']} audit before launch: {exc}") from exc
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cal._write_json(attempt / "receipt.json", {"status": "running", **expectations})
    try:
        cal._write_json(attempt / "host_resource_gate.json", gate)
        cal._write_json(attempt / "resolved_terminations.json", resolved_raw)
        resolved = _validate_resolved_terminations(attempt, spec)
        cal._write_json(attempt / "command.json", {"command": command})
        rc, log = launch_fn(pkl, attempt / "eval", int(spec["n_motions"]),
                            int(spec["physics_seed"]), timeout_s, list(spec["extra_overrides"]))
        exp022._atomic_text(attempt / "sonic.log", log)
        process_result = _write_process_result(
            attempt, returncode=rc, log_path=attempt / "sonic.log", spec=spec,
            motion_pkl_sha256=pkl_sha256)
        if rc != 0:
            raise RuntimeError(f"SONIC returned {rc}")
        record, rollouts = validate_attempt(attempt, spec["motion_keys"])
        log_terms = parse_log_termination_terms(log)
        if sorted(log_terms) != resolved["audit"]["active_terms"]:
            raise RuntimeError(f"SONIC log terms {sorted(log_terms)} differ from the resolved "
                               f"config {resolved['audit']['active_terms']}")
        rollout_check = _check_rollouts_against_expectation(spec, rollouts, full_valid_length)
        record.update({
            "status": "complete", **expectations, "returncode": rc,
            "elapsed_s": float(time.monotonic() - started), "recovered_or_resumed": False,
            "returncode_observed": True, "process_result": process_result,
            "resolved_terminations": resolved, "log_termination_terms": log_terms,
            "rollout_expectation_check": rollout_check, "host_resource_gate": gate,
            "command_sha256": _sha256(attempt / "command.json"),
        })
        cal._write_json(attempt / "receipt.json", record)
        record["attempt_receipt_sha256"] = _sha256(attempt / "receipt.json")
        return record, rollouts
    except Exception as exc:
        cal._write_json(attempt / "receipt.json", {
            "status": "failed", **expectations, "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_s": float(time.monotonic() - started),
        })
        raise BridgeAbort(f"SONIC launch {spec['name']} failed: {exc}") from exc


def _forbid_launch(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
    raise BridgeAbort("the analysis stage must never launch SONIC; a launch is incomplete")


def load_completed_launch(spec: Mapping[str, Any], output: Path, *,
                          full_valid_length: int | None) -> tuple[dict[str, Any], list[Any]]:
    """Adopt a completed launch's archive through the full attempt audit, never launching."""
    return run_or_resume_launch(
        spec, output / "launches" / str(spec["name"]) / "motions.pkl", output,
        launch_fn=_forbid_launch, timeout_s=0, compose_fn=_forbid_launch,
        host_gate_fn=_forbid_launch, full_valid_length=full_valid_length,
        rewrite_receipt=False)


# ------------------------------------------------------------------------------ geometry

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[0], q[1:]
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def quat_error_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Isaac Lab ``quat_error_magnitude``: |log(q1 * conj(q2))| with the sign normalised."""
    diff = _quat_mul(np.asarray(q1, float), _quat_conj(np.asarray(q2, float)))
    if diff[0] < 0:
        diff = -diff
    return float(2.0 * np.arctan2(np.linalg.norm(diff[1:]), diff[0]))


def heading_quat(q: np.ndarray) -> np.ndarray:
    """SONIC ``get_heading_q``: zero the x/y components and renormalise (yaw only)."""
    h = np.array([q[0], 0.0, 0.0, q[3]], dtype=float)
    return h / np.linalg.norm(h)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def up_z(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, float)
    return 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)


def resample_reference_50hz(qpos_25fps: np.ndarray) -> np.ndarray:
    """SONIC's 25->50 fps reference grid: 2*(T-1) steps, midpoints at odd steps."""
    q = np.asarray(qpos_25fps, float)
    if q.ndim != 2 or q.shape[1] != QPOS_WIDTH or len(q) < 2:
        raise ValueError(f"reference qpos must be (T>=2, {QPOS_WIDTH}), got {q.shape}")
    out = np.empty((2 * (len(q) - 1), QPOS_WIDTH), dtype=float)
    out[0::2] = q[:-1]
    mid = 0.5 * (q[:-1] + q[1:])
    for i in range(len(q) - 1):
        mid[i, 3:7] = _slerp(q[i, 3:7], q[i + 1, 3:7], 0.5)
    out[1::2] = mid
    return out


class EvaluatorGeometry:
    """MuJoCo FK of the bodies SONIC's termination terms read (see EVALUATOR_FORMULAS)."""

    BODIES = (ANCHOR_BODY,) + EE_HEIGHT_BODIES

    def __init__(self, body: G1Body | None = None):
        import mujoco
        self.body = body if body is not None else G1Body(None)
        self.ids = {name: int(mujoco.mj_name2id(self.body.model, mujoco.mjtObj.mjOBJ_BODY, name))
                    for name in self.BODIES}
        missing = [name for name, index in self.ids.items() if index < 0]
        if missing:
            raise ValueError(f"g1.xml lacks the evaluator bodies {missing}")

    def positions(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        self.body.fk(np.asarray(qpos, float))
        return {name: self.body.data.xpos[index].copy() for name, index in self.ids.items()}


def evaluator_terms(geometry: EvaluatorGeometry, reference_50hz: np.ndarray,
                    achieved: np.ndarray, *, thresholds: Mapping[str, float] = EVALUATOR_TERM_THRESHOLDS,
                    sample_dt_s: float = SAMPLE_DT_S) -> dict[str, Any]:
    """Per-sample offline recomputation of SONIC's four tracking-error terminations."""
    ref = np.asarray(reference_50hz, float)
    ach = np.asarray(achieved, float)
    n = len(ach)
    if n == 0:
        raise ValueError("achieved trajectory is empty")
    if n > len(ref):
        raise ValueError(f"achieved trajectory ({n} samples) outruns the 50 Hz reference ({len(ref)})")
    values = {name: np.zeros(n, dtype=float) for name in thresholds}
    ref_pos = [geometry.positions(ref[i]) for i in range(n)]
    ach_pos = [geometry.positions(ach[i]) for i in range(n)]
    for i in range(n):
        values["anchor_pos"][i] = abs(ref[i, 2] - ach[i, 2])
        values["anchor_ori_full"][i] = quat_error_angle(ref[i, 3:7], ach[i, 3:7]) ** 2
        values["ee_body_pos"][i] = max(abs(ref_pos[i][b][2] - ach_pos[i][b][2]) for b in EE_HEIGHT_BODIES)
        previous = ach[i - 1] if i > 0 else ref[0]
        delta_pos = np.array([previous[0], previous[1], ref[i, 2]])
        delta_ori = heading_quat(_quat_mul(previous[3:7], _quat_conj(ref[i, 3:7])))
        values["foot_pos_xyz"][i] = max(
            np.linalg.norm(delta_pos + _quat_rotate(delta_ori, ref_pos[i][b] - ref_pos[i][ANCHOR_BODY])
                           - ach_pos[i][b])
            for b in FOOT_BODIES)
    exceeded = {name: values[name] > float(thresholds[name]) for name in thresholds}
    per_term = {}
    for name in thresholds:
        hits = np.flatnonzero(exceeded[name])
        first = int(hits[0]) if len(hits) else None
        per_term[name] = {
            "threshold": float(thresholds[name]),
            "max_value": float(values[name].max()),
            "max_ratio_to_threshold": float(values[name].max() / thresholds[name]),
            "n_samples_exceeding": int(len(hits)),
            "first_exceed_sample": first,
            "first_exceed_time_s": None if first is None else float((first + 1) * sample_dt_s),
        }
    firsts = [(info["first_exceed_sample"], name) for name, info in per_term.items()
              if info["first_exceed_sample"] is not None]
    if firsts:
        first_sample = min(sample for sample, _ in firsts)
        firing = sorted(name for sample, name in firsts if sample == first_sample)
    else:
        first_sample, firing = None, []
    last = {name: float(values[name][-1] / thresholds[name]) for name in thresholds}
    return {
        "values": values,
        "exceeded": exceeded,
        "per_term": per_term,
        "first_firing_sample": first_sample,
        "first_firing_time_s": None if first_sample is None else float((first_sample + 1) * sample_dt_s),
        "first_firing_reference_step": first_sample,
        "first_firing_terms": firing,
        "any_term_exceeded": bool(firsts),
        "closest_term_at_last_sample": max(last, key=last.get),
        "ratio_to_threshold_at_last_sample": last,
        "n_samples": n,
    }


# ------------------------------------------------------------------------------ outcomes

def fall_detection(qpos: np.ndarray) -> dict[str, Any]:
    q = np.asarray(qpos, float)
    pelvis_z = q[:, 2]
    up = up_z(q[:, 3:7])
    low = pelvis_z < FELL_PELVIS_Z_M
    tilted = up < FELL_UP_Z
    hits = np.flatnonzero(low | tilted)
    first = int(hits[0]) if len(hits) else None
    return {
        "fell": bool(len(hits)),
        "fell_first_sample": first,
        "fell_first_time_s": None if first is None else float((first + 1) * SAMPLE_DT_S),
        "fell_by_pelvis_height": bool(low.any()),
        "fell_by_tilt": bool(tilted.any()),
        "min_pelvis_z_m": float(pelvis_z.min()),
        "min_up_z": float(up.min()),
    }


def classify_outcome(*, fell: bool, reached_front_edge: bool, passed_obstacle: bool,
                     passed_within_corridor: bool, finished_beyond: bool,
                     exact_clear: bool) -> str:
    """Preregistered order: fell > stalled > walked_through > cleared (+ named residuals)."""
    if fell:
        return "fell"
    if not reached_front_edge:
        return "stalled"
    if passed_obstacle and not exact_clear:
        return "walked_through"
    if passed_obstacle and passed_within_corridor and finished_beyond and exact_clear:
        return "cleared"
    if not passed_obstacle:
        return "residual_stopped_at_box"
    return "residual_bypassed_or_reversed"


def physical_outcome(qpos: np.ndarray, *, scorer: Callable[..., Mapping[str, Any]] = exp022.score_trajectory,
                     obstacle_x_m: float = STAGED_OBSTACLE_X_M) -> dict[str, Any]:
    """Physical outcome class per graded height for one full-length achieved trajectory."""
    q = np.asarray(qpos, float)
    fall = fall_detection(q)
    score = dict(scorer(q, obstacle_x_m, terminated=False, reported_progress=1.0))
    max_x = float(q[:, 0].max())
    reached = bool(max_x >= BOX_FRONT_EDGE_X_M)
    classes = {}
    for height in GRADED_HEIGHTS_M:
        key = f"{height:g}"
        classes[key] = classify_outcome(
            fell=fall["fell"], reached_front_edge=reached,
            passed_obstacle=bool(score["passed_obstacle"]),
            passed_within_corridor=bool(score["passed_within_lateral_corridor"]),
            finished_beyond=bool(score["finished_beyond_obstacle"]),
            exact_clear=bool(score["exact_clears"][key]))
    physical_state = "fell" if fall["fell"] else ("stalled" if not reached else "reached_box")
    return {
        **fall,
        "physical_state": physical_state,
        "reached_box_front_edge": reached,
        "box_front_edge_x_m": BOX_FRONT_EDGE_X_M,
        "max_root_x_m": max_x,
        "final_root_x_m": float(q[-1, 0]),
        "max_lateral_deviation_m": float(np.abs(q[:, 1]).max()),
        "collides_at_height": {f"{h:g}": not bool(score["exact_clears"][f"{h:g}"])
                               for h in GRADED_HEIGHTS_M},
        "outcome_class": classes,
        "staged_score": score,
    }


def wilson(k: int, n: int) -> list[float] | None:
    return None if n == 0 else contract.wilson(int(k), int(n))


def agreement_matrix(labels_by_seed: Mapping[int, Mapping[str, bool]]) -> dict[str, Any]:
    seeds = sorted(labels_by_seed)
    keys = sorted(set.intersection(*(set(labels_by_seed[s]) for s in seeds)))
    n = len(keys)
    pairwise: dict[str, Any] = {}
    matrix = []
    for a in seeds:
        row = []
        for b in seeds:
            agree = sum(bool(labels_by_seed[a][k]) == bool(labels_by_seed[b][k]) for k in keys)
            row.append(agree / n if n else None)
            if a < b:
                pairwise[f"{a}-{b}"] = {"agree": agree, "n": n, "fraction": agree / n if n else None,
                                        "wilson95": wilson(agree, n)}
        matrix.append(row)
    unanimous = sum(len({bool(labels_by_seed[s][k]) for s in seeds}) == 1 for k in keys)
    per_seed = {str(s): {"terminated": sum(bool(v) for v in labels_by_seed[s].values()), "n": n}
                for s in seeds}
    pair_fractions = [p["fraction"] for p in pairwise.values() if p["fraction"] is not None]
    return {
        "seeds": seeds, "n_clips": n, "matrix": matrix, "pairwise": pairwise,
        "unanimous": {"agree": unanimous, "n": n, "fraction": unanimous / n if n else None,
                      "wilson95": wilson(unanimous, n)},
        "mean_pairwise_fraction": float(np.mean(pair_fractions)) if pair_fractions else None,
        "terminated_per_seed": per_seed,
        "agreement_definition": ("unanimous = identical termination label across all seeds; "
                                 "pairwise = label agreement between two seeds"),
    }


# ------------------------------------------------------------------------------ rows

def _contract_features(body: G1Body, qpos: np.ndarray, thresholds: Mapping[str, Any],
                       fps: float) -> dict[str, Any]:
    return contract.features(body, np.asarray(qpos, float),
                             float(thresholds["support_height_m"]),
                             float(thresholds["support_speed_mps"]), fps=fps)


def build_part_a_rows(
    rollouts: Mapping[str, Any], *, clips: Mapping[str, np.ndarray],
    source_rows: Mapping[int, Mapping[str, Any]], thresholds: Mapping[str, Any],
    geometry: EvaluatorGeometry | None, body: G1Body | None,
    scorer: Callable[..., Mapping[str, Any]] = exp022.score_trajectory,
    contract_fn: Callable[..., Mapping[str, Any]] | None = None,
    evaluator_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    term_arrays: dict[str, np.ndarray] = {}
    for seed in POOL_SEEDS:
        key = f"s{seed}"
        rollout = rollouts[key]
        achieved = np.asarray(rollout.qpos, float)
        reference = np.asarray(clips[key], float)
        outcome = physical_outcome(achieved, scorer=scorer)
        reference_score = dict(scorer(reference, STAGED_OBSTACLE_X_M, terminated=False,
                                      reported_progress=1.0))
        row = {
            "part": "A", "seed": seed, "motion_key": key, "motion_id": int(rollout.motion_id),
            "physics_seed": PART_A_PHYSICS_SEED, "config": "termination_free",
            "tracker_terminated": bool(rollout.terminated),
            "tracker_reported_progress": float(rollout.progress),
            "valid_frames": int(rollout.valid_length),
            "valid_time_s": float(rollout.valid_length * SAMPLE_DT_S),
            "sample_dt_s": SAMPLE_DT_S,
            "source_qpos_content_sha256": source_rows[seed]["qpos_content_sha256"],
            "reference_exact_clears": reference_score["exact_clears"],
            "obstacle_x_m": STAGED_OBSTACLE_X_M, "obstacle_depth_m": OBSTACLE_DEPTH_M,
            **{k: v for k, v in outcome.items() if k != "staged_score"},
            "achieved_staged_score": outcome["staged_score"],
        }
        if contract_fn is not None:
            row["achieved_contract"] = dict(contract_fn(achieved, ACHIEVED_FPS))
            row["reference_contract"] = dict(contract_fn(reference, float(FPS)))
        elif body is not None:
            row["achieved_contract"] = _contract_features(body, achieved, thresholds, ACHIEVED_FPS)
            row["reference_contract"] = _contract_features(body, reference, thresholds, float(FPS))
        if evaluator_fn is not None:
            terms = dict(evaluator_fn(reference, achieved))
        elif geometry is not None:
            terms = evaluator_terms(geometry, resample_reference_50hz(reference), achieved)
        else:
            terms = None
        if terms is not None:
            row["evaluator_terms"] = {k: v for k, v in terms.items() if k not in ("values", "exceeded")}
            for name, values in terms.get("values", {}).items():
                term_arrays[f"{key}__{name}"] = np.asarray(values, dtype=np.float32)
        rows.append(row)
    return rows, term_arrays


def build_part_b_rows(
    rollouts_by_seed: Mapping[int, Mapping[str, Any]], *, clips: Mapping[str, np.ndarray],
    source_rows: Mapping[int, Mapping[str, Any]], geometry: EvaluatorGeometry | None,
    scorer: Callable[..., Mapping[str, Any]] = exp022.score_trajectory,
    evaluator_fn: Callable[..., Mapping[str, Any]] | None = None,
    seed_sources: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for physics_seed in sorted(rollouts_by_seed):
        rollouts = rollouts_by_seed[physics_seed]
        for seed in POOL_SEEDS:
            key = f"s{seed}"
            rollout = rollouts[key]
            achieved = np.asarray(rollout.qpos, float)
            reference = np.asarray(clips[key], float)
            score = dict(scorer(achieved, STAGED_OBSTACLE_X_M, terminated=bool(rollout.terminated),
                                reported_progress=float(rollout.progress)))
            reference_score = dict(scorer(reference, STAGED_OBSTACLE_X_M, terminated=False,
                                          reported_progress=1.0))
            row = {
                "part": "B", "seed": seed, "motion_key": key, "motion_id": int(rollout.motion_id),
                "physics_seed": int(physics_seed), "config": "release",
                "rollout_source": (seed_sources or {}).get(physics_seed, "exp028_launch"),
                "tracker_terminated": bool(rollout.terminated),
                "tracker_reported_progress": float(rollout.progress),
                "valid_frames": int(rollout.valid_length),
                "valid_time_s": float(rollout.valid_length * SAMPLE_DT_S),
                "sample_dt_s": SAMPLE_DT_S,
                "source_qpos_content_sha256": source_rows[seed]["qpos_content_sha256"],
                "reference_exact_clears": reference_score["exact_clears"],
                "obstacle_x_m": STAGED_OBSTACLE_X_M, "obstacle_depth_m": OBSTACLE_DEPTH_M,
                "achieved_staged_score": score,
                "achieved_replay_clear_after_passing": score["achieved_replay_clear_after_passing"],
            }
            if len(achieved):
                if evaluator_fn is not None:
                    terms = dict(evaluator_fn(reference, achieved))
                elif geometry is not None:
                    terms = evaluator_terms(geometry, resample_reference_50hz(reference), achieved)
                else:
                    terms = None
                if terms is not None:
                    row["evaluator_terms"] = {k: v for k, v in terms.items()
                                              if k not in ("values", "exceeded")}
            rows.append(row)
    return rows


def summarize_part_a(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = list(OUTCOME_CLASSES) + list(RESIDUAL_CLASSES)
    by_height: dict[str, Any] = {}
    subset_by_height: dict[str, Any] = {}
    for height in GRADED_HEIGHTS_M:
        key = f"{height:g}"
        counts = {c: sum(row["outcome_class"][key] == c for row in rows) for c in classes}
        by_height[key] = {"n": len(rows), **counts,
                          "cleared_wilson95": wilson(counts["cleared"], len(rows))}
        subset = [row for row in rows if row["reference_exact_clears"][key]]
        sub_counts = {c: sum(row["outcome_class"][key] == c for row in subset) for c in classes}
        subset_by_height[key] = {
            "n_reference_exact_clear": len(subset), **sub_counts,
            "fell_or_walked_through": sub_counts["fell"] + sub_counts["walked_through"],
            "seeds": [row["seed"] for row in subset],
            "classes": {row["motion_key"]: row["outcome_class"][key] for row in subset},
        }
    firing = {}
    for row in rows:
        terms = row.get("evaluator_terms")
        if not terms:
            continue
        label = "+".join(terms["first_firing_terms"]) if terms["first_firing_terms"] else "none"
        firing[label] = firing.get(label, 0) + 1
    states = {s: sum(row["physical_state"] == s for row in rows) for s in ("fell", "stalled", "reached_box")}
    return {
        "n": len(rows),
        "physical_state": states,
        "tracker_terminated": sum(bool(row["tracker_terminated"]) for row in rows),
        "outcome_class_by_height": by_height,
        "reference_exact_clear_subset_by_height": subset_by_height,
        "evaluator_first_firing_terms": firing,
        "first_firing_time_s": [row["evaluator_terms"]["first_firing_time_s"] for row in rows
                                if row.get("evaluator_terms")],
        "class_definitions": {
            "fell": f"pelvis z < {FELL_PELVIS_Z_M} m or up-vector z < {FELL_UP_Z} at any sample",
            "stalled": f"max root x < {BOX_FRONT_EDGE_X_M} m (box front edge)",
            "walked_through": "passed the expanded rear edge with whole-body collision at the height",
            "cleared": "passed inside the lateral corridor, finished beyond, exactly collision-free",
            "residual_stopped_at_box": "reached the front edge but never passed the rear edge "
                                       "(not one of the four preregistered classes)",
            "residual_bypassed_or_reversed": "passed collision-free but outside the corridor or "
                                             "reversed (not one of the four preregistered classes)",
        },
    }


def summarize_part_b(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted({int(row["physics_seed"]) for row in rows})
    labels = {s: {row["motion_key"]: bool(row["tracker_terminated"]) for row in rows
                  if int(row["physics_seed"]) == s} for s in seeds}
    lengths = {s: {row["motion_key"]: int(row["valid_frames"]) for row in rows
                   if int(row["physics_seed"]) == s} for s in seeds}
    agreement = agreement_matrix(labels)
    retention: dict[str, Any] = {}
    for height in GRADED_HEIGHTS_M:
        key = f"{height:g}"
        clearing = sorted({row["motion_key"] for row in rows if row["reference_exact_clears"][key]})
        per_seed = {}
        total = 0
        for s in seeds:
            retained = [row["motion_key"] for row in rows
                        if int(row["physics_seed"]) == s and row["motion_key"] in clearing
                        and row["achieved_replay_clear_after_passing"][key]]
            per_seed[str(s)] = {"retained": len(retained), "n": len(clearing), "keys": retained}
            total += len(retained)
        n_rollouts = len(clearing) * len(seeds)
        retention[key] = {"n_reference_exact_clear": len(clearing), "n_seeds": len(seeds),
                          "n_rollouts": n_rollouts, "retained_rollouts": total,
                          "retention_fraction": total / n_rollouts if n_rollouts else None,
                          "wilson95": wilson(total, n_rollouts), "per_seed": per_seed}
    closest = {}
    for row in rows:
        terms = row.get("evaluator_terms")
        if terms and row["tracker_terminated"]:
            name = terms["closest_term_at_last_sample"]
            closest[name] = closest.get(name, 0) + 1
    return {
        "seeds": seeds, "n_clips": len(labels[seeds[0]]) if seeds else 0,
        "terminated_per_seed": {str(s): sum(labels[s].values()) for s in seeds},
        "terminated_rate_wilson95_per_seed": {str(s): wilson(sum(labels[s].values()), len(labels[s]))
                                              for s in seeds},
        "valid_frames_per_seed": {str(s): lengths[s] for s in seeds},
        "agreement": agreement,
        "retention_over_all_seeds": retention,
        "closest_term_at_last_alive_sample_terminated": closest,
    }


def offline_term_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Under the release evaluator every archived sample was alive: no term may exceed."""
    checked = [row for row in rows if row.get("evaluator_terms")]
    exceeding = [row["motion_key"] for row in checked if row["evaluator_terms"]["any_term_exceeded"]]
    n_samples = sum(row["evaluator_terms"]["n_samples"] for row in checked)
    n_exceeding = sum(sum(t["n_samples_exceeding"] for t in row["evaluator_terms"]["per_term"].values())
                      for row in checked)
    return {"n_rollouts": len(checked), "n_alive_samples": n_samples,
            "n_samples_exceeding_any_threshold": n_exceeding,
            "rollouts_with_false_positive": exceeding, "consistent": not exceeding}


def evaluate_decision_rules(part_a: Mapping[str, Any], part_b: Mapping[str, Any]) -> dict[str, Any]:
    subset = part_a["reference_exact_clear_subset_by_height"]
    primary = subset["0.05"]
    n = primary["n_reference_exact_clear"]
    fell_or_walked = primary["fell_or_walked_through"]
    agreement = part_b["agreement"]["unanimous"]["fraction"]
    return {
        "not_tracked_wording": {
            "rule": "allowed only if >= 9/12 exact-clearing (5 cm, x=1.2 m) references end fell "
                    "or walked_through with terminations disabled",
            "n_reference_exact_clear_5cm": n,
            "fell_or_walked_through": fell_or_walked,
            "cleared": primary["cleared"],
            "allowed": bool(n == 12 and fell_or_walked >= 9),
            "most_cleared_rewrite_required": bool(n and primary["cleared"] > n / 2),
            "by_height": {h: {"n": v["n_reference_exact_clear"],
                              "fell_or_walked_through": v["fell_or_walked_through"],
                              "cleared": v["cleared"]} for h, v in subset.items()},
        },
        "test_retest": {
            "rule": "agreement < 0.80 => every single-seed zero is a lower bound only",
            "agreement_fraction_unanimous": agreement,
            "agreement_wilson95": part_b["agreement"]["unanimous"]["wilson95"],
            "mean_pairwise_fraction": part_b["agreement"]["mean_pairwise_fraction"],
            "single_seed_zeros_are_lower_bounds": bool(agreement is not None and agreement < 0.80),
        },
    }


# ------------------------------------------------------------------------------ campaign

def _persist(output: Path, receipt: dict[str, Any], *, started: float,
             part_a_rows: Sequence[Mapping[str, Any]] | None = None,
             part_b_rows: Sequence[Mapping[str, Any]] | None = None,
             summary: Mapping[str, Any] | None = None) -> None:
    anchors = receipt.setdefault("evidence_anchors", {})
    if part_a_rows is not None:
        cal._write_jsonl(output / "part_a_rows.jsonl", part_a_rows)
        anchors["part_a_rows"] = {"n_rows": len(part_a_rows), "logical_sha256": cal._json_hash(part_a_rows),
                                  "file_sha256": _sha256(output / "part_a_rows.jsonl")}
    if part_b_rows is not None:
        cal._write_jsonl(output / "part_b_rows.jsonl", part_b_rows)
        anchors["part_b_rows"] = {"n_rows": len(part_b_rows), "logical_sha256": cal._json_hash(part_b_rows),
                                  "file_sha256": _sha256(output / "part_b_rows.jsonl")}
    if summary is not None:
        cal._write_json(output / "summary.json", summary)
        anchors["summary"] = {"logical_sha256": cal._json_hash(summary),
                              "file_sha256": _sha256(output / "summary.json")}
    receipt["wall_clock_s"] = float(time.monotonic() - started)
    cal._write_json(output / "receipt.json", receipt)


def _smoke_receipt_path(output: Path) -> Path:
    return output / "smoke_receipt.json"


def smoke_passed(output: Path, part: str) -> bool:
    path = _smoke_receipt_path(output)
    if not path.is_file():
        return False
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    entry = receipt.get(part) if isinstance(receipt, dict) else None
    return bool(isinstance(entry, dict) and entry.get("pass") is True
                and receipt.get("schema") == SMOKE_RECEIPT_SCHEMA)


def validate_completed_output(output: Path, receipt: Mapping[str, Any]) -> None:
    """Revalidate a completed campaign's evidence anchors before an idempotent resume returns it."""
    anchors = receipt.get("evidence_anchors", {})
    for name, expected_n in (("part_a_rows", 64), ("part_b_rows", 192)):
        path = output / f"{name}.jsonl"
        rows = _read_jsonl(path)
        anchor = anchors.get(name, {})
        if (len(rows) != expected_n or anchor.get("n_rows") != expected_n
                or cal._json_hash(rows) != anchor.get("logical_sha256")
                or _sha256(path) != anchor.get("file_sha256")):
            raise BridgeAbort(f"completed {name} no longer matches its evidence anchor")
    try:
        summary = json.loads((output / "summary.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BridgeAbort(f"completed summary is unreadable: {exc}") from exc
    anchor = anchors.get("summary", {})
    if (cal._json_hash(summary) != anchor.get("logical_sha256")
            or _sha256(output / "summary.json") != anchor.get("file_sha256")
            or receipt.get("summary") != summary):
        raise BridgeAbort("completed summary no longer matches its evidence anchor")
    for spec in smoke_launch_plan() + campaign_launch_plan():
        record = receipt.get("launches", {}).get(spec["name"])
        if not isinstance(record, dict) or record.get("status") != "complete":
            raise BridgeAbort(f"completed receipt lacks launch {spec['name']}")
        attempt = Path(record.get("attempt", ""))
        if (_sha256(attempt / "receipt.json") != record.get("attempt_receipt_sha256")
                or _sha256(attempt / "process_result.json") != record.get("process_result", {}).get("file_sha256")):
            raise BridgeAbort(f"completed launch artifacts changed for {spec['name']}")


def _campaign_identity(project: Mapping[str, Any], tracker: Mapping[str, Any],
                       source: Mapping[str, Any], exp022a: Mapping[str, Any],
                       protocol: Mapping[str, Any], thresholds: Mapping[str, Any]) -> str:
    return cal._json_hash({
        "schema": SCHEMA_VERSION,
        "project_source_sha256": project["source_sha256"],
        "project_commit": project["git"].get("commit"),
        "physical_model": project["physical_model"],
        "tracker": bound_tracker_identity(tracker),
        "source_exp021": source, "exp022a": exp022a,
        "protocol_sha256": protocol["sha256"], "thresholds": thresholds,
        "plan": smoke_launch_plan() + campaign_launch_plan(),
        "overrides": list(TERMINATION_FREE_OVERRIDES),
        "release_thresholds": RELEASE_THRESHOLDS,
        "heights": list(GRADED_HEIGHTS_M), "obstacle_x_m": STAGED_OBSTACLE_X_M,
    })


def run_campaign(
    *,
    stage: str,
    out: str | Path = DEFAULT_OUT,
    resume: bool = False,
    dry_run: bool = False,
    source_dir: str | Path = SOURCE_OUT,
    exp022a_dir: str | Path = EXP022A_OUT,
    timeout_s: int = 2400,
    launch_fn: Callable[..., tuple[int, str]] = launch_sonic,
    export_fn: Callable[..., Path] = write_motion_pkl,
    compose_fn: Callable[[Sequence[str]], Mapping[str, Any]] = compose_resolved_terminations,
    host_gate_fn: Callable[..., Mapping[str, Any]] = host_gate.require_host_resources,
    host_report_fn: Callable[..., Mapping[str, Any]] = host_gate.host_resource_report,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    tracker_identity_fn: Callable[[], Mapping[str, Any]] = tracker_identity,
    protocol_identity_fn: Callable[[], Mapping[str, Any]] = protocol_identity,
    scorer: Callable[..., Mapping[str, Any]] = exp022.score_trajectory,
    contract_fn: Callable[..., Mapping[str, Any]] | None = None,
    evaluator_fn: Callable[..., Mapping[str, Any]] | None = None,
    mj_model: Any = None,
    require_preregistered: bool = True,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    output = Path(out)
    source = exp022.load_source_bundle(source_dir)
    exp022a = exp022a_identity(exp022a_dir)
    exp022a_rollouts = load_exp022a_rollouts(exp022a)
    full_valid_length = full_reference_valid_length(exp022a_rollouts)
    validate_smoke_keys(exp022a_rollouts)
    thresholds = threshold_identity()
    protocol = dict(protocol_identity_fn())
    current_project = project_identity(code_state_fn=code_state_fn)
    tracker = dict(tracker_identity_fn())
    plan = smoke_launch_plan() + campaign_launch_plan()

    if dry_run:
        report = dict(host_report_fn(require_no_isaac=True))
        commands = {spec["name"]: build_sonic_command(
            output / "launches" / spec["name"] / "motions.pkl",
            output / "launches" / spec["name"] / "attempt-000/eval",
            spec["n_motions"], spec["physics_seed"], spec["extra_overrides"]) for spec in plan}
        return {
            "schema": SCHEMA_VERSION, "experiment": "exp028_termination_free_rollouts",
            "status": "dry_run", "writes_performed": False,
            "project_dirty_observed": current_project["git"].get("dirty"),
            "protocol": protocol, "tracker": bound_tracker_identity(tracker),
            "tracker_dirty_paths": tracker.get("dirty_paths"),
            "source": source["identity"], "exp022a": exp022a, "thresholds": thresholds,
            "launch_plan": plan, "commands": commands,
            "termination_free_overrides": list(TERMINATION_FREE_OVERRIDES),
            "host_resource_gate": report, "full_reference_valid_length": full_valid_length,
            "campaign_identity_sha256": _campaign_identity(
                current_project, tracker, source["identity"], exp022a, protocol, thresholds),
        }

    if require_preregistered and protocol.get("status") != "preregistered":
        raise BridgeAbort(f"EXP-028 protocol status is {protocol.get('status')!r}; commit it as "
                          "'preregistered' before the first launch")
    existing_receipt = output / "receipt.json"
    old: dict[str, Any] | None = None
    resume_project_check: dict[str, Any] | None = None
    if resume:
        if not existing_receipt.is_file():
            raise BridgeAbort(f"--resume requires an existing EXP-028 receipt in {output}")
        try:
            old = json.loads(existing_receipt.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise BridgeAbort(f"existing campaign receipt is unreadable: {exc}") from exc
        if not isinstance(old, dict):
            raise BridgeAbort("existing campaign receipt is not an object")
        if old.get("status") == "blocked" or old.get("schema") == FAILURE_SCHEMA_VERSION:
            raise BridgeAbort("existing EXP-028 campaign is blocked; preserve it and use fresh output")
        if old.get("schema") != SCHEMA_VERSION:
            raise BridgeAbort("existing output is not a resumable EXP-028 campaign")
        pinned_project = old.get("provenance", {}).get("project")
        if not isinstance(pinned_project, dict) or pinned_project.get("git", {}).get("dirty") is not False:
            raise BridgeAbort("existing receipt lacks a clean pinned Scene2Motion identity")
        try:
            resume_project_check = validate_project_recheck(pinned_project, current_project, output)
        except ValueError as exc:
            raise BridgeAbort(str(exc)) from exc
        project = dict(pinned_project)
        if old.get("provenance", {}).get("protocol", {}).get("sha256") != protocol["sha256"]:
            raise BridgeAbort("EXP-028 protocol changed since the campaign was created")
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise BridgeAbort(f"refusing non-empty output for a fresh campaign: {output} "
                              "(pass --resume to continue an EXP-028 campaign)")
        if current_project["git"].get("dirty") is not False:
            raise BridgeAbort("EXP-028 requires an exactly clean Scene2Motion worktree")
        project = current_project

    campaign_identity = _campaign_identity(project, tracker, source["identity"], exp022a,
                                           protocol, thresholds)
    if old is not None and old.get("campaign_identity_sha256") != campaign_identity:
        raise BridgeAbort("existing EXP-028 output has a different campaign identity")
    if old is not None and old.get("status") == "complete":
        validate_completed_output(output, old)
        return old

    started = time.monotonic()
    if old is None:
        # Host gate before the output directory exists: a failed gate leaves nothing behind.
        try:
            initial_gate = dict(host_gate_fn(require_no_isaac=True))
        except host_gate.HostResourceGateFailed as exc:
            raise BridgeAbort(f"host-resource gate failed before the campaign was created: {exc}") from exc
        output.mkdir(parents=True, exist_ok=False)
        receipt: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "experiment": "exp028_termination_free_rollouts",
            "status": "running", "complete": False, "blocked": False, "stage": "preflight",
            "campaign_identity_sha256": campaign_identity,
            "resume_supported": True,
            "claim_scope": ("achieved-state replay of the archived EXP-021 pool: Part A with every "
                            "tracking-error termination unreachable, Part B physics-seed re-roll "
                            "under the release evaluator; no obstacle in Isaac"),
            "not_contact_rich_execution": True,
            "actual_ardy_samples": 0, "reused_archived_ardy_samples": 64,
            "design": {
                "launch_plan": plan,
                "termination_free_overrides": list(TERMINATION_FREE_OVERRIDES),
                "release_thresholds": RELEASE_THRESHOLDS,
                "expected_tracking_terms": list(EXPECTED_TRACKING_TERMS),
                "time_out_term": TIME_OUT_TERM,
                "obstacle": {"x_m": STAGED_OBSTACLE_X_M, "depth_m": OBSTACLE_DEPTH_M,
                             "front_edge_x_m": BOX_FRONT_EDGE_X_M,
                             "half_width_m": exp022.OBSTACLE_HALF_WIDTH_M},
                "graded_heights_m": list(GRADED_HEIGHTS_M),
                "fell": {"pelvis_z_m": FELL_PELVIS_Z_M, "up_z": FELL_UP_Z},
                "smoke_motion_keys": list(SMOKE_MOTION_KEYS),
                "full_reference_valid_length": full_valid_length,
                "evaluator_formulas": EVALUATOR_FORMULAS,
                "outcome_classes": list(OUTCOME_CLASSES), "residual_classes": list(RESIDUAL_CLASSES),
            },
            "provenance": {
                "project": project, "protocol": protocol, "source_exp021": source["identity"],
                "exp022a": exp022a, "tracker": tracker, "threshold_receipt": thresholds,
                "initial_host_resource_gate": initial_gate,
            },
            "stages_complete": {"smoke_a": False, "smoke_b": False, "part_a": False,
                                "part_b": False, "analysis": False},
            "resume_project_check": resume_project_check,
            "post_launch_revalidation": {}, "launches": {}, "host_gate_blocks": [],
        }
        _persist(output, receipt, started=started)
    else:
        receipt = old
        receipt["resume_project_check"] = resume_project_check
        receipt["provenance"]["tracker_at_resume"] = tracker
        receipt.setdefault("host_gate_blocks", [])

    part_a_rows = _read_jsonl(output / "part_a_rows.jsonl") if (output / "part_a_rows.jsonl").is_file() else []
    part_b_rows = _read_jsonl(output / "part_b_rows.jsonl") if (output / "part_b_rows.jsonl").is_file() else []
    model = mj_model if mj_model is not None else G1Body(None).model
    source_rows = {int(row["seed"]): row for row in source["rows"]}

    def revalidate(name: str) -> None:
        if exp022.load_source_bundle(source_dir)["identity"] != source["identity"]:
            raise ValueError("EXP-021 source artifacts changed during SONIC execution")
        if exp022a_identity(exp022a_dir) != exp022a:
            raise ValueError("EXP-022A artifacts changed during SONIC execution")
        current_tracker = dict(tracker_identity_fn())
        if bound_tracker_identity(current_tracker) != bound_tracker_identity(tracker):
            raise ValueError("SONIC checkout/checkpoint/config changed during execution")
        git_check = validate_project_recheck(project, project_identity(code_state_fn=code_state_fn), output)
        receipt["post_launch_revalidation"][name] = {
            "source_exp021_unchanged": True, "exp022a_unchanged": True,
            "tracker_bound_identity_unchanged": True,
            "tracker_dirty_paths_now": current_tracker.get("dirty_paths"), "project": git_check,
        }

    def launch_part(part: str) -> dict[str, list[Any]]:
        rollouts_by_launch: dict[str, list[Any]] = {}
        for spec in launches_for_part(part):
            receipt["stage"] = f"launching_{spec['name']}"
            _persist(output, receipt, started=started)
            pkl = ensure_motion_pkl(spec, source["clips"], output, export_fn=export_fn, mj_model=model)
            if spec["chunk"] is not None:
                expected = EXP022A_MOTION_PKL_SHA256[f"chunk{spec['chunk']:02d}_seed0"]
                observed = _sha256(pkl)
                if export_fn is write_motion_pkl and observed != expected:
                    raise ValueError(f"{spec['name']} motion pickle {observed} differs from "
                                     f"EXP-022A's {expected}")
            try:
                record, rollouts = run_or_resume_launch(
                    spec, pkl, output, launch_fn=launch_fn, timeout_s=timeout_s,
                    compose_fn=compose_fn, host_gate_fn=host_gate_fn,
                    full_valid_length=full_valid_length)
            except host_gate.HostResourceGateFailed as exc:
                receipt["host_gate_blocks"].append({
                    "launch": spec["name"], "note": "blocked_host_gate", "error": str(exc),
                    "at_unix_s": time.time()})
                receipt["stage"] = f"blocked_host_gate_{spec['name']}"
                _persist(output, receipt, started=started)
                raise CampaignPaused(f"blocked_host_gate: {exc}") from exc
            receipt["launches"][spec["name"]] = record
            rollouts_by_launch[spec["name"]] = rollouts
            if not record.get("recovered_or_resumed"):
                revalidate(spec["name"])
            _persist(output, receipt, started=started)
        return rollouts_by_launch

    try:
        stages = [stage] if stage != "all" else ["smoke", "part_a", "part_b", "analyze"]
        for current in stages:
            if current == "smoke":
                smoke: dict[str, Any] = {"schema": SMOKE_RECEIPT_SCHEMA}
                for part in ("smoke_a", "smoke_b"):
                    spec = launches_for_part(part)[0]
                    try:
                        by_launch = launch_part(part)
                        record = receipt["launches"][spec["name"]]
                        smoke[part] = {
                            "pass": True, "launch": spec["name"],
                            "config_expectation": spec["config_expectation"],
                            "resolved_terminations_sha256": record["resolved_terminations"]["file_sha256"],
                            "audit": record["resolved_terminations"]["audit"],
                            "log_termination_terms": record["log_termination_terms"],
                            "rollout_expectation_check": record["rollout_expectation_check"],
                            "n_rollouts": len(by_launch[spec["name"]]),
                        }
                        receipt["stages_complete"][part] = True
                    except CampaignPaused:
                        raise
                    except BridgeAbort as exc:
                        smoke[part] = {"pass": False, "launch": spec["name"], "error": str(exc)}
                        cal._write_json(_smoke_receipt_path(output), smoke)
                        raise
                    cal._write_json(_smoke_receipt_path(output), smoke)
                receipt["smoke_receipt_sha256"] = _sha256(_smoke_receipt_path(output))
                _persist(output, receipt, started=started)
            elif current == "part_a":
                if not smoke_passed(output, "smoke_a"):
                    raise CampaignPaused("Part A refuses to launch without a passing smoke_a "
                                         "receipt (run --stage smoke first)")
                launch_part("part_a")
                receipt["stages_complete"]["part_a"] = True
                _persist(output, receipt, started=started)
            elif current == "part_b":
                if not smoke_passed(output, "smoke_b"):
                    raise CampaignPaused("Part B refuses to launch without a passing smoke_b "
                                         "receipt (run --stage smoke first)")
                launch_part("part_b")
                receipt["stages_complete"]["part_b"] = True
                _persist(output, receipt, started=started)
            elif current == "analyze":
                if not (receipt["stages_complete"]["part_a"] and receipt["stages_complete"]["part_b"]):
                    raise CampaignPaused("analysis requires complete Part A and Part B launches")
                receipt["stage"] = "analysis"
                _persist(output, receipt, started=started)
                geometry = None if evaluator_fn is not None else EvaluatorGeometry()
                body = None if contract_fn is not None else (geometry.body if geometry else G1Body(None))
                part_a_rollouts: dict[str, Any] = {}
                for spec in launches_for_part("part_a"):
                    _, rollouts = load_completed_launch(spec, output, full_valid_length=full_valid_length)
                    part_a_rollouts.update({r.motion_key: r for r in rollouts})
                part_b_by_seed: dict[int, dict[str, Any]] = {0: dict(exp022a_rollouts)}
                for spec in launches_for_part("part_b"):
                    _, rollouts = load_completed_launch(spec, output, full_valid_length=full_valid_length)
                    part_b_by_seed.setdefault(int(spec["physics_seed"]), {}).update(
                        {r.motion_key: r for r in rollouts})
                expected_keys = {f"s{seed}" for seed in POOL_SEEDS}
                if set(part_a_rollouts) != expected_keys or any(
                        set(v) != expected_keys for v in part_b_by_seed.values()):
                    raise ValueError("launch archives do not cover all 64 archived motions")
                part_a_rows, term_arrays = build_part_a_rows(
                    part_a_rollouts, clips=source["clips"], source_rows=source_rows,
                    thresholds=thresholds, geometry=geometry, body=body, scorer=scorer,
                    contract_fn=contract_fn, evaluator_fn=evaluator_fn)
                if term_arrays:
                    cal._atomic_write(output / "part_a_evaluator_terms.npz",
                                      lambda handle: np.savez_compressed(handle, **term_arrays))
                part_b_rows = build_part_b_rows(
                    part_b_by_seed, clips=source["clips"], source_rows=source_rows,
                    geometry=geometry, scorer=scorer, evaluator_fn=evaluator_fn,
                    seed_sources={0: "exp022a_archive", 1: "exp028_launch", 2: "exp028_launch"})
                # Cross-check the seed-0 restatement against EXP-022A's own achieved rows
                # (only meaningful with the exact geometry scorer).
                seed0_matches: bool | None = None
                if scorer is exp022.score_trajectory:
                    exp022a_rows = {row["motion_key"]: row for row in
                                    _read_jsonl(Path(exp022a_dir) / "achieved_rows.jsonl")
                                    if row["obstacle_label"] == "staged"}
                    mismatches = []
                    for row in part_b_rows:
                        if row["physics_seed"] != 0:
                            continue
                        ref = exp022a_rows[row["motion_key"]]
                        if (bool(ref["tracker_terminated"]) != row["tracker_terminated"]
                                or ref["achieved_replay_clear_after_passing"]
                                != row["achieved_replay_clear_after_passing"]):
                            mismatches.append(row["motion_key"])
                    if mismatches:
                        raise ValueError(f"seed-0 restatement disagrees with EXP-022A rows: {mismatches}")
                    seed0_matches = True
                summary_a = summarize_part_a(part_a_rows)
                summary_b = summarize_part_b(part_b_rows)
                summary = {
                    "status": "complete",
                    "part_a": summary_a, "part_b": summary_b,
                    "offline_term_consistency_release_evaluator": offline_term_consistency(
                        [row for row in part_b_rows if row["physics_seed"] == 0]),
                    "decision_rules": evaluate_decision_rules(summary_a, summary_b),
                    "seed0_restatement_matches_exp022a": seed0_matches,
                    "interpretation_guard": (
                        "SONIC tracked references without an Isaac obstacle; achieved states "
                        "were replayed against Scene2Motion geometry. Part A outcome classes are "
                        "physical only in the sense of the achieved-state archive; no "
                        "contact-rich execution claim is licensed."),
                }
                receipt["stages_complete"]["analysis"] = True
                receipt.update({"status": "complete", "complete": True, "stage": "complete",
                                "sonic_rollouts_requested": 6 * 32 + 2 * SMOKE_NUM_ENVS,
                                "sonic_rollouts_returned": sum(
                                    int(r.get("n_rollouts", 0)) for r in receipt["launches"].values()),
                                "summary": summary})
                _persist(output, receipt, started=started, part_a_rows=part_a_rows,
                         part_b_rows=part_b_rows, summary=summary)
        if receipt["status"] != "complete":
            receipt["stage"] = f"{stage}_complete"
        _persist(output, receipt, started=started)
        return receipt
    except Exception as exc:
        if isinstance(exc, CampaignPaused):
            raise
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
            "blocked": True, "failed_stage": receipt.get("stage"),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        _persist(output, receipt, started=started)
        if isinstance(exc, BridgeAbort):
            raise
        raise BridgeAbort(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing EXP-028 campaign directory")
    parser.add_argument("--source", default=str(SOURCE_OUT))
    parser.add_argument("--exp022a", default=str(EXP022A_OUT))
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the launch plan, commands and host gate; write nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_campaign(
            stage=args.stage, out=args.out, resume=args.resume, dry_run=args.dry_run,
            source_dir=args.source, exp022a_dir=args.exp022a, timeout_s=args.timeout_s)
    except (BridgeAbort, host_gate.HostResourceGateFailed) as exc:
        print(json.dumps({"status": "aborted", "error_type": type(exc).__name__,
                          "error": str(exc)}, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps({
            "status": receipt["status"], "writes_performed": False,
            "protocol": receipt["protocol"], "host_resource_gate": receipt["host_resource_gate"],
            "tracker": receipt["tracker"], "tracker_dirty_paths": receipt["tracker_dirty_paths"],
            "launch_plan": [{k: v for k, v in spec.items() if k not in ("seeds", "motion_keys")}
                            | {"first_key": spec["motion_keys"][0], "last_key": spec["motion_keys"][-1]}
                            for spec in receipt["launch_plan"]],
            "commands": {name: " ".join(cmd) for name, cmd in receipt["commands"].items()},
        }, indent=2))
        return 0
    print(json.dumps({"status": receipt["status"], "stage": receipt.get("stage"),
                      "stages_complete": receipt.get("stages_complete"),
                      "launches": len(receipt.get("launches", {}))}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
