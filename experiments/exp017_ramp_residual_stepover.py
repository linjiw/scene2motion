"""EXP-017: absolute packets versus phase-aligned residual packets for step-over.

This is the first representation experiment for Scene2Motion-RAMP.  It deliberately has
only two evaluated arms.  Both arms use the same step-over prompt, evaluation seed, sampler
settings, fixed obstacle, nominal target gait, physical target-phase receipt, and exact ARDY
constraint support.  They differ only in whether a coherent donor packet is transported as
an absolute pose packet or as an adapted-minus-neutral residual packet.

The generation budget is explicit in *samples* (not Python calls)::

    2D + N + 2NP

``D`` matched discovery seeds produce one STEP-adapted and one WALK-neutral source clip.
``N`` disjoint held-out seeds produce one WALK nominal target clip each.  The final two arms
are generated for every nominal seed and each of ``P`` predeclared, seed-independent scenes.
The complete program manifest is written and hashed before any final-arm output is sampled.

This file implements a kinematic pilot only.  A paper claim about physical execution still
requires equal-budget SONIC replay for both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec, build_conditions, channel_usage  # noqa: E402
from scene2motion.ramp import (  # noqa: E402
    PacketControls,
    constraint_support_digest,
    extract_packet_pair,
    render_packet,
)
from scene2motion.ramp.step_phase import (  # noqa: E402
    StepPhaseCycle,
    align_step_phase_cycles,
    align_step_target_phase,
    enumerate_step_phase_cycles_from_qpos,
)
from scene2motion.robot import BODY_MARGIN, G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import (  # noqa: E402
    BoxHeightProbe,
    StepOverThresholds,
    foot_kinematics_series,
    motion_metrics,
)


WALK = "A person walks forward."
STEP = "A person steps over an obstacle."
ARMS = ("absolute", "residual")
EVENT_SELECTOR = "qpos-step-cycle-adapted-lift-neutral-progress-v1"
DEFAULT_OBSTACLE_X = (2.4, 3.6, 4.8)
ALLOWED_NONZERO_CHANNELS = {
    "root_2d", "root_y_pos", "global_joints_rots",
}


class ExperimentAbort(RuntimeError):
    """A fail-closed experiment stop after writing a diagnostic receipt."""


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _identity(schema: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical, self-describing identity with no reference to mutable output paths."""
    payload = {"schema": schema, "fields": dict(fields)}
    # Round-trip rejects non-JSON or non-finite values and detaches caller-owned mappings.
    normalized = json.loads(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return {**normalized, "sha256": _json_hash(normalized)}


def _array_hash(arrays: Mapping[str, Any]) -> str:
    """Hash ndarray content independent of NPZ container metadata and key order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            continue
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sample_hash(sample: Mapping[str, Any]) -> str:
    arrays = {str(name): value for name, value in sample.items()
              if isinstance(value, np.ndarray)}
    if not arrays:
        raise ValueError("motion sample contains no ndarray payloads to hash")
    return _array_hash(arrays)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL,
        )
        return {
            "commit": commit,
            "dirty": bool(status),
            "status": status,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": [],
                "tracked_diff_sha256": None}


def _checkpoint_identity(runner: ArdyRunner) -> dict[str, Any]:
    """Resolve and content-hash the released denoiser used by the live runner."""
    model_name = str(runner.model_name)
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = [
        hub / f"models--nvidia--{model_name}",
        hub / f"models--nvidia--{model_name.upper()}",
        hub / f"models--nvidia--ARDY-{model_name.upper()}",
    ]
    for model_dir in candidates:
        ref = model_dir / "refs" / "main"
        revision = ref.read_text().strip() if ref.exists() else None
        checkpoint = (model_dir / "snapshots" / revision / "denoiser.safetensors"
                      if revision else None)
        if checkpoint is not None and checkpoint.exists():
            return {
                "generator_id": f"nvidia/{model_name}@{revision}",
                "model_name": model_name,
                "hf_revision": revision,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
    raise ValueError(f"could not resolve denoiser.safetensors for model {model_name!r}")


def _base_spec(root_xz: np.ndarray) -> ConstraintSpec:
    """Root-path-only substrate shared by source and held-out nominal generation."""
    return ConstraintSpec(
        root_xz=np.asarray(root_xz, dtype=float),
        heading=None,
        root_y=None,
        first_heading=0.0,
    )


def _scene_id(obstacle_x: float, obstacle_height: float, obstacle_depth: float) -> str:
    return (f"step_x{float(obstacle_x):.3f}_h{float(obstacle_height):.3f}_"
            f"d{float(obstacle_depth):.3f}")


def _load_thresholds(path: Path) -> tuple[StepOverThresholds, dict[str, Any]]:
    try:
        receipt = json.loads(path.read_text())
        if receipt.get("experiment") != "stepover_threshold_calibration":
            raise ValueError("unexpected calibration experiment identity")
        if receipt.get("status") != "calibrated":
            raise ValueError("threshold receipt status is not 'calibrated'")
        calibration = receipt["calibration"]
        if not isinstance(calibration, dict):
            raise ValueError("calibration must be an object")
        n_clips = calibration.get("n_clips")
        n_accepted = calibration.get("n_accepted")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (n_clips, n_accepted)):
            raise ValueError("calibration counts must be integers")
        if n_clips <= 0 or n_accepted <= 0 or n_accepted > n_clips:
            raise ValueError("calibration counts must be nonzero and internally valid")
        clips = calibration.get("clips")
        outliers = calibration.get("outlier_clip_indices")
        if not isinstance(clips, list) or len(clips) != n_clips:
            raise ValueError("calibration n_clips does not match its clip diagnostics")
        if (not isinstance(outliers, list)
                or any(isinstance(index, bool) or not isinstance(index, int)
                       for index in outliers)
                or len(set(outliers)) != len(outliers)
                or any(index < 0 or index >= n_clips for index in outliers)
                or n_accepted != n_clips - len(outliers)):
            raise ValueError("calibration accepted/outlier counts are inconsistent")

        corpus = receipt["corpus"]
        if not isinstance(corpus, dict):
            raise ValueError("calibration corpus must be an object")
        for name in ("exp1c_dir", "selection"):
            if not isinstance(corpus.get(name), str) or not corpus[name].strip():
                raise ValueError(f"calibration corpus field {name!r} is missing")
        arms = corpus.get("arms")
        if (not isinstance(arms, list) or not arms
                or any(not isinstance(arm, str) or not arm.strip() for arm in arms)
                or len(set(arms)) != len(arms)):
            raise ValueError("calibration corpus arms must be unique non-empty strings")
        sources = corpus.get("sources")
        if sources not in ("both", "reference", "achieved"):
            raise ValueError("calibration corpus has an invalid sources mode")
        for name in ("reference", "achieved", "excluded_achieved"):
            if not isinstance(corpus.get(name), list):
                raise ValueError(f"calibration corpus field {name!r} must be a list")
        if sources in ("both", "reference") and not corpus["reference"]:
            raise ValueError("reference calibration source is empty")
        if sources in ("both", "achieved") and not corpus["achieved"]:
            raise ValueError("achieved calibration source is empty")
        for entry in corpus["reference"]:
            if (not isinstance(entry, dict)
                    or not isinstance(entry.get("path"), str)
                    or not entry["path"].strip()
                    or not _is_sha256(entry.get("sha256"))):
                raise ValueError("calibration reference corpus entry is invalid")
        for entry in corpus["achieved"]:
            if (not isinstance(entry, dict)
                    or not isinstance(entry.get("archive"), str)
                    or not entry["archive"].strip()
                    or not _is_sha256(entry.get("sha256"))):
                raise ValueError("calibration achieved corpus entry is invalid")
        reference_fps = corpus.get("reference_fps")
        if (isinstance(reference_fps, bool) or not isinstance(reference_fps, (int, float))
                or not np.isfinite(reference_fps) or reference_fps <= 0):
            raise ValueError("calibration corpus reference_fps must be positive")

        provenance = receipt["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("calibration provenance must be an object")
        code = provenance.get("code")
        if not isinstance(code, dict) or not isinstance(code.get("commit"), str):
            raise ValueError("calibration provenance lacks a code commit")
        if not code["commit"].strip() or not isinstance(code.get("dirty"), bool):
            raise ValueError("calibration code provenance is incomplete")
        if not _is_sha256(code.get("tracked_diff_sha256")):
            raise ValueError("calibration tracked-diff hash is invalid")
        source_hashes = provenance.get("source_sha256")
        required_sources = {
            "experiments/calibrate_stepover_thresholds.py",
            "scene2motion/stepover_eval.py",
            "scene2motion/sonic_state_export.py",
        }
        if not isinstance(source_hashes, dict) or not required_sources <= set(source_hashes):
            raise ValueError("calibration provenance lacks required source hashes")
        if any(not _is_sha256(source_hashes[name]) for name in required_sources):
            raise ValueError("calibration provenance contains an invalid source hash")

        thresholds = StepOverThresholds(**receipt["stepover_thresholds"])
        thresholds.validate()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid threshold calibration receipt: {exc}") from exc
    return thresholds, {
        "path": str(path),
        "sha256": _sha256(path),
        "experiment": receipt.get("experiment"),
        "status": receipt.get("status"),
        "n_clips": n_clips,
        "n_accepted": n_accepted,
        "normalized_receipt_sha256": _json_hash(receipt),
    }


def _phase_cycles(
    body: Any,
    qpos: np.ndarray,
    fps: float,
    thresholds: StepOverThresholds,
    *,
    swing_side: str | None = None,
    support_window_s: float,
    min_stance_support_fraction: float,
    min_relative_lift_m: float,
) -> tuple[StepPhaseCycle, ...]:
    return enumerate_step_phase_cycles_from_qpos(
        body,
        qpos,
        fps,
        frame_window=(int(0.1 * len(qpos)), int(0.9 * len(qpos))),
        swing_side=swing_side,
        thresholds=thresholds,
        support_window_s=support_window_s,
        min_stance_support_fraction=min_stance_support_fraction,
        min_relative_lift_m=min_relative_lift_m,
    )


def _route_progress(qpos: np.ndarray, frame: int) -> float:
    forward = np.asarray(qpos, dtype=float)[:, 0]
    distance = float(forward[-1] - forward[0])
    if distance <= 1e-8:
        raise ValueError("source clip has no positive route progress")
    return float((forward[int(frame)] - forward[0]) / distance)


def _prescribed_progress_ratio(qpos: np.ndarray, root_xz: np.ndarray) -> float:
    """Final realized forward progress divided by the prescribed route distance."""
    actual = float(np.asarray(qpos, dtype=float)[-1, 0]
                   - np.asarray(qpos, dtype=float)[0, 0])
    planned = float(np.asarray(root_xz, dtype=float)[-1, 1]
                    - np.asarray(root_xz, dtype=float)[0, 1])
    if planned <= 1e-8:
        raise ValueError("prescribed route has no positive forward distance")
    return actual / planned


def _select_source_pair(
    source_rows: list[dict[str, Any]],
    *,
    half_window_frames: int,
) -> tuple[dict[str, Any], StepPhaseCycle, StepPhaseCycle, Any]:
    """Select by adapted quality; neutral motion contributes feasibility and phase only."""
    feasible: list[tuple[tuple[Any, ...], dict[str, Any], StepPhaseCycle,
                         StepPhaseCycle, Any]] = []
    for row in source_rows:
        adapted_qpos = row.pop("_adapted_qpos")
        neutral_qpos = row.pop("_neutral_qpos")
        adapted_cycles = row.pop("_adapted_cycles")
        neutral_cycles = row.pop("_neutral_cycles")
        pair_rows = []
        for adapted in adapted_cycles:
            adapted_progress = _route_progress(adapted_qpos, adapted.apex_frame)
            for neutral in neutral_cycles:
                if neutral.swing_side != adapted.swing_side:
                    continue
                neutral_progress = _route_progress(neutral_qpos, neutral.apex_frame)
                try:
                    phase_match = align_step_phase_cycles(
                        adapted, neutral, half_window_frames=half_window_frames)
                except ValueError as exc:
                    pair_rows.append({
                        "adapted_apex_frame": adapted.apex_frame,
                        "neutral_apex_frame": neutral.apex_frame,
                        "swing_side": adapted.swing_side,
                        "eligible": False,
                        "reason": str(exc),
                    })
                    continue
                progress_delta = abs(adapted_progress - neutral_progress)
                pair_rows.append({
                    "adapted_apex_frame": adapted.apex_frame,
                    "neutral_apex_frame": neutral.apex_frame,
                    "swing_side": adapted.swing_side,
                    "adapted_progress_fraction": adapted_progress,
                    "neutral_progress_fraction": neutral_progress,
                    "progress_delta": progress_delta,
                    "eligible": True,
                })
                # Lower tuple wins. Adapted lift is the only quality term; neutral affects
                # only correspondence feasibility and a deterministic progress tie-break.
                key = (
                    -float(adapted.event.relative_lift_m),
                    progress_delta,
                    int(row["seed"]),
                    int(adapted.apex_frame),
                    int(neutral.apex_frame),
                    adapted.swing_side,
                )
                feasible.append((key, row, adapted, neutral, phase_match))
        row["candidate_pairs"] = pair_rows
    if not feasible:
        raise ValueError("no donor seed contains a phase-alignable adapted/neutral step pair")
    _, row, adapted, neutral, phase_match = min(feasible, key=lambda value: value[0])
    return row, adapted, neutral, phase_match


def _actual_channel_usage(runner: ArdyRunner, spec: ConstraintSpec) -> dict[str, int]:
    _, mask = build_conditions(runner.model, spec, runner.device)
    return {name: count for name, count in channel_usage(runner.model, mask).items()
            if count}


def _assert_matched_programs(
    absolute_spec: ConstraintSpec,
    residual_spec: ConstraintSpec,
    absolute_info: Any,
    residual_info: Any,
    absolute_usage: Mapping[str, int],
    residual_usage: Mapping[str, int],
) -> None:
    if absolute_info.support_hash != residual_info.support_hash:
        raise ValueError("absolute/residual programs have different support hashes")
    if constraint_support_digest(absolute_spec) != constraint_support_digest(residual_spec):
        raise ValueError("absolute/residual ConstraintSpecs have different support")
    for name in ("rot_frames", "rot_joints"):
        if not np.array_equal(getattr(absolute_spec, name), getattr(residual_spec, name)):
            raise ValueError(f"absolute/residual programs differ in {name}")
    if absolute_spec.pos_frames is not None or residual_spec.pos_frames is not None:
        raise ValueError("E1 packets must not use the joint-position channel")
    if absolute_spec.heading is not None or residual_spec.heading is not None:
        raise ValueError("E1 packets must leave the body-heading feature free")
    if dict(absolute_usage) != dict(residual_usage):
        raise ValueError("absolute/residual programs have different actual ARDY mask usage")
    unexpected = set(absolute_usage) - ALLOWED_NONZERO_CHANNELS
    if unexpected:
        raise ValueError(f"E1 writes unexpected ARDY channels: {sorted(unexpected)}")
    missing = ALLOWED_NONZERO_CHANNELS - set(absolute_usage)
    if missing:
        raise ValueError(f"E1 failed to write required ARDY channels: {sorted(missing)}")
    if absolute_info.target_phase_match_hash != residual_info.target_phase_match_hash:
        raise ValueError("absolute/residual programs use different target phase receipts")


def _target_assignment(
    cycles: Sequence[StepPhaseCycle],
    qpos: np.ndarray,
    foot_kinematics: Mapping[str, Mapping[str, np.ndarray]],
    root_xz: np.ndarray,
    obstacle_x: Sequence[float],
    packet: Any,
    *,
    source_measurement_protocol_hash: str,
    half_window_frames: int,
    max_center_shift_frames: int,
) -> list[dict[str, Any]]:
    """One-to-one nominal-cycle assignment to fixed scenes, using nominal data only."""
    candidates = [cycle for cycle in cycles if cycle.swing_side == packet.swing_side]
    if len(candidates) < len(obstacle_x):
        raise ValueError(
            f"nominal clip has {len(candidates)} usable {packet.swing_side} cycles for "
            f"{len(obstacle_x)} fixed scenes"
        )
    best: tuple[tuple[Any, ...], list[dict[str, Any]]] | None = None
    for ordered in itertools.permutations(candidates, len(obstacle_x)):
        assignments: list[dict[str, Any]] = []
        cost = 0.0
        valid = True
        for scene_index, (x, cycle) in enumerate(zip(obstacle_x, ordered)):
            try:
                target_match = align_step_target_phase(
                    packet.phase_knots,
                    cycle,
                    source_measurement_protocol_hash=source_measurement_protocol_hash,
                    expected_swing_side=packet.swing_side,
                    search_half_window_frames=half_window_frames,
                )
            except ValueError:
                valid = False
                break
            side = cycle.swing_side
            foot = foot_kinematics[side]["forward_representative_m"]
            foot_offset = float(foot[cycle.apex_frame] - qpos[cycle.apex_frame, 0])
            desired_root = float(x) - foot_offset
            desired_frame = int(np.argmin(np.abs(root_xz[:, 1] - desired_root)))
            shift = desired_frame - cycle.apex_frame
            if abs(shift) > max_center_shift_frames:
                valid = False
                break
            target_queries = np.asarray(
                target_match.target_query_offsets_frames, dtype=float)
            first_target = desired_frame + int(np.ceil(target_queries[0] - 1e-12))
            last_target = desired_frame + int(np.floor(target_queries[-1] + 1e-12))
            if first_target < 0 or last_target >= len(root_xz):
                valid = False
                break
            predicted_foot = float(root_xz[desired_frame, 1] + foot_offset)
            spatial_error = predicted_foot - float(x)
            cost += abs(spatial_error) + 1e-6 * abs(shift)
            assignments.append({
                "scene_index": scene_index,
                "obstacle_x_m": float(x),
                "cycle": cycle,
                "target_phase_match": target_match,
                "controls": PacketControls(
                    strength=1.0,
                    center_shift_frames=int(shift),
                    duration_scale=1.0,
                ),
                "nominal_foot_forward_offset_m": foot_offset,
                "desired_root_frame": desired_frame,
                "predicted_foot_center_m": predicted_foot,
                "predicted_spatial_error_m": spatial_error,
            })
        if not valid:
            continue
        tie = tuple((entry["cycle"].apex_frame,
                     entry["controls"].center_shift_frames) for entry in assignments)
        key = (round(cost, 12), tie)
        if best is None or key < best[0]:
            best = (key, assignments)
    if best is None:
        raise ValueError("no bounded one-to-one target-cycle assignment for fixed scenes")
    return best[1]


def _score(
    body: Any,
    probe: Any,
    qpos: np.ndarray,
    root_xz: np.ndarray,
    obstacle_x: float,
    swing_side: str,
    fps: float,
    thresholds: StepOverThresholds,
    obstacle_height: float,
) -> dict[str, Any]:
    metrics = motion_metrics(
        body,
        probe.body(obstacle_height),
        probe,
        qpos,
        root_xz,
        obstacle_x,
        swing_side,
        fps=fps,
        thresholds=thresholds,
    )
    gates = metrics["local_step"]["gates"]
    metrics["kinematic_traversal_success"] = bool(
        gates["whole_body_collision_free"]
        and gates["root_traversal"]
        and gates["lateral_corridor"]
        and metrics["progress_ratio"] >= 0.8
    )
    metrics["kinematic_step_success"] = bool(
        metrics["local_step_success"] and metrics["progress_ratio"] >= 0.8
    )
    return metrics


def _program_deformation(
    spec: ConstraintSpec,
    nominal: Mapping[str, Any],
    fps: float,
) -> dict[str, Any]:
    """Pre-outcome program magnitude relative to the held-out nominal substrate.

    Root-height deformation is integrated as ``sum(abs(delta_y))/fps`` in metre-seconds
    and normalized as a mean over the full clip. Rotation deformation is the principal
    SO(3) geodesic angle for every constrained (frame, joint) pair, reported as both a sum
    and a mean over those pairs.
    """
    nominal_root = np.asarray(nominal["smooth_root_pos"], dtype=float)
    nominal_rot = np.asarray(nominal["global_rot_mats"], dtype=float)
    if spec.root_y is None or spec.rot_frames is None or spec.rot_joints is None:
        raise ValueError("RAMP E1 deformation requires root-y and rotation constraints")
    root_y = np.asarray(spec.root_y, dtype=float)
    if root_y.shape != (len(nominal_root),):
        raise ValueError("program root-y does not align with nominal clip")
    root_delta = root_y - nominal_root[:, 1]
    frames = np.asarray(spec.rot_frames, dtype=int)
    joints = np.asarray(spec.rot_joints, dtype=int)
    target = np.asarray(spec.rot_targets, dtype=float)
    reference = nominal_rot[frames[:, None], joints[None, :]]
    relative = target @ np.swapaxes(reference, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    angles = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return {
        "normalization": {
            "root_height": "full-clip frames; integral=sum(abs(delta_y))/fps",
            "rotation": "all constrained frame-joint pairs; principal SO(3) angle",
        },
        "root_height_integrated_abs_m_s": float(np.sum(np.abs(root_delta)) / fps),
        "root_height_mean_abs_m": float(np.mean(np.abs(root_delta))),
        "root_height_max_abs_m": float(np.max(np.abs(root_delta))),
        "rotation_sum_geodesic_rad": float(np.sum(angles)),
        "rotation_mean_geodesic_rad": float(np.mean(angles)),
        "rotation_max_geodesic_rad": float(np.max(angles)),
        "rotation_n_frame_joint_pairs": int(angles.size),
        "rotation_n_frames": int(len(frames)),
        "rotation_n_joints": int(len(joints)),
    }


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive residual-minus-absolute effects for crossed placement strata.

    The pilot reuses every evaluation seed at every fixed path position and varies neither
    obstacle geometry nor topology.  Those positions are therefore not independent scenes
    that license a scene-cluster confidence interval.  Confirmatory inference is deferred to
    a campaign with independently sampled scene geometries/topologies.
    """
    paths = {
        "max_box_height_lower_bound_m": ("max_box_height_lower_bound_m",),
        "obstacle_min_clearance_m": ("obstacle_min_clearance_m",),
        "kinematic_step_success": ("kinematic_step_success",),
        "progress_ratio": ("progress_ratio",),
        "mean_support_feet": ("mean_support_feet",),
        "bilateral_flight_fraction": ("bilateral_flight_fraction",),
        "max_foot_floor_penetration_m": ("max_foot_floor_penetration_m",),
        "root_height_integrated_abs_m_s": (
            "program_deformation", "root_height_integrated_abs_m_s"),
        "rotation_mean_geodesic_rad": (
            "program_deformation", "rotation_mean_geodesic_rad"),
    }

    def get(row: Mapping[str, Any], path: tuple[str, ...]) -> float:
        value: Any = row
        for key in path:
            value = value[key]
        return float(value)

    keyed = {(str(row["scene_id"]), int(row["seed"]), str(row["arm"])): row
             for row in rows}
    placements = sorted({str(row["scene_id"]) for row in rows})
    metrics: dict[str, Any] = {}
    for label, path in paths.items():
        placement_effects: dict[str, float] = {}
        for placement in placements:
            seeds = sorted({int(row["seed"]) for row in rows
                            if str(row["scene_id"]) == placement})
            diffs = []
            for eval_seed in seeds:
                absolute = keyed.get((placement, eval_seed, "absolute"))
                residual = keyed.get((placement, eval_seed, "residual"))
                if absolute is None or residual is None:
                    raise ValueError("paired summary found an incomplete arm pair")
                diffs.append(get(residual, path) - get(absolute, path))
            placement_effects[placement] = float(np.mean(diffs))
        values = np.asarray(list(placement_effects.values()), dtype=float)
        if not np.isfinite(values).all() or len(values) == 0:
            raise ValueError(f"paired summary metric {label} is empty or non-finite")
        metrics[label] = {
            "direction": "residual_minus_absolute",
            "placement_effects_after_collapsing_seeds": placement_effects,
            "descriptive_mean_over_placements": float(np.mean(values)),
        }
    return {
        "inference_unit": "descriptive only; fixed placements crossed with evaluation seeds",
        "interpretation": (
            "No confidence interval or p-value: the fixed positions share obstacle geometry, "
            "donor bundle, and evaluation seeds and are not independent scenes."
        ),
        "n_fixed_placements": len(placements),
        "metrics": metrics,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def _packet_arrays(pair: Any) -> dict[str, np.ndarray]:
    return {
        "absolute_rotation_payload": pair.absolute.rotation_payload,
        "absolute_root_height_payload_m": pair.absolute.root_height_payload_m,
        "residual_rotation_payload": pair.residual.rotation_payload,
        "residual_root_height_payload_m": pair.residual.root_height_payload_m,
        "source_offsets_frames": pair.absolute.source_offsets_frames,
        "taper": pair.absolute.taper,
        "phase_knots": pair.absolute.phase_knots,
        "adapted_query_offsets_frames": pair.absolute.adapted_query_offsets_frames,
        "neutral_query_offsets_frames": pair.residual.neutral_query_offsets_frames,
        "parent_indices": pair.absolute.parent_indices,
    }


def run_experiment(
    args: argparse.Namespace,
    *,
    runner: ArdyRunner | None = None,
    body: Any | None = None,
) -> dict[str, Any]:
    """Run exp017; dependency injection exists solely for CPU orchestration tests."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ExperimentAbort(
            f"single-shot non-resumable pilot refuses non-empty output directory: {out}")
    started = time.time()
    counters = {
        "donor_source_samples": 0,
        "nominal_samples": 0,
        "paired_evaluation_samples": 0,
        "generate_invocations": 0,
    }
    repo = Path(__file__).resolve().parents[1]
    stage = "validation"
    try:
        if args.n_donors < 1 or args.n_seeds < 1:
            raise ValueError("n_donors and n_seeds must be positive")
        if not args.obstacle_x:
            raise ValueError("at least one fixed obstacle placement is required")
        if len(set(args.obstacle_x)) != len(args.obstacle_x):
            raise ValueError("fixed obstacle placements must be unique")
        if args.half_window_frames < 1 or args.max_center_shift_frames < 0:
            raise ValueError("invalid packet window or center-shift bound")
        if args.duration <= 0 or args.speed <= 0:
            raise ValueError("duration and speed must be positive")
        if args.obstacle_height <= 0 or args.obstacle_depth <= 0:
            raise ValueError("obstacle dimensions must be positive")
        if not 0 < args.min_source_progress_ratio <= 1:
            raise ValueError("min_source_progress_ratio must lie in (0, 1]")
        if not 0 < args.min_nominal_progress_ratio <= 1:
            raise ValueError("min_nominal_progress_ratio must lie in (0, 1]")
        if not np.isfinite(np.asarray(args.obstacle_x, dtype=float)).all():
            raise ValueError("fixed obstacle placements must be finite")
        if not np.isfinite(np.asarray(args.cfg_weight, dtype=float)).all():
            raise ValueError("cfg_weight must be finite")
        donor_seeds = list(range(args.donor_seed_start,
                                 args.donor_seed_start + args.n_donors))
        eval_seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
        if set(donor_seeds) & set(eval_seeds):
            raise ValueError("donor and evaluation seed ranges must be disjoint")
        thresholds, threshold_identity = _load_thresholds(
            Path(args.threshold_calibration_receipt))
        min_stance_support_fraction = float(
            thresholds.min_contralateral_support_fraction)
        code = _git_state(repo)
        if code.get("dirty") is not False:
            raise ValueError(
                "exp017 requires an exactly clean git worktree before source generation"
            )

        runner = runner or ArdyRunner(cache_path="outputs/text_cache.npz")
        if (isinstance(runner.noise_stream_version, bool)
                or not isinstance(runner.noise_stream_version, (int, np.integer))
                or int(runner.noise_stream_version) != 2):
            raise ValueError("exp017 requires ARDY noise_stream_version == 2")
        body = body or G1Body(None)
        checkpoint = _checkpoint_identity(runner)
        fps = float(runner.fps)
        frames = int(round(args.duration * fps))
        if frames < 2 * args.half_window_frames + 3:
            raise ValueError("clip is too short for the locked packet window")
        root_xz = np.stack([
            np.zeros(frames), np.linspace(0.0, args.speed * args.duration, frames)
        ], axis=-1)
        route_heading = np.zeros(frames)
        route_lo, route_hi = float(root_xz[0, 1]), float(root_xz[-1, 1])
        for obstacle_x in args.obstacle_x:
            expanded_half = args.obstacle_depth / 2 + BODY_MARGIN
            if not (
                route_lo
                < float(obstacle_x) - expanded_half
                < float(obstacle_x) + expanded_half
                < route_hi
            ):
                raise ValueError(
                    f"expanded obstacle at x={float(obstacle_x):.3f} lies outside "
                    f"prescribed route [{route_lo:.3f}, {route_hi:.3f}]"
                )
        fixed_scenes = [{
            "scene_id": _scene_id(float(x), args.obstacle_height, args.obstacle_depth),
            "obstacle_x_m": float(x),
            "obstacle_height_m": args.obstacle_height,
            "obstacle_depth_m": args.obstacle_depth,
            "simulation_body_margin_m": BODY_MARGIN,
        } for x in args.obstacle_x]
        base = _base_spec(root_xz)
        code_revision = (
            f"{code.get('commit')}:{code.get('tracked_diff_sha256')}:"
            f"dirty={code.get('dirty')}"
        )
        cfg_weight = (float(args.cfg_weight[0]), float(args.cfg_weight[1]))
        generation_settings = {
            "runner_class": type(runner).__name__,
            "diffusion_steps": int(args.diffusion_steps),
            "cfg_weight": list(cfg_weight),
            "noise_stream_version": int(runner.noise_stream_version),
            "sampler": getattr(
                runner, "sampler_name",
                "ARDY deterministic DDIM, eta=0 (Scene2Motion runner contract)"),
            "history_frames": getattr(runner, "history_frames", None),
            "generation_horizon_frames": getattr(
                getattr(runner, "model", None), "gen_horizon_len", None),
            "fps": fps,
        }

        # ---- D matched source pairs -------------------------------------------------
        stage = "source_discovery"
        prompts: list[str] = []
        specs: list[ConstraintSpec] = []
        seeds: list[int] = []
        for seed in donor_seeds:
            prompts.extend((STEP, WALK))
            specs.extend((base, base))
            seeds.extend((seed, seed))
        generated = runner.generate(
            prompts, specs, frames, args.diffusion_steps,
            cfg_weight=cfg_weight, seeds=seeds)
        counters["generate_invocations"] += 1
        counters["donor_source_samples"] += len(generated)
        if len(generated) != 2 * args.n_donors:
            raise ValueError("runner returned the wrong number of donor source samples")
        source_rows: list[dict[str, Any]] = []
        source_samples: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for index, seed in enumerate(donor_seeds):
            adapted, neutral = generated[2 * index:2 * index + 2]
            adapted_qpos = runner.to_qpos(adapted)
            neutral_qpos = runner.to_qpos(neutral)
            adapted_progress_ratio = _prescribed_progress_ratio(adapted_qpos, root_xz)
            neutral_progress_ratio = _prescribed_progress_ratio(neutral_qpos, root_xz)
            row: dict[str, Any] = {
                "seed": seed,
                "adapted_prompt": STEP,
                "neutral_prompt": WALK,
                "adapted_clip_sha256": _sample_hash(adapted),
                "neutral_clip_sha256": _sample_hash(neutral),
                "adapted_final_progress_ratio_vs_prescribed_route": adapted_progress_ratio,
                "neutral_final_progress_ratio_vs_prescribed_route": neutral_progress_ratio,
            }
            try:
                if (
                    adapted_progress_ratio < args.min_source_progress_ratio
                    or neutral_progress_ratio < args.min_source_progress_ratio
                ):
                    raise ValueError(
                        "source progress below locked prescribed-route ratio "
                        f"{args.min_source_progress_ratio:.3f}"
                    )
                adapted_cycles = _phase_cycles(
                    body, adapted_qpos, fps, thresholds,
                    support_window_s=args.support_window_s,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=args.min_relative_lift_m,
                )
                neutral_cycles = _phase_cycles(
                    body, neutral_qpos, fps, thresholds,
                    support_window_s=args.support_window_s,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=args.min_relative_lift_m,
                )
                row.update({
                    "adapted_cycles": [cycle.as_dict() for cycle in adapted_cycles],
                    "neutral_cycles": [cycle.as_dict() for cycle in neutral_cycles],
                    "_adapted_qpos": adapted_qpos,
                    "_neutral_qpos": neutral_qpos,
                    "_adapted_cycles": adapted_cycles,
                    "_neutral_cycles": neutral_cycles,
                })
                source_samples[seed] = (adapted, neutral)
            except ValueError as exc:
                row.update({"adapted_cycles": [], "neutral_cycles": [],
                            "ineligible_reason": str(exc),
                            "_adapted_qpos": adapted_qpos,
                            "_neutral_qpos": neutral_qpos,
                            "_adapted_cycles": (), "_neutral_cycles": ()})
            source_rows.append(row)
        try:
            selected_row, adapted_cycle, neutral_cycle, source_phase_match = (
                _select_source_pair(source_rows, half_window_frames=args.half_window_frames)
            )
            selected_seed = int(selected_row["seed"])
            selected_row["selected"] = True
        finally:
            for row in source_rows:
                row.setdefault("selected", False)
            _write_jsonl(out / "donor_candidates.jsonl", source_rows)
        adapted_sample, neutral_sample = source_samples[selected_seed]
        selected_adapted_qpos = runner.to_qpos(adapted_sample)
        selected_neutral_qpos = runner.to_qpos(neutral_sample)
        selected_source_qpos = {
            "adapted_qpos": np.asarray(selected_adapted_qpos, dtype=np.float32),
            "neutral_qpos": np.asarray(selected_neutral_qpos, dtype=np.float32),
        }
        np.savez(out / "selected_source_qpos.npz", **selected_source_qpos)

        parents = np.asarray(runner.skeleton.joint_parents.detach().cpu().numpy(), dtype=int)
        base_provenance = {
            "adapted_clip_sha256": selected_row["adapted_clip_sha256"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "generator_id": checkpoint["generator_id"],
            "sampler_seed": selected_seed,
            "noise_stream_version": int(runner.noise_stream_version),
            "event_selector": EVENT_SELECTOR,
            "code_revision": code_revision,
        }
        pair = extract_packet_pair(
            adapted_sample,
            neutral_sample,
            adapted_cycle.event,
            neutral_cycle.event,
            adapted_route_heading=route_heading,
            neutral_route_heading=route_heading,
            phase_match=source_phase_match,
            joint_names=runner.joint_names,
            parent_indices=parents,
            root_idx=int(runner.skeleton.root_idx),
            source_fps=fps,
            half_window_frames=args.half_window_frames,
            absolute_provenance=dict(base_provenance),
            residual_provenance={
                **base_provenance,
                "neutral_clip_sha256": selected_row["neutral_clip_sha256"],
            },
        )
        if pair.absolute.measurement_protocol_hash != adapted_cycle.measurement_protocol_hash:
            raise ValueError("serialized packet lost its source measurement protocol identity")
        packet_arrays = _packet_arrays(pair)
        np.savez(out / "packet_pair.npz", **packet_arrays)
        packet_record = {
            **pair.metadata(),
            "absolute": pair.absolute.metadata(),
            "residual": pair.residual.metadata(),
            "adapted_cycle": adapted_cycle.as_dict(),
            "neutral_cycle": neutral_cycle.as_dict(),
            "source_phase_match": source_phase_match.as_dict(),
            "payload_content_sha256": _array_hash(packet_arrays),
            "npz_sha256": _sha256(out / "packet_pair.npz"),
        }
        _write_json(out / "packet_pair.json", packet_record)
        experiment_identity = _identity("exp017-experiment-v1", {
            "experiment": "exp017_ramp_residual_stepover",
            "execution_mode": "single-shot_non_resumable_pilot",
            "generator_id": checkpoint["generator_id"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "code_commit": code["commit"],
            "code_tracked_diff_sha256": code["tracked_diff_sha256"],
            "threshold_receipt_sha256": threshold_identity["sha256"],
            "threshold_normalized_receipt_sha256": (
                threshold_identity["normalized_receipt_sha256"]),
            "thresholds": asdict(thresholds),
            "generation_settings": generation_settings,
            "prompts": {"adapted_source": STEP, "neutral_source": WALK,
                        "nominal": WALK, "paired_evaluation": STEP},
            "donor_seeds": donor_seeds,
            "evaluation_seeds": eval_seeds,
            "fixed_scenes": fixed_scenes,
            "route_content_sha256": _array_hash({
                "root_xz": root_xz, "route_heading": route_heading}),
            "packet_pair_hash": pair.digest(),
            "measurement_protocol_hash": pair.absolute.measurement_protocol_hash,
            "selected_adapted_clip_sha256": selected_row["adapted_clip_sha256"],
            "selected_neutral_clip_sha256": selected_row["neutral_clip_sha256"],
            "selected_source_qpos_content_sha256": _array_hash(selected_source_qpos),
            "D": args.n_donors,
            "N": args.n_seeds,
            "P": len(args.obstacle_x),
        })

        # ---- N held-out nominal clips ------------------------------------------------
        stage = "nominal_discovery"
        nominals = runner.generate(
            [WALK] * args.n_seeds,
            [base] * args.n_seeds,
            frames,
            args.diffusion_steps,
            cfg_weight=cfg_weight,
            seeds=eval_seeds,
        )
        counters["generate_invocations"] += 1
        counters["nominal_samples"] += len(nominals)
        if len(nominals) != args.n_seeds:
            raise ValueError("runner returned the wrong number of nominal samples")

        nominal_rows: list[dict[str, Any]] = []
        nominal_qpos_archive: dict[str, np.ndarray] = {}
        program_records: list[dict[str, Any]] = []
        rendered: dict[tuple[int, int, str], ConstraintSpec] = {}
        for seed, nominal in zip(eval_seeds, nominals):
            qpos = runner.to_qpos(nominal)
            nominal_progress_ratio = _prescribed_progress_ratio(qpos, root_xz)
            if nominal_progress_ratio < args.min_nominal_progress_ratio:
                raise ValueError(
                    f"nominal seed {seed} progress ratio {nominal_progress_ratio:.4f} "
                    f"is below locked {args.min_nominal_progress_ratio:.4f}"
                )
            cycles = _phase_cycles(
                body, qpos, fps, thresholds,
                swing_side=pair.absolute.swing_side,
                support_window_s=args.support_window_s,
                min_stance_support_fraction=min_stance_support_fraction,
                min_relative_lift_m=args.min_relative_lift_m,
            )
            feet = foot_kinematics_series(body, qpos, fps)
            assignments = _target_assignment(
                cycles, qpos, feet, root_xz, args.obstacle_x, pair.absolute,
                source_measurement_protocol_hash=(
                    pair.absolute.measurement_protocol_hash),
                half_window_frames=args.half_window_frames,
                max_center_shift_frames=args.max_center_shift_frames,
            )
            nominal_hash = _sample_hash(nominal)
            nominal_qpos_archive[f"s{seed}"] = np.asarray(qpos, dtype=np.float32)
            nominal_rows.append({
                "seed": seed,
                "prompt": WALK,
                "clip_sha256": nominal_hash,
                "qpos_sha256": _array_hash({"qpos": qpos}),
                "final_progress_ratio_vs_prescribed_route": nominal_progress_ratio,
                "cycles": [cycle.as_dict() for cycle in cycles],
                "assignments": [{
                    key: (value.as_dict() if hasattr(value, "as_dict") else value)
                    for key, value in assignment.items() if key != "cycle"
                } | {"cycle": assignment["cycle"].as_dict()}
                    for assignment in assignments],
            })
            for assignment in assignments:
                scene_index = int(assignment["scene_index"])
                event = assignment["cycle"].event
                common = dict(
                    target_nominal=nominal,
                    target_event=event,
                    joint_names=runner.joint_names,
                    root_xz=root_xz,
                    route_heading=route_heading,
                    target_phase_match=assignment["target_phase_match"],
                    nominal_route_heading=route_heading,
                    target_fps=fps,
                    controls=assignment["controls"],
                    first_heading=0.0,
                )
                absolute_spec, absolute_info = render_packet(pair.absolute, **common)
                residual_spec, residual_info = render_packet(pair.residual, **common)
                absolute_usage = _actual_channel_usage(runner, absolute_spec)
                residual_usage = _actual_channel_usage(runner, residual_spec)
                _assert_matched_programs(
                    absolute_spec, residual_spec, absolute_info, residual_info,
                    absolute_usage, residual_usage,
                )
                scene_id = _scene_id(
                    float(assignment["obstacle_x_m"]),
                    args.obstacle_height,
                    args.obstacle_depth,
                )
                for arm, spec, info, usage in (
                    ("absolute", absolute_spec, absolute_info, absolute_usage),
                    ("residual", residual_spec, residual_info, residual_usage),
                ):
                    key = (seed, scene_index, arm)
                    rendered[key] = spec
                    deformation = _program_deformation(spec, nominal, fps)
                    program = {
                        "scene_id": scene_id,
                        "scene_index": scene_index,
                        "obstacle_x_m": float(assignment["obstacle_x_m"]),
                        "seed": seed,
                        "arm": arm,
                        "prompt": STEP,
                        "nominal_clip_sha256": nominal_hash,
                        "packet_hash": info.packet_hash,
                        "packet_pair_hash": pair.digest(),
                        "program_hash": info.program_hash,
                        "support_hash": info.support_hash,
                        "target_phase_match_hash": info.target_phase_match_hash,
                        "target_phase_match": assignment["target_phase_match"].as_dict(),
                        "target_cycle": assignment["cycle"].as_dict(),
                        "controls": assignment["controls"].as_dict(),
                        "target_frames": list(info.target_frames),
                        "joint_indices": list(info.joint_indices),
                        "channel_usage": dict(usage),
                        "deformation_vs_nominal": deformation,
                        "predicted_spatial_error_m": float(
                            assignment["predicted_spatial_error_m"]),
                    }
                    program_identity = _identity("exp017-program-v1", {
                        "experiment_identity_sha256": experiment_identity["sha256"],
                        "scene_id": scene_id,
                        "scene_index": scene_index,
                        "seed": seed,
                        "arm": arm,
                        "prompt": STEP,
                        "nominal_clip_sha256": nominal_hash,
                        "packet_hash": info.packet_hash,
                        "packet_pair_hash": pair.digest(),
                        "program_hash": info.program_hash,
                        "support_hash": info.support_hash,
                        "target_phase_match_hash": info.target_phase_match_hash,
                        "target_cycle_sha256": _json_hash(
                            assignment["cycle"].as_dict()),
                        "controls": assignment["controls"].as_dict(),
                        "channel_usage": dict(usage),
                        "deformation_sha256": _json_hash(deformation),
                    })
                    program.update({
                        "experiment_identity_sha256": experiment_identity["sha256"],
                        "program_identity": program_identity,
                        "program_identity_sha256": program_identity["sha256"],
                    })
                    program_records.append(program)
        _write_jsonl(out / "nominal_rows.jsonl", nominal_rows)
        np.savez(out / "nominal_qpos.npz", **nominal_qpos_archive)
        _write_jsonl(out / "programs.jsonl", program_records)

        # Freeze every decision informed by source/nominal outputs before final sampling.
        stage = "manifest"
        planned_samples = 2 * args.n_donors + args.n_seeds + (
            2 * args.n_seeds * len(args.obstacle_x))
        manifest = {
            "experiment": "exp017_ramp_residual_stepover",
            "design": "paired absolute-vs-phase-aligned-residual E1 pilot",
            "execution_mode": "single-shot_non_resumable_pilot",
            "resume_supported": False,
            "arms": list(ARMS),
            "prompts": {"adapted_source": STEP, "neutral_source": WALK,
                        "nominal": WALK, "paired_evaluation": STEP},
            "donor_seeds": donor_seeds,
            "evaluation_seeds": eval_seeds,
            "fixed_scenes": fixed_scenes,
            "D": args.n_donors,
            "N": args.n_seeds,
            "P": len(args.obstacle_x),
            "planned_ardy_samples": planned_samples,
            "budget_formula": "2D+N+2NP",
            "diffusion_steps": args.diffusion_steps,
            "generation_settings": generation_settings,
            "noise_stream_version": int(runner.noise_stream_version),
            "packet_pair": packet_record,
            "experiment_identity": experiment_identity,
            "experiment_identity_sha256": experiment_identity["sha256"],
            "threshold_calibration": threshold_identity,
            "thresholds": asdict(thresholds),
            "progress_gates": {
                "minimum_source_final_progress_ratio_vs_prescribed_route": (
                    args.min_source_progress_ratio),
                "minimum_nominal_final_progress_ratio_vs_prescribed_route": (
                    args.min_nominal_progress_ratio),
            },
            "qpos_evidence": {
                "selected_source_archive": "selected_source_qpos.npz",
                "selected_source_content_sha256": _array_hash(selected_source_qpos),
                "selected_source_archive_sha256": _sha256(
                    out / "selected_source_qpos.npz"),
                "nominal_archive": "nominal_qpos.npz",
                "nominal_content_sha256": _array_hash(nominal_qpos_archive),
                "nominal_archive_sha256": _sha256(out / "nominal_qpos.npz"),
            },
            "programs": program_records,
            "contains_final_outcomes": False,
            "provenance": {"checkpoint": checkpoint, "code": code},
        }
        _write_json(out / "manifest.json", manifest)
        manifest_sha = _sha256(out / "manifest.json")

        # ---- 2NP paired final outputs ------------------------------------------------
        stage = "paired_evaluation"
        probes = {
            index: BoxHeightProbe(float(x), args.obstacle_depth)
            for index, x in enumerate(args.obstacle_x)
        }
        rows: list[dict[str, Any]] = []
        qpos_archive: dict[str, np.ndarray] = {}
        record_by_key = {
            (int(row["seed"]), int(row["scene_index"]), str(row["arm"])): row
            for row in program_records
        }
        for seed in eval_seeds:
            for scene_index, x in enumerate(args.obstacle_x):
                # Duplicate per-sample seeds give both arms byte-identical advancing random
                # streams while preserving ARDY's batch-position independence guarantee.
                outputs = runner.generate(
                    [STEP, STEP],
                    [rendered[(seed, scene_index, "absolute")],
                     rendered[(seed, scene_index, "residual")]],
                    frames,
                    args.diffusion_steps,
                    cfg_weight=cfg_weight,
                    seeds=[seed, seed],
                )
                counters["generate_invocations"] += 1
                counters["paired_evaluation_samples"] += len(outputs)
                if len(outputs) != 2:
                    raise ValueError("runner returned the wrong paired-arm sample count")
                for arm, sample in zip(ARMS, outputs):
                    program = record_by_key[(seed, scene_index, arm)]
                    qpos = runner.to_qpos(sample)
                    metrics = _score(
                        body, probes[scene_index], qpos, root_xz, float(x),
                        pair.absolute.swing_side, fps, thresholds, args.obstacle_height,
                    )
                    motion_key = f"{program['scene_id']}__{arm}__s{seed}"
                    qpos_archive[motion_key] = np.asarray(qpos, dtype=np.float32)
                    sample_sha = _sample_hash(sample)
                    qpos_sha = _array_hash({"qpos": qpos})
                    metrics_sha = _json_hash(metrics)
                    output_identity = _identity("exp017-output-v1", {
                        "experiment_identity_sha256": experiment_identity["sha256"],
                        "program_identity_sha256": program["program_identity_sha256"],
                        "motion_key": motion_key,
                        "sample_sha256": sample_sha,
                        "qpos_sha256": qpos_sha,
                        "metrics_sha256": metrics_sha,
                        "seed": seed,
                        "arm": arm,
                        "scene_id": program["scene_id"],
                    })
                    rows.append({
                        **metrics,
                        "motion_key": motion_key,
                        "scene_id": program["scene_id"],
                        "scene_index": scene_index,
                        "seed": seed,
                        "arm": arm,
                        "prompt": STEP,
                        "sample_sha256": sample_sha,
                        "qpos_sha256": qpos_sha,
                        "metrics_sha256": metrics_sha,
                        "program_hash": program["program_hash"],
                        "support_hash": program["support_hash"],
                        "packet_hash": program["packet_hash"],
                        "target_phase_match_hash": program["target_phase_match_hash"],
                        "program_deformation": program["deformation_vs_nominal"],
                        "experiment_identity_sha256": experiment_identity["sha256"],
                        "program_identity_sha256": program["program_identity_sha256"],
                        "output_identity": output_identity,
                        "output_identity_sha256": output_identity["sha256"],
                        "manifest_sha256": manifest_sha,
                    })
        expected_final = 2 * args.n_seeds * len(args.obstacle_x)
        if len(rows) != expected_final or counters["paired_evaluation_samples"] != expected_final:
            raise ValueError("paired evaluation did not consume exactly 2NP samples")
        np.savez(out / "qpos.npz", **qpos_archive)
        _write_jsonl(out / "rows.jsonl", rows)
        summary = _paired_summary(rows)
        _write_json(out / "summary.json", summary)

        total_spent = sum(counters[name] for name in (
            "donor_source_samples", "nominal_samples", "paired_evaluation_samples"))
        if total_spent != planned_samples:
            raise ValueError("actual ARDY sample count differs from 2D+N+2NP")
        receipt = {
            "experiment": "exp017_ramp_residual_stepover",
            "status": "pilot_kinematics_complete_sonic_not_run",
            "claim_scope": "step-event representation E1; physical execution not evaluated",
            "execution_mode": "single-shot_non_resumable_pilot",
            "resume_supported": False,
            "manifest_sha256": manifest_sha,
            "D": args.n_donors,
            "N": args.n_seeds,
            "P": len(args.obstacle_x),
            "budget_formula": "2D+N+2NP",
            "planned_ardy_samples": planned_samples,
            "actual_ardy_samples": total_spent,
            "query_accounting": counters,
            "n_programs": len(program_records),
            "n_rows": len(rows),
            "packet_pair_hash": pair.digest(),
            "experiment_identity_sha256": experiment_identity["sha256"],
            "output_identity_set_sha256": _json_hash([
                row["output_identity_sha256"] for row in rows]),
            "rows_sha256": _sha256(out / "rows.jsonl"),
            "qpos_content_sha256": _array_hash(qpos_archive),
            "qpos_archive_sha256": _sha256(out / "qpos.npz"),
            "summary_sha256": _sha256(out / "summary.json"),
            "paired_summary": summary,
            "threshold_calibration": threshold_identity,
            "provenance": {
                "checkpoint": checkpoint,
                "code": code,
                "source_sha256": {
                    "experiments/exp017_ramp_residual_stepover.py": _sha256(
                        Path(__file__).resolve()),
                    "scene2motion/ramp/packet.py": _sha256(
                        repo / "scene2motion" / "ramp" / "packet.py"),
                    "scene2motion/ramp/phase.py": _sha256(
                        repo / "scene2motion" / "ramp" / "phase.py"),
                    "scene2motion/ramp/step_phase.py": _sha256(
                        repo / "scene2motion" / "ramp" / "step_phase.py"),
                },
            },
            "wall_clock_s": round(time.time() - started, 3),
        }
        _write_json(out / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        failure = {
            "experiment": "exp017_ramp_residual_stepover",
            "status": "failed_closed",
            "execution_mode": "single-shot_non_resumable_pilot",
            "resume_supported": False,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "query_accounting": counters,
            "wall_clock_s": round(time.time() - started, 3),
        }
        _write_json(out / "receipt.json", failure)
        if isinstance(exc, ExperimentAbort):
            raise
        raise ExperimentAbort(f"{stage}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/exp017_ramp_residual_stepover")
    parser.add_argument("--n_donors", type=int, default=12)
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--donor_seed_start", type=int, default=2600)
    parser.add_argument("--seed_start", type=int, default=2800)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--speed", type=float, default=0.90)
    parser.add_argument("--obstacle_x", type=float, nargs="+",
                        default=list(DEFAULT_OBSTACLE_X))
    parser.add_argument("--obstacle_height", type=float, default=0.08)
    parser.add_argument("--obstacle_depth", type=float, default=0.20)
    parser.add_argument("--diffusion_steps", type=int, default=5)
    parser.add_argument("--cfg_weight", type=float, nargs=2, default=(2.0, 2.0),
                        metavar=("TEXT", "CONSTRAINT"))
    parser.add_argument("--half_window_frames", type=int, default=2)
    parser.add_argument("--max_center_shift_frames", type=int, default=8)
    parser.add_argument("--support_window_s", type=float, default=0.24)
    parser.add_argument("--min_relative_lift_m", type=float, default=0.04)
    parser.add_argument("--min_source_progress_ratio", type=float, default=0.75)
    parser.add_argument("--min_nominal_progress_ratio", type=float, default=0.75)
    parser.add_argument("--threshold_calibration_receipt", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        receipt = run_experiment(args)
    except ExperimentAbort as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({
        "status": receipt["status"],
        "actual_ardy_samples": receipt["actual_ardy_samples"],
        "manifest_sha256": receipt["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
