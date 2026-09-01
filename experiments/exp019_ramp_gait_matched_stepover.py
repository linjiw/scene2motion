"""EXP-019: gait-matched placement and the paired absolute-vs-residual comparison.

Third placement design in the E1 family (``docs/ramp-e1-gait-matched-protocol.md``).
exp017 waited for a gait event to occur at a predeclared obstacle (1/8 eligible);
exp018 warped route timing to move the root to one (1/6 persisting, because the prior
re-plans its gait when root-path timing changes).  exp019 inverts the lever: the
obstacle is placed at the nominal's own swing apex on the nominal's own generating
route, so the exp017 center shift is zero by construction, no nominal is regenerated,
and no route is warped.

Per-seed placement supports the paired representation comparison and nothing else; it
is not a fixed scene and supports no scene-specified-placement claim.  The unmodified
nominal clip is scored against the same obstacle as a free third reference arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp017_ramp_residual_stepover as e17  # noqa: E402
from experiments import exp018_ramp_route_warped_stepover as e18  # noqa: E402
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.ramp import extract_packet_pair, render_packet  # noqa: E402
from scene2motion.robot import BODY_MARGIN, G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import BoxHeightProbe, foot_kinematics_series  # noqa: E402


SCHEMA_VERSION = "exp019-gait-matched-stepover-v2"
FAILURE_SCHEMA_VERSION = "exp019-gait-matched-stepover-failure-v2"

WALK = e17.WALK
STEP = e17.STEP
PACKET_ARMS = e17.ARMS
FPS = cal.FPS
N_FRAMES = cal.N_FRAMES

DONOR_SEEDS = e18.DONOR_SEEDS
# v2 pool, sized to the constructibility rate the v1 attempts measured (5/16 seeds; see
# docs/ramp-exp019-constructibility-2026-09-01.md).  K=32 fresh seeds yields ~10 expected
# constructible seeds against the unchanged N=8 requirement; the v1 seeds are closed.
POOL_SEEDS = tuple(range(4000, 4032))
POOL_BATCH_SIZE = 8
N_SELECT = 8
EXPECTED_SWING_SIDE = e18.EXPECTED_SWING_SIDE
STRATUM_ORDER = cal.STRATUM_SELECTION_ORDER

OBSTACLE_HEIGHT_M = e18.OBSTACLE_HEIGHT_M
OBSTACLE_DEPTH_M = e18.OBSTACLE_DEPTH_M
HALF_WINDOW_FRAMES = e18.HALF_WINDOW_FRAMES
SUPPORT_WINDOW_S = e18.SUPPORT_WINDOW_S
DONOR_MIN_RELATIVE_LIFT_M = e18.DONOR_MIN_RELATIVE_LIFT_M
TARGET_MIN_RELATIVE_LIFT_M = e18.TARGET_MIN_RELATIVE_LIFT_M
MIN_SOURCE_PROGRESS_RATIO = e18.MIN_SOURCE_PROGRESS_RATIO
OBSTACLE_HALF_EXTENT_M = OBSTACLE_DEPTH_M / 2 + BODY_MARGIN

SOURCE_FILES = (
    "experiments/exp019_ramp_gait_matched_stepover.py",
    "experiments/exp018_ramp_route_warped_stepover.py",
    "experiments/exp017_ramp_residual_stepover.py",
    "experiments/calibrate_ramp_route_phase.py",
    "scene2motion/ramp/packet.py",
    "scene2motion/ramp/phase_observability.py",
    "scene2motion/ramp/route_phase.py",
    "scene2motion/ramp/step_phase.py",
    "scene2motion/stepover_eval.py",
    "scene2motion/robot.py",
    "scene2motion/constraints.py",
    "scene2motion/runner.py",
)


class PilotAbort(RuntimeError):
    """Fail-closed pilot stop after durable evidence has been written."""


def _source_hashes(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in SOURCE_FILES:
        digest = cal._sha256(repo / name)
        if digest is None:
            raise ValueError(f"required source file is missing: {name}")
        result[name] = digest
    return result


def pool_batch_plan() -> tuple[dict[str, Any], ...]:
    """One batch invocation per (seed block, calibrated speed), eight samples each."""
    blocks = tuple(
        POOL_SEEDS[start:start + POOL_BATCH_SIZE]
        for start in range(0, len(POOL_SEEDS), POOL_BATCH_SIZE)
    )
    if any(len(block) != POOL_BATCH_SIZE for block in blocks):
        raise RuntimeError("locked pool no longer splits into blocks of eight")
    batches: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        for speed_label, speed in cal.SPEEDS:
            batches.append({
                "index": len(batches),
                "seed_block": block_index,
                "speed_label": speed_label,
                "requested_speed_mps": float(speed),
                "seeds": list(block),
            })
    return tuple(batches)


def _mean_of_defined(rows: Any, metric: str) -> float | None:
    """Mean over rows where the metric is present and not None.

    ``lead_matches_donor_side`` is None whenever no crossing was detected, so a bare
    ``float(row[metric])`` raises and takes the whole summary with it.
    """
    values = [
        float(row[metric]) for row in rows
        if row.get(metric) is not None
    ]
    return float(np.mean(values)) if values else None


def support_footfall_positions(
    foot_kinematics: Mapping[str, Mapping[str, np.ndarray]],
    thresholds: Any,
) -> np.ndarray:
    """Forward positions of both feet whenever they are in physical support.

    An obstacle footprint containing any of these is unwinnable for every arm: the
    robot puts a foot down inside it, so no whole-body box clears the ground there.
    """
    positions: list[np.ndarray] = []
    for side in ("left", "right"):
        clearance = np.asarray(
            foot_kinematics[side]["bottom_clearance_m"], dtype=float)
        speed = np.asarray(foot_kinematics[side]["planar_speed_mps"], dtype=float)
        forward = np.asarray(
            foot_kinematics[side]["forward_representative_m"], dtype=float)
        if clearance.shape != speed.shape or clearance.shape != forward.shape:
            raise ValueError("foot kinematics series are misaligned")
        supported = (
            (clearance <= thresholds.support_height_m)
            & (speed <= thresholds.support_speed_mps)
        )
        positions.append(forward[supported])
    return np.concatenate(positions) if positions else np.zeros(0, dtype=float)


def placeable_candidates(
    clip: Any,
    qpos: np.ndarray,
    foot_kinematics: Mapping[str, Mapping[str, np.ndarray]],
    route_xz: np.ndarray,
    *,
    target_min_prominence_m: float,
    thresholds: Any,
    swing_side: str = EXPECTED_SWING_SIDE,
) -> list[dict[str, Any]]:
    """Gait-matched obstacle candidates: one per complete, placeable swing cycle.

    The obstacle is placed at ``route_progress(apex) + foot_offset``, the swing foot's
    apex position measured against the *prescribed* route rather than the achieved root.
    exp017's assignment then solves ``desired_root = obstacle_x - foot_offset =
    route_progress(apex)``, whose nearest route frame is the apex itself, so the center
    shift is exactly zero rather than zero up to the ~1.4 cm root-tracking residual.
    Selection sees no arm outcome.
    """
    route = np.asarray(route_xz, dtype=float)
    low, high = float(route[0, 1]), float(route[-1, 1])
    midpoint = 0.5 * (low + high)
    forward = np.asarray(
        foot_kinematics[swing_side]["forward_representative_m"], dtype=float)
    if forward.shape != (len(qpos),) or not np.isfinite(forward).all():
        raise ValueError("swing-foot forward representative is invalid")
    footfalls = support_footfall_positions(foot_kinematics, thresholds)
    candidates: list[dict[str, Any]] = []
    for cycle in clip.cycles:
        row: dict[str, Any] = {
            "seed": clip.seed,
            "speed_label": clip.speed_label,
            "swing_side": cycle.swing_side,
            "apex_frame": cycle.apex_frame,
            "prominence_m": cycle.prominence_m,
            "packet_window_valid": cycle.packet_window_valid,
            "cycle_evidence_digest": cycle.evidence_digest,
            "placeable": False,
            "rejection": None,
        }
        if cycle.swing_side != swing_side:
            row["rejection"] = "wrong_swing_side"
            candidates.append(row)
            continue
        if cycle.prominence_m < target_min_prominence_m:
            row["rejection"] = "prominence_below_frozen_target_gate"
            candidates.append(row)
            continue
        if not cycle.packet_window_valid:
            row["rejection"] = "packet_half_window_two_not_supported"
            candidates.append(row)
            continue
        foot_offset = float(forward[cycle.apex_frame]) - float(
            qpos[cycle.apex_frame, 0])
        obstacle_x = float(route[cycle.apex_frame, 1]) + foot_offset
        row["nominal_foot_forward_offset_m"] = foot_offset
        row["achieved_foot_x_m"] = float(forward[cycle.apex_frame])
        row["route_progress_at_apex_m"] = float(route[cycle.apex_frame, 1])
        row["obstacle_x_m"] = obstacle_x
        row["route_low_m"] = low
        row["route_high_m"] = high
        row["obstacle_half_extent_m"] = OBSTACLE_HALF_EXTENT_M
        if not (
            low < obstacle_x - OBSTACLE_HALF_EXTENT_M
            and obstacle_x + OBSTACLE_HALF_EXTENT_M < high
        ):
            row["rejection"] = "expanded_obstacle_outside_route"
            candidates.append(row)
            continue
        nearest_footfall = (
            float(np.min(np.abs(footfalls - obstacle_x))) if footfalls.size
            else float("inf")
        )
        row["nearest_support_footfall_m"] = nearest_footfall
        if nearest_footfall <= OBSTACLE_HALF_EXTENT_M:
            # The swing arcs over this x, but a footfall of either foot lands inside the
            # expanded footprint, so no arm - packet or nominal - can clear the box.
            row["rejection"] = "support_footfall_inside_obstacle_footprint"
            candidates.append(row)
            continue
        row.update({
            "placeable": True,
            "distance_from_route_midpoint_m": abs(obstacle_x - midpoint),
            "selection_key": [
                abs(obstacle_x - midpoint),
                -float(cycle.prominence_m),
                int(STRATUM_ORDER[clip.speed_label]),
                int(cycle.apex_frame),
                cycle.evidence_digest,
            ],
        })
        candidates.append(row)
    return candidates


def select_placement(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Frozen outcome-free key: mid-route, then prominence, stratum, apex, digest.

    Only candidates that already survived the constructibility probe are eligible.
    """
    placeable = [
        row for row in candidates
        if row.get("placeable") and row.get("constructible")
    ]
    if not placeable:
        return None
    return dict(min(placeable, key=lambda row: tuple(row["selection_key"])))


def probe_constructibility(
    candidate: Mapping[str, Any],
    *,
    pair: Any,
    sample: Mapping[str, Any],
    qpos: np.ndarray,
    foot_kinematics: Mapping[str, Mapping[str, np.ndarray]],
    route_xz: np.ndarray,
    route_heading: np.ndarray,
    thresholds: Any,
    min_stance_support_fraction: float,
    runner: Any,
    body: Any,
) -> dict[str, Any]:
    """Can this candidate actually be turned into a matched pair of programs?

    Placement eligibility is measured with the phase-observability enumerator, but the
    packet transport consumes ``step_phase`` cycles and then a rendered ConstraintSpec.
    Those are different gates, so a candidate that passes the first can still be
    unbuildable.  This probe runs the real assignment, both renders, and the channel
    assertions, so selection can never choose something the render stage will reject.

    It is outcome-free: it consumes only the nominal clip and the frozen packet, and no
    generated arm response, collision result, or traversal outcome exists yet.
    """
    result: dict[str, Any] = {"constructible": False, "construct_rejection": None}
    try:
        cycles = e17._phase_cycles(
            body, qpos, FPS, thresholds, swing_side=EXPECTED_SWING_SIDE,
            support_window_s=SUPPORT_WINDOW_S,
            min_stance_support_fraction=min_stance_support_fraction,
            min_relative_lift_m=TARGET_MIN_RELATIVE_LIFT_M)
    except ValueError as exc:
        result["construct_rejection"] = f"step_phase_cycles: {exc}"
        return result
    matching = [
        cycle for cycle in cycles
        if int(cycle.apex_frame) == int(candidate["apex_frame"])
    ]
    if not matching:
        result["construct_rejection"] = (
            "no step_phase cycle at the observability apex "
            f"{candidate['apex_frame']} (available: "
            f"{sorted(int(c.apex_frame) for c in cycles)})"
        )
        return result
    try:
        assignments, _ = e17._target_assignment(
            matching, qpos, foot_kinematics, route_xz,
            [float(candidate["obstacle_x_m"])], pair.absolute,
            source_common_physical_protocol_hash=(
                pair.absolute.common_physical_protocol_hash),
            half_window_frames=HALF_WINDOW_FRAMES,
            max_center_shift_frames=0)
    except (e17.TargetAssignmentError, ValueError) as exc:
        result["construct_rejection"] = f"target_assignment: {exc}"
        return result
    assignment = assignments[0]
    if int(assignment["controls"].center_shift_frames) != 0:
        result["construct_rejection"] = "nonzero center shift"
        return result
    common = dict(
        target_nominal=sample, target_event=assignment["cycle"].event,
        joint_names=runner.joint_names, root_xz=route_xz,
        route_heading=route_heading,
        target_phase_match=assignment["target_phase_match"],
        nominal_route_heading=route_heading, target_fps=FPS,
        controls=assignment["controls"], first_heading=0.0)
    try:
        absolute_spec, absolute_info = render_packet(pair.absolute, **common)
        residual_spec, residual_info = render_packet(pair.residual, **common)
        absolute_usage = e17._actual_channel_usage(runner, absolute_spec)
        residual_usage = e17._actual_channel_usage(runner, residual_spec)
        e17._assert_matched_programs(
            absolute_spec, residual_spec, absolute_info, residual_info,
            absolute_usage, residual_usage)
    except (TypeError, ValueError) as exc:
        result["construct_rejection"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update({
        "constructible": True,
        "_assignment": assignment,
        "_specs": {"absolute": absolute_spec, "residual": residual_spec},
        "_infos": {"absolute": absolute_info, "residual": residual_info},
        "_usages": {"absolute": absolute_usage, "residual": residual_usage},
    })
    return result


def run_pilot(
    *,
    out: str | Path,
    threshold_receipt: str | Path,
    v3_receipt: str | Path = e18.V3_RECEIPT_PATH,
    runner: Any | None = None,
    body: Any | None = None,
    code_state_fn: Any = cal._git_state,
    generator_identity_fn: Any = cal._generator_identity,
    cache_path: str | Path = "outputs/text_cache.npz",
) -> dict[str, Any]:
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise PilotAbort(f"refusing nonempty pilot output directory: {output}")
    repo = Path(__file__).resolve().parents[1]
    code = dict(code_state_fn(repo))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "preflight"
    sample_count_exact = True
    counters = {
        "generate_invocations": 0,
        "donor_samples_launched": 0,
        "donor_samples_returned": 0,
        "pool_samples_launched": 0,
        "pool_samples_returned": 0,
        "arm_samples_launched": 0,
        "arm_samples_returned": 0,
    }
    donor_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    program_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    qpos_archives: dict[str, dict[str, np.ndarray]] = {
        "donor_qpos": {}, "pool_qpos": {}, "arm_qpos": {},
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp019_ramp_gait_matched_stepover",
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": stage,
        "resume_supported": False,
        "design": {
            "placement_lever": (
                "obstacle placed at the nominal's own swing apex on the nominal's own "
                "generating route; exp017 center shift is zero by construction"
            ),
            "scope": (
                "per-seed placement supports the paired representation comparison only; "
                "it is not a fixed scene and supports no scene-specified-placement claim"
            ),
            "donor_seeds": list(DONOR_SEEDS),
            "pool_seeds": list(POOL_SEEDS),
            "n_select": N_SELECT,
            "expected_swing_side": EXPECTED_SWING_SIDE,
            "arms": ["nominal", *PACKET_ARMS],
            "nominal_arm_is_free": (
                "the unmodified pool clip scored against the same obstacle; no sample"
            ),
            "obstacle_height_m": OBSTACLE_HEIGHT_M,
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "obstacle_half_extent_m": OBSTACLE_HALF_EXTENT_M,
            "half_window_frames": HALF_WINDOW_FRAMES,
            "center_shift_frames": 0,
            "selection_key": [
                "abs(obstacle_x - route midpoint)", "-prominence",
                "stratum order (reference, slow, fast)", "apex frame",
                "phase-evidence receipt digest",
            ],
            "pool_batches": [dict(batch) for batch in pool_batch_plan()],
            "budget_equation": "2D + 3K + 2N",
        },
        "query_accounting": counters,
        "sample_count_exact": True,
        "provenance": {},
    }

    def persist() -> None:
        for name, rows in (
            ("donor_candidates", donor_rows), ("pool_rows", pool_rows),
            ("placement_rows", placement_rows), ("program_rows", program_rows),
            ("arm_rows", arm_rows),
        ):
            cal._write_jsonl(output / f"{name}.jsonl", rows)
        for name, archive in qpos_archives.items():
            cal._persist_qpos(output / f"{name}.npz", archive)
        receipt["stage"] = stage
        receipt["query_accounting"] = dict(counters)
        receipt["sample_count_exact"] = sample_count_exact
        receipt["evidence_anchors"] = {
            **{
                name: {
                    "path": f"{name}.jsonl", "n_rows": len(rows),
                    "logical_sha256": cal._json_hash(rows),
                    "file_sha256": cal._sha256(output / f"{name}.jsonl"),
                }
                for name, rows in (
                    ("donor_candidates", donor_rows), ("pool_rows", pool_rows),
                    ("placement_rows", placement_rows),
                    ("program_rows", program_rows), ("arm_rows", arm_rows),
                )
            },
            **{
                name: {
                    "path": f"{name}.npz", "n_arrays": len(archive),
                    "content_sha256": cal._array_hash(archive) if archive else None,
                    "archive_sha256": cal._sha256(output / f"{name}.npz"),
                }
                for name, archive in qpos_archives.items()
            },
        }
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        cal._write_json(output / "receipt.json", receipt)

    def generate(prompts, specs, seeds, *, launched_counter, returned_counter):
        nonlocal sample_count_exact
        counters["generate_invocations"] += 1
        counters[launched_counter] += len(seeds)
        persist()
        try:
            returned = runner.generate(
                list(prompts), list(specs), N_FRAMES, cal.DIFFUSION_STEPS,
                cfg_weight=cal.CFG_WEIGHT, seeds=list(seeds),
            )
        except Exception:
            sample_count_exact = False
            raise
        counters[returned_counter] += len(returned)
        if len(returned) != len(seeds):
            sample_count_exact = False
            raise ValueError(
                f"runner returned {len(returned)} samples for {len(seeds)} planned")
        return list(returned)

    try:
        if code.get("dirty") is not False:
            raise ValueError("exp019 requires an exactly clean git worktree")
        if os.environ.get("CHECKPOINTS_DIR"):
            raise ValueError("exp019 forbids ambient CHECKPOINTS_DIR")
        receipt["provenance"]["code"] = code
        source_hashes = _source_hashes(repo)
        receipt["provenance"]["source_sha256"] = source_hashes
        thresholds, threshold_dependency = cal._load_physical_threshold_dependency(
            Path(threshold_receipt))
        receipt["provenance"]["physical_threshold_dependency"] = threshold_dependency
        min_stance_support_fraction = float(
            thresholds.min_contralateral_support_fraction)
        calibration = e18.load_v3_calibration(Path(v3_receipt))
        calibration.pop("route_timing_bounds")
        pmin = float(calibration["target_min_prominence_m"])
        receipt["provenance"]["route_phase_calibration"] = {
            **calibration,
            "consumed_quantities": (
                "frozen target prominence gate only; exp019 warps no route, so the "
                "timing caps and placement set do not apply"
            ),
        }
        archived = e18.load_archived_donor_bundle(repo)
        receipt["provenance"]["archived_donor_bundle"] = archived
        physical_model = cal._physical_model_identity()
        receipt["provenance"]["physical_model"] = physical_model

        runner = runner or ArdyRunner(cache_path=cache_path)
        if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
            raise ValueError(f"exp019 requires runner fps == {FPS:g}")
        if (
            isinstance(runner.noise_stream_version, (bool, np.bool_))
            or not isinstance(runner.noise_stream_version, (int, np.integer))
            or int(runner.noise_stream_version) != cal.NOISE_STREAM_VERSION
        ):
            raise ValueError("exp019 requires ARDY noise_stream_version == 2")
        body = body or G1Body(None)
        generator_identity = cal._validated_generator_identity(
            generator_identity_fn(runner))
        runtime_identity = cal._runtime_identity()
        receipt["provenance"]["generator"] = generator_identity
        receipt["provenance"]["runtime"] = runtime_identity
        code_revision = (
            f"{code.get('commit')}:{code.get('tracked_diff_sha256')}:"
            f"dirty={code.get('dirty')}")

        def revalidate_identities() -> dict[str, Any]:
            current = dict(code_state_fn(repo))
            if current.get("commit") != code.get("commit"):
                raise ValueError("git commit changed during the pilot")
            if current.get("tracked_diff_sha256") != code.get("tracked_diff_sha256"):
                raise ValueError("tracked git diff changed during the pilot")
            if _source_hashes(repo) != source_hashes:
                raise ValueError("source content changed during the pilot")
            if cal._physical_model_identity() != physical_model:
                raise ValueError("physical G1 model changed during the pilot")
            if (
                cal._validated_generator_identity(generator_identity_fn(runner))
                != generator_identity
            ):
                raise ValueError("generator identity changed during the pilot")
            if cal._runtime_identity() != runtime_identity:
                raise ValueError("runtime identity changed during the pilot")
            return {"commit_unchanged": True, "sources_unchanged": True,
                    "physical_model_unchanged": True, "generator_unchanged": True,
                    "runtime_unchanged": True}

        # ---- donor bank ------------------------------------------------------------
        stage = "donor_generation"
        persist()
        reference_route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)
        route_heading = np.zeros(N_FRAMES, dtype=float)
        base_spec = e17._base_spec(reference_route)
        prompts: list[str] = []
        specs: list[ConstraintSpec] = []
        seeds: list[int] = []
        for seed in DONOR_SEEDS:
            prompts.extend((STEP, WALK))
            specs.extend((base_spec, base_spec))
            seeds.extend((seed, seed))
        generated = generate(
            prompts, specs, seeds,
            launched_counter="donor_samples_launched",
            returned_counter="donor_samples_returned")

        stage = "donor_selection"
        source_samples: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        mismatches: list[str] = []
        for index, seed in enumerate(DONOR_SEEDS):
            adapted, neutral = generated[2 * index:2 * index + 2]
            adapted_qpos = np.array(runner.to_qpos(adapted), copy=True)
            neutral_qpos = np.array(runner.to_qpos(neutral), copy=True)
            qpos_archives["donor_qpos"][f"s{seed}__adapted"] = adapted_qpos
            qpos_archives["donor_qpos"][f"s{seed}__neutral"] = neutral_qpos
            adapted_hash = e17._sample_hash(adapted)
            neutral_hash = e17._sample_hash(neutral)
            archived_adapted, archived_neutral = archived[
                "archived_clip_sha256_by_seed"][str(seed)]
            if adapted_hash != archived_adapted:
                mismatches.append(f"seed {seed} adapted")
            if neutral_hash != archived_neutral:
                mismatches.append(f"seed {seed} neutral")
            adapted_ratio = e17._prescribed_progress_ratio(
                adapted_qpos, reference_route)
            neutral_ratio = e17._prescribed_progress_ratio(
                neutral_qpos, reference_route)
            row: dict[str, Any] = {
                "seed": seed, "adapted_prompt": STEP, "neutral_prompt": WALK,
                "adapted_clip_sha256": adapted_hash,
                "neutral_clip_sha256": neutral_hash,
                "archived_clip_hashes_match": (
                    adapted_hash == archived_adapted
                    and neutral_hash == archived_neutral),
                "adapted_final_progress_ratio_vs_prescribed_route": adapted_ratio,
                "neutral_final_progress_ratio_vs_prescribed_route": neutral_ratio,
            }
            try:
                if (
                    adapted_ratio < MIN_SOURCE_PROGRESS_RATIO
                    or neutral_ratio < MIN_SOURCE_PROGRESS_RATIO
                ):
                    raise ValueError(
                        "source progress below locked prescribed-route ratio "
                        f"{MIN_SOURCE_PROGRESS_RATIO:.3f}")
                adapted_cycles = e17._phase_cycles(
                    body, adapted_qpos, FPS, thresholds,
                    support_window_s=SUPPORT_WINDOW_S,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=DONOR_MIN_RELATIVE_LIFT_M)
                neutral_cycles = e17._phase_cycles(
                    body, neutral_qpos, FPS, thresholds,
                    support_window_s=SUPPORT_WINDOW_S,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=DONOR_MIN_RELATIVE_LIFT_M)
                row.update({
                    "adapted_cycles": [c.as_dict() for c in adapted_cycles],
                    "neutral_cycles": [c.as_dict() for c in neutral_cycles],
                    "_adapted_qpos": adapted_qpos, "_neutral_qpos": neutral_qpos,
                    "_adapted_cycles": adapted_cycles,
                    "_neutral_cycles": neutral_cycles,
                })
                source_samples[seed] = (adapted, neutral)
            except ValueError as exc:
                row.update({
                    "adapted_cycles": [], "neutral_cycles": [],
                    "ineligible_reason": str(exc),
                    "_adapted_qpos": adapted_qpos, "_neutral_qpos": neutral_qpos,
                    "_adapted_cycles": (), "_neutral_cycles": (),
                })
            donor_rows.append(row)
        if mismatches:
            raise ValueError(
                "donor regeneration is not byte-identical to the archived bundle "
                "(determinism refusal): " + "; ".join(mismatches))
        try:
            selected_row, adapted_cycle, neutral_cycle, source_phase_match = (
                e17._select_source_pair(
                    donor_rows, half_window_frames=HALF_WINDOW_FRAMES))
            selected_donor_seed = int(selected_row["seed"])
            selected_row["selected"] = True
        finally:
            for row in donor_rows:
                row.setdefault("selected", False)
            persist()
        if selected_donor_seed != e18.EXPECTED_SELECTED_DONOR_SEED:
            raise ValueError(
                f"donor selection chose seed {selected_donor_seed}, not the archived "
                f"seed {e18.EXPECTED_SELECTED_DONOR_SEED}")
        adapted_sample, neutral_sample = source_samples[selected_donor_seed]

        stage = "packet_extraction"
        parents = np.asarray(
            runner.skeleton.joint_parents.detach().cpu().numpy(), dtype=int)
        base_provenance = {
            "adapted_clip_sha256": selected_row["adapted_clip_sha256"],
            "checkpoint_sha256": generator_identity["checkpoint"]["checkpoint_sha256"],
            "generator_id": generator_identity["checkpoint"]["generator_id"],
            "sampler_seed": selected_donor_seed,
            "noise_stream_version": int(runner.noise_stream_version),
            "event_selector": e17.EVENT_SELECTOR,
            "code_revision": code_revision,
        }
        pair = extract_packet_pair(
            adapted_sample, neutral_sample, adapted_cycle.event, neutral_cycle.event,
            adapted_route_heading=route_heading, neutral_route_heading=route_heading,
            phase_match=source_phase_match, joint_names=runner.joint_names,
            parent_indices=parents, root_idx=int(runner.skeleton.root_idx),
            source_fps=FPS, half_window_frames=HALF_WINDOW_FRAMES,
            absolute_provenance=dict(base_provenance),
            residual_provenance={
                **base_provenance,
                "neutral_clip_sha256": selected_row["neutral_clip_sha256"],
            },
        )
        if pair.absolute.swing_side != EXPECTED_SWING_SIDE:
            raise ValueError("regenerated packet swing side is not the locked side")
        if int(pair.absolute.adapted_center_frame) != e18.EXPECTED_ADAPTED_CENTER_FRAME:
            raise ValueError("regenerated packet center frame differs from the archive")
        packet_arrays = e17._packet_arrays(pair)
        payload_content = e17._array_hash(packet_arrays)
        if payload_content != e18.EXPECTED_PACKET_PAYLOAD_CONTENT_SHA256:
            raise ValueError(
                "regenerated packet payload differs from the archived bundle while clip "
                f"hashes match (measurement-code drift refusal): {payload_content}")
        np.savez(output / "packet_pair.npz", **packet_arrays)
        cal._write_json(output / "packet_pair.json", {
            **pair.metadata(),
            "absolute": pair.absolute.metadata(),
            "residual": pair.residual.metadata(),
            "adapted_cycle": adapted_cycle.as_dict(),
            "neutral_cycle": neutral_cycle.as_dict(),
            "source_phase_match": source_phase_match.as_dict(),
            "payload_content_sha256": payload_content,
            "payload_matches_archive": True,
            "npz_sha256": cal._sha256(output / "packet_pair.npz"),
        })
        receipt["packet_pair"] = {
            "pair_hash": pair.digest(),
            "payload_content_sha256": payload_content,
            "swing_side": pair.absolute.swing_side,
            "selected_donor_seed": selected_donor_seed,
        }
        persist()

        # ---- nominal pool ----------------------------------------------------------
        stage = "pool_generation"
        clips_by_key: dict[tuple[int, str], Any] = {}
        samples_by_key: dict[tuple[int, str], Any] = {}
        qpos_by_key: dict[tuple[int, str], np.ndarray] = {}
        routes_by_label = {
            label: cal.route_xz_for_speed(speed) for label, speed in cal.SPEEDS
        }
        for batch in pool_batch_plan():
            spec = cal.root_only_walk_spec(batch["requested_speed_mps"])
            returned = generate(
                [WALK] * POOL_BATCH_SIZE, [spec] * POOL_BATCH_SIZE, batch["seeds"],
                launched_counter="pool_samples_launched",
                returned_counter="pool_samples_returned")
            for seed, sample in zip(batch["seeds"], returned):
                qpos = np.asarray(runner.to_qpos(sample), dtype=float)
                key = f"pilot__{batch['speed_label']}__seed{seed}"
                qpos_archives["pool_qpos"][key] = np.array(qpos, copy=True)
                clip = cal.analyze_generated_clip(
                    body=body, qpos=qpos, sample=sample, split="pilot", seed=seed,
                    speed_label=batch["speed_label"],
                    requested_speed_mps=batch["requested_speed_mps"],
                    thresholds=thresholds, qpos_archive_key=key)
                clips_by_key[(seed, batch["speed_label"])] = clip
                samples_by_key[(seed, batch["speed_label"])] = sample
                qpos_by_key[(seed, batch["speed_label"])] = np.array(qpos, copy=True)
                pool_rows.append({"batch_index": batch["index"], **clip.as_dict()})
            persist()

        # ---- outcome-free gait-matched placement -----------------------------------
        stage = "placement_selection"
        receipt["provenance"]["post_pool_identity_revalidation"] = (
            revalidate_identities())
        selection_by_seed: dict[int, dict[str, Any]] = {}
        probe_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
        for seed in POOL_SEEDS:
            all_candidates: list[dict[str, Any]] = []
            for label, _ in cal.SPEEDS:
                clip = clips_by_key[(seed, label)]
                qpos = qpos_by_key[(seed, label)]
                try:
                    feet = foot_kinematics_series(body, qpos, FPS)
                except (KeyError, TypeError, ValueError) as exc:
                    all_candidates.append({
                        "seed": seed, "speed_label": label, "placeable": False,
                        "rejection": f"foot_kinematics: {exc}",
                    })
                    continue
                for candidate in placeable_candidates(
                    clip, qpos, feet, routes_by_label[label],
                    target_min_prominence_m=pmin, thresholds=thresholds,
                ):
                    if candidate.get("placeable"):
                        probe = probe_constructibility(
                            candidate, pair=pair,
                            sample=samples_by_key[(seed, label)], qpos=qpos,
                            foot_kinematics=feet, route_xz=routes_by_label[label],
                            route_heading=route_heading, thresholds=thresholds,
                            min_stance_support_fraction=min_stance_support_fraction,
                            runner=runner, body=body)
                        probe_by_key[
                            (seed, label, int(candidate["apex_frame"]))] = probe
                        candidate.update({
                            key: value for key, value in probe.items()
                            if not key.startswith("_")
                        })
                    all_candidates.append(candidate)
            chosen = select_placement(all_candidates)
            placement_rows.append({
                "seed": seed,
                "n_candidates": len(all_candidates),
                "n_placeable": sum(1 for c in all_candidates if c.get("placeable")),
                "n_constructible": sum(
                    1 for c in all_candidates if c.get("constructible")),
                "eligible": chosen is not None,
                "selected": chosen,
                "candidates": all_candidates,
            })
            if chosen is not None:
                selection_by_seed[seed] = chosen
        eligible_seeds = sorted(selection_by_seed)
        selected_seeds = eligible_seeds[:N_SELECT]
        receipt["placement_selection"] = {
            "frozen_pmin_m": pmin,
            "eligibility_definition": (
                "phase-observability cycle above frozen Pmin with a valid packet "
                "window, an in-route obstacle at route_progress(apex)+foot_offset, and "
                "a successful outcome-free constructibility probe (step_phase cycle at "
                "the same apex, zero-shift assignment, both renders, channel assertions)"
            ),
            "eligible_seeds": eligible_seeds,
            "eligibility": f"{len(eligible_seeds)}/{len(POOL_SEEDS)}",
            "selected_seeds": selected_seeds,
            "per_seed": {
                str(seed): {
                    key: selection_by_seed[seed][key]
                    for key in ("speed_label", "apex_frame", "obstacle_x_m",
                                "prominence_m", "distance_from_route_midpoint_m")
                }
                for seed in selected_seeds
            },
        }
        persist()
        if len(eligible_seeds) < N_SELECT:
            raise ValueError(
                f"pool has {len(eligible_seeds)} placeable {EXPECTED_SWING_SIDE}-side "
                f"seeds; requires N={N_SELECT} from frozen K={len(POOL_SEEDS)}")

        # ---- programs (shift 0 by construction) ------------------------------------
        stage = "program_render"
        rendered: dict[tuple[int, str], ConstraintSpec] = {}
        context: dict[int, dict[str, Any]] = {}
        for seed in selected_seeds:
            chosen = selection_by_seed[seed]
            label = str(chosen["speed_label"])
            route = routes_by_label[label]
            obstacle_x = float(chosen["obstacle_x_m"])
            sample = samples_by_key[(seed, label)]
            qpos = qpos_by_key[(seed, label)]
            clip = clips_by_key[(seed, label)]
            # The selection probe already built these outcome-free; reuse them so the
            # rendered programs are byte-identical to what eligibility was decided on.
            probe = probe_by_key[(seed, label, int(chosen["apex_frame"]))]
            if not probe.get("constructible"):
                raise RuntimeError("selected candidate is not constructible")
            assignment = probe["_assignment"]
            scene_id = e17._scene_id(obstacle_x, OBSTACLE_HEIGHT_M, OBSTACLE_DEPTH_M)
            context[seed] = {
                "speed_label": label, "route": route, "obstacle_x_m": obstacle_x,
                "scene_id": scene_id, "qpos": qpos, "clip": clip,
                "target_apex_frame": int(assignment["cycle"].apex_frame),
            }
            for arm in PACKET_ARMS:
                spec = probe["_specs"][arm]
                info = probe["_infos"][arm]
                usage = probe["_usages"][arm]
                rendered[(seed, arm)] = spec
                program_rows.append({
                    "seed": seed, "arm": arm, "prompt": STEP, "scene_id": scene_id,
                    "obstacle_x_m": obstacle_x, "speed_label": label,
                    "target_apex_frame": int(assignment["cycle"].apex_frame),
                    "center_shift_frames": 0,
                    "predicted_spatial_error_m": float(
                        assignment["predicted_spatial_error_m"]),
                    "packet_hash": info.packet_hash,
                    "program_hash": info.program_hash,
                    "support_hash": info.support_hash,
                    "target_phase_match_hash": info.target_phase_match_hash,
                    "controls": assignment["controls"].as_dict(),
                    "channel_usage": dict(usage),
                    "deformation_vs_nominal": e17._program_deformation(
                        spec, sample, FPS),
                })
        persist()

        # ---- free nominal reference arm --------------------------------------------
        stage = "nominal_scoring"
        for seed in selected_seeds:
            info = context[seed]
            row: dict[str, Any] = {
                "seed": seed, "arm": "nominal", "prompt": WALK,
                "scene_id": info["scene_id"], "obstacle_x_m": info["obstacle_x_m"],
                "speed_label": info["speed_label"], "costs_no_sample": True,
                "status": "returned",
            }
            arm_rows.append(row)
            try:
                metrics = e17._score(
                    body, BoxHeightProbe(info["obstacle_x_m"], OBSTACLE_DEPTH_M),
                    info["qpos"], info["route"], info["obstacle_x_m"],
                    EXPECTED_SWING_SIDE, FPS, thresholds, OBSTACLE_HEIGHT_M)
                row.update({**metrics, "status": "completed"})
            except Exception as exc:  # noqa: BLE001 - kept in the denominator
                row.update({"status": "failed", "error_type": type(exc).__name__,
                            "error": str(exc)})
            persist()

        # ---- paired packet arms ----------------------------------------------------
        stage = "arm_generation"
        arm_plan = [(seed, arm) for seed in selected_seeds for arm in PACKET_ARMS]
        arm_returned = generate(
            [STEP] * len(arm_plan), [rendered[key] for key in arm_plan],
            [seed for seed, _ in arm_plan],
            launched_counter="arm_samples_launched",
            returned_counter="arm_samples_returned")

        stage = "arm_scoring"
        program_by_key = {(row["seed"], row["arm"]): row for row in program_rows}
        for (seed, arm), sample in zip(arm_plan, arm_returned):
            program = program_by_key[(seed, arm)]
            info = context[seed]
            row = {
                "seed": seed, "arm": arm, "prompt": STEP,
                "scene_id": program["scene_id"],
                "obstacle_x_m": program["obstacle_x_m"],
                "speed_label": program["speed_label"], "costs_no_sample": False,
                "status": "returned", "sample_sha256": e17._sample_hash(sample),
            }
            arm_rows.append(row)
            try:
                qpos = np.array(runner.to_qpos(sample), copy=True)
                key = f"s{seed}__{arm}"
                qpos_archives["arm_qpos"][key] = np.asarray(qpos, dtype=np.float32)
                row["qpos_content_sha256"] = e17._array_hash(
                    {key: qpos_archives["arm_qpos"][key]})
                metrics = e17._score(
                    body, BoxHeightProbe(info["obstacle_x_m"], OBSTACLE_DEPTH_M),
                    qpos, info["route"], info["obstacle_x_m"], EXPECTED_SWING_SIDE,
                    FPS, thresholds, OBSTACLE_HEIGHT_M)
                row.update({
                    **metrics,
                    "program_hash": program["program_hash"],
                    "support_hash": program["support_hash"],
                    "packet_hash": program["packet_hash"],
                    "program_deformation": program["deformation_vs_nominal"],
                    "status": "completed",
                })
            except Exception as exc:  # noqa: BLE001 - kept in the denominator
                row.update({"status": "failed", "error_type": type(exc).__name__,
                            "error": str(exc)})
            persist()

        stage = "summary"
        receipt["provenance"]["post_arm_identity_revalidation"] = (
            revalidate_identities())
        completed = [row for row in arm_rows if row["status"] == "completed"]
        by_arm = {
            arm: {row["seed"]: row for row in completed if row["arm"] == arm}
            for arm in ("nominal", *PACKET_ARMS)
        }
        complete_seeds = sorted(
            set(by_arm["absolute"]) & set(by_arm["residual"]))
        metrics_of_interest = (
            "max_box_height_lower_bound_m", "obstacle_min_clearance_m",
            "obstacle_collision_free", "kinematic_step_success", "progress_ratio",
            "phase_error_m", "lead_matches_donor_side",
        )

        def delta(metric: str, high: str, low: str) -> dict[str, Any]:
            values: dict[str, float] = {}
            for seed in complete_seeds:
                if seed not in by_arm[high] or seed not in by_arm[low]:
                    continue
                try:
                    values[str(seed)] = float(by_arm[high][seed][metric]) - float(
                        by_arm[low][seed][metric])
                except (KeyError, TypeError, ValueError):
                    continue
            array = np.asarray(list(values.values()), dtype=float)
            return {
                "per_seed": values,
                "n": int(array.size),
                "n_positive": int(np.sum(array > 0)) if array.size else 0,
                "n_negative": int(np.sum(array < 0)) if array.size else 0,
                "median": float(np.median(array)) if array.size else None,
                "mean": float(np.mean(array)) if array.size else None,
            }

        receipt["summary"] = {
            "inference": (
                "descriptive, paired on seed; per-seed placements are strata, not "
                "independent scenes. No confidence interval or p-value."
            ),
            "planned_arm_denominator_per_packet_arm": len(selected_seeds),
            "completed_counts": {
                arm: len(by_arm[arm]) for arm in ("nominal", *PACKET_ARMS)},
            "seeds_with_complete_packet_pairs": complete_seeds,
            "arm_means": {
                arm: {
                    metric: _mean_of_defined(by_arm[arm].values(), metric)
                    for metric in metrics_of_interest
                }
                for arm in ("nominal", *PACKET_ARMS)
            },
            "residual_minus_absolute": {
                metric: delta(metric, "residual", "absolute")
                for metric in metrics_of_interest
            },
            "absolute_minus_nominal": {
                metric: delta(metric, "absolute", "nominal")
                for metric in metrics_of_interest
            },
            "residual_minus_nominal": {
                metric: delta(metric, "residual", "nominal")
                for metric in metrics_of_interest
            },
        }

        stage = "complete"
        total = (counters["donor_samples_returned"]
                 + counters["pool_samples_returned"]
                 + counters["arm_samples_returned"])
        expected = (2 * len(DONOR_SEEDS) + 3 * len(POOL_SEEDS)
                    + 2 * len(selected_seeds))
        receipt.update({
            "status": "complete", "complete": True, "blocked": False, "stage": stage,
            "actual_ardy_samples": total,
            "conservative_charged_ardy_samples": total,
            "budget_check": {"equation": "2D + 3K + 2N", "expected": expected,
                             "actual": total},
        })
        if expected != total:
            raise RuntimeError("final sample accounting does not match the design")
        persist()
        return receipt
    except Exception as exc:
        launched = sum(counters[n] for n in counters if n.endswith("_launched"))
        returned = sum(counters[n] for n in counters if n.endswith("_returned"))
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION, "status": "blocked", "complete": False,
            "blocked": True, "failed_stage": stage, "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_ardy_samples": returned if sample_count_exact else None,
            "returned_ardy_samples_lower_bound": returned,
            "conservative_charged_ardy_samples": max(launched, returned),
        })
        persist()
        if isinstance(exc, PilotAbort):
            raise
        raise PilotAbort(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/exp019_gait_matched_stepover")
    parser.add_argument(
        "--threshold-receipt",
        default="outputs/exp016_threshold_calibration/receipt.json")
    parser.add_argument("--v3-receipt", default=e18.V3_RECEIPT_PATH)
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    receipt = run_pilot(
        out=args.out, threshold_receipt=args.threshold_receipt,
        v3_receipt=args.v3_receipt, cache_path=args.cache_path)
    print(json.dumps({
        "status": receipt["status"],
        "actual_ardy_samples": receipt.get("actual_ardy_samples"),
        "eligibility": receipt.get("placement_selection", {}).get("eligibility"),
    }, indent=2))


if __name__ == "__main__":
    main()
