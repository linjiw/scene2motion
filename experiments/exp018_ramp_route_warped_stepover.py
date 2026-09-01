"""EXP-018: route-warped placement persistence and paired packet arms.

Successor to the closed exp017 fixed-frame pool on the calibrated route-phase
foundation (``docs/ramp-e1-route-warped-protocol.md``).  Stage A regenerates each
selected nominal seed under its outcome-free calibrated route-progress warp and
measures whether a complete same-side swing still meets the fixed obstacle within
exp017's unchanged +/-8-frame placement gate.  Stage B generates one absolute and one
residual packet arm for every persisting seed on the same warped route and scores the
exp017 E1a endpoint vector.

Frozen dependencies are hash-locked: the exp016 physical support thresholds, the v3
route-phase calibration receipt (Pmin, timing caps, placement set, selection-key
semantics), and exp017's archived donor bundle.  The donor bank is regenerated
deterministically and must reproduce the archived clip hashes and packet payload
content exactly; a clip-hash mismatch is a determinism refusal and a payload mismatch
with matching clips is a measurement-code-drift refusal.  Every stage persists
incremental hash-anchored evidence and every failure is fail-closed with the stage
recorded.
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
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.ramp import extract_packet_pair, render_packet  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import (  # noqa: E402
    BoxHeightProbe,
    foot_kinematics_series,
)


SCHEMA_VERSION = "exp018-route-warped-stepover-v1"
FAILURE_SCHEMA_VERSION = "exp018-route-warped-stepover-failure-v1"

WALK = e17.WALK
STEP = e17.STEP
ARMS = e17.ARMS
FPS = cal.FPS
N_FRAMES = cal.N_FRAMES
ROUTE_LENGTH_M = cal.PILOT_ROUTE_LENGTH_M
DIFFUSION_STEPS = cal.DIFFUSION_STEPS
CFG_WEIGHT = cal.CFG_WEIGHT
NOISE_STREAM_VERSION = cal.NOISE_STREAM_VERSION

DONOR_SEEDS = (2600, 2601, 2602, 2603)
POOL_SEEDS = tuple(range(3800, 3816))
POOL_BATCH_SIZE = 8
N_SELECT = 6
EXPECTED_SWING_SIDE = "left"

OBSTACLE_HEIGHT_M = 0.08
OBSTACLE_DEPTH_M = 0.20
HALF_WINDOW_FRAMES = 2
MAX_CENTER_SHIFT_FRAMES = 8
SUPPORT_WINDOW_S = 0.24
DONOR_MIN_RELATIVE_LIFT_M = 0.04
# Target discovery deliberately does not inherit the donor-quality lift gate: the
# calibration protocol separates neutral-swing discovery from packet-source quality.
TARGET_MIN_RELATIVE_LIFT_M = 0.0
MIN_SOURCE_PROGRESS_RATIO = 0.75
MIN_WARPED_PROGRESS_RATIO = 0.75

V3_RECEIPT_PATH = "outputs/calibrate_ramp_route_phase_v3/receipt.json"
V3_RECEIPT_FILE_SHA256 = (
    "745c8ad3c7784c686ba03434a84c980b5f1a6be65b5b48fc16fa6973e31c2b58"
)
ARCHIVED_PACKET_PAIR_JSON = "outputs/exp017_ramp_pool_d4_k8_n2_p1/packet_pair.json"
ARCHIVED_PACKET_PAIR_JSON_SHA256 = (
    "3bdaace6d9d1161cb579fb469fa14e733808cc78bc4033f254c17e836d7da773"
)
ARCHIVED_DONOR_CANDIDATES_JSONL = (
    "outputs/exp017_ramp_pool_d4_k8_n2_p1/donor_candidates.jsonl"
)
EXPECTED_SELECTED_DONOR_SEED = 2603
EXPECTED_ADAPTED_CLIP_SHA256 = (
    "928870d0caacd5c89c24bbf2919b1ff1b819cd7a856187d6e54011ea0dd404db"
)
EXPECTED_NEUTRAL_CLIP_SHA256 = (
    "b243a4e0950fca9732b4c5d3e61d1df915fb3c20f4010dad9da8136b80a8cab1"
)
EXPECTED_ADAPTED_CENTER_FRAME = 55
EXPECTED_PACKET_PAYLOAD_CONTENT_SHA256 = (
    "2f70610b3d19e316f71ab6f5a8101413e17e6b6f393aacdab6c4995b00209aca"
)

SOURCE_FILES = (
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
    """Six batch invocations of eight: two seed blocks x three calibrated speeds."""
    blocks = (POOL_SEEDS[:POOL_BATCH_SIZE], POOL_SEEDS[POOL_BATCH_SIZE:])
    if any(len(block) != POOL_BATCH_SIZE for block in blocks):
        raise RuntimeError("locked pool no longer splits into two blocks of eight")
    batches: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        for speed_label, speed in cal.SPEEDS:
            batches.append(
                {
                    "index": len(batches),
                    "seed_block": block_index,
                    "speed_label": speed_label,
                    "requested_speed_mps": float(speed),
                    "seeds": list(block),
                }
            )
    return tuple(batches)


def pmin_from_receipt(receipt: Mapping[str, Any]) -> float:
    """Verify and extract the frozen target prominence gate from the v3 receipt."""
    target = receipt["target_prominence_receipt"]
    if target["schema"] != cal.PROMINENCE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("v3 target-prominence receipt has an unexpected schema")
    if target["sha256"] != cal._json_hash(
        {"schema": target["schema"], "fields": target["fields"]}
    ):
        raise ValueError("v3 target-prominence receipt digest mismatch")
    pmin = float(target["fields"]["target_min_prominence_m"])
    if not np.isfinite(pmin) or pmin <= 0.0:
        raise ValueError("frozen Pmin is not a positive finite value")
    return pmin


def load_v3_calibration(path: Path) -> dict[str, Any]:
    """Load and hash-verify the passing v3 calibration; return frozen quantities."""
    digest = cal._sha256(path)
    if digest != V3_RECEIPT_FILE_SHA256:
        raise ValueError(
            "route-phase calibration receipt file hash is not the locked v3 artifact"
        )
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != cal.CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("v3 calibration receipt has an unexpected campaign schema")
    if receipt.get("status") != "complete" or receipt.get("complete") is not True:
        raise ValueError("v3 calibration receipt is not a completed passing campaign")
    if receipt.get("validation", {}).get("passed") is not True:
        raise ValueError("v3 calibration validation did not pass")
    design_placements = receipt["campaign_design"]["fields"]["route_program"][
        "event_placements_m"
    ]
    if tuple(float(v) for v in design_placements) != cal.EVENT_PLACEMENTS_M:
        raise ValueError("v3 placement set does not match the module constant")
    bounds = cal.timing_bounds_from_receipt(receipt["route_timing_receipt"])
    return {
        "path": str(path),
        "file_sha256": digest,
        "target_min_prominence_m": pmin_from_receipt(receipt),
        "route_timing_bounds": bounds,
        "route_timing_receipt_sha256": receipt["route_timing_receipt"]["sha256"],
        "target_prominence_receipt_sha256": receipt["target_prominence_receipt"][
            "sha256"
        ],
        "event_placements_m": list(cal.EVENT_PLACEMENTS_M),
    }


def warped_route_xz(program: Any) -> np.ndarray:
    """Materialize the straight pilot route under one frozen progress program."""
    progress = np.asarray(program.progress_m, dtype=float)
    if progress.shape != (N_FRAMES,) or not np.isfinite(progress).all():
        raise ValueError("route program progress does not cover the locked clip")
    if abs(progress[0]) > 1e-9 or abs(progress[-1] - ROUTE_LENGTH_M) > 1e-6:
        raise ValueError("route program endpoints do not span the locked pilot route")
    if np.any(np.diff(progress) <= 0.0):
        raise ValueError("route program progress is not strictly increasing")
    return np.stack([np.zeros(N_FRAMES, dtype=float), progress], axis=-1)


def load_archived_donor_bundle(repo: Path) -> dict[str, Any]:
    """Load the hash-locked exp017 donor archive that exp018 must reproduce."""
    packet_path = repo / ARCHIVED_PACKET_PAIR_JSON
    digest = cal._sha256(packet_path)
    if digest != ARCHIVED_PACKET_PAIR_JSON_SHA256:
        raise ValueError("archived packet_pair.json hash is not the locked artifact")
    record = json.loads(packet_path.read_text())
    absolute = record["absolute"]
    if absolute["swing_side"] != EXPECTED_SWING_SIDE:
        raise ValueError("archived donor packet swing side is not the locked side")
    if int(absolute["provenance"]["sampler_seed"]) != EXPECTED_SELECTED_DONOR_SEED:
        raise ValueError("archived donor packet seed is not the locked donor")
    donors_path = repo / ARCHIVED_DONOR_CANDIDATES_JSONL
    donor_rows = [json.loads(line) for line in donors_path.open()]
    archived_clips = {
        int(row["seed"]): (row["adapted_clip_sha256"], row["neutral_clip_sha256"])
        for row in donor_rows
    }
    if sorted(archived_clips) != sorted(DONOR_SEEDS):
        raise ValueError("archived donor candidates do not cover the locked bank")
    return {
        "packet_pair_json": str(packet_path),
        "packet_pair_json_sha256": digest,
        "archived_pair_hash": record["pair_hash"],
        "archived_payload_content_sha256": record["payload_content_sha256"],
        "archived_measurement_protocol_hash": record["measurement_protocol_hash"],
        "donor_candidates_jsonl": str(donors_path),
        "donor_candidates_jsonl_sha256": cal._sha256(donors_path),
        "archived_clip_sha256_by_seed": {
            str(seed): list(archived_clips[seed]) for seed in sorted(archived_clips)
        },
    }


def run_pilot(
    *,
    out: str | Path,
    threshold_receipt: str | Path,
    v3_receipt: str | Path = V3_RECEIPT_PATH,
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
        "warped_samples_launched": 0,
        "warped_samples_returned": 0,
        "arm_samples_launched": 0,
        "arm_samples_returned": 0,
    }
    donor_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    warped_rows: list[dict[str, Any]] = []
    program_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    qpos_archives: dict[str, dict[str, np.ndarray]] = {
        "donor_qpos": {},
        "pool_qpos": {},
        "warped_qpos": {},
        "warped_substrate": {},
        "arm_qpos": {},
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp018_ramp_route_warped_stepover",
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": stage,
        "resume_supported": False,
        "design": {
            "donor_seeds": list(DONOR_SEEDS),
            "pool_seeds": list(POOL_SEEDS),
            "n_select": N_SELECT,
            "expected_swing_side": EXPECTED_SWING_SIDE,
            "obstacle_height_m": OBSTACLE_HEIGHT_M,
            "obstacle_depth_m": OBSTACLE_DEPTH_M,
            "half_window_frames": HALF_WINDOW_FRAMES,
            "max_center_shift_frames": MAX_CENTER_SHIFT_FRAMES,
            "support_window_s": SUPPORT_WINDOW_S,
            "donor_min_relative_lift_m": DONOR_MIN_RELATIVE_LIFT_M,
            "target_min_relative_lift_m": TARGET_MIN_RELATIVE_LIFT_M,
            "min_source_progress_ratio": MIN_SOURCE_PROGRESS_RATIO,
            "min_warped_progress_ratio": MIN_WARPED_PROGRESS_RATIO,
            "prompts": {"adapted_source": STEP, "neutral_source": WALK,
                        "pool": WALK, "warped_nominal": WALK, "arms": STEP},
            "pool_batches": [dict(batch) for batch in pool_batch_plan()],
            "budget_equation": "2D + 3K + N + 2M = 8 + 48 + N + 2M",
        },
        "query_accounting": counters,
        "sample_count_exact": True,
        "provenance": {},
    }

    def persist() -> None:
        cal._write_jsonl(output / "donor_candidates.jsonl", donor_rows)
        cal._write_jsonl(output / "pool_rows.jsonl", pool_rows)
        cal._write_jsonl(output / "warped_rows.jsonl", warped_rows)
        cal._write_jsonl(output / "program_rows.jsonl", program_rows)
        cal._write_jsonl(output / "arm_rows.jsonl", arm_rows)
        for name, archive in qpos_archives.items():
            cal._persist_qpos(output / f"{name}.npz", archive)
        receipt["stage"] = stage
        receipt["query_accounting"] = dict(counters)
        receipt["sample_count_exact"] = sample_count_exact
        receipt["evidence_anchors"] = {
            **{
                name: {
                    "path": f"{name}.jsonl",
                    "n_rows": len(rows),
                    "logical_sha256": cal._json_hash(rows),
                    "file_sha256": cal._sha256(output / f"{name}.jsonl"),
                }
                for name, rows in (
                    ("donor_candidates", donor_rows),
                    ("pool_rows", pool_rows),
                    ("warped_rows", warped_rows),
                    ("program_rows", program_rows),
                    ("arm_rows", arm_rows),
                )
            },
            **{
                name: {
                    "path": f"{name}.npz",
                    "n_arrays": len(archive),
                    "content_sha256": (
                        cal._array_hash(archive) if archive else None
                    ),
                    "archive_sha256": cal._sha256(output / f"{name}.npz"),
                }
                for name, archive in qpos_archives.items()
            },
        }
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        cal._write_json(output / "receipt.json", receipt)

    def generate(
        prompts: Sequence[str],
        specs: Sequence[ConstraintSpec],
        seeds: Sequence[int],
        *,
        launched_counter: str,
        returned_counter: str,
    ) -> list[Any]:
        nonlocal sample_count_exact
        counters["generate_invocations"] += 1
        counters[launched_counter] += len(seeds)
        persist()
        try:
            returned = runner.generate(
                list(prompts), list(specs), N_FRAMES, DIFFUSION_STEPS,
                cfg_weight=CFG_WEIGHT, seeds=list(seeds),
            )
        except Exception:
            sample_count_exact = False
            raise
        counters[returned_counter] += len(returned)
        if len(returned) != len(seeds):
            sample_count_exact = False
            raise ValueError(
                f"runner returned {len(returned)} samples for {len(seeds)} planned"
            )
        return list(returned)

    try:
        if code.get("dirty") is not False:
            raise ValueError("exp018 requires an exactly clean git worktree")
        if os.environ.get("CHECKPOINTS_DIR"):
            raise ValueError("exp018 forbids ambient CHECKPOINTS_DIR")
        receipt["provenance"]["code"] = code
        source_hashes = _source_hashes(repo)
        receipt["provenance"]["source_sha256"] = source_hashes
        thresholds, threshold_dependency = cal._load_physical_threshold_dependency(
            Path(threshold_receipt)
        )
        receipt["provenance"]["physical_threshold_dependency"] = threshold_dependency
        min_stance_support_fraction = float(
            thresholds.min_contralateral_support_fraction
        )
        calibration = load_v3_calibration(Path(v3_receipt))
        frozen_bounds = calibration.pop("route_timing_bounds")
        pmin = float(calibration["target_min_prominence_m"])
        receipt["provenance"]["route_phase_calibration"] = {
            **calibration,
            "route_timing_bounds": frozen_bounds.as_dict(),
        }
        archived = load_archived_donor_bundle(repo)
        receipt["provenance"]["archived_donor_bundle"] = archived
        physical_model = cal._physical_model_identity()
        receipt["provenance"]["physical_model"] = physical_model

        runner = runner or ArdyRunner(cache_path=cache_path)
        if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
            raise ValueError(f"exp018 requires runner fps == {FPS:g}")
        if (
            isinstance(runner.noise_stream_version, (bool, np.bool_))
            or not isinstance(runner.noise_stream_version, (int, np.integer))
            or int(runner.noise_stream_version) != NOISE_STREAM_VERSION
        ):
            raise ValueError("exp018 requires ARDY noise_stream_version == 2")
        body = body or G1Body(None)
        generator_identity = cal._validated_generator_identity(
            generator_identity_fn(runner)
        )
        runtime_identity = cal._runtime_identity()
        receipt["provenance"]["generator"] = generator_identity
        receipt["provenance"]["runtime"] = runtime_identity
        code_revision = (
            f"{code.get('commit')}:{code.get('tracked_diff_sha256')}:"
            f"dirty={code.get('dirty')}"
        )

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
            return {
                "commit_unchanged": True,
                "sources_unchanged": True,
                "physical_model_unchanged": True,
                "generator_unchanged": True,
                "runtime_unchanged": True,
            }

        # ---- donor bank: deterministic regeneration of the archived bundle --------
        stage = "donor_generation"
        persist()
        linear_reference_route = cal.route_xz_for_speed(cal.REFERENCE_SPEED_MPS)
        route_heading = np.zeros(N_FRAMES, dtype=float)
        base_spec = e17._base_spec(linear_reference_route)
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
            returned_counter="donor_samples_returned",
        )

        stage = "donor_selection"
        source_samples: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        clip_hash_mismatches: list[str] = []
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
                clip_hash_mismatches.append(f"seed {seed} adapted")
            if neutral_hash != archived_neutral:
                clip_hash_mismatches.append(f"seed {seed} neutral")
            adapted_ratio = e17._prescribed_progress_ratio(
                adapted_qpos, linear_reference_route)
            neutral_ratio = e17._prescribed_progress_ratio(
                neutral_qpos, linear_reference_route)
            row: dict[str, Any] = {
                "seed": seed,
                "adapted_prompt": STEP,
                "neutral_prompt": WALK,
                "adapted_clip_sha256": adapted_hash,
                "neutral_clip_sha256": neutral_hash,
                "archived_clip_hashes_match": (
                    adapted_hash == archived_adapted
                    and neutral_hash == archived_neutral
                ),
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
                        f"{MIN_SOURCE_PROGRESS_RATIO:.3f}"
                    )
                adapted_cycles = e17._phase_cycles(
                    body, adapted_qpos, FPS, thresholds,
                    support_window_s=SUPPORT_WINDOW_S,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=DONOR_MIN_RELATIVE_LIFT_M,
                )
                neutral_cycles = e17._phase_cycles(
                    body, neutral_qpos, FPS, thresholds,
                    support_window_s=SUPPORT_WINDOW_S,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=DONOR_MIN_RELATIVE_LIFT_M,
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
                row.update({
                    "adapted_cycles": [], "neutral_cycles": [],
                    "ineligible_reason": str(exc),
                    "_adapted_qpos": adapted_qpos, "_neutral_qpos": neutral_qpos,
                    "_adapted_cycles": (), "_neutral_cycles": (),
                })
            donor_rows.append(row)
        if clip_hash_mismatches:
            raise ValueError(
                "donor regeneration is not byte-identical to the archived bundle "
                "(determinism refusal): " + "; ".join(clip_hash_mismatches)
            )
        try:
            selected_row, adapted_cycle, neutral_cycle, source_phase_match = (
                e17._select_source_pair(
                    donor_rows, half_window_frames=HALF_WINDOW_FRAMES)
            )
            selected_donor_seed = int(selected_row["seed"])
            selected_row["selected"] = True
        finally:
            for row in donor_rows:
                row.setdefault("selected", False)
            persist()
        if selected_donor_seed != EXPECTED_SELECTED_DONOR_SEED:
            raise ValueError(
                f"donor selection chose seed {selected_donor_seed}, not the archived "
                f"seed {EXPECTED_SELECTED_DONOR_SEED}"
            )
        adapted_sample, neutral_sample = source_samples[selected_donor_seed]

        stage = "packet_extraction"
        parents = np.asarray(
            runner.skeleton.joint_parents.detach().cpu().numpy(), dtype=int)
        base_provenance = {
            "adapted_clip_sha256": selected_row["adapted_clip_sha256"],
            "checkpoint_sha256": generator_identity["checkpoint"][
                "checkpoint_sha256"],
            "generator_id": generator_identity["checkpoint"]["generator_id"],
            "sampler_seed": selected_donor_seed,
            "noise_stream_version": int(runner.noise_stream_version),
            "event_selector": e17.EVENT_SELECTOR,
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
            source_fps=FPS,
            half_window_frames=HALF_WINDOW_FRAMES,
            absolute_provenance=dict(base_provenance),
            residual_provenance={
                **base_provenance,
                "neutral_clip_sha256": selected_row["neutral_clip_sha256"],
            },
        )
        if pair.absolute.swing_side != EXPECTED_SWING_SIDE:
            raise ValueError("regenerated packet swing side is not the locked side")
        if int(pair.absolute.adapted_center_frame) != EXPECTED_ADAPTED_CENTER_FRAME:
            raise ValueError(
                "regenerated packet center frame differs from the archived bundle"
            )
        packet_arrays = e17._packet_arrays(pair)
        payload_content = e17._array_hash(packet_arrays)
        if payload_content != EXPECTED_PACKET_PAYLOAD_CONTENT_SHA256:
            raise ValueError(
                "regenerated packet payload differs from the archived bundle while "
                "clip hashes match (measurement-code drift refusal): "
                f"{payload_content}"
            )
        np.savez(output / "packet_pair.npz", **packet_arrays)
        packet_record = {
            **pair.metadata(),
            "absolute": pair.absolute.metadata(),
            "residual": pair.residual.metadata(),
            "adapted_cycle": adapted_cycle.as_dict(),
            "neutral_cycle": neutral_cycle.as_dict(),
            "source_phase_match": source_phase_match.as_dict(),
            "payload_content_sha256": payload_content,
            "archived_payload_content_sha256": (
                archived["archived_payload_content_sha256"]),
            "payload_matches_archive": True,
            "npz_sha256": cal._sha256(output / "packet_pair.npz"),
        }
        cal._write_json(output / "packet_pair.json", packet_record)
        receipt["packet_pair"] = {
            "pair_hash": pair.digest(),
            "payload_content_sha256": payload_content,
            "swing_side": pair.absolute.swing_side,
            "selected_donor_seed": selected_donor_seed,
            "common_physical_protocol_hash": (
                pair.absolute.common_physical_protocol_hash),
        }
        persist()

        # ---- nominal pool: K seeds x three calibrated strata -----------------------
        stage = "pool_generation"
        clips: list[Any] = []
        for batch in pool_batch_plan():
            spec = cal.root_only_walk_spec(batch["requested_speed_mps"])
            returned = generate(
                [WALK] * POOL_BATCH_SIZE,
                [spec] * POOL_BATCH_SIZE,
                batch["seeds"],
                launched_counter="pool_samples_launched",
                returned_counter="pool_samples_returned",
            )
            for seed, sample in zip(batch["seeds"], returned):
                qpos = np.asarray(runner.to_qpos(sample), dtype=float)
                key = f"pilot__{batch['speed_label']}__seed{seed}"
                qpos_archives["pool_qpos"][key] = np.array(qpos, copy=True)
                clip = cal.analyze_generated_clip(
                    body=body,
                    qpos=qpos,
                    sample=sample,
                    split="pilot",
                    seed=seed,
                    speed_label=batch["speed_label"],
                    requested_speed_mps=batch["requested_speed_mps"],
                    thresholds=thresholds,
                    qpos_archive_key=key,
                )
                clips.append(clip)
                pool_rows.append({"batch_index": batch["index"], **clip.as_dict()})
            persist()

        stage = "pool_selection"
        receipt["provenance"]["post_pool_identity_revalidation"] = (
            revalidate_identities()
        )
        candidates, selected = cal.build_and_select_programs(
            clips,
            target_min_prominence_m=pmin,
            timing_bounds=frozen_bounds,
            pool_speed_strata=True,
        )
        left_by_seed = {
            item.cycle.seed: item
            for item in selected
            if item.cycle.swing_side == EXPECTED_SWING_SIDE
        }
        eligible_seeds = sorted(left_by_seed)
        selected_seeds = eligible_seeds[:N_SELECT]
        selection_record = {
            "frozen_pmin_m": pmin,
            "frozen_route_timing_bounds": frozen_bounds.as_dict(),
            "selection": (
                "v3 frozen key, strata pooled per seed/side, restricted to the "
                f"packet swing side ({EXPECTED_SWING_SIDE}); first "
                f"{N_SELECT} eligible seeds in predeclared order"
            ),
            "candidate_rows": [item.compact_dict() for item in candidates],
            "selected_programs_all_sides": [item.as_dict() for item in selected],
            "eligible_seeds": eligible_seeds,
            "eligibility": f"{len(eligible_seeds)}/{len(POOL_SEEDS)}",
            "selected_seeds": selected_seeds,
            "per_seed_selection": {
                str(seed): {
                    "placement_m": left_by_seed[seed].placement_m,
                    "speed_label": left_by_seed[seed].cycle.speed_label,
                    "apex_frame": left_by_seed[seed].cycle.apex_frame,
                    "prominence_m": left_by_seed[seed].cycle.prominence_m,
                    "program_digest": left_by_seed[seed].program.digest(),
                    "cycle_evidence_digest": left_by_seed[seed].cycle.evidence_digest,
                }
                for seed in selected_seeds
            },
        }
        cal._write_json(output / "pool_selection.json", selection_record)
        receipt["pool_selection"] = {
            key: selection_record[key]
            for key in (
                "frozen_pmin_m", "eligible_seeds", "eligibility",
                "selected_seeds", "per_seed_selection",
            )
        }
        persist()
        if len(eligible_seeds) < N_SELECT:
            raise ValueError(
                f"pool has {len(eligible_seeds)} eligible {EXPECTED_SWING_SIDE}-side "
                f"seeds; requires N={N_SELECT} from frozen K={len(POOL_SEEDS)}"
            )

        # ---- stage A: warped nominals and placement persistence --------------------
        stage = "warped_generation"
        warped_routes: dict[int, np.ndarray] = {}
        for seed in selected_seeds:
            warped_routes[seed] = warped_route_xz(left_by_seed[seed].program)
        warped_specs = [
            ConstraintSpec(
                root_xz=warped_routes[seed], heading=None, root_y=None,
                first_heading=0.0,
            )
            for seed in selected_seeds
        ]
        warped_returned = generate(
            [WALK] * len(selected_seeds),
            warped_specs,
            selected_seeds,
            launched_counter="warped_samples_launched",
            returned_counter="warped_samples_returned",
        )

        stage = "persistence_classification"
        warped_by_seed: dict[int, tuple[Mapping[str, Any], np.ndarray]] = {}
        assignment_by_seed: dict[int, dict[str, Any]] = {}
        for seed, sample in zip(selected_seeds, warped_returned):
            qpos = np.array(runner.to_qpos(sample), copy=True)
            qpos_archives["warped_qpos"][f"s{seed}"] = qpos
            rotations = np.asarray(sample.get("global_rot_mats"))
            smooth_root = np.asarray(sample.get("smooth_root_pos"))
            if (
                not np.issubdtype(rotations.dtype, np.number)
                or rotations.shape != (N_FRAMES, len(runner.joint_names), 3, 3)
                or not np.isfinite(rotations).all()
                or not np.issubdtype(smooth_root.dtype, np.number)
                or smooth_root.shape != (N_FRAMES, 3)
                or not np.isfinite(smooth_root).all()
            ):
                raise ValueError(f"warped seed {seed} has an invalid substrate")
            qpos_archives["warped_substrate"][f"s{seed}__global_rot_mats"] = np.array(
                rotations, copy=True)
            qpos_archives["warped_substrate"][f"s{seed}__smooth_root_pos"] = np.array(
                smooth_root, copy=True)
            selection = left_by_seed[seed]
            placement = float(selection.placement_m)
            row: dict[str, Any] = {
                "seed": seed,
                "prompt": WALK,
                "clip_sha256": e17._sample_hash(sample),
                "qpos_content_sha256": e17._array_hash({f"s{seed}": qpos}),
                "placement_m": placement,
                "selected_stratum": selection.cycle.speed_label,
                "selected_linear_apex_frame": selection.cycle.apex_frame,
                "program_digest": selection.program.digest(),
                "persisting": False,
                "attrition_stage": None,
                "attrition_reason": None,
            }
            warped_rows.append(row)
            try:
                ratio = e17._prescribed_progress_ratio(qpos, warped_routes[seed])
                row["final_progress_ratio_vs_warped_route"] = ratio
                if ratio < MIN_WARPED_PROGRESS_RATIO:
                    raise ValueError(
                        f"warped progress ratio {ratio:.4f} below locked "
                        f"{MIN_WARPED_PROGRESS_RATIO:.2f}"
                    )
                cycles = e17._phase_cycles(
                    body, qpos, FPS, thresholds,
                    swing_side=EXPECTED_SWING_SIDE,
                    support_window_s=SUPPORT_WINDOW_S,
                    min_stance_support_fraction=min_stance_support_fraction,
                    min_relative_lift_m=TARGET_MIN_RELATIVE_LIFT_M,
                )
                row["cycles"] = [cycle.as_dict() for cycle in cycles]
                feet = foot_kinematics_series(body, qpos, FPS)
                assignments, diagnostics = e17._target_assignment(
                    cycles, qpos, feet, warped_routes[seed], [placement],
                    pair.absolute,
                    source_common_physical_protocol_hash=(
                        pair.absolute.common_physical_protocol_hash),
                    half_window_frames=HALF_WINDOW_FRAMES,
                    max_center_shift_frames=MAX_CENTER_SHIFT_FRAMES,
                )
                row["assignment_diagnostics"] = diagnostics
                assignment = assignments[0]
                row.update({
                    "persisting": True,
                    "target_apex_frame": int(assignment["cycle"].apex_frame),
                    "center_shift_frames": int(
                        assignment["controls"].center_shift_frames),
                    "predicted_spatial_error_m": float(
                        assignment["predicted_spatial_error_m"]),
                })
                warped_by_seed[seed] = (sample, qpos)
                assignment_by_seed[seed] = assignment
            except e17.TargetAssignmentError as exc:
                row.update({
                    "attrition_stage": "target_assignment",
                    "attrition_reason": str(exc),
                    "assignment_diagnostics": exc.diagnostics,
                })
            except ValueError as exc:
                row.update({
                    "attrition_stage": "progress_or_phase",
                    "attrition_reason": str(exc),
                })
            persist()
        persisting_seeds = sorted(assignment_by_seed)
        receipt["stage_a_persistence"] = {
            "primary_endpoint": (
                "per-seed placement persistence of the calibrated route warp under "
                "the unchanged exp017 +/-8-frame assignment gate"
            ),
            "n_selected": len(selected_seeds),
            "n_persisting": len(persisting_seeds),
            "persisting_seeds": persisting_seeds,
            "persistence_rate": (
                len(persisting_seeds) / len(selected_seeds)
                if selected_seeds else None
            ),
            "per_seed": {
                str(row["seed"]): {
                    "persisting": row["persisting"],
                    "attrition_stage": row["attrition_stage"],
                    "center_shift_frames": row.get("center_shift_frames"),
                    "predicted_spatial_error_m": row.get(
                        "predicted_spatial_error_m"),
                    "progress_ratio": row.get(
                        "final_progress_ratio_vs_warped_route"),
                }
                for row in warped_rows
            },
        }
        persist()
        if not persisting_seeds:
            raise ValueError(
                "stage A found zero persisting seeds: the calibrated route warp did "
                "not place any complete swing within the locked assignment gate"
            )

        # ---- stage B: paired packet arms on the warped routes ----------------------
        stage = "program_render"
        rendered: dict[tuple[int, str], ConstraintSpec] = {}
        for seed in persisting_seeds:
            sample, _ = warped_by_seed[seed]
            assignment = assignment_by_seed[seed]
            placement = float(left_by_seed[seed].placement_m)
            common = dict(
                target_nominal=sample,
                target_event=assignment["cycle"].event,
                joint_names=runner.joint_names,
                root_xz=warped_routes[seed],
                route_heading=route_heading,
                target_phase_match=assignment["target_phase_match"],
                nominal_route_heading=route_heading,
                target_fps=FPS,
                controls=assignment["controls"],
                first_heading=0.0,
            )
            absolute_spec, absolute_info = render_packet(pair.absolute, **common)
            residual_spec, residual_info = render_packet(pair.residual, **common)
            absolute_usage = e17._actual_channel_usage(runner, absolute_spec)
            residual_usage = e17._actual_channel_usage(runner, residual_spec)
            e17._assert_matched_programs(
                absolute_spec, residual_spec, absolute_info, residual_info,
                absolute_usage, residual_usage,
            )
            scene_id = e17._scene_id(placement, OBSTACLE_HEIGHT_M, OBSTACLE_DEPTH_M)
            for arm, spec, info, usage in (
                ("absolute", absolute_spec, absolute_info, absolute_usage),
                ("residual", residual_spec, residual_info, residual_usage),
            ):
                rendered[(seed, arm)] = spec
                program_rows.append({
                    "seed": seed,
                    "arm": arm,
                    "prompt": STEP,
                    "scene_id": scene_id,
                    "placement_m": placement,
                    "selected_stratum": left_by_seed[seed].cycle.speed_label,
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

        stage = "arm_generation"
        arm_plan = [
            (seed, arm) for seed in persisting_seeds for arm in ARMS
        ]
        arm_returned = generate(
            [STEP] * len(arm_plan),
            [rendered[key] for key in arm_plan],
            [seed for seed, _ in arm_plan],
            launched_counter="arm_samples_launched",
            returned_counter="arm_samples_returned",
        )

        stage = "arm_scoring"
        program_by_key = {
            (row["seed"], row["arm"]): row for row in program_rows
        }
        for (seed, arm), sample in zip(arm_plan, arm_returned):
            program = program_by_key[(seed, arm)]
            placement = float(program["placement_m"])
            row: dict[str, Any] = {
                "seed": seed,
                "arm": arm,
                "prompt": STEP,
                "scene_id": program["scene_id"],
                "placement_m": placement,
                "status": "returned",
                "sample_sha256": e17._sample_hash(sample),
            }
            arm_rows.append(row)
            try:
                qpos = np.array(runner.to_qpos(sample), copy=True)
                key = f"s{seed}__{arm}"
                qpos_archives["arm_qpos"][key] = np.asarray(qpos, dtype=np.float32)
                row["qpos_content_sha256"] = e17._array_hash(
                    {key: qpos_archives["arm_qpos"][key]})
                metrics = e17._score(
                    body, BoxHeightProbe(placement, OBSTACLE_DEPTH_M), qpos,
                    warped_routes[seed], placement, EXPECTED_SWING_SIDE, FPS,
                    thresholds, OBSTACLE_HEIGHT_M,
                )
                row.update({
                    **metrics,
                    "program_hash": program["program_hash"],
                    "support_hash": program["support_hash"],
                    "packet_hash": program["packet_hash"],
                    "program_deformation": program["deformation_vs_nominal"],
                    "status": "completed",
                })
            except Exception as exc:  # noqa: BLE001 - preserved in the denominator
                row.update({
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            persist()

        stage = "summary"
        receipt["provenance"]["post_arm_identity_revalidation"] = (
            revalidate_identities()
        )
        completed = [row for row in arm_rows if row["status"] == "completed"]
        complete_pairs = sorted(
            seed for seed in persisting_seeds
            if sum(1 for row in completed if row["seed"] == seed) == 2
        )
        summary: dict[str, Any] = {
            "planned_arm_denominator_per_arm": len(persisting_seeds),
            "completed_arm_counts": {
                arm: sum(1 for row in completed if row["arm"] == arm)
                for arm in ARMS
            },
            "seeds_with_complete_pairs": complete_pairs,
        }
        if complete_pairs:
            paired_rows = [
                row for row in completed if row["seed"] in complete_pairs
            ]
            summary["paired_descriptive"] = e17._paired_summary(paired_rows)
            summary["per_seed_deltas_residual_minus_absolute"] = {
                str(seed): {
                    metric: (
                        float(next(
                            row[metric] for row in paired_rows
                            if row["seed"] == seed and row["arm"] == "residual"))
                        - float(next(
                            row[metric] for row in paired_rows
                            if row["seed"] == seed and row["arm"] == "absolute"))
                    )
                    for metric in (
                        "max_box_height_lower_bound_m",
                        "progress_ratio",
                    )
                }
                for seed in complete_pairs
            }
        receipt["stage_b_summary"] = summary

        stage = "complete"
        total = (
            counters["donor_samples_returned"]
            + counters["pool_samples_returned"]
            + counters["warped_samples_returned"]
            + counters["arm_samples_returned"]
        )
        receipt.update({
            "status": "complete",
            "complete": True,
            "blocked": False,
            "stage": stage,
            "actual_ardy_samples": total,
            "conservative_charged_ardy_samples": total,
            "budget_check": {
                "equation": "2D + 3K + N + 2M",
                "expected": (
                    2 * len(DONOR_SEEDS) + 3 * len(POOL_SEEDS)
                    + len(selected_seeds) + 2 * len(persisting_seeds)
                ),
                "actual": total,
            },
        })
        if receipt["budget_check"]["expected"] != total:
            raise RuntimeError("final sample accounting does not match the design")
        persist()
        return receipt
    except Exception as exc:
        launched = sum(
            counters[name] for name in counters if name.endswith("_launched")
        )
        returned = sum(
            counters[name] for name in counters if name.endswith("_returned")
        )
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION,
            "status": "blocked",
            "complete": False,
            "blocked": True,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
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
    parser.add_argument("--out", default="outputs/exp018_route_warped_stepover")
    parser.add_argument(
        "--threshold-receipt",
        default="outputs/exp016_threshold_calibration/receipt.json",
    )
    parser.add_argument("--v3-receipt", default=V3_RECEIPT_PATH)
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    receipt = run_pilot(
        out=args.out,
        threshold_receipt=args.threshold_receipt,
        v3_receipt=args.v3_receipt,
        cache_path=args.cache_path,
    )
    print(json.dumps({
        "status": receipt["status"],
        "actual_ardy_samples": receipt.get("actual_ardy_samples"),
        "persistence": receipt.get("stage_a_persistence", {}).get(
            "persistence_rate"),
    }, indent=2))


if __name__ == "__main__":
    main()
