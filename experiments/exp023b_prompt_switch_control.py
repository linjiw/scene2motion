"""EXP-023b: WALK->SQUEEZE prompt-switch positive control under the EXP-023 handoff.

EXP-023 found that a WALK->STEP prompt switch at frame 52 or 104, delivered through ARDY's
released ``autoregressive_step`` interface at its GUI-default minimum history (one four-frame
token), elicited no whole-body-clearable step in 0/8 seeds against 6/8 from frame 0.  That
result cannot distinguish "the prompt is only read while the rollout context is being
established" from "a four-frame history handoff attenuates every prompt".  This driver runs
the identical handoff with the cached SQUEEZE prompt on fresh seeds: if *some* prompt switches
under the same contract, the STEP timing result is behaviour-specific; if none does, the
abstract is scoped to the handoff as driven.

The generation contract, chunk-audit merge, step-event detector, fixed-box scorer, record
validators and pin checks are imported unchanged from
:mod:`experiments.exp023_prompt_handoff`, whose source hash is bound into the receipt.  New
here: the four-arm plan on seeds 4640-4647, the preregistered three-signature sidestep
detector, a prompt-agnostic paired handoff-transmission statistic, the shared host-resource
gate, and the protocol's decision rule evaluated mechanically into the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import calibrate_ramp_route_phase as cal  # noqa: E402
from experiments import exp023_prompt_handoff as e23  # noqa: E402
from experiments.analyze_exp021_exact_addressability import wilson_interval  # noqa: E402
from scene2motion import host_gate as hg  # noqa: E402
from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import foot_kinematics_series  # noqa: E402


SCHEMA_VERSION = "exp023b-prompt-switch-control-v1"
FAILURE_SCHEMA_VERSION = "exp023b-prompt-switch-control-failure-v1"

WALK = e23.WALK
STEP = e23.STEP
SQUEEZE = "A person steps sideways through a narrow gap."
PROMPTS = (WALK, STEP, SQUEEZE)

FPS = e23.FPS
N_FRAMES = e23.N_FRAMES
HORIZON = e23.HORIZON
N_WINDOWS = e23.N_WINDOWS
PADDED_FRAMES = e23.PADDED_FRAMES
POST_ONSET_FRAMES = e23.POST_ONSET_FRAMES
FROZEN_LATENCY_FRAMES = e23.FROZEN_LATENCY_FRAMES
DIFFUSION_STEPS = e23.DIFFUSION_STEPS
CFG_WEIGHT = e23.CFG_WEIGHT
NOISE_STREAM_VERSION = e23.NOISE_STREAM_VERSION
CHUNK_SEED_COUNT = e23.CHUNK_SEED_COUNT
GRADED_HEIGHTS_M = e23.GRADED_HEIGHTS_M

SEEDS = tuple(range(4640, 4648))
ARMS = ("all_walk", "squeeze_0", "squeeze_52", "step_52")
PROMPT_ARMS = ("squeeze_0", "squeeze_52", "step_52")
REFERENCE_ARM = "all_walk"
CHUNK_ROWS = CHUNK_SEED_COUNT * len(ARMS)
SCHEDULES: Mapping[str, tuple[str, ...]] = {
    "all_walk": (WALK, WALK, WALK, WALK),
    "squeeze_0": (SQUEEZE, SQUEEZE, SQUEEZE, SQUEEZE),
    "squeeze_52": (WALK, SQUEEZE, SQUEEZE, SQUEEZE),
    "step_52": (WALK, STEP, STEP, STEP),
}
ONSETS: Mapping[str, int | None] = {
    "all_walk": None,
    "squeeze_0": 0,
    "squeeze_52": 52,
    "step_52": 52,
}
SWITCH_FRAME = 52
# Delayed arms whose accepted transcript must be byte-identical to all_walk before the fork.
FORK_FRAMES: Mapping[str, int] = {"squeeze_52": SWITCH_FRAME, "step_52": SWITCH_FRAME}
CONTROL_ONSETS = (0, SWITCH_FRAME)

# Sidestep detector (preregistered; composite = any signature).
HEADING_SIDLE_MIN_DEVIATION_DEG = 45.0
LATERAL_EXCURSION_MIN_M = 0.15
SUSTAINED_MIN_FRAMES = 13  # smallest integer frame count >= 0.5 s at 25 fps
FOOT_CROSSING_MIN_M = 0.10
FOOT_CROSSING_MIN_FRAMES = 3
SIDESTEP_SIGNATURES = ("heading_sidle", "lateral_excursion", "foot_crossing")
COMPOSITE_RULE = "any"

# Gates and decision rule (protocol: docs/ramp-exp023b-prompt-switch-positive-control-protocol.md).
MIN_SQUEEZE0_SIDESTEPS = 4
MAX_ALL_WALK_SEEDS_WITH_ANY_SIDESTEP = 1
MAX_ALL_WALK_SEEDS_WITH_ANY_STEP_EVENT = 1
DECISION_MIN_SQUEEZE52_SIDESTEPS = 4
DECISION_MAX_STEP52_EVENTS = 1
DECISION_MAX_SQUEEZE52_SIDESTEPS_NO_SWITCH = 1
VERDICT_TRANSMITS = "handoff_transmits_prompts_timing_claim_behaviour_specific"
VERDICT_NO_SWITCH = "no_prompt_switched_under_minimum_history_handoff"
VERDICT_INDETERMINATE = "indeterminate_at_n8"
VERDICT_STEP_REPLICATION_FAILED = "step_replication_failed_exp023_zero_not_replicated"
DECISION_STEP52_REPLICATION_FAIL_MIN = 2  # step_52 >= 2/8 events: EXP-023's 0/8 did not replicate

HOST_GATE = dict(hg.ARDY_GENERATION_GATE)  # ARDY-only campaign; Isaac co-tenancy is recorded, not gated.
HOST_GATE_REQUIRE_NO_ISAAC = bool(HOST_GATE["require_no_isaac"])

PROTOCOL_PATH = "docs/ramp-exp023b-prompt-switch-positive-control-protocol.md"
SOURCE_FILES = (
    PROTOCOL_PATH,
    "experiments/exp023b_prompt_switch_control.py",
    "experiments/exp023_prompt_handoff.py",
    "experiments/calibrate_ramp_route_phase.py",
    "experiments/exp017_ramp_residual_stepover.py",
    "experiments/analyze_exp021_exact_addressability.py",
    "scene2motion/host_gate.py",
    "scene2motion/runner.py",
    "scene2motion/constraints.py",
    "scene2motion/robot.py",
    "scene2motion/stepover_eval.py",
)

# Reused EXP-023 machinery (arm-agnostic).
route_xz = e23.route_xz
root_only_spec = e23.root_only_spec
_array_sha256 = e23._array_sha256
_raw_row_sha256 = e23._raw_row_sha256
_json_safe = e23._json_safe
_validate_pins = e23._validate_pins
_actual_channel_usage = e23._actual_channel_usage
_merge_chunk_audits = e23._merge_chunk_audits
_serialized_noise_evidence = e23._serialized_noise_evidence
detect_prompt_event = e23.detect_prompt_event
score_exact_boxes = e23.score_exact_boxes
supporting_motion_metrics = e23.supporting_motion_metrics
_validated_event_record = e23._validated_event_record
_validated_exact_box_record = e23._validated_exact_box_record
_validated_motion_metrics = e23._validated_motion_metrics


class PromptSwitchAbort(RuntimeError):
    """Fail-closed stop after durable campaign evidence has been written."""


class MeasurementRefusal(RuntimeError):
    """A preregistered substrate/specificity gate refused the measurement."""


class HostGateRefusal(RuntimeError):
    """The host-resource gate failed before anything was written; carries the report."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


# --- locked plan -----------------------------------------------------------------------


def locked_row_plan() -> list[dict[str, Any]]:
    """Seed-major 32-row plan; its order binds feature and latent-audit rows."""
    pairs = [(seed, arm) for seed in SEEDS for arm in ARMS]
    return [
        {
            "row_index": index,
            "seed": int(seed),
            "arm": arm,
            "prompt_schedule": list(SCHEDULES[arm]),
            "onset_frame": ONSETS[arm],
        }
        for index, (seed, arm) in enumerate(pairs)
    ]


def locked_chunk_plan(plan: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Four seed-paired B=8 chunks; no same-seed causal comparison crosses a call."""
    rows = list(plan if plan is not None else locked_row_plan())
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(SEEDS), CHUNK_SEED_COUNT)):
        seeds = tuple(SEEDS[start:start + CHUNK_SEED_COUNT])
        chunk_rows = [dict(row) for row in rows if int(row["seed"]) in seeds]
        if (
            len(seeds) != CHUNK_SEED_COUNT
            or len(chunk_rows) != CHUNK_ROWS
            or {str(row["arm"]) for row in chunk_rows} != set(ARMS)
            or {int(row["seed"]) for row in chunk_rows} != set(seeds)
        ):
            raise ValueError("locked EXP-023b chunk plan no longer forms paired B=8 calls")
        chunks.append({
            "chunk_index": index,
            "name": f"chunk{index:02d}_seeds{seeds[0]}_{seeds[-1]}",
            "seeds": list(seeds),
            "row_indices": [int(row["row_index"]) for row in chunk_rows],
            "rows": chunk_rows,
        })
    if len(chunks) != len(SEEDS) // CHUNK_SEED_COUNT or sorted(
        row_index for chunk in chunks for row_index in chunk["row_indices"]
    ) != list(range(len(rows))):
        raise ValueError("locked EXP-023b chunks do not partition the 32-row plan")
    return chunks


# --- provenance ------------------------------------------------------------------------


def _source_hashes(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        digest = cal._sha256(repo / relative)
        if digest is None:
            raise ValueError(f"required EXP-023b source is missing: {relative}")
        hashes[relative] = digest
    return hashes


def _prompt_cache_identity(runner: Any, cache_path: Path) -> dict[str, Any]:
    """Bind all three cached prompt arrays (WALK, STEP, SQUEEZE) in memory and on disk."""
    if not cache_path.is_file():
        raise ValueError("EXP-023b prompt cache is missing")
    prompts: dict[str, Any] = {}
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            for prompt in PROMPTS:
                key = hashlib.sha1(prompt.encode()).hexdigest()
                if key not in cache.files:
                    raise ValueError(f"cached embedding is missing for {prompt!r}")
                file_value = np.array(cache[key], copy=True)
                memory_value = np.array(runner._text_cache[key], copy=True)
                if (
                    file_value.size == 0
                    or not np.isfinite(file_value).all()
                    or not np.array_equal(file_value, memory_value)
                ):
                    raise ValueError(
                        f"in-memory embedding does not byte-match cache for {prompt!r}"
                    )
                prompts[prompt] = {
                    "cache_key_sha1": key,
                    "content_sha256": _array_sha256(file_value, key),
                    "shape": list(file_value.shape),
                    "dtype": str(file_value.dtype),
                }
    except (AttributeError, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid EXP-023b prompt cache: {exc}") from exc
    fields = {
        "path": str(cache_path),
        "file_sha256": cal._sha256(cache_path),
        "prompts": prompts,
    }
    return cal._identity("exp023b-walk-step-squeeze-prompt-cache-v1", fields)


def _once(fn: Callable[[], Any]) -> Callable[[], Any]:
    cache: list[Any] = []

    def call() -> Any:
        if not cache:
            cache.append(fn())
        return cache[0]

    return call


def evaluate_host_gate(
    *,
    vram_fn: Callable[[], Mapping[str, Any]] = hg.query_free_vram_mib,
    ram_fn: Callable[[], Mapping[str, Any]] = hg.query_available_ram_mib,
    isaac_fn: Callable[[], Sequence[Mapping[str, Any]]] = hg.concurrent_isaac_processes,
) -> dict[str, Any]:
    """Preregistered ARDY-only host gate (``scene2motion.host_gate.ARDY_GENERATION_GATE``):
    >= 4 GiB free VRAM and >= 8 GiB available RAM, a ~4x margin on the measured B=8 need
    (1,076 MiB CUDA reserved, 2,297 MiB host RSS on 2026-09-02).

    Measures each probe once, evaluates through the shared
    :func:`scene2motion.host_gate.require_host_resources`, and either returns the report to
    bind into the receipt or raises :class:`HostGateRefusal` carrying the same measurements.
    Concurrent Isaac processes are recorded for the record but not gated (ARDY-only campaign).
    """
    measured = {"vram_fn": _once(vram_fn), "ram_fn": _once(ram_fn), "isaac_fn": _once(isaac_fn)}
    try:
        report = dict(hg.require_host_resources(**HOST_GATE, **measured))
    except hg.HostResourceGateFailed as exc:
        report = hg.host_resource_report(**HOST_GATE, **measured)
        report["concurrent_isaac_processes_informational"] = [
            dict(item) for item in measured["isaac_fn"]()
        ]
        raise HostGateRefusal(str(exc), report) from exc
    report["concurrent_isaac_processes_informational"] = [
        dict(item) for item in measured["isaac_fn"]()
    ]
    return report


# --- sidestep detector -----------------------------------------------------------------


def yaw_from_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    """Yaw about world +Z of MuJoCo (w, x, y, z) root quaternions; 0 rad faces +X.

    Matches ``_heading`` in ``experiments/analyze_trackability_contract.py``.  On this route
    +X is the travel (forward) axis and +Y the lateral axis (``qpos[:, 0]`` runs 0 -> 7.2 m,
    ``qpos[:, 1]`` stays within about 0.02 m in WALK clips).
    """
    q = np.asarray(quat_wxyz, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternions must be (T, 4) in (w, x, y, z) order")
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def route_travel_heading(route: np.ndarray) -> np.ndarray:
    """Per-frame travel direction of the (lateral, forward) route in the MuJoCo yaw convention."""
    exact = np.asarray(route, dtype=float)
    if exact.ndim != 2 or exact.shape[1] != 2 or len(exact) < 2 or not np.isfinite(exact).all():
        raise ValueError("route must be a finite (T, 2) array with at least two frames")
    lateral = np.gradient(exact[:, 0])
    forward = np.gradient(exact[:, 1])
    if np.any(np.hypot(lateral, forward) <= 0.0):
        raise ValueError("route travel direction is undefined at a stationary frame")
    return np.arctan2(lateral, forward)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _sustained_signature(mask: np.ndarray, min_run_frames: int, onset: int) -> dict[str, Any]:
    runs = _runs(mask)
    qualifying = [(start, end) for start, end in runs if end - start >= min_run_frames]
    return {
        "present": bool(qualifying),
        "min_run_frames": int(min_run_frames),
        "longest_run_frames": int(max((end - start for start, end in runs), default=0)),
        "qualifying_run_count": len(qualifying),
        "first_frame": int(onset + qualifying[0][0]) if qualifying else None,
        "first_latency_frames": int(qualifying[0][0]) if qualifying else None,
        "flagged_frame_count": int(np.count_nonzero(mask)),
    }


def detect_sidestep(
    body: Any,
    qpos: np.ndarray,
    route: np.ndarray,
    onset_frame: int,
    *,
    foot_series_fn: Callable[[Any, np.ndarray, float], Mapping[str, Any]] = (
        foot_kinematics_series
    ),
) -> dict[str, Any]:
    """Preregistered three-signature sidestep composite in the 96 frames after onset.

    (i) ``heading_sidle``: the root yaw deviates from the route travel direction by
        >= 45 deg for >= 13 consecutive frames (>= 0.5 s at 25 fps).
    (ii) ``lateral_excursion``: |root Y - route lateral offset at onset| >= 0.15 m for
        >= 13 consecutive frames.
    (iii) ``foot_crossing``: in the pelvis-yaw-aligned frame (identical to the route's lateral
        axis whenever the heading tracks the route), the left foot's lateral representative
        lies >= 0.10 m to the right of the right foot's for >= 3 consecutive frames.  The
        nominal G1 stance facing +X has the left foot at +Y (separation about +0.12 m in WALK).
    The composite is ``any``; every signature is reported separately.
    """
    exact_qpos = np.asarray(qpos, dtype=float)
    exact_route = np.asarray(route, dtype=float)
    start = int(onset_frame)
    end = start + POST_ONSET_FRAMES
    if (
        exact_qpos.ndim != 2
        or exact_qpos.shape[1] < 7
        or exact_route.shape != (N_FRAMES, 2)
        or start not in CONTROL_ONSETS
        or end > N_FRAMES
        or len(exact_qpos) < N_FRAMES
        or not np.isfinite(exact_qpos[:N_FRAMES]).all()
    ):
        raise ValueError("sidestep detector received an invalid clip, route, or onset")
    segment = exact_qpos[start:end]

    yaw = yaw_from_quaternion(segment[:, 3:7])
    travel = route_travel_heading(exact_route)[start:end]
    deviation = _wrap_angle(yaw - travel)
    heading = {
        **_sustained_signature(
            np.abs(deviation) >= np.radians(HEADING_SIDLE_MIN_DEVIATION_DEG),
            SUSTAINED_MIN_FRAMES, start),
        "threshold_deg": HEADING_SIDLE_MIN_DEVIATION_DEG,
        "max_abs_deviation_deg": float(np.degrees(np.max(np.abs(deviation)))),
        "deviation_at_onset_deg": float(np.degrees(deviation[0])),
        "rule": "|root yaw - route travel direction| >= threshold for >= min_run_frames",
    }

    reference_lateral = float(exact_route[start, 0])
    excursion = segment[:, 1] - reference_lateral
    peak = int(np.argmax(np.abs(excursion)))
    lateral = {
        **_sustained_signature(
            np.abs(excursion) >= LATERAL_EXCURSION_MIN_M, SUSTAINED_MIN_FRAMES, start),
        "threshold_m": LATERAL_EXCURSION_MIN_M,
        "route_lateral_reference_m": reference_lateral,
        "root_lateral_at_onset_m": float(segment[0, 1]),
        "max_abs_excursion_m": float(np.abs(excursion[peak])),
        "signed_excursion_at_peak_m": float(excursion[peak]),
        "rule": "|root Y - route lateral offset at onset| >= threshold for >= min_run_frames",
    }

    feet = foot_series_fn(body, segment, FPS)
    positions: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        if side not in feet:
            raise ValueError(f"sidestep foot series lacks {side}")
        names = ("forward_representative_m", "lateral_representative_m")
        arrays = [np.asarray(feet[side][name], dtype=float) for name in names]
        if any(
            array.shape != (POST_ONSET_FRAMES,) or not np.isfinite(array).all()
            for array in arrays
        ):
            raise ValueError(f"sidestep foot series is invalid for {side}")
        positions[side] = np.stack(arrays, axis=-1)
    body_left_axis = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
    body_lateral = {
        side: np.einsum("tk,tk->t", positions[side], body_left_axis) for side in positions
    }
    separation = body_lateral["left"] - body_lateral["right"]
    world_separation = positions["left"][:, 1] - positions["right"][:, 1]
    crossing = {
        **_sustained_signature(
            separation <= -FOOT_CROSSING_MIN_M, FOOT_CROSSING_MIN_FRAMES, start),
        "threshold_m": FOOT_CROSSING_MIN_M,
        "frame": "pelvis-yaw-aligned lateral axis (left positive)",
        "nominal_ordering": "left foot at +Y when facing +X; separation > 0",
        "separation_at_onset_m": float(separation[0]),
        "min_separation_m": float(np.min(separation)),
        "min_world_lateral_separation_m": float(np.min(world_separation)),
        "rule": "left minus right body-lateral representative <= -threshold for >= min_run_frames",
    }

    signatures = {
        "heading_sidle": heading,
        "lateral_excursion": lateral,
        "foot_crossing": crossing,
    }
    present_names = [name for name in SIDESTEP_SIGNATURES if signatures[name]["present"]]
    first_frames = [int(signatures[name]["first_frame"]) for name in present_names]
    first_frame = min(first_frames) if first_frames else None
    return {
        "present": bool(present_names),
        "composite_rule": COMPOSITE_RULE,
        "present_signatures": present_names,
        "first_signature_frame": first_frame,
        "first_signature_latency_frames": (
            None if first_frame is None else int(first_frame - start)),
        "signatures": signatures,
        "analysis_window_start_frame": start,
        "analysis_window_end_frame": end - 1,
        "analysis_window_frames": POST_ONSET_FRAMES,
        "fps": FPS,
        "axis_convention": {
            "forward": "MuJoCo +X = qpos[:, 0] = route[:, 1]",
            "lateral": "MuJoCo +Y = qpos[:, 1] = route[:, 0]",
            "yaw": "about +Z from qpos[:, 3:7] (w, x, y, z); 0 faces +X",
        },
    }


def _validated_sidestep_record(value: Mapping[str, Any], onset: int) -> dict[str, Any]:
    """Validate even dependency-injected detector output before it reaches a gate."""
    record = dict(value)
    present = record.get("present")
    if not isinstance(present, (bool, np.bool_)):
        raise ValueError("sidestep record requires boolean present")
    if record.get("composite_rule") != COMPOSITE_RULE:
        raise ValueError("sidestep record does not use the locked composite rule")
    signatures = record.get("signatures")
    if not isinstance(signatures, Mapping) or set(signatures) != set(SIDESTEP_SIGNATURES):
        raise ValueError("sidestep record must report exactly the three locked signatures")
    locked = {
        "heading_sidle": (SUSTAINED_MIN_FRAMES, "threshold_deg", HEADING_SIDLE_MIN_DEVIATION_DEG),
        "lateral_excursion": (SUSTAINED_MIN_FRAMES, "threshold_m", LATERAL_EXCURSION_MIN_M),
        "foot_crossing": (FOOT_CROSSING_MIN_FRAMES, "threshold_m", FOOT_CROSSING_MIN_M),
    }
    present_names: list[str] = []
    first_frames: list[int] = []
    for name in SIDESTEP_SIGNATURES:
        signature = signatures[name]
        min_run, threshold_key, threshold = locked[name]
        if not isinstance(signature, Mapping):
            raise ValueError(f"sidestep signature {name} is not a mapping")
        flag = signature.get("present")
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(f"sidestep signature {name} requires boolean present")
        if int(signature.get("min_run_frames", -1)) != min_run:
            raise ValueError(f"sidestep signature {name} changed its locked run length")
        if not np.isclose(float(signature.get(threshold_key, np.nan)), threshold,
                          atol=0.0, rtol=0.0):
            raise ValueError(f"sidestep signature {name} changed its locked threshold")
        longest = signature.get("longest_run_frames")
        if not isinstance(longest, (int, np.integer)) or int(longest) < 0:
            raise ValueError(f"sidestep signature {name} has an invalid longest run")
        if flag:
            frame = signature.get("first_frame")
            if int(longest) < min_run:
                raise ValueError(f"present sidestep signature {name} lacks a sustained run")
            if not isinstance(frame, (int, np.integer)) or not (
                int(onset) <= int(frame) < int(onset) + POST_ONSET_FRAMES
            ):
                raise ValueError(
                    f"sidestep signature {name} lies outside the locked 96-frame window")
            if int(signature.get("first_latency_frames", -1)) != int(frame) - int(onset):
                raise ValueError(f"sidestep signature {name} frame and latency disagree")
            present_names.append(name)
            first_frames.append(int(frame))
        else:
            if int(longest) >= min_run or signature.get("first_frame") is not None:
                raise ValueError(f"absent sidestep signature {name} reports a sustained run")
    if bool(present) != bool(present_names):
        raise ValueError("sidestep composite disagrees with its signatures")
    if list(record.get("present_signatures", [])) != present_names:
        raise ValueError("sidestep present_signatures disagrees with its signatures")
    expected_first = min(first_frames) if first_frames else None
    if record.get("first_signature_frame") != expected_first:
        raise ValueError("sidestep first_signature_frame disagrees with its signatures")
    return _json_safe(record)


# --- paired handoff-transmission statistic ---------------------------------------------


TRANSMISSION_PAIRS: Mapping[str, tuple[str, str]] = {
    "squeeze_52_vs_all_walk": ("squeeze_52", "all_walk"),
    "step_52_vs_all_walk": ("step_52", "all_walk"),
    "squeeze_0_vs_all_walk": ("squeeze_0", "all_walk"),
    "step_52_vs_squeeze_52": ("step_52", "squeeze_52"),
}


def handoff_transmission(qpos_by_key: Mapping[str, np.ndarray], seed: int) -> dict[str, Any]:
    """Prompt-agnostic RMS joint-angle difference over frames 52..147 for one seed.

    ``qpos[:, 7:]`` are the joint coordinates; the root pose is excluded because the dense
    root-XZ constraint pins it in every arm.  Descriptive only.
    """
    window = slice(SWITCH_FRAME, SWITCH_FRAME + POST_ONSET_FRAMES)
    prefix = slice(0, SWITCH_FRAME)
    clips: dict[str, np.ndarray] = {}
    for arm in ARMS:
        clip = np.asarray(qpos_by_key[f"s{int(seed)}_{arm}"], dtype=float)
        if clip.ndim != 2 or clip.shape[0] < N_FRAMES or clip.shape[1] <= 7:
            raise ValueError(f"handoff transmission requires (>=200, >7) qpos for {arm}")
        if not np.isfinite(clip[:N_FRAMES]).all():
            raise ValueError(f"handoff transmission received non-finite qpos for {arm}")
        clips[arm] = clip[:N_FRAMES, 7:]
    widths = {clip.shape[1] for clip in clips.values()}
    if len(widths) != 1:
        raise ValueError("handoff transmission arms have different joint widths")

    def rms(first: str, second: str, frames: slice) -> float:
        difference = clips[first][frames] - clips[second][frames]
        return float(np.sqrt(np.mean(np.square(difference))))

    return {
        "seed": int(seed),
        "frames": [SWITCH_FRAME, SWITCH_FRAME + POST_ONSET_FRAMES - 1],
        "joint_dofs": int(widths.pop()),
        "joint_rms_rad": {
            name: rms(first, second, window)
            for name, (first, second) in TRANSMISSION_PAIRS.items()
        },
        "prefix_joint_rms_rad_frames_0_51": {
            f"{arm}_vs_all_walk": rms(arm, "all_walk", prefix) for arm in FORK_FRAMES
        },
    }


# --- causal audits ---------------------------------------------------------------------


def _checked_hashes(value: Any, expected: int, label: str) -> list[str]:
    hashes = list(value if isinstance(value, (list, tuple)) else [])
    if len(hashes) != expected or any(not cal._is_sha256(item) for item in hashes):
        raise ValueError(f"{label} contains invalid or missing row hashes")
    return hashes


def _check_window_audit(
    window: int, audit: Mapping[str, Any], exact: np.ndarray
) -> tuple[list[int], list[str], dict[str, Any]]:
    """EXP-023's per-window history-policy and transcript-hash checks (arm-agnostic)."""
    n_rows = int(exact.shape[0])
    expected = {
        "window_index": window,
        "global_history_start_frame": e23.EXPECTED_GLOBAL_HISTORY_START_FRAMES[window],
        "accepted_transcript_frames_before": (
            e23.EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE[window]),
        "input_history_frames": e23.EXPECTED_INPUT_HISTORY_FRAMES[window],
        "model_num_frames": e23.EXPECTED_MODEL_NUM_FRAMES[window],
        "transcript_frames": e23.EXPECTED_TRANSCRIPT_FRAMES[window],
    }
    for field, value in expected.items():
        if int(audit.get(field, -1)) != value:
            raise ValueError(f"history policy field {field} mismatch at window {window}")
    hashes = _checked_hashes(audit.get("row_sha256"), n_rows, f"latent audit window {window}")
    shape = list(audit.get("shape", []))
    if len(shape) != 3 or shape[0] != n_rows or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"latent audit row/shape mismatch at window {window}")
    stable = _checked_hashes(
        audit.get("stable_transcript_row_sha256"), n_rows,
        f"stable transcript audit window {window}")
    transcript_frames = expected["transcript_frames"]
    if stable != [_raw_row_sha256(exact[row, :transcript_frames]) for row in range(n_rows)]:
        raise ValueError(
            f"stable transcript hashes do not match archived features at window {window}")
    expected_history_rows = 0 if window == 0 else n_rows
    input_hashes = _checked_hashes(
        audit.get("input_history_row_sha256"), expected_history_rows,
        f"input history audit window {window}")
    _checked_hashes(
        audit.get("returned_input_history_row_sha256"), expected_history_rows,
        f"returned history audit window {window}")
    history_start = expected["global_history_start_frame"]
    accepted_before = expected["accepted_transcript_frames_before"]
    expected_inputs = [] if window == 0 else [
        _raw_row_sha256(exact[row, history_start:accepted_before]) for row in range(n_rows)
    ]
    if input_hashes != expected_inputs:
        raise ValueError(f"window {window} did not consume the locked visible transcript suffix")
    reconstruction_exact = list(audit.get("returned_history_reconstruction_exact", []))
    reconstruction_max_abs = list(audit.get("returned_history_reconstruction_max_abs", []))
    if (
        len(reconstruction_exact) != n_rows
        or any(not isinstance(value, (bool, np.bool_)) for value in reconstruction_exact)
        or len(reconstruction_max_abs) != n_rows
        or not np.isfinite(np.asarray(reconstruction_max_abs, dtype=float)).all()
        or any(float(value) < 0.0 for value in reconstruction_max_abs)
    ):
        raise ValueError(f"history reconstruction audit is invalid at window {window}")
    if window == 0 and (
        not all(bool(value) for value in reconstruction_exact)
        or any(float(value) != 0.0 for value in reconstruction_max_abs)
    ):
        raise ValueError("first window cannot report a reconstructed input history")
    if window and any(
        bool(is_exact) and float(max_abs) != 0.0
        for is_exact, max_abs in zip(reconstruction_exact, reconstruction_max_abs)
    ):
        raise ValueError(f"exact history reconstruction has nonzero error at window {window}")
    record = {
        "window": window,
        **{key: value for key, value in expected.items() if key != "window_index"},
        "n_returned_prefixes_exact": int(sum(bool(value) for value in reconstruction_exact)),
        "max_returned_prefix_error": float(max(
            (float(value) for value in reconstruction_max_abs), default=0.0)),
        "stable_hashes_match_archived_features": True,
        "input_hashes_match_visible_transcript_suffix": True,
    }
    return shape, hashes, record


def _first_divergence_frame(first: np.ndarray, second: np.ndarray) -> int | None:
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape:
        raise ValueError("divergence check requires equal shapes")
    differs = np.any(a != b, axis=tuple(range(1, a.ndim))) if a.ndim > 1 else (a != b)
    frames = np.flatnonzero(differs)
    return int(frames[0]) if len(frames) else None


def validate_noise_and_feature_forks(
    features: np.ndarray,
    noise_audit: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Hard causal-design gates before decoding or scientific scoring.

    All four arms of a seed must share corresponding-window noise; every row must draw fresh
    noise per window; distinct seeds must differ; and each delayed arm's accepted feature
    transcript must be byte-identical to ``all_walk`` before its fork frame.
    """
    exact = np.asarray(features)
    if exact.ndim != 3 or exact.shape[:2] != (len(plan), PADDED_FRAMES):
        raise ValueError(
            f"scheduled features must be ({len(plan)}, {PADDED_FRAMES}, D), got {exact.shape}")
    if exact.shape[2] < 1 or not np.isfinite(exact).all():
        raise ValueError("scheduled features are empty or nonfinite")
    if len(noise_audit) != N_WINDOWS:
        raise ValueError(
            f"EXP-023b expected {N_WINDOWS} audited latent draws, got {len(noise_audit)}")
    by_key = {(int(item["seed"]), str(item["arm"])): int(item["row_index"]) for item in plan}
    planned_seeds = tuple(dict.fromkeys(int(item["seed"]) for item in plan))
    if (
        len(by_key) != len(plan)
        or set(by_key) != {(seed, arm) for seed in planned_seeds for arm in ARMS}
        or sorted(by_key.values()) != list(range(len(plan)))
    ):
        raise ValueError("EXP-023b causal plan is not a complete local seed/arm product")

    shapes: list[list[int]] = []
    hashes_by_window: list[list[str]] = []
    history_reconstruction: list[dict[str, Any]] = []
    for window, audit in enumerate(noise_audit):
        shape, hashes, record = _check_window_audit(window, audit, exact)
        shapes.append(shape)
        hashes_by_window.append(hashes)
        history_reconstruction.append(record)
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("initial-latent shape changed across autoregressive windows")

    paired_noise: list[dict[str, Any]] = []
    for seed in planned_seeds:
        indices = [by_key[(seed, arm)] for arm in ARMS]
        for window, hashes in enumerate(hashes_by_window):
            values = [hashes[index] for index in indices]
            if len(set(values)) != 1:
                raise ValueError(
                    f"same-seed arms received unequal noise at seed {seed}, window {window}")
            paired_noise.append({
                "seed": seed,
                "window": window,
                "sha256": values[0],
                "all_arms_equal": True,
            })
        for arm, index in zip(ARMS, indices):
            stream = [hashes[index] for hashes in hashes_by_window]
            if len(set(stream)) != N_WINDOWS:
                raise ValueError(f"latent replay detected for seed {seed}, arm {arm}: {stream}")
    for window, hashes in enumerate(hashes_by_window):
        per_seed = [hashes[by_key[(seed, REFERENCE_ARM)]] for seed in planned_seeds]
        if len(set(per_seed)) != len(planned_seeds):
            raise ValueError(f"distinct seeds collided in latent audit at window {window}")

    feature_forks: list[dict[str, Any]] = []
    for seed in planned_seeds:
        walk = exact[by_key[(seed, REFERENCE_ARM)]]
        record: dict[str, Any] = {"seed": seed, "reference_arm": REFERENCE_ARM}
        for arm, fork in FORK_FRAMES.items():
            delayed = exact[by_key[(seed, arm)]]
            if not np.array_equal(walk[:fork], delayed[:fork]):
                raise ValueError(f"{arm} feature prefix differs from WALK for seed {seed}")
            record[arm] = {
                "fork_frame": int(fork),
                "prefix_sha256": _array_sha256(walk[:fork], "prefix"),
                "exact_through_fork": True,
                "first_divergence_frame_from_all_walk": _first_divergence_frame(walk, delayed),
            }
        record["squeeze_0"] = {
            "fork_frame": 0,
            "first_divergence_frame_from_all_walk": _first_divergence_frame(
                walk, exact[by_key[(seed, "squeeze_0")]]),
        }
        feature_forks.append(record)

    return {
        "n_windows": N_WINDOWS,
        "arms": list(ARMS),
        "fork_frames": {arm: int(fork) for arm, fork in FORK_FRAMES.items()},
        "latent_shapes": shapes,
        "paired_noise": paired_noise,
        "feature_forks": feature_forks,
        "history_reconstruction": history_reconstruction,
        "corresponding_window_noise_equal": True,
        "noise_fresh_across_windows": True,
        "distinct_seeds_differ": True,
        "history_inputs_match_visible_transcript_suffixes": True,
        "stable_transcript_hashes_match_archived_features": True,
        "feature_prefixes_exact": True,
    }


def _validate_qpos_forks(qpos_by_key: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        walk = np.asarray(qpos_by_key[f"s{seed}_{REFERENCE_ARM}"])
        record: dict[str, Any] = {"seed": seed, "reference_arm": REFERENCE_ARM}
        for arm, fork in FORK_FRAMES.items():
            delayed = np.asarray(qpos_by_key[f"s{seed}_{arm}"])
            if not np.array_equal(walk[:fork], delayed[:fork]):
                raise ValueError(f"{arm} decoded qpos prefix differs for seed {seed}")
            record[arm] = {
                "fork_frame": int(fork),
                "prefix_sha256": _array_sha256(walk[:fork], "qpos_prefix"),
                "exact_through_fork": True,
                "first_divergence_frame_from_all_walk": _first_divergence_frame(walk, delayed),
            }
        rows.append(record)
    return rows


# --- summary, gates, decision rule -----------------------------------------------------


def _rate_record(present: int, planned: int) -> dict[str, Any]:
    low, high = wilson_interval(int(present), int(planned))
    return {
        "present": int(present),
        "planned": int(planned),
        "missing": int(planned) - int(present),
        "rate": float(present / planned),
        "wilson95": [float(low), float(high)],
    }


def _spread(values: Sequence[float]) -> dict[str, Any]:
    exact = [float(value) for value in values]
    return {
        "values": exact,
        "n": len(exact),
        "median": float(np.median(exact)) if exact else None,
        "min": float(min(exact)) if exact else None,
        "max": float(max(exact)) if exact else None,
    }


def _two_by_two(first: Sequence[bool], second: Sequence[bool]) -> dict[str, int]:
    pairs = list(zip(first, second))
    return {
        "both": int(sum(a and b for a, b in pairs)),
        "first_only": int(sum(a and not b for a, b in pairs)),
        "second_only": int(sum(b and not a for a, b in pairs)),
        "neither": int(sum(not a and not b for a, b in pairs)),
        "planned": len(pairs),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    route: np.ndarray,
    qpos_by_key: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    by_seed_arm = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    if set(by_seed_arm) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("summary requires every planned (seed, arm) row exactly once")
    walk_rows = [by_seed_arm[(seed, REFERENCE_ARM)] for seed in SEEDS]

    sidestep_rates: dict[str, Any] = {}
    step_rates: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    fixed_box_rates: dict[str, Any] = {}
    traversal_rates: dict[str, Any] = {}
    route_fidelity: dict[str, Any] = {}
    for arm in PROMPT_ARMS:
        arm_rows = [by_seed_arm[(seed, arm)] for seed in SEEDS]
        sidesteps = [row["sidestep"] for row in arm_rows]
        events = [row["event"] for row in arm_rows]
        sidestep_rates[arm] = {
            "composite": _rate_record(
                sum(bool(item["present"]) for item in sidesteps), len(arm_rows)),
            "signatures": {
                name: _rate_record(
                    sum(bool(item["signatures"][name]["present"]) for item in sidesteps),
                    len(arm_rows))
                for name in SIDESTEP_SIGNATURES
            },
        }
        step_rates[arm] = _rate_record(
            sum(bool(item["present"]) for item in events), len(arm_rows))
        latency[arm] = {
            "sidestep_first_signature_latency_frames": _spread([
                item["first_signature_latency_frames"]
                for item in sidesteps if item["present"]
            ]),
            "step_event_latency_frames": _spread([
                item["latency_frames"] for item in events if item["present"]
            ]),
        }
        fixed_box_rates[arm] = {
            f"{height:g}": _rate_record(
                sum(bool(row["fixed_box"]["clears"][f"{height:g}"]) for row in arm_rows),
                len(arm_rows))
            for height in GRADED_HEIGHTS_M
        }
        traversal_rates[arm] = _rate_record(
            sum(bool(row["fixed_box"]["traversal"]["traversed"]) for row in arm_rows),
            len(arm_rows))
    for arm in ARMS:
        arm_rows = [by_seed_arm[(seed, arm)] for seed in SEEDS]
        route_fidelity[arm] = {
            name: _spread([row["supporting_motion"][name] for row in arm_rows])
            for name in ("progress_ratio", "route_path_mae_m", "max_foot_floor_penetration_m")
        }

    walk_sidestep_per_window = {
        str(onset): _rate_record(
            sum(bool(row["control_sidesteps"][str(onset)]["present"]) for row in walk_rows),
            len(walk_rows))
        for onset in CONTROL_ONSETS
    }
    walk_sidestep_seed_hits = int(sum(
        any(item["present"] for item in row["control_sidesteps"].values())
        for row in walk_rows
    ))
    walk_signature_seed_hits = {
        name: int(sum(
            any(item["signatures"][name]["present"]
                for item in row["control_sidesteps"].values())
            for row in walk_rows
        ))
        for name in SIDESTEP_SIGNATURES
    }
    walk_step_seed_hits = int(sum(
        any(item["present"] for item in row["control_events"].values())
        for row in walk_rows
    ))
    walk_step_window_hits = int(sum(
        int(item["present"]) for row in walk_rows for item in row["control_events"].values()
    ))
    fixed_box_rates["all_walk_matched_windows"] = {
        str(onset): {
            f"{height:g}": _rate_record(
                sum(bool(row["control_fixed_boxes"][str(onset)]["clears"][f"{height:g}"])
                    for row in walk_rows),
                len(walk_rows))
            for height in GRADED_HEIGHTS_M
        }
        for onset in CONTROL_ONSETS
    }
    traversal_rates["all_walk_matched_windows"] = {
        str(onset): _rate_record(
            sum(bool(row["control_fixed_boxes"][str(onset)]["traversal"]["traversed"])
                for row in walk_rows),
            len(walk_rows))
        for onset in CONTROL_ONSETS
    }

    paired_seed_table: list[dict[str, Any]] = []
    for seed in SEEDS:
        walk = by_seed_arm[(seed, REFERENCE_ARM)]
        entry: dict[str, Any] = {
            "seed": seed,
            "all_walk": {
                **{
                    f"sidestep_window_{onset}": bool(
                        walk["control_sidesteps"][str(onset)]["present"])
                    for onset in CONTROL_ONSETS
                },
                **{
                    f"step_event_window_{onset}": bool(
                        walk["control_events"][str(onset)]["present"])
                    for onset in CONTROL_ONSETS
                },
            },
        }
        for arm in PROMPT_ARMS:
            row = by_seed_arm[(seed, arm)]
            entry[arm] = {
                "sidestep": bool(row["sidestep"]["present"]),
                "sidestep_signatures": list(row["sidestep"]["present_signatures"]),
                "step_event": bool(row["event"]["present"]),
            }
        paired_seed_table.append(entry)

    def column(arm: str, field: str) -> list[bool]:
        return [bool(entry[arm][field]) for entry in paired_seed_table]

    paired_counts = {
        "squeeze_0_sidestep_vs_squeeze_52_sidestep": _two_by_two(
            column("squeeze_0", "sidestep"), column("squeeze_52", "sidestep")),
        "squeeze_52_sidestep_vs_step_52_step_event": _two_by_two(
            column("squeeze_52", "sidestep"), column("step_52", "step_event")),
        "squeeze_0_sidestep_vs_step_52_step_event": _two_by_two(
            column("squeeze_0", "sidestep"), column("step_52", "step_event")),
        "squeeze_52_sidestep_vs_all_walk_window_52_sidestep": _two_by_two(
            column("squeeze_52", "sidestep"), column("all_walk", "sidestep_window_52")),
    }

    transmission_rows = [handoff_transmission(qpos_by_key, seed) for seed in SEEDS]
    transmission = {
        "definition": (
            "RMS over 96 post-switch frames (52..147) and all joint coordinates qpos[:, 7:] "
            "of the per-frame joint-angle difference between two arms of the same seed"
        ),
        "per_seed": transmission_rows,
        "joint_rms_rad": {
            name: _spread([item["joint_rms_rad"][name] for item in transmission_rows])
            for name in TRANSMISSION_PAIRS
        },
        "prefix_joint_rms_rad_frames_0_51": {
            name: _spread([
                item["prefix_joint_rms_rad_frames_0_51"][name] for item in transmission_rows
            ])
            for name in (f"{arm}_vs_all_walk" for arm in FORK_FRAMES)
        },
        "inference": "descriptive only; no interval or test is claimed",
    }

    return {
        "sidestep_rates_missing_retained": sidestep_rates,
        "step_event_rates_missing_retained": step_rates,
        "prompt_relative_latency": latency,
        "all_walk_sidestep_specificity": {
            "seeds_with_any_signature_in_any_window": walk_sidestep_seed_hits,
            "planned_seeds": len(walk_rows),
            "window_composites": int(sum(
                int(item["present"]) for row in walk_rows
                for item in row["control_sidesteps"].values())),
            "planned_windows": len(walk_rows) * len(CONTROL_ONSETS),
            "per_window": walk_sidestep_per_window,
            "seeds_with_signature": walk_signature_seed_hits,
        },
        "all_walk_step_specificity": {
            "seeds_with_any_event": walk_step_seed_hits,
            "planned_seeds": len(walk_rows),
            "window_events": walk_step_window_hits,
            "planned_windows": len(walk_rows) * len(CONTROL_ONSETS),
        },
        "fixed_box_rates": fixed_box_rates,
        "fixed_box_traversal_rates": traversal_rates,
        "route_fidelity_reported_not_enforced": route_fidelity,
        "paired_seed_table": paired_seed_table,
        "paired_counts": paired_counts,
        "handoff_transmission": transmission,
        "inference": (
            "planned-denominator rates with Wilson 95% intervals; paired counts across arms; "
            "descriptive at n=8; no interval is claimed on any paired difference"
        ),
    }


def evaluate_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    squeeze0 = int(summary["sidestep_rates_missing_retained"]["squeeze_0"]["composite"]["present"])
    walk_sidestep = int(
        summary["all_walk_sidestep_specificity"]["seeds_with_any_signature_in_any_window"])
    walk_step = int(summary["all_walk_step_specificity"]["seeds_with_any_event"])
    gates = {
        "squeeze0_substrate": {
            "required_min_present": MIN_SQUEEZE0_SIDESTEPS,
            "observed_present": squeeze0,
            "planned": len(SEEDS),
            "pass": squeeze0 >= MIN_SQUEEZE0_SIDESTEPS,
        },
        "all_walk_sidestep_specificity": {
            "allowed_max_seeds_with_any_signature": MAX_ALL_WALK_SEEDS_WITH_ANY_SIDESTEP,
            "observed_seeds_with_any_signature": walk_sidestep,
            "planned": len(SEEDS),
            "pass": walk_sidestep <= MAX_ALL_WALK_SEEDS_WITH_ANY_SIDESTEP,
        },
        "all_walk_step_specificity": {
            "allowed_max_seeds_with_any_event": MAX_ALL_WALK_SEEDS_WITH_ANY_STEP_EVENT,
            "observed_seeds_with_any_event": walk_step,
            "planned": len(SEEDS),
            "pass": walk_step <= MAX_ALL_WALK_SEEDS_WITH_ANY_STEP_EVENT,
        },
        "delayed_arm_absence_is_an_outcome_not_a_gate": True,
    }
    gates["all_pass"] = bool(
        gates["squeeze0_substrate"]["pass"]
        and gates["all_walk_sidestep_specificity"]["pass"]
        and gates["all_walk_step_specificity"]["pass"]
    )
    return gates


def evaluate_decision_rule(
    summary: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    """The protocol's decision rule, evaluated mechanically; valid only if every gate passed."""
    rates = summary["sidestep_rates_missing_retained"]
    squeeze52 = int(rates["squeeze_52"]["composite"]["present"])
    squeeze0 = int(rates["squeeze_0"]["composite"]["present"])
    step52 = int(summary["step_event_rates_missing_retained"]["step_52"]["present"])
    transmits = (
        squeeze52 >= DECISION_MIN_SQUEEZE52_SIDESTEPS and step52 <= DECISION_MAX_STEP52_EVENTS
    )
    no_switch = (
        squeeze52 <= DECISION_MAX_SQUEEZE52_SIDESTEPS_NO_SWITCH
        and squeeze0 >= MIN_SQUEEZE0_SIDESTEPS
    )
    replication_failed = step52 >= DECISION_STEP52_REPLICATION_FAIL_MIN
    if replication_failed:
        # EXP-023's delayed-arm zero has not replicated on fresh seeds: the delayed-prompt
        # sentence is withdrawn regardless of squeeze_52 (protocol, decision rules).
        verdict = VERDICT_STEP_REPLICATION_FAILED
    elif transmits:
        verdict = VERDICT_TRANSMITS
    elif no_switch:
        verdict = VERDICT_NO_SWITCH
    else:
        verdict = VERDICT_INDETERMINATE
    gates_pass = bool(gates.get("all_pass"))
    return {
        "inputs": {
            "squeeze_52_sidestep_present": squeeze52,
            "squeeze_0_sidestep_present": squeeze0,
            "step_52_step_event_present": step52,
            "planned": len(SEEDS),
        },
        "rules": [
            {
                "verdict": VERDICT_STEP_REPLICATION_FAILED,
                "condition": (
                    f"step_52 step event >= {DECISION_STEP52_REPLICATION_FAIL_MIN}/8 "
                    "(checked first; withdraws the delayed-prompt sentence)"
                ),
                "satisfied": bool(replication_failed),
            },
            {
                "verdict": VERDICT_TRANSMITS,
                "condition": (
                    f"squeeze_52 sidestep >= {DECISION_MIN_SQUEEZE52_SIDESTEPS}/8 and "
                    f"step_52 step event <= {DECISION_MAX_STEP52_EVENTS}/8"
                ),
                "satisfied": bool(transmits),
            },
            {
                "verdict": VERDICT_NO_SWITCH,
                "condition": (
                    f"squeeze_52 sidestep <= {DECISION_MAX_SQUEEZE52_SIDESTEPS_NO_SWITCH}/8 "
                    f"with squeeze_0 sidestep >= {MIN_SQUEEZE0_SIDESTEPS}/8"
                ),
                "satisfied": bool(no_switch),
            },
        ],
        "verdict": verdict,
        "gates_all_pass": gates_pass,
        "verdict_valid": gates_pass,
        "step_52_replicates_exp023_zero_of_eight": bool(step52 <= DECISION_MAX_STEP52_EVENTS),
        "pooled_step_52_with_exp023": {
            "exp023_step_52_present_of_8": 0,
            "exp023b_step_52_present_of_8": step52,
            "pooled_present_of_16": step52,
            "wilson95": list(wilson_interval(step52, 16)),
        },
        "note": (
            "Rules are evaluated in order; in-between counts report no binary verdict at n=8. "
            "A verdict is scientifically valid only when every preregistered gate passed."
        ),
    }


def dry_run_report(host_gate_fn: Callable[[], Mapping[str, Any]] = evaluate_host_gate) -> dict:
    """Locked plan plus the live host-gate report; writes nothing."""
    plan = locked_row_plan()
    chunks = locked_chunk_plan(plan)
    repo = Path(__file__).resolve().parents[1]
    try:
        report: Mapping[str, Any] | None = host_gate_fn()
        gate_pass = True
    except HostGateRefusal as exc:
        report = exc.report
        gate_pass = False
    return {
        "dry_run": True,
        "nothing_written": True,
        "experiment": "exp023b_prompt_switch_control",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "schedules": {arm: list(SCHEDULES[arm]) for arm in ARMS},
        "onsets": {arm: ONSETS[arm] for arm in ARMS},
        "row_plan": plan,
        "row_plan_sha256": cal._json_hash(plan),
        "chunk_plan": [
            {key: value for key, value in chunk.items() if key != "rows"} for chunk in chunks
        ],
        "protocol": {"path": PROTOCOL_PATH, "sha256": cal._sha256(repo / PROTOCOL_PATH)},
        "host_resource_gate": _json_safe(report),
        "host_gate_pass": bool(gate_pass),
    }


# --- campaign --------------------------------------------------------------------------


def run_campaign(
    *,
    out: str | Path,
    runner: Any | None = None,
    runner_factory: Callable[[], Any] | None = None,
    body: Any | None = None,
    cache_path: str | Path = "outputs/text_cache.npz",
    host_gate_fn: Callable[[], Mapping[str, Any]] = evaluate_host_gate,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = cal._git_state,
    source_hashes_fn: Callable[[Path], Mapping[str, str]] = _source_hashes,
    generator_identity_fn: Callable[[Any], Mapping[str, Any]] = cal._generator_identity,
    generator_identity_validator_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        cal._validated_generator_identity
    ),
    runtime_identity_fn: Callable[[], Mapping[str, Any]] = cal._runtime_identity,
    physical_identity_fn: Callable[[], Mapping[str, Any]] = cal._physical_model_identity,
    prompt_identity_fn: Callable[[Any, Path], Mapping[str, Any]] = _prompt_cache_identity,
    pin_validator_fn: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], None] = (
        _validate_pins
    ),
    channel_usage_fn: Callable[[Any, ConstraintSpec], Mapping[str, int]] = _actual_channel_usage,
    event_detector_fn: Callable[[Any, np.ndarray, np.ndarray, int], Mapping[str, Any]] = (
        detect_prompt_event
    ),
    sidestep_detector_fn: Callable[[Any, np.ndarray, np.ndarray, int], Mapping[str, Any]] = (
        detect_sidestep
    ),
    exact_box_fn: Callable[[Any, np.ndarray, float], Mapping[str, Any]] = score_exact_boxes,
    motion_metrics_fn: Callable[[Any, np.ndarray, np.ndarray], Mapping[str, Any]] = (
        supporting_motion_metrics
    ),
) -> dict[str, Any]:
    """Run the locked prompt-switch control or raise after durable refusal.

    Order of operations: refuse a non-empty output directory; evaluate the host gate (a
    failure raises :class:`HostGateRefusal` with nothing written); persist the empty ledger;
    only then construct the runner.
    """
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise PromptSwitchAbort(f"refusing nonempty EXP-023b output directory: {output}")
    # The host gate precedes every write and the runner construction; its failure must leave
    # the output directory untouched so the same directory can be launched later.
    try:
        host_report = dict(host_gate_fn())
    except hg.HostResourceGateFailed as exc:
        raise HostGateRefusal(str(exc)) from exc
    if host_report.get("pass") is not True:
        raise HostGateRefusal("host-resource gate report does not pass", host_report)

    repo = Path(__file__).resolve().parents[1]
    code = dict(code_state_fn(repo))
    injected_components = []
    if runner is not None:
        injected_components.append("runner_instance")
    if runner_factory is not None:
        injected_components.append("runner_factory")
    if body is not None:
        injected_components.append("body")
    callbacks = (
        ("host_gate_fn", host_gate_fn, evaluate_host_gate),
        ("code_state_fn", code_state_fn, cal._git_state),
        ("source_hashes_fn", source_hashes_fn, _source_hashes),
        ("generator_identity_fn", generator_identity_fn, cal._generator_identity),
        ("generator_identity_validator_fn", generator_identity_validator_fn,
         cal._validated_generator_identity),
        ("runtime_identity_fn", runtime_identity_fn, cal._runtime_identity),
        ("physical_identity_fn", physical_identity_fn, cal._physical_model_identity),
        ("prompt_identity_fn", prompt_identity_fn, _prompt_cache_identity),
        ("pin_validator_fn", pin_validator_fn, _validate_pins),
        ("channel_usage_fn", channel_usage_fn, _actual_channel_usage),
        ("event_detector_fn", event_detector_fn, detect_prompt_event),
        ("sidestep_detector_fn", sidestep_detector_fn, detect_sidestep),
        ("exact_box_fn", exact_box_fn, score_exact_boxes),
        ("motion_metrics_fn", motion_metrics_fn, supporting_motion_metrics),
    )
    injected_components.extend(
        name for name, value, default in callbacks if value is not default
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "preflight"
    sample_count_exact = True
    rows: list[dict[str, Any]] = []
    qpos_archive: dict[str, np.ndarray] = {}
    feature_archive: dict[str, np.ndarray] = {}
    noise_evidence: list[dict[str, Any]] = []
    plan = locked_row_plan()
    chunks = locked_chunk_plan(plan)
    spent_seeds: list[int] = []
    route = route_xz()
    route_digest = _array_sha256(route, "route_xz")
    counters = {
        "schedule_invocations_planned": len(chunks),
        "schedule_invocations_started": 0,
        "schedule_invocations_completed": 0,
        "autoregressive_window_calls_planned": len(chunks) * N_WINDOWS,
        "autoregressive_window_calls_completed": 0,
        "trajectories_planned": len(plan),
        "trajectories_launched": 0,
        "trajectories_returned": 0,
        "trajectories_converted_to_qpos": 0,
        "trajectories_analyzed": 0,
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "experiment": "exp023b_prompt_switch_control",
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": stage,
        "sample_count_exact": sample_count_exact,
        "actual_ardy_samples": 0,
        "host_resource_gate": _json_safe(host_report),
        "campaign_design": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "prompt_arms": list(PROMPT_ARMS),
            "reference_arm": REFERENCE_ARM,
            "prompts": {"WALK": WALK, "STEP": STEP, "SQUEEZE": SQUEEZE},
            "schedules": {arm: list(SCHEDULES[arm]) for arm in ARMS},
            "onsets": {arm: ONSETS[arm] for arm in ARMS},
            "control_onsets": list(CONTROL_ONSETS),
            "fork_frames": {arm: int(fork) for arm, fork in FORK_FRAMES.items()},
            "row_plan": plan,
            "row_plan_sha256": cal._json_hash(plan),
            "chunk_plan": [
                {key: value for key, value in chunk.items() if key != "rows"}
                for chunk in chunks
            ],
            "chunk_seed_count": CHUNK_SEED_COUNT,
            "chunk_batch_rows": CHUNK_ROWS,
            "n_schedule_trajectories": len(plan),
            "sample_accounting_definition": (
                "one complete prompt schedule is one frozen-prior sample; four B=8 chunk "
                "calls each make four Horizon52 window calls, counted separately"
            ),
            "history_policy": {
                "name": "ARDY interactive GUI default minimum history (EXP-023 contract)",
                "visible_history_frames": list(e23.EXPECTED_INPUT_HISTORY_FRAMES),
                "global_history_start_frames": list(e23.EXPECTED_GLOBAL_HISTORY_START_FRAMES),
                "accepted_transcript_frames_before": list(
                    e23.EXPECTED_ACCEPTED_TRANSCRIPT_FRAMES_BEFORE),
                "accepted_transcript_frames_after": list(e23.EXPECTED_TRANSCRIPT_FRAMES),
                "model_visible_num_frames": list(e23.EXPECTED_MODEL_NUM_FRAMES),
                "full_accepted_transcript_is_immutable": True,
            },
            "fps": FPS,
            "scored_frames": N_FRAMES,
            "archived_frames": PADDED_FRAMES,
            "post_onset_frames": POST_ONSET_FRAMES,
            "frozen_latency_frames": FROZEN_LATENCY_FRAMES,
            "diffusion_steps": DIFFUSION_STEPS,
            "cfg_weight": list(CFG_WEIGHT),
            "route_sha256": route_digest,
            "conditioning": "dense root XZ only for frames 0:200; 200:208 unconditioned",
            "sidestep_detector": {
                "composite_rule": COMPOSITE_RULE,
                "signatures": list(SIDESTEP_SIGNATURES),
                "heading_sidle_min_deviation_deg": HEADING_SIDLE_MIN_DEVIATION_DEG,
                "lateral_excursion_min_m": LATERAL_EXCURSION_MIN_M,
                "sustained_min_frames": SUSTAINED_MIN_FRAMES,
                "foot_crossing_min_m": FOOT_CROSSING_MIN_M,
                "foot_crossing_min_frames": FOOT_CROSSING_MIN_FRAMES,
            },
            "event_min_clearance_m": e23.EVENT_MIN_CLEARANCE_M,
            "event_profile_points": e23.PROFILE_POINTS,
            "graded_heights_m": list(GRADED_HEIGHTS_M),
            "substrate_gate": f"squeeze_0 sidestep composite >= {MIN_SQUEEZE0_SIDESTEPS}/8",
            "specificity_gates": (
                "all_walk seeds with any sidestep signature in either control window "
                f"<= {MAX_ALL_WALK_SEEDS_WITH_ANY_SIDESTEP}/8; all_walk seeds with any step "
                f"event <= {MAX_ALL_WALK_SEEDS_WITH_ANY_STEP_EVENT}/8"
            ),
            "host_gate": {
                "preset": "scene2motion.host_gate.ARDY_GENERATION_GATE",
                "min_free_vram_mib": HOST_GATE["min_free_vram_mib"],
                "min_available_ram_mib": HOST_GATE["min_available_ram_mib"],
                "require_no_concurrent_isaac": HOST_GATE_REQUIRE_NO_ISAAC,
            },
        },
        "query_accounting": dict(counters),
        "generation_chunks": {
            str(chunk["name"]): {
                "status": "planned",
                "seeds": list(chunk["seeds"]),
                "row_indices": list(chunk["row_indices"]),
            }
            for chunk in chunks
        },
        "provenance": {"code": code},
        "execution_mode": {
            "dependency_injections": injected_components,
            "scientific_evidence_eligible": not injected_components,
            "pre_model_construction_evidence_guaranteed": runner is None,
            "note": (
                "Dependency injection exists for CPU tests; any injected run is explicitly "
                "non-evidentiary. Production constructs the ARDY runner only after the empty "
                "evidence bundle is durable."
            ),
        },
    }

    artifacts_dirty = {"qpos": True, "features": True, "noise": True}
    artifact_content_hash: dict[str, str | None] = {
        "qpos": None,
        "features": None,
        "noise": None,
    }

    def persist() -> None:
        cal._write_jsonl(output / "rows.jsonl", rows)
        if artifacts_dirty["qpos"]:
            cal._persist_qpos(output / "qpos.npz", qpos_archive)
            artifact_content_hash["qpos"] = (
                cal._array_hash(qpos_archive) if qpos_archive else None)
            artifacts_dirty["qpos"] = False
        if artifacts_dirty["features"]:
            cal._persist_qpos(output / "features.npz", feature_archive)
            artifact_content_hash["features"] = (
                cal._array_hash(feature_archive) if feature_archive else None)
            artifacts_dirty["features"] = False
        if artifacts_dirty["noise"]:
            cal._write_json(output / "noise_audit.json", noise_evidence)
            artifact_content_hash["noise"] = cal._json_hash(noise_evidence)
            artifacts_dirty["noise"] = False
        receipt["stage"] = stage
        receipt["query_accounting"] = dict(counters)
        receipt["sample_count_exact"] = sample_count_exact
        receipt["actual_ardy_samples"] = int(counters["trajectories_returned"])
        receipt["spent_seeds"] = list(spent_seeds)
        receipt["unlaunched_locked_seeds"] = [
            seed for seed in SEEDS if seed not in spent_seeds]
        receipt["seeds_spent_and_must_not_be_reused"] = bool(spent_seeds)
        receipt["evidence_anchors"] = {
            "rows": {
                "path": "rows.jsonl",
                "n_rows": len(rows),
                "logical_sha256": cal._json_hash(rows),
                "file_sha256": cal._sha256(output / "rows.jsonl"),
            },
            "qpos": {
                "path": "qpos.npz",
                "n_arrays": len(qpos_archive),
                "content_sha256": artifact_content_hash["qpos"],
                "file_sha256": cal._sha256(output / "qpos.npz"),
            },
            "features": {
                "path": "features.npz",
                "n_arrays": len(feature_archive),
                "content_sha256": artifact_content_hash["features"],
                "file_sha256": cal._sha256(output / "features.npz"),
            },
            "noise_audit": {
                "path": "noise_audit.json",
                "n_records": len(noise_evidence),
                "logical_sha256": artifact_content_hash["noise"],
                "file_sha256": cal._sha256(output / "noise_audit.json"),
            },
        }
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        cal._write_json(output / "receipt.json", receipt)

    # Required before runner construction or any GPU generation.
    persist()

    try:
        if runner is not None and runner_factory is not None:
            raise ValueError("provide either runner or runner_factory, not both")
        if code.get("dirty") is not False:
            raise ValueError("EXP-023b requires an exactly clean git worktree")
        if not isinstance(code.get("commit"), str) or not code["commit"].strip():
            raise ValueError("EXP-023b requires a concrete git commit")
        if os.environ.get("CHECKPOINTS_DIR"):
            raise ValueError("EXP-023b forbids ambient CHECKPOINTS_DIR")

        source_hashes = dict(source_hashes_fn(repo))
        protocol_sha256 = source_hashes.get(PROTOCOL_PATH)
        if not cal._is_sha256(protocol_sha256):
            raise ValueError("EXP-023b protocol content hash is missing or invalid")
        receipt["provenance"]["source_sha256"] = source_hashes
        receipt["provenance"]["protocol"] = {
            "path": PROTOCOL_PATH,
            "sha256": protocol_sha256,
        }
        if runner is None:
            runner = (
                runner_factory() if runner_factory is not None
                else ArdyRunner(cache_path=cache_path)
            )
        if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
            raise ValueError(f"EXP-023b requires runner fps == {FPS:g}")
        if int(runner.noise_stream_version) != NOISE_STREAM_VERSION:
            raise ValueError("EXP-023b requires noise_stream_version == 2")
        if int(runner.model.gen_horizon_len) != HORIZON:
            raise ValueError("EXP-023b requires the Horizon52 checkpoint")
        token = int(runner.model.num_frames_per_token)
        if token <= 0 or HORIZON % token or PADDED_FRAMES % token:
            raise ValueError("EXP-023b checkpoint token size is incompatible with Horizon52")
        if token != e23.EXPECTED_INPUT_HISTORY_FRAMES[1]:
            raise ValueError(
                "EXP-023b locks the released GUI-default four-frame history token")

        generator_identity = dict(generator_identity_validator_fn(
            generator_identity_fn(runner)))
        runtime_identity = dict(runtime_identity_fn())
        physical_identity = dict(physical_identity_fn())
        prompt_identity = dict(prompt_identity_fn(runner, Path(cache_path)))
        pin_validator_fn(generator_identity, runtime_identity, physical_identity)
        receipt["provenance"].update({
            "generator": generator_identity,
            "runtime": runtime_identity,
            "physical_model": physical_identity,
            "walk_step_squeeze_prompt_cache": prompt_identity,
        })
        body = body or G1Body(None)

        spec = root_only_spec(route)
        channel_usage = {str(name): int(value)
                         for name, value in channel_usage_fn(runner, spec).items()}
        if channel_usage != {"root_pos": 2 * N_FRAMES}:
            raise ValueError(
                "EXP-023b requires exactly dense root-XZ conditioning; observed "
                f"{channel_usage}"
            )
        receipt["campaign_design"]["actual_channel_usage"] = channel_usage

        def revalidate_bound_identities() -> dict[str, Any]:
            current_code = dict(code_state_fn(repo))
            git_check = cal._verify_completion_git_state(
                code, current_code, repo=repo, output=output)
            if dict(source_hashes_fn(repo)) != source_hashes:
                raise ValueError("EXP-023b source content changed during the campaign")
            current_generator = dict(generator_identity_validator_fn(
                generator_identity_fn(runner)))
            current_runtime = dict(runtime_identity_fn())
            current_physical = dict(physical_identity_fn())
            current_prompts = dict(prompt_identity_fn(runner, Path(cache_path)))
            if current_generator != generator_identity:
                raise ValueError("EXP-023b checkpoint identity changed")
            if current_runtime != runtime_identity:
                raise ValueError("EXP-023b ARDY/numerical runtime identity changed")
            if current_physical != physical_identity:
                raise ValueError("EXP-023b G1 physical model identity changed")
            if current_prompts != prompt_identity:
                raise ValueError("EXP-023b cached prompt identity changed")
            pin_validator_fn(current_generator, current_runtime, current_physical)
            if (
                float(runner.fps) != FPS
                or int(runner.noise_stream_version) != NOISE_STREAM_VERSION
                or int(runner.model.gen_horizon_len) != HORIZON
                or int(runner.model.num_frames_per_token)
                != e23.EXPECTED_INPUT_HISTORY_FRAMES[1]
            ):
                raise ValueError("EXP-023b runner contract changed")
            return {
                "git": git_check,
                "sources_unchanged": True,
                "checkpoint_unchanged": True,
                "runtime_unchanged": True,
                "physical_model_unchanged": True,
                "prompt_cache_unchanged": True,
                "runner_contract_unchanged": True,
            }

        chunk_audits: list[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = []
        for chunk in chunks:
            chunk_name = str(chunk["name"])
            stage = f"generation_{chunk_name}"
            chunk_rows = list(chunk["rows"])
            local_plan = [
                {
                    **dict(item),
                    "global_row_index": int(item["row_index"]),
                    "row_index": local_index,
                }
                for local_index, item in enumerate(chunk_rows)
            ]
            counters["schedule_invocations_started"] += 1
            counters["trajectories_launched"] += len(chunk_rows)
            spent_seeds.extend(seed for seed in chunk["seeds"] if seed not in spent_seeds)
            receipt["generation_chunks"][chunk_name].update({
                "status": "running",
                "local_row_plan": local_plan,
            })
            persist()
            try:
                chunk_features, raw_chunk_audit = runner.generate_prompt_schedule(
                    [item["prompt_schedule"] for item in chunk_rows],
                    [spec] * len(chunk_rows),
                    N_FRAMES,
                    DIFFUSION_STEPS,
                    cfg_weight=CFG_WEIGHT,
                    seeds=[int(item["seed"]) for item in chunk_rows],
                    history_frames=token,
                )
            except Exception:
                sample_count_exact = False
                receipt["generation_chunks"][chunk_name]["status"] = (
                    "generation_exception_window_count_unknown")
                persist()
                raise

            exact_chunk = np.asarray(chunk_features)
            counters["schedule_invocations_completed"] += 1
            try:
                observed_window_calls = len(raw_chunk_audit)
            except TypeError:
                observed_window_calls = 0
            counters["autoregressive_window_calls_completed"] += observed_window_calls
            returned_count = int(exact_chunk.shape[0]) if exact_chunk.ndim >= 1 else 0
            counters["trajectories_returned"] += returned_count
            receipt["generation_chunks"][chunk_name].update({
                "status": "returned_unvalidated",
                "trajectories_returned": returned_count,
                "window_calls_returned": observed_window_calls,
            })

            # Archive this chunk before any shape or causal gate. A malformed return still
            # consumes its two fresh seeds and remains durable evidence.
            if exact_chunk.ndim == 0:
                feature_archive[f"{chunk_name}_raw_scalar_return"] = np.asarray(exact_chunk)
            else:
                for local_index in range(returned_count):
                    if local_index < len(chunk_rows):
                        item = chunk_rows[local_index]
                        key = f"s{item['seed']}_{item['arm']}"
                    else:
                        key = f"{chunk_name}_unexpected_returned_row_{local_index}"
                    feature_archive[key] = np.array(exact_chunk[local_index], copy=True)
            artifacts_dirty["features"] = True
            noise_evidence.append({
                "kind": "raw_chunk_runner_audit_before_validation",
                "chunk": chunk_name,
                "seeds": list(chunk["seeds"]),
                "audit": _json_safe(raw_chunk_audit),
            })
            artifacts_dirty["noise"] = True
            persist()

            local_audit = validate_noise_and_feature_forks(
                exact_chunk, raw_chunk_audit, local_plan)
            if returned_count != len(chunk_rows):
                raise ValueError(f"{chunk_name} returned the wrong trajectory count")
            chunk_audits.append((chunk, raw_chunk_audit))
            receipt["generation_chunks"][chunk_name].update({
                "status": "complete",
                "causal_pairing_audit": local_audit,
                "post_chunk_identity_revalidation": revalidate_bound_identities(),
            })
            persist()

        expected_feature_keys = {f"s{item['seed']}_{item['arm']}" for item in plan}
        if set(feature_archive) != expected_feature_keys:
            raise ValueError("EXP-023b feature archive does not exactly match its row plan")
        exact_features = np.stack([
            feature_archive[f"s{item['seed']}_{item['arm']}"] for item in plan
        ])
        merged_noise_audit = _merge_chunk_audits(chunk_audits, len(plan))
        fork_audit = validate_noise_and_feature_forks(
            exact_features, merged_noise_audit, plan)
        receipt["causal_pairing_audit"] = fork_audit
        noise_evidence[:] = _serialized_noise_evidence(merged_noise_audit, plan)
        artifacts_dirty["noise"] = True
        receipt["provenance"]["post_generation_identity_revalidation"] = (
            revalidate_bound_identities())
        persist()

        stage = "decode"
        decoded_count = 0
        for index, sample in enumerate(runner.decode_features(exact_features)):
            if index >= len(plan):
                raise ValueError("EXP-023b decoder returned extra samples")
            item = plan[index]
            key = f"s{item['seed']}_{item['arm']}"
            qpos = np.asarray(runner.to_qpos(sample))
            if qpos.ndim != 2 or qpos.shape[0] != PADDED_FRAMES or not np.isfinite(qpos).all():
                raise ValueError(f"EXP-023b decoded invalid qpos for {key}: {qpos.shape}")
            # Preserve native decoded precision for the exact fork gate.
            qpos_archive[key] = np.array(qpos, copy=True)
            artifacts_dirty["qpos"] = True
            counters["trajectories_converted_to_qpos"] += 1
            decoded_count += 1
            persist()
        if decoded_count != len(plan):
            raise ValueError("EXP-023b decoder returned the wrong number of samples")
        receipt["causal_pairing_audit"]["qpos_forks"] = _validate_qpos_forks(qpos_archive)
        receipt["causal_pairing_audit"]["qpos_prefixes_exact"] = True
        persist()

        stage = "analysis"
        predicted_centres = {
            onset: float(route[onset + FROZEN_LATENCY_FRAMES, 1]) for onset in CONTROL_ONSETS
        }
        receipt["campaign_design"]["predicted_box_centres_m"] = {
            str(onset): value for onset, value in predicted_centres.items()
        }
        noise_hash_by_row = {
            int(item["row_index"]): [
                str(merged_noise_audit[window]["row_sha256"][int(item["row_index"])])
                for window in range(N_WINDOWS)
            ]
            for item in plan
        }
        for item in plan:
            seed = int(item["seed"])
            arm = str(item["arm"])
            index = int(item["row_index"])
            key = f"s{seed}_{arm}"
            full_qpos = np.asarray(qpos_archive[key], dtype=float)
            scored_qpos = full_qpos[:N_FRAMES]
            row: dict[str, Any] = {
                **dict(item),
                "archive_key": key,
                "archived_frames": PADDED_FRAMES,
                "scored_frames": N_FRAMES,
                "noise_sha256_by_window": noise_hash_by_row[index],
                "features_sha256": _array_sha256(feature_archive[key], "features"),
                "qpos_sha256": _array_sha256(qpos_archive[key], "qpos"),
                "supporting_motion": _validated_motion_metrics(
                    motion_metrics_fn(body, scored_qpos, route)),
            }
            if arm == REFERENCE_ARM:
                row.update({"event": None, "fixed_box": None, "sidestep": None})
                row["control_events"] = {
                    str(onset): _validated_event_record(
                        event_detector_fn(body, scored_qpos, route, onset), onset)
                    for onset in CONTROL_ONSETS
                }
                row["control_fixed_boxes"] = {
                    str(onset): _validated_exact_box_record(
                        exact_box_fn(body, scored_qpos, predicted_centres[onset]),
                        predicted_centres[onset])
                    for onset in CONTROL_ONSETS
                }
                row["control_sidesteps"] = {
                    str(onset): _validated_sidestep_record(
                        sidestep_detector_fn(body, scored_qpos, route, onset), onset)
                    for onset in CONTROL_ONSETS
                }
            else:
                onset = int(ONSETS[arm])
                row["event"] = _validated_event_record(
                    event_detector_fn(body, scored_qpos, route, onset), onset)
                row["fixed_box"] = _validated_exact_box_record(
                    exact_box_fn(body, scored_qpos, predicted_centres[onset]),
                    predicted_centres[onset])
                row["sidestep"] = _validated_sidestep_record(
                    sidestep_detector_fn(body, scored_qpos, route, onset), onset)
                row.update({
                    "control_events": None,
                    "control_fixed_boxes": None,
                    "control_sidesteps": None,
                })
            rows.append(row)
            counters["trajectories_analyzed"] += 1
            persist()

        if len(rows) != len(plan) or counters["trajectories_analyzed"] != len(plan):
            raise ValueError("EXP-023b analysis did not preserve the planned denominator")
        summary = summarize_rows(rows, route, qpos_archive)
        gates = evaluate_gates(summary)
        receipt["summary"] = summary
        receipt["measurement_gates"] = gates
        receipt["decision_rule"] = evaluate_decision_rule(summary, gates)
        receipt["provenance"]["post_analysis_identity_revalidation"] = (
            revalidate_bound_identities())
        refusals = (
            ("squeeze0_substrate", "SQUEEZE-from-start substrate gate failed"),
            ("all_walk_sidestep_specificity", "all-WALK sidestep-specificity gate failed"),
            ("all_walk_step_specificity", "all-WALK step-specificity gate failed"),
        )
        for gate_name, message in refusals:
            if not gates[gate_name]["pass"]:
                receipt.update({
                    "schema": FAILURE_SCHEMA_VERSION,
                    "status": "refused",
                    "blocked": True,
                    "refusal_reason": f"{gate_name}_gate_failed",
                    "seeds_spent_and_must_not_be_reused": True,
                })
                persist()
                raise MeasurementRefusal(message)

        stage = "complete"
        receipt["provenance"]["completion_identity_revalidation"] = (
            revalidate_bound_identities())
        if any((
            counters["schedule_invocations_completed"] != len(chunks),
            counters["autoregressive_window_calls_completed"] != len(chunks) * N_WINDOWS,
            counters["trajectories_returned"] != len(plan),
            counters["trajectories_converted_to_qpos"] != len(plan),
            counters["trajectories_analyzed"] != len(plan),
        )):
            raise ValueError("EXP-023b completion accounting is not exact")
        receipt.update({
            "status": "complete",
            "complete": True,
            "blocked": False,
            "stage": stage,
            "actual_ardy_samples": len(plan),
        })
        persist()
        return receipt
    except Exception as exc:
        if isinstance(exc, MeasurementRefusal):
            persist()
            raise PromptSwitchAbort(str(exc)) from exc
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION,
            "status": "blocked",
            "complete": False,
            "blocked": True,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_ardy_samples": int(counters["trajectories_returned"]),
        })
        persist()
        if isinstance(exc, PromptSwitchAbort):
            raise
        raise PromptSwitchAbort(str(exc)) from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    host_gate_fn: Callable[[], Mapping[str, Any]] = evaluate_host_gate,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/exp023b_prompt_switch_control")
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the locked row/chunk plan and the host-gate report as JSON; write nothing",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run_report(host_gate_fn), indent=2, sort_keys=True))
        return 0
    try:
        receipt = run_campaign(
            out=args.out, cache_path=args.cache_path, host_gate_fn=host_gate_fn)
    except HostGateRefusal as exc:
        print(json.dumps({
            "status": "host_gate_failed",
            "error": str(exc),
            "host_resource_gate": _json_safe(exc.report),
            "output_untouched": True,
        }, indent=2, sort_keys=True))
        return 2
    summary = receipt["summary"]
    print(json.dumps({
        "status": receipt["status"],
        "actual_ardy_samples": receipt["actual_ardy_samples"],
        "sidestep_rates": {
            arm: summary["sidestep_rates_missing_retained"][arm]["composite"]
            for arm in PROMPT_ARMS
        },
        "step_event_rates": summary["step_event_rates_missing_retained"],
        "all_walk_sidestep_specificity": summary["all_walk_sidestep_specificity"],
        "measurement_gates": receipt["measurement_gates"],
        "decision_rule": receipt["decision_rule"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
