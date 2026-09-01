"""Neutral-WALK calibration v2 for RAMP phase observability and route timing.

This campaign is deliberately locked rather than configurable.  It spends exactly 72
frozen ARDY samples: calibration seeds 3400--3415 and validation seeds 3500--3507, each
paired across three root-XZ-only WALK speeds in nine batches of eight.  Calibration fixes
the target swing-prominence gate and route-timing derivative caps.  Validation may reject
those quantities, but can never widen or otherwise change them.

v2 supersedes the v1 campaign after its hash-anchored refusal (see
``docs/ramp-route-phase-calibration-refusal-2026-09-01.md``): calibration tolerates
measured neutral-substrate attrition instead of requiring every seed, the single fixed
event progress becomes a frozen five-point placement set with outcome-free
minimum-deformation selection, timing caps pool all three speed strata, and validation
kill rules gate only the reference stratum the E1 pilot consumes while the endpoint
strata are measured and reported descriptively.  All resizing is grounded in the v1
refusal evidence (``experiments/analyze_calibrate_ramp_route_phase_v1.py``) and frozen in
``docs/ramp-route-phase-calibration-protocol-v2.md`` before any v2 sample is generated.

The physical step/support threshold receipt is a fixed common-support dependency.  This
run is not new confirmatory evidence for those thresholds.  Likewise, the 4 cm donor-step
quality gate is carried as a separate identity and is never used to discover or accept a
neutral target swing.

All GPU-facing orchestration is dependency injectable so the call plan, accounting,
incremental evidence, and refusal paths can be tested on CPU.  The default executable path
still uses exact MuJoCo foot kinematics and the released ARDY runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene2motion.constraints import ConstraintSpec  # noqa: E402
from scene2motion.ramp.phase_observability import (  # noqa: E402
    PHASE_BACKGROUND_METHOD,
    PHASE_OBSERVABILITY_SCHEMA_VERSION,
    PHASE_PROMINENCE_METHOD,
    measure_phase_observability,
)
from scene2motion.ramp.route_phase import (  # noqa: E402
    ROUTE_PROGRESS_METHOD,
    ROUTE_PROGRESS_SCHEMA_VERSION,
    ROUTE_PROGRESS_SLOPE_POLICY,
    RouteProgressProgram,
    RouteTimingBounds,
    reparameterize_route_progress,
)
from scene2motion.robot import ARDY_G1_XML, G1Body  # noqa: E402
from scene2motion.runner import ArdyRunner  # noqa: E402
from scene2motion.stepover_eval import (  # noqa: E402
    StepOverThresholds,
    foot_kinematics_series,
)


CAMPAIGN_SCHEMA_VERSION = "ramp-route-phase-calibration-v2"
FAILURE_SCHEMA_VERSION = "ramp-route-phase-calibration-failure-v2"
PROMINENCE_RECEIPT_SCHEMA_VERSION = "ramp-target-prominence-calibration-v2"
TIMING_RECEIPT_SCHEMA_VERSION = "ramp-route-timing-calibration-v2"
DONOR_GATE_SCHEMA_VERSION = "ramp-donor-step-quality-dependency-v1"
WALK_PROMPT = "A person walks forward."
FPS = 25.0
N_FRAMES = 200
DURATION_S = (N_FRAMES - 1) / FPS
PILOT_ROUTE_LENGTH_M = 7.2
PILOT_EVENT_OBSTACLE_PROGRESS_M = 3.6
EVENT_PLACEMENTS_M = (3.0, 3.3, 3.6, 3.9, 4.2)
REFERENCE_SPEED_MPS = PILOT_ROUTE_LENGTH_M / DURATION_S
SPEEDS: tuple[tuple[str, float], ...] = (
    ("slow", 0.6),
    ("reference", REFERENCE_SPEED_MPS),
    ("fast", 1.2),
)
CALIBRATION_SEEDS = tuple(range(3400, 3416))
VALIDATION_SEEDS = tuple(range(3500, 3508))
BATCH_SIZE = 8
N_BATCHES = 9
PLANNED_SAMPLES = 72
DIFFUSION_STEPS = 5
CFG_WEIGHT = (2.0, 2.0)
NOISE_STREAM_VERSION = 2
PACKET_HALF_WINDOW_FRAMES = 2
SUPPORT_WINDOW_S = 0.24
DONOR_MIN_RELATIVE_LIFT_M = 0.04
PHYSICAL_THRESHOLD_RECEIPT_FILE_SHA256 = (
    "f6dba8be84a9d5d0b76c8114d4b93b1707bc1bb8a6fec1a26a22aa1780a6e9bf"
)
PHYSICAL_THRESHOLD_RECEIPT_NORMALIZED_SHA256 = (
    "0f6365718b6dd31aace2addca613777a6bb18bb343a1de92ed16694f6d024e7c"
)
PROMINENCE_QUANTUM_M = 0.001
ACCELERATION_QUANTUM_MPS2 = 0.01
JERK_QUANTUM_MPS3 = 0.1
ENDPOINT_SPEED_QUANTUM_MPS = 0.01
CALIBRATION_QUANTILE = 0.95
HEADROOM = 1.25
BROAD_ACCELERATION_CAP_MPS2 = 1.0e9
MIN_ROUTE_SPEED_MPS = 0.6
MAX_ROUTE_SPEED_MPS = 1.2
MIN_CALIBRATION_BACKGROUND_SEEDS = 12
MIN_TIMING_CONTRIBUTING_SEEDS = 10
VALIDATION_MAX_REFERENCE_ATTRITION = 3
VALIDATION_MAX_BACKGROUND_EXCEEDANCE = 1
VALIDATION_MIN_FULL_FEASIBLE_ABSOLUTE = 4
VALIDATION_MIN_FULL_FEASIBLE_FRACTION = 0.6
VALIDATION_MIN_SELECTED_SIGNALS_PER_SIDE = 3
VALIDATION_SIGNAL_QUANTILE = 0.25
VALIDATION_BACKGROUND_QUANTILE = 0.95
VALIDATION_SEPARATION_HEADROOM = 1.25


class CalibrationAbort(RuntimeError):
    """Fail-closed stop after durable campaign evidence has been written."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(schema: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema": str(schema), "fields": dict(fields)}
    normalized = json.loads(_canonical_json(payload))
    return {**normalized, "sha256": _json_hash(normalized)}


def _array_hash(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in sorted(arrays):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            continue
        found = True
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(_canonical_json(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    if not found:
        raise ValueError("array payload contains no ndarray values")
    return digest.hexdigest()


def _sample_hash(sample: Mapping[str, Any]) -> str:
    return _array_hash({str(name): value for name, value in sample.items()})


def _file_manifest(root: Path) -> dict[str, str]:
    """Hash every regular file reachable from one immutable model snapshot."""
    if not root.is_dir():
        raise ValueError(f"snapshot root is not a directory: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"snapshot contains no files: {root}")
    manifest: dict[str, str] = {}
    for path in files:
        digest = _sha256(path)
        if digest is None:
            raise ValueError(f"snapshot file disappeared while hashing: {path}")
        manifest[path.relative_to(root).as_posix()] = digest
    return manifest


def _locked_text_cache_identity(runner: Any, cache_path: Path) -> dict[str, Any]:
    """Bind the exact WALK embedding used in memory to its authoritative cache file."""
    if not cache_path.is_file():
        raise ValueError("locked WALK text-cache file is missing")
    prompt_cache_key = hashlib.sha1(WALK_PROMPT.encode()).hexdigest()
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if prompt_cache_key not in cache.files:
                raise ValueError("locked WALK prompt is absent from the text cache")
            file_embedding = np.array(cache[prompt_cache_key], copy=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid locked WALK text cache: {exc}") from exc
    try:
        memory_embedding = np.array(runner._text_cache[prompt_cache_key], copy=True)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "runner memory does not contain the locked WALK prompt embedding"
        ) from exc
    if (
        file_embedding.size == 0
        or memory_embedding.size == 0
        or not np.isfinite(file_embedding).all()
        or not np.isfinite(memory_embedding).all()
    ):
        raise ValueError("locked WALK prompt embedding is empty or nonfinite")
    file_digest = _array_hash({prompt_cache_key: file_embedding})
    memory_digest = _array_hash({prompt_cache_key: memory_embedding})
    if memory_digest != file_digest:
        raise ValueError(
            "runner in-memory WALK embedding does not byte-match the authoritative cache"
        )
    return {
        "path": str(cache_path),
        "sha256": _sha256(cache_path),
        "walk_prompt": WALK_PROMPT,
        "walk_prompt_cache_key_sha1": prompt_cache_key,
        "walk_prompt_embedding_content_sha256": file_digest,
        "runner_memory_embedding_content_sha256": memory_digest,
        "runner_memory_byte_matches_file": True,
        "walk_prompt_embedding_shape": list(file_embedding.shape),
        "walk_prompt_embedding_dtype": str(file_embedding.dtype),
    }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def nearest_rank(values: Iterable[float], quantile: float) -> float:
    """Nearest-rank empirical quantile with one-indexed ``ceil(q*n)`` indexing."""
    exact = np.asarray(tuple(float(value) for value in values), dtype=float)
    q = _finite(quantile, "quantile")
    if exact.ndim != 1 or len(exact) == 0 or not np.isfinite(exact).all():
        raise ValueError("nearest-rank values must be a non-empty finite sequence")
    if not 0.0 < q <= 1.0:
        raise ValueError("nearest-rank quantile must lie in (0, 1]")
    ordered = np.sort(exact)
    rank = max(1, int(math.ceil(q * len(ordered))))
    return float(ordered[rank - 1])


def ceil_outward(value: float, quantum: float, *, positive_floor: bool = False) -> float:
    """Round a nonnegative bound upward without binary-float boundary ambiguity."""
    exact = _finite(value, "value")
    step = _finite(quantum, "quantum")
    if exact < 0.0 or step <= 0.0:
        raise ValueError("outward rounding requires value >= 0 and quantum > 0")
    units = (Decimal(str(exact)) / Decimal(str(step))).to_integral_value(
        rounding=ROUND_CEILING
    )
    rounded = float(units * Decimal(str(step)))
    if positive_floor and rounded == 0.0:
        return step
    return rounded


def calibrated_upper_bound(
    values: Iterable[float], *, quantum: float, positive_floor: bool = False
) -> dict[str, Any]:
    exact = tuple(_finite(value, "calibration value") for value in values)
    empirical = nearest_rank(exact, CALIBRATION_QUANTILE)
    expanded = HEADROOM * empirical
    rounded = ceil_outward(expanded, quantum, positive_floor=positive_floor)
    return {
        "n": len(exact),
        "per_seed_maxima": list(exact),
        "nearest_rank_quantile": CALIBRATION_QUANTILE,
        "nearest_rank_value": empirical,
        "headroom": HEADROOM,
        "expanded_value": expanded,
        "rounding": "decimal-ceiling-outward",
        "quantum": quantum,
        "positive_quantum_floor": positive_floor,
        "value": rounded,
    }


@dataclass(frozen=True)
class CalibrationBatch:
    index: int
    split: str
    seed_block: int
    speed_label: str
    requested_speed_mps: float
    seeds: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        route = route_xz_for_speed(self.requested_speed_mps)
        return {
            "index": self.index,
            "split": self.split,
            "seed_block": self.seed_block,
            "speed_label": self.speed_label,
            "requested_speed_mps": self.requested_speed_mps,
            "seeds": list(self.seeds),
            "n_samples": len(self.seeds),
            "route_endpoint_m": float(route[-1, 1]),
            "root_xz_content_sha256": _array_hash({"root_xz": route}),
        }

    @property
    def digest(self) -> str:
        return _json_hash(self.as_dict())


def locked_batch_plan() -> tuple[CalibrationBatch, ...]:
    """Return the exact nine-batch order, pairing each seed block across speeds."""
    batches: list[CalibrationBatch] = []
    index = 0
    split_blocks = (
        ("calibration", tuple(CALIBRATION_SEEDS[:8]), 0),
        ("calibration", tuple(CALIBRATION_SEEDS[8:]), 1),
        ("validation", tuple(VALIDATION_SEEDS), 0),
    )
    for split, seeds, block in split_blocks:
        if len(seeds) != BATCH_SIZE:
            raise RuntimeError("locked seed block no longer has eight samples")
        for speed_label, speed in SPEEDS:
            batches.append(
                CalibrationBatch(
                    index=index,
                    split=split,
                    seed_block=block,
                    speed_label=speed_label,
                    requested_speed_mps=float(speed),
                    seeds=seeds,
                )
            )
            index += 1
    if len(batches) != N_BATCHES or sum(len(batch.seeds) for batch in batches) != PLANNED_SAMPLES:
        raise RuntimeError("locked batch plan no longer contains nine batches / 72 samples")
    return tuple(batches)


def route_xz_for_speed(speed_mps: float) -> np.ndarray:
    speed = _finite(speed_mps, "speed_mps")
    if speed <= 0.0:
        raise ValueError("speed_mps must be positive")
    return np.stack(
        [
            np.zeros(N_FRAMES, dtype=float),
            np.linspace(0.0, speed * DURATION_S, N_FRAMES, dtype=float),
        ],
        axis=-1,
    )


def root_only_walk_spec(speed_mps: float) -> ConstraintSpec:
    return ConstraintSpec(
        root_xz=route_xz_for_speed(speed_mps),
        heading=None,
        root_y=None,
        first_heading=0.0,
    )


@dataclass(frozen=True)
class ObservedCycle:
    split: str
    seed: int
    speed_label: str
    requested_speed_mps: float
    swing_side: str
    takeoff_frame: int
    apex_frame: int
    landing_frame: int
    prominence_m: float
    background_contrasts_m: tuple[float, ...]
    background_window_identities: tuple[Mapping[str, Any], ...]
    nominal_foot_forward_offset_m: float
    evidence_digest: str
    phase_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.split not in ("calibration", "validation"):
            raise ValueError("cycle split must be calibration or validation")
        object.__setattr__(self, "seed", _strict_int(self.seed, "seed"))
        if self.speed_label not in {label for label, _ in SPEEDS}:
            raise ValueError("cycle speed label is not in the locked design")
        object.__setattr__(
            self,
            "requested_speed_mps",
            _finite(self.requested_speed_mps, "requested_speed_mps"),
        )
        if self.swing_side not in ("left", "right"):
            raise ValueError("swing_side must be left or right")
        for name in ("takeoff_frame", "apex_frame", "landing_frame"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if not 0 <= self.takeoff_frame <= self.apex_frame <= self.landing_frame < N_FRAMES:
            raise ValueError("cycle frames must be ordered within the locked clip")
        for name in ("prominence_m", "nominal_foot_forward_offset_m"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        backgrounds = tuple(
            _finite(value, "background contrast") for value in self.background_contrasts_m
        )
        if len(backgrounds) != 2 or any(value < 0.0 for value in backgrounds):
            raise ValueError("each observed cycle requires two nonnegative null contrasts")
        object.__setattr__(self, "background_contrasts_m", backgrounds)
        identities = tuple(
            json.loads(_canonical_json(dict(value)))
            for value in self.background_window_identities
        )
        if len(identities) != len(backgrounds):
            raise ValueError("every null contrast requires one physical-window identity")
        required = {"baseline_support_side", "window_start_frame", "window_end_frame", "method"}
        if any(set(identity) != required for identity in identities):
            raise ValueError("background window identity has unexpected fields")
        if any(
            identity["baseline_support_side"] not in ("left", "right")
            or not isinstance(identity["window_start_frame"], int)
            or not isinstance(identity["window_end_frame"], int)
            or identity["window_end_frame"] < identity["window_start_frame"]
            or not isinstance(identity["method"], str)
            or not identity["method"]
            for identity in identities
        ):
            raise ValueError("background window identity is invalid")
        object.__setattr__(self, "background_window_identities", identities)
        if not _is_sha256(self.evidence_digest):
            raise ValueError("evidence_digest must be a lowercase SHA-256 digest")
        detached = json.loads(_canonical_json(dict(self.phase_evidence)))
        object.__setattr__(self, "phase_evidence", detached)

    @property
    def packet_window_valid(self) -> bool:
        return (
            self.apex_frame - PACKET_HALF_WINDOW_FRAMES >= self.takeoff_frame
            and self.apex_frame + PACKET_HALF_WINDOW_FRAMES <= self.landing_frame
        )

    @property
    def max_background_contrast_m(self) -> float:
        return max(self.background_contrasts_m)

    def event_root_progress_m(self, placement_m: float) -> float:
        placement = _finite(placement_m, "placement_m")
        if placement not in EVENT_PLACEMENTS_M:
            raise ValueError("placement_m is not in the frozen event placement set")
        return placement - self.nominal_foot_forward_offset_m

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seed": self.seed,
            "speed_label": self.speed_label,
            "requested_speed_mps": self.requested_speed_mps,
            "swing_side": self.swing_side,
            "takeoff_frame": self.takeoff_frame,
            "apex_frame": self.apex_frame,
            "landing_frame": self.landing_frame,
            "prominence_m": self.prominence_m,
            "background_contrasts_m": list(self.background_contrasts_m),
            "background_window_identities": list(self.background_window_identities),
            "background_window_identity_sha256": [
                _json_hash(identity) for identity in self.background_window_identities
            ],
            "max_background_contrast_m": self.max_background_contrast_m,
            "packet_half_window_frames": PACKET_HALF_WINDOW_FRAMES,
            "packet_window_valid": self.packet_window_valid,
            "nominal_foot_forward_offset_m": self.nominal_foot_forward_offset_m,
            "event_root_progress_equation": (
                "event_root_progress_m = placement_m "
                "- nominal_foot_forward_offset_m"
            ),
            "event_placements_m": list(EVENT_PLACEMENTS_M),
            "event_root_progress_by_placement_m": {
                f"{placement:g}": self.event_root_progress_m(placement)
                for placement in EVENT_PLACEMENTS_M
            },
            "evidence_digest": self.evidence_digest,
            "phase_evidence": self.phase_evidence,
        }


@dataclass(frozen=True)
class AnalyzedClip:
    split: str
    seed: int
    speed_label: str
    requested_speed_mps: float
    sample_sha256: str
    qpos_content_sha256: str
    qpos_archive_key: str
    cycles: tuple[ObservedCycle, ...]
    measurement_rejection: str | None = None

    def __post_init__(self) -> None:
        for name in ("sample_sha256", "qpos_content_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        object.__setattr__(self, "cycles", tuple(self.cycles))
        if not self.cycles and not self.measurement_rejection:
            raise ValueError("a clip without measured cycles must preserve its rejection")
        for cycle in self.cycles:
            if (
                cycle.split != self.split
                or cycle.seed != self.seed
                or cycle.speed_label != self.speed_label
            ):
                raise ValueError("observed cycle does not belong to its clip")

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seed": self.seed,
            "speed_label": self.speed_label,
            "requested_speed_mps": self.requested_speed_mps,
            "sample_sha256": self.sample_sha256,
            "qpos_content_sha256": self.qpos_content_sha256,
            "qpos_archive_key": self.qpos_archive_key,
            "n_cycles": len(self.cycles),
            "measurement_rejection": self.measurement_rejection,
            "cycles": [cycle.as_dict() for cycle in self.cycles],
        }


def analyze_generated_clip(
    *,
    body: Any,
    qpos: np.ndarray,
    sample: Mapping[str, Any],
    split: str,
    seed: int,
    speed_label: str,
    requested_speed_mps: float,
    thresholds: StepOverThresholds,
    qpos_archive_key: str,
) -> AnalyzedClip:
    """Measure exact physical-foot phase evidence for one generated neutral WALK."""
    exact_qpos = np.asarray(qpos, dtype=float)
    if exact_qpos.ndim != 2 or exact_qpos.shape[0] != N_FRAMES:
        raise ValueError(f"qpos must have shape ({N_FRAMES}, nq)")
    if not np.isfinite(exact_qpos).all():
        raise ValueError("qpos contains non-finite values")
    sample_digest = _sample_hash(sample)
    qpos_digest = _array_hash({qpos_archive_key: exact_qpos})
    try:
        kinematics = foot_kinematics_series(body, exact_qpos, FPS)
        evidence = measure_phase_observability(
            kinematics,
            fps=FPS,
            thresholds=thresholds,
            support_window_s=SUPPORT_WINDOW_S,
        )
    except ValueError as exc:
        return AnalyzedClip(
            split=split,
            seed=seed,
            speed_label=speed_label,
            requested_speed_mps=requested_speed_mps,
            sample_sha256=sample_digest,
            qpos_content_sha256=qpos_digest,
            qpos_archive_key=qpos_archive_key,
            cycles=(),
            measurement_rejection=f"{type(exc).__name__}: {exc}",
        )

    cycles: list[ObservedCycle] = []
    for item in evidence:
        receipt = item.prominence
        cycle = receipt.cycle
        apex = int(cycle.apex_frame)
        side = str(cycle.swing_side)
        forward = np.asarray(kinematics[side]["forward_representative_m"], dtype=float)
        if forward.shape != (N_FRAMES,) or not np.isfinite(forward).all():
            raise ValueError("exact foot forward representative is invalid")
        foot_offset = float(forward[apex] - exact_qpos[apex, 0])
        backgrounds = tuple(float(contrast.contrast_m) for contrast in item.background_contrasts)
        background_identities = tuple(
            {
                "baseline_support_side": str(contrast.baseline_support_side),
                "window_start_frame": int(contrast.window_start_frame),
                "window_end_frame": int(contrast.window_end_frame),
                "method": str(contrast.method),
            }
            for contrast in item.background_contrasts
        )
        cycles.append(
            ObservedCycle(
                split=split,
                seed=seed,
                speed_label=speed_label,
                requested_speed_mps=requested_speed_mps,
                swing_side=side,
                takeoff_frame=int(cycle.landmarks.takeoff_frame),
                apex_frame=apex,
                landing_frame=int(cycle.landmarks.landing_frame),
                prominence_m=float(receipt.prominence_m),
                background_contrasts_m=backgrounds,
                background_window_identities=background_identities,
                nominal_foot_forward_offset_m=foot_offset,
                evidence_digest=item.receipt_digest,
                phase_evidence=item.as_dict(),
            )
        )
    return AnalyzedClip(
        split=split,
        seed=seed,
        speed_label=speed_label,
        requested_speed_mps=requested_speed_mps,
        sample_sha256=sample_digest,
        qpos_content_sha256=qpos_digest,
        qpos_archive_key=qpos_archive_key,
        cycles=tuple(cycles),
    )


def deduplicated_clip_backgrounds(
    clip: AnalyzedClip,
) -> tuple[dict[str, Any], ...]:
    """Deduplicate nulls by their physical support-window identity within one clip."""
    unique: dict[str, dict[str, Any]] = {}
    raw_count = 0
    for cycle in clip.cycles:
        for identity, contrast in zip(
            cycle.background_window_identities, cycle.background_contrasts_m
        ):
            raw_count += 1
            digest = _json_hash(identity)
            row = {
                "identity": dict(identity),
                "identity_sha256": digest,
                "contrast_m": float(contrast),
            }
            previous = unique.get(digest)
            if previous is not None and not np.isclose(
                previous["contrast_m"], contrast, atol=1e-12, rtol=0.0
            ):
                raise ValueError(
                    "one physical background window has conflicting contrast values"
                )
            unique[digest] = row
    return tuple(unique[digest] for digest in sorted(unique))


def background_deduplication_receipt(
    clips: Sequence[AnalyzedClip], *, split: str | None = None
) -> dict[str, Any]:
    selected = [clip for clip in clips if split is None or clip.split == split]
    rows: list[dict[str, Any]] = []
    for clip in selected:
        unique = deduplicated_clip_backgrounds(clip)
        raw_count = sum(len(cycle.background_contrasts_m) for cycle in clip.cycles)
        rows.append(
            {
                "split": clip.split,
                "seed": clip.seed,
                "speed_label": clip.speed_label,
                "raw_count": raw_count,
                "unique_count": len(unique),
                "duplicate_count": raw_count - len(unique),
                "unique_windows": list(unique),
            }
        )
    return {
        "deduplication_scope": "within each generated clip",
        "physical_window_identity": [
            "baseline_support_side",
            "window_start_frame",
            "window_end_frame",
            "method",
        ],
        "prominence_receipt_digest_is_not_part_of_null_identity": True,
        "raw_count": sum(row["raw_count"] for row in rows),
        "unique_count": sum(row["unique_count"] for row in rows),
        "duplicate_count": sum(row["duplicate_count"] for row in rows),
        "clips": rows,
    }


def calibration_seed_background_maxima(
    clips: Sequence[AnalyzedClip], *, split: str
) -> dict[int, float]:
    seeds = CALIBRATION_SEEDS if split == "calibration" else VALIDATION_SEEDS
    result: dict[int, float] = {}
    for seed in seeds:
        values = [
            float(background["contrast_m"])
            for clip in clips
            if clip.split == split and clip.seed == seed
            for background in deduplicated_clip_backgrounds(clip)
        ]
        if values:
            result[int(seed)] = max(values)
    return result


def freeze_target_prominence(clips: Sequence[AnalyzedClip]) -> dict[str, Any]:
    maxima = calibration_seed_background_maxima(clips, split="calibration")
    missing = [seed for seed in CALIBRATION_SEEDS if seed not in maxima]
    present = [seed for seed in CALIBRATION_SEEDS if seed in maxima]
    if len(present) < MIN_CALIBRATION_BACKGROUND_SEEDS:
        raise ValueError(
            "target-prominence calibration has background evidence for only "
            f"{len(present)}/{len(CALIBRATION_SEEDS)} seeds; requires "
            f">={MIN_CALIBRATION_BACKGROUND_SEEDS} (missing {missing})"
        )
    ordered = [maxima[seed] for seed in present]
    bound = calibrated_upper_bound(ordered, quantum=PROMINENCE_QUANTUM_M)
    if bound["value"] <= 0.0:
        raise ValueError(
            "target-prominence calibration is degenerate: exact rounded Pmin is zero"
        )
    fields = {
        "role": "target-neutral-swing-observability-gate",
        "not_a_donor_quality_gate": True,
        "phase_observability_schema_version": PHASE_OBSERVABILITY_SCHEMA_VERSION,
        "signal_method": PHASE_PROMINENCE_METHOD,
        "background_method": PHASE_BACKGROUND_METHOD,
        "background_reduction": (
            "per-seed maximum across physical-window-deduplicated own-foot nulls and all "
            "three speeds"
        ),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "seeds_with_background_evidence": present,
        "seeds_without_background_evidence": missing,
        "min_required_background_seeds": MIN_CALIBRATION_BACKGROUND_SEEDS,
        "missing_seeds_are_recorded_attrition_not_imputed": True,
        "per_seed_background_maxima_m": {
            str(seed): maxima[seed] for seed in present
        },
        "nearest_rank_quantile": CALIBRATION_QUANTILE,
        "nearest_rank_q95_m": bound["nearest_rank_value"],
        "headroom": HEADROOM,
        "rounding": "ceil-to-1mm-decimal-outward",
        "rounding_quantum_m": PROMINENCE_QUANTUM_M,
        "target_min_prominence_m": bound["value"],
        "calibration_details": bound,
        "background_null_deduplication": background_deduplication_receipt(
            clips, split="calibration"
        ),
    }
    return _identity(PROMINENCE_RECEIPT_SCHEMA_VERSION, fields)


@dataclass(frozen=True)
class ProgramCandidate:
    cycle: ObservedCycle
    placement_m: float
    program: RouteProgressProgram | None
    rejection: str | None

    def __post_init__(self) -> None:
        placement = _finite(self.placement_m, "placement_m")
        if placement not in EVENT_PLACEMENTS_M:
            raise ValueError("placement_m is not in the frozen event placement set")
        object.__setattr__(self, "placement_m", placement)

    @property
    def feasible(self) -> bool:
        return self.program is not None and self.rejection is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle.as_dict(),
            "placement_m": self.placement_m,
            "status": "feasible" if self.feasible else "rejected",
            "rejection": self.rejection,
            "program": None if self.program is None else self.program.diagnostics(),
        }

    def compact_dict(self) -> dict[str, Any]:
        """Candidate row without the embedded cycle evidence (rows.jsonl carries it)."""
        cycle = self.cycle
        return {
            "split": cycle.split,
            "seed": cycle.seed,
            "speed_label": cycle.speed_label,
            "swing_side": cycle.swing_side,
            "apex_frame": cycle.apex_frame,
            "prominence_m": cycle.prominence_m,
            "cycle_evidence_digest": cycle.evidence_digest,
            "placement_m": self.placement_m,
            "status": "feasible" if self.feasible else "rejected",
            "rejection": self.rejection,
            "program_digest": None if self.program is None else self.program.digest(),
        }


def placement_selection_key(
    candidate: ProgramCandidate,
) -> tuple[float, float, float, float, float, str]:
    """Outcome-free lexicographic key: deformation cost, placement ties, digest."""
    if not candidate.feasible or candidate.program is None:
        raise ValueError("selection key requires a feasible program candidate")
    return (
        *candidate.program.selection_cost,
        abs(candidate.placement_m - PILOT_EVENT_OBSTACLE_PROGRESS_M),
        candidate.placement_m,
        candidate.cycle.evidence_digest,
    )


def broad_timing_bounds() -> RouteTimingBounds:
    return RouteTimingBounds(
        fps=FPS,
        min_discrete_route_progress_speed_mps=MIN_ROUTE_SPEED_MPS,
        max_discrete_route_progress_speed_mps=MAX_ROUTE_SPEED_MPS,
        max_abs_route_progress_acceleration_mps2=BROAD_ACCELERATION_CAP_MPS2,
        max_abs_discrete_route_progress_jerk_mps3=None,
        reference_route_progress_speed_mps=REFERENCE_SPEED_MPS,
        max_endpoint_route_progress_speed_deviation_mps=None,
    )


def make_program_candidate(
    cycle: ObservedCycle, placement_m: float, *, target_min_prominence_m: float,
    timing_bounds: RouteTimingBounds | None = None,
) -> ProgramCandidate:
    pmin = _finite(target_min_prominence_m, "target_min_prominence_m")
    if cycle.prominence_m < pmin:
        return ProgramCandidate(
            cycle, placement_m, None, "prominence_below_frozen_target_gate"
        )
    if not cycle.packet_window_valid:
        return ProgramCandidate(
            cycle, placement_m, None, "packet_half_window_two_not_supported"
        )
    try:
        program = reparameterize_route_progress(
            np.asarray([[0.0, 0.0], [0.0, PILOT_ROUTE_LENGTH_M]], dtype=float),
            n_frames=N_FRAMES,
            event_frame=cycle.apex_frame,
            event_root_progress_m=cycle.event_root_progress_m(placement_m),
            timing_bounds=timing_bounds or broad_timing_bounds(),
        )
    except (TypeError, ValueError) as exc:
        return ProgramCandidate(
            cycle, placement_m, None, f"{type(exc).__name__}: {exc}"
        )
    return ProgramCandidate(cycle, placement_m, program, None)


def build_and_select_programs(
    clips: Sequence[AnalyzedClip], *, target_min_prominence_m: float,
    timing_bounds: RouteTimingBounds | None = None,
) -> tuple[tuple[ProgramCandidate, ...], tuple[ProgramCandidate, ...]]:
    """Enumerate every (valid cycle x placement) program; select one per seed/speed/side."""
    candidates = tuple(
        make_program_candidate(
            cycle,
            placement,
            target_min_prominence_m=target_min_prominence_m,
            timing_bounds=timing_bounds,
        )
        for clip in clips
        for cycle in clip.cycles
        for placement in EVENT_PLACEMENTS_M
    )
    grouped: dict[tuple[str, int, str, str], list[ProgramCandidate]] = {}
    for candidate in candidates:
        if candidate.feasible:
            cycle = candidate.cycle
            grouped.setdefault(
                (cycle.split, cycle.seed, cycle.speed_label, cycle.swing_side), []
            ).append(candidate)
    selected: list[ProgramCandidate] = []
    for key in sorted(grouped):
        selected.append(min(grouped[key], key=placement_selection_key))
    return candidates, tuple(selected)


def freeze_timing_bounds(selected: Sequence[ProgramCandidate]) -> dict[str, Any]:
    calibration = [
        candidate
        for candidate in selected
        if candidate.feasible and candidate.cycle.split == "calibration"
    ]
    per_seed: dict[int, dict[str, float]] = {}
    for seed in CALIBRATION_SEEDS:
        programs = [item.program for item in calibration if item.cycle.seed == seed]
        programs = [program for program in programs if program is not None]
        if not programs:
            continue
        if any(
            program.max_abs_discrete_route_progress_jerk_mps3 is None
            or program.endpoint_route_progress_speed_deviation_mps is None
            for program in programs
        ):
            raise ValueError("broad route program omitted required timing diagnostics")
        per_seed[seed] = {
            "max_abs_scalar_progress_acceleration_mps2": max(
                max(
                    program.max_abs_continuous_route_progress_acceleration_mps2,
                    program.max_abs_discrete_route_progress_acceleration_mps2,
                )
                for program in programs
            ),
            "max_abs_fps_discrete_scalar_progress_jerk_mps3": max(
                float(program.max_abs_discrete_route_progress_jerk_mps3)
                for program in programs
            ),
            "max_endpoint_scalar_progress_speed_deviation_mps": max(
                float(program.endpoint_route_progress_speed_deviation_mps)
                for program in programs
            ),
        }
    contributing = sorted(per_seed)
    missing = [seed for seed in CALIBRATION_SEEDS if seed not in per_seed]
    if len(contributing) < MIN_TIMING_CONTRIBUTING_SEEDS:
        raise ValueError(
            "route-timing calibration has selected programs for only "
            f"{len(contributing)}/{len(CALIBRATION_SEEDS)} seeds; requires "
            f">={MIN_TIMING_CONTRIBUTING_SEEDS} (missing {missing})"
        )
    acceleration = calibrated_upper_bound(
        [per_seed[seed]["max_abs_scalar_progress_acceleration_mps2"] for seed in contributing],
        quantum=ACCELERATION_QUANTUM_MPS2,
        positive_floor=True,
    )
    jerk = calibrated_upper_bound(
        [per_seed[seed]["max_abs_fps_discrete_scalar_progress_jerk_mps3"] for seed in contributing],
        quantum=JERK_QUANTUM_MPS3,
        positive_floor=True,
    )
    endpoint = calibrated_upper_bound(
        [per_seed[seed]["max_endpoint_scalar_progress_speed_deviation_mps"] for seed in contributing],
        quantum=ENDPOINT_SPEED_QUANTUM_MPS,
    )
    bounds = RouteTimingBounds(
        fps=FPS,
        min_discrete_route_progress_speed_mps=MIN_ROUTE_SPEED_MPS,
        max_discrete_route_progress_speed_mps=MAX_ROUTE_SPEED_MPS,
        max_abs_route_progress_acceleration_mps2=acceleration["value"],
        max_abs_discrete_route_progress_jerk_mps3=jerk["value"],
        reference_route_progress_speed_mps=REFERENCE_SPEED_MPS,
        max_endpoint_route_progress_speed_deviation_mps=endpoint["value"],
    )
    fields = {
        "role": "fixed-route-timing-bounds",
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "contributing_seeds": contributing,
        "seeds_without_selected_program": missing,
        "min_required_contributing_seeds": MIN_TIMING_CONTRIBUTING_SEEDS,
        "pooled_speed_strata": [label for label, _ in SPEEDS],
        "selection": (
            "minimum route deformation per seed/speed/side over the frozen placement "
            "set; placement then receipt digest tie break"
        ),
        "selection_is_outcome_free": True,
        "event_placements_m": list(EVENT_PLACEMENTS_M),
        "per_seed_maxima": {str(seed): per_seed[seed] for seed in contributing},
        "speed_envelope_mps": [MIN_ROUTE_SPEED_MPS, MAX_ROUTE_SPEED_MPS],
        "reference_speed_mps": REFERENCE_SPEED_MPS,
        "acceleration": acceleration,
        "jerk": jerk,
        "endpoint_speed_deviation": endpoint,
        "route_timing_bounds": bounds.as_dict(),
        "route_timing_bounds_digest": bounds.digest(),
        "derivative_semantics": {
            "acceleration": (
                "maximum of analytic continuous and fps-discrete scalar "
                "route-progress acceleration"
            ),
            "c1_event": (
                "scalar route progress and speed are C1; scalar acceleration may jump "
                "at the event anchor"
            ),
            "jerk": (
                "fps-discrete scalar route-progress jerk; no claim of continuous jerk "
                "through the C1 event"
            ),
            "endpoint": "analytic scalar progress endpoint-slope deviation",
        },
    }
    return _identity(TIMING_RECEIPT_SCHEMA_VERSION, fields)


def timing_bounds_from_receipt(receipt: Mapping[str, Any]) -> RouteTimingBounds:
    try:
        if receipt["schema"] != TIMING_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unexpected timing receipt schema")
        if receipt["sha256"] != _json_hash(
            {"schema": receipt["schema"], "fields": receipt["fields"]}
        ):
            raise ValueError("timing receipt digest mismatch")
        return RouteTimingBounds(**receipt["fields"]["route_timing_bounds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid frozen timing receipt: {exc}") from exc


def _deduplicated_seed_speed_backgrounds(
    clips: Sequence[AnalyzedClip], *, split: str, seed: int, speed_label: str
) -> tuple[dict[str, Any], ...]:
    return tuple(
        background
        for clip in clips
        if clip.split == split
        and clip.seed == seed
        and clip.speed_label == speed_label
        for background in deduplicated_clip_backgrounds(clip)
    )


def validation_reference_separation(
    clips: Sequence[AnalyzedClip],
    selected_frozen: Sequence[ProgramCandidate],
    *,
    target_min_prominence_m: float,
    non_attrited_seeds: Sequence[int],
) -> dict[str, Any]:
    """Per-side robust separation of route-selected signal over own-foot background."""
    side_results: dict[str, Any] = {}
    for side in ("left", "right"):
        background_by_seed: dict[int, float] = {}
        signal_by_seed: dict[int, float] = {}
        for seed in non_attrited_seeds:
            side_backgrounds = [
                float(background["contrast_m"])
                for background in _deduplicated_seed_speed_backgrounds(
                    clips, split="validation", seed=seed, speed_label="reference"
                )
                if background["identity"]["baseline_support_side"] == side
            ]
            if side_backgrounds:
                background_by_seed[seed] = max(side_backgrounds)
            selected_signal = [
                item.cycle.prominence_m
                for item in selected_frozen
                if item.feasible
                and item.cycle.split == "validation"
                and item.cycle.seed == seed
                and item.cycle.speed_label == "reference"
                and item.cycle.swing_side == side
            ]
            if selected_signal:
                if len(selected_signal) != 1:
                    raise ValueError(
                        "validation separation received multiple selected programs for "
                        "one seed/side"
                    )
                signal_by_seed[seed] = selected_signal[0]
        n_signals = len(signal_by_seed)
        if n_signals < VALIDATION_MIN_SELECTED_SIGNALS_PER_SIDE or not background_by_seed:
            side_results[side] = {
                "passed": False,
                "reason": (
                    f"side has {n_signals} selected signals "
                    f"(requires >={VALIDATION_MIN_SELECTED_SIGNALS_PER_SIDE}) or no "
                    "side-specific background evidence"
                ),
                "n_selected_signals": n_signals,
                "background_by_seed_m": {
                    str(k): v for k, v in background_by_seed.items()
                },
                "selected_signal_by_seed_m": {
                    str(k): v for k, v in signal_by_seed.items()
                },
            }
            continue
        background_q95 = nearest_rank(
            background_by_seed.values(), VALIDATION_BACKGROUND_QUANTILE
        )
        signal_q25 = nearest_rank(
            signal_by_seed.values(), VALIDATION_SIGNAL_QUANTILE
        )
        required = max(
            target_min_prominence_m,
            VALIDATION_SEPARATION_HEADROOM * background_q95,
        )
        side_results[side] = {
            "n_selected_signals": n_signals,
            "background_by_seed_m": {
                str(k): v for k, v in background_by_seed.items()
            },
            "selected_signal_by_seed_m": {
                str(k): v for k, v in signal_by_seed.items()
            },
            "background_q95_m": background_q95,
            "selected_signal_q25_m": signal_q25,
            "required_signal_floor_m": required,
            "passed": signal_q25 >= required,
        }
    return {
        "rule": (
            "for each swing side, nearest-rank Q25 of the outcome-free route-selected "
            "per-seed reference prominence must be >= max(frozen Pmin, 1.25 * "
            "nearest-rank Q95 of the same-side per-seed maximum "
            "physical-window-deduplicated reference background), with at least "
            f"{VALIDATION_MIN_SELECTED_SIGNALS_PER_SIDE} selected signals per side"
        ),
        "unused_high_prominence_cycles_cannot_satisfy_separation": True,
        "signal_quantile": VALIDATION_SIGNAL_QUANTILE,
        "background_quantile": VALIDATION_BACKGROUND_QUANTILE,
        "separation_headroom": VALIDATION_SEPARATION_HEADROOM,
        "side_results": side_results,
        "passed": all(result.get("passed", False) for result in side_results.values()),
    }


def _descriptive_stratum(
    clips: Sequence[AnalyzedClip],
    *,
    speed_label: str,
    target_min_prominence_m: float,
    frozen_bounds: RouteTimingBounds,
) -> dict[str, Any]:
    """Measured, reported, never gated: endpoint-stratum coverage quantities."""
    stratum_clips = [
        clip
        for clip in clips
        if clip.split == "validation" and clip.speed_label == speed_label
    ]
    attrited = sorted(
        seed
        for seed in VALIDATION_SEEDS
        if not any(clip.seed == seed and clip.cycles for clip in stratum_clips)
    )
    broad_candidates, broad_selected = build_and_select_programs(
        stratum_clips, target_min_prominence_m=target_min_prominence_m
    )
    frozen_candidates, frozen_selected = build_and_select_programs(
        stratum_clips,
        target_min_prominence_m=target_min_prominence_m,
        timing_bounds=frozen_bounds,
    )
    return {
        "gated": False,
        "speed_label": speed_label,
        "attrited_seeds": attrited,
        "n_attrited": len(attrited),
        "complete_seeds": sorted(
            {
                clip.seed
                for clip in stratum_clips
                for cycle in clip.cycles
                if cycle.packet_window_valid
                and cycle.prominence_m >= target_min_prominence_m
            }
        ),
        "broad_feasible_seeds": sorted(
            {item.cycle.seed for item in broad_selected if item.feasible}
        ),
        "full_frozen_feasible_seeds": sorted(
            {item.cycle.seed for item in frozen_selected if item.feasible}
        ),
        "selected_placement_usage_m": {
            f"{placement:g}": sum(
                1 for item in frozen_selected if item.placement_m == placement
            )
            for placement in EVENT_PLACEMENTS_M
        },
        "n_broad_candidates": len(broad_candidates),
        "n_frozen_candidates": len(frozen_candidates),
        "background_null_deduplication": background_deduplication_receipt(stratum_clips),
    }


def validate_frozen_calibration(
    clips: Sequence[AnalyzedClip],
    *,
    target_min_prominence_m: float,
    frozen_bounds: RouteTimingBounds,
) -> dict[str, Any]:
    """Apply the reference-stratum kill rules; measure endpoint strata descriptively."""
    reasons: list[str] = []
    reference_clips = [
        clip
        for clip in clips
        if clip.split == "validation" and clip.speed_label == "reference"
    ]
    attrited = sorted(
        seed
        for seed in VALIDATION_SEEDS
        if not any(clip.seed == seed and clip.cycles for clip in reference_clips)
    )
    non_attrited = [seed for seed in VALIDATION_SEEDS if seed not in attrited]
    n_ok = len(non_attrited)
    if len(attrited) > VALIDATION_MAX_REFERENCE_ATTRITION:
        reasons.append(
            f"reference: substrate attrition {len(attrited)}/{len(VALIDATION_SEEDS)} > "
            f"{VALIDATION_MAX_REFERENCE_ATTRITION}/{len(VALIDATION_SEEDS)}"
        )

    background: dict[int, float] = {}
    for seed in non_attrited:
        values = [
            float(item["contrast_m"])
            for item in _deduplicated_seed_speed_backgrounds(
                clips, split="validation", seed=seed, speed_label="reference"
            )
        ]
        if values:
            background[seed] = max(values)
    background_exceed_or_missing = [
        seed
        for seed in non_attrited
        if seed not in background or background[seed] > target_min_prominence_m
    ]
    if len(background_exceed_or_missing) > VALIDATION_MAX_BACKGROUND_EXCEEDANCE:
        reasons.append(
            "reference: background exceedance/missing count "
            f"{len(background_exceed_or_missing)} > "
            f"{VALIDATION_MAX_BACKGROUND_EXCEEDANCE} among non-attrited seeds"
        )

    frozen_candidates, frozen_selected = build_and_select_programs(
        reference_clips,
        target_min_prominence_m=target_min_prominence_m,
        timing_bounds=frozen_bounds,
    )
    full_feasible = sorted(
        {item.cycle.seed for item in frozen_selected if item.feasible}
    )
    required_full = max(
        VALIDATION_MIN_FULL_FEASIBLE_ABSOLUTE,
        int(math.ceil(VALIDATION_MIN_FULL_FEASIBLE_FRACTION * n_ok)),
    )
    if len(full_feasible) < required_full:
        reasons.append(
            f"reference: full-frozen feasible coverage {len(full_feasible)}/{n_ok} < "
            f"{required_full}"
        )

    separation = validation_reference_separation(
        clips,
        frozen_selected,
        target_min_prominence_m=target_min_prominence_m,
        non_attrited_seeds=non_attrited,
    )
    if not separation.get("passed", False):
        reasons.append("reference: locked robust quantile separation failed")

    descriptive = {
        speed_label: _descriptive_stratum(
            clips,
            speed_label=speed_label,
            target_min_prominence_m=target_min_prominence_m,
            frozen_bounds=frozen_bounds,
        )
        for speed_label in ("slow", "fast")
    }

    return {
        "validation_cannot_widen_calibration": True,
        "gated_stratum": "reference",
        "frozen_target_min_prominence_m": target_min_prominence_m,
        "frozen_route_timing_bounds": frozen_bounds.as_dict(),
        "kill_rules": {
            "attrition": (
                f"at most {VALIDATION_MAX_REFERENCE_ATTRITION}/"
                f"{len(VALIDATION_SEEDS)} reference validation seeds with zero "
                "measured complete cycles"
            ),
            "background": (
                f"at most {VALIDATION_MAX_BACKGROUND_EXCEEDANCE} non-attrited seed "
                "with missing reference background or background above frozen Pmin"
            ),
            "full_frozen_feasibility": (
                "full-feasible seeds >= max("
                f"{VALIDATION_MIN_FULL_FEASIBLE_ABSOLUTE}, "
                f"ceil({VALIDATION_MIN_FULL_FEASIBLE_FRACTION} x non-attrited))"
            ),
            "side_evidence_and_separation": separation["rule"],
        },
        "reference": {
            "attrited_seeds": attrited,
            "n_attrited": len(attrited),
            "non_attrited_seeds": non_attrited,
            "background_by_seed_m": {str(k): v for k, v in background.items()},
            "background_exceed_or_missing_seeds": background_exceed_or_missing,
            "full_frozen_feasible_seeds": full_feasible,
            "required_full_frozen_feasible": required_full,
            "selected_programs": [item.as_dict() for item in frozen_selected],
            "selected_placement_usage_m": {
                f"{placement:g}": sum(
                    1 for item in frozen_selected if item.placement_m == placement
                )
                for placement in EVENT_PLACEMENTS_M
            },
            "n_frozen_candidates": len(frozen_candidates),
            "separation": separation,
            "background_null_deduplication": background_deduplication_receipt(
                reference_clips
            ),
        },
        "descriptive_strata": descriptive,
        "passed": not reasons,
        "kill_reasons": reasons,
    }


def donor_quality_dependency() -> dict[str, Any]:
    return _identity(
        DONOR_GATE_SCHEMA_VERSION,
        {
            "role": "adapted-donor-step-quality-only",
            "min_relative_lift_m": DONOR_MIN_RELATIVE_LIFT_M,
            "not_used_for_neutral_target_swing_discovery": True,
            "not_used_for_target_prominence_calibration": True,
        },
    )


def campaign_design() -> dict[str, Any]:
    batches = locked_batch_plan()
    separation_rule = {
        "signal_unit": (
            "per-side, per-seed reference prominence of the outcome-free "
            "minimum-deformation route-selected program under frozen bounds"
        ),
        "background_unit": (
            "same-side per-seed maximum physical-window-deduplicated own-foot "
            "reference null over non-attrited seeds"
        ),
        "signal_quantile": VALIDATION_SIGNAL_QUANTILE,
        "background_quantile": VALIDATION_BACKGROUND_QUANTILE,
        "headroom": VALIDATION_SEPARATION_HEADROOM,
        "min_selected_signals_per_side": VALIDATION_MIN_SELECTED_SIGNALS_PER_SIDE,
        "pass": (
            "for each side, selected_signal_Q25 >= "
            "max(frozen_Pmin, 1.25*same_side_background_Q95)"
        ),
        "unused_cycles_cannot_supply_signal": True,
    }
    fields = {
        "prompt": WALK_PROMPT,
        "constraint": "neutral-root-xz-only-walk",
        "fps": FPS,
        "n_frames": N_FRAMES,
        "duration_definition": "(n_frames-1)/fps",
        "duration_s": DURATION_S,
        "diffusion_steps": DIFFUSION_STEPS,
        "cfg_weight": list(CFG_WEIGHT),
        "noise_stream_version": NOISE_STREAM_VERSION,
        "speeds_mps": {label: speed for label, speed in SPEEDS},
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "same_seed_paired_across_speeds": True,
        "batches": [batch.as_dict() for batch in batches],
        "n_batches": N_BATCHES,
        "batch_size": BATCH_SIZE,
        "planned_samples": PLANNED_SAMPLES,
        "phase_discovery_min_relative_lift_m": 0.0,
        "packet_half_window_frames": PACKET_HALF_WINDOW_FRAMES,
        "target_prominence_calibration": {
            "unit": (
                "16 calibration-seed maxima across speeds after within-clip physical "
                "support-window null deduplication"
            ),
            "quantile": CALIBRATION_QUANTILE,
            "headroom": HEADROOM,
            "rounding_quantum_m": PROMINENCE_QUANTUM_M,
            "zero_rounded_threshold_policy": (
                "abort as degenerate observability; do not accept flat zero-prominence "
                "cycles and do not silently add an unregistered floor"
            ),
        },
        "route_program": {
            "route_length_m": PILOT_ROUTE_LENGTH_M,
            "event_placements_m": list(EVENT_PLACEMENTS_M),
            "reference_placement_m": PILOT_EVENT_OBSTACLE_PROGRESS_M,
            "foot_anchor_equation": (
                "event_root_progress_m = placement_m "
                "- nominal_foot_forward_offset_m"
            ),
            "speed_envelope_mps": [MIN_ROUTE_SPEED_MPS, MAX_ROUTE_SPEED_MPS],
            "reference_speed_mps": REFERENCE_SPEED_MPS,
            "broad_acceleration_cap_mps2": BROAD_ACCELERATION_CAP_MPS2,
            "broad_jerk_cap": None,
            "broad_endpoint_cap": None,
            "selection": (
                "minimum deformation per seed/speed/side over the frozen placement "
                "set; |placement-3.6|, placement, then phase-evidence receipt digest "
                "tie break"
            ),
        },
        "timing_cap_calibration": {
            "per_seed_reduction": "maximum across selected speed/side programs",
            "quantile": CALIBRATION_QUANTILE,
            "headroom": HEADROOM,
            "acceleration_quantum_mps2": ACCELERATION_QUANTUM_MPS2,
            "jerk_quantum_mps3": JERK_QUANTUM_MPS3,
            "endpoint_quantum_mps": ENDPOINT_SPEED_QUANTUM_MPS,
        },
        "validation_separation_rule": separation_rule,
        "validation_gated_stratum": "reference",
        "validation_kill_rules": {
            "reference_attrition_max": VALIDATION_MAX_REFERENCE_ATTRITION,
            "reference_background_exceedance_or_missing_max": (
                VALIDATION_MAX_BACKGROUND_EXCEEDANCE
            ),
            "reference_full_frozen_min": (
                f"max({VALIDATION_MIN_FULL_FEASIBLE_ABSOLUTE}, "
                f"ceil({VALIDATION_MIN_FULL_FEASIBLE_FRACTION}*non_attrited))"
            ),
            "min_selected_signals_per_side": VALIDATION_MIN_SELECTED_SIGNALS_PER_SIDE,
        },
        "endpoint_strata_are_descriptive_only": True,
        "calibration_attrition_tolerances": {
            "min_background_seeds": MIN_CALIBRATION_BACKGROUND_SEEDS,
            "min_timing_contributing_seeds": MIN_TIMING_CONTRIBUTING_SEEDS,
        },
        "program_dependencies": {
            "route_progress_schema": ROUTE_PROGRESS_SCHEMA_VERSION,
            "route_progress_method": ROUTE_PROGRESS_METHOD,
            "route_progress_slope_policy": ROUTE_PROGRESS_SLOPE_POLICY,
            "phase_observability_schema": PHASE_OBSERVABILITY_SCHEMA_VERSION,
            "phase_prominence_method": PHASE_PROMINENCE_METHOD,
            "phase_background_method": PHASE_BACKGROUND_METHOD,
        },
    }
    return _identity(CAMPAIGN_SCHEMA_VERSION, fields)


def _load_physical_threshold_dependency(path: Path) -> tuple[StepOverThresholds, dict[str, Any]]:
    try:
        receipt = json.loads(path.read_text())
        file_digest = _sha256(path)
        normalized_digest = _json_hash(receipt)
        if file_digest != PHYSICAL_THRESHOLD_RECEIPT_FILE_SHA256:
            raise ValueError("physical threshold receipt file hash is not the locked artifact")
        if normalized_digest != PHYSICAL_THRESHOLD_RECEIPT_NORMALIZED_SHA256:
            raise ValueError(
                "physical threshold normalized identity is not the locked artifact"
            )
        if receipt.get("experiment") != "stepover_threshold_calibration":
            raise ValueError("unexpected physical threshold experiment")
        if receipt.get("status") != "calibrated":
            raise ValueError("physical threshold receipt is not calibrated")
        thresholds = StepOverThresholds(**receipt["stepover_thresholds"])
        thresholds.validate()
        calibration = receipt["calibration"]
        if not isinstance(calibration, dict):
            raise ValueError("physical threshold calibration block is invalid")
        if (
            isinstance(calibration.get("n_clips"), bool)
            or not isinstance(calibration.get("n_clips"), int)
            or calibration["n_clips"] <= 0
        ):
            raise ValueError("physical threshold receipt has no calibration clips")
        if (
            isinstance(calibration.get("n_accepted"), bool)
            or not isinstance(calibration.get("n_accepted"), int)
            or not 0 < calibration["n_accepted"] <= calibration["n_clips"]
        ):
            raise ValueError("physical threshold accepted count is invalid")
        provenance = receipt["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("physical threshold provenance is invalid")
        if not isinstance(provenance.get("code"), dict):
            raise ValueError("physical threshold code provenance is missing")
        if not isinstance(provenance.get("source_sha256"), dict):
            raise ValueError("physical threshold source hashes are missing")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid physical threshold receipt: {exc}") from exc
    fields = {
        "path": str(path),
        "file_sha256": file_digest,
        "locked_expected_file_sha256": PHYSICAL_THRESHOLD_RECEIPT_FILE_SHA256,
        "normalized_receipt_sha256": normalized_digest,
        "locked_expected_normalized_receipt_sha256": (
            PHYSICAL_THRESHOLD_RECEIPT_NORMALIZED_SHA256
        ),
        "experiment": receipt["experiment"],
        "status": receipt["status"],
        "n_clips": calibration["n_clips"],
        "n_accepted": calibration["n_accepted"],
        "thresholds": vars(thresholds),
        "evidentiary_role": "fixed-common-support-dependency-only",
        "new_confirmatory_evidence_for_support_thresholds": False,
    }
    return thresholds, _identity("ramp-fixed-common-support-dependency-v1", fields)


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not identify git state") from exc
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _verify_completion_git_state(
    initial: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    repo: Path,
    output: Path,
) -> dict[str, Any]:
    """Allow only this campaign's newly written output beneath an unchanged commit/diff."""
    if current.get("commit") != initial.get("commit"):
        raise ValueError("git commit changed during the calibration campaign")
    if current.get("tracked_diff_sha256") != initial.get("tracked_diff_sha256"):
        raise ValueError("tracked git diff changed during the calibration campaign")
    try:
        relative_output = output.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        relative_output = None
    unexpected: list[str] = []
    allowed: list[str] = []
    status = current.get("status")
    if not isinstance(status, list) or any(not isinstance(line, str) for line in status):
        raise ValueError("completion git status is invalid")
    for line in status:
        path = line[3:] if len(line) >= 4 else ""
        if relative_output is not None and (
            path == relative_output or path.startswith(relative_output + "/")
        ):
            allowed.append(line)
        else:
            unexpected.append(line)
    if unexpected:
        raise ValueError(
            "worktree changed outside the campaign output during calibration: "
            + "; ".join(unexpected)
        )
    return {
        "commit_unchanged": True,
        "tracked_diff_unchanged": True,
        "allowed_output_status": allowed,
        "unexpected_status": unexpected,
    }


def _generator_identity(runner: Any) -> dict[str, Any]:
    model_name = str(runner.model_name)
    cache_path = Path(runner.cache_path) if getattr(runner, "cache_path", None) else None
    if cache_path is None:
        raise ValueError("locked WALK text-cache file is missing")
    text_cache_identity = _locked_text_cache_identity(runner, cache_path)
    if os.environ.get("CHECKPOINTS_DIR"):
        raise ValueError(
            "locked released-prior calibration forbids ambient CHECKPOINTS_DIR"
        )
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        stats_folder = Path(runner.model.motion_rep.stats.folder)
    except (AttributeError, TypeError) as exc:
        raise ValueError("runner does not expose its loaded motion-statistics path") from exc
    if stats_folder.name != "motion" or stats_folder.parent.name != "stats":
        raise ValueError("loaded ARDY motion-statistics path has unexpected structure")
    snapshot = stats_folder.parent.parent
    try:
        snapshot.resolve().relative_to(Path(HF_HUB_CACHE).resolve())
    except ValueError as exc:
        raise ValueError(
            "loaded ARDY model is not inside the configured Hugging Face cache"
        ) from exc
    if (
        snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != f"models--nvidia--{model_name}"
    ):
        raise ValueError("loaded ARDY snapshot path does not match the resolved model name")
    revision = snapshot.name
    if not revision:
        raise ValueError("loaded ARDY snapshot has no revision identity")
    manifest = _file_manifest(snapshot)
    required_model_files = {
        "config.yaml",
        "denoiser.safetensors",
        "tokenizer.safetensors",
        "stats/motion/mean.npy",
        "stats/motion/std.npy",
        "stats/post_quantization/mean.npy",
        "stats/post_quantization/std.npy",
        "stats/pre_quantization/mean.npy",
        "stats/pre_quantization/std.npy",
    }
    missing = sorted(required_model_files - set(manifest))
    if missing:
        raise ValueError(
            "released ARDY snapshot lacks model-consumed files: " + ", ".join(missing)
        )
    checkpoint_identity = {
        "generator_id": f"nvidia/{model_name}@{revision}",
        "model_name": model_name,
        "hf_revision": revision,
        "snapshot_path": str(snapshot),
        "hf_hub_cache": str(Path(HF_HUB_CACHE)),
        "snapshot_path_derived_from_loaded_motion_stats": True,
        "snapshot_file_count": len(manifest),
        "snapshot_file_sha256": manifest,
        "snapshot_manifest_sha256": _json_hash(manifest),
        "required_model_files": sorted(required_model_files),
        "denoiser_path": str(snapshot / "denoiser.safetensors"),
        "checkpoint_sha256": manifest["denoiser.safetensors"],
    }
    return {
        "checkpoint": checkpoint_identity,
        "text_cache": text_cache_identity,
        "runner_class": type(runner).__name__,
        "runner_fps": float(runner.fps),
        "noise_stream_version": int(runner.noise_stream_version),
        "sampler": getattr(
            runner,
            "sampler_name",
            "ARDY deterministic DDIM, eta=0 (Scene2Motion runner contract)",
        ),
        "history_frames": getattr(runner, "history_frames", None),
        "generation_horizon_frames": getattr(
            getattr(runner, "model", None), "gen_horizon_len", None
        ),
    }


def _source_hashes(repo: Path) -> dict[str, str]:
    relative = (
        "experiments/calibrate_ramp_route_phase.py",
        "scene2motion/ramp/route_phase.py",
        "scene2motion/ramp/phase_observability.py",
        "scene2motion/ramp/step_phase.py",
        "scene2motion/stepover_eval.py",
        "scene2motion/robot.py",
        "scene2motion/constraints.py",
        "scene2motion/runner.py",
    )
    result: dict[str, str] = {}
    for name in relative:
        digest = _sha256(repo / name)
        if digest is None:
            raise ValueError(f"required source file is missing: {name}")
        result[name] = digest
    return result


def _physical_model_identity() -> dict[str, Any]:
    path = Path(ARDY_G1_XML)
    digest = _sha256(path)
    if digest is None:
        raise ValueError(f"released G1 MJCF is missing: {path}")
    return _identity(
        "ramp-exact-foot-kinematics-model-v1",
        {
            "path": str(path),
            "sha256": digest,
            "role": "physical-foot-geometries-and-forward-kinematics",
        },
    )


def _validated_generator_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(dict(value)))
        checkpoint = normalized["checkpoint"]
        text_cache = normalized["text_cache"]
        if not _is_sha256(checkpoint["checkpoint_sha256"]):
            raise ValueError("checkpoint content hash is invalid")
        if not _is_sha256(checkpoint["snapshot_manifest_sha256"]):
            raise ValueError("checkpoint snapshot-manifest hash is invalid")
        snapshot_files = checkpoint["snapshot_file_sha256"]
        required_model_files = checkpoint["required_model_files"]
        if (
            not isinstance(snapshot_files, dict)
            or not snapshot_files
            or any(
                not isinstance(name, str) or not _is_sha256(digest)
                for name, digest in snapshot_files.items()
            )
            or checkpoint["snapshot_manifest_sha256"] != _json_hash(snapshot_files)
            or not isinstance(required_model_files, list)
            or any(name not in snapshot_files for name in required_model_files)
        ):
            raise ValueError("checkpoint snapshot manifest is invalid")
        for name in ("sha256", "walk_prompt_embedding_content_sha256"):
            if not _is_sha256(text_cache[name]):
                raise ValueError(f"text-cache field {name!r} is invalid")
        if (
            not _is_sha256(text_cache["runner_memory_embedding_content_sha256"])
            or text_cache["runner_memory_embedding_content_sha256"]
            != text_cache["walk_prompt_embedding_content_sha256"]
            or text_cache["runner_memory_byte_matches_file"] is not True
        ):
            raise ValueError("runner memory/cache embedding identity is invalid")
        if text_cache["walk_prompt"] != WALK_PROMPT:
            raise ValueError("text-cache identity does not bind the locked WALK prompt")
        expected_key = hashlib.sha1(WALK_PROMPT.encode()).hexdigest()
        if text_cache["walk_prompt_cache_key_sha1"] != expected_key:
            raise ValueError("text-cache prompt key does not match the locked WALK prompt")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid generator/checkpoint/text-cache identity: {exc}") from exc
    return normalized


def _runtime_identity() -> dict[str, Any]:
    """Bind external ARDY inference code and numerical/FK runtime versions."""
    import ardy
    import mujoco
    import torch

    package_root = Path(ardy.__file__).resolve().parent
    source_files = sorted(package_root.rglob("*.py"))
    if not source_files:
        raise ValueError("could not enumerate ARDY Python runtime sources")
    sources: dict[str, str] = {}
    for path in source_files:
        digest = _sha256(path)
        if digest is None:
            raise ValueError(f"ARDY runtime source disappeared: {path}")
        sources[path.relative_to(package_root).as_posix()] = digest
    repository = package_root.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
        tracked_status = subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not bind the external ARDY git identity") from exc
    fields = {
        "python": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "mujoco_version": mujoco.__version__,
        "ardy_package_root": str(package_root),
        "ardy_git_commit": commit,
        "ardy_tracked_status": tracked_status,
        "ardy_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "ardy_python_source_count": len(sources),
        "ardy_python_source_manifest_sha256": _json_hash(sources),
        "ardy_python_source_sha256": sources,
    }
    return _identity("ramp-ardy-runtime-identity-v1", fields)


def _atomic_write(path: Path, writer: Callable[[Any], None]) -> None:
    """Durably replace one evidence file without truncating its prior version."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_write(path, lambda handle: handle.write(payload))


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        _canonical_json(dict(value)) + "\n" for value in values
    ).encode()
    _atomic_write(path, lambda handle: handle.write(payload))


def _persist_qpos(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    _atomic_write(path, lambda handle: np.savez(handle, **normalized))


def run_campaign(
    *,
    out: str | Path,
    threshold_receipt: str | Path,
    runner: Any | None = None,
    body: Any | None = None,
    code_state_fn: Callable[[Path], Mapping[str, Any]] = _git_state,
    generator_identity_fn: Callable[[Any], Mapping[str, Any]] = _generator_identity,
    analyze_clip_fn: Callable[..., AnalyzedClip] = analyze_generated_clip,
    cache_path: str | Path = "outputs/text_cache.npz",
) -> dict[str, Any]:
    """Run the locked campaign, returning its complete receipt or raising after blocking."""
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise CalibrationAbort(
            f"refusing nonempty calibration output directory: {output}"
        )
    repo = Path(__file__).resolve().parents[1]
    # Capture the launch state before creating the campaign directory.  Otherwise a real
    # in-repository run observes its own untracked batch-plan/output files and falsely
    # reports that the pre-existing worktree was dirty.
    code = dict(code_state_fn(repo))
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stage = "preflight"
    sample_count_exact = True
    rows: list[dict[str, Any]] = []
    clips: list[AnalyzedClip] = []
    qpos_archive: dict[str, np.ndarray] = {}
    unexpected_sample_hashes: list[str] = []
    counters = {
        "generate_invocations_planned": N_BATCHES,
        "generate_invocations_started": 0,
        "generate_invocations_completed": 0,
        "samples_planned": PLANNED_SAMPLES,
        "samples_launched": 0,
        "samples_returned": 0,
        "samples_converted_to_qpos": 0,
        "samples_analyzed": 0,
    }
    design = campaign_design()
    plan = locked_batch_plan()
    attempt_plan = [
        {
            "batch_index": batch.index,
            "batch_digest": batch.digest,
            "batch_position": position,
            "split": batch.split,
            "seed_block": batch.seed_block,
            "seed": seed,
            "speed_label": batch.speed_label,
            "requested_speed_mps": batch.requested_speed_mps,
        }
        for batch in plan
        for position, seed in enumerate(batch.seeds)
    ]
    attempts = [{**item, "status": "planned"} for item in attempt_plan]
    attempt_by_key = {
        (item["batch_index"], item["batch_position"]): item for item in attempts
    }
    receipt: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA_VERSION,
        "experiment": "calibrate_ramp_route_phase",
        "status": "running",
        "complete": False,
        "blocked": False,
        "stage": stage,
        "resume_supported": False,
        "campaign_design": design,
        "donor_step_quality_dependency": donor_quality_dependency(),
        "query_accounting": counters,
        "sample_count_exact": sample_count_exact,
        "rows": [],
        "provenance": {},
    }

    def persist() -> None:
        _write_jsonl(output / "rows.jsonl", rows)
        _write_jsonl(output / "attempts.jsonl", attempts)
        _persist_qpos(output / "qpos.npz", qpos_archive)
        receipt["stage"] = stage
        receipt["query_accounting"] = dict(counters)
        receipt["sample_count_exact"] = sample_count_exact
        receipt["rows"] = list(rows)
        receipt["evidence_anchors"] = {
            "rows": {
                "path": "rows.jsonl",
                "n_rows": len(rows),
                "logical_sha256": _json_hash(rows),
                "file_sha256": _sha256(output / "rows.jsonl"),
            },
            "attempts": {
                "path": "attempts.jsonl",
                "n_rows": len(attempts),
                "plan_logical_sha256": _json_hash(attempt_plan),
                "logical_sha256": _json_hash(attempts),
                "file_sha256": _sha256(output / "attempts.jsonl"),
            },
            "qpos": {
                "path": "qpos.npz",
                "n_arrays": len(qpos_archive),
                "content_sha256": (
                    _array_hash(qpos_archive) if qpos_archive else None
                ),
                "archive_sha256": _sha256(output / "qpos.npz"),
            },
            "unexpected_returned_samples": {
                "count": len(unexpected_sample_hashes),
                "sample_sha256": list(unexpected_sample_hashes),
            },
        }
        receipt["attempt_status_counts"] = {
            status: sum(item["status"] == status for item in attempts)
            for status in sorted({item["status"] for item in attempts})
        }
        receipt["wall_clock_s"] = float(time.monotonic() - started)
        _write_json(output / "receipt.json", receipt)

    try:
        _write_json(output / "batch_plan.json", {
            "campaign_design_sha256": design["sha256"],
            "batches": [batch.as_dict() for batch in plan],
            "attempt_plan": attempt_plan,
            "attempt_plan_logical_sha256": _json_hash(attempt_plan),
        })
        thresholds, physical_dependency = _load_physical_threshold_dependency(
            Path(threshold_receipt)
        )
        receipt["physical_threshold_dependency"] = physical_dependency
        receipt["provenance"]["code"] = code
        source_hashes = _source_hashes(repo)
        physical_model = _physical_model_identity()
        receipt["provenance"]["source_sha256"] = source_hashes
        receipt["provenance"]["physical_model"] = physical_model
        if code.get("dirty") is not False:
            raise ValueError("calibration requires an exactly clean git worktree")
        if not isinstance(code.get("commit"), str) or not code["commit"].strip():
            raise ValueError("calibration requires a concrete git commit")

        if os.environ.get("CHECKPOINTS_DIR"):
            raise ValueError(
                "locked released-prior calibration forbids ambient CHECKPOINTS_DIR"
            )
        runner = runner or ArdyRunner(cache_path=cache_path)
        if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
            raise ValueError(f"calibration requires runner fps == {FPS:g}")
        if (
            isinstance(runner.noise_stream_version, (bool, np.bool_))
            or not isinstance(runner.noise_stream_version, (int, np.integer))
            or int(runner.noise_stream_version) != NOISE_STREAM_VERSION
        ):
            raise ValueError("calibration requires ARDY noise_stream_version == 2")
        body = body or G1Body(None)
        generator_identity = _validated_generator_identity(
            generator_identity_fn(runner)
        )
        runtime_identity = _runtime_identity()
        receipt["provenance"]["generator"] = generator_identity
        receipt["provenance"]["runtime"] = runtime_identity
        receipt["provenance"]["generation_settings"] = {
            "prompt": WALK_PROMPT,
            "fps": FPS,
            "n_frames": N_FRAMES,
            "diffusion_steps": DIFFUSION_STEPS,
            "cfg_weight": list(CFG_WEIGHT),
            "noise_stream_version": NOISE_STREAM_VERSION,
        }

        def revalidate_bound_identities() -> dict[str, Any]:
            current_code = dict(code_state_fn(repo))
            git_check = _verify_completion_git_state(
                code, current_code, repo=repo, output=output
            )
            if _source_hashes(repo) != source_hashes:
                raise ValueError("source content changed during the calibration campaign")
            if _physical_model_identity() != physical_model:
                raise ValueError("physical G1 model changed during the calibration campaign")
            if (
                _validated_generator_identity(generator_identity_fn(runner))
                != generator_identity
            ):
                raise ValueError(
                    "checkpoint or locked WALK text-cache identity changed during calibration"
                )
            if _runtime_identity() != runtime_identity:
                raise ValueError("ARDY or numerical/FK runtime changed during calibration")
            if not np.isclose(float(runner.fps), FPS, atol=0.0, rtol=0.0):
                raise ValueError("runner fps changed during the calibration campaign")
            if (
                isinstance(runner.noise_stream_version, (bool, np.bool_))
                or not isinstance(runner.noise_stream_version, (int, np.integer))
                or int(runner.noise_stream_version) != NOISE_STREAM_VERSION
            ):
                raise ValueError("runner noise-stream version changed during calibration")
            return {
                "git": git_check,
                "sources_unchanged": True,
                "physical_model_unchanged": True,
                "checkpoint_and_text_cache_unchanged": True,
                "ardy_and_numerical_runtime_unchanged": True,
                "runner_contract_unchanged": True,
            }

        stage = "generation"
        persist()

        for batch in plan:
            spec = root_only_walk_spec(batch.requested_speed_mps)
            specs = [spec] * BATCH_SIZE
            prompts = [WALK_PROMPT] * BATCH_SIZE
            seeds = list(batch.seeds)
            batch_attempts = [
                attempt_by_key[(batch.index, position)] for position in range(BATCH_SIZE)
            ]
            for attempt in batch_attempts:
                if attempt["status"] != "planned":
                    raise RuntimeError("attempt ledger is not in its immutable planned state")
                attempt["status"] = "launched"
            counters["generate_invocations_started"] += 1
            counters["samples_launched"] += BATCH_SIZE
            persist()
            try:
                returned = runner.generate(
                    prompts,
                    specs,
                    N_FRAMES,
                    DIFFUSION_STEPS,
                    cfg_weight=CFG_WEIGHT,
                    seeds=seeds,
                )
            except Exception:
                sample_count_exact = False
                raise
            if not isinstance(returned, Sequence):
                sample_count_exact = False
                raise ValueError("runner returned a non-sequence batch")
            counters["samples_returned"] += len(returned)
            if len(returned) == BATCH_SIZE:
                counters["generate_invocations_completed"] += 1
            else:
                sample_count_exact = False

            returned_hash_errors: list[str] = []
            for position, sample in enumerate(returned):
                if position >= BATCH_SIZE:
                    try:
                        if not isinstance(sample, Mapping):
                            raise TypeError("unexpected returned sample is not a mapping")
                        unexpected_sample_hashes.append(_sample_hash(sample))
                    except (TypeError, ValueError) as exc:
                        unexpected_sample_hashes.append(
                            _json_hash(
                                {
                                    "position": position,
                                    "type": type(sample).__name__,
                                    "hash_error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        )
                    continue
                attempt = batch_attempts[position]
                attempt["returned_batch_size"] = len(returned)
                attempt["status"] = "returned_unprocessed"
                try:
                    if not isinstance(sample, Mapping):
                        raise TypeError("runner sample is not a mapping")
                    attempt["sample_sha256"] = _sample_hash(sample)
                except (TypeError, ValueError) as exc:
                    sample_count_exact = False
                    attempt["status"] = "returned_hash_failed"
                    attempt["sample_hash_error"] = f"{type(exc).__name__}: {exc}"
                    returned_hash_errors.append(
                        f"batch {batch.index} position {position}: {exc}"
                    )
            persist()
            if returned_hash_errors:
                raise ValueError(
                    "returned sample pre-hash failed: " + "; ".join(returned_hash_errors)
                )

            for position, sample in enumerate(returned[:BATCH_SIZE]):
                seed = seeds[position]
                attempt = batch_attempts[position]
                qpos_key = (
                    f"{batch.split}__{batch.speed_label}__seed{seed}"
                )
                qpos = np.asarray(runner.to_qpos(sample), dtype=float)
                if qpos.ndim != 2 or qpos.shape[0] != N_FRAMES or not np.isfinite(qpos).all():
                    raise ValueError("runner qpos conversion violates the 200-frame contract")
                qpos_archive[qpos_key] = np.array(qpos, copy=True)
                counters["samples_converted_to_qpos"] += 1
                attempt.update(
                    {
                        "status": "qpos_archived",
                        "qpos_archive_key": qpos_key,
                        "qpos_content_sha256": _array_hash(
                            {qpos_key: qpos_archive[qpos_key]}
                        ),
                    }
                )
                persist()
                clip = analyze_clip_fn(
                    body=body,
                    qpos=qpos,
                    sample=sample,
                    split=batch.split,
                    seed=seed,
                    speed_label=batch.speed_label,
                    requested_speed_mps=batch.requested_speed_mps,
                    thresholds=thresholds,
                    qpos_archive_key=qpos_key,
                )
                if not isinstance(clip, AnalyzedClip):
                    raise TypeError("analyze_clip_fn must return AnalyzedClip")
                expected_sample_hash = _sample_hash(sample)
                expected_qpos_hash = _array_hash({qpos_key: qpos_archive[qpos_key]})
                if clip.sample_sha256 != expected_sample_hash:
                    raise ValueError("analyzer sample hash does not bind the returned sample")
                if clip.qpos_content_sha256 != expected_qpos_hash:
                    raise ValueError("analyzer qpos hash does not bind the archived qpos")
                if clip.qpos_archive_key != qpos_key:
                    raise ValueError("analyzer qpos archive key does not match the batch plan")
                clips.append(clip)
                rows.append({
                    "batch_index": batch.index,
                    "batch_digest": batch.digest,
                    "batch_position": position,
                    **clip.as_dict(),
                })
                counters["samples_analyzed"] += 1
                attempt.update(
                    {
                        "status": "analyzed",
                        "row_index": len(rows) - 1,
                        "n_measured_cycles": len(clip.cycles),
                        "measurement_rejection": clip.measurement_rejection,
                    }
                )
                persist()
            persist()
            if len(returned) != BATCH_SIZE:
                raise ValueError(
                    f"runner returned {len(returned)} samples for an eight-sample batch"
                )

        if counters["samples_returned"] != PLANNED_SAMPLES or len(clips) != PLANNED_SAMPLES:
            raise ValueError("completed generation does not contain exactly 72 samples")

        stage = "post_generation_identity_revalidation"
        receipt["provenance"]["post_generation_identity_revalidation"] = (
            revalidate_bound_identities()
        )
        persist()

        stage = "calibration"
        target_receipt = freeze_target_prominence(clips)
        pmin = float(target_receipt["fields"]["target_min_prominence_m"])
        if target_receipt["sha256"] == receipt["donor_step_quality_dependency"]["sha256"]:
            raise RuntimeError("donor and target threshold identities unexpectedly collide")
        receipt["target_prominence_receipt"] = target_receipt
        calibration_clips = [clip for clip in clips if clip.split == "calibration"]
        candidates, selected = build_and_select_programs(
            calibration_clips, target_min_prominence_m=pmin
        )
        receipt["program_candidates"] = [item.compact_dict() for item in candidates]
        receipt["selected_programs"] = [item.as_dict() for item in selected]
        receipt["program_attrition"] = {
            "scope": "calibration split; cycle x placement enumeration",
            "measured_cycles": sum(len(clip.cycles) for clip in calibration_clips),
            "candidate_rows": len(candidates),
            "broad_feasible": sum(item.feasible for item in candidates),
            "selected_seed_speed_side": len(selected),
            "selected_placement_usage_m": {
                f"{placement:g}": sum(
                    1 for item in selected if item.placement_m == placement
                )
                for placement in EVENT_PLACEMENTS_M
            },
            "rejection_counts": {
                reason: sum(item.rejection == reason for item in candidates)
                for reason in sorted(
                    {item.rejection for item in candidates if item.rejection is not None}
                )
            },
        }
        timing_receipt = freeze_timing_bounds(selected)
        receipt["route_timing_receipt"] = timing_receipt
        frozen = timing_bounds_from_receipt(timing_receipt)
        persist()

        stage = "validation"
        validation = validate_frozen_calibration(
            clips,
            target_min_prominence_m=pmin,
            frozen_bounds=frozen,
        )
        receipt["validation"] = validation
        stage = "post_analysis_identity_revalidation"
        receipt["provenance"]["post_analysis_identity_revalidation"] = (
            revalidate_bound_identities()
        )
        persist()
        if not validation["passed"]:
            raise ValueError(
                "locked validation kill rule failed: "
                + "; ".join(validation["kill_reasons"])
            )
        if counters != {
            "generate_invocations_planned": N_BATCHES,
            "generate_invocations_started": N_BATCHES,
            "generate_invocations_completed": N_BATCHES,
            "samples_planned": PLANNED_SAMPLES,
            "samples_launched": PLANNED_SAMPLES,
            "samples_returned": PLANNED_SAMPLES,
            "samples_converted_to_qpos": PLANNED_SAMPLES,
            "samples_analyzed": PLANNED_SAMPLES,
        }:
            raise RuntimeError("final query accounting does not equal the frozen campaign")
        stage = "complete"
        receipt.update({
            "status": "complete",
            "complete": True,
            "blocked": False,
            "stage": stage,
            "actual_ardy_samples": PLANNED_SAMPLES,
            "conservative_charged_ardy_samples": PLANNED_SAMPLES,
        })
        receipt["provenance"]["completion_identity_revalidation"] = (
            receipt["provenance"]["post_analysis_identity_revalidation"]
        )
        persist()
        return receipt
    except Exception as exc:
        receipt.update({
            "schema": FAILURE_SCHEMA_VERSION,
            "status": "blocked",
            "complete": False,
            "blocked": True,
            "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_ardy_samples": (
                counters["samples_returned"] if sample_count_exact else None
            ),
            "returned_ardy_samples_lower_bound": counters["samples_returned"],
            "conservative_charged_ardy_samples": max(
                counters["samples_launched"], counters["samples_returned"]
            ),
        })
        persist()
        if isinstance(exc, CalibrationAbort):
            raise
        raise CalibrationAbort(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/calibrate_ramp_route_phase")
    parser.add_argument(
        "--threshold-receipt",
        default="outputs/exp016_threshold_calibration/receipt.json",
    )
    parser.add_argument("--cache-path", default="outputs/text_cache.npz")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run_campaign(
        out=args.out,
        threshold_receipt=args.threshold_receipt,
        cache_path=args.cache_path,
    )


if __name__ == "__main__":
    main()
