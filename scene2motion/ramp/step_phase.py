"""Physical step-cycle evidence for RAMP phase alignment.

The generic aligners in :mod:`scene2motion.ramp.phase` deliberately accept only an
already measured phase trace and explicit stance evidence.  This module is the production
bridge from the exact qpos-derived physical-foot envelopes in
:mod:`scene2motion.stepover_eval` to those inputs.

A trace covers one *observed swing interval*, from the first unsupported swing-foot frame
through the first subsequent supported frame.  Its fixed convention is takeoff ``0.25``,
apex ``0.50``, and landing ``0.75``.  The trace is piecewise linear in time between those
physical landmarks; it does not invent phase outside the measured swing.  Consequently,
all alignment helpers fail rather than extrapolate beyond takeoff or landing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..stepover_eval import (
    KinematicStepEvent,
    StepOverThresholds,
    foot_kinematics_series,
)
from .packet import PhaseMatch
from .phase import (
    StanceEvidence,
    TargetPhaseMatch,
    align_cyclic_phase_windows,
    align_target_phase_window,
)


Side = Literal["left", "right"]
_SIDES: tuple[Side, Side] = ("left", "right")
STEP_PHASE_METHOD = "qpos-physical-foot-swing-landmarks-v1"
STANCE_EVIDENCE_METHOD = "qpos-physical-foot-support-mask-v1"
STEP_PHASE_PROTOCOL_VERSION = "qpos-step-phase-measurement-v2"
COMMON_STEP_PHASE_PROTOCOL_VERSION = "qpos-step-phase-common-physics-v1"
TAKEOFF_PHASE = 0.25
APEX_PHASE = 0.50
LANDING_PHASE = 0.75


def _protocol_hash(value: Any) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError("measurement_protocol_hash must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _unit_fraction(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite fraction")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite fraction") from exc
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not np.isfinite(result) or not lower_ok or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _stable_support_samples(duration_s: float, fps: float) -> int:
    """Samples needed so first-to-last elapsed time is at least ``duration_s``."""
    return int(math.ceil(_positive_finite(duration_s, "support dwell duration") * fps)) + 1


def step_phase_measurement_protocol_hash(
    thresholds: StepOverThresholds,
    *,
    support_window_s: float,
    min_stance_support_fraction: float,
    min_relative_lift_m: float,
) -> str:
    """Canonical cross-rate identity for every physical step-phase measurement gate."""
    thresholds.validate()
    payload = {
        "protocol_version": STEP_PHASE_PROTOCOL_VERSION,
        "support_height_m": float(thresholds.support_height_m),
        "support_speed_mps": float(thresholds.support_speed_mps),
        "min_stance_support_fraction": float(min_stance_support_fraction),
        "support_window_s": float(support_window_s),
        "max_floor_penetration_m": float(thresholds.max_floor_penetration_m),
        "stable_support_dwell_s": float(thresholds.landing_dwell_s),
        "min_relative_lift_m": float(min_relative_lift_m),
        "phase_convention": {
            "method": STEP_PHASE_METHOD,
            "takeoff": TAKEOFF_PHASE,
            "apex": APEX_PHASE,
            "landing": LANDING_PHASE,
        },
    }
    for name, value in payload.items():
        if name not in ("protocol_version", "phase_convention") and not np.isfinite(value):
            raise ValueError(f"measurement protocol field {name!r} must be finite")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def step_phase_common_physical_protocol_hash(
    thresholds: StepOverThresholds,
    *,
    support_window_s: float,
    min_stance_support_fraction: float,
) -> str:
    """Identity of shared physical phase measurement, excluding quality prominence.

    ``min_relative_lift_m`` decides whether a measured swing is prominent enough for a
    particular role.  It is intentionally absent here so a high-quality donor and an
    ordinary target gait can share one physical phase convention while retaining distinct
    gate-specific :func:`step_phase_measurement_protocol_hash` receipts.
    """
    thresholds.validate()
    payload = {
        "protocol_version": COMMON_STEP_PHASE_PROTOCOL_VERSION,
        "support_height_m": float(thresholds.support_height_m),
        "support_speed_mps": float(thresholds.support_speed_mps),
        "min_stance_support_fraction": float(min_stance_support_fraction),
        "support_window_s": float(support_window_s),
        "max_floor_penetration_m": float(thresholds.max_floor_penetration_m),
        "stable_support_dwell_s": float(thresholds.landing_dwell_s),
        "phase_convention": {
            "method": STEP_PHASE_METHOD,
            "takeoff": TAKEOFF_PHASE,
            "apex": APEX_PHASE,
            "landing": LANDING_PHASE,
        },
    }
    for name, value in payload.items():
        if name not in ("protocol_version", "phase_convention") and not np.isfinite(value):
            raise ValueError(f"common physical protocol field {name!r} must be finite")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _series(kinematics: Mapping[str, Any], side: Side, name: str) -> np.ndarray:
    try:
        values = np.asarray(kinematics[side][name], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"kinematics lacks {side} {name}") from exc
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{side} {name} must be a non-empty finite 1-D series")
    return values


def _validated_kinematics(
    kinematics: Mapping[str, Any],
) -> tuple[dict[Side, dict[str, np.ndarray]], int]:
    result: dict[Side, dict[str, np.ndarray]] = {}
    lengths: set[int] = set()
    for side in _SIDES:
        result[side] = {}
        for name in ("bottom_clearance_m", "planar_speed_mps"):
            values = _series(kinematics, side, name)
            result[side][name] = values
            lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError("left/right kinematic series must share one frame count")
    return result, lengths.pop()


def _support_masks(
    kinematics: Mapping[Side, Mapping[str, np.ndarray]],
    thresholds: StepOverThresholds,
) -> dict[Side, np.ndarray]:
    return {
        side: (
            (kinematics[side]["bottom_clearance_m"] <= thresholds.support_height_m)
            & (kinematics[side]["planar_speed_mps"] <= thresholds.support_speed_mps)
        )
        for side in _SIDES
    }


def _false_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Inclusive runs in which ``mask`` is false."""
    padded = np.r_[True, np.asarray(mask, dtype=bool), True].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == -1)
    ends = np.flatnonzero(edges == 1) - 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _centered_window(center: int, length: int, seconds: float, fps: float) -> tuple[int, int]:
    half = int(round(0.5 * seconds * fps))
    first, last = center - half, center + half
    if first < 0 or last >= length:
        raise ValueError("locked stance-support window is truncated by the clip boundary")
    return first, last


def _phase_trace(takeoff: int, apex: int, landing: int) -> np.ndarray:
    # Two samples per side keep the largest possible phase advance strictly below the
    # generic aligner's ambiguity ceiling of 0.25 cycles/frame.
    if not takeoff < apex < landing:
        raise ValueError("takeoff/apex/landing landmarks are nonmonotone")
    if apex - takeoff < 2 or landing - apex < 2:
        raise ValueError("takeoff/apex/landing landmarks have insufficient phase samples")
    rising = np.linspace(TAKEOFF_PHASE, APEX_PHASE, apex - takeoff + 1)
    falling = np.linspace(APEX_PHASE, LANDING_PHASE, landing - apex + 1)[1:]
    phase = np.concatenate((rising, falling))
    if np.any(np.diff(phase) <= 0.0) or np.any((phase < 0.0) | (phase >= 1.0)):
        raise ValueError("constructed physical phase trace is not strictly monotone")
    return phase


@dataclass(frozen=True)
class StepPhaseLandmarks:
    """Absolute physical-foot landmarks for one complete unilateral swing.

    ``takeoff_frame`` is the first unsupported frame and ``landing_frame`` is the first
    subsequent re-supported frame under the locked height-and-speed support test.
    """

    takeoff_frame: int
    apex_frame: int
    landing_frame: int
    takeoff_phase: float = TAKEOFF_PHASE
    apex_phase: float = APEX_PHASE
    landing_phase: float = LANDING_PHASE

    def __post_init__(self) -> None:
        for name in ("takeoff_frame", "apex_frame", "landing_frame"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if not self.takeoff_frame < self.apex_frame < self.landing_frame:
            raise ValueError("takeoff/apex/landing landmarks are nonmonotone")
        phases = np.asarray(
            [self.takeoff_phase, self.apex_phase, self.landing_phase], dtype=float
        )
        if not np.isfinite(phases).all() or np.any(np.diff(phases) <= 0.0):
            raise ValueError("landmark phases must be finite and strictly increasing")
        if np.any((phases < 0.0) | (phases >= 1.0)):
            raise ValueError("landmark phases must lie in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepPhaseCycle:
    """Immutable audit receipt for one qpos-derived unilateral swing cycle."""

    method: str
    event: KinematicStepEvent
    landmarks: StepPhaseLandmarks
    fps: float
    phase_trace: tuple[float, ...]
    stance_side: Side
    stance_support_fraction: float
    min_stance_support_fraction: float
    support_window_s: float
    support_window_start_frame: int
    support_window_end_frame: int
    support_height_m: float
    support_speed_mps: float
    max_floor_penetration_m: float
    landing_dwell_s: float
    takeoff_dwell_frames: int
    landing_dwell_frames: int
    min_relative_lift_m: float
    measurement_protocol_hash: str
    common_physical_protocol_hash: str

    def __post_init__(self) -> None:
        if self.method != STEP_PHASE_METHOD:
            raise ValueError(f"step phase method must be {STEP_PHASE_METHOD!r}")
        if not isinstance(self.event, KinematicStepEvent):
            raise ValueError("event must be a KinematicStepEvent")
        if not isinstance(self.landmarks, StepPhaseLandmarks):
            raise ValueError("landmarks must be StepPhaseLandmarks")
        if (
            self.landmarks.takeoff_phase != TAKEOFF_PHASE
            or self.landmarks.apex_phase != APEX_PHASE
            or self.landmarks.landing_phase != LANDING_PHASE
        ):
            raise ValueError("landmarks do not use the locked 0.25/0.50/0.75 phase convention")
        if self.event.frame != self.landmarks.apex_frame:
            raise ValueError("event frame must equal the physical swing apex")
        if self.event.side not in _SIDES or self.stance_side not in _SIDES:
            raise ValueError("swing and stance sides must be 'left' or 'right'")
        expected_stance: Side = "right" if self.event.side == "left" else "left"
        if self.stance_side != expected_stance:
            raise ValueError("stance side must be contralateral to the swing foot")

        _strict_int(self.event.frame, "event frame")
        _strict_int(self.event.support_window_start_frame, "event support-window start")
        _strict_int(self.event.support_window_end_frame, "event support-window end")
        for name in (
            "relative_lift_m",
            "swing_height_m",
            "stance_height_m",
            "stance_support_fraction",
            "swing_planar_speed_mps",
            "stance_planar_speed_mps",
        ):
            value = float(getattr(self.event, name))
            if not np.isfinite(value):
                raise ValueError(f"event field {name!r} must be finite")
        if not 0.0 <= self.event.stance_support_fraction <= 1.0:
            raise ValueError("event stance support must lie in [0, 1]")
        if self.event.swing_planar_speed_mps < 0.0 or self.event.stance_planar_speed_mps < 0.0:
            raise ValueError("event planar speeds must be non-negative")
        measured_lift = self.event.swing_height_m - self.event.stance_height_m
        if not np.isclose(
            self.event.relative_lift_m, measured_lift, atol=1e-12, rtol=0.0
        ):
            raise ValueError("event relative lift does not match swing-minus-stance height")

        object.__setattr__(self, "fps", _positive_finite(self.fps, "fps"))
        object.__setattr__(
            self,
            "stance_support_fraction",
            _unit_fraction(self.stance_support_fraction, "stance_support_fraction"),
        )
        object.__setattr__(
            self,
            "min_stance_support_fraction",
            _unit_fraction(
                self.min_stance_support_fraction,
                "min_stance_support_fraction",
                positive=True,
            ),
        )
        if self.stance_support_fraction < self.min_stance_support_fraction:
            raise ValueError("contralateral stance support is below the locked threshold")
        object.__setattr__(
            self,
            "support_window_s",
            _positive_finite(self.support_window_s, "support_window_s"),
        )
        for name in ("support_window_start_frame", "support_window_end_frame"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if not (
            self.support_window_start_frame
            <= self.event.frame
            <= self.support_window_end_frame
        ):
            raise ValueError("stance-support window does not contain the event")
        if (
            self.event.support_window_start_frame != self.support_window_start_frame
            or self.event.support_window_end_frame != self.support_window_end_frame
        ):
            raise ValueError("event and phase receipt use different stance-support windows")
        if not np.isclose(
            self.event.stance_support_fraction, self.stance_support_fraction
        ):
            raise ValueError("event and phase receipt report different stance support")
        expected_first, expected_last = _centered_window(
            self.event.frame,
            max(self.event.frame + 1, self.support_window_end_frame + 1),
            self.support_window_s,
            self.fps,
        )
        if (
            self.support_window_start_frame != expected_first
            or self.support_window_end_frame != expected_last
        ):
            raise ValueError("stance-support window does not match its duration and fps")

        phase = tuple(float(value) for value in self.phase_trace)
        if len(phase) != self.landmarks.landing_frame - self.landmarks.takeoff_frame + 1:
            raise ValueError("phase trace does not span the observed swing interval")
        values = np.asarray(phase)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values >= 1.0)):
            raise ValueError("phase trace must be finite and lie in [0, 1)")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("physical phase trace must be strictly monotone")
        local_apex = self.local_apex_frame
        if not (
            np.isclose(values[0], self.landmarks.takeoff_phase)
            and np.isclose(values[local_apex], self.landmarks.apex_phase)
            and np.isclose(values[-1], self.landmarks.landing_phase)
        ):
            raise ValueError("phase trace does not preserve its physical landmarks")
        expected_phase = _phase_trace(
            self.landmarks.takeoff_frame,
            self.landmarks.apex_frame,
            self.landmarks.landing_frame,
        )
        if not np.array_equal(values, expected_phase):
            raise ValueError("phase trace is not the exact locked landmark interpolation")
        object.__setattr__(self, "phase_trace", phase)

        for name in (
            "support_height_m",
            "support_speed_mps",
            "max_floor_penetration_m",
            "min_relative_lift_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "landing_dwell_s", _positive_finite(self.landing_dwell_s, "landing_dwell_s")
        )
        takeoff_dwell_frames = _strict_int(
            self.takeoff_dwell_frames, "takeoff_dwell_frames"
        )
        dwell_frames = _strict_int(self.landing_dwell_frames, "landing_dwell_frames")
        expected_dwell_frames = _stable_support_samples(self.landing_dwell_s, self.fps)
        if takeoff_dwell_frames != expected_dwell_frames or dwell_frames != expected_dwell_frames:
            raise ValueError("support dwell sample counts do not match duration and fps")
        object.__setattr__(self, "takeoff_dwell_frames", takeoff_dwell_frames)
        object.__setattr__(self, "landing_dwell_frames", dwell_frames)
        if self.event.relative_lift_m < self.min_relative_lift_m:
            raise ValueError("event lift is below the locked relative-lift threshold")
        expected_protocol_hash = step_phase_measurement_protocol_hash(
            StepOverThresholds(
                support_height_m=self.support_height_m,
                support_speed_mps=self.support_speed_mps,
                min_contralateral_support_fraction=self.min_stance_support_fraction,
                max_floor_penetration_m=self.max_floor_penetration_m,
                landing_dwell_s=self.landing_dwell_s,
            ),
            support_window_s=self.support_window_s,
            min_stance_support_fraction=self.min_stance_support_fraction,
            min_relative_lift_m=self.min_relative_lift_m,
        )
        if _protocol_hash(self.measurement_protocol_hash) != expected_protocol_hash:
            raise ValueError("measurement_protocol_hash does not match the receipt gates")
        expected_common_hash = step_phase_common_physical_protocol_hash(
            StepOverThresholds(
                support_height_m=self.support_height_m,
                support_speed_mps=self.support_speed_mps,
                min_contralateral_support_fraction=self.min_stance_support_fraction,
                max_floor_penetration_m=self.max_floor_penetration_m,
                landing_dwell_s=self.landing_dwell_s,
            ),
            support_window_s=self.support_window_s,
            min_stance_support_fraction=self.min_stance_support_fraction,
        )
        if _protocol_hash(self.common_physical_protocol_hash) != expected_common_hash:
            raise ValueError(
                "common_physical_protocol_hash does not match the receipt physics"
            )

    @property
    def swing_side(self) -> Side:
        return self.event.side

    @property
    def apex_frame(self) -> int:
        return self.landmarks.apex_frame

    @property
    def local_apex_frame(self) -> int:
        return self.landmarks.apex_frame - self.landmarks.takeoff_frame

    @property
    def support_source(self) -> str:
        return (
            f"{STANCE_EVIDENCE_METHOD};height_m<={self.support_height_m:.9g};"
            f"speed_mps<={self.support_speed_mps:.9g};"
            f"frames={self.support_window_start_frame}:{self.support_window_end_frame};"
            f"fps={self.fps:.9g}"
        )

    def stance_evidence(self, *, absolute_frame: bool = True) -> StanceEvidence:
        """Return measured support evidence in absolute or trace-local coordinates."""
        center = self.apex_frame if absolute_frame else self.local_apex_frame
        return StanceEvidence(
            side=self.stance_side,
            center_frame=center,
            support_fraction=self.stance_support_fraction,
            source=self.support_source,
        )

    def require_phase_window(self, half_window_frames: int) -> None:
        half = _strict_int(half_window_frames, "half_window_frames")
        if half < 1:
            raise ValueError("half_window_frames must be at least one")
        if self.local_apex_frame - half < 0 or self.local_apex_frame + half >= len(
            self.phase_trace
        ):
            raise ValueError("requested phase window extends beyond physical takeoff/landing")

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "event": asdict(self.event),
            "landmarks": self.landmarks.as_dict(),
            "fps": self.fps,
            "phase_trace": list(self.phase_trace),
            "phase_trace_start_frame": self.landmarks.takeoff_frame,
            "stance_side": self.stance_side,
            "stance_support_fraction": self.stance_support_fraction,
            "min_stance_support_fraction": self.min_stance_support_fraction,
            "support_window_s": self.support_window_s,
            "support_window_start_frame": self.support_window_start_frame,
            "support_window_end_frame": self.support_window_end_frame,
            "support_height_m": self.support_height_m,
            "support_speed_mps": self.support_speed_mps,
            "max_floor_penetration_m": self.max_floor_penetration_m,
            "landing_dwell_s": self.landing_dwell_s,
            "takeoff_dwell_frames": self.takeoff_dwell_frames,
            "landing_dwell_frames": self.landing_dwell_frames,
            "min_relative_lift_m": self.min_relative_lift_m,
            "measurement_protocol_hash": self.measurement_protocol_hash,
            "common_physical_protocol_hash": self.common_physical_protocol_hash,
            "support_source": self.support_source,
        }


def _event_for_apex(
    kinematics: Mapping[Side, Mapping[str, np.ndarray]],
    support: Mapping[Side, np.ndarray],
    *,
    side: Side,
    apex: int,
    support_window_s: float,
    fps: float,
) -> KinematicStepEvent:
    stance: Side = "right" if side == "left" else "left"
    length = len(support[side])
    first, last = _centered_window(apex, length, support_window_s, fps)
    swing_height = float(kinematics[side]["bottom_clearance_m"][apex])
    stance_height = float(kinematics[stance]["bottom_clearance_m"][apex])
    fraction = float(np.mean(support[stance][first : last + 1]))
    return KinematicStepEvent(
        frame=apex,
        side=side,
        relative_lift_m=swing_height - stance_height,
        swing_height_m=swing_height,
        stance_height_m=stance_height,
        stance_support_fraction=fraction,
        swing_planar_speed_mps=float(kinematics[side]["planar_speed_mps"][apex]),
        stance_planar_speed_mps=float(kinematics[stance]["planar_speed_mps"][apex]),
        support_window_start_frame=first,
        support_window_end_frame=last,
    )


def _check_same_event(observed: KinematicStepEvent, expected: KinematicStepEvent) -> None:
    if observed.frame != expected.frame or observed.side != expected.side:
        raise ValueError("event does not identify the physical swing apex")
    if (
        observed.support_window_start_frame != expected.support_window_start_frame
        or observed.support_window_end_frame != expected.support_window_end_frame
    ):
        raise ValueError("event was not measured with the locked stance-support window")
    for name in (
        "relative_lift_m",
        "swing_height_m",
        "stance_height_m",
        "stance_support_fraction",
        "swing_planar_speed_mps",
        "stance_planar_speed_mps",
    ):
        if not np.isclose(float(getattr(observed, name)), float(getattr(expected, name))):
            raise ValueError(f"event field {name!r} does not match exact foot kinematics")


def _cycle_from_event(
    kinematics: Mapping[Side, Mapping[str, np.ndarray]],
    support: Mapping[Side, np.ndarray],
    event: KinematicStepEvent,
    *,
    fps: float,
    thresholds: StepOverThresholds,
    support_window_s: float,
    min_stance_support_fraction: float,
    min_relative_lift_m: float,
    verify_event: bool,
) -> StepPhaseCycle:
    if not isinstance(event, KinematicStepEvent):
        raise ValueError("event must be a KinematicStepEvent")
    if event.side not in _SIDES:
        raise ValueError("event side must be 'left' or 'right'")
    frame = _strict_int(event.frame, "event frame")
    length = len(support[event.side])
    if frame < 0 or frame >= length:
        raise ValueError("event frame is outside the clip")
    if support[event.side][frame]:
        raise ValueError("event frame is not inside an unsupported swing-foot interval")

    run = next(
        ((start, end) for start, end in _false_runs(support[event.side]) if start <= frame <= end),
        None,
    )
    if run is None:  # Defensive: the event was already found unsupported above.
        raise ValueError("event has no physical swing interval")
    first_air, last_air = run
    if first_air == 0:
        raise ValueError("swing cycle has no observed takeoff before the clip boundary")
    if last_air == length - 1:
        raise ValueError("swing cycle has no observed landing before the clip boundary")
    takeoff, landing = first_air, last_air + 1
    support_dwell_frames = _stable_support_samples(thresholds.landing_dwell_s, fps)
    takeoff_dwell_start = takeoff - support_dwell_frames
    if takeoff_dwell_start < 0:
        raise ValueError("swing takeoff dwell is truncated by the clip boundary")
    if not np.all(support[event.side][takeoff_dwell_start:takeoff]):
        raise ValueError("swing cycle has no stable observed pre-takeoff support dwell")
    landing_dwell_frames = support_dwell_frames
    landing_dwell_end = landing + landing_dwell_frames
    if landing_dwell_end > length:
        raise ValueError("swing landing dwell is truncated by the clip boundary")
    if not np.all(support[event.side][landing:landing_dwell_end]):
        raise ValueError("swing cycle has no stable observed landing dwell")

    stance: Side = "right" if event.side == "left" else "left"
    relative = (
        kinematics[event.side]["bottom_clearance_m"]
        - kinematics[stance]["bottom_clearance_m"]
    )
    apex = first_air + int(np.argmax(relative[first_air : last_air + 1]))
    canonical = _event_for_apex(
        kinematics,
        support,
        side=event.side,
        apex=apex,
        support_window_s=support_window_s,
        fps=fps,
    )
    if verify_event:
        _check_same_event(event, canonical)
    event = canonical

    if event.relative_lift_m < min_relative_lift_m:
        raise ValueError("physical swing apex is below the locked relative-lift threshold")
    phase = _phase_trace(takeoff, apex, landing)

    # A unilateral step must retain at least one support foot on every observed swing frame.
    bilateral_flight = ~(support["left"] | support["right"])
    if np.any(bilateral_flight[takeoff : landing + 1]):
        raise ValueError("swing cycle contains bilateral flight")
    if event.stance_support_fraction < min_stance_support_fraction:
        raise ValueError("contralateral stance support is below the locked threshold")
    lowest = min(
        float(np.min(kinematics[side]["bottom_clearance_m"][takeoff : landing + 1]))
        for side in _SIDES
    )
    if lowest < -thresholds.max_floor_penetration_m:
        raise ValueError("swing cycle exceeds the locked physical-foot penetration bound")

    return StepPhaseCycle(
        method=STEP_PHASE_METHOD,
        event=event,
        landmarks=StepPhaseLandmarks(takeoff, apex, landing),
        fps=fps,
        phase_trace=tuple(float(value) for value in phase),
        stance_side=stance,
        stance_support_fraction=event.stance_support_fraction,
        min_stance_support_fraction=min_stance_support_fraction,
        support_window_s=support_window_s,
        support_window_start_frame=event.support_window_start_frame,
        support_window_end_frame=event.support_window_end_frame,
        support_height_m=thresholds.support_height_m,
        support_speed_mps=thresholds.support_speed_mps,
        max_floor_penetration_m=thresholds.max_floor_penetration_m,
        landing_dwell_s=thresholds.landing_dwell_s,
        takeoff_dwell_frames=support_dwell_frames,
        landing_dwell_frames=landing_dwell_frames,
        min_relative_lift_m=min_relative_lift_m,
        measurement_protocol_hash=step_phase_measurement_protocol_hash(
            thresholds,
            support_window_s=support_window_s,
            min_stance_support_fraction=min_stance_support_fraction,
            min_relative_lift_m=min_relative_lift_m,
        ),
        common_physical_protocol_hash=step_phase_common_physical_protocol_hash(
            thresholds,
            support_window_s=support_window_s,
            min_stance_support_fraction=min_stance_support_fraction,
        ),
    )


def validate_step_phase_cycle(
    kinematics: Mapping[str, Any],
    event: KinematicStepEvent,
    *,
    fps: float,
    thresholds: StepOverThresholds | None = None,
    support_window_s: float = 0.24,
    min_stance_support_fraction: float | None = None,
    min_relative_lift_m: float = 0.04,
) -> StepPhaseCycle:
    """Validate an existing exact kinematic event and derive its phase receipt."""
    fps = _positive_finite(fps, "fps")
    support_window_s = _positive_finite(support_window_s, "support_window_s")
    thresholds = thresholds or StepOverThresholds()
    thresholds.validate()
    threshold_fraction = _unit_fraction(
        thresholds.min_contralateral_support_fraction,
        "thresholds.min_contralateral_support_fraction",
        positive=True,
    )
    if min_stance_support_fraction is None:
        min_stance_support_fraction = threshold_fraction
    else:
        min_stance_support_fraction = _unit_fraction(
            min_stance_support_fraction, "min_stance_support_fraction", positive=True
        )
        if min_stance_support_fraction != threshold_fraction:
            raise ValueError(
                "explicit min_stance_support_fraction conflicts with StepOverThresholds"
            )
    min_relative_lift_m = float(min_relative_lift_m)
    if not np.isfinite(min_relative_lift_m) or min_relative_lift_m < 0.0:
        raise ValueError("min_relative_lift_m must be finite and non-negative")
    exact, _ = _validated_kinematics(kinematics)
    support = _support_masks(exact, thresholds)
    return _cycle_from_event(
        exact,
        support,
        event,
        fps=fps,
        thresholds=thresholds,
        support_window_s=support_window_s,
        min_stance_support_fraction=min_stance_support_fraction,
        min_relative_lift_m=min_relative_lift_m,
        verify_event=True,
    )


def step_phase_cycle_from_qpos(
    body: Any,
    qpos: np.ndarray,
    fps: float,
    event: KinematicStepEvent,
    **kwargs: Any,
) -> StepPhaseCycle:
    """Derive a phase receipt from qpos using exact physical-foot envelopes."""
    return validate_step_phase_cycle(
        foot_kinematics_series(body, qpos, fps), event, fps=fps, **kwargs
    )


def enumerate_step_phase_cycles(
    kinematics: Mapping[str, Any],
    *,
    fps: float,
    frame_window: tuple[int, int] | None = None,
    swing_side: Side | None = None,
    thresholds: StepOverThresholds | None = None,
    support_window_s: float = 0.24,
    min_stance_support_fraction: float | None = None,
    min_relative_lift_m: float = 0.04,
) -> tuple[StepPhaseCycle, ...]:
    """Enumerate complete, physically supported swing cycles in deterministic order.

    The frame window is half-open and filters apex frames.  Invalid unsupported runs are
    never returned.  If no run survives, the function raises with the observed rejection
    reasons rather than returning an unaudited fallback event.
    """
    fps = _positive_finite(fps, "fps")
    support_window_s = _positive_finite(support_window_s, "support_window_s")
    thresholds = thresholds or StepOverThresholds()
    thresholds.validate()
    threshold_fraction = _unit_fraction(
        thresholds.min_contralateral_support_fraction,
        "thresholds.min_contralateral_support_fraction",
        positive=True,
    )
    if min_stance_support_fraction is None:
        min_stance_support_fraction = threshold_fraction
    else:
        min_stance_support_fraction = _unit_fraction(
            min_stance_support_fraction, "min_stance_support_fraction", positive=True
        )
        if min_stance_support_fraction != threshold_fraction:
            raise ValueError(
                "explicit min_stance_support_fraction conflicts with StepOverThresholds"
            )
    min_relative_lift_m = float(min_relative_lift_m)
    if not np.isfinite(min_relative_lift_m) or min_relative_lift_m < 0.0:
        raise ValueError("min_relative_lift_m must be finite and non-negative")
    if swing_side is not None and swing_side not in _SIDES:
        raise ValueError("swing_side must be 'left', 'right', or None")
    exact, length = _validated_kinematics(kinematics)
    support = _support_masks(exact, thresholds)

    if frame_window is None:
        lo, hi = 0, length
    else:
        if len(frame_window) != 2:
            raise ValueError("frame_window must contain exactly two bounds")
        lo = _strict_int(frame_window[0], "frame_window start")
        hi = _strict_int(frame_window[1], "frame_window end")
        if lo < 0 or hi > length or lo >= hi:
            raise ValueError("frame_window must be a non-empty half-open clip interval")

    sides: Sequence[Side] = _SIDES if swing_side is None else (swing_side,)
    cycles: list[StepPhaseCycle] = []
    rejections: list[str] = []
    for side in sides:
        stance: Side = "right" if side == "left" else "left"
        relative = (
            exact[side]["bottom_clearance_m"] - exact[stance]["bottom_clearance_m"]
        )
        for first_air, last_air in _false_runs(support[side]):
            apex = first_air + int(np.argmax(relative[first_air : last_air + 1]))
            if not lo <= apex < hi:
                continue
            try:
                event = _event_for_apex(
                    exact,
                    support,
                    side=side,
                    apex=apex,
                    support_window_s=support_window_s,
                    fps=fps,
                )
                cycles.append(
                    _cycle_from_event(
                        exact,
                        support,
                        event,
                        fps=fps,
                        thresholds=thresholds,
                        support_window_s=support_window_s,
                        min_stance_support_fraction=min_stance_support_fraction,
                        min_relative_lift_m=min_relative_lift_m,
                        verify_event=False,
                    )
                )
            except ValueError as exc:
                rejections.append(f"{side} run {first_air}:{last_air}: {exc}")
    cycles.sort(key=lambda cycle: (cycle.apex_frame, cycle.swing_side))
    if not cycles:
        detail = "; ".join(rejections) if rejections else "no unsupported swing interval"
        raise ValueError(f"no valid complete unilateral swing cycle: {detail}")
    return tuple(cycles)


def enumerate_step_phase_cycles_from_qpos(
    body: Any,
    qpos: np.ndarray,
    fps: float,
    **kwargs: Any,
) -> tuple[StepPhaseCycle, ...]:
    """Enumerate phase receipts after computing exact physical-foot qpos kinematics."""
    return enumerate_step_phase_cycles(
        foot_kinematics_series(body, qpos, fps), fps=fps, **kwargs
    )


def _same_source_lock(adapted: StepPhaseCycle, neutral: StepPhaseCycle) -> None:
    fields = (
        "fps",
        "support_window_s",
        "min_stance_support_fraction",
        "support_height_m",
        "support_speed_mps",
        "max_floor_penetration_m",
        "landing_dwell_s",
        "landing_dwell_frames",
        "min_relative_lift_m",
    )
    if any(getattr(adapted, name) != getattr(neutral, name) for name in fields):
        raise ValueError("adapted and neutral cycles do not share one locked phase protocol")


def align_step_phase_cycles(
    adapted: StepPhaseCycle,
    neutral: StepPhaseCycle,
    *,
    half_window_frames: int,
    max_phase_error: float = 0.05,
    min_common_phase_span_per_side: float = 0.02,
) -> PhaseMatch:
    """Build a RAMP source-pair receipt from two physical step cycles."""
    if not isinstance(adapted, StepPhaseCycle) or not isinstance(neutral, StepPhaseCycle):
        raise ValueError("adapted and neutral must be StepPhaseCycle receipts")
    if adapted.swing_side != neutral.swing_side:
        raise ValueError("adapted and neutral cycles use different swing sides")
    _same_source_lock(adapted, neutral)
    adapted.require_phase_window(half_window_frames)
    neutral.require_phase_window(half_window_frames)
    return align_cyclic_phase_windows(
        adapted.phase_trace,
        neutral.phase_trace,
        adapted_center_frame=adapted.local_apex_frame,
        neutral_center_frame=neutral.local_apex_frame,
        half_window_frames=half_window_frames,
        swing_side=adapted.swing_side,
        adapted_stance=adapted.stance_evidence(absolute_frame=False),
        neutral_stance=neutral.stance_evidence(absolute_frame=False),
        measurement_protocol_hash=adapted.common_physical_protocol_hash,
        min_stance_support_fraction=adapted.min_stance_support_fraction,
        support_window_s=adapted.support_window_s,
        max_phase_error=max_phase_error,
        min_common_phase_span_per_side=min_common_phase_span_per_side,
    )


def align_step_target_phase(
    packet_phase_knots: Sequence[float],
    target: StepPhaseCycle,
    *,
    expected_swing_side: Side,
    source_common_physical_protocol_hash: str,
    search_half_window_frames: int,
    max_phase_error: float = 0.05,
    min_packet_phase_span_per_side: float = 0.02,
) -> TargetPhaseMatch:
    """Place a packet phase grid on a held-out, physically measured target cycle.

    The generic aligner operates on the cycle-local trace.  The returned immutable receipt
    is translated back to the target clip's absolute apex frame for direct consumption by
    :func:`scene2motion.ramp.packet.render_packet`.
    """
    if not isinstance(target, StepPhaseCycle):
        raise ValueError("target must be a StepPhaseCycle receipt")
    if expected_swing_side not in _SIDES:
        raise ValueError("expected_swing_side must be 'left' or 'right'")
    if target.swing_side != expected_swing_side:
        raise ValueError("target cycle swing side does not match the packet")
    source_protocol_hash = _protocol_hash(source_common_physical_protocol_hash)
    if target.common_physical_protocol_hash != source_protocol_hash:
        raise ValueError("target and source use different common physical protocols")
    target.require_phase_window(search_half_window_frames)
    receipt = align_target_phase_window(
        target.phase_trace,
        packet_phase_knots,
        target_center_frame=target.local_apex_frame,
        search_half_window_frames=search_half_window_frames,
        swing_side=expected_swing_side,
        target_stance=target.stance_evidence(absolute_frame=False),
        measurement_protocol_hash=source_protocol_hash,
        min_stance_support_fraction=target.min_stance_support_fraction,
        support_window_s=target.support_window_s,
        max_phase_error=max_phase_error,
        min_packet_phase_span_per_side=min_packet_phase_span_per_side,
    )
    return replace(receipt, target_center_frame=target.apex_frame)


__all__ = [
    "APEX_PHASE",
    "COMMON_STEP_PHASE_PROTOCOL_VERSION",
    "LANDING_PHASE",
    "STANCE_EVIDENCE_METHOD",
    "STEP_PHASE_METHOD",
    "STEP_PHASE_PROTOCOL_VERSION",
    "TAKEOFF_PHASE",
    "StepPhaseCycle",
    "StepPhaseLandmarks",
    "align_step_phase_cycles",
    "align_step_target_phase",
    "enumerate_step_phase_cycles",
    "enumerate_step_phase_cycles_from_qpos",
    "step_phase_cycle_from_qpos",
    "step_phase_common_physical_protocol_hash",
    "step_phase_measurement_protocol_hash",
    "validate_step_phase_cycle",
]
