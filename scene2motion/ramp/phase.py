"""Strict local gait-phase alignment for coherent RAMP packets.

An adapted clip and a neutral clip may traverse the same gait event at different
cadences.  Equal random seeds or equal frame indices do not establish phase
correspondence.  This module instead inverts two measured, locally monotone cyclic
phase traces onto one event-relative phase grid.  Inversion is bounded by the full
source windows: it never extrapolates.

Phases are normalized cycles in ``[0, 1)`` and must advance in the positive
direction.  A large modular frame-to-frame advance is indistinguishable from a
backward step, so the helper rejects it as ambiguous rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .packet import PhaseMatch


Side = Literal["left", "right"]
PHASE_ALIGNMENT_METHOD = "bounded-local-cyclic-inverse-v1"
TARGET_PHASE_ALIGNMENT_METHOD = "bounded-target-cyclic-inverse-v1"


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


def _validate_phase_grid_progression(values: Any, name: str) -> None:
    phase = np.asarray(values, dtype=float)
    forward = np.mod(np.diff(phase), 1.0)
    if np.any(forward <= 1e-8) or np.any(forward >= 0.5):
        raise ValueError(f"{name} must advance strictly and unambiguously")


@dataclass(frozen=True)
class StanceEvidence:
    """Event-local evidence that one named foot is supporting the character.

    ``source`` is deliberately required so callers cannot silently derive stance
    from a seed, nominal frame number, or the phase value being aligned.
    """

    side: Side
    center_frame: int
    support_fraction: float
    source: str

    def __post_init__(self) -> None:
        if self.side not in ("left", "right"):
            raise ValueError("stance-evidence side must be 'left' or 'right'")
        object.__setattr__(
            self, "center_frame", _strict_int(self.center_frame, "stance-evidence center_frame")
        )
        if isinstance(self.support_fraction, (bool, np.bool_)) or not np.isfinite(
            self.support_fraction
        ):
            raise ValueError("stance-evidence support_fraction must be finite")
        if not 0.0 <= self.support_fraction <= 1.0:
            raise ValueError("stance-evidence support_fraction must lie in [0, 1]")
        object.__setattr__(self, "support_fraction", float(self.support_fraction))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("stance-evidence source must be non-empty")


@dataclass(frozen=True)
class TargetPhaseMatch:
    """Immutable receipt for placing one packet phase grid on a target gait."""

    method: str
    target_center_frame: int
    swing_side: Side
    target_query_offsets_frames: tuple[float, ...]
    packet_phase_knots: tuple[float, ...]
    target_phase_knots: tuple[float, ...]
    max_phase_error: float
    target_stance_side: Side
    target_stance_support_fraction: float
    min_stance_support_fraction: float
    support_window_s: float
    target_stance_source: str
    search_half_window_frames: int
    measurement_protocol_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("target phase-match method must be non-empty")
        object.__setattr__(
            self,
            "measurement_protocol_hash",
            _protocol_hash(self.measurement_protocol_hash),
        )
        object.__setattr__(
            self,
            "target_center_frame",
            _strict_int(self.target_center_frame, "target_center_frame"),
        )
        half_window = _strict_int(self.search_half_window_frames, "search_half_window_frames")
        if half_window < 1:
            raise ValueError("search_half_window_frames must be at least one")
        object.__setattr__(self, "search_half_window_frames", half_window)
        if self.swing_side not in ("left", "right") or self.target_stance_side not in (
            "left",
            "right",
        ):
            raise ValueError("target phase-match sides must be 'left' or 'right'")
        expected_stance: Side = "right" if self.swing_side == "left" else "left"
        if self.target_stance_side != expected_stance:
            raise ValueError("target stance is not contralateral to the swing foot")

        for name in (
            "target_query_offsets_frames",
            "packet_phase_knots",
            "target_phase_knots",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, values)
        count = len(self.packet_phase_knots)
        if count < 3 or count % 2 != 1:
            raise ValueError(
                "target phase receipt requires an odd packet grid of at least 3 knots"
            )
        if len(self.target_query_offsets_frames) != count or len(self.target_phase_knots) != count:
            raise ValueError("target phase receipt arrays must have equal lengths")
        query = np.asarray(self.target_query_offsets_frames)
        center_index = count // 2
        if np.any(np.diff(query) <= 0) or not np.isclose(query[center_index], 0.0):
            raise ValueError("target phase queries must increase strictly through offset zero")
        if query[0] < -half_window or query[-1] > half_window:
            raise ValueError("target phase queries exceed the bounded search window")
        for name in ("packet_phase_knots", "target_phase_knots"):
            values = np.asarray(getattr(self, name))
            if np.any((values < 0.0) | (values >= 1.0)):
                raise ValueError(f"{name} must lie in [0, 1)")
            _validate_phase_grid_progression(values, name)

        for name in (
            "max_phase_error",
            "target_stance_support_fraction",
            "min_stance_support_fraction",
            "support_window_s",
        ):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.max_phase_error < 0.5:
            raise ValueError("max_phase_error must lie in [0, 0.5)")
        if not 0.0 <= self.target_stance_support_fraction <= 1.0:
            raise ValueError("target_stance_support_fraction must lie in [0, 1]")
        if not 0.0 < self.min_stance_support_fraction <= 1.0:
            raise ValueError("min_stance_support_fraction must lie in (0, 1]")
        if self.target_stance_support_fraction < self.min_stance_support_fraction:
            raise ValueError("target stance support is below the locked threshold")
        if self.support_window_s <= 0.0:
            raise ValueError("support_window_s must be positive")
        if not isinstance(self.target_stance_source, str) or not self.target_stance_source.strip():
            raise ValueError("target_stance_source must be non-empty")
        if max(self.phase_errors, default=float("inf")) > self.max_phase_error:
            raise ValueError("target phase error exceeds the locked tolerance")

    @property
    def phase_errors(self) -> tuple[float, ...]:
        packet = np.asarray(self.packet_phase_knots)
        target = np.asarray(self.target_phase_knots)
        raw = np.abs(packet - target)
        return tuple(float(value) for value in np.minimum(raw, 1.0 - raw))

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target_center_frame": self.target_center_frame,
            "swing_side": self.swing_side,
            "target_query_offsets_frames": list(self.target_query_offsets_frames),
            "packet_phase_knots": list(self.packet_phase_knots),
            "target_phase_knots": list(self.target_phase_knots),
            "phase_errors": list(self.phase_errors),
            "max_phase_error": self.max_phase_error,
            "target_stance_side": self.target_stance_side,
            "target_stance_support_fraction": self.target_stance_support_fraction,
            "min_stance_support_fraction": self.min_stance_support_fraction,
            "support_window_s": self.support_window_s,
            "target_stance_source": self.target_stance_source,
            "search_half_window_frames": self.search_half_window_frames,
            "measurement_protocol_hash": self.measurement_protocol_hash,
        }


def _phase_array(value: Any, name: str) -> np.ndarray:
    phase = np.asarray(value, dtype=float)
    if phase.ndim != 1 or len(phase) < 3:
        raise ValueError(f"{name} must be a one-dimensional array with at least three frames")
    if not np.isfinite(phase).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any((phase < 0.0) | (phase >= 1.0)):
        raise ValueError(f"{name} must contain normalized cyclic phases in [0, 1)")
    return phase


def _local_relative_phase(
    phase: np.ndarray,
    *,
    center: int,
    half_window: int,
    min_advance: float,
    max_advance: float,
    name: str,
) -> np.ndarray:
    first = center - half_window
    last = center + half_window
    if first < 0 or last >= len(phase):
        raise ValueError(f"{name} does not contain the requested full phase window")

    local = phase[first : last + 1]
    forward = np.mod(np.diff(local), 1.0)
    if np.any(forward <= min_advance):
        raise ValueError(f"{name} is locally non-monotonic or phase-stalled")
    if np.any(forward > max_advance):
        raise ValueError(f"{name} has a backward or phase-ambiguous local transition")

    unwrapped = np.concatenate(([0.0], np.cumsum(forward)))
    return unwrapped - unwrapped[half_window]


def _cyclic_error(left: float, right: float) -> float:
    delta = abs(float(left) - float(right))
    return min(delta, 1.0 - delta)


def _query_offsets(relative_phase: np.ndarray, knots: np.ndarray, half_window: int) -> np.ndarray:
    frame_offsets = np.arange(-half_window, half_window + 1, dtype=float)
    query = np.interp(knots, relative_phase, frame_offsets)
    query[len(knots) // 2] = 0.0
    return query


def align_target_phase_window(
    target_phase: Any,
    packet_phase_knots: Any,
    *,
    target_center_frame: int,
    search_half_window_frames: int,
    swing_side: Side,
    target_stance: StanceEvidence,
    measurement_protocol_hash: str,
    min_stance_support_fraction: float = 0.8,
    support_window_s: float = 0.12,
    max_phase_error: float = 0.05,
    min_phase_advance_per_frame: float = 1e-6,
    max_phase_advance_per_frame: float = 0.25,
    min_packet_phase_span_per_side: float = 0.02,
) -> TargetPhaseMatch:
    """Invert a measured target gait onto an existing packet phase grid.

    ``packet_phase_knots`` should be the canonical/adapted knot sequence archived
    with the source packet.  The target trace is inspected only inside the full
    event-centred search window.  Every packet knot must fall inside that measured
    phase support; extrapolation, nearest-frame substitution, and seed-based phase
    assumptions are rejected.
    """

    protocol_hash = _protocol_hash(measurement_protocol_hash)
    target = _phase_array(target_phase, "target_phase")
    packet_grid = _phase_array(packet_phase_knots, "packet_phase_knots")
    if len(packet_grid) % 2 != 1:
        raise ValueError("packet_phase_knots must contain an odd number of knots")
    packet_center_index = len(packet_grid) // 2
    target_center = _strict_int(target_center_frame, "target_center_frame")
    search_half_window = _strict_int(
        search_half_window_frames, "search_half_window_frames"
    )
    if search_half_window < 1:
        raise ValueError("search_half_window_frames must be at least one")

    if swing_side not in ("left", "right"):
        raise ValueError("swing_side must be 'left' or 'right'")
    if not isinstance(target_stance, StanceEvidence):
        raise ValueError("target_stance must be explicit StanceEvidence")
    if target_stance.center_frame != target_center:
        raise ValueError("target stance evidence is for a different event frame")
    expected_stance: Side = "right" if swing_side == "left" else "left"
    if target_stance.side != expected_stance:
        raise ValueError("target stance evidence is not contralateral to the swing foot")

    for value, name in (
        (max_phase_error, "max_phase_error"),
        (min_phase_advance_per_frame, "min_phase_advance_per_frame"),
        (max_phase_advance_per_frame, "max_phase_advance_per_frame"),
        (min_packet_phase_span_per_side, "min_packet_phase_span_per_side"),
        (min_stance_support_fraction, "min_stance_support_fraction"),
        (support_window_s, "support_window_s"),
    ):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_phase_error < 0.5:
        raise ValueError("max_phase_error must lie in [0, 0.5)")
    if min_phase_advance_per_frame < 0.0:
        raise ValueError("min_phase_advance_per_frame must be non-negative")
    if not min_phase_advance_per_frame < max_phase_advance_per_frame < 0.5:
        raise ValueError(
            "max_phase_advance_per_frame must lie between the minimum advance and 0.5"
        )
    if min_packet_phase_span_per_side <= 0.0:
        raise ValueError("min_packet_phase_span_per_side must be positive")
    if not 0.0 < min_stance_support_fraction <= 1.0:
        raise ValueError("min_stance_support_fraction must lie in (0, 1]")
    if target_stance.support_fraction < min_stance_support_fraction:
        raise ValueError(
            f"target stance support {target_stance.support_fraction:.4f} is below "
            f"the locked threshold {min_stance_support_fraction:.4f}"
        )
    if support_window_s <= 0.0:
        raise ValueError("support_window_s must be positive")

    packet_relative = _local_relative_phase(
        packet_grid,
        center=packet_center_index,
        half_window=packet_center_index,
        min_advance=min_phase_advance_per_frame,
        max_advance=max_phase_advance_per_frame,
        name="packet_phase_knots",
    )
    target_relative = _local_relative_phase(
        target,
        center=target_center,
        half_window=search_half_window,
        min_advance=min_phase_advance_per_frame,
        max_advance=max_phase_advance_per_frame,
        name="target_phase",
    )
    if (
        -packet_relative[0] < min_packet_phase_span_per_side
        or packet_relative[-1] < min_packet_phase_span_per_side
    ):
        raise ValueError("packet phase grid has insufficient event context")
    tolerance = 1e-12
    if (
        packet_relative[0] < target_relative[0] - tolerance
        or packet_relative[-1] > target_relative[-1] + tolerance
    ):
        raise ValueError("target window has insufficient phase overlap for the packet grid")

    center_error = _cyclic_error(
        packet_grid[packet_center_index], target[target_center]
    )
    if center_error > max_phase_error:
        raise ValueError(
            f"target event-center phase error {center_error:.4f} exceeds "
            f"{max_phase_error:.4f}"
        )
    target_query = _query_offsets(
        target_relative, packet_relative, search_half_window
    )
    target_knots = np.mod(target[target_center] + packet_relative, 1.0)
    return TargetPhaseMatch(
        method=TARGET_PHASE_ALIGNMENT_METHOD,
        target_center_frame=target_center,
        swing_side=swing_side,
        target_query_offsets_frames=tuple(float(value) for value in target_query),
        packet_phase_knots=tuple(float(value) for value in packet_grid),
        target_phase_knots=tuple(float(value) for value in target_knots),
        max_phase_error=float(max_phase_error),
        target_stance_side=target_stance.side,
        target_stance_support_fraction=target_stance.support_fraction,
        min_stance_support_fraction=float(min_stance_support_fraction),
        support_window_s=float(support_window_s),
        target_stance_source=target_stance.source,
        search_half_window_frames=search_half_window,
        measurement_protocol_hash=protocol_hash,
    )


def align_cyclic_phase_windows(
    adapted_phase: Any,
    neutral_phase: Any,
    *,
    adapted_center_frame: int,
    neutral_center_frame: int,
    half_window_frames: int,
    swing_side: Side,
    adapted_stance: StanceEvidence,
    neutral_stance: StanceEvidence,
    measurement_protocol_hash: str,
    min_stance_support_fraction: float = 0.8,
    support_window_s: float = 0.12,
    max_phase_error: float = 0.05,
    min_phase_advance_per_frame: float = 1e-6,
    max_phase_advance_per_frame: float = 0.25,
    min_common_phase_span_per_side: float = 0.02,
) -> PhaseMatch:
    """Align two event windows through measured cyclic phase.

    A common set of event-relative phase knots is selected from the intersection
    of the two bounded windows.  Each locally unwrapped trajectory is inverted to
    fractional frame offsets at those knots.  Small center-phase disagreement is
    retained in the returned audit fields; it is not hidden by re-labelling either
    trace.

    The function fails closed if either full window is unavailable, the phase trace
    cannot be unwrapped uniquely and monotonically, the events disagree in phase,
    the common phase support is too small, or contralateral stance contact is not
    explicitly evidenced for both events.
    """

    protocol_hash = _protocol_hash(measurement_protocol_hash)
    adapted = _phase_array(adapted_phase, "adapted_phase")
    neutral = _phase_array(neutral_phase, "neutral_phase")
    adapted_center = _strict_int(adapted_center_frame, "adapted_center_frame")
    neutral_center = _strict_int(neutral_center_frame, "neutral_center_frame")
    half_window = _strict_int(half_window_frames, "half_window_frames")
    if half_window < 1:
        raise ValueError("half_window_frames must be at least one")

    if swing_side not in ("left", "right"):
        raise ValueError("swing_side must be 'left' or 'right'")
    expected_stance: Side = "right" if swing_side == "left" else "left"
    for name, evidence, center in (
        ("adapted", adapted_stance, adapted_center),
        ("neutral", neutral_stance, neutral_center),
    ):
        if not isinstance(evidence, StanceEvidence):
            raise ValueError(f"{name}_stance must be explicit StanceEvidence")
        if evidence.center_frame != center:
            raise ValueError(f"{name} stance evidence is for a different event frame")
        if evidence.side != expected_stance:
            raise ValueError(f"{name} stance evidence is not contralateral to the swing foot")

    for value, name in (
        (max_phase_error, "max_phase_error"),
        (min_phase_advance_per_frame, "min_phase_advance_per_frame"),
        (max_phase_advance_per_frame, "max_phase_advance_per_frame"),
        (min_common_phase_span_per_side, "min_common_phase_span_per_side"),
        (min_stance_support_fraction, "min_stance_support_fraction"),
        (support_window_s, "support_window_s"),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 <= max_phase_error < 0.5:
        raise ValueError("max_phase_error must lie in [0, 0.5)")
    if min_phase_advance_per_frame < 0.0:
        raise ValueError("min_phase_advance_per_frame must be non-negative")
    if not min_phase_advance_per_frame < max_phase_advance_per_frame < 0.5:
        raise ValueError(
            "max_phase_advance_per_frame must lie between the minimum advance and 0.5"
        )
    if min_common_phase_span_per_side <= 0.0:
        raise ValueError("min_common_phase_span_per_side must be positive")
    if not 0.0 < min_stance_support_fraction <= 1.0:
        raise ValueError("min_stance_support_fraction must lie in (0, 1]")
    if support_window_s <= 0.0:
        raise ValueError("support_window_s must be positive")
    for name, evidence in (("adapted", adapted_stance), ("neutral", neutral_stance)):
        if evidence.support_fraction < min_stance_support_fraction:
            raise ValueError(
                f"{name} stance support {evidence.support_fraction:.4f} is below "
                f"the locked threshold {min_stance_support_fraction:.4f}"
            )

    adapted_relative = _local_relative_phase(
        adapted,
        center=adapted_center,
        half_window=half_window,
        min_advance=min_phase_advance_per_frame,
        max_advance=max_phase_advance_per_frame,
        name="adapted_phase",
    )
    neutral_relative = _local_relative_phase(
        neutral,
        center=neutral_center,
        half_window=half_window,
        min_advance=min_phase_advance_per_frame,
        max_advance=max_phase_advance_per_frame,
        name="neutral_phase",
    )

    center_error = _cyclic_error(adapted[adapted_center], neutral[neutral_center])
    if center_error > max_phase_error:
        raise ValueError(
            f"event-center phase error {center_error:.4f} exceeds {max_phase_error:.4f}"
        )

    left_span = min(-adapted_relative[0], -neutral_relative[0])
    right_span = min(adapted_relative[-1], neutral_relative[-1])
    if (
        left_span < min_common_phase_span_per_side
        or right_span < min_common_phase_span_per_side
    ):
        raise ValueError("adapted and neutral windows have insufficient common phase overlap")

    negative = np.linspace(-left_span, 0.0, half_window + 1)[:-1]
    positive = np.linspace(0.0, right_span, half_window + 1)
    relative_knots = np.concatenate((negative, positive))
    adapted_query = _query_offsets(adapted_relative, relative_knots, half_window)
    neutral_query = _query_offsets(neutral_relative, relative_knots, half_window)

    adapted_knots = np.mod(adapted[adapted_center] + relative_knots, 1.0)
    neutral_knots = np.mod(neutral[neutral_center] + relative_knots, 1.0)
    match = PhaseMatch(
        method=PHASE_ALIGNMENT_METHOD,
        adapted_query_offsets_frames=tuple(float(value) for value in adapted_query),
        neutral_query_offsets_frames=tuple(float(value) for value in neutral_query),
        adapted_phase_knots=tuple(float(value) for value in adapted_knots),
        neutral_phase_knots=tuple(float(value) for value in neutral_knots),
        max_phase_error=float(max_phase_error),
        adapted_stance_side=adapted_stance.side,
        neutral_stance_side=neutral_stance.side,
        adapted_stance_source=adapted_stance.source,
        neutral_stance_source=neutral_stance.source,
        adapted_stance_support_fraction=adapted_stance.support_fraction,
        neutral_stance_support_fraction=neutral_stance.support_fraction,
        min_stance_support_fraction=float(min_stance_support_fraction),
        support_window_s=float(support_window_s),
        measurement_protocol_hash=protocol_hash,
    )
    source_offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    match.validate(source_offsets, swing_side)
    return match


__all__ = [
    "PHASE_ALIGNMENT_METHOD",
    "TARGET_PHASE_ALIGNMENT_METHOD",
    "StanceEvidence",
    "TargetPhaseMatch",
    "align_cyclic_phase_windows",
    "align_target_phase_window",
]
