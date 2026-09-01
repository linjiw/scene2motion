"""Phase-align a fixed spatial route with a measured gait event.

The route geometry is immutable.  V1 accepts an arbitrary-heading, forward-collinear
``root_xz`` polyline and changes only the rate at which scalar arc-length progress is
traversed so that one predeclared event frame reaches one predeclared root-progress anchor.
The schedule is a deterministic, solver-free C1 cubic Hermite curve through three anchors.
Its endpoint slopes equal the adjacent segment means and its event slope is their
arithmetic mean.

All feasibility checks fail closed.  In particular, satisfying the three anchors is not
enough: both analytic continuous derivatives and the sampled frame-rate schedule must
respect the supplied timing bounds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


ROUTE_PROGRESS_SCHEMA_VERSION = "ramp-route-progress-program-v1"
ROUTE_PROGRESS_METHOD = "three-anchor-c1-hermite-v1"
ROUTE_PROGRESS_SLOPE_POLICY = "adjacent-segment-means-arithmetic-event-v1"
ROUTE_GEOMETRY_CONVENTION = "forward-collinear-root-xz-polyline-v1"
ROUTE_DERIVATIVE_CONVENTION = "scalar-arc-length-route-progress-v1"
_DENSE_SUBSTEPS_PER_FRAME = 16
_ATOL = 1e-10


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode())
    hasher.update(str(array.dtype).encode())
    hasher.update(_canonical_json(array.shape).encode())
    hasher.update(array.tobytes())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class RouteTimingBounds:
    """Calibrated frame-rate and derivative limits for route reparameterisation."""

    fps: float
    min_discrete_route_progress_speed_mps: float
    max_discrete_route_progress_speed_mps: float
    max_abs_route_progress_acceleration_mps2: float
    max_abs_discrete_route_progress_jerk_mps3: float | None = None
    reference_route_progress_speed_mps: float | None = None
    max_endpoint_route_progress_speed_deviation_mps: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "fps",
            "min_discrete_route_progress_speed_mps",
            "max_discrete_route_progress_speed_mps",
            "max_abs_route_progress_acceleration_mps2",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if self.min_discrete_route_progress_speed_mps < 0.0:
            raise ValueError("min_discrete_route_progress_speed_mps must be nonnegative")
        if (
            self.max_discrete_route_progress_speed_mps
            <= self.min_discrete_route_progress_speed_mps
        ):
            raise ValueError(
                "maximum discrete route-progress speed must exceed its minimum"
            )
        if self.max_abs_route_progress_acceleration_mps2 <= 0.0:
            raise ValueError("max_abs_route_progress_acceleration_mps2 must be positive")

        for name in (
            "max_abs_discrete_route_progress_jerk_mps3",
            "reference_route_progress_speed_mps",
            "max_endpoint_route_progress_speed_deviation_mps",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_float(value, name))
        if (
            self.max_abs_discrete_route_progress_jerk_mps3 is not None
            and self.max_abs_discrete_route_progress_jerk_mps3 <= 0.0
        ):
            raise ValueError(
                "max_abs_discrete_route_progress_jerk_mps3 must be positive"
            )
        if (
            self.reference_route_progress_speed_mps is not None
            and self.reference_route_progress_speed_mps <= 0.0
        ):
            raise ValueError("reference_route_progress_speed_mps must be positive")
        if self.max_endpoint_route_progress_speed_deviation_mps is not None:
            if self.max_endpoint_route_progress_speed_deviation_mps < 0.0:
                raise ValueError(
                    "max_endpoint_route_progress_speed_deviation_mps must be nonnegative"
                )
            if self.reference_route_progress_speed_mps is None:
                raise ValueError(
                    "an endpoint route-progress speed deviation bound requires "
                    "reference_route_progress_speed_mps"
                )

    def as_dict(self) -> dict[str, float | None]:
        return {
            "fps": self.fps,
            "min_discrete_route_progress_speed_mps": (
                self.min_discrete_route_progress_speed_mps
            ),
            "max_discrete_route_progress_speed_mps": (
                self.max_discrete_route_progress_speed_mps
            ),
            "max_abs_route_progress_acceleration_mps2": (
                self.max_abs_route_progress_acceleration_mps2
            ),
            "max_abs_discrete_route_progress_jerk_mps3": (
                self.max_abs_discrete_route_progress_jerk_mps3
            ),
            "reference_route_progress_speed_mps": (
                self.reference_route_progress_speed_mps
            ),
            "max_endpoint_route_progress_speed_deviation_mps": (
                self.max_endpoint_route_progress_speed_deviation_mps
            ),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class RouteProgressProgram:
    """Immutable dense route schedule suitable for both RAMP representation arms."""

    route_xz: tuple[tuple[float, float], ...]
    route_cumulative_progress_m: tuple[float, ...]
    progress_m: tuple[float, ...]
    root_xz: tuple[tuple[float, float], ...]
    route_heading_rad: tuple[float, ...]
    event_frame: int
    event_root_progress_m: float
    route_length_m: float
    segment_mean_route_progress_speeds_mps: tuple[float, float]
    anchor_route_progress_slopes_mps: tuple[float, float, float]
    continuous_interval_route_progress_speed_ranges_mps: tuple[
        tuple[float, float], tuple[float, float]
    ]
    continuous_interval_route_progress_acceleration_endpoints_mps2: tuple[
        tuple[float, float], tuple[float, float]
    ]
    continuous_route_progress_speed_range_mps: tuple[float, float]
    max_abs_continuous_route_progress_acceleration_mps2: float
    discrete_route_progress_speed_range_mps: tuple[float, float]
    max_abs_discrete_route_progress_acceleration_mps2: float
    max_abs_discrete_route_progress_jerk_mps3: float | None
    endpoint_route_progress_speed_deviation_mps: float | None
    mean_abs_progress_deformation_m: float
    integrated_abs_progress_deformation_m_s: float
    normalized_progress_deformation: float
    rms_route_progress_speed_deviation_mps: float
    max_path_projection_error_m: float
    path_projection_tolerance_m: float
    timing_bounds: RouteTimingBounds
    route_hash: str
    schema_version: str = ROUTE_PROGRESS_SCHEMA_VERSION
    method: str = ROUTE_PROGRESS_METHOD
    slope_policy: str = ROUTE_PROGRESS_SLOPE_POLICY
    route_geometry_convention: str = ROUTE_GEOMETRY_CONVENTION
    route_derivative_convention: str = ROUTE_DERIVATIVE_CONVENTION
    caller_foot_centered_anchor_equation: str = (
        "event_root_progress_m = obstacle_progress_m - nominal_foot_forward_offset_m"
    )
    progress_continuity: str = "C1 scalar route progress at the event"
    acceleration_continuity: str = (
        "piecewise-continuous scalar route-progress acceleration; may jump at the event"
    )
    discrete_route_progress_jerk_definition: str = (
        "first difference of discrete scalar route-progress acceleration scaled by fps; "
        "equivalently second difference of speed scaled by fps^2"
    )
    forward_heading_convention: str = "atan2(delta_root_x,delta_root_z)"
    dense_validation_substeps_per_frame: int = _DENSE_SUBSTEPS_PER_FRAME
    packet_center_shift_frames: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.progress_m)

    @property
    def selection_cost(self) -> tuple[float, float, float]:
        """Outcome-free cost prefix for deterministic target-cycle selection."""
        return (
            self.normalized_progress_deformation,
            self.rms_route_progress_speed_deviation_mps,
            self.max_abs_discrete_route_progress_acceleration_mps2,
        )

    def _identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "slope_policy": self.slope_policy,
            "packet_center_shift_frames": self.packet_center_shift_frames,
            "event_frame": self.event_frame,
            "event_root_progress_m": self.event_root_progress_m,
            "route_length_m": self.route_length_m,
            "segment_mean_route_progress_speeds_mps": list(
                self.segment_mean_route_progress_speeds_mps
            ),
            "anchor_route_progress_slopes_mps": list(
                self.anchor_route_progress_slopes_mps
            ),
            "continuous_interval_route_progress_speed_ranges_mps": [
                list(value)
                for value in self.continuous_interval_route_progress_speed_ranges_mps
            ],
            "continuous_interval_route_progress_acceleration_endpoints_mps2": [
                list(value)
                for value in (
                    self.continuous_interval_route_progress_acceleration_endpoints_mps2
                )
            ],
            "continuous_route_progress_speed_range_mps": list(
                self.continuous_route_progress_speed_range_mps
            ),
            "max_abs_continuous_route_progress_acceleration_mps2": (
                self.max_abs_continuous_route_progress_acceleration_mps2
            ),
            "discrete_route_progress_speed_range_mps": list(
                self.discrete_route_progress_speed_range_mps
            ),
            "max_abs_discrete_route_progress_acceleration_mps2": (
                self.max_abs_discrete_route_progress_acceleration_mps2
            ),
            "max_abs_discrete_route_progress_jerk_mps3": (
                self.max_abs_discrete_route_progress_jerk_mps3
            ),
            "endpoint_route_progress_speed_deviation_mps": (
                self.endpoint_route_progress_speed_deviation_mps
            ),
            "mean_abs_progress_deformation_m": self.mean_abs_progress_deformation_m,
            "integrated_abs_progress_deformation_m_s": (
                self.integrated_abs_progress_deformation_m_s
            ),
            "normalized_progress_deformation": self.normalized_progress_deformation,
            "rms_route_progress_speed_deviation_mps": (
                self.rms_route_progress_speed_deviation_mps
            ),
            "max_path_projection_error_m": self.max_path_projection_error_m,
            "path_projection_tolerance_m": self.path_projection_tolerance_m,
            "route_geometry_convention": self.route_geometry_convention,
            "route_derivative_convention": self.route_derivative_convention,
            "caller_foot_centered_anchor_equation": (
                self.caller_foot_centered_anchor_equation
            ),
            "progress_continuity": self.progress_continuity,
            "acceleration_continuity": self.acceleration_continuity,
            "discrete_route_progress_jerk_definition": (
                self.discrete_route_progress_jerk_definition
            ),
            "forward_heading_convention": self.forward_heading_convention,
            "selection_cost_components": {
                "normalized_progress_deformation": self.normalized_progress_deformation,
                "rms_route_progress_speed_deviation_mps": (
                    self.rms_route_progress_speed_deviation_mps
                ),
                "max_abs_discrete_route_progress_acceleration_mps2": (
                    self.max_abs_discrete_route_progress_acceleration_mps2
                ),
            },
            "selection_cost": list(self.selection_cost),
            "timing_bounds": self.timing_bounds.as_dict(),
            "timing_bounds_hash": self.timing_bounds.digest(),
            "route_hash": self.route_hash,
            "dense_validation_substeps_per_frame": (
                self.dense_validation_substeps_per_frame
            ),
        }

    def digest(self) -> str:
        hasher = hashlib.sha256(_canonical_json(self._identity()).encode())
        _hash_array(hasher, "route_xz", np.asarray(self.route_xz, dtype=np.float64))
        _hash_array(
            hasher,
            "route_cumulative_progress_m",
            np.asarray(self.route_cumulative_progress_m, dtype=np.float64),
        )
        _hash_array(hasher, "progress_m", np.asarray(self.progress_m, dtype=np.float64))
        _hash_array(hasher, "root_xz", np.asarray(self.root_xz, dtype=np.float64))
        _hash_array(
            hasher,
            "route_heading_rad",
            np.asarray(self.route_heading_rad, dtype=np.float64),
        )
        return hasher.hexdigest()

    def __hash__(self) -> int:
        return int(self.digest()[:16], 16)

    def diagnostics(self) -> dict[str, Any]:
        """Return a canonical-JSON-safe feasibility and identity receipt."""
        return {
            **self._identity(),
            "program_hash": self.digest(),
            "n_frames": self.n_frames,
            "selection_cost": list(self.selection_cost),
        }

    def diagnostics_json(self) -> str:
        return _canonical_json(self.diagnostics())


def route_progress_selection_key(
    program: RouteProgressProgram, candidate_digest: str
) -> tuple[float, float, float, int, str, str]:
    """Stable outcome-free tie-break key for otherwise feasible cycle candidates."""
    if not isinstance(program, RouteProgressProgram):
        raise TypeError("program must be a RouteProgressProgram")
    if not _is_sha256(candidate_digest):
        raise ValueError("candidate_digest must be a lowercase SHA-256 digest")
    return (
        *program.selection_cost,
        program.event_frame,
        candidate_digest,
        program.digest(),
    )


def _validate_route(route_xz: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    route = np.asarray(route_xz, dtype=float)
    if route.ndim != 2 or route.shape[1:] != (2,) or len(route) < 2:
        raise ValueError("route_xz must have shape (V, 2) with at least two vertices")
    if not np.isfinite(route).all():
        raise ValueError("route_xz must contain only finite values")
    route = np.array(route, dtype=np.float64, copy=True)
    scale = max(1.0, float(np.max(np.abs(route))))
    tol = 1e-12 * scale
    for first in range(len(route)):
        distance = np.linalg.norm(route[first + 1 :] - route[first], axis=1)
        if np.any(distance <= tol):
            raise ValueError("route_xz contains duplicate vertices")
    delta = np.diff(route, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= tol):
        raise ValueError("route_xz contains a zero-length segment")
    direction = delta[0] / lengths[0]
    unit = delta / lengths[:, None]
    collinearity_error = np.abs(unit[:, 0] * direction[1] - unit[:, 1] * direction[0])
    if np.any(collinearity_error > 1e-10):
        raise ValueError(
            "route_xz must be forward-collinear in route-progress v1"
        )
    if np.any(unit @ direction <= 0.0):
        raise ValueError("route_xz must not reverse direction")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return route, lengths, cumulative


def _hermite_interval(
    query_frames: np.ndarray,
    *,
    first_frame: int,
    last_frame: int,
    first_progress: float,
    last_progress: float,
    first_slope_mps: float,
    last_slope_mps: float,
    fps: float,
) -> np.ndarray:
    duration_s = (last_frame - first_frame) / fps
    u = (query_frames - first_frame) / (last_frame - first_frame)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return (
        h00 * first_progress
        + h10 * duration_s * first_slope_mps
        + h01 * last_progress
        + h11 * duration_s * last_slope_mps
    )


def _derivative_extrema(
    *, mean_speed: float, first_slope: float, last_slope: float, duration_s: float
) -> tuple[float, float, tuple[float, float]]:
    # v(u) = A*u^2 + B*u + C, where u is normalized interval time.
    a = -6.0 * mean_speed + 3.0 * first_slope + 3.0 * last_slope
    b = 6.0 * mean_speed - 4.0 * first_slope - 2.0 * last_slope
    c = first_slope
    queries = [0.0, 1.0]
    if abs(a) > np.finfo(float).eps:
        stationary = -b / (2.0 * a)
        if 0.0 < stationary < 1.0:
            queries.append(float(stationary))
    velocity = np.asarray([a * u * u + b * u + c for u in queries])
    acceleration = (float(b / duration_s), float((2.0 * a + b) / duration_s))
    return (
        float(np.min(velocity)),
        float(np.max(velocity)),
        acceleration,
    )


def _route_points(
    progress: np.ndarray,
    route: np.ndarray,
    lengths: np.ndarray,
    cumulative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    segment = np.searchsorted(cumulative, progress, side="right") - 1
    segment = np.clip(segment, 0, len(lengths) - 1)
    fraction = (progress - cumulative[segment]) / lengths[segment]
    points = route[segment] + fraction[:, None] * (route[segment + 1] - route[segment])
    tangent = (route[segment + 1] - route[segment]) / lengths[segment, None]
    # ARDY root_xz: +Z forward and +X right, hence atan2(dx, dz).
    heading = np.unwrap(np.arctan2(tangent[:, 0], tangent[:, 1]))
    return points, heading


def _max_projection_error(points: np.ndarray, route: np.ndarray) -> float:
    best = np.full(len(points), np.inf)
    for first, last in zip(route[:-1], route[1:]):
        delta = last - first
        alpha = np.clip(((points - first) @ delta) / float(delta @ delta), 0.0, 1.0)
        projected = first + alpha[:, None] * delta
        best = np.minimum(best, np.linalg.norm(points - projected, axis=1))
    return float(np.max(best))


def _check_range(value: float, low: float, high: float, name: str) -> None:
    if value < low - _ATOL or value > high + _ATOL:
        raise ValueError(
            f"{name} violates the calibrated scalar route-progress speed bounds"
        )


def reparameterize_route_progress(
    route_xz: Any,
    *,
    n_frames: int,
    event_frame: int,
    event_root_progress_m: float,
    timing_bounds: RouteTimingBounds,
    path_projection_tolerance_m: float = 1e-9,
) -> RouteProgressProgram:
    """Build a deterministic phase-aligned dense route schedule.

    ``event_root_progress_m`` is arc length from the first route vertex.  For a
    foot-centered obstacle, the caller must first subtract the nominal foot-forward offset
    from obstacle progress.  The geometry-only primitive intentionally does not infer that
    experiment-specific convention.
    """
    if not isinstance(timing_bounds, RouteTimingBounds):
        raise TypeError("timing_bounds must be a validated RouteTimingBounds")
    frames = _strict_int(n_frames, "n_frames")
    event = _strict_int(event_frame, "event_frame")
    if frames < 3:
        raise ValueError("n_frames must be at least three")
    if not 0 < event < frames - 1:
        raise ValueError("event_frame must lie strictly inside the clip")
    event_progress = _finite_float(event_root_progress_m, "event_root_progress_m")
    projection_tolerance = _finite_float(
        path_projection_tolerance_m, "path_projection_tolerance_m"
    )
    if projection_tolerance < 0.0:
        raise ValueError("path_projection_tolerance_m must be nonnegative")

    route, lengths, cumulative = _validate_route(route_xz)
    route_length = float(cumulative[-1])
    if not 0.0 < event_progress < route_length:
        raise ValueError("event_root_progress_m must lie strictly inside the route")

    fps = timing_bounds.fps
    durations = (event / fps, (frames - 1 - event) / fps)
    means = (
        event_progress / durations[0],
        (route_length - event_progress) / durations[1],
    )
    for index, mean in enumerate(means):
        _check_range(
            mean,
            timing_bounds.min_discrete_route_progress_speed_mps,
            timing_bounds.max_discrete_route_progress_speed_mps,
            f"necessary segment-{index} mean speed",
        )

    slopes = (means[0], 0.5 * (means[0] + means[1]), means[1])
    first_frames = np.arange(event + 1, dtype=float)
    second_frames = np.arange(event, frames, dtype=float)
    first_progress = _hermite_interval(
        first_frames,
        first_frame=0,
        last_frame=event,
        first_progress=0.0,
        last_progress=event_progress,
        first_slope_mps=slopes[0],
        last_slope_mps=slopes[1],
        fps=fps,
    )
    second_progress = _hermite_interval(
        second_frames,
        first_frame=event,
        last_frame=frames - 1,
        first_progress=event_progress,
        last_progress=route_length,
        first_slope_mps=slopes[1],
        last_slope_mps=slopes[2],
        fps=fps,
    )
    progress = np.concatenate([first_progress[:-1], second_progress])
    if not np.isfinite(progress).all():
        raise ValueError("Hermite route progress is nonfinite")
    if not (
        abs(progress[0]) <= _ATOL
        and abs(progress[event] - event_progress) <= _ATOL
        and abs(progress[-1] - route_length) <= _ATOL
    ):
        raise ValueError("Hermite route progress does not preserve its anchors")
    derivative = [
        _derivative_extrema(
            mean_speed=means[index],
            first_slope=slopes[index],
            last_slope=slopes[index + 1],
            duration_s=durations[index],
        )
        for index in range(2)
    ]
    continuous_min = min(value[0] for value in derivative)
    continuous_max = max(value[1] for value in derivative)
    continuous_acceleration = max(
        abs(acceleration)
        for value in derivative
        for acceleration in value[2]
    )
    _check_range(
        continuous_min,
        timing_bounds.min_discrete_route_progress_speed_mps,
        timing_bounds.max_discrete_route_progress_speed_mps,
        "analytic continuous minimum speed",
    )
    _check_range(
        continuous_max,
        timing_bounds.min_discrete_route_progress_speed_mps,
        timing_bounds.max_discrete_route_progress_speed_mps,
        "analytic continuous maximum speed",
    )
    if continuous_min <= 0.0:
        raise ValueError("analytic Hermite derivative reverses or stalls")
    if (
        continuous_acceleration
        > timing_bounds.max_abs_route_progress_acceleration_mps2 + _ATOL
    ):
        raise ValueError("analytic continuous acceleration exceeds the calibrated bound")
    if np.any(np.diff(progress) <= 0.0):
        raise ValueError("Hermite route progress is not strictly monotone")

    # Independent dense validation catches implementation errors between integer frames.
    for index, (first, last) in enumerate(((0, event), (event, frames - 1))):
        query = np.linspace(
            first,
            last,
            (last - first) * _DENSE_SUBSTEPS_PER_FRAME + 1,
        )
        dense = _hermite_interval(
            query,
            first_frame=first,
            last_frame=last,
            first_progress=(0.0, event_progress)[index],
            last_progress=(event_progress, route_length)[index],
            first_slope_mps=slopes[index],
            last_slope_mps=slopes[index + 1],
            fps=fps,
        )
        low, high = (0.0, event_progress) if index == 0 else (
            event_progress,
            route_length,
        )
        if not np.isfinite(dense).all():
            raise ValueError("dense Hermite validation contains nonfinite values")
        if np.any(np.diff(dense) <= 0.0):
            raise ValueError("dense Hermite derivative reverses or stalls")
        if np.min(dense) < low - _ATOL or np.max(dense) > high + _ATOL:
            raise ValueError("dense Hermite progress overshoots an anchor interval")

    speed = np.diff(progress) * fps
    acceleration = np.diff(speed) * fps
    jerk = np.diff(acceleration) * fps
    discrete_min, discrete_max = float(np.min(speed)), float(np.max(speed))
    _check_range(
        discrete_min,
        timing_bounds.min_discrete_route_progress_speed_mps,
        timing_bounds.max_discrete_route_progress_speed_mps,
        "discrete minimum speed",
    )
    _check_range(
        discrete_max,
        timing_bounds.min_discrete_route_progress_speed_mps,
        timing_bounds.max_discrete_route_progress_speed_mps,
        "discrete maximum speed",
    )
    max_acceleration = float(np.max(np.abs(acceleration)))
    if (
        max_acceleration
        > timing_bounds.max_abs_route_progress_acceleration_mps2 + _ATOL
    ):
        raise ValueError("discrete acceleration exceeds the calibrated bound")
    max_jerk = float(np.max(np.abs(jerk))) if len(jerk) else None
    if (
        timing_bounds.max_abs_discrete_route_progress_jerk_mps3 is not None
        and max_jerk is not None
        and max_jerk
        > timing_bounds.max_abs_discrete_route_progress_jerk_mps3 + _ATOL
    ):
        raise ValueError("discrete jerk exceeds the calibrated bound")

    endpoint_deviation = None
    if timing_bounds.reference_route_progress_speed_mps is not None:
        endpoint_deviation = max(
            abs(slopes[0] - timing_bounds.reference_route_progress_speed_mps),
            abs(slopes[2] - timing_bounds.reference_route_progress_speed_mps),
        )
    if (
        timing_bounds.max_endpoint_route_progress_speed_deviation_mps is not None
        and endpoint_deviation is not None
        and endpoint_deviation
        > timing_bounds.max_endpoint_route_progress_speed_deviation_mps + _ATOL
    ):
        raise ValueError("endpoint speed deviation exceeds the calibrated bound")

    points, heading = _route_points(progress, route, lengths, cumulative)
    projection_error = _max_projection_error(points, route)
    if projection_error > projection_tolerance + _ATOL:
        raise ValueError("dense root_xz does not project onto the prescribed route")

    linear = np.linspace(0.0, route_length, frames)
    deformation = np.abs(progress - linear)
    reference_speed = (
        timing_bounds.reference_route_progress_speed_mps
        if timing_bounds.reference_route_progress_speed_mps is not None
        else route_length / ((frames - 1) / fps)
    )
    mean_deformation = float(np.mean(deformation))
    integrated_deformation = float(np.sum(deformation) / fps)
    normalized_deformation = mean_deformation / route_length
    rms_speed_deviation = float(np.sqrt(np.mean((speed - reference_speed) ** 2)))

    route_hasher = hashlib.sha256()
    _hash_array(route_hasher, "route_xz", route)
    _hash_array(route_hasher, "route_cumulative_progress_m", cumulative)
    route_hash = route_hasher.hexdigest()
    return RouteProgressProgram(
        route_xz=tuple(tuple(float(value) for value in row) for row in route),
        route_cumulative_progress_m=tuple(float(value) for value in cumulative),
        progress_m=tuple(float(value) for value in progress),
        root_xz=tuple(tuple(float(value) for value in row) for row in points),
        route_heading_rad=tuple(float(value) for value in heading),
        event_frame=event,
        event_root_progress_m=event_progress,
        route_length_m=route_length,
        segment_mean_route_progress_speeds_mps=tuple(float(value) for value in means),
        anchor_route_progress_slopes_mps=tuple(float(value) for value in slopes),
        continuous_interval_route_progress_speed_ranges_mps=tuple(
            (float(value[0]), float(value[1])) for value in derivative
        ),
        continuous_interval_route_progress_acceleration_endpoints_mps2=tuple(
            tuple(float(acceleration) for acceleration in value[2])
            for value in derivative
        ),
        continuous_route_progress_speed_range_mps=(continuous_min, continuous_max),
        max_abs_continuous_route_progress_acceleration_mps2=continuous_acceleration,
        discrete_route_progress_speed_range_mps=(discrete_min, discrete_max),
        max_abs_discrete_route_progress_acceleration_mps2=max_acceleration,
        max_abs_discrete_route_progress_jerk_mps3=max_jerk,
        endpoint_route_progress_speed_deviation_mps=endpoint_deviation,
        mean_abs_progress_deformation_m=mean_deformation,
        integrated_abs_progress_deformation_m_s=integrated_deformation,
        normalized_progress_deformation=normalized_deformation,
        rms_route_progress_speed_deviation_mps=rms_speed_deviation,
        max_path_projection_error_m=projection_error,
        path_projection_tolerance_m=projection_tolerance,
        timing_bounds=timing_bounds,
        route_hash=route_hash,
    )


__all__ = [
    "ROUTE_PROGRESS_METHOD",
    "ROUTE_PROGRESS_SCHEMA_VERSION",
    "ROUTE_PROGRESS_SLOPE_POLICY",
    "RouteProgressProgram",
    "RouteTimingBounds",
    "reparameterize_route_progress",
    "route_progress_selection_key",
]
