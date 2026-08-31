"""Event-aligned coherent motion packets for frozen humanoid priors.

The legacy semantic scaffold copies an absolute world-space pose from one donor. This module
implements the first RAMP primitive: compare an adapted donor with an independently aligned
neutral donor, transport only that coherent adaptation onto a held-out nominal gait, and emit
the same ARDY channels for the absolute and residual experimental arms.

Rotations are represented in the kinematic hierarchy. For prescribed route heading h,
L_root = H(h)^T G_root and L_j = G_parent(j)^T G_j. The transported residual is
D_j = L_adapt,j L_neutral,j^T and L_target,j* = D_j^strength L_target,j.

The source route heading is explicit. ARDY's generated global_root_heading describes the
realised body and can disagree with the decoded rotation channel; using it as route yaw would
erase precisely the sidling/turning residual that RAMP needs to retain.

Version 1 is deliberately strict and step-event scoped. Adapted and neutral windows must be
resampled onto an explicitly supplied common gait-phase grid, preserve the same swing side,
and provide contralateral stance evidence. Duck/double-support and squeeze events need their
own event/contact contract rather than silently weakening this one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..constraints import ConstraintSpec


PacketRepresentation = Literal["absolute", "residual"]
Side = Literal["left", "right"]
PACKET_SCHEMA_VERSION = 1
MAX_PACKET_STRENGTH = 2.0
SO3_BRANCH_MARGIN_RAD = 1e-5

_BASE_PROVENANCE_FIELDS = (
    "adapted_clip_sha256",
    "checkpoint_sha256",
    "generator_id",
    "sampler_seed",
    "noise_stream_version",
    "event_selector",
    "code_revision",
)


class EventLike(Protocol):
    """The v1 step-event contract needs an event centre and swing side."""

    frame: int
    side: Side


class TargetPhaseMatchLike(Protocol):
    """Structural target-phase receipt; defined concretely in ramp.phase."""

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
    measurement_protocol_hash: str

    def as_dict(self) -> dict[str, Any]: ...


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _readonly_array(value: Any, dtype: np.dtype) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _readonly_integer_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers")
    int64 = np.iinfo(np.int64)
    if raw.size and (np.any(raw < int64.min) or np.any(raw > int64.max)):
        raise ValueError(f"{name} contains a value outside signed int64")
    return _readonly_array(raw, np.int64)


def _validate_cyclic_progression(values: Any, name: str) -> None:
    phase = np.asarray(values, dtype=float)
    if phase.ndim != 1 or len(phase) < 3 or not np.isfinite(phase).all():
        raise ValueError(f"{name} must be a finite one-dimensional phase grid")
    forward = np.mod(np.diff(phase), 1.0)
    if np.any(forward <= 1e-8) or np.any(forward >= 0.5):
        raise ValueError(f"{name} must advance strictly and unambiguously")


def _assert_so3(name: str, matrices: np.ndarray, atol: float = 2e-4) -> None:
    matrices = np.asarray(matrices, dtype=float)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError(f"{name} must contain finite 3x3 rotation matrices")
    flat = matrices.reshape(-1, 3, 3)
    gram = np.swapaxes(flat, -1, -2) @ flat
    if not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(f"{name} contains a non-orthonormal matrix")
    if not np.allclose(np.linalg.det(flat), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{name} contains an improper rotation")


def _principal_angles(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    rotvec = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    return np.linalg.norm(rotvec, axis=-1).reshape(matrices.shape[:-2])


def _event(event: EventLike) -> tuple[int, Side]:
    frame = _strict_int(event.frame, "event frame")
    side = str(event.side)
    if side not in ("left", "right"):
        raise ValueError("event side must be 'left' or 'right'")
    return frame, side  # type: ignore[return-value]


def _yaw_matrices(heading: np.ndarray) -> np.ndarray:
    """ARDY Y-up yaw: +Z rotates toward +X as heading increases."""
    heading = np.asarray(heading, dtype=float)
    c, s = np.cos(heading), np.sin(heading)
    out = np.zeros(heading.shape + (3, 3), dtype=float)
    out[..., 0, 0], out[..., 0, 2] = c, s
    out[..., 1, 1] = 1.0
    out[..., 2, 0], out[..., 2, 2] = -s, c
    return out


def _topological_order(parents: np.ndarray, root_idx: int) -> tuple[int, ...]:
    parents = np.asarray(parents)
    if parents.dtype.kind not in "iu" or parents.dtype.kind == "b":
        raise ValueError("parent_indices must contain integers")
    parents = parents.astype(np.int64, copy=False)
    root_idx = _strict_int(root_idx, "root_idx")
    if parents.ndim != 1 or not len(parents):
        raise ValueError("parent_indices must be a non-empty 1-D array")
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1 or int(roots[0]) != root_idx:
        raise ValueError("parent_indices must contain exactly the declared root")
    if np.any((parents < -1) | (parents >= len(parents))):
        raise ValueError("parent index is outside the joint array")
    if any(int(parent) == joint for joint, parent in enumerate(parents)):
        raise ValueError("a joint cannot parent itself")

    children: list[list[int]] = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(child)
    order: list[int] = []
    frontier = [root_idx]
    while frontier:
        joint = frontier.pop(0)
        order.append(joint)
        frontier.extend(children[joint])
    if len(order) != len(parents):
        raise ValueError("parent_indices contain a cycle or disconnected component")
    return tuple(order)


def _sample_arrays(
    sample: Mapping[str, Any], *, expected_joints: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a decoded prior sample.

    The returned heading is the generated body-heading feature and is diagnostic only. Route
    canonicalisation always uses the separately supplied route heading.
    """
    rotations = np.asarray(sample.get("global_rot_mats"), dtype=float)
    root = np.asarray(sample.get("smooth_root_pos"), dtype=float)
    heading_cs = np.asarray(sample.get("global_root_heading"), dtype=float)
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("global_rot_mats must have shape (T, J, 3, 3)")
    frames, joints = rotations.shape[:2]
    if expected_joints is not None and joints != expected_joints:
        raise ValueError(f"sample has {joints} joints, expected {expected_joints}")
    if root.shape != (frames, 3):
        raise ValueError("smooth_root_pos must have shape (T, 3)")
    if heading_cs.shape != (frames, 2):
        raise ValueError("global_root_heading must have shape (T, 2)")
    if not np.isfinite(root).all() or not np.isfinite(heading_cs).all():
        raise ValueError("sample root and heading arrays must be finite")
    norm = np.linalg.norm(heading_cs, axis=-1)
    if np.any(norm < 1e-8):
        raise ValueError("global_root_heading contains a zero vector")
    _assert_so3("global_rot_mats", rotations)
    generated_body_heading = np.arctan2(heading_cs[:, 1], heading_cs[:, 0])
    return rotations, root, generated_body_heading


def _route_heading_series(value: Any, length: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        raw = np.full(length, float(raw), dtype=float)
    if raw.shape != (length,) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite scalar or a length-{length} array")
    return raw


def _global_to_local(
    global_rotations: np.ndarray,
    route_headings: np.ndarray,
    parents: np.ndarray,
    root_idx: int,
) -> np.ndarray:
    local = np.empty_like(global_rotations)
    root_yaw = _yaw_matrices(route_headings)
    local[:, root_idx] = np.swapaxes(root_yaw, -1, -2) @ global_rotations[:, root_idx]
    for joint, parent in enumerate(parents):
        if joint == root_idx:
            continue
        local[:, joint] = (
            np.swapaxes(global_rotations[:, int(parent)], -1, -2)
            @ global_rotations[:, joint]
        )
    _assert_so3("hierarchy-local rotations", local)
    return local


def _local_to_global(
    local_rotations: np.ndarray,
    route_headings: np.ndarray,
    parents: np.ndarray,
    root_idx: int,
) -> np.ndarray:
    order = _topological_order(parents, root_idx)
    out = np.empty_like(local_rotations)
    out[:, root_idx] = _yaw_matrices(route_headings) @ local_rotations[:, root_idx]
    for joint in order:
        if joint == root_idx:
            continue
        out[:, joint] = out[:, int(parents[joint])] @ local_rotations[:, joint]
    _assert_so3("rendered global rotations", out)
    return out


def _window_offsets(half_window_frames: int) -> tuple[np.ndarray, np.ndarray]:
    half_window_frames = _strict_int(half_window_frames, "half_window_frames")
    if half_window_frames < 1:
        raise ValueError("a coherent packet requires at least one context frame per side")
    offsets = np.arange(-half_window_frames, half_window_frames + 1, dtype=np.int64)
    taper = 0.5 * (1.0 + np.cos(np.pi * np.abs(offsets) / half_window_frames))
    taper[0] = taper[-1] = 0.0
    return offsets, taper


def _bounded_query(center: int, offsets: np.ndarray, length: int, name: str) -> np.ndarray:
    query = float(center) + np.asarray(offsets, dtype=float)
    if query[0] < 0 or query[-1] > length - 1:
        raise ValueError(f"{name} does not have the requested full context window")
    return query


def _canonical_provenance_json(value: Mapping[str, Any] | str) -> str:
    try:
        raw = json.loads(value) if isinstance(value, str) else dict(value)
        if not isinstance(raw, dict):
            raise TypeError("provenance must encode a JSON object")
        return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("packet provenance must be finite JSON data") from exc


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"provenance field {name!r} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"provenance field {name!r} must be a SHA-256 hex digest") from exc


def _validate_provenance(raw: Mapping[str, Any], representation: PacketRepresentation) -> None:
    required = list(_BASE_PROVENANCE_FIELDS)
    if representation == "residual":
        required.append("neutral_clip_sha256")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"packet provenance is missing required fields: {', '.join(missing)}")
    _validate_sha256(raw["adapted_clip_sha256"], "adapted_clip_sha256")
    _validate_sha256(raw["checkpoint_sha256"], "checkpoint_sha256")
    if representation == "residual":
        _validate_sha256(raw["neutral_clip_sha256"], "neutral_clip_sha256")
    _strict_int(raw["sampler_seed"], "provenance sampler_seed")
    noise_version = raw["noise_stream_version"]
    if isinstance(noise_version, (bool, np.bool_)) or not (
        isinstance(noise_version, (int, np.integer)) and int(noise_version) >= 1
    ):
        raise ValueError(
            "provenance field 'noise_stream_version' must be a positive integer"
        )
    for name in ("generator_id", "event_selector", "code_revision"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ValueError(f"provenance field {name!r} must be a non-empty string")


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    value = np.ascontiguousarray(value)
    hasher.update(name.encode())
    hasher.update(str(value.dtype).encode())
    hasher.update(json.dumps(value.shape).encode())
    hasher.update(value.tobytes())


@dataclass(frozen=True)
class PhaseMatch:
    """Auditable common-phase alignment for an adapted/neutral step-event window."""

    method: str
    adapted_query_offsets_frames: tuple[float, ...]
    neutral_query_offsets_frames: tuple[float, ...]
    adapted_phase_knots: tuple[float, ...]
    neutral_phase_knots: tuple[float, ...]
    max_phase_error: float
    adapted_stance_side: Side
    neutral_stance_side: Side
    adapted_stance_source: str
    neutral_stance_source: str
    adapted_stance_support_fraction: float
    neutral_stance_support_fraction: float
    min_stance_support_fraction: float
    support_window_s: float
    measurement_protocol_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("phase-match method must be non-empty")
        if not _is_sha256(self.measurement_protocol_hash):
            raise ValueError("measurement_protocol_hash must be a lowercase SHA-256 digest")
        for name in (
            "adapted_query_offsets_frames",
            "neutral_query_offsets_frames",
            "adapted_phase_knots",
            "neutral_phase_knots",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain finite values")
            object.__setattr__(self, name, values)
        if not np.isfinite(self.max_phase_error) or not 0 <= self.max_phase_error < 0.5:
            raise ValueError("max_phase_error must lie in [0, 0.5)")
        object.__setattr__(self, "max_phase_error", float(self.max_phase_error))
        if self.adapted_stance_side not in ("left", "right") or self.neutral_stance_side not in (
            "left",
            "right",
        ):
            raise ValueError("stance sides must be 'left' or 'right'")
        for name in ("adapted_stance_source", "neutral_stance_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty evidence source")
        _validate_cyclic_progression(self.adapted_phase_knots, "adapted_phase_knots")
        _validate_cyclic_progression(self.neutral_phase_knots, "neutral_phase_knots")
        for name in (
            "adapted_stance_support_fraction",
            "neutral_stance_support_fraction",
            "min_stance_support_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.support_window_s) or self.support_window_s <= 0:
            raise ValueError("support_window_s must be positive and finite")
        object.__setattr__(self, "support_window_s", float(self.support_window_s))
        if (
            self.min_stance_support_fraction <= 0
            or self.adapted_stance_support_fraction < self.min_stance_support_fraction
            or self.neutral_stance_support_fraction < self.min_stance_support_fraction
        ):
            raise ValueError("phase match stance support is below the locked threshold")

    @property
    def phase_errors(self) -> tuple[float, ...]:
        adapted = np.asarray(self.adapted_phase_knots)
        neutral = np.asarray(self.neutral_phase_knots)
        if adapted.shape != neutral.shape:
            return ()
        raw = np.abs(adapted - neutral)
        return tuple(float(value) for value in np.minimum(raw, 1.0 - raw))

    def validate(self, source_offsets: np.ndarray, swing_side: Side) -> None:
        expected_length = len(source_offsets)
        arrays = (
            self.adapted_query_offsets_frames,
            self.neutral_query_offsets_frames,
            self.adapted_phase_knots,
            self.neutral_phase_knots,
        )
        if any(len(value) != expected_length for value in arrays):
            raise ValueError("phase-match arrays must align with the packet source window")
        adapted_query = np.asarray(self.adapted_query_offsets_frames)
        neutral_query = np.asarray(self.neutral_query_offsets_frames)
        if np.any(np.diff(adapted_query) <= 0) or np.any(np.diff(neutral_query) <= 0):
            raise ValueError("phase-match query offsets must be strictly increasing")
        center_index = int(np.flatnonzero(source_offsets == 0)[0])
        if not np.isclose(adapted_query[center_index], 0.0) or not np.isclose(
            neutral_query[center_index], 0.0
        ):
            raise ValueError("phase-match queries must place both event centres at offset zero")
        for name in ("adapted_phase_knots", "neutral_phase_knots"):
            phases = np.asarray(getattr(self, name))
            if np.any((phases < 0) | (phases >= 1)):
                raise ValueError(f"{name} must lie in [0, 1)")
        if not self.phase_errors or max(self.phase_errors) > self.max_phase_error:
            error = max(self.phase_errors, default=float("inf"))
            raise ValueError(
                f"window phase error {error:.4f} exceeds {self.max_phase_error:.4f}"
            )
        expected_stance: Side = "right" if swing_side == "left" else "left"
        if (
            self.adapted_stance_side != expected_stance
            or self.neutral_stance_side != expected_stance
        ):
            raise ValueError("phase match does not preserve contralateral stance contact")

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "adapted_query_offsets_frames": list(self.adapted_query_offsets_frames),
            "neutral_query_offsets_frames": list(self.neutral_query_offsets_frames),
            "adapted_phase_knots": list(self.adapted_phase_knots),
            "neutral_phase_knots": list(self.neutral_phase_knots),
            "phase_errors": list(self.phase_errors),
            "max_phase_error": self.max_phase_error,
            "adapted_stance_side": self.adapted_stance_side,
            "neutral_stance_side": self.neutral_stance_side,
            "adapted_stance_source": self.adapted_stance_source,
            "neutral_stance_source": self.neutral_stance_source,
            "adapted_stance_support_fraction": self.adapted_stance_support_fraction,
            "neutral_stance_support_fraction": self.neutral_stance_support_fraction,
            "min_stance_support_fraction": self.min_stance_support_fraction,
            "support_window_s": self.support_window_s,
            "measurement_protocol_hash": self.measurement_protocol_hash,
        }


@dataclass(frozen=True)
class CoherentMotionPacket:
    """Dense full-body rotations and root height centred on one step event."""

    representation: PacketRepresentation
    swing_side: Side
    source_fps: float
    source_offsets_frames: np.ndarray
    joint_names: tuple[str, ...]
    parent_indices: np.ndarray
    root_idx: int
    rotation_payload: np.ndarray
    root_height_payload_m: np.ndarray
    taper: np.ndarray
    phase_knots: np.ndarray
    adapted_query_offsets_frames: np.ndarray
    adapted_route_heading_rad: np.ndarray
    adapted_center_frame: int
    measurement_protocol_hash: str
    neutral_query_offsets_frames: np.ndarray | None = None
    neutral_route_heading_rad: np.ndarray | None = None
    neutral_center_frame: int | None = None
    phase_match: PhaseMatch | None = None
    provenance_json: str = field(repr=False, default="{}")

    schema_version: ClassVar[int] = PACKET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.representation not in ("absolute", "residual"):
            raise ValueError("representation must be 'absolute' or 'residual'")
        if self.swing_side not in ("left", "right"):
            raise ValueError("swing_side must be 'left' or 'right'")
        if not _is_sha256(self.measurement_protocol_hash):
            raise ValueError("measurement_protocol_hash must be a lowercase SHA-256 digest")
        if not np.isfinite(self.source_fps) or self.source_fps <= 0:
            raise ValueError("source_fps must be positive and finite")
        names = tuple(str(name) for name in self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be non-empty and unique")
        parents = _readonly_integer_array(self.parent_indices, "parent_indices")
        root_idx = _strict_int(self.root_idx, "root_idx")
        _topological_order(parents, root_idx)
        if len(parents) != len(names):
            raise ValueError("parent_indices must align with joint_names")
        offsets = _readonly_integer_array(self.source_offsets_frames, "source_offsets_frames")
        if offsets.ndim != 1 or len(offsets) < 3 or not np.all(np.diff(offsets) > 0):
            raise ValueError("source offsets must be a strictly increasing context window")
        if np.count_nonzero(offsets == 0) != 1:
            raise ValueError("source offsets must contain exactly one event centre")
        rotations = _readonly_array(self.rotation_payload, np.float64)
        expected = (len(offsets), len(names), 3, 3)
        if rotations.shape != expected:
            raise ValueError(f"rotation_payload must have shape {expected}")
        _assert_so3("rotation_payload", rotations)
        heights = _readonly_array(self.root_height_payload_m, np.float64)
        taper = _readonly_array(self.taper, np.float64)
        phase_knots = _readonly_array(self.phase_knots, np.float64)
        adapted_query = _readonly_array(self.adapted_query_offsets_frames, np.float64)
        adapted_route = _readonly_array(self.adapted_route_heading_rad, np.float64)
        if (
            heights.shape != offsets.shape
            or taper.shape != offsets.shape
            or phase_knots.shape != offsets.shape
        ):
            raise ValueError("height, taper, and phase knots must align with source offsets")
        if adapted_query.shape != offsets.shape or adapted_route.shape != offsets.shape:
            raise ValueError("adapted query and route heading must align with source offsets")
        if not (
            np.isfinite(heights).all()
            and np.isfinite(taper).all()
            and np.isfinite(phase_knots).all()
            and np.isfinite(adapted_query).all()
            and np.isfinite(adapted_route).all()
        ):
            raise ValueError("packet numeric payloads must be finite")
        if np.any((phase_knots < 0) | (phase_knots >= 1)):
            raise ValueError("packet phase knots must lie in [0, 1)")
        _validate_cyclic_progression(phase_knots, "packet phase knots")
        center_index = int(np.flatnonzero(offsets == 0)[0])
        if np.any(np.diff(adapted_query) <= 0) or not np.isclose(
            adapted_query[center_index], 0.0
        ):
            raise ValueError("adapted queries must increase strictly through event offset zero")
        if np.any((taper < 0) | (taper > 1)):
            raise ValueError("taper weights must lie in [0, 1]")
        if not np.isclose(taper[center_index], 1.0):
            raise ValueError("taper must peak at one on the event centre")
        if not np.isclose(taper[0], 0.0) or not np.isclose(taper[-1], 0.0):
            raise ValueError("taper must close to zero at both packet boundaries")

        adapted_center = _strict_int(self.adapted_center_frame, "adapted_center_frame")
        neutral_center: int | None = None
        neutral_query: np.ndarray | None = None
        neutral_route: np.ndarray | None = None
        if self.representation == "absolute":
            if (
                self.neutral_center_frame is not None
                or self.neutral_query_offsets_frames is not None
                or self.neutral_route_heading_rad is not None
                or self.phase_match is not None
            ):
                raise ValueError("an absolute packet has no neutral event or phase match")
        else:
            if (
                self.neutral_center_frame is None
                or self.neutral_query_offsets_frames is None
                or self.neutral_route_heading_rad is None
                or self.phase_match is None
            ):
                raise ValueError("a residual packet must record neutral and phase alignment data")
            neutral_center = _strict_int(self.neutral_center_frame, "neutral_center_frame")
            neutral_query = _readonly_array(self.neutral_query_offsets_frames, np.float64)
            neutral_route = _readonly_array(self.neutral_route_heading_rad, np.float64)
            if neutral_query.shape != offsets.shape or neutral_route.shape != offsets.shape:
                raise ValueError("neutral query and route heading must align with source offsets")
            if not np.isfinite(neutral_query).all() or not np.isfinite(neutral_route).all():
                raise ValueError("neutral query and route heading must be finite")
            if not np.allclose(
                neutral_query, self.phase_match.neutral_query_offsets_frames,
                atol=0.0, rtol=0.0,
            ):
                raise ValueError("neutral queries must equal the recorded phase alignment")
            self.phase_match.validate(offsets, self.swing_side)
            if self.phase_match.measurement_protocol_hash != self.measurement_protocol_hash:
                raise ValueError("packet and phase match use different measurement protocols")
            if not np.allclose(
                adapted_query, self.phase_match.adapted_query_offsets_frames,
                atol=0.0, rtol=0.0,
            ):
                raise ValueError("adapted queries must equal the recorded phase alignment")
            max_angle = float(np.max(_principal_angles(rotations)))
            if max_angle >= np.pi - SO3_BRANCH_MARGIN_RAD:
                raise ValueError("residual packet is too close to the ambiguous SO(3) pi branch")

        provenance_json = _canonical_provenance_json(self.provenance_json)
        provenance = json.loads(provenance_json)
        _validate_provenance(provenance, self.representation)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "source_offsets_frames", offsets)
        object.__setattr__(self, "rotation_payload", rotations)
        object.__setattr__(self, "root_height_payload_m", heights)
        object.__setattr__(self, "taper", taper)
        object.__setattr__(self, "phase_knots", phase_knots)
        object.__setattr__(self, "adapted_query_offsets_frames", adapted_query)
        object.__setattr__(self, "adapted_route_heading_rad", adapted_route)
        object.__setattr__(self, "neutral_query_offsets_frames", neutral_query)
        object.__setattr__(self, "neutral_route_heading_rad", neutral_route)
        object.__setattr__(self, "root_idx", root_idx)
        object.__setattr__(self, "adapted_center_frame", adapted_center)
        object.__setattr__(self, "neutral_center_frame", neutral_center)
        object.__setattr__(self, "source_fps", float(self.source_fps))
        object.__setattr__(self, "provenance_json", provenance_json)

    @property
    def joint_indices(self) -> np.ndarray:
        indices = np.arange(len(self.joint_names), dtype=np.int64)
        indices.setflags(write=False)
        return indices

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a detached JSON object; callers cannot mutate packet identity."""
        return json.loads(self.provenance_json)

    @property
    def max_residual_angle_rad(self) -> float | None:
        if self.representation != "residual":
            return None
        return float(np.max(_principal_angles(self.rotation_payload)))

    def digest(self) -> str:
        hasher = hashlib.sha256()
        header = {
            "representation": self.representation,
            "schema_version": self.schema_version,
            "swing_side": self.swing_side,
            "source_fps": self.source_fps,
            "joint_names": self.joint_names,
            "root_idx": self.root_idx,
            "adapted_center_frame": self.adapted_center_frame,
            "neutral_center_frame": self.neutral_center_frame,
            "phase_match": self.phase_match.as_dict() if self.phase_match else None,
            "measurement_protocol_hash": self.measurement_protocol_hash,
            "provenance": self.provenance,
        }
        hasher.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        _hash_array(hasher, "source_offsets_frames", self.source_offsets_frames)
        _hash_array(hasher, "parent_indices", self.parent_indices)
        _hash_array(hasher, "rotation_payload", self.rotation_payload)
        _hash_array(hasher, "root_height_payload_m", self.root_height_payload_m)
        _hash_array(hasher, "taper", self.taper)
        _hash_array(hasher, "phase_knots", self.phase_knots)
        _hash_array(
            hasher, "adapted_query_offsets_frames", self.adapted_query_offsets_frames)
        _hash_array(hasher, "adapted_route_heading_rad", self.adapted_route_heading_rad)
        if self.neutral_query_offsets_frames is not None:
            _hash_array(
                hasher,
                "neutral_query_offsets_frames",
                self.neutral_query_offsets_frames,
            )
        if self.neutral_route_heading_rad is not None:
            _hash_array(hasher, "neutral_route_heading_rad", self.neutral_route_heading_rad)
        return hasher.hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Return compact JSON-ready identity; numeric payloads belong in an NPZ artifact."""
        return {
            "schema_version": self.schema_version,
            "representation": self.representation,
            "swing_side": self.swing_side,
            "source_fps": self.source_fps,
            "source_offsets_frames": self.source_offsets_frames.tolist(),
            "joint_names": list(self.joint_names),
            "parent_indices": self.parent_indices.tolist(),
            "root_idx": self.root_idx,
            "adapted_center_frame": self.adapted_center_frame,
            "neutral_center_frame": self.neutral_center_frame,
            "phase_match": self.phase_match.as_dict() if self.phase_match else None,
            "measurement_protocol_hash": self.measurement_protocol_hash,
            "provenance": self.provenance,
            "packet_hash": self.digest(),
            "max_residual_angle_rad": self.max_residual_angle_rad,
            "rotation_payload_shape": list(self.rotation_payload.shape),
            "root_height_payload_shape": list(self.root_height_payload_m.shape),
            "phase_knots": self.phase_knots.tolist(),
            "adapted_query_offsets_frames": self.adapted_query_offsets_frames.tolist(),
            "adapted_route_heading_rad": self.adapted_route_heading_rad.tolist(),
            "neutral_query_offsets_frames": (
                self.neutral_query_offsets_frames.tolist()
                if self.neutral_query_offsets_frames is not None
                else None
            ),
            "neutral_route_heading_rad": (
                self.neutral_route_heading_rad.tolist()
                if self.neutral_route_heading_rad is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PacketControls:
    """Low-dimensional controls exposed to the later response optimizer."""

    strength: float = 1.0
    center_shift_frames: int = 0
    duration_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.strength)
            or self.strength < 0
            or self.strength > MAX_PACKET_STRENGTH
        ):
            raise ValueError(
                f"strength must be finite and lie in [0, {MAX_PACKET_STRENGTH}]"
            )
        center_shift = _strict_int(self.center_shift_frames, "center_shift_frames")
        if not np.isfinite(self.duration_scale) or self.duration_scale <= 0:
            raise ValueError("duration_scale must be positive and finite")
        object.__setattr__(self, "strength", float(self.strength))
        object.__setattr__(self, "center_shift_frames", center_shift)
        object.__setattr__(self, "duration_scale", float(self.duration_scale))

    def as_dict(self) -> dict[str, float | int]:
        return {
            "strength": self.strength,
            "center_shift_frames": self.center_shift_frames,
            "duration_scale": self.duration_scale,
        }


@dataclass(frozen=True)
class PacketRenderInfo:
    """Serializable identity and footprint of one rendered packet program."""

    representation: PacketRepresentation
    swing_side: Side
    target_center_frame: int
    target_frames: tuple[int, ...]
    joint_indices: tuple[int, ...]
    weights: tuple[float, ...]
    source_query_frames: tuple[float, ...]
    target_phase_query_offsets_frames: tuple[float, ...]
    target_phase_errors: tuple[float, ...]
    target_phase_match_hash: str
    target_phase_match_json: str
    measurement_protocol_hash: str
    target_fps: float
    controls: PacketControls
    packet_hash: str
    support_hash: str
    program_hash: str


@dataclass(frozen=True)
class CoherentPacketPair:
    """Causally matched absolute/residual packets for the E1 representation test."""

    absolute: CoherentMotionPacket
    residual: CoherentMotionPacket

    def __post_init__(self) -> None:
        if (
            self.absolute.representation != "absolute"
            or self.residual.representation != "residual"
        ):
            raise ValueError("packet pair must contain one absolute and one residual packet")
        scalar_fields = (
            "swing_side",
            "source_fps",
            "joint_names",
            "root_idx",
            "adapted_center_frame",
            "measurement_protocol_hash",
        )
        for name in scalar_fields:
            if getattr(self.absolute, name) != getattr(self.residual, name):
                raise ValueError(f"packet pair differs in {name}")
        array_fields = (
            "source_offsets_frames",
            "parent_indices",
            "taper",
            "phase_knots",
            "adapted_query_offsets_frames",
            "adapted_route_heading_rad",
        )
        for name in array_fields:
            if not np.array_equal(getattr(self.absolute, name), getattr(self.residual, name)):
                raise ValueError(f"packet pair differs in {name}")
        absolute_provenance = self.absolute.provenance
        residual_provenance = self.residual.provenance
        for name in _BASE_PROVENANCE_FIELDS:
            if absolute_provenance[name] != residual_provenance[name]:
                raise ValueError(f"packet pair provenance differs in {name}")

    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.absolute.digest().encode())
        hasher.update(self.residual.digest().encode())
        return hasher.hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "pair_hash": self.digest(),
            "absolute_packet_hash": self.absolute.digest(),
            "residual_packet_hash": self.residual.digest(),
            "adapted_query_offsets_frames": (
                self.absolute.adapted_query_offsets_frames.tolist()
            ),
            "source_offsets_frames": self.absolute.source_offsets_frames.tolist(),
            "measurement_protocol_hash": self.absolute.measurement_protocol_hash,
        }


def _resample_rotations(
    key_frames: np.ndarray, rotations: np.ndarray, query_frames: np.ndarray
) -> np.ndarray:
    key_frames = np.asarray(key_frames, dtype=float)
    query_frames = np.asarray(query_frames, dtype=float)
    if (
        key_frames.ndim != 1
        or len(key_frames) != len(rotations)
        or np.any(np.diff(key_frames) <= 0)
    ):
        raise ValueError("rotation key frames must be one-dimensional and strictly increasing")
    if not np.isfinite(query_frames).all() or np.any(query_frames < key_frames[0]) or np.any(
        query_frames > key_frames[-1]
    ):
        raise ValueError("rotation queries must lie inside the source clip")
    out = np.empty((len(query_frames),) + rotations.shape[1:], dtype=float)
    for joint in range(rotations.shape[1]):
        out[:, joint] = Slerp(key_frames, Rotation.from_matrix(rotations[:, joint]))(
            query_frames
        ).as_matrix()
    _assert_so3("resampled rotations", out)
    return out


def _interpolate_angles(
    key_frames: np.ndarray, angles: np.ndarray, query: np.ndarray
) -> np.ndarray:
    return np.interp(query, key_frames, np.unwrap(np.asarray(angles, dtype=float)))


def _resample_source_window(
    rotations: np.ndarray,
    root: np.ndarray,
    route_heading: np.ndarray,
    query: np.ndarray,
    parents: np.ndarray,
    root_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_frames = np.arange(len(rotations), dtype=float)
    source_local = _global_to_local(rotations, route_heading, parents, root_idx)
    sampled_local = _resample_rotations(key_frames, source_local, query)
    sampled_height = np.interp(query, key_frames, root[:, 1])
    sampled_route = _interpolate_angles(key_frames, route_heading, query)
    return sampled_local, sampled_height, sampled_route


def extract_absolute_packet(
    adapted: Mapping[str, Any],
    adapted_event: EventLike,
    *,
    adapted_route_heading: np.ndarray | float,
    adapted_query_offsets_frames: Sequence[float],
    adapted_phase_knots: Sequence[float],
    joint_names: Sequence[str],
    parent_indices: Sequence[int],
    root_idx: int,
    source_fps: float,
    half_window_frames: int,
    measurement_protocol_hash: str,
    provenance: Mapping[str, Any],
) -> CoherentMotionPacket:
    """Extract a route-frame absolute full-body packet for the paired E1 baseline."""
    names = tuple(joint_names)
    parents = _readonly_integer_array(parent_indices, "parent_indices")
    root_idx = _strict_int(root_idx, "root_idx")
    _topological_order(parents, root_idx)
    if len(names) != len(parents):
        raise ValueError("joint_names and parent_indices must align")
    rotations, root, _ = _sample_arrays(adapted, expected_joints=len(names))
    route = _route_heading_series(adapted_route_heading, len(rotations), "adapted_route_heading")
    center, side = _event(adapted_event)
    offsets, taper = _window_offsets(half_window_frames)
    query_offsets = np.asarray(adapted_query_offsets_frames, dtype=float)
    if query_offsets.shape != offsets.shape:
        raise ValueError("adapted queries must align with the packet source window")
    query = _bounded_query(center, query_offsets, len(rotations), "adapted event")
    local, height, sampled_route = _resample_source_window(
        rotations, root, route, query, parents, root_idx
    )
    return CoherentMotionPacket(
        representation="absolute",
        swing_side=side,
        source_fps=float(source_fps),
        source_offsets_frames=offsets,
        joint_names=names,
        parent_indices=parents,
        root_idx=root_idx,
        rotation_payload=local,
        root_height_payload_m=height,
        taper=taper,
        phase_knots=np.asarray(adapted_phase_knots, dtype=float),
        adapted_query_offsets_frames=query_offsets,
        adapted_route_heading_rad=sampled_route,
        adapted_center_frame=center,
        measurement_protocol_hash=measurement_protocol_hash,
        provenance_json=_canonical_provenance_json(provenance),
    )


def extract_residual_packet(
    adapted: Mapping[str, Any],
    neutral: Mapping[str, Any],
    adapted_event: EventLike,
    neutral_event: EventLike,
    *,
    adapted_route_heading: np.ndarray | float,
    neutral_route_heading: np.ndarray | float,
    phase_match: PhaseMatch,
    joint_names: Sequence[str],
    parent_indices: Sequence[int],
    root_idx: int,
    source_fps: float,
    half_window_frames: int,
    provenance: Mapping[str, Any],
) -> CoherentMotionPacket:
    """Extract an adapted-minus-neutral packet on an explicit common phase grid."""
    names = tuple(joint_names)
    parents = _readonly_integer_array(parent_indices, "parent_indices")
    root_idx = _strict_int(root_idx, "root_idx")
    _topological_order(parents, root_idx)
    if len(names) != len(parents):
        raise ValueError("joint_names and parent_indices must align")
    adapted_rot, adapted_root, _ = _sample_arrays(adapted, expected_joints=len(names))
    neutral_rot, neutral_root, _ = _sample_arrays(neutral, expected_joints=len(names))
    adapted_route = _route_heading_series(
        adapted_route_heading, len(adapted_rot), "adapted_route_heading"
    )
    neutral_route = _route_heading_series(
        neutral_route_heading, len(neutral_rot), "neutral_route_heading"
    )
    adapted_center, adapted_side = _event(adapted_event)
    neutral_center, neutral_side = _event(neutral_event)
    if adapted_side != neutral_side:
        raise ValueError("adapted and neutral events must use the same swing side")
    offsets, taper = _window_offsets(half_window_frames)
    phase_match.validate(offsets, adapted_side)
    adapted_query = _bounded_query(
        adapted_center,
        np.asarray(phase_match.adapted_query_offsets_frames),
        len(adapted_rot),
        "adapted phase-aligned event",
    )
    neutral_query = _bounded_query(
        neutral_center,
        np.asarray(phase_match.neutral_query_offsets_frames),
        len(neutral_rot),
        "neutral phase-aligned event",
    )
    adapted_local, adapted_height, sampled_adapted_route = _resample_source_window(
        adapted_rot, adapted_root, adapted_route, adapted_query, parents, root_idx
    )
    neutral_local, neutral_height, sampled_neutral_route = _resample_source_window(
        neutral_rot, neutral_root, neutral_route, neutral_query, parents, root_idx
    )
    residual = adapted_local @ np.swapaxes(neutral_local, -1, -2)
    _assert_so3("local rotation residual", residual)
    return CoherentMotionPacket(
        representation="residual",
        swing_side=adapted_side,
        source_fps=float(source_fps),
        source_offsets_frames=offsets,
        joint_names=names,
        parent_indices=parents,
        root_idx=root_idx,
        rotation_payload=residual,
        root_height_payload_m=adapted_height - neutral_height,
        taper=taper,
        phase_knots=np.asarray(phase_match.adapted_phase_knots, dtype=float),
        adapted_query_offsets_frames=np.asarray(
            phase_match.adapted_query_offsets_frames, dtype=float),
        adapted_route_heading_rad=sampled_adapted_route,
        neutral_query_offsets_frames=np.asarray(
            phase_match.neutral_query_offsets_frames, dtype=float),
        neutral_route_heading_rad=sampled_neutral_route,
        adapted_center_frame=adapted_center,
        measurement_protocol_hash=phase_match.measurement_protocol_hash,
        neutral_center_frame=neutral_center,
        phase_match=phase_match,
        provenance_json=_canonical_provenance_json(provenance),
    )


def extract_packet_pair(
    adapted: Mapping[str, Any],
    neutral: Mapping[str, Any],
    adapted_event: EventLike,
    neutral_event: EventLike,
    *,
    adapted_route_heading: np.ndarray | float,
    neutral_route_heading: np.ndarray | float,
    phase_match: PhaseMatch,
    joint_names: Sequence[str],
    parent_indices: Sequence[int],
    root_idx: int,
    source_fps: float,
    half_window_frames: int,
    absolute_provenance: Mapping[str, Any],
    residual_provenance: Mapping[str, Any],
) -> CoherentPacketPair:
    """Build the E1 arms from byte-identical adapted source queries.

    The two packets intentionally differ only in representation and neutral subtraction.
    The pair object rejects any mismatch in the adapted phase queries, route frame, skeleton,
    taper, source event, checkpoint, seed, or noise-stream provenance.
    """
    absolute = extract_absolute_packet(
        adapted,
        adapted_event,
        adapted_route_heading=adapted_route_heading,
        adapted_query_offsets_frames=phase_match.adapted_query_offsets_frames,
        adapted_phase_knots=phase_match.adapted_phase_knots,
        joint_names=joint_names,
        parent_indices=parent_indices,
        root_idx=root_idx,
        source_fps=source_fps,
        half_window_frames=half_window_frames,
        measurement_protocol_hash=phase_match.measurement_protocol_hash,
        provenance=absolute_provenance,
    )
    residual = extract_residual_packet(
        adapted,
        neutral,
        adapted_event,
        neutral_event,
        adapted_route_heading=adapted_route_heading,
        neutral_route_heading=neutral_route_heading,
        phase_match=phase_match,
        joint_names=joint_names,
        parent_indices=parent_indices,
        root_idx=root_idx,
        source_fps=source_fps,
        half_window_frames=half_window_frames,
        provenance=residual_provenance,
    )
    return CoherentPacketPair(absolute=absolute, residual=residual)


def _rotation_power(matrices: np.ndarray, powers: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    powers = np.asarray(powers, dtype=float)
    if powers.shape != matrices.shape[:-2]:
        powers = np.broadcast_to(powers, matrices.shape[:-2])
    rotvec = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    powered = Rotation.from_rotvec(rotvec * powers.reshape(-1, 1)).as_matrix()
    out = powered.reshape(matrices.shape)
    _assert_so3("powered rotation residual", out)
    return out


def _program_digest(
    packet_hash: str,
    controls: PacketControls,
    spec: ConstraintSpec,
    target_frames: np.ndarray,
    nominal_route_heading: np.ndarray,
    output_route_heading: np.ndarray,
    target_phase_match_json: str,
    target_fps: float,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(packet_hash.encode())
    hasher.update(
        json.dumps(controls.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    )
    _hash_array(hasher, "root_xz", np.asarray(spec.root_xz, dtype=np.float64))
    _hash_array(
        hasher, "output_route_heading", np.asarray(output_route_heading, dtype=np.float64))
    _hash_array(hasher, "root_y", np.asarray(spec.root_y, dtype=np.float64))
    _hash_array(
        hasher,
        "nominal_route_heading",
        np.asarray(nominal_route_heading, dtype=np.float64),
    )
    _hash_array(hasher, "rot_frames", np.asarray(target_frames, dtype=np.int64))
    _hash_array(hasher, "rot_joints", np.asarray(spec.rot_joints, dtype=np.int64))
    _hash_array(hasher, "rot_targets", np.asarray(spec.rot_targets, dtype=np.float64))
    hasher.update(target_phase_match_json.encode())
    hasher.update(json.dumps({"target_fps": target_fps}, allow_nan=False).encode())
    hasher.update(
        json.dumps(
            {"first_heading": spec.first_heading},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    return hasher.hexdigest()


def constraint_support_digest(spec: ConstraintSpec) -> str:
    """Hash exactly which ARDY channels, frames, and joints a spec constrains."""
    root_frames = (
        np.asarray(spec.root_frames, dtype=np.int64)
        if spec.root_frames is not None
        else np.arange(spec.T, dtype=np.int64)
    )
    support = {
        "T": spec.T,
        "root_2d": True,
        "heading": spec.heading is not None,
        "root_y": spec.root_y is not None,
        "position_channel": spec.pos_frames is not None,
        "rotation_channel": spec.rot_frames is not None,
    }
    hasher = hashlib.sha256(json.dumps(support, sort_keys=True, separators=(",", ":")).encode())
    _hash_array(hasher, "root_frames", root_frames)
    if spec.pos_frames is not None:
        _hash_array(hasher, "pos_frames", np.asarray(spec.pos_frames, dtype=np.int64))
        _hash_array(hasher, "pos_joints", np.asarray(spec.pos_joints, dtype=np.int64))
    if spec.rot_frames is not None:
        _hash_array(hasher, "rot_frames", np.asarray(spec.rot_frames, dtype=np.int64))
        _hash_array(hasher, "rot_joints", np.asarray(spec.rot_joints, dtype=np.int64))
    return hasher.hexdigest()


def _validate_target_phase_match(
    packet: CoherentMotionPacket,
    target_event_center: int,
    target_event_side: Side,
    match: TargetPhaseMatchLike,
) -> tuple[np.ndarray, tuple[float, ...], str, str]:
    if not isinstance(match.method, str) or not match.method.strip():
        raise ValueError("target phase-match method must be non-empty")
    if _strict_int(match.target_center_frame, "target phase-match center") != target_event_center:
        raise ValueError("target phase receipt belongs to a different event centre")
    if match.swing_side != target_event_side:
        raise ValueError("target phase receipt belongs to a different swing side")
    if not _is_sha256(match.measurement_protocol_hash):
        raise ValueError("target phase receipt has an invalid measurement protocol hash")
    if match.measurement_protocol_hash != packet.measurement_protocol_hash:
        raise ValueError("target and packet use different measurement protocols")
    query = np.asarray(match.target_query_offsets_frames, dtype=float)
    packet_knots = np.asarray(match.packet_phase_knots, dtype=float)
    target_knots = np.asarray(match.target_phase_knots, dtype=float)
    expected_shape = packet.source_offsets_frames.shape
    if (
        query.shape != expected_shape
        or packet_knots.shape != expected_shape
        or target_knots.shape != expected_shape
    ):
        raise ValueError("target phase receipt must align with the packet phase grid")
    center_index = int(np.flatnonzero(packet.source_offsets_frames == 0)[0])
    if (
        not np.isfinite(query).all()
        or np.any(np.diff(query) <= 0)
        or not np.isclose(query[center_index], 0.0)
    ):
        raise ValueError("target phase queries must increase strictly through event offset zero")
    if not np.array_equal(packet_knots, packet.phase_knots):
        raise ValueError("target phase receipt was constructed for a different packet grid")
    if not np.isfinite(target_knots).all() or np.any((target_knots < 0) | (target_knots >= 1)):
        raise ValueError("target phase knots must be finite and lie in [0, 1)")
    _validate_cyclic_progression(target_knots, "target phase knots")
    raw_error = np.abs(packet.phase_knots - target_knots)
    errors = np.minimum(raw_error, 1.0 - raw_error)
    max_error = float(match.max_phase_error)
    if not np.isfinite(max_error) or not 0 <= max_error < 0.5 or np.max(errors) > max_error:
        raise ValueError("target phase error exceeds the locked tolerance")
    expected_stance: Side = "right" if target_event_side == "left" else "left"
    if match.target_stance_side != expected_stance:
        raise ValueError("target phase receipt lacks contralateral stance evidence")
    support = float(match.target_stance_support_fraction)
    threshold = float(match.min_stance_support_fraction)
    window = float(match.support_window_s)
    if (
        not np.isfinite(support)
        or not np.isfinite(threshold)
        or not np.isfinite(window)
        or not 0 <= support <= 1
        or not 0 < threshold <= 1
        or support < threshold
        or window <= 0
    ):
        raise ValueError("target stance evidence does not meet its locked gate")
    if not isinstance(match.target_stance_source, str) or not match.target_stance_source.strip():
        raise ValueError("target stance evidence source must be non-empty")
    try:
        receipt_json = json.dumps(
            match.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("target phase receipt must be finite JSON data") from exc
    receipt_hash = hashlib.sha256(receipt_json.encode()).hexdigest()
    return query, tuple(float(error) for error in errors), receipt_json, receipt_hash


def render_packet(
    packet: CoherentMotionPacket,
    target_nominal: Mapping[str, Any],
    target_event: EventLike,
    *,
    joint_names: Sequence[str],
    root_xz: np.ndarray,
    route_heading: np.ndarray,
    target_phase_match: TargetPhaseMatchLike,
    nominal_route_heading: np.ndarray | float | None = None,
    target_fps: float | None = None,
    controls: PacketControls | None = None,
    first_heading: float | None = None,
) -> tuple[ConstraintSpec, PacketRenderInfo]:
    """Transport a packet onto a held-out nominal gait and return an ARDY request."""
    controls = controls or PacketControls()
    names = tuple(joint_names)
    if names != packet.joint_names:
        raise ValueError("target joint ordering does not match the packet skeleton")
    rotations, nominal_root, _ = _sample_arrays(
        target_nominal, expected_joints=len(packet.joint_names))
    frames = len(rotations)
    root_xz = np.asarray(root_xz, dtype=float)
    route_heading = np.asarray(route_heading, dtype=float)
    # RAMP transports the adaptation onto this seed's realised nominal substrate.  Taking
    # root height from the sample here prevents a harness from silently replacing it with
    # a population constant and turning residual transport back into an absolute schedule.
    root_y = np.asarray(nominal_root[:, 1], dtype=float)
    if (
        root_xz.shape != (frames, 2)
        or route_heading.shape != (frames,)
        or root_y.shape != (frames,)
    ):
        raise ValueError("root_xz, route_heading, and root_y must match the target clip length")
    if not (
        np.isfinite(root_xz).all()
        and np.isfinite(route_heading).all()
        and np.isfinite(root_y).all()
    ):
        raise ValueError("target route arrays must be finite")
    nominal_route = _route_heading_series(
        route_heading if nominal_route_heading is None else nominal_route_heading,
        frames,
        "nominal_route_heading",
    )
    fps = packet.source_fps if target_fps is None else float(target_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    if isinstance(first_heading, (bool, np.bool_)):
        raise ValueError("first_heading must be a finite angle")
    if first_heading is not None:
        first_heading = float(first_heading)
        if not np.isfinite(first_heading):
            raise ValueError("first_heading must be a finite angle")
    event_center, event_side = _event(target_event)
    if event_side != packet.swing_side:
        raise ValueError("target event swing side does not match the packet")
    (
        target_phase_queries,
        target_phase_errors,
        target_phase_match_json,
        target_phase_match_hash,
    ) = _validate_target_phase_match(packet, event_center, event_side, target_phase_match)
    center = event_center + controls.center_shift_frames

    scale = controls.duration_scale
    scaled_lo = float(target_phase_queries[0]) * scale
    scaled_hi = float(target_phase_queries[-1]) * scale
    if not np.isfinite(scale) or not np.isfinite(scaled_lo) or not np.isfinite(scaled_hi):
        raise ValueError("duration scale produces a non-finite target window")
    lo = int(np.ceil(scaled_lo - 1e-12))
    hi = int(np.floor(scaled_hi + 1e-12))
    if lo >= 0 or hi <= 0 or hi - lo + 1 < 3:
        raise ValueError("duration scale collapses the packet context window")
    first_target, last_target = center + lo, center + hi
    if first_target < 0 or last_target >= frames:
        raise ValueError("rendered packet falls outside the target clip")
    target_offsets = np.arange(lo, hi + 1, dtype=np.int64)
    target_frames = center + target_offsets
    nominal_offsets = target_offsets / controls.duration_scale
    query = np.interp(
        nominal_offsets, target_phase_queries, packet.source_offsets_frames)
    payload = _resample_rotations(packet.source_offsets_frames, packet.rotation_payload, query)
    height_payload = np.interp(query, packet.source_offsets_frames, packet.root_height_payload_m)
    taper = np.interp(query, packet.source_offsets_frames, packet.taper)
    taper[0] = taper[-1] = 0.0
    weights = controls.strength * taper

    target_source_local = _global_to_local(
        rotations, nominal_route, packet.parent_indices, packet.root_idx)
    nominal_query_frames = event_center + nominal_offsets
    phase_sampled_local = _resample_rotations(
        np.arange(frames, dtype=float), target_source_local, nominal_query_frames)
    original_local = target_source_local[target_frames]
    warp_delta = phase_sampled_local @ np.swapaxes(original_local, -1, -2)
    warp_angles = _principal_angles(warp_delta)
    if np.max(warp_angles) >= np.pi - SO3_BRANCH_MARGIN_RAD:
        raise ValueError("target phase warp crosses the ambiguous SO(3) pi branch")
    target_local = _rotation_power(warp_delta, taper[:, None]) @ original_local
    if packet.representation == "residual":
        delta = payload
    else:
        delta = payload @ np.swapaxes(target_local, -1, -2)
    raw_angles = _principal_angles(delta)
    if np.max(raw_angles) >= np.pi - SO3_BRANCH_MARGIN_RAD:
        raise ValueError("resampled packet crosses the ambiguous SO(3) pi branch")
    effective_angles = raw_angles * weights[:, None]
    if np.max(effective_angles) >= np.pi - SO3_BRANCH_MARGIN_RAD:
        raise ValueError("packet strength crosses the SO(3) principal branch; reduce strength")
    powered = _rotation_power(delta, weights[:, None])
    edited_local = powered @ target_local
    rotation_targets = _local_to_global(
        edited_local, route_heading[target_frames], packet.parent_indices, packet.root_idx
    )

    nominal_query_height = np.interp(
        nominal_query_frames, np.arange(frames, dtype=float), nominal_root[:, 1])
    original_height = nominal_root[target_frames, 1]
    phase_warped_height = original_height + taper * (
        nominal_query_height - original_height)
    edited_height = np.array(root_y, dtype=float, copy=True)
    edited_height[target_frames] = phase_warped_height
    if packet.representation == "residual":
        edited_height[target_frames] += weights * height_payload
    else:
        edited_height[target_frames] += weights * (
            height_payload - phase_warped_height
        )
    spec = ConstraintSpec(
        root_xz=np.array(root_xz, copy=True),
        # Route tangent and ARDY's hip-derived body-heading feature are not the same
        # quantity.  V1 leaves the latter free rather than contradicting packet root/body
        # rotations; the straight-route step pilot remains located by dense root_xz.
        heading=None,
        root_y=edited_height,
        rot_frames=target_frames,
        rot_joints=np.array(packet.joint_indices, copy=True),
        rot_targets=rotation_targets,
        first_heading=first_heading,
    )
    packet_hash = packet.digest()
    info = PacketRenderInfo(
        representation=packet.representation,
        swing_side=packet.swing_side,
        target_center_frame=center,
        target_frames=tuple(int(frame) for frame in target_frames),
        joint_indices=tuple(int(index) for index in packet.joint_indices),
        weights=tuple(float(weight) for weight in weights),
        source_query_frames=tuple(float(frame) for frame in query),
        target_phase_query_offsets_frames=tuple(
            float(frame) for frame in target_phase_queries),
        target_phase_errors=target_phase_errors,
        target_phase_match_hash=target_phase_match_hash,
        target_phase_match_json=target_phase_match_json,
        measurement_protocol_hash=packet.measurement_protocol_hash,
        target_fps=fps,
        controls=controls,
        packet_hash=packet_hash,
        support_hash=constraint_support_digest(spec),
        program_hash=_program_digest(
            packet_hash,
            controls,
            spec,
            target_frames,
            nominal_route,
            route_heading,
            target_phase_match_json,
            fps,
        ),
    )
    return spec, info
