"""Threshold-free measurement of neutral-walk gait-phase observability.

Donor adaptation quality and target gait-phase observability answer different questions.
The former may require a large step-over lift; the latter asks only whether a complete,
physically supported neutral swing exposes a clearance peak that can be localized.  This
module therefore discovers swings with the source-quality lift gate disabled and measures
their world-frame swing-foot clearance prominence without applying an acceptance threshold.

For swing-foot bottom clearance ``c(t)`` above the fixed world floor, four stable support
samples immediately before takeoff and four beginning at landing form the locked baseline
windows.  The prominence is

``c(apex) - max(median(c_pre), median(c_post))``.

Each validated swing-support baseline supplies one preregistered null: its largest
swing-foot clearance minus its median.  This produces exactly two deterministic background
contrasts per swing.  Contralateral height and motion cannot inflate the signal or either
null.  No data-dependent threshold or quantile is selected here; calibration and later
target acceptance remain downstream concerns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np

from ..stepover_eval import StepOverThresholds
from .step_phase import StepPhaseCycle, enumerate_step_phase_cycles


Side = Literal["left", "right"]
_SIDES: tuple[Side, Side] = ("left", "right")
PHASE_OBSERVABILITY_SCHEMA_VERSION = "ramp-own-foot-phase-observability-v1"
PHASE_PROMINENCE_METHOD = "neutral-swing-world-floor-clearance-prominence-v1"
PHASE_BACKGROUND_METHOD = "swing-support-clearance-window-max-minus-median-v1"
PHASE_BASELINE_SUPPORT_METHOD = "qpos-physical-foot-support-mask-v1"
PHASE_BASELINE_FRAMES = 4
_DISCOVERY_MIN_RELATIVE_LIFT_M = 0.0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _validate_digest(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


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


def _positive_finite(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _series(kinematics: Mapping[str, Any], side: Side, name: str) -> np.ndarray:
    try:
        values = np.asarray(kinematics[side][name], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"kinematics lacks {side} {name}") from exc
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{side} {name} must be a non-empty finite 1-D series")
    return np.array(values, dtype=float, copy=True)


def _validated_kinematics(
    kinematics: Mapping[str, Any],
) -> tuple[dict[Side, dict[str, np.ndarray]], int]:
    exact: dict[Side, dict[str, np.ndarray]] = {}
    lengths: set[int] = set()
    for side in _SIDES:
        exact[side] = {}
        for name in ("bottom_clearance_m", "planar_speed_mps"):
            values = _series(kinematics, side, name)
            exact[side][name] = values
            lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError("left/right kinematic series must share one frame count")
    return exact, lengths.pop()


def _kinematics_digest(kinematics: Mapping[Side, Mapping[str, np.ndarray]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(PHASE_OBSERVABILITY_SCHEMA_VERSION.encode())
    for side in _SIDES:
        for name in ("bottom_clearance_m", "planar_speed_mps"):
            header = {"side": side, "name": name, "shape": list(kinematics[side][name].shape)}
            hasher.update(_canonical_json(header).encode())
            values = np.ascontiguousarray(kinematics[side][name], dtype="<f8")
            hasher.update(values.tobytes(order="C"))
    return hasher.hexdigest()


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


def phase_observability_measurement_hash(
    thresholds: StepOverThresholds,
    *,
    fps: float,
    support_window_s: float,
    min_stance_support_fraction: float | None = None,
) -> str:
    """Return the common measurement identity, excluding downstream quality gates.

    The identity intentionally contains neither a donor minimum lift nor a target minimum
    prominence.  Four *samples* define each baseline, so the sampling rate is part of the
    identity rather than being normalized away.
    """
    if not isinstance(thresholds, StepOverThresholds):
        raise ValueError("thresholds must be StepOverThresholds")
    thresholds.validate()
    fps = _positive_finite(fps, "fps")
    support_window_s = _positive_finite(support_window_s, "support_window_s")
    locked_fraction = float(thresholds.min_contralateral_support_fraction)
    if min_stance_support_fraction is None:
        min_stance_support_fraction = locked_fraction
    min_stance_support_fraction = _finite(
        min_stance_support_fraction, "min_stance_support_fraction"
    )
    if not 0.0 < min_stance_support_fraction <= 1.0:
        raise ValueError("min_stance_support_fraction must lie in (0, 1]")
    if min_stance_support_fraction != locked_fraction:
        raise ValueError(
            "explicit min_stance_support_fraction conflicts with StepOverThresholds"
        )
    payload = {
        "schema_version": PHASE_OBSERVABILITY_SCHEMA_VERSION,
        "signal_method": PHASE_PROMINENCE_METHOD,
        "background_method": PHASE_BACKGROUND_METHOD,
        "fps": fps,
        "baseline_frames": PHASE_BASELINE_FRAMES,
        "support_height_m": float(thresholds.support_height_m),
        "support_speed_mps": float(thresholds.support_speed_mps),
        "support_window_s": support_window_s,
        "min_stance_support_fraction": min_stance_support_fraction,
        "max_floor_penetration_m": float(thresholds.max_floor_penetration_m),
        "stable_support_dwell_s": float(thresholds.landing_dwell_s),
        "swing_discovery_quality_gate": "disabled",
        "signal_baseline_support": "cycle-swing-side-all-samples",
        "background_windows": ["pre_takeoff", "post_landing"],
        "signal_quantity": "swing-foot-world-frame-bottom-clearance-m",
        "background_statistic": "same-foot-window-max-minus-window-median",
    }
    return _digest(payload)


def _finite_tuple(values: Any, name: str, expected_length: int) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite values") from exc
    if len(result) != expected_length or not np.isfinite(np.asarray(result)).all():
        raise ValueError(f"{name} must contain exactly {expected_length} finite values")
    return result


def _baseline_support_source(cycle: StepPhaseCycle) -> str:
    pre_start = cycle.landmarks.takeoff_frame - PHASE_BASELINE_FRAMES
    pre_end = cycle.landmarks.takeoff_frame - 1
    post_start = cycle.landmarks.landing_frame
    post_end = cycle.landmarks.landing_frame + PHASE_BASELINE_FRAMES - 1
    return (
        f"{PHASE_BASELINE_SUPPORT_METHOD};side={cycle.swing_side};"
        f"height_m<={cycle.support_height_m:.9g};"
        f"speed_mps<={cycle.support_speed_mps:.9g};"
        f"pre_frames={pre_start}:{pre_end};post_frames={post_start}:{post_end};"
        f"fps={cycle.fps:.9g}"
    )


@dataclass(frozen=True)
class PhaseProminenceReceipt:
    """Auditable own-foot world-floor prominence for one complete neutral swing."""

    method: str
    cycle: StepPhaseCycle
    measurement_protocol_hash: str
    kinematics_digest: str
    baseline_support_side: Side
    baseline_support_source: str
    baseline_frames: int
    pre_window_start_frame: int
    pre_window_end_frame: int
    post_window_start_frame: int
    post_window_end_frame: int
    pre_clearance_m: tuple[float, ...]
    post_clearance_m: tuple[float, ...]
    apex_clearance_m: float
    pre_median_clearance_m: float
    post_median_clearance_m: float
    background_reference_clearance_m: float
    prominence_m: float

    def __post_init__(self) -> None:
        if self.method != PHASE_PROMINENCE_METHOD:
            raise ValueError(f"method must be {PHASE_PROMINENCE_METHOD!r}")
        if not isinstance(self.cycle, StepPhaseCycle):
            raise ValueError("cycle must be a StepPhaseCycle")
        if self.cycle.min_relative_lift_m != _DISCOVERY_MIN_RELATIVE_LIFT_M:
            raise ValueError("neutral swing discovery must disable the source-quality lift gate")
        _validate_digest(self.measurement_protocol_hash, "measurement_protocol_hash")
        _validate_digest(self.kinematics_digest, "kinematics_digest")
        if self.baseline_support_side != self.cycle.swing_side:
            raise ValueError("baseline support side must be the cycle-certified swing foot")
        if self.baseline_support_source != _baseline_support_source(self.cycle):
            raise ValueError("baseline support source does not match the cycle and locked windows")
        baseline_frames = _strict_int(self.baseline_frames, "baseline_frames")
        if baseline_frames != PHASE_BASELINE_FRAMES:
            raise ValueError(f"baseline_frames must equal the locked value {PHASE_BASELINE_FRAMES}")
        object.__setattr__(self, "baseline_frames", baseline_frames)

        frame_fields = (
            "pre_window_start_frame",
            "pre_window_end_frame",
            "post_window_start_frame",
            "post_window_end_frame",
        )
        for name in frame_fields:
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if (
            self.pre_window_start_frame != self.cycle.landmarks.takeoff_frame - baseline_frames
            or self.pre_window_end_frame != self.cycle.landmarks.takeoff_frame - 1
            or self.post_window_start_frame != self.cycle.landmarks.landing_frame
            or self.post_window_end_frame
            != self.cycle.landmarks.landing_frame + baseline_frames - 1
        ):
            raise ValueError("baseline windows do not match the locked swing-event geometry")
        if self.pre_window_start_frame < 0:
            raise ValueError("pre-takeoff baseline window is outside the clip")

        pre = _finite_tuple(self.pre_clearance_m, "pre_clearance_m", baseline_frames)
        post = _finite_tuple(self.post_clearance_m, "post_clearance_m", baseline_frames)
        object.__setattr__(self, "pre_clearance_m", pre)
        object.__setattr__(self, "post_clearance_m", post)
        apex = _finite(self.apex_clearance_m, "apex_clearance_m")
        pre_median = _finite(self.pre_median_clearance_m, "pre_median_clearance_m")
        post_median = _finite(self.post_median_clearance_m, "post_median_clearance_m")
        background = _finite(
            self.background_reference_clearance_m,
            "background_reference_clearance_m",
        )
        prominence = _finite(self.prominence_m, "prominence_m")
        expected_pre = float(np.median(pre))
        expected_post = float(np.median(post))
        expected_background = max(expected_pre, expected_post)
        if not np.isclose(pre_median, expected_pre, atol=1e-12, rtol=0.0):
            raise ValueError("pre_median_clearance_m does not match the four-frame median")
        if not np.isclose(post_median, expected_post, atol=1e-12, rtol=0.0):
            raise ValueError("post_median_clearance_m does not match the four-frame median")
        if not np.isclose(background, expected_background, atol=1e-12, rtol=0.0):
            raise ValueError(
                "background_reference_clearance_m must be the larger baseline median"
            )
        if not np.isclose(prominence, apex - background, atol=1e-12, rtol=0.0):
            raise ValueError("prominence_m does not match apex-clearance-minus-background")
        for name, value in (
            ("apex_clearance_m", apex),
            ("pre_median_clearance_m", pre_median),
            ("post_median_clearance_m", post_median),
            ("background_reference_clearance_m", background),
            ("prominence_m", prominence),
        ):
            object.__setattr__(self, name, value)

    @property
    def swing_side(self) -> Side:
        return self.cycle.swing_side

    @property
    def stance_side(self) -> Side:
        return self.cycle.stance_side

    @property
    def receipt_digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE_OBSERVABILITY_SCHEMA_VERSION,
            "method": self.method,
            "cycle": self.cycle.as_dict(),
            "measurement_protocol_hash": self.measurement_protocol_hash,
            "kinematics_digest": self.kinematics_digest,
            "baseline_support_side": self.baseline_support_side,
            "baseline_support_source": self.baseline_support_source,
            "baseline_frames": self.baseline_frames,
            "pre_window_start_frame": self.pre_window_start_frame,
            "pre_window_end_frame": self.pre_window_end_frame,
            "post_window_start_frame": self.post_window_start_frame,
            "post_window_end_frame": self.post_window_end_frame,
            "pre_clearance_m": list(self.pre_clearance_m),
            "post_clearance_m": list(self.post_clearance_m),
            "apex_clearance_m": self.apex_clearance_m,
            "pre_median_clearance_m": self.pre_median_clearance_m,
            "post_median_clearance_m": self.post_median_clearance_m,
            "background_reference_clearance_m": self.background_reference_clearance_m,
            "prominence_m": self.prominence_m,
        }
        if include_digest:
            result["receipt_digest"] = self.receipt_digest
        return result


@dataclass(frozen=True)
class PhaseBackgroundContrast:
    """Own-foot max-minus-median null on one locked swing-support baseline."""

    method: str
    source_prominence_digest: str
    measurement_protocol_hash: str
    kinematics_digest: str
    baseline_support_side: Side
    baseline_support_source: str
    window_label: Literal["pre_takeoff", "post_landing"]
    window_start_frame: int
    window_end_frame: int
    baseline_frames: int
    clearance_m: tuple[float, ...]
    peak_frame: int
    peak_clearance_m: float
    median_clearance_m: float
    contrast_m: float

    def __post_init__(self) -> None:
        if self.method != PHASE_BACKGROUND_METHOD:
            raise ValueError(f"method must be {PHASE_BACKGROUND_METHOD!r}")
        _validate_digest(self.source_prominence_digest, "source_prominence_digest")
        _validate_digest(self.measurement_protocol_hash, "measurement_protocol_hash")
        _validate_digest(self.kinematics_digest, "kinematics_digest")
        if self.baseline_support_side not in _SIDES:
            raise ValueError("baseline_support_side must be 'left' or 'right'")
        if not (
            isinstance(self.baseline_support_source, str)
            and self.baseline_support_source.startswith(
                f"{PHASE_BASELINE_SUPPORT_METHOD};side={self.baseline_support_side};"
            )
        ):
            raise ValueError("baseline support source does not identify its support side")
        if self.window_label not in ("pre_takeoff", "post_landing"):
            raise ValueError("window_label must be 'pre_takeoff' or 'post_landing'")
        for name in ("window_start_frame", "window_end_frame", "peak_frame"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        baseline_frames = _strict_int(self.baseline_frames, "baseline_frames")
        if baseline_frames != PHASE_BASELINE_FRAMES:
            raise ValueError(f"baseline_frames must equal the locked value {PHASE_BASELINE_FRAMES}")
        object.__setattr__(self, "baseline_frames", baseline_frames)
        if self.window_start_frame < 0 or (
            self.window_end_frame - self.window_start_frame + 1 != baseline_frames
        ):
            raise ValueError("background window must contain exactly four in-bounds frames")
        values = _finite_tuple(
            self.clearance_m,
            "clearance_m",
            baseline_frames,
        )
        object.__setattr__(self, "clearance_m", values)
        expected_peak_offset = int(np.argmax(values))
        expected_peak_frame = self.window_start_frame + expected_peak_offset
        if self.peak_frame != expected_peak_frame:
            raise ValueError("peak_frame must be the first deterministic window maximum")
        peak = _finite(self.peak_clearance_m, "peak_clearance_m")
        median = _finite(self.median_clearance_m, "median_clearance_m")
        contrast = _finite(self.contrast_m, "contrast_m")
        expected_peak = float(values[expected_peak_offset])
        expected_median = float(np.median(values))
        if not np.isclose(peak, expected_peak, atol=1e-12, rtol=0.0):
            raise ValueError("peak_clearance_m does not match the window maximum")
        if not np.isclose(median, expected_median, atol=1e-12, rtol=0.0):
            raise ValueError("median_clearance_m does not match the window median")
        if not np.isclose(contrast, peak - median, atol=1e-12, rtol=0.0):
            raise ValueError("contrast_m does not match window-maximum-minus-median")
        for name, value in (
            ("peak_clearance_m", peak),
            ("median_clearance_m", median),
            ("contrast_m", contrast),
        ):
            object.__setattr__(self, name, value)

    @property
    def receipt_digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE_OBSERVABILITY_SCHEMA_VERSION,
            **asdict(self),
            "clearance_m": list(self.clearance_m),
        }
        if include_digest:
            result["receipt_digest"] = self.receipt_digest
        return result


@dataclass(frozen=True)
class PhaseObservabilityEvidence:
    """One own-foot prominence and its two locked support-window null contrasts."""

    prominence: PhaseProminenceReceipt
    background_contrasts: tuple[PhaseBackgroundContrast, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prominence, PhaseProminenceReceipt):
            raise ValueError("prominence must be a PhaseProminenceReceipt")
        backgrounds = tuple(self.background_contrasts)
        if len(backgrounds) != 2:
            raise ValueError("phase observability evidence requires exactly two contrasts")
        for contrast in backgrounds:
            if not isinstance(contrast, PhaseBackgroundContrast):
                raise ValueError("background_contrasts must contain PhaseBackgroundContrast")
            if contrast.source_prominence_digest != self.prominence.receipt_digest:
                raise ValueError("background contrast belongs to a different signal prominence")
            if (
                contrast.measurement_protocol_hash
                != self.prominence.measurement_protocol_hash
                or contrast.kinematics_digest != self.prominence.kinematics_digest
            ):
                raise ValueError("signal and background contrasts do not share one measurement")
            if (
                contrast.baseline_support_side != self.prominence.baseline_support_side
                or contrast.baseline_support_source
                != self.prominence.baseline_support_source
            ):
                raise ValueError("background contrast does not share the swing-support evidence")
            if contrast.window_label == "pre_takeoff":
                expected_start = self.prominence.pre_window_start_frame
                expected_end = self.prominence.pre_window_end_frame
                signal_values = np.asarray(self.prominence.pre_clearance_m)
            else:
                expected_start = self.prominence.post_window_start_frame
                expected_end = self.prominence.post_window_end_frame
                signal_values = np.asarray(self.prominence.post_clearance_m)
            if (
                contrast.window_start_frame != expected_start
                or contrast.window_end_frame != expected_end
            ):
                raise ValueError("background contrast does not use its locked signal baseline")
            if not np.array_equal(np.asarray(contrast.clearance_m), signal_values):
                raise ValueError("background contrast does not use the own-foot signal clearance")
        labels = [contrast.window_label for contrast in backgrounds]
        if labels != ["pre_takeoff", "post_landing"]:
            raise ValueError("background contrasts must use both locked baselines in order")
        object.__setattr__(self, "background_contrasts", backgrounds)

    @property
    def receipt_digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE_OBSERVABILITY_SCHEMA_VERSION,
            "prominence": self.prominence.as_dict(),
            "background_contrasts": [item.as_dict() for item in self.background_contrasts],
        }
        if include_digest:
            result["receipt_digest"] = self.receipt_digest
        return result


def _swing_clearance(
    kinematics: Mapping[Side, Mapping[str, np.ndarray]],
    swing_side: Side,
) -> np.ndarray:
    values = kinematics[swing_side]["bottom_clearance_m"]
    if not np.isfinite(values).all():
        raise ValueError("swing-foot clearance trace contains non-finite values")
    return values


def _prominence_terms(
    clearance: np.ndarray,
    *,
    apex: int,
    takeoff: int,
    landing: int,
) -> tuple[tuple[float, ...], tuple[float, ...], float, float, float, float, float]:
    pre = tuple(
        float(value)
        for value in clearance[takeoff - PHASE_BASELINE_FRAMES : takeoff]
    )
    post = tuple(
        float(value) for value in clearance[landing : landing + PHASE_BASELINE_FRAMES]
    )
    if len(pre) != PHASE_BASELINE_FRAMES or len(post) != PHASE_BASELINE_FRAMES:
        raise ValueError("locked four-frame baseline is truncated by the clip boundary")
    apex_value = float(clearance[apex])
    values = np.asarray((*pre, *post, apex_value), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("phase prominence inputs contain non-finite values")
    pre_median = float(np.median(pre))
    post_median = float(np.median(post))
    background = max(pre_median, post_median)
    prominence = apex_value - background
    return pre, post, apex_value, pre_median, post_median, background, prominence


def enumerate_phase_prominences(
    kinematics: Mapping[str, Any],
    *,
    fps: float,
    frame_window: tuple[int, int] | None = None,
    swing_side: Side | None = None,
    thresholds: StepOverThresholds | None = None,
    support_window_s: float = 0.24,
    min_stance_support_fraction: float | None = None,
) -> tuple[PhaseProminenceReceipt, ...]:
    """Measure every complete neutral swing that has locked stable baseline windows.

    Swing discovery always passes ``min_relative_lift_m=0`` to the common physical-cycle
    detector.  The returned prominence is a measurement, not an acceptance decision.
    """
    exact, length = _validated_kinematics(kinematics)
    thresholds = thresholds or StepOverThresholds()
    measurement_hash = phase_observability_measurement_hash(
        thresholds,
        fps=fps,
        support_window_s=support_window_s,
        min_stance_support_fraction=min_stance_support_fraction,
    )
    support = _support_masks(exact, thresholds)
    cycles = enumerate_step_phase_cycles(
        exact,
        fps=fps,
        frame_window=frame_window,
        swing_side=swing_side,
        thresholds=thresholds,
        support_window_s=support_window_s,
        min_stance_support_fraction=min_stance_support_fraction,
        min_relative_lift_m=_DISCOVERY_MIN_RELATIVE_LIFT_M,
    )
    input_digest = _kinematics_digest(exact)
    receipts: list[PhaseProminenceReceipt] = []
    rejections: list[str] = []
    for cycle in cycles:
        takeoff = cycle.landmarks.takeoff_frame
        apex = cycle.landmarks.apex_frame
        landing = cycle.landmarks.landing_frame
        pre_start = takeoff - PHASE_BASELINE_FRAMES
        post_end = landing + PHASE_BASELINE_FRAMES
        if pre_start < 0 or post_end > length:
            rejections.append(f"{cycle.swing_side} apex {apex}: baseline window is truncated")
            continue
        swing_support = support[cycle.swing_side]
        if not (
            np.all(swing_support[pre_start:takeoff])
            and np.all(swing_support[landing:post_end])
        ):
            rejections.append(
                f"{cycle.swing_side} apex {apex}: baseline window lacks stable swing-foot support"
            )
            continue
        clearance = _swing_clearance(exact, cycle.swing_side)
        terms = _prominence_terms(
            clearance,
            apex=apex,
            takeoff=takeoff,
            landing=landing,
        )
        receipts.append(
            PhaseProminenceReceipt(
                method=PHASE_PROMINENCE_METHOD,
                cycle=cycle,
                measurement_protocol_hash=measurement_hash,
                kinematics_digest=input_digest,
                baseline_support_side=cycle.swing_side,
                baseline_support_source=_baseline_support_source(cycle),
                baseline_frames=PHASE_BASELINE_FRAMES,
                pre_window_start_frame=pre_start,
                pre_window_end_frame=takeoff - 1,
                post_window_start_frame=landing,
                post_window_end_frame=post_end - 1,
                pre_clearance_m=terms[0],
                post_clearance_m=terms[1],
                apex_clearance_m=terms[2],
                pre_median_clearance_m=terms[3],
                post_median_clearance_m=terms[4],
                background_reference_clearance_m=terms[5],
                prominence_m=terms[6],
            )
        )
    if not receipts:
        detail = "; ".join(rejections) if rejections else "no measured swing survived"
        raise ValueError(f"no observable neutral swing prominence: {detail}")
    return tuple(receipts)


def enumerate_background_phase_contrasts(
    kinematics: Mapping[str, Any],
    prominence: PhaseProminenceReceipt,
    *,
    thresholds: StepOverThresholds | None = None,
) -> tuple[PhaseBackgroundContrast, ...]:
    """Measure two preregistered own-foot nulls on the locked support baselines.

    The pre-takeoff and post-landing windows have already passed the swing-foot support
    gate.  That gate is recomputed here from the bound input.  Within each window, the
    first maximum is used on ties.
    """
    if not isinstance(prominence, PhaseProminenceReceipt):
        raise ValueError("prominence must be a PhaseProminenceReceipt")
    exact, _ = _validated_kinematics(kinematics)
    input_digest = _kinematics_digest(exact)
    if input_digest != prominence.kinematics_digest:
        raise ValueError("kinematics do not match the signal prominence receipt")
    thresholds = thresholds or StepOverThresholds()
    expected_hash = phase_observability_measurement_hash(
        thresholds,
        fps=prominence.cycle.fps,
        support_window_s=prominence.cycle.support_window_s,
        min_stance_support_fraction=prominence.cycle.min_stance_support_fraction,
    )
    if expected_hash != prominence.measurement_protocol_hash:
        raise ValueError("support thresholds do not match the signal measurement protocol")
    support = _support_masks(exact, thresholds)
    swing_support = support[prominence.baseline_support_side]
    clearance = _swing_clearance(exact, prominence.baseline_support_side)
    windows = (
        (
            "pre_takeoff",
            prominence.pre_window_start_frame,
            prominence.pre_window_end_frame,
        ),
        (
            "post_landing",
            prominence.post_window_start_frame,
            prominence.post_window_end_frame,
        ),
        )
    receipts: list[PhaseBackgroundContrast] = []
    for window_label, window_start, window_end in windows:
        window_slice = slice(window_start, window_end + 1)
        if len(swing_support[window_slice]) != PHASE_BASELINE_FRAMES or not np.all(
            swing_support[window_slice]
        ):
            raise ValueError(
                f"{window_label} background window lacks stable swing-foot support"
            )
        values = tuple(float(value) for value in clearance[window_slice])
        if len(values) != PHASE_BASELINE_FRAMES or not np.isfinite(values).all():
            raise ValueError(f"{window_label} background values are truncated or non-finite")
        peak_offset = int(np.argmax(values))
        peak = float(values[peak_offset])
        median = float(np.median(values))
        receipts.append(
            PhaseBackgroundContrast(
                method=PHASE_BACKGROUND_METHOD,
                source_prominence_digest=prominence.receipt_digest,
                measurement_protocol_hash=prominence.measurement_protocol_hash,
                kinematics_digest=input_digest,
                baseline_support_side=prominence.baseline_support_side,
                baseline_support_source=prominence.baseline_support_source,
                window_label=window_label,
                window_start_frame=window_start,
                window_end_frame=window_end,
                baseline_frames=PHASE_BASELINE_FRAMES,
                clearance_m=values,
                peak_frame=window_start + peak_offset,
                peak_clearance_m=peak,
                median_clearance_m=median,
                contrast_m=peak - median,
            )
        )
    if len(receipts) != 2:  # Defensive against future changes to the locked windows.
        raise RuntimeError("locked phase background construction did not produce two receipts")
    return tuple(receipts)


def measure_phase_observability(
    kinematics: Mapping[str, Any],
    **kwargs: Any,
) -> tuple[PhaseObservabilityEvidence, ...]:
    """Return complete signal-and-background evidence for all observable neutral swings."""
    thresholds = kwargs.get("thresholds")
    prominences = enumerate_phase_prominences(kinematics, **kwargs)
    evidence: list[PhaseObservabilityEvidence] = []
    rejections: list[str] = []
    for prominence in prominences:
        try:
            backgrounds = enumerate_background_phase_contrasts(
                kinematics,
                prominence,
                thresholds=thresholds,
            )
        except ValueError as exc:
            rejections.append(
                f"{prominence.swing_side} apex {prominence.cycle.apex_frame}: {exc}"
            )
            continue
        evidence.append(PhaseObservabilityEvidence(prominence, backgrounds))
    if not evidence:
        detail = "; ".join(rejections) if rejections else "no complete evidence"
        raise ValueError(f"no complete phase-observability evidence: {detail}")
    return tuple(evidence)


__all__ = [
    "PHASE_BACKGROUND_METHOD",
    "PHASE_BASELINE_FRAMES",
    "PHASE_OBSERVABILITY_SCHEMA_VERSION",
    "PHASE_PROMINENCE_METHOD",
    "PhaseBackgroundContrast",
    "PhaseObservabilityEvidence",
    "PhaseProminenceReceipt",
    "enumerate_background_phase_contrasts",
    "enumerate_phase_prominences",
    "measure_phase_observability",
    "phase_observability_measurement_hash",
]
