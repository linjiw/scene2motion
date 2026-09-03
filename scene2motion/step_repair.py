"""Obstacle-relative footstep repair for a frozen humanoid reference.

The released motion prior is left untouched.  This module operates on its exported G1
``qpos`` reference and changes only the twelve leg joints.  It turns a scene measurement into
two explicit foot targets:

* a foot that is supporting the body is kept on the same side of the obstacle, and
* a swinging foot whose envelope overlaps the obstacle is lifted above its inflated top.

The targets are dilated and smoothed in time, then projected with bounded per-frame inverse
kinematics.  The resulting joint deformation is smoothed once more before it is scored.  Root
translation and orientation, the upper body, clip length, and frame rate are preserved exactly.
A result is admitted only when the whole-body collision check clears, the predeclared support
screen still passes, both target residuals are bounded, and the edit stays inside the declared
joint-change and *pointwise* joint-speed budgets.

This is deliberately a *reference* repair.  Passing it is not evidence of controller tracking
or obstacle traversal; those are separate obstacle-present rollout endpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.optimize import least_squares

from .robot import BODY_MARGIN, G1Body
from .stepover_eval import foot_kinematics_series, step_scene


Side = Literal["left", "right"]
SIDES: tuple[Side, Side] = ("left", "right")
LEG_JOINTS: dict[Side, tuple[str, ...]] = {
    "left": (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ),
    "right": (
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ),
}

_FORWARD = np.array([1.0, 0.0, 0.0])
_LATERAL = np.array([0.0, 1.0, 0.0])
_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class SupportRule:
    """The frozen reference-screen operating rule used to admit a repaired candidate."""

    support_height_m: float
    support_speed_mps: float
    max_unsupported_run_s: float = 0.20

    def validate(self) -> None:
        values = (self.support_height_m, self.support_speed_mps,
                  self.max_unsupported_run_s)
        if not all(np.isfinite(values)) or any(value < 0 for value in values):
            raise ValueError("support-rule values must be finite and non-negative")


@dataclass(frozen=True)
class FootstepRepairConfig:
    """Fixed numerical choices for the local foot-envelope projection.

    The values are method parameters, not learned quantities.  Experiments must bind the full
    dataclass in their receipt and must not tune it on rollout outcomes.
    """

    clearance_buffer_m: float = 0.008
    temporal_dilation_frames: int = 5
    temporal_smoothing_sigma_frames: float = 1.0
    descriptor_weight: float = 8.0
    joint_fidelity_weight: float = 0.12
    temporal_continuity_weight: float = 0.05
    previous_solution_fraction: float = 0.30
    active_offset_tolerance_m: float = 1e-4
    max_ik_evaluations: int = 100
    max_ik_target_residual_m: float = 0.005
    joint_delta_smoothing_sigma_frames: float = 1.0
    max_post_smoothing_target_residual_m: float = 0.025
    max_joint_delta_rad: float = 0.50
    max_joint_speed_increase_rads: float = 2.0
    numeric_tolerance: float = 1e-8

    def validate(self) -> None:
        positive = (
            self.clearance_buffer_m, self.temporal_smoothing_sigma_frames,
            self.descriptor_weight, self.joint_fidelity_weight,
            self.active_offset_tolerance_m, self.max_ik_target_residual_m,
            self.joint_delta_smoothing_sigma_frames,
            self.max_post_smoothing_target_residual_m,
            self.max_joint_delta_rad, self.max_joint_speed_increase_rads,
            self.numeric_tolerance,
        )
        if not all(np.isfinite(positive)) or any(value <= 0 for value in positive):
            raise ValueError("positive repair parameters must be finite and positive")
        if self.temporal_dilation_frames < 1 or self.max_ik_evaluations < 1:
            raise ValueError("dilation and IK evaluation budgets must be positive integers")
        if not 0 <= self.previous_solution_fraction <= 1:
            raise ValueError("previous_solution_fraction must lie in [0, 1]")
        if self.temporal_continuity_weight < 0:
            raise ValueError("temporal_continuity_weight must be non-negative")


@dataclass(frozen=True)
class FootTargetPlan:
    """Smoothed world-frame target offsets for one foot."""

    side: Side
    forward_offset_m: np.ndarray
    vertical_offset_m: np.ndarray
    target_descriptor_m: np.ndarray

    @property
    def active_frames(self) -> np.ndarray:
        return np.flatnonzero(
            np.abs(self.forward_offset_m) + self.vertical_offset_m > 0
        )


@dataclass(frozen=True)
class FootstepRepairResult:
    """The repaired trajectory and its serialisable audit record."""

    qpos: np.ndarray
    record: dict[str, Any]


def _longest_true_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in np.asarray(mask, dtype=bool):
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def support_report(body: G1Body, qpos: np.ndarray, fps: float,
                   rule: SupportRule) -> dict[str, Any]:
    """Evaluate the named support rule without implying physical contact forces."""

    rule.validate()
    kinematics = foot_kinematics_series(body, qpos, fps)
    masks = {
        side: (
            (kinematics[side]["bottom_clearance_m"] <= rule.support_height_m)
            & (kinematics[side]["planar_speed_mps"] <= rule.support_speed_mps)
        )
        for side in SIDES
    }
    unsupported = ~(masks["left"] | masks["right"])
    longest_frames = _longest_true_run(unsupported)
    longest_s = float(longest_frames / fps)
    return {
        "support_height_m": float(rule.support_height_m),
        "support_speed_mps": float(rule.support_speed_mps),
        "max_allowed_unsupported_run_s": float(rule.max_unsupported_run_s),
        "longest_unsupported_run_frames": longest_frames,
        "longest_unsupported_run_s": longest_s,
        "passes": bool(longest_s <= rule.max_unsupported_run_s + 1e-12),
        "supported_fraction": {
            side: float(masks[side].mean()) for side in SIDES
        },
    }


def _foot_geom_ids(body: G1Body, side: Side) -> list[int]:
    ids = [
        geom for geom in body.robot_geoms
        if "foot" in body.geom_name[geom].lower()
        and side in body.geom_name[geom].lower()
    ]
    if not ids:
        raise ValueError(f"G1 collision model has no named {side} foot geoms")
    return ids


def _foot_descriptor(body: G1Body, qpos: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    """World ``(forward midpoint, lateral midpoint, bottom)`` of a physical foot."""

    body.fk(qpos)
    forward = [body.geom_extent(g, _FORWARD, extra_margin=0.0) for g in geom_ids]
    lateral = [body.geom_extent(g, _LATERAL, extra_margin=0.0) for g in geom_ids]
    vertical = [body.geom_extent(g, _UP, extra_margin=0.0) for g in geom_ids]
    return np.array([
        0.5 * (min(extent[0] for extent in forward)
               + max(extent[1] for extent in forward)),
        0.5 * (min(extent[0] for extent in lateral)
               + max(extent[1] for extent in lateral)),
        min(extent[0] for extent in vertical),
    ])


def _smooth_signed_offsets(raw: np.ndarray, config: FootstepRepairConfig) -> np.ndarray:
    """Dilate signed requirements without cancelling opposite directions."""

    size = int(config.temporal_dilation_frames)
    sigma = float(config.temporal_smoothing_sigma_frames)
    positive = gaussian_filter1d(maximum_filter1d(np.maximum(raw, 0.0), size=size), sigma)
    negative = -gaussian_filter1d(maximum_filter1d(np.maximum(-raw, 0.0), size=size), sigma)
    return np.where(raw > 0, np.maximum(positive, raw),
                    np.where(raw < 0, np.minimum(negative, raw), positive + negative))


def _smooth_positive_offsets(raw: np.ndarray, config: FootstepRepairConfig) -> np.ndarray:
    size = int(config.temporal_dilation_frames)
    sigma = float(config.temporal_smoothing_sigma_frames)
    spread = gaussian_filter1d(maximum_filter1d(np.maximum(raw, 0.0), size=size), sigma)
    return np.maximum(spread, raw)


def plan_foot_targets(body: G1Body, qpos: np.ndarray, *, fps: float,
                      obstacle_x_m: float, obstacle_height_m: float,
                      obstacle_depth_m: float, support_rule: SupportRule,
                      config: FootstepRepairConfig) -> dict[Side, FootTargetPlan]:
    """Translate obstacle overlap into smoothed stance-placement and swing-height targets."""

    support_rule.validate()
    config.validate()
    if obstacle_height_m <= 0 or obstacle_depth_m <= 0:
        raise ValueError("obstacle height and depth must be positive")
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] < 19 or len(qpos) < 2:
        raise ValueError("qpos must have shape (T, >=19) with at least two frames")
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("qpos and fps must be finite, with fps positive")

    kinematics = foot_kinematics_series(body, qpos, fps)
    near = float(obstacle_x_m - (0.5 * obstacle_depth_m + BODY_MARGIN))
    far = float(obstacle_x_m + (0.5 * obstacle_depth_m + BODY_MARGIN))
    top = float(obstacle_height_m + BODY_MARGIN)
    plans: dict[Side, FootTargetPlan] = {}

    for side in SIDES:
        foot = kinematics[side]
        overlap = ((foot["forward_max_m"] >= near)
                   & (foot["forward_min_m"] <= far))
        stance = ((foot["bottom_clearance_m"] <= support_rule.support_height_m)
                  & (foot["planar_speed_mps"] <= support_rule.support_speed_mps))
        forward = np.zeros(len(qpos), dtype=float)
        vertical = np.zeros(len(qpos), dtype=float)

        for frame in np.flatnonzero(overlap):
            if stance[frame]:
                if foot["forward_representative_m"][frame] < obstacle_x_m:
                    forward[frame] = (
                        near - config.clearance_buffer_m - foot["forward_max_m"][frame]
                    )
                else:
                    forward[frame] = (
                        far + config.clearance_buffer_m - foot["forward_min_m"][frame]
                    )
            elif foot["bottom_clearance_m"][frame] < top + config.clearance_buffer_m:
                vertical[frame] = (
                    top + config.clearance_buffer_m
                    - foot["bottom_clearance_m"][frame]
                )

        forward = _smooth_signed_offsets(forward, config)
        vertical = _smooth_positive_offsets(vertical, config)
        geom_ids = _foot_geom_ids(body, side)
        original = np.stack([_foot_descriptor(body, pose, geom_ids) for pose in qpos])
        target = original.copy()
        target[:, 0] += forward
        target[:, 2] += vertical
        plans[side] = FootTargetPlan(side, forward, vertical, target)
    return plans


def _leg_addresses_and_bounds(body: G1Body, side: Side
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    addresses: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for name in LEG_JOINTS[side]:
        joint = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or not body.model.jnt_limited[joint]:
            raise ValueError(f"missing or unlimited G1 leg joint: {name}")
        addresses.append(int(body.model.jnt_qposadr[joint]))
        lower.append(float(body.model.jnt_range[joint, 0]))
        upper.append(float(body.model.jnt_range[joint, 1]))
    return np.asarray(addresses), np.asarray(lower), np.asarray(upper)


def _project_leg(body: G1Body, qpos: np.ndarray, plan: FootTargetPlan,
                 config: FootstepRepairConfig) -> dict[str, Any]:
    addresses, lower, upper = _leg_addresses_and_bounds(body, plan.side)
    geom_ids = _foot_geom_ids(body, plan.side)
    active = np.flatnonzero(
        np.abs(plan.forward_offset_m) + plan.vertical_offset_m
        > config.active_offset_tolerance_m
    )
    previous: np.ndarray | None = None
    previous_frame: int | None = None
    residuals: list[float] = []
    solver_successes = 0
    total_evaluations = 0

    for frame_value in active:
        frame = int(frame_value)
        base = qpos[frame].copy()
        original_joints = base[addresses].copy()
        consecutive = previous is not None and previous_frame == frame - 1
        initial = original_joints.copy()
        if consecutive:
            fraction = config.previous_solution_fraction
            initial = (1.0 - fraction) * initial + fraction * previous
        initial = np.clip(initial, lower + config.numeric_tolerance,
                          upper - config.numeric_tolerance)
        target = plan.target_descriptor_m[frame]

        def residual(joints: np.ndarray) -> np.ndarray:
            pose = base.copy()
            pose[addresses] = joints
            values = [
                *(config.descriptor_weight
                  * (_foot_descriptor(body, pose, geom_ids) - target)),
                *(config.joint_fidelity_weight * (joints - original_joints)),
            ]
            if consecutive:
                values.extend(config.temporal_continuity_weight * (joints - previous))
            return np.asarray(values, dtype=float)

        solution = least_squares(
            residual, initial, bounds=(lower, upper),
            max_nfev=config.max_ik_evaluations,
            xtol=1e-9, ftol=1e-9, gtol=1e-9,
        )
        qpos[frame, addresses] = solution.x
        achieved = _foot_descriptor(body, qpos[frame], geom_ids)
        residuals.append(float(np.linalg.norm(achieved - target)))
        solver_successes += int(solution.success)
        total_evaluations += int(solution.nfev)
        previous = solution.x.copy()
        previous_frame = frame

    return {
        "side": plan.side,
        "n_active_frames": int(len(active)),
        "active_first_frame": int(active[0]) if len(active) else None,
        "active_last_frame": int(active[-1]) if len(active) else None,
        "max_forward_offset_m": float(np.abs(plan.forward_offset_m).max()),
        "max_vertical_offset_m": float(plan.vertical_offset_m.max()),
        "max_target_residual_m": float(max(residuals, default=0.0)),
        "solver_successes": int(solver_successes),
        "solver_calls": int(len(active)),
        "total_function_evaluations": int(total_evaluations),
    }


def _max_joint_speed(qpos: np.ndarray, fps: float) -> float:
    if len(qpos) < 2:
        return 0.0
    return float(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * fps)


def _smooth_leg_deformation(body: G1Body, original: np.ndarray, projected: np.ndarray,
                            config: FootstepRepairConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Smooth the local IK deformation while returning to zero outside the edited window."""

    repaired = projected.copy()
    records: list[dict[str, Any]] = []
    sigma = float(config.joint_delta_smoothing_sigma_frames)
    for side in SIDES:
        addresses, lower, upper = _leg_addresses_and_bounds(body, side)
        raw_delta = projected[:, addresses] - original[:, addresses]
        smooth_delta = gaussian_filter1d(
            raw_delta, sigma=sigma, axis=0, mode="constant", cval=0.0,
        )
        values = np.clip(original[:, addresses] + smooth_delta, lower, upper)
        repaired[:, addresses] = values
        records.append({
            "side": side,
            "sigma_frames": sigma,
            "max_raw_joint_delta_rad": float(np.abs(raw_delta).max()),
            "max_smoothed_joint_delta_rad": float(
                np.abs(values - original[:, addresses]).max()
            ),
        })
    return repaired, records


def _target_residual(body: G1Body, qpos: np.ndarray, plan: FootTargetPlan,
                     config: FootstepRepairConfig) -> float:
    geom_ids = _foot_geom_ids(body, plan.side)
    active = np.flatnonzero(
        np.abs(plan.forward_offset_m) + plan.vertical_offset_m
        > config.active_offset_tolerance_m
    )
    return float(max((
        np.linalg.norm(
            _foot_descriptor(body, qpos[int(frame)], geom_ids)
            - plan.target_descriptor_m[int(frame)]
        )
        for frame in active
    ), default=0.0))


def _pointwise_dynamics_change(original: np.ndarray, repaired: np.ndarray, *, fps: float,
                               addresses: np.ndarray) -> dict[str, float]:
    """Largest local velocity/acceleration increase on a matched joint-time element."""

    before_velocity = np.diff(original[:, addresses], axis=0) * fps
    after_velocity = np.diff(repaired[:, addresses], axis=0) * fps
    velocity_increase = np.maximum(
        np.abs(after_velocity) - np.abs(before_velocity), 0.0,
    )
    before_acceleration = np.diff(before_velocity, axis=0) * fps
    after_acceleration = np.diff(after_velocity, axis=0) * fps
    acceleration_increase = np.maximum(
        np.abs(after_acceleration) - np.abs(before_acceleration), 0.0,
    )
    return {
        "max_pointwise_joint_speed_increase_rads": float(
            velocity_increase.max(initial=0.0)
        ),
        "max_pointwise_joint_acceleration_increase_rads2": float(
            acceleration_increase.max(initial=0.0)
        ),
    }


def _collision_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "collision_free": bool(report["collision_free"]),
        "penetration_frames": int(report["penetration_frames"]),
        "max_penetration_m": float(report["max_penetration_m"]),
        "min_clearance_m": float(report["min_clearance_m"]),
        "worst": dict(report["worst"]),
        "culprit_geoms": list(report.get("culprit_geoms", [])),
    }


def repair_step_reference(qpos: np.ndarray, *, fps: float, obstacle_x_m: float,
                          obstacle_height_m: float, obstacle_depth_m: float,
                          support_rule: SupportRule,
                          config: FootstepRepairConfig | None = None,
                          body: G1Body | None = None,
                          obstacle_body: G1Body | None = None,
                          ) -> FootstepRepairResult:
    """Apply bounded foot-envelope IK and return the candidate plus its full decision record.

    A reference that already fails the support rule is refused without modification.  This
    matters experimentally: the repair is intended to bridge a geometry deficit on a plausible
    substrate, not to hide an unsupported generated motion behind a successful collision query.
    """

    config = config or FootstepRepairConfig()
    config.validate()
    support_rule.validate()
    original = np.asarray(qpos, dtype=float)
    if original.ndim != 2 or original.shape[1] < 36 or len(original) < 2:
        raise ValueError("qpos must have shape (T, >=36) with at least two frames")
    if not np.isfinite(original).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("qpos and fps must be finite, with fps positive")

    body = body or G1Body(None)
    obstacle_body = obstacle_body or G1Body(
        step_scene(obstacle_x_m, obstacle_height_m, obstacle_depth_m)
    )
    before_support = support_report(body, original, fps, support_rule)
    before_collision_full = obstacle_body.trajectory_report(original)
    before_floor = float(body.trajectory_report(original)["max_foot_floor_penetration_m"])
    before_speed = _max_joint_speed(original, fps)

    base_record: dict[str, Any] = {
        "schema_version": "step-foot-envelope-ik-v2",
        "method": (
            "obstacle-relative, support-screened foot-envelope IK with smoothed leg-joint "
            "deformation; root and upper body unchanged"
        ),
        "scene": {
            "obstacle_x_m": float(obstacle_x_m),
            "obstacle_height_m": float(obstacle_height_m),
            "obstacle_depth_m": float(obstacle_depth_m),
            "body_margin_m": float(BODY_MARGIN),
        },
        "support_rule": asdict(support_rule),
        "config": asdict(config),
        "before": {
            "collision": _collision_summary(before_collision_full),
            "support": before_support,
            "max_foot_floor_penetration_m": before_floor,
            "max_joint_speed_rads": before_speed,
        },
    }

    if not before_support["passes"]:
        base_record.update({
            "status": "refused",
            "accepted": False,
            "reasons": ["input_support_screen_failed"],
            "after": None,
            "deformation": None,
            "feet": [],
        })
        return FootstepRepairResult(original.copy(), base_record)

    plans = plan_foot_targets(
        body, original, fps=fps, obstacle_x_m=obstacle_x_m,
        obstacle_height_m=obstacle_height_m, obstacle_depth_m=obstacle_depth_m,
        support_rule=support_rule, config=config,
    )
    projected = original.copy()
    foot_records = [
        _project_leg(body, projected, plans[side], config) for side in SIDES
    ]
    repaired, smoothing_records = _smooth_leg_deformation(
        body, original, projected, config,
    )
    for side, record in zip(SIDES, foot_records):
        record["post_smoothing_max_target_residual_m"] = _target_residual(
            body, repaired, plans[side], config,
        )

    after_collision_full = obstacle_body.trajectory_report(repaired)
    after_support = support_report(body, repaired, fps, support_rule)
    after_floor = float(body.trajectory_report(repaired)["max_foot_floor_penetration_m"])
    after_speed = _max_joint_speed(repaired, fps)
    leg_addresses = np.concatenate([
        _leg_addresses_and_bounds(body, side)[0] for side in SIDES
    ])
    max_delta = float(np.max(np.abs(repaired[:, leg_addresses] - original[:, leg_addresses])))
    max_raw_residual = float(max(record["max_target_residual_m"] for record in foot_records))
    max_final_residual = float(max(
        record["post_smoothing_max_target_residual_m"] for record in foot_records
    ))
    dynamics_change = _pointwise_dynamics_change(
        original, repaired, fps=fps, addresses=leg_addresses,
    )

    reasons: list[str] = []
    if not after_collision_full["collision_free"]:
        reasons.append("whole_body_collision_remains")
    if not after_support["passes"]:
        reasons.append("support_screen_failed_after_projection")
    if max_raw_residual > config.max_ik_target_residual_m + config.numeric_tolerance:
        reasons.append("ik_target_residual_exceeded")
    if (max_final_residual
            > config.max_post_smoothing_target_residual_m + config.numeric_tolerance):
        reasons.append("post_smoothing_target_residual_exceeded")
    if max_delta > config.max_joint_delta_rad + config.numeric_tolerance:
        reasons.append("joint_delta_budget_exceeded")
    if (dynamics_change["max_pointwise_joint_speed_increase_rads"]
            > config.max_joint_speed_increase_rads + config.numeric_tolerance):
        reasons.append("joint_speed_increase_budget_exceeded")
    if not np.array_equal(repaired[:, :7], original[:, :7]):
        reasons.append("root_changed_internal_error")
    if not np.array_equal(repaired[:, 19:], original[:, 19:]):
        reasons.append("upper_body_changed_internal_error")

    accepted = not reasons
    base_record.update({
        "status": "accepted" if accepted else "rejected",
        "accepted": bool(accepted),
        "reasons": reasons,
        "feet": foot_records,
        "joint_delta_smoothing": smoothing_records,
        "after": {
            "collision": _collision_summary(after_collision_full),
            "support": after_support,
            "max_foot_floor_penetration_m": after_floor,
            "max_joint_speed_rads": after_speed,
        },
        "deformation": {
            "max_leg_joint_delta_rad": max_delta,
            "max_ik_target_residual_m": max_raw_residual,
            "max_post_smoothing_target_residual_m": max_final_residual,
            "max_global_joint_speed_change_rads": float(after_speed - before_speed),
            **dynamics_change,
            "root_exactly_unchanged": bool(np.array_equal(repaired[:, :7], original[:, :7])),
            "upper_body_exactly_unchanged": bool(
                np.array_equal(repaired[:, 19:], original[:, 19:])
            ),
        },
    })
    return FootstepRepairResult(repaired, base_record)
