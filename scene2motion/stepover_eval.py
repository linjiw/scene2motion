"""Operational geometry metrics for step-over motion clips.

The primary quantity is not a toe peak but the tallest corridor-spanning floor box the whole
MuJoCo body clears at a prescribed route position.  Contact and phase descriptors explain why
a clip failed without substituting for that collision test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import mujoco

from .robot import BODY_MARGIN, G1Body
from .scenes import Box, Scene


Side = Literal["left", "right"]
_SIDES: tuple[Side, Side] = ("left", "right")
_FORWARD = np.array([1.0, 0.0, 0.0])
_LATERAL = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class StepOverThresholds:
    """Predeclared gates for a local, kinematic step-over.

    The defaults are deliberately conservative starting points, not fitted constants.  A
    confirmatory experiment should lock them from tracker-successful neutral walks before
    looking at step-over outcomes.
    """

    support_height_m: float = 0.02
    support_speed_mps: float = 0.20
    min_contralateral_support_fraction: float = 0.90
    max_unsupported_run_frames: int = 2
    landing_dwell_frames: int = 3
    landing_horizon_s: float = 0.75
    max_floor_penetration_m: float = 0.02
    lateral_corridor_half_width_m: float = 0.30
    corridor_longitudinal_pad_m: float = 0.30

    def validate(self) -> None:
        if self.support_height_m < 0 or self.support_speed_mps < 0:
            raise ValueError("support height and speed thresholds must be non-negative")
        if not 0 <= self.min_contralateral_support_fraction <= 1:
            raise ValueError("minimum contralateral support fraction must be in [0, 1]")
        if self.max_unsupported_run_frames < 0 or self.landing_dwell_frames < 1:
            raise ValueError("run bound must be non-negative and landing dwell positive")
        if self.landing_horizon_s <= 0 or self.max_floor_penetration_m < 0:
            raise ValueError("landing horizon must be positive and penetration non-negative")
        if self.lateral_corridor_half_width_m < 0 or self.corridor_longitudinal_pad_m < 0:
            raise ValueError("corridor dimensions must be non-negative")


@dataclass(frozen=True)
class KinematicStepEvent:
    """A qpos-derived unilateral step event suitable for donor selection.

    Its first five fields intentionally match :class:`semantic_scaffold.StepEvent`, so it
    can be passed directly to the scaffold builder without consulting ARDY's generated
    ``foot_contacts`` channel.
    """

    frame: int
    side: Side
    relative_lift_m: float
    swing_height_m: float
    stance_height_m: float
    stance_support_fraction: float
    swing_planar_speed_mps: float
    stance_planar_speed_mps: float
    support_window_start_frame: int
    support_window_end_frame: int


def step_scene(obstacle_x: float, height: float, depth: float) -> Scene:
    """One corridor-spanning floor obstacle, without unrelated wall contacts."""
    return Scene(
        scene_id=f"step_probe_h{height:.3f}",
        family="low_obstacle",
        boxes=[Box((obstacle_x, 0.0, height / 2),
                   (depth / 2, 1.4, height / 2), "low_box")],
        start=(0.0, 0.0),
        goal=(8.0, 0.0),
        required_adaptation="step_over",
    )


class BoxHeightProbe:
    """Constant-memory binary search over floor-box height for many trajectories."""

    def __init__(self, obstacle_x: float, depth: float, hi: float = 0.40,
                 tol: float = 0.005):
        if depth <= 0 or hi <= 0 or tol <= 0 or tol >= hi:
            raise ValueError("depth, hi, and tol must be positive, with tol < hi")
        self.obstacle_x, self.depth = float(obstacle_x), float(depth)
        self.hi, self.tol = float(hi), float(tol)
        # One mutable MuJoCo model. Caching a mesh-heavy G1 model for every binary-search
        # height costs several GB over a modest experiment. The obstacle is the only scene
        # geom, so changing its centre/half-height and refreshing model constants is exact
        # for the requested box while keeping memory constant.
        self._body = G1Body(step_scene(self.obstacle_x, self.hi, self.depth))
        if len(self._body.scene_geoms) != 1:
            raise RuntimeError("step probe expected exactly one scene geom")
        self._geom = self._body.scene_geoms[0]

    def body(self, height: float) -> G1Body:
        height = float(height)
        if height <= 0:
            raise ValueError("height must be positive")
        self._body.model.geom_pos[self._geom, 2] = height / 2
        self._body.model.geom_size[self._geom, 2] = height / 2 + self._body.body_margin
        mujoco.mj_setConst(self._body.model, self._body.data)
        return self._body

    def clears(self, qpos: np.ndarray, height: float) -> bool:
        return bool(self.body(height).trajectory_report(qpos)["collision_free"])

    def probe(self, qpos: np.ndarray) -> float:
        """A ``tol``-resolution lower bound, capped at ``hi`` (not an exact maximum)."""
        if not self.clears(qpos, self.tol):
            return 0.0
        if self.clears(qpos, self.hi):
            return self.hi
        lo, hi = self.tol, self.hi
        while hi - lo > self.tol:
            mid = 0.5 * (lo + hi)
            if self.clears(qpos, mid):
                lo = mid
            else:
                hi = mid
        return float(lo)

    def metadata(self) -> dict:
        return {"quantity": "max_box_height_lower_bound_m",
                "resolution_m": self.tol, "cap_m": self.hi,
                "body_margin_included": True}


def _foot_geom_ids(body: G1Body) -> dict[Side, list[int]]:
    geoms: dict[Side, list[int]] = {
        side: [g for g in body.robot_geoms
               if "foot" in body.geom_name[g].lower()
               and side in body.geom_name[g].lower()]
        for side in _SIDES
    }
    if any(not ids for ids in geoms.values()):
        raise ValueError("G1 collision model has no named foot geoms for one or both sides")
    return geoms


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=1)


def foot_kinematics_series(body: G1Body, qpos: np.ndarray, fps: float
                           ) -> dict[Side, dict[str, np.ndarray]]:
    """Exact physical-foot envelopes and finite-difference planar velocities.

    All envelopes use the complete named collision-pad set for a side and
    :meth:`G1Body.geom_extent` with ``extra_margin=0``.  They therefore describe the
    physical foot primitives, not the conservative visual-mesh expansion used by the
    obstacle certificate.  ``forward_representative_m`` and
    ``lateral_representative_m`` are the midpoints of the union envelope.
    """
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or len(qpos) == 0:
        raise ValueError("qpos must be a non-empty (T, nq) array")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")

    geoms = _foot_geom_ids(body)
    raw: dict[Side, dict[str, np.ndarray]] = {}
    for side in _SIDES:
        raw[side] = {
            name: np.empty(len(qpos), dtype=float)
            for name in ("forward_min_m", "forward_max_m", "lateral_min_m",
                         "lateral_max_m", "bottom_clearance_m")
        }

    for t, q in enumerate(qpos):
        body.fk(q)
        for side, ids in geoms.items():
            forward = [body.geom_extent(g, _FORWARD, extra_margin=0.0) for g in ids]
            lateral = [body.geom_extent(g, _LATERAL, extra_margin=0.0) for g in ids]
            vertical = [body.geom_extent(g, body._UP, extra_margin=0.0) for g in ids]
            raw[side]["forward_min_m"][t] = min(x[0] for x in forward)
            raw[side]["forward_max_m"][t] = max(x[1] for x in forward)
            raw[side]["lateral_min_m"][t] = min(x[0] for x in lateral)
            raw[side]["lateral_max_m"][t] = max(x[1] for x in lateral)
            raw[side]["bottom_clearance_m"][t] = min(x[0] for x in vertical)

    for side in _SIDES:
        k = raw[side]
        forward_rep = 0.5 * (k["forward_min_m"] + k["forward_max_m"])
        lateral_rep = 0.5 * (k["lateral_min_m"] + k["lateral_max_m"])
        planar_pos = np.stack([forward_rep, lateral_rep], axis=-1)
        planar_vel = _finite_difference(planar_pos, float(fps))
        k["forward_representative_m"] = forward_rep
        k["lateral_representative_m"] = lateral_rep
        k["planar_velocity_mps"] = planar_vel
        k["planar_speed_mps"] = np.linalg.norm(planar_vel, axis=-1)
    return raw


def foot_clearance_series(body: G1Body, qpos: np.ndarray) -> dict[str, np.ndarray]:
    """Lowest collision-pad surface for each foot at every frame.

    This compatibility API intentionally has no frame-rate argument.  Velocity is irrelevant
    to the returned quantity, so a unit rate is used internally.
    """
    kinematics = foot_kinematics_series(body, qpos, fps=1.0)
    return {side: kinematics[side]["bottom_clearance_m"] for side in _SIDES}


def _support_masks(kinematics: dict[Side, dict[str, np.ndarray]],
                   thresholds: StepOverThresholds) -> dict[Side, np.ndarray]:
    return {
        side: ((kinematics[side]["bottom_clearance_m"] <= thresholds.support_height_m) &
               (kinematics[side]["planar_speed_mps"] <= thresholds.support_speed_mps))
        for side in _SIDES
    }


def select_kinematic_step_event(
        body: G1Body,
        qpos: np.ndarray,
        fps: float,
        frame_window: tuple[int, int] | None = None,
        *,
        thresholds: StepOverThresholds | None = None,
        support_window_s: float = 0.24,
        min_stance_support_fraction: float = 0.90,
        min_relative_lift_m: float = 0.04,
) -> KinematicStepEvent:
    """Select the strongest exact unilateral step without generated contact labels.

    A candidate needs an airborne swing foot, positive relative lift, a contralateral foot
    that satisfies both the height and speed support test for most of a centred window, and
    bounded physical-foot penetration throughout that window.  Selection is deterministic:
    relative lift, then stance stability, then earlier frame, then left side.
    """
    thresholds = thresholds or StepOverThresholds()
    thresholds.validate()
    if support_window_s < 0:
        raise ValueError("support_window_s must be non-negative")
    if not 0 <= min_stance_support_fraction <= 1:
        raise ValueError("min_stance_support_fraction must be in [0, 1]")
    if min_relative_lift_m < 0:
        raise ValueError("min_relative_lift_m must be non-negative")

    kinematics = foot_kinematics_series(body, qpos, fps)
    support = _support_masks(kinematics, thresholds)
    T = len(qpos)
    lo, hi = frame_window if frame_window is not None else (int(0.1 * T), int(0.9 * T))
    lo, hi = max(0, int(lo)), min(T, int(hi))
    if lo >= hi:
        raise ValueError("frame_window contains no frames")
    half_window = int(round(0.5 * support_window_s * fps))

    candidates: list[KinematicStepEvent] = []
    for side, stance in (("left", "right"), ("right", "left")):
        swing_h = kinematics[side]["bottom_clearance_m"]
        stance_h = kinematics[stance]["bottom_clearance_m"]
        for frame in range(lo, hi):
            w0, w1 = max(0, frame - half_window), min(T, frame + half_window + 1)
            stance_fraction = float(support[stance][w0:w1].mean())
            relative_lift = float(swing_h[frame] - stance_h[frame])
            bounded_penetration = min(
                float(kinematics[s]["bottom_clearance_m"][w0:w1].min())
                for s in _SIDES
            ) >= -thresholds.max_floor_penetration_m
            if (support[side][frame] or not support[stance][frame] or
                    stance_fraction < min_stance_support_fraction or
                    relative_lift < min_relative_lift_m or not bounded_penetration):
                continue
            candidates.append(KinematicStepEvent(
                frame=int(frame), side=side, relative_lift_m=relative_lift,
                swing_height_m=float(swing_h[frame]),
                stance_height_m=float(stance_h[frame]),
                stance_support_fraction=stance_fraction,
                swing_planar_speed_mps=float(
                    kinematics[side]["planar_speed_mps"][frame]),
                stance_planar_speed_mps=float(
                    kinematics[stance]["planar_speed_mps"][frame]),
                support_window_start_frame=w0,
                support_window_end_frame=w1 - 1,
            ))
    if not candidates:
        raise ValueError("qpos contains no stable kinematic unilateral-support step event")
    return max(candidates, key=lambda e: (e.relative_lift_m,
                                          e.stance_support_fraction,
                                          -e.frame, e.side == "left"))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive start/end indices of contiguous true runs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _first_dwell(mask: np.ndarray, earliest: int, latest: int,
                 dwell_frames: int) -> tuple[int | None, int | None]:
    for start, end in _true_runs(mask):
        start = max(start, earliest)
        end = min(end, latest)
        if end - start + 1 >= dwell_frames:
            return int(start), int(start + dwell_frames - 1)
    return None, None


def _foot_crossing(
        side: Side,
        kinematics: dict[Side, dict[str, np.ndarray]],
        support: dict[Side, np.ndarray],
        obstacle_lo: float,
        obstacle_hi: float,
        fps: float,
        thresholds: StepOverThresholds,
) -> dict:
    """Find a sampled before→overlap→after transition for one exact foot envelope."""
    foot = kinematics[side]
    other: Side = "right" if side == "left" else "left"
    before = foot["forward_max_m"] < obstacle_lo
    overlap = ((foot["forward_max_m"] >= obstacle_lo) &
               (foot["forward_min_m"] <= obstacle_hi))
    after = foot["forward_min_m"] > obstacle_hi
    selected: tuple[int, int] | None = None
    # With finite-rate data, demand observed envelope overlap.  A before→after teleport
    # between adjacent frames is not evidence of a physically traversed step.
    if bool(before[0]):
        for start, end in _true_runs(overlap):
            if start > 0 and end + 1 < len(overlap) and before[start - 1] and after[end + 1]:
                selected = (start, end)
                break

    crossing = selected is not None
    start, end = selected if selected is not None else (None, None)
    after_frame = (end + 1) if end is not None else None
    before_frame = (start - 1) if start is not None else None
    landing_start = landing_end = None
    contralateral_fraction: float | None = None
    peak_frame: int | None = None
    if crossing:
        assert start is not None and end is not None and after_frame is not None
        overlap_frames = np.arange(start, end + 1)
        contralateral_fraction = float(support[other][overlap_frames].mean())
        peak_frame = int(overlap_frames[
            np.argmax(foot["bottom_clearance_m"][overlap_frames])])
        latest = min(len(after) - 1,
                     after_frame + int(np.ceil(thresholds.landing_horizon_s * fps)))
        landing_start, landing_end = _first_dwell(
            support[side] & after, after_frame, latest, thresholds.landing_dwell_frames)
    else:
        overlap_frames = np.flatnonzero(overlap)
        if len(overlap_frames):
            peak_frame = int(overlap_frames[
                np.argmax(foot["bottom_clearance_m"][overlap_frames])])

    return {
        "side": side,
        "initially_before": bool(before[0]),
        "finishes_after": bool(after[-1]),
        "crossed_before_over_after": bool(crossing),
        "before_frame": before_frame,
        "overlap_start_frame": start,
        "overlap_end_frame": end,
        "overlap_frame_count": int(end - start + 1) if crossing else 0,
        "after_frame": after_frame,
        "contralateral_support_fraction": contralateral_fraction,
        "contralateral_support_gate": bool(
            contralateral_fraction is not None and
            contralateral_fraction >= thresholds.min_contralateral_support_fraction),
        "landing_start_frame": landing_start,
        "landing_end_frame": landing_end,
        "landing_dwell_satisfied": landing_start is not None,
        "peak_over_obstacle_frame": peak_frame,
        "peak_over_obstacle_bottom_clearance_m": (
            float(foot["bottom_clearance_m"][peak_frame])
            if peak_frame is not None else None),
        "minimum_bottom_clearance_m": float(foot["bottom_clearance_m"].min()),
        "maximum_planar_speed_mps": float(foot["planar_speed_mps"].max()),
    }


def _longest_true_run(mask: np.ndarray) -> int:
    runs = _true_runs(mask)
    return max((end - start + 1 for start, end in runs), default=0)


def evaluate_local_step(
        free_body: G1Body,
        obstacle_body: G1Body,
        qpos: np.ndarray,
        obstacle_x: float,
        obstacle_depth: float,
        fps: float,
        *,
        corridor_center_y: float = 0.0,
        expected_lead_side: Side | None = None,
        obstacle_margin_m: float | None = None,
        thresholds: StepOverThresholds | None = None,
) -> dict:
    """Evaluate whether a clip contains a local, contact-preserving step-over.

    Collision freedom alone cannot distinguish a step from stopping, jumping, or walking
    around a box.  This evaluator combines the conservative whole-body obstacle certificate
    with exact physical-foot geometry and a temporal contact topology:

    * both feet are observed before, overlapping, and after the expanded obstacle slab;
    * the lead/trail order is coherent and the other foot supports each overlap;
    * unsupported runs are bounded and both feet exhibit a stable landing beyond the slab;
    * the root traverses the slab inside a lateral corridor; and
    * foot-floor penetration remains bounded.

    The returned nested dictionary is JSON-serializable and keeps every component gate.  The
    expected donor side is diagnostic only: either valid lead side counts as capability.
    """
    thresholds = thresholds or StepOverThresholds()
    thresholds.validate()
    if obstacle_depth <= 0 or not np.isfinite(obstacle_depth):
        raise ValueError("obstacle_depth must be positive and finite")
    if expected_lead_side not in (None, "left", "right"):
        raise ValueError("expected_lead_side must be None, 'left', or 'right'")
    if not np.isfinite(obstacle_x) or not np.isfinite(corridor_center_y):
        raise ValueError("obstacle and corridor coordinates must be finite")

    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or len(qpos) == 0 or qpos.shape[1] < 2:
        raise ValueError("qpos must be a non-empty (T, nq>=2) array")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains non-finite values")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    margin = (float(getattr(obstacle_body, "body_margin", BODY_MARGIN))
              if obstacle_margin_m is None else float(obstacle_margin_m))
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("obstacle_margin_m must be non-negative and finite")
    obstacle_lo = float(obstacle_x - 0.5 * obstacle_depth - margin)
    obstacle_hi = float(obstacle_x + 0.5 * obstacle_depth + margin)

    kinematics = foot_kinematics_series(free_body, qpos, fps)
    support = _support_masks(kinematics, thresholds)
    feet = {
        side: _foot_crossing(side, kinematics, support, obstacle_lo, obstacle_hi,
                             float(fps), thresholds)
        for side in _SIDES
    }

    both_cross = all(feet[s]["crossed_before_over_after"] for s in _SIDES)
    lead_side: Side | None = None
    trail_side: Side | None = None
    lead_trail_order = False
    lead_landing_timing = False
    if both_cross:
        left_after = int(feet["left"]["after_frame"])
        right_after = int(feet["right"]["after_frame"])
        if left_after != right_after:
            lead_side = "left" if left_after < right_after else "right"
            trail_side = "right" if lead_side == "left" else "left"
            lead_trail_order = bool(
                int(feet[lead_side]["overlap_start_frame"]) <
                int(feet[trail_side]["overlap_start_frame"]))
            landing_end = feet[lead_side]["landing_end_frame"]
            trail_overlap_end = feet[trail_side]["overlap_end_frame"]
            lead_landing_timing = bool(
                landing_end is not None and trail_overlap_end is not None and
                int(landing_end) <= int(trail_overlap_end))

    if both_cross:
        local_start = min(int(feet[s]["overlap_start_frame"]) for s in _SIDES)
        local_end = max(
            int(feet[s]["landing_end_frame"]
                if feet[s]["landing_end_frame"] is not None
                else feet[s]["after_frame"])
            for s in _SIDES)
    else:
        overlap_indices = np.flatnonzero(
            ((kinematics["left"]["forward_max_m"] >= obstacle_lo) &
             (kinematics["left"]["forward_min_m"] <= obstacle_hi)) |
            ((kinematics["right"]["forward_max_m"] >= obstacle_lo) &
             (kinematics["right"]["forward_min_m"] <= obstacle_hi)))
        local_start = int(overlap_indices[0]) if len(overlap_indices) else 0
        local_end = int(overlap_indices[-1]) if len(overlap_indices) else len(qpos) - 1
    local = slice(local_start, local_end + 1)
    unsupported = ~(support["left"] | support["right"])
    unsupported_local = unsupported[local]
    max_unsupported_run = _longest_true_run(unsupported_local)

    root_forward = qpos[:, 0]
    root_lateral = qpos[:, 1]
    root_traversal = bool(root_forward[0] < obstacle_lo and root_forward[-1] > obstacle_hi)
    corridor_lo = obstacle_lo - thresholds.corridor_longitudinal_pad_m
    corridor_hi = obstacle_hi + thresholds.corridor_longitudinal_pad_m
    corridor_frames = np.flatnonzero(
        (root_forward >= corridor_lo) & (root_forward <= corridor_hi))
    if len(corridor_frames):
        lateral_deviation = np.abs(root_lateral[corridor_frames] - corridor_center_y)
        max_lateral_deviation: float | None = float(lateral_deviation.max())
    else:
        max_lateral_deviation = None
    lateral_corridor = bool(
        max_lateral_deviation is not None and
        max_lateral_deviation <= thresholds.lateral_corridor_half_width_m)

    min_foot_bottom = min(
        float(kinematics[s]["bottom_clearance_m"].min()) for s in _SIDES)
    max_floor_penetration = max(0.0, -min_foot_bottom)
    collision = obstacle_body.trajectory_report(qpos)
    whole_body_collision_free = bool(collision["collision_free"])
    both_finish = all(feet[s]["finishes_after"] for s in _SIDES)

    lead_support_gate = bool(
        lead_side is not None and feet[lead_side]["contralateral_support_gate"])
    trail_support_gate = bool(
        trail_side is not None and feet[trail_side]["contralateral_support_gate"])
    lead_landing = bool(
        lead_side is not None and feet[lead_side]["landing_dwell_satisfied"])
    trail_landing = bool(
        trail_side is not None and feet[trail_side]["landing_dwell_satisfied"])
    gates = {
        "whole_body_collision_free": whole_body_collision_free,
        "root_traversal": root_traversal,
        "lateral_corridor": lateral_corridor,
        "both_feet_cross_before_over_after": both_cross,
        "lead_trail_order": lead_trail_order,
        "lead_overlap_has_trailing_support": lead_support_gate,
        "trail_overlap_has_lead_support": trail_support_gate,
        "bounded_unsupported_run": (
            max_unsupported_run <= thresholds.max_unsupported_run_frames),
        "lead_landing_dwell": lead_landing,
        "trail_landing_dwell": trail_landing,
        "lead_lands_before_or_during_trailing_overlap": lead_landing_timing,
        "both_feet_finish_beyond": both_finish,
        "bounded_floor_penetration": (
            max_floor_penetration <= thresholds.max_floor_penetration_m),
    }
    local_step_success = bool(all(gates.values()))
    expected_match = (None if expected_lead_side is None or lead_side is None
                      else bool(lead_side == expected_lead_side))

    return {
        "local_step_success": local_step_success,
        "gates": gates,
        "thresholds": asdict(thresholds),
        "obstacle": {
            "center_x_m": float(obstacle_x),
            "nominal_depth_m": float(obstacle_depth),
            "conservative_margin_m": margin,
            "expanded_forward_min_m": obstacle_lo,
            "expanded_forward_max_m": obstacle_hi,
        },
        "collision": {
            "collision_free": whole_body_collision_free,
            "min_clearance_m": (float(collision["min_clearance_m"])
                                  if "min_clearance_m" in collision else None),
            "max_penetration_m": (float(collision["max_penetration_m"])
                                   if "max_penetration_m" in collision else None),
        },
        "root": {
            "starts_before": bool(root_forward[0] < obstacle_lo),
            "finishes_after": bool(root_forward[-1] > obstacle_hi),
            "traverses": root_traversal,
            "corridor_center_y_m": float(corridor_center_y),
            "max_lateral_deviation_m": max_lateral_deviation,
            "inside_lateral_corridor": lateral_corridor,
        },
        "crossing": {
            "lead_side": lead_side,
            "trail_side": trail_side,
            "lead_trail_order_valid": lead_trail_order,
            "expected_lead_side": expected_lead_side,
            "lead_matches_expected": expected_match,
            "lead_landing_timing_valid": lead_landing_timing,
        },
        "support": {
            "left_support_fraction_local": float(support["left"][local].mean()),
            "right_support_fraction_local": float(support["right"][local].mean()),
            "unsupported_fraction_local": float(unsupported_local.mean()),
            "max_unsupported_run_frames": int(max_unsupported_run),
            "max_unsupported_run_s": float(max_unsupported_run / fps),
            "local_start_frame": int(local_start),
            "local_end_frame": int(local_end),
        },
        "floor": {
            "minimum_foot_bottom_m": min_foot_bottom,
            "max_foot_floor_penetration_m": max_floor_penetration,
        },
        "feet": feet,
    }


# Descriptive alias for callers that treat the result as a metric bundle.
local_step_metrics = evaluate_local_step


def motion_metrics(free_body: G1Body, obstacle_body: G1Body, probe: BoxHeightProbe,
                   qpos: np.ndarray, root_xz: np.ndarray, obstacle_x: float,
                   swing_side: str, interaction_half_width: float = 0.8, *,
                   fps: float = 25.0,
                   thresholds: StepOverThresholds | None = None) -> dict:
    """Operational traversal and foot-local spatial descriptors for one generated clip.

    ``interaction_half_width`` remains for source compatibility but is no longer used to
    define the event.  The older pelvis-centred +/-0.8 m window could span two gait cycles and
    systematically displaced a foot-scaffold phase measurement.  Event frames now come from
    exact physical-foot overlap with the conservative obstacle slab.
    """
    if swing_side not in ("left", "right"):
        raise ValueError("swing_side must be 'left' or 'right'")
    qpos = np.asarray(qpos)
    root_xz = np.asarray(root_xz)
    if len(qpos) != len(root_xz):
        raise ValueError("qpos and root_xz must have the same number of frames")
    thresholds = thresholds or StepOverThresholds()
    thresholds.validate()
    local_step = evaluate_local_step(
        free_body, obstacle_body, qpos, obstacle_x, probe.depth, fps,
        expected_lead_side=swing_side, thresholds=thresholds)
    fixed = local_step["collision"]
    floor = free_body.trajectory_report(qpos)
    feet = foot_kinematics_series(free_body, qpos, fps)
    observed_side = local_step["crossing"]["lead_side"] or swing_side
    obstacle_lo = local_step["obstacle"]["expanded_forward_min_m"]
    obstacle_hi = local_step["obstacle"]["expanded_forward_max_m"]
    overlap = np.where(
        (feet[observed_side]["forward_max_m"] >= obstacle_lo) &
        (feet[observed_side]["forward_min_m"] <= obstacle_hi))[0]
    if not len(overlap):
        overlap = np.asarray([int(np.argmin(np.abs(
            feet[observed_side]["forward_representative_m"] - obstacle_x)))])
    cross = int(overlap[np.argmin(np.abs(
        feet[observed_side]["forward_representative_m"][overlap] - obstacle_x))])
    peak = int(overlap[np.argmax(feet[observed_side]["bottom_clearance_m"][overlap])])
    support = _support_masks(feet, thresholds)
    local_start = local_step["support"]["local_start_frame"]
    local_end = local_step["support"]["local_end_frame"]
    local_frames = np.arange(local_start, local_end + 1)
    supported = np.stack([support["left"][local_frames],
                          support["right"][local_frames]], -1)
    progress = float(qpos[-1, 0] - qpos[0, 0])
    planned = float(root_xz[-1, 1] - root_xz[0, 1])
    path_world = np.stack([qpos[:, 1], qpos[:, 0]], -1)
    return {
        "obstacle_collision_free": bool(fixed["collision_free"]),
        "obstacle_min_clearance_m": float(fixed["min_clearance_m"]),
        "max_box_height_lower_bound_m": float(probe.probe(qpos)),
        "progress_m": progress,
        "progress_ratio": progress / max(abs(planned), 1e-9),
        "path_error_m": float(np.linalg.norm(path_world - root_xz, axis=-1).mean()),
        "mean_support_feet": float(supported.sum(-1).mean()),
        "bilateral_flight_fraction": float((supported.sum(-1) == 0).mean()),
        "selected_lead_side": observed_side,
        "lead_matches_donor_side": local_step["crossing"]["lead_matches_expected"],
        "swing_foot_peak_m": float(feet[observed_side]["bottom_clearance_m"][peak]),
        "swing_foot_at_crossing_m": float(
            feet[observed_side]["bottom_clearance_m"][cross]),
        "swing_peak_frame": peak,
        "crossing_frame": cross,
        "phase_error_frames": int(peak - cross),
        "phase_error_m": float(
            feet[observed_side]["forward_representative_m"][peak] - obstacle_x),
        "max_foot_floor_penetration_m": float(floor["max_foot_floor_penetration_m"]),
        "local_step_success": bool(local_step["local_step_success"]),
        "local_step": local_step,
    }
