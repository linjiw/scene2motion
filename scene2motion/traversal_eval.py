"""Scene-level traversal evaluation: start, goal and obstacles in; an outcome class out.

This is the endpoint for obstacle-present execution studies (EXP-029 onward). It answers, for
one tracked trajectory in one :class:`~scene2motion.scenes.Scene`, the two questions the project
cares about:

* **did it collide** — with the obstacle, or with a corridor wall, and
* **did it complete the local traversal** — pass through the specified corridor, past the
  obstacle, and reach the goal, upright and within the time limit.

**Local traversal is not navigation.** Completion here requires passing the obstacle *inside the
corridor*: walking around it does not count. Navigation — reaching a destination where going
around may be the best answer — is a different task with a different success label, and this
module deliberately does not score it.

Relationship to the existing evaluators, none of which this replaces:

* ``experiments/exp022_exact_tracking_bridge.score_trajectory`` is the single-obstacle,
  route-progress endpoint that the landed EXP-022A/028 receipts depend on. It is frozen; this
  module is the general scene version and does not modify it.
* ``experiments/exp028_termination_free_rollouts`` owns the preregistered physical-outcome
  ordering ``fell > stalled > walked_through > cleared``. The fall thresholds here are the same
  constants (:data:`FALL_PELVIS_Z_M`, :data:`FALL_UP_Z`), pinned by a test, and ``fell`` keeps
  its precedence.

What this module adds is the goal, the corridor and an **obstacle-versus-wall split** of the
collision, so that "hit the beam" and "left the corridor" are never the same number.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scene2motion.scenes import Box, Scene

# Fall thresholds, identical to the preregistered EXP-028 constants.
FALL_PELVIS_Z_M = 0.50
FALL_UP_Z = 0.70

#: Outcome classes, in the precedence order :func:`classify` applies.  ``rejected`` is never
#: produced from a trajectory: it is the class a driver records for an assigned trial that was
#: screened out and never executed, so that rates stay over *all assigned trials*.
OUTCOMES = (
    "rejected",
    "fell",
    "collided_obstacle",
    "collided_wall",
    "cutoff",
    "timeout",
    "stalled",
    "completed",
)

WALL_LABEL_PREFIX = "wall"


def up_z(quat_wxyz: np.ndarray) -> np.ndarray:
    """Vertical component of the body up-axis; matches EXP-028's helper."""
    q = np.asarray(quat_wxyz, dtype=float)
    return 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)


@dataclass(frozen=True)
class TraversalCriteria:
    """Everything that must be fixed before a trial is run."""

    goal_tolerance_m: float = 0.5
    #: Half-width of the corridor the robot must stay inside while passing the obstacle.
    #: ``None`` takes it from ``scene.meta["corridor_half"]``, else from the wall boxes.
    corridor_half_width_m: float | None = None
    #: Wall clock the rollout is allowed; ``None`` disables the timeout class.
    time_limit_s: float | None = None
    fall_pelvis_z_m: float = FALL_PELVIS_Z_M
    fall_up_z: float = FALL_UP_Z
    #: Extra clearance required beyond simple collision freedom, for a graded endpoint.
    required_clearance_m: float = 0.0


def is_wall(box: Box) -> bool:
    return str(box.label).startswith(WALL_LABEL_PREFIX)


def obstacle_boxes(scene: Scene) -> list[Box]:
    return [b for b in scene.boxes if not is_wall(b)]


def wall_boxes(scene: Scene) -> list[Box]:
    return [b for b in scene.boxes if is_wall(b)]


def sub_scene(scene: Scene, boxes: Sequence[Box]) -> Scene:
    return replace(scene, boxes=list(boxes))


def obstacle_span_x(scene: Scene) -> tuple[float, float] | None:
    """(near edge, far edge) in x over the non-wall boxes; ``None`` when there are none."""
    boxes = obstacle_boxes(scene)
    if not boxes:
        return None
    lo = min(float(b.center[0]) - float(b.half[0]) for b in boxes)
    hi = max(float(b.center[0]) + float(b.half[0]) for b in boxes)
    return lo, hi


def corridor_half_width(scene: Scene, criteria: TraversalCriteria) -> float | None:
    """Explicit criterion, else the scene's own metadata, else the inner face of the walls."""
    if criteria.corridor_half_width_m is not None:
        return float(criteria.corridor_half_width_m)
    meta_half = (scene.meta or {}).get("corridor_half")
    if meta_half is not None:
        return float(meta_half)
    walls = wall_boxes(scene)
    if not walls:
        return None
    return min(abs(float(b.center[1])) - float(b.half[1]) for b in walls)


def fall_report(qpos: np.ndarray, criteria: TraversalCriteria,
                sample_dt_s: float) -> dict[str, Any]:
    """Same rule as EXP-028: pelvis below the height bound, or the up-axis tilted past it."""
    q = np.asarray(qpos, dtype=float)
    pelvis_z = q[:, 2]
    up = up_z(q[:, 3:7])
    low = pelvis_z < criteria.fall_pelvis_z_m
    tilted = up < criteria.fall_up_z
    hits = np.flatnonzero(low | tilted)
    first = int(hits[0]) if len(hits) else None
    return {
        "fell": bool(len(hits)),
        "fell_first_sample": first,
        "fell_first_time_s": None if first is None else float((first + 1) * sample_dt_s),
        "fell_by_pelvis_height": bool(low.any()),
        "fell_by_tilt": bool(tilted.any()),
        "min_pelvis_z_m": float(pelvis_z.min()),
        "min_up_z": float(up.min()),
    }


def _default_collision_report(scene: Scene, qpos: np.ndarray) -> Mapping[str, Any]:
    """Whole-body collision of the trajectory against the scene's boxes (MuJoCo)."""
    from scene2motion.robot import G1Body

    return G1Body(scene=scene).trajectory_report(np.asarray(qpos, dtype=float))


def _empty_collision_report() -> dict[str, Any]:
    return {"collision_free": True, "penetration_frames": 0, "max_penetration_m": 0.0,
            "min_clearance_m": float("inf"), "worst": {"frame": -1, "depth_m": 0.0}}


def classify(
    *,
    fell: bool,
    collided_obstacle: bool,
    collided_wall: bool,
    terminated: bool,
    timed_out: bool,
    reached_goal: bool,
    passed_obstacle_in_corridor: bool,
    clearance_ok: bool,
) -> str:
    """Precedence: fell > obstacle collision > wall collision > cutoff > timeout > stalled.

    ``fell`` keeps the first position it holds in EXP-028's preregistered ordering. The two
    collision classes are separated so that hitting the obstacle and leaving the corridor are
    never reported as one number. Everything that is not a named failure and did not complete
    is ``stalled``; a trial that satisfies every condition is ``completed``.
    """
    if fell:
        return "fell"
    if collided_obstacle:
        return "collided_obstacle"
    if collided_wall:
        return "collided_wall"
    if terminated:
        return "cutoff"
    if timed_out:
        return "timeout"
    if reached_goal and passed_obstacle_in_corridor and clearance_ok:
        return "completed"
    return "stalled"


def rejected_record(reason: str, **extra: Any) -> dict[str, Any]:
    """The record for an assigned trial that was screened out and never executed.

    Drivers must emit one of these instead of dropping the trial, so that a rule which rejects
    everything cannot read as perfect task performance.
    """
    return {"outcome": "rejected", "rejection_reason": str(reason), "executed": False, **extra}


def evaluate_traversal(
    qpos: np.ndarray,
    scene: Scene,
    *,
    terminated: bool = False,
    sample_dt_s: float = 0.02,
    criteria: TraversalCriteria | None = None,
    collision_fn: Callable[[Scene, np.ndarray], Mapping[str, Any]] = _default_collision_report,
) -> dict[str, Any]:
    """Classify one tracked trajectory against a scene's start, goal and obstacles.

    ``qpos`` is a finite (T, >=7) array whose first seven columns are the root position and
    the root quaternion in (w, x, y, z) order — the archived achieved-state layout.
    ``terminated`` is the tracker evaluator's cutoff flag, which is ``False`` by construction in
    a termination-free study.
    """
    criteria = criteria or TraversalCriteria()
    q = np.asarray(qpos, dtype=float)
    if q.ndim != 2 or q.shape[1] < 7:
        raise ValueError(f"qpos must be (T, >=7); got {q.shape}")
    if len(q) == 0:
        raise ValueError("qpos is empty; record a rejected or failed-launch trial instead")
    if not np.isfinite(q[:, :7]).all():
        raise ValueError("qpos root columns must be finite")

    duration_s = float(len(q) * sample_dt_s)
    timed_out = bool(criteria.time_limit_s is not None and duration_s > criteria.time_limit_s)

    # --- collision, split into the obstacle and the corridor walls
    obstacles, walls = obstacle_boxes(scene), wall_boxes(scene)
    obstacle_report = (dict(collision_fn(sub_scene(scene, obstacles), q)) if obstacles
                       else _empty_collision_report())
    wall_report = (dict(collision_fn(sub_scene(scene, walls), q)) if walls
                   else _empty_collision_report())
    collided_obstacle = not bool(obstacle_report["collision_free"])
    collided_wall = not bool(wall_report["collision_free"])
    obstacle_clearance = float(obstacle_report.get("min_clearance_m", float("inf")))
    clearance_ok = bool(not collided_obstacle
                        and obstacle_clearance >= criteria.required_clearance_m)

    # --- progress toward the goal, in world space
    start = np.asarray(scene.start, dtype=float)
    goal = np.asarray(scene.goal, dtype=float)
    root_xy = q[:, :2]
    goal_distance = float(np.linalg.norm(root_xy - goal, axis=1).min())
    final_goal_distance = float(np.linalg.norm(root_xy[-1] - goal))
    reached_goal = bool(goal_distance <= criteria.goal_tolerance_m)
    planned = float(np.linalg.norm(goal - start))
    travelled = float(np.linalg.norm(root_xy[-1] - start))
    toward_goal = float(np.dot(root_xy[-1] - start, goal - start) / planned) if planned else 0.0

    # --- passage through the obstacle, inside the corridor
    span = obstacle_span_x(scene)
    half_width = corridor_half_width(scene, criteria)
    if span is None:
        passed_obstacle = True
        pass_sample: int | None = None
        stayed_in_corridor = True
    else:
        _, far_edge = span
        beyond = np.flatnonzero(q[:, 0] >= far_edge)
        pass_sample = int(beyond[0]) if len(beyond) else None
        passed_obstacle = pass_sample is not None
        if half_width is None or pass_sample is None:
            stayed_in_corridor = passed_obstacle
        else:
            near_edge = span[0]
            inside_span = (q[:, 0] >= near_edge) & (q[:, 0] <= far_edge)
            lateral = np.abs(q[inside_span, 1]) if inside_span.any() else np.abs(
                q[[pass_sample], 1])
            stayed_in_corridor = bool(lateral.max() <= half_width)
    passed_in_corridor = bool(passed_obstacle and stayed_in_corridor)

    fall = fall_report(q, criteria, sample_dt_s)
    outcome = classify(
        fell=fall["fell"],
        collided_obstacle=collided_obstacle,
        collided_wall=collided_wall,
        terminated=bool(terminated),
        timed_out=timed_out,
        reached_goal=reached_goal,
        passed_obstacle_in_corridor=passed_in_corridor,
        clearance_ok=clearance_ok,
    )

    return {
        "outcome": outcome,
        "executed": True,
        "scene_id": scene.scene_id,
        "samples": int(len(q)),
        "duration_s": duration_s,
        "timed_out": timed_out,
        "tracker_terminated": bool(terminated),
        # collision, obstacle and corridor kept apart
        "collided_obstacle": collided_obstacle,
        "collided_wall": collided_wall,
        "obstacle_min_clearance_m": obstacle_clearance,
        "obstacle_max_penetration_m": float(obstacle_report.get("max_penetration_m", 0.0)),
        "obstacle_first_collision_sample": (
            int(obstacle_report.get("worst", {}).get("frame", -1)) if collided_obstacle else None),
        "wall_max_penetration_m": float(wall_report.get("max_penetration_m", 0.0)),
        "clearance_ok": clearance_ok,
        "required_clearance_m": criteria.required_clearance_m,
        # traversal geometry
        "passed_obstacle": passed_obstacle,
        "passed_within_corridor": passed_in_corridor,
        "pass_sample": pass_sample,
        "corridor_half_width_m": half_width,
        "obstacle_span_x_m": list(span) if span is not None else None,
        # goal, in world space
        "reached_goal": reached_goal,
        "goal_distance_m": goal_distance,
        "final_goal_distance_m": final_goal_distance,
        "planned_distance_m": planned,
        "travelled_distance_m": travelled,
        "progress_toward_goal_m": toward_goal,
        "progress_fraction": float(toward_goal / planned) if planned else 0.0,
        **fall,
    }


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Outcome breakdown over **all assigned trials**, including rejected ones.

    ``completion_rate`` divides by every assigned trial, so a rule that rejects everything
    scores zero rather than looking perfect. ``completion_rate_of_executed`` is reported beside
    it, never instead of it.
    """
    total = len(records)
    counts = {name: 0 for name in OUTCOMES}
    for record in records:
        outcome = str(record.get("outcome", ""))
        if outcome not in counts:
            raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
        counts[outcome] += 1
    executed = total - counts["rejected"]
    return {
        "n_assigned_trials": total,
        "n_executed": executed,
        "n_rejected_before_execution": counts["rejected"],
        "outcomes": counts,
        "completion_rate": (counts["completed"] / total) if total else 0.0,
        "completion_rate_of_executed": (counts["completed"] / executed) if executed else 0.0,
        "collision_rate": ((counts["collided_obstacle"] + counts["collided_wall"]) / total
                           if total else 0.0),
    }
