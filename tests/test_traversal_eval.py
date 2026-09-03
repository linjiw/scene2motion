"""CPU tests for the scene-level traversal evaluator.

Pins the outcome precedence, the obstacle-versus-wall collision split, the corridor rule that
makes walking around a failure, and the all-assigned-trials denominator.
"""

from __future__ import annotations

import numpy as np
import pytest

from scene2motion import traversal_eval as te
from scene2motion.scenes import Box, Scene


# ---------------------------------------------------------------- fixtures

def _scene(corridor_half=1.4, beam_x=4.0, goal_x=8.0):
    """A beam across a walled corridor, start at the origin, goal beyond the beam."""
    boxes = [
        Box((3.5, +corridor_half + 0.05, 1.25), (4.5, 0.05, 1.25), "wall_left"),
        Box((3.5, -corridor_half - 0.05, 1.25), (4.5, 0.05, 1.25), "wall_right"),
        Box((beam_x, 0.0, 1.30), (0.12, corridor_half, 0.125), "beam"),
    ]
    return Scene("beam_test", "overhead_beam", boxes, start=(0.0, 0.0), goal=(goal_x, 0.0),
                 meta={"corridor_half": corridor_half, "beam_x": beam_x})


def _qpos(xs, ys=None, z=0.75, up=1.0, n=None):
    """Root trajectory with an upright quaternion unless `up` says otherwise."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs) if n is None else n
    ys = np.zeros(n) if ys is None else np.asarray(ys, dtype=float)
    q = np.zeros((n, 36))
    q[:, 0], q[:, 1], q[:, 2] = xs, ys, z
    # up_z = 1 - 2(qy^2 + qz^2): solve for a pitch-only quaternion giving the wanted up.
    q[:, 3] = 1.0
    q[:, 5] = np.sqrt(max(0.0, (1.0 - up) / 2.0))
    return q


def _collision(obstacle_hit=False, wall_hit=False, clearance=0.5):
    """Injected collision reporter keyed on whether the sub-scene holds walls."""
    def report(scene, qpos):
        walls = any(te.is_wall(b) for b in scene.boxes)
        hit = wall_hit if walls else obstacle_hit
        return {
            "collision_free": not hit,
            "penetration_frames": 3 if hit else 0,
            "max_penetration_m": 0.02 if hit else 0.0,
            "min_clearance_m": -0.02 if hit else clearance,
            "worst": {"frame": 11 if hit else -1, "depth_m": 0.02 if hit else 0.0},
        }
    return report


def _evaluate(qpos, scene=None, **kwargs):
    kwargs.setdefault("collision_fn", _collision())
    return te.evaluate_traversal(qpos, scene or _scene(), **kwargs)


# ---------------------------------------------------------------- the happy path

def test_a_run_that_passes_the_beam_and_reaches_the_goal_completes():
    result = _evaluate(_qpos(np.linspace(0.0, 8.0, 60)))
    assert result["outcome"] == "completed"
    assert result["passed_obstacle"] and result["passed_within_corridor"]
    assert result["reached_goal"]
    assert result["collided_obstacle"] is False and result["collided_wall"] is False
    assert result["progress_fraction"] == pytest.approx(1.0, abs=0.01)


def test_pass_sample_is_the_first_sample_beyond_the_far_edge():
    result = _evaluate(_qpos(np.linspace(0.0, 8.0, 81)))          # 0.1 m per sample
    far_edge = result["obstacle_span_x_m"][1]
    assert result["pass_sample"] is not None
    xs = np.linspace(0.0, 8.0, 81)
    assert xs[result["pass_sample"]] >= far_edge
    assert xs[result["pass_sample"] - 1] < far_edge


# ---------------------------------------------------------------- local traversal, not navigation

def test_walking_around_the_beam_is_not_a_completed_traversal():
    """Reaching the goal outside the corridor must not score as local traversal."""
    xs = np.linspace(0.0, 8.0, 60)
    ys = np.where((xs > 3.0) & (xs < 5.0), 2.5, 0.0)             # detour past the corridor
    result = _evaluate(_qpos(xs, ys))
    assert result["reached_goal"] is True
    assert result["passed_obstacle"] is True
    assert result["passed_within_corridor"] is False
    assert result["outcome"] == "stalled"                         # not "completed"


def test_staying_inside_the_corridor_is_judged_across_the_whole_obstacle_span():
    xs = np.linspace(0.0, 8.0, 200)
    ys = np.where((xs > 3.8) & (xs < 3.95), 2.0, 0.0)            # brief excursion inside the span
    result = _evaluate(_qpos(xs, ys))
    assert result["passed_within_corridor"] is False


# ---------------------------------------------------------------- collision split

def test_hitting_the_obstacle_and_hitting_a_wall_are_different_outcomes():
    xs = np.linspace(0.0, 8.0, 60)
    obstacle = _evaluate(_qpos(xs), collision_fn=_collision(obstacle_hit=True))
    wall = _evaluate(_qpos(xs), collision_fn=_collision(wall_hit=True))
    assert obstacle["outcome"] == "collided_obstacle"
    assert obstacle["collided_wall"] is False
    assert wall["outcome"] == "collided_wall"
    assert wall["collided_obstacle"] is False


def test_obstacle_collision_records_its_depth_and_first_sample():
    result = _evaluate(_qpos(np.linspace(0.0, 8.0, 60)),
                       collision_fn=_collision(obstacle_hit=True))
    assert result["obstacle_max_penetration_m"] == pytest.approx(0.02)
    assert result["obstacle_first_collision_sample"] == 11
    assert result["clearance_ok"] is False


def test_a_graded_clearance_requirement_can_fail_a_collision_free_run():
    xs = np.linspace(0.0, 8.0, 60)
    criteria = te.TraversalCriteria(required_clearance_m=0.18)
    tight = _evaluate(_qpos(xs), criteria=criteria, collision_fn=_collision(clearance=0.05))
    loose = _evaluate(_qpos(xs), criteria=criteria, collision_fn=_collision(clearance=0.25))
    assert tight["collided_obstacle"] is False and tight["clearance_ok"] is False
    assert tight["outcome"] == "stalled"
    assert loose["outcome"] == "completed"


# ---------------------------------------------------------------- the other failure classes

def test_a_fall_outranks_every_other_class():
    xs = np.linspace(0.0, 8.0, 60)
    result = _evaluate(_qpos(xs, z=0.30), terminated=True,
                       collision_fn=_collision(obstacle_hit=True))
    assert result["outcome"] == "fell"
    assert result["fell_by_pelvis_height"] is True


def test_a_tilt_counts_as_a_fall_at_the_exp028_threshold():
    xs = np.linspace(0.0, 8.0, 60)
    assert _evaluate(_qpos(xs, up=0.60))["outcome"] == "fell"
    assert _evaluate(_qpos(xs, up=0.80))["outcome"] == "completed"


def test_an_evaluator_cutoff_is_reported_when_nothing_worse_happened():
    result = _evaluate(_qpos(np.linspace(0.0, 8.0, 60)), terminated=True)
    assert result["outcome"] == "cutoff"
    assert result["tracker_terminated"] is True


def test_a_run_that_never_reaches_the_beam_stalls():
    result = _evaluate(_qpos(np.linspace(0.0, 2.0, 60)))
    assert result["outcome"] == "stalled"
    assert result["passed_obstacle"] is False
    assert result["reached_goal"] is False
    assert result["progress_fraction"] == pytest.approx(0.25, abs=0.01)


def test_exceeding_the_time_limit_is_its_own_class():
    xs = np.linspace(0.0, 8.0, 600)                               # 12 s at 50 Hz
    criteria = te.TraversalCriteria(time_limit_s=10.0)
    result = _evaluate(_qpos(xs), criteria=criteria)
    assert result["timed_out"] is True
    assert result["outcome"] == "timeout"


def test_passing_and_reaching_the_goal_inside_the_limit_is_not_a_timeout():
    criteria = te.TraversalCriteria(time_limit_s=10.0)
    result = _evaluate(_qpos(np.linspace(0.0, 8.0, 400)), criteria=criteria)
    assert result["timed_out"] is False and result["outcome"] == "completed"


# ---------------------------------------------------------------- scene helpers

def test_walls_and_obstacles_are_split_by_label():
    scene = _scene()
    assert [b.label for b in te.obstacle_boxes(scene)] == ["beam"]
    assert sorted(b.label for b in te.wall_boxes(scene)) == ["wall_left", "wall_right"]


def test_obstacle_span_covers_every_non_wall_box():
    scene = _scene()
    scene.boxes.append(Box((5.0, 0.0, 0.05), (0.2, 1.0, 0.05), "low_box"))
    lo, hi = te.obstacle_span_x(scene)
    assert lo == pytest.approx(3.88)
    assert hi == pytest.approx(5.2)


def test_corridor_half_width_prefers_the_criterion_then_meta_then_the_walls():
    scene = _scene(corridor_half=1.4)
    assert te.corridor_half_width(scene, te.TraversalCriteria(corridor_half_width_m=0.5)) == 0.5
    assert te.corridor_half_width(scene, te.TraversalCriteria()) == pytest.approx(1.4)
    bare = Scene("bare", "f", te.wall_boxes(scene), start=(0.0, 0.0), goal=(8.0, 0.0))
    assert te.corridor_half_width(bare, te.TraversalCriteria()) == pytest.approx(1.4)


def test_a_scene_without_obstacles_is_passable_by_definition():
    scene = Scene("empty", "f", [], start=(0.0, 0.0), goal=(8.0, 0.0))
    result = _evaluate(_qpos(np.linspace(0.0, 8.0, 60)), scene=scene)
    assert result["obstacle_span_x_m"] is None
    assert result["outcome"] == "completed"


# ---------------------------------------------------------------- input validation

def test_an_empty_trajectory_is_rejected_rather_than_scored():
    with pytest.raises(ValueError, match="empty"):
        _evaluate(np.zeros((0, 36)))


def test_a_non_finite_root_is_rejected():
    q = _qpos(np.linspace(0.0, 8.0, 10))
    q[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _evaluate(q)


def test_a_too_narrow_trajectory_is_rejected():
    with pytest.raises(ValueError, match=r"\(T, >=7\)"):
        _evaluate(np.zeros((10, 5)))


# ---------------------------------------------------------------- reporting over all trials

def test_rejected_trials_carry_a_reason_and_are_not_executed():
    record = te.rejected_record("screen predicts an evaluator cutoff", seed=4400)
    assert record["outcome"] == "rejected"
    assert record["executed"] is False
    assert record["seed"] == 4400


def test_rejecting_everything_scores_zero_not_perfect():
    """The denominator rule: a rule that executes nothing must not look flawless."""
    records = [te.rejected_record("screened out") for _ in range(8)]
    summary = te.summarise(records)
    assert summary["n_assigned_trials"] == 8
    assert summary["n_executed"] == 0
    assert summary["completion_rate"] == 0.0
    assert summary["completion_rate_of_executed"] == 0.0


def test_completion_rate_uses_all_assigned_trials_and_reports_the_executed_one_beside_it():
    records = (
        [{"outcome": "completed"}] * 2
        + [{"outcome": "collided_obstacle"}]
        + [te.rejected_record("screened out")] * 5
    )
    summary = te.summarise(records)
    assert summary["n_assigned_trials"] == 8
    assert summary["completion_rate"] == pytest.approx(2 / 8)
    assert summary["completion_rate_of_executed"] == pytest.approx(2 / 3)
    assert summary["collision_rate"] == pytest.approx(1 / 8)


def test_summarise_rejects_an_unknown_outcome():
    with pytest.raises(ValueError, match="unknown outcome"):
        te.summarise([{"outcome": "succeeded"}])


def test_every_classify_branch_is_reachable_and_ordered():
    base = dict(fell=False, collided_obstacle=False, collided_wall=False, terminated=False,
                timed_out=False, reached_goal=True, passed_obstacle_in_corridor=True,
                clearance_ok=True)
    assert te.classify(**base) == "completed"
    assert te.classify(**{**base, "timed_out": True}) == "timeout"
    assert te.classify(**{**base, "terminated": True, "timed_out": True}) == "cutoff"
    assert te.classify(**{**base, "collided_wall": True, "terminated": True}) == "collided_wall"
    assert te.classify(**{**base, "collided_obstacle": True,
                          "collided_wall": True}) == "collided_obstacle"
    assert te.classify(**{**base, "fell": True, "collided_obstacle": True}) == "fell"
    assert te.classify(**{**base, "reached_goal": False}) == "stalled"


# ---------------------------------------------------------------- consistency with EXP-028

def test_fall_thresholds_match_the_preregistered_exp028_constants():
    from experiments import exp028_termination_free_rollouts as e28

    assert te.FALL_PELVIS_Z_M == e28.FELL_PELVIS_Z_M
    assert te.FALL_UP_Z == e28.FELL_UP_Z


def test_up_z_matches_exp028s_helper():
    from experiments import exp028_termination_free_rollouts as e28

    quats = np.array([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0],
                      [0.9, 0.1, 0.2, 0.3]])
    assert np.allclose(te.up_z(quats), e28.up_z(quats))
