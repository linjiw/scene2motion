import numpy as np
import pytest

from scene2motion.stepover_eval import (
    StepOverThresholds,
    evaluate_local_step,
    foot_clearance_series,
    foot_kinematics_series,
    motion_metrics,
    select_kinematic_step_event,
)


class FakeFootBody:
    """Two axis-aligned physical foot boxes encoded directly in synthetic qpos."""

    robot_geoms = [0, 1]
    geom_name = {0: "left_foot_pad", 1: "right_foot_pad"}
    _UP = np.array([0.0, 0.0, 1.0])
    body_margin = 0.0

    def __init__(self):
        self._q = None

    def fk(self, q):
        self._q = np.asarray(q)

    def geom_extent(self, geom, direction, extra_margin=0.0):
        assert extra_margin == 0.0
        if geom == 0:
            x, y, bottom = self._q[2:5]
        else:
            x, y, bottom = self._q[5:8]
        direction = np.asarray(direction)
        if np.allclose(direction, [1, 0, 0]):
            return float(x - 0.05), float(x + 0.05)
        if np.allclose(direction, [0, 1, 0]):
            return float(y - 0.04), float(y + 0.04)
        if np.allclose(direction, [0, 0, 1]):
            return float(bottom), float(bottom + 0.04)
        raise AssertionError(f"unexpected direction {direction}")

    def trajectory_report(self, _qpos):
        return {"max_foot_floor_penetration_m": 0.0}


class FakeObstacleBody:
    body_margin = 0.0

    def __init__(self, collision_free=True):
        self.collision_free = collision_free

    def trajectory_report(self, _qpos):
        return {
            "collision_free": self.collision_free,
            "min_clearance_m": 0.01 if self.collision_free else -0.01,
            "max_penetration_m": 0.0 if self.collision_free else 0.01,
        }


class FakeProbe:
    depth = 0.2

    def probe(self, _qpos):
        return 0.08


def _qpos(root_x, left_x, left_z, right_x, right_z, root_y=None,
          left_y=-0.10, right_y=0.10):
    root_x = np.asarray(root_x, dtype=float)
    T = len(root_x)
    root_y = np.zeros(T) if root_y is None else np.broadcast_to(root_y, (T,))
    return np.stack([
        root_x, root_y,
        np.broadcast_to(left_x, (T,)), np.broadcast_to(left_y, (T,)),
        np.broadcast_to(left_z, (T,)),
        np.broadcast_to(right_x, (T,)), np.broadcast_to(right_y, (T,)),
        np.broadcast_to(right_z, (T,)),
    ], axis=-1)


def valid_step_qpos():
    """Left foot crosses and lands, then right foot crosses and lands."""
    T = 25
    root = np.linspace(0.30, 1.70, T)
    left_x = np.full(T, 0.40)
    left_z = np.zeros(T)
    # Exact states for a [0.9, 1.1] slab and +/-5 cm foot envelope:
    # before through frame 5, overlap 6:9, after from frame 10.
    left_x[6:11] = [0.86, 0.94, 1.04, 1.10, 1.16]
    left_x[11:] = 1.16
    left_z[6:10] = [0.08, 0.15, 0.15, 0.08]

    right_x = np.full(T, 0.35)
    right_z = np.zeros(T)
    # The trailing overlap begins after the lead has completed a 3-frame support dwell.
    right_x[15:19] = [0.86, 0.96, 1.08, 1.16]
    right_x[19:] = 1.16
    right_z[15:18] = [0.08, 0.15, 0.08]
    return _qpos(root, left_x, left_z, right_x, right_z)


def relaxed_thresholds(**changes):
    values = dict(
        support_height_m=0.02,
        support_speed_mps=0.20,
        min_contralateral_support_fraction=0.90,
        max_unsupported_run_frames=2,
        landing_dwell_frames=3,
        landing_horizon_s=0.75,
        max_floor_penetration_m=0.02,
        lateral_corridor_half_width_m=0.30,
        corridor_longitudinal_pad_m=0.30,
    )
    values.update(changes)
    return StepOverThresholds(**values)


def test_foot_kinematics_are_surface_extents_and_finite_difference_velocity():
    body = FakeFootBody()
    q = _qpos([0, 0, 0], [0.0, 0.1, 0.1], [0.01, 0.02, 0.03],
              [0.5, 0.5, 0.5], 0.0)

    k = foot_kinematics_series(body, q, fps=10.0)

    np.testing.assert_allclose(k["left"]["forward_min_m"], [-0.05, 0.05, 0.05])
    np.testing.assert_allclose(k["left"]["forward_max_m"], [0.05, 0.15, 0.15])
    np.testing.assert_allclose(k["left"]["lateral_representative_m"], -0.10)
    np.testing.assert_allclose(k["left"]["bottom_clearance_m"], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(k["left"]["planar_velocity_mps"][:, 0], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(foot_clearance_series(body, q)["left"],
                               [0.01, 0.02, 0.03])


def test_exact_local_step_gate_accepts_ordered_supported_crossing():
    result = evaluate_local_step(
        FakeFootBody(), FakeObstacleBody(), valid_step_qpos(),
        obstacle_x=1.0, obstacle_depth=0.2, fps=10.0,
        expected_lead_side="left", obstacle_margin_m=0.0,
        thresholds=relaxed_thresholds(),
    )

    assert result["local_step_success"]
    assert all(result["gates"].values())
    assert result["crossing"]["lead_side"] == "left"
    assert result["crossing"]["trail_side"] == "right"
    assert result["crossing"]["lead_matches_expected"] is True
    assert result["feet"]["left"]["overlap_start_frame"] == 6
    assert result["feet"]["left"]["after_frame"] == 10
    assert result["feet"]["left"]["landing_start_frame"] == 11
    assert result["feet"]["right"]["overlap_start_frame"] == 15
    assert result["support"]["max_unsupported_run_frames"] == 0


def test_motion_metrics_phase_is_measured_from_foot_not_pelvis():
    q = valid_step_qpos()
    planned = np.stack([q[:, 1], q[:, 0]], axis=-1)
    result = motion_metrics(
        FakeFootBody(), FakeObstacleBody(), FakeProbe(), q, planned,
        obstacle_x=1.0, swing_side="left", fps=10.0,
        thresholds=relaxed_thresholds(),
    )

    # First equally high lead-foot peak is at foot centre x=.94, while the pelvis is much
    # farther back. This guards the exact spatial-registration defect in the legacy metric.
    assert result["phase_error_m"] == pytest.approx(-0.06)
    assert result["crossing_frame"] in range(6, 10)
    assert result["selected_lead_side"] == "left"
    assert result["local_step_success"]


def test_simultaneous_hop_fails_order_support_and_unsupported_gates():
    q = valid_step_qpos()
    # Make the right foot duplicate the left swing exactly: both feet are airborne and cross
    # at once, even though endpoint progress and whole-body collision are nominally valid.
    q[:, 5] = q[:, 2]
    q[:, 7] = q[:, 4]
    result = evaluate_local_step(
        FakeFootBody(), FakeObstacleBody(), q, 1.0, 0.2, 10.0,
        obstacle_margin_m=0.0, thresholds=relaxed_thresholds(),
    )

    assert not result["local_step_success"]
    assert not result["gates"]["lead_trail_order"]
    assert not result["gates"]["lead_overlap_has_trailing_support"]
    assert not result["gates"]["bounded_unsupported_run"]


def test_peak_without_true_before_overlap_after_does_not_count_as_crossing():
    q = valid_step_qpos()
    # The trailing foot lifts high over the obstacle but returns to the approach side.
    q[15:19, 5] = [0.86, 0.96, 1.08, 0.35]
    q[19:, 5] = 0.35
    result = evaluate_local_step(
        FakeFootBody(), FakeObstacleBody(), q, 1.0, 0.2, 10.0,
        obstacle_margin_m=0.0, thresholds=relaxed_thresholds(),
    )

    assert not result["feet"]["right"]["crossed_before_over_after"]
    assert not result["gates"]["both_feet_cross_before_over_after"]
    assert not result["gates"]["both_feet_finish_beyond"]
    assert not result["local_step_success"]


def test_lateral_detour_and_floor_penetration_are_independent_failures():
    q = valid_step_qpos()
    local = (q[:, 0] > 0.6) & (q[:, 0] < 1.4)
    q[local, 1] = 0.45
    q[12, 4] = -0.04
    result = evaluate_local_step(
        FakeFootBody(), FakeObstacleBody(), q, 1.0, 0.2, 10.0,
        obstacle_margin_m=0.0, thresholds=relaxed_thresholds(),
    )

    assert not result["gates"]["lateral_corridor"]
    assert not result["gates"]["bounded_floor_penetration"]
    assert result["floor"]["max_foot_floor_penetration_m"] == pytest.approx(0.04)
    assert not result["local_step_success"]


def test_whole_body_collision_is_required_even_with_valid_foot_topology():
    result = evaluate_local_step(
        FakeFootBody(), FakeObstacleBody(collision_free=False), valid_step_qpos(),
        1.0, 0.2, 10.0, obstacle_margin_m=0.0,
        thresholds=relaxed_thresholds(),
    )
    assert not result["gates"]["whole_body_collision_free"]
    assert not result["local_step_success"]


def test_qpos_donor_selector_uses_height_speed_and_stable_stance():
    q = valid_step_qpos()
    event = select_kinematic_step_event(
        FakeFootBody(), q, fps=10.0, frame_window=(4, 13),
        thresholds=relaxed_thresholds(), support_window_s=0.2,
    )

    assert event.side == "left"
    assert event.frame in (7, 8)
    assert event.relative_lift_m == pytest.approx(0.15)
    assert event.stance_support_fraction == 1.0
    assert event.stance_planar_speed_mps == 0.0


def test_qpos_donor_selector_rejects_bilateral_flight():
    q = valid_step_qpos()
    q[:, 5] = q[:, 2]
    q[:, 7] = q[:, 4]
    with pytest.raises(ValueError, match="no stable kinematic"):
        select_kinematic_step_event(
            FakeFootBody(), q, fps=10.0, frame_window=(5, 10),
            thresholds=relaxed_thresholds(), support_window_s=0.2,
        )


@pytest.mark.parametrize("bad_depth,bad_fps", [(0.0, 10.0), (0.2, 0.0)])
def test_local_step_input_validation(bad_depth, bad_fps):
    with pytest.raises(ValueError):
        evaluate_local_step(
            FakeFootBody(), FakeObstacleBody(), valid_step_qpos(),
            1.0, bad_depth, bad_fps, obstacle_margin_m=0.0,
        )
