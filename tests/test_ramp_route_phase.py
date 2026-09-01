import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import scene2motion.ramp.route_phase as route_phase
from scene2motion.ramp import (
    ROUTE_PROGRESS_METHOD,
    ROUTE_PROGRESS_SLOPE_POLICY,
    RouteProgressProgram,
    RouteTimingBounds,
    reparameterize_route_progress,
    route_progress_selection_key,
)


def bounds(**changes):
    values = {
        "fps": 10.0,
        "min_discrete_route_progress_speed_mps": 0.5,
        "max_discrete_route_progress_speed_mps": 1.5,
        "max_abs_route_progress_acceleration_mps2": 2.0,
        "reference_route_progress_speed_mps": 1.0,
    }
    values.update(changes)
    return RouteTimingBounds(**values)


def straight_program(**changes):
    values = {
        "route_xz": np.array([[0.0, 0.0], [0.0, 2.0]]),
        "n_frames": 21,
        "event_frame": 10,
        "event_root_progress_m": 1.0,
        "timing_bounds": bounds(),
    }
    values.update(changes)
    return reparameterize_route_progress(**values)


def test_straight_route_exact_anchors_and_zero_packet_shift_contract():
    program = straight_program()
    assert isinstance(program, RouteProgressProgram)
    assert program.progress_m[0] == pytest.approx(0.0)
    assert program.progress_m[program.event_frame] == pytest.approx(1.0)
    assert program.progress_m[-1] == pytest.approx(2.0)
    np.testing.assert_allclose(program.root_xz, np.stack([
        np.zeros(21), np.linspace(0.0, 2.0, 21)
    ], axis=-1))
    np.testing.assert_allclose(program.route_heading_rad, 0.0)
    assert program.anchor_route_progress_slopes_mps == pytest.approx((1.0, 1.0, 1.0))
    assert program.packet_center_shift_frames == 0
    assert program.method == ROUTE_PROGRESS_METHOD
    assert program.slope_policy == ROUTE_PROGRESS_SLOPE_POLICY
    receipt = json.loads(program.diagnostics_json())
    assert receipt["packet_center_shift_frames"] == 0
    assert receipt["forward_heading_convention"] == "atan2(delta_root_x,delta_root_z)"
    assert receipt["program_hash"] == program.digest()


def test_ninety_degree_route_is_rejected_by_v1_geometry_contract():
    with pytest.raises(ValueError, match="forward-collinear"):
        reparameterize_route_progress(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            n_frames=21,
            event_frame=10,
            event_root_progress_m=1.0,
            timing_bounds=bounds(),
        )


def test_arbitrary_heading_forward_collinear_vertices_are_accepted():
    segment_length = np.sqrt(5.0)
    program = reparameterize_route_progress(
        np.array([[1.0, -2.0], [2.0, 0.0], [3.0, 2.0]]),
        n_frames=21,
        event_frame=10,
        event_root_progress_m=segment_length,
        timing_bounds=bounds(
            min_discrete_route_progress_speed_mps=1.5,
            max_discrete_route_progress_speed_mps=3.0,
            reference_route_progress_speed_mps=segment_length,
        ),
    )
    points = np.asarray(program.root_xz)
    np.testing.assert_allclose(points[0], [1.0, -2.0])
    np.testing.assert_allclose(points[10], [2.0, 0.0])
    np.testing.assert_allclose(points[-1], [3.0, 2.0])
    np.testing.assert_allclose(program.route_heading_rad, np.arctan2(1.0, 2.0))
    assert program.max_path_projection_error_m < 1e-12


def test_hash_is_deterministic_json_safe_and_tracks_identity():
    first = straight_program()
    second = straight_program()
    assert first == second
    assert hash(first) == hash(second)
    assert first.digest() == second.digest()
    assert first.diagnostics_json() == second.diagnostics_json()
    json.dumps(first.diagnostics(), allow_nan=False)

    changed_route = straight_program(route_xz=[[0.1, 0.0], [0.1, 2.0]])
    changed_bound = straight_program(
        timing_bounds=bounds(max_abs_route_progress_acceleration_mps2=2.1)
    )
    assert changed_route.digest() != first.digest()
    assert changed_bound.digest() != first.digest()


def test_receipt_states_exact_anchor_geometry_and_derivative_semantics():
    receipt = straight_program().diagnostics()
    assert receipt["route_geometry_convention"] == (
        "forward-collinear-root-xz-polyline-v1"
    )
    assert receipt["route_derivative_convention"] == (
        "scalar-arc-length-route-progress-v1"
    )
    assert receipt["caller_foot_centered_anchor_equation"] == (
        "event_root_progress_m = obstacle_progress_m - "
        "nominal_foot_forward_offset_m"
    )
    assert receipt["progress_continuity"] == "C1 scalar route progress at the event"
    assert receipt["acceleration_continuity"] == (
        "piecewise-continuous scalar route-progress acceleration; may jump at the event"
    )
    assert receipt["discrete_route_progress_jerk_definition"] == (
        "first difference of discrete scalar route-progress acceleration scaled by fps; "
        "equivalently second difference of speed scaled by fps^2"
    )
    assert receipt["forward_heading_convention"] == (
        "atan2(delta_root_x,delta_root_z)"
    )


def test_every_stored_diagnostic_and_selection_component_changes_digest():
    program = straight_program()
    replacements = {
        "continuous_interval_route_progress_speed_ranges_mps": (
            (0.99, 1.0),
            (1.0, 1.0),
        ),
        "continuous_interval_route_progress_acceleration_endpoints_mps2": (
            (0.01, 0.0),
            (0.0, 0.0),
        ),
        "continuous_route_progress_speed_range_mps": (0.99, 1.0),
        "max_abs_continuous_route_progress_acceleration_mps2": 0.01,
        "discrete_route_progress_speed_range_mps": (0.99, 1.0),
        "max_abs_discrete_route_progress_acceleration_mps2": 0.01,
        "max_abs_discrete_route_progress_jerk_mps3": 0.01,
        "endpoint_route_progress_speed_deviation_mps": 0.01,
        "mean_abs_progress_deformation_m": 0.01,
        "integrated_abs_progress_deformation_m_s": 0.01,
        "normalized_progress_deformation": 0.01,
        "rms_route_progress_speed_deviation_mps": 0.01,
        "max_path_projection_error_m": 1e-12,
        "path_projection_tolerance_m": 2e-9,
        "route_geometry_convention": "changed-route-geometry-v1",
        "route_derivative_convention": "changed-route-derivative-v1",
        "caller_foot_centered_anchor_equation": "changed-anchor-equation",
        "progress_continuity": "changed-progress-continuity",
        "acceleration_continuity": "changed-acceleration-continuity",
        "discrete_route_progress_jerk_definition": "changed-jerk-definition",
        "forward_heading_convention": "changed-heading-convention",
        "dense_validation_substeps_per_frame": 17,
    }
    for field, value in replacements.items():
        assert replace(program, **{field: value}).digest() != program.digest(), field

    # These three fields are the stable selection-cost prefix and are also covered above.
    for field in (
        "normalized_progress_deformation",
        "rms_route_progress_speed_deviation_mps",
        "max_abs_discrete_route_progress_acceleration_mps2",
    ):
        changed = replace(program, **{field: getattr(program, field) + 0.01})
        assert changed.selection_cost != program.selection_cost
        assert changed.digest() != program.digest()


def test_input_mutation_cannot_change_program_or_hash():
    route = np.array([[0.0, 0.0], [0.0, 2.0]])
    program = straight_program(route_xz=route)
    digest = program.digest()
    route[:] = 19.0
    assert program.route_xz == ((0.0, 0.0), (0.0, 2.0))
    assert program.digest() == digest
    with pytest.raises(FrozenInstanceError):
        program.event_frame = 11
    with pytest.raises(TypeError):
        program.root_xz[0][0] = 2.0


@pytest.mark.parametrize(
    ("event_frame", "event_progress", "pattern"),
    [
        (1, 1.0, "segment-0 mean speed"),
        (19, 1.0, "segment-1 mean speed"),
    ],
)
def test_infeasible_early_or_late_event_fails_necessary_average_gate(
    event_frame, event_progress, pattern
):
    with pytest.raises(ValueError, match=pattern):
        straight_program(event_frame=event_frame, event_root_progress_m=event_progress)


def test_speed_bounds_are_closed_at_boundary_and_fail_outside_epsilon():
    exact_min = straight_program(
        timing_bounds=bounds(
            min_discrete_route_progress_speed_mps=1.0,
            max_discrete_route_progress_speed_mps=1.5,
        )
    )
    assert exact_min.discrete_route_progress_speed_range_mps[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="speed bounds"):
        straight_program(
            timing_bounds=bounds(
                min_discrete_route_progress_speed_mps=1.0 + 1e-7,
                max_discrete_route_progress_speed_mps=1.5,
            )
        )

    exact_max = straight_program(
        timing_bounds=bounds(
            min_discrete_route_progress_speed_mps=0.5,
            max_discrete_route_progress_speed_mps=1.0,
        )
    )
    assert exact_max.discrete_route_progress_speed_range_mps[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="speed bounds"):
        straight_program(
            timing_bounds=bounds(
                min_discrete_route_progress_speed_mps=0.5,
                max_discrete_route_progress_speed_mps=1.0 - 1e-7,
            )
        )


def test_acceleration_and_jerk_bounds_are_closed_then_fail_below_epsilon():
    loose = straight_program(event_frame=8, event_root_progress_m=0.7)
    exact_acceleration = loose.max_abs_continuous_route_progress_acceleration_mps2
    exact = straight_program(
        event_frame=8,
        event_root_progress_m=0.7,
        timing_bounds=bounds(
            max_abs_route_progress_acceleration_mps2=exact_acceleration
        ),
    )
    assert exact.max_abs_continuous_route_progress_acceleration_mps2 == pytest.approx(
        exact_acceleration
    )
    with pytest.raises(ValueError, match="continuous acceleration"):
        straight_program(
            event_frame=8,
            event_root_progress_m=0.7,
            timing_bounds=bounds(
                max_abs_route_progress_acceleration_mps2=exact_acceleration - 1e-7
            ),
        )

    assert loose.max_abs_discrete_route_progress_jerk_mps3 is not None
    straight_program(
        event_frame=8,
        event_root_progress_m=0.7,
        timing_bounds=bounds(
            max_abs_discrete_route_progress_jerk_mps3=(
                loose.max_abs_discrete_route_progress_jerk_mps3
            )
        ),
    )
    with pytest.raises(ValueError, match="discrete jerk"):
        straight_program(
            event_frame=8,
            event_root_progress_m=0.7,
            timing_bounds=bounds(
                max_abs_discrete_route_progress_jerk_mps3=(
                    loose.max_abs_discrete_route_progress_jerk_mps3 - 1e-7
                )
            ),
        )


def test_hidden_between_frame_reversal_and_overshoot_fail_closed():
    # Both segment means are positive and inside these deliberately broad bounds, but
    # the fixed event slope makes the first cubic reverse between integer anchors.
    with pytest.raises(ValueError, match="continuous minimum speed"):
        reparameterize_route_progress(
            [[0.0, 0.0], [0.0, 505.0]],
            n_frames=101,
            event_frame=50,
            event_root_progress_m=5.0,
            timing_bounds=RouteTimingBounds(
                fps=1.0,
                min_discrete_route_progress_speed_mps=0.0,
                max_discrete_route_progress_speed_mps=20.0,
                max_abs_route_progress_acceleration_mps2=100.0,
            ),
        )


@pytest.mark.parametrize("event_frame", [0, 20, -1, 21, 1.5, True])
def test_event_frame_boundaries_and_type_are_rejected(event_frame):
    with pytest.raises(ValueError, match="event_frame"):
        straight_program(event_frame=event_frame)


@pytest.mark.parametrize("event_progress", [0.0, 2.0, -0.1, 2.1, np.nan])
def test_event_progress_boundaries_and_nonfinite_are_rejected(event_progress):
    with pytest.raises(ValueError, match="event_root_progress_m"):
        straight_program(event_root_progress_m=event_progress)


@pytest.mark.parametrize(
    ("route", "pattern"),
    [
        ([[0.0, 0.0]], "at least two"),
        ([[0.0, 0.0], [0.0, 0.0]], "duplicate"),
        ([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], "duplicate"),
        ([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]], "reverse direction"),
        (
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            "forward-collinear",
        ),
        ([[0.0, 0.0], [np.inf, 1.0]], "finite"),
    ],
)
def test_bad_route_geometry_fails_closed(route, pattern):
    with pytest.raises(ValueError, match=pattern):
        straight_program(route_xz=route)


def test_path_projection_gate_is_independent(monkeypatch):
    original = route_phase._route_points

    def displaced(*args, **kwargs):
        points, heading = original(*args, **kwargs)
        points[:, 0] += 0.01
        return points, heading

    monkeypatch.setattr(route_phase, "_route_points", displaced)
    with pytest.raises(ValueError, match="project onto"):
        straight_program(path_projection_tolerance_m=1e-4)


def test_endpoint_speed_deviation_is_reported_and_optionally_gated():
    program = straight_program(
        event_frame=8,
        event_root_progress_m=0.7,
        timing_bounds=bounds(reference_route_progress_speed_mps=1.0),
    )
    assert program.endpoint_route_progress_speed_deviation_mps is not None
    limit = program.endpoint_route_progress_speed_deviation_mps
    straight_program(
        event_frame=8,
        event_root_progress_m=0.7,
        timing_bounds=bounds(
            reference_route_progress_speed_mps=1.0,
            max_endpoint_route_progress_speed_deviation_mps=limit,
        ),
    )
    with pytest.raises(ValueError, match="endpoint speed deviation"):
        straight_program(
            event_frame=8,
            event_root_progress_m=0.7,
            timing_bounds=bounds(
                reference_route_progress_speed_mps=1.0,
                max_endpoint_route_progress_speed_deviation_mps=limit - 1e-7,
            ),
        )


def test_asymmetric_regression_passes_when_three_knot_pchip_overshoots_speed():
    # A simple asymmetric anchor demonstrates why the old three-knot PCHIP endpoint
    # heuristic is not the method contract.
    frames, fps, event = 200, 25.0, 119
    route_length = 7.2
    event_progress = 3.6
    locked = RouteTimingBounds(
        fps=fps,
        min_discrete_route_progress_speed_mps=0.6,
        max_discrete_route_progress_speed_mps=1.2,
        max_abs_route_progress_acceleration_mps2=0.3,
        reference_route_progress_speed_mps=0.9,
    )
    program = reparameterize_route_progress(
        [[0.0, 0.0], [0.0, route_length]],
        n_frames=frames,
        event_frame=event,
        event_root_progress_m=event_progress,
        timing_bounds=locked,
    )
    assert program.continuous_route_progress_speed_range_mps == pytest.approx(
        (0.6948529411764708, 1.1864495798319328)
    )

    anchors_t = np.array([0.0, event / fps, (frames - 1) / fps])
    anchors_s = np.array([0.0, event_progress, route_length])
    pchip = PchipInterpolator(anchors_t, anchors_s)
    dense_t = np.linspace(anchors_t[0], anchors_t[-1], 20001)
    pchip_speed = pchip.derivative()(dense_t)
    assert (
        float(np.min(pchip_speed))
        < locked.min_discrete_route_progress_speed_mps
    )
    assert (
        float(np.max(pchip_speed))
        > locked.max_discrete_route_progress_speed_mps
    )


def test_archived_seed_2806_foot_centered_root_progress_regression():
    # exp017 desired_root = obstacle progress 3.6 - nominal foot-forward offset
    # -0.15709712249081598.  Experiment-layer provenance will bind both inputs; this
    # geometry-only program binds their resulting root-progress anchor.
    program = reparameterize_route_progress(
        [[0.0, 0.0], [0.0, 7.2]],
        n_frames=200,
        event_frame=119,
        event_root_progress_m=3.757097122490816,
        timing_bounds=RouteTimingBounds(
            fps=25.0,
            min_discrete_route_progress_speed_mps=0.6,
            max_discrete_route_progress_speed_mps=1.2,
            max_abs_route_progress_acceleration_mps2=0.3,
            reference_route_progress_speed_mps=0.9,
        ),
    )
    assert program.event_root_progress_m == pytest.approx(3.757097122490816)
    assert program.progress_m[119] == pytest.approx(program.event_root_progress_m)
    assert program.continuous_route_progress_speed_range_mps == pytest.approx(
        (0.7415392796617928, 1.1236739877301665)
    )
    assert json.loads(program.diagnostics_json())["event_root_progress_m"] == pytest.approx(
        3.757097122490816
    )


def test_analytic_route_progress_derivatives_match_dense_evaluation():
    program = straight_program(event_frame=8, event_root_progress_m=0.7)
    fps = program.timing_bounds.fps
    anchors = (
        (0, program.event_frame, 0.0, program.event_root_progress_m),
        (
            program.event_frame,
            program.n_frames - 1,
            program.event_root_progress_m,
            program.route_length_m,
        ),
    )
    dense_ranges = []
    dense_acceleration_endpoints = []
    for index, (first, last, start_progress, end_progress) in enumerate(anchors):
        duration = (last - first) / fps
        mean = (end_progress - start_progress) / duration
        first_slope = program.anchor_route_progress_slopes_mps[index]
        last_slope = program.anchor_route_progress_slopes_mps[index + 1]
        a = -6.0 * mean + 3.0 * first_slope + 3.0 * last_slope
        b = 6.0 * mean - 4.0 * first_slope - 2.0 * last_slope
        u = np.linspace(0.0, 1.0, 200001)
        velocity = a * u**2 + b * u + first_slope
        acceleration = (2.0 * a * u + b) / duration
        dense_ranges.append((float(np.min(velocity)), float(np.max(velocity))))
        dense_acceleration_endpoints.append(
            (float(acceleration[0]), float(acceleration[-1]))
        )
    np.testing.assert_allclose(
        program.continuous_interval_route_progress_speed_ranges_mps,
        dense_ranges,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        program.continuous_interval_route_progress_acceleration_endpoints_mps2,
        dense_acceleration_endpoints,
        atol=1e-12,
    )


def test_event_acceleration_jump_and_fps_discrete_progress_jerk_are_explicit():
    program = straight_program(event_frame=8, event_root_progress_m=0.7)
    acceleration_endpoints = (
        program.continuous_interval_route_progress_acceleration_endpoints_mps2
    )
    left_event_acceleration = acceleration_endpoints[0][1]
    right_event_acceleration = acceleration_endpoints[1][0]
    assert left_event_acceleration != pytest.approx(right_event_acceleration)

    progress = np.asarray(program.progress_m)
    fps = program.timing_bounds.fps
    discrete_speed = np.diff(progress) * fps
    discrete_acceleration = np.diff(discrete_speed) * fps
    discrete_jerk = np.diff(discrete_acceleration) * fps
    assert program.max_abs_discrete_route_progress_jerk_mps3 == pytest.approx(
        np.max(np.abs(discrete_jerk))
    )
    assert program.max_abs_discrete_route_progress_jerk_mps3 > 0.0
    assert "may jump at the event" in program.diagnostics()["acceleration_continuity"]


def test_selection_key_has_stable_digest_tie_break_and_rejects_bad_digest():
    program = straight_program()
    low = "1" * 64
    high = "2" * 64
    assert route_progress_selection_key(program, low) < route_progress_selection_key(
        program, high
    )
    assert route_progress_selection_key(program, low) == route_progress_selection_key(
        program, low
    )
    assert route_progress_selection_key(program, low)[:3] == program.selection_cost
    assert route_progress_selection_key(program, low) == (*program.selection_cost, low)
    with pytest.raises(ValueError, match="SHA-256"):
        route_progress_selection_key(program, "bad")


@pytest.mark.parametrize(
    "bad",
    [
        {"fps": 0.0},
        {"min_discrete_route_progress_speed_mps": -0.1},
        {"max_discrete_route_progress_speed_mps": 0.5},
        {"max_abs_route_progress_acceleration_mps2": 0.0},
        {"max_abs_discrete_route_progress_jerk_mps3": 0.0},
        {"reference_route_progress_speed_mps": 0.0},
        {
            "max_endpoint_route_progress_speed_deviation_mps": 0.1,
            "reference_route_progress_speed_mps": None,
        },
    ],
)
def test_timing_bounds_validation(bad):
    with pytest.raises(ValueError):
        bounds(**bad)


def test_timing_bounds_are_immutable_hash_bound_and_json_safe():
    timing = bounds()
    with pytest.raises(FrozenInstanceError):
        timing.fps = 20.0
    json.dumps(timing.as_dict(), allow_nan=False)
    assert timing.digest() == replace(timing).digest()
    assert timing.digest() != replace(timing, fps=20.0).digest()
