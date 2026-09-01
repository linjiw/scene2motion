import json
from pathlib import Path

import numpy as np
import pytest

from experiments import calibrate_ramp_route_phase as cal
from experiments import exp017_ramp_residual_stepover as e17
from experiments import exp019_ramp_gait_matched_stepover as pilot


REPO = Path(pilot.__file__).resolve().parents[1]


def _cycle(*, seed=3900, speed_label="reference", side="left", apex=100,
           prominence=0.06, takeoff=None, landing=None):
    return cal.ObservedCycle(
        split="pilot", seed=seed, speed_label=speed_label,
        requested_speed_mps=dict(cal.SPEEDS)[speed_label], swing_side=side,
        takeoff_frame=apex - 9 if takeoff is None else takeoff,
        apex_frame=apex,
        landing_frame=apex + 9 if landing is None else landing,
        prominence_m=prominence, background_contrasts_m=(0.005, 0.006),
        background_window_identities=(
            {"baseline_support_side": side, "window_start_frame": apex - 13,
             "window_end_frame": apex - 10, "method": cal.PHASE_BACKGROUND_METHOD},
            {"baseline_support_side": side, "window_start_frame": apex + 9,
             "window_end_frame": apex + 12, "method": cal.PHASE_BACKGROUND_METHOD},
        ),
        nominal_foot_forward_offset_m=0.15,
        evidence_digest=cal._json_hash({"seed": seed, "apex": apex, "side": side,
                                        "speed": speed_label}),
        phase_evidence={"receipt_digest": "x"},
    )


def _clip(cycles, *, seed=3900, speed_label="reference"):
    return cal.AnalyzedClip(
        split="pilot", seed=seed, speed_label=speed_label,
        requested_speed_mps=dict(cal.SPEEDS)[speed_label],
        sample_sha256="1" * 64, qpos_content_sha256="2" * 64,
        qpos_archive_key=f"pilot__{speed_label}__seed{seed}",
        cycles=tuple(cycles),
        measurement_rejection=None if cycles else "no cycles",
    )


class _Thresholds:
    support_height_m = 0.02
    support_speed_mps = 0.15


def _scene(speed_label="reference", *, foot_offset=0.15, support_frames=()):
    """Route, qpos and foot kinematics with a known root-tracking residual.

    ``support_frames`` marks frames where both feet are in physical support, which is
    what the footfall-clearance rule reads.
    """
    route = cal.route_xz_for_speed(dict(cal.SPEEDS)[speed_label])
    qpos = np.zeros((cal.N_FRAMES, 8), dtype=float)
    # Achieved root lags the prescribed route by 2 cm, as ARDY's tracking does.
    qpos[:, 0] = route[:, 1] - 0.02
    forward = qpos[:, 0] + foot_offset
    clearance = np.full(cal.N_FRAMES, 1.0)
    speed = np.full(cal.N_FRAMES, 1.0)
    for frame in support_frames:
        clearance[frame] = 0.0
        speed[frame] = 0.0
    feet = {
        side: {
            "forward_representative_m": forward.copy(),
            "bottom_clearance_m": clearance.copy(),
            "planar_speed_mps": speed.copy(),
        }
        for side in ("left", "right")
    }
    return route, qpos, feet


def test_pool_plan_and_seeds_are_fresh_and_budget_is_exact():
    batches = pilot.pool_batch_plan()
    assert len(batches) == 3 * len(pilot.POOL_SEEDS) // 8
    assert all(len(b["seeds"]) == 8 for b in batches)
    seeds = sorted(seed for batch in batches for seed in batch["seeds"])
    assert seeds == sorted(list(pilot.POOL_SEEDS) * 3)
    assert [b["speed_label"] for b in batches[:3]] == ["slow", "reference", "fast"]
    prior = set(range(3200, 3216)) | set(range(3300, 3308)) | \
        set(range(3400, 3416)) | set(range(3500, 3508)) | \
        set(range(3600, 3616)) | set(range(3700, 3708)) | \
        set(range(3800, 3816)) | set(range(3900, 3916)) | \
        set(range(2800, 2808)) | set(pilot.DONOR_SEEDS)
    assert set(pilot.POOL_SEEDS).isdisjoint(prior)
    # v2 pool sized to the measured 5/16 constructibility rate: K=32 -> ~10 expected
    # against the unchanged N=8 requirement.
    assert len(pilot.POOL_SEEDS) == 32 and pilot.N_SELECT == 8
    assert 2 * len(pilot.DONOR_SEEDS) + 3 * len(pilot.POOL_SEEDS) \
        + 2 * pilot.N_SELECT == 120


def test_placement_is_route_anchored_so_exp017_shift_is_exactly_zero():
    route, qpos, feet = _scene()
    clip = _clip([_cycle(apex=100)])
    candidates = pilot.placeable_candidates(
        clip, qpos, feet, route, target_min_prominence_m=0.042,
        thresholds=_Thresholds())
    assert len(candidates) == 1
    chosen = candidates[0]
    assert chosen["placeable"] is True
    # Route-anchored, not achieved-foot-anchored: the 2 cm tracking residual shows up.
    assert chosen["obstacle_x_m"] == pytest.approx(
        float(route[100, 1]) + 0.15)
    assert chosen["achieved_foot_x_m"] == pytest.approx(
        chosen["obstacle_x_m"] - 0.02)
    # exp017's own assignment arithmetic must land exactly on the apex frame.
    desired_root = chosen["obstacle_x_m"] - chosen[
        "nominal_foot_forward_offset_m"]
    desired_frame = int(np.argmin(np.abs(route[:, 1] - desired_root)))
    assert desired_frame == 100


def test_achieved_foot_anchoring_would_not_give_a_zero_shift():
    """Regression guard for the anchoring choice this design depends on."""
    route, qpos, feet = _scene()
    forward = feet["left"]["forward_representative_m"]
    naive_obstacle = float(forward[100])
    foot_offset = naive_obstacle - float(qpos[100, 0])
    desired_frame = int(np.argmin(
        np.abs(route[:, 1] - (naive_obstacle - foot_offset))))
    assert desired_frame != 100


def test_placement_rejects_prominence_window_side_and_route_margin():
    route, qpos, feet = _scene()
    clip = _clip([
        _cycle(apex=100, side="right"),
        _cycle(apex=100, prominence=0.01),
        _cycle(apex=100, takeoff=99, landing=101),
        # Late apex: the expanded obstacle would overrun the end of the route.
        _cycle(apex=196, takeoff=187, landing=cal.N_FRAMES - 1),
    ])
    rejections = {
        row["rejection"]
        for row in pilot.placeable_candidates(
            clip, qpos, feet, route, target_min_prominence_m=0.042,
        thresholds=_Thresholds())
    }
    assert "wrong_swing_side" in rejections
    assert "prominence_below_frozen_target_gate" in rejections
    assert "packet_half_window_two_not_supported" in rejections
    assert "expanded_obstacle_outside_route" in rejections


def test_placement_rejects_obstacles_containing_a_support_footfall():
    """A footfall inside the footprint makes the scene unwinnable for every arm.

    exp019's v3 run placed all eight obstacles at swing apexes whose footprints still
    contained a footfall (nearest 0.007-0.115 m against a 0.140 m half-extent), so the
    nominal reference arm collided exactly like both packet arms and the comparison
    could not speak to representation.
    """
    route, qpos, feet = _scene()
    forward = feet["left"]["forward_representative_m"]
    obstacle_x = float(route[100, 1]) + 0.15
    # Plant a foot ~5 cm from the obstacle centre, well inside the expanded footprint.
    contact = int(np.argmin(np.abs(forward - (obstacle_x - 0.05))))
    route_c, qpos_c, feet_c = _scene(support_frames=(contact,))
    rows = pilot.placeable_candidates(
        _clip([_cycle(apex=100)]), qpos_c, feet_c, route_c,
        target_min_prominence_m=0.042, thresholds=_Thresholds())
    assert rows[0]["placeable"] is False
    assert rows[0]["rejection"] == "support_footfall_inside_obstacle_footprint"
    assert rows[0]["nearest_support_footfall_m"] <= pilot.OBSTACLE_HALF_EXTENT_M

    # The same cycle with the footfall well clear of the footprint stays placeable.
    far = int(np.argmin(np.abs(forward - (obstacle_x - 1.0))))
    rows_clear = pilot.placeable_candidates(
        _clip([_cycle(apex=100)]), *_scene(support_frames=(far,))[1:],
        route_xz=route, target_min_prominence_m=0.042, thresholds=_Thresholds())
    assert rows_clear[0]["placeable"] is True
    assert rows_clear[0]["nearest_support_footfall_m"] > pilot.OBSTACLE_HALF_EXTENT_M


def test_support_footfall_positions_reads_both_feet():
    _, _, feet = _scene(support_frames=(10, 11))
    positions = pilot.support_footfall_positions(feet, _Thresholds())
    assert positions.size == 4  # two frames x two feet


def test_mean_of_defined_skips_missing_and_none_metrics():
    rows = [{"m": 1.0}, {"m": None}, {}, {"m": 3.0}]
    assert pilot._mean_of_defined(rows, "m") == pytest.approx(2.0)
    assert pilot._mean_of_defined(rows, "absent") is None


def _constructible(candidates):
    return [{**row, "constructible": True} if row.get("placeable") else row
            for row in candidates]


def test_selection_prefers_mid_route_then_prominence_then_stratum():
    route, qpos, feet = _scene()
    far = _clip([_cycle(apex=40)])
    near_low = _clip([_cycle(apex=100, prominence=0.05)])
    near_high = _clip([_cycle(apex=100, prominence=0.09)])
    candidates = []
    for clip in (far, near_low, near_high):
        candidates.extend(pilot.placeable_candidates(
            clip, qpos, feet, route, target_min_prominence_m=0.042,
        thresholds=_Thresholds()))
    chosen = pilot.select_placement(_constructible(candidates))
    assert chosen["apex_frame"] == 100
    assert chosen["prominence_m"] == pytest.approx(0.09)

    # Same cost, different stratum: reference wins by the frozen stratum order.
    fast_route, fast_qpos, fast_feet = _scene("fast")
    fast = pilot.placeable_candidates(
        _clip([_cycle(apex=100, speed_label="fast")], speed_label="fast"),
        fast_qpos, fast_feet, fast_route, target_min_prominence_m=0.042,
        thresholds=_Thresholds())
    assert pilot.STRATUM_ORDER["reference"] < pilot.STRATUM_ORDER["fast"]
    assert fast and fast[0]["placeable"]


def test_select_placement_returns_none_without_placeable_candidates():
    route, qpos, feet = _scene()
    clip = _clip([_cycle(apex=100, prominence=0.001)])
    candidates = pilot.placeable_candidates(
        clip, qpos, feet, route, target_min_prominence_m=0.042,
        thresholds=_Thresholds())
    assert pilot.select_placement(_constructible(candidates)) is None


def test_placeable_but_unconstructible_candidates_are_never_selected():
    """Placement eligibility and packet constructibility are different gates.

    exp019's first run selected a seed whose observability apex had no step_phase
    cycle, and only found out at render time.  Selection must require the probe.
    """
    route, qpos, feet = _scene()
    candidates = pilot.placeable_candidates(
        _clip([_cycle(apex=100)]), qpos, feet, route,
        target_min_prominence_m=0.042,
        thresholds=_Thresholds())
    assert candidates[0]["placeable"] is True
    # Placeable but not probed / probe failed: not eligible.
    assert pilot.select_placement(candidates) is None
    assert pilot.select_placement(
        [{**candidates[0], "constructible": False,
          "construct_rejection": "no step_phase cycle"}]) is None
    assert pilot.select_placement(_constructible(candidates)) is not None


def test_selection_is_deterministic_under_candidate_reordering():
    route, qpos, feet = _scene()
    candidates = []
    for apex in (60, 100, 140):
        candidates.extend(pilot.placeable_candidates(
            _clip([_cycle(apex=apex)]), qpos, feet, route,
            target_min_prominence_m=0.042,
        thresholds=_Thresholds()))
    candidates = _constructible(candidates)
    first = pilot.select_placement(candidates)
    second = pilot.select_placement(list(reversed(candidates)))
    assert first["selection_key"] == second["selection_key"]


def test_channel_guard_now_uses_the_live_model_namespace():
    assert e17.ALLOWED_NONZERO_CHANNELS == {"root_pos", "global_rot_data"}


def test_run_pilot_refuses_nonempty_output_and_dirty_worktree(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")
    with pytest.raises(pilot.PilotAbort, match="nonempty"):
        pilot.run_pilot(
            out=out,
            threshold_receipt=REPO / "outputs/exp016_threshold_calibration/receipt.json")
    assert (out / "stale").read_text() == "x"

    with pytest.raises(pilot.PilotAbort, match="clean git worktree"):
        pilot.run_pilot(
            out=tmp_path / "clean",
            threshold_receipt=REPO / "outputs/exp016_threshold_calibration/receipt.json",
            code_state_fn=lambda _r: {"commit": "c" * 40, "dirty": True,
                                      "status": ["?? x"],
                                      "tracked_diff_sha256": "d" * 64})
    receipt = json.loads((tmp_path / "clean" / "receipt.json").read_text())
    assert receipt["schema"] == pilot.FAILURE_SCHEMA_VERSION
    assert receipt["failed_stage"] == "preflight"
    assert receipt["conservative_charged_ardy_samples"] == 0
