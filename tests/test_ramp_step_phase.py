import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import scene2motion.ramp.step_phase as step_phase_module
from scene2motion.stepover_eval import StepOverThresholds
from scene2motion.ramp.step_phase import (
    APEX_PHASE,
    LANDING_PHASE,
    STEP_PHASE_METHOD,
    TAKEOFF_PHASE,
    StepPhaseCycle,
    align_step_phase_cycles,
    align_step_target_phase,
    enumerate_step_phase_cycles,
    enumerate_step_phase_cycles_from_qpos,
    step_phase_cycle_from_qpos,
    step_phase_common_physical_protocol_hash,
    step_phase_measurement_protocol_hash,
    validate_step_phase_cycle,
)


def swing_kinematics(
    *,
    length=25,
    side="left",
    takeoff=7,
    apex=10,
    landing=14,
    flight_frames=(),
    stance_gaps=(),
    peak_at_takeoff=False,
):
    result = {
        foot: {
            "bottom_clearance_m": np.zeros(length, dtype=float),
            "planar_speed_mps": np.zeros(length, dtype=float),
        }
        for foot in ("left", "right")
    }
    if landing is None:
        last_air = length - 1
    else:
        last_air = landing - 1
    if peak_at_takeoff:
        heights = np.linspace(0.16, 0.04, last_air - takeoff + 1)
    else:
        before = np.linspace(0.04, 0.16, apex - takeoff + 1)
        after = np.linspace(0.16, 0.04, last_air - apex + 1)[1:]
        heights = np.concatenate((before, after))
    result[side]["bottom_clearance_m"][takeoff : last_air + 1] = heights
    result[side]["planar_speed_mps"][takeoff : last_air + 1] = 1.0

    stance = "right" if side == "left" else "left"
    for frame in stance_gaps:
        result[stance]["planar_speed_mps"][frame] = 1.0
    for frame in flight_frames:
        result[stance]["bottom_clearance_m"][frame] = 0.08
        result[stance]["planar_speed_mps"][frame] = 1.0
    return result


def cycle(**kwargs):
    kin = swing_kinematics(**kwargs)
    return enumerate_step_phase_cycles(
        kin,
        fps=10.0,
        swing_side=kwargs.get("side", "left"),
        support_window_s=0.4,
    )[0]


def test_enumeration_derives_auditable_physical_landmarks_and_phase():
    receipt = cycle()

    assert isinstance(receipt, StepPhaseCycle)
    assert receipt.method == STEP_PHASE_METHOD
    assert receipt.landmarks.takeoff_frame == 7
    assert receipt.landmarks.apex_frame == 10
    assert receipt.landmarks.landing_frame == 14
    assert receipt.phase_trace[0] == pytest.approx(TAKEOFF_PHASE)
    assert receipt.phase_trace[receipt.local_apex_frame] == pytest.approx(APEX_PHASE)
    assert receipt.phase_trace[-1] == pytest.approx(LANDING_PHASE)
    assert np.all(np.diff(receipt.phase_trace) > 0.0)
    assert receipt.event.frame == receipt.apex_frame
    assert receipt.event.side == "left"
    assert receipt.stance_side == "right"
    assert receipt.stance_support_fraction == pytest.approx(1.0)
    assert receipt.event.support_window_start_frame == 8
    assert receipt.event.support_window_end_frame == 12

    evidence = receipt.stance_evidence()
    assert evidence.center_frame == 10
    assert evidence.side == "right"
    assert evidence.support_fraction == pytest.approx(1.0)
    assert "height_m<=0.02" in evidence.source
    assert "speed_mps<=0.2" in evidence.source
    assert "frames=8:12" in evidence.source
    archived = receipt.as_dict()
    assert archived["support_window_s"] == pytest.approx(0.4)
    assert archived["landing_dwell_s"] == pytest.approx(0.12)
    assert archived["takeoff_dwell_frames"] == 3
    assert archived["landing_dwell_frames"] == 3
    assert archived["min_relative_lift_m"] == pytest.approx(0.04)
    assert archived["measurement_protocol_hash"] == receipt.measurement_protocol_hash
    assert (
        archived["common_physical_protocol_hash"]
        == receipt.common_physical_protocol_hash
    )
    json.dumps(archived, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        receipt.fps = 25.0


def test_existing_event_is_revalidated_against_exact_kinematics_and_lock():
    kin = swing_kinematics()
    discovered = enumerate_step_phase_cycles(
        kin, fps=10.0, swing_side="left", support_window_s=0.4
    )[0]
    checked = validate_step_phase_cycle(
        kin, discovered.event, fps=10.0, support_window_s=0.4
    )
    assert checked == discovered

    stale = replace(discovered.event, relative_lift_m=discovered.event.relative_lift_m + 0.01)
    with pytest.raises(ValueError, match="does not match exact foot kinematics"):
        validate_step_phase_cycle(kin, stale, fps=10.0, support_window_s=0.4)
    with pytest.raises(ValueError, match="locked stance-support window"):
        validate_step_phase_cycle(kin, discovered.event, fps=10.0, support_window_s=0.8)


def test_source_alignment_uses_landmark_phase_not_equal_frames():
    adapted = cycle(takeoff=7, apex=10, landing=14)
    neutral = cycle(takeoff=5, apex=10, landing=16)

    match = align_step_phase_cycles(adapted, neutral, half_window_frames=2)

    assert match.adapted_query_offsets_frames != match.neutral_query_offsets_frames
    assert match.adapted_query_offsets_frames[2] == 0.0
    assert match.neutral_query_offsets_frames[2] == 0.0
    np.testing.assert_allclose(match.adapted_phase_knots, match.neutral_phase_knots)
    assert match.adapted_stance_support_fraction == pytest.approx(1.0)
    assert match.neutral_stance_support_fraction == pytest.approx(1.0)


def test_target_alignment_returns_absolute_event_receipt_for_renderer():
    adapted = cycle(takeoff=7, apex=10, landing=14)
    neutral = cycle(takeoff=5, apex=10, landing=16)
    source = align_step_phase_cycles(adapted, neutral, half_window_frames=2)
    target = cycle(length=31, takeoff=8, apex=15, landing=22)

    receipt = align_step_target_phase(
        source.adapted_phase_knots,
        target,
        expected_swing_side="left",
        source_common_physical_protocol_hash=adapted.common_physical_protocol_hash,
        search_half_window_frames=5,
    )

    assert receipt.target_center_frame == 15
    assert receipt.swing_side == "left"
    assert receipt.target_stance_side == "right"
    assert receipt.target_stance_support_fraction == pytest.approx(1.0)
    assert receipt.target_query_offsets_frames[len(receipt.target_query_offsets_frames) // 2] == 0
    assert "qpos-physical-foot-support-mask" in receipt.target_stance_source


def test_qpos_wrappers_always_call_exact_foot_kinematics(monkeypatch):
    kin = swing_kinematics()
    discovered = enumerate_step_phase_cycles(
        kin, fps=10.0, swing_side="left", support_window_s=0.4
    )[0]
    calls = []

    def fake_kinematics(body, qpos, fps):
        calls.append((body, np.asarray(qpos).shape, fps))
        return kin

    monkeypatch.setattr(step_phase_module, "foot_kinematics_series", fake_kinematics)
    body = object()
    qpos = np.zeros((25, 3))
    checked = step_phase_cycle_from_qpos(
        body, qpos, 10.0, discovered.event, support_window_s=0.4
    )
    enumerated = enumerate_step_phase_cycles_from_qpos(
        body, qpos, 10.0, swing_side="left", support_window_s=0.4
    )
    assert checked == discovered
    assert enumerated == (discovered,)
    assert calls == [(body, (25, 3), 10.0), (body, (25, 3), 10.0)]


@pytest.mark.parametrize(
    "kinematics,message",
    [
        (
            swing_kinematics(takeoff=0, apex=5, landing=10),
            "no observed takeoff",
        ),
        (
            swing_kinematics(takeoff=12, apex=17, landing=None),
            "no observed landing",
        ),
        (
            swing_kinematics(flight_frames=(9,)),
            "bilateral flight",
        ),
        (
            swing_kinematics(peak_at_takeoff=True),
            "nonmonotone",
        ),
    ],
)
def test_invalid_or_truncated_swing_cycles_fail_closed(kinematics, message):
    with pytest.raises(ValueError, match=message):
        enumerate_step_phase_cycles(
            kinematics,
            fps=10.0,
            swing_side="left",
            support_window_s=0.4,
        )


def test_insufficient_contralateral_support_fails_separately_from_flight():
    # The right-foot support gaps are outside the left-foot swing itself, so this is not
    # bilateral flight. They do fall inside the independently locked, longer evidence window.
    kin = swing_kinematics(stance_gaps=(4, 5, 15, 16))
    with pytest.raises(ValueError, match="stance support is below"):
        enumerate_step_phase_cycles(
            kin,
            fps=10.0,
            swing_side="left",
            support_window_s=1.2,
            min_stance_support_fraction=0.90,
        )


def test_transient_contact_is_not_accepted_as_a_stable_landing():
    kin = swing_kinematics()
    # Frame 14 is the first apparent landing, but support disappears again immediately.
    kin["left"]["planar_speed_mps"][15] = 1.0
    thresholds = StepOverThresholds(landing_dwell_s=0.3)
    with pytest.raises(ValueError, match="stable observed landing dwell"):
        enumerate_step_phase_cycles(
            kin,
            fps=10.0,
            swing_side="left",
            support_window_s=0.4,
            thresholds=thresholds,
        )


def test_elapsed_support_dwell_is_required_before_takeoff_and_after_landing():
    receipt = cycle()
    # ceil(0.12 s * 10 Hz) + 1 samples span 0.2 s, which is >= the lock.
    assert receipt.takeoff_dwell_frames == 3
    assert receipt.landing_dwell_frames == 3

    kin = swing_kinematics()
    kin["left"]["planar_speed_mps"][5] = 1.0
    with pytest.raises(ValueError, match="pre-takeoff support dwell"):
        enumerate_step_phase_cycles(
            kin, fps=10.0, swing_side="left", support_window_s=0.4
        )


@pytest.mark.parametrize(
    "change,message",
    [
        ({"phase_trace": (0.25, 0.34, 0.43, 0.50, 0.56, 0.65, 0.70, 0.75)}, "exact locked"),
        ({"takeoff_dwell_frames": 2}, "sample counts"),
        ({"support_window_s": 0.5}, "measurement_protocol_hash"),
    ],
)
def test_frozen_receipt_rejects_forged_derived_invariants(change, message):
    with pytest.raises(ValueError, match=message):
        replace(cycle(), **change)


def test_frozen_receipt_rejects_forged_landmarks_and_event_evidence():
    receipt = cycle()
    with pytest.raises(ValueError, match="locked 0.25/0.50/0.75"):
        replace(
            receipt,
            landmarks=replace(receipt.landmarks, takeoff_phase=0.20),
        )
    with pytest.raises(ValueError, match="relative-lift threshold"):
        replace(
            receipt,
            event=replace(
                receipt.event,
                relative_lift_m=0.0,
                swing_height_m=receipt.event.stance_height_m,
            ),
        )
    with pytest.raises(ValueError, match="swing-minus-stance"):
        replace(
            receipt,
            event=replace(
                receipt.event,
                relative_lift_m=receipt.event.relative_lift_m + 0.01,
            ),
        )
    with pytest.raises(ValueError, match="must be finite"):
        replace(receipt, event=replace(receipt.event, swing_height_m=float("nan")))


def test_threshold_fraction_is_derived_and_conflicting_override_is_rejected():
    kin = swing_kinematics()
    thresholds = StepOverThresholds(min_contralateral_support_fraction=0.85)
    receipt = enumerate_step_phase_cycles(
        kin,
        fps=10.0,
        swing_side="left",
        support_window_s=0.4,
        thresholds=thresholds,
    )[0]
    assert receipt.min_stance_support_fraction == pytest.approx(0.85)
    with pytest.raises(ValueError, match="conflicts"):
        enumerate_step_phase_cycles(
            kin,
            fps=10.0,
            swing_side="left",
            support_window_s=0.4,
            thresholds=thresholds,
            min_stance_support_fraction=0.90,
        )


def test_measurement_protocol_identity_is_rate_independent_but_dwell_samples_are_not():
    kin = swing_kinematics()
    at_10_hz = enumerate_step_phase_cycles(
        kin, fps=10.0, swing_side="left", support_window_s=0.4
    )[0]
    at_20_hz = enumerate_step_phase_cycles(
        kin, fps=20.0, swing_side="left", support_window_s=0.4
    )[0]
    assert at_10_hz.measurement_protocol_hash == at_20_hz.measurement_protocol_hash
    assert (
        at_10_hz.common_physical_protocol_hash
        == at_20_hz.common_physical_protocol_hash
    )
    assert at_10_hz.landing_dwell_frames == 3
    assert at_20_hz.landing_dwell_frames == 4


def test_common_physical_protocol_excludes_only_relative_lift_quality_gate():
    thresholds = StepOverThresholds()
    common = step_phase_common_physical_protocol_hash(
        thresholds,
        support_window_s=0.4,
        min_stance_support_fraction=0.8,
    )
    assert common == step_phase_common_physical_protocol_hash(
        thresholds,
        support_window_s=0.4,
        min_stance_support_fraction=0.8,
    )
    strict = step_phase_measurement_protocol_hash(
        thresholds,
        support_window_s=0.4,
        min_stance_support_fraction=0.8,
        min_relative_lift_m=0.04,
    )
    permissive = step_phase_measurement_protocol_hash(
        thresholds,
        support_window_s=0.4,
        min_stance_support_fraction=0.8,
        min_relative_lift_m=0.0,
    )
    assert strict != permissive
    assert common not in (strict, permissive)


def test_target_with_distinct_lift_gate_aligns_under_common_physics():
    adapted = cycle(takeoff=7, apex=10, landing=14)
    neutral = cycle(takeoff=5, apex=10, landing=16)
    source = align_step_phase_cycles(adapted, neutral, half_window_frames=2)

    target_kinematics = swing_kinematics(length=31, takeoff=8, apex=15, landing=22)
    rise = np.linspace(0.01, 0.03, 8)
    fall = np.linspace(0.03, 0.01, 7)[1:]
    target_kinematics["left"]["bottom_clearance_m"][8:22] = np.concatenate(
        (rise, fall)
    )
    target = enumerate_step_phase_cycles(
        target_kinematics,
        fps=10.0,
        swing_side="left",
        support_window_s=0.4,
        min_relative_lift_m=0.0,
    )[0]
    assert target.event.relative_lift_m < 0.04
    assert target.measurement_protocol_hash != adapted.measurement_protocol_hash
    assert target.common_physical_protocol_hash == adapted.common_physical_protocol_hash

    receipt = align_step_target_phase(
        source.adapted_phase_knots,
        target,
        expected_swing_side="left",
        source_common_physical_protocol_hash=source.measurement_protocol_hash,
        search_half_window_frames=5,
    )
    assert receipt.target_center_frame == 15
    assert receipt.measurement_protocol_hash == target.common_physical_protocol_hash


def test_source_pair_keeps_same_lift_quality_gate_despite_common_identity():
    strict = cycle()
    permissive = enumerate_step_phase_cycles(
        swing_kinematics(),
        fps=10.0,
        swing_side="left",
        support_window_s=0.4,
        min_relative_lift_m=0.0,
    )[0]
    assert strict.common_physical_protocol_hash == permissive.common_physical_protocol_hash
    assert strict.measurement_protocol_hash != permissive.measurement_protocol_hash
    with pytest.raises(ValueError, match="one locked phase protocol"):
        align_step_phase_cycles(strict, permissive, half_window_frames=2)


def test_target_alignment_rejects_actual_common_physical_gate_mismatch():
    source_cycle = cycle()
    source = align_step_phase_cycles(source_cycle, source_cycle, half_window_frames=2)
    target = enumerate_step_phase_cycles(
        swing_kinematics(),
        fps=10.0,
        swing_side="left",
        support_window_s=0.4,
        thresholds=StepOverThresholds(support_speed_mps=0.3),
        min_relative_lift_m=0.0,
    )[0]
    assert target.common_physical_protocol_hash != source_cycle.common_physical_protocol_hash
    with pytest.raises(ValueError, match="different common physical protocols"):
        align_step_target_phase(
            source.adapted_phase_knots,
            target,
            expected_swing_side="left",
            source_common_physical_protocol_hash=source.measurement_protocol_hash,
            search_half_window_frames=2,
        )


def test_alignment_never_extrapolates_outside_physical_landmarks():
    receipt = cycle()
    with pytest.raises(ValueError, match="beyond physical takeoff/landing"):
        align_step_phase_cycles(receipt, receipt, half_window_frames=4)

    source = align_step_phase_cycles(receipt, receipt, half_window_frames=2)
    with pytest.raises(ValueError, match="beyond physical takeoff/landing"):
        align_step_target_phase(
            source.adapted_phase_knots,
            receipt,
            expected_swing_side="left",
            source_common_physical_protocol_hash=receipt.common_physical_protocol_hash,
            search_half_window_frames=4,
        )


def test_alignment_rejects_side_or_protocol_mismatch():
    left = cycle()
    right = cycle(side="right")
    with pytest.raises(ValueError, match="different swing sides"):
        align_step_phase_cycles(left, right, half_window_frames=2)
    with pytest.raises(ValueError, match="does not match the packet"):
        align_step_target_phase(
            left.phase_trace,
            right,
            expected_swing_side="left",
            source_common_physical_protocol_hash=left.common_physical_protocol_hash,
            search_half_window_frames=2,
        )

    with pytest.raises(ValueError, match="measurement_protocol_hash"):
        replace(left, support_window_s=0.5)
    with pytest.raises(ValueError, match="different common physical protocols"):
        align_step_target_phase(
            left.phase_trace,
            left,
            expected_swing_side="left",
            source_common_physical_protocol_hash="2" * 64,
            search_half_window_frames=2,
        )


def test_input_shapes_and_frame_window_are_strict():
    kin = swing_kinematics()
    bad = {side: dict(values) for side, values in kin.items()}
    bad["right"] = dict(bad["right"])
    bad["right"]["planar_speed_mps"] = np.zeros(24)
    with pytest.raises(ValueError, match="share one frame count"):
        enumerate_step_phase_cycles(bad, fps=10.0)
    with pytest.raises(ValueError, match="half-open"):
        enumerate_step_phase_cycles(kin, fps=10.0, frame_window=(20, 10))
    with pytest.raises(ValueError, match="no valid complete"):
        enumerate_step_phase_cycles(kin, fps=10.0, frame_window=(0, 5))
