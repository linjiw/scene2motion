import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from scene2motion.ramp.phase_observability import (
    PHASE_BACKGROUND_METHOD,
    PHASE_BASELINE_FRAMES,
    PHASE_PROMINENCE_METHOD,
    PhaseBackgroundContrast,
    PhaseObservabilityEvidence,
    PhaseProminenceReceipt,
    enumerate_background_phase_contrasts,
    enumerate_phase_prominences,
    measure_phase_observability,
    phase_observability_measurement_hash,
)
from scene2motion.ramp.step_phase import enumerate_step_phase_cycles
from scene2motion.stepover_eval import StepOverThresholds


def neutral_swing(
    *,
    length=80,
    side="left",
    takeoff=35,
    apex=39,
    landing=44,
    peak=0.035,
):
    kinematics = {
        foot: {
            "bottom_clearance_m": np.zeros(length, dtype=float),
            "planar_speed_mps": np.zeros(length, dtype=float),
        }
        for foot in ("left", "right")
    }
    last_air = landing - 1
    rising = np.linspace(0.021, peak, apex - takeoff + 1)
    falling = np.linspace(peak, 0.021, last_air - apex + 1)[1:]
    kinematics[side]["bottom_clearance_m"][takeoff:landing] = np.concatenate(
        (rising, falling)
    )
    kinematics[side]["planar_speed_mps"][takeoff:landing] = 1.0
    return kinematics


def set_swing_baselines(kinematics, side="left"):
    kinematics[side]["bottom_clearance_m"][31:35] = [0.000, 0.002, 0.004, 0.006]
    kinematics[side]["bottom_clearance_m"][44:48] = [0.002, 0.004, 0.006, 0.008]


def prominence(kinematics, side="left"):
    return enumerate_phase_prominences(
        kinematics,
        fps=10.0,
        swing_side=side,
        support_window_s=0.4,
    )[0]


def evidence(kinematics, side="left"):
    return measure_phase_observability(
        kinematics,
        fps=10.0,
        swing_side=side,
        support_window_s=0.4,
    )[0]


def test_signal_uses_world_floor_clearance_and_locked_own_support_baselines():
    kin = neutral_swing()
    set_swing_baselines(kin)
    receipt = prominence(kin)

    assert isinstance(receipt, PhaseProminenceReceipt)
    assert receipt.method == PHASE_PROMINENCE_METHOD
    assert receipt.cycle.min_relative_lift_m == 0.0
    assert receipt.cycle.landmarks.takeoff_frame == 35
    assert receipt.cycle.landmarks.apex_frame == 39
    assert receipt.cycle.landmarks.landing_frame == 44
    assert receipt.baseline_support_side == "left"
    assert "side=left" in receipt.baseline_support_source
    assert "height_m<=0.02" in receipt.baseline_support_source
    assert "speed_mps<=0.2" in receipt.baseline_support_source
    assert "pre_frames=31:34" in receipt.baseline_support_source
    assert "post_frames=44:47" in receipt.baseline_support_source
    assert receipt.baseline_frames == PHASE_BASELINE_FRAMES == 4
    assert receipt.pre_clearance_m == pytest.approx((0.000, 0.002, 0.004, 0.006))
    assert receipt.post_clearance_m == pytest.approx((0.002, 0.004, 0.006, 0.008))
    assert receipt.pre_median_clearance_m == pytest.approx(0.003)
    assert receipt.post_median_clearance_m == pytest.approx(0.005)
    assert receipt.background_reference_clearance_m == pytest.approx(0.005)
    assert receipt.apex_clearance_m == pytest.approx(0.035)
    assert receipt.prominence_m == pytest.approx(0.030)
    assert len(receipt.receipt_digest) == 64
    archived = receipt.as_dict()
    assert "pre_clearance_m" in archived
    assert "pre_relative_clearance_m" not in archived
    json.dumps(archived, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        receipt.prominence_m = 1.0


def test_left_and_right_swings_use_the_same_own_foot_measurement():
    left = neutral_swing(side="left")
    right = neutral_swing(side="right")
    set_swing_baselines(left, "left")
    set_swing_baselines(right, "right")
    left_evidence = evidence(left, "left")
    right_evidence = evidence(right, "right")

    assert left_evidence.prominence.baseline_support_side == "left"
    assert right_evidence.prominence.baseline_support_side == "right"
    assert left_evidence.prominence.apex_clearance_m == pytest.approx(
        right_evidence.prominence.apex_clearance_m
    )
    assert left_evidence.prominence.prominence_m == pytest.approx(
        right_evidence.prominence.prominence_m
    )
    assert [item.contrast_m for item in left_evidence.background_contrasts] == pytest.approx(
        [item.contrast_m for item in right_evidence.background_contrasts]
    )


def test_contralateral_height_and_motion_do_not_change_signal_or_nulls():
    base = neutral_swing()
    changed = neutral_swing()
    set_swing_baselines(base)
    set_swing_baselines(changed)
    baseline_frames = np.r_[31:35, 44:48]
    changed["right"]["bottom_clearance_m"][baseline_frames] = 0.019
    changed["right"]["planar_speed_mps"][baseline_frames] = 1.0
    base_evidence = evidence(base)
    changed_evidence = evidence(changed)

    assert changed_evidence.prominence.prominence_m == pytest.approx(
        base_evidence.prominence.prominence_m
    )
    assert changed_evidence.prominence.pre_clearance_m == pytest.approx(
        base_evidence.prominence.pre_clearance_m
    )
    assert changed_evidence.prominence.post_clearance_m == pytest.approx(
        base_evidence.prominence.post_clearance_m
    )
    assert [item.contrast_m for item in changed_evidence.background_contrasts] == pytest.approx(
        [item.contrast_m for item in base_evidence.background_contrasts]
    )
    assert changed_evidence.prominence.kinematics_digest != base_evidence.prominence.kinematics_digest


def test_own_foot_baseline_instability_fails_closed():
    kin = neutral_swing()
    kin["left"]["planar_speed_mps"][32] = 1.0
    with pytest.raises(ValueError, match="stable observed .*support dwell"):
        prominence(kin)


def test_observability_disables_donor_lift_gate_and_applies_no_target_gate():
    kin = neutral_swing(peak=0.015)
    kin["left"]["bottom_clearance_m"][31:35] = 0.020
    kin["left"]["bottom_clearance_m"][44:48] = 0.020
    kin["left"]["bottom_clearance_m"][35:44] = [
        0.005,
        0.008,
        0.011,
        0.013,
        0.015,
        0.013,
        0.010,
        0.008,
        0.005,
    ]
    with pytest.raises(ValueError, match="relative-lift threshold"):
        enumerate_step_phase_cycles(
            kin,
            fps=10.0,
            swing_side="left",
            support_window_s=0.4,
            min_relative_lift_m=0.04,
        )
    receipt = prominence(kin)
    assert receipt.cycle.min_relative_lift_m == 0.0
    assert receipt.prominence_m == pytest.approx(-0.005)


def test_protocol_identity_contains_measurement_gates_not_quality_thresholds():
    thresholds = StepOverThresholds()
    identity = phase_observability_measurement_hash(
        thresholds,
        fps=25.0,
        support_window_s=0.24,
    )
    assert len(identity) == 64
    assert identity == phase_observability_measurement_hash(
        thresholds,
        fps=25.0,
        support_window_s=0.24,
    )
    assert identity != phase_observability_measurement_hash(
        thresholds,
        fps=50.0,
        support_window_s=0.24,
    )
    assert identity != phase_observability_measurement_hash(
        replace(thresholds, support_height_m=0.03),
        fps=25.0,
        support_window_s=0.24,
    )
    with pytest.raises(TypeError):
        phase_observability_measurement_hash(
            thresholds,
            fps=25.0,
            support_window_s=0.24,
            donor_min_relative_lift_m=0.04,
        )
    with pytest.raises(TypeError):
        phase_observability_measurement_hash(
            thresholds,
            fps=25.0,
            support_window_s=0.24,
            target_min_phase_prominence_m=0.01,
        )


def test_signal_measurement_fails_closed_on_nonfinite_data():
    kin = neutral_swing()
    kin["left"]["bottom_clearance_m"][39] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prominence(kin)


def test_signal_measurement_fails_closed_on_truncated_four_frame_baseline():
    kin = neutral_swing(length=25, takeoff=3, apex=6, landing=10)
    with pytest.raises(ValueError, match="truncated"):
        prominence(kin)


def test_background_nulls_are_exactly_two_own_foot_window_max_minus_median():
    kin = neutral_swing()
    kin["left"]["bottom_clearance_m"][31:35] = [0.000, 0.006, 0.006, 0.002]
    kin["left"]["bottom_clearance_m"][44:48] = [0.002, 0.004, 0.006, 0.008]
    signal = prominence(kin)
    first = enumerate_background_phase_contrasts(kin, signal)
    second = enumerate_background_phase_contrasts(kin, signal)

    assert first == second
    assert len(first) == 2
    assert all(isinstance(item, PhaseBackgroundContrast) for item in first)
    assert all(item.method == PHASE_BACKGROUND_METHOD for item in first)
    assert [item.window_label for item in first] == ["pre_takeoff", "post_landing"]
    assert [(item.window_start_frame, item.window_end_frame) for item in first] == [
        (31, 34),
        (44, 47),
    ]
    assert [item.baseline_support_side for item in first] == ["left", "left"]
    assert [item.peak_frame for item in first] == [32, 47]
    assert [item.peak_clearance_m for item in first] == pytest.approx([0.006, 0.008])
    assert [item.median_clearance_m for item in first] == pytest.approx([0.004, 0.005])
    assert [item.contrast_m for item in first] == pytest.approx([0.002, 0.003])
    assert len({item.receipt_digest for item in first}) == 2
    archived = [item.as_dict() for item in first]
    assert all("clearance_m" in item for item in archived)
    assert all("signed_order" not in item and "relative_clearance_m" not in item for item in archived)
    json.dumps(archived, sort_keys=True)


def test_complete_evidence_is_deterministic_and_isolated_from_input_mutation():
    kin = neutral_swing()
    result = evidence(kin)
    assert isinstance(result, PhaseObservabilityEvidence)
    archived = result.as_dict()
    digest = result.receipt_digest
    kin["left"]["bottom_clearance_m"][:] = 99.0
    assert result.as_dict() == archived
    assert result.receipt_digest == digest
    assert len(result.background_contrasts) == 2


def test_background_requires_matching_input_and_protocol():
    kin = neutral_swing()
    signal = prominence(kin)
    altered = neutral_swing()
    altered["right"]["bottom_clearance_m"][0] = 0.001
    with pytest.raises(ValueError, match="do not match"):
        enumerate_background_phase_contrasts(altered, signal)
    with pytest.raises(ValueError, match="protocol"):
        enumerate_background_phase_contrasts(
            kin,
            signal,
            thresholds=StepOverThresholds(support_height_m=0.03),
        )


def test_short_realistic_support_runs_still_produce_two_window_nulls():
    kin = neutral_swing(length=25, takeoff=7, apex=10, landing=14)
    result = evidence(kin)
    assert len(result.background_contrasts) == 2
    assert [item.window_label for item in result.background_contrasts] == [
        "pre_takeoff",
        "post_landing",
    ]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"baseline_frames": 3}, "locked value"),
        ({"prominence_m": 1.0}, "apex-clearance-minus-background"),
        ({"pre_median_clearance_m": 1.0}, "four-frame median"),
        ({"measurement_protocol_hash": "bad"}, "SHA-256"),
        ({"baseline_support_side": "right"}, "cycle-certified swing foot"),
        ({"baseline_support_source": "bad"}, "support source"),
    ],
)
def test_prominence_receipt_rejects_forged_invariants(change, message):
    signal = prominence(neutral_swing())
    with pytest.raises(ValueError, match=message):
        replace(signal, **change)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"window_label": "other"}, "window_label"),
        ({"window_end_frame": 99}, "exactly four"),
        ({"peak_frame": 99}, "first deterministic"),
        ({"peak_clearance_m": 1.0}, "window maximum"),
        ({"median_clearance_m": 1.0}, "window median"),
        ({"contrast_m": 1.0}, "maximum-minus-median"),
        ({"baseline_support_side": "right"}, "support source"),
    ],
)
def test_background_receipt_rejects_forged_invariants(change, message):
    result = evidence(neutral_swing())
    with pytest.raises(ValueError, match=message):
        replace(result.background_contrasts[0], **change)


def test_evidence_rejects_missing_duplicate_or_unbound_window_nulls():
    result = evidence(neutral_swing())
    with pytest.raises(ValueError, match="exactly two"):
        replace(result, background_contrasts=result.background_contrasts[:1])
    with pytest.raises(ValueError, match="both locked baselines"):
        replace(
            result,
            background_contrasts=(
                result.background_contrasts[0],
                result.background_contrasts[0],
            ),
        )

    first, second = result.background_contrasts
    shifted_window = replace(
        first,
        window_start_frame=first.window_start_frame + 1,
        window_end_frame=first.window_end_frame + 1,
        peak_frame=first.peak_frame + 1,
    )
    with pytest.raises(ValueError, match="locked signal baseline"):
        replace(result, background_contrasts=(shifted_window, second))

    offset = 0.001
    shifted_values = tuple(value + offset for value in first.clearance_m)
    shifted_signal = replace(
        first,
        clearance_m=shifted_values,
        peak_clearance_m=first.peak_clearance_m + offset,
        median_clearance_m=first.median_clearance_m + offset,
    )
    with pytest.raises(ValueError, match="own-foot signal clearance"):
        replace(result, background_contrasts=(shifted_signal, second))
