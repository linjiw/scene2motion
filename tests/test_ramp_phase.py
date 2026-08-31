import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from scene2motion.ramp.phase import (
    PHASE_ALIGNMENT_METHOD,
    TARGET_PHASE_ALIGNMENT_METHOD,
    StanceEvidence,
    TargetPhaseMatch,
    align_cyclic_phase_windows,
    align_target_phase_window,
)


PROTOCOL_HASH = "1" * 64


def phase_trace(length, center, center_phase, cadence):
    frames = np.arange(length, dtype=float)
    return np.mod(center_phase + cadence * (frames - center), 1.0)


def stance(side, center, *, support=1.0, source="synthetic-contact-sensor-v1"):
    return StanceEvidence(
        side=side,
        center_frame=center,
        support_fraction=support,
        source=source,
    )


def align(adapted, neutral, *, adapted_center=7, neutral_center=7, half=3, **kwargs):
    return align_cyclic_phase_windows(
        adapted,
        neutral,
        adapted_center_frame=adapted_center,
        neutral_center_frame=neutral_center,
        half_window_frames=half,
        swing_side="left",
        adapted_stance=stance("right", adapted_center),
        neutral_stance=stance("right", neutral_center),
        measurement_protocol_hash=PROTOCOL_HASH,
        **kwargs,
    )


def test_different_cadences_invert_to_fractional_offsets_on_common_knots():
    adapted = phase_trace(15, 7, 0.40, 0.08)
    neutral = phase_trace(15, 7, 0.40, 0.04)

    match = align(adapted, neutral)

    assert match.method == PHASE_ALIGNMENT_METHOD
    np.testing.assert_allclose(
        match.adapted_query_offsets_frames,
        (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        match.neutral_query_offsets_frames,
        (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(match.adapted_phase_knots, match.neutral_phase_knots)
    match.validate(np.arange(-3, 4), "left")


def test_wrap_around_is_unwrapped_locally_without_extrapolation():
    adapted = phase_trace(15, 7, 0.98, 0.06)
    neutral = phase_trace(17, 9, 0.01, 0.04)

    match = align(
        adapted,
        neutral,
        adapted_center=7,
        neutral_center=9,
        max_phase_error=0.04,
    )

    assert max(match.phase_errors) == pytest.approx(0.03)
    assert match.adapted_query_offsets_frames[3] == 0.0
    assert match.neutral_query_offsets_frames[3] == 0.0
    assert min(match.adapted_phase_knots) >= 0.0
    assert max(match.adapted_phase_knots) < 1.0
    assert min(match.neutral_phase_knots) >= 0.0
    assert max(match.neutral_phase_knots) < 1.0
    assert match.adapted_query_offsets_frames[0] >= -3.0
    assert match.adapted_query_offsets_frames[-1] <= 3.0
    assert match.neutral_query_offsets_frames[0] >= -3.0
    assert match.neutral_query_offsets_frames[-1] <= 3.0


def test_event_center_phase_mismatch_fails_instead_of_relabelling_phases():
    adapted = phase_trace(15, 7, 0.20, 0.05)
    neutral = phase_trace(15, 7, 0.31, 0.05)

    with pytest.raises(ValueError, match="event-center phase error"):
        align(adapted, neutral, max_phase_error=0.05)


@pytest.mark.parametrize(
    "bad_local",
    [
        # A small backwards step looks like a 0.96-cycle forward jump.
        (0.20, 0.25, 0.30, 0.26, 0.36, 0.41, 0.46),
        # Repeated samples provide no invertible phase coordinate.
        (0.20, 0.25, 0.30, 0.30, 0.40, 0.45, 0.50),
    ],
)
def test_nonmonotone_or_stalled_local_phase_fails_closed(bad_local):
    neutral = phase_trace(15, 7, 0.26, 0.05)
    adapted = neutral.copy()
    adapted[4:11] = bad_local

    with pytest.raises(ValueError, match="non-monotonic|phase-stalled|phase-ambiguous"):
        align(adapted, neutral)


def test_large_transition_is_phase_ambiguous_even_if_modulo_direction_is_positive():
    adapted = phase_trace(15, 7, 0.40, 0.05)
    neutral = adapted.copy()
    adapted[8] = 0.78

    with pytest.raises(ValueError, match="phase-ambiguous"):
        align(adapted, neutral)


def test_insufficient_common_phase_overlap_fails_closed():
    adapted = phase_trace(15, 7, 0.40, 0.05)
    neutral = phase_trace(15, 7, 0.40, 0.004)

    with pytest.raises(ValueError, match="insufficient common phase overlap"):
        align(adapted, neutral, min_common_phase_span_per_side=0.02)


@pytest.mark.parametrize("adapted_center,neutral_center", [(2, 7), (7, 12)])
def test_source_bounds_require_a_full_window(adapted_center, neutral_center):
    adapted = phase_trace(15, 7, 0.40, 0.05)
    neutral = phase_trace(15, 7, 0.40, 0.05)

    with pytest.raises(ValueError, match="full phase window"):
        align(
            adapted,
            neutral,
            adapted_center=adapted_center,
            neutral_center=neutral_center,
        )


@pytest.mark.parametrize(
    "which,evidence,message",
    [
        ("adapted", stance("left", 7), "not contralateral"),
        ("neutral", stance("right", 7, support=0.6), "below the locked threshold"),
        ("adapted", stance("right", 6), "different event frame"),
    ],
)
def test_stance_must_be_explicit_event_matched_and_contralateral(which, evidence, message):
    adapted = phase_trace(15, 7, 0.40, 0.05)
    neutral = phase_trace(15, 7, 0.40, 0.05)
    args = {
        "adapted_stance": stance("right", 7),
        "neutral_stance": stance("right", 7),
    }
    args[f"{which}_stance"] = evidence

    with pytest.raises(ValueError, match=message):
        align_cyclic_phase_windows(
            adapted,
            neutral,
            adapted_center_frame=7,
            neutral_center_frame=7,
            half_window_frames=3,
            swing_side="left",
            measurement_protocol_hash=PROTOCOL_HASH,
            **args,
        )


@pytest.mark.parametrize(
    "bad_phase,message",
    [
        (np.array([0.0, 0.1, np.nan]), "finite"),
        (np.array([0.0, 0.1, 1.0]), r"\[0, 1\)"),
        (np.zeros((3, 1)), "one-dimensional"),
    ],
)
def test_phase_domain_is_explicit_and_validated(bad_phase, message):
    valid = phase_trace(15, 7, 0.40, 0.05)
    with pytest.raises(ValueError, match=message):
        align(bad_phase, valid)


def test_stance_evidence_rejects_implicit_or_untraceable_claims():
    with pytest.raises(ValueError, match="support_fraction"):
        StanceEvidence("right", 7, 1.1, "sensor")
    with pytest.raises(ValueError, match="source"):
        StanceEvidence("right", 7, 1.0, "")
    with pytest.raises(ValueError, match="integer"):
        StanceEvidence("right", 7.0, 1.0, "sensor")


def test_support_receipt_records_measured_fractions_and_locked_window():
    adapted = phase_trace(15, 7, 0.40, 0.05)
    neutral = phase_trace(15, 7, 0.40, 0.05)
    match = align_cyclic_phase_windows(
        adapted,
        neutral,
        adapted_center_frame=7,
        neutral_center_frame=7,
        half_window_frames=3,
        swing_side="left",
        adapted_stance=stance("right", 7, support=0.92),
        neutral_stance=stance("right", 7, support=0.87),
        min_stance_support_fraction=0.85,
        support_window_s=0.16,
        measurement_protocol_hash=PROTOCOL_HASH,
    )

    assert match.adapted_stance_support_fraction == pytest.approx(0.92)
    assert match.neutral_stance_support_fraction == pytest.approx(0.87)
    assert match.min_stance_support_fraction == pytest.approx(0.85)
    assert match.support_window_s == pytest.approx(0.16)


def target_align(target, packet_knots, *, center=10, half=5, **kwargs):
    return align_target_phase_window(
        target,
        packet_knots,
        target_center_frame=center,
        search_half_window_frames=half,
        swing_side="left",
        target_stance=stance("right", center, support=0.93),
        measurement_protocol_hash=PROTOCOL_HASH,
        **kwargs,
    )


def test_target_trace_is_inverted_onto_packet_grid_with_fractional_offsets():
    packet_knots = phase_trace(7, 3, 0.40, 0.04)
    target = phase_trace(21, 10, 0.40, 0.03)

    receipt = target_align(target, packet_knots)

    assert isinstance(receipt, TargetPhaseMatch)
    assert receipt.method == TARGET_PHASE_ALIGNMENT_METHOD
    np.testing.assert_allclose(
        receipt.target_query_offsets_frames,
        (-4.0, -8.0 / 3.0, -4.0 / 3.0, 0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(receipt.packet_phase_knots, receipt.target_phase_knots)
    assert receipt.target_stance_support_fraction == pytest.approx(0.93)
    assert receipt.min_stance_support_fraction == pytest.approx(0.8)
    assert receipt.support_window_s == pytest.approx(0.12)
    assert receipt.target_stance_source == "synthetic-contact-sensor-v1"


def test_target_alignment_handles_wrap_and_retains_center_error():
    packet_knots = phase_trace(7, 3, 0.98, 0.04)
    target = phase_trace(21, 10, 0.01, 0.03)

    receipt = target_align(target, packet_knots, max_phase_error=0.04)

    assert max(receipt.phase_errors) == pytest.approx(0.03)
    assert min(receipt.target_phase_knots) >= 0.0
    assert max(receipt.target_phase_knots) < 1.0
    assert receipt.target_query_offsets_frames[0] >= -5.0
    assert receipt.target_query_offsets_frames[-1] <= 5.0


def test_target_alignment_is_json_ready_and_immutable():
    packet_knots = phase_trace(7, 3, 0.40, 0.04)
    target = phase_trace(21, 10, 0.40, 0.03)
    receipt = target_align(target, packet_knots)

    json.dumps(receipt.as_dict(), sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        receipt.target_center_frame = 9


def test_target_alignment_rejects_insufficient_overlap_without_extrapolating():
    packet_knots = phase_trace(7, 3, 0.40, 0.04)
    target = phase_trace(21, 10, 0.40, 0.02)

    with pytest.raises(ValueError, match="insufficient phase overlap"):
        target_align(target, packet_knots, half=5)


def test_target_alignment_rejects_phase_mismatch_and_nonmonotone_trace():
    packet_knots = phase_trace(7, 3, 0.40, 0.04)
    mismatched = phase_trace(21, 10, 0.52, 0.03)
    with pytest.raises(ValueError, match="event-center phase error"):
        target_align(mismatched, packet_knots)

    nonmonotone = phase_trace(21, 10, 0.40, 0.03)
    nonmonotone[11] = nonmonotone[10] - 0.02
    with pytest.raises(ValueError, match="phase-ambiguous"):
        target_align(nonmonotone, packet_knots)


def test_target_alignment_rejects_bounds_ambiguous_grid_and_bad_stance():
    packet_knots = phase_trace(7, 3, 0.40, 0.04)
    target = phase_trace(21, 10, 0.40, 0.03)
    with pytest.raises(ValueError, match="full phase window"):
        target_align(target, packet_knots, center=3, half=5)

    ambiguous_grid = packet_knots.copy()
    ambiguous_grid[4] = 0.90
    with pytest.raises(ValueError, match="phase-ambiguous"):
        target_align(target, ambiguous_grid)

    with pytest.raises(ValueError, match="below the locked threshold"):
        align_target_phase_window(
            target,
            packet_knots,
            target_center_frame=10,
            search_half_window_frames=5,
            swing_side="left",
            target_stance=stance("right", 10, support=0.79),
            measurement_protocol_hash=PROTOCOL_HASH,
        )
