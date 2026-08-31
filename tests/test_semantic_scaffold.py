import numpy as np
import pytest

from scene2motion.semantic_scaffold import (
    StepEvent,
    build_transplanted_scaffold,
    select_unilateral_step_event,
)


NAMES = [
    "pelvis",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_toe_base",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_toe_base",
    "torso_skel",
]


def sample(T=10):
    root = np.stack([np.zeros(T), np.full(T, 0.78), np.arange(T) * 0.1], -1)
    offsets = np.zeros((len(NAMES), 3))
    offsets[:, 1] = [0.0, 0.4, 0.10, 0.05, 0.4, 0.10, 0.05, 0.7]
    posed = root[:, None, :] + offsets[None, :, :]
    contacts = np.ones((T, 4), dtype=bool)
    rotations = np.broadcast_to(np.eye(3), (T, len(NAMES), 3, 3)).copy()
    return {
        "smooth_root_pos": root,
        "posed_joints": posed,
        "foot_contacts": contacts,
        "global_rot_mats": rotations,
        "global_root_heading": np.tile([1.0, 0.0], (T, 1)),
    }


def test_selects_relative_unilateral_lift_not_bilateral_flight():
    s = sample()
    # A strong left step with the right foot in contact.
    s["foot_contacts"][4, :2] = False
    s["posed_joints"][4, 2:4, 1] += 0.30
    # A smaller right step.
    s["foot_contacts"][6, 2:4] = False
    s["posed_joints"][6, 5:7, 1] += 0.10
    # A high bilateral hop is deliberately ineligible.
    s["foot_contacts"][5] = False
    s["posed_joints"][5, 2:4, 1] += 1.0

    e = select_unilateral_step_event(s, NAMES)
    assert e.frame == 4
    assert e.side == "left"
    assert e.relative_lift_m > 0.0


def test_select_requires_unilateral_support():
    s = sample()
    s["foot_contacts"][:] = False
    with pytest.raises(ValueError, match="unilateral-support"):
        select_unilateral_step_event(s, NAMES)


def test_transplant_translates_full_body_to_target_root():
    s = sample()
    event = StepEvent(3, "left", 0.2, 0.3, 0.1)
    T = 10
    root_xz = np.stack([np.full(T, 0.25), np.arange(T) * 0.2], -1)
    root_y = np.full(T, 0.72)
    heading = np.zeros(T)

    spec, info = build_transplanted_scaffold(
        s, NAMES, 0, event, 5, root_xz, heading, root_y, "fullbody_pos",
        half_window_frames=1,
    )

    np.testing.assert_array_equal(spec.pos_frames, [4, 5, 6])
    np.testing.assert_array_equal(info.donor_frames, [2, 3, 4])
    assert 0 not in spec.pos_joints
    # Every transplanted pose keeps its donor joint offset from the pelvis, but its pelvis
    # is aligned to the prescribed target root.
    donor_offset = (s["posed_joints"][2, 1] - s["smooth_root_pos"][2])
    expected = np.array([root_xz[4, 0], root_y[4], root_xz[4, 1]]) + donor_offset
    np.testing.assert_allclose(spec.pos_targets[0, 0], expected)
    assert spec.rot_frames is None


def test_leg_only_and_position_rotation_are_distinct_channels():
    s = sample()
    event = StepEvent(3, "left", 0.2, 0.3, 0.1)
    T = 10
    root_xz = np.stack([np.zeros(T), np.arange(T) * 0.2], -1)
    root_y = np.full(T, 0.78)
    heading = np.zeros(T)

    leg, leg_info = build_transplanted_scaffold(
        s, NAMES, 0, event, 5, root_xz, heading, root_y, "leg_pos")
    both, both_info = build_transplanted_scaffold(
        s, NAMES, 0, event, 5, root_xz, heading, root_y, "fullbody_posrot")

    assert [NAMES[i] for i in leg.pos_joints] == [
        "left_knee_skel", "left_ankle_pitch_skel", "left_toe_base"
    ]
    assert leg.rot_frames is None
    np.testing.assert_array_equal(both.rot_frames, both.pos_frames)
    np.testing.assert_array_equal(both.pos_joints, np.arange(1, len(NAMES)))
    np.testing.assert_array_equal(both.rot_joints, np.arange(len(NAMES)))
    assert both.rot_targets.shape == (1, len(NAMES), 3, 3)
    assert 0 in both.rot_joints
    assert not leg_info.carries_rotations
    assert both_info.carries_rotations


def test_none_is_root_only_and_does_not_require_donor_arrays():
    T = 4
    root_xz = np.zeros((T, 2))
    spec, info = build_transplanted_scaffold(
        {}, NAMES, 0, StepEvent(0, "left", 0.0, 0.0, 0.0), 2,
        root_xz, np.zeros(T), np.full(T, 0.78), "none",
    )
    assert spec.pos_frames is None and spec.rot_frames is None
    assert info.target_frames == ()


def test_transplant_rotates_offsets_and_global_rotations_to_target_heading():
    s = sample()
    # Give one donor joint a +Z (forward) offset from the root.
    s["posed_joints"][:, 1] = s["smooth_root_pos"] + np.array([0.0, 0.2, 1.0])
    event = StepEvent(3, "left", 0.2, 0.3, 0.1)
    T = 10
    root_xz = np.zeros((T, 2))
    heading = np.full(T, np.pi / 2)  # target forward is +X in ARDY ground coordinates
    spec, _ = build_transplanted_scaffold(
        s, NAMES, 0, event, 5, root_xz, heading, np.full(T, 0.78),
        "fullbody_posrot",
    )

    joint_slot = list(spec.pos_joints).index(1)
    np.testing.assert_allclose(spec.pos_targets[0, joint_slot], [1.0, 0.98, 0.0], atol=1e-7)
    root_rot_slot = list(spec.rot_joints).index(0)
    expected_yaw = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(spec.rot_targets[0, root_rot_slot], expected_yaw, atol=1e-7)
