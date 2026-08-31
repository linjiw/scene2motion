"""Semantic-to-geometric scaffolds for spatially placing an ARDY behaviour.

Text can elicit a coordinated motion without putting it at the place where a scene needs it.
Conversely, an isolated joint-position request can name the place while omitting the support
and posture context that made that request coherent in motion capture.  This module keeps those
two factors separate: it finds a unilateral step in a text-generated donor and can transplant
either only the swing-leg positions or the donor's full-body pose to a prescribed route frame.

The donor and evaluation seeds are deliberately managed by the calling experiment.  Nothing in
this module selects on an evaluation output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .constraints import ConstraintSpec


ScaffoldMode = Literal["none", "leg_pos", "fullbody_pos", "fullbody_posrot"]
SCAFFOLD_MODES: tuple[ScaffoldMode, ...] = (
    "none", "leg_pos", "fullbody_pos", "fullbody_posrot"
)


@dataclass(frozen=True)
class StepEvent:
    """A donor frame with one foot airborne and the other foot supporting the body."""

    frame: int
    side: Literal["left", "right"]
    # Mean ankle/toe height of the swing side minus that of the stance side.  A relative
    # quantity prevents a whole-body hop from winning merely because the pelvis moved up.
    relative_lift_m: float
    swing_height_m: float
    stance_height_m: float


@dataclass(frozen=True)
class ScaffoldInfo:
    """Serializable provenance for one transplanted scaffold."""

    mode: ScaffoldMode
    donor_frames: tuple[int, ...]
    target_frames: tuple[int, ...]
    joint_indices: tuple[int, ...]
    rotation_joint_indices: tuple[int, ...]
    swing_side: str | None
    carries_rotations: bool


def _side_joint_indices(joint_names: Sequence[str], side: str,
                        tokens: tuple[str, ...]) -> np.ndarray:
    prefix = f"{side}_"
    return np.asarray([
        i for i, name in enumerate(joint_names)
        if name.startswith(prefix) and any(token in name for token in tokens)
    ], dtype=int)


def select_unilateral_step_event(sample: dict, joint_names: Sequence[str],
                                  frame_window: tuple[int, int] | None = None) -> StepEvent:
    """Find the strongest contact-preserving step frame in an ARDY output.

    ``foot_contacts`` is used only to identify unilateral support.  The ranking statistic is
    measured from ``posed_joints`` and is relative to the stance foot, so bilateral flight and
    pelvis bob cannot masquerade as a step.  Ties are resolved by the earlier frame, making
    donor selection deterministic.
    """
    posed = np.asarray(sample["posed_joints"], dtype=float)
    contacts = np.asarray(sample["foot_contacts"], dtype=bool)
    if posed.ndim != 3 or posed.shape[-1] != 3:
        raise ValueError("posed_joints must have shape (T, J, 3)")
    if contacts.ndim != 2 or contacts.shape[0] != posed.shape[0] or contacts.shape[1] < 4:
        raise ValueError("foot_contacts must have shape (T, >=4) aligned with posed_joints")
    if posed.shape[1] != len(joint_names):
        raise ValueError("joint_names must match the joint dimension of posed_joints")

    feet = {
        side: _side_joint_indices(joint_names, side, ("ankle", "toe"))
        for side in ("left", "right")
    }
    if any(len(v) == 0 for v in feet.values()):
        raise ValueError("could not identify ankle/toe joints for both sides")

    T = len(posed)
    lo, hi = frame_window if frame_window is not None else (int(0.1 * T), int(0.9 * T))
    lo, hi = max(0, int(lo)), min(T, int(hi))
    if lo >= hi:
        raise ValueError("frame_window contains no frames")
    on_ground = {
        "left": contacts[:, 0:2].any(-1),
        "right": contacts[:, 2:4].any(-1),
    }

    candidates: list[StepEvent] = []
    for side, stance in (("left", "right"), ("right", "left")):
        valid = np.where(~on_ground[side] & on_ground[stance])[0]
        for frame in valid[(valid >= lo) & (valid < hi)]:
            swing_h = float(posed[frame, feet[side], 1].mean())
            stance_h = float(posed[frame, feet[stance], 1].mean())
            candidates.append(StepEvent(int(frame), side, swing_h - stance_h,
                                        swing_h, stance_h))
    if not candidates:
        raise ValueError("donor contains no unilateral-support frame in the requested window")
    return max(candidates, key=lambda e: (e.relative_lift_m, -e.frame,
                                          e.side == "left"))


def _aligned_frames(donor_center: int, target_center: int, donor_length: int,
                    target_length: int, half_window_frames: int,
                    stride_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if half_window_frames < 0:
        raise ValueError("half_window_frames must be non-negative")
    if stride_frames < 1:
        raise ValueError("stride_frames must be positive")
    offsets = list(range(-half_window_frames, half_window_frames + 1, stride_frames))
    if 0 not in offsets:
        offsets.append(0)
    offsets = np.asarray(sorted(set(offsets)), dtype=int)
    donor = donor_center + offsets
    target = target_center + offsets
    keep = ((donor >= 0) & (donor < donor_length) &
            (target >= 0) & (target < target_length))
    donor, target = donor[keep], target[keep]
    if not len(donor):
        raise ValueError("donor and target blocks do not overlap their clips")
    return donor, target


def build_transplanted_scaffold(
        sample: dict,
        joint_names: Sequence[str],
        root_idx: int,
        event: StepEvent,
        target_frame: int,
        root_xz: np.ndarray,
        heading: np.ndarray,
        root_y: np.ndarray,
        mode: ScaffoldMode,
        *,
        half_window_frames: int = 0,
        stride_frames: int = 1,
        first_heading: float | None = None,
) -> tuple[ConstraintSpec, ScaffoldInfo]:
    """Place a donor step pose/block at ``target_frame`` on a prescribed root path.

    Position targets are translated per frame so the donor pelvis coincides with the target
    root.  ``leg_pos`` carries only the airborne leg's knee/ankle/toe positions;
    ``fullbody_pos`` additionally says what the stance leg, arms, and torso are doing;
    ``fullbody_posrot`` writes the decoder-facing rotation channel as a separate diagnostic.
    Position keyframes exclude the root because :class:`ConstraintSpec` supplies it from
    ``root_xz`` and ``root_y``; the rotation keyframe includes the root so that the mode is
    genuinely full-body.
    """
    if mode not in SCAFFOLD_MODES:
        raise ValueError(f"unknown scaffold mode {mode!r}; expected one of {SCAFFOLD_MODES}")
    root_xz = np.asarray(root_xz, dtype=float)
    heading = np.asarray(heading, dtype=float)
    root_y = np.asarray(root_y, dtype=float)
    if root_xz.ndim != 2 or root_xz.shape[1] != 2:
        raise ValueError("root_xz must have shape (T, 2)")
    T = len(root_xz)
    if heading.shape != (T,) or root_y.shape != (T,):
        raise ValueError("heading and root_y must each have shape (T,)")
    if not 0 <= int(root_idx) < len(joint_names):
        raise ValueError("root_idx is outside joint_names")

    base = ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                          first_heading=first_heading)
    if mode == "none":
        return base, ScaffoldInfo(mode, (), (), (), (), None, False)

    posed = np.asarray(sample["posed_joints"], dtype=float)
    donor_root = np.asarray(sample["smooth_root_pos"], dtype=float)
    if posed.ndim != 3 or posed.shape[1:] != (len(joint_names), 3):
        raise ValueError("posed_joints must have shape (Td, len(joint_names), 3)")
    if donor_root.shape != (len(posed), 3):
        raise ValueError("smooth_root_pos must have shape (Td, 3)")
    donor_heading_cs = np.asarray(sample.get("global_root_heading"), dtype=float)
    if donor_heading_cs.shape != (len(posed), 2):
        raise ValueError("global_root_heading must have shape (Td, 2)")

    donor_frames, target_frames = _aligned_frames(
        event.frame, int(target_frame), len(posed), T,
        int(half_window_frames), int(stride_frames))
    if mode == "leg_pos":
        joints = _side_joint_indices(joint_names, event.side, ("knee", "ankle", "toe"))
        if not len(joints):
            raise ValueError(f"could not identify {event.side} leg joints")
    else:
        joints = np.asarray([i for i in range(len(joint_names)) if i != root_idx], dtype=int)

    target_root = np.stack([
        root_xz[target_frames, 0], root_y[target_frames], root_xz[target_frames, 1]
    ], axis=-1)
    donor_heading = np.arctan2(donor_heading_cs[donor_frames, 1],
                               donor_heading_cs[donor_frames, 0])
    delta_heading = heading[target_frames] - donor_heading
    c, s = np.cos(delta_heading), np.sin(delta_heading)
    yaw = np.zeros((len(target_frames), 3, 3), dtype=float)
    yaw[:, 0, 0], yaw[:, 0, 2] = c, s
    yaw[:, 1, 1] = 1.0
    yaw[:, 2, 0], yaw[:, 2, 2] = -s, c
    offsets = posed[donor_frames][:, joints, :] - donor_root[donor_frames, None, :]
    pos_targets = target_root[:, None, :] + np.einsum("fij,fkj->fki", yaw, offsets)

    kwargs = dict(
        root_xz=root_xz,
        heading=heading,
        root_y=root_y,
        pos_frames=target_frames,
        pos_joints=joints,
        pos_targets=pos_targets,
        first_heading=first_heading,
    )
    carries_rotations = mode == "fullbody_posrot"
    rot_joints = np.asarray([], dtype=int)
    if carries_rotations:
        rotations = np.asarray(sample.get("global_rot_mats"), dtype=float)
        if rotations.shape != (len(posed), len(joint_names), 3, 3):
            raise ValueError("global_rot_mats must have shape (Td, J, 3, 3)")
        # Rotation constraints do not need the position channel's automatic-root rule, so
        # include the root as well: this is genuinely a full-body rotation keyframe. Rotate
        # every donor global rotation into the target route heading.
        rot_joints = np.arange(len(joint_names), dtype=int)
        rot_targets = np.einsum(
            "fij,fkjl->fkil", yaw, rotations[donor_frames][:, rot_joints])
        kwargs.update(rot_frames=target_frames, rot_joints=rot_joints,
                      rot_targets=rot_targets)
    spec = ConstraintSpec(**kwargs)
    info = ScaffoldInfo(
        mode=mode,
        donor_frames=tuple(int(x) for x in donor_frames),
        target_frames=tuple(int(x) for x in target_frames),
        joint_indices=tuple(int(x) for x in joints),
        rotation_joint_indices=tuple(int(x) for x in rot_joints),
        swing_side=event.side,
        carries_rotations=carries_rotations,
    )
    return spec, info
