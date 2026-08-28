# Scene2Motion-G1: the ARDY-native constraint ACTION SPACE.
#
# This module is the crux of the V1 formulation. Rather than retraining a 280M-parameter
# motion prior to be scene-aware, we ask: what is the largest set of body adaptations that
# can already be REQUESTED of the frozen prior through its existing conditioning channels?
#
# ARDY conditions by overwriting slices of a 414-d per-frame feature vector with observed
# values, gated by a per-(frame, channel) binary mask
# (ardy/motion_rep/reps/ardy_motionrep.py:284 create_conditions -> _fill_*_constraints).
# The channels that _fill_* can write, and what each buys us:
#
#   name                  feature slice              maskable per     expresses
#   -------------------------------------------------------------------------------------
#   root_2d               root_pos[0], root_pos[2]   frame            WHERE to go
#   root_y_pos            root_pos[1]                frame            DUCK  (pelvis height)
#   global_root_heading   [3:5]  (cos,sin)           frame            TURN / SIDLE
#   global_joints_rots    [104:308], 6-d per joint   (frame, joint)   limb orientation
#   global_joints_positions -> local_joints_positions[5:104]
#                                                    (frame, joint)   TUCK / STEP-OVER
#
# Two facts about local_joints_positions matter and are easy to get wrong
# (ardy/motion_rep/reps/ardy_motionrep.py:137 __call__ and :366 _fill_global_position_constraints,
#  verified to agree):
#   local_joints_positions[t, j] = global_pos[t, j] - pelvis_pos[t] + [0, pelvis_y[t], 0]
# i.e. they are root-relative in the GROUND PLANE but ABSOLUTE IN HEIGHT, and they are NOT
# rotated into the heading frame. So a world-height target for a joint is expressible
# directly, which is exactly what overhead clearance needs.
# Also: the filler asserts that the ROOT joint is supplied at every frame that carries any
# joint-position constraint, and that root_2d is constrained at those frames.
#
# Measured effect of each channel on the frozen ARDY-G1-RP-25FPS-Horizon52 prior
# (experiments/exp001_capability_envelope):
#   root_2d alone     -> pelvis path tracked to ~1.4 cm mean / 3.1 cm max
#   root_y_pos -0.30  -> knee flexion 130 deg -> 77 deg, top-of-head 1.31 m -> ~1.05 m,
#                        feet stay planted, forward progress unchanged
#   joint tuck 0.35   -> lateral half-width 25.0 cm -> 19.2 cm (hands 22 -> 6 cm)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ConstraintSpec:
    """A scene-conditioned request to the frozen prior, in raw physical units.

    All arrays are ARDY-frame (Y-up, ground plane XZ, metres) and length T unless noted.
    This dataclass IS the V1 model's output space: a network that emits one of these is a
    scene-conditioned planner, and the prior turns it into motion.
    """

    root_xz: np.ndarray                     # (T, 2) pelvis ground path
    heading: np.ndarray | None = None       # (T,) radians from +Z; None -> free
    root_y: np.ndarray | None = None        # (T,) pelvis height; None -> free (duck channel)
    # Sparse joint position targets: which frames, which joints, and the world targets.
    pos_frames: np.ndarray | None = None    # (K,) frame indices
    pos_joints: np.ndarray | None = None    # (J,) joint indices (root is added automatically)
    pos_targets: np.ndarray | None = None   # (K, J, 3) world positions, ARDY frame

    def __post_init__(self):
        T = len(self.root_xz)
        for name in ("heading", "root_y"):
            v = getattr(self, name)
            if v is not None and len(v) != T:
                raise ValueError(f"{name} has length {len(v)}, expected {T}")
        if (self.pos_frames is None) != (self.pos_targets is None):
            raise ValueError("pos_frames and pos_targets must be given together")
        if self.pos_frames is not None:
            if self.pos_targets.shape[:2] != (len(self.pos_frames), len(self.pos_joints)):
                raise ValueError("pos_targets must be (K, J, 3) matching pos_frames/pos_joints")
            if self.root_y is None:
                # The filler needs a root height at every position-constrained frame and
                # takes it from the root target, so leaving root_y free there is a silent
                # inconsistency. Surface it rather than letting it corrupt the request.
                raise ValueError("joint-position constraints require root_y to be specified")

    @property
    def T(self) -> int:
        return len(self.root_xz)


class ArdyConstraintSet:
    """Adapter from a ConstraintSpec to ARDY's ``update_constraints`` protocol.

    ARDY's own constraint classes (Root2DConstraintSet, FullBodyConstraintSet, ...) all
    require a reference MOTION to copy from — they are built for "sample constraints from an
    existing clip". We need to author constraints from a PLAN, with nothing to copy, so we
    speak the same duck-typed protocol directly:
    ``update_constraints(data_dict, index_dict)`` appending into the keys the fillers read.
    """

    name = "scene2motion"

    def __init__(self, spec: ConstraintSpec, root_idx: int, device: str):
        self.spec, self.root_idx, self.device = spec, root_idx, device

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def update_constraints(self, data_dict: dict, index_dict: dict) -> None:
        s = self.spec
        fi = torch.arange(s.T, device=self.device)

        data_dict["root_2d"].append(self._t(s.root_xz))
        index_dict["root_2d"].append(fi)

        if s.heading is not None:
            h = self._t(s.heading)
            data_dict["global_root_heading"].append(torch.stack([torch.cos(h), torch.sin(h)], -1))
            index_dict["global_root_heading"].append(fi)

        if s.root_y is not None:
            data_dict["root_y_pos"].append(self._t(s.root_y))
            index_dict["root_y_pos"].append(fi)

        if s.pos_frames is not None:
            pf = self._t(s.pos_frames, torch.long)
            # The root must appear at every position-constrained frame, and its target must
            # agree with root_xz / root_y or the two requests fight each other.
            joints = self._t(np.concatenate([[self.root_idx], s.pos_joints]), torch.long)
            root_tgt = torch.stack([
                self._t(s.root_xz[s.pos_frames, 0]),
                self._t(s.root_y[s.pos_frames]),
                self._t(s.root_xz[s.pos_frames, 1]),
            ], -1)                                             # (K, 3)
            targets = torch.cat([root_tgt[:, None], self._t(s.pos_targets)], 1)  # (K, 1+J, 3)
            pairs = torch.stack([
                pf[:, None].expand(-1, len(joints)).reshape(-1),
                joints[None].expand(len(pf), -1).reshape(-1),
            ], -1)
            data_dict["global_joints_positions"].append(targets.reshape(-1, 3))
            index_dict["global_joints_positions"].append(pairs)


def build_conditions(model, spec: ConstraintSpec, device: str):
    """(observed_motion, motion_mask) ready to pass to ``Ardy.__call__``."""
    cs = ArdyConstraintSet(spec, model.skeleton.root_idx, device)
    return model.motion_rep.create_conditions_from_constraints_batched(
        [cs], torch.tensor([spec.T], device=device), to_normalize=True, device=device
    )


def channel_usage(model, mask: torch.Tensor) -> dict[str, int]:
    """How many (frame, channel) entries each named feature block actually constrains.

    Cheap guard against the most common authoring bug: writing a channel the fillers never
    read, and so silently generating an unconstrained motion.
    """
    return {k: int(mask[0][:, sl].sum().item()) for k, sl in model.motion_rep.slice_dict.items()}
