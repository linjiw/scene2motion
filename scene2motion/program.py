# Scene2Motion-G1: the CONSTRAINT PROGRAM — the output space of p_phi(C | S, s, g).
#
# This is the object the generator emits. It is deliberately NOT motion (34 joints x T) and
# NOT a dense per-frame constraint tensor: the multimodality this project exists to model
# lives in the DECISION (duck under vs. walk around), not in centimetres of knee angle, so
# the generative capacity belongs in a small structured program and the motion manifold
# stays with the frozen prior.
#
# Two measurements fix the design:
#
#  * EXP-003b. Sparsifying the REQUEST degrades the duck sharply (collision-free 100% dense
#    -> 94% at 8 waypoints -> 0% at 4) while the route survives to 4 waypoints. Crucially the
#    failure tracks control points INSIDE the adaptation window, not total waypoint count. So
#    the model's output is sparse and the request handed to ARDY is DENSE: those are
#    different sparsities, and conflating them is what kills the crouch.
#
#  * EXP-001. The body modes in outputs/body_modes.json are a coarse quantisation (5 duck
#    levels x 2 tuck x 2 lift) of axes that were measured continuously at 11 x 8 x 6 x 5
#    levels. Quantising costs unnecessary over-crouch, and the discrete table cannot express
#    duck-and-tuck at once — a real limitation, since `beam_and_gap` needs exactly that. The
#    program therefore carries four CONTINUOUS adaptation axes and no discrete mode at all.
#
# Layout (39 emitted numbers, all normalised to [-1, 1] for the generator):
#     lat    (14,)   signed lateral offset from the start->goal chord at 16 knots,
#                    endpoints pinned to 0. Translation- and rotation-invariant by
#                    construction, and left-vs-right homotopy is just the sign of the
#                    middle knots.
#     slot   (4, 5)  four adaptation slots, each [s_mid, s_half, dip, tuck, lift], where s_*
#                    are positions along the path in [0, 1]. An inactive slot is one whose
#                    intensities are ~0, so slot activity needs no discrete head.
#                    There is no `sidle` field: it was identically 0.000 in all 278 corpus
#                    programs (every calibrated mode has sidle_deg = 0), and EXP-001b measured
#                    that sidling makes G1 WIDER rather than narrower. A channel that is
#                    always zero is capacity the generator would spend learning to emit zero.
#     speed  ()      traversal speed, which sets the clip duration.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constraints import ConstraintSpec
from .planner import (LEAD_S, MODE_BY_NAME, Plan, _dilate_channel, _limb_targets,
                      _path_channels, _resample, _smooth)

# N_LAT=24 halves the geometric route error against 16 for eight extra numbers; 32 buys only
# another 2 cm. Measured Hausdorff p95 over all rungs x strategies: 0.117 (16) / 0.076 (24) /
# 0.056 (32).
N_LAT, M_SLOT, N_AX = 24, 4, 5
DIM_C = (N_LAT - 2) + M_SLOT * N_AX + 1                   # 22 + 20 + 1 = 43

NOMINAL_PELVIS = MODE_BY_NAME["stand"].pelvis_y
# Lead time is shared with the planner renderer (planner.LEAD_S) so a program and the oracle
# plan it was fitted to are rendered identically. EXP-004b measured the requirement at
# 0.2-0.3 s; 0.8 s sits mid-plateau.
SMOOTH_S = 0.6

# Ranges are the MEASURED reachable extents, so a program in [-1, 1] cannot ask for an
# envelope the prior was never shown to produce (EXP-001 dip 0-0.50, EXP-001b tuck 0-0.85,
# EXP-001c lift 0-0.55).
# LAT_SCALE is 1.50, not 0.60: measured max |lateral offset| over the corpus is 1.458 m
# (p95 = 1.00), so 0.60 pushed to_vec() to [-2.17, +2.08] and broke the [-1, 1] contract the
# generator's output activation depends on.
LAT_SCALE, DIP_MAX, TUCK_MAX, LIFT_MAX = 1.50, 0.50, 0.85, 0.55
SHALF_MAX = 0.35
SPEED_MID, SPEED_HALF = 0.9, 0.4                          # -> [0.5, 1.3] m/s
ACTIVE = {"dip": 0.03, "tuck": 0.05, "lift": 0.05}
_ARC_SAMPLES = 2048


@dataclass
class ConstraintProgram:
    """A whole-body traversal strategy as 39 numbers."""

    lat: np.ndarray = field(default_factory=lambda: np.zeros(N_LAT - 2))
    slot: np.ndarray = field(default_factory=lambda: np.zeros((M_SLOT, N_AX)))
    speed: float = 0.9

    # -- generator interface ------------------------------------------------------------

    def to_vec(self) -> np.ndarray:
        """(DIM_C,) in [-1, 1]."""
        s = self.slot
        return np.concatenate([
            self.lat / LAT_SCALE,
            np.stack([2 * s[:, 0] - 1,
                      2 * s[:, 1] / SHALF_MAX - 1,
                      2 * s[:, 2] / DIP_MAX - 1,
                      2 * s[:, 3] / TUCK_MAX - 1,
                      2 * s[:, 4] / LIFT_MAX - 1], -1).ravel(),
            [(self.speed - SPEED_MID) / SPEED_HALF],
        ]).astype(np.float32)

    @staticmethod
    def from_vec(v: np.ndarray) -> "ConstraintProgram":
        # The box IS the capability set: every axis is normalised by a MEASURED reachable
        # extent, so clipping to [-1, 1] is what makes an out-of-envelope program
        # unrepresentable rather than merely discouraged.
        v = np.clip(np.asarray(v, float), -1.0, 1.0)
        i = N_LAT - 2
        s = v[i:i + M_SLOT * N_AX].reshape(M_SLOT, N_AX)
        slot = np.stack([(s[:, 0] + 1) / 2,
                         (s[:, 1] + 1) / 2 * SHALF_MAX,
                         (s[:, 2] + 1) / 2 * DIP_MAX,
                         (s[:, 3] + 1) / 2 * TUCK_MAX,
                         (s[:, 4] + 1) / 2 * LIFT_MAX], -1)
        return ConstraintProgram(lat=v[:i] * LAT_SCALE, slot=slot,
                                 speed=float(v[-1] * SPEED_HALF + SPEED_MID))

    @property
    def active_slots(self) -> list[int]:
        """Slots asking for a real adaptation, by the measured activity thresholds."""
        return [k for k in range(M_SLOT)
                if (self.slot[k, 2] > ACTIVE["dip"] or self.slot[k, 3] > ACTIVE["tuck"]
                    or self.slot[k, 4] > ACTIVE["lift"])]


# ---------------------------------------------------------------------------------------
# Encode: an oracle Plan -> a program
# ---------------------------------------------------------------------------------------

def _chord_frame(scene) -> tuple[np.ndarray, np.ndarray, float]:
    s, g = np.asarray(scene.start, float), np.asarray(scene.goal, float)
    d = g - s
    L = float(np.linalg.norm(d))
    e = d / max(L, 1e-9)
    return s, np.array([-e[1], e[0]]), L        # origin, unit normal, chord length


def encode(plan: Plan, scene, fps: float, speed: float = 0.9) -> ConstraintProgram:
    """Fit a program to a planner solution.

    The route becomes lateral offsets from the chord; each contiguous run of a given
    adaptation becomes one slot. Runs are taken from the mode schedule the planner produced,
    then converted to the continuous axes so the discrete vocabulary does not survive into
    the program.
    """
    origin, normal, L = _chord_frame(scene)
    T = max(int(2 * fps), int(round(plan.length / speed * fps)))
    xy, modes = _resample(plan.xy, plan.modes, T)

    # lateral offsets at N_LAT uniform chord abscissae, endpoints pinned
    along = (xy - origin) @ (np.array([normal[1], -normal[0]]))
    u = np.clip(along / max(L, 1e-9), 0, 1)
    lat_full = (xy - origin) @ normal
    knots = np.linspace(0, 1, N_LAT)
    order = np.argsort(u)
    lat = np.interp(knots, u[order], lat_full[order])
    lat[0] = lat[-1] = 0.0

    # adaptation runs -> slots
    axes = np.stack([
        NOMINAL_PELVIS - np.array([m.pelvis_y for m in modes]),     # dip
        np.array([m.tuck for m in modes]),
        np.array([m.lift for m in modes]),
    ], -1)
    active = ((axes[:, 0] > ACTIVE["dip"]) | (axes[:, 1] > ACTIVE["tuck"])
              | (axes[:, 2] > ACTIVE["lift"]))
    slot = np.zeros((M_SLOT, N_AX))
    runs, i = [], 0
    while i < T:
        if active[i]:
            j = i
            while j + 1 < T and active[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    # Keep the strongest runs when there are more than slots, then order the KEPT runs by
    # position. Sorting slots by intensity makes slot identity unstable -- a deep duck before
    # a shallow tuck swaps rows against the same scene with the intensities reversed -- and a
    # set predictor matching slots to targets would be fighting that permutation.
    runs.sort(key=lambda r: -axes[r[0]:r[1] + 1].max())
    for k, (a, b) in enumerate(sorted(runs[:M_SLOT])):
        mid, half = (a + b) / 2 / max(T - 1, 1), (b - a) / 2 / max(T - 1, 1)
        v = axes[a:b + 1].max(axis=0)
        slot[k] = [mid, min(half, SHALF_MAX), min(v[0], DIP_MAX),
                   min(v[1], TUCK_MAX), min(v[2], LIFT_MAX)]
    return ConstraintProgram(lat=lat[1:-1], slot=slot, speed=speed)


# ---------------------------------------------------------------------------------------
# Decode: a program -> the dense ConstraintSpec the frozen prior consumes
# ---------------------------------------------------------------------------------------

def decode(prog: ConstraintProgram, scene, fps: float, nominal: dict | None = None,
           joint_names: list[str] | None = None,
           duration: float | None = None) -> ConstraintSpec:
    """Render a program densely. The model's output is sparse; ARDY's request is not."""
    origin, normal, L = _chord_frame(scene)
    tangent = np.array([normal[1], -normal[0]])
    T = max(int(2 * fps),
            int(round((duration if duration is not None else L / max(prog.speed, 0.1)) * fps)))

    # The knots are uniform in CHORD abscissa; the plan they were fitted to is uniform in
    # ARC LENGTH. Sampling the decode uniformly in abscissa therefore lags the plan by up to
    # 0.30 m wherever the route bends, and no number of knots removes it -- it is a
    # parameterisation mismatch, not a resolution one. So render densely, then resample by
    # arc length to match how the planner lays frames along a path.
    knots = np.linspace(0, 1, N_LAT)
    lat = np.concatenate([[0.0], prog.lat, [0.0]])
    a = np.linspace(0, 1, _ARC_SAMPLES)
    dense = origin + np.outer(a * L, tangent) + np.outer(np.interp(a, knots, lat), normal)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, arc[-1], T)
    xy = np.stack([np.interp(want, arc, dense[:, 0]),
                   np.interp(want, arc, dense[:, 1])], -1)
    root_xz, heading = _path_channels(xy, fps)

    # Slots -> per-frame axes, then the same channel dilation and smoothing the planner
    # renderer uses, so a program and an oracle plan reach ARDY through identical machinery.
    u = np.linspace(0, 1, T)
    dip = np.zeros(T)
    tuck = np.zeros(T)
    lift = np.zeros(T)
    for k in prog.active_slots:
        mid, half, d, tk, lf = prog.slot[k]
        m = np.abs(u - mid) <= max(half, 1.0 / T)
        dip[m] = np.maximum(dip[m], d)
        tuck[m] = np.maximum(tuck[m], tk)
        lift[m] = np.maximum(lift[m], lf)

    w = int(round(LEAD_S * fps))
    sm = max(3, int(SMOOTH_S * fps) | 1)
    root_y = _smooth(NOMINAL_PELVIS - _dilate_channel(dip, w, "max"), sm)
    tuck = _smooth(_dilate_channel(tuck, w, "max"), sm)
    lift = _smooth(_dilate_channel(lift, w, "max"), sm)


    pos_frames = pos_joints = pos_targets = None
    if nominal is not None and joint_names is not None and (tuck.max() > ACTIVE["tuck"]
                                                            or lift.max() > ACTIVE["lift"]):
        pos_frames, pos_joints, pos_targets = _limb_targets(
            root_xz, tuck, lift, nominal, joint_names, fps, root_y=root_y)

    return ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                          pos_frames=pos_frames, pos_joints=pos_joints,
                          pos_targets=pos_targets, first_heading=float(heading[0]))
