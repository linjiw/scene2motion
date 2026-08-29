# Scene2Motion-G1: measuring HOW the body traverses, separately from WHERE it goes.
#
# EXP-005e found that the corridor-exclusion enumerator is near-complete over routes (2.9 % of
# its misses involve a new route) and structurally blind to a different body adaptation along
# the SAME route (93.6 %). That factorises traversal into
#
#     route  = RoutePlanner(S, s, g)          classical, and saturated
#     body   ~ q(b | S, route)                the open problem
#
# and makes "how the body went" the quantity that has to be measured properly. This module is
# that measurement. Four traps it exists to avoid, each already paid for once in this repo:
#
#  1. BANDS DECIDE THE ANSWER. Quantising a smooth feasible interval into ten bins yields ten
#     "strategies"; two bins yield two. So the signature is HIERARCHICAL: a discrete active-set
#     (which adaptations are engaged at all) over a continuous descriptor (how much).
#
#  2. NOISE MASQUERADES AS DIVERSITY. Two ARDY samples of the SAME program differ. Distances
#     are therefore whitened by the seed covariance and reported in units of ARDY-noise sigma,
#     so "different" means "further apart than the prior's own scatter".
#
#  3. CONFOUNDS MASQUERADE AS ADAPTATION. EXP-001 once concluded that ducking makes the robot
#     wider, from comparing a clip against its own opening gait. Every quantity here is a
#     PAIRED DELTA against a neutral-body program on the same route with the same seed.
#
#  4. JOINTS ARE NOT THE BODY. G1's head sits 23.6 cm above ARDY's highest joint (EXP-000), so
#     every extent is read off the inflated MuJoCo collision geometry.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Descriptor channels, in order. Units are metres except the two times (seconds) and the
# heading (radians). All are DELTAS against the matched control unless stated.
CHANNELS = ("dh_top", "dw_left", "dw_right", "dz_foot_left", "dz_foot_right",
            "t_lead", "t_duration", "dpsi")
N_CHANNELS = len(CHANNELS)

# Physical floors, below which a change is not worth calling an adaptation whatever the
# statistics say: 2 cm of envelope, 3 cm of foot, 5 degrees of heading. The statistical
# threshold is computed from matched-control noise and the larger of the two is used.
PHYSICAL_FLOOR = {"dh_top": 0.02, "dw_left": 0.02, "dw_right": 0.02,
                  "dz_foot_left": 0.03, "dz_foot_right": 0.03, "dpsi": np.deg2rad(5.0)}


@dataclass
class Interaction:
    """Where along the route the body has to do something, in metres of travel."""

    x_center: float
    half_width: float = 0.7

    def mask(self, qpos: np.ndarray) -> np.ndarray:
        return np.abs(qpos[:, 0] - self.x_center) <= self.half_width


def _signed_extents(body, qpos_t: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """(left, right) surface extent about the pelvis along `normal`, both positive.

    Side-specific, because tucking one arm is a different strategy from tucking both and a
    symmetric half-width cannot tell them apart.
    """
    body.fk(qpos_t)
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    pel = float(body.data.qpos[:3] @ n)
    lo = hi = 0.0
    for g in body.robot_geoms:
        a, b = body._extent(g, n)
        hi = max(hi, b - pel)
        lo = min(lo, a - pel)
    return float(hi), float(-lo)


def heading_normal(qpos_t: np.ndarray) -> np.ndarray:
    """The robot's OWN lateral axis at one frame, from its root quaternion (MuJoCo wxyz)."""
    w, x, y, z = qpos_t[3:7]
    # yaw of the root frame; the lateral axis is the body +y expressed in world
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([-np.sin(yaw), np.cos(yaw), 0.0])


def envelope_series(body, qpos: np.ndarray,
                    normal=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """(T, 3) per-frame (top, left extent, right extent). Computed once per clip.

    `matched_delta` needs this for both members of a pair; recomputing the control's series
    for every adapted clip it is compared against costs a factor of the ladder size, which on
    a 36-program sweep is most of the run.

    `normal` may be the string "heading", which measures each frame along the ROBOT'S OWN
    lateral axis instead of a fixed world axis.  The fixed axis is wrong in principle: rotating
    a standing G1 by 26 degrees -- the q99 of the yaw channel -- with no posture change at all
    moves the apparent left extent by 11.6 cm, while along the robot's own axis it moves by
    exactly zero.  Measured on EXP-005f's 1296 paired residuals the heading term accounts for
    only R^2 = 0.021 of the width noise and removing it moves q99 by 3 %, so this is a
    correctness fix rather than a rescue of any result.

    The DEFAULT is deliberately still the fixed world axis, because every artefact currently on
    disk -- the EXP-005f noise floor, the EXP-005g ledger and everything whitened against it --
    was measured that way, and silently changing the ruler underneath them would make the
    ledger and anything selected from it incomparable.  The frozen final run flips it.
    """
    per_frame = isinstance(normal, str) and normal == "heading"
    out = np.empty((len(qpos), 3))
    for t, q in enumerate(qpos):
        n = heading_normal(q) if per_frame else normal
        L, R = _signed_extents(body, q, n)
        out[t] = (body.top_height(q), L, R)
    return out


def raw_descriptor(body, qpos: np.ndarray, interaction: Interaction,
                   fps: float, normal=np.array([0.0, 1.0, 0.0])) -> dict:
    """`normal` may be "heading" -- see `envelope_series` for why the default is not."""
    """Absolute morphology quantities for one clip. Deltas are taken later.

    `t_lead` and `t_duration` are left as NaN here: they are properties of a DIFFERENCE from a
    control, not of a single clip, and are filled in by `matched_delta`.
    """
    m = interaction.mask(qpos)
    if m.sum() < 3:
        m = np.ones(len(qpos), bool)
    tops, wl, wr, fl, fr = [], [], [], [], []
    foot = [g for g in body.robot_geoms if "foot" in body.geom_name[g]]
    for t in np.flatnonzero(m):
        q = qpos[t]
        tops.append(body.top_height(q))
        L, R = _signed_extents(body, q, normal)
        wl.append(L)
        wr.append(R)
        body.fk(q)
        z = body.data.geom_xpos[foot][:, 2] if foot else np.array([0.0])
        half = len(foot) // 2 or 1
        fl.append(float(z[:half].max()))
        fr.append(float(z[half:].max()) if len(z) > half else float(z.max()))
    from scipy.spatial.transform import Rotation
    yaw = Rotation.from_quat(qpos[m][:, 3:7][:, [1, 2, 3, 0]]).as_euler("zyx")[:, 0]
    return {"top": float(np.max(tops)), "w_left": float(np.max(wl)),
            "w_right": float(np.max(wr)), "foot_left": float(np.max(fl)),
            "foot_right": float(np.max(fr)), "yaw": float(np.mean(yaw)),
            "_series_top": np.array(tops), "_mask": m}


def matched_delta(adapted: dict, control: dict, qpos_adapted: np.ndarray,
                  qpos_control: np.ndarray, interaction: Interaction, fps: float,
                  body=None, env_adapted: np.ndarray | None = None,
                  env_control: np.ndarray | None = None) -> np.ndarray:
    """(N_CHANNELS,) paired delta of `adapted` against its matched control.

    Signs are chosen so that POSITIVE always means "more adaptation": the body got shorter,
    narrower, lifted its feet higher, turned further.
    """
    dh = control["top"] - adapted["top"]
    dwl = control["w_left"] - adapted["w_left"]
    dwr = control["w_right"] - adapted["w_right"]
    dzl = adapted["foot_left"] - control["foot_left"]
    dzr = adapted["foot_right"] - control["foot_right"]
    dpsi = abs(adapted["yaw"] - control["yaw"])

    # Timing, from the whole-clip envelope difference rather than the interaction window: lead
    # time is precisely the part that happens BEFORE the window.
    #
    # The series must cover EVERY envelope channel, not just height. Measuring timing on top
    # height alone gave a tuck-only program a lead of -4.72 s: tucking does not change height,
    # so the threshold fired on noise and the onset landed at a random frame. The norm over
    # (top, left width, right width) responds to whichever adaptation is actually present.
    t_lead = t_dur = 0.0
    if env_adapted is None and body is not None:
        env_adapted = envelope_series(body, qpos_adapted)
    if env_control is None and body is not None:
        env_control = envelope_series(body, qpos_control)
    if env_adapted is not None and env_control is not None:
        n = min(len(env_adapted), len(env_control))
        d = np.abs(env_adapted[:n] - env_control[:n]).sum(axis=1)
        thr = max(PHYSICAL_FLOOR["dh_top"], 0.25 * d.max()) if d.max() > 0 else np.inf
        on = np.flatnonzero(d > thr)
        if len(on):
            enter = np.flatnonzero(interaction.mask(qpos_adapted[:n]))
            # Lead is clamped at zero: an adaptation that starts INSIDE the window has no
            # lead, and a negative number here would be a measurement artefact, not late
            # anticipation.
            t_lead = float(max(0, enter[0] - on[0]) / fps) if len(enter) else 0.0
            t_dur = float((on[-1] - on[0] + 1) / fps)
    return np.array([dh, dwl, dwr, dzl, dzr, t_lead, t_dur, dpsi], float)


def active_set(delta: np.ndarray, noise_q99: np.ndarray) -> tuple:
    """Layer 1: WHICH adaptations are engaged, as a discrete tuple.

    A channel is active when its delta clears both the matched-control noise quantile and the
    physical floor. Using the measured noise rather than a round number is what stops ordinary
    gait variation from being labelled a new body strategy.
    """
    def on(i: int, name: str) -> bool:
        return bool(delta[i] > max(noise_q99[i], PHYSICAL_FLOOR.get(name, 0.0)))
    duck = on(0, "dh_top")
    tuck = on(1, "dw_left") or on(2, "dw_right")
    liftL, liftR = on(3, "dz_foot_left"), on(4, "dz_foot_right")
    yaw = on(7, "dpsi")
    # Lift ORDER distinguishes left-foot-first from right-foot-first step-overs. It is only
    # meaningful when the two feet differ by MORE than the noise on the foot channels: a bare
    # sign comparison assigns an order to every clip, and since which foot is higher near an
    # obstacle depends mostly on gait phase, that turns seed noise into a strategy label. It
    # was inflating the distinct-active-set count before this threshold was added.
    order = 0
    if liftL and liftR:
        gap = abs(delta[3] - delta[4])
        if gap > max(noise_q99[3], noise_q99[4]):
            order = 1 if delta[3] >= delta[4] else -1
    return (duck, tuck, liftL, liftR, order, yaw)


def seed_statistics(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, covariance) of a program's descriptor over its ARDY seeds."""
    d = np.atleast_2d(deltas)
    return d.mean(axis=0), np.cov(d, rowvar=False) if len(d) > 1 else np.zeros(
        (d.shape[1], d.shape[1]))


def whitener(seed_covs: list[np.ndarray], ridge: float = 1e-6) -> np.ndarray:
    """Inverse square root of the pooled seed covariance.

    Pooling across programs gives one noise scale for the whole comparison, so `d_morph` is
    "how many ARDY-noise standard deviations apart" and is comparable between scenes.
    """
    valid = [c for c in seed_covs if np.all(np.isfinite(c)) and c.shape[0] == N_CHANNELS]
    S = np.mean(valid, axis=0) if valid else np.eye(N_CHANNELS)
    S = S + ridge * np.eye(N_CHANNELS) * max(1.0, float(np.trace(S)) / N_CHANNELS)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, ridge)
    return V @ np.diag(w ** -0.5) @ V.T


def d_morph(a: np.ndarray, b: np.ndarray, W: np.ndarray) -> float:
    """Whitened distance in units of ARDY seed-noise sigma."""
    return float(np.linalg.norm(W @ (np.asarray(a) - np.asarray(b))))


def stability(signatures: list[tuple]) -> float:
    """Fraction of seeds landing on the program's modal active set.

    A program that gives `duck` half the time and `duck+tuck` the other half is diverse but not
    controllably addressable, and this project needs addressable modes.
    """
    if not signatures:
        return 0.0
    from collections import Counter
    return Counter(signatures).most_common(1)[0][1] / len(signatures)


def epsilon_net(points: np.ndarray, W: np.ndarray, eps: float) -> list[int]:
    """Greedy eps-net indices: representatives mutually more than `eps` sigma apart.

    Deduplicating BEFORE counting is what stops a densely-sampled duck region from dominating
    a recall number simply because the oracle happened to sample it more finely.
    """
    keep: list[int] = []
    for i, p in enumerate(points):
        if all(d_morph(p, points[j], W) > eps for j in keep):
            keep.append(i)
    return keep


def morph_recall_at_k(reference: np.ndarray, proposed: np.ndarray, W: np.ndarray,
                      eps: float, K: int) -> float:
    """Fraction of an eps-net reference covered by the first K proposals."""
    if len(reference) == 0:
        return float("nan")
    p = proposed[:K]
    if len(p) == 0:
        return 0.0
    return float(np.mean([min(d_morph(r, q, W) for q in p) <= eps for r in reference]))
