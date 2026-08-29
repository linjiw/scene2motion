"""The model's input: what the scene looks like ALONG a given route.

Not a BEV grid. `scene_encoding.py` renders the scene in the chord frame, which is right for a
model that must also choose the route; here the route is already chosen and the only question
left is what the body must do along it. So the representation is route-local and
one-dimensional: 64 samples in arc length, each carrying what the body decision at that point
actually depends on.

Every channel is computed from the SCENE and the ROUTE only. The planner's mode choice is the
label and never appears here -- if it leaked in, the model would learn to copy an answer it is
supposed to predict.
"""

from __future__ import annotations

import numpy as np

from ..planner import PELVIS_BAND
from ..scenes import Scene

N_SAMPLES = 64
FOOTPRINT_R = 0.38      # m, standing G1 half-width; what "above the robot" means laterally
CEILING = 2.6           # m, room height: the clearance value when nothing is overhead
MAX_LATERAL = 2.0       # m, cap on the left/right free-distance channels
LOOKAHEAD_S = 3.0       # s, horizon for the time-to-restriction channel

CHANNELS = ("overhead_clearance", "floor_height", "left_clearance", "right_clearance",
            "curvature", "dist_to_goal", "time_to_restriction")
N_CHANNELS = len(CHANNELS)


def _arc_resample(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to `n` points uniform in ARC LENGTH, not in index."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return np.repeat(xy.reshape(1, 2) if len(xy) else np.zeros((1, 2)), n, axis=0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(want, s, xy[:, 0]), np.interp(want, s, xy[:, 1])], -1)


def _overhead_and_floor(scene: Scene, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest obstacle underside above each point, and the highest floor obstacle under it.

    A box counts as overhead if its footprint comes within FOOTPRINT_R of the route point --
    the robot is not a point, and a beam it passes beside at 5 cm still hits its shoulder.
    """
    over = np.full(len(pts), CEILING)
    floor = np.zeros(len(pts))
    for b in scene.boxes:
        lo, hi = b.lo, b.hi
        dx = np.maximum(np.maximum(lo[0] - pts[:, 0], 0.0), pts[:, 0] - hi[0])
        dy = np.maximum(np.maximum(lo[1] - pts[:, 1], 0.0), pts[:, 1] - hi[1])
        near = np.hypot(dx, dy) <= FOOTPRINT_R
        if not near.any():
            continue
        if lo[2] > PELVIS_BAND[0]:            # its underside is above knee height: overhead
            over[near] = np.minimum(over[near], lo[2])
        if lo[2] <= 0.05:                      # sits on the floor: something to step over
            floor[near] = np.maximum(floor[near], hi[2])
    return over, floor


def _lateral_free(scene: Scene, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Free distance to the left (+y) and right (-y) at pelvis height, capped."""
    left = np.full(len(pts), MAX_LATERAL)
    right = np.full(len(pts), MAX_LATERAL)
    for b in scene.boxes:
        lo, hi = b.lo, b.hi
        if hi[2] < PELVIS_BAND[0] or lo[2] > PELVIS_BAND[1]:
            continue                            # not in the band the torso occupies
        span = (pts[:, 0] >= lo[0] - FOOTPRINT_R) & (pts[:, 0] <= hi[0] + FOOTPRINT_R)
        if not span.any():
            continue
        dl = lo[1] - pts[:, 1]                  # obstacle to the left
        m = span & (dl >= 0)
        left[m] = np.minimum(left[m], dl[m])
        dr = pts[:, 1] - hi[1]                  # obstacle to the right
        m = span & (dr >= 0)
        right[m] = np.minimum(right[m], dr[m])
    return np.clip(left, 0, MAX_LATERAL), np.clip(right, 0, MAX_LATERAL)


def _curvature(pts: np.ndarray) -> np.ndarray:
    d1 = np.gradient(pts, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.power(np.sum(d1 ** 2, axis=1), 1.5) + 1e-9
    return np.clip(num / den, 0.0, 10.0)


def profile(scene: Scene, xy: np.ndarray, speed: float = 0.9,
            n: int = N_SAMPLES) -> np.ndarray:
    """(n, N_CHANNELS) route-local scene profile, in physical units."""
    pts = _arc_resample(xy, n)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1]) if s[-1] > 0 else 1.0

    over, floor = _overhead_and_floor(scene, pts)
    left, right = _lateral_free(scene, pts)
    curv = _curvature(pts)
    to_goal = total - s

    # Time until the route next passes under a restriction low enough to matter. This is what
    # makes anticipation learnable: the label ducks BEFORE the beam, and nothing else in the
    # profile at that moment says a beam is coming.
    restricted = over < 1.35
    t_restrict = np.full(n, LOOKAHEAD_S)
    if restricted.any():
        idx = np.where(restricted)[0]
        for i in range(n):
            ahead = idx[idx >= i]
            if len(ahead):
                t_restrict[i] = min(LOOKAHEAD_S, (s[ahead[0]] - s[i]) / max(speed, 1e-6))
            else:
                t_restrict[i] = LOOKAHEAD_S
    return np.stack([over, floor, left, right, curv, to_goal, t_restrict], -1)


# Per-channel normalisation, fixed rather than fitted so a model trained on one split is
# directly comparable to one trained on another.
NORM_SCALE = np.array([CEILING, 0.5, MAX_LATERAL, MAX_LATERAL, 2.0, 10.0, LOOKAHEAD_S])


def normalise(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float32) / NORM_SCALE.astype(np.float32)
