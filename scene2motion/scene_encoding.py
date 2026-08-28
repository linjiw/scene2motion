# Scene2Motion-G1: the scene as a tensor.
#
# Rendered in the CHORD FRAME (start -> goal), the same frame the constraint program's lateral
# knots live in. That is not a convenience: it makes both the input and the output invariant to
# where the scene sits in the world and which way it points, so the model never has to spend
# capacity learning that a corridor rotated by 30 degrees is the same corridor.
#
# The channels are chosen from what EXP-000 through EXP-002 established actually matters, and
# nothing else:
#
#   0  pelvis-band occupancy   what a 2D (x, y, theta) planner would see. Kept because the
#                              PELVIS baseline is defined by it, and because a model that
#                              cannot reproduce that baseline is not learning navigation.
#   1  free height             the lowest obstacle underside above the floor, normalised. This
#                              is the channel a 2D planner throws away and the one the whole
#                              overhead-clearance result depends on.
#   2  floor obstacle height   what must be stepped over rather than ducked under.
#   3  lateral free width      how much room there is across the corridor at each point,
#                              which is what the tuck axis trades against.
#
# A full 3D voxel grid would carry all of this and much more, at ~30x the memory, and the extra
# capacity would be spent on the 2.6 m of empty air above the robot. Four BEV channels retain
# what the task uses.

from __future__ import annotations

import numpy as np

from .scenes import Scene

# Chord-frame extent. Along-track runs a little past both endpoints so approach and departure
# are visible; across-track covers the widest corridor the generator produces (1.8 m half).
N_ALONG, N_ACROSS = 64, 32
ALONG_LO, ALONG_HI = -0.10, 1.10        # fractions of the chord
ACROSS_LO, ACROSS_HI = -2.0, 2.0        # metres
N_CHANNELS = 4
MAX_HEIGHT = 2.0                        # normaliser for the height channels
ROBOT_TOP = 1.34                        # standing envelope, EXP-001d 90 % bound
PELVIS_BAND = (0.60, 0.95)


def chord_frame(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(origin, tangent, normal, length) of the start -> goal chord."""
    s = np.asarray(scene.start, float)
    g = np.asarray(scene.goal, float)
    d = g - s
    L = float(np.linalg.norm(d))
    t = d / max(L, 1e-9)
    return s, t, np.array([-t[1], t[0]]), L


def encode_scene(scene: Scene) -> np.ndarray:
    """(4, N_ALONG, N_ACROSS) float32 BEV of `scene` in its own chord frame."""
    origin, tangent, normal, L = chord_frame(scene)
    u = np.linspace(ALONG_LO, ALONG_HI, N_ALONG)
    v = np.linspace(ACROSS_LO, ACROSS_HI, N_ACROSS)
    U, V = np.meshgrid(u, v, indexing="ij")
    # world position of every cell
    P = origin[None, None, :] + U[..., None] * L * tangent + V[..., None] * normal

    pelvis = np.zeros((N_ALONG, N_ACROSS), np.float32)
    ceiling = np.full((N_ALONG, N_ACROSS), MAX_HEIGHT, np.float32)
    floor = np.zeros((N_ALONG, N_ACROSS), np.float32)
    blocked_full = np.zeros((N_ALONG, N_ACROSS), bool)

    for b in scene.boxes:
        lo, hi = b.lo, b.hi
        inside = ((P[..., 0] >= lo[0]) & (P[..., 0] <= hi[0])
                  & (P[..., 1] >= lo[1]) & (P[..., 1] <= hi[1]))
        if not inside.any():
            continue
        z0, z1 = float(lo[2]), float(hi[2])
        if z1 > PELVIS_BAND[0] and z0 < PELVIS_BAND[1]:
            pelvis[inside] = 1.0
        if z0 > 0.02:                       # hangs above the floor: it defines a ceiling
            ceiling[inside] = np.minimum(ceiling[inside], z0)
        else:                               # rests on the floor: it must be stepped over
            floor[inside] = np.maximum(floor[inside], z1)
        if z0 < ROBOT_TOP and z1 > 0.02:
            blocked_full[inside] = True

    # Lateral free width: how far a cell is from the nearest blocking cell across the
    # corridor, in metres, capped. Computed per along-track row because that is the direction
    # a gap actually constrains.
    dv = (ACROSS_HI - ACROSS_LO) / (N_ACROSS - 1)
    free_w = np.zeros((N_ALONG, N_ACROSS), np.float32)
    for i in range(N_ALONG):
        row = blocked_full[i]
        if not row.any():
            free_w[i] = 1.0
            continue
        idx = np.flatnonzero(row)
        for j in range(N_ACROSS):
            if row[j]:
                continue
            left = idx[idx < j]
            right = idx[idx > j]
            dl = (j - left[-1]) * dv if len(left) else 2.0
            dr = (right[0] - j) * dv if len(right) else 2.0
            free_w[i, j] = min(min(dl, dr) / 1.0, 1.0)

    return np.stack([
        pelvis,
        np.clip(ceiling / MAX_HEIGHT, 0, 1),
        np.clip(floor / 0.5, 0, 1),
        free_w,
    ]).astype(np.float32)


def scene_scalars(scene: Scene) -> np.ndarray:
    """Conditioning that is not spatial: chord length and corridor half-width."""
    _, _, _, L = chord_frame(scene)
    half = float(scene.meta.get("corridor_half", 1.4))
    return np.array([L / 10.0, half / 2.0], np.float32)
