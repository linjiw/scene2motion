# Scene2Motion-G1: evaluation metrics.
#
# The rule that shapes this module: NEVER score a motion by the label that was requested.
# Coverage measured against commanded strategy names is a tautology — it counts what we asked
# for, not what the frozen prior delivered, and a decoder that silently ignored half its
# requests would score perfectly. Every signature here is read off the generated qpos.

from __future__ import annotations

import numpy as np

from .scenes import Scene
from .strategies import _choice_obstacles

# A pelvis this far below its standing height means the clip genuinely ducked rather than
# bobbing. Standing pelvis is 0.78 m; the shallowest calibrated duck requests 0.63 m.
DUCK_THRESHOLD = 0.68
# Lateral dead zone for "which side", in metres. Below this the body straddled the obstacle's
# centre line and the side is genuinely undefined rather than noisy.
SIDE_DEAD_ZONE = 0.12


def realised_signature(qpos: np.ndarray, scene: Scene) -> tuple[tuple[int, ...], int]:
    """(which side of each choice obstacle, did it duck) — measured from the motion.

    The homotopy half uses only obstacles a standing body fits past on BOTH sides, because a
    corridor-spanning beam offers no side to choose and counting lateral position there would
    turn one strategy into three.
    """
    homotopy = []
    for x, oy in _choice_obstacles(scene):
        i = int(np.argmin(np.abs(qpos[:, 0] - x)))
        lo, hi = max(0, i - 12), min(len(qpos), i + 13)
        d = float(np.mean(qpos[lo:hi, 1])) - oy
        homotopy.append(0 if abs(d) < SIDE_DEAD_ZONE else int(np.sign(d)))
    return tuple(homotopy), int(qpos[:, 2].min() < DUCK_THRESHOLD)


def distinct_signatures(qposes: list[np.ndarray], scene: Scene,
                        ok: list[bool] | None = None) -> int:
    """How many genuinely different verified traversals a set of clips contains.

    Only clips that passed verification count: a model that produces four distinct ways of
    colliding has not covered four strategies.
    """
    sigs = set()
    for i, q in enumerate(qposes):
        if ok is not None and not ok[i]:
            continue
        sigs.add(realised_signature(q, scene))
    return len(sigs)


def epsilon_net_count(vectors: np.ndarray, eps: float) -> int:
    """Greedy eps-net size: how many programs are more than `eps` apart.

    A diversity number that jitter cannot buy. `eps` should be set to the decoder's own noise
    floor — the distance below which two programs render to the same motion — so that
    resampling the same strategy does not count as covering a second one.
    """
    if len(vectors) == 0:
        return 0
    centres = [vectors[0]]
    for v in vectors[1:]:
        if min(float(np.linalg.norm(v - c)) for c in centres) > eps:
            centres.append(v)
    return len(centres)


def envelope_series(body, qpos: np.ndarray, normal=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """(T, 3) clearance envelope: top height, lateral half-width, pelvis height.

    Phase-robust, unlike raw joint positions: EXP-004's first version differenced collision
    primitive positions and got SNR < 1 because gait-phase drift moves a swinging wrist
    further than a 10 cm beam change moves the body.
    """
    out = np.empty((len(qpos), 3))
    for i, q in enumerate(qpos):
        out[i] = (body.top_height(q), body.half_width(q, normal), q[2])
    return out


def locality_ratio(env_a: np.ndarray, env_b: np.ndarray, qpos_a: np.ndarray,
                   obstacle_x: float, half_width: float = 0.7) -> float:
    """How much more the envelope differs inside the encounter window than outside it.

    1.0 is a uniformly smeared difference; higher is surgical. A strawman that holds the
    adaptation for the whole clip scores BELOW 1.0, which is the property that makes this
    metric able to tell the two apart.
    """
    n = min(len(env_a), len(env_b))
    d = np.abs(env_a[:n] - env_b[:n]).sum(axis=1)
    inside = np.abs(qpos_a[:n, 0] - obstacle_x) <= half_width
    if inside.sum() < 3 or (~inside).sum() < 3:
        return float("nan")
    return float(d[inside].mean() / max(d[~inside].mean(), 1e-6))
