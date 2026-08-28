# Scene2Motion-G1: strategy enumeration.
#
# The generator we are building is p(C | S, s, g), a DISTRIBUTION over constraint programs,
# so every training scene needs a SET of valid labels rather than one. This module is the
# operator that produces that set.
#
# EXP-003 enumerated strategies by hand: "forbid the bypass so it must duck" and "forbid every
# adaptation so it must go around". That works for one scene family and does not generalise.
# What actually distinguishes two strategies is a pair of discrete facts:
#
#   HOMOTOPY   which side of each obstacle the root path passes
#   MORPHOLOGY which body adaptations the traversal uses
#
# Together they form a signature. Two plans with the same signature are the same strategy no
# matter how their waypoints differ; two plans with different signatures are genuinely
# different ways through the scene. Enumeration is then: search, record the signature, forbid
# the corridor that solution used, search again, until nothing new appears.
#
# A candidate is NOT a label until it has been generated through the frozen prior and checked
# against the scene geometry. A plan that A* likes but the prior cannot execute would be a
# poisoned training target — it would teach the generator to propose motions that collide.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planner import MODE_BY_NAME, Plan, plan
from .scenes import Scene

STAND_ONLY = ("stand",)

# Mode sets to search under. Restricting to standing forces the planner to find a route that
# needs no body adaptation, which is how the "go around" alternative to a duck is discovered;
# without it A* simply takes whichever is cheaper and the alternative is never seen.
MODE_SETS: tuple[tuple[str, ...] | None, ...] = (None, STAND_ONLY)

# Half-width of the corridor excluded after a solution is found, in metres. Based on the
# STANDING envelope rather than the widest mode: duck_deep's half-width is an outlier from a
# single seed flinging its arms out, and using it would exclude most of the corridor and make
# every alternative infeasible.
EXCLUSION_HALF_WIDTH = MODE_BY_NAME["stand"].half_width + 0.05
# How far along travel the exclusion extends either side of an obstacle. The alternative route
# only has to differ WHERE THE CHOICE IS MADE; forcing it to differ everywhere would rule out
# routes that rejoin the centre line, which is most of them.
EXCLUSION_HALF_DEPTH = 1.0


@dataclass(frozen=True)
class Strategy:
    """One distinct way through a scene: a plan plus the signature that makes it distinct."""

    name: str
    plan: Plan
    homotopy: tuple[int, ...]      # sign of lateral offset at each obstacle, in x order
    morphology: tuple[str, ...]    # sorted non-standing modes used
    detour_m: float                # path length beyond the straight start->goal distance

    @property
    def signature(self) -> tuple:
        return (self.homotopy, self.morphology)


def _choice_obstacles(scene: Scene) -> list[tuple[float, float]]:
    """(x, y) of the obstacles a path genuinely has to pick a SIDE of.

    Two exclusions, both necessary or the strategy count inflates with things that are not
    choices:
      - side walls: passing a wall on one side is not a decision.
      - obstacles that do not leave room for a standing body on BOTH sides. A beam spanning
        the whole corridor has no "left" and "right" -- you must go under it, and where you
        happen to be laterally while doing so is a nuisance parameter. Counting it as a
        homotopy class made one duck look like three strategies.
    """
    need = 2 * MODE_BY_NAME["stand"].half_width
    y_lo, y_hi = scene.bounds[2], scene.bounds[3]
    out = []
    for b in scene.boxes:
        if b.label.startswith("wall_"):
            continue
        if (y_hi - b.hi[1]) >= need and (b.lo[1] - y_lo) >= need:
            out.append((float(b.center[0]), float(b.center[1])))
    return sorted(set(out))


def _lateral_at(p: Plan, x: float) -> float:
    """Mean lateral (world y) position of the path as it passes `x`."""
    if len(p.xy) == 0:
        return 0.0
    d = np.abs(p.xy[:, 0] - x)
    k = np.argsort(d)[:max(3, len(p.xy) // 20)]
    return float(p.xy[k, 1].mean())


def signature_of(p: Plan, scene: Scene, dead_zone: float = 0.12) -> tuple[tuple[int, ...],
                                                                          tuple[str, ...]]:
    """(homotopy, morphology) for a plan.

    `dead_zone` keeps paths that thread almost exactly through an obstacle's centre line from
    flipping sign on numerical noise and inflating the strategy count.
    """
    homotopy = []
    for x, oy in _choice_obstacles(scene):
        # Side is measured relative to the OBSTACLE, not to the world centre line: a pillar
        # offset to +0.30 is passed on the left at y > 0.30, not at y > 0.
        d = _lateral_at(p, x) - oy
        homotopy.append(0 if abs(d) < dead_zone else int(np.sign(d)))
    morphology = tuple(sorted({m.name for m in p.modes} - {"stand"}))
    return tuple(homotopy), morphology


def enumerate_strategies(scene: Scene, max_k: int = 6, res: float = 0.05) -> list[Strategy]:
    """All distinct strategies for `scene`, by iterated diverse re-planning.

    Returns them ordered by plan cost proxy (path length), so `[0]` is what a plain
    cost-minimising planner would have produced and everything after it is an alternative
    that would otherwise never be seen.
    """
    straight = float(np.hypot(scene.goal[0] - scene.start[0], scene.goal[1] - scene.start[1]))
    found: dict[tuple, Strategy] = {}
    # Exclusions are placed at EVERY obstacle (so the re-plan is really pushed elsewhere),
    # even though only some obstacles carry a homotopy bit in the signature.
    xs = [float(b.center[0]) for b in scene.boxes if not b.label.startswith("wall_")]

    for modes in MODE_SETS:
        forbidden: list[tuple[float, float, float, float]] = []
        for _ in range(max_k):
            p = plan(scene, "adaptive", res=res, allow_modes=modes,
                     forbid_boxes=forbidden or None)
            if not p.feasible:
                break
            homotopy, morphology = signature_of(p, scene)
            sig = (homotopy, morphology)
            if sig not in found:
                tag = ("around" if morphology == () else "+".join(morphology))
                side = "".join("LCR"[h + 1] for h in homotopy) or "-"
                found[sig] = Strategy(f"{tag}:{side}", p, homotopy, morphology,
                                      p.length - straight)
            # Exclude the corridor this solution used at the obstacles, so the next search
            # has to commit to a different side rather than returning a jittered twin.
            if not xs:
                break
            for x in xs:
                y = _lateral_at(p, x)
                forbidden.append((x - EXCLUSION_HALF_DEPTH, x + EXCLUSION_HALF_DEPTH,
                                  y - EXCLUSION_HALF_WIDTH, y + EXCLUSION_HALF_WIDTH))

    return sorted(found.values(), key=lambda s: s.plan.length)


def summarise(strats: list[Strategy]) -> str:
    return " | ".join(f"{s.name} {s.plan.length:.2f}m (+{s.detour_m:.2f})" for s in strats)
