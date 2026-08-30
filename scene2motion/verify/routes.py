"""Candidate routes for a scene, so route choice becomes a comparison rather than a default.

The Phase 1-3 planner returned ONE route per preference and the body layer coped with it.
Choosing a route for what the body will have to do needs alternatives to choose between, and
A* already has the hooks: `forbid_boxes` blocks rectangles, so forcing the path through a
particular lateral band near the obstacles enumerates genuinely different traversals rather
than perturbations of one.

Candidates are generated deterministically from the scene, so the same scene always yields the
same list in the same order and a selector's choice is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..planner import MODES, Plan, plan
from ..scenes import Scene
from .cost import evaluate as score

UPRIGHT_MODES = ("stand", "narrow", "step_over")
# Half-width of the lateral slot a candidate is forced through. The widest body mode is
# 0.378 m half-width, so 0.45 leaves a slot that a route can actually use rather than one
# that is infeasible for every candidate.
SLOT_HALF = 0.45


@dataclass
class Route:
    label: str
    plan: Plan
    lateral_y: float | None      # the band this candidate was pushed through, if any
    upright_only: bool

    @property
    def xy(self) -> np.ndarray:
        return self.plan.xy

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.plan.xy, axis=0), axis=1).sum())


def _beam_span(scene: Scene) -> tuple[float, float]:
    xs = [b for b in scene.boxes if b.label.startswith("partial_beam")]
    if not xs:
        return (0.0, 0.0)
    return (float(min(b.lo[0] for b in xs)) - 0.4, float(max(b.hi[0] for b in xs)) + 0.4)


def candidates(scene: Scene, k: int = 8, res: float = 0.05) -> list[Route]:
    """Up to `k` distinct routes, ordered deterministically.

    Half the budget is spent on upright-only variants and half on duck-permitted ones, each
    swept across lateral bands near the obstacles. Duplicates -- different constraints that
    happen to yield the same path -- are dropped, so `k` is a ceiling and not a quota.
    """
    x_lo, x_hi = _beam_span(scene)
    half = scene.meta.get("corridor_half", 1.2)
    n_lat = max(1, k // 2)
    # Lateral SLOTS the route is forced through while it is level with the obstacles. A
    # narrow forbidden band does nothing here -- the corridor is wide enough that A* simply
    # routes around it and returns the same path. Forbidding the COMPLEMENT of a slot is what
    # actually enumerates distinct traversals: each candidate must cross the obstacle stretch
    # at a chosen lateral position, and whether that costs a detour, a duck, or is infeasible
    # is exactly what the selector should be weighing.
    offsets = np.linspace(-half + SLOT_HALF, half - SLOT_HALF, n_lat) if n_lat > 1 \
        else np.array([0.0])

    out: list[Route] = []
    seen: set[bytes] = set()

    def add(label, p, y, upright):
        if not p.feasible or len(p.xy) < 2:
            return
        key = np.round(p.xy, 3).tobytes()
        if key in seen:
            return
        seen.add(key)
        out.append(Route(label=label, plan=p, lateral_y=y, upright_only=upright))

    for upright in (False, True):
        modes = UPRIGHT_MODES if upright else None
        tag = "upright" if upright else "any"
        add(f"{tag}:free", plan(scene, allow_modes=modes, res=res), None, upright)
        for y in offsets:
            far = half + 1.0
            boxes = [(x_lo, x_hi, -far, float(y) - SLOT_HALF),
                     (x_lo, x_hi, float(y) + SLOT_HALF, far)]
            add(f"{tag}:slot_y{y:+.2f}",
                plan(scene, allow_modes=modes, forbid_boxes=boxes, res=res), float(y), upright)
        if len(out) >= k:
            break
    return out[:k]


def route_score(route: Route, u: np.ndarray, s_m: np.ndarray, c_min_m: float,
                preference: str = "balanced"):
    """J for one (route, schedule) pair. Thin wrapper so callers stay honest about arguments."""
    return score(u, s_m, c_min_m, preference)
