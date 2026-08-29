"""Three user preferences, expressed as three argument sets to the SAME planner.

There is no new search here and there should not be. `planner.plan` already does A* over
(x, y, body mode) against a conformally calibrated envelope, and already exposes exactly the
two hooks a preference needs: `allow_modes` restricts the body volumes it may hold, and
`forbid_boxes` blocks regions of the floor. A preference is therefore a pair of arguments,
which is why this module is short and why the demo inherits every geometric guarantee the
planner already has -- including its refusal behaviour.

    Shortest Path       everything allowed -> the direct line, ducking under the beam
    Stay Upright        no duck modes      -> A* must find the bypass, upright
    Maximum Clearance   no duck modes AND the beam footprint dilated -> a wider berth
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..planner import LEAD_S, MODE_BY_NAME, MODES, Plan, plan
from ..scenes import Scene
from .scene_builder import beam_footprint

# Body modes that keep the head at standing height. Anything else is a duck.
UPRIGHT_MODES = ("stand", "narrow", "step_over")

# How far outside the beam footprint "Maximum Clearance" insists on staying. Sized against
# the geometry rather than picked round: the beam edge sits at y = -0.175 and a standing G1
# (half-width 0.378 m) can reach y = -0.822 before its shoulder meets the wall, so the
# dilation has to exceed the ~0.43 m the plain upright route already leaves or the preference
# is indistinguishable from "Stay Upright". 0.55 m bites without making the scene infeasible.
CLEARANCE_DILATION = 0.55  # m

# "Shortest Path" DISCOUNTS the body-mode cost rather than removing it, and the difference
# matters. Under the shipped costs a deep duck held for ~1.5 m outweighs a 0.5 m detour, so
# the planner walks around a beam it could duck under -- right for its objective, wrong for
# this label. But flattening every cost to 1.0 removes any reason to ever stand back up:
# switching modes costs MODE_SWITCH_COST, so the cheapest plan is then to duck at the start
# and stay down for all 8 m. Discounting keeps the ordering (standing is still preferred
# where it fits) while making route length dominant, which is what the label promises.
SHORTEST_COST_ALPHA = 0.15
SHORTEST_MODES = tuple(replace(m, cost=1.0 + SHORTEST_COST_ALPHA * (m.cost - 1.0))
                       for m in MODES)

PREFERENCES = ("shortest", "upright", "clearance")
PREFERENCE_LABEL = {
    "shortest": "Shortest Path",
    "upright": "Stay Upright",
    "clearance": "Maximum Clearance",
}


@dataclass
class Strategy:
    """A planned traversal plus the facts the decision panel is allowed to state."""

    preference: str
    plan: Plan
    feasible: bool
    path_length_m: float
    goes_under_beam: bool
    duck_required: bool
    min_top_clearance_m: float | None      # beam underside minus the tallest mode held under it
    deepest_mode: str
    duck_start_s: float | None
    duck_end_s: float | None = None
    duck_held_throughout: bool = False
    lead_s: float = LEAD_S
    note: str = ""
    refusal: dict | None = None
    modes_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "preference": self.preference,
            "preference_label": PREFERENCE_LABEL[self.preference],
            "feasible": self.feasible,
            "path_length_m": round(self.path_length_m, 3),
            "goes_under_beam": self.goes_under_beam,
            "duck_required": self.duck_required,
            "min_top_clearance_m": (None if self.min_top_clearance_m is None
                                    else round(self.min_top_clearance_m, 3)),
            "deepest_mode": self.deepest_mode,
            "duck_start_s": (None if self.duck_start_s is None
                             else round(self.duck_start_s, 2)),
            "duck_end_s": (None if self.duck_end_s is None else round(self.duck_end_s, 2)),
            "duck_held_throughout": self.duck_held_throughout,
            "lead_s": self.lead_s,
            "modes_used": self.modes_used,
            "note": self.note,
            "refusal": self.refusal,
        }


def _plan_args(preference: str, scene: Scene) -> dict:
    if preference == "shortest":
        return {"modes_override": SHORTEST_MODES}
    if preference == "upright":
        return {"allow_modes": UPRIGHT_MODES}
    if preference == "clearance":
        b = beam_footprint(scene)
        d = CLEARANCE_DILATION
        return {"allow_modes": UPRIGHT_MODES,
                "forbid_boxes": [(b["x_lo"] - d, b["x_hi"] + d, b["y_lo"] - d, b["y_hi"] + d)]}
    raise ValueError(f"unknown preference {preference!r}")


def _path_length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def _under_beam(xy: np.ndarray, b: dict) -> np.ndarray:
    return ((xy[:, 0] >= b["x_lo"]) & (xy[:, 0] <= b["x_hi"])
            & (xy[:, 1] >= b["y_lo"]) & (xy[:, 1] <= b["y_hi"]))


def evaluate(scene: Scene, preference: str, speed: float = 0.9,
             res: float = 0.05) -> Strategy:
    """Plan `scene` under `preference` and extract only facts the panel can defend."""
    p = plan(scene, "adaptive", res=res, **_plan_args(preference, scene))
    b = beam_footprint(scene)
    xy = np.asarray(p.xy, float).reshape(-1, 2) if p.xy is not None else np.zeros((0, 2))
    length = _path_length(xy)

    if not p.feasible or len(xy) == 0:
        return Strategy(preference=preference, plan=p, feasible=False, path_length_m=length,
                        goes_under_beam=False, duck_required=False, min_top_clearance_m=None,
                        deepest_mode="-", duck_start_s=None, refusal=p.refusal,
                        note=p.note or "no route under this preference")

    under = _under_beam(xy, b)
    modes = list(p.modes)
    names = [m.name for m in modes]
    ducking = [i for i, m in enumerate(modes) if m.name not in UPRIGHT_MODES]
    deepest = min(modes, key=lambda m: m.top).name if modes else "-"

    clearance = None
    if under.any():
        tops = [modes[i].top for i in np.where(under)[0] if i < len(modes)]
        if tops:
            clearance = float(b["z_lo"] - max(tops))

    # Report the duck SPAN, not just its start. A shallow duck can be cheaper to hold for the
    # whole path than to switch out of and back into (MODE_SWITCH_COST is 2.5 m-equivalent),
    # so "duck starts at 0.0 s" is sometimes the truth rather than a bug, and the panel should
    # say which case it is instead of implying a localised crouch that did not happen.
    duck_start_s = duck_end_s = None
    held_throughout = False
    if ducking:
        first, last = min(ducking), max(ducking)
        duck_start_s = float(_path_length(xy[:first + 1]) / max(speed, 1e-6))
        duck_end_s = float(_path_length(xy[:last + 1]) / max(speed, 1e-6))
        held_throughout = (first == 0 and last == len(modes) - 1)

    return Strategy(
        preference=preference, plan=p, feasible=True, path_length_m=length,
        goes_under_beam=bool(under.any()),
        duck_required=bool(ducking),
        min_top_clearance_m=clearance,
        deepest_mode=deepest,
        duck_start_s=duck_start_s, duck_end_s=duck_end_s,
        duck_held_throughout=held_throughout,
        modes_used=sorted(set(names), key=lambda n: -MODE_BY_NAME[n].top),
        note=p.note or "",
    )


def evaluate_all(scene: Scene, speed: float = 0.9) -> dict[str, Strategy]:
    return {k: evaluate(scene, k, speed=speed) for k in PREFERENCES}
