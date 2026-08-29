"""The one demo scene, parameterised by the two dimensions the UI exposes.

`scenes.build_partial_beam` randomises beam position, goal distance and the beam's lateral
edge from a seed, which is right for a counterfactual ladder and wrong for a demo: moving a
slider must change ONE thing. This builds the same geometry with beam height and beam width
as explicit arguments and everything else fixed, so a user turning "beam height" down sees
the planner's decision change for that reason alone.

Geometry is identical in kind to the family the rest of the project measured on, so the
calibrated body modes in `outputs/body_modes.json` apply unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..scenes import WALL_T, Box, Scene, _side_walls

# Fixed so that only the two exposed sliders move.
BEAM_X = 4.0             # m, beam centre along the corridor
GOAL_X = 8.0             # m
CORRIDOR_HALF = 1.2      # m, half-width of the free corridor
BEAM_THICK_X = 0.12      # m, half-extent of the beam along travel
BEAM_THICK_Z = 0.125     # m, half-extent of the beam in height

DEFAULTS = {"beam_height": 1.00, "beam_width": 1.45, "n_beams": 1, "gap": 3.0}
PRESETS = ("single", "two")


@dataclass(frozen=True)
class BeamParams:
    """The two dimensions the UI exposes, plus everything needed to reproduce the scene."""

    beam_height: float          # m, underside of the beam above the floor
    beam_width: float           # m, how far the beam reaches across the corridor
    n_beams: int = 1            # 1 = Single Beam preset, 2 = Two Beams
    gap: float = DEFAULTS["gap"]  # m, spacing between beam centres when n_beams == 2

    def clamped(self) -> "BeamParams":
        span = 2 * CORRIDOR_HALF
        return BeamParams(
            n_beams=int(min(max(self.n_beams, 1), 2)),
            # Gap floor keeps the two beams from overlapping into one wide beam; the ceiling
            # keeps the second beam clear of the goal.
            gap=float(min(max(self.gap, 0.8), 5.5)),
            beam_height=float(min(max(self.beam_height, 0.60), 1.60)),
            # Leave at least 0.15 m of corridor so "around" is never trivially impossible,
            # and require at least 0.30 m of beam so "under" is never trivially avoidable.
            beam_width=float(min(max(self.beam_width, 0.30), span - 0.15)),
        )

    def key(self) -> str:
        k = f"h{self.beam_height:.3f}_w{self.beam_width:.3f}"
        return k if self.n_beams == 1 else f"{k}_n{self.n_beams}_g{self.gap:.2f}"

    @property
    def beam_xs(self) -> list[float]:
        return [BEAM_X + i * self.gap for i in range(self.n_beams)]


def build(params: BeamParams) -> Scene:
    """A partial beam spanning `beam_width` from the left wall, leaving a bypass on the right.

    The bypass is what makes the scene a genuine choice: under the beam is shorter, around it
    keeps the robot upright, and both reach the goal.
    """
    p = params.clamped()
    left = CORRIDOR_HALF + WALL_T / 2
    edge = left - p.beam_width                      # right-hand edge of the beam
    cy, hy = 0.5 * (edge + left), 0.5 * (left - edge)
    xs = p.beam_xs
    # The corridor grows with the beams so the goal stays a clear 3.5 m past the last one --
    # otherwise widening the gap would silently shorten the run-out and change two things at
    # once, which is the failure the fixed-geometry scene builder exists to prevent.
    goal_x = max(GOAL_X, xs[-1] + 3.5)
    boxes = _side_walls(CORRIDOR_HALF, -1.0, goal_x + 1.0)
    for i, bx in enumerate(xs):
        boxes.append(Box((bx, cy, p.beam_height + BEAM_THICK_Z),
                         (BEAM_THICK_X, hy, BEAM_THICK_Z),
                         "partial_beam" if i == 0 else f"partial_beam_{i}"))
    return Scene(
        scene_id=f"demo_partial_beam_{p.key()}",
        family="partial_beam",
        boxes=boxes,
        start=(0.0, 0.0),
        goal=(goal_x, 0.0),
        start_heading=0.0,
        param_name="beam_underside_height",
        param_value=p.beam_height,
        required_adaptation="duck" if p.beam_height < 1.07 else "none",
        bounds=(-1.0, goal_x + 1.0, -CORRIDOR_HALF - 0.3, CORRIDOR_HALF + 0.3),
        meta={"beam_x": BEAM_X, "beam_xs": xs, "n_beams": p.n_beams, "gap": p.gap,
              "beam_edge_y": edge, "corridor_half": CORRIDOR_HALF,
              "beam_width": p.beam_width, "bypass_width": edge + CORRIDOR_HALF,
              "goal_x": goal_x, "demo": True},
    )


def beam_footprints(scene: Scene) -> list[dict]:
    """Every beam box, for drawing. World-frame extents, ordered along the corridor."""
    out = [{"x_lo": float(b.lo[0]), "x_hi": float(b.hi[0]),
            "y_lo": float(b.lo[1]), "y_hi": float(b.hi[1]),
            "z_lo": float(b.lo[2]), "z_hi": float(b.hi[2])}
           for b in scene.boxes if b.label.startswith("partial_beam")]
    if not out:
        raise ValueError("scene has no partial_beam box")
    return sorted(out, key=lambda d: d["x_lo"])


def beam_footprint(scene: Scene) -> dict:
    """The FIRST beam. Kept so single-beam callers are unchanged."""
    return beam_footprints(scene)[0]


def save_default(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = BeamParams(**DEFAULTS).clamped()
    payload = {
        "name": "partial_beam",
        "description": "One overhead beam covering part of the corridor: duck under, or walk "
                       "around. Both reach the goal.",
        "defaults": DEFAULTS,
        "ranges": {"beam_height": [0.60, 1.60], "beam_width": [0.30, 2 * CORRIDOR_HALF - 0.15]},
        "fixed": {"beam_x": BEAM_X, "goal_x": GOAL_X, "corridor_half": CORRIDOR_HALF,
                  "start": [0.0, 0.0]},
        "scene": build(p).to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
