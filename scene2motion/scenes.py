# Scene2Motion-G1: procedural scene generator.
#
# Conventions
# -----------
# Scenes are authored in a WORLD frame that is Z-UP (the MuJoCo/robotics convention):
#   x = forward, y = lateral, z = height.
# ARDY's internal frame is Y-UP with the ground plane in XZ:
#   ardy_x = world_y, ardy_y = world_z, ardy_z = world_x.
# `world_to_ardy` / `ardy_to_world` in geom.py are the only place that mapping lives.
#
# Every scene is a start pose, a goal position, and a list of axis-aligned boxes. Scene
# families are parameterised so that a family + a swept parameter yields a COUNTERFACTUAL
# LADDER: the same scene with only the clearance-critical dimension changed. That ladder is
# the supervision signal for "geometric change -> necessary body adaptation".

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal

import numpy as np

# G1 nominal standing dimensions, measured from ARDY-generated free-walking motion
# (see experiments/exp000_body_envelope). Overridden by measurement at build time.
G1_NOMINAL_HEIGHT = 1.07  # m, top of the rig's highest joint while walking
G1_NOMINAL_HALF_WIDTH = 0.25  # m, max |lateral offset from root| over all joints
G1_PELVIS_HEIGHT = 0.78  # m

Adaptation = Literal["none", "duck", "narrow", "step_over", "detour", "sidle"]


@dataclass
class Box:
    """Axis-aligned box in the Z-up world frame, given as centre + half-extents."""

    center: tuple[float, float, float]
    half: tuple[float, float, float]
    label: str = ""

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.center) - np.asarray(self.half)

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.center) + np.asarray(self.half)

    def contains(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        return np.all((p >= self.lo) & (p <= self.hi), axis=-1)


@dataclass
class Scene:
    """A traversal problem: get the G1 from `start` to `goal` through `boxes`."""

    scene_id: str
    family: str
    boxes: list[Box]
    start: tuple[float, float]  # world (x, y) ground position
    goal: tuple[float, float]
    start_heading: float = 0.0  # radians, 0 = +x
    # The parameter that was swept to build the counterfactual ladder, and its value.
    param_name: str = ""
    param_value: float = 0.0
    # The adaptation the scene was DESIGNED to require. Ground truth for the
    # counterfactual-adaptation metric; never fed to any model.
    required_adaptation: Adaptation = "none"
    # Room bounds (x_min, x_max, y_min, y_max) for the occupancy grid.
    bounds: tuple[float, float, float, float] = (-1.0, 9.0, -3.0, 3.0)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["boxes"] = [asdict(b) for b in self.boxes]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        d = dict(d)
        d["boxes"] = [Box(**b) for b in d["boxes"]]
        d["start"] = tuple(d["start"])
        d["goal"] = tuple(d["goal"])
        d["bounds"] = tuple(d["bounds"])
        return Scene(**d)


def _sid(family: str, param: str, value: float, seed: int) -> str:
    h = hashlib.sha1(f"{family}|{param}|{value:.4f}|{seed}".encode()).hexdigest()[:8]
    return f"{family}_{param}{value:.2f}_{h}"


# --------------------------------------------------------------------------------------
# Scene families.
#
# Each builder takes the swept parameter and a seed for the nuisance randomisation
# (room width, obstacle x-offset, goal distance) that must NOT change the required
# adaptation. Holding the seed fixed while sweeping the parameter is what makes two
# scenes a matched counterfactual pair.
# --------------------------------------------------------------------------------------

WALL_T = 0.15  # wall thickness
ROOM_H = 2.6  # ceiling height


def _side_walls(y_half: float, x_lo: float, x_hi: float) -> list[Box]:
    cx, hx = 0.5 * (x_lo + x_hi), 0.5 * (x_hi - x_lo)
    return [
        Box((cx, +y_half + WALL_T / 2, ROOM_H / 2), (hx, WALL_T / 2, ROOM_H / 2), "wall_left"),
        Box((cx, -y_half - WALL_T / 2, ROOM_H / 2), (hx, WALL_T / 2, ROOM_H / 2), "wall_right"),
    ]


def build_overhead_beam(height: float, seed: int, corridor_half: float = 1.4) -> Scene:
    """A horizontal beam spanning the corridor at `height` (its UNDERSIDE).

    A 2D planner sees a free corridor: the beam does not touch the floor, so the pelvis
    path is unobstructed. The body must duck. This is the family where root-only planning
    is provably blind.
    """
    rng = np.random.default_rng(seed)
    bx = float(rng.uniform(3.6, 4.4))
    goal_x = float(rng.uniform(7.5, 8.5))
    beam_t = 0.25
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    boxes.append(
        Box((bx, 0.0, height + beam_t / 2), (0.12, corridor_half, beam_t / 2), "beam")
    )
    req: Adaptation = "duck" if height < G1_NOMINAL_HEIGHT else "none"
    return Scene(
        _sid("overhead_beam", "h", height, seed), "overhead_beam", boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="beam_underside_height", param_value=height,
        required_adaptation=req,
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"beam_x": bx, "corridor_half": corridor_half},
    )


def build_narrow_gap(width: float, seed: int) -> Scene:
    """A full-height wall across the corridor with a vertical slot of `width`.

    The pelvis fits through any slot wider than ~0.2 m, so a 2D point-planner declares
    success at widths that physically trap the shoulders and swinging arms. The body must
    narrow (arm tuck) and, below the shoulder width, turn (sidle).
    """
    rng = np.random.default_rng(seed)
    wx = float(rng.uniform(3.6, 4.4))
    goal_x = float(rng.uniform(7.5, 8.5))
    corridor_half = 1.4
    gap_y = float(rng.uniform(-0.35, 0.35))
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    # wall in two panels either side of the slot
    for sign, name in ((+1, "gapwall_left"), (-1, "gapwall_right")):
        inner = gap_y + sign * width / 2
        outer = sign * (corridor_half + WALL_T)
        if abs(outer - inner) < 1e-3:
            continue
        cy, hy = 0.5 * (inner + outer), 0.5 * abs(outer - inner)
        boxes.append(Box((wx, cy, ROOM_H / 2), (WALL_T / 2, hy, ROOM_H / 2), name))
    if width >= 2 * G1_NOMINAL_HALF_WIDTH:
        req: Adaptation = "none"
    elif width >= 0.34:  # still passable by tucking the arms; shoulders ~0.30 m wide
        req = "narrow"
    else:
        req = "sidle"
    return Scene(
        _sid("narrow_gap", "w", width, seed), "narrow_gap", boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="gap_width", param_value=width, required_adaptation=req,
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"wall_x": wx, "gap_y": gap_y, "corridor_half": corridor_half},
    )


def build_low_obstacle(height: float, seed: int) -> Scene:
    """A floor-standing box of `height` spanning the corridor: step over or detour."""
    rng = np.random.default_rng(seed)
    bx = float(rng.uniform(3.6, 4.4))
    goal_x = float(rng.uniform(7.5, 8.5))
    corridor_half = 1.4
    depth = float(rng.uniform(0.18, 0.32))
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    boxes.append(Box((bx, 0.0, height / 2), (depth / 2, corridor_half, height / 2), "low_box"))
    req: Adaptation = "none" if height < 0.06 else "step_over"
    return Scene(
        _sid("low_obstacle", "h", height, seed), "low_obstacle", boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="box_height", param_value=height, required_adaptation=req,
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"box_x": bx, "depth": depth, "corridor_half": corridor_half},
    )


def build_pillar(offset: float, seed: int) -> Scene:
    """A full-height pillar offset laterally by `offset` from the straight line.

    The CONTROL family: the required adaptation is a pure 2D detour that a root planner
    already solves. If a method fails here it is broken, not interesting. It is also the
    family with genuine bimodality (go left or go right), which is where a generative
    model should beat a regressor.
    """
    rng = np.random.default_rng(seed)
    px = float(rng.uniform(3.6, 4.4))
    goal_x = float(rng.uniform(7.5, 8.5))
    corridor_half = 1.6
    r = float(rng.uniform(0.22, 0.34))
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    boxes.append(Box((px, offset, ROOM_H / 2), (r, r, ROOM_H / 2), "pillar"))
    req: Adaptation = "detour" if abs(offset) < r + G1_NOMINAL_HALF_WIDTH else "none"
    return Scene(
        _sid("pillar", "off", offset, seed), "pillar", boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="pillar_lateral_offset", param_value=offset, required_adaptation=req,
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"pillar_x": px, "radius": r, "corridor_half": corridor_half},
    )


def build_beam_and_gap(height: float, seed: int) -> Scene:
    """Compound: a narrow gap followed by an overhead beam. Two adaptations in sequence.

    Tests whether adaptation stays LOCAL — the duck must not smear back over the squeeze.
    """
    rng = np.random.default_rng(seed)
    goal_x = float(rng.uniform(8.5, 9.5))
    corridor_half = 1.4
    wx, bx = 2.8, 5.6
    width = 0.62
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    for sign, name in ((+1, "gapwall_left"), (-1, "gapwall_right")):
        inner, outer = sign * width / 2, sign * (corridor_half + WALL_T)
        boxes.append(
            Box((wx, 0.5 * (inner + outer), ROOM_H / 2),
                (WALL_T / 2, 0.5 * abs(outer - inner), ROOM_H / 2), name)
        )
    boxes.append(Box((bx, 0.0, height + 0.125), (0.12, corridor_half, 0.125), "beam"))
    return Scene(
        _sid("beam_and_gap", "h", height, seed), "beam_and_gap", boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="beam_underside_height", param_value=height,
        required_adaptation="duck" if height < G1_NOMINAL_HEIGHT else "narrow",
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"wall_x": wx, "beam_x": bx, "gap_width": width, "corridor_half": corridor_half},
    )


def build_partial_beam(height: float, seed: int) -> Scene:
    """A beam covering only PART of the corridor width: duck under it, or walk around it.

    The ambiguity is deliberate and genuine -- both strategies are collision-free and reach
    the goal, and they are topologically distinct (one passes under the obstacle, the other
    to the side of it). Families like this are where a generative prior can show something
    a single deterministic policy structurally cannot: a DISTRIBUTION over traversal
    strategies rather than one behaviour per geometry.
    """
    rng = np.random.default_rng(seed)
    bx = float(rng.uniform(3.6, 4.4))
    goal_x = float(rng.uniform(7.5, 8.5))
    corridor_half = 1.2
    # The beam covers the corridor from the left wall to `edge`, leaving a bypass on the right
    # wide enough to walk through upright, but far enough off the direct line that going
    # around is a real detour rather than a free alternative.
    edge = float(rng.uniform(-0.25, 0.05))
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    cy, hy = 0.5 * (edge + corridor_half + WALL_T), 0.5 * (corridor_half + WALL_T - edge)
    boxes.append(Box((bx, cy, height + 0.125), (0.12, hy, 0.125), "partial_beam"))
    return Scene(
        _sid("partial_beam", "h", height, seed), "partial_beam", boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="beam_underside_height", param_value=height,
        required_adaptation="duck" if height < G1_NOMINAL_HEIGHT else "none",
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"beam_x": bx, "beam_edge_y": edge, "corridor_half": corridor_half,
              "bypass_width": corridor_half - edge},
    )


BUILDERS = {
    "partial_beam": build_partial_beam,
    "overhead_beam": build_overhead_beam,
    "narrow_gap": build_narrow_gap,
    "low_obstacle": build_low_obstacle,
    "pillar": build_pillar,
    "beam_and_gap": build_beam_and_gap,
}

# Counterfactual ladders: family -> the swept parameter values, ordered from
# "no adaptation needed" to "adaptation required". Adjacent rungs sharing a seed are a
# matched pair, so the difference between their motions isolates the effect of the
# geometry change alone.
LADDERS: dict[str, list[float]] = {
    # beam underside height: 1.30 clears the 1.07 m rig, 0.75 forces a deep crouch
    "overhead_beam": [1.30, 1.20, 1.10, 1.00, 0.90, 0.80],
    # gap width: 0.70 is free, 0.30 needs a turn
    "narrow_gap": [0.70, 0.60, 0.50, 0.42, 0.36, 0.30],
    # box height: 0.02 is free, 0.35 needs a real step-over
    "low_obstacle": [0.02, 0.08, 0.15, 0.22, 0.30, 0.38],
    # pillar lateral offset: 1.20 is out of the way, 0.0 is dead ahead
    "pillar": [1.20, 0.90, 0.60, 0.30, 0.10, 0.00],
    "beam_and_gap": [1.30, 1.10, 0.95, 0.85],
    # Low enough to force a real choice: standing under the beam is never an option, so
    # every rung genuinely admits both "duck under" and "go around".
    "partial_beam": [1.15, 1.05, 0.95, 0.85],
}


def build_suite(seeds_per_rung: int = 8, families: Iterable[str] | None = None) -> list[Scene]:
    """The full counterfactual suite: every family x every rung x `seeds_per_rung` seeds.

    Seeds are shared across the rungs of a family, so scenes with the same (family, seed)
    and different rungs form a matched ladder.
    """
    families = list(families or BUILDERS)
    scenes: list[Scene] = []
    for fam in families:
        for seed in range(seeds_per_rung):
            for val in LADDERS[fam]:
                scenes.append(BUILDERS[fam](val, seed))
    return scenes


def save_scenes(path: str, scenes: list[Scene]) -> None:
    with open(path, "w") as fh:
        for s in scenes:
            fh.write(json.dumps(s.to_dict()) + "\n")


def load_scenes(path: str) -> list[Scene]:
    with open(path) as fh:
        return [Scene.from_dict(json.loads(line)) for line in fh if line.strip()]


# --------------------------------------------------------------------------------------
# Randomised scenes for TRAINING.
#
# The six families above are counterfactual ladders: one clearance dimension swept with
# everything else pinned. That is exactly what evaluation needs and exactly what training
# must not be limited to -- a generator trained on six shapes learns six shapes.
#
# So training scenes draw their obstacle set and every dimension continuously. Evaluation
# stays on the ladders, and `is_heldout` keeps the two disjoint: any sampled scene that lands
# close to an evaluation configuration is rejected, so a train/eval overlap cannot be
# mistaken for generalisation.
# --------------------------------------------------------------------------------------

OBSTACLE_KINDS = ("beam", "partial_beam", "pillar", "slot", "floor_box")


def _rand_obstacle(rng: np.random.Generator, kind: str, x: float,
                   corridor_half: float) -> list[Box]:
    """One obstacle of `kind` centred at `x`, spanning what that kind spans."""
    if kind == "beam":
        h = float(rng.uniform(0.80, 1.45))
        return [Box((x, 0.0, h + 0.125), (0.12, corridor_half, 0.125), "beam")]
    if kind == "partial_beam":
        h = float(rng.uniform(0.80, 1.30))
        edge = float(rng.uniform(-0.45, 0.45))
        side = 1 if rng.random() < 0.5 else -1
        outer = side * (corridor_half + WALL_T)
        cy, hy = 0.5 * (edge + outer), 0.5 * abs(outer - edge)
        return [Box((x, cy, h + 0.125), (0.12, hy, 0.125), "partial_beam")]
    if kind == "pillar":
        r = float(rng.uniform(0.18, 0.38))
        y = float(rng.uniform(-corridor_half + r, corridor_half - r))
        return [Box((x, y, ROOM_H / 2), (r, r, ROOM_H / 2), "pillar")]
    if kind == "slot":
        w = float(rng.uniform(0.40, 1.10))
        y = float(rng.uniform(-0.4, 0.4))
        out = []
        for sign, nm in ((+1, "slotwall_left"), (-1, "slotwall_right")):
            inner, outer = y + sign * w / 2, sign * (corridor_half + WALL_T)
            if abs(outer - inner) < 1e-3:
                continue
            out.append(Box((x, 0.5 * (inner + outer), ROOM_H / 2),
                           (WALL_T / 2, 0.5 * abs(outer - inner), ROOM_H / 2), nm))
        return out
    h = float(rng.uniform(0.02, 0.30))
    return [Box((x, 0.0, h / 2), (float(rng.uniform(0.09, 0.16)), corridor_half, h / 2),
                "floor_box")]


def random_scene(seed: int, n_obstacles: tuple[int, int] = (1, 3)) -> Scene:
    """A randomised corridor traversal problem, for training only."""
    rng = np.random.default_rng(seed)
    corridor_half = float(rng.uniform(1.0, 1.8))
    goal_x = float(rng.uniform(6.5, 9.5))
    k = int(rng.integers(n_obstacles[0], n_obstacles[1] + 1))
    # Keep obstacles apart along travel: two adaptations inside one stride is not a
    # traversal problem, it is a wall.
    xs = np.sort(rng.uniform(2.2, goal_x - 1.4, size=k))
    while k > 1 and np.min(np.diff(xs)) < 1.8:
        xs = np.sort(rng.uniform(2.2, goal_x - 1.4, size=k))
    kinds = [OBSTACLE_KINDS[int(i)] for i in rng.integers(0, len(OBSTACLE_KINDS), size=k)]
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    for x, kind in zip(xs, kinds):
        boxes += _rand_obstacle(rng, kind, float(x), corridor_half)
    return Scene(
        f"rand_{seed:06d}", "random", boxes,
        start=(0.0, float(rng.uniform(-0.25, 0.25))),
        goal=(goal_x, float(rng.uniform(-0.35, 0.35))), start_heading=0.0,
        param_name="", param_value=0.0, required_adaptation="none",
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"corridor_half": corridor_half, "kinds": kinds,
              "obstacle_xs": [float(x) for x in xs]},
    )


def is_heldout(sc: Scene, tol: float = 0.04) -> bool:
    """True if a random scene sits too close to an evaluation ladder configuration.

    Only the clearance-critical dimension is compared, because that is the dimension the
    ladders sweep and therefore the one on which train/eval leakage would matter.
    """
    for b in sc.boxes:
        if b.label in ("beam", "partial_beam"):
            underside = b.center[2] - b.half[2]
            rungs = LADDERS["overhead_beam"] + LADDERS["partial_beam"]
            if any(abs(underside - r) < tol for r in rungs):
                return True
        elif b.label == "floor_box":
            if any(abs(2 * b.half[2] - r) < tol for r in LADDERS["low_obstacle"]):
                return True
    return False


def sample_train_scenes(n: int, seed0: int = 1_000_000) -> list[Scene]:
    """`n` random scenes, none of which collides with an evaluation ladder rung."""
    out, s = [], seed0
    while len(out) < n:
        sc = random_scene(s)
        s += 1
        if not is_heldout(sc):
            out.append(sc)
    return out
