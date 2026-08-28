# Scene2Motion-G1: planners.
#
# The scientific point of this module is the CONTRAST between three planners that differ
# only in what body volume they treat as blocking:
#
#   PELVIS      a cell is blocked only if something occupies the pelvis band. This is what
#               ordinary (x, y, theta) humanoid navigation reduces to. An overhead beam is
#               invisible to it, so it walks the robot's head straight into the beam.
#   STANDING    a cell is blocked if anything occupies the standing body volume. Safe, but
#               it declares INFEASIBLE every scene whose only route is under something —
#               a corridor-spanning beam has no way around.
#   ADAPTIVE    A* over (x, y, BODY MODE). Modes are body envelopes the frozen prior can
#               actually reach, measured rather than assumed (see BodyMode). Traversing a
#               low-clearance cell is allowed but costs more, and the resulting mode
#               schedule converts directly into ARDY constraint channels.
#
# ADAPTIVE is the V0 oracle: no learning, just search in the augmented configuration space.
# It is the baseline a learned V1 scene->constraint model has to beat, and the gap between
# PELVIS and ADAPTIVE is the size of the problem this research is about.

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .constraints import ConstraintSpec
from .scenes import Scene


@dataclass(frozen=True)
class BodyMode:
    """A body envelope the prior can hold, as an upright cylinder plus a cost.

    `half_width`, `top` and `max_step` are MEASURED from ARDY generations under the
    corresponding constraint request (experiments/exp001*, aggregated by
    experiments/derive_modes.py), using the robot's own MuJoCo collision primitives rather
    than the ARDY joint set — G1's head reaches 23 cm above its highest ARDY joint, so a
    joint-derived envelope is 23 cm too optimistic.

    Aggregation is WORST CASE over seeds. A planner that assumes the average clearance
    routes the robot through gaps it clears only half the time.
    """

    name: str
    half_width: float    # m, lateral half-extent about the pelvis
    top: float           # m, top of the highest collision primitive
    pelvis_y: float      # m, pelvis height this mode requests
    cost: float          # per-step multiplier; standing is 1.0
    tuck: float = 0.0    # arm-tuck strength in [0, 1); 0 = arms free
    sidle_deg: float = 0.0
    max_step: float = 0.0  # m, tallest floor obstacle this mode steps over
    lift: float = 0.0      # m, requested swing-foot lift that achieves max_step


_MODES_JSON = Path(__file__).resolve().parents[1] / "outputs" / "body_modes.json"

# Fallback used only if the calibration has not been run; kept obviously conservative so a
# missing calibration degrades to "plans little" rather than "plans confidently and wrong".
_FALLBACK = (
    BodyMode("stand", 0.34, 1.30, 0.78, 1.00),
    BodyMode("duck", 0.34, 1.09, 0.53, 1.45),
)


def load_modes(path: Path = _MODES_JSON) -> tuple[BodyMode, ...]:
    if not path.exists():
        return _FALLBACK
    raw = json.loads(path.read_text())["modes"]
    keep = {f.name for f in fields(BodyMode)}
    return tuple(BodyMode(**{k: v for k, v in m.items() if k in keep}) for m in raw)


MODES: tuple[BodyMode, ...] = load_modes()
MODE_BY_NAME = {m.name: m for m in MODES}

# Cost of switching body mode between adjacent cells. Without it the planner flickers
# between modes frame to frame, which is both physically silly and destroys the locality
# the counterfactual metric measures.
MODE_SWITCH_COST = 2.5

PELVIS_BAND = (0.60, 0.95)  # what an (x, y, theta) navigation planner implicitly protects


def _rect_dist_xy(px: np.ndarray, py: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance in the ground plane from points to an axis-aligned rectangle (0 inside)."""
    dx = np.maximum(np.maximum(lo[0] - px, 0.0), px - hi[0])
    dy = np.maximum(np.maximum(lo[1] - py, 0.0), py - hi[1])
    return np.hypot(dx, dy)


class Grid:
    """Occupancy over a scene, queryable per body mode."""

    def __init__(self, scene: Scene, res: float = 0.05):
        self.scene, self.res = scene, res
        x0, x1, y0, y1 = scene.bounds
        self.nx = int(np.ceil((x1 - x0) / res)) + 1
        self.ny = int(np.ceil((y1 - y0) / res)) + 1
        self.x0, self.y0 = x0, y0
        xs = x0 + res * np.arange(self.nx)
        ys = y0 + res * np.arange(self.ny)
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        # For each box: ground-plane distance from every cell, plus its vertical span.
        self._dist = []
        self._zspan = []
        for b in scene.boxes:
            lo, hi = b.lo, b.hi
            self._dist.append(_rect_dist_xy(self.X, self.Y, lo[:2], hi[:2]))
            self._zspan.append((float(lo[2]), float(hi[2])))
        self._blocked_cache: dict[str, np.ndarray] = {}

    def world(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + i * self.res, self.y0 + j * self.res

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res)))

    def blocked(self, mode: BodyMode) -> np.ndarray:
        """(nx, ny) bool: is a mode-`mode` body at this cell in collision with the scene?"""
        if mode.name in self._blocked_cache:
            return self._blocked_cache[mode.name]
        out = np.zeros((self.nx, self.ny), dtype=bool)
        for d, (zlo, zhi) in zip(self._dist, self._zspan):
            # A box blocks this mode unless it is entirely above the body's top, or low
            # enough that the mode's measured step-over clears it.
            if zlo >= mode.top or zhi <= mode.max_step:
                continue
            out |= d <= mode.half_width
        self._blocked_cache[mode.name] = out
        return out

    def blocked_pelvis_band(self, radius: float = 0.30) -> np.ndarray:
        """What a pelvis-only (x, y, theta) planner sees."""
        out = np.zeros((self.nx, self.ny), dtype=bool)
        for d, (zlo, zhi) in zip(self._dist, self._zspan):
            if zhi <= PELVIS_BAND[0] or zlo >= PELVIS_BAND[1]:
                continue
            out |= d <= radius
        return out

    def free_height(self, radius: float = 0.30) -> np.ndarray:
        """Lowest obstacle underside above the floor within `radius` of each cell.

        This is the map an overhead-aware planner needs and a 2D one throws away.
        """
        out = np.full((self.nx, self.ny), np.inf)
        for d, (zlo, zhi) in zip(self._dist, self._zspan):
            near = d <= radius
            if zlo <= 0.02:  # floor-standing: it does not define a ceiling, it blocks
                continue
            out = np.where(near, np.minimum(out, zlo), out)
        return out


_NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]


def _astar(grid: Grid, start: tuple[int, int], goal: tuple[int, int],
           modes: tuple[BodyMode, ...]) -> list[tuple[int, int, int]] | None:
    """A* over (i, j, mode_index). Returns the cell/mode path, or None if unreachable."""
    blocked = [grid.blocked(m) for m in modes]
    if all(b[start] for b in blocked) or all(b[goal] for b in blocked):
        return None
    res = grid.res
    cheapest = min(m.cost for m in modes)

    def h(i, j):  # admissible: straight-line at the cheapest possible per-metre cost
        return cheapest * res * np.hypot(i - goal[0], j - goal[1])

    starts = [(mi, m) for mi, m in enumerate(modes) if not blocked[mi][start]]
    openq = [(h(*start), 0.0, (start[0], start[1], mi)) for mi, _ in starts]
    heapq.heapify(openq)
    g = {s[2]: 0.0 for s in openq}
    came: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if gc > g.get(cur, np.inf):
            continue
        i, j, mi = cur
        if (i, j) == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for di, dj, w in _NB:
            ni, nj = i + di, j + dj
            if not (0 <= ni < grid.nx and 0 <= nj < grid.ny):
                continue
            for nmi, nm in enumerate(modes):
                if blocked[nmi][ni, nj]:
                    continue
                step = w * res * nm.cost + (MODE_SWITCH_COST * res if nmi != mi else 0.0)
                nxt = (ni, nj, nmi)
                ng = gc + step
                if ng < g.get(nxt, np.inf):
                    g[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(openq, (ng + h(ni, nj), ng, nxt))
    return None


@dataclass
class Plan:
    """A planned traversal: a ground path plus the body mode held at each waypoint."""

    xy: np.ndarray          # (N, 2) world ground path
    modes: list[BodyMode]   # length N
    planner: str
    feasible: bool
    note: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum()) if len(self.xy) > 1 else 0.0


def plan(scene: Scene, planner: str = "adaptive", res: float = 0.05,
         allow_modes: tuple[str, ...] | None = None,
         forbid_y: tuple[float, float] | list[tuple[float, float]] | None = None,
         forbid_boxes: list[tuple[float, float, float, float]] | None = None) -> Plan:
    """Plan a traversal of `scene` under one of the three body-volume assumptions.

    `allow_modes`, `forbid_y` and `forbid_boxes` exist to ENUMERATE strategies rather than
    take the single cheapest one. `forbid_y` blocks a lateral band over the WHOLE corridor;
    `forbid_boxes` blocks (x_lo, x_hi, y_lo, y_hi) rectangles, which is what diverse
    re-planning needs -- a global band through the middle makes every alternative infeasible
    when start and goal both sit on the centre line, because the path can no longer leave and
    return to it. Forbidding every duck mode forces the planner to find a route around an
    obstacle; forbidding the lateral band that contains the bypass forces it to go under.
    Whether both then survive generation and collision checking is the multimodality
    question -- a single deterministic controller has only ever one answer to give.
    """
    grid = Grid(scene, res)
    s = grid.cell(*scene.start)
    t = grid.cell(*scene.goal)
    if planner == "pelvis":
        modes = (MODE_BY_NAME["stand"],)
        # The defining blindness: only the pelvis band counts as an obstacle.
        grid._blocked_cache["stand"] = grid.blocked_pelvis_band(radius=0.30)
    elif planner == "standing":
        modes = (MODE_BY_NAME["stand"],)
    elif planner == "adaptive":
        modes = MODES
    else:
        raise ValueError(f"unknown planner {planner!r}")

    if allow_modes is not None:
        modes = tuple(m for m in modes if m.name in allow_modes)
        if not modes:
            return Plan(np.zeros((0, 2)), [], planner, False, "no modes left after filtering")

    if forbid_y is not None:
        bands = [forbid_y] if isinstance(forbid_y[0], (int, float)) else list(forbid_y)
        band = np.zeros_like(grid.X, dtype=bool)
        for lo, hi in bands:
            band |= (grid.Y >= lo) & (grid.Y <= hi)
        # Never forbid the start or the goal cell itself, or every re-plan is trivially
        # infeasible once a band happens to cover an endpoint.
        for wx, wy in (scene.start, scene.goal):
            i, j = grid.cell(wx, wy)
            band[max(0, i - 3):i + 4, max(0, j - 3):j + 4] = False
        for m in modes:
            grid._blocked_cache[m.name] = grid.blocked(m) | band

    if forbid_boxes:
        box = np.zeros_like(grid.X, dtype=bool)
        for xl, xh, yl, yh in forbid_boxes:
            box |= (grid.X >= xl) & (grid.X <= xh) & (grid.Y >= yl) & (grid.Y <= yh)
        for wx, wy in (scene.start, scene.goal):
            i, j = grid.cell(wx, wy)
            box[max(0, i - 3):i + 4, max(0, j - 3):j + 4] = False
        for m in modes:
            grid._blocked_cache[m.name] = grid.blocked(m) | box

    path = _astar(grid, s, t, modes)
    if path is None:
        return Plan(np.zeros((0, 2)), [], planner, False, "no path in this planner's C-space")
    xy = np.array([grid.world(i, j) for i, j, _ in path])
    used = [modes[mi] for _, _, mi in path]
    return Plan(xy, used, planner, True)


# ---------------------------------------------------------------------------------------
# Plan -> ARDY constraint request
# ---------------------------------------------------------------------------------------

def _resample(xy: np.ndarray, modes: list[BodyMode], n: int) -> tuple[np.ndarray, list[BodyMode]]:
    """Arc-length resample the path to `n` samples, carrying the mode of the nearest node."""
    if len(xy) < 2:
        return np.repeat(xy, n, axis=0), [modes[0]] * n
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(want, s, xy[:, 0]), np.interp(want, s, xy[:, 1])], -1)
    idx = np.clip(np.searchsorted(s, want), 0, len(modes) - 1)
    return out, [modes[i] for i in idx]


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with edge padding; keeps endpoints put."""
    if win <= 1:
        return a
    k = np.ones(win) / win
    pad = win // 2
    p = np.pad(a, ((pad, pad),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    if a.ndim == 1:
        return np.convolve(p, k, mode="valid")[: len(a)]
    return np.stack([np.convolve(p[:, c], k, mode="valid")[: len(a)] for c in range(a.shape[1])], -1)

def plan_to_path_spec(p: Plan, fps: float, speed: float = 0.9,
                      duration: float | None = None) -> ConstraintSpec:
    """The PATH-ONLY request: where to go and which way to face, no body adaptation.

    Generating this first serves two purposes. It is the matched control the counterfactual
    metric differences against, and its joint trajectories and foot-contact schedule are
    what the adapted request is built from — so the adaptation is always expressed as a
    local edit of a motion the prior already produced for this exact path, rather than a
    pose invented from outside its manifold.
    """
    T = _n_frames(p, fps, speed, duration)
    xy, _ = _resample(p.xy, p.modes, T)
    root_xz, heading = _path_channels(xy, fps)
    return ConstraintSpec(root_xz=root_xz, heading=heading,
                          root_y=np.full(T, MODE_BY_NAME["stand"].pelvis_y),
                          first_heading=float(heading[0]))


def _dilate(modes: list[BodyMode], w: int) -> list[BodyMode]:
    """Spread each adaptation `w` frames either side, keeping the most demanding one.

    "Most demanding" is ordered by (lowest top, largest tuck, largest lift) -- the same
    ranking the planner used to pay for the mode -- so dilation never silently relaxes an
    adaptation, it only starts it earlier and ends it later.
    """
    if w <= 0:
        return modes
    n = len(modes)
    rank = [(-m.top, m.tuck, m.lift) for m in modes]
    out = []
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        out.append(modes[max(range(lo, hi), key=lambda k: rank[k])])
    return out


def plan_to_spec(p: Plan, fps: float, nominal: dict, joint_names: list[str],
                 speed: float = 0.9, duration: float | None = None,
                 lead_s: float = 0.8) -> ConstraintSpec:
    """The ADAPTED request: the mode schedule rendered into ARDY constraint channels.

    Each mode contributes the channel it was calibrated on:
      pelvis_y   -> root_y_pos                       (duck)
      tuck       -> arm joint-position targets       (narrow)
      sidle_deg  -> heading offset from path tangent (narrow, when calibrated with one)
      lift       -> swing-leg joint-position targets (step_over)

    The schedule is smoothed before it is sent. The planner emits a step function over
    modes; handed a discontinuity the prior resolves it as a stumble, which then shows up
    as a collision or a tracking failure and gets blamed on the method.
    """
    T = _n_frames(p, fps, speed, duration)
    xy, modes = _resample(p.xy, p.modes, T)
    root_xz, heading = _path_channels(xy, fps)

    # ANTICIPATION. A* labels only the cells physically under the obstacle, so a naive
    # rendering asks the prior to crouch during the ~0.3 s it spends beneath a beam. The
    # body cannot change envelope that fast, and the resulting motion clips the obstacle on
    # the way in. Dilating each adaptation backwards and forwards in time by `lead_s` is
    # what a person does: you duck BEFORE the beam and stand up after it. Without this the
    # planner is correct in space and wrong in time.
    modes = _dilate(modes, int(round(lead_s * fps)))

    ry_win = max(3, int(0.6 * fps) | 1)
    root_y = _smooth(np.array([m.pelvis_y for m in modes]), ry_win)
    tuck = _smooth(np.array([m.tuck for m in modes]), ry_win)
    lift = _smooth(np.array([m.lift for m in modes]), ry_win)
    sidle = _smooth(np.array([np.deg2rad(m.sidle_deg) for m in modes]), ry_win)
    heading = heading + sidle

    idx = {n: i for i, n in enumerate(joint_names)}
    nom_j = nominal["posed_joints"]           # (Tn, J, 3) ardy frame
    nom_r = nominal["smooth_root_pos"]        # (Tn, 3)
    contacts = nominal["foot_contacts"]       # (Tn, 4) L_heel L_toe R_heel R_toe
    Tn = len(nom_j)

    def nf(f: np.ndarray) -> np.ndarray:      # nominal frame index for a request frame
        return np.clip(f, 0, Tn - 1)

    # frame -> {joint index: world-height offset to add}
    lifts: dict[int, dict[int, float]] = {}
    if lift.max() > 1e-3:
        for side, pair in (("left", (0, 2)), ("right", (2, 4))):
            legs = [idx[f"{side}_{j}"] for j in
                    ("knee_skel", "ankle_pitch_skel", "ankle_roll_skel", "toe_base")]
            air = ~contacts[:, pair[0]:pair[1]].any(-1)
            for f in np.where(lift > 0.02)[0]:
                if air[nf(np.array(f))]:      # only lift a leg that is actually airborne
                    for j in legs:
                        lifts.setdefault(int(f), {})[j] = float(lift[f])

    arm_joints = [idx[n] for n in
                  ("left_shoulder_roll_skel", "left_elbow_skel", "left_wrist_roll_skel",
                   "left_hand_roll_skel", "right_shoulder_roll_skel", "right_elbow_skel",
                   "right_wrist_roll_skel", "right_hand_roll_skel")]
    tuck_frames = np.where(tuck > 0.05)[0]

    pos_frames = pos_joints = pos_targets = None
    want = sorted(set(tuck_frames.tolist()) | set(lifts))
    if want:
        step = max(1, int(0.12 * fps))        # ~8 targets/s: enough to steer, sparse enough
        want = np.array(want[::step]) if len(want) > step else np.array(want)
        joints = np.array(sorted(set(arm_joints) | {j for d in lifts.values() for j in d}))
        k = nf(want)
        off = nom_j[k][:, joints, :] - nom_r[k][:, None, :]
        # Tucking shrinks the arm offset along BOTH ground axes: once the body turns, the
        # fore-aft swing arc is what sets the corridor width (EXP-001b), so shrinking only
        # the lateral component cannot narrow a sidling robot.
        shrink = np.ones((len(want), len(joints)))
        arm_cols = np.array([i for i, j in enumerate(joints) if j in set(arm_joints)])
        if len(arm_cols):
            shrink[:, arm_cols] = (1.0 - tuck[want])[:, None]
        height = nom_j[k][:, joints, 1].copy()
        for i, f in enumerate(want):
            for j, dy in lifts.get(int(f), {}).items():
                col = int(np.where(joints == j)[0][0])
                height[i, col] += dy
        pos_frames, pos_joints = want, joints
        pos_targets = np.stack([
            root_xz[want][:, None, 0] + off[:, :, 0] * shrink,
            height,
            root_xz[want][:, None, 1] + off[:, :, 2] * shrink,
        ], -1)

    return ConstraintSpec(root_xz=root_xz, heading=heading, root_y=root_y,
                          pos_frames=pos_frames, pos_joints=pos_joints,
                          pos_targets=pos_targets, first_heading=float(heading[0]))


def _n_frames(p: Plan, fps: float, speed: float, duration: float | None) -> int:
    T = int(round((duration if duration is not None else p.length / speed) * fps))
    return max(T, int(2 * fps))


def _path_channels(xy: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Ground path and heading in the ARDY frame (ardy_x = world_y, ardy_z = world_x)."""
    root_xz = _smooth(np.stack([xy[:, 1], xy[:, 0]], -1), max(3, int(0.4 * fps) | 1))
    d = np.gradient(root_xz, axis=0)
    heading = _smooth(np.unwrap(np.arctan2(d[:, 0], d[:, 1])), max(3, int(0.4 * fps) | 1))
    return root_xz, heading
