# Scene2Motion-G1: BODY-ENUMERATE -- classical enumeration of the feasible BODY set at a
# FIXED route.  This module is the strong classical baseline EXP-005g gates against.
#
# THE OBJECT
# ----------
# For a scene S and a route r fixed by the shipped planner,
#
#     F_B(S, r) = { b : ARDY(r, b) is collision-free and reaches the goal }
#
# `planner._astar` returns exactly ONE point of F_B -- the cost-minimal mode assignment on a
# 7-mode lattice (planner.py:296-311).  EXP-005e measured that this is where the enumerator's
# incompleteness lives: 93.6 % of its misses are "new BODY, same route", and they are
# cost-competitive (median 1.005x the best route).  That indicts the CORRIDOR-EXCLUSION rule
# in strategies.py:143-148, which operates on spatial regions and therefore cannot see a
# different body along one corridor.  It does NOT indict classical search, and a reviewer will
# say so.  This module is the answer to that reviewer.
#
# Four baselines, then their composition:
#
#   A  enumerate_kbest       exact k-best over the fixed-route body chain
#   B  enumerate_nogood      iterated body no-good cuts + whole-route MODE-FLOOR
#   C  enumerate_weight_sweep decomposed objective, preference simplex, ties enumerated
#   D  refine_continuous     tube-constrained continuous refinement, verified through ARDY
#   *  enumerate_composite   k-best skeletons seeded into continuous refinement
#
# Every entry point has the signature (scene, route_plan, K, runner, ...) and returns
# `list[ConstraintProgram]`.  Every one holds the ROUTE FIXED: `prog.lat` is byte-identical to
# `encode(route_plan, ...).lat` in every candidate, which is what makes ONE matched
# neutral-body control valid for all of them (morphology.py:23 requires the paired delta be
# taken on the same route and the same seed).
#
# ---------------------------------------------------------------------------------------
# WHAT THE ADVERSARIAL REVIEWS CHANGED, AND WHY
# ---------------------------------------------------------------------------------------
# Four enumerators were proposed and each was reviewed as a strawman.  Where the review named
# a stronger version, that version is what is implemented here.  The substitutions, with the
# measurement that forced each:
#
# A.  Proposed: quotiented k-best over (x, y, mode), grouped by route, with `pads_m=(0.0,)`
#     and a `purposeful_axes` relevance filter on by default.
#     Measured against it: |pool| == 1 on 61.9 % of the suite (K-curve flat by construction);
#     at most ONE slot axis varied per scene; the headline MorphRecall was scored against the
#     enumerator's own pool (2.38 members), and widening the reference with a `pads_m` sweep
#     the same module already accepted took recall@8 from 1.000 to 0.684.
#     ADOPTED: timing is a first-class enumerated axis, not a hook (`TIMING_PADS_M`,
#     `TIMING_SHIFTS_M`); windows may be SPLIT so a body can vary within one interaction;
#     relevance filtering is moved OUT of the enumerator into a reported attribute
#     (`BodySkeleton.pareto_minimal`) so it cannot pre-empt EXP-005h; and no route k-best runs
#     at all, because the route is fixed by the object's definition, which also removes the
#     measured 0.155 m of route drift that invalidated the matched control.
#
# B.  Proposed: no-good cuts over per-site mode sets, where a site is a run where the CHEAPEST
#     mode cannot be held.
#     Measured against it: that definition can only propose a body where geometry forces one,
#     so 100 % of its own misses (52 "no_site_route" + 71 "off_site" + ZERO "at_site") are
#     outside its candidate space; a 25-line MODE-FLOOR arm using the shipped
#     `plan(..., allow_modes=)` keyword scored recall@8 0.773 vs 0.476 at the same CPU cost.
#     ADOPTED: `mode_floor_candidates` -- "you must be at least this adapted EVERYWHERE" -- is
#     part of baseline B, and `_slots_from_runs` tiles a long hold across consecutive slots
#     rather than letting `SHALF_MAX = 0.35` silently clip it to 70 % of the clip (a clipped
#     route-long duck+tuck was GPU-measured at 6.6 cm of penetration, i.e. NOT in F_B).
#     Also adopted: the mode table is built through `envelope.get_envelope()`, never by reading
#     `outputs/exp001d/envelope.json` directly -- the raw rows are non-monotone and up to 3.9 cm
#     NARROWER than the shipped certificate at dip 0.25-0.35, against a 4 cm body margin.
#
# C.  Proposed: sweep weights on a decomposed objective and take the argmin.
#     Measured against it: an LP over the reachable set found 68.9 % of the within-route
#     reference is WEAKLY supported -- it only ever TIES for optimal -- so no single-argmin
#     sweep reaches it at any budget, and direct enumeration beat the sweep 0.238 vs 0.106 at
#     recall@8 while being 4.6x cheaper.
#     ADOPTED: every scalarisation is solved with an exact top-`t` chain enumeration rather
#     than an argmin (`topk_per_weight`), which is precisely the tie mass the LP identified;
#     `supportedness` ships as a diagnostic so the ceiling is reported rather than assumed.
#
# D.  Proposed: MAP-Elites over the 20-D slot block with a hardcoded seed-noise sigma measured
#     from one deep-duck program per scene, scoring single realisations, and a uniform-Sobol
#     ablation.
#     Measured against it: the repo's own EXP-005f residuals (1296 paired rows) give a per-seed
#     sd 6-10x larger on t_lead / t_duration, under which 96 % of SAME-PROGRAM, different-seed
#     pairs read as ">2 sigma apart" -- i.e. regenerating one program eight times would have
#     scored "K=8 distinct bodies"; and the honest cheap ablation (tube-constrained stratified
#     sampling) was 15/15 feasible at 16 ARDY calls where uniform Sobol was 5/63.
#     ADOPTED: `SEED_SIGMA_PRIOR` comes from `outputs/exp005f/per_scene.json` and is only a
#     PRIOR -- the gate re-measures it in-run; candidates are scored on seed MEANS with
#     `Stability` reported; the archive is keyed and whitened on the SAME channel set; and the
#     tube-constrained stratified arm is the seeding stage of D rather than a strawman.
#
# ---------------------------------------------------------------------------------------
# FACTS THIS MODULE MAY NOT CONTRADICT
# ---------------------------------------------------------------------------------------
#  * Generation ~0.2 s/clip batched, MuJoCo check ~0.07 s/clip.  ARDY CALLS ARE THE BUDGET.
#  * `half_width` RISES with dip (0.378 -> 0.510): ducking makes G1 WIDER (EXP-001d).  So
#    "buy headroom" and "buy lateral margin" genuinely fight, and a deeper duck can need a
#    laterally wider corridor than a shallower one.
#  * `step_over.max_step` is 0.028 m and is NOT conformally calibrated.  The lift axis moves
#    the descriptor but clears almost nothing; it is never interpolated into a step-height
#    curve here.
#  * EXP-005f retired two axes on noise grounds: TUCK never fires (achieved ~5 cm against
#    16-18 cm of seed noise on width) and foot ORDER is never controllable.  Both are still
#    ENUMERATED -- a proposer must be free to ask -- but `CHANNEL_SETS` lets the gate report
#    coverage restricted to the channels the prior can actually deliver, and the tuck axis is
#    never counted as evidence of diversity on its own.
#  * 128-scene suite: adaptive feasibility 65.6 % (84/128); every feasible plan is
#    collision-free; all failures are refusals.

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .envelope import Envelope, get_envelope
from .morphology import (N_CHANNELS, Interaction, active_set, d_morph, envelope_series,
                         matched_delta, raw_descriptor, stability)
from .planner import (LEAD_S, MODE_SWITCH_COST, MODES, Plan, _rect_dist_xy, _smooth,
                      plan_to_path_spec)
from .program import (ACTIVE, DIP_MAX, LIFT_MAX, M_SLOT, N_AX, N_LAT, NOMINAL_PELVIS,
                      SHALF_MAX, SMOOTH_S, TUCK_MAX, ConstraintProgram, decode, encode)

__all__ = [
    "BodySkeleton", "BodyCandidate", "RouteTube", "RouteContext", "BodyEvaluator",
    "Evaluation", "route_context", "route_tube", "body_lattice", "j_body",
    "predicted_descriptor", "select_k", "enumerate_kbest", "enumerate_nogood",
    "enumerate_weight_sweep", "refine_continuous", "enumerate_composite",
    "reference_random_restart", "null_seed_baseline", "supportedness", "BASELINES",
    "SEED_SIGMA_PRIOR", "CHANNEL_SETS", "PROMPT",
]

PROMPT = "A person walks forward."
SPEED, GOAL_TOL, MAX_DURATION = 0.9, 0.5, 14.0        # matches strategies.validate_strategies

# Above this a floor-standing box is a wall, not a step (planner.py:181 records the same trap
# producing "step_height short by 257 cm" for a 2.6 m wall panel).
MAX_PLAUSIBLE_STEP = 0.60

# ---------------------------------------------------------------------------------------
# Noise scale.  A PRIOR only -- the gate re-measures it from the clips it actually generates.
# ---------------------------------------------------------------------------------------
# Pooled within-program per-seed standard deviation over EXP-005f's 1296 paired residual rows
# (outputs/exp005f/per_scene.json: 6 scenes x 36 programs x 6 seeds, every delta paired against
# a neutral-body control on the same route and the same seed).  This is the number that decides
# whether two candidates are different bodies or the same body twice, so it must never be
# estimated from a single strong program: doing that under-reads t_lead by 9.5x, and under a
# 9.5x-optimistic sigma 96 % of SAME-PROGRAM seed pairs read as ">2 sigma apart".
SEED_SIGMA_PRIOR = np.array([0.0386, 0.0602, 0.0600, 0.0848, 0.1036, 0.7270, 1.1708, 0.1608])

# Channel subsets coverage may be reported over.  Pre-register both; the primary number is
# "all" (the guidance's descriptor), and "controllable" is the ablation that answers "did the
# baseline look bad only because of channels no proposer can address?".
CHANNEL_SETS = {
    "all": tuple(range(N_CHANNELS)),
    # the channels the ConstraintProgram has an axis for: dip -> dh_top, tuck -> dw_*,
    # lift -> dz_foot_*.  There is no heading channel (program.py:31-34) and t_lead/t_duration
    # are only indirectly addressable through slot extent.
    "controllable": (0, 1, 2, 3, 4),
    # the two axes EXP-005f left alive after the noise floor: duck and lift.
    "envelope": (0, 3, 4),
}


# =======================================================================================
# 1.  The fixed route, and what it DEMANDS of the body
# =======================================================================================

def route_program(scene, route_plan: Plan, fps: float, speed: float = SPEED,
                  duration: float | None = None) -> ConstraintProgram:
    """`encode(route_plan)` -- the route as a program.  Its `lat` is THE fixed route."""
    return encode(route_plan, scene, fps, speed)


def n_frames(route_plan: Plan, fps: float, speed: float = SPEED,
             max_duration: float = MAX_DURATION) -> int:
    return min(int(max_duration * fps),
               max(int(2 * fps), int(round(route_plan.length / max(speed, 0.1) * fps))))


def decoded_route(prog: ConstraintProgram, scene, T: int) -> np.ndarray:
    """(T, 2) world route `decode` will actually render.  Mirrors program.py:203-212.

    The enumeration MUST run on this, not on `route_plan.xy`.  Measured on the 128-scene
    suite: the 24-knot smooth curve corner-cuts the A* grid polyline, and on 28 of 40 scenes
    the decoded route is stand-blocked at frames where the plan is not (7 frames on
    `partial_beam`, 2 on `pillar`).  Certifying a body against the polyline and then rendering
    the curve certifies a route ARDY never walks.  The residual is under 5 cm -- about one
    grid cell -- and iterative knot refitting does NOT remove it (measured: 7 blocked frames
    before and after), because it is a parameterisation difference and not a fitting error.
    """
    s, g = np.asarray(scene.start, float), np.asarray(scene.goal, float)
    d = g - s
    L = float(np.linalg.norm(d))
    e = d / max(L, 1e-9)
    normal = np.array([-e[1], e[0]])
    knots = np.linspace(0, 1, N_LAT)
    lat = np.concatenate([[0.0], prog.lat, [0.0]])
    a = np.linspace(0, 1, 2048)
    dense = s + np.outer(a * L, e) + np.outer(np.interp(a, knots, lat), normal)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, arc[-1], T)
    return np.stack([np.interp(want, arc, dense[:, 0]),
                     np.interp(want, arc, dense[:, 1])], -1)


@dataclass(eq=False)
class RouteTube:
    """Body-clearance tube along the fixed route, in ARC LENGTH (the guidance's x(s))."""

    xy: np.ndarray            # (T, 2) world
    u: np.ndarray             # (T,) arc-length fraction -- the coordinate slot[:, 0] lives in
    s: np.ndarray             # (T,) metres of travel
    length: float
    c_top: np.ndarray         # (T,) lowest overhead underside above the body (inf if none)
    c_lat: np.ndarray         # (T,) ground-plane distance to the nearest blocking box
    h_floor: np.ndarray       # (T,) tallest floor obstacle under the body
    dip_req: np.ndarray       # (T,) minimum dip that clears the overhead
    tuck_req: np.ndarray      # (T,) minimum tuck that fits the channel at that dip
    lift_req: np.ndarray      # (T,) minimum lift the calibrated step-over could ask for
    windows: list[tuple[int, int]]      # contiguous frame runs where STANDING is blocked
    obstacle_x: list[float]             # x of every non-wall box, deduplicated
    box_dist: np.ndarray                # (T, n_boxes) ground-plane distance, per frame
    box_zlo: np.ndarray                 # (n_boxes,)
    box_zhi: np.ndarray                 # (n_boxes,)

    def blocked(self, body: "Body") -> np.ndarray:
        """(T,) bool -- EXACTLY `planner.Grid.blocked` (planner.py:141-144), per frame.

        Feasibility must be this predicate and not a comparison against the scalar
        `c_top`/`c_lat` summaries.  Those are computed at the frame's REQUIRED dip, so just
        outside a beam they report a lateral clearance measured against the beam itself -- and
        a body that ducks under the beam does not need lateral clearance from it at all.  With
        the scalar test the enumerator concluded that upright walking was infeasible in open
        corridor two frames before the beam and made every free-segment option a duck; with
        this one the free segments are free, which is what they are.
        """
        if self.box_dist.size == 0:
            return np.zeros(len(self.u), bool)
        counts = (self.box_zlo < body.top) & (self.box_zhi > body.max_step)
        if not counts.any():
            return np.zeros(len(self.u), bool)
        return (self.box_dist[:, counts] <= body.half_width).any(axis=1)

    def clearance(self, body: "Body", a: int, b: int) -> tuple[float, float]:
        """(vertical, lateral) margin this body has over [a, b].  inf where nothing applies."""
        over = self.box_zlo > 0.02
        head = float(np.min(np.where(over, self.box_zlo, np.inf)) - body.top) if over.any() \
            else float("inf")
        counts = (self.box_zlo < body.top) & (self.box_zhi > body.max_step)
        if counts.any():
            lat = float(np.min(self.box_dist[a:b + 1][:, counts]) - body.half_width)
        else:
            lat = float("inf")
        return head, lat

    def window_u(self, w: tuple[int, int]) -> tuple[float, float]:
        return float(self.u[w[0]]), float(self.u[w[1]])


def route_tube(scene, prog: ConstraintProgram, T: int, *, merge_gap_m: float = 0.5,
               max_windows: int = M_SLOT, envelope: Envelope | None = None) -> RouteTube:
    """Read the clearance tube off the scene along the DECODED route.

    Boxes are used RAW, exactly as `planner.Grid` does (planner.py:116-119): the 4 cm geometry
    margin is already inside the calibrated envelope, so inflating here double-counts it.
    """
    env = envelope or get_envelope()
    xy = decoded_route(prog, scene, T)
    px, py = xy[:, 0], xy[:, 1]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1) if T > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s[-1]) if T > 1 else 0.0

    dists, zspans = [], []
    for b in scene.boxes:
        dists.append(_rect_dist_xy(px, py, b.lo[:2], b.hi[:2]))
        zspans.append((float(b.lo[2]), float(b.hi[2])))

    h_floor = np.zeros(T)
    for d, (zlo, zhi) in zip(dists, zspans):
        if zlo <= 0.02 and zhi <= MAX_PLAUSIBLE_STEP:
            h_floor = np.where(d <= float(env.half_width(0.0, 0.0)),
                               np.maximum(h_floor, zhi), h_floor)
    # The lift axis is a step function at the single measured point (body_modes.json:
    # lift 0.45 -> max_step 0.028 m, "NOT yet conformally calibrated").  Interpolating a
    # step-height curve out of one point would invent a capability.
    lift_req = np.where(h_floor > 1e-6, LIFT_STEP_POINT, 0.0)
    step_have = np.where(lift_req >= LIFT_STEP_POINT, STEP_AT_POINT, 0.0)

    # Overhead, at the width the duck itself needs: a deeper duck is WIDER, so probing at the
    # standing half-width under-reads which boxes are overhead.  Three fixed-point passes.
    dip_req = np.zeros(T)
    c_top = np.full(T, np.inf)
    top0 = float(env.top(0.0))
    for _ in range(3):
        w = np.asarray(env.half_width(dip_req, 0.0), float)
        ct = np.full(T, np.inf)
        for d, (zlo, zhi) in zip(dists, zspans):
            if zlo <= 0.02:
                continue
            ct = np.where(d <= w, np.minimum(ct, zlo), ct)
        c_top = ct
        need = np.isfinite(ct) & (ct < top0)
        dip_req = np.clip(np.where(need, _invert_top(env, np.minimum(ct, top0)), 0.0),
                          0.0, DIP_MAX)

    c_lat = np.full(T, np.inf)
    top_at = np.asarray(env.top(dip_req), float)
    for d, (zlo, zhi) in zip(dists, zspans):
        overlaps = (zlo < top_at) & (zhi > step_have)
        c_lat = np.where(overlaps, np.minimum(c_lat, d), c_lat)

    hw_at = np.asarray(env.half_width(dip_req, 0.0), float)
    need = np.isfinite(c_lat) & (c_lat < hw_at)
    tuck_req = np.zeros(T)
    if need.any():
        tuck_req = np.where(need, _invert_hw(env, dip_req, np.minimum(c_lat, hw_at)), 0.0)

    # A demand window is where the STANDING body is blocked -- the same predicate A* faces
    # (planner.py:141-144), evaluated on the decoded route.
    D = np.stack(dists, -1) if dists else np.zeros((T, 0))
    zlo_a = np.array([z[0] for z in zspans])
    zhi_a = np.array([z[1] for z in zspans])
    stand_body = Body(0.0, 0.0, 0.0, float(env.top(0.0)), float(env.half_width(0.0, 0.0)),
                      0.0, 1.0)
    if D.size:
        counts = (zlo_a < stand_body.top) & (zhi_a > stand_body.max_step)
        act = ((D[:, counts] <= stand_body.half_width).any(axis=1) if counts.any()
               else np.zeros(T, bool))
    else:
        act = np.zeros(T, bool)
    windows = _runs(act, s, merge_gap_m)
    if len(windows) > max_windows:                     # keep the longest demands
        windows = sorted(sorted(windows, key=lambda r: -(r[1] - r[0]))[:max_windows])
    obs = sorted({round(float(b.center[0]), 3) for b in scene.boxes
                  if not b.label.startswith("wall_")})
    return RouteTube(xy, np.linspace(0, 1, T), s, length, c_top, c_lat, h_floor,
                     dip_req, tuck_req, lift_req, windows, obs, D, zlo_a, zhi_a)


LIFT_STEP_POINT, STEP_AT_POINT = 0.45, 0.028


def _invert_top(env: Envelope, ceiling: np.ndarray) -> np.ndarray:
    """Smallest dip whose bounded top clears `ceiling`.  `top` DECREASES, so invert on -top."""
    return np.interp(-np.asarray(ceiling, float), -env.top_dip, env.dips)


def _invert_hw(env: Envelope, dip: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Smallest tuck whose bounded half-width at this dip fits `want`."""
    grid = np.linspace(0.0, TUCK_MAX, 35)
    out = np.zeros_like(np.asarray(dip, float))
    for i, (d, w) in enumerate(zip(np.atleast_1d(dip), np.atleast_1d(want))):
        hw = np.asarray(env.half_width(float(d), grid), float)
        ok = np.flatnonzero(hw <= w + 1e-9)
        out[i] = float(grid[ok[0]]) if len(ok) else TUCK_MAX
    return out


def _runs(mask: np.ndarray, s: np.ndarray, merge_gap_m: float) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    out: list[list[int]] = []
    for t in idx:
        if out and s[t] - s[out[-1][1]] <= merge_gap_m:
            out[-1][1] = int(t)
        else:
            out.append([int(t), int(t)])
    return [(a, b) for a, b in out]


# =======================================================================================
# 2.  The calibrated body lattice
# =======================================================================================

# Excess per-metre cost, fitted to outputs/body_modes.json so a dense-lattice schedule is
# comparable with A*'s on the planner's own scale:
#   dip  0/.15/.25/.35/.50 -> 1.00/1.20/1.45/1.80/2.40   (1 + 1.0 d + 3.6 d^2, max err 0.03)
#   tuck 0.85 -> 1.60, lift 0.45 -> 1.70
# Cross terms are additive and monotone -- the conservative choice, since a duck-and-tuck is
# then never made to look cheaper than either alone.  They are extrapolation, not measurement.
COST_DIP = (1.0, 3.6)
COST_TUCK = 0.60 / TUCK_MAX
COST_LIFT = 0.70 / LIFT_STEP_POINT

DEFAULT_DIPS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_TUCKS = (0.0, 0.40, 0.60, 0.85)
DEFAULT_LIFTS = (0.0, 0.45)


@dataclass(frozen=True)
class Body:
    """One point of the calibrated (dip, tuck, lift) lattice, with its certified envelope."""

    dip: float
    tuck: float
    lift: float
    top: float
    half_width: float
    max_step: float
    cost: float

    @property
    def axes(self) -> tuple[float, float, float]:
        return (self.dip, self.tuck, self.lift)

    @property
    def effort(self) -> float:
        return self.cost - 1.0

    @property
    def name(self) -> str:
        return f"d{self.dip:.2f}t{self.tuck:.2f}l{self.lift:.2f}"


def mode_cost(dip: float, tuck: float, lift: float) -> float:
    return (1.0 + COST_DIP[0] * dip + COST_DIP[1] * dip * dip
            + COST_TUCK * tuck + COST_LIFT * lift)


def body_lattice(dips: Sequence[float] = DEFAULT_DIPS,
                 tucks: Sequence[float] = DEFAULT_TUCKS,
                 lifts: Sequence[float] = DEFAULT_LIFTS,
                 envelope: Envelope | None = None) -> tuple[Body, ...]:
    """The calibrated capability envelope, sampled on a (dip, tuck, lift) grid.

    Built through `envelope.get_envelope()` and NEVER by reading `outputs/exp001d/envelope.json`
    directly.  That matters: `Envelope.__init__` applies `np.maximum.accumulate` to the
    half-width curve (envelope.py:67-69) precisely so interpolation cannot invert the physical
    relationship on a noisy level, and the raw rows are non-monotone -- up to 3.9 cm NARROWER
    than the shipped certificate at dip 0.25-0.35, against a stated 4 cm body margin.  A table
    built from the raw rows plans through gaps the certified table refuses.

    Using the function rather than `outputs/body_modes.json`'s five-point quantisation also
    supplies the duck x tuck cross term, which the shipped table cannot express at all
    (program.py:19-20 names `beam_and_gap` as the scene that needs exactly that).  The cross
    term is `Envelope.half_width(dip, tuck)`, the repo's own deliberately conservative
    synthesis -- and envelope.py:79 says plainly it is "the weakest link in the envelope",
    so every duck-and-tuck candidate rests on an approximation and the MuJoCo check, not the
    certificate, is what accepts it.
    """
    env = envelope or get_envelope()
    out: list[Body] = []
    for d in dips:
        top = float(env.top(d))
        for t in tucks:
            hw = float(env.half_width(d, t))
            for lf in lifts:
                out.append(Body(dip=float(d), tuck=float(t), lift=float(lf), top=top,
                                half_width=hw,
                                max_step=STEP_AT_POINT if lf >= LIFT_STEP_POINT else 0.0,
                                cost=mode_cost(d, t, lf)))
    return tuple(sorted(out, key=lambda b: (b.cost, b.dip, b.tuck, b.lift)))


def shipped_lattice() -> tuple[Body, ...]:
    """The seven calibrated modes, as lattice points.  The nesting check for K=1."""
    return tuple(Body(dip=round(NOMINAL_PELVIS - m.pelvis_y, 4), tuck=m.tuck, lift=m.lift,
                      top=m.top, half_width=m.half_width, max_step=m.max_step, cost=m.cost)
                 for m in MODES)


def feasible_bodies(tube: RouteTube, lattice: Sequence[Body],
                    a: int, b: int) -> list[int]:
    """Lattice indices that clear EVERY frame of [a, b] on the fixed route."""
    return [i for i, bd in enumerate(lattice) if not tube.blocked(bd)[a:b + 1].any()]


def pareto_minimal(idx: Sequence[int], lattice: Sequence[Body]) -> list[int]:
    """The componentwise-minimal feasible bodies in (dip, tuck, lift).

    REPORTED, NOT FILTERED.  Baseline A's original default dropped every body engaging an axis
    no minimal body needed, which took the mean feasible set from 69.4 to 3.5 bodies per scene
    and the mean distinct discrete active set from 2.55 to 1.08 -- i.e. the filter, not the
    geometry, produced the finding "there is nothing to enumerate".  Deciding which candidates
    are decision-relevant is EXP-005h's job (ParetoRecall over the objective vector f(b)), so
    it is attached to each candidate and left for the gate to condition on.
    """
    ax = {i: (round(lattice[i].dip, 6), round(lattice[i].tuck, 6), round(lattice[i].lift, 6))
          for i in idx}
    out = []
    for i in idx:
        a = ax[i]
        if not any(j != i and ax[j] != a and all(x <= y for x, y in zip(ax[j], a))
                   for j in idx):
            out.append(i)
    return out


# =======================================================================================
# 3.  Skeleton -> ConstraintProgram
# =======================================================================================

@dataclass(frozen=True)
class Block:
    """One contiguous body hold: frames [a, b] at lattice body `body`."""

    a: int
    b: int
    body: int


@dataclass
class BodySkeleton:
    """A body schedule on the fixed route, before it is rendered to a program."""

    blocks: tuple[Block, ...]
    origin: str
    cost: float = float("nan")            # planner currency: excess cost-metres + switches
    cost_ratio: float = 1.0               # / cheapest feasible body on this route
    pareto_minimal: bool = True           # EXP-005h attribute; never a filter here
    note: str = ""

    def key(self) -> tuple:
        return tuple((b.a, b.b, b.body) for b in self.blocks)


def _slots_from_runs(runs: Sequence[tuple[int, int, tuple[float, float, float]]], T: int
                     ) -> np.ndarray:
    """(M_SLOT, N_AX) slot block, TILING any hold too long for one slot.

    `program.encode` caps a slot's half-extent at `SHALF_MAX = 0.35` (program.py:66), so a body
    held for the whole route decodes to 70 % of the clip plus the 0.8 s dilation, not 100 %.
    That is not a cosmetic loss: a GPU check of a route-long duck+tuck whose slot had been
    clipped came back COLLIDING on both seeds (6.6 cm of penetration) because the truncated
    slot pulled the duck away from the beam.  So a run whose half-extent exceeds SHALF_MAX is
    split across consecutive slots, which is what makes baseline B's MODE-FLOOR arm express
    the thing it exists to express.
    """
    slot = np.zeros((M_SLOT, N_AX))
    denom = max(T - 1, 1)
    pieces: list[tuple[float, float, tuple[float, float, float]]] = []
    for a, b, ax in runs:
        half = (b - a) / 2 / denom
        n = max(1, int(np.ceil(half / SHALF_MAX - 1e-9)))
        edges = np.linspace(a, b, n + 1)
        for k in range(n):
            lo, hi = edges[k], edges[k + 1]
            pieces.append((((lo + hi) / 2) / denom,
                           min(max((hi - lo) / 2 / denom, 1.5 / denom), SHALF_MAX), ax))
    # More pieces than slots: keep the strongest, as encode does (program.py:175), and say so.
    pieces.sort(key=lambda p: -max(p[2]))
    for k, (mid, half, ax) in enumerate(sorted(pieces[:M_SLOT])):
        slot[k] = [float(np.clip(mid, 0.0, 1.0)), float(half),
                   min(ax[0], DIP_MAX), min(ax[1], TUCK_MAX), min(ax[2], LIFT_MAX)]
    return slot


def skeleton_to_program(sk: BodySkeleton, base: ConstraintProgram,
                        lattice: Sequence[Body], T: int,
                        pad_m: float = 0.0, shift_m: float = 0.0,
                        tube: RouteTube | None = None) -> ConstraintProgram:
    """Render a skeleton into a ConstraintProgram, with the ROUTE COPIED VERBATIM.

    `pad_m` widens every hold symmetrically and `shift_m` moves its centre; both are metres of
    travel, converted to arc-length fraction through the tube.  They are the ONSET and OFFSET
    axes of `b`, which the guidance names as part of the body program and which A* can never
    choose on its own because `MODE_SWITCH_COST` makes an early or long adaptation strictly
    more expensive.
    """
    L = max(tube.length, 1e-6) if tube is not None else 1.0
    dpad, dshift = pad_m / L, shift_m / L
    denom = max(T - 1, 1)
    runs = []
    for blk in sk.blocks:
        bd = lattice[blk.body]
        a = blk.a - dpad * denom + dshift * denom
        b = blk.b + dpad * denom + dshift * denom
        runs.append((float(np.clip(a, 0, denom)), float(np.clip(b, 0, denom)),
                     (bd.dip, bd.tuck, bd.lift)))
    slot = _slots_from_runs(runs, T)
    return ConstraintProgram(lat=base.lat.copy(), slot=slot, speed=base.speed)


# =======================================================================================
# 4.  Cost, in the planner's own currency
# =======================================================================================

def _dilate_max(v: np.ndarray, w: int) -> np.ndarray:
    """Vectorised `planner._dilate_channel(v, w, "max")`.

    Bit-identical (checked in the module's self-test): the planner truncates its window at the
    array edges and edge-padding replicates the same extremum, so the maxima agree everywhere.
    It is duplicated rather than imported because the shipped loop is O(T*w) in Python and the
    enumerator renders thousands of candidate programs per scene -- it was 90 % of baseline A's
    CPU time before this.
    """
    if w <= 0:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    p = np.pad(v, (w, w), mode="edge")
    return sliding_window_view(p, 2 * w + 1).max(axis=-1)[:len(v)]


def rendered_axes(prog: ConstraintProgram, T: int, fps: float) -> tuple[np.ndarray, ...]:
    """(dip, tuck, lift) per frame exactly as `decode` will send them (program.py:215-230).

    Measuring effort on the RENDERED channels rather than the slot numbers is what makes "held
    a deep duck for the whole clip" cost more than "ducked briefly", and it is also the only
    place a post-dilation two-axis overlap becomes visible (see `uncalibrated_overlap`).
    """
    u = np.linspace(0, 1, T)
    dip = np.zeros(T)
    tuck = np.zeros(T)
    lift = np.zeros(T)
    for k in prog.active_slots:
        mid, half, d, tk, lf = prog.slot[k]
        m = np.abs(u - mid) <= max(half, 1.0 / T)
        dip[m] = np.maximum(dip[m], d)
        tuck[m] = np.maximum(tuck[m], tk)
        lift[m] = np.maximum(lift[m], lf)
    w = int(round(LEAD_S * fps))
    sm = max(3, int(SMOOTH_S * fps) | 1)
    return (_smooth(_dilate_max(dip, w), sm),
            _smooth(_dilate_max(tuck, w), sm),
            _smooth(_dilate_max(lift, w), sm))


def j_body(prog: ConstraintProgram, T: int, fps: float, route_len: float,
           switch_cost: bool = True, axes: tuple | None = None) -> float:
    """Excess body effort in the planner's units: extra cost-metres over walking upright.

    Includes a mode-switch term, so it really is the planner's objective and not a rename:
    `planner._astar` charges `MODE_SWITCH_COST * res` per transition (planner.py:311), and a
    schedule that flickers must not look free.  The switch count is read off the rendered
    channels at `ACTIVE` thresholds.
    """
    dip, tuck, lift = axes if axes is not None else rendered_axes(prog, T, fps)
    per_m = (COST_DIP[0] * dip + COST_DIP[1] * dip ** 2 + COST_TUCK * tuck + COST_LIFT * lift)
    j = float(per_m.mean() * route_len)
    if switch_cost:
        on = (dip > ACTIVE["dip"]) | (tuck > ACTIVE["tuck"]) | (lift > ACTIVE["lift"])
        j += MODE_SWITCH_COST * 0.05 * int(np.sum(on[1:] != on[:-1]))
    return j


def uncalibrated_overlap(prog: ConstraintProgram, T: int, fps: float,
                         axes: tuple | None = None) -> bool:
    """Does the RENDERED request command two adaptation axes at once?

    Checked after dilation, not on `encode`'s runs.  `decode` dilates each slot by
    `LEAD_S = 0.8 s` per side, so two slots on different axes within ~1.6 s re-merge in the
    ConstraintSpec even though they were separate slots: a certificate checked on the slot
    partition passed 31 of 201 candidates that commanded dip AND tuck simultaneously after
    dilation.  This is a DIAGNOSTIC, not a filter -- `Envelope.half_width(dip, tuck)` does
    certify the combination (conservatively), and the per-instance guarantee is the MuJoCo
    check either way.  It is reported so a collision can be attributed.
    """
    dip, tuck, lift = axes if axes is not None else rendered_axes(prog, T, fps)
    on = np.stack([dip > ACTIVE["dip"], tuck > ACTIVE["tuck"], lift > ACTIVE["lift"]])
    return bool((on.sum(axis=0) > 1).any())


# =======================================================================================
# 5.  The proposal-side descriptor surrogate (CPU only; NEVER a gate metric)
# =======================================================================================

def predicted_descriptor(prog: ConstraintProgram, tube: RouteTube, T: int, fps: float,
                         envelope: Envelope | None = None,
                         axes: tuple | None = None) -> np.ndarray:
    """(n_windows * N_CHANNELS,) prediction of `morphology.matched_delta`, from the plan alone.

    Same channels, same order, same sign convention (positive = more adaptation), pushed
    through the SAME calibrated envelope the planner certifies against, so `dw` correctly comes
    out NEGATIVE for a deep duck -- ducking makes G1 wider (EXP-001d), and a surrogate that
    gets that backwards would order candidates the wrong way round on the lateral axis.

    THIS IS A SCREENING SURROGATE.  It was measured against the achieved descriptor at rank
    correlation 0.34 over 18 pairs: precision is decent (17 of 18 pairs it called distinct were
    >2 sigma apart after generation) and ordering is not usable.  So it deduplicates
    proposals before they are paid for in ARDY calls, and it never appears in a reported
    coverage number.
    """
    env = envelope or get_envelope()
    dip, tuck, lift = axes if axes is not None else rendered_axes(prog, T, fps)
    top0 = float(env.top(0.0))
    hw0 = float(env.half_width(0.0, 0.0))
    dtop = top0 - np.asarray(env.top(dip), float)
    dw = hw0 - np.asarray(env.half_width(dip, tuck), float)
    dz = np.where(lift >= LIFT_STEP_POINT, STEP_AT_POINT, lift * 0.0)
    act = np.maximum.reduce([dtop / 0.02, np.abs(dw) / 0.02, dz / 0.03])
    wins = tube.windows or [(0, T - 1)]
    rows = []
    for (a, b) in wins:
        lo = max(0, a - int(round(2 * LEAD_S * fps)))
        seg = slice(lo, min(T, b + 1 + int(round(LEAD_S * fps))))
        on = np.flatnonzero(act[seg] > max(1.0, 0.25 * act[seg].max()) if act[seg].max() > 0
                            else np.inf)
        t_lead = float((a - (seg.start + on[0])) / fps) if len(on) else 0.0
        t_dur = float((on[-1] - on[0] + 1) / fps) if len(on) else 0.0
        w = slice(a, b + 1)
        rows.append([float(dtop[w].max()), float(dw[w].max()), float(dw[w].max()),
                     float(dz[w].max()), float(dz[w].max()),
                     max(t_lead, 0.0), t_dur, 0.0])
    return np.asarray(rows, float).ravel()


# =======================================================================================
# 6.  Shared context
# =======================================================================================

@dataclass(eq=False)
class RouteContext:
    """Everything every baseline needs about ONE fixed route.  Built once, shared."""

    scene: object
    route_plan: Plan
    base: ConstraintProgram          # the route; `lat` is copied verbatim into every candidate
    tube: RouteTube
    lattice: tuple[Body, ...]
    T: int
    fps: float
    speed: float
    per_window: list[list[int]]      # feasible lattice indices per demand window
    free_segments: list[tuple[int, int]]   # segments of the route that demand NOTHING
    per_free: list[list[int]]        # feasible lattice indices per free segment
    background: list[int]            # feasible lattice indices everywhere OUTSIDE the windows
    stand: int                       # lattice index of the cheapest body (upright)
    feasible: bool
    note: str = ""

    @property
    def n_windows(self) -> int:
        return len(self.tube.windows)

    def interactions(self, half_width: float = 0.7) -> list[Interaction]:
        """Where the descriptor is measured.  ONE per demand window, deduplicated.

        NOT one per box.  On `beam_and_gap` the two gap walls share an x-centre, so a
        box-centre rule produces two bit-identical descriptor blocks (verified: the 8- and
        16-dimensional slices are `np.allclose`) and then weights the dead duplicate 2:1
        against the beam while the block whitener divides by sqrt(3) as if three independent
        windows existed.  Windows come from the tube, in arc length, which is the coordinate
        the guidance asks for.
        """
        if not self.tube.windows:
            return [Interaction(float(np.median(self.tube.xy[:, 0])), 1e9)]
        out, seen = [], []
        for (a, b) in self.tube.windows:
            x = float(self.tube.xy[a:b + 1, 0].mean())
            if all(abs(x - v) > half_width for v in seen):
                seen.append(x)
                out.append(Interaction(x, half_width))
        return out


def route_context(scene, route_plan: Plan, *, fps: float = 25.0, speed: float = SPEED,
                  lattice: Sequence[Body] | None = None,
                  max_duration: float = MAX_DURATION,
                  merge_gap_m: float = 0.5, max_free_segments: int = 2,
                  min_free_m: float = 1.0) -> RouteContext:
    """Fix the route and read everything off it.  Zero ARDY calls, ~1-3 ms."""
    lat = tuple(lattice) if lattice is not None else body_lattice()
    if not route_plan.feasible or len(route_plan.xy) < 2:
        return RouteContext(scene, route_plan, ConstraintProgram(), None, lat, 0, fps, speed,
                            [], [], [], [], 0, False, "route infeasible")
    T = n_frames(route_plan, fps, speed, max_duration)
    base = route_program(scene, route_plan, fps, speed)
    tube = route_tube(scene, base, T, merge_gap_m=merge_gap_m)
    per_window = [feasible_bodies(tube, lat, a, b) for (a, b) in tube.windows]
    free = np.ones(T, bool)
    for (a, b) in tube.windows:
        free[a:b + 1] = False
    background = ([i for i, bd in enumerate(lat) if not tube.blocked(bd)[free].any()]
                  if free.any() else feasible_bodies(tube, lat, 0, T - 1))
    # FREE SEGMENTS: stretches the geometry demands nothing of.  They exist because a true
    # k-best over the (x, y, mode) graph does NOT stop at the demand windows -- once the
    # variants under the beam are exhausted it starts spending cost on adaptations in open
    # corridor, and those are real, cost-competitive points of F_B.  A site-based enumerator
    # that omits them returns exactly ONE candidate on every route with no demand window
    # (measured: 61.9 % of the 128-scene suite -- pillar and partial_beam), which makes its
    # K-curve flat by construction rather than by geometry.
    free_segments: list[tuple[int, int]] = []
    n_free = max(0, max_free_segments if tube.windows else max_free_segments)
    if n_free:
        edges = [0] + [x for (a, b) in tube.windows for x in (a, b)] + [T - 1]
        gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
        gaps = [(a, b) for (a, b) in gaps if tube.s[b] - tube.s[a] >= min_free_m]
        gaps.sort(key=lambda g: -(tube.s[g[1]] - tube.s[g[0]]))
        free_segments = sorted(gaps[:n_free])
    per_free = [feasible_bodies(tube, lat, a, b) for (a, b) in free_segments]
    ok = all(len(w) for w in per_window) and bool(background)
    note = "" if ok else "some window or the background admits no calibrated body"
    stand = int(np.argmin([b.cost for b in lat]))
    return RouteContext(scene, route_plan, base, tube, lat, T, fps, speed,
                        per_window, free_segments, per_free, background or [stand],
                        stand, ok, note)


def astar_program(ctx: RouteContext) -> ConstraintProgram:
    """The shipped planner's own answer, as a program.  Candidate 0 of every baseline.

    Pinning this makes BODY-ENUMERATE@1 the current system for EVERY baseline, so the K=1
    column of the gate is a single shared number and no baseline can look good at K=1 by
    quietly returning something the planner does not produce.
    """
    return ConstraintProgram(lat=ctx.base.lat.copy(), slot=ctx.base.slot.copy(),
                             speed=ctx.base.speed)


# =======================================================================================
# 7.  Timing axis (shared by A, B, C)
# =======================================================================================

# Metres of travel.  The renderer already dilates every adaptation by LEAD_S = 0.8 s ~ 0.72 m
# either side, so a timing variant smaller than that is erased before ARDY sees it; and
# EXP-005f measured the seed noise on t_lead at 0.73 s (sd) / 2.29 s (q99), so even these are
# only ~1 sigma apart.  They are enumerated because onset/offset are part of `b` by definition
# and a baseline that pins them cannot be said to have enumerated the body set -- but the gate
# must not be surprised when they fail to separate.
TIMING_PADS_M = (0.0, 0.75, 1.5)
TIMING_SHIFTS_M = (0.0, -0.75, 0.75)


def _timing_variants(pads: Sequence[float], shifts: Sequence[float]) -> list[tuple[float, float]]:
    return [(p, s) for p in pads for s in shifts]


# =======================================================================================
# 8.  BASELINE A -- exact k-best over the fixed-route body chain
# =======================================================================================

def _merge_product(per: Sequence[Sequence[tuple]], limit: int) -> list[tuple]:
    """Exact k-best over a product of independent, individually sorted option lists.

    Each element of `per` is `[(cost, blocks_or_block), ...]` sorted by cost.  Because the
    windows are separated by free corridor the total cost is additive, so the product can be
    expanded in true cost order with a heap without ever materialising it -- Lawler
    partitioning on a chain.  Returns `[(total_cost, tuple_of_blocks), ...]`.
    """
    if not per:
        return [(0.0, ())]
    start = tuple(0 for _ in per)
    base = sum(p[0][0] for p in per)
    seen = {start}
    heap = [(base, start)]
    out: list[tuple] = []
    while heap and len(out) < limit:
        c, idx = heapq.heappop(heap)
        blocks: list[Block] = []
        for w, k in enumerate(idx):
            v = per[w][k][1]
            blocks.extend(v if isinstance(v, (list, tuple)) else [v])
        out.append((float(c), tuple(blocks)))
        for w in range(len(per)):
            if idx[w] + 1 < len(per[w]):
                nxt = idx[:w] + (idx[w] + 1,) + idx[w + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (c - per[w][idx[w]][0] + per[w][idx[w] + 1][0], nxt))
    return out


def _chain_assignments(ctx: RouteContext, blocks_per_window: list[list[list[Block]]],
                       limit: int) -> Iterable[tuple[float, tuple[Block, ...]]]:
    """k-best over the product of per-window options, cheapest first, by heap merge.

    At a FIXED route the (x, y, mode) product graph collapses to a CHAIN: the windows are
    separated by free corridor, so total cost is additive over windows and the product can be
    expanded in cost order without solving it.  This is Lawler partitioning / k-best on a
    chain, which is what "k-best over the augmented graph, grouped by route" reduces to once
    the route is fixed -- and it is exact, where the textbook form is not merely expensive but
    DEGENERATE: measured on `partial_beam`, plain k-labels-per-node k-best at K=16 returns
    sixteen goal paths at identical cost 8.947 with identical body word (one-cell wiggles),
    and on `overhead_beam` / `beam_and_gap` it burns 11-12 s and reaches the goal ZERO times.
    """
    per = []
    for opts in blocks_per_window:
        per.append(sorted(((sum(_block_cost(ctx, b) for b in combo), tuple(combo))
                           for combo in opts), key=lambda t: t[0]))
    if not per:
        yield 0.0, ()
        return
    for c, blocks in _merge_product(per, limit):
        yield float(c), tuple(blocks)


def _block_cost(ctx: RouteContext, blk: Block) -> float:
    bd = ctx.lattice[blk.body]
    metres = float(ctx.tube.s[blk.b] - ctx.tube.s[blk.a])
    return bd.effort * max(metres, ctx.tube.length / max(ctx.T - 1, 1))


def _window_block_options(ctx: RouteContext, w: int, split: bool) -> list[list[Block]]:
    """Every way to fill demand window `w`: one body, or two bodies split at peak demand.

    Splitting is what lets the enumerator express a body that VARIES WITHIN one interaction --
    a duck that deepens through the beam -- which "one mode per demand block" cannot, and which
    was named as the family baseline A structurally could not reach.
    """
    a, b = ctx.tube.windows[w]
    out = [[Block(a, b, m)] for m in ctx.per_window[w]]
    if split and b - a >= 6:
        c = int(a + np.argmax(ctx.tube.dip_req[a:b + 1] + ctx.tube.tuck_req[a:b + 1]))
        c = int(np.clip(c, a + 2, b - 2))
        left = feasible_bodies(ctx.tube, ctx.lattice, a, c)
        right = feasible_bodies(ctx.tube, ctx.lattice, c + 1, b)
        for m1 in left:
            for m2 in right:
                if m1 != m2:
                    out.append([Block(a, c, m1), Block(c + 1, b, m2)])
    return out


def _free_block_options(ctx: RouteContext, k: int, max_bodies: int = 12) -> list[list[Block]]:
    """Ways to fill a free segment: leave it alone, or hold a body there anyway."""
    a, b = ctx.free_segments[k]
    opts: list[list[Block]] = [[]]
    bodies = sorted(ctx.per_free[k], key=lambda i: ctx.lattice[i].cost)
    for m in bodies[:max_bodies]:
        if ctx.lattice[m].effort > 1e-9:
            opts.append([Block(a, b, m)])
    return opts


def kbest_skeletons(ctx: RouteContext, *, max_skeletons: int = 400,
                    split_windows: bool = True,
                    gratuitous: bool = True) -> list[BodySkeleton]:
    """Baseline A's skeletons: exact k-best body schedules on the fixed route."""
    if not ctx.feasible:
        return []
    opts = [_window_block_options(ctx, w, split_windows) for w in range(ctx.n_windows)]
    if gratuitous:
        opts += [_free_block_options(ctx, k) for k in range(len(ctx.free_segments))]
    out: list[BodySkeleton] = []
    minimal = {w: set(pareto_minimal(ctx.per_window[w], ctx.lattice))
               for w in range(ctx.n_windows)}
    for cost, blocks in _chain_assignments(ctx, opts, max_skeletons):
        pm = (len(blocks) == ctx.n_windows
              and all(blk.body in minimal[w] for w, blk in enumerate(blocks)))
        out.append(BodySkeleton(blocks=blocks, origin="kbest", cost=cost,
                                pareto_minimal=pm))
    if not out:
        out = [BodySkeleton(blocks=(), origin="kbest", cost=0.0)]
    c0 = max(min(s.cost for s in out), 1e-9)
    for s in out:
        s.cost_ratio = s.cost / c0
    return out


def enumerate_kbest(scene, route_plan: Plan, K: int = 8, runner=None, *,
                    ctx: RouteContext | None = None, fps: float = 25.0,
                    speed: float = SPEED, lattice: Sequence[Body] | None = None,
                    max_skeletons: int = 400, split_windows: bool = True,
                    gratuitous: bool = True,
                    pads_m: Sequence[float] = TIMING_PADS_M,
                    shifts_m: Sequence[float] = TIMING_SHIFTS_M,
                    cost_bloat: float = 0.35, anchor_astar: bool = True,
                    return_candidates: bool = False):
    """BASELINE A.  k-best over the (x, y, body-mode) graph, restricted to the fixed route.

    Zero ARDY calls; `runner` is accepted for signature uniformity and unused.
    """
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    sks = kbest_skeletons(ctx, max_skeletons=max_skeletons, split_windows=split_windows,
                          gratuitous=gratuitous)
    cands = _expand(ctx, sks, pads_m, shifts_m, "kbest")
    return _finish(ctx, cands, K, cost_bloat, anchor_astar, return_candidates)


# =======================================================================================
# 9.  BASELINE B -- body no-good cuts, plus the whole-route MODE FLOOR
# =======================================================================================

def nogood_skeletons(ctx: RouteContext, *, max_rounds: int = 24) -> list[BodySkeleton]:
    """Iterated no-good: forbid the per-window body set just used, re-solve, repeat.

    The no-good is over the PER-WINDOW body, not the frame-exact schedule.  That is the
    coarsest cut that is still sound: `decode` takes a per-channel MAX over the 0.8 s dilation
    window (planner.py:_dilate_channel), so anything finer than "which envelope was engaged at
    this interaction" is erased before ARDY sees the request, and a one-frame relabelling would
    evade a finer cut while producing the identical clip.
    """
    if not ctx.feasible:
        return []
    forbidden: list[set[int]] = [set() for _ in range(ctx.n_windows)]
    out: list[BodySkeleton] = []
    for _ in range(max_rounds):
        blocks, cost = [], 0.0
        ok = True
        for w, (a, b) in enumerate(ctx.tube.windows):
            avail = [m for m in ctx.per_window[w] if m not in forbidden[w]]
            if not avail:
                ok = False
                break
            m = min(avail, key=lambda i: ctx.lattice[i].cost)
            blk = Block(a, b, m)
            blocks.append(blk)
            cost += _block_cost(ctx, blk)
        if not ok:
            break
        out.append(BodySkeleton(tuple(blocks), "nogood", cost))
        if not ctx.n_windows:
            break
        # Lawler partition: the negation of a conjunction is a disjunction over windows, so
        # cut the window whose next-cheapest alternative is cheapest.
        best_w, best_d = None, float("inf")
        for w, blk in enumerate(blocks):
            avail = [m for m in ctx.per_window[w]
                     if m not in forbidden[w] and m != blk.body]
            if not avail:
                continue
            d = min(ctx.lattice[m].cost for m in avail) - ctx.lattice[blk.body].cost
            if d < best_d:
                best_w, best_d = w, d
        if best_w is None:
            break
        forbidden[best_w].add(blocks[best_w].body)
    return out


def mode_floor_skeletons(ctx: RouteContext) -> list[BodySkeleton]:
    """"You must be at least this adapted EVERYWHERE" -- one skeleton per lattice floor.

    THIS IS THE ARM THE REVIEW OF BASELINE B NAMED, and it is not an invention: `plan()` has
    taken an `allow_modes` argument since strategies.py shipped (planner.py:355, 389-392) and
    `strategies.MODE_SETS = (None, STAND_ONLY)` is already its two-point version.  Restricted
    to a fixed route it becomes a floor on the body: forbid every lattice body cheaper than
    `m` over the whole route.

    Why it matters more than the no-good it accompanies: a site-based enumerator can only
    propose a body where geometry FORCES one, so on a route with no demand window it returns
    exactly one candidate, and every one of its own measured misses (52 "gratuitous adaptation
    on a route that needs none" + 71 "held an adaptation through free space", zero "at a
    site") lies outside its candidate space.  Adding this arm took recall@8 from 0.476 to
    0.773 against the same reference at the same CPU cost.  A route-long hold is exactly what
    `_slots_from_runs` has to tile, which is why that function exists.
    """
    if not ctx.feasible:
        return []
    out: list[BodySkeleton] = []
    order = sorted(range(len(ctx.lattice)), key=lambda i: ctx.lattice[i].cost)
    for m in order:
        bd = ctx.lattice[m]
        allow = [i for i in order if ctx.lattice[i].cost >= bd.cost - 1e-12]
        # feasible everywhere on the route under this floor?
        floor_bg = [i for i in ctx.background if i in allow]
        if not floor_bg:
            continue
        bg = min(floor_bg, key=lambda i: ctx.lattice[i].cost)
        blocks = [Block(0, ctx.T - 1, bg)]
        cost = _block_cost(ctx, blocks[0])
        for w, (a, b) in enumerate(ctx.tube.windows):
            avail = [i for i in ctx.per_window[w] if i in allow]
            if not avail:
                blocks = []
                break
            mw = min(avail, key=lambda i: ctx.lattice[i].cost)
            if mw != bg:
                blocks.append(Block(a, b, mw))
                cost += _block_cost(ctx, Block(a, b, mw))
        if not blocks:
            continue
        out.append(BodySkeleton(tuple(blocks), "mode_floor", cost,
                                note=f"floor={bd.name}"))
    return out


def enumerate_nogood(scene, route_plan: Plan, K: int = 8, runner=None, *,
                     ctx: RouteContext | None = None, fps: float = 25.0,
                     speed: float = SPEED, lattice: Sequence[Body] | None = None,
                     max_rounds: int = 24, mode_floor: bool = True,
                     pads_m: Sequence[float] = TIMING_PADS_M,
                     shifts_m: Sequence[float] = TIMING_SHIFTS_M,
                     cost_bloat: float = 0.35, anchor_astar: bool = True,
                     return_candidates: bool = False):
    """BASELINE B.  Body no-good cuts at a fixed route, plus the whole-route mode floor.

    Zero ARDY calls.
    """
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    sks = nogood_skeletons(ctx, max_rounds=max_rounds)
    if mode_floor:
        sks += mode_floor_skeletons(ctx)
    c0 = max(min((s.cost for s in sks), default=1.0), 1e-9)
    for s in sks:
        s.cost_ratio = s.cost / c0
    cands = _expand(ctx, sks, pads_m, shifts_m, "nogood")
    return _finish(ctx, cands, K, cost_bloat, anchor_astar, return_candidates)


# =======================================================================================
# 10.  BASELINE C -- decomposed objective, preference simplex, ties ENUMERATED
# =======================================================================================

FEATURES = ("dip", "tuck", "lift", "effort", "clear_top", "clear_lat", "hold", "switch")
N_FEAT = len(FEATURES)
C_MAX = 0.30                       # clearance saturates; past this no body choice is better


def _feature_matrix(ctx: RouteContext) -> np.ndarray:
    """(n_windows + 1, n_lattice, N_FEAT) per-block feature densities.

    Row -1 is the background (outside every window).  `clear_top` and `clear_lat` genuinely
    FIGHT, because the calibrated envelope has half-width rising with dip: buying headroom by
    ducking costs lateral margin.  That tension is what makes a preference sweep return more
    than one adaptation family at all.
    """
    W = ctx.n_windows + len(ctx.free_segments)
    F = np.zeros((W + 1, len(ctx.lattice), N_FEAT))
    spans = list(ctx.tube.windows) + list(ctx.free_segments)
    free = np.ones(ctx.T, bool)
    for a, b in spans:
        free[a:b + 1] = False
    spans = spans + [(0, ctx.T - 1)]
    for r, (a, b) in enumerate(spans):
        metres = float(ctx.tube.s[b] - ctx.tube.s[a]) or ctx.tube.length / max(ctx.T - 1, 1)
        for i, bd in enumerate(ctx.lattice):
            head, lat = ctx.tube.clearance(bd, a, b)
            F[r, i] = [bd.dip / DIP_MAX, bd.tuck / TUCK_MAX, bd.lift / LIFT_MAX,
                       bd.effort / 1.4,
                       1.0 - float(np.clip(head / C_MAX, 0, 1)),
                       1.0 - float(np.clip(lat / C_MAX, 0, 1)),
                       metres / max(ctx.tube.length, 1e-9) * (bd.effort > 1e-9),
                       0.0]
            F[r, i] *= metres
    return F


def _solve_weight(ctx: RouteContext, F: np.ndarray, w: np.ndarray, topk: int
                  ) -> list[tuple[float, tuple[Block, ...]]]:
    """Top-`topk` schedules under preference `w`, per window independently (a chain).

    An ARGMIN here would be the strawman: an LP over the reachable set measured 68.9 % of the
    within-route reference as WEAKLY supported -- it only ever TIES for optimal -- so no
    single-argmin sweep can return it at any budget, and 3.6 % more is phi-degenerate.  Taking
    the top-`topk` per scalarisation is precisely the tie mass that would otherwise be
    structurally unreachable, and it costs nothing on a chain.
    """
    per = []
    spans = list(ctx.tube.windows) + list(ctx.free_segments)
    avail = list(ctx.per_window) + list(ctx.per_free)
    for wi, (a, b) in enumerate(spans):
        scored = sorted(((float(F[wi, m] @ w), m) for m in avail[wi]), key=lambda t: t[0])
        per.append([(c, Block(a, b, m)) for c, m in scored[:topk]])
    if not per:
        scored = sorted(((float(F[-1, m] @ w), m) for m in ctx.background))
        return [(scored[0][0], ())] if scored else []
    # Merge the per-span option lists with a heap, NOT with itertools.product plus a break:
    # lexicographic product order fixes span 0 at its own argmin for its first |rest| combos,
    # so truncating it returned `topk` schedules that all shared the window body and differed
    # only in free-segment noise -- which collapsed this baseline to ONE distinct candidate on
    # scenes where it should have had dozens.
    return [(c, tuple(b for b in blocks))
            for c, blocks in _merge_product(per, topk)]


def sobol_dirichlet(n: int, alpha: float = 0.35, seed: int = 0) -> np.ndarray:
    """(n, N_FEAT) low-discrepancy preference vectors.  alpha<1 concentrates near faces."""
    from scipy.stats import gamma, qmc
    eng = qmc.Sobol(d=N_FEAT, scramble=True, seed=seed)
    u = eng.random_base2(int(np.ceil(np.log2(max(n, 2)))))[:n]
    g = np.maximum(gamma.ppf(np.clip(u, 1e-6, 1 - 1e-6), a=alpha), 1e-12)
    return g / g.sum(1, keepdims=True)


def supportedness(phis: np.ndarray, i: int, tol: float = 1e-9) -> str:
    """Is row `i` the STRICT argmin for SOME non-negative preference vector?

    The certificate that says whether a miss is a budget problem or a structural one:
        max delta  s.t.  (Phi_j - Phi_i) . w >= delta  for all j,  w >= 0,  sum w = 1
    'strict' (delta>0) -- reachable, more weights help; 'weak' (delta=0) -- only ever ties, so
    a single-argmin sweep never returns it; 'non_supported' (delta<0) -- in a dent of the lower
    convex hull, unreachable at any budget.  Measured on 62 scenes: 27.6 / 68.9 / 0.0 %, with
    3.6 % phi-degenerate.  Report it; it is this baseline's honest ceiling.
    """
    from scipy.optimize import linprog
    D = phis - phis[i]
    D = D[np.linalg.norm(D, axis=1) > 1e-9]
    if len(D) == 0:
        return "strict"
    F = phis.shape[1]
    c = np.zeros(F + 1)
    c[-1] = -1.0
    r = linprog(c, np.hstack([-D, np.ones((len(D), 1))]), np.zeros(len(D)),
                np.concatenate([np.ones((1, F)), np.zeros((1, 1))], axis=1), [1.0],
                bounds=[(0, None)] * F + [(None, None)], method="highs")
    if not r.success:
        return "infeasible"
    d = -float(r.fun)
    return "strict" if d > tol else ("weak" if d > -tol else "non_supported")


def enumerate_weight_sweep(scene, route_plan: Plan, K: int = 8, runner=None, *,
                           ctx: RouteContext | None = None, fps: float = 25.0,
                           speed: float = SPEED, lattice: Sequence[Body] | None = None,
                           n_sobol: int = 96, alpha: float = 0.35, topk_per_weight: int = 4,
                           pads_m: Sequence[float] = TIMING_PADS_M,
                           shifts_m: Sequence[float] = TIMING_SHIFTS_M,
                           cost_bloat: float = 0.35, anchor_astar: bool = True,
                           seed: int = 0, return_candidates: bool = False):
    """BASELINE C.  J = w_d J_dip + w_t J_tuck + w_l J_lift + w_c J_clearance, swept.

    Zero ARDY calls.  The route is fixed, so the search under each weight is a chain and the
    "A* call" per weight is ~0.1 ms rather than ~130 ms.
    """
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    F = _feature_matrix(ctx)
    ws = [np.eye(N_FEAT)[k] for k in range(N_FEAT)]
    ws.append(np.eye(N_FEAT)[FEATURES.index("effort")])
    ws += list(sobol_dirichlet(n_sobol, alpha, seed))
    seen: dict[tuple, BodySkeleton] = {}
    for w in ws:
        for cost, blocks in _solve_weight(ctx, F, np.asarray(w, float), topk_per_weight):
            sk = BodySkeleton(blocks, "wsweep", sum(_block_cost(ctx, b) for b in blocks))
            seen.setdefault(sk.key(), sk)
    sks = list(seen.values())
    c0 = max(min((s.cost for s in sks), default=1.0), 1e-9)
    for s in sks:
        s.cost_ratio = s.cost / c0
    cands = _expand(ctx, sks, pads_m, shifts_m, "wsweep")
    return _finish(ctx, cands, K, cost_bloat, anchor_astar, return_candidates)


# =======================================================================================
# 11.  Candidates, and how K of them are chosen
# =======================================================================================

@dataclass(eq=False)
class BodyCandidate:
    """One proposed point of F_B(S, r), before ARDY has been asked."""

    program: ConstraintProgram
    origin: str
    cost: float                    # excess body effort, planner currency (cost-metres)
    total_cost: float              # route_len + cost -- the quantity dJ is measured on
    cost_ratio: float = 1.0        # total_cost / cheapest candidate's total_cost
    predicted: np.ndarray | None = None
    pareto_minimal: bool = True
    overlap: bool = False          # rendered request commands two axes at once
    note: str = ""

    def key(self) -> bytes:
        return np.round(self.program.slot, 4).tobytes()


def make_candidate(ctx: RouteContext, prog: ConstraintProgram, origin: str,
                   pareto_minimal: bool = True, note: str = "") -> BodyCandidate:
    """Score one program: rendered axes ONCE, then cost, overlap and the surrogate."""
    axes = rendered_axes(prog, ctx.T, ctx.fps)
    j = j_body(prog, ctx.T, ctx.fps, ctx.tube.length, axes=axes)
    return BodyCandidate(program=prog, origin=origin, cost=j,
                         total_cost=ctx.tube.length + j, pareto_minimal=pareto_minimal,
                         overlap=uncalibrated_overlap(prog, ctx.T, ctx.fps, axes=axes),
                         predicted=predicted_descriptor(prog, ctx.tube, ctx.T, ctx.fps,
                                                        axes=axes),
                         note=note)


def _expand(ctx: RouteContext, skeletons: Sequence[BodySkeleton],
            pads_m: Sequence[float], shifts_m: Sequence[float], origin: str,
            timing_budget: int = 48, max_candidates: int = 900) -> list[BodyCandidate]:
    """Skeletons x timing variants -> deduplicated candidates, cost-ordered.

    The timing grid is applied in full to the `timing_budget` cheapest skeletons and only at
    (0, 0) to the rest.  Rendering a program costs ~0.4 ms, so an unbudgeted 400 x 9 product is
    seconds per scene for candidates the cost filter would drop anyway.
    """
    pool: dict[bytes, BodyCandidate] = {}
    ranked = sorted(skeletons, key=lambda s: s.cost)
    tv = _timing_variants(pads_m, shifts_m)
    for n, sk in enumerate(ranked):
        for pad, shift in (tv if n < timing_budget else [(0.0, 0.0)]):
            if not sk.blocks and (pad or shift):
                continue
            prog = skeleton_to_program(sk, ctx.base, ctx.lattice, ctx.T, pad, shift, ctx.tube)
            cand = make_candidate(ctx, prog, f"{origin}:{sk.origin}", sk.pareto_minimal,
                                  f"{sk.note} pad={pad:+.2f} shift={shift:+.2f}".strip())
            k = cand.key()
            if k not in pool or cand.total_cost < pool[k].total_cost:
                pool[k] = cand
        if len(pool) >= max_candidates:
            break
    out = sorted(pool.values(), key=lambda c: c.total_cost)
    if out:
        c0 = max(out[0].total_cost, 1e-9)
        for c in out:
            c.cost_ratio = c.total_cost / c0
    return out


def _finish(ctx: RouteContext, cands: list[BodyCandidate], K: int, cost_bloat: float,
            anchor_astar: bool, return_candidates: bool):
    if anchor_astar:
        cands = _prepend_astar(ctx, cands)
    sel = select_k(cands, K, cost_bloat=cost_bloat, anchor_first=anchor_astar)
    return sel if return_candidates else [c.program for c in sel]


def _prepend_astar(ctx: RouteContext, cands: list[BodyCandidate]) -> list[BodyCandidate]:
    anchor = make_candidate(ctx, astar_program(ctx), "astar",
                            note="shipped planner's own answer")
    rest = [c for c in cands if c.key() != anchor.key()]
    if rest:
        c0 = max(min([anchor.total_cost] + [c.total_cost for c in rest]), 1e-9)
        anchor.cost_ratio = anchor.total_cost / c0
        for c in rest:
            c.cost_ratio = c.total_cost / c0
    return [anchor] + rest


def _predicted_active(pred: np.ndarray, floor: np.ndarray | None = None) -> tuple:
    """Discrete signature predicted from the plan, for proposal-side spread only."""
    f = SEED_SIGMA_PRIOR if floor is None else floor
    rows = np.asarray(pred, float).reshape(-1, N_CHANNELS)
    return tuple((bool(r[0] > 2 * f[0]), bool(max(r[1], r[2]) > 2 * max(f[1], f[2])),
                  bool(max(r[3], r[4]) > 2 * max(f[3], f[4]))) for r in rows)


def select_k(cands: Sequence[BodyCandidate], K: int, *, cost_bloat: float = 0.35,
             anchor_first: bool = True,
             scale: np.ndarray | None = None) -> list[BodyCandidate]:
    """Cheapest first, then uncovered discrete signature, then farthest-point.

    Three rules, in that order, each answering an objection:

    1. `[0]` is the shipped planner's own answer when `anchor_first`, so the K=1 column of the
       gate is the current system for every baseline and no K-curve can look good by giving up
       the baseline it is supposed to extend.
    2. Cover distinct predicted DISCRETE signatures next.  A pure farthest-point rule optimises
       the continuous layer and will spend K=2..4 on three depths of one duck, because
       amplitude extremes maximise a Euclidean spread; the guidance makes discrete-mode recall
       the first quantity to report.
    3. Fill the rest by farthest-point (the greedy 2-approximation to max-min spread).

    The cost filter is on TOTAL traversal cost (`route_len + j_body`), not on body excess.
    That is not pedantry: on `partial_beam` the cheapest body has `j_body` EXACTLY 0 (the route
    goes around), so a ratio on the excess admits nothing but itself and K collapses to 1.
    `1.35` matches the filter EXP-005e used, and EXP-005e's own misses ran at median 1.005x,
    so it is loose rather than self-serving.
    """
    if not cands:
        return []
    sc = SEED_SIGMA_PRIOR if scale is None else np.asarray(scale, float)
    pool = [c for c in cands if c.cost_ratio <= 1.0 + cost_bloat + 1e-9]
    if len(pool) < min(K, len(cands)):          # always spend the budget if candidates exist
        have = {id(c) for c in pool}
        extra = [c for c in cands if id(c) not in have]
        pool = pool + sorted(extra, key=lambda c: c.cost_ratio)[:K - len(pool)]
    pool = sorted(pool, key=lambda c: (0 if c.origin == "astar" and anchor_first else 1,
                                       c.total_cost))
    chosen = [pool[0]]
    taken = {id(pool[0])}

    def vec(c: BodyCandidate) -> np.ndarray:
        p = np.asarray(c.predicted, float).reshape(-1, N_CHANNELS)
        return (p / sc).ravel()

    sigs = {_predicted_active(chosen[0].predicted)}
    for c in pool[1:]:
        if len(chosen) >= K:
            break
        s = _predicted_active(c.predicted)
        if s not in sigs:
            sigs.add(s)
            chosen.append(c)
            taken.add(id(c))
    while len(chosen) < K:
        best, bd = None, -1.0
        for c in pool:
            if id(c) in taken:
                continue
            n = min(_euclid(vec(c), vec(q)) for q in chosen)
            if n > bd:
                best, bd = c, n
        if best is None or bd <= 1e-9:
            break
        chosen.append(best)
        taken.add(id(best))
    return chosen[:K]


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance, zero-padding to the longer of the two."""
    n = max(len(a), len(b))
    x = np.zeros(n)
    y = np.zeros(n)
    x[:len(a)], y[:len(b)] = a, b
    return float(np.linalg.norm(x - y))


# =======================================================================================
# 12.  Evaluation through the frozen prior  (the only place ARDY is called)
# =======================================================================================

@dataclass(eq=False)
class Evaluation:
    """What ARDY did with one program, over several seeds."""

    program: ConstraintProgram
    origin: str
    seeds: list[int]
    deltas: np.ndarray                  # (n_seeds, n_interactions * N_CHANNELS)
    mu: np.ndarray                      # seed MEAN -- the guidance's mu_b
    feasible: list[bool]
    feasible_rate: float
    signatures: list[tuple]             # per seed: tuple of per-interaction active sets
    signature: tuple                    # modal signature
    stability: float
    j_body: float
    total_cost: float
    cost_ratio: float = 1.0
    overlap: bool = False        # rendered request commands two adaptation axes at once
    reports: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def addressable(self) -> bool:
        """Entry condition for the reference set: reliably feasible AND stable.

        The guidance pre-commits `Stability >= 0.8`: "a program that produces duck on half the
        seeds and duck+tuck on the other half is diverse, but it is not controllably
        addressable".  A candidate that fails this is not a body strategy, it is a coin flip.
        """
        return self.feasible_rate >= 0.75 and self.stability >= 0.8


class BodyEvaluator:
    """Generates candidates through frozen ARDY and scores them against a matched control.

    ONE control set per (scene, route) serves every candidate of every baseline, because every
    candidate shares `lat` exactly.  That is what makes `Delta m(b, z) = m(M(r,b,z)) -
    m(M(r,b_0,z))` well defined (guidance: same route, same seed), and it is why the whole
    module refuses to let the route drift.
    """

    def __init__(self, runner, ctx: RouteContext, seeds: Sequence[int] = (100, 101, 102),
                 *, prompt: str = PROMPT, diffusion_steps: int = 10,
                 goal_tol: float = GOAL_TOL, body=None, batch: int = 16,
                 half_width: float = 0.7):
        from .robot import G1Body
        self.runner, self.ctx = runner, ctx
        self.seeds = list(seeds)
        self.prompt, self.steps, self.goal_tol, self.batch = prompt, diffusion_steps, goal_tol, batch
        self.fps = runner.fps
        self.T = ctx.T
        self.body = body or G1Body(ctx.scene)
        self.interactions = ctx.interactions(half_width)
        self.n_ardy = 0                # CHARGED calls (what the cost table counts)
        self.n_unique = 0              # clips actually generated (cache hits are free)
        self.gpu_s = 0.0
        self._cache: dict[tuple, dict] = {}

        # `nominal`: one path-only clip.  `_limb_targets` edits THIS motion, so the adaptation
        # is a local edit of something the prior already produced for this exact path rather
        # than a pose invented from outside its manifold (planner.py:452-461).
        t0 = time.time()
        ref = runner.generate([prompt],
                              [plan_to_path_spec(ctx.route_plan, self.fps, ctx.speed,
                                                 duration=self.T / self.fps)],
                              self.T, diffusion_steps, seeds=[self.seeds[0]])[0]
        self.gpu_s += time.time() - t0
        self.n_ardy += 1
        self.n_unique += 1
        self.nominal = ref

        self.neutral = ConstraintProgram(lat=ctx.base.lat.copy(),
                                         slot=np.zeros((M_SLOT, N_AX)), speed=ctx.base.speed)
        self._ctrl: dict[int, tuple] = {}
        self.ensure_controls(self.seeds)

    def ensure_controls(self, seeds: Sequence[int]) -> None:
        """Generate the neutral-body control at every seed that will be scored against it.

        The matched delta is only defined seed-for-seed: `Delta m(b, z) = m(M(r,b,z)) -
        m(M(r,b_0,z))` (guidance: same route, SAME SEED).  Any arm that uses seeds outside the
        evaluator's default block -- the NULL-SEED control does, by construction, because its K
        copies need disjoint seeds -- must therefore pay for its own controls, or its
        descriptors carry an adapted-vs-control seed mismatch on top of the effect and the
        measured noise floor comes out inflated.  Inflating the noise floor would raise `eps`
        and make every baseline look worse, so this is not a harmless shortcut in either
        direction.
        """
        todo = [int(z) for z in dict.fromkeys(seeds) if int(z) not in self._ctrl]
        if not todo:
            return
        outs = self._generate([self.neutral] * len(todo), todo)
        for z, o in zip(todo, outs):
            q = self.runner.to_qpos(o)
            self._ctrl[z] = (q, [raw_descriptor(self.body, q, it, self.fps)
                                 for it in self.interactions],
                             envelope_series(self.body, q))

    @property
    def ctrl_q(self) -> list:
        return [self._ctrl[z][0] for z in self.seeds]

    # -- generation -------------------------------------------------------------------

    def _generate(self, progs: Sequence[ConstraintProgram], seeds: Sequence[int]) -> list[dict]:
        keys = [(np.round(p.slot, 5).tobytes(), int(s)) for p, s in zip(progs, seeds)]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        self.n_ardy += len(progs)
        for i in range(0, len(todo), self.batch):
            grp = todo[i:i + self.batch]
            specs = [decode(progs[j], self.ctx.scene, self.fps, self.nominal,
                            self.runner.joint_names, duration=self.T / self.fps) for j in grp]
            t0 = time.time()
            outs = self.runner.generate([self.prompt] * len(grp), specs, self.T, self.steps,
                                        seeds=[int(seeds[j]) for j in grp])
            self.gpu_s += time.time() - t0
            self.n_unique += len(grp)
            for j, o in zip(grp, outs):
                self._cache[keys[j]] = o
        return [self._cache[k] for k in keys]

    # -- scoring ----------------------------------------------------------------------

    def evaluate(self, programs: Sequence[ConstraintProgram],
                 origins: Sequence[str] | None = None,
                 seeds: Sequence[Sequence[int]] | None = None,
                 noise_q99: np.ndarray | None = None) -> list[Evaluation]:
        """Generate every (program, seed) and return one Evaluation per program.

        `seeds` may give a DIFFERENT seed block per program.  That exists for the NULL-SEED
        control arm, which submits the same program K times and must get K different seeds;
        every other caller leaves it None and shares the evaluator's seed block, which is what
        makes the matched control valid.
        """
        if not len(programs):
            return []
        origins = list(origins or ["?"] * len(programs))
        blocks = [list(self.seeds) for _ in programs] if seeds is None else [list(s) for s in seeds]
        flat_p, flat_s, owner = [], [], []
        for i, (p, sb) in enumerate(zip(programs, blocks)):
            for s in sb:
                flat_p.append(p)
                flat_s.append(s)
                owner.append(i)
        self.ensure_controls(flat_s)
        outs = self._generate(flat_p, flat_s)
        q99 = SEED_SIGMA_PRIOR * 2.33 if noise_q99 is None else np.asarray(noise_q99, float)
        per: dict[int, list] = {}
        for k, o in enumerate(outs):
            i = owner[k]
            ctrl_q, ctrl_desc, ctrl_env = self._ctrl[int(flat_s[k])]
            q = self.runner.to_qpos(o)
            rep = self.body.trajectory_report(q)
            goal = bool(np.linalg.norm(q[-1, :2] - np.asarray(self.ctx.scene.goal))
                        < self.goal_tol)
            env_a = envelope_series(self.body, q)
            d = []
            for it, cd in zip(self.interactions, ctrl_desc):
                ad = raw_descriptor(self.body, q, it, self.fps)
                # `env_adapted` / `env_control` are passed explicitly so `matched_delta` does
                # not recompute the control's whole-clip envelope once per interaction
                # (morphology.py:122-125 exists for exactly this).
                d.append(matched_delta(ad, cd, q, ctrl_q, it, self.fps, body=self.body,
                                       env_adapted=env_a, env_control=ctrl_env))
            per.setdefault(i, []).append(
                (np.concatenate(d), bool(goal and rep["collision_free"]), rep))
        out = []
        for i, p in enumerate(programs):
            rows = per.get(i, [])
            D = np.array([r[0] for r in rows])
            oks = [r[1] for r in rows]
            sigs = [tuple(active_set(row[j * N_CHANNELS:(j + 1) * N_CHANNELS], q99)
                          for j in range(len(self.interactions))) for row in D]
            from collections import Counter
            modal = Counter(sigs).most_common(1)[0][0] if sigs else ()
            j = j_body(p, self.T, self.fps, self.ctx.tube.length)
            out.append(Evaluation(program=p, origin=origins[i], seeds=blocks[i], deltas=D,
                                  mu=D.mean(axis=0) if len(D) else np.zeros(0),
                                  feasible=oks, feasible_rate=float(np.mean(oks)) if oks else 0.0,
                                  signatures=sigs, signature=modal,
                                  stability=stability(sigs), j_body=j,
                                  total_cost=self.ctx.tube.length + j,
                                  overlap=uncalibrated_overlap(p, self.T, self.fps),
                                  reports=[r[2] for r in rows]))
        c0 = max(min([e.total_cost for e in out]), 1e-9)
        for e in out:
            e.cost_ratio = e.total_cost / c0
        return out


# =======================================================================================
# 13.  BASELINE D -- continuous body-only refinement at a fixed route
# =======================================================================================

_AXIS_SIGMA = np.array([0.06, 0.06, 0.28, 0.28, 0.28])       # normalised [-1,1] slot units
SLOT_LO, SLOT_HI = N_LAT - 2, N_LAT - 2 + M_SLOT * N_AX      # 22, 42


def tube_seeds(ctx: RouteContext, n: int, rng: np.random.Generator,
               gratuitous: float = 0.25) -> list[ConstraintProgram]:
    """Tube-constrained stratified body programs: the seeding stage of baseline D.

    `s_mid` / `s_half` are pinned by the demand window (with timing jitter) and the intensity
    axes are stratified from the geometric REQUIREMENT up to the reachable ceiling.  That drops
    the effective dimension from 20 to ~3 per window and is what makes the arm feasible at all:
    uniform Sobol over the 20-D slot box was measured at 5/63 and 1/63 feasible clips, while
    the tube-constrained arm was 15/15 on the same two scenes at 16 ARDY calls.  So "uniform
    sampling is the honest cheap baseline" is true and it is not a viable baseline; this is.

    `gratuitous` puts a fraction of the sample on adaptations the geometry does NOT demand
    (a body held where nothing requires it), because that is where a site-based enumerator's
    own miss taxonomy said all of its misses live.
    """
    out = []
    wins = ctx.tube.windows or [(int(0.35 * ctx.T), int(0.65 * ctx.T))]
    L = max(ctx.tube.length, 1e-6)
    for _ in range(n):
        runs = []
        for (a, b) in wins:
            if rng.random() < 0.15 and len(wins) > 1:
                continue                                   # sometimes skip a window entirely
            d_req = float(ctx.tube.dip_req[a:b + 1].max())
            t_req = float(ctx.tube.tuck_req[a:b + 1].max())
            l_req = float(ctx.tube.lift_req[a:b + 1].max())
            dip = float(rng.uniform(d_req, DIP_MAX)) if d_req > ACTIVE["dip"] else (
                float(rng.uniform(0, DIP_MAX)) if rng.random() < gratuitous else 0.0)
            tuck = float(rng.uniform(t_req, TUCK_MAX)) if t_req > ACTIVE["tuck"] else (
                float(rng.uniform(0, TUCK_MAX)) if rng.random() < gratuitous else 0.0)
            lift = l_req if l_req > 0 else (LIFT_STEP_POINT if rng.random() < gratuitous / 2
                                            else 0.0)
            pad = float(rng.uniform(0.0, 1.5)) / L * max(ctx.T - 1, 1)
            shift = float(rng.uniform(-0.75, 0.75)) / L * max(ctx.T - 1, 1)
            runs.append((float(np.clip(a - pad + shift, 0, ctx.T - 1)),
                         float(np.clip(b + pad + shift, 0, ctx.T - 1)), (dip, tuck, lift)))
        if not runs:
            runs = [(0.0, float(ctx.T - 1), (0.0, 0.0, 0.0))]
        out.append(ConstraintProgram(lat=ctx.base.lat.copy(),
                                     slot=_slots_from_runs(runs, ctx.T),
                                     speed=ctx.base.speed))
    return out


def _iso_line(vi: np.ndarray, vj: np.ndarray, rng, iso: float = 0.12,
              line: float = 0.30) -> np.ndarray:
    """Iso+LineDD.  No covariance to adapt, which is why it is chosen over CMA-ME here:

    at ~64 evaluations CMA-ES has not adapted a 20x20 covariance (it needs 4+3ln(20) ~ 13
    samples per generation merely to begin) and degenerates to an expensive Gaussian, while
    Iso+LineDD's line term traces the dip<->tuck substitution ridge directly.
    """
    out = vi.copy()
    sig = np.tile(_AXIS_SIGMA, M_SLOT)
    out[SLOT_LO:SLOT_HI] += (iso * sig * rng.standard_normal(SLOT_HI - SLOT_LO)
                             + line * rng.standard_normal()
                             * (vj[SLOT_LO:SLOT_HI] - vi[SLOT_LO:SLOT_HI]))
    return np.clip(out, -1.0, 1.0)


def _structural(v: np.ndarray, rng) -> np.ndarray:
    """Cross a discrete active-set boundary on purpose.

    Gaussian mutation almost never flips layer 1: it has to push an axis across its activity
    threshold and the isotropic step is smaller than the gap.  Dropping or switching on a whole
    channel is the operator that reaches "duck instead of tuck", which is the axis EXP-005e
    says the corridor enumerator cannot see.
    """
    out = v.copy()
    s = out[SLOT_LO:SLOT_HI].reshape(M_SLOT, N_AX)
    live = [k for k in range(M_SLOT) if np.any(s[k, 2:] > -0.9)]
    k = int(rng.choice(live)) if live else int(rng.integers(M_SLOT))
    ax = 2 + int(rng.integers(3))
    r = rng.random()
    if r < 0.45:
        s[k, ax] = -1.0
    elif r < 0.85:
        s[k, ax] = float(rng.uniform(-0.4, 1.0))
    else:
        free = [j for j in range(M_SLOT) if not np.any(s[j, 2:] > -0.9)]
        if free:
            s[free[0]] = s[k]
            s[free[0], 0] = float(np.clip(s[k, 0] + rng.choice([-1, 1]) * rng.uniform(.1, .3),
                                          -1, 1))
    out[SLOT_LO:SLOT_HI] = s.ravel()
    return np.clip(out, -1.0, 1.0)


def _cell(ev: Evaluation, sigma: np.ndarray, bin_sigma: float, n_win: int,
          channels: Sequence[int]) -> tuple:
    """MAP-Elites key: discrete active set, then live channels binned in seed sigmas.

    Binned on the SAME channel subset the archive is whitened on.  Binning eight channels and
    whitening eight while declaring two of them uninformative is the inconsistency that let
    43 % of the selection distance come from channels the search itself called dead.
    """
    key = []
    for j in range(n_win):
        m = ev.mu[j * N_CHANNELS:(j + 1) * N_CHANNELS]
        row = [tuple(ev.signature[j]) if j < len(ev.signature) else ()]
        for c in channels:
            row.append(int(np.clip(np.floor(m[c] / max(bin_sigma * sigma[c], 1e-9)), -2, 12)))
        key.append(tuple(row))
    return tuple(key)


def refine_continuous(scene, route_plan: Plan, K: int = 8, runner=None, *,
                      ctx: RouteContext | None = None, fps: float = 25.0,
                      speed: float = SPEED, lattice: Sequence[Body] | None = None,
                      budget: int = 96, n_seeds: int = 3, batch: int = 16,
                      seed: int = 0, sigma: np.ndarray | None = None,
                      bin_sigma: float = 1.0, eps: float = 2.0, cost_bloat: float = 0.35,
                      channels: str = "all", p_structural: float = 0.30,
                      evaluator: BodyEvaluator | None = None, anchor_astar: bool = True,
                      return_evaluations: bool = False):
    """BASELINE D.  Continuous body-only refinement at a fixed route, VERIFIED as it goes.

    `budget` is in ARDY CLIPS, including the matched control and every replicate seed, because
    that is the currency the gate compares on.  A candidate is admitted to the archive only if
    it is feasible on EVERY seed and its descriptor is scored on the seed MEAN -- not on one
    realisation.  That is not fastidiousness: under an optimistic noise scale, 96 % of
    same-program different-seed pairs read as ">2 sigma apart", so a single-realisation archive
    would happily return "K=8 distinct bodies" for one program generated eight times.
    """
    if runner is None:
        raise ValueError("baseline D calls ARDY; pass a runner")
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    rng = np.random.default_rng(seed)
    sig = SEED_SIGMA_PRIOR if sigma is None else np.asarray(sigma, float)
    chan = CHANNEL_SETS[channels]
    ev = evaluator or BodyEvaluator(runner, ctx, seeds=[100 + i for i in range(n_seeds)],
                                    batch=batch)
    n_win = len(ev.interactions)
    W = _block_whitener(sig, n_win, chan)

    archive: dict[tuple, Evaluation] = {}
    evaluated: list[Evaluation] = []

    def admit(evals):
        for e in evals:
            evaluated.append(e)
            if e.feasible_rate < 1.0:
                continue
            c = _cell(e, sig, bin_sigma, n_win, chan)
            cur = archive.get(c)
            if cur is None or e.j_body < cur.j_body:
                archive[c] = e

    per_cand = max(n_seeds, 1)
    init = []
    if anchor_astar:
        init.append(astar_program(ctx))
    init += tube_seeds(ctx, max(0, min(12, (budget - ev.n_ardy) // per_cand - 1)), rng)
    if init:
        tags = (["astar"] + ["tube"] * (len(init) - 1)) if anchor_astar else ["tube"] * len(init)
        admit(ev.evaluate(init, tags))
    while ev.n_ardy + per_cand <= budget:
        n = max(1, min((budget - ev.n_ardy) // per_cand, max(1, batch // per_cand)))
        parents = list(archive.values()) or evaluated
        if not parents:
            break
        vs = [p.program.to_vec() for p in parents]
        props, tags = [], []
        for _ in range(n):
            i = int(rng.integers(len(vs)))
            if rng.random() < p_structural:
                props.append(_structural(vs[i], rng))
                tags.append("structural")
            else:
                props.append(_iso_line(vs[i], vs[int(rng.integers(len(vs)))], rng))
                tags.append("iso_line")
        admit(ev.evaluate([_fix_route(ConstraintProgram.from_vec(v), ctx) for v in props], tags))

    elites = sorted(archive.values(), key=lambda e: e.j_body)
    sel = select_k_measured(elites, K, W, eps=eps, cost_bloat=cost_bloat,
                            anchor=astar_program(ctx) if anchor_astar else None)
    if return_evaluations:
        return sel, {"archive": archive, "evaluated": evaluated, "evaluator": ev}
    return [e.program for e in sel]


def _fix_route(prog: ConstraintProgram, ctx: RouteContext) -> ConstraintProgram:
    """Restore the fixed route after any operation in the normalised 43-vector."""
    return ConstraintProgram(lat=ctx.base.lat.copy(), slot=prog.slot, speed=ctx.base.speed)


def _block_whitener(sigma: np.ndarray, n_blocks: int,
                    channels: Sequence[int]) -> np.ndarray:
    """(n_blocks*8, n_blocks*8) diagonal whitener, RMS-normalised over interactions.

    Channels outside `channels` get weight 0, so the bins and the distance agree.  The
    1/sqrt(n_blocks) is what stops a three-obstacle scene from looking more diverse than a
    one-obstacle scene purely because it has more blocks: without it the same 2-sigma
    threshold means "2 sigma" on one and "1.15 sigma per window" on the other.
    """
    d = np.zeros(N_CHANNELS)
    for c in channels:
        d[c] = 1.0 / max(sigma[c], 1e-9)
    return np.diag(np.tile(d, n_blocks)) / np.sqrt(max(n_blocks, 1))


def select_k_measured(evals: Sequence[Evaluation], K: int, W: np.ndarray, *,
                      eps: float = 2.0, cost_bloat: float = 0.35,
                      anchor: ConstraintProgram | None = None) -> list[Evaluation]:
    """Cheapest first (anchored on A* if it survived), then farthest-point in seed sigmas."""
    if not evals:
        return []
    pool = [e for e in evals if e.cost_ratio <= 1.0 + cost_bloat + 1e-9] or list(evals)
    pool = sorted(pool, key=lambda e: e.j_body)
    first = 0
    if anchor is not None:
        ak = np.round(anchor.slot, 4).tobytes()
        for i, e in enumerate(pool):
            if np.round(e.program.slot, 4).tobytes() == ak:
                first = i
                break
    chosen = [pool[first]]
    taken = {first}
    while len(chosen) < K:
        best, bd = None, -1.0
        for i, e in enumerate(pool):
            if i in taken:
                continue
            d = min(d_morph(e.mu, q.mu, W) for q in chosen)
            if d > bd:
                best, bd = i, d
        if best is None or bd <= eps:
            break
        chosen.append(pool[best])
        taken.add(best)
    return chosen


# =======================================================================================
# 14.  COMPOSITE -- k-best body skeletons seeded into continuous refinement
# =======================================================================================

def enumerate_composite(scene, route_plan: Plan, K: int = 8, runner=None, *,
                        ctx: RouteContext | None = None, fps: float = 25.0,
                        speed: float = SPEED, lattice: Sequence[Body] | None = None,
                        budget: int = 96, n_seeds: int = 3, batch: int = 16, seed: int = 0,
                        sigma: np.ndarray | None = None, eps: float = 2.0,
                        bin_sigma: float = 1.0, cost_bloat: float = 0.35,
                        channels: str = "all", n_skeleton: int = 16,
                        evaluator: BodyEvaluator | None = None,
                        return_evaluations: bool = False):
    """THE STRONGEST CLASSICAL BASELINE the guidance names: k-best skeletons + refinement.

    Stage 1 (CPU, zero ARDY calls): union the skeleton sets of A, B and C, deduplicate on the
    proposal-side surrogate descriptor, and keep the `n_skeleton` most spread.  This is where
    the DISCRETE alternatives come from -- a different body at an interaction, a whole-route
    floor, a split adaptation -- and it costs milliseconds.

    Stage 2 (ARDY): verify those skeletons at `n_seeds` seeds each, then spend the remaining
    budget refining CONTINUOUSLY around the ones that survived, using D's emitters seeded from
    the verified elites rather than from a blind sample.  A skeleton that ARDY refuses is not
    a point of F_B and must not seed anything.

    The split matters because the two stages fail differently: the lattice cannot express an
    off-lattice amplitude (A* must round a 0.383 m dip requirement up to `duck_max`'s 0.500 m,
    a 12 cm over-crouch), and continuous refinement cannot find a different discrete strategy
    (a Gaussian step essentially never crosses an active-set boundary).
    """
    if runner is None:
        raise ValueError("the composite calls ARDY; pass a runner")
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    rng = np.random.default_rng(seed)
    sig = SEED_SIGMA_PRIOR if sigma is None else np.asarray(sigma, float)
    chan = CHANNEL_SETS[channels]

    sks = (kbest_skeletons(ctx, max_skeletons=200)
           + nogood_skeletons(ctx) + mode_floor_skeletons(ctx))
    cands = _expand(ctx, sks, TIMING_PADS_M, TIMING_SHIFTS_M, "composite")
    cands = _prepend_astar(ctx, cands)
    skeleton_progs = [c.program for c in select_k(cands, n_skeleton, cost_bloat=cost_bloat)]

    ev = evaluator or BodyEvaluator(runner, ctx, seeds=[100 + i for i in range(n_seeds)],
                                    batch=batch)
    n_win = len(ev.interactions)
    W = _block_whitener(sig, n_win, chan)
    per_cand = max(n_seeds, 1)
    room = max(0, (budget - ev.n_ardy) // per_cand)
    stage1 = skeleton_progs[:max(1, min(len(skeleton_progs), room - room // 3))]
    evals = ev.evaluate(stage1, ["skeleton"] * len(stage1))

    archive: dict[tuple, Evaluation] = {}
    for e in evals:
        if e.feasible_rate >= 1.0:
            archive.setdefault(_cell(e, sig, bin_sigma, n_win, chan), e)
    while ev.n_ardy + per_cand <= budget:
        n = max(1, min((budget - ev.n_ardy) // per_cand, max(1, batch // per_cand)))
        parents = list(archive.values()) or [e for e in evals if e.feasible_rate > 0] or evals
        vs = [p.program.to_vec() for p in parents]
        props, tags = [], []
        for _ in range(n):
            i = int(rng.integers(len(vs)))
            if rng.random() < 0.3:
                props.append(_structural(vs[i], rng))
                tags.append("refine_structural")
            else:
                props.append(_iso_line(vs[i], vs[int(rng.integers(len(vs)))], rng))
                tags.append("refine_iso")
        new = ev.evaluate([_fix_route(ConstraintProgram.from_vec(v), ctx) for v in props], tags)
        evals += new
        for e in new:
            if e.feasible_rate >= 1.0:
                c = _cell(e, sig, bin_sigma, n_win, chan)
                if c not in archive or e.j_body < archive[c].j_body:
                    archive[c] = e

    elites = sorted(archive.values(), key=lambda e: e.j_body) or sorted(
        [e for e in evals if e.feasible_rate > 0], key=lambda e: e.j_body)
    sel = select_k_measured(elites, K, W, eps=eps, cost_bloat=cost_bloat,
                            anchor=astar_program(ctx))
    if return_evaluations:
        return sel, {"archive": archive, "evaluated": evals, "evaluator": ev}
    return [e.program for e in sel]


# =======================================================================================
# 15.  The reference proposer, and the NULL control arm
# =======================================================================================

def reference_random_restart(scene, route_plan: Plan, K: int = 32, runner=None, *,
                             ctx: RouteContext | None = None, fps: float = 25.0,
                             speed: float = SPEED, lattice: Sequence[Body] | None = None,
                             seed: int = 0) -> list[ConstraintProgram]:
    """A deliberately expensive body search that shares NO rule with any baseline.

    Its job is to be able to reach points of F_B that every baseline's construction excludes,
    so that the reference set is not the union of the things the baselines happen to find.
    It samples, per restart, INDEPENDENTLY of cost:
      * a random whole-route posture floor (including "none"),
      * per window, an OFF-LATTICE continuous (dip, tuck, lift) drawn above the requirement,
      * a random window split, so a body may vary within one interaction,
      * random gratuitous adaptations where the geometry demands none,
      * random onset/offset up to +-1.5 m of travel.
    This mirrors `exp005e.reference_signatures` -- random exclusions, random mode subsets --
    moved from ROUTES to BODIES, which is the axis EXP-005e showed the misses live on.
    """
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    rng = np.random.default_rng(seed)
    L = max(ctx.tube.length, 1e-6)
    denom = max(ctx.T - 1, 1)
    wins = ctx.tube.windows or [(int(0.35 * ctx.T), int(0.65 * ctx.T))]
    out: dict[bytes, ConstraintProgram] = {}
    for _ in range(K * 4):
        runs = []
        if rng.random() < 0.25:                                    # whole-route floor
            runs.append((0.0, float(denom), (float(rng.uniform(0, DIP_MAX)) * (rng.random() < .6),
                                             float(rng.uniform(0, TUCK_MAX)) * (rng.random() < .4),
                                             LIFT_STEP_POINT * (rng.random() < .2))))
        for (a, b) in wins:
            if rng.random() < 0.12:
                continue
            d_req = float(ctx.tube.dip_req[a:b + 1].max())
            t_req = float(ctx.tube.tuck_req[a:b + 1].max())
            l_req = float(ctx.tube.lift_req[a:b + 1].max())
            n_split = 2 if (rng.random() < 0.3 and b - a >= 8) else 1
            edges = np.linspace(a, b, n_split + 1)
            for k in range(n_split):
                dip = float(rng.uniform(d_req, DIP_MAX)) if (d_req > ACTIVE["dip"] or
                                                             rng.random() < 0.3) else 0.0
                tuck = float(rng.uniform(t_req, TUCK_MAX)) if (t_req > ACTIVE["tuck"] or
                                                               rng.random() < 0.25) else 0.0
                lift = l_req if l_req > 0 else (LIFT_STEP_POINT if rng.random() < 0.15 else 0.0)
                pad = float(rng.uniform(0, 1.5)) / L * denom
                shift = float(rng.uniform(-1.5, 1.5)) / L * denom
                runs.append((float(np.clip(edges[k] - pad + shift, 0, denom)),
                             float(np.clip(edges[k + 1] + pad + shift, 0, denom)),
                             (dip, tuck, lift)))
        if not runs:
            continue
        p = ConstraintProgram(lat=ctx.base.lat.copy(), slot=_slots_from_runs(runs, ctx.T),
                              speed=ctx.base.speed)
        out.setdefault(np.round(p.slot, 4).tobytes(), p)
        if len(out) >= K:
            break
    return list(out.values())[:K]


def null_seed_baseline(scene, route_plan: Plan, K: int = 8, runner=None, *,
                       ctx: RouteContext | None = None, fps: float = 25.0,
                       speed: float = SPEED, lattice: Sequence[Body] | None = None
                       ) -> list[ConstraintProgram]:
    """THE CONTROL ARM: the SAME program, K times.  Its coverage is the noise floor.

    The gate must generate candidate k of this arm at a DIFFERENT seed block (see
    `BodyEvaluator.evaluate(seeds=...)`).  Whatever discrete-mode recall and MorphRecall@8 this
    arm scores is what a proposer earns for emitting nothing at all, so any baseline's number
    is only meaningful above it.  If this arm scores materially above zero, `eps` is below the
    measured noise floor and every coverage number in the table is measuring ARDY's sampler.
    """
    ctx = ctx or route_context(scene, route_plan, fps=fps, speed=speed, lattice=lattice)
    if not ctx.feasible:
        return []
    return [astar_program(ctx) for _ in range(K)]


BASELINES: dict[str, Callable] = {
    "A-KBEST": enumerate_kbest,
    "B-NOGOOD": enumerate_nogood,
    "C-WSWEEP": enumerate_weight_sweep,
    "D-REFINE": refine_continuous,
    "COMPOSITE": enumerate_composite,
    "NULL-SEED": null_seed_baseline,
}

# Which baselines spend ARDY calls to PROPOSE (as opposed to only to verify).
SPENDS_ARDY = {"D-REFINE", "COMPOSITE"}
