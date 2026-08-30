"""The aligned clearance trace: what the generated motion actually cleared, per route position.

`G1Body.trajectory_report` returns one scalar minimum over the whole clip. That is enough to
say "this collided" and useless for saying WHERE, which is what a local repair needs. This
produces clearance as a function of route position on the SAME 64-sample grid the schedule
lives on, so a deficit at sample i maps to a command at sample i with no re-indexing.

Three things are kept apart, and must stay apart:

    collision            measured clearance < 0 anywhere -- the body intersected the scene
    overhead deficit     headroom below the target margin. THIS is what a duck repair acts on
    lateral deficit      side clearance below the target. A route problem, not a body problem

The third distinction is not decorative. Measured on the m018 system, a 3-beam scene reported
a total minimum clearance of 76 mm against a 180 mm target -- while its overhead clearance was
341 mm. The robot was squeezing past a wall, not grazing a beam. A repair loop driven by the
undifferentiated minimum would have answered that by crouching, which buys exactly nothing.

`MARGIN_M` is likewise an OVERHEAD margin: the scheduler solves `z >= g_inv(c - margin)` with
`c` the route profile's overhead channel. Judging it against a whole-scene minimum would fail
routes the system was never asked to keep clear -- the shipped single-beam demo clips sit at
0.13-0.17 m of total clearance purely from corridor width.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..learn.route_profile import N_SAMPLES
from ..robot import G1Body
from ..scenes import Scene


def schedule_hash(q) -> str:
    """Identity of a command schedule, rounded so float noise cannot fork the cache.

    Every provenance record and every cache key that claims to describe a schedule uses this,
    so "repaired" can never name a clip generated from the pre-repair commands.
    """
    return hashlib.sha1(np.round(np.asarray(q, float), 5).tobytes()).hexdigest()[:16]


@dataclass
class ClearanceTrace:
    s_m: np.ndarray            # (N,) route distance of each sample
    clearance: np.ndarray      # (N,) minimum clearance to ANY scene geom
    overhead: np.ndarray       # (N,) minimum clearance to geometry above the body
    min_clearance_m: float     # whole-clip minimum over everything (matches trajectory_report)
    min_overhead_m: float
    collision_free: bool
    goal_error_m: float
    culprits: tuple = ()

    def deficit(self, target_m: float) -> np.ndarray:
        """e(s) = max(0, target - overhead(s)). The headroom a duck repair can buy back."""
        return np.maximum(target_m - self.overhead, 0.0)

    def lateral_deficit(self, target_m: float) -> np.ndarray:
        """Where the body is tight against something a duck cannot help with."""
        return np.maximum(target_m - self.clearance, 0.0) * (self.deficit(target_m) <= 1e-9)

    def below_margin(self, target_m: float) -> bool:
        return bool(self.deficit(target_m).max() > 1e-9)

    def to_dict(self, target_m: float) -> dict:
        e, lat = self.deficit(target_m), self.lateral_deficit(target_m)
        return {"min_clearance_m": round(self.min_clearance_m, 5),
                "min_overhead_m": round(self.min_overhead_m, 5),
                "collision_free": self.collision_free,
                "goal_error_m": round(self.goal_error_m, 4),
                "target_m": target_m,
                "meets_target": not self.below_margin(target_m),
                "max_deficit_m": round(float(e.max()), 5),
                "deficit_support": int((e > 1e-9).sum()),
                "max_lateral_deficit_m": round(float(lat.max()), 5),
                "culprits": list(self.culprits),
                "overhead": np.round(self.overhead, 4).tolist(),
                "clearance": np.round(self.clearance, 4).tolist()}


def clearance_trace(scene: Scene, qpos: np.ndarray, xy: np.ndarray,
                    n: int = N_SAMPLES, body: G1Body | None = None) -> ClearanceTrace:
    """Per-route-position clearance of an actual generated motion, overhead split out.

    Frames are mapped to route position by the robot's own along-route travel rather than by
    frame index, because the prior does not traverse the path at a constant rate -- it
    accelerates from rest and settles at the goal. Indexing by frame would shift the deficit
    relative to the schedule by a variable amount and put the repair in the wrong place.
    """
    body = body or G1Body(scene)
    rep = body.trajectory_report(qpos)
    per_frame = np.asarray(rep["per_frame_min_clearance"], float)
    per_over = np.asarray(rep["per_frame_overhead_clearance"], float)

    xy = np.asarray(xy, float).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total = float(seg.sum()) or 1.0
    p = np.asarray(qpos)[:, :2]
    travelled = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    travelled = np.clip(travelled, 0.0, total)

    want = np.linspace(0.0, total, n)
    nf = len(per_frame)
    # Worst clearance in the frame window belonging to each sample, so a brief dip between
    # sample centres is not missed by point sampling.
    edges = np.clip(np.searchsorted(
        travelled, np.concatenate([[0.0], (want[1:] + want[:-1]) / 2, [total]])), 0, nf)
    frame_of = np.clip(np.searchsorted(travelled, want), 0, nf - 1)
    clr, ovr = np.empty(n), np.empty(n)
    for i in range(n):
        # The window must be non-empty AND inside the clip: `travelled` saturates at the route
        # length once the robot stops advancing, so the last several samples can all map past
        # the final frame and produce an empty slice.
        a = int(min(edges[i], nf - 1))
        b = int(min(max(edges[i + 1], a + 1), nf))
        clr[i] = float(per_frame[a:b].min())
        ovr[i] = float(per_over[a:b].min())

    return ClearanceTrace(
        s_m=want, clearance=clr, overhead=ovr,
        min_clearance_m=float(rep["min_clearance_m"]),
        min_overhead_m=float(per_over.min()),
        collision_free=bool(rep["collision_free"]),
        goal_error_m=float(np.linalg.norm(np.asarray(qpos)[-1, :2] - np.asarray(scene.goal))),
        culprits=tuple(rep.get("culprit_geoms", ())),
    )
