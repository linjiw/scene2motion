"""Generate -> verify -> repair -> accept/reject, with provenance for every attempt.

The contract this file exists to enforce: a clip labelled "repaired" was regenerated from the
repaired schedule and independently reverified. Nothing else may claim that word.

Two mechanisms hold it. The cache key includes the schedule hash and the repair iteration, so
a pre-repair clip cannot be served for a post-repair request even when the route, seed, model
and scene are identical. And every attempt is recorded with the hash of the schedule that
produced it, so the claim is checkable after the fact rather than trusted.

Outcomes are three-valued, not two:

    accepted            collision-free AND meets the target margin
    accepted_margin     collision-free, below the target margin, no repair budget left
    rejected            collision, or still colliding after MAX_REPAIRS
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..demo.ardy_runner import (DEFAULT_SEED, DIFFUSION_STEPS, MAX_DURATION_S, PROMPT, SPEED,
                                get_runner, program_bytes)
from ..demo.cache import ClipCache, key_for
from ..learn.predictor import spec_from_dip
from ..optim.response import DIP_MAX
from ..optim.scheduler import MARGIN_M
from ..planner import Plan, plan_to_path_spec
from ..scenes import Scene
from .repair import MAX_REPAIRS, repair
from .trace import ClearanceTrace, clearance_trace, schedule_hash


@dataclass
class Attempt:
    iteration: int
    schedule_hash: str
    source: str                # 'cache' or 'generated'
    key: str
    trace: ClearanceTrace
    peak_dip_m: float
    gen_s: float

    def to_dict(self, target_m: float) -> dict:
        return {"iteration": self.iteration, "schedule_hash": self.schedule_hash,
                "source": self.source, "clip_key": self.key,
                "peak_dip_m": round(self.peak_dip_m, 4), "gen_s": round(self.gen_s, 2),
                **self.trace.to_dict(target_m)}


@dataclass
class LoopResult:
    scene_id: str
    outcome: str = "unverified"
    attempts: list[Attempt] = field(default_factory=list)
    repairs: list = field(default_factory=list)
    # Generations the method REQUIRED (one per attempt), and the subset that actually hit
    # the GPU. The first is the method's cost; the second is a fact about this cache.
    ardy_calls: int = 0
    cache_hits: int = 0
    reason: str = ""
    provenance: dict = field(default_factory=dict)
    # The command schedule the FINAL attempt was generated from. Cost metrics are charged
    # against this, not against the proposal, so a repaired clip pays for the dip it added.
    q_final: np.ndarray | None = None

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]

    @property
    def repaired(self) -> bool:
        """True only if the final clip came from a schedule that a repair produced."""
        return bool(self.repairs) and self.final.schedule_hash == self.repairs[-1].q_after_hash

    def to_dict(self, target_m: float = MARGIN_M) -> dict:
        return {"scene_id": self.scene_id, "outcome": self.outcome, "reason": self.reason,
                # 2 per attempt: the path reference clip and the adapted request.
                "ardy_calls": 2 * len(self.attempts), "ardy_calls_executed": self.ardy_calls,
                "cache_hits": self.cache_hits, "n_attempts": len(self.attempts),
                "repaired": self.repaired,
                "n_repairs": len(self.repairs), "target_m": target_m,
                "attempts": [a.to_dict(target_m) for a in self.attempts],
                "repairs": [r.to_dict() for r in self.repairs],
                "provenance": self.provenance}


def _n_frames(p: Plan, fps: float, speed: float) -> int:
    length = float(np.linalg.norm(np.diff(p.xy, axis=0), axis=1).sum())
    return min(int(MAX_DURATION_S * fps), max(int(2 * fps), int(round(length / speed * fps))))


def generate_from_schedule(scene: Scene, p: Plan, q: np.ndarray, cache: ClipCache,
                           *, iteration: int, preference: str, seed: int = DEFAULT_SEED,
                           speed: float = SPEED, allow_generate: bool = True) -> dict:
    """Run the frozen prior on one specific command schedule.

    The cache key carries `schedule_hash(q)` and the repair iteration. Both matter: the hash
    stops a differently-scheduled clip being reused, and the iteration keeps attempt 0 and
    attempt 1 distinguishable in the cache even in the degenerate case where a repair is a
    no-op, so "which attempt produced this file" is never ambiguous.
    """
    runner = get_runner()
    fps = runner.fps
    T = _n_frames(p, fps, speed)
    qh = schedule_hash(q)
    key = key_for(scene_id=scene.scene_id, preference=f"{preference}|verify|it{iteration}|{qh}",
                  program_bytes=program_bytes(p, scene, fps) + np.round(q, 5).tobytes(),
                  model=runner.model_name, seed=seed, steps=DIFFUSION_STEPS, fps=fps, n_frames=T)
    hit = cache.get(key)
    if hit is not None:
        return {"qpos": hit[0], "source": "cache", "key": key, "gen_s": 0.0,
                "schedule_hash": qh, "meta": hit[1]}
    if not allow_generate:
        return {"qpos": None, "source": "miss", "key": key, "gen_s": 0.0, "schedule_hash": qh}

    t0 = time.time()
    ref = runner.generate([PROMPT], [plan_to_path_spec(p, fps, speed, duration=T / fps)],
                          T, DIFFUSION_STEPS, seeds=[seed])[0]
    spec = spec_from_dip(scene, p.xy, q * DIP_MAX, fps, speed=speed, duration=T / fps)
    out = runner.generate([PROMPT], [spec], T, DIFFUSION_STEPS, seeds=[seed])[0]
    qpos = runner.to_qpos(out)
    gen_s = time.time() - t0
    meta = {"scene_id": scene.scene_id, "preference": preference, "repair_iteration": iteration,
            "schedule_hash": qh, "model": runner.model_name, "seed": seed,
            "steps": DIFFUSION_STEPS, "fps": fps, "n_frames": T,
            "peak_dip_m": round(float(q.max() * DIP_MAX), 4), "gen_s": round(gen_s, 2)}
    cache.put(key, qpos, meta)
    return {"qpos": qpos, "source": "generated", "key": key, "gen_s": gen_s,
            "schedule_hash": qh, "meta": meta}


def run(scene: Scene, p: Plan, q0: np.ndarray, resp, cache: ClipCache, *,
        preference: str = "clearance", target_m: float = MARGIN_M, seed: int = DEFAULT_SEED,
        speed: float = SPEED, max_repairs: int = MAX_REPAIRS,
        allow_generate: bool = True, provenance: dict | None = None,
        lead_taus: float | None = None) -> LoopResult:
    """The full loop. At most `max_repairs` corrections, then a verdict."""
    res = LoopResult(scene_id=scene.scene_id, provenance=dict(provenance or {}))
    res.provenance.update({"initial_schedule_hash": schedule_hash(q0), "seed": seed,
                           "steps": DIFFUSION_STEPS, "target_m": target_m,
                           "preference": preference, "speed_mps": speed,
                           "max_repairs": max_repairs})
    q = np.clip(np.asarray(q0, float), 0.0, 1.0)

    for it in range(max_repairs + 1):
        g = generate_from_schedule(scene, p, q, cache, iteration=it, preference=preference,
                                   seed=seed, speed=speed, allow_generate=allow_generate)
        if g["qpos"] is None:
            res.outcome, res.reason = "unverified", "clip not cached and generation disabled"
            return res
        if g["source"] == "generated":
            res.ardy_calls += 2          # the path reference and the adapted request
        else:
            res.cache_hits += 1
        tr = clearance_trace(scene, g["qpos"], p.xy)
        res.attempts.append(Attempt(iteration=it, schedule_hash=g["schedule_hash"],
                                    source=g["source"], key=g["key"], trace=tr,
                                    peak_dip_m=float(q.max() * DIP_MAX), gen_s=g["gen_s"]))

        if tr.collision_free and not tr.below_margin(target_m):
            res.outcome = "accepted"
            res.reason = f"collision-free, meets {target_m:.2f} m target"
            break
        if it == max_repairs:
            if tr.collision_free:
                res.outcome = "accepted_margin"
                res.reason = (f"collision-free but {tr.deficit(target_m).max()*1000:.0f} mm "
                              f"short of the {target_m:.2f} m target after {it} repair(s)")
            else:
                res.outcome = "rejected"
                res.reason = (f"still colliding ({tr.min_clearance_m*1000:.0f} mm) after "
                              f"{it} repair(s); select around this route")
            break
        q, step = repair(q, tr.deficit(target_m), resp, tr.s_m, speed, it + 1,
                         lead_taus=lead_taus)
        res.repairs.append(step)

    res.q_final = q
    res.provenance["final_schedule_hash"] = res.final.schedule_hash
    res.provenance["dip_initial_m"] = np.round(np.clip(q0, 0, 1) * DIP_MAX, 4).tolist()
    res.provenance["dip_final_m"] = np.round(q * DIP_MAX, 4).tolist()
    return res
