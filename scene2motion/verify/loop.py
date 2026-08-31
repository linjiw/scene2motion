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
from ..planner import Plan
from ..scenes import Scene
from .repair import MAX_REPAIRS, repair
from .trace import ClearanceTrace, clearance_trace, schedule_hash

# Cache identity for the frozen G1 checkpoint. Resolve the public alias without constructing
# the CUDA model, so cache-only inventory does not contend for the GPU it explicitly avoids.
VERIFY_MODEL_ALIAS = "g1"
VERIFY_FPS = 25.0


@dataclass
class Attempt:
    iteration: int
    schedule_hash: str
    source: str                # 'cache' or 'generated'
    key: str
    trace: ClearanceTrace
    peak_dip_m: float
    gen_s: float
    seed: int | None = None

    def to_dict(self, target_m: float) -> dict:
        return {"iteration": self.iteration, "schedule_hash": self.schedule_hash,
                "source": self.source, "clip_key": self.key,
                "seed": self.seed,
                "peak_dip_m": round(self.peak_dip_m, 4), "gen_s": round(self.gen_s, 2),
                **self.trace.to_dict(target_m)}


@dataclass
class LoopResult:
    scene_id: str
    outcome: str = "unverified"
    attempts: list[Attempt] = field(default_factory=list)
    repairs: list = field(default_factory=list)
    # Cache-dependent execution counters. One candidate attempt is now one ARDY generation;
    # noise-stream v1 also generated an unused path-reference clip per attempt. That historical
    # cost remains exposed separately by ``legacy_ardy_calls`` in :meth:`to_dict`.
    ardy_calls: int = 0
    adapted_generations_executed: int = 0
    cache_hits: int = 0
    reason: str = ""
    provenance: dict = field(default_factory=dict)
    # The command schedule the FINAL attempt was generated from. Cost metrics are charged
    # against this, not against the proposal, so a repaired clip pays for the dip it added.
    q_final: np.ndarray | None = None
    # The dip schedule (metres) behind EVERY attempt, in order. The demo plots these, so what
    # the reader sees is what was generated from rather than a reconstruction.
    dips_m: list = field(default_factory=list)
    # Resampling records attempts chronologically but may select an earlier, better clip.
    # None preserves the repair loop's historical convention that the last attempt is final.
    selected_attempt: int | None = None

    @property
    def final(self) -> Attempt:
        if self.selected_attempt is None:
            return self.attempts[-1]
        return self.attempts[self.selected_attempt]

    @property
    def repaired(self) -> bool:
        """True only if the final clip came from a schedule that a repair produced."""
        return bool(self.repairs) and self.final.schedule_hash == self.repairs[-1].q_after_hash

    def to_dict(self, target_m: float = MARGIN_M) -> dict:
        selected = (self.selected_attempt if self.selected_attempt is not None
                    else (len(self.attempts) - 1 if self.attempts else None))
        return {"scene_id": self.scene_id, "outcome": self.outcome, "reason": self.reason,
                # Logical method cost is independent of cache state; executed is diagnostic.
                "ardy_calls": len(self.attempts), "ardy_calls_executed": self.ardy_calls,
                "legacy_ardy_calls": 2 * len(self.attempts),
                "necessary_adapted_generations": len(self.attempts),
                "adapted_generations_executed": self.adapted_generations_executed,
                "cache_hits": self.cache_hits, "n_attempts": len(self.attempts),
                "selected_attempt": selected,
                "repaired": self.repaired,
                "n_repairs": len(self.repairs), "target_m": target_m,
                "attempts": [a.to_dict(target_m) for a in self.attempts],
                "dips_m": self.dips_m,
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
    from ardy.model.registry import resolve_model_name
    fps = VERIFY_FPS
    model_name = resolve_model_name(VERIFY_MODEL_ALIAS)
    T = _n_frames(p, fps, speed)
    qh = schedule_hash(q)
    key = key_for(scene_id=scene.scene_id, preference=f"{preference}|verify|it{iteration}|{qh}",
                  program_bytes=program_bytes(p, scene, fps) + np.round(q, 5).tobytes(),
                  model=model_name, seed=seed, steps=DIFFUSION_STEPS, fps=fps, n_frames=T)
    hit = cache.get(key)
    if hit is not None:
        return {"qpos": hit[0], "source": "cache", "key": key, "gen_s": 0.0,
                "schedule_hash": qh, "meta": hit[1]}
    if not allow_generate:
        return {"qpos": None, "source": "miss", "key": key, "gen_s": 0.0, "schedule_hash": qh}

    # Loading the prior is intentionally after the cache lookup. A cache hit and every
    # `allow_generate=False` request remain CPU-only.
    runner = get_runner()
    if runner.model_name != model_name or abs(float(runner.fps) - fps) > 1e-9:
        raise RuntimeError(f"cache identity {model_name}@{fps} does not match runner "
                           f"{runner.model_name}@{runner.fps}")
    t0 = time.time()
    spec = spec_from_dip(scene, p.xy, q * DIP_MAX, fps, speed=speed, duration=T / fps)
    out = runner.generate([PROMPT], [spec], T, DIFFUSION_STEPS, seeds=[seed])[0]
    qpos = runner.to_qpos(out)
    gen_s = time.time() - t0
    meta = {"scene_id": scene.scene_id, "preference": preference, "repair_iteration": iteration,
            "schedule_hash": qh, "model": runner.model_name, "seed": seed,
            "noise_stream_version": runner.noise_stream_version,
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
                           "max_repairs": max_repairs,
                           "max_adapted_generation_budget": max_repairs + 1,
                           "feedback_mode": "verification_guided_repair",
                           "attempt_seeds_planned": [seed] * (max_repairs + 1)})
    q = np.clip(np.asarray(q0, float), 0.0, 1.0)

    for it in range(max_repairs + 1):
        g = generate_from_schedule(scene, p, q, cache, iteration=it, preference=preference,
                                   seed=seed, speed=speed, allow_generate=allow_generate)
        if g["qpos"] is None:
            res.outcome, res.reason = "unverified", "clip not cached and generation disabled"
            if res.attempts:
                # The newly repaired schedule was never generated. Keep the last verified
                # candidate selected and never attach its metrics to the missing schedule.
                res.q_final = np.asarray(res.dips_m[-1], float) / DIP_MAX
                res.provenance["selected_attempt"] = len(res.attempts) - 1
                res.provenance["selected_seed"] = res.final.seed
                res.provenance["attempt_seeds_executed"] = [a.seed for a in res.attempts]
                res.provenance["final_schedule_hash"] = res.final.schedule_hash
                res.provenance["dip_initial_m"] = np.round(
                    np.clip(q0, 0, 1) * DIP_MAX, 4).tolist()
                res.provenance["dip_final_m"] = list(res.dips_m[-1])
            return res
        if g["source"] == "generated":
            res.ardy_calls += 1
            res.adapted_generations_executed += 1
        else:
            res.cache_hits += 1
        res.dips_m.append(np.round(q * DIP_MAX, 4).tolist())
        tr = clearance_trace(scene, g["qpos"], p.xy)
        res.attempts.append(Attempt(iteration=it, schedule_hash=g["schedule_hash"],
                                    source=g["source"], key=g["key"], trace=tr,
                                    peak_dip_m=float(q.max() * DIP_MAX), gen_s=g["gen_s"],
                                    seed=seed))

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
    res.provenance["selected_attempt"] = len(res.attempts) - 1
    res.provenance["selected_seed"] = res.final.seed
    res.provenance["attempt_seeds_executed"] = [a.seed for a in res.attempts]
    res.provenance["final_schedule_hash"] = res.final.schedule_hash
    res.provenance["dip_initial_m"] = np.round(np.clip(q0, 0, 1) * DIP_MAX, 4).tolist()
    res.provenance["dip_final_m"] = np.round(q * DIP_MAX, 4).tolist()
    return res


RESAMPLE_SEED_STRIDE = 10_000


def resample_seeds(seed: int, max_samples: int,
                   stride: int = RESAMPLE_SEED_STRIDE) -> list[int]:
    """Deterministic independent seeds for an unchanged-proposal compute control.

    A large stride prevents adjacent top-level experiment seeds (for example 100 and 101)
    from silently sharing their second sample. The first seed is deliberately unchanged so
    repair and resampling are paired on exactly the same initial generation.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be at least one")
    if stride < 1:
        raise ValueError("stride must be positive")
    return [int(seed + i * stride) for i in range(max_samples)]


def _sample_score(a: Attempt, target_m: float) -> tuple:
    """Predeclared verify/select order for unchanged-proposal samples.

    Target satisfaction dominates collision-free status, followed by realised overhead,
    whole-body clearance, and goal accuracy. The attempt index is the final tie-break so the
    result is stable even when deterministic generation produces identical measurements.
    """
    tr = a.trace
    return (int(tr.collision_free and not tr.below_margin(target_m)),
            int(tr.collision_free), float(tr.min_overhead_m), float(tr.min_clearance_m),
            -float(tr.goal_error_m), -int(a.iteration))


def run_resample(scene: Scene, p: Plan, q0: np.ndarray, cache: ClipCache, *,
                 preference: str = "clearance", target_m: float = MARGIN_M,
                 seed: int = DEFAULT_SEED, speed: float = SPEED, max_samples: int = 3,
                 seed_stride: int = RESAMPLE_SEED_STRIDE, allow_generate: bool = True,
                 provenance: dict | None = None) -> LoopResult:
    """Verify/select independent samples from one byte-identical proposal.

    This is the equal-compute control for feedback repair. It may spend the same maximum
    number of adapted generations, but it receives no geometric correction: every attempt
    uses exactly `q0`, and only the generator seed changes. Like repair, it stops as soon as
    an attempt is collision-free and meets the target. If none does, the predeclared score
    selects the best measured candidate rather than privileging the final draw.
    """
    seeds = resample_seeds(seed, max_samples, seed_stride)
    q = np.clip(np.asarray(q0, float), 0.0, 1.0)
    qh = schedule_hash(q)
    res = LoopResult(scene_id=scene.scene_id, provenance=dict(provenance or {}))
    res.provenance.update({"initial_schedule_hash": qh, "seed": seed,
                           "attempt_seeds_planned": seeds, "steps": DIFFUSION_STEPS,
                           "target_m": target_m, "preference": preference,
                           "speed_mps": speed, "max_samples": max_samples,
                           "max_adapted_generation_budget": max_samples,
                           "feedback_mode": "independent_resample_select",
                           "proposal_unchanged": True,
                           "selection_rule": ["meets_target", "collision_free",
                                              "min_overhead_m", "min_clearance_m",
                                              "negative_goal_error_m", "earlier_attempt"]})

    for it, attempt_seed in enumerate(seeds):
        g = generate_from_schedule(scene, p, q, cache, iteration=it, preference=preference,
                                   seed=attempt_seed, speed=speed,
                                   allow_generate=allow_generate)
        if g["qpos"] is None:
            res.outcome = "unverified"
            res.reason = "clip not cached and generation disabled"
            res.selected_attempt = (max(range(len(res.attempts)),
                                        key=lambda i: _sample_score(res.attempts[i], target_m))
                                    if res.attempts else None)
            if res.attempts:
                res.q_final = q
                res.provenance["selected_attempt"] = res.selected_attempt
                res.provenance["selected_seed"] = res.final.seed
                res.provenance["attempt_seeds_executed"] = [a.seed for a in res.attempts]
                res.provenance["final_schedule_hash"] = res.final.schedule_hash
                res.provenance["dip_initial_m"] = np.round(q * DIP_MAX, 4).tolist()
                res.provenance["dip_final_m"] = np.round(q * DIP_MAX, 4).tolist()
            return res
        if g["source"] == "generated":
            res.ardy_calls += 1
            res.adapted_generations_executed += 1
        else:
            res.cache_hits += 1
        res.dips_m.append(np.round(q * DIP_MAX, 4).tolist())
        tr = clearance_trace(scene, g["qpos"], p.xy)
        res.attempts.append(Attempt(iteration=it, schedule_hash=g["schedule_hash"],
                                    source=g["source"], key=g["key"], trace=tr,
                                    peak_dip_m=float(q.max() * DIP_MAX), gen_s=g["gen_s"],
                                    seed=attempt_seed))
        if tr.collision_free and not tr.below_margin(target_m):
            res.selected_attempt = it
            res.outcome = "accepted"
            res.reason = (f"sample {it + 1}/{max_samples} is collision-free and meets "
                          f"the {target_m:.2f} m target")
            break

    if res.outcome != "accepted":
        res.selected_attempt = max(range(len(res.attempts)),
                                   key=lambda i: _sample_score(res.attempts[i], target_m))
        tr = res.final.trace
        if tr.collision_free:
            res.outcome = "accepted_margin"
            res.reason = (f"best of {len(res.attempts)} independent samples is collision-free "
                          f"but {tr.deficit(target_m).max()*1000:.0f} mm short of the "
                          f"{target_m:.2f} m target")
        else:
            res.outcome = "rejected"
            res.reason = (f"all {len(res.attempts)} independent samples collide; best is "
                          f"{tr.min_clearance_m*1000:.0f} mm")

    res.q_final = q
    res.provenance["selected_attempt"] = res.selected_attempt
    res.provenance["selected_seed"] = res.final.seed
    res.provenance["attempt_seeds_executed"] = [a.seed for a in res.attempts]
    res.provenance["final_schedule_hash"] = res.final.schedule_hash
    res.provenance["dip_initial_m"] = np.round(q * DIP_MAX, 4).tolist()
    res.provenance["dip_final_m"] = np.round(q * DIP_MAX, 4).tolist()
    return res
