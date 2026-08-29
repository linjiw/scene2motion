"""Generation for the demo: plan -> constraint program -> frozen ARDY -> qpos, cached.

The model is loaded lazily and only when a cache miss actually needs it, so the UI starts
instantly and a fully-cached session never touches the GPU. Generation runs at 5 denoising
steps: EXP-006 measured 5 to beat the default 10 on validity (0.536 vs 0.448, paired McNemar
p = 0.0023), seed scatter, roughness and contact consistency, at 1.9x the speed -- it is not
a speed/quality trade, it is strictly better on this checkpoint.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..planner import LEAD_S, Plan, plan_to_path_spec, plan_to_spec
from ..robot import G1Body
from ..scenes import Scene
from .cache import ClipCache, key_for

PROMPT = "A person walks forward."
SPEED = 0.9
DIFFUSION_STEPS = 5
MAX_DURATION_S = 14.0
DEFAULT_SEED = 100

_lock = threading.Lock()
_runner = None


def get_runner():
    """The shared ArdyRunner, built on first use. Never constructed at import time."""
    global _runner
    with _lock:
        if _runner is None:
            from ..runner import ArdyRunner
            _runner = ArdyRunner(cache_path="outputs/text_cache.npz")
        return _runner


def n_frames(p: Plan, fps: float, speed: float = SPEED) -> int:
    length = float(np.linalg.norm(np.diff(p.xy, axis=0), axis=1).sum())
    return min(int(MAX_DURATION_S * fps), max(int(2 * fps), int(round(length / speed * fps))))


def program_bytes(p: Plan, scene: Scene, fps: float) -> bytes:
    """Identity of the BODY REQUEST, so the cache keys on what was actually asked for."""
    from ..program import encode
    prog = encode(p, scene, fps, SPEED)
    return (np.round(prog.lat, 5).tobytes() + np.round(prog.slot, 5).tobytes()
            + np.round(np.float64(prog.speed), 5).tobytes())


def _dip_for(scene: Scene, p: Plan, body_layer: str):
    """The dip schedule a body layer would command, or None for the heuristic renderer."""
    if body_layer not in ("learned", "optimized"):
        return None
    from ..learn.predictor import predict_dip
    if body_layer == "optimized":
        from .schedules import all_schedules, dip_for_layer
        d = dip_for_layer(all_schedules(scene, p, speed=SPEED), "optimized")
        if d is not None:
            return d
    return predict_dip(scene, p.xy, speed=SPEED)


def generate(scene: Scene, p: Plan, preference: str, cache: ClipCache,
             seed: int = DEFAULT_SEED, allow_generate: bool = True,
             body_layer: str = "heuristic") -> dict:
    """Return {qpos, source, key, meta}. `source` is 'cache' or 'generated'.

    `body_layer` selects who fills the duck channel: the heuristic mode lattice rendered by
    `plan_to_spec`, or the learned CNN's continuous schedule. The ROUTE is identical either
    way -- this is the route/body split, so a difference in the clip is attributable to the
    body layer alone. It is part of the cache key, or the two would collide.
    """
    from ..runner import ArdyRunner  # noqa: F401  (import cost only on this path)
    runner = get_runner()
    fps = runner.fps
    T = n_frames(p, fps)

    # The cache key must cover the BODY SCHEDULE, not just the route program and the layer
    # name. `program_bytes` describes the route, and "optimized" is a name whose meaning
    # changes with the clearance margin, the response fit and the checkpoint -- so a clip
    # generated at margin 0.12 would have been served for a 0.18 request under the same key.
    # Hashing the schedule itself makes the key describe what was actually asked for.
    dip = _dip_for(scene, p, body_layer)
    dip_bytes = b"" if dip is None else np.round(dip, 5).tobytes()
    key = key_for(scene_id=scene.scene_id, preference=f"{preference}|{body_layer}",
                  program_bytes=program_bytes(p, scene, fps) + dip_bytes,
                  model=runner.model_name, seed=seed, steps=DIFFUSION_STEPS,
                  fps=fps, n_frames=T)
    hit = cache.get(key)
    if hit is not None:
        qpos, meta = hit
        return {"qpos": qpos, "source": "cache", "key": key, "meta": meta}
    if not allow_generate:
        return {"qpos": None, "source": "miss", "key": key, "meta": {}}

    t0 = time.time()
    # The path-only clip first: it is the matched control the adapted request is BUILT from,
    # so the duck is a local edit of a motion the prior already produced for this exact path
    # rather than a pose invented from outside its manifold.
    ref = runner.generate([PROMPT], [plan_to_path_spec(p, fps, SPEED, duration=T / fps)],
                          T, DIFFUSION_STEPS, seeds=[seed])[0]
    if dip is not None:
        from ..learn.predictor import spec_from_dip
        spec = spec_from_dip(scene, p.xy, dip, fps, speed=SPEED, duration=T / fps)
    else:
        spec = plan_to_spec(p, fps, ref, runner.joint_names, speed=SPEED,
                            duration=T / fps, lead_s=LEAD_S)
    out = runner.generate([PROMPT], [spec], T, DIFFUSION_STEPS, seeds=[seed])[0]
    qpos = runner.to_qpos(out)
    gen_s = time.time() - t0

    body = G1Body(scene)
    rep = body.trajectory_report(qpos)
    goal_err = float(np.linalg.norm(qpos[-1, :2] - np.asarray(scene.goal)))
    meta = {"scene_id": scene.scene_id, "preference": preference,
            "body_layer": body_layer, "seed": seed,
            "steps": DIFFUSION_STEPS, "model": runner.model_name, "fps": fps,
            "n_frames": T, "generate_s": round(gen_s, 3),
            "collision_free": bool(rep["collision_free"]),
            "min_clearance_m": round(float(rep["min_clearance_m"]), 4),
            "max_penetration_m": round(float(rep["max_penetration_m"]), 4),
            "goal_error_m": round(goal_err, 3),
            "prompt": PROMPT}
    cache.put(key, qpos, meta)
    return {"qpos": qpos, "source": "generated", "key": key, "meta": meta}
