"""Turn a plan and a qpos clip into compact drawing data for the browser.

No new rendering framework, and deliberately no offscreen GL: a headless box may or may not
have a working EGL context, and the demo must start reliably. What it draws instead is the
robot's ACTUAL COLLISION GEOMETRY -- the same MuJoCo primitives `robot.trajectory_report`
tests against the scene -- projected into a side view (x, z) and a bird's-eye view (x, y).
That is honest in a way a prettier mesh render would not be: what you see is exactly what the
collision status refers to.

Capsules and cylinders are emitted as segments with a radius so a limb reads as a limb rather
than a blob; spheres and boxes collapse to a circle at their centre.
"""

from __future__ import annotations

import numpy as np

from ..planner import MODE_BY_NAME
from ..robot import G1Body
from ..scenes import Scene
from .scene_builder import beam_footprint

MAX_FRAMES = 110          # animation frames sent to the browser
UPRIGHT_PELVIS_Z = 0.70   # above this the state reads UPRIGHT, below it DUCK


def _geom_segments(body: G1Body) -> list[dict]:
    """Static per-geom description: which are segments, which are points, and their radii."""
    import mujoco
    m = body.model
    out = []
    for g in body.robot_geoms:
        t = int(m.geom_type[g])
        size = m.geom_size[g]
        r = float(size[0])
        half = float(size[1]) if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                       int(mujoco.mjtGeom.mjGEOM_CYLINDER)) else 0.0
        out.append({"g": int(g), "r": r, "half": half,
                    "name": body.geom_name.get(g, str(g))})
    return out


def frames(scene: Scene, qpos: np.ndarray, max_frames: int = MAX_FRAMES) -> dict:
    """Compact animation payload: per-frame geometry, pelvis path, and posture state."""
    body = G1Body(scene)
    spec = _geom_segments(body)
    n = len(qpos)
    idx = np.unique(np.linspace(0, n - 1, min(max_frames, n)).astype(int))
    out_frames = []
    for t in idx:
        body.fk(qpos[t])
        pos = body.data.geom_xpos
        mat = body.data.geom_xmat.reshape(-1, 3, 3)
        shapes = []
        for s in spec:
            g = s["g"]
            c = pos[g]
            if s["half"] > 1e-6:
                axis = mat[g][:, 2] * s["half"]      # capsule/cylinder long axis is local z
                a, b = c - axis, c + axis
                shapes.append([round(float(a[0]), 3), round(float(a[1]), 3),
                               round(float(a[2]), 3), round(float(b[0]), 3),
                               round(float(b[1]), 3), round(float(b[2]), 3),
                               round(s["r"], 3)])
            else:
                shapes.append([round(float(c[0]), 3), round(float(c[1]), 3),
                               round(float(c[2]), 3), round(float(c[0]), 3),
                               round(float(c[1]), 3), round(float(c[2]), 3),
                               round(s["r"], 3)])
        pel = qpos[t][:3]
        top = float(body.top_height(qpos[t]))
        out_frames.append({
            "t": int(t),
            "s": shapes,
            "pelvis": [round(float(pel[0]), 3), round(float(pel[1]), 3),
                       round(float(pel[2]), 3)],
            "top": round(top, 3),
            "state": "DUCK" if float(pel[2]) < UPRIGHT_PELVIS_Z else "UPRIGHT",
        })
    return {"n_shapes": len(spec), "frames": out_frames,
            "pelvis_path": [[round(float(q[0]), 3), round(float(q[1]), 3)] for q in qpos]}


def bev(scene: Scene, routes: dict[str, np.ndarray], selected: str) -> dict:
    """Bird's-eye data: bounds, walls, beam footprint, start/goal, and each route."""
    b = beam_footprint(scene)
    walls = [{"x_lo": float(x.lo[0]), "x_hi": float(x.hi[0]),
              "y_lo": float(x.lo[1]), "y_hi": float(x.hi[1])}
             for x in scene.boxes if x.label.startswith("wall")]
    return {
        "bounds": list(scene.bounds),
        "beam": b,
        "walls": walls,
        "start": list(scene.start),
        "goal": list(scene.goal),
        "routes": {k: [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in v[::2]]
                   for k, v in routes.items() if v is not None and len(v)},
        "selected": selected,
    }


def side(scene: Scene) -> dict:
    """Side-view static geometry: floor line and the beam's (x, z) rectangle."""
    b = beam_footprint(scene)
    return {"x_lo": float(scene.bounds[0]), "x_hi": float(scene.bounds[1]),
            "beam": {"x_lo": b["x_lo"], "x_hi": b["x_hi"],
                     "z_lo": b["z_lo"], "z_hi": b["z_hi"],
                     "y_lo": b["y_lo"], "y_hi": b["y_hi"]}}


def mode_envelope(mode_name: str) -> dict:
    m = MODE_BY_NAME[mode_name]
    return {"name": m.name, "top": m.top, "half_width": m.half_width,
            "pelvis_y": m.pelvis_y, "cost": m.cost}
