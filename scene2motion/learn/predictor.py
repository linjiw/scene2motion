"""The learned planner as a drop-in for the heuristic's body layer.

Same signature in and same object out: scene + route -> a dip schedule the renderer can turn
into a ConstraintSpec. The ROUTE still comes from A*; only the body decision along it is
learned, which is the whole point of the route/body split.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

from ..planner import MODE_BY_NAME
from ..scenes import Scene
from .model import MODELS
from .route_profile import N_SAMPLES, normalise, profile

NOMINAL_PELVIS = MODE_BY_NAME["stand"].pelvis_y
DEFAULT_CKPT = Path("outputs/duck_model/cnn.pt")


@functools.lru_cache(maxsize=4)
def load_model(arch: str = "cnn", ckpt: str | None = None):
    m = MODELS[arch]()
    m.load_state_dict(torch.load(Path(ckpt or DEFAULT_CKPT), map_location="cpu"))
    m.eval()
    return m


def predict_dip(scene: Scene, xy: np.ndarray, arch: str = "cnn",
                ckpt: str | None = None, speed: float = 0.9) -> np.ndarray:
    """(N_SAMPLES,) predicted dip in metres along the route."""
    p = normalise(profile(scene, xy, speed=speed))
    with torch.no_grad():
        d = load_model(arch, ckpt)(torch.from_numpy(p).float()[None])[0].numpy()
    return np.clip(d, 0.0, 0.55)


def dip_to_pelvis(dip: np.ndarray) -> np.ndarray:
    return NOMINAL_PELVIS - np.asarray(dip, float)


def resample_to_frames(dip: np.ndarray, xy: np.ndarray, T: int) -> np.ndarray:
    """Route-position dip -> per-frame dip, uniform in arc length like the renderer expects."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1]) if s[-1] > 0 else 1.0
    src = np.linspace(0.0, total, len(dip))
    return np.interp(np.linspace(0.0, total, T), src, dip)


def spec_from_dip(scene: Scene, xy, dip: np.ndarray, fps: float, speed: float = 0.9,
                  duration: float | None = None):
    """Render ANY dip schedule into the duck channel, whoever produced it.

    This is the single integration point that makes heuristic and learned planners
    comparable: both hand over a dip schedule in route-position space and everything
    downstream -- the root path, the heading, the frame count, the prior -- is identical. A
    difference in the generated motion is then attributable to the schedule and to nothing
    else.

    Only `root_y` is written. Duck is the one body axis this project has shown to be both
    addressable and trackable, so the demo's learned layer does not touch tuck or lift.
    """
    from ..constraints import ConstraintSpec
    from ..planner import Plan, _n_frames, plan_to_path_spec

    p = Plan(np.asarray(xy, float), [MODE_BY_NAME["stand"]] * len(xy), "adaptive", True)
    T = _n_frames(p, fps, speed, duration)
    base = plan_to_path_spec(p, fps, speed, duration=duration)
    root_y = dip_to_pelvis(resample_to_frames(dip, xy, T))
    return ConstraintSpec(root_xz=base.root_xz, heading=base.heading,
                          root_y=np.asarray(root_y, float),
                          first_heading=base.first_heading)
