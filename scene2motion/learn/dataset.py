"""Procedural dataset: route-local profile -> duck schedule, with geometry splits.

Labels come from the system that already works -- plan the scene with the heuristic planner,
render its mode schedule through the SAME dilation and smoothing `plan_to_spec` uses, and
resample to route position. So the target is not the raw A* lattice decision but the dip
channel the prior is actually asked for, including the 0.8 s anticipation lead. A model that
reproduces this is a drop-in replacement for the middle layer, not an approximation of an
intermediate variable nobody consumes.

Splits are by GEOMETRY, never by random row. Beam heights and positions are disjoint between
train and test, so a test score is a statement about unseen geometry rather than unseen noise:

    train   heights 0.80 0.90 1.00 1.10 1.20      positions 3.0 4.0 5.0
    dev     heights 0.85 1.15                     positions 3.5
    test    heights 0.95 1.05                     positions 4.5
    plus    no-beam controls in every split (the model must learn NOT to duck)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..demo.strategy_planner import SHORTEST_MODES
from ..planner import LEAD_S, MODE_BY_NAME, _dilate_channel, _smooth, plan
from ..scenes import WALL_T, Box, Scene, _side_walls
from .route_profile import N_CHANNELS, N_SAMPLES, normalise, profile

NOMINAL_PELVIS = MODE_BY_NAME["stand"].pelvis_y
SPEED = 0.9
CORRIDOR_HALF = 1.2
GOAL_X = 8.0

SPLITS = {
    "train": {"heights": [0.80, 0.90, 1.00, 1.10, 1.20], "positions": [3.0, 4.0, 5.0]},
    "dev":   {"heights": [0.85, 1.15], "positions": [3.5]},
    "test":  {"heights": [0.95, 1.05], "positions": [4.5]},
}
WIDTHS = [0.90, 1.20, 1.45, 1.75, 2.00]
# Secondary geometry, varied per sample so the model sees more than a product grid. These are
# NOT split-defining: only beam height and beam position separate train from test, so a test
# score is about unseen beam geometry rather than an unseen corridor width.
CORRIDOR_HALVES = [1.0, 1.2, 1.4]
GOAL_XS = [7.0, 8.0, 9.0]
REPEATS = 3
# A beam this high restricts nothing: the label must be a flat zero and the model must learn it.
NO_BEAM_HEIGHT = 2.40


def build_scene(height: float, beam_x: float, width: float, seed: int = 0,
                second_beam: tuple[float, float] | None = None,
                corridor_half: float = CORRIDOR_HALF, goal_x: float = GOAL_X) -> Scene:
    """One or two partial beams in a corridor. Only the named parameters move."""
    left = corridor_half + WALL_T / 2
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)

    def add(h: float, x: float, w: float, tag: str) -> None:
        edge = left - w
        cy, hy = 0.5 * (edge + left), 0.5 * (left - edge)
        boxes.append(Box((x, cy, h + 0.125), (0.12, hy, 0.125), tag))

    add(height, beam_x, min(width, 2 * corridor_half - 0.15), "partial_beam")
    if second_beam is not None:
        add(second_beam[0], second_beam[1], width, "partial_beam_2")
    sid = hashlib.sha1(f"{height:.3f}_{beam_x:.3f}_{width:.3f}_{corridor_half:.2f}_"
                       f"{goal_x:.1f}_{second_beam}_{seed}".encode()).hexdigest()[:10]
    return Scene(
        scene_id=f"lb_{sid}", family="partial_beam", boxes=boxes,
        start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
        param_name="beam_underside_height", param_value=height,
        required_adaptation="duck" if height < 1.07 else "none",
        bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
        meta={"beam_x": beam_x, "beam_h": height, "beam_w": width,
              "corridor_half": corridor_half, "goal_x": goal_x,
              "second_beam": second_beam},
    )


def duck_label(p, n: int = N_SAMPLES, speed: float = SPEED,
               lead_s: float = LEAD_S) -> np.ndarray:
    """The dip schedule the prior is actually asked for, in route-position space.

    Same dilation and smoothing as `plan_to_spec`, applied over arc-length samples rather than
    frames so the label does not depend on clip length. The dilation width is `lead_s` of
    travel converted to samples, which is what makes the anticipation lead present in the
    target instead of something a downstream renderer adds later.
    """
    xy = np.asarray(p.xy, float).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1]) if s[-1] > 0 else 1.0
    want = np.linspace(0.0, total, n)
    idx = np.clip(np.searchsorted(s, want), 0, len(p.modes) - 1)
    pelvis = np.array([p.modes[i].pelvis_y for i in idx])

    w = max(1, int(round(lead_s * speed / max(total, 1e-6) * n)))
    win = max(3, (n // 12) | 1)
    pelvis = _smooth(_dilate_channel(pelvis, w, "min"), win)
    return np.clip(NOMINAL_PELVIS - pelvis, 0.0, None).astype(np.float32)


@dataclass
class Sample:
    scene_id: str
    split: str
    height: float
    beam_x: float
    width: float
    profile: np.ndarray      # (N_SAMPLES, N_CHANNELS) physical units
    label: np.ndarray        # (N_SAMPLES,) dip in metres
    route_len: float
    goes_under: bool


def _grid(split: str, seed: int = 0) -> list[dict]:
    cfg = SPLITS[split]
    rng = np.random.default_rng(hash(split) % (2 ** 31) + seed)
    combos = []
    for h in cfg["heights"] + [NO_BEAM_HEIGHT]:
        for x in cfg["positions"]:
            for w in WIDTHS:
                for _ in range(REPEATS):
                    combos.append({
                        "height": float(h), "beam_x": float(x), "width": float(w),
                        "corridor_half": float(rng.choice(CORRIDOR_HALVES)),
                        "goal_x": float(rng.choice(GOAL_XS)),
                    })
    return combos


def build_split(split: str, verbose: bool = False) -> list[Sample]:
    out, skipped = [], 0
    for g in _grid(split):
        scene = build_scene(**g)
        # The SHORTEST-preference planner, which is the one that ducks under a beam rather
        # than walking around it. Labelling with the default cost model would teach the model
        # to predict "no duck" almost everywhere, because under the shipped mode costs a deep
        # duck held for ~1.5 m is dearer than a 0.5 m detour.
        p = plan(scene, "adaptive", modes_override=SHORTEST_MODES)
        if not p.feasible or len(p.xy) < 2:
            skipped += 1
            continue
        prof = profile(scene, p.xy, speed=SPEED)
        lab = duck_label(p)
        xy = np.asarray(p.xy)
        h, x, w = g["height"], g["beam_x"], g["width"]
        under = bool(((xy[:, 0] > x - 0.2) & (xy[:, 0] < x + 0.2)
                      & (xy[:, 1] > g["corridor_half"] + WALL_T / 2 - w)).any())
        out.append(Sample(scene.scene_id, split, h, x, w, prof, lab,
                          float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), under))
    if verbose:
        ducking = sum(1 for s in out if s.label.max() > 0.02)
        print(f"  {split:5s}: {len(out):3d} samples ({ducking} with a duck, "
              f"{len(out)-ducking} flat), {skipped} infeasible")
    return out


def to_arrays(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([normalise(s.profile) for s in samples]).astype(np.float32)
    Y = np.stack([s.label for s in samples]).astype(np.float32)
    return X, Y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/duck_dataset")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("building procedural duck dataset (CPU only, no ARDY)")
    meta = {"n_samples": N_SAMPLES, "n_channels": N_CHANNELS, "speed": SPEED,
            "lead_s": LEAD_S, "splits": SPLITS, "widths": WIDTHS,
            "no_beam_height": NO_BEAM_HEIGHT, "label": "dip (m) after lead-dilation+smoothing",
            "built_s": None, "counts": {}}
    for split in SPLITS:
        s = build_split(split, verbose=True)
        X, Y = to_arrays(s)
        np.savez_compressed(out / f"{split}.npz", X=X, Y=Y,
                            height=np.array([x.height for x in s]),
                            beam_x=np.array([x.beam_x for x in s]),
                            width=np.array([x.width for x in s]),
                            route_len=np.array([x.route_len for x in s]),
                            goes_under=np.array([x.goes_under for x in s]))
        meta["counts"][split] = len(s)
    meta["built_s"] = round(time.time() - t0, 2)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} in {meta['built_s']}s")


if __name__ == "__main__":
    main()
