"""Optimisation-teacher dataset: 0-3 beams, geometry- and composition-disjoint splits.

The teacher is the QP, not the heuristic. That is the whole difference from Phase 2: the
targets contain minimal-crouch depth, emergent anticipation and emergent merge/split, so a
model distilling them can do things the mode lattice cannot.

Splits are disjoint on TWO axes, and the distinction matters for what a test number means:

  geometry   beam heights and positions never shared between train and test
  composition  train sees 0, 1 and a frozen subset of 2-beam gaps; test additionally sees
               unseen gap-speed pairs and THREE beams, which no training sample contains

So a test score mixes interpolation (unseen heights on seen beam counts) with genuine
compositional extrapolation (unseen beam counts), and the two are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..demo.strategy_planner import SHORTEST_MODES
from ..learn.route_profile import N_SAMPLES, normalise, profile
from ..planner import plan
from ..scenes import WALL_T, Box, Scene, _side_walls
from .response import DuckResponse
from .scheduler import MARGIN_M, dt_for, solve

CORRIDOR_HALF, WIDTH = 1.2, 1.9
BEAM_HALF_X, BEAM_HALF_Z = 0.12, 0.125

SPLITS = {
    "train": {"heights": [0.95, 1.05, 1.15, 1.25], "x0": [3.0, 4.0],
              "gaps": [1.0, 2.0, 3.5, 5.0], "speeds": [0.7, 0.9, 1.1],
              "counts": [0, 1, 2]},
    # Dev was originally one height, one position, one speed -- 8 samples, which is far too
    # thin to choose a checkpoint on and was quoted as though it meant something. Widened to
    # its own disjoint heights, positions and speeds and frozen here BEFORE any selection.
    "dev":   {"heights": [1.00, 1.30], "x0": [3.5, 5.0], "gaps": [1.5, 3.0, 4.5],
              "speeds": [0.8, 1.05], "counts": [0, 1, 2]},
    "test":  {"heights": [1.10, 1.20], "x0": [4.5], "gaps": [1.2, 2.5, 4.5, 6.0],
              "speeds": [0.6, 1.0, 1.2], "counts": [0, 1, 2, 3]},
}
REPEATS = 2


def build_beams(heights: list[float], xs: list[float], goal_x: float,
                corridor_half: float = CORRIDOR_HALF) -> Scene:
    left = corridor_half + WALL_T / 2
    boxes = _side_walls(corridor_half, -1.0, goal_x + 1.0)
    edge = left - min(WIDTH, 2 * corridor_half - 0.15)
    cy, hy = 0.5 * (edge + left), 0.5 * (left - edge)
    for i, (h, x) in enumerate(zip(heights, xs)):
        boxes.append(Box((x, cy, h + BEAM_HALF_Z), (BEAM_HALF_X, hy, BEAM_HALF_Z),
                         f"beam_{i}"))
    sid = hashlib.sha1(f"{heights}_{xs}_{goal_x}_{corridor_half}".encode()).hexdigest()[:10]
    return Scene(scene_id=f"v3_{sid}", family="partial_beam", boxes=boxes,
                 start=(0.0, 0.0), goal=(goal_x, 0.0), start_heading=0.0,
                 param_name="beam_underside_height",
                 param_value=heights[0] if heights else 2.6,
                 required_adaptation="duck" if heights else "none",
                 bounds=(-1.0, goal_x + 1.0, -corridor_half - 0.3, corridor_half + 0.3),
                 meta={"heights": heights, "xs": xs, "n_beams": len(heights),
                       "goal_x": goal_x})


def _cases(split: str) -> list[dict]:
    cfg = SPLITS[split]
    rng = np.random.default_rng(abs(hash(split)) % (2 ** 31))
    out = []
    for n_beams in cfg["counts"]:
        for h in cfg["heights"]:
            for x0 in cfg["x0"]:
                for sp in cfg["speeds"]:
                    gaps = cfg["gaps"] if n_beams >= 2 else [0.0]
                    for gap in gaps:
                        for _ in range(REPEATS):
                            if n_beams == 0:
                                hs, xs = [], []
                            else:
                                hs = [float(h + rng.uniform(-0.03, 0.03))
                                      for _ in range(n_beams)]
                                xs = [x0 + k * gap for k in range(n_beams)]
                            goal_x = float(max(8.0, (xs[-1] if xs else 4.0) + 3.5))
                            out.append({"heights": hs, "xs": xs, "goal_x": goal_x,
                                        "speed": float(sp), "n_beams": n_beams,
                                        "gap": float(gap)})
    return out


def build_split(split: str, resp: DuckResponse, verbose=False) -> dict:
    X, Y, QREQ, META = [], [], [], []
    skipped = infeas = 0
    for c in _cases(split):
        scene = build_beams(c["heights"], c["xs"], c["goal_x"])
        p = plan(scene, "adaptive", modes_override=SHORTEST_MODES)
        if not p.feasible or len(p.xy) < 2:
            skipped += 1
            continue
        prof = profile(scene, p.xy, speed=c["speed"])
        route_len = float(np.linalg.norm(np.diff(np.asarray(p.xy), axis=0), axis=1).sum())
        dt = dt_for(route_len, N_SAMPLES, c["speed"])
        sched = solve(prof[:, 0], resp, dt)
        if not sched.feasible:
            infeas += 1
            continue
        q_req = resp.g_inv(prof[:, 0] - MARGIN_M)
        X.append(normalise(prof))
        Y.append(sched.q.astype(np.float32))
        QREQ.append(q_req.astype(np.float32))
        META.append({**c, "scene_id": scene.scene_id, "route_len": route_len, "dt": dt,
                     "objective": sched.objective, "max_violation_m": sched.max_violation_m,
                     "status": sched.status, "n_iter": sched.n_iter})
    if verbose:
        nb = np.array([m["n_beams"] for m in META])
        print(f"  {split:5s}: {len(X):4d} samples "
              f"({', '.join(f'{int((nb==k).sum())}x{k}-beam' for k in sorted(set(nb)))}), "
              f"{skipped} unroutable, {infeas} infeasible")
    return {"X": np.stack(X), "Y": np.stack(Y), "QREQ": np.stack(QREQ), "meta": META}


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    """Build ATOMICALLY into a fresh versioned directory.

    An interrupted in-place rebuild previously left train and dev at margin 0.18 beside a test
    split and a metadata file at 0.12 -- training and evaluation silently at different margins,
    with meta.json describing neither (it recorded 281 train samples for a file holding 286).
    Three properties prevent that recurring, and none of them is optional:

      * the build goes to a TEMP directory and is moved into place only once every split is
        written, so a partial build can never be mistaken for a complete one;
      * the destination is VERSIONED by margin, so a 0.12 and a 0.18 dataset cannot occupy the
        same path and an old file cannot survive under a new name;
      * metadata is written LAST and validated by reloading every split from disk and checking
        its length against the count recorded, plus a content hash per file.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--tag", default=None, help="defaults to m<margin>")
    args = ap.parse_args()
    resp = DuckResponse.load()
    tag = args.tag or f"m{MARGIN_M:.2f}".replace(".", "")
    final = Path(args.root) / f"duck_dataset_v3_{tag}"
    tmp = Path(args.root) / f".duck_dataset_v3_{tag}.building"
    if final.exists():
        raise SystemExit(f"{final} already exists; refusing to overwrite a built dataset. "
                         f"Remove or rename it deliberately.")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0 = time.time()
    print(f"optimisation-teacher dataset (tau={resp.tau_s:.2f}s, margin={MARGIN_M} m) -> {final}")

    counts, files = {}, {}
    for split in SPLITS:
        d = build_split(split, resp, verbose=True)
        f = tmp / f"{split}.npz"
        np.savez_compressed(f, X=d["X"], Y=d["Y"], QREQ=d["QREQ"],
                            n_beams=np.array([m["n_beams"] for m in d["meta"]]),
                            gap=np.array([m["gap"] for m in d["meta"]]),
                            speed=np.array([m["speed"] for m in d["meta"]]),
                            dt=np.array([m["dt"] for m in d["meta"]]))
        (tmp / f"{split}_meta.json").write_text(json.dumps(d["meta"], indent=1))
        counts[split] = len(d["X"])

    # Validate against what is actually on disk before any metadata claims it.
    for split, n in counts.items():
        z = np.load(tmp / f"{split}.npz")
        if len(z["Y"]) != n:
            raise SystemExit(f"{split}: wrote {n} but file holds {len(z['Y'])}")
        if not np.isfinite(z["X"]).all() or not np.isfinite(z["Y"]).all():
            raise SystemExit(f"{split}: non-finite values")
        files[split] = {"n": int(n), "sha256_16": sha256_of(tmp / f"{split}.npz")}

    meta = {"splits": SPLITS, "repeats": REPEATS, "counts": counts, "margin_m": MARGIN_M,
            "tau_s": resp.tau_s, "response": "outputs/duck_response/response.json",
            "teacher": "convex QP (scheduler.solve)", "files": files,
            "n_samples": N_SAMPLES, "built_s": round(time.time() - t0, 1)}
    meta["dataset_hash"] = __import__("hashlib").sha256(
        json.dumps({k: v["sha256_16"] for k, v in files.items()},
                   sort_keys=True).encode()).hexdigest()[:16]
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(final)
    print(f"validated {counts}, hash {meta['dataset_hash']}")
    print(f"wrote {final} in {meta['built_s']}s")


if __name__ == "__main__":
    main()
