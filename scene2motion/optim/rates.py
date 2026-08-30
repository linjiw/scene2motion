"""Phase 4C: how fast can the frozen prior actually change its body height?

The Phase 3 scheduler models the body as a first-order lag and nothing else, so it is free to
write a schedule with an arbitrarily steep edge and trust the lag to smooth it. The lag does
smooth it -- but the prior is not a lag, and a command edge steeper than anything it can
follow simply does not arrive. This measures the ceiling directly, from clips the prior
produced, so the scheduler can be told about it in the units it plans in.

Two rates, because they are not the same motion:

    r_down   how fast the commanded duck may INCREASE -- crouching, which gravity assists
    r_up     how fast it may DECREASE -- standing back up, which it does not

Both are expressed as command units per second, so a constraint reads
`u[i+1] - u[i] <= r_down * dt_i` with dt in seconds and no hidden scaling.

Method: for each cached clip, take the measured top-of-body height per frame, convert it back
through the fitted gain's inverse into the command that would have produced it, and look at
the distribution of its time derivative. The high quantile of the positive part is what the
prior can achieve going down, the negative part going up. A quantile rather than a maximum,
because one frame of diffusion noise should not set a bound the scheduler then relies on.

Phase 3's earlier attempt at a time-parameterised objective is NOT repeated here. That version
divided a jerk term by dt^3, which at 64 samples over a 14 m route made the jerk weight ~125x
everything else and turned the objective into a smoothness solver with a nominal effort term.
The objective below is dimensioned so every term is an integral over time with units that do
not move when the route length or the speed does.

    python -m scene2motion.optim.rates --fit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

OUT = Path("outputs/duck_rates")
CACHE = Path("scene2motion/demo_outputs/clips")
QUANTILE = 0.95
# Below this much movement the frame-to-frame difference is dominated by diffusion noise
# rather than by an intended change of height, and including it would bias both bounds.
MIN_RATE = 0.02          # command units per second


def rates_from_clip(qpos: np.ndarray, resp, fps: float, body) -> tuple[np.ndarray, np.ndarray]:
    """(descent rates, recovery rates) in command units per second, from one clip."""
    T = len(qpos)
    top = np.array([body.top_height(qpos[t]) for t in range(T)])
    u = resp.g_inv(top)                       # the command that explains this height
    du = np.diff(u) * fps
    return du[du > MIN_RATE], -du[du < -MIN_RATE]


def fit(cache_root: Path = CACHE, limit: int | None = None, quantile: float = QUANTILE) -> dict:
    """Measure both bounds over the cached clip corpus."""
    from ..demo.scene_builder import BeamParams, build
    from ..optim.response import DuckResponse
    from ..robot import G1Body

    resp = DuckResponse.load(Path("outputs/duck_response/response.json"))
    keys = sorted(p.stem for p in Path(cache_root).glob("*.json"))
    if limit:
        keys = keys[:limit]

    down, up, used, skipped = [], [], [], 0
    bodies: dict[str, object] = {}
    for k in keys:
        try:
            meta = json.loads((Path(cache_root) / f"{k}.json").read_text())
            qpos = np.load(Path(cache_root) / f"{k}.npy")
            sid = meta["scene_id"]
            if not sid.startswith("demo_partial_beam_h"):
                skipped += 1
                continue
            if sid not in bodies:
                h = float(sid.split("_h")[1].split("_w")[0])
                w = float(sid.split("_w")[1].split("_")[0])
                n = int(sid.split("_n")[1].split("_")[0]) if "_n" in sid else 1
                g = float(sid.split("_g")[1]) if "_g" in sid else 3.0
                bodies[sid] = G1Body(build(BeamParams(h, w, n, g).clamped()))
            d, u_ = rates_from_clip(qpos, resp, float(meta.get("fps", 25.0)), bodies[sid])
            down.append(d)
            up.append(u_)
            used.append(k)
        except Exception:
            skipped += 1

    d = np.concatenate([x for x in down if len(x)]) if any(len(x) for x in down) else np.zeros(1)
    u = np.concatenate([x for x in up if len(x)]) if any(len(x) for x in up) else np.zeros(1)
    return {"generated_at": time.time(), "n_clips": len(used), "n_skipped": skipped,
            "quantile": quantile, "min_rate_floor": MIN_RATE,
            "r_down": round(float(np.quantile(d, quantile)), 4),
            "r_up": round(float(np.quantile(u, quantile)), 4),
            "r_down_median": round(float(np.median(d)), 4),
            "r_up_median": round(float(np.median(u)), 4),
            "r_down_max": round(float(d.max()), 4), "r_up_max": round(float(u.max()), 4),
            "n_down_samples": int(len(d)), "n_up_samples": int(len(u))}


def load(path: Path = OUT / "rates.json") -> dict | None:
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quantile", type=float, default=QUANTILE)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    r = fit(limit=a.limit, quantile=a.quantile)
    (a.out / "rates.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    print(f"-> {a.out / 'rates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
