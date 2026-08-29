"""Populate the demo clip cache. Deliberately small by default.

The demo is usable with two clips (one upright, one ducking); the grid exists so a recorded
walkthrough never hits a miss. Runs one clip at a time and reports each, so it can be stopped
at any point and the cache stays valid -- entries are content-addressed and independent.

    python -m scene2motion.demo.seed_cache --heights 1.30 1.00 --preferences shortest upright
"""

from __future__ import annotations

import argparse
import time

from .ardy_runner import generate
from .cache import ClipCache
from .scene_builder import DEFAULTS, BeamParams, build
from .strategy_planner import PREFERENCES, evaluate
from .app import CACHE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heights", type=float, nargs="+", default=[1.30, 1.00])
    ap.add_argument("--widths", type=float, nargs="+", default=[DEFAULTS["beam_width"]])
    ap.add_argument("--preferences", nargs="+", default=["shortest", "upright"])
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    made = hit = skipped = 0
    for w in args.widths:
        for h in args.heights:
            scene = build(BeamParams(h, w))
            for pref in args.preferences:
                if pref not in PREFERENCES:
                    continue
                s = evaluate(scene, pref)
                if not s.feasible:
                    print(f"  h={h:.2f} w={w:.2f} {pref:10s} refused — skipped", flush=True)
                    skipped += 1
                    continue
                r = generate(scene, s.plan, pref, CACHE, seed=args.seed)
                made += r["source"] == "generated"
                hit += r["source"] == "cache"
                m = r["meta"]
                print(f"  h={h:.2f} w={w:.2f} {pref:10s} {r['source']:9s} "
                      f"{'clear' if m.get('collision_free') else 'COLLISION':9s} "
                      f"min_clr={m.get('min_clearance_m')} goal_err={m.get('goal_error_m')} "
                      f"{r['key']}", flush=True)
    print(f"\n{made} generated, {hit} cached, {skipped} skipped in {time.time()-t0:.1f}s")
    print("cache:", CACHE.stats()["n_entries"], "entries")


if __name__ == "__main__":
    main()
