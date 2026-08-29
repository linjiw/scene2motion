"""Exercise the running demo against the V1 acceptance criteria and print evidence.

Hits the live HTTP API rather than the Python functions, so what is checked is what a user
actually gets. Run with the server up:  python -m scene2motion.demo.acceptance
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8765"
OK, BAD = "  PASS  ", "  FAIL  "


def get(path: str, timeout: int = 300) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    globals()["BASE"] = args.base
    fails = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print((OK if cond else BAD) + name + (f"  — {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    print("V1 ACCEPTANCE\n" + "-" * 78)

    # 1. both under/duck and around/upright render for the SAME scene
    s = get("/api/plan?height=1.00&width=1.45&preference=shortest")
    u = get("/api/plan?height=1.00&width=1.45&preference=upright")
    check("under and around routes exist for one scene",
          s["panel"]["goes_under_beam"] and not u["panel"]["goes_under_beam"],
          f"under {s['panel']['path_length_m']} m / around {u['panel']['path_length_m']} m")
    check("both routes are drawn in the BEV payload",
          len(s["bev"]["routes"]) >= 2 and all(len(v) > 2 for v in s["bev"]["routes"].values()),
          f"{ {k: len(v) for k, v in s['bev']['routes'].items()} }")

    # 2. the three preferences select the intended strategy
    a = get("/api/plan?height=1.00&width=1.45&preference=clearance")["all"]
    check("Shortest Path goes under and is shortest",
          a["shortest"]["goes_under_beam"] and
          a["shortest"]["path_length_m"] < a["upright"]["path_length_m"])
    check("Stay Upright goes around without ducking",
          not a["upright"]["goes_under_beam"] and not a["upright"]["duck_required"])
    check("Maximum Clearance goes around and pays more path for it",
          not a["clearance"]["goes_under_beam"] and
          a["clearance"]["path_length_m"] > a["upright"]["path_length_m"],
          f"{a['clearance']['path_length_m']} m vs {a['upright']['path_length_m']} m")

    # 3. beam height and width change the decision consistently
    tops = [get(f"/api/plan?height={h}&width=1.45&preference=shortest")["panel"]
            for h in (1.60, 1.30, 1.15, 1.00)]
    check("higher beam needs no duck; lower beam needs a deeper one",
          (not tops[0]["duck_required"]) and tops[1]["duck_required"]
          and tops[3]["deepest_mode"] != tops[1]["deepest_mode"],
          " -> ".join(t["deepest_mode"] for t in tops))
    widths = [get(f"/api/plan?height=1.00&width={w}&preference=shortest")["panel"]
              for w in (0.40, 1.45)]
    check("a narrow beam is sidestepped, a wide one is ducked under",
          (not widths[0]["goes_under_beam"]) and widths[1]["goes_under_beam"])

    # 4. cached playback is deterministic
    g1 = get("/api/generate?height=1.00&width=1.45&preference=shortest")
    g2 = get("/api/generate?height=1.00&width=1.45&preference=shortest")
    check("repeat request is a cache hit with an identical key",
          g1["source"] == "cache" and g1["cache_key"] == g2["cache_key"], g1["cache_key"])
    check("repeat request returns identical geometry",
          g1["anim"]["frames"][50] == g2["anim"]["frames"][50])

    # 5. one neutral and one duck motion load and animate
    duck = g1["anim"]["frames"]
    up = get("/api/generate?height=1.00&width=1.45&preference=upright")["anim"]["frames"]
    dz = [f["pelvis"][2] for f in duck]
    uz = [f["pelvis"][2] for f in up]
    check("duck clip animates and actually ducks",
          len(duck) > 50 and any(f["state"] == "DUCK" for f in duck) and min(dz) < 0.55,
          f"pelvis {min(dz):.3f}..{max(dz):.3f} m")
    check("upright clip animates and never ducks",
          len(up) > 50 and all(f["state"] == "UPRIGHT" for f in up) and min(uz) > 0.70,
          f"pelvis {min(uz):.3f}..{max(uz):.3f} m")

    # 6. the panel matches the route and the program
    p = g1["clip"]
    pan = s["panel"]
    check("panel duck window starts before the beam",
          pan["duck_required"] and pan["duck_start_s"] < 4.0 / 0.9,
          f"duck {pan['duck_start_s']}s -> {pan['duck_end_s']}s, beam at ~4.4 s")
    check("panel deepest mode is a duck mode when it says a duck is required",
          pan["deepest_mode"].startswith("duck") == pan["duck_required"],
          pan["deepest_mode"])

    # 7. the demo never overclaims
    v = g1["validation"]
    check("validation separates kinematic, tracked and physics-validated",
          v["kinematic_collision_free"] is True and v["sonic_tracked"] is False
          and v["physics_validated"] is False, v["statement"])
    check("clip metadata records provenance",
          all(k in p for k in ("model", "seed", "steps", "fps", "n_frames", "prompt")),
          f"{p['model']} seed={p['seed']} steps={p['steps']}")

    # 8. cache miss is reported rather than silently generated
    m = get("/api/generate?height=1.45&width=0.85&preference=clearance&allow_generate=0")
    check("uncached request with generation disabled reports a miss",
          m["ok"] is False and m["source"] == "miss")

    print("-" * 78)
    print(f"{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
