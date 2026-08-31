"""Live acceptance for Demo V3.1: the AUTO verify/repair path, against a running server.

Every check here is about a claim the UI makes. The load-bearing ones are the last few: that
a scene labelled repaired really was regenerated from the repaired schedule, that the
no-repair scene really needed no repair, and that the two are distinguishable from the
response alone rather than from the screenshot.

    python -m scene2motion.demo.accept_v31 --port 8077
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request

# The two committed demo cases, both on the final m018 model, both 3-6 beam OOD scenes.
REPAIR_CASE = {"height": 1.05, "width": 2.25, "n_beams": 5, "gap": 2.5,
               "preference": "shortest", "body_layer": "auto"}
CLEAN_CASE = {"height": 0.95, "width": 2.25, "n_beams": 3, "gap": 1.5,
              "preference": "shortest", "body_layer": "auto"}


def get(base: str, path: str, **params):
    url = f"{base}{path}?{urllib.parse.urlencode(params)}" if params else f"{base}{path}"
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    checks: list[tuple[str, bool, str]] = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    page = urllib.request.urlopen(base, timeout=30).read().decode()
    ck("page serves", "Route First, Body Next" in page)
    ck("AUTO is the default body layer", 'layer="auto"' in page)
    ck("one-shot layers moved to Advanced", 'id="layers2"' in page and "Advanced" in page)
    ck("repair timeline present", 'id="timeline"' in page)
    ck("cost breakdown present", 'id="cost"' in page)
    ck("beam count 1-6 selectable", '["6","6"]' in page)

    p = get(base, "/api/plan", height=1.05, width=2.25, n_beams=5, gap=2.5, preference="shortest")
    ck("plan for a 5-beam scene", p["params"]["n_beams"] == 5, p["scene_id"])
    ck("plan carries schedules", bool(p.get("schedules", {}).get("schedules")))

    r = get(base, "/api/generate", **REPAIR_CASE)
    ck("repair case generates", r.get("ok"), r.get("reason", ""))
    ck("repair case is the AUTO layer", r.get("body_layer") == "auto")
    a0 = r["attempts"][0]
    ck("repair case really fails one-shot", not a0["collision_free"],
        f"attempt 0 overhead {a0['min_overhead_m']:+.4f} m")
    ck("repair case ends collision-free", r["validation"]["kinematic_collision_free"],
        f"final overhead {r['validation']['min_overhead_m']:+.4f} m")
    ck("repair case meets the target margin", r["validation"]["meets_target_margin"])
    ck("repair case is labelled repaired", r["repaired"] is True)
    ck("repaired label matches the final schedule",
       r["provenance"]["final_schedule_hash"] != r["provenance"]["initial_schedule_hash"]
       and r["attempts"][-1]["schedule_hash"] == r["provenance"]["final_schedule_hash"])
    ck("each attempt has its own clip",
       len({x["clip_key"] for x in r["attempts"]}) == len(r["attempts"]),
       " ".join(x["clip_key"][:8] for x in r["attempts"]))
    ck("plotted schedules are the ones that ran",
       [x["schedule_hash"] for x in r["schedules_run"]]
       == [x["schedule_hash"] for x in r["attempts"]])
    ck("one candidate-producing ARDY call per attempt",
       r["ardy_calls"] == r["n_attempts"],
       f"{r['ardy_calls']} calls / {r['n_attempts']} attempts")
    ck("cost breakdown sums", abs(r["cost"]["w_route_term"] + r["cost"]["w_body_term"]
                                  + r["cost"]["w_clear_term"] - r["cost"]["total"]) < 1e-6)
    ck("repair case never claims physics validation",
       r["validation"]["physics_validated"] is False
       and r["validation"]["sonic_tracked"] is False)

    c = get(base, "/api/generate", **CLEAN_CASE)
    ck("clean case generates", c.get("ok"), c.get("reason", ""))
    ck("clean case needs no repair", c["repaired"] is False and len(c["repairs"]) == 0)
    ck("clean case verified on the first attempt", c["n_attempts"] == 1
       and c["validation"]["kinematic_collision_free"] and c["validation"]["meets_target_margin"],
       f"overhead {c['validation']['min_overhead_m']:+.4f} m")
    ck("clean and repair cases are distinguishable from the response",
       c["repaired"] != r["repaired"] and c["n_attempts"] != r["n_attempts"])

    d = get(base, "/api/generate", **{**REPAIR_CASE, "body_layer": "optimized"})
    ck("one-shot layer still available for comparison", d.get("ok"))
    ck("one-shot layer makes no margin claim",
       "meets_target_margin" not in d.get("validation", {}))

    ok = sum(1 for _, c_, _ in checks if c_)
    for name, c_, det in checks:
        print(f"  {'PASS' if c_ else 'FAIL'}  {name}{('  — ' + det) if det else ''}")
    print(f"\n{ok}/{len(checks)} checks passed")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
