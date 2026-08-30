"""Dependency-light HTTP UI for the partial-beam demo.

Standard library only -- `http.server` plus a single self-contained page. No web framework,
no build step, no npm. Planning is live on every slider move (three A* plans take ~130 ms on
CPU); GENERATION is separate and cached, so the UI never blocks on the GPU and a fully-cached
session never loads the model at all.

Run:  python -m scene2motion.demo.app --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..scenes import Scene
from . import renderer
from .cache import ClipCache
from .schedules import LAYER_LABEL, all_schedules
from .scene_builder import DEFAULTS, BeamParams, build
from .strategy_planner import PREFERENCE_LABEL, PREFERENCES, evaluate, evaluate_all

ROOT = Path(__file__).resolve().parents[1]
CACHE = ClipCache(ROOT / "demo_outputs" / "clips")
_gen_lock = threading.Lock()


# ---------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------

def _params(q: dict) -> BeamParams:
    return BeamParams(
        beam_height=float(q.get("height", [DEFAULTS["beam_height"]])[0]),
        beam_width=float(q.get("width", [DEFAULTS["beam_width"]])[0]),
        n_beams=int(float(q.get("n_beams", [DEFAULTS["n_beams"]])[0])),
        gap=float(q.get("gap", [DEFAULTS["gap"]])[0]),
    ).clamped()


def api_plan(q: dict) -> dict:
    params = _params(q)
    pref = q.get("preference", ["shortest"])[0]
    if pref not in PREFERENCES:
        pref = "shortest"
    scene = build(params)
    strategies = evaluate_all(scene)
    sel = strategies[pref]
    routes = {k: (s.plan.xy if s.feasible else None) for k, s in strategies.items()}
    sched = all_schedules(scene, sel.plan) if sel.feasible else None
    return {
        "scene_id": scene.scene_id,
        "params": {"beam_height": params.beam_height, "beam_width": params.beam_width,
                   "n_beams": params.n_beams, "gap": params.gap},
        "schedules": sched,
        "preference": pref,
        "bev": renderer.bev(scene, routes, pref),
        "side": renderer.side(scene),
        "panel": sel.to_dict(),
        "all": {k: {"feasible": s.feasible, "path_length_m": round(s.path_length_m, 2),
                    "goes_under_beam": s.goes_under_beam, "duck_required": s.duck_required,
                    "label": PREFERENCE_LABEL[k]}
                for k, s in strategies.items()},
    }


def api_auto(q: dict) -> dict:
    """AUTO: propose, generate, verify against real geometry, repair, decide.

    This is the only layer that reports what the motion ACTUALLY cleared rather than what the
    schedule asked for. Everything it returns about a repair is derived from a clip that was
    regenerated from the repaired schedule and reverified -- the attempt list carries each
    clip's key and the hash of the schedule it came from, so the claim is checkable here and
    not merely asserted.
    """
    import numpy as np

    from ..optim.response import DIP_MAX
    from ..optim.scheduler import MARGIN_M
    from ..verify import cost as costmod
    from ..verify.loop import run as run_loop
    from .schedules import SPEED, dip_for_layer, response

    params = _params(q)
    pref = q.get("preference", ["shortest"])[0]
    if pref not in PREFERENCES:
        pref = "shortest"
    allow = q.get("allow_generate", ["1"])[0] != "0"
    max_repairs = max(0, min(2, int(float(q.get("max_repairs", ["2"])[0]))))
    scene = build(params)
    strat = evaluate(scene, pref)
    if not strat.feasible:
        return {"ok": False, "reason": "no route under this preference",
                "refusal": strat.refusal}
    resp = response()
    sched = all_schedules(scene, strat.plan, speed=SPEED)
    dip0 = dip_for_layer(sched, "optimized")
    if resp is None or dip0 is None:
        return {"ok": False, "reason": "no Phase-3 schedule available for this scene"}

    s_m = np.asarray(sched["s_m"], float)
    q0 = np.clip(np.asarray(dip0, float) / DIP_MAX, 0.0, 1.0)
    with _gen_lock:                      # one GPU job at a time; LUCID shares this device
        res = run_loop(scene, strat.plan, q0, resp, CACHE, preference=pref,
                       max_repairs=max_repairs, allow_generate=allow,
                       provenance={"tcn_dataset_hash": sched.get("tcn_dataset_hash"),
                                   "tcn_margin_m": sched.get("tcn_margin_m")})
    if not res.attempts:
        return {"ok": False, "reason": res.reason or "not cached", "source": "miss"}

    final = res.final
    anim = renderer.frames(scene, CACHE.get(final.key)[0])
    d = res.to_dict()
    # Every schedule the loop actually ran, in metres, on the plot's grid. The first is the
    # TCN's proposal; each later one is what a repair produced and what was regenerated from.
    ladder = [{"iteration": a.iteration, "schedule_hash": a.schedule_hash,
               "clip_key": a.key, "dip_m": d["dips_m"][a.iteration],
               "collision_free": a.trace.collision_free,
               "min_overhead_m": round(a.trace.min_overhead_m, 4)}
              for a in res.attempts]
    u_final = np.asarray(d["provenance"]["dip_final_m"], float) / DIP_MAX
    breakdown = costmod.evaluate(u_final, s_m, final.trace.min_overhead_m, pref).to_dict()
    return {"ok": True, "body_layer": "auto", "anim": anim, "outcome": d["outcome"],
            "reason": d["reason"], "repaired": d["repaired"], "ardy_calls": d["ardy_calls"],
            "n_attempts": d["n_attempts"], "cache_hits": d["cache_hits"],
            "attempts": d["attempts"], "repairs": d["repairs"], "provenance": d["provenance"],
            "target_m": MARGIN_M, "cache_key": final.key, "schedules_run": ladder,
            "cost": breakdown, "source": final.source,
            "validation": _auto_validation(res, MARGIN_M)}


def _auto_validation(res, target_m: float) -> dict:
    """The claims AUTO is allowed to make, each tied to a measured quantity.

    Collision-free and meets-target are separate keys because they are separate facts, and
    `repaired` is only true when the accepted clip's schedule hash matches the last repair's
    output -- so the word cannot attach to a pre-repair clip.
    """
    tr = res.final.trace
    return {"kinematic_collision_free": bool(tr.collision_free),
            "meets_target_margin": not tr.below_margin(target_m),
            "min_overhead_m": round(tr.min_overhead_m, 4),
            "min_clearance_m": round(tr.min_clearance_m, 4),
            "max_deficit_m": round(float(tr.deficit(target_m).max()), 4),
            "max_lateral_deficit_m": round(float(tr.lateral_deficit(target_m).max()), 4),
            "goal_error_m": round(tr.goal_error_m, 4),
            "repaired": res.repaired,
            "verified_against": "MuJoCo collision geometry, per route position",
            "sonic_tracked": False,
            "physics_validated": False,
            "statement": ("kinematic collision-free" if tr.collision_free
                          else "kinematic COLLISION") + " · not physics validated"}


def api_generate(q: dict) -> dict:
    from .ardy_runner import generate
    params = _params(q)
    pref = q.get("preference", ["shortest"])[0]
    allow = q.get("allow_generate", ["1"])[0] != "0"
    body_layer = q.get("body_layer", ["heuristic"])[0]
    if body_layer == "auto":
        return api_auto(q)
    if body_layer not in ("heuristic", "learned", "optimized"):
        body_layer = "heuristic"
    scene = build(params)
    strat = evaluate(scene, pref)
    if not strat.feasible:
        return {"ok": False, "reason": "no route under this preference",
                "refusal": strat.refusal}
    with _gen_lock:                      # one GPU job at a time; LUCID shares this device
        res = generate(scene, strat.plan, pref, CACHE, allow_generate=allow,
                       body_layer=body_layer)
    if res["qpos"] is None:
        return {"ok": False, "reason": "not cached", "cache_key": res["key"],
                "source": "miss"}
    anim = renderer.frames(scene, res["qpos"])
    meta = res["meta"]
    return {"ok": True, "source": res["source"], "cache_key": res["key"],
            "body_layer": body_layer, "anim": anim, "clip": meta,
            "validation": _validation(meta)}


def _validation(meta: dict) -> dict:
    """The three status levels the demo is allowed to claim, kept explicitly distinct.

    A kinematically collision-free clip has NOT been shown to be physically executable, and
    the UI must never let one imply the other. SONIC tracking is a separate stage that this
    demo does not run inline.
    """
    return {
        "kinematic_collision_free": bool(meta.get("collision_free", False)),
        "min_clearance_m": meta.get("min_clearance_m"),
        "goal_error_m": meta.get("goal_error_m"),
        "sonic_tracked": False,
        "physics_validated": False,
        "statement": ("kinematic collision-free" if meta.get("collision_free")
                      else "kinematic COLLISION") + " · not physics validated",
    }


def api_cache(_q: dict) -> dict:
    return CACHE.stats()


ROUTES = {"/api/plan": api_plan, "/api/generate": api_generate, "/api/auto": api_auto,
          "/api/cache": api_cache}


# ---------------------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):      # keep the console readable
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        fn = ROUTES.get(u.path)
        if fn is None:
            return self._send(404, b'{"error":"not found"}', "application/json")
        try:
            payload = fn(parse_qs(u.query))
            body = json.dumps(payload).encode()
            return self._send(200, body, "application/json")
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(),
                              "application/json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"scene2motion demo  ->  http://{args.host}:{args.port}")
    print(f"clip cache: {CACHE.root}  ({CACHE.stats()['n_entries']} entries)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Route First, Body Next — partial beam</title>
<style>
:root{
  --ink:#e8eaed; --dim:#9aa3ad; --line:#2a3038; --bg:#12151a; --panel:#181c22;
  --under:#f0a04b; --around:#5bc8af; --sel:#e8eaed; --warn:#e0574f; --ok:#5bc8af;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
       align-items:baseline;gap:16px;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
header .sub{color:var(--dim);font-size:12px}
.wrap{display:grid;grid-template-columns:minmax(300px,1fr) minmax(360px,1.35fr) 320px;
      gap:1px;background:var(--line);min-height:calc(100vh - 56px)}
.col{background:var(--bg);padding:16px;overflow:auto}
.col h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
        margin:0 0 12px;font-weight:600}
.controls{display:flex;flex-direction:column;gap:14px;margin-bottom:18px}
.ctl label{display:flex;justify-content:space-between;font-size:12px;color:var(--dim);
           margin-bottom:5px}
.ctl label b{color:var(--ink);font-weight:600}
input[type=range]{width:100%;accent-color:var(--under)}
.prefs{display:flex;gap:6px;flex-wrap:wrap}
.prefs button{flex:1 1 auto;background:var(--panel);color:var(--dim);border:1px solid var(--line);
  padding:7px 9px;border-radius:5px;cursor:pointer;font:inherit;font-size:11.5px}
.prefs button[aria-pressed=true]{background:#232a33;color:var(--ink)}
.prefs button[aria-pressed=true]{border-color:#4a5560;color:var(--ink)}
#gen{width:100%;margin-top:4px;background:var(--under);color:#1a1205;border:0;padding:10px;
     border-radius:5px;font:inherit;font-weight:700;cursor:pointer}
#gen[disabled]{opacity:.5;cursor:progress}
svg{display:block;width:100%;height:auto;background:var(--panel);border:1px solid var(--line);
    border-radius:6px}
.legend{display:flex;gap:14px;font-size:11px;color:var(--dim);margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
.rows{border-top:1px solid var(--line)}
.row{display:flex;justify-content:space-between;gap:10px;padding:7px 0;
     border-bottom:1px solid var(--line);font-size:12.5px}
.row span{color:var(--dim)} .row b{font-weight:600;text-align:right}
.badge{display:inline-block;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;
       letter-spacing:.03em}
.b-ok{background:rgba(91,200,175,.15);color:var(--ok)}
.b-warn{background:rgba(224,87,79,.15);color:var(--warn)}
.b-mute{background:#232a33;color:var(--dim)}
.status{margin-top:12px;padding:10px;background:var(--panel);border:1px solid var(--line);
        border-radius:6px;font-size:11.5px;color:var(--dim);line-height:1.7}
.playbar{display:flex;align-items:center;gap:10px;margin-top:10px}
.playbar button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:5px;padding:5px 11px;font:inherit;cursor:pointer}
#scrub{flex:1;accent-color:var(--around)}
.state{font-weight:700;letter-spacing:.06em}
.hint{color:var(--dim);font-size:11.5px;margin-top:10px}
details#adv{margin-top:8px} details#adv summary{color:var(--dim);font-size:11.5px;cursor:pointer}
.tl{display:flex;flex-direction:column;gap:0}
.tl .step{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--line)}
.tl .dot{flex:0 0 9px;height:9px;border-radius:50%;margin-top:5px}
.tl .body{flex:1;font-size:12px;line-height:1.6}
.tl .body .h{font-weight:600} .tl .body .m{color:var(--dim);font-size:11px}
.tl code{color:var(--dim);font-size:10.5px}
.tl .arrow{color:var(--dim);padding:2px 0 2px 19px;font-size:11px}
</style>
<header>
  <h1>Route First, Body Next</h1>
  <span class="sub">scene → route → body adaptation → frozen ARDY → G1</span>
  <span class="sub" id="cachestat"></span>
</header>
<div class="wrap">
  <div class="col">
    <h2>Scene &amp; routes</h2>
    <div class="controls">
      <div class="ctl">
        <label>beams <b><span id="nbv">1</span></b></label>
        <div class="prefs" id="presets"></div>
      </div>
      <div class="ctl">
        <label>beam height <b><span id="hv">1.00</span> m</b></label>
        <input type="range" id="h" min="0.60" max="1.60" step="0.05" value="1.00">
      </div>
      <div class="ctl" id="gapctl" hidden>
        <label>beam gap <b><span id="gv">3.00</span> m</b></label>
        <input type="range" id="g" min="0.80" max="5.50" step="0.10" value="3.00">
      </div>
      <div class="ctl">
        <label>beam width <b><span id="wv">1.45</span> m</b></label>
        <input type="range" id="w" min="0.30" max="2.25" step="0.05" value="1.45">
      </div>
      <div class="ctl">
        <label>preference</label>
        <div class="prefs" id="prefs"></div>
      </div>
      <div class="ctl">
        <label>body layer</label>
        <div class="prefs" id="layers"></div>
        <details id="adv">
          <summary>Advanced · compare body layers</summary>
          <div class="prefs" id="layers2" style="margin-top:8px"></div>
          <div class="hint">These generate once and report what came back. Only AUTO
            verifies the result against the scene and repairs it.</div>
        </details>
      </div>
      <button id="gen">Generate motion</button>
    </div>
    <svg id="bev" viewBox="0 0 420 260" role="img" aria-label="Bird's eye view of the scene and planned routes"></svg>
    <div class="legend">
      <span><i style="background:var(--under)"></i>under (duck)</span>
      <span><i style="background:var(--around)"></i>around (upright)</span>
      <span><i style="background:var(--sel);height:5px"></i>selected</span>
    </div>
  </div>

  <div class="col">
    <h2>G1 collision geometry <span id="statebadge"></span></h2>
    <svg id="side" viewBox="0 0 560 300" role="img" aria-label="Side view animation of the G1 passing the beam"></svg>
    <div class="playbar">
      <button id="play">▶</button>
      <input type="range" id="scrub" min="0" max="0" value="0">
      <span class="sub" id="frameno" style="color:var(--dim);font-size:11.5px"></span>
    </div>
    <svg id="bevbot" viewBox="0 0 560 150" role="img" aria-label="Bird's eye view of the robot during the motion" style="margin-top:10px"></svg>
    <h2 style="margin-top:18px">Duck schedule</h2>
    <svg id="sched" viewBox="0 0 560 170" role="img"
         aria-label="Commanded duck depth against distance along the route"></svg>
    <div class="legend" id="schedlegend"></div>
    <div class="hint">Shapes are the robot's own MuJoCo collision primitives — the same
      geometry the collision status refers to, not a decorative mesh.</div>
  </div>

  <div class="col">
    <h2>Planner decision</h2>
    <div class="rows" id="panel"></div>
    <div class="status" id="valid"></div>
    <div id="repairwrap" hidden>
      <h2 style="margin-top:20px">Verify &amp; repair</h2>
      <div id="timeline"></div>
      <div class="hint" id="repairnote"></div>
    </div>
    <div id="costwrap" hidden>
      <h2 style="margin-top:20px">Route / body cost</h2>
      <div class="rows" id="cost"></div>
    </div>
    <h2 style="margin-top:20px">All preferences</h2>
    <div class="rows" id="allprefs"></div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s), PREFS=[["shortest","Shortest Path"],["upright","Stay Upright"],["clearance","Maximum Clearance"]];
let pref="shortest", layer="auto", planData=null, anim=null, playing=false, fi=0, timer=null;
let lastAuto=null;
// AUTO is the public default: it is the only layer that checks its own output. The three
// one-shot layers stay available for comparison, behind a disclosure.
const AUTO_LAYER=[["auto","AUTO — TCN + Verify/Repair"]];
const LAYERS=[["heuristic","Heuristic"],["learned","Phase-2 Learned"],["optimized","Phase-3 Optimized"]];
const ALL_LAYERS=AUTO_LAYER.concat(LAYERS);
const PRESETS_SCENE=[["1","1"],["2","2"],["3","3"],["4","4"],["5","5"],["6","6"]];
let preset="1";
const SCHED_COLOUR={heuristic:"#9aa3ad",learned:"#7aa2f7",optimizer:"#f0a04b",optimized:"#5bc8af"};
const SCHED_LABEL={heuristic:"heuristic",learned:"Phase-2 learned",
                   optimizer:"optimizer (teacher)",optimized:"Phase-3 optimized"};

PREFS.forEach(([k,l])=>{const b=document.createElement("button");b.textContent=l;b.dataset.k=k;
  b.onclick=()=>{pref=k;syncPrefs();refresh();};$("#prefs").appendChild(b);});
function syncPrefs(){[...$("#prefs").children].forEach(b=>b.setAttribute("aria-pressed",b.dataset.k===pref));}
syncPrefs();
function addLayer(host,k,l){const b=document.createElement("button");b.textContent=l;b.dataset.k=k;
  b.onclick=()=>{layer=k;lastAuto=null;syncLayers();showAuto(null);if(planData)drawSched(planData.schedules);};
  $(host).appendChild(b);}
AUTO_LAYER.forEach(([k,l])=>addLayer("#layers",k,l));
LAYERS.forEach(([k,l])=>addLayer("#layers2",k,l));
function syncLayers(){[...$("#layers").children,...$("#layers2").children]
  .forEach(b=>b.setAttribute("aria-pressed",b.dataset.k===layer));
  $("#adv").open = layer!=="auto";}
syncLayers();
PRESETS_SCENE.forEach(([k,l])=>{const b=document.createElement("button");b.textContent=l;b.dataset.k=k;
  b.onclick=()=>{preset=k;syncPreset();refresh();};$("#presets").appendChild(b);});
function syncPreset(){[...$("#presets").children].forEach(b=>b.setAttribute("aria-pressed",b.dataset.k===preset));
  $("#gapctl").hidden = preset==="1"; $("#nbv").textContent=preset;}
function nBeams(){return +preset;}
syncPreset();

const fmt=(v,u="",d=2)=>v==null?"—":(typeof v==="number"?v.toFixed(d):v)+u;
function rows(el,items){el.innerHTML=items.map(([k,v])=>
  `<div class="row"><span>${k}</span><b>${v}</b></div>`).join("");}

// ---- bird's eye -------------------------------------------------------------------
function drawBEV(svg,bev,W,H,withRobot){
  const [x0,x1,y0,y1]=bev.bounds, pad=10;
  const sx=v=>pad+(v-x0)/(x1-x0)*(W-2*pad), sy=v=>H/2-(v)/(y1-y0)*(H-2*pad);
  let s=`<rect x="0" y="0" width="${W}" height="${H}" fill="#181c22"/>`;
  bev.walls.forEach(w=>{s+=`<rect x="${sx(w.x_lo)}" y="${sy(w.y_hi)}" width="${sx(w.x_hi)-sx(w.x_lo)}" height="${sy(w.y_lo)-sy(w.y_hi)}" fill="#232a33"/>`;});
  const b=bev.beam;
  s+=`<rect x="${sx(b.x_lo)}" y="${sy(b.y_hi)}" width="${Math.max(2,sx(b.x_hi)-sx(b.x_lo))}" height="${sy(b.y_lo)-sy(b.y_hi)}" fill="rgba(240,160,75,.30)" stroke="var(--under)" stroke-width="1.2"/>`;
  const colour={shortest:"var(--under)",upright:"var(--around)",clearance:"#7aa2f7"};
  for(const [k,pts] of Object.entries(bev.routes)){
    if(!pts||pts.length<2)continue;
    const d=pts.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");
    const on=k===bev.selected;
    s+=`<path d="${d}" fill="none" stroke="${colour[k]||'#888'}" stroke-width="${on?3.2:1.4}" opacity="${on?1:.45}" stroke-linecap="round"/>`;
  }
  s+=`<circle cx="${sx(bev.start[0])}" cy="${sy(bev.start[1])}" r="4.5" fill="#e8eaed"/>`;
  s+=`<circle cx="${sx(bev.goal[0])}" cy="${sy(bev.goal[1])}" r="4.5" fill="none" stroke="#e8eaed" stroke-width="2"/>`;
  if(withRobot&&anim){const f=anim.frames[fi];
    s+=`<circle cx="${sx(f.pelvis[0])}" cy="${sy(f.pelvis[1])}" r="5" fill="var(--around)"/>`;}
  svg.innerHTML=s;
}

// ---- side view --------------------------------------------------------------------
function drawSide(){
  const svg=$("#side"),W=560,H=300,pad=14;
  if(!planData){svg.innerHTML="";return;}
  const sd=planData.side, x0=sd.x_lo, x1=sd.x_hi, zmax=2.0;
  const sx=v=>pad+(v-x0)/(x1-x0)*(W-2*pad), sz=v=>H-pad-(v/zmax)*(H-2*pad);
  let s=`<rect x="0" y="0" width="${W}" height="${H}" fill="#181c22"/>`;
  s+=`<line x1="${pad}" y1="${sz(0)}" x2="${W-pad}" y2="${sz(0)}" stroke="#3a424c" stroke-width="1.5"/>`;
  const b=sd.beam;
  s+=`<rect x="${sx(b.x_lo)}" y="${sz(b.z_hi)}" width="${Math.max(3,sx(b.x_hi)-sx(b.x_lo))}" height="${sz(b.z_lo)-sz(b.z_hi)}" fill="rgba(240,160,75,.35)" stroke="var(--under)" stroke-width="1.2"/>`;
  s+=`<line x1="${pad}" y1="${sz(b.z_lo)}" x2="${W-pad}" y2="${sz(b.z_lo)}" stroke="var(--under)" stroke-dasharray="3 4" stroke-width="1" opacity=".5"/>`;
  if(anim){
    const path=anim.pelvis_path;
    s+=`<path d="${path.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sz(0.02).toFixed(1)).join(" ")}" fill="none" stroke="#3a424c" stroke-width="1"/>`;
    const f=anim.frames[fi];
    for(const sh of f.s){
      const [ax,ay,az,bx,by,bz,r]=sh, rr=Math.max(1.5,r/zmax*(H-2*pad));
      if(Math.abs(ax-bx)<1e-6&&Math.abs(az-bz)<1e-6)
        s+=`<circle cx="${sx(ax).toFixed(1)}" cy="${sz(az).toFixed(1)}" r="${rr.toFixed(1)}" fill="rgba(122,162,247,.55)"/>`;
      else
        s+=`<line x1="${sx(ax).toFixed(1)}" y1="${sz(az).toFixed(1)}" x2="${sx(bx).toFixed(1)}" y2="${sz(bz).toFixed(1)}" stroke="rgba(122,162,247,.55)" stroke-width="${(2*rr).toFixed(1)}" stroke-linecap="round"/>`;
    }
    s+=`<circle cx="${sx(f.pelvis[0])}" cy="${sz(f.pelvis[2])}" r="3.5" fill="var(--around)"/>`;
    s+=`<text x="${W-pad}" y="${pad+12}" text-anchor="end" fill="#9aa3ad" font-size="11">top ${f.top.toFixed(2)} m</text>`;
  }
  svg.innerHTML=s;
}

function tick(){ if(!anim)return; fi=(fi+1)%anim.frames.length; $("#scrub").value=fi; paint(); }
function paint(){
  drawSide();
  if(planData) drawBEV($("#bevbot"),planData.bev,560,150,true);
  if(anim){const f=anim.frames[fi];
    $("#statebadge").innerHTML=`<span class="badge ${f.state==="DUCK"?"b-warn":"b-mute"} state">${f.state}</span>`;
    $("#frameno").textContent=`frame ${fi+1}/${anim.frames.length}`;}
}
$("#play").onclick=()=>{playing=!playing;$("#play").textContent=playing?"❚❚":"▶";
  if(timer)clearInterval(timer); if(playing)timer=setInterval(tick,55);};
$("#scrub").oninput=e=>{fi=+e.target.value;paint();};

// ---- duck-schedule plot -------------------------------------------------------------
function drawSched(sc){
  const svg=$("#sched"),W=560,H=170,padL=46,padR=12,padT=12,padB=26;
  if(!sc||!sc.schedules){svg.innerHTML="";$("#schedlegend").innerHTML="";return;}
  const S=sc.s_m, L=sc.route_len_m||1;
  let ymax=0.05; for(const v of Object.values(sc.schedules)) ymax=Math.max(ymax,...v);
  ymax=Math.max(0.10,Math.ceil(ymax*20)/20);
  const sx=v=>padL+(v/L)*(W-padL-padR), sy=v=>H-padB-(v/ymax)*(H-padT-padB);
  let s=`<rect x="0" y="0" width="${W}" height="${H}" fill="#181c22"/>`;
  // beam spans, so the reader can see the schedule leading the obstacle
  for(const b of (sc.beams||[]))
    s+=`<rect x="${sx(b.s_lo)}" y="${padT}" width="${Math.max(2,sx(b.s_hi)-sx(b.s_lo))}" height="${H-padT-padB}" fill="rgba(240,160,75,.18)"/>`;
  s+=`<line x1="${padL}" y1="${sy(0)}" x2="${W-padR}" y2="${sy(0)}" stroke="#3a424c"/>`;
  s+=`<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${sy(0)}" stroke="#3a424c"/>`;
  for(const f of [0,0.5,1]){const v=ymax*f;
    s+=`<text x="${padL-6}" y="${sy(v)+4}" text-anchor="end" fill="#9aa3ad" font-size="10">${(v*100).toFixed(0)}</text>`;}
  s+=`<text x="${padL-6}" y="${padT+2}" text-anchor="end" fill="#9aa3ad" font-size="9">cm</text>`;
  for(let d=0; d<=L; d+=2)
    s+=`<text x="${sx(d)}" y="${H-8}" text-anchor="middle" fill="#9aa3ad" font-size="10">${d}</text>`;
  s+=`<text x="${W-padR}" y="${H-8}" text-anchor="end" fill="#9aa3ad" font-size="10">route distance (m)</text>`;
  // The schedules AUTO actually generated from, drawn over the proposals. Iteration 0 is
  // the TCN's proposal; each later line is what a repair produced and was regenerated from.
  if(lastAuto&&lastAuto.schedules_run){
    lastAuto.schedules_run.forEach((run,i)=>{
      const d=run.dip_m.map((y,j)=>(j?"L":"M")+sx(S[j]).toFixed(1)+" "+sy(y).toFixed(1)).join(" ");
      const col=run.collision_free?"#5bc8af":"#e0574f";
      s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="${i===lastAuto.schedules_run.length-1?3:1.6}"`
        +` opacity="${i===lastAuto.schedules_run.length-1?1:.55}"`
        +` ${i?"":'stroke-dasharray="4 3"'}/>`;
    });
  }
  const order=["optimizer","heuristic","learned","optimized"];
  for(const k of order){
    const v=sc.schedules[k]; if(!v) continue;
    const d=v.map((y,i)=>(i?"L":"M")+sx(S[i]).toFixed(1)+" "+sy(y).toFixed(1)).join(" ");
    const sel=(k===layer)||(k==="optimizer"&&layer==="optimized"&&!sc.schedules.optimized);
    const dash=(k==="heuristic")?'stroke-dasharray="5 4"':(k==="optimizer"?'stroke-dasharray="2 3"':"");
    s+=`<path d="${d}" fill="none" stroke="${SCHED_COLOUR[k]}" stroke-width="${sel?2.6:1.3}" opacity="${sel?1:.65}" ${dash}/>`;
  }
  svg.innerHTML=s;
  let leg = order.filter(k=>sc.schedules[k]).map(k=>
    `<span><i style="background:${SCHED_COLOUR[k]}"></i>${SCHED_LABEL[k]}${k===layer?" (selected)":""}</span>`);
  if(lastAuto&&lastAuto.schedules_run) leg = lastAuto.schedules_run.map(r=>
    `<span><i style="background:${r.collision_free?"#5bc8af":"#e0574f"}"></i>`
    +`${r.iteration?`repair ${r.iteration}`:"TCN proposal"} · ${(r.min_overhead_m*100).toFixed(1)} cm`
    +`${r.collision_free?"":" · COLLISION"}</span>`).concat(leg);
  $("#schedlegend").innerHTML=leg.join("");
}

// ---- verify / repair timeline -------------------------------------------------------
// One row per ARDY call, each stating what came back and what was done about it. A row that
// says "repaired" names the clip key it produced, so the claim points at bytes.
function showAuto(d){
  lastAuto=d;
  $("#repairwrap").hidden = !d; $("#costwrap").hidden = !d;
  if(!d) return;
  const T=d.target_m, steps=[];
  d.attempts.forEach((a,i)=>{
    const ok=a.collision_free&&a.meets_target;
    const col=a.collision_free?(a.meets_target?"var(--ok)":"#f0a04b"):"var(--warn)";
    const verdict = !a.collision_free ? "COLLISION"
        : (a.meets_target ? `meets the ${(T*100).toFixed(0)} cm target` : `short of target`);
    steps.push(`<div class="step"><div class="dot" style="background:${col}"></div><div class="body">
      <div class="h">${i?`Attempt ${i} — repaired schedule`:"Attempt 0 — TCN proposal"}</div>
      <div>${verdict} · headroom <b>${(a.min_overhead_m*100).toFixed(1)} cm</b>
        ${a.max_deficit_m>0?`· deficit ${(a.max_deficit_m*1000).toFixed(0)} mm`:""}</div>
      <div class="m">peak dip ${(a.peak_dip_m*100).toFixed(1)} cm · ${a.source}</div>
      <code>schedule ${a.schedule_hash} → clip ${a.clip_key}</code></div></div>`);
    const r=d.repairs[i];
    if(r) steps.push(`<div class="arrow">↓ repair ${r.iteration}: +${(r.repair_magnitude_m*100).toFixed(1)} cm dip,
      starts ${Math.abs(r.onset_shift_m*100).toFixed(0)} cm ${r.onset_shift_m<=0?"earlier":"later"},
      held ${(r.duration_change_m*100).toFixed(0)} cm longer${r.slope_floor_bound?" · gain floor bound":""}</div>`);
  });
  $("#timeline").innerHTML=`<div class="tl">${steps.join("")}</div>`;
  $("#repairnote").innerHTML = d.repaired
    ? `The clip shown was regenerated from the repaired schedule
       <code>${d.provenance.final_schedule_hash}</code> and reverified independently. It is not
       the proposal clip <code>${d.provenance.initial_schedule_hash}</code>.`
    : `No repair was applied — the proposal verified on the first attempt.
       ${d.ardy_calls} ARDY calls.`;
  const c=d.cost;
  rows($("#cost"),[
    ["route length",`${c.route_len_m.toFixed(2)} m`],
    ["body effort ∫u²",c.j_effort.toFixed(3)],
    ["body rate ∫(du/ds)²",c.j_rate.toFixed(3)],
    ["body smoothness ∫(d²u/ds²)²",c.j_smooth.toFixed(3)],
    ["J_body",c.j_body.toFixed(2)],
    ["worst headroom C_min",`${(c.c_min_m*100).toFixed(1)} cm`],
    ["weights",`w_L ${c.weights.w_L} · w_B ${c.weights.w_B} · w_C ${c.weights.w_C}`],
    ["J = w_L·L + w_B·J_body − w_C·C_min",
     `${c.w_route_term.toFixed(1)} + ${c.w_body_term.toFixed(1)} − ${(-c.w_clear_term).toFixed(2)} = <b>${c.total.toFixed(1)}</b>`],
  ]);
}

// ---- data -------------------------------------------------------------------------
async function refresh(){
  const h=$("#h").value,w=$("#w").value,g=$("#g").value;
  const nb = nBeams();
  $("#hv").textContent=(+h).toFixed(2); $("#wv").textContent=(+w).toFixed(2);
  $("#gv").textContent=(+g).toFixed(2);
  const r=await fetch(`/api/plan?height=${h}&width=${w}&n_beams=${nb}&gap=${g}&preference=${pref}`);
  planData=await r.json();
  drawBEV($("#bev"),planData.bev,420,260,false);
  drawSched(planData.schedules);
  const p=planData.panel;
  const duck = !p.duck_required ? "not required"
    : (p.duck_held_throughout ? `held throughout (from ${fmt(p.duck_start_s,"s")})`
       : `${fmt(p.duck_start_s,"s")} → ${fmt(p.duck_end_s,"s")}`);
  rows($("#panel"),[
    ["strategy",p.preference_label],
    ["route found",p.feasible?'<span class="badge b-ok">yes</span>':'<span class="badge b-warn">refused</span>'],
    ["path length",fmt(p.path_length_m," m")],
    ["passes under beam",p.goes_under_beam?"yes":"no"],
    ["duck required",p.duck_required?"yes":"no"],
    ["duck window",duck],
    ["anticipation",fmt(p.lead_s,"s",1)],
    ["deepest body mode",p.deepest_mode],
    ["headroom under beam",p.min_top_clearance_m==null?"—":fmt(p.min_top_clearance_m," m")],
  ]);
  rows($("#allprefs"),Object.entries(planData.all).map(([k,v])=>
    [v.label,`${v.feasible?fmt(v.path_length_m," m"):"refused"} · ${v.goes_under_beam?"under":"around"}`]));
  $("#valid").innerHTML=`<b>Motion not generated yet.</b><br>Plan is geometric only —
     press <b>Generate motion</b> to run the frozen prior and collision-check the result.`;
  anim=null; $("#scrub").max=0; $("#statebadge").innerHTML=""; showAuto(null); paint();
  fetch("/api/cache").then(r=>r.json()).then(c=>{$("#cachestat").textContent=`cache: ${c.n_entries} clips`;});
}
$("#h").oninput=refresh; $("#w").oninput=refresh; $("#g").oninput=refresh;

// Deep-linking: ?height=1.0&width=1.45&preference=shortest&auto=1 restores a state and can
// auto-generate, so a specific view is linkable, screenshot-able and recordable.
function applyURL(){
  const q=new URLSearchParams(location.search);
  if(q.has("height")) $("#h").value=q.get("height");
  if(q.has("width"))  $("#w").value=q.get("width");
  if(q.has("preference")&&PREFS.some(([k])=>k===q.get("preference"))) pref=q.get("preference");
  if(q.has("body_layer")&&ALL_LAYERS.some(([k])=>k===q.get("body_layer"))) layer=q.get("body_layer");
  if(q.has("n_beams")){const n=Math.max(1,Math.min(6,+q.get("n_beams")||1)); preset=String(n);}
  if(q.has("gap")) $("#g").value=q.get("gap");
  syncPrefs(); syncLayers(); syncPreset();
  return {auto:q.get("auto")==="1", frame:q.has("frame")?+q.get("frame"):null};
}

$("#gen").onclick=async()=>{
  const b=$("#gen"); b.disabled=true; b.textContent="Generating…";
  try{
    const h=$("#h").value,w=$("#w").value;
    const nb=nBeams(), g=$("#g").value;
    const r=await fetch(`/api/generate?height=${h}&width=${w}&n_beams=${nb}&gap=${g}&preference=${pref}&body_layer=${layer}`);
    const d=await r.json();
    if(!d.ok){$("#valid").innerHTML=`<span class="badge b-warn">no motion</span> ${d.reason}`;
              showAuto(null);}
    else if(d.body_layer==="auto"){
      anim=d.anim; fi=0; $("#scrub").max=anim.frames.length-1; $("#scrub").value=0;
      const v=d.validation;
      // Three separate claims, never merged: it did not collide, it did or did not leave the
      // margin that was asked for, and nothing here has been physically validated.
      $("#valid").innerHTML=
        `<span class="badge ${v.kinematic_collision_free?"b-ok":"b-warn"}">`+
        `${v.kinematic_collision_free?"kinematic collision-free":"KINEMATIC COLLISION"}</span>
         <span class="badge ${v.meets_target_margin?"b-ok":"b-mute"}">`+
        `${v.meets_target_margin?`meets ${(d.target_m*100).toFixed(0)} cm target`:`below ${(d.target_m*100).toFixed(0)} cm target`}</span>
         <span class="badge b-mute">not physics validated</span><br>
         <b>${d.outcome}</b> — ${d.reason}<br>
         headroom ${fmt(v.min_overhead_m," m")} · nearest anything ${fmt(v.min_clearance_m," m")} ·
         goal error ${fmt(v.goal_error_m," m")}<br>
         ${v.max_lateral_deficit_m>0?`<span style="opacity:.75">lateral clearance is
           ${(v.max_lateral_deficit_m*1000).toFixed(0)} mm inside the target — a route question,
           not something a duck can fix.</span><br>`:""}
         ${d.ardy_calls} ARDY calls over ${d.n_attempts} attempt${d.n_attempts>1?"s":""}
         (${d.cache_hits} served from cache)<br>
         <span style="opacity:.7">clip ${d.cache_key} · verified against ${v.verified_against}</span><br>
         SONIC tracked: <b>no</b> — tracking is a separate offline stage.`;
      showAuto(d); drawSched(planData&&planData.schedules); paint();
    }
    else{
      anim=d.anim; fi=0; $("#scrub").max=anim.frames.length-1; $("#scrub").value=0;
      showAuto(null);
      const v=d.validation, c=d.clip;
      $("#valid").innerHTML=
        `<span class="badge ${v.kinematic_collision_free?"b-ok":"b-warn"}">`+
        `${v.kinematic_collision_free?"kinematic collision-free":"KINEMATIC COLLISION"}</span>
         <span class="badge b-mute">not verified against the target margin</span>
         <span class="badge b-mute">not physics validated</span><br>
         min clearance ${fmt(v.min_clearance_m," m")} · goal error ${fmt(v.goal_error_m," m")}<br>
         body layer <b>${d.body_layer}</b> · source <b>${d.source}</b> · ${c.n_frames} frames @ ${c.fps} fps · ${c.steps} steps
         ${c.generate_s?`· ${c.generate_s.toFixed(1)} s`:""}<br>
         <span style="opacity:.7">key ${d.cache_key}</span><br>
         SONIC tracked: <b>no</b> — tracking is a separate offline stage.`;
      paint();
      fetch("/api/cache").then(r=>r.json()).then(c=>{$("#cachestat").textContent=`cache: ${c.n_entries} clips`;});
    }
  }catch(e){$("#valid").textContent="error: "+e;}
  b.disabled=false; b.textContent="Generate motion";
};
(async()=>{
  const o=applyURL(); await refresh();
  if(o.auto){ await $("#gen").onclick();
    if(o.frame!=null && anim){ fi=Math.max(0,Math.min(anim.frames.length-1,o.frame));
                               $("#scrub").value=fi; paint(); } }
})();
</script>
"""

if __name__ == "__main__":
    main()
