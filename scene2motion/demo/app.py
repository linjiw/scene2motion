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
from .scene_builder import DEFAULTS, BeamParams, build
from .strategy_planner import PREFERENCE_LABEL, PREFERENCES, evaluate, evaluate_all

ROOT = Path(__file__).resolve().parents[1]
CACHE = ClipCache(ROOT / "demo_outputs" / "clips")
_gen_lock = threading.Lock()


# ---------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------

def api_plan(q: dict) -> dict:
    params = BeamParams(
        beam_height=float(q.get("height", [DEFAULTS["beam_height"]])[0]),
        beam_width=float(q.get("width", [DEFAULTS["beam_width"]])[0]),
    ).clamped()
    pref = q.get("preference", ["shortest"])[0]
    if pref not in PREFERENCES:
        pref = "shortest"
    scene = build(params)
    strategies = evaluate_all(scene)
    sel = strategies[pref]
    routes = {k: (s.plan.xy if s.feasible else None) for k, s in strategies.items()}
    return {
        "scene_id": scene.scene_id,
        "params": {"beam_height": params.beam_height, "beam_width": params.beam_width},
        "preference": pref,
        "bev": renderer.bev(scene, routes, pref),
        "side": renderer.side(scene),
        "panel": sel.to_dict(),
        "all": {k: {"feasible": s.feasible, "path_length_m": round(s.path_length_m, 2),
                    "goes_under_beam": s.goes_under_beam, "duck_required": s.duck_required,
                    "label": PREFERENCE_LABEL[k]}
                for k, s in strategies.items()},
    }


def api_generate(q: dict) -> dict:
    from .ardy_runner import generate
    params = BeamParams(
        beam_height=float(q.get("height", [DEFAULTS["beam_height"]])[0]),
        beam_width=float(q.get("width", [DEFAULTS["beam_width"]])[0]),
    ).clamped()
    pref = q.get("preference", ["shortest"])[0]
    allow = q.get("allow_generate", ["1"])[0] != "0"
    scene = build(params)
    strat = evaluate(scene, pref)
    if not strat.feasible:
        return {"ok": False, "reason": "no route under this preference",
                "refusal": strat.refusal}
    with _gen_lock:                      # one GPU job at a time; LUCID shares this device
        res = generate(scene, strat.plan, pref, CACHE, allow_generate=allow)
    if res["qpos"] is None:
        return {"ok": False, "reason": "not cached", "cache_key": res["key"],
                "source": "miss"}
    anim = renderer.frames(scene, res["qpos"])
    meta = res["meta"]
    return {"ok": True, "source": res["source"], "cache_key": res["key"],
            "anim": anim, "clip": meta,
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


ROUTES = {"/api/plan": api_plan, "/api/generate": api_generate, "/api/cache": api_cache}


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
        <label>beam height <b><span id="hv">1.00</span> m</b></label>
        <input type="range" id="h" min="0.60" max="1.60" step="0.05" value="1.00">
      </div>
      <div class="ctl">
        <label>beam width <b><span id="wv">1.45</span> m</b></label>
        <input type="range" id="w" min="0.30" max="2.25" step="0.05" value="1.45">
      </div>
      <div class="ctl">
        <label>preference</label>
        <div class="prefs" id="prefs"></div>
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
    <div class="hint">Shapes are the robot's own MuJoCo collision primitives — the same
      geometry the collision status refers to, not a decorative mesh.</div>
  </div>

  <div class="col">
    <h2>Planner decision</h2>
    <div class="rows" id="panel"></div>
    <div class="status" id="valid"></div>
    <h2 style="margin-top:20px">All preferences</h2>
    <div class="rows" id="allprefs"></div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s), PREFS=[["shortest","Shortest Path"],["upright","Stay Upright"],["clearance","Maximum Clearance"]];
let pref="shortest", planData=null, anim=null, playing=false, fi=0, timer=null;

PREFS.forEach(([k,l])=>{const b=document.createElement("button");b.textContent=l;b.dataset.k=k;
  b.onclick=()=>{pref=k;syncPrefs();refresh();};$("#prefs").appendChild(b);});
function syncPrefs(){[...$("#prefs").children].forEach(b=>b.setAttribute("aria-pressed",b.dataset.k===pref));}
syncPrefs();

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

// ---- data -------------------------------------------------------------------------
async function refresh(){
  const h=$("#h").value,w=$("#w").value;
  $("#hv").textContent=(+h).toFixed(2); $("#wv").textContent=(+w).toFixed(2);
  const r=await fetch(`/api/plan?height=${h}&width=${w}&preference=${pref}`);
  planData=await r.json();
  drawBEV($("#bev"),planData.bev,420,260,false);
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
  anim=null; $("#scrub").max=0; $("#statebadge").innerHTML=""; paint();
  fetch("/api/cache").then(r=>r.json()).then(c=>{$("#cachestat").textContent=`cache: ${c.n_entries} clips`;});
}
$("#h").oninput=refresh; $("#w").oninput=refresh;

// Deep-linking: ?height=1.0&width=1.45&preference=shortest&auto=1 restores a state and can
// auto-generate, so a specific view is linkable, screenshot-able and recordable.
function applyURL(){
  const q=new URLSearchParams(location.search);
  if(q.has("height")) $("#h").value=q.get("height");
  if(q.has("width"))  $("#w").value=q.get("width");
  if(q.has("preference")&&PREFS.some(([k])=>k===q.get("preference"))) pref=q.get("preference");
  syncPrefs();
  return {auto:q.get("auto")==="1", frame:q.has("frame")?+q.get("frame"):null};
}

$("#gen").onclick=async()=>{
  const b=$("#gen"); b.disabled=true; b.textContent="Generating…";
  try{
    const h=$("#h").value,w=$("#w").value;
    const r=await fetch(`/api/generate?height=${h}&width=${w}&preference=${pref}`);
    const d=await r.json();
    if(!d.ok){$("#valid").innerHTML=`<span class="badge b-warn">no motion</span> ${d.reason}`;}
    else{
      anim=d.anim; fi=0; $("#scrub").max=anim.frames.length-1; $("#scrub").value=0;
      const v=d.validation, c=d.clip;
      $("#valid").innerHTML=
        `<span class="badge ${v.kinematic_collision_free?"b-ok":"b-warn"}">`+
        `${v.kinematic_collision_free?"kinematic collision-free":"KINEMATIC COLLISION"}</span>
         <span class="badge b-mute">not physics validated</span><br>
         min clearance ${fmt(v.min_clearance_m," m")} · goal error ${fmt(v.goal_error_m," m")}<br>
         source <b>${d.source}</b> · ${c.n_frames} frames @ ${c.fps} fps · ${c.steps} steps
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
