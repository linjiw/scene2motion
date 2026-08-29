"""Behavioural probes A and B: does the learned planner do the physically right thing?

These are not accuracy metrics. Each asks whether a specific relationship the heuristic
encodes survived distillation, and each is checkable by eye as well as by number.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..demo.strategy_planner import SHORTEST_MODES
from ..planner import plan
from .dataset import NO_BEAM_HEIGHT, build_scene, duck_label
from .predictor import predict_dip
from .route_profile import N_SAMPLES

SPEED = 0.9


def _pair(scene, arch="cnn"):
    p = plan(scene, "adaptive", modes_override=SHORTEST_MODES)
    if not p.feasible:
        return None
    return p, duck_label(p), predict_dip(scene, p.xy, arch=arch)


def _onset(dip: np.ndarray, xy: np.ndarray, thresh: float = 0.03) -> float | None:
    """Arc length at which the dip first exceeds `thresh`."""
    idx = np.where(dip > thresh)[0]
    if not len(idx):
        return None
    seg = np.linalg.norm(np.diff(np.asarray(xy, float), axis=0), axis=1)
    total = float(seg.sum())
    return float(idx[0] / (len(dip) - 1) * total)


def probe_depth_vs_height(arch="cnn") -> list[dict]:
    """A lower beam must produce a deeper duck."""
    rows = []
    for h in (1.20, 1.10, 1.00, 0.90):
        r = _pair(build_scene(h, 4.0, 1.75), arch)
        if r is None:
            continue
        p, y, d = r
        rows.append({"beam_h": h, "label_peak": float(y.max()),
                     "pred_peak": float(d.max())})
    return rows


def probe_no_beam(arch="cnn") -> list[dict]:
    """No beam means no duck. The model must have learned to do nothing."""
    rows = []
    for x in (3.0, 4.0, 5.0):
        r = _pair(build_scene(NO_BEAM_HEIGHT, x, 1.75), arch)
        if r is None:
            continue
        p, y, d = r
        rows.append({"beam_x": x, "label_peak": float(y.max()),
                     "pred_peak": float(d.max())})
    return rows


def probe_anticipation(arch="cnn") -> list[dict]:
    """The duck must START BEFORE the beam, and move with the beam."""
    rows = []
    for x in (3.0, 3.5, 4.0, 4.5, 5.0):
        r = _pair(build_scene(1.00, x, 1.75), arch)
        if r is None:
            continue
        p, y, d = r
        rows.append({"beam_x": x,
                     "label_onset_m": _onset(y, p.xy), "pred_onset_m": _onset(d, p.xy)})
    return rows


def probe_stands_after(arch="cnn") -> list[dict]:
    """After the beam the robot must come back up."""
    rows = []
    for h in (1.00, 1.10):
        r = _pair(build_scene(h, 4.0, 1.75), arch)
        if r is None:
            continue
        p, y, d = r
        tail = slice(int(0.85 * N_SAMPLES), None)
        rows.append({"beam_h": h, "label_tail_dip": float(y[tail].max()),
                     "pred_tail_dip": float(d[tail].max())})
    return rows


def probe_two_beams(arch="cnn") -> list[dict]:
    """Two consecutive beams: two duck events, and the model has never seen this."""
    rows = []
    for gap in (2.0, 2.5, 3.0):
        sc = build_scene(1.05, 3.0, 1.75, second_beam=(1.05, 3.0 + gap), goal_x=9.0)
        r = _pair(sc, arch)
        if r is None:
            continue
        p, y, d = r

        def peaks(v, thresh=0.03):
            on = v > thresh
            return int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0))
        rows.append({"gap_m": gap, "label_events": peaks(y), "pred_events": peaks(d),
                     "label_peak": float(y.max()), "pred_peak": float(d.max())})
    return rows


def probe_latency(arch="cnn", n: int = 40) -> dict:
    """Body-layer latency: the heuristic's mode lattice against a forward pass."""
    scenes = [build_scene(1.00, 4.0, 1.75) for _ in range(n)]
    routes = [plan(s, "adaptive", modes_override=SHORTEST_MODES) for s in scenes]
    ok = [(s, p) for s, p in zip(scenes, routes) if p.feasible]
    t0 = time.perf_counter()
    for s, p in ok:
        duck_label(p)
    heur = (time.perf_counter() - t0) / len(ok)
    predict_dip(ok[0][0], ok[0][1].xy, arch=arch)          # warm
    t0 = time.perf_counter()
    for s, p in ok:
        predict_dip(s, p.xy, arch=arch)
    learn = (time.perf_counter() - t0) / len(ok)
    return {"n": len(ok), "heuristic_ms": round(heur * 1e3, 3),
            "learned_ms": round(learn * 1e3, 3)}


def smoothness(v: np.ndarray) -> float:
    """Mean |second difference| -- lower is smoother, in metres per sample squared."""
    return float(np.abs(np.diff(np.asarray(v, float), n=2)).mean()) if len(v) > 2 else 0.0


def main() -> None:
    out = {"depth_vs_height": probe_depth_vs_height(),
           "no_beam": probe_no_beam(),
           "anticipation": probe_anticipation(),
           "stands_after": probe_stands_after(),
           "two_beams": probe_two_beams(),
           "latency": probe_latency()}

    print("A1  lower beam -> deeper duck")
    for r in out["depth_vs_height"]:
        print(f"    h={r['beam_h']:.2f}  label {r['label_peak']*100:5.1f} cm   "
              f"pred {r['pred_peak']*100:5.1f} cm")
    lp = [r["pred_peak"] for r in out["depth_vs_height"]]
    print(f"    monotone in prediction: {all(a <= b for a, b in zip(lp, lp[1:]))}")

    print("\nA2  no beam -> no duck")
    for r in out["no_beam"]:
        print(f"    x={r['beam_x']:.1f}  label {r['label_peak']*100:5.2f} cm   "
              f"pred {r['pred_peak']*100:5.2f} cm")

    print("\nA3  duck onset tracks the beam (anticipation)")
    for r in out["anticipation"]:
        lo, po = r["label_onset_m"], r["pred_onset_m"]
        print(f"    beam at {r['beam_x']:.1f} m   label onset {lo if lo is None else round(lo,2)} m"
              f"   pred onset {po if po is None else round(po,2)} m")

    print("\nA4  stands up after the beam (dip in the last 15% of the route)")
    for r in out["stands_after"]:
        print(f"    h={r['beam_h']:.2f}  label {r['label_tail_dip']*100:5.2f} cm   "
              f"pred {r['pred_tail_dip']*100:5.2f} cm")

    print("\nB   two consecutive beams (never seen in training)")
    for r in out["two_beams"]:
        print(f"    gap {r['gap_m']:.1f} m   label {r['label_events']} events "
              f"(peak {r['label_peak']*100:.1f} cm)   pred {r['pred_events']} events "
              f"(peak {r['pred_peak']*100:.1f} cm)")

    lat = out["latency"]
    print(f"\nC   body-layer latency over {lat['n']} routes: "
          f"heuristic {lat['heuristic_ms']} ms   learned {lat['learned_ms']} ms")

    Path("outputs/duck_model").mkdir(parents=True, exist_ok=True)
    Path("outputs/duck_model/probes.json").write_text(json.dumps(out, indent=2))
    print("\nwrote outputs/duck_model/probes.json")


if __name__ == "__main__":
    main()
