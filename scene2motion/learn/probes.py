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
from ..planner import MODE_BY_NAME, plan
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


# Two-beam geometry. The corridor is long enough (goal at 11 m) and the first beam early
# enough (2.5 m) that the widest gap still fits, and the gaps span the range where the label
# itself changes from one merged duck to two separate ones -- at 0.72 m of anticipation lead
# either side, two ducks merge below about a 3.5 m gap. A ladder that stopped at 3.0 m would
# only ever ask about the merged case and could not show whether the model SEPARATES events.
TWO_BEAM_H = 1.05
TWO_BEAM_X = 2.5
TWO_BEAM_GOAL_X = 11.0
TWO_BEAM_GAPS = (2.0, 3.0, 4.0, 5.0, 6.0)


def probe_two_beams(arch="cnn") -> list[dict]:
    """Two consecutive beams: how many duck events, and the model has never seen this."""
    rows = []
    for gap in TWO_BEAM_GAPS:
        sc = build_scene(TWO_BEAM_H, TWO_BEAM_X, 1.75,
                         second_beam=(TWO_BEAM_H, TWO_BEAM_X + gap),
                         goal_x=TWO_BEAM_GOAL_X)
        r = _pair(sc, arch)
        if r is None:
            continue
        p, y, d = r

        def peaks(v, thresh=0.03):
            on = v > thresh
            return int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0))
        rows.append({"gap_m": gap, "label_events": peaks(y), "pred_events": peaks(d),
                     "label_peak": float(y.max()), "pred_peak": float(d.max()),
                     "event_count_correct": peaks(y) == peaks(d)})
    return rows


# Standing top-of-head, from the calibrated envelope. The minimum dip that clears a beam at
# height h is (STAND_TOP - h): anything beyond that is crouching the robot does not need to do.
STAND_TOP = MODE_BY_NAME["stand"].top


def probe_excess_crouch(arch="cnn") -> list[dict]:
    """How much deeper than necessary does each planner crouch?

    The heuristic can only pick from a lattice of five dips {0, .15, .25, .35, .50}, so between
    steps it must round UP and over-duck. The obvious prediction is that a continuous model
    wins here. This measures it rather than assuming it.
    """
    rows = []
    for h in np.arange(0.95, 1.31, 0.05):
        r = _pair(build_scene(float(h), 4.0, 1.75), arch)
        if r is None:
            continue
        p, y, d = r
        need = max(0.0, STAND_TOP - float(h))
        rows.append({"beam_h": round(float(h), 2), "needed_m": need,
                     "heuristic_peak_m": float(y.max()), "learned_peak_m": float(d.max()),
                     "heuristic_excess_m": float(y.max()) - need,
                     "learned_excess_m": float(d.max()) - need})
    return rows


def probe_latency(arch="cnn", n: int = 40, repeats: int = 7) -> dict:
    """Body-layer latency: the heuristic's mode lattice against a forward pass.

    Median over `repeats` passes, not a single timing. This is a sub-millisecond microbenchmark
    on a machine that is simultaneously training another model, and a single pass moved by 25 %
    between runs -- enough to make a quoted figure unreproducible. The spread is reported so the
    number can be read with the right precision.
    """
    scenes = [build_scene(1.00, 4.0, 1.75) for _ in range(n)]
    routes = [plan(s, "adaptive", modes_override=SHORTEST_MODES) for s in scenes]
    ok = [(s, p) for s, p in zip(scenes, routes) if p.feasible]
    predict_dip(ok[0][0], ok[0][1].xy, arch=arch)          # warm

    def timed(fn) -> float:
        t0 = time.perf_counter()
        for s, p in ok:
            fn(s, p)
        return (time.perf_counter() - t0) / len(ok) * 1e3

    heur = sorted(timed(lambda s, p: duck_label(p)) for _ in range(repeats))
    learn = sorted(timed(lambda s, p: predict_dip(s, p.xy, arch=arch))
                   for _ in range(repeats))
    mid = repeats // 2
    return {"n": len(ok), "repeats": repeats,
            "heuristic_ms": round(heur[mid], 3),
            "learned_ms": round(learn[mid], 3),
            "heuristic_ms_range": [round(heur[0], 3), round(heur[-1], 3)],
            "learned_ms_range": [round(learn[0], 3), round(learn[-1], 3)]}


def smoothness(v: np.ndarray) -> float:
    """Mean |second difference| -- lower is smoother, in metres per sample squared."""
    return float(np.abs(np.diff(np.asarray(v, float), n=2)).mean()) if len(v) > 2 else 0.0


def main() -> None:
    out = {"depth_vs_height": probe_depth_vs_height(),
           "no_beam": probe_no_beam(),
           "anticipation": probe_anticipation(),
           "stands_after": probe_stands_after(),
           "two_beams": probe_two_beams(),
           "excess_crouch": probe_excess_crouch(),
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
              f"(peak {r['pred_peak']*100:.1f} cm)   "
              f"{'ok' if r['event_count_correct'] else 'MISMATCH'}")
    ok = sum(r["event_count_correct"] for r in out["two_beams"])
    print(f"    event count correct on {ok}/{len(out['two_beams'])} gaps")

    ex = out["excess_crouch"]
    hx = float(np.mean([r["heuristic_excess_m"] for r in ex]))
    lx = float(np.mean([r["learned_excess_m"] for r in ex]))
    print("\nC1  excess crouch beyond what the beam requires "
          "(heuristic lattice is {0, .15, .25, .35, .50} m)")
    for r in ex:
        print(f"    h={r['beam_h']:.2f}  needed {r['needed_m']*100:5.1f} cm   "
              f"heuristic {r['heuristic_excess_m']*100:+5.1f} cm   "
              f"learned {r['learned_excess_m']*100:+5.1f} cm")
    print(f"    mean excess: heuristic {hx*100:.1f} cm   learned {lx*100:.1f} cm   "
          f"-> {'learned' if lx < hx else 'heuristic'} is tighter")

    lat = out["latency"]
    print(f"\nC2  body-layer latency, median of {lat['repeats']} passes over {lat['n']} routes")
    print(f"    heuristic {lat['heuristic_ms']:.3f} ms  (range {lat['heuristic_ms_range']})")
    print(f"    learned   {lat['learned_ms']:.3f} ms  (range {lat['learned_ms_range']})")

    Path("outputs/duck_model").mkdir(parents=True, exist_ok=True)
    Path("outputs/duck_model/probes.json").write_text(json.dumps(out, indent=2))
    print("\nwrote outputs/duck_model/probes.json")


if __name__ == "__main__":
    main()
