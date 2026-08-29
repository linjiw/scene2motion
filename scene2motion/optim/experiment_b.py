"""Experiment B: multi-beam composition, and the recovery boundary as a phase diagram.

The organising claim is that merge-versus-split is governed by TEMPORAL gap, not distance: two
beams 3 m apart are one crouch at 1.2 m/s and two crouches at 0.6 m/s, because what matters is
whether there is time to stand up and come back down against the body's 0.19 s lag and the
jerk cost of doing so. Distance alone cannot express that, so the diagram is drawn over
gap-time = gap / speed.

CPU only -- the optimiser and the TCN both run without ARDY, and the kinematic verification of
selected cells is Experiment A's job.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..demo.scene_builder import BeamParams, build
from ..demo.strategy_planner import SHORTEST_MODES
from ..learn.route_profile import N_SAMPLES, normalise, profile
from ..optim.model_v3 import DuckTCN
from ..planner import plan
from .response import DIP_MAX, DuckResponse
from .scheduler import MARGIN_M, dt_for, solve

GAPS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
SPEEDS = (0.6, 0.9, 1.2)
BEAM_H, WIDTH = 1.05, 1.75
SPLIT_THRESH = 0.05          # command must fall below this between beams to count as a split


def events(v: np.ndarray, t: float = 0.05) -> int:
    on = np.asarray(v) > t
    return int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/duck_model_v3_m018")
    args = ap.parse_args()
    resp = DuckResponse.load()
    tcn = None
    ck = Path("outputs/duck_model_v3_m018/tcn.pt")
    if ck.exists():
        tcn = DuckTCN()
        tcn.load_state_dict(torch.load(ck, map_location="cpu"))
        tcn.eval()

    rows, t0 = [], time.time()
    for speed in SPEEDS:
        for gap in GAPS:
            sc = build(BeamParams(BEAM_H, WIDTH, n_beams=2, gap=gap))
            p = plan(sc, "adaptive", modes_override=SHORTEST_MODES)
            if not p.feasible:
                continue
            prof = profile(sc, p.xy, speed=speed)
            L = float(np.linalg.norm(np.diff(np.asarray(p.xy), axis=0), axis=1).sum())
            dt = dt_for(L, N_SAMPLES, speed)
            sol = solve(prof[:, 0], resp, dt)
            if not sol.feasible:
                continue
            s_m = np.linspace(0, L, N_SAMPLES)
            i0 = int(np.argmin(np.abs(s_m - 4.0)))
            i1 = int(np.argmin(np.abs(s_m - (4.0 + gap))))
            mid = slice(min(i0, i1) + 1, max(i0, i1))
            row = {"gap_m": gap, "speed": speed, "gap_time_s": gap / speed,
                   "opt_events": events(sol.q), "opt_peak": float(sol.q.max()),
                   "opt_between_min": float(sol.q[mid].min()) if mid.stop > mid.start else 0.0,
                   "opt_objective": float(sol.objective)}
            row["opt_split"] = bool(row["opt_between_min"] < SPLIT_THRESH)
            if tcn is not None:
                q_req = resp.g_inv(prof[:, 0] - MARGIN_M)
                with torch.no_grad():
                    q = tcn(torch.from_numpy(normalise(prof)).float()[None],
                            torch.from_numpy(q_req).float()[None])[0].numpy()
                row.update({"tcn_events": events(q), "tcn_peak": float(q.max()),
                            "tcn_between_min": float(q[mid].min()) if mid.stop > mid.start else 0.0,
                            "tcn_mae_to_opt": float(np.abs(q - sol.q).mean() * DIP_MAX)})
                row["tcn_split"] = bool(row["tcn_between_min"] < SPLIT_THRESH)
                row["split_agree"] = bool(row["tcn_split"] == row["opt_split"])
            rows.append(row)

    print(f"{len(rows)} cells in {time.time()-t0:.1f}s   (split = command returns below "
          f"{SPLIT_THRESH} between beams)\n")
    print(f"{'speed':>6s} " + " ".join(f"{g:>5.1f}" for g in GAPS) + "   gap in metres")
    for tag, key in (("optimizer", "opt_split"), ("Phase-3 TCN", "tcn_split")):
        if key.split("_")[0] == "tcn" and tcn is None:
            continue
        print(f"\n{tag}: M = merged crouch, S = split (stand between)")
        for sp in SPEEDS:
            cells = []
            for g in GAPS:
                r = next((x for x in rows if x["speed"] == sp and x["gap_m"] == g), None)
                cells.append("  -  " if r is None or key not in r
                             else ("  S  " if r[key] else "  M  "))
            print(f"{sp:6.1f} " + "".join(cells))
    print("\ngap TIME at the merge/split boundary, per speed:")
    for sp in SPEEDS:
        R = sorted([r for r in rows if r["speed"] == sp], key=lambda r: r["gap_time_s"])
        sw = [r for r in R if r["opt_split"]]
        print(f"  {sp:.1f} m/s: splits from {sw[0]['gap_time_s']:.2f} s "
              f"(gap {sw[0]['gap_m']:.1f} m)" if sw else f"  {sp:.1f} m/s: never splits")
    if tcn is not None:
        agree = [r["split_agree"] for r in rows if "split_agree" in r]
        mae = [r["tcn_mae_to_opt"] for r in rows if "tcn_mae_to_opt" in r]
        print(f"\nTCN vs optimiser: merge/split agreement {np.mean(agree):.3f} "
              f"({int(np.sum(agree))}/{len(agree)}), schedule MAE {np.mean(mae)*1000:.1f} mm")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_b.json").write_text(json.dumps(
        {"gaps": list(GAPS), "speeds": list(SPEEDS), "beam_h": BEAM_H,
         "split_thresh": SPLIT_THRESH, "rows": rows,
         "wall_clock_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\nwrote {out}/experiment_b.json")


if __name__ == "__main__":
    main()
