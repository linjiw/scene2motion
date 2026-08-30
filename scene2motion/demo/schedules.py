"""Every body planner's duck schedule for one route, in one comparable frame.

All four are returned as dip in METRES against route DISTANCE, resampled to the same grid, so
the plot compares like with like and the line the user sees is the line that generated the
clip. Keeping this in one place is what stops the plot and the generator drifting apart.

    heuristic   the mode lattice, dilated and smoothed by `plan_to_spec`
    learned     Phase 2's imitation CNN
    optimizer   the Phase 3 convex QP -- the teacher, solved live in ~50 ms
    tcn         Phase 3's residual TCN, if a checkpoint exists
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..learn.dataset import duck_label
from ..learn.route_profile import N_SAMPLES, normalise, profile
from ..optim.response import DIP_MAX, DuckResponse
from ..optim.scheduler import MARGIN_M, dt_for, solve
from ..planner import Plan
from ..scenes import Scene
from .scene_builder import beam_footprints

SPEED = 0.9
# Versioned: the checkpoint path carries the margin it was trained for, so a model trained
# against one clearance margin cannot be served by a demo configured for another.
TCN_DIR = Path("outputs/duck_model_v3_m018")
TCN_CKPT = TCN_DIR / "tcn.pt"
RESP_PATH = Path("outputs/duck_response/response.json")

BODY_LAYERS = ("heuristic", "learned", "optimized")
LAYER_LABEL = {"heuristic": "Heuristic Planner", "learned": "Phase-2 Learned",
               "optimized": "Phase-3 Optimized"}


def _arc(xy: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(np.asarray(xy, float), axis=0), axis=1)
    return np.linspace(0.0, float(seg.sum()), n)


def response() -> DuckResponse | None:
    return DuckResponse.load(RESP_PATH) if RESP_PATH.exists() else None


def all_schedules(scene: Scene, p: Plan, speed: float = SPEED,
                  tcn_dir: Path | None = None) -> dict:
    """dip (m) per planner on a shared route-distance grid, plus beam spans."""
    s = _arc(p.xy, N_SAMPLES)
    route_len = float(s[-1])
    prof = profile(scene, p.xy, speed=speed)
    out = {"s_m": np.round(s, 3).tolist(), "route_len_m": round(route_len, 3),
           "speed_mps": speed, "dip_max_m": DIP_MAX, "schedules": {}, "beams": []}

    for f in beam_footprints(scene):
        # Beam extent expressed in route distance: the route is a straight run down the
        # corridor here, so x maps to arc length directly enough for a plot span.
        out["beams"].append({"s_lo": round(float(f["x_lo"]), 3),
                             "s_hi": round(float(f["x_hi"]), 3),
                             "height_m": round(float(f["z_lo"]), 3)})

    out["schedules"]["heuristic"] = np.round(duck_label(p, n=N_SAMPLES, speed=speed), 4).tolist()

    try:
        from ..learn.predictor import predict_dip
        out["schedules"]["learned"] = np.round(predict_dip(scene, p.xy, speed=speed), 4).tolist()
    except Exception:
        pass

    r = response()
    if r is not None:
        dt = dt_for(route_len, N_SAMPLES, speed)
        sol = solve(prof[:, 0], r, dt)
        out["optimizer_feasible"] = bool(sol.feasible)
        out["optimizer_status"] = sol.status
        out["margin_m"] = MARGIN_M
        if sol.feasible:
            out["schedules"]["optimizer"] = np.round(sol.q * DIP_MAX, 4).tolist()
        d = Path(tcn_dir) if tcn_dir is not None else TCN_DIR
        if (d / "tcn.pt").exists():
            try:
                import json as _json
                import torch
                from ..optim.model_v3 import DuckTCN
                mj = _json.loads((d / "tcn.json").read_text())
                # Refuse to plot a schedule from a model trained for a different margin than
                # the optimiser is currently solving with -- that mismatch is exactly what
                # produced the mixed-artifact state this guard exists to prevent.
                if abs(float(mj.get("margin_m", -1)) - MARGIN_M) > 1e-9:
                    raise ValueError(f"checkpoint margin {mj.get('margin_m')} != {MARGIN_M}")
                out["tcn_margin_m"] = mj["margin_m"]
                out["tcn_dataset_hash"] = mj.get("dataset_hash")
                out["tcn_dir"] = str(d)
                m = DuckTCN()
                m.load_state_dict(torch.load(d / "tcn.pt", map_location="cpu"))
                m.eval()
                q_req = r.g_inv(prof[:, 0] - MARGIN_M)
                with torch.no_grad():
                    q = m(torch.from_numpy(normalise(prof)).float()[None],
                          torch.from_numpy(q_req).float()[None])[0].numpy()
                out["schedules"]["optimized"] = np.round(q * DIP_MAX, 4).tolist()
            except Exception:
                pass
    return out


def dip_for_layer(sched: dict, layer: str) -> np.ndarray | None:
    """The schedule a given body layer would actually hand to the renderer."""
    key = {"heuristic": "heuristic", "learned": "learned",
           "optimized": "optimized"}.get(layer, "heuristic")
    v = sched["schedules"].get(key)
    if v is None and key == "optimized":
        v = sched["schedules"].get("optimizer")      # fall back to the teacher
    return None if v is None else np.asarray(v, float)
