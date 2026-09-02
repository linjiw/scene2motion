"""Post-hoc, CPU-only: do reference-clip contact features predict SONIC termination?

Status: exploratory analysis over archived, already-tracked clips. It generates nothing,
launches nothing, and is not a preregistered campaign. It exists because EXP-022A
(``outputs/exp022_exact_tracking_bridge``) found zero reference-to-achieved clearance
retention and named "step amplitude / unconstrained root distribution / the tracker" as
un-isolated causes. This script asks the cheaper question first: is there a kinematic
signature in the *reference* that predicts the tracker outcome, and does the calibrated
local-step contact gate (``outputs/exp016_threshold_calibration/receipt.json``:
``max_unsupported_run_s`` = 0.2 s) already encode it?

Families (all noise-stream v2, physics seed 0):
  * exp021 STEP references (64)  x  EXP-022A achieved rollouts   (out-of-sample for the gate)
  * exp1c WALK ctrl / position-lift references (288) x their achieved rollouts.  NOTE: the
    exp1c control arms are the corpus on which the gate itself was calibrated (tracker-
    successful selection); they carry no predictive claim and are reported separately.
  * exp023 rows (32; NOT tracked) get contract features only, as a prediction, plus the
    free-root WALK pelvis baseline (all_walk arm).

Per clip, from the reference qpos (25 fps) through the exact G1 foot primitives
(``stepover_eval.foot_kinematics_series``): support masks from the calibrated thresholds,
longest bilateral no-support run, flight fraction, mean support feet, foot/root extrema and
speeds, tilt and yaw rates, and a ballistic ratio (run duration / ballistic flight time of the
pelvis rise inside that run).  Against the achieved-state archives: termination, alignment of
the termination time with the reference's first no-support onset, and a fall-versus-cutoff
snapshot at the last valid sample.

"Terminated" means SONIC's evaluation configuration ended the episode on a tracking-error
threshold (pelvis height / orientation, ankle-wrist height, ankle position, time-out); it is an
evaluator cutoff, not a measured fall.  The firing term is not logged by SONIC.

Run:  source env.sh && $S2M_PY experiments/analyze_trackability_contract.py \
          --out outputs/analysis_trackability_contract
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scene2motion.robot import G1Body  # noqa: E402
from scene2motion.stepover_eval import foot_kinematics_series  # noqa: E402
from scene2motion.sonic_state_export import load_sonic_state_rollouts  # noqa: E402

FPS = 25.0
SAMPLE_DT = 0.02  # SONIC achieved-state archive: qpos[i] is the state after step i+1
G = 9.81
THRESHOLD_RECEIPT = REPO / "outputs/exp016_threshold_calibration/receipt.json"
THRESHOLD_RECEIPT_SHA256 = "f6dba8be84a9d5d0b76c8114d4b93b1707bc1bb8a6fec1a26a22aa1780a6e9bf"
EXP021 = REPO / "outputs/exp021_elicited_lift_distribution_v2"
EXP022A = REPO / "outputs/exp022_exact_tracking_bridge"
EXP1C = REPO / "outputs/exp1c_stepover"
EXP023 = REPO / "outputs/exp023_prompt_handoff"
FEATS = ["max_unsupported_run_s", "bilateral_flight_frac", "mean_support_feet", "min_foot_bottom_m",
         "max_foot_clearance_m", "max_foot_planar_speed", "max_foot_vert_speed", "root_z_min", "root_z_max",
         "root_z_range", "max_root_planar_speed", "mean_root_planar_speed", "max_root_vert_speed",
         "max_tilt_deg", "max_yaw_rate_dps", "heading_range_deg", "max_joint_speed_rads", "p99_joint_speed_rads"]
SWEEP_S = [0.12, 0.2, 0.24, 0.28, 0.32, 0.4, 0.5]
RNG = np.random.default_rng(20260901)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _up_z(q):
    x, y = q[:, 1], q[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def _heading(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def support_masks(body: G1Body, q: np.ndarray, sup_h: float, sup_v: float, fps: float = FPS):
    k = foot_kinematics_series(body, q, fps)
    L, R = k["left"], k["right"]
    supL = (L["bottom_clearance_m"] <= sup_h) & (L["planar_speed_mps"] <= sup_v)
    supR = (R["bottom_clearance_m"] <= sup_h) & (R["planar_speed_mps"] <= sup_v)
    return k, supL, supR


def features(body: G1Body, q: np.ndarray, sup_h: float, sup_v: float, fps: float = FPS) -> dict:
    q = np.asarray(q, dtype=float)
    k, supL, supR = support_masks(body, q, sup_h, sup_v, fps)
    L, R = k["left"], k["right"]
    flight = (~supL) & (~supR)
    fr = runs_of(flight)
    longest = max(fr, key=lambda r: r[1] - r[0]) if fr else None
    bottom = np.minimum(L["bottom_clearance_m"], R["bottom_clearance_m"])
    top = np.maximum(L["bottom_clearance_m"], R["bottom_clearance_m"])
    z = q[:, 2]
    f = {
        "max_unsupported_run_s": float((longest[1] - longest[0]) / fps) if longest else 0.0,
        "bilateral_flight_frac": float(flight.mean()),
        "mean_support_feet": float((supL.astype(int) + supR.astype(int)).mean()),
        "min_foot_bottom_m": float(bottom.min()),
        "max_foot_clearance_m": float(top.max()),
        "max_foot_planar_speed": float(max(L["planar_speed_mps"].max(), R["planar_speed_mps"].max())),
    }
    # first no-support run of at least the calibrated gate length (5 frames at 25 fps)
    gate_frames = int(np.ceil(0.2 * fps))
    first = next((r for r in fr if r[1] - r[0] >= gate_frames), None)
    f["first_nosupport_onset_s"] = float(first[0] / fps) if first else None
    f["first_nosupport_run_s"] = float((first[1] - first[0]) / fps) if first else None
    # ballistic ratio of the longest run: duration / ballistic flight time for the pelvis rise in it
    if longest:
        a, b = longest
        rise = float(z[a:b].max() - z[a])
        f["longest_run_pelvis_rise_m"] = rise
        t_ball = 2.0 * np.sqrt(2.0 * max(rise, 0.0) / G)
        f["ballistic_ratio"] = float((b - a) / fps / t_ball) if t_ball > 0 else None
    else:
        f["longest_run_pelvis_rise_m"] = 0.0
        f["ballistic_ratio"] = None
    vz = np.abs(np.diff(np.stack([L["bottom_clearance_m"], R["bottom_clearance_m"]], -1), axis=0)) * fps
    f["max_foot_vert_speed"] = float(vz.max())
    xy = q[:, :2]
    f.update(root_z_min=float(z.min()), root_z_max=float(z.max()), root_z_mean=float(z.mean()),
             root_z_range=float(z.max() - z.min()))
    vp = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    f.update(max_root_planar_speed=float(vp.max()), mean_root_planar_speed=float(vp.mean()),
             max_root_vert_speed=float(np.abs(np.diff(z)).max() * fps))
    f["max_tilt_deg"] = float(np.degrees(np.arccos(np.clip(_up_z(q[:, 3:7]).min(), -1, 1))))
    hd = np.unwrap(_heading(q[:, 3:7]))
    f["max_yaw_rate_dps"] = float(np.degrees(np.abs(np.diff(hd)).max() * fps))
    f["heading_range_deg"] = float(np.degrees(hd.max() - hd.min()))
    jv = np.abs(np.diff(q[:, 7:], axis=0)) * fps
    f["max_joint_speed_rads"] = float(jv.max())
    f["p99_joint_speed_rads"] = float(np.quantile(jv, 0.99))
    return f


def snapshot_at_termination(body: G1Body, ref_q: np.ndarray, ro, sup_h: float, sup_v: float) -> dict:
    """Fall-versus-cutoff snapshot at the last valid achieved sample.

    Achieved sample i is the state after control step i+1 at 50 Hz, so achieved sample
    ``valid-1`` corresponds to reference frame ``round(valid/2)`` (25 fps).
    """
    valid = int(ro.valid_length)
    aq = np.asarray(ro.qpos[:valid], float)
    if valid < 2:
        return {"valid_samples": valid}
    last = aq[-1]
    rf = min(int(round(valid / 2.0)), len(ref_q) - 1)
    ref_pair = ref_q[rf:rf + 2] if rf + 1 < len(ref_q) else ref_q[rf - 1:rf + 1]
    ach_pair = aq[-2:]
    kr, _, _ = support_masks(body, np.asarray(ref_pair, float), sup_h, sup_v, FPS)
    ka, _, _ = support_masks(body, ach_pair, sup_h, sup_v, 1.0 / SAMPLE_DT)
    i_r = 0 if rf + 1 < len(ref_q) else 1
    ref_feet = [float(kr[s]["bottom_clearance_m"][i_r]) for s in ("left", "right")]
    ach_feet = [float(ka[s]["bottom_clearance_m"][-1]) for s in ("left", "right")]
    return {
        "valid_samples": valid,
        "termination_time_s": float(valid * SAMPLE_DT),
        "achieved_pelvis_z_m": float(last[2]),
        "achieved_up_z": float(_up_z(last[None, 3:7])[0]),
        "achieved_tilt_deg": float(np.degrees(np.arccos(np.clip(_up_z(last[None, 3:7])[0], -1, 1)))),
        "achieved_max_root_x_m": float(aq[:, 0].max()),
        "reference_frame": rf,
        "reference_pelvis_z_m": float(ref_q[rf, 2]),
        "pelvis_height_error_m": float(abs(last[2] - ref_q[rf, 2])),
        "reference_feet_bottom_m": ref_feet,
        "achieved_feet_bottom_m": ach_feet,
        "max_foot_height_error_m": float(max(abs(a - b) for a, b in zip(ref_feet, ach_feet))),
    }


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def auc_ci(v: np.ndarray, y: np.ndarray, n_boot: int = 2000) -> list[float]:
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yy, vv = y[idx], v[idx]
        if 0 < yy.sum() < n:
            vals.append(auc(vv[yy == 1], vv[yy == 0]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [float("nan")] * 2


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def logistic(X, y, l2=1.0, iters=200):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.diag(np.r_[np.ones(d), 0])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def loo_auc(X, y) -> float:
    n = len(y)
    scores = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logistic((X[tr] - mu) / sd, y[tr])
        scores[i] = np.r_[(X[i] - mu) / sd, 1.0] @ w
    return auc(scores[y == 1], scores[y == 0])


def load_achieved(dir_glob: str) -> dict:
    out = {}
    for p in sorted(glob.glob(dir_glob, recursive=True)):
        for ro in load_sonic_state_rollouts(os.path.dirname(p)):
            out[ro.motion_key] = ro
    return out


def gate_table(rows: list[dict], thr: float) -> dict:
    y = np.array([1 if r["terminated"] else 0 for r in rows])
    run = np.array([r["max_unsupported_run_s"] for r in rows])
    above = run > thr
    tp, fn = int((above & (y == 1)).sum()), int((~above & (y == 1)).sum())
    fp, tn = int((above & (y == 0)).sum()), int((~above & (y == 0)).sum())
    return {"threshold_s": thr, "terminated_above": tp, "terminated_n": int(y.sum()), "survived_above": fp,
            "survived_n": int((y == 0).sum()), "sensitivity": [tp / max(tp + fn, 1), wilson(tp, tp + fn)],
            "specificity": [tn / max(tn + fp, 1), wilson(tn, tn + fp)]}


def analyse(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("terminated") is not None]
    y = np.array([1 if r["terminated"] else 0 for r in rows])
    out = {"n": len(rows), "terminated": int(y.sum()), "terminated_rate_wilson": wilson(int(y.sum()), len(rows)),
           "single_feature_auc": {}, "gate_0p2s_primary": None, "sweep_max_unsupported_run_s": []}
    if 0 < y.sum() < len(y):
        for f in FEATS:
            v = np.array([r[f] for r in rows], float)
            out["single_feature_auc"][f] = {"auc": auc(v[y == 1], v[y == 0]), "ci95_bootstrap": auc_ci(v, y)}
        out["gate_0p2s_primary"] = gate_table(rows, 0.2)
        out["sweep_max_unsupported_run_s"] = [gate_table(rows, t) for t in SWEEP_S]
        X = np.array([[r[f] for f in FEATS] for r in rows], float)
        out["loo_logistic_auc_all_features"] = loo_auc(X, y)
        run = np.array([r["max_unsupported_run_s"] for r in rows])
        out["max_run_among_survivors_s"] = float(run[y == 0].max())
        out["min_run_among_terminated_s"] = float(run[y == 1].min())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs/analysis_trackability_contract"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if sha256_file(THRESHOLD_RECEIPT) != THRESHOLD_RECEIPT_SHA256:
        raise SystemExit("threshold receipt hash mismatch")
    thr = json.load(open(THRESHOLD_RECEIPT))["stepover_thresholds"]
    sup_h, sup_v, gate_run = thr["support_height_m"], thr["support_speed_mps"], thr["max_unsupported_run_s"]
    body = G1Body(None)
    rows: list[dict] = []

    # ---- exp021 x EXP-022A ----
    z = np.load(EXP021 / "qpos.npz")
    r021 = {r["seed"]: r for r in map(json.loads, open(EXP021 / "rows.jsonl"))}
    ref22 = {r["motion_key"]: r for r in map(json.loads, open(EXP022A / "reference_rows.jsonl"))
             if r["obstacle_label"] == "staged"}
    ach22 = load_achieved(str(EXP022A / "launches/*/attempt-*/**/achieved_qpos*.npz"))
    for key in z.files:
        seed = int(key[1:])
        ro, rr, r = ach22.get(key), ref22.get(key, {}), r021.get(seed, {})
        rec = {"family": "exp021_step", "key": key, "seed": seed,
               "lift_height_m": r.get("lift_height_m"), "lift_x_m": r.get("lift_x_m"),
               "exact_clear_5cm_x1.2": rr.get("exact_clears", {}).get("0.05"),
               "exact_clear_8cm_x1.2": rr.get("exact_clears", {}).get("0.08"),
               "terminated": None if ro is None else bool(ro.terminated),
               "valid_length": None if ro is None else int(ro.valid_length),
               "tracker_progress": None if ro is None else float(ro.progress)}
        rec.update(features(body, z[key], sup_h, sup_v))
        if ro is not None:
            rec["termination_snapshot"] = snapshot_at_termination(body, np.asarray(z[key], float), ro, sup_h, sup_v)
            if rec["first_nosupport_onset_s"] is not None:
                rec["termination_minus_first_onset_s"] = float(ro.valid_length * SAMPLE_DT - rec["first_nosupport_onset_s"])
        rows.append(rec)

    # ---- exp1c ----
    # Note: ``Path / "*-l*/"`` drops the trailing slash and would also match the ``*-l*.pkl``
    # motion files, which are pickled and must not be opened as achieved-state archives.
    for arm_dir in sorted(p for p in glob.glob(str(EXP1C) + "/*-l*/") if os.path.isdir(p)):
        label = os.path.basename(arm_dir.rstrip("/"))
        ros = {ro.motion_key: ro for ro in load_sonic_state_rollouts(arm_dir)}
        for qp in sorted(glob.glob(str(EXP1C / f"qpos/{label}__s*.npy"))):
            key = os.path.basename(qp)[:-4]
            ro = ros.get(key)
            if ro is None:
                cands = [k for k in ros if k.endswith(key) or key.endswith(k)]
                ro = ros[cands[0]] if cands else None
            q = np.load(qp)
            rec = {"family": "exp1c_" + label.split("-")[0], "arm": label, "key": key,
                   "seed": int(key.split("__s")[1]), "lift_req_m": float(label.split("-l")[1]) / 100,
                   "terminated": None if ro is None else bool(ro.terminated),
                   "valid_length": None if ro is None else int(ro.valid_length),
                   "tracker_progress": None if ro is None else float(ro.progress),
                   "achieved_max_root_x_m": None if ro is None else float(np.asarray(ro.qpos[:int(ro.valid_length)])[:, 0].max())}
            rec.update(features(body, q, sup_h, sup_v))
            rows.append(rec)

    # ---- exp023 (untracked): contract prediction + free-root WALK pelvis baseline ----
    z23 = np.load(EXP023 / "qpos.npz")
    for key in z23.files:
        seed, arm = key.split("_", 1)
        rec = {"family": "exp023_untracked", "key": key, "seed": int(seed[1:]), "arm": arm, "terminated": None}
        rec.update(features(body, z23[key], sup_h, sup_v))
        rows.append(rec)

    with open(out / "rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    e21 = [r for r in rows if r["family"] == "exp021_step"]
    e1c = [r for r in rows if r["family"].startswith("exp1c")]
    e23 = [r for r in rows if r["family"] == "exp023_untracked"]
    tr21 = [r for r in e21 if r["terminated"] is not None]
    tr1c = [r for r in e1c if r["terminated"] is not None]
    summary = {
        "definitions": {
            "no_support": f"neither foot inside the calibrated support envelope (bottom <= {sup_h:.4f} m and planar speed <= {sup_v:.3f} m/s)",
            "max_unsupported_run_s": "longest bilateral no-support run in the reference",
            "gate": f"calibrated local-step gate max_unsupported_run_s = {gate_run} s (exp016 receipt, frozen before exp021); primary predictor",
            "terminated": "SONIC evaluation configuration ended the episode on a tracking-error threshold (evaluator cutoff, term not logged)",
            "physics_seed": 0, "families_note": "exp1c control arms are the gate's calibration corpus (no predictive claim)"},
        "exp021_step": analyse(e21),
        "exp1c_all": analyse(e1c),
        "exp1c_ctrl_calibration_corpus": analyse([r for r in e1c if r["arm"].startswith("ctrl")]),
        "exp1c_lift": analyse([r for r in e1c if r["arm"].startswith("lift")]),
    }

    # baselines the reviewers asked for: lift-height-only, and the run feature inside the non-lifting stratum
    y21 = np.array([1 if r["terminated"] else 0 for r in tr21])
    lift = np.array([r["lift_height_m"] or 0.0 for r in tr21])
    summary["exp021_lift_height_only"] = {"auc": auc(lift[y21 == 1], lift[y21 == 0]), "ci95_bootstrap": auc_ci(lift, y21)}
    nl = [r for r in tr21 if (r["lift_height_m"] or 0.0) < 0.03]
    ynl = np.array([1 if r["terminated"] else 0 for r in nl])
    vnl = np.array([r["max_unsupported_run_s"] for r in nl])
    summary["exp021_non_lifting_stratum"] = {"n": len(nl), "terminated": int(ynl.sum()),
                                             "run_auc": auc(vnl[ynl == 1], vnl[ynl == 0]), "ci95_bootstrap": auc_ci(vnl, ynl)}

    def fit_apply(train, test):
        Xtr = np.array([[r[f] for f in FEATS] for r in train], float)
        ytr = np.array([1 if r["terminated"] else 0 for r in train])
        Xte = np.array([[r[f] for f in FEATS] for r in test], float)
        yte = np.array([1 if r["terminated"] else 0 for r in test])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = logistic((Xtr - mu) / sd, ytr)
        s = np.hstack([(Xte - mu) / sd, np.ones((len(yte), 1))]) @ w
        return auc(s[yte == 1], s[yte == 0])
    lift1c = [r for r in tr1c if r["arm"].startswith("lift")]
    summary["transfer_auc"] = {"exp1c_all_to_exp021": fit_apply(tr1c, tr21), "exp021_to_exp1c_all": fit_apply(tr21, tr1c),
                               "exp1c_lift_to_exp021": fit_apply(lift1c, tr21), "exp021_to_exp1c_lift": fit_apply(tr21, lift1c)}

    # the calibrated gate against placed / clearing clips, with ballistic ratios
    lifted = [r for r in tr21 if (r["lift_height_m"] or 0) >= 0.03]
    for h in ("5cm", "8cm"):
        clr = [r for r in tr21 if r[f"exact_clear_{h}_x1.2"]]
        summary[f"exact_clear_{h}_at_x1.2"] = {
            "n": len(clr), "wilson95_of_64": wilson(len(clr), 64), "all_terminated": all(r["terminated"] for r in clr),
            "max_unsupported_run_s_sorted": sorted(round(r["max_unsupported_run_s"], 2) for r in clr),
            "n_within_calibrated_gate": int(sum(r["max_unsupported_run_s"] <= gate_run for r in clr)),
            "longest_run_pelvis_rise_m_sorted": sorted(round(r["longest_run_pelvis_rise_m"], 3) for r in clr),
            "ballistic_ratio_sorted": sorted(round(r["ballistic_ratio"], 1) for r in clr if r["ballistic_ratio"] is not None),
            "root_z_max_mean_min_max": [float(np.mean([r["root_z_max"] for r in clr])), float(min(r["root_z_max"] for r in clr)),
                                        float(max(r["root_z_max"] for r in clr))] if clr else None,
            "termination_minus_first_onset_s": sorted(round(r.get("termination_minus_first_onset_s", float("nan")), 2) for r in clr),
            "achieved_max_root_x_at_termination_m": sorted(round(r["termination_snapshot"].get("achieved_max_root_x_m", float("nan")), 2) for r in clr)}
    summary["exp021_lift_ge_3cm"] = {"n": len(lifted), "terminated": int(sum(r["terminated"] for r in lifted)),
                                     "n_within_calibrated_gate": int(sum(r["max_unsupported_run_s"] <= gate_run for r in lifted))}
    summary["exp021_within_calibrated_gate"] = {"n": int(sum(r["max_unsupported_run_s"] <= gate_run for r in tr21)),
                                                "survived": int(sum(1 for r in tr21 if r["max_unsupported_run_s"] <= gate_run and not r["terminated"]))}

    # alignment of termination with the first >= 0.2 s no-support onset (terminated clips)
    term = [r for r in tr21 if r["terminated"]]
    d = np.array([r["termination_minus_first_onset_s"] for r in term if r.get("termination_minus_first_onset_s") is not None])
    summary["termination_vs_first_nosupport_onset"] = {
        "n_terminated": len(term), "n_with_onset": int(len(d)),
        "n_terminated_without_any_gate_length_run": int(sum(1 for r in term if r.get("termination_minus_first_onset_s") is None)),
        "median_s": float(np.median(d)) if len(d) else None, "iqr_s": [float(np.quantile(d, .25)), float(np.quantile(d, .75))] if len(d) else None,
        "within_0p1s": int((np.abs(d) <= 0.1).sum()), "within_0p2s": int((np.abs(d) <= 0.2).sum()), "within_0p3s": int((np.abs(d) <= 0.3).sum()),
        "before_onset": int((d < 0).sum()), "before_onset_by_more_than_0p1s": int((d < -0.1).sum())}
    # old lift-apex alignment kept for comparison
    dl = np.array([r["valid_length"] * SAMPLE_DT - r["lift_x_m"] / 0.9045 for r in lifted if r["terminated"] and r["lift_x_m"] is not None])
    summary["termination_vs_lift_apex_nominal_speed"] = {"n": int(len(dl)), "median_s": float(np.median(dl)) if len(dl) else None,
                                                         "frac_within_pm1s": float((np.abs(dl) <= 1).mean()) if len(dl) else None}

    # fall versus cutoff at the last valid sample
    snaps = [r["termination_snapshot"] for r in term if "termination_snapshot" in r and "achieved_pelvis_z_m" in r["termination_snapshot"]]
    if snaps:
        pz = np.array([s["achieved_pelvis_z_m"] for s in snaps]); up = np.array([s["achieved_up_z"] for s in snaps])
        pe = np.array([s["pelvis_height_error_m"] for s in snaps]); fe = np.array([s["max_foot_height_error_m"] for s in snaps])
        rff = np.array([max(s["reference_feet_bottom_m"]) for s in snaps])
        summary["fall_vs_cutoff_at_last_sample"] = {
            "n": len(snaps), "achieved_pelvis_z_min_median_max": [float(pz.min()), float(np.median(pz)), float(pz.max())],
            "n_pelvis_below_0p5m": int((pz < 0.5).sum()), "n_up_z_below_0p92": int((up < 0.92).sum()),
            "n_pelvis_height_error_gt_0p25": int((pe > 0.25).sum()), "n_pelvis_height_error_gt_0p15": int((pe > 0.15).sum()),
            "n_max_foot_height_error_gt_0p25": int((fe > 0.25).sum()), "n_max_foot_height_error_gt_0p2": int((fe > 0.2).sum()),
            "reference_higher_foot_bottom_at_termination_min_median_max": [float(rff.min()), float(np.median(rff)), float(rff.max())],
            "note": "pelvis/orientation terms use 0.25 m / 1.0 rad under the tracking/eval override; foot terms 0.25 m (height) and 0.2 m (position); the firing term is not logged"}

    summary["exp023_step_0_max_unsupported_run_s"] = {r["key"]: round(r["max_unsupported_run_s"], 2) for r in e23 if r["arm"] == "step_0"}
    walk23 = [r["root_z_max"] for r in e23 if r["arm"] == "all_walk"]
    summary["root_z_max"] = {
        "exp021_exact_clear_5cm_mean_min_max": summary["exact_clear_5cm_at_x1.2"]["root_z_max_mean_min_max"],
        "exp021_survived_mean_min_max": [float(np.mean([r["root_z_max"] for r in tr21 if not r["terminated"]])),
                                         float(min(r["root_z_max"] for r in tr21 if not r["terminated"])), float(max(r["root_z_max"] for r in tr21 if not r["terminated"]))],
        "exp023_all_walk_free_root_median_min_max": [float(np.median(walk23)), float(min(walk23)), float(max(walk23))],
        "exp1c_ctrl_root_pinned_quantiles": np.quantile([r["root_z_max"] for r in e1c if r["arm"].startswith("ctrl")], [.1, .5, .9]).round(3).tolist(),
        "note": "walking comparator = exp023 all_walk (free root, same route); exp1c ctrl arms had root_y pinned at 0.78 m"}
    ctrl = [r for r in tr1c if r["arm"].startswith("ctrl")]
    summary["exp1c_ctrl_forward_progress"] = {"n": len(ctrl), "n_advanced_less_than_1m": int(sum(r["achieved_max_root_x_m"] < 1.0 for r in ctrl)),
                                              "median_max_root_x_m": float(np.median([r["achieved_max_root_x_m"] for r in ctrl]))}

    receipt = {
        "analysis": "trackability_contract", "status": "post_hoc_exploratory",
        "generated_at_unix": time.time(), "wall_clock_s": time.time() - t0,
        "code": git_state(), "script_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "threshold_receipt": {"path": str(THRESHOLD_RECEIPT), "sha256": THRESHOLD_RECEIPT_SHA256,
                                   "support_height_m": sup_h, "support_speed_mps": sup_v, "max_unsupported_run_s": gate_run},
            "exp021_qpos_sha256": sha256_file(EXP021 / "qpos.npz"), "exp021_rows_sha256": sha256_file(EXP021 / "rows.jsonl"),
            "exp022a_reference_rows_sha256": sha256_file(EXP022A / "reference_rows.jsonl"),
            "exp022a_receipt_sha256": sha256_file(EXP022A / "receipt.json"),
            "exp1c_receipt_sha256": sha256_file(EXP1C / "receipt.json"),
            "exp023_qpos_sha256": sha256_file(EXP023 / "qpos.npz")},
        "scope": "kinematic reference features vs evaluator termination on archived v2 clips (physics seed 0, one rollout per clip); "
                 "no new generation; not a preregistered campaign; obstacle absent from Isaac in every source; exp1c control arms are the gate's calibration corpus",
        "n_rows": len(rows), "summary": summary,
    }
    json.dump(receipt, open(out / "receipt.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", out, "in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
