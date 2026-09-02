"""Fig. 5 — the float and the gate, regenerated from committed analysis outputs only.

Inputs: ``outputs/analysis_trackability_contract/{rows.jsonl,receipt.json}`` (post hoc CPU
analysis; physics seed 0; one scene).  No model, MuJoCo or GPU dependency: run with any
interpreter that has numpy + matplotlib, e.g.
``/home/linjiw/isaaclab-install/env_isaaclab/bin/python experiments/fig_contract_gate.py``.

Panels: (a) longest bilateral no-support run per reference, split by SONIC evaluator outcome,
for the prompt-elicited family (exp021) and the position-channel lift family (exp1c); the
calibrated 0.2 s gate and the post hoc 0.32 s sweep optimum are drawn and labelled as such.
(b) ROC of that single feature on exp021 with a bootstrap band and both operating points.
(c) State at the evaluator cutoff: achieved pelvis height against the reference's higher foot
(every terminated robot upright).  (d) Cutoff time relative to the reference's first >= 0.2 s
no-support onset.  Every number drawn is also written to ``fig5_numbers.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/analysis_trackability_contract"
OUT = REPO / "docs/figures"

# Validated two-slot categorical palette (dataviz reference instance; validator: ALL PASS).
C_TERM = "#eb6834"   # evaluator-terminated
C_SURV = "#2a78d6"   # survived
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e6e5e1"
GATE_S = 0.20        # calibrated gate flags run > 0.20 s (>= 6 frames at 25 fps)
POSTHOC_CUT_S = 0.28  # post hoc sweep optimum flags run > 0.28 s, i.e. >= 0.32 s (8 frames)
POSTHOC_LABEL = "≥ 0.32 s"
RNG = np.random.default_rng(20260901)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def roc(v: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thr = np.unique(np.concatenate([[-np.inf], np.sort(v), [np.inf]]))
    tpr, fpr = [], []
    for t in thr:
        flag = v > t
        tpr.append(flag[y == 1].mean())
        fpr.append(flag[y == 0].mean())
    return np.asarray(fpr), np.asarray(tpr), thr


def roc_band(v: np.ndarray, y: np.ndarray, n_boot: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0, 1, 101)
    curves = []
    idx = np.arange(len(v))
    for _ in range(n_boot):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].sum() in (0, len(b)):
            continue
        f, t, _ = roc(v[b], y[b])
        order = np.argsort(f, kind="stable")
        curves.append(np.interp(grid, f[order], t[order]))
    curves = np.asarray(curves)
    return grid, np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=7, width=0.6, length=3)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (src / "rows.jsonl").read_text().splitlines() if line]
    receipt = json.loads((src / "receipt.json").read_text())
    summary = receipt["summary"]

    e21 = [r for r in rows if r["family"] == "exp021_step"]
    e1c = [r for r in rows if r["family"] == "exp1c_lift"]
    numbers: dict = {"inputs": {"rows_sha256": sha256(src / "rows.jsonl"),
                                "receipt_sha256": sha256(src / "receipt.json")}}

    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 8, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---- (a) no-support runs by outcome ------------------------------------------------
    strips = [("exp021 STEP prompt", e21), ("exp1c position-channel lifts", e1c)]
    ytick, ylab = [], []
    y0 = 0
    numbers["panel_a"] = {}
    for label, fam in strips:
        for outcome, colour, name in ((True, C_TERM, "terminated"), (False, C_SURV, "survived")):
            vals = np.array([r["max_unsupported_run_s"] for r in fam if r["terminated"] is outcome])
            jitter = RNG.uniform(-0.28, 0.28, size=len(vals))
            ring = np.array([bool(r.get("exact_clear_5cm_x1.2")) for r in fam if r["terminated"] is outcome])
            ax_a.scatter(vals, y0 + jitter, s=9, color=colour, alpha=0.75, linewidths=0, zorder=3)
            if ring.any():
                ax_a.scatter(vals[ring], (y0 + jitter)[ring], s=26, facecolors="none",
                             edgecolors=C_INK, linewidths=0.6, zorder=4)
            n_flag = int((vals > GATE_S).sum())
            ytick.append(y0)
            ylab.append(f"{label.split()[0]}{' lift' if 'exp1c' in label else ''}\n{name} {len(vals)}\nflagged {n_flag}")
            numbers["panel_a"][f"{label}|{name}"] = {
                "n": int(len(vals)), "flagged_gt_0p2s": n_flag,
                "flagged_ge_0p32s": int((vals > POSTHOC_CUT_S).sum()),
                "median_s": float(np.median(vals)) if len(vals) else None}
            y0 -= 1
        y0 -= 0.6
    # Lines sit between grid points so the 0.04 s-quantised runs fall cleanly on one side.
    ax_a.axvline(0.22, color=C_INK, linewidth=0.8, zorder=2)
    ax_a.axvline(0.30, color=C_INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax_a.text(0.42, 0.95, "solid: calibrated gate, run > 0.20 s\ndashed: post hoc sweep optimum, run ≥ 0.32 s (8 frames)",
              fontsize=6.3, ha="left", va="top", color=C_INK2)
    ax_a.set_yticks(ytick)
    ax_a.set_yticklabels(ylab, fontsize=6.3)
    ax_a.set_xlim(-0.05, 4.6)
    ax_a.set_ylim(y0 + 0.4, 1.1)
    ax_a.set_xlabel("longest bilateral no-support run in the reference (s)")
    ax_a.set_title("(a) The float, by evaluator outcome", loc="left")
    style_axis(ax_a)
    ax_a.yaxis.grid(False)
    ax_a.scatter([], [], s=9, color=C_TERM, label="evaluator-terminated")
    ax_a.scatter([], [], s=9, color=C_SURV, label="survived")
    ax_a.scatter([], [], s=26, facecolors="none", edgecolors=C_INK, linewidths=0.6,
                 label="exactly clears 5 cm box at x = 1.2 m")
    ax_a.legend(loc="lower right", fontsize=6, frameon=False, handletextpad=0.3,
                bbox_to_anchor=(1.0, 0.16))

    # ---- (b) ROC ------------------------------------------------------------------------
    v = np.array([r["max_unsupported_run_s"] for r in e21]); y = np.array([int(r["terminated"]) for r in e21])
    fpr, tpr, thr = roc(v, y)
    order = np.argsort(fpr, kind="stable")
    grid, lo, hi = roc_band(v, y)
    ax_b.fill_between(grid, lo, hi, color=C_INK2, alpha=0.12, linewidth=0, label="95 % bootstrap band")
    ax_b.plot(fpr[order], tpr[order], color=C_INK, linewidth=1.4, label="exp021 STEP (n = 64)")
    v1 = np.array([r["max_unsupported_run_s"] for r in e1c]); y1 = np.array([int(r["terminated"]) for r in e1c])
    f1, t1, _ = roc(v1, y1); o1 = np.argsort(f1, kind="stable")
    ax_b.plot(f1[o1], t1[o1], color=C_INK2, linewidth=0.9, linestyle=(0, (2, 2)),
              label="exp1c lift arms (n = 144)")
    ax_b.plot([0, 1], [0, 1], color=C_GRID, linewidth=0.6)
    pts = {}
    for cut, marker, name, ty in ((GATE_S, "o", "gate > 0.20 s", 0.80),
                                  (POSTHOC_CUT_S, "s", f"post hoc {POSTHOC_LABEL}", 0.64)):
        flag = v > cut
        pt = (flag[y == 0].mean(), flag[y == 1].mean())
        pts[name] = {"fpr": float(pt[0]), "tpr": float(pt[1]),
                     "flagged_terminated": int(flag[y == 1].sum()), "terminated": int(y.sum()),
                     "flagged_survivors": int(flag[y == 0].sum()), "survivors": int((y == 0).sum())}
        ax_b.scatter([pt[0]], [pt[1]], s=30, marker=marker, color=C_TERM if cut == GATE_S else C_SURV,
                     edgecolors=C_INK, linewidths=0.5, zorder=5)
        ax_b.annotate(f"{name}: {pts[name]['flagged_terminated']}/{pts[name]['terminated']} flagged, "
                      f"{pts[name]['survivors'] - pts[name]['flagged_survivors']}/{pts[name]['survivors']} passed",
                      xy=pt, xytext=(0.30, ty), textcoords="axes fraction",
                      fontsize=6.3, color=C_INK,
                      arrowprops=dict(arrowstyle="-", color=C_INK2, linewidth=0.5))
    a21 = summary["exp021_step"]["single_feature_auc"]["max_unsupported_run_s"]
    a1c = summary["exp1c_lift"]["single_feature_auc"]["max_unsupported_run_s"]
    ax_b.text(0.98, 0.30, f"AUC exp021 {a21['auc']:.3f} ({a21['ci95_bootstrap'][0]:.3f}–{a21['ci95_bootstrap'][1]:.2f})\n"
              f"AUC exp1c lift {a1c['auc']:.2f} ({a1c['ci95_bootstrap'][0]:.2f}–{a1c['ci95_bootstrap'][1]:.2f})",
              ha="right", va="bottom", fontsize=6.3, color=C_INK2, transform=ax_b.transAxes)
    ax_b.set_xlim(-0.02, 1.02); ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("survivors flagged (false-alarm rate)")
    ax_b.set_ylabel("terminated flagged (sensitivity)")
    ax_b.set_title("(b) One feature predicts the cutoff", loc="left")
    ax_b.legend(loc="lower right", fontsize=6, frameon=False)
    style_axis(ax_b)
    numbers["panel_b"] = {"auc_exp021": a21, "auc_exp1c_lift": a1c, "operating_points": pts,
                          "auc_recomputed_exp021": auc(v[y == 1], v[y == 0])}

    # ---- (c) state at cutoff ----------------------------------------------------------
    term = [r for r in e21 if r["terminated"]]
    ref_foot = np.array([max(r["termination_snapshot"]["reference_feet_bottom_m"]) for r in term])
    ach_pelvis = np.array([r["termination_snapshot"]["achieved_pelvis_z_m"] for r in term])
    clear = np.array([bool(r.get("exact_clear_5cm_x1.2")) for r in term])
    ax_c.scatter(ref_foot, ach_pelvis, s=12, color=C_TERM, alpha=0.8, linewidths=0, zorder=3)
    ax_c.scatter(ref_foot[clear], ach_pelvis[clear], s=34, facecolors="none", edgecolors=C_INK,
                 linewidths=0.6, zorder=4)
    ax_c.axhline(0.50, color=C_INK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax_c.text(0.02, 0.505, "fall threshold used by EXP-028 (pelvis < 0.50 m): 0/53 below it",
              fontsize=6.3, va="bottom", color=C_INK)
    ax_c.set_xlim(0, 0.85); ax_c.set_ylim(0.40, 1.0)
    ax_c.set_xlabel("reference: higher foot bottom at the cutoff frame (m)")
    ax_c.set_ylabel("achieved pelvis height at cutoff (m)")
    ax_c.set_title("(c) At the cutoff: robot upright, reference airborne", loc="left")
    style_axis(ax_c)
    numbers["panel_c"] = {"n_terminated": int(len(term)),
                          "achieved_pelvis_z_min_median_max": [float(ach_pelvis.min()), float(np.median(ach_pelvis)), float(ach_pelvis.max())],
                          "reference_higher_foot_min_median_max": [float(ref_foot.min()), float(np.median(ref_foot)), float(ref_foot.max())],
                          "n_pelvis_below_0p5": int((ach_pelvis < 0.5).sum())}

    # ---- (d) cutoff timing ------------------------------------------------------------
    dt = np.array([r["termination_minus_first_onset_s"] for r in term if r.get("termination_minus_first_onset_s") is not None])
    bins = np.arange(-0.55, 0.75, 0.1)
    ax_d.hist(dt, bins=bins, color=C_TERM, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
    med = float(np.median(dt))
    ax_d.axvline(med, color=C_INK, linewidth=0.8)
    ax_d.text(med + 0.03, ax_d.get_ylim()[1] * 0.55, f"median +{med:.2f} s", fontsize=6.5, color=C_INK)
    within = int((np.abs(dt) <= 0.2).sum())
    ax_d.text(0.03, 0.95, f"{within}/{len(dt)} within ±0.2 s\n{int((dt < 0).sum())} before onset",
              ha="left", va="top", fontsize=6.5, color=C_INK2, transform=ax_d.transAxes)
    ax_d.set_xlabel("cutoff time − first ≥ 0.2 s no-support onset (s)")
    ax_d.set_ylabel("terminated references")
    ax_d.set_title("(d) Cutoff time relative to float onset", loc="left")
    style_axis(ax_d)
    numbers["panel_d"] = {"n": int(len(dt)), "median_s": med, "within_0p2s": within,
                          "before_onset": int((dt < 0).sum()), "bins_s": [float(b) for b in bins]}

    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    fig.savefig(out / "fig5_contract_gate.pdf")
    fig.savefig(out / "fig5_contract_gate.png", dpi=220)
    (out / "fig5_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: numbers[k] for k in ("panel_b", "panel_c", "panel_d")}, indent=1, sort_keys=True)[:1500])


if __name__ == "__main__":
    main()
