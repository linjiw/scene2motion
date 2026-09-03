"""Fig. 7 — correcting measured clearance errors vs equal-budget resampling, from committed
outputs only.

Inputs (read-only, both hashed into ``fig7_numbers.json``):
  * ``outputs/analysis_repair_paired_bootstrap/summary.json`` — the descriptive paired
    scene-cluster bootstrap (36 scenes x 8 seeds, 30,000 resamples) of two corrections minus
    best-of-three resampling.
  * ``outputs/phase4e_architecture_v2_s8/experiment.json`` — the benchmark those rows come
    from; used here only for the per-arm generation caps (``method_specs[*].
    max_adapted_generations``) and the mean generation calls actually used
    (``summary[*].mean_ardy_calls``), i.e. the budget-parity annotation.

Panels: (a) reference-geometry rate of each arm over the 288 scene-seed trials; (b) the paired
per-scene difference with its 95 % cluster-bootstrap interval; (c) the discordant trials, split
by which arm alone succeeded.  The heuristic 18 cm margin row — where resampling beats
correction, 41 trials to 18 — is highlighted in all three panels: this comparison is reported
with its loss, not without it.

No model, MuJoCo or GPU dependency: run with any interpreter that has numpy + matplotlib, e.g.
``/home/linjiw/isaaclab-install/env_isaaclab/bin/python experiments/fig_repair_vs_resampling.py``.
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
SRC_PAIRED = REPO / "outputs/analysis_repair_paired_bootstrap/summary.json"
SRC_BENCH = REPO / "outputs/phase4e_architecture_v2_s8/experiment.json"
OUT = REPO / "docs/figures"

# Same validated two-slot categorical palette and ink/grid constants as fig4/fig5/fig6.
C_REPAIR = "#2a78d6"   # two corrections (arm A)
C_RESAMP = "#eb6834"   # best-of-three resampling (arm B)
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e6e5e1"
C_LOSS_BAND = "#fbeade"  # tint behind the row where resampling wins

# Row order, top to bottom, and the short y-label for each (proposer, endpoint) pair.
ROW_LABELS = {
    ("tcn", "collision_free"): "learned (TCN)\ncollision-free",
    ("tcn", "meets_target"): "learned (TCN)\n18 cm margin",
    ("qp", "collision_free"): "QP teacher\ncollision-free",
    ("qp", "meets_target"): "QP teacher\n18 cm margin",
    ("heuristic", "collision_free"): "heuristic\ncollision-free",
    ("heuristic", "meets_target"): "heuristic\n18 cm margin",
}
LOSS_ROW = ("heuristic", "meets_target")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=7, width=0.6, length=3)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", default=str(SRC_PAIRED))
    ap.add_argument("--bench", default=str(SRC_BENCH))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    paired_path, bench_path, out = Path(args.paired), Path(args.bench), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    paired = json.loads(paired_path.read_text())
    bench = json.loads(bench_path.read_text())
    caps = {m["name"]: int(m["max_adapted_generations"]) for m in bench["method_specs"]}
    calls = {k: float(v["mean_ardy_calls"]) for k, v in bench["summary"].items()}
    n_trials = paired["n_scenes"] * paired["n_seeds"]

    rows = sorted(paired["results"], key=lambda r: list(ROW_LABELS).index((r["proposer"], r["endpoint"])))
    ys = np.arange(len(rows))[::-1].astype(float)  # first row at the top

    numbers: dict = {
        "figure": "fig7_repair_vs_resampling",
        "inputs": {
            "paired_bootstrap_summary": {
                "path": str(paired_path.relative_to(REPO)), "sha256": sha256(paired_path)},
            "phase4e_benchmark": {
                "path": str(bench_path.relative_to(REPO)), "sha256": sha256(bench_path)},
        },
        "provenance": {
            "analysis": paired["analysis"],
            "descriptive_only": bool(paired["descriptive_only"]),
            "n_scenes": paired["n_scenes"], "n_seeds": paired["n_seeds"],
            "n_scene_seed_trials": n_trials,
            "n_boot": paired["n_boot"], "rng_seed": paired["rng_seed"],
            "benchmark_sha256_recorded_by_paired_analysis": paired["source"]["sha256"],
            "benchmark_sha256_matches_file": paired["source"]["sha256"] == sha256(bench_path),
        },
        "budget_parity": {
            "compared_arms": {r["arm_a"]: caps[r["arm_a"]] for r in rows}
                             | {r["arm_b"]: caps[r["arm_b"]] for r in rows},
            "all_compared_arms_capped_at_three_generations":
                all(caps[r["arm_a"]] == 3 and caps[r["arm_b"]] == 3 for r in rows),
            "max_adapted_generations_all_arms": caps,
            "mean_generation_calls_used": {r["arm_a"]: calls[r["arm_a"]] for r in rows}
                                          | {r["arm_b"]: calls[r["arm_b"]] for r in rows},
            "note": ("Only the two-correction (+2) and best-of-three (-resample3) arms share a "
                     "cap of three generations; the uncorrected arms are capped at one and the "
                     "one-correction / best-of-two arms at two, so those are not budget-matched "
                     "and are not compared here. Equal caps are not equal computation: the mean "
                     "calls actually used differ."),
        },
        "panel_a": {}, "panel_b": {}, "panel_c": {},
    }

    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 8, "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(7.0, 3.35), sharey=True,
        gridspec_kw={"width_ratios": [1.30, 1.0, 1.05]})

    # Row highlight: the one comparison correction loses, drawn identically in all three panels.
    loss_y = float(ys[[(r["proposer"], r["endpoint"]) for r in rows].index(LOSS_ROW)])
    for ax in (ax_a, ax_b, ax_c):
        ax.axhspan(loss_y - 0.5, loss_y + 0.5, color=C_LOSS_BAND, zorder=0, linewidth=0)

    # ---- (a) rates of the two equal-cap arms ------------------------------------------
    bar_h = 0.34
    for i, (r, y) in enumerate(zip(rows, ys)):
        for rate, off, colour, series in ((r["rate_a"], +0.19, C_REPAIR, "two corrections (+2)"),
                                          (r["rate_b"], -0.19, C_RESAMP, "best of three (resample3)")):
            ax_a.barh(y + off, rate, height=bar_h, color=colour, linewidth=0, zorder=3)
            ax_a.text(rate + 0.02, y + off, f"{rate:.3f}", va="center", ha="left",
                      fontsize=6.2, color=C_INK, zorder=4)
            if i == 0:  # direct series labels on the first row instead of a legend box
                ax_a.text(0.025, y + off, series, va="center", ha="left", fontsize=6.0,
                          color=C_INK, zorder=5)
        numbers["panel_a"][r["label"]] = {
            "arm_a": r["arm_a"], "rate_a": r["rate_a"], "cap_a": caps[r["arm_a"]],
            "mean_generation_calls_a": calls[r["arm_a"]],
            "arm_b": r["arm_b"], "rate_b": r["rate_b"], "cap_b": caps[r["arm_b"]],
            "mean_generation_calls_b": calls[r["arm_b"]],
            "n_scene_seed_trials": n_trials}
    ax_a.set_xlim(0, 1.30)
    ax_a.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([ROW_LABELS[(r["proposer"], r["endpoint"])] for r in rows], fontsize=6.4)
    for tick, r in zip(ax_a.get_yticklabels(), rows):  # emphasise the row correction loses
        if (r["proposer"], r["endpoint"]) == LOSS_ROW:
            tick.set_color(C_INK)
            tick.set_fontweight("bold")
    ax_a.set_ylim(ys.min() - 1.05, ys.max() + 0.62)
    ax_a.set_xlabel(f"reference-geometry rate over {n_trials} scene–seed trials\n"
                    "(both arms capped at three generations)", fontsize=6.8)
    ax_a.set_title("(a) Reference-geometry rates", loc="left", fontsize=7.5)
    style_axis(ax_a)

    # ---- (b) paired per-scene difference with cluster-bootstrap interval ----------------
    for r, y in zip(rows, ys):
        d = r["paired_difference_pp"]
        lo, hi = r["bootstrap_95_pp"]
        colour = C_REPAIR if d > 0 else (C_RESAMP if d < 0 else C_INK2)
        ax_b.plot([lo, hi], [y, y], color=colour, linewidth=1.4, solid_capstyle="butt", zorder=3)
        for xend in (lo, hi):
            ax_b.plot([xend, xend], [y - 0.14, y + 0.14], color=colour, linewidth=1.0, zorder=3)
        ax_b.scatter([d], [y], s=26, color=colour, edgecolors="#fcfcfb", linewidths=0.6, zorder=4)
        ax_b.text(hi + 2.5 if d >= 0 else lo - 2.5, y, f"{d:+.1f}",
                  va="center", ha="left" if d >= 0 else "right", fontsize=6.4, color=C_INK,
                  zorder=4)
        numbers["panel_b"][r["label"]] = {
            "paired_difference_pp": d, "bootstrap_95_pp": [lo, hi],
            "sign": "correction" if d > 0 else ("resampling" if d < 0 else "tie")}
    ax_b.axvline(0, color=C_INK, linewidth=0.8, zorder=2)
    ax_b.set_xlim(-32, 70)
    ax_b.set_xticks([-20, 0, 20, 40, 60])
    ax_b.set_xlabel("paired difference (pp), 95 % CI\ncorrection − best-of-three", fontsize=6.8)
    ax_b.set_title("(b) Paired difference", loc="left", fontsize=7.5)
    style_axis(ax_b)

    # ---- (c) discordant trials ----------------------------------------------------------
    for r, y in zip(rows, ys):
        a_only, b_only = r["discordant_a_only"], r["discordant_b_only"]
        ax_c.barh(y, a_only, height=bar_h * 1.5, color=C_REPAIR, linewidth=0, zorder=3)
        ax_c.barh(y, -b_only, height=bar_h * 1.5, color=C_RESAMP, linewidth=0, zorder=3)
        if a_only or b_only:
            if a_only:
                ax_c.text(a_only + 5, y, f"{a_only}", va="center", ha="left", fontsize=6.4,
                          color=C_INK, zorder=4)
            if b_only:
                ax_c.text(-b_only - 5, y, f"{b_only}", va="center", ha="right", fontsize=6.4,
                          color=C_INK, zorder=4)
        else:
            ax_c.text(5, y, "0 : 0", va="center", ha="left", fontsize=6.4, color=C_INK2, zorder=4)
        numbers["panel_c"][r["label"]] = {
            "discordant_correction_only": a_only, "discordant_resampling_only": b_only,
            "tile": f"{a_only}:{b_only}", "n_scene_seed_trials": n_trials}
    ax_c.axvline(0, color=C_INK, linewidth=0.8, zorder=2)
    ax_c.set_xlim(-150, 150)
    ax_c.set_xticks([-100, -50, 0, 50, 100])
    ax_c.set_xticklabels(["100", "50", "0", "50", "100"])
    ax_c.set_xlabel(f"discordant trials of {n_trials}\n"
                    "← best-of-three    two corrections →", fontsize=6.6)
    ax_c.set_title("(c) Discordant trials", loc="left", fontsize=7.5)
    style_axis(ax_c)
    ax_c.text(-41, loss_y - 0.72, "↑ resampling wins here", va="center", ha="center",
              fontsize=6.3, color=C_INK, zorder=4)

    fig.tight_layout(w_pad=1.0, rect=(0, 0.155, 1, 1))
    for i, line in enumerate((
            "Equal cap is not equal computation: mean generation calls actually used "
            "(correction vs best-of-three) are TCN 2.72/2.99, QP 2.67/2.98, heuristic 1.84/1.81.",
            "The uncorrected arms (cap 1) and the one-correction / best-of-two arms (cap 2) are "
            "not budget-matched and are not compared here.",
            "36-scene cluster bootstrap, 30,000 resamples, descriptive. Reference geometry only: "
            "no tracked or obstacle-present outcome is claimed.")):
        fig.text(0.012, 0.115 - 0.042 * i, line, fontsize=6.0, color=C_INK2, ha="left", va="top")

    fig.savefig(out / "fig7_repair_vs_resampling.pdf")
    fig.savefig(out / "fig7_repair_vs_resampling.png", dpi=220)
    (out / "fig7_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"panel_b": numbers["panel_b"], "panel_c": numbers["panel_c"]},
                     indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
