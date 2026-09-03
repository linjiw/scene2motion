"""Fig. 6 — the reference screen across corpora, from committed analysis outputs only.

Inputs: ``outputs/analysis_trackability_contract/receipt.json`` (step family: the prompt-elicited
pool and the position-channel ladder) and ``outputs/analysis_duck_contract/receipt.json``
(EXP-026, the duck family).  No model, MuJoCo or GPU dependency: run with any interpreter that
has numpy + matplotlib, e.g.
``/home/linjiw/isaaclab-install/env_isaaclab/bin/python experiments/fig_screen_across_families.py``.

Panels: (a) how well the longest bilateral no-support run ranks the controller's cutoffs in each
corpus, with bootstrap intervals — three corpora spanning two behaviour families, two actuation
channels and two generation pipelines.  (b) inside the duck family, the same ranking for each
preregistered feature group across every stratum, which is where the contact feature is the only
one that stays above chance.  (c) the screen used as a filter at its calibrated 0.20 s threshold:
a usable filter for stepping, not for ducking.  Every number drawn is written to
``fig6_numbers.json``.
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
CONTRACT = REPO / "outputs/analysis_trackability_contract"
DUCK = REPO / "outputs/analysis_duck_contract"
OUT = REPO / "docs/figures"

# Validated categorical slots (dataviz reference instance): blue / orange / aqua.
C_SPEED, C_CROUCH, C_CONTACT = "#2a78d6", "#eb6834", "#1baf7a"
C_INK, C_INK2, C_GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SCREEN_S = 0.20

# The three corpora the screen has been measured on, in evidence order.
CORPORA = (
    ("exp021_step", "stepping · text prompt\n64 refs", CONTRACT),
    ("exp1c_lift", "stepping · position channel\n144 refs", CONTRACT),
    ("duck", "ducking · planner + repair\n526 refs", DUCK),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=7, width=0.6, length=3)
    ax.set_axisbelow(True)


def corpus_rows(contract: dict, duck: dict) -> list[dict]:
    """One record per corpus: AUC with interval, screen operating point, denominators."""
    rows: list[dict] = []
    for key, label, _ in CORPORA:
        if key == "duck":
            summary = duck["summary"]
            group = summary["pooled_auc_by_group"]["contact"]
            screen = summary["screen"]["calibrated_0p20s"]
            rows.append({
                "key": key, "label": label,
                "auc": float(group["auc"]),
                "ci95": [float(v) for v in group["cluster_bootstrap"]["ci95"]],
                "interval_kind": "cluster bootstrap over 36 scenes",
                "n": int(summary["n_clips"]), "terminated": int(summary["n_terminated"]),
                "sensitivity": float(screen["sensitivity"]),
                "specificity": float(screen["specificity"]),
                "flagged_terminated": int(screen["flagged_terminated"]),
                "flagged_survivors": int(screen["flagged_survivors"]),
                "survivors": int(screen["survivors"]),
                "within_scene_auc": float(
                    summary["within_scene_auc_by_group"]["contact"]["weighted_mean_auc"]),
            })
            continue
        summary = contract["summary"][key]
        auc = summary["single_feature_auc"]["max_unsupported_run_s"]
        gate = summary["gate_0p2s_primary"]
        rows.append({
            "key": key, "label": label,
            "auc": float(auc["auc"]), "ci95": [float(v) for v in auc["ci95_bootstrap"]],
            "interval_kind": "clip bootstrap (single scene)",
            "n": int(summary["n"]), "terminated": int(summary["terminated"]),
            "sensitivity": float(gate["sensitivity"][0]),
            "specificity": float(gate["specificity"][0]),
            "flagged_terminated": int(gate["terminated_above"]),
            "flagged_survivors": int(gate["survived_above"]),
            "survivors": int(gate["survived_n"]),
            "within_scene_auc": None,
        })
    return rows


def duck_strata(duck: dict) -> list[dict]:
    """Every evaluable duck stratum with the three group AUCs, dip bins then beam counts."""
    out: list[dict] = []
    strata = duck["summary"]["strata"]
    for name, prefix in (("dip_bins", "dip "), ("route_classes", "")):
        for entry in strata[name]["strata"]:
            if not entry.get("evaluable"):
                continue
            label = (f"{prefix}{entry['stratum']}" if name == "dip_bins"
                     else f"{entry['stratum']} beams")
            out.append({"label": label, "n": int(entry["n"]),
                        "terminated": int(entry["terminated"]), **entry["auc"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(CONTRACT))
    parser.add_argument("--duck", default=str(DUCK))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    contract_dir, duck_dir, out = Path(args.contract), Path(args.duck), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((contract_dir / "receipt.json").read_text())
    duck = json.loads((duck_dir / "receipt.json").read_text())

    rows = corpus_rows(contract, duck)
    strata = duck_strata(duck)
    numbers = {
        "inputs": {
            "analysis_trackability_contract": {
                "path": str((contract_dir / "receipt.json").relative_to(REPO)),
                "sha256": sha256(contract_dir / "receipt.json")},
            "analysis_duck_contract": {
                "path": str((duck_dir / "receipt.json").relative_to(REPO)),
                "sha256": sha256(duck_dir / "receipt.json")},
        },
        "screen_threshold_s": SCREEN_S,
        "corpora": rows,
        "duck_strata": strata,
        "duck_contact_minus_speed": {
            "pooled": duck["summary"]["contact_minus_speed_pooled"],
            "within_scene": duck["summary"]["contact_minus_speed_within_scene"],
        },
    }

    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 8,
                         "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 4.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.85, wspace=0.42)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    # ---- (a) how well the run ranks cutoffs, per corpus ---------------------------------
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        low, high = row["ci95"]
        colour = C_CONTACT if row["key"] == "duck" else C_INK
        ax_a.plot([low, high], [y, y], color=colour, linewidth=1.6, solid_capstyle="round")
        ax_a.scatter([row["auc"]], [y], s=34, color=colour, zorder=4)
        ax_a.text(row["auc"], y + 0.24, f"{row['auc']:.3f}", ha="center", fontsize=7,
                  color=colour)
        ax_a.text(0.362, y, f"{row['terminated']}/{row['n']} cut off", va="center",
                  ha="left", fontsize=6.5, color=C_INK2)
        if row["within_scene_auc"] is not None:
            ax_a.scatter([row["within_scene_auc"]], [y], s=30, marker="D", facecolors="none",
                         edgecolors=colour, linewidths=1.0, zorder=4)
            ax_a.text(row["within_scene_auc"], y - 0.34, "within scene", ha="center",
                      fontsize=6.3, color=colour)
    ax_a.axvline(0.5, color=C_INK2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax_a.text(0.5, len(rows) - 0.35, "chance", fontsize=6.5, ha="center", color=C_INK2)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([r["label"] for r in rows], fontsize=6.8)
    ax_a.set_xlim(0.355, 1.035)
    ax_a.set_ylim(-0.75, len(rows) - 0.15)
    ax_a.set_xlabel("how often a cut-off reference has the longer no-support run (AUC, 95 % bootstrap)")
    ax_a.set_title("(a) One reference feature ranks the controller's cutoffs in three corpora",
                   loc="left")
    style_axis(ax_a)
    ax_a.xaxis.grid(True, color=C_GRID, linewidth=0.5)

    # ---- (b) duck strata, per feature group --------------------------------------------
    ys_b = np.arange(len(strata))[::-1]
    for group, colour, label in (("speed", C_SPEED, "speed (the confound)"),
                                 ("crouch", C_CROUCH, "crouch depth"),
                                 ("contact", C_CONTACT, "contact (the screen)")):
        ax_b.scatter([s[group] for s in strata], ys_b, s=22, color=colour, label=label,
                     zorder=3, linewidths=0)
    ax_b.axvline(0.5, color=C_INK2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax_b.set_yticks(ys_b)
    ax_b.set_yticklabels([f"{s['label']} ({s['n']})" for s in strata], fontsize=6.3)
    ax_b.set_xlim(0.3, 0.99)
    ax_b.set_xlabel("AUC within the stratum")
    ax_b.set_title("(b) Duck family: only the contact feature\nstays above chance everywhere",
                   loc="left")
    ax_b.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.3,
                borderpad=0.2)
    style_axis(ax_b)
    ax_b.xaxis.grid(True, color=C_GRID, linewidth=0.5)

    # ---- (c) the screen as a filter ----------------------------------------------------
    offsets = {"exp021_step": 0.028, "exp1c_lift": -0.055, "duck": -0.055}
    for row in rows:
        colour = C_CONTACT if row["key"] == "duck" else C_INK
        ax_c.scatter([row["specificity"]], [row["sensitivity"]], s=40, color=colour, zorder=4)
        ax_c.text(row["specificity"], row["sensitivity"] + offsets[row["key"]],
                  row["label"].split(" · ")[1].split("\n")[0],
                  fontsize=6.3, color=colour, ha="center")
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0.53, 1.08)
    ax_c.set_xlabel("survivors the screen keeps (specificity)")
    ax_c.set_ylabel("cutoffs the screen catches (sensitivity)")
    ax_c.set_title("(c) As a filter at 0.20 s: usable for\nstepping, not for ducking", loc="left")
    ax_c.text(0.03, 0.555, "useful toward the top right: skips launches\nthat would be cut off, keeps those that survive",
              fontsize=6, color=C_INK2)
    style_axis(ax_c)
    ax_c.grid(True, color=C_GRID, linewidth=0.5)

    fig.subplots_adjust(left=0.205, right=0.985, top=0.9, bottom=0.115)
    fig.savefig(out / "fig6_screen_across_families.pdf")
    fig.savefig(out / "fig6_screen_across_families.png", dpi=220)
    (out / "fig6_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(json.dumps({r["label"].replace("\n", " "): {"auc": r["auc"], "ci95": r["ci95"],
                                                      "sens": r["sensitivity"],
                                                      "spec": r["specificity"]}
                      for r in rows}, indent=2))


if __name__ == "__main__":
    main()
