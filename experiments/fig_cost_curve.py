"""Fig. 4 (kinematic half) — exact obstacle-centred hit rate along the route, from committed
outputs only (``outputs/analysis_exact_centre_cost_curve/curve.jsonl`` + receipt).

Left: exact whole-body clearance rate of the 64 archived exp021 STEP clips versus box centre
for 3/5/8 cm boxes (Wilson 95 % bands), with the post hoc / EXP-024-preregistered centre
(1.2 m) and the mid-route control (3.6 m) marked, and EXP-022A's paired guarded retention
(zero) at both.  Right: the kinematic best-of-N curve at 1.2 m for 5 and 8 cm against the
executed retention.  The EXP-028 physical outcome classes are added when that campaign lands.
Run with any interpreter that has numpy + matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/analysis_exact_centre_cost_curve"
OUT = REPO / "docs/figures"
# Sequential single-hue ramp (blue) for the ordered heights: light -> dark = low -> high box.
C_H = {0.03: "#86b6ef", 0.05: "#2a78d6", 0.08: "#104281"}
C_INK, C_INK2, C_GRID, C_TERM = "#0b0b0b", "#52514e", "#e6e5e1", "#eb6834"


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=7, width=0.6, length=3)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC)); ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(); src, out = Path(args.src), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (src / "curve.jsonl").read_text().splitlines() if l]
    receipt = json.loads((src / "receipt.json").read_text())
    heights = sorted({r["height_m"] for r in rows})
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 8, "axes.titleweight": "bold",
                         "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})
    numbers = {"inputs": receipt["evidence_anchors"], "heights_m": heights}
    for h in heights:
        lvl = sorted((r for r in rows if r["height_m"] == h), key=lambda r: r["centre_x_m"])
        x = np.array([r["centre_x_m"] for r in lvl]); p = np.array([r["exact_hit_rate"] for r in lvl])
        lo = np.array([r["wilson95"][0] for r in lvl]); hi = np.array([r["wilson95"][1] for r in lvl])
        ax.fill_between(x, lo, hi, color=C_H[h], alpha=0.12, linewidth=0)
        ax.plot(x, p, color=C_H[h], linewidth=1.4, label=f"{int(round(h * 100))} cm box")
    centres = receipt["at_exp022a_centres"]
    for label, entry in centres.items():
        xc = entry["centre_x_m"]
        ax.axvline(xc, color=C_INK2, linewidth=0.7, linestyle=(0, (3, 2)))
        hits5 = entry["reference_exact_hits"].get("0.05", entry["reference_exact_hits"].get(0.05))
        ret5 = entry["paired_guarded_retention"]["0.05"]["achieved_guarded_clear"]
        ax.scatter([xc], [ret5 / 64], s=34, marker="x", color=C_TERM, linewidths=1.2, zorder=5)
        ax.annotate(f"{'staged' if label == 'staged' else 'control'} x = {xc:.1f} m\nreference 5 cm: {hits5}/64\n"
                    f"SONIC-retained: {ret5}/64", xy=(xc, ret5 / 64), xytext=(xc + 0.15, 0.33 if label == "staged" else 0.30),
                    fontsize=6.3, color=C_INK, arrowprops=dict(arrowstyle="-", color=C_INK2, linewidth=0.5))
        numbers[label] = entry
    ax.set_xlim(0.5, 6.9); ax.set_ylim(0, 0.42)
    ax.set_xlabel("box centre along the route (m)")
    ax.set_ylabel("exact whole-body clearance rate (64 clips)")
    ax.set_title("(a) Where the elicited float clears a fixed box", loc="left")
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax.text(0.99, 0.56, "Wilson 95 % bands; one route, one scene;\ncentres other than 1.2/3.6 m are descriptive",
            ha="right", va="top", fontsize=6, color=C_INK2, transform=ax.transAxes)
    style_axis(ax)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 0.9045, lambda t: t * 0.9045))
    sec.set_xlabel("nominal arrival time at 0.9045 m/s (s)", fontsize=7, color=C_INK2)
    sec.tick_params(colors=C_INK2, labelsize=6.5, width=0.6, length=3)
    sec.spines["top"].set_color(C_INK2); sec.spines["top"].set_linewidth(0.6)

    staged = centres["staged"]; ns = [1, 2, 4, 8, 16, 32]
    for h in (0.05, 0.08):
        row = next(r for r in rows if abs(r["centre_x_m"] - staged["centre_x_m"]) < 1e-9 and r["height_m"] == h)
        curve = [row["independent_plugin_best_of_n"][f"N={n}"] for n in ns]
        bx.plot(ns, curve, marker="o", markersize=3.5, linewidth=1.2, color=C_H[h],
                label=f"kinematic, {int(round(h*100))} cm (p = {row['successes']}/64)")
        numbers[f"best_of_n_{h}"] = {"ns": ns, "curve": curve, "n90": row["independent_plugin_n90"]}
    bx.plot(ns, [0] * len(ns), marker="x", markersize=4, linewidth=1.2, color=C_TERM,
            label="SONIC-retained (EXP-022A), any N")
    bx.set_xscale("log", base=2); bx.set_xticks(ns); bx.set_xticklabels([str(n) for n in ns])
    bx.set_ylim(-0.03, 1.03); bx.set_xlabel("N independent draws at x = 1.2 m")
    bx.set_ylabel("P(at least one clears)")
    bx.set_title("(b) Best-of-N at x = 1.2 m", loc="left")
    bx.legend(loc="center right", fontsize=6, frameon=False)
    style_axis(bx)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out / "fig4_cost_curve.pdf"); fig.savefig(out / "fig4_cost_curve.png", dpi=220)
    (out / "fig4_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
