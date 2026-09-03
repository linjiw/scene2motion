"""Fig. 8 — scene-level outcome classes by box height, from one committed output only.

Input: ``outputs/analysis_traversal_outcomes/summary.json`` (the v1 traversal evaluator re-scoring
the 64 archived EXP-022A achieved states against a start, a goal and a corridor-spanning box at
x = 1.2 m).  No model, MuJoCo or GPU dependency; run with any interpreter that has numpy +
matplotlib, e.g.
``/home/linjiw/isaaclab-install/env_isaaclab/bin/python experiments/fig_traversal_outcomes.py``.

Panel (a): stacked outcome classes over **all 64 assigned trials** at each of the six graded box
heights, in the evaluator's precedence order (fell > collided_obstacle > collided_wall > cutoff >
timeout > stalled > completed).  The obstacle-collision class is drawn split by whether the
rollout got past the obstacle position, because that split is what resolves the apparent tension
between "14 pass the corridor and finish beyond" and "0 satisfy the endpoint": all 14 got past by
going *through* the box, and exactly one of them (s4434) reached the goal region — also through
the box.  Panel (b): the full eight-class ledger, with the **timeout class drawn as "not
assessed"** rather than as a zero, because ``analyze_traversal_outcomes.py`` builds its
``TraversalCriteria`` without ``time_limit_s`` and ``None`` disables the class
(``scene2motion/traversal_eval.py``); its zero is the absence of a measurement, not the absence
of timeouts.

Scope, and it is on the figure: the obstacle was **absent from the physics scene**, so
``collided_obstacle`` means the recorded motion intersects the box geometry in replay — the robot
never felt contact and its controller was never perturbed by a box.  One route, one scene,
physics seed 0, one rollout per reference; descriptive.

Every number drawn is also written to ``docs/figures/fig8_numbers.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/analysis_traversal_outcomes/summary.json"
OUT = REPO / "docs/figures"

# Class colours.  C_TERM is the project's evaluator-cutoff orange (fig4/fig5); the obstacle
# collision gets its own hue as a two-step ramp for the past/short-of split.
C_TERM = "#eb6834"    # evaluator cutoff
C_HIT_PAST = "#5c3d99"   # intersected the box, and got past the obstacle position
C_HIT_SHORT = "#b7a5dd"  # intersected the box, short of the obstacle position
C_ZERO = "#eceae5"    # a class that was assessed and observed zero times
C_NA = "#c9c8c4"      # a class that was not assessed at all
C_INK, C_INK2, C_GRID = "#0b0b0b", "#52514e", "#e6e5e1"

#: Precedence order of :data:`scene2motion.traversal_eval.OUTCOMES`, mirrored here so the figure
#: script reads only the committed summary.  Asserted against the summary's own keys.
PRECEDENCE = ("rejected", "fell", "collided_obstacle", "collided_wall",
              "cutoff", "timeout", "stalled", "completed")
#: The one class the analysis did not assess: no ``time_limit_s`` was configured.
NOT_ASSESSED = ("timeout",)

#: Panel (a) stack, bottom to top: (field, colour, in-bar text colour, legend label).  The
#: obstacle-collision class is split by whether the rollout got past the obstacle position.
STACK = (
    ("collided_obstacle_past_position", C_HIT_PAST, "white",
     "intersected the box, past the obstacle position"),
    ("collided_obstacle_short_of_position", C_HIT_SHORT, C_INK,
     "intersected the box, short of the obstacle position"),
    ("cutoff", C_TERM, "white", "evaluator cutoff (none of them past the obstacle)"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_INK2); ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=7, width=0.6, length=3)
    ax.set_axisbelow(True)


def collect(summary: dict) -> dict:
    """Every drawn quantity, recomputed from the summary and checked for self-consistency."""
    heights = sorted(summary["by_height"], key=float)
    n_assigned = None
    per_height: dict[str, dict] = {}
    for key in heights:
        blk = summary["by_height"][key]
        counts = blk["outcomes"]
        if tuple(counts) != PRECEDENCE:
            raise SystemExit(f"outcome keys {tuple(counts)} != precedence {PRECEDENCE}")
        clips = blk["per_clip"]
        n = int(blk["n_assigned_trials"])
        n_assigned = n if n_assigned is None else n_assigned
        if n != n_assigned or len(clips) != n or sum(counts.values()) != n:
            raise SystemExit(f"{key}: trials do not close over all assigned trials")
        if int(blk["n_rejected_before_execution"]) != int(counts["rejected"]):
            raise SystemExit(f"{key}: rejected count disagrees with the rejected class")

        passed = [c for c in clips if c["passed_obstacle"]]
        passed_hit = [c for c in passed if c["collided_obstacle"]]
        goal = [c for c in clips if c["reached_goal"]]
        # The fact the figure annotates: everything that got past the obstacle position went
        # through the box, and the goal-reaching rollout is one of those.
        if len(passed_hit) != len(passed):
            raise SystemExit(f"{key}: {len(passed) - len(passed_hit)} passed without intersecting")
        if not all(c["collided_obstacle"] and c["passed_obstacle"] for c in goal):
            raise SystemExit(f"{key}: a goal-reaching rollout did not pass through the box")
        short = int(counts["collided_obstacle"]) - len(passed_hit)
        if short < 0:
            raise SystemExit(f"{key}: more passed-and-hit rows than obstacle collisions")

        terminated = [c for c in clips if c["tracker_terminated"]]
        preempted = [c for c in terminated if c["outcome"] != "cutoff"]
        if int(counts["cutoff"]) != len(terminated) - len(preempted):
            raise SystemExit(f"{key}: cutoff count does not follow from the precedence order")
        # The legend says no cutoff row got past the obstacle; keep that label from going stale.
        cutoff_past = [c for c in clips if c["outcome"] == "cutoff" and c["passed_obstacle"]]
        if cutoff_past:
            raise SystemExit(f"{key}: {len(cutoff_past)} cutoff rows past the obstacle; relabel")

        per_height[key] = {
            "obstacle_height_m": float(blk["obstacle_height_m"]),
            "n_assigned_trials": n,
            "n_executed": int(blk["n_executed"]),
            "outcomes": {k: int(v) for k, v in counts.items()},
            "collided_obstacle_past_position": len(passed_hit),
            "collided_obstacle_short_of_position": short,
            "n_passed_obstacle_position": len(passed),
            "n_passed_without_cutoff": sum(1 for c in passed if not c["tracker_terminated"]),
            "n_never_reached_obstacle_position": n - len(passed),
            "n_reached_goal_region": len(goal),
            "goal_region_clips": [c["motion_key"] for c in goal],
            "n_cutoff_past_obstacle_position": len(cutoff_past),
            "n_tracker_terminated": len(terminated),
            "n_terminated_preempted_by_collision": len(preempted),
            "collision_rate_over_assigned": float(blk["collision_rate"]),
            "completion_rate_over_assigned": float(blk["completion_rate"]),
        }
    return {"heights": heights, "n_assigned_trials": int(n_assigned), "per_height": per_height}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC)); ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(src.read_text())
    data = collect(summary)
    heights, per_height = data["heights"], data["per_height"]
    n_assigned = data["n_assigned_trials"]
    labels = [f"{round(per_height[h]['obstacle_height_m'] * 100):d}" for h in heights]

    # The annotated facts are constant across heights only because passing the obstacle position
    # is a root-trajectory property; assert that before drawing one line for all six bars.
    n_past = {per_height[h]["collided_obstacle_past_position"] for h in heights}
    n_goal = {per_height[h]["n_reached_goal_region"] for h in heights}
    goal_clips = {tuple(per_height[h]["goal_region_clips"]) for h in heights}
    if len(n_past) != 1 or len(n_goal) != 1 or len(goal_clips) != 1:
        raise SystemExit("past-the-obstacle or goal-region set varies with height; redraw per bar")
    past = n_past.pop(); goal_n = n_goal.pop(); goal_clip = goal_clips.pop()
    n_terms = {per_height[h]["n_tracker_terminated"] for h in heights}
    if len(n_terms) != 1:
        raise SystemExit("the tracker-cutoff count varies with height; restate the footnote")
    n_term = n_terms.pop()
    pre = [per_height[h]["n_terminated_preempted_by_collision"] for h in heights]
    pre_lo, pre_hi = min(pre), max(pre)

    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "text.color": C_INK,
                         "axes.labelcolor": C_INK, "axes.titlesize": 8, "axes.titleweight": "bold",
                         "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.35),
                                 gridspec_kw={"width_ratios": [1.22, 1.0]})

    # ---- (a) stacked classes over all assigned trials -------------------------------------
    xs = list(range(len(heights)))
    bottoms = [0.0] * len(heights)
    for field, colour, textc, _label in STACK:
        vals = [per_height[h]["outcomes"][field] if field in PRECEDENCE else per_height[h][field]
                for h in heights]
        ax.bar(xs, vals, bottom=bottoms, width=0.66, color=colour, linewidth=0.0)
        for x, v, b in zip(xs, vals, bottoms):
            if v > 0:
                ax.text(x, b + v / 2, str(v), ha="center", va="center", fontsize=6.8,
                        color=textc, fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    if any(abs(b - n_assigned) > 1e-9 for b in bottoms):
        raise SystemExit("stacked segments do not sum to all assigned trials")

    # The callout lives in blank headroom above the bars; the left spine is trimmed to the data
    # range so the axis still reads as capped at all assigned trials.
    head = n_assigned * 1.34
    ax.axhline(past, color=C_INK, linewidth=0.7, linestyle=(0, (3, 2)))
    # No leader line: the dashed rule carries its own y tick, and a leader would cross the bars.
    ax.text(-0.55, head * 0.995,
            f"dashed rule — {past} of {n_assigned} got past the obstacle position,\n"
            f"all {past} of them by going through the box, and {goal_n} of those\n"
            f"({goal_clip[0]}) reached the goal region, also through the box",
            fontsize=6.3, color=C_INK, ha="left", va="top")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlim(-0.62, len(heights) - 0.38); ax.set_ylim(0, head)
    ax.set_yticks([0, past, 32, 48, n_assigned])
    ax.set_xlabel("box height (cm), obstacle at x = 1.2 m")
    ax.set_ylabel(f"trials (all {n_assigned} assigned)")
    ax.set_title("(a) Outcome class over all assigned trials", loc="left")
    style_axis(ax)
    ax.spines["left"].set_bounds(0, n_assigned)
    rate = ax.secondary_yaxis("right", functions=(lambda v: v / n_assigned,
                                                  lambda p: p * n_assigned))
    rate.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    rate.set_ylabel("share of assigned trials", fontsize=7, color=C_INK2)
    rate.tick_params(colors=C_INK2, labelsize=6.5, width=0.6, length=3)
    rate.spines["right"].set_color(C_INK2); rate.spines["right"].set_linewidth(0.6)
    rate.spines["right"].set_bounds(0.0, 1.0)
    handles = [Patch(facecolor=c, label=lab) for _f, c, _t, lab in STACK]
    handles.append(Patch(facecolor=C_NA, hatch="////", edgecolor="white",
                         label="timeout — not assessed (no time limit configured)"))
    handles.append(Patch(facecolor=C_ZERO, edgecolor=C_GRID,
                         label="rejected / fell / collided_wall / stalled / completed — 0"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, -0.20), fontsize=6,
              frameon=False, handlelength=1.4, handleheight=0.9, labelspacing=0.35, borderpad=0.0)

    # ---- (b) the full eight-class ledger, timeout drawn as not assessed --------------------
    fill = {"collided_obstacle": C_HIT_PAST, "cutoff": C_TERM}
    for row, name in enumerate(PRECEDENCE):
        y = len(PRECEDENCE) - 1 - row
        for col, h in enumerate(heights):
            cell = dict(xy=(col - 0.46, y - 0.40), width=0.92, height=0.80, linewidth=0.5)
            if name in NOT_ASSESSED:
                bx.add_patch(Rectangle(**cell, facecolor=C_NA, edgecolor="white", hatch="////"))
                bx.text(col, y, "n/a", ha="center", va="center", fontsize=6.4, color=C_INK)
                continue
            v = per_height[h]["outcomes"][name]
            if v:
                bx.add_patch(Rectangle(**cell, facecolor=fill[name], edgecolor="white",
                                       alpha=0.16 + 0.62 * v / n_assigned))
            else:
                bx.add_patch(Rectangle(**cell, facecolor=C_ZERO, edgecolor="white"))
            bx.text(col, y, str(v), ha="center", va="center", fontsize=6.6,
                    color=C_INK if v else C_INK2)
    bx.set_xlim(-0.5, len(heights) - 0.5); bx.set_ylim(-0.6, len(PRECEDENCE) - 0.4)
    bx.set_xticks(xs); bx.set_xticklabels(labels)
    bx.set_yticks(list(range(len(PRECEDENCE))))
    bx.set_yticklabels(list(PRECEDENCE)[::-1], fontsize=6.6)
    bx.set_xlabel("box height (cm)")
    # Shifted over the class labels so the full title fits inside the figure.
    bx.set_title("(b) All eight classes, in precedence order", loc="left").set_x(-0.30)
    for side in ("top", "right", "left", "bottom"):
        bx.spines[side].set_visible(False)
    bx.tick_params(colors=C_INK2, labelsize=7, width=0.0, length=0)
    bx.text(0.0, -0.20, "timeout is not a zero: the analysis set no\n"
                        "time limit, which disables the class, so no\n"
                        "trial could be classed as timed out.  “stalled”\n"
                        "is the residual class, preempted by both\n"
                        "the collision and the cutoff classes.",
            transform=bx.transAxes, fontsize=6, color=C_INK2, ha="left", va="top")

    fig.text(0.008, 0.012,
             "Obstacle ABSENT from the physics scene: “intersected the box” means the recorded "
             "motion intersects the box geometry in replay — the robot never\nfelt contact and "
             "its controller was never perturbed by a box.  One route, one scene, physics seed 0, "
             f"one rollout per reference; descriptive.\n{n_assigned} archived EXP-022A achieved "
             "states; "
             f"the tracker's evaluator cut off {n_term} of them at every height, of which "
             f"{pre_lo}–{pre_hi} are classed as collisions by precedence.",
             fontsize=5.9, color=C_INK2, ha="left", va="bottom")
    fig.tight_layout(w_pad=1.6, rect=(0, 0.075, 1, 1))
    fig.savefig(out / "fig8_traversal_outcomes.pdf")
    fig.savefig(out / "fig8_traversal_outcomes.png", dpi=220)

    numbers = {
        "inputs": {"summary_path": str(src.relative_to(REPO)) if src.is_relative_to(REPO)
                   else str(src), "summary_sha256": sha256(src)},
        "analysis": summary["analysis"],
        "descriptive_only": summary["descriptive_only"],
        "scope": summary["scope"],
        "scene": summary["scene"],
        "n_clips": summary["n_clips"],
        "n_assigned_trials": n_assigned,
        "heights_m": [per_height[h]["obstacle_height_m"] for h in heights],
        "outcome_classes_precedence": list(PRECEDENCE),
        "classes_not_assessed": {
            "timeout": "not assessed: analyze_traversal_outcomes.py builds TraversalCriteria "
                       "without time_limit_s, and None disables the timeout class in "
                       "scene2motion/traversal_eval.py; the zero is the absence of a "
                       "measurement, not the absence of timeouts"},
        "panel_a_stack_fields_bottom_to_top": [f for f, _c, _t, _l in STACK],
        "annotation": {
            "n_passed_obstacle_position": past,
            "n_passed_and_intersected_box": past,
            "n_passed_without_intersecting_box": 0,
            "n_reached_goal_region": goal_n,
            "goal_region_clips": list(goal_clip),
            "constant_across_all_heights": True,
        },
        "by_height": per_height,
    }
    (out / "fig8_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(json.dumps({h: {k: per_height[h][k] for k in
                          ("outcomes", "collided_obstacle_past_position",
                           "collided_obstacle_short_of_position", "n_reached_goal_region")}
                      for h in heights}, indent=1, sort_keys=True))
    print("ok")


if __name__ == "__main__":
    main()
