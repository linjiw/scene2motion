# Kimodo-G1 reduced audit — provenance gap and recovery trail (2026-08-31)

## The claim

`docs/paper-draft-v0.md:228-237` (frozen baseline draft; owned by another agent, not edited
here) quotes a reduced capability audit on Kimodo-G1-RP-v1: 12-program battery, 6 paired
seeds + 6-null calibration, **84 clips, 173 s**; naive counting rules 8.0 / 9.0 / 9.0 modes,
calibrated rule **2.0** → **4.5× overstatement**; duck ladder monotone with stability 1.00
(Δtop 0.010 → 0.067 m for 0.10 → 0.40 m requests); tuck at the null q99 with stability
0.33–0.67; position lift over-responding ~1.6×.

## The gap

**No `outputs/` directory in this repository backs those numbers**, and no copy of the run's
receipt exists anywhere on this machine. Searched 2026-08-31: `outputs/` (all 70+ dirs),
`run/`, `scene2motion/demo_outputs/`, `~/kimodo` (checkout exists, no audit artifacts),
`~/ardy`, `~/lucid` (different project's data root), `git log -S Kimodo` (no tracked history),
git stashes/worktrees (none), and `/tmp` session scratchpads.

## Where the run actually happened

The run is real and its provenance was located in a Claude session transcript:

- **Session**: project `-home-linjiw-ardy`, session id
  `f4440d67-ed27-4331-be07-dc169754a80c`; transcript at
  `~/.claude/projects/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c.jsonl`
  (entries timestamped 2026-08-31T04:37–04:41Z).
- **Authoring**: subagent "Draft KimodoRunner adapter"
  (`~/.claude/projects/-home-linjiw-ardy/f4440d67-…/subagents/agent-ac06b53618f8a2379.jsonl`)
  wrote `kimodo_runner.py`, `smoke_kimodo.py`, `kimodo_reduced_audit.py` and `NOTES.md`.
- **Artifacts** were written to the session's temporary scratchpad:
  `/tmp/claude-1000/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c/scratchpad/kimodo/`
  including `audit_out/receipt.json` and `audit_out/per_program.json`.
- **That directory no longer exists.** `/tmp/claude-1000/-home-linjiw-ardy/` was removed by
  session-scratchpad cleanup. Nothing was copied into any repository first.

## What the transcript preserves (verbatim capture of the receipt and console)

Receipt fields read back in-session and captured in the transcript:
`experiment: kimodo_reduced_audit`, `model: kimodo-g1-rp`, `n_clips: 84`,
`wall_clock_s: 172.7`, `fps: 30.0`, `diffusion_steps: 100`,
`prompt: "A person walks forward at a steady pace."`,
route 6.0 s @ 0.9 m/s, T = 180, root_y 0.78, window [0.35, 0.65],
`seeds_paired: [100–105]` (null arm on disjoint seeds 200–205), and
`counts: {">1 mm": 8.0, "round 1 cm": 9.0, "1 cm drop-never-valid": 9.0,
"6-seed paired q99-calibrated stability >= 0.8": 2.0}` → 4.5×. Per-program console lines
(duck 0.10–0.40 stability 1.00, tuck 0.30/0.60 stability 0.33/0.67, lift 0.08/0.20, rot ±30,
two combos) and the null q99 per channel are also captured verbatim. These match the paper
draft digit for digit, so the draft's numbers are *attested* — but not *re-derivable*: no
ledger, no per-seed deltas, no code file survives outside the transcripts.

## Status against the project's evidence bar

REPORT §8.2(e): "No number should appear that a committed script cannot re-derive from a
committed ledger." The Kimodo row currently fails that bar. It is recorded as a provenance
gap in `docs/REPORT.md` §24; anywhere the numbers are quoted they must be labelled
**transcript-sourced** until rerun.

## Rerun checklist (to close the gap; NOT run today — no GPU launches authorized)

1. Recover `kimodo_runner.py` and `kimodo_reduced_audit.py` from
   `agent-ac06b53618f8a2379.jsonl` (full file contents are in its Write tool calls), or
   re-author from the NOTES captured in the main transcript.
2. Commit them under `experiments/` (e.g. `experiments/kimodo_reduced_audit.py`) with the
   vendored-descriptor caveats kept in the receipt.
3. Run with the same battery/seeds; land `outputs/kimodo_reduced_audit/{receipt.json,
   per_program.json}` in-repo. Expected cost per the prior run: ~3 GPU-min (84 clips, 173 s).
4. Diff against the transcript-captured numbers; if they reproduce, drop the
   transcript-sourced label in REPORT §24 and the paper draft (draft edit belongs to its
   owner).
