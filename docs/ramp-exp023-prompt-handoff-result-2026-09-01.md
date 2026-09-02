# EXP-023 result — a delayed WALK→STEP prompt does not move the step event

`outputs/exp023_prompt_handoff/`, schema `exp023-prompt-handoff-v1`, status **`complete`**,
**32/32 frozen-prior samples** (seeds 4500–4507 × {all_walk, step_0, step_52, step_104}),
four B=8 schedule calls, 16/16 Horizon52 window calls, no seed reused. Protocol:
`docs/ramp-exp023-prompt-handoff-protocol.md` (sha `c9368bb0…`, bound in the receipt).
Generation at Scene2Motion commit `d888225` (clean); the CPU analysis was finished by the
committed resume script at `f7eb604` after the host process was killed mid-analysis (see
"Provenance" below). Kinematic stage only; no SONIC, no cross-prior claim.

## Preregistered gates (both passed; no threshold or seed changed)

| gate | requirement | observed |
|---|---|---|
| substrate | ≥ 4/8 `step_0` rows contain the frozen event in frames 0–95 | **6/8** |
| specificity | ≤ 1/8 `all_walk` seeds contain an event in any matched window | **0/8** (0/24 windows) |

## Primary result: planned-denominator event rates

| arm | prompt onset | events present | planned | latency frames (present only) |
|---|---:|---:|---:|---|
| `step_0` | 0 | **6** | 8 | 18, 33, 39, 50, 59, 64 (median 44.5 = 1.78 s) |
| `step_52` | 52 | **0** | 8 | — |
| `step_104` | 104 | **0** | 8 | — |
| `all_walk` (3 matched windows) | none | 0 | 8 seeds / 24 windows | — |

Each delayed arm shares byte-identical WALK history with its `all_walk` sibling up to the
fork and corresponding-window noise with all three siblings (feature and decoded-qpos
prefixes verified exact, 8/8 seeds). Under the released interactive interface's default
minimum history (one four-frame token), switching the prompt to STEP at frame 52 or 104
produced **no whole-body-clearable step event (≥ 0.03 m) in the 96-frame post-onset window
in any of the eight paired seeds**, while the same seeds prompted with STEP from frame 0
produced one in six of eight. The delayed arms' largest unfiltered whole-body clearable
height anywhere in their windows was 0.030 m once (seed 4503, `step_52`), 0.017 m once, 0.005 m
once and 0.000 m in the remaining 13 rows; the three `all_walk` windows read 0.000 m in all 24.
No present-event slope, per-seed slope, or paired difference exists to report: complete-case
pairs are 0/8 for both delayed arms. Per protocol no binary "timed prompting" verdict is
assigned at n = 8; the planned-denominator contrast 0/8 vs 6/8 is the result.

## Secondary: exact fixed box at the frozen-latency prediction

The obstacle centre for each onset was fixed in advance at the route position of frame
`onset + 34` (1.23, 3.11, 4.99 m), 0.20 m deep, scored over the full 200 frames with both-feet
before-over-after traversal required.

| arm | traversed | clears 3 cm | 5 cm | 8 cm | ≥ 12 cm |
|---|---:|---:|---:|---:|---:|
| `step_0` (x = 1.23 m) | 8/8 | **1/8** | 0/8 | 0/8 | 0/8 |
| `step_52` (x = 3.11 m) | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| `step_104` (x = 4.99 m) | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| `all_walk` at each centre | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |

Every trajectory walked through every box slab, so these zeros are collisions, not
nonarrivals. On fresh seeds the frozen 34-frame anchor catches the `step_0` lift at 3 cm in
one seed and at 5 cm in none: the six `step_0` events sit at frames 18–64 and route positions
0.94–2.4 m, not at one predictable frame. This is a second, prospective negative for
"fixed-latency staging" on top of EXP-022A's exact-centre audit (12/64 at 5 cm on exp021).

## Supporting endpoints

Progress ratio 0.994–1.006 and route MAE 0.012–0.035 m for all 32 rows: every arm walked the
prescribed 7.2 m route; the prompt change did not stall or derail the rollout. Maximum
physical-foot floor penetration was 0.000–0.057 m (above the calibrated 0.02 m local-step
floor gate in 9 rows, including `all_walk` rows), so the archived clips are not all
contact-clean; this affects no primary endpoint, which is whole-body clearance. Wall-clock:
60.4 s for generation, decode and the first five rows; 337 s for the resumed analysis.

## What this settles and what it does not

* Under the locked policy, the step event is tied to the **rollout origin**, not to prompt
  onset: a prompt that first appears at a Horizon52 boundary does not elicit the behaviour
  within the next 96 frames (3.84 s). Together with exp017/018/019/020 (no spatial channel
  moves the event) this closes the native-interface timing claim for the Horizon52
  checkpoint at the released minimum history.
* It does **not** establish what happens under longer history crops, under the Horizon8
  checkpoint, with a repeated or re-issued prompt, or through a latent-state API. Those are
  different interface contracts and would need their own preregistration.
* It cannot separate "the prompt is only read while the rollout context is being
  established" from "a four-frame history handoff attenuates the prompt": both are
  consistent with 0/8, and the campaign was not designed to tell them apart.
* n = 8 paired seeds; descriptive. The 6/8 `step_0` rate agrees with exp021's 49/64
  elicitation; the `step_0` latency spread (18–64 frames) is wider than the provisional
  exp021 median-34 anchor implies.

## Provenance

The production run archived all 32 normalized-feature and qpos trajectories, the latent /
history audit and the causal-pairing audit, then the host process was terminated during the
CPU analysis after 5/32 rows (receipt preserved as `receipt.interrupted-analysis.json`, rows as
`rows.interrupted-analysis.jsonl`, both hashed in the final receipt). The committed
`experiments/exp023_prompt_handoff_resume_analysis.py` (sha `741ef473…`) re-scored the archives:
it required every frozen source file (protocol, harness, calibration, runner, constraints,
robot, scorers) to hash identically to the generation-time manifest, revalidated the G1 model,
ARDY runtime, and prompt-cache identities, recomputed the qpos fork audit, recomputed the
five archived rows byte-identically before scoring the rest, and applied the preregistered
gates unchanged. Zero trajectories were regenerated; zero new ARDY samples; the generator was
never reloaded (the analysis needs none). An earlier resume attempt was itself killed before
writing anything; the final receipt records that.

## Decision

1. Record the native-interface timing claim as **landed negative, scoped**: neither a spatial
   channel nor a delayed prompt (minimum-history handoff) moves the STEP event.
2. Do not run a Horizon8 or long-history variant for the ICRA deadline unless it is
   preregistered as a distinct interface contract; it is not required for the current claim.
3. The paper's mechanism section reports EXP-023 beside exp021 as the timing evidence, with
   the scope statements above verbatim.
