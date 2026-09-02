# EXP-023b protocol — prompt-switch positive control (DRAFT, not yet preregistered)

**Status:** draft written 2026-09-01 (evening) after the internal review of the plan of record.
Becomes preregistered when committed with `Status: preregistered` and its sha256 bound into the
receipt before the first sample. Seeds **4640–4647** are reserved. Reuses the EXP-023 harness
(`experiments/exp023_prompt_handoff.py`) unchanged except for the prompt table.

## Why this control is required

EXP-023 found that a WALK→STEP prompt switch at frame 52 or 104, delivered through ARDY's
released `autoregressive_step` interface at its default minimum history (one four-frame token),
elicited no step in 0/8 seeds against 6/8 from frame 0. Its own result note says the campaign
"cannot distinguish 'the prompt is only read while the rollout context is being established'
from 'a four-frame history handoff attenuates the prompt'". ARDY's README states that smaller
history crops facilitate faster adaptation to new prompts, and its interactive demo visibly
switches behaviours. Without a demonstration that *some* prompt switches under the identical
handoff, "delayed prompting fails" is not established and harness attenuation is an equally
likely reading. The SQUEEZE prompt (`A person steps sideways through a narrow gap.`) is already
in the embedding cache, so the control needs no encoder.

## Locked design

Identical to EXP-023's `step_52` schedule with STEP replaced by SQUEEZE, plus its `all_walk`
and from-start siblings, on fresh seeds 4640–4647 (8), B = 8 = 2 seeds × 4 arms per call:

| arm | frames 0–51 | 52–207 | prompt onset |
|---|---|---|---:|
| `all_walk` | WALK | WALK | none |
| `squeeze_0` | SQUEEZE | SQUEEZE | 0 |
| `squeeze_52` | WALK | SQUEEZE | 52 |
| `step_52` | WALK | STEP | 52 (replication of EXP-023's arm on new seeds) |

Same route (7.2 m at 0.9045 m/s, 200 frames, `root_xz` only), same sampler (v2, 5 steps,
cfg (2, 2)), same minimum-history continuation contract, byte-identical WALK prefix and
corresponding-window noise across the four arms of a seed (the EXP-023 noise and fork audits are
reused verbatim). 32 samples.

## Endpoints (kinematic; planned denominators)

- **Sidestep detector** (preregistered): a lateral root excursion |Δy| ≥ 0.15 m sustained for
  ≥ 0.5 s within the 96 frames after onset, with heading deviation ≤ 45° (i.e. a sideways
  translation, not a turn), OR a bilateral foot-crossing lateral pattern from
  `foot_kinematics_series` (lateral representative of one foot passing the other's by ≥ 0.10 m
  while the root moves laterally). Both are scored; the root-excursion rule is primary.
- Step detector as in EXP-023 for `step_52` (whole-body-clearable lift ≥ 0.03 m in the
  post-onset window).
- Route fidelity (progress ratio, route MAE) for every arm; the SQUEEZE arms are expected to
  deviate laterally, so the route-MAE gate is reported, not enforced, for them.

## Predictions and decision rules

- Substrate gate: `squeeze_0` shows the sidestep in ≥ 4/8 (if not, the SQUEEZE prompt is not a
  usable positive control on this route and the campaign is reported as inconclusive).
- Specificity gate: `all_walk` shows it in ≤ 1/8.
- **If `squeeze_52` ≥ 4/8 and `step_52` ≤ 1/8:** the handoff transmits prompts; the STEP timing
  result is behaviour-specific and the abstract may say "a later prompt does not elicit the step,
  although the same handoff switches other behaviours".
- **If `squeeze_52` ≤ 1/8 (with `squeeze_0` ≥ 4/8):** under the minimum-history handoff no prompt
  switched; the abstract is downgraded to "under our minimum-history handoff, no delayed prompt
  switched behaviour" with attenuation named as the leading alternative, and the timing claim is
  scoped to "the released interface as we drove it".
- In between: report the counts; no binary verdict at n = 8.

## Gates, budget, statistics

As EXP-023 (clean worktree, empty output dir, v2 sampler, identity revalidation, 32/32
accounting, rows before summary, protocol sha bound). Host-resource gate: launch only with
≥ 12 GB free VRAM and ≥ 18 GB available RAM, recorded in the receipt. ≈ 1 GPU-minute of
generation plus CPU scoring. Wilson 95 % intervals on every per-arm rate; paired counts across
arms; descriptive at n = 8.
