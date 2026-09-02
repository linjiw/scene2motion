# EXP-023b protocol — prompt-switch positive control (DRAFT, not yet preregistered)

**Status:** draft written 2026-09-01 (evening) after the internal review of the plan of record.
Becomes preregistered when committed with `Status: preregistered` and its sha256 bound into the
receipt before the first sample. Seeds **4640–4647** are reserved. Driver:
`experiments/exp023b_prompt_switch_control.py`, which imports EXP-023's generation contract,
causal audits, step detector, fixed-box scorer and pin checks from
`experiments/exp023_prompt_handoff.py` unchanged (that file's sha256 is bound into the receipt)
and adds only the plan below, the sidestep detector, the transmission statistic and the host
gate. `--dry-run` prints the locked row/chunk plan and the host-gate report without writing.

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
corresponding-window noise across the four arms of a seed (the EXP-023 per-window history and
transcript audits and chunk merge are reused verbatim; the fork audit is generalized so that
`squeeze_52` and `step_52` must be byte-identical to `all_walk` through frame 51 in both the
feature transcript and the decoded qpos, all four arms of a seed share corresponding-window
noise, and distinct seeds differ). 32 samples.

## Endpoints (kinematic; planned denominators; a missing value is a failure, never imputed)

- **Sidestep detector** (preregistered composite of three signatures, each reported
  separately; the composite is `any` and is the gate quantity). Every signature is scored in
  the 96 frames after the arm's onset, and for `all_walk` at both control onsets 0 and 52.
  Axis convention, verified on the archived EXP-023 clips: MuJoCo `qpos[:, 0]` is the route's
  forward axis (+X, `route[:, 1]`, 0 → 7.2 m), `qpos[:, 1]` the lateral axis (+Y,
  `route[:, 0]`), and yaw is taken about +Z from the root quaternion `qpos[:, 3:7]`
  (w, x, y, z) with 0 rad facing +X.
  - (i) *heading-deviation sidle*: |root yaw − route travel direction| ≥ 45° for a run of
    ≥ 13 consecutive frames (≥ 0.5 s at 25 fps). Added because the dense `root_xz`
    constraint pins the root path, so a lateral root excursion may be unreachable even when
    the prompt switches the gait; a sidle then shows as the body turning across the route.
  - (ii) *lateral root excursion*: |root Y − route lateral offset at onset| ≥ 0.15 m for
    ≥ 13 consecutive frames (the original primary rule; heading no longer bounds it).
  - (iii) *foot crossing*: in the pelvis-yaw-aligned frame (identical to the route's lateral
    axis whenever the heading tracks the route), the left foot's `lateral_representative_m`
    from `foot_kinematics_series` lies ≥ 0.10 m to the right of the right foot's for ≥ 3
    consecutive frames (nominal G1 stance facing +X: left foot at +Y, separation +0.08 to
    +0.31 m on the archived WALK clips).
  Detector behaviour on the archived EXP-023 clips (post hoc check of specificity, not
  evidence for this campaign): the WALK clips give maximum heading deviation 5–13°, lateral
  excursion ≤ 0.02 m and foot separation ≥ +0.015 m at both control onsets, and the STEP clips
  (heading ≤ 28°, separation ≥ −0.012 m) also produce no signature.
- Step detector as in EXP-023 (`detect_prompt_event`, whole-body-clearable lift ≥ 0.03 m
  with a two-foot traversal in the post-onset window) for every arm at its onset and for
  `all_walk` at both control onsets; fixed-box exact scoring at the predicted centres
  `route[onset + 34]` at the six graded heights, as in EXP-023.
- **Paired handoff-transmission statistic** (descriptive auxiliary endpoint, prompt-agnostic):
  for each seed the RMS joint-angle difference (`qpos[:, 7:]`, frames 52–147) between
  `squeeze_52` and `all_walk` and between `step_52` and `all_walk`, and, as full-prompt
  references over the same frames, between `squeeze_0` and `all_walk` and between `step_52`
  and `squeeze_52`. Reported per seed with medians; the frames 0–51 difference of each delayed
  arm from `all_walk` must be exactly zero (fork audit). No interval or test is claimed.
- Route fidelity (progress ratio, route MAE, floor penetration) for every arm; the SQUEEZE
  arms are expected to deviate, so the route-MAE gate is reported, not enforced, for them.

## Predictions and decision rules

- Substrate gate: `squeeze_0` shows the sidestep in ≥ 4/8 (if not, the SQUEEZE prompt is not a
  usable positive control on this route and the campaign is reported as inconclusive).
- Specificity gate: `all_walk` shows the sidestep composite (any signature) in ≤ 1/8 seeds
  across both control windows; the EXP-023 step-event specificity (`all_walk` step events in
  ≤ 1/8 seeds) is carried over unchanged.
- **If `squeeze_52` ≥ 4/8 and `step_52` ≤ 1/8:** the handoff transmits prompts; the STEP timing
  result is behaviour-specific and the abstract may say "a later prompt does not elicit the step,
  although the same handoff switches other behaviours".
- **If `squeeze_52` ≤ 1/8 (with `squeeze_0` ≥ 4/8):** under the minimum-history handoff no prompt
  switched; the abstract is downgraded to "under our minimum-history handoff, no delayed prompt
  switched behaviour" with attenuation named as the leading alternative, and the timing claim is
  scoped to "the released interface as we drove it".
- **If `step_52` ≥ 2/8:** EXP-023's delayed-arm zero has not replicated on fresh seeds. The
  delayed-prompt sentence is withdrawn from the abstract regardless of `squeeze_52`, the pooled
  16-seed `step_52` rate (EXP-023 + EXP-023b) is reported with its Wilson interval, and the
  campaign's verdict is `step_replication_failed` (the transmission reading is then moot).
- In between: report the counts; no binary verdict at n = 8.
- **Scope statements fixed in advance.** (a) Every delayed arm is scored in the 96 frames after
  its onset (3.84 s), as in EXP-023; a switch that manifests later is scored absent and this is
  a stated limitation. (b) Signature (ii) is expected to be largely unreachable under the dense
  `root_xz` constraint; the substrate gate is a statement about the composite, and the per-
  signature counts are reported so a reader can see which signature carried it. (c) A `squeeze_0`
  clip counts toward the substrate gate on the composite alone; its progress ratio and route MAE
  are reported beside it and a stalled squeeze (progress ratio < 0.5) is still a behaviour
  switch for the question this control asks (whether the prompt was read), but is labelled as
  such and never counted as a traversal. There is no constructibility probe for the SQUEEZE
  prompt on this route; the free `all_walk` arm is the nominal control.

## Gates, budget, statistics

As EXP-023 (clean worktree, empty output dir, v2 sampler, identity revalidation, 32/32
accounting, rows before summary, protocol sha bound). Host-resource gate: the driver calls
`scene2motion.host_gate.require_host_resources(require_no_isaac=False)` (≥ 12 GiB free VRAM,
≥ 18 GiB available RAM) before creating anything in `--out` and before constructing the
runner; a failure exits non-zero, prints the measured report as JSON and leaves `--out` empty;
on pass the report is bound into the receipt under `host_resource_gate` (concurrent Isaac
processes are recorded for the record, not gated, in this ARDY-only campaign: the VRAM and RAM
thresholds are the binding conditions for ARDY generation; the "no concurrent Isaac job"
condition of the plan applies to the SONIC campaigns). ≈ 1 GPU-minute
of generation plus CPU scoring. Wilson 95 % intervals on every per-arm rate; paired counts
across arms; the decision rule above is evaluated mechanically into the receipt
(`decision_rule.verdict`, valid only when every gate passed); descriptive at n = 8.

**Code:** driver `experiments/exp023b_prompt_switch_control.py`; tests
`tests/test_exp023b_prompt_switch_control.py`; reused unchanged and hash-bound:
`experiments/exp023_prompt_handoff.py`, `scene2motion/host_gate.py`,
`experiments/analyze_exp021_exact_addressability.py` (`wilson_interval`).
