# EXP-017 protocol: paired absolute-vs-residual step-over packets

**Status: candidate-pool harness reviewed and ready; three prior GPU attempts stopped before
arms, 2026-08-31.** The representation/phase implementation and CPU-tested exp017
orchestration exist. A `D=1`
preflight had no eligible donor; a `D=4` retry found a donor but rejected nominal seed 2800;
and its exact-design replay identified the locked placement-shift gate as binding. Therefore
no absolute-vs-residual outcome or SONIC comparison exists. This document preserves those
refusals and locks the first scientifically fair, explicitly conditional arm test. It does not
claim that either arm works.

## Question and deliberately narrow scope

On a straight prescribed route, does subtracting a phase-aligned neutral gait from a coherent
step donor and transporting that residual onto a held-out nominal gait improve a spatially
placed step-over relative to copying the same adapted donor as an absolute packet?

This experiment tests **representation**, not response repair. It has two final arms:

1. `absolute`: the adapted donor packet from `CoherentPacketPair.absolute`;
2. `residual`: the hierarchy-local adapted-minus-neutral packet from
   `CoherentPacketPair.residual`.

Both use `strength=1`, `duration_scale=1`, and the same geometry-derived integer event shift.
No local optimizer, RepairNet, best-of-N selection, or post-outcome adjustment is allowed.
Version 1 is one unilateral step event only. It cannot support claims about duck,
squeeze/turn, multi-event traversal, route choice, or cross-prior transfer.

## Units, locked placements, and splits

Let:

* `D` be the number of donor seeds;
* `K` be the number of predeclared held-out nominal-candidate seeds;
* `N` be the number of first-eligible candidates selected for paired evaluation, with `K >= N`;
* `P` be the number of route-placement strata.

Donor and nominal-candidate seed sets are disjoint. Exp017 uses fixed, seed-independent obstacle
positions; the descriptive pilot defaults to `x = (2.4, 3.6, 4.8) m`. The old exp016 frame
anchors `(56, 78, 100, 108, 130, 152)` remain an autoregressive-history diagnostic, not the
scene definition for this representation test. For each held-out nominal clip, exp017 locates
complete qpos-derived physical swing cycles, makes a deterministic one-to-one assignment to
the fixed scenes, constructs one `TargetPhaseMatch` per scene, and derives a bounded integer
path-placement shift using only the prescribed route and nominal physical-foot offset. No
final-arm outcome enters this assignment.

All `K` nominal candidates are generated, archived, and classified before the manifest is
frozen. A candidate is eligible only if it admits every `P` assignment. A missing cycle, phase
extrapolation, protocol mismatch, or out-of-bound shift rejects that candidate under a recorded
attrition reason; it never triggers nearest-frame substitution. The deterministic selector
uses the first `N` eligible candidates in predeclared seed order. If fewer than `N` are
eligible, the campaign selects no partial cohort and stops before final-arm generation. It
never silently reduces `N` or `P`, extends `K`, or changes a gate. The later confirmatory design
must add genuinely different scene geometries/topologies rather than treating three positions
of one box as a broad scene distribution.

The pilot is conditional on the selected donor bundle. Donor-bank uncertainty is not folded
into seed uncertainty. A later confirmatory experiment must repeat the result across donor
bundles before making a donor-general claim.

## Exact frozen-prior sample budget

Every **completed candidate-pool** campaign spends exactly

$$
B_{\mathrm{physical}} = 2D + K + 2NP.
$$

Cached clips still count toward this budget. A failed preflight reports launched and returned
samples separately; if a generator call raises, exact physical spend is unknown and the
receipt reports both a returned-sample lower bound and a conservative launched charge.
Kinematic analysis, qpos phase extraction, packet
construction, and SONIC replay do not count as frozen-prior samples, but their wall time is
reported separately. Python `generate(...)` invocation count is reported separately from
samples.

| Stage | Samples | Locked purpose |
|---|---:|---|
| Donor bank | `2D` | For every donor seed/noise stream, generate one adapted clip with the exact step-over prompt and one neutral clip with the exact walk prompt. |
| Held-out nominal pool | `K` | Generate and archive one walk-prompt nominal per predeclared candidate seed. Classify all candidates under common gates, then freeze the first `N` eligible seeds. |
| Final paired arms | `2NP` | For every selected seed and placement, make exactly one absolute call and one residual call. Every manifest-planned arm remains in the denominator, including a failed attempt. |

The adapted source clip is byte-identically shared by the absolute and residual packets; the
residual arm alone uses the paired neutral clip for subtraction. Donor eligibility and the
deterministic selection order are locked from physical qpos evidence only—complete swing,
minimum relative lift, contralateral stance support, penetration bound, and common phase
support—before any held-out final generation. Evaluation outcomes cannot affect donor choice.

## Pairing and provenance invariants

The only intended difference between final calls is the packet **value representation**.
Before launching either call, the harness must assert and archive all of the following.

### Source-packet lock

`extract_packet_pair` must construct one `CoherentPacketPair`. Its two arms share the same:

* adapted clip and physical adapted event;
* adapted fractional phase queries, packet phase knots, source window, and taper;
* source route-heading frame, skeleton ordering, hierarchy, and root index;
* frozen generator ID, checkpoint SHA-256, adapted sampler seed, integer noise-stream
  version, event-selector string, and code revision.

The residual provenance additionally pins the neutral clip SHA-256. Archive the pair hash,
both packet hashes, both source `StepPhaseCycle` receipts, and the source `PhaseMatch` receipt.
Never build the absolute arm through the legacy position scaffold in
`scene2motion/semantic_scaffold.py`; that would change channels as well as representation.

### Held-out target lock

For every `(evaluation seed, placement)` pair, both final arms must share the exact same:

* target nominal clip SHA-256 and qpos-derived `StepPhaseCycle`;
* `TargetPhaseMatch` JSON and SHA-256, swing side, contralateral stance side, measured stance
  fraction, locked support threshold/window, and physical evidence source;
* prescribed `root_xz`, route heading, target FPS, first heading, obstacle geometry, event
  center, and geometry-derived `center_shift_frames`;
* final prompt string, sampler seed, DDIM settings, and per-sample noise stream v2.

Use the exact step-over prompt for both final arms; prompt wording is a locked constant, not an
experimental factor. The matched nominal call uses the exact walk prompt. A prompt factorial
belongs to exp016 and must not be smuggled into this representation comparison.

### Exact conditioning-support lock

Render both packets before generation and require equal `constraint_support_digest` values.
The v1 renderer must give both arms dense root XZ, dense nominal-root-height plus packet
payload, and full-body global rotations at identical frames and joints. It uses no joint
position channel. The ARDY body-heading feature remains free in both arms because it is not
the route tangent; route heading is used for hierarchy-local canonicalization and global
rotation composition. `program_hash` values are expected to differ because target values
differ, while `support_hash` values must be equal. A support mismatch is a harness error and
invalidates the pair before generation.

Exp017 is a single-shot, non-resumable pilot and refuses a non-empty output directory. A
canonical experiment identity hashes repository/checkpoint/calibration/generation settings,
prompts, splits, fixed scenes, route, source evidence, packet pair, and measurement protocol.
Every program and final-output row carries a schema-tagged canonical identity whose payload
includes the experiment identity plus its own scene/seed/arm, clip, packet/program/support,
phase/control/channel, and output hashes. Final rows additionally carry the frozen manifest
SHA-256 as a foreign key; the manifest is deliberately absent from identity payloads to avoid
a hash cycle. The final receipt anchors the complete rows file, output-identity set, qpos
content, and qpos archive by SHA-256. Noise-stream version 2 is asserted before source
generation, not merely logged. A progressive `run_provenance` identity is complete before the
first source call and is copied into success or failure receipts; it binds code/checkpoint,
calibration, prompts/settings, exact budget, splits, fixed scenes, and route even when no
manifest can be created.

## Physical phase and contact receipts

The generic phase arrays are not labels inferred from seed or frame number. Exp017 must use
`scene2motion/ramp/step_phase.py` to derive them from
`foot_kinematics_series` and locked `StepOverThresholds`:

* takeoff is the first unsupported swing-foot frame after a stable support dwell;
* apex is the maximum physical relative foot lift within that unsupported run;
* landing is the first subsequently supported frame followed by a stable support dwell;
* phase is monotone and fixed at `0.25 / 0.50 / 0.75` for takeoff/apex/landing;
* contralateral stance support is measured over the locked event-centred window;
* a canonical measurement-protocol hash binds support height/speed, support duration/window,
  stance threshold, penetration/lift gates, and the phase convention across source and target
  while allowing their FPS-derived sample counts to differ.

The source adapted and neutral cycles must have the same swing side and one locked phase/contact
protocol. An eligible held-out nominal target must have the packet's swing side. A donor-side
failure is an eligibility rejection recorded in `donor_candidates.jsonl`; expected target-side
gate failures reject individual candidates and enter `nominal_rows.jsonl` plus
`nominal_selection.json`. Pool exhaustion or an unexpected evidence/protocol error aborts
before the manifest/final arms and is recorded in the partial-spend `receipt.json`. Truncated
takeoff or landing, bilateral flight, insufficient lift, excessive
floor penetration, weak stance support, ambiguous phase progression, incomplete common phase
support, or target extrapolation is never repaired by a nearest-frame fallback.

Support/contact thresholds are locked from tracker-successful neutral walks before final-arm
outcomes, using the existing threshold-calibration receipt machinery. The current kinematic
stage uses the same receipt for donor, target, and reference-output evaluation; the future
SONIC stage must reuse that exact receipt for achieved-state evaluation.

## Launch sequence

Exp017 refuses a dirty worktree. The first real-model preflight was locked as:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d1_n1_p1 \
  --n_donors 1 --n_nominal_candidates 1 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

It failed closed in source discovery after exactly two donor samples: both adapted and neutral
cycle lists were empty for seed 2600, with no nominal or final-arm samples. This is a
schema/eligibility failure, not a result. The v1 failure ledger is archived at
`outputs/exp017_ramp_preflight_d1_n1_p1/`. It preserved hashes and rejection text but not the
rejected qpos, so it cannot distinguish clearance from speed as the failed support sub-gate.
The harness now archives and hashes every candidate donor qpos before eligibility and emits
diagnostic-only support fractions, clearance, and speed summaries without changing thresholds
or selection.

Before inspecting any additional donor outcome, the second attempt is locked to the next
wider **sequential** bank, not a cherry-picked seed:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d4_n1_p1 \
  --n_donors 4 --n_nominal_candidates 1 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

It used donor seeds 2600–2603, evaluation seed 2800, the unchanged calibration receipt and
gates, and one fixed placement. Three donor pairs were eligible and the deterministic selector
chose seed 2603: a coherent left-swing source with 0.232 m adapted relative lift. The run then
failed target assignment for nominal seed 2800 after nine samples (`8 + 1`), before its
manifest or either final arm. The ledger is archived at
`outputs/exp017_ramp_preflight_d4_n1_p1/`. This establishes source eligibility for the fixed
design, not packet performance.

The second failure exposed the same evidence-order issue on the target side: nominal qpos and
per-cycle assignment checks were not written before the exception. The harness now archives
all returned nominal qpos before any gate, records phase/shift/window rejections per cycle,
and hash-anchors the finalized nominal ledger. Before inspecting any new nominal outcome, the
third attempt is locked as an exact-design replay whose only intended difference is logging:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_preflight_d4_n1_p1_v2 \
  --n_donors 4 --n_nominal_candidates 1 --n_seeds 1 --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

The replay keeps donor seeds 2600–2603, evaluation seed 2800, placement, thresholds, shift
bound, prompts, and sampler fixed. All regenerated samples count again. If assignment repeats
its failure, the run stops after nine samples and the archived checks determine the binding
gate; if it reaches both arms, it spends eleven.

The replay ran from clean commit `dad93dd` and repeated the failure after nine samples. Its
adapted and neutral source clip hashes, paired source-qpos hash, selected donor event, and
packet payload were identical to the preceding retry. Nominal seed 2800 progressed through
the prescribed route (`1.013` final-progress ratio) and contained one complete left-swing
cycle with full contralateral stance. All five alignment phase errors were exactly zero and
the transported render window, frames 98--100, was inside the clip. The fixed obstacle at
`x=3.6 m`, however, required target root frame 99 while the nominal apex was frame 34: a
`+65`-frame center shift, beyond the unchanged `+/-8` bound. The binding rejection is therefore
`center_shift_bound`, not phase or packet-window eligibility. The complete replay ledger is
`outputs/exp017_ramp_preflight_d4_n1_p1_v2/`.

Moving the obstacle to frame 34 or widening the bound after observing `+65` would change the
intervention rather than repair the harness. Seed 2800 remains a refusal. Because neither
final arm has yet been generated, the next pilot may separate nominal addressability from the
within-addressable representation comparison without arm-outcome leakage. This redesign was
adopted **after** the nominal-eligibility preflight and is labeled exploratory, not part of the
original preregistration.

### Locked exploratory nominal-pool design

Let `K` be a predeclared ordered pool of held-out nominal seeds, with `K >= N`. The exploratory
pilot is locked to `D=4`, `K=8`, `N=2`, `P=1`, donor seeds 2600--2603, nominal pool seeds
2800--2807, and the same `x=3.6 m` scene, prompts, sampler, calibration, phase limits, and
`+/-8` placement bound. The harness must generate, archive, and classify **all K** nominal
candidates before final-arm generation. It then selects the first `N` candidates in the
predeclared seed order that pass the common pre-arm progress, physical phase/contact, bounded
placement, and packet-window gates for every `P` scene. It may not rank candidates by lift or
response, replace a selected seed after an arm failure, extend `K`, or relax a gate.

The reviewed implementation is frozen for launch as:

```bash
source env.sh
$S2M_PY experiments/exp017_ramp_residual_stepover.py \
  --out outputs/exp017_ramp_pool_d4_k8_n2_p1 \
  --n_donors 4 --n_nominal_candidates 8 --n_seeds 2 \
  --obstacle_x 3.6 \
  --threshold_calibration_receipt outputs/exp016_threshold_calibration/receipt.json
```

A completed campaign spends exactly

$$
B_{\mathrm{physical}} = 2D + K + 2NP = 20
$$

frozen-prior samples. Source failure stops after eight; pool exhaustion with fewer than `N`
eligible candidates stops after all 16 source-plus-pool samples; a selected-arm failure still
counts in the fixed `NP` denominator for that arm. The ledger must report `E/K` nominal
eligibility with every rejection reason, `N/E` selected coverage when a full cohort exists,
actual attempted/evaluated seed and arm counts, and `NP` planned outcomes per arm.
All `K` screening calls and their latency count in later test-time-budget comparisons.

The primary estimand is conditional: among the first `N` members of this frozen pool satisfying
the common pre-arm gates for all fixed placements, does residual transport improve the endpoint
over same-support absolute transport? Screening is nondifferential between arms, but may select
gaits whose nominal phase or seed-linked random stream also correlates with prompt response.
Consequently the pilot cannot establish unconditional ARDY-seed robustness or end-to-end
planner success. Low `E/K` is itself negative method evidence. Confirmation requires a newly
frozen, disjoint pool, independent geometries, and multiple donor bundles rather than repeated
expansion of this pool.

### Locked pool outcome

The run from clean commit `3e4ba7f` stopped after exactly 16 launched and returned samples:
eight donor-source samples and all eight nominal candidates, with no final-arm call. Only seed
2805 passed (`E/K=1/8`), exactly at the `+8`-frame placement boundary. Four candidates failed
target phase-cycle eligibility and three had valid cycles but no assignment within `+/-8`;
therefore `E<N`, the selector chose no partial cohort, and no manifest or absolute/residual
outcome exists. The complete hash-verified ledger is
`outputs/exp017_ramp_pool_d4_k8_n2_p1/`.

The pool is closed: do not extend `K`, move the obstacle, relax the shift bound, or relabel the
single eligible seed as E1. Retrospective same-qpos mechanism checks indicate that neither a
right-swing source packet nor lowering the target-only lift threshold changes eligibility for
these eight seeds; those checks are post-outcome diagnostics and do not alter the recorded
labels. The result instead motivates a new program component that couples route timing to gait
phase at fixed path progress. Any such redesign requires a new protocol and disjoint seeds.

A completed preflight remains a schema/eligibility check, not a representation result. Inspect
`receipt.json`, `manifest.json`, both packet/program support hashes, physical cycle receipts,
and all row counts before scheduling the larger kinematic pilot. The current harness
intentionally stops at kinematics; an equal-budget `2NP` SONIC achieved-qpos stage must be
added and run before reporting execution success.

## Endpoints

The primary kinematic quantity is the obstacle-centred whole-body
`max_box_height_lower_bound_m` from `BoxHeightProbe`, evaluated against the robot's simulation
collision primitives with the separately measured body margin. It is not swing-foot peak.
For a fixed predeclared obstacle, also report `obstacle_collision_free` and the full
`evaluate_local_step` gate bundle.

The required E1a kinematic endpoint vector for every arm is:

* maximum whole-body box-height lower bound and fixed-obstacle minimum clearance;
* local-step success, root progress ratio, and path error;
* crossing-position error (`phase_error_m`) and crossing-frame error;
* selected lead/swing side and whether it matches the packet side;
* contralateral support fractions, maximum unsupported duration, bilateral-flight fraction,
  landing dwell, and maximum foot-floor penetration;
* integrated rotation/root-height deformation and frozen-prior calls/wall-clock time.

E1b adds the same crossing, contact, progress, clearance, and local-step readouts after equal-
budget SONIC achieved-qpos replay.

Until an obstacle is instantiated in Isaac, “SONIC execution success” means replaying achieved
qpos against the Scene2Motion simulation collision model and the physical support/progress
gates. It is **not** an Isaac obstacle-contact result. Report reference and achieved-state
endpoints separately; never substitute tracking success or foot peak for traversal success.

## Statistical analysis

E1a crosses the same held-out nominal seeds with fixed path positions of one box geometry and
uses one selected donor bundle. Those positions are **placement strata, not independent
scenes**. The pilot therefore reports raw paired residual-minus-absolute effects per placement
after averaging repeated seeds, the descriptive mean over placements, raw paired counts, and
no confidence interval or p-value. It cannot support population inference.

The attempted-seed ledger separately reports source eligibility and pre-manifest target
attrition from partial-spend receipts. These cases are not assigned to an arm that was never
generated. After the manifest is frozen, every planned absolute/residual pair remains in the
arm denominator, including generation/scoring failures; no outcome-dependent complete-case
filtering is allowed.

E1b must introduce independently sampled scene geometries/topologies and repeat donor bundles.
For each true scene, collapse repeated seeds before a cluster bootstrap over scenes, or use a
predeclared crossed/mixed-effects model that represents seed and scene reuse. Seeds are never
pooled as independent observations. Primary interpretation at both stages uses the joint
endpoint vector: better clearance without stance, crossing, and progress is not a successful
kinematic step; execution claims additionally require achieved-state retention.

## Staged decision and kill condition

E1a advances the packet representation to a **local-response-optimizer implementation** only
if the residual arm shows a descriptive whole-body box-clearance advantage over the exact-
support absolute arm without degrading swing-side, stance/contact, crossing, and progress
gates, or produces a valid kinematic local step where the absolute arm does not. This is only
permission to implement the optimizer; it is not evidence for execution, RepairNet,
cross-prior transfer, route planning, duck/squeeze, or the full E1 claim.

E1b, not the current harness, adjudicates the public-interface path. If independently varied
scenes/donors with equal-budget SONIC replay show neither a residual advantage over same-
support absolute packets nor any physically valid achieved step off the floor, stop expanding
this packet path. Record the negative result and pivot to an output-space residual adapter
`x_adapt = x_nominal ⊕ A_psi(S, x_nominal)` or per-window latent optimization. Do not retreat
to an audit-paper claim.
