# Native duck correction: next-stage work, 2026-09-06

The active research direction is now bounded, execution-aware native duck correction
with frozen ARDY-G1 and SONIC/default. The September1 method-closure plan and the earlier
two-backend ASTRA plan are historical. A5 and all earlier outcome labels are unchanged.

## Contact event continuation completed

The previously prepared, never-launched continuation ran from the original clean
`/tmp/astra-contact-6vd0OZ/repo` checkout. No source, gate, physics, batch or seed changes.
Receipt: `outputs/astra_contact_event_v2/fixed3_default_present/event_verification.json`.
All32 achieved states, apparatus records and runtime modes exactly match the frozen
A5 replay; all valid-prefix contact arrays remain zero. The sparse event export records
**257 later-excluded-action events, 2,018 contact-point observations**, total normal
impulse169.190546 N s. No valid-prefix or first-excluded-action events were found.
Later events may follow reset and cannot qualify or disqualify the original maneuver.
This closes excluded-event attribution, not controlled positive G1 contact sensitivity
or episode-wide contact freedom. Historical A5 labels remain unchanged.

One operational launch,32 reused executions, zero new scientific units/generations;
65.236 s wall,6125.387 MiB peak child RSS, no abort or retry. Earlier failed attempts
and admission refusals remain in their original directories. Contact sensitivity and
coverage through the prospective continuation interval remain outstanding.

## A6 implementation and preregistration

`scene2motion/scene_relative_duck.py` implements native depth/onset/recovery requests,
scene-relative beam geometry, final whole-body exit and censored continuation,
separate signed nominal/inflated geometry, paired ridge response fitting and a bounded
constrained proposal. The historical beam maps exactly to fixed3. Reference qpos is
never deformed. The proposed correction still requires regeneration and its own absent
preview; this implementation alone is not a demonstrated method improvement.

`experiments/astra_a6_response.py` binds12 development scenes,48 distinct motion seeds,
336 response candidates and48 free-WALK controls. It preserves exact B=8 noise/command
plans, failed samples, completed-scene hashes,32-slot absent apparatus/mode readback,
and resource-gated launches. All descendants remain in development groups. No held-out
or obstacle-present method evaluations are authorized by the A6 identification protocol.

The prospective held-out policy, response-transfer rule, error allowances, baselines and
endpoint still require development evidence and a separate freeze. The0.45 m per second
A6 descriptor derives from half the commanded speed and is explicitly development-only.

## Archived full-route suffix inspected once

`outputs/astra_a5_suffix_v1/summary.json` accounts for all32 fixed3/default present
records. Ten terminate; none of the17 historical local passes terminate. All17 retain
states through8.26 s, with4.28–5.76 s observed after local dwell. Their achieved final
x is2.955–5.834 m (median5.229), versus reference final x7.391–7.494 m. Maximum lateral
excursion among these17 is0.391 m. Final-second displacement ranges−0.0234–0.6658 m
(median0.5722): many still progress at the clip horizon, while some stall later.
Thus local-pass full-route failures reflect substantial longitudinal shortfall and
some later stalling, without cutoff or gross lateral corridor drift in these17.
This is a descriptive archived inspection, not a new long-route endpoint or a causal
claim about why the tracker loses progress. Keep long-route completion secondary.

## A6 d0 collected; slot confound caught before fitting

Atd4607e4, all384 references were generated once. Three32-slot absent batches completed
(96 previews;179.132 s simulator wall). Their apparatus/default-mode checks pass.
Runtime readback shows that baseline and perturbations of the same unit can select
different internal encoders when placed in different slots. This invalidates a paired
command-response interpretation. No response model or clearance/progress analysis was
fit to d0. Preserve the96 as confounded diagnostics; cancel the288 unlaunched d0 previews.
Orchestration stopped during post-process archive loading, after s02's simulator exited;
its read-only verification was then completed. No simulator was killed or retried.

The versioned [paired continuation protocol](astra-a6-paired-response-protocol.md)
reuses all384 references and puts a unit in the same slot across all native variants.
It requires exact default encoder-mask equality against baseline. The unchanged32-env
workload requires128 labelled padding duplicates for the final16 units:512 new actual
previews,384 scientific observations. Total cap608 operational previews including d0;
no additional generations, held-out observations or present outcomes.

Preparation also exposed archive-relative paths in the resource receipt. The pinned
d0 preparation verified them unchanged from the original resource archive directory.
The maintained driver now accepts an explicit resource source and binds its absolute
paths/hashes; a regression test covers cwd independence and tampering. Frozen d0 source
and generated artifacts remain unchanged.

Validation before first generation: full CPU1197 passed in210.18 s. The final17 focused
tests cover the subsequent path-resolution and paired-layout fixes. A new full-suite
check is run for the paired continuation before execution.

## A6 paired collection complete: useful tradeoffs, no method gain yet

All16 paired launches passed apparatus, slot-membership and default encoder-mask checks.
The corpus now contains **12 development scenes,48 motion units,336 native response
candidates and48 free controls**. All384 scientific previews are measured;128 padding
previews remain archived and separately charged. With the96 preserved d0 diagnostics,
A6 used608 operational previews and no obstacle-present or held-out evaluations.

[Campaign receipt](../outputs/astra_a6_campaign_summary_v1/receipt.json),
[complete response analysis](../outputs/astra_a6_response_analysis_v1/summary.json),
[response directions](../outputs/astra_a6_response_figures_v1/response_directions.png).
Generation/decode took29.028 s across48 B=8 calls, excluding model load and analysis.
Paired simulator wall919.980 s; A6 simulator total1099.112 s including d0; peak child
RSS6112.563 MiB. All512 paired previews completed without resource abort or retry.
The separate contact diagnostic's32 replays and65.236 s are not included in A6 costs.

Each row below has48 assigned scene–seed units. These are **absent-preview development
counts**, not obstacle-present passage rates, prospective endpoint results, or A5 relabels.
The development descriptor includes the declared0.45 m continuation, scene-relative exit
deadline, inflated beam nonpenetration and corridor/pelvis/upright checks. Termination
can occur outside the local interval and must not be called a fall.

| Native variant | Complete exit + one-second observation | Development descriptor passes | Tracking terminations |
| --- | ---: | ---: | ---: |
| Baseline |22|7|19|
| Shallower depth |31|8|10|
| Deeper depth |18|3|25|
| Later onset |26|5|12|
| Earlier onset |23|7|14|
| Earlier recovery |24|6|16|
| Later recovery |22|7|15|
| Free nominal |27|0|7|

All336 native references are nonviolating under inflated **beam geometry**, which does
not imply reference motion quality or achieved passage. The execution data exposes
failures not captured by that clearance label. It does not yet prove that an execution-aware
correction chooses better commands than a strong reference/scene-aware control.

Positive-minus-negative paired effects, using every pair where the named metric is
observed (different metrics have different observed subsets):

| Change | Median inflated clearance change | Median exit-time change | Median continuation change |
| --- | --- | --- | --- |
| Deeper duck (0.32 vs0.24 m) |+60.0 mm,29 pairs|+0.38 s,17 pairs|−64.5 mm,16 pairs|
| Earlier onset (lead1.26 vs0.90 m) |−8.1 mm,32 pairs|+0.07 s,16 pairs|+24.5 mm,15 pairs|
| Later recovery (offset1.56 vs1.20 m) |+3.1 mm,35 pairs|+0.06 s,20 pairs|−62.9 mm,20 pairs|

These are conditional development descriptions, not uncertainty intervals or a joint
“typical” response. Missing pairs stay missing. Response signs vary across units; a
universal deeper/earlier/longer correction is not supported. Only **10/48 units across
6/12 scenes** have complete observations for all seven native commands. In particular,
complete-case local Jacobians do not cover the whole development envelope.

**Decision:** do not open held-out confirmation or claim a working transferred adapter.
The next development milestone is a simple pooled response model/table that retains
arrival/censoring status and partial observations, followed by actual regeneration and
exact-preview validation of one bounded correction. Compare it with scene-aware geometry,
a conservative native posture and a sensible budget-matched two-candidate search. Keep
stepping, mode searches and fixed2/fixed3 ranker training paused. Preserve the current
A6 descriptor; any prospective endpoint needs its own pre-test freeze.

## Achieved-state video artifacts and final checks

[Historical local passage](../outputs/astra_a5_achieved_videos_v1/historical_local_pass.mp4)
and [historical local failure](../outputs/astra_a5_achieved_videos_v1/historical_local_failure.mp4)
render archived SONIC states against nominal beam geometry, with explicit historical
labels and no resimulation/contact claim. The deterministic selection uses the upper
median final achieved x within the17-pass and15-failure subsets: a5u7/repeat1 and
a5u3/repeat1. Both posters were visually inspected; ffprobe confirms640×456,25 fps,
207/60 frames. The videos are local artifacts, not a new method evaluation.

Final paired implementation validation: **1202 CPU tests passed in213.55 s**, including
all17 focused response/pairing tests. The later analysis, figure and video scripts ran
successfully; the figure and video posters were inspected. Test logs are retained in
`outputs/astra_a6_validation_v1`. No A6 simulator or campaign poller remains.
