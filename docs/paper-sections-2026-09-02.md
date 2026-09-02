# Paper section drafts — §4, §6, §7 (ICRA 2027, 2026-09-02)

**Status:** first full drafts of the three results/mechanism sections, written from committed
ledgers only. Every number carries its ledger in brackets; numbers from campaigns still running
(EXP-024 prospective test, EXP-028) are marked ⟨pending⟩ with the slot they fill. Terminology
locked by the framing note: "execution-audited", never "certified"; "evaluator cutoff", never
"fall", until EXP-028 says otherwise; the calibrated gate is `run > 0.20 s`, the post hoc
optimum `run > 0.28 s (≥ 0.32 s, 8 frames)`. Section numbers follow the current outline:
§3 protocol and endpoints, **§4 the trackability contract and audit**, §5 addressability results,
**§6 interface mechanism**, **§7 executed evaluation**, §8 cost curves and dataset, §9 limitations.

---

## §4 The trackability contract and the audit it closes

**Problem.** A frozen motion prior returns a kinematic reference. Whether a released tracker can
follow it is decided downstream, one rollout at a time, and a rollout costs 52–94 s of Isaac time
per launch of at most 36 references [CLAUDE.md, exp1b/EXP-022A receipts]. A generator-side
predictor of the tracker's verdict is therefore the piece that turns a prior into a usable
substrate: it says which references deserve a launch and, when it refuses, why. We call the
predictor a *trackability contract* when it is (i) computed from the reference alone, (ii) fixed
before the tracking data it is tested on exist, and (iii) tied to a specific released tracker's
termination behaviour by measurement rather than by argument.

**Definitions.** For a reference sampled at 25 fps we compute, from the archived joint
configuration and the calibrated collision model, a per-frame *support mask*: a foot is in the
support envelope when its lowest collision primitive is within 4.65 cm of the floor and its
planar speed is below 1.18 m/s [exp016 receipt, sha `f6dba8be…`, frozen 2026-08-31 on the exp1c
calibration corpus before any prompt-elicited clip existed]. A frame with neither foot in the
envelope is *no-support*. The primary contract feature is the longest bilateral no-support run,
`max_unsupported_run_s`; the calibrated **gate** flags a reference when `run > 0.20 s` (six or
more frames at 25 fps). Seventeen further features (bilateral flight fraction, mean supporting
feet, pelvis height range and peak, ballistic ratio, joint-speed quantiles, tilt, yaw rate; full
list in the released schema) are computed for the ablations of §4.3 but play no part in the gate.
A no-support run is a **float** when it is longer than the ballistic flight time of the pelvis
rise inside it, `T_ball = 2·sqrt(2·Δz/g)`; the *ballistic ratio* is the run length divided by
`T_ball`.

**The elicited step is a float (Fig. 5a).** Under the STEP prompt the prior lifts a foot inside a
bilateral no-support phase. In the twelve exp021 clips that exactly clear the 5 cm box at the
scene-fixed centre, the run lasts 0.44–1.04 s in nine clips and 2.4–3.1 s in three, while the
pelvis rises only 0.02–0.26 m inside it: ballistic ratios of 1.3–15.6, root peaks of 0.97 m on
average against 0.78 m in free-root walking on the same route [`outputs/analysis_trackability_contract/`,
exp023 `all_walk`]. None of the 44 clips with a lift of 3 cm or more passes the gate (upper 95 %
bound 0.08); the eight clips that pass the gate are all among the 20 that never lift, and all eight survived tracking. This is not a hop: SONIC
tracks real jumps, and the reference here asks for a body to hang in the air without the vertical
impulse that would carry it.

**The gate predicts the evaluator's verdict (Fig. 5b, Tab. 4).** Every one of the 64 exp021
references was tracked once by SONIC under its release evaluation terminations (physics seed 0,
achieved states archived at 50 Hz) [EXP-022A receipt]. Fifty-three were terminated. The
calibrated gate flags 53/53 of them (sensitivity 1.00, Wilson 0.93–1.00) and passes 8/11 survivors
(specificity 0.73, 0.43–0.90); the single feature reaches AUC 0.997 (bootstrap 0.987–1.00), and a
leave-one-out logistic model over all 18 features reaches 0.976, so nothing beyond the run length
is needed. The post hoc sweep over the threshold gives its optimum at `run > 0.28 s` (≥ 0.32 s,
eight frames): 51/53 flagged and 11/11 passed; we report it as a sweep, never as the gate. Two
controls separate "float" from "did it lift": lift height alone reaches AUC 0.92 (0.86–0.97), and
inside the 20 clips that never lift, ten of which were terminated, the run length still separates
the outcomes (AUC 0.98). The same feature transfers to a second family with a different actuation
channel — the 144 position-channel lift references of EXP-1C (135 terminated, 9 survivors) — at
AUC 0.92 (0.82–0.98); a logistic model fitted on either family scores the other at 0.92 and 0.90.
The 144 EXP-1C control references are the corpus the thresholds were calibrated on and carry no
predictive claim.

**What "terminated" means, and does not mean (Fig. 5c–d).** SONIC's `tracking/eval` configuration
ends an episode on a pelvis-height error above 0.25 m, a pelvis-orientation error above 1.0 rad, an
ankle/wrist height error above 0.25 m or an ankle position error above 0.2 m; the firing term is
not logged. In the terminated rollouts the cutoff lands within 0.2 s of the reference's first
no-support run of at least 0.2 s in 47/53 (median +0.04 s, IQR 0–0.12 s), and at the last
archived sample every robot is upright: pelvis 0.56–0.95 m, none below 0.50 m, one tilted beyond
23°, while the reference's higher foot is a median 0.40 m in the air and the foot-height error
exceeds 0.2 m in 24/53 [`termination_snapshot` in the contract rows]. We therefore describe the
outcome as an **evaluator cutoff on a support-envelope violation while upright**, not as a fall.
Whether the cutoff-free robot falls, stalls, walks through the box or clears it is a separate
measurement ⟨EXP-028, pending: outcome class per clip, first-firing term recomputed offline,
physics-seed test–retest⟩.

**Prospective test.** Everything above is post hoc on the exp021 pool. The contract's status as a
predictor rests on ⟨EXP-024, pending: 128 fresh references under four native root contracts,
per-clip gate predictions committed before any launch; preregistered P1: ≥ 90 % of flagged
references terminate, ≤ 30 % of passed ones do, AUC ≥ 0.90; a stricter predeclared level at
AUC ≥ 0.95⟩.

**Why this is the pipeline's launch-budget filter.** At the calibrated gate the pipeline would have
skipped every launch the evaluator cut off (53/53) at the cost of three of eleven survivors; at
the post hoc optimum, 51/53 at no survivor cost. The filter costs one collision-model pass per
reference (about 2–7 s on a CPU core, dominated by the box-height scan the audit needs anyway) and
no tracker time. Its refusals are labelled: a reference is rejected *because* its support envelope
is empty for a stated number of frames, which is the same quantity a generator would have to fix.

---

## §6 Interface mechanism: what each native channel moves, and what it cannot

**Question.** ARDY-G1 exposes five writable conditioning channels (root planar path, root height,
heading, joint rotations, joint positions) and a text prompt, and its released streaming interface
re-prompts window by window. §5 reported that the STEP prompt elicits a whole-body-clearable lift of at least 3 cm
in 44/64 clips (69 %, Wilson 0.57–0.79; any positive lift 49/64) whose tallest lift lands early: the root reaches the lift position
at frame 35 in the median clip (q10–q90 21–55; 1.4 s), inside the first 50 frames in 40/49
(0.69–0.90), after frame 60 in 4/49 [`outputs/analysis_event_frames/`, committed analyser].
Thirty of the 49 clips with any lift contain more than one lift region, so "early" is a statement
about the tallest region rather than about a single emission; under the windowed streaming
interface the first clearable lift lies within 3.8 s of the start in 6/8 fresh seeds while the
tallest region can come later (5/8) [exp023 `step_0`, A0 analyser]. Here we ask which channel, if any, moves that event to a scene-specified place or time.

**Sparse rotations and phase-aligned packets do not place it, and they attenuate it.** Writing the
donor's step as a coherent, gait-phase-aligned rotation packet on the nominal walk (exp019 v7;
8 seeds × 3 arms) elicits the step in 8/8 absolute and 8/8 residual clips but lands it a
median 2.33 m / 1.32–1.59 m from the commanded obstacle: 0/24 collision-free at the obstacle in
every arm, placement gain ≈ 0 (slopes −0.44 and +0.02), and the rotation compliance at the
constrained frames is *negative* (−0.26 / −0.44: the clips end farther from the command than the
unconstrained walk), with no response lag in [−40, +80] frames that beats the nominal baseline.
The prior is also not translation-equivariant along the route: a shifted start is absorbed as a
speed change [exp019 v7 receipt; `docs/ramp-e1a-controls-2026-09-01.md`]. Added to the prompt, the
packet does not merely fail to help: on matched seeds it lowers the mean lift from 0.121 m to
0.072 / 0.061 m and removes every 5 cm clearance (3/8 → 0/8; paired loss 6.0 / 6.8 cm in 6/8
seeds) [exp020 receipt]. The structured channel is read as a style cue, not as a constraint.

**Route timing and waiting do not place it either.** Warping the route so that the root arrives at
the obstacle when the prior's own step would occur preserves the intended gait in 1/6 clips
[exp018], and waiting for the gait event at a predeclared obstacle succeeds in 1/8 [exp017]: the
prior re-plans its gait under root-path timing changes.

**The position channel lifts the foot but produces flight.** Requesting the lift directly through
the joint-position channel (EXP-1C, the preregistered amplitude ladder, 288 references) produces
bilateral flight at every amplitude and no tracked step-over; those 144 lift references are the
transfer family of §4 (AUC 0.92) [`outputs/exp1c_stepover/`].

**A delayed prompt elicits the step less often, later, and unplaced.** Through the released
minimum-history streaming interface (one four-frame token of history per window), switching the
prompt from WALK to STEP at frame 52 elicited a whole-body-clearable step in 0/8 seeds in EXP-023
and in 3/8 fresh seeds in EXP-023b — pooled 3/16 (0.07–0.43) against 6/8 (0.41–0.93) from frame 0 —
at latencies of 0.8–3.0 s after the switch; none of the delayed events cleared a box at the
predicted centre (0/8 at every height), against 1/8 at 3 cm for the from-start arm [exp023,
exp023b receipts]. The handoff transmits the prompt: the delayed arms diverge from the all-WALK
control by 0.14–0.20 rad RMS over the post-switch joints (the pre-switch prefix is byte-identical),
and the SQUEEZE prompt delivered from frame 0 elicits a lift in 5/8. What the interface does not
own is *when* the behaviour lands relative to the prompt and *where* relative to the scene. We
previously wrote that the event is "tied to the rollout origin"; EXP-023b's 3/8 withdrew that
mechanism claim, and the statement we retain is the measured one.

**The SQUEEZE prompt is a lifting gait, not a sidle.** EXP-023b's positive control failed at its
substrate gate: with the root pinned to the route, "steps sideways through a narrow gap" produced
no heading deviation above 28°, no lateral excursion above 7 cm and no foot crossing in 0/8 clips,
but a whole-body-clearable lift in 5/8. A positive control for prompt switching needs a behaviour
this prior can express under a pinned root; that control is still owed and needs fresh seeds.

**Reading.** Every native channel that reaches the decoder either leaves the event where the prior
puts it or attenuates it; the one channel that changes *what* is emitted (text) does not control
*when* or *where*. This is the addressability result the contract of §4 completes: even where the
event can be staged onto the obstacle post hoc (12/64 exactly clear a 5 cm box at 1.2 m, the
unique maximum of the exact-centre curve; Fig. 4), the reference it produces is a float that the
tracker's evaluator rejects.

---

## §7 Executed evaluation: paired retention under a released tracker

**Endpoint.** For each reference we launch SONIC once (release checkpoint `sonic_release/last.pt`,
sha `e6bdab3f…`; release evaluation terminations; physics seed 0; the obstacle absent from Isaac)
and replay the achieved joint states against the same collision model and the same box. A
reference is *retained* at height h and centre x when the rollout is not terminated, passes the
box inside the lateral corridor, finishes beyond it, and is exactly collision-free at h. We report
paired reference→achieved retention, never raw achieved clearance, because an achieved trajectory
that never reaches the box clears it vacuously (43/64 raw "clears" at 5 cm in EXP-022A are of that
kind).

**Result (Tab. 1, Fig. 4b).** Of the 64 exp021 references, 13/12/11/7/6/2 exactly clear the
staged box at 3/5/8/12/20/30 cm; the paired guarded retention is **0** at every height
(one-sided 95 % bound 0.21 on the 3 cm rate), 53/64 rollouts are cut off by the evaluator, and
every 5 cm-clearing reference is cut off at t = 0.2–1.2 s with the achieved root 0.18–1.12 m down
the route — before the box's front edge at 1.10 m [EXP-022A receipt, summary]. Dropping the
non-termination guard recovers nothing: no achieved trajectory both passes the box and clears it.
At the mid-route control centre (3.6 m) the reference clears 2/64 and retention is again 0. The
kinematic best-of-N curve at 1.2 m (N90 = 12 draws at 5 cm) buys reference clearance, not
execution (Fig. 4b).

**What the cutoff is (Fig. 5c–d).** As established in §4, the cutoffs are evaluator decisions on a
support-envelope violation taken while the robot is upright, within 0.2 s of the reference's first
no-support run in 47/53. They are the correct decision for a tracker that refuses to follow a body
into the air without an impulse; they are not evidence that the robot fell. ⟨EXP-028, pending:
with every tracking-error termination raised to an unreachable threshold, the outcome class of
each of the 64 rollouts — fell / stalled / walked through / cleared — with the first-firing term
and its timestamp recomputed offline, and the termination agreement across physics seeds 0–2 that
bounds what any single-seed label in this paper can claim.⟩

**The duck axis, for scale.** The same executed endpoint on the multi-scene duck-under-beam family
(36 scenes, 526 references, 859 rollouts, 24 launches) gives 0/859 traversals, with 554/859
terminated or short of the progress criterion and only 200 reaching the first beam [exp1b_v2
receipt]. That campaign's 14 s clip cap forced references at 1.04–1.79 m/s and even zero-dip walks
stalled at 2.15 m of 7.2 m, so it is "no traversal under this protocol", not "the duck is
untrackable"; the execution-calibrated clearance-loss model it was meant to fit is not
identifiable (`gate.json: insufficient_data`). We keep it as the honest denominator of the
execution tier.

**Prospective and contract-ablation results.** ⟨EXP-024, pending: 128 fresh references under
`free`, `pin_y 0.78 m`, `pin_h 0`, `pin_yh`; elicitation, exact clearance at 1.2 / 3.6 m, gate
predictions committed before SONIC, four launches of 32, guarded retention; P1–P4 and the
constructibility of the pinned arms.⟩ Whatever they return, the paired-retention endpoint, the
contract and the physical-outcome classes are what a tracker or a generator would have to satisfy
for the loop to close; the paper ships them as the specification.

---

### Numbers still owed to these sections (fill from receipts, never from memory)

| slot | source | fills |
|---|---|---|
| EXP-024 prospective 2×2, AUC, P1/P1-strong, arm constructibility, `free` replication of 12/64 | `outputs/exp024_reference_contract/receipt.json` (analyze stage) | §4 prospective test; §7 last paragraph; Tab. 4 |
| EXP-028 outcome classes, first-firing term, agreement | `outputs/exp028_termination_free_rollouts/` | §4 "what terminated means"; §7 "what the cutoff is"; Fig. 1 caption; title wording |
| A0 on EXP-023b's three delayed events (are they floats?) | `experiments/analyze_event_frames.py` extended to the 023b archive | §6 delayed-prompt paragraph |
| re-designed prompt-switch positive control | new protocol, fresh seeds, CPU encoder for new phrasings | §6 SQUEEZE paragraph |
