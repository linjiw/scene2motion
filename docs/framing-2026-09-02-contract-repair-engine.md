# Framing note — the trackability contract, measured-deficit repair, and the verified data engine

**Written:** 2026-09-02, in response to the principal-investigator guidance of the same date
("Uncompromising empirical rigor" + "actionable methodological and systems contribution";
three pillars: Trackability Contract, Measured-Deficit Closed-Loop Repair, Verified Dataset
Engine). **Status:** working framing for the ICRA 2027 draft. It does not replace the plan of
record (`docs/plan-2026-09-01-icra2027.md`); it binds the three-pillar narrative to receipts and
records where the guidance's wording must be tightened to stay inside them. Receipts outrank
this note; this note outranks the plan's §2 wording where the two differ.

## 0. The reconciliation in one table

| guidance says | receipts support today | pending | wording the paper may use |
|---|---|---|---|
| Pillar 1: a *trackability contract* — kinematic constraints that predict controller survivability | yes, **post hoc**: the calibrated 0.2 s contact gate (frozen in the exp016 receipt before exp021 existed) flags 53/53 SONIC-terminated exp021 clips and passes 8/11 survivors; longest no-support run AUC 0.997 (bootstrap 0.987–1.00); transfer to the position-channel family 0.92, cross-family logistic 0.92/0.90 (`outputs/analysis_trackability_contract/`) | **EXP-024** prospective test on 128 fresh clips with predictions committed before SONIC; **EXP-028** physical meaning of "terminated"; **EXP-024b** whether a real jump survives (flight per se vs float); **EXP-025** cross-prior | "calibrated contract", "predicts the release evaluator's cutoff"; after EXP-024: "prospectively validated"; after EXP-028: "predicts [physical outcome class]" |
| "Testing kinematic clearance without dynamic support certification is a mirage" | yes: 12/64 exact 5 cm clears → 0 retained under the release evaluator (EXP-022A); every clearing clip lifts inside a non-ballistic bilateral no-support phase 0.44–3.12 s long (ballistic ratio 1.3–15.6) | EXP-028 decides whether the cut-off robot would have fallen, stalled, walked through or cleared | "kinematic mirage" is fine as a phrase; **"certification" is a dead word** in this repo (plan §1.2, CLAUDE.md) — use "support-phase gate", "contract", "audit" |
| Pillar 2: measured-deficit closed-loop repair Δq = e/\|g′\| with 3τv anticipation rescues reachable modes and refuses unphysical ones | yes, **kinematic, multi-scene**: Table 2 (`outputs/phase4e_architecture_v2_s8`, 36 scenes × 8 seeds × 15 arms): +1 repair lifts every proposer to ≥ 0.91 and +2 to ≥ 0.993 collision-free; repair beats equal-budget resampling wherever the proposal is misshaped (TCN 63 vs 0, QP 112 vs 1 discordant rows) and **loses to it on margin for the already-right heuristic proposer** (41 vs 18); forward secant within 4 mm of the exact inverse where the centred one undershoots by 27 %; the 3τ lead confirmed by the 1.5–8τ sweep; bounded iterations + monotonicity; 26/300 quantified refusals in the corpus pilot | nothing planned before Sep 15 for the duck axis; the *executed* duck result is 0/859 traversals under a protocol confounded by the 14 s clip cap (EXP-1B), and τ(d) is `insufficient_data` | "measured-deficit repair operator", "kinematically verified across 36 scenes", "refuses with a quantified deficit"; **not** "execution-certified repair", **not** τ(d) as a released label |
| The repair "strictly refuses unphysical modes (the float)" | yes, by construction and by measurement: the operator acts only on the overhead channel and refuses on saturation; for the step family no native channel yields a contact-consistent step (EXP-1C position lifts = bilateral flight, killed; exp020 packets suppress amplitude 0.121 → 0.072/0.061 m); 0/44 lifting exp021 clips pass the 0.2 s gate (Wilson upper bound 0.08) | EXP-024 asks whether any native *root contract* (pin height / heading) yields a contact-consistent, clearing, retained step | "the gate refuses every lifting clip"; "no native channel tried yields a supported step" (scoped to the channels tried) |
| Pillar 3: ≈10⁵ verified, labelled, refusal-annotated clips per GPU-day | **only at the kinematically-verified tier and only by extrapolation**: corpus pilot v2 produced 268 kinematically verified duck records (192 accepted at 0.18 m + 76 below margin), 6 rejected, 26 refused-with-reason, from 300 scenes in 301.9 s on one RTX 5080 → 7.7 × 10⁴ per GPU-day *if* that 5-minute rate held; executed-traversal tier: 0/859 (duck), 0/12 (step) | EXP-028 adds physical outcome classes; EXP-024 adds 128 gate-labelled, tracked records | the four-tier table with wall-clock (house rule 10); **never** "10⁵ verified clips per GPU-day" as one number (plan §1.2) |
| downstream benefit from "verified execution margins τ(d)" | **no**: τ(d) could not be fit (`docs/exec-gate-audit.md`, 0 uncensored loss observations) | a τ(d) needs an interaction-reaching campaign; not before Sep 15 | "achieved-state archives and per-reference evaluator outcomes", "contract features as labels"; τ(d) is future work |
| "calibrated, execution-certified whole-body clearance pipeline" | calibrated: yes (thresholds from the exp016 receipt); execution-*audited*: yes (859 + 352 tracked rollouts with achieved-state replay); execution-certified: **no** (zero retained traversals in either family) | — | **"calibrated, execution-audited whole-body clearance pipeline"** |
| EXP-023b positive control on "heading turn or speed change" | **run 2026-09-02:** SQUEEZE does not sidle under the pinned root (0/8; substrate gate refused) but lifts in 5/8; delayed STEP elicited 3/8 (EXP-023: 0/8) → pooled 3/16 vs 6/8; delayed prompts move the joints 0.14–0.20 rad RMS | a re-designed control on fresh seeds; turn/speed phrasings need the CPU encoder (EXP-027) | "a later prompt elicits the step less often, later, and unplaced"; never "does not elicit" |
| EXP-028: "divergence timestamp, joint torque saturation, base tipping angle" | divergence timestamp and tipping angle: yes from schema-v2 achieved qpos (offline recomputation of the evaluator terms; up-vector from the root quaternion); **torque saturation: no** — the archive stores qpos only | schema-v3 export (qvel + applied torque) is a half-day change to `sonic_state_export.py`; listed as *should*, after EXP-028 lands on v2 | report "tipping angle and first-firing evaluator term with its timestamp"; torque saturation only if v3 lands |
| EXP-024: "AUC > 0.95 prospectively" | post hoc 0.997 on exp021; 0.92 on transfer | drafted P1 is AUC ≥ 0.90 with sensitivity ≥ 0.90 and false-alarm ≤ 0.30; a stricter "strongly confirmed" level at ≥ 0.95 is added to the protocol before preregistration | report whichever level is reached, both predeclared |

## 1. What each pillar can say, with evidence

### Pillar 1 — the trackability contract

*Definition (paper §3).* A reference is **contact-consistent** when no bilateral no-support run
exceeds 0.2 s (5 frames at 25 fps), with *no-support* = neither foot inside the calibrated
support envelope (foot bottom ≤ 4.65 cm and planar speed ≤ 1.18 m/s; thresholds from the
exp016 receipt, sha `f6dba8be…`). A no-support run is a **float** when its duration exceeds the
ballistic flight time of the pelvis rise inside it. The contract is the gate plus the feature
set of `experiments/analyze_trackability_contract.py::features`.

*Result (paper §5).* On the prompt-elicited family (exp021, 64 clips, one scene, physics seed
0): the gate flags 53/53 evaluator-terminated rollouts (sensitivity 1.00, Wilson 0.93–1.00) and
passes 8/11 survivors (specificity 0.73, 0.43–0.90); the single feature reaches AUC 0.997
(0.987–1.00); the leave-one-out logistic over 18 features 0.976; lift height alone 0.92, and the
feature still separates the 20 non-lifting clips (AUC 0.98), so it is not "did it lift". All 12
clips that exactly clear the scene-fixed 5 cm box (and all 11 at 8 cm) are floats: no-support
runs 0.44–1.04 s (nine) or 2.4–3.1 s (three), pelvis rise 0.02–0.26 m inside the run, ballistic
ratio 1.3–15.6, root peak 0.97 m vs 0.78 m in free-root WALK on the same route. Transfer: exp1c
lift arms (144 clips, 9 survivors) AUC 0.92 (0.82–0.98); cross-family logistic 0.92 / 0.90.

*Why the evaluator, not the robot, ends the episode (paper §5.2).* SONIC's `tracking/eval`
override terminates on pelvis height error > 0.25 m, orientation > 1.0 rad, ankle/wrist height
error > 0.25 m or ankle position error > 0.2 m. Termination lands within 0.2 s of the reference's
first ≥ 0.2 s no-support onset in 47/53 (median +0.04 s; 11/12 clearing clips); at the last
archived sample every terminated robot is upright (pelvis 0.56–0.95 m, 0/53 below 0.5 m, 1/53
tilted > 23°) while the reference's higher foot is a median 0.40 m in the air and foot-height
error exceeds 0.2 m in 24/53. **"Evaluator cutoff" and "dynamic loss of balance" are therefore
distinguished in every sentence**: EXP-022A measured the former; EXP-028 measures what happens
when the cutoff is removed.

*Information-flow sentence.* Resampling cannot fix what the contract measures: 0/44 lifting
clips pass the gate (upper 95 % bound 0.08), so a best-of-N selector on this prompt has nothing
contact-consistent to select, however large N — the kinematic best-of-N budget (N90 = 12 at the
exact centre) buys clearance in the reference and zero retention in execution.

### Pillar 2 — measured-deficit closed-loop repair

*Operator (paper §3.2; `scene2motion/verify/loop.py`, REPORT §10).*
e(s) = max(0, target − measured_overhead(s)); Δq(s) = e(s)/|g′(q(s))| with a **forward** secant
(within 4 mm of the exact inverse over q ∈ [0, 0.7], where the centred secant undershoots by up
to 27 %); slope floored at 0.15 m/unit so saturation yields a deliberately partial repair; the
correction is shifted early by 3τ·v with τ the generator's measured first-order lag (the
1.5/3/5/8τ sweep: 3 ≡ 5 at 1.000/0.417, 1.5 worse, 8 worse on margin at 0.250); Δq ≥ 0 and q is
clipped (monotone: a repair can only deepen or extend); two iterations, then refuse with the
quantified deficit. Contacts are split by normal into overhead (the schedule's job) and lateral
(the route's job), because an undifferentiated minimum tells a repair loop to crouch at walls.

*Result (paper §4; Table 2).* 36 OOD scenes × 8 paired seeds × 15 arms = 4,320 rows, proposal
held byte-identical across the resampling arms at equal budget:

| proposer | one-shot | +1 repair | +2 repairs | best-of-2 | best-of-3 |
|---|---|---|---|---|---|
| heuristic | .983 / .535 | 1.000 / .622 | 1.000 / .646 | .993 / .656 | 1.000 / **.726** |
| QP | .872 / .007 | 1.000 / .326 | 1.000 / .417 | .944 / .017 | .972 / .031 |
| TCN | .729 / .003 | .913 / .278 | .993 / .375 | .764 / .007 | .774 / .014 |

(collision-free / meets 0.18 m margin; scene is the inference unit.) Repair beats equal-budget
resampling wherever the proposal is misshaped (TCN +2 vs best-of-3: 63 vs 0 discordant rows on
collision, 104 vs 0 on margin; QP 112 vs 1) and **loses on margin for the already-right heuristic
proposer** (41 vs 18 in resampling's favour). That asymmetry *is* the information-flow claim:
measurement supplies the structural correction a wrong proposal needs, and supplies nothing a
right one lacks. The paper states both halves.

*Refusal as an output.* Corpus pilot v2: 26/300 scenes refused with a named deficit; the
step-over family is refused as a class by the contract gate (above). The operator never acts on
a no-support run: there is no scalar command on this prior that shortens a float (EXP-1C, exp020,
exp019 v7), and EXP-024 tests the last native root contracts (pinned height / heading).

*Scope the paper must keep.* Kinematic. The executed duck campaign (EXP-1B, 859 rollouts, 36
scenes) reached 0/859 traversals because the 14 s clip cap forced 1.04–1.79 m/s references and
even zero-dip walks stalled at 2.15 m of 7.2 m; it is "no traversal under this protocol", not
"the repair does not execute". No τ(d) exists.

### Pillar 3 — the verified dataset engine

*Tiers (house rule 10), all on one RTX 5080 16 GB:*

| tier | duck family (corpus pilot v2, heuristic, ≤ 2 repairs) | step family (v2 sampler) |
|---|---|---|
| generated | 300 scenes / 301.9 s (≈ 1 scene/s incl. verification) | 5,393 references; ≈ 0.2 s/clip batched |
| kinematically verified | 268 (192 accepted at 0.18 m + 76 collision-free below margin); 6 rejected; 26 refused with reason | exp021: 64 scored at 7.2 s/clip incl. the box-height profile scan |
| gate-accepted (contract) | n/a (duck contract = EXP-026, CPU, appendix) | 8/64 pass the 0.2 s gate; 0/44 lifting clips |
| tracker-executed | 859 rollouts / 526 references; 0 traversals | 352 references (64 prompt + 288 position-channel); 0 retained at the box |

Extrapolated rate at the kinematically-verified tier: 7.7 × 10⁴ records per GPU-day from a
302-second pilot — reported as an extrapolation with its tier, never as "10⁵ verified clips".
Record schema (paper §6): scene, request, response, signed spatial error, contract features,
kinematic outcome, evaluator outcome, physics seed, resolved termination config, refusal reason,
provenance (HF revision, denoiser sha, runtime commit, `g1.xml` sha, tracker commit + ckpt).
Downstream use is stated without a learned model: the 0.2 s gate would have skipped every
terminating SONIC launch (53/53) at the cost of 3/11 survivors; the post hoc 0.32 s cut 51/53
at no survivor cost — a launch-budget filter, not a training claim.

## 2. Text drafts (three-pillar framing, inside the receipts)

**Title.** *A Trackability Contract for Frozen Humanoid Motion Priors: Verifying, Gating and
Repairing Generated Traversal References Before a Tracker Sees Them.* (Shorter: *Verify, Gate,
Repair: What a Frozen Humanoid Prior Can and Cannot Be Asked to Traverse.*)

**Abstract (≈200 words).** Frozen humanoid motion priors promise natural whole-body motion for
scene traversal, but a generated reference is not an executable one. We present a calibrated,
execution-audited pipeline that turns a released prior (ARDY-G1) into verified traversal
references for a Unitree G1, and use it to measure what a downstream builder must know before
planning on such a prior. (1) A *trackability contract*: kinematic support-phase features,
calibrated before the data existed, that predict whether a released whole-body tracker (SONIC)
follows a reference. The text prompt "steps over an obstacle" elicits a clearable lift in 49/64
clips, yet every clip that clears a scene-fixed 5 cm box does so inside a non-ballistic
bilateral no-support phase 0.4–3.1 s long; the tracker's release evaluator cuts off all of them,
upright, at the onset of that phase, and the contract's 0.2 s gate flags 53/53 cut-off clips
while passing 8/11 survivors (AUC 0.997; cross-family transfer 0.90–0.92). (2) A
*measured-deficit repair operator* — forward-secant gain inversion with lag-derived anticipation
and strict monotonicity — that closes overhead-clearance deficits the prior can reach (36 scenes
× 8 seeds: ≥ 0.993 collision-free after two repairs, beating equal-budget resampling for the
two misshaped proposers) and refuses, with a quantified deficit, the ones it cannot, including
the float. (3) A *tiered, refusal-annotated data engine* with generated / verified /
gate-accepted / tracker-executed accounting. We release the protocol, the gate, and 5,393
provenance-bound references, 878 with tracked outcomes.

**Intro, paragraph on interceptions (verbatim candidate).** "Terminated" in a tracker's
evaluation harness is an interception, not a fall. In every cut-off rollout of our
prompt-elicited step family the robot was upright at the last archived state (pelvis 0.56–0.95
m); what ended the episode was the reference asking both feet to leave the ground for
0.4–3.1 s while rising a few centimetres — a motion no controller can track because it is not
a motion a body can perform. We therefore report two outcomes for every reference: the
evaluator's cutoff under the release configuration, and the physical outcome class with the
cutoff removed (fell / stalled / walked through / cleared).

**Intro, information-flow paragraph.** Open-loop imitation and naive resampling treat the prior
as a black box that is sampled until something passes. Measurement adds what sampling cannot:
a *signed, located deficit* that names the channel to correct (overhead vs lateral), the
command increment that corrects it (Δq = e/|g′|), and — when the deficit is not reachable by any
native channel — a refusal with its reason. On the duck axis this turns a 0.73 one-shot
proposer into 0.993 in two repairs where equal-budget resampling reaches 0.77; on the step-over
axis it turns "clears in 12/64" into "0/44 contact-consistent", which no budget repairs.

**Contribution bullets.**
- C1 — *Trackability contract and audit protocol.* A preregistered, fail-closed protocol
  (elicitation → exact scene-relative clearance → event timing → channel gain/compliance/lag →
  delayed prompt with positive control → contact gate → paired tracker retention with a guarded
  endpoint) and the calibrated gate, prospectively tested (EXP-024) and physically grounded
  (EXP-028).
- C2 — *Measured-deficit repair.* The forward-secant, lag-anticipated, monotone repair operator
  with bounded iterations and quantified refusal; 36-scene × 8-seed evidence against equal-budget
  resampling; the refusal of the float as its negative half.
- C3 — *Verified data engine and dataset.* Tiered accounting, record schema, provenance binding,
  and the released references with achieved-state archives.

**Limitations (must appear).** One prior for every step-over rate (cross-prior only if EXP-025
lands), one prompt phrasing, one route, one scene; physics seed 0 until EXP-028's re-roll;
"terminated" is an evaluator cutoff whose firing term is not logged (EXP-028 recomputes it
offline); the obstacle is absent from Isaac; the repair operator is validated kinematically and
its executed duck campaign is protocol-limited; the delayed-prompt result is scoped to Horizon52
at the released minimum history.

## 3. Execution status and runbook (2026-09-02)

**Host.** The GPU is held by two co-tenant Isaac jobs (≈ 11.9 GB used, ≈ 4.4 GB free; 8 GB RAM
available) — every protocol's host gate (≥ 12 GB free VRAM, ≥ 18 GB RAM, no Isaac process for
SONIC) fails. Nothing was launched. `scene2motion/host_gate.py` is the shared gate; every new
driver calls it before touching its output directory.

**2026-09-02 07:30–08:00 EDT update.** All three protocols are preregistered (`976cd91`).
An ARDY-only host-gate preset (4 GiB free VRAM / 8 GiB RAM, measured: 1,076 MiB CUDA reserved,
2,297 MiB RSS for one B=8 call) replaced the Isaac-sized blanket figure for ARDY generation.
The host still carries two co-tenant Isaac jobs (6.3 + 6.8 GB VRAM; 1.6 GB free; 6 GB RAM), so
EXP-023b's first attempt was refused with `--out` untouched; `experiments/launch_when_host_free.sh`
polls the host and launches it in the first window that passes. SONIC campaigns (EXP-028,
EXP-024 sonic stage) stay blocked until no Isaac co-tenant runs. The record schema the engine
emits is specified in `docs/dataset-engine-record-schema.md`.

**Built today (CPU, committed, tests green):** the host gate; the EXP-023b driver
(`experiments/exp023b_prompt_switch_control.py`), the EXP-028 driver
(`experiments/exp028_termination_free_rollouts.py`), the A0 event-frame analyser
(`experiments/analyze_event_frames.py` → `outputs/analysis_event_frames/`), and the EXP-024
driver (`experiments/exp024_reference_contract.py`) — see the commit log for the exact state of
each and their tests.

**Launch order when the host frees (each ≤ 30 GPU-min; protocols flipped to preregistered and
committed immediately before their first sample):**
1. EXP-023b (≈ 1 GPU-min ARDY) — `--dry-run` first, then `--out outputs/exp023b_prompt_switch_control`.
2. EXP-028 `--stage smoke` (two motions, resolved termination config dumped and asserted), then
   `part_a` (2 launches), `part_b` (4 launches), `analyze`.
3. EXP-024 `generate` → `score` → `predict`; **commit `predictions.jsonl`**; then `sonic`
   (4 launches) → `analyze`; then EXP-028's termination-free pass on the 128 EXP-024 clips.
4. EXP-024b (shipped one-leg jump + one retargeted forward jump) if the GPU is idle.
5. EXP-025 Part A if the Kimodo runner recovery lands by Sep 4 noon.

## 4. Decisions taken under this note (PI may overrule)

1. **"Certified" is replaced by "audited"** everywhere the guidance uses it; "certificate" stays a
   dead word. Reason: zero retained traversals in both families.
2. **The 10⁵/GPU-day figure is reported as a tiered extrapolation (7.7 × 10⁴ at the
   kinematically-verified tier from a 302 s pilot)**, never as one number.
3. **EXP-024 keeps the drafted confirmation bar (AUC ≥ 0.90, sensitivity ≥ 0.90, false-alarm ≤
   0.30) and adds a predeclared "strongly confirmed" level at AUC ≥ 0.95**, so the guidance's
   0.95 is testable without loosening anything if it is missed.
4. **EXP-028 delivers divergence timestamp (first-firing evaluator term, offline) and base tipping
   angle from schema-v2 archives; torque saturation is deferred to a schema-v3 export** (qvel +
   applied torque), a *should* after EXP-028 lands.
5. **EXP-023b's positive control is SQUEEZE** (the only cached non-STEP prompt); heading-turn and
   speed-change phrasings join EXP-027 once the CPU encoder can run.
6. The paper's spine stays the measurement paper of the plan of record; the three pillars are
   the *presentation* of its three contributions, with Pillar 2 carried by the duck-axis Table 2
   evidence and the float refusal, not by any new method run before Sep 15.
