# E2 protocol (draft — ready to preregister)

**Status: draft with its budget now measured; not yet preregistered.** exp021 has landed
(`docs/ramp-exp021-addressable-region-2026-09-01.md`) and supplies the selection budget
and the staging geometry. Nothing here licenses a claim.

## What E1a and its controls already decided

The RAMP refocus planned E2 as a local response optimizer followed by an amortized repair
policy, and the external guidance ordered it as (1) measure the placement response gain,
(2) a re-anchoring fixed-point loop, (3) selection, (4) attenuation, (5) latent
optimization, (6) an output-space adapter. Steps 1 and 2 are now answered without
spending a campaign, and answered negative
(`docs/ramp-e1a-controls-2026-09-01.md`, `REPORT.md` §36):

* **Placement gain is zero.** Slope of realized lift position on requested position is
  −0.44 (absolute) and +0.02 (residual). The signed error is not consistently signed
  (sd 1.7–2.4 m), so no open-loop lead exists to fit.
* **Re-anchoring presumes a command that is honored somewhere; none is.** Rotation
  compliance at the constrained pairs is negative (−0.26 / −0.44), and no lag in
  [−40, +80] frames beats the nominal baseline for any of 16 clips. A fixed-point loop
  over an unread command has no fixed point to find.
* **Latent optimization is not indicated.** Its precondition is a smooth objective in a
  channel the model responds to; the response is absent, not rough.
* **World-anchoring is closed.** ARDY is not translation-equivariant along the route, so
  a lift at `s` cannot be moved onto an obstacle at `s*` by starting `Δ = s* − s` earlier.
* **The packet itself is a liability.** Against the same prompt with route conditioning
  only, adding the packet halves the elicited amplitude (paired median −6.0/−6.8 cm, 6/8
  seeds) and removes the prompt's only clearances (3/8 → 0/8).

So E2 is not a controller. **E2 is selection**, and its scientific content is the cost
curve: how many generator calls buy a placed, executable traversal, measured honestly,
against the baselines the project already built.

## Design

**Generator side.** STEP prompt, route conditioning only — the arm that won the control.
No packet, no rotation channel, no root-height channel. Fresh seeds, disjoint from every
prior campaign.

**Scene side.** A scene-specified obstacle at a predeclared route position: unlike E1a's
per-seed gait-matched placement, the obstacle does **not** follow the gait, because the
whole point is scene-conditioned traversal. Graded box heights 0.03–0.30 m, with 0.05 m
as the primary operating point and 0.08 m secondary.

**Staging (the placement mechanism).** exp021 measured that the prior emits its
text-requested step-over once, at median clip frame 34 (1.36 s), with 80 % inside the
first 50 frames, and that the per-clip hit rate against an obstacle is 0.25–0.31 at
1.0–1.4 m of route and **zero beyond 3.4 m**. E2 therefore does not command placement: it
**stages** it, generating each clip from a route that begins ≈ 1.2 m before the obstacle.
An arm with the obstacle at mid-route (3.6 m) is the negative control that shows staging
is doing the work — exp021 predicts it scores ~0.

**Selection rule.** Sample N clips; accept the first whose measured whole-body box
clearance at the obstacle meets the height threshold; report calls spent. Selection is on
*measured geometry*, not on a learned verifier — the point is to price the measurement,
and a learned verifier is Texedo's contribution, not ours.

**Arms.**
1. `one_shot` — N = 1 staged, the honest floor (exp021 predicts ≈ 0.31 at 5 cm).
2. `selection` — staged, **N = 8** with geometric acceptance; exp021 predicts ≈ 0.90 at
   5 cm and ≈ 0.90 at 8 cm from per-clip 0.312 / 0.266. N = 8 is preregistered rather than
   fitted per height.
3. `unstaged_selection` — the same N = 8 with the obstacle at mid-route: the control that
   isolates staging from resampling. Predicted ≈ 0.
4. `packet_selection` — the same N = 8 spent on packet arms, to show the representation
   does not become useful when resampled (E1a: per-clip 0/16 against text's 3/8).
5. `nominal` — WALK, free, the no-elicitation floor.

**Endpoints.** E1a's vector, graded: box clearance at the obstacle at each height,
collision-free, crossing error, stance/contact, progress, plus **generator calls and
wall-clock**, which are the quantity being traded. Every arm keeps its planned denominator.

**Execution.** The tracker enters the endpoint here, not later: the kinematic endpoint is
ambiguous at 5–7 cm, and the guidance is explicit that a 0.06 m kinematic lift is exactly
the amplitude at which the gated step-over tracked at 0.375. The SONIC surface is verified
present and callable (`exp011_tracked_addressability.run_sonic`,
`scene2motion.sonic_export.write_motion_pkl`, checkpoint
`sonic_release/last.pt`). Accepted clips are replayed and scored on achieved qpos against
the same geometry, with the exp016 caveat stated: until obstacles are instantiated in
Isaac, executed clearance is a replay against our collision model, not Isaac contact.

**Statistics.** Scene is the inference unit; seeds are the resampling budget within a
scene, never pooled as independent observations. Wilson intervals on every rate. Report
the risk–coverage of the acceptance rule, not a single number.

**Decision rule.** Selection ships if it reaches the 0.05 m operating point at a call
budget the project is willing to pay, *and* the accepted clips survive tracking. If the
per-clip rate is so low that the budget is unreasonable, say so and report the cost curve
as the result — that is a real finding about frozen priors as planners, and it is the
honest end of this line rather than a further controller.

## Open items before this becomes preregistered

* ~~Fill N from the exp021 best-of-N curve~~ — done: N = 8, staged at 1.2 m, r = 0.25 m.
* Decide whether the obstacle position is drawn per scene or fixed, and state whether the
  17 % eligibility figure from E1a is seed-dependent or scene-dependent — a deployed
  system can redraw seeds but not scenes.
* Size the SONIC stage against the GPU the lucid training job is already using.

## Not in E2

The **output-space residual adapter** stays last and outside E2, as the guidance ordered,
and it must be preceded by its own heuristic: a procedural contact-preserving stitch of
the donor swing into the nominal, no learning. E1a is direct evidence for that ordering —
the packet failed precisely by pulling motion off the prior's coordinated manifold, which
is the adapter's characteristic risk, so its judge must be SONIC rather than the kinematic
endpoint. And if it wins, the paper's claim changes and should be stated plainly: the
prior generates locomotion while a separate module authors the event.
