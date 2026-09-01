# E1a result: the coherent packet elicits the step-over but does not place it

**Status: completed campaign, descriptive result.** The gait-matched pilot
(`docs/ramp-e1-gait-matched-protocol.md`) ran to completion at K = 64:
`outputs/exp019_gait_matched_stepover_v7/`, schema `exp019-gait-matched-stepover-v5`,
status `complete`, **216/216 frozen-prior samples with exact accounting**, 13/64 eligible
seeds, N = 8 evaluated, three arms each (nominal free, absolute, residual). This is the
first completed absolute-vs-residual comparison in the E1 family, after exp017's pool
exhaustion and exp018's route-warp negative.

## The measurement works: a real step-over is visible, walking is not

| clip | best whole-body box height clearable, anywhere on its route |
|---|---:|
| donor step-over, seed 2603 (the selected packet source) | **0.3013 m** |
| donor step-over, seed 2600 | 0.2889 m |
| the same donors' neutral WALK clips | 0.0000 m |
| the 8 evaluated nominal WALK clips | 0.0000 m (7/8), 0.0050 m (1/8) |

Ordinary walking clears nothing, because the swing foot sweeps through every route
position at low height; a genuine step-over clears 0.30 m. The endpoint is discriminative,
and a nominal walk scoring zero is the **expected baseline**, not a defective scene.

## The result

At the commanded obstacle, over 8 paired seeds:

| metric | nominal | absolute | residual |
|---|---:|---:|---:|
| whole-body box clearance at the obstacle (m) | 0.0000 | **0.0000** | **0.0000** |
| obstacle collision-free | 0/8 | 0/8 | 0/8 |
| kinematic step success | 0/8 | 0/8 | 0/8 |
| obstacle min clearance (m) | −0.0788 | −0.0902 | −0.0956 |
| lead side matches the donor | 1.000 | 0.375 | 0.625 |
| progress ratio | 0.998 | 1.005 | 1.004 |

Residual − absolute is noise on every endpoint (box clearance identically 0; min clearance
median −1.7 mm, 3/5 split; crossing error median −9 mm, 3/5). Both packets slightly worsen
min clearance against the nominal and **break the swing-side match** the nominal had
(absolute 5/8 wrong, residual 3/8).

## Why: the behavior is transported, but not to the commanded place

The packets are not inert. Scanning each generated clip's whole route:

| arm | clips with any liftable clearance | mean best clearance | median distance from that lift to the commanded obstacle |
|---|---:|---:|---:|
| nominal | 1/8 | 0.0006 m | — |
| absolute | **8/8** | 0.0698 m | **2.33 m** |
| residual | **8/8** | 0.0609 m | **1.62 m** |

Every packet arm produces a step-over-like lift somewhere; none produces it at the
obstacle. The lift lands one to three metres away, and is itself weaker than the donor's
0.30 m.

**This is the same failure mode as text elicitation, through the strongest structured
channel available.** exp015 found that a step-over prompt elicits the behavior with box
clearance 0.0 and random placement (8/8). Here the request is not a prompt but an
event-aligned coherent packet: the target swing is identified from measured physical gait
phase, the obstacle is anchored to that swing's own apex on the nominal's own route, the
center shift is exactly zero, and both arms are rendered with identical channel support.
The behavior still does not land where it was placed. Spatial placement of an elicited
behavior is therefore not a text-conditioning artefact — it survives phase-matched,
coherent, residual-or-absolute packet transport.

## Scope and what this does not say

- One donor bundle, one box geometry, per-seed placements, 8 paired seeds, kinematic stage
  only. Descriptive; no confidence intervals, p-values, or population claims.
- It does **not** show the residual representation is worthless — it shows that on this
  endpoint vector neither packet form places the behavior, and that they are
  indistinguishable from each other. A donor whose swing duration matched the prior's own
  walk swings (see the constructibility note) might transport differently.
- Applicability bound, measured separately: a scene where this packet is both
  constructible and placeable exists for ~17 % of walk seeds (6/32, 6/32, 5/32, 13/64
  across four independent pools).

## Consequence for the RAMP program

exp017's decision rule says the residual arm advances to the local response optimizer only
on a descriptive clearance advantage without degrading the other gates. It shows none, and
degrades swing-side matching. The rule's stated fallback therefore applies: **stop
expanding this open-loop packet path and pivot to correcting the realized response** —
either the output-space residual adapter `x_adapt = x_nominal ⊕ A_ψ(S, x_nominal)` or
per-window latent optimization, per exp017 §Staged decision and the RAMP refocus guidance.

There is a strong positive reading available for the method: the packet reliably *elicits*
the behavior (8/8, versus 1/8 for the nominal), and what it fails at is *placement*. That
is precisely a response-space error — measurable from the generated clip, with a signed
distance and a direction — which is exactly the signal a response-conditioned repair
policy is supposed to consume. E1's negative is therefore a direct motivation for E2
rather than a dead end: the open-loop program cannot place the event, so the placement
error itself becomes the feedback signal.
