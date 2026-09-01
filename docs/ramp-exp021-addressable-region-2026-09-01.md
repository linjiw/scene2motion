# EXP-021: kinematic addressable-window map, with corrected fixed-obstacle scope

`outputs/exp021_elicited_lift_distribution_v2/`, schema
`exp021-elicited-lift-distribution-v1`, status `complete`, **64/64 samples**. STEP prompt,
route conditioning only, the reference-speed 7.2 m straight route, fresh seeds 4400–4463.
Kinematic stage only.

This measures where text-elicited whole-body clearance appears on one fixed route. The
original note interpreted the resulting tolerance map as a placement method; the correction
below withdraws that interpretation pending fixed-obstacle fresh seeds and tracking.

## CORRECTION, 2026-09-01 — exact fixed-obstacle clearance is lower

**This correction supersedes the 7–8-call traversal-budget interpretation below.** The
historical analysis used `clears(profile, target, radius=0.25, height)`, which counts a hit
when *any* hypothetical box centre within ±0.25 m of the target clears. That is a useful
spatial-tolerance/addressable-window analysis, but it is not the project's primary endpoint:
whole-body `BoxHeightProbe` clearance at one fixed scene obstacle.

`experiments/analyze_exp021_exact_addressability.py` replays all 64 archived qpos at the
historical target x = 1.2 m, with the 0.20 m-deep box fixed there and no neighbouring-centre
search. It validates the completed receipt, the exact `s4400`–`s4463` key set, all three
64-sample count claims, and qpos content hash `50c27a6b…` before scoring. It launches no
generator and is print-only unless `--write` or `--out` is explicit.

Binary hits below call `BoxHeightProbe.clears(qpos, height)` at the exact height. The
continuous `probe()` value is a conservative 5 mm-resolution lower bound; thresholding
that lower bound would undercount one 8 cm clip and one 20 cm clip.

| fixed-box height | exact hits | per clip (Wilson 95 %) | independent N=8 plug-in | N for 90 % | N for 95 % | disjoint sequential N=8 blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 3 cm | 13/64 | 0.203 [0.123, 0.317] | 0.837 | 11 | 14 | 7/8 |
| **5 cm** | **12/64** | **0.188 [0.111, 0.300]** | **0.810** | **12** | **15** | **6/8** |
| **8 cm** | **11/64** | **0.172 [0.099, 0.282]** | **0.779** | **13** | **16** | **5/8** |
| 12 cm | 7/64 | 0.109 [0.054, 0.209] | 0.604 | 20 | 26 | 4/8 |
| 20 cm | 6/64 | 0.094 [0.044, 0.190] | 0.545 | 24 | 31 | 3/8 |
| 30 cm | 2/64 | 0.031 [0.009, 0.107] | 0.224 | 73 | 95 | 1/8 |

The independent columns apply `1 − (1 − p)^N` to the same-pool point estimate; the final
column instead groups the frozen seed order into eight non-overlapping blocks of eight and
asks whether each block contains at least one exact hit. Both are descriptive. **The 1.2 m
target was selected post hoc on these same clips, and every budget in this correction is
therefore post hoc too.** Fresh seeds must validate the exact fixed-box rate, and SONIC must
enter the endpoint, before this becomes a traversal or execution claim.

The addressable-region tables below are preserved as the original tolerance analysis. Read
“within 0.25 m” literally; do not relabel those rates as exact obstacle clearance.

## The prior's spontaneous step-over

* **Elicitation 0.77** (49/64 clips contain a liftable clearance).
* **The archive contains high-clearance capability.** Lift heights: median 0.079 m, q75
  0.184 m, q90 0.251 m, and some clips **reach the probe's 0.400 m cap**. Clips clearing a box *somewhere*: 44/64 at 3 cm, 37 at 5 cm,
  31 at 8 cm, 24 at 12 cm, 14 at 20 cm, 5 at 30 cm. The prior can produce a full
  step-over; E1a's 0.06–0.07 m was the packet suppressing it, not a ceiling.
* **Timing appears early but is not yet a committed result.** A historical, uncommitted
  conversion put the median near frame 34 and roughly 80–86% of detected lifts in the first
  50 frames. Exact counts change with the frame-conversion rule (including 3 versus 4 events
  after frame 60), and event frames are not stored in the EXP-021 rows. Do not cite the exact
  timing numbers until a committed archive-to-frame analysis freezes the conversion.

The spatial distribution is consistent with an early-event concentration, but it does not
show that event timing is immutable. ARDY's native interactive API changes prompts between
autoregressive windows; EXP-023 must test WALK→STEP handoff before any claim that no
conditioning operation moves the event time.

## Original lower-bound tolerance map (not fixed-obstacle success)

Per-clip rate at which the 5 mm-resolution lower-bound profile has a peak within 0.25 m of
the target. These are not exact-height fixed-box outcomes:

| target on route | 3 cm | 5 cm | 8 cm | 12 cm | 20 cm |
|---:|---:|---:|---:|---:|---:|
| 0.6 m | 0.078 | 0.078 | 0.062 | 0.062 | 0.031 |
| **1.0 m** | **0.312** | **0.281** | **0.250** | 0.203 | 0.125 |
| **1.4 m** | 0.297 | 0.250 | 0.203 | 0.141 | 0.109 |
| 1.8 m | 0.141 | 0.094 | 0.062 | 0.062 | 0.031 |
| 2.2 m | 0.047 | 0.047 | 0.016 | 0 | 0 |
| 3.0 m | 0.016 | 0.016 | 0.016 | 0 | 0 |
| ≥ 3.4 m | **0** | **0** | **0** | 0 | 0 |

The last row is zero under this historical lower-bound+tolerance calculation. It does not
contradict exact binary replay at x=3.6 m, where 2/64 clips clear both the 5 cm and 8 cm boxes.

Optimizing the target position on the same inspected pool gave this historical, post-hoc
tolerance operating point:

| box height | best target | per-clip rate | N for 90 % |
|---:|---:|---:|---:|
| 3 cm | 1.2 m | 0.344 | **6** |
| 5 cm | 1.2 m | 0.312 | **7** |
| 8 cm | 1.2 m | 0.266 | **8** |
| 12 cm | 1.1 m | 0.219 | 10 |
| 20 cm | 1.2 m | 0.141 | 16 |

Averaged over obstacle positions spread across the whole route, the same tolerance rates fall
to 0.04–0.10 and the budget rises to N ≈ 16–32. The defensible conclusion is descriptive:
the archive has a narrow addressable window and low measured yield outside it.

## Exploratory hypothesis this suggested — not a landed method

One possible planner would not tell the prior *where* to step; it would measure candidates
and arrange the approach so a candidate meets a fixed obstacle:

1. approach the obstacle to a staging point ≈ 1.2 m (≈ 1.36 s of walking) before it;
2. generate a fresh clip from that staging point under the STEP prompt;
3. measure whole-body box clearance at the obstacle; accept or resample;
4. use a prospectively frozen budget; the corrected post-hoc plug-in needs N=12/13 for 90%
   at 5/8 cm and N=15/16 for 95%, not 7–8 calls.

This is a hypothesis for EXP-022B, after EXP-022A and EXP-023 determine tracking retention
and the native timing lever. It is *not* the same as translation-equivariance, which is
separately measured false: the route offset must come from re-planning the approach, not
from shifting an existing clip's route.

## Scope

Kinematic stage, one prompt, one straight route at the reference speed, one box depth, no
tracker. The provisional early window is specific to this prior and configuration; prompt
onset and history policy are untested in this archive. The staging
approach assumes the corridor allows a 1.2 m run-up, which multi-obstacle scenes will not
always give; the honest generalization of step 1 above is receding-horizon replanning at
each obstacle, where the per-clip rate becomes the cost driver. No
execution claim: 5–8 cm is precisely the amplitude at which tracking previously consumed
most of the margin, so the tracker must enter the endpoint before any of this is called a
traversal.
