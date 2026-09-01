# EXP-021: the prior emits the requested behavior once, early — and that is placeable

`outputs/exp021_elicited_lift_distribution_v2/`, schema
`exp021-elicited-lift-distribution-v1`, status `complete`, **64/64 samples**. STEP prompt,
route conditioning only, the reference-speed 7.2 m straight route, fresh seeds 4400–4463.
Kinematic stage only.

This measures what E1a's controls left standing: if nothing places the behavior on
command, where does the prior put it on its own, and what does it cost to select?

## The prior's spontaneous step-over

* **Elicitation 0.77** (49/64 clips contain a liftable clearance).
* **Amplitude is not the problem.** Lift heights: median 0.079 m, q75 0.184 m, q90
  0.251 m, **max 0.400 m**. Clips clearing a box *somewhere*: 44/64 at 3 cm, 37 at 5 cm,
  31 at 8 cm, 24 at 12 cm, 14 at 20 cm, 5 at 30 cm. The prior can produce a full
  step-over; E1a's 0.06–0.07 m was the packet suppressing it, not a ceiling.
* **Timing is the problem, and it is sharp.** Converting each lift to its clip frame:
  median **frame 34** (1.36 s), q10–q90 **21–54**, and **80 % of all lifts fall in the
  first 50 frames** of a 200-frame clip. Restricting to lifts ≥ 0.08 m does not move it
  (median frame 33, q10–q90 22–50). Only 3 of 49 lifts occur after frame 60.

**The prior expresses the prompt once, shortly after its context is established, then
returns to walking.** That single fact explains the whole E1 family: the event's *when* is
set by the rollout, not by the conditioning, so no channel that specifies *where* can move
it. exp015's "text elicits but places randomly" was never random — it was early, and the
obstacles were elsewhere.

## The addressable region, and what selection costs

Per-clip hit rate against a scene-specified obstacle (peak within 0.25 m of the target):

| target on route | 3 cm | 5 cm | 8 cm | 12 cm | 20 cm |
|---:|---:|---:|---:|---:|---:|
| 0.6 m | 0.078 | 0.078 | 0.062 | 0.062 | 0.031 |
| **1.0 m** | **0.312** | **0.281** | **0.250** | 0.203 | 0.125 |
| **1.4 m** | 0.297 | 0.250 | 0.203 | 0.141 | 0.109 |
| 1.8 m | 0.141 | 0.094 | 0.062 | 0.062 | 0.031 |
| 2.2 m | 0.047 | 0.047 | 0.016 | 0 | 0 |
| 3.0 m | 0.016 | 0.016 | 0.016 | 0 | 0 |
| ≥ 3.4 m | **0** | **0** | **0** | 0 | 0 |

Optimizing the target position gives the operating point and its selection budget:

| box height | best target | per-clip rate | N for 90 % |
|---:|---:|---:|---:|
| 3 cm | 1.2 m | 0.344 | **6** |
| 5 cm | 1.2 m | 0.312 | **7** |
| 8 cm | 1.2 m | 0.266 | **8** |
| 12 cm | 1.1 m | 0.219 | 10 |
| 20 cm | 1.2 m | 0.141 | 16 |

Averaged over obstacle positions spread across the whole route, the same rates fall to
0.04–0.10 and the budget rises to N ≈ 16–32 — which is what the campaign's headline table
reports, and why it understates the achievable method. **The prior is not uniformly
unaddressable; it has a narrow addressable window and is nearly useless outside it.**

## The method this implies

Not a controller. The planner does not tell the prior *where* to step; it measures where
the prior steps and arranges the scene around it:

1. approach the obstacle to a staging point ≈ 1.2 m (≈ 1.36 s of walking) before it;
2. generate a fresh clip from that staging point under the STEP prompt;
3. measure whole-body box clearance at the obstacle; accept or resample;
4. expect ≈ 7–8 generator calls for 90 % at a 5–8 cm obstacle.

This is the project's "measure, don't command" thesis in its strongest available form, and
it is cheap relative to E1a's packet, whose per-clip rate at the obstacle was 0/16
(Wilson95 upper 0.19). It is *not* the same as translation-equivariance, which is
separately measured false: the route offset must come from re-planning the approach, not
from shifting an existing clip's route.

## Scope

Kinematic stage, one prompt, one straight route at the reference speed, one box depth, no
tracker. The frame-34 window is a property of this prior with these settings — the
history/horizon configuration is the obvious confound and is untested here. The staging
approach assumes the corridor allows a 1.2 m run-up, which multi-obstacle scenes will not
always give; the honest generalization of step 1 above is receding-horizon replanning at
each obstacle, which is exactly where the per-clip rate becomes the cost driver. No
execution claim: 5–8 cm is precisely the amplitude at which tracking previously consumed
most of the margin, so the tracker must enter the endpoint before any of this is called a
traversal.
