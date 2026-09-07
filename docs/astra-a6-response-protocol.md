# A6: local native duck execution response

**Status: preregistered**. Development collection only, 2026-09-06. Commit before launch.

Question: do independent native depth, onset and recovery edits produce useful achieved
clearance/progress responses? Frozen ARDY-G1 Horizon52, v2 noise, WALK prompt, 208 frames,
25 Hz, 0.9 m/s straight route, five diffusion steps, CFG [2,2], frozen SONIC release at
the patched A1 tracker identity, default encoder weights [1,1,1]. No qpos editing.

## Assignments and budget

12 scene configurations in nested order: beam x=(1.2,1.6,2.0) m, underside=(1.10,1.20) m,
route length=(0.24,0.60) m. Beam width 2.25 m, thickness 0.25 m. These are **development**
scenes, not confirmation or broad geometry-OOD. Four distinct motion seeds per scene:
64000+4*scene_index+unit_index, 64000–64047 inclusive. All descendants stay in their
scene/unit development group. Physics seed 0, one preview per generated reference.

Theta=(depth, onset lead before front edge, recovery offset after rear edge), metres.
Baseline=(0.28,1.08,1.38). Perturbations: ±(0.04,0.18,0.18), one axis at a time.
Box bounds=[0.24,0.32] × [0.90,1.26] × [1.20,1.56]. Fixed quintic onset/recovery ramps
0.45/0.90 m. On the historical x=1.2,length=0.24 beam, the baseline is exactly fixed3.
Negative start coordinates mean the duck has already begun at the route origin; preserve
this request and measure its execution, without an unreported clamp or trajectory warp.

Each unit has one B=8 generation call in order baseline, depth−/+, onset−/+, recovery−/+,
free_nominal. All eight rows use the unit's same v2 latent noise seed. Free has no native
height conditioning; root path and heading are identical. Bind exact plans and batch order.
Per scene:32 generated references and one32-slot absent launch in the same order. Total:
**336 response candidates +48 free controls =384 generations and384 absent previews**.
Zero obstacle-present evaluations, held-out seed consumption, mode sweeps or stepping.
Free controls are apparatus/development diagnostics; retain failures and exclude no units.
This is response identification, not a budget-matched method comparison.

## Measurements and development decisions

Score valid achieved samples at (i+1)*dt; reference i*dt. Preserve disconnected whole-body
footprint-overlap intervals. Final forward rear-body exit must be observed; require an
entire subsequent one-second interval for continuation. Non-arrival, exit censoring and
recovery censoring are explicit statuses, never favorable numerical clearance targets.
Record signed nominal and 40 mm inflated geometry separately. Apply the existing margin
once: inflated-clearance feasibility threshold is zero. Check geometry, uprightness and
corridor over approach through continuation. Simulator contact remains null for absent
previews and cannot be inferred from geometry.

An explicitly **development-only descriptor** requires inflated nonpenetration, observed
exit by rear_edge/0.9+2.5 s, one-second continuation >=0.45 m (half commanded task speed),
and existing 1.2 m corridor, pelvis>=0.5 m, up_z>=0.70 throughout that prefix. This does
not modify A5, freeze the held-out endpoint, or certify contact/quality. Record native
joint velocity/acceleration diagnostics. Supported-foot slip awaits actual support signals;
near-floor kinematics must not be relabelled as supported slip or effort.

Fit regularized paired local linear differences only for complete observations, retaining
all missing/failed units in accounting. Three independent axes are required. Inspect
response directions and proposed corrections, not just R². Residual allowances require
grouped development validation; no hard-coded allowance becomes a safety guarantee.
The proposed grid optimizer produces one command under clearance, deadline, continuation
and bounded-edit constraints. It is not yet a validated policy. A separate preregistration
must bind the response transfer rule and exact-final-candidate preview before corrections
are evaluated. Development revisions may reuse these development units with versioned
protocols and fresh outputs; no historical gate is loosened or confirmation seed reused.

## Resources, provenance, resume and stops

Use the existing measured ARDY gate4096 MiB VRAM/8192 MiB RAM and the validated compact
32-slot SONIC profile: admission5500/9216 MiB, runtime floors1200/2500 MiB, timeout900 s.
One ASTRA lock. Do not touch co-tenants, change physics or batch size to fit resources,
or silently retry failed launches. No background poller; admission refusal stops the
invocation and leaves unlaunched assignments pending.

Clean immutable execution checkout; external fresh output. Bind protocol, project,
generator revision/denoiser/runtime, robot XML, tracker source/checkpoint, prompt cache,
resource-profile evidence, command arrays, ordered seeds and launched/returned counts.
Persist every generation start before sampling and every batch array before analysis.
Completed scenes may be resumed only after hash verification; any incomplete spent
scene or failed simulator launch stops for diagnosis. Apparatus and runtime default
mode validation are required before proceeding. A defect requires a versioned successor
beside preserved attempts. Scientific failure/censoring stays in the assigned denominator.

Run from the pinned clean checkout after tests:

```bash
source env.sh
$S2M_PY -m experiments.astra_a6_response --out /absolute/external/a6 --stage generate
$S2M_PY -m experiments.astra_a6_response --out /absolute/external/a6 --stage prepare
$S2M_PY -m experiments.astra_a6_response --out /absolute/external/a6 --stage execute
$S2M_PY -m experiments.astra_a6_response --out /absolute/external/a6 --stage analyze
```

`--scenes 0` restricts an invocation to the first preregistered scene; the total allocation
and unexecuted assignments remain unchanged. Stop expansion if apparatus/provenance is
invalid; scientific response failures are measured outcomes, not harness failures.
