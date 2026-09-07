# A19: fresh-scene native depth–forward-timing candidate coverage

**Status: preregistered**

Assigned before any A19 generation or execution. This is development discovery,
not a two-preview compiler evaluation. A18 stopped at its force-sensitivity gate;
A19 uses no physical obstacle and does not depend on force calibration.

## Assignments and frozen controls

Eight scenes are the ordered Cartesian product of center x=(1.35,1.85)m,
underside=(1.12,1.18)m and beam length=(0.32,0.48)m. Width2.25m,
thickness0.25m, straight route, corridor half-width1.2m. These geometry tuples are
absent from A6's12 reused scenes. Two seeds per scene:81000+2*scene_index+unit_index.
Seeds are carriers, not independent scenes. Physics seed0, default encoder,
208 reference frames at25Hz; same released checkpoints, five diffusion steps,
CFG(2,2), noise streamv2, cached historical prompt, and no postprocessing.

Each unit receives exact historical nominal DuckCommand(.28,1.08,1.38) and
shallow DuckCommand(.26,1.08,1.38), generated through unchanged make_spec.
New commands form depth=(.26,.28,.30)m × beta=(-.24,0,.24)s. A new-grid
center is not assumed equivalent to either historical comparator.

The committed joint_duck_timing module defines a smooth spatial height window
and monotone time law. Achieved nominal first body-envelope overlap and last
rear-envelope exit initialize descent/recovery on the nominal native command
clock. Missing/out-of-support events invoke its explicit geometry fallback.
The window is frozen for all nine commands of that unit. A beta perturbation
changes native forward timing and the height values sampled along that motion;
route endpoints, observation horizon, heading, prompt and spatial path remain
fixed. No generated joint trajectory is subsequently rescaled.

## Budgets and simulator context

176 assigned scientific absent previews:16×(9+2). Maximum162 separate32-env
jobs =5184 operational rollouts. The first job evaluates all16 new nominal
references plus16 fixed nominal duplicates; an identical second job must replay
all32 qpos traces, termination flags, timestep and encoder signatures exactly.
The remaining160 jobs change one target slot in that frozen nominal batch:
16 shallow plus144 joint candidates. Other slots are context workload, not
independent measurements. Budget decomposition:176 scientific previews,
4976 background rollouts and32 replay-control rollouts. No whole-context cache
reuse is planned; identities are reported, never inferred from parameter labels.

Comparator generation: four batches of8 (four units×two commands each).
Joint generation: per unit one batch of8 joint candidates, then one batch with
the ninth plus seven nominal regeneration controls. Maximum288 generated
outputs:176 scientific references and112 exact regeneration controls. All seven
controls must equal their original nominal qpos. Incomplete/spent batches never
retry. Record actual native arrays, seed, request digest, qpos and decoded
reference identity before preview. ARDY revision/runtime/checkpoints match A6.

The simulator runtime stays /tmp/s2m-a15-88fcc45; only prepared input/output
paths change from its verified g0 job contract. Fresh preparation checkout must
be clean and committed. Canonical batch compilation changes one target while
preserving every non-target field and metadata value. Validate all32 apparatus
records, slot membership, encoder masks and action/time alignment after each job.
Target encoder masks must equal nominal; do not compare unedited background
states across edited contexts as if global batch independence were established.

Compact gates unchanged: RAM9216MiB/VRAM5500MiB admission, runtime floors
2500/1200MiB,900s/job, exclusive campaign lock. ARDY gates8192/4096MiB.
No persistent numerical/model worker during Isaac execution. Resource refusal
spends no job and stops dispatch; no resident poller or co-tenant interruption.
Any failed apparatus, identity, replay or generation-control gate stops dispatch.
Completed prefixes may resume only after exact validation, never relaunching a
spent job. Stop-file support at each completed-job boundary.

## Endpoints and prospective engineering decision

Unchanged scene_relative_duck.measure_motion endpoint: conservative whole-body
clearance, observed rear exit within beam.rear/.9+2.5s, complete one-second suffix
with >=.45m continuation, corridor, uprightness, and no prefix hazard. Preserve
all nominal/inflated geometry, exit/censoring, continuation, termination and costs.
Reference geometry, obstacle-absent executed qualification and obstacle-present
passage remain distinct. No missing continuation is imputed as a successful suffix.

For each unit report nine-candidate coverage, beta=0 depth-only coverage,
depth=.28 timing-only coverage, and exact nominal/shallow outcomes. Plot executed
inflated clearance against continuation by scene and carrier with labeled controls;
retain censored points separately. Report distinct requests, qpos references,
compiled contexts and achieved traces, plus normalized finite-difference responses.

The engineering supply gate requires at least TWO independent scene groups each
with a unit where a nonzero-beta joint candidate qualifies while both historical
comparators and every beta=0 new-depth candidate fail. This is a conservative
repeated timing-supply gate, not a significance test or a statement that other
patterns have no value. If it fails, do not fit a predictor or open a present
bridge for this frozen design. Report all response data and diagnose the mechanism.

If it passes, select one qualified joint candidate per assigned unit by maximum
inflated-clearance slack, then maximum continuation, then minimum |beta|, then
minimum |depth-.28|, then candidate key. Abstain if none qualifies. Selection
uses absent data only and is frozen here. A separately preregistered present
capability bridge compares these selections, shallow and fixed3/default with
at most48 target executions. Its nine-candidate discovery advantage is charged
explicitly; it is not evidence for an equal-budget two-preview method. Force
claims remain limited by A18. No predictor, holdout, learner or hardware trial
is assigned by this protocol.
