# A13 completed after two resource refusals

The campaign subsequently completed64 exact observer replays and54 pure policy
calls. See the [final result](astra-a13-result-2026-09-06.md) and independent
audit in `outputs/astra_a13_completion_v1`. The preparation/resumption notes
below are historical; do not relaunch the completed campaign.

Latest continuation: a second admission attempt refused at RAM7796MiB and
VRAM3868MiB, still zero launches. A new [archive coverage diagnostic](astra-a13-trace-coverage-2026-09-06.md)
independently reconstructs all39 unchanged-target pairs:15 of29 divergences occur
inside the frozen eight-step window,14 later. Current A13 remains unchanged;
subsequent fix validation must cover the late cases and both encoder modes.

The next research stage is implemented and preregistered at `ad569b5`. It addresses
A12's unresolved comparison error: an identical target reference can execute
differently when other motion rows change. Until this is resolved, another motion
correction comparison could confound the edit with its simulation context.

The observer records policy inputs, encoder tokens, raw/applied actions and robot
states for warm-up plus eight physics steps. The campaign requires each observed
32-slot execution to reproduce its original A12 archive exactly. Only after both
contexts pass does the policy intervention test a fixed target row against changed
background inputs. Whole-input replay must first reproduce all live actions exactly.
See the [preregistered protocol](astra-a13-batch-observer-protocol.md).

## Validation and first admission

- Full CPU suite:1267 passed in207.47s. One additional observer regression was added
  after that run's test collection; the latest focused suite has11 passes.
- Focused tests cover forbidden job mutations, exact rather than tolerant numeric
  equality, no extra live inference calls, copied states without tensor aliasing,
  independent/coupled policy probes, and refusal of hybrids when controls fail.
- Preparation succeeded from clean `/tmp/s2m-a13-ad569b5`, preserving original
  source pins and binding the new observer/protocol plus runtime policy modules.
- The first host admission refused before launch: RAM8252MiB versus9216 required;
  VRAM4327MiB versus5500 required. The refusal is preserved with the prepared jobs.
- Assigned budget:64 operational executions, zero new generations/independent
  units, at most54 post-simulation policy forwards. Actual A13 executions:0.

Artifacts: `outputs/astra_a13_batch_observer_v1`; validation receipt:
`outputs/astra_a13_validation_v1/summary.json`. No policy mechanism, apparatus fix,
performance gain, contact result, or held-out result is claimed from this stage.

## Resume the frozen campaign

Run from `/tmp/s2m-a13-ad569b5` when host resources meet the unchanged gates:

```bash
source env.sh
env LD_LIBRARY_PATH= OPENBLAS_NUM_THREADS=1 "$S2M_PY" experiments/astra_a13_batch_observer.py \
  --stage execute --out /home/linjiw/scene2motion/outputs/astra_a13_batch_observer_v1
```

The first attempt spent no seeds. Resume preserves the refusal and the same job;
after any actual launch, failures stop the campaign without automatic retry.
The second job is bound to the first verified trace before it can launch. No
background poller is active, and no other workload was interrupted.

Next, interpret the first differing field only if observer preservation passes,
then implement and separately validate the smallest justified apparatus change.
Keep A12's failed method comparison and the broader plan: constructive duck gain,
controlled G1 contact sensitivity, held-out beam and multi-beam tests, stepping,
and downstream utility. This diagnostic supplies no cross-family success claim.
