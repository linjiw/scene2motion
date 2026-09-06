# Which controller interface did the replay use?

## Measured result

The read-only callback preserved **64/64 archived achieved trajectories exactly** in
two Isaac/SONIC operational replays of A4's fixed-three-second references. In **each
32-slot batch, 14 slots used the G1 encoder and 18 used the teleoperation encoder**.
Here “teleoperation” names a pretrained encoder branch; no human operated these runs.
All paired absent/present slot assignments agree, and no encoder mask changes within
any valid execution prefix. This resolves runtime mode uncertainty for these replays;
it does not establish that either mode caused A4's recovery deficit or performs better.

The [protocol](astra-controller-readback-protocol.md) was committed at `c7c864f` before
launch. Execution used the clean `/tmp/astra-mode-wJmzDl/repo` clone, tracker `7c63c53`,
unchanged frozen checkpoint, physics seed 0, original pickle/order and 32 isolated slots.
Eight archived motion carriers × four slots × two apparatus conditions = 64 operational
executions, **zero additional independent scientific samples**. No references were generated,
no controller weights changed, and historical A4 outcomes were not relabelled.

## Claim–evidence map

| Claim | Measurement and denominator | Evidence | Boundary |
| --- | --- | --- | --- |
| Logging preserved execution | Exact valid qpos, lengths, termination, timestep and apparatus in 64/64 replays | Per-batch `verification.json` | These two fixed3 batches only; not every callback/workload |
| Runtime interface was mixed | G1 14/32, teleop 18/32 in each batch; zero transitions over 26,432 valid action samples total | Per-batch `eval/controller_modes.json` | Frames/slots are not independent scenes; other A4 arms were not instrumented |
| Readback was aligned | Policy-observation and pre-step command masks agree everywhere; command steps contiguous from zero | Same readback and summary | This is command information, not contact sensing or motion-quality validation |
| Probe fit the compact profile | 114.70 s simulator wall time total; maximum child peak RSS 6067.31 MiB | `process.json`, `resource_samples.json` | Two launches, no speedup claim; available RAM/VRAM include co-tenants |

The [summary](../outputs/astra_controller_readback_v1/summary.json) binds source and
measurement hashes. Minimum observed available RAM was 11,786 MiB and free VRAM
4711 MiB; supervisor peak RSS was 16.63 MiB. Both launches completed without resource
abort or retry. Admission remained 9216/5500 MiB and running floors 2500/1200 MiB.

## Instrument and interpretation

`AstraControllerExportCallback` reads the flattened tokenizer observation already
consumed by the policy before stepping physics. Runtime term dimensions locate
`encoder_index`; the command's ordered names decode the mask. The logger supports
multi-hot masks and stores an independent live-command mask. It makes no random calls,
changes no commands, and trims exactly the same valid prefix as achieved qpos.
This avoids assigning an automatic-reset mode to the preceding action.

Resolved weights were `[1, 1, 1]` for `[g1, teleop, smpl]`, but no SMPL-active mask was
observed. Configured weights are not observed frequencies. In particular, G1-specific
input analysis alone does not describe all executions of the observed mixed interface.
The result does **not** invalidate A4's default-stack counts: paired replay modes match,
and states reproduce exactly. It also does not retrospectively turn missing historical
mode fields into measured labels. Keep the replay linked as an operational variant of
the original source group, never as another training/test-independent example.

## Decision and next test — proposed, not preregistered

Keep the logger for new campaigns. Before another timing-repair comparison, isolate
the encoder choice with a matched explicit-G1 / explicit-teleop / default-sampler
comparison on fresh carrier groups. Retain the checkpoint, reference arrays, scene,
clock, evaluator and resource profile; verify the commanded mode actually reaches every
policy step. A mode override is a new experimental condition, not a silent A4 fix.

The falsifiable question is whether fixing the encoder changes executed progress,
clearance and recovery for the same reference. Paired mode effects would support
mode-conditioned carrier qualification; comparable failures would redirect work toward
the reference/progress interface. Neither outcome licenses a new repair-success claim.
Keep fixed2/fixed3 comparators and qualify the **final edited reference**, not its parent.
Freeze fresh seeds, launch budget and analysis unit before that test; keep development
and held-out groups separate. Do not choose a mode from post hoc replay winner counts.

Contact instrumentation, achieved-motion quality and geometry-disjoint testing remain
required before releasing a high-quality positive demonstration bank. This milestone
adds a measured controller-interface field and an equivalence-tested instrument, not a
new navigation method or a completed dataset contribution.

## Rebuild

```bash
source env.sh
LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_controller_readback \
  --out outputs/astra_controller_readback_v1
LD_LIBRARY_PATH= "$S2M_PY" -m pytest tests/test_astra_controller_export.py \
  tests/test_astra_controller_readback.py tests/test_astra_controller_summary.py -q
```

The rebuild verifies existing receipts; it never launches the simulator. The 19 focused
tests cover observation slicing, multi-hot masks, malformed input, pre-reset capture,
valid-prefix trimming, command-only mutation, mode-record alignment and summary integrity.
Full CPU regression: **1051 passed in 220.16 s**; the two subsequently added summary
tests pass within the 19-test focused run. Summary rebuild is exact. No ASTRA worker remains.
