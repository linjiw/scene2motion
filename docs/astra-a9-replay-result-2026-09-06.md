# A9: both same-slot jobs reproduce exactly

The [preregistered A9 probe](astra-a9-same-slot-replay-protocol.md), committed at
`95a0731`, completed64 operational replays. **All64 achieved-state arrays, valid
lengths and termination labels match their original A6 executions exactly.**
Default encoder signatures, apparatus and sample clocks also match. There are zero
observation-status or development-descriptor disagreements. This closes the exact
replay check for these two jobs, not the constructive-method or contact-sensitivity gates.

| Replay arm | Assigned pairs | Exact qpos pairs | Observed clearance/exit/continuation pairs |
|---|---:|---:|---:|
| Native baseline | 32 | 32 | 26 / 18 / 16 |
| Free-WALK | 32 | 32 | 23 / 18 / 18 |

Every available numeric difference is zero. Unobserved outcomes stay unobserved;
they are not counted as zero-error numeric pairs. These are original development
scenes a6s00–a6s07, seeds64000–64031, physics seed0,32 matched slots, default mode,
with no new reference generation, independent units or obstacle-present evaluation.
Repeating a failure exactly is repeatability evidence, not traversal success.

Evidence: [complete result](../outputs/astra_a9_same_slot_replay_v1/summary.json)
(SHA256 `f08ff8b21e73351a42454175b920e2c2d9990299e9d963ef0ad20dc4284f714a`),
[independent array verification](../outputs/astra_a9_same_slot_replay_v1/independent_verification.json),
and per-arm process, runtime-mode, apparatus and comparison receipts in the same
directory. The independent check reloaded both old/new archives and compared all64
arrays, lengths and termination labels, in addition to the driver's measurement checks.

Two simulator processes completed without runtime resource abort:42.454 s baseline
and44.233 s free-WALK, total86.687 s. Peak child RSS6069.32 MiB. The initial sandbox
invocation preserved a preflight refusal:6064 MiB available RAM and unavailable GPU
readback (`nvidia-smi` rc9), with no launch. After authorized host-level execution,
the unchanged admission gate passed at23444 MiB available RAM and15105 MiB free VRAM
for the baseline job. No resource threshold, simulator setting or input was changed.
There was one initial refusal, two successful launches, no failed simulator attempt,
and no poller. The execution driver and its verification children have exited.

The clean preparation checkout `/tmp/s2m-a9-95a0731` and original simulator-source
checkout `/tmp/s2m-a6-paired-d9ac674` must remain unchanged. New jobs inherited the
original simulation contract and only changed the evaluation output directory;
the new protocol/verifier provenance was added separately. Prepared-job checks also
reject altered runtime arguments, environment and original source bindings.

## Research implication and next work

The [different-slot diagnostic](astra-a8-response-result-2026-09-06.md) observed
substantial variation for duplicate references even with matching encoder masks.
A9 shows that variation is absent in the tested same-slot baseline/free replays.
It does not prove determinism for every edited command, hardware state or simulator
configuration. A8's failure remains a measured failure of its response model; these
replays do not turn it into an optimizer gain or identify the cause of its errors.

The next constructive-development task is a separately preregistered **smaller-step
native response experiment**, retaining no-edit regeneration/replay controls and
original-scale controls. Half-size coordinates are depth±0.02 m and onset/recovery
±0.09 m around the existing baseline, with smooth native conditioning and frozen
generator/controller. Preserve generator batch shape/order, noise stream, encoder
slots and partial outcomes. Compare response magnitude, midpoint departure, status
transitions and scene-held-out prediction against no-change/constant/scene controls
before fitting a bounded inverse. Compound-command regeneration and final-preview
qualification still require their own justified pilot. Controlled positive G1 contact
sensitivity remains open. Untouched confirmation data remain untouched.

Validation:1236 CPU tests passed in208.03 s. Focused28 tests passed in1.48 s, including
three replay-contract guards added after full-suite collection. A8 and its midpoint
diagnostic rebuild exactly; all figure/source hashes match; the new different-slot
analysis also rebuilds exactly. All existing unrelated probe/report changes and
pending A8 source artifacts were preserved.
