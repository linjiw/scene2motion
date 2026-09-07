# A13 completes: first observed divergence is downstream of policy output

Both instrumented32-slot runs exactly reproduce their original A12 trajectories,
lengths, termination/progress records and clocks:64/64. Apparatus and encoder
checks pass. All18 captured whole-input cases reproduce their live actions and
exact repeats, using36 pure forwards. All18 additional fixed-target/background
substitutions preserve the target action exactly. Total54 pure forwards.

The observer is therefore admissible for this diagnostic. For the selected
group0/slot19 G1 target, the first measured difference appears after the second
physics step, before its observations or actions differ.

| Event | Zero-based policy step | Time at boundary |
| --- | ---: | ---: |
| Warm-up, inputs/actions and initial state exact | -1 | 0s, no physics |
| First post-step state exact | 0 | 0.02s |
| Pre-state and applied actions still exact; post-state differs | 1 | 0.04s after step |
| First observed input difference | 2 | 0.04s before step |
| First raw/applied action difference | 3 | 0.06s before step |
| First assembled token difference | 5 | 0.10s before step |

The first joint-position difference is5.960464477539063e-8rad; the first maximum
joint-velocity difference is3.039836883544922e-6rad/s. At step5, assembled token
difference reaches0.0625 and maximum action difference reaches0.05045078694820404.
This sequence is consistent with closed-loop amplification following a small
execution difference. It is not a controlled proof that quantization causes the
long-horizon0.302m separation, nor a diagnosis of a specific PhysX kernel.

The state trace omits hidden contact/solver caches and intermediate drive targets.
Equal observed generalized state and policy output do not prove equal complete
simulator state. The constrained conclusion is that the first observed onset is
inside the environment step, downstream of the matching policy output. The
controlled pure-input intervention also finds no target policy dependence on
other captured input rows in these18 tests.

A subsequent post hoc read of all25 unchanged group0 targets finds15 early
post-state divergences. In all15, the observed pre-state, full input row, raw
action and applied action match at onset; all first appear after step1. The
remaining10 have no state divergence in the captured window. This supports the
same measured ordering beyond the single stress case, but the controlled input
substitution experiment still covers slot19 only. Receipt:
`outputs/astra_a13_completion_v1/early_cohort_diagnostic.json`.

## Execution and evidence

- Frozen observer checkout: `/tmp/s2m-a13-ad569b5`.
- Campaign: `outputs/astra_a13_batch_observer_v1/summary.json` and per-arm
  `repeat_comparison.json`, `verification.json`, `eval/policy_trace.{json,npz}`.
- Simulator job time:51.3556s +50.2616s =101.6172s.
- Peak child RSS6073.11MiB; minimum available RAM7918MiB across both jobs.
  Both jobs completed without timeout or resource abort.
- Preserve the two earlier admission refusals. No new generations or independent
  units; no present-obstacle, contact, or motion-performance claim.
- Independent archive and trace reconstruction passes:
  `outputs/astra_a13_completion_v1/summary.json`.
- The [39-pair coverage diagnostic](astra-a13-trace-coverage-2026-09-06.md) remains
  applicable: fourteen of29 onsets are outside this early trace. The stress-case
  mechanism localization must not be generalized to those unobserved windows.

## Constructive next step

Control the incoming comparison context explicitly: put one candidate into its
original slot while holding all31 background references fixed. This prevents
unrelated incoming candidates from changing the workload used for that target.
It retains the current GPU simulator and frozen policy, at the cost of only one
useful candidate per32-env job. It does not promise independence from changing
the declared canonical background itself.

The [A14 protocol](astra-a14-canonical-context-protocol.md) separately preregisters
two actual edited targets, four jobs/128 operational executions, exact whole-prefix
repeats, and byte-level construction checks for all39 unchanged source targets.
It supplies a controlled evaluation adapter before returning to motion correction.

NVIDIA documents limited scene determinism and marks its enhanced-determinism
flag as unsupported on GPU in both the [IsaacLab-linked5.4.1 API](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxSceneFlag.html)
and [5.6.1 API](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/_api_build/structPxSceneFlag.html).
These documents motivate controlling context; they do not establish the cause
of this particular observed divergence.
