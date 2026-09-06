# Does fixing the controller encoder improve local beam passage?

## Result and decision

**G1-only did not outperform the default sampler on the primary three-second duck.**
Present local passage was 17/32 for both, versus 15/32 for teleop-only. G1 gained two
matched passes and lost none against teleop; against default it gained one and lost one.
Only two of eight motion groups improve over teleop, each by one of four repeats.
This is a small, one-geometry development result, not a significant improvement or a
new repair method. Keep default fixed3 as the established comparator; do not expand
mode sweeps or train a mode selector from these present outcomes.

All **24 fresh references and 576 assigned executions** are retained in the
[receipt](../outputs/astra_a5_encoder_d0/summary.json). The
[protocol](astra-a5-encoder-protocol.md) and implementation were committed at `df69290`
before generation. Seeds 63000–63007 are now spent. All 18 launches completed without
retry, timeout, resource abort, mode mismatch or apparatus failure. No training occurred.

## Matched results

Each row has eight source-motion groups with four slot repeats, one physics seed and
one reused 1.15 m beam. Absent progress uses the historical raised virtual beam;
present success uses the target beam and unchanged whole-body recovery endpoint.

| Reference | Mode | Absent progress /32 | Present local passage /32 | Present units passing at least 3/4 repeats /8 |
| --- | --- | ---: | ---: | ---: |
| Free walk | Default | 10 | 0 | 0 |
| Free walk | G1 | 10 | 0 | 0 |
| Free walk | Teleop | 10 | 0 | 0 |
| Three-second duck | Default | 18 | 17 | 4 |
| Three-second duck | G1 | 18 | 17 | 4 |
| Three-second duck | Teleop | 14 | 15 | 3 |
| Two-second duck | Default | 13 | 14 | 3 |
| Two-second duck | G1 | 14 | 15 | 3 |
| Two-second duck | Teleop | 12 | 13 | 3 |

All full-route completion counts are zero. The primary G1-minus-teleop per-unit
success-fraction differences, a5u0 through a5u7, are
`[0.25, 0, 0, 0, 0.25, 0, 0, 0]`: mean +6.25 percentage points. No paired-difference
confidence interval or significance claim is made at n=8. The receipt includes the
preregistered Wilson intervals; slot intervals are descriptive, not independent-scene
population intervals. Unit-level intervals are broad: G1/default 4/8, 21.5–78.5%;
teleop 3/8, 13.7–69.4%. No geometry generalization is established.

The free-walk reachability gate passed in each mode, with a5u2 and a5u4 passing all
four absent repeats. This permitted the entire present stage; no other unit was removed.
An absent/present count increase is not proof that the obstacle helps: the conditions
produce different dynamics, and geometry alone does not identify the physical cause.

## What the intervention measured

Runtime checks covered **199,966 valid action samples**. Explicit arms used their
requested one-hot mode throughout, matching live command and contiguous command time.
Default used measured G1/teleop masks. All encoder keys, modules and checkpoint stayed
fixed; only sampling weights changed. “Teleop” is an encoder branch, not a human operator.
Mode-dependent tensor shapes are part of the tested configuration, so this does not
isolate neural representation from numerical batch effects.

Equal endpoint counts do not mean identical trajectories. G1 and default have one
paired win and one loss on fixed3. As a **post hoc diagnostic**, every fixed3 present
failure lacks the whole-body dwell by 4 s: 15 default, 15 G1, 17 teleop. Of these,
9/7/10 respectively have no archived state at 4 s. Do not impute their last positions
or infer a unique failure cause; collision and termination can precede missing recovery.
The mode intervention has not removed the progress/recovery bottleneck.

Actual contact, torque and high-quality demonstration qualification remain **not
measured**. Local margin-preserving passage is not a contact-free safety certificate.
All variants share `astra-a5/a5uN` split groups and remain development data. These
execution-labelled records do not complete the dataset's quality, OOD or utility gates.

## Resources and reproducibility

Generation and simulation ran in separate workers from clean clone
`/tmp/astra-a5-TH5KkJ/repo`, tracker `7c63c53`, frozen release checkpoint. All arrays were
frozen before execution. Nine absent batches preceded the saved reachability receipt;
nine present batches followed in the preregistered order. Compact allocation and
32 environments were retained, with admission 9216/5500 MiB RAM/VRAM and runtime floors
2500/1200 MiB. No co-tenant was stopped.

- Simulator wall time: **1112.65 s (18.54 minutes)** across 18 launches.
- Maximum simulator child RSS: **6104.58 MiB (5.96 GiB)**; supervisor 16.88 MiB.
- Minimum observed available RAM/free VRAM: **8825/3232 MiB**.
- Generation/decode timers sum to 1.89 s; these exclude model loading, geometry scoring,
  preparation, verification and analysis, and are not end-to-end throughput.

These are observed costs, not a new memory-equivalence or speedup experiment. For this
commit, preparation resolves resource-profile evidence paths relative to the working
directory: both preparation commands ran in
`/home/linjiw/scene2motion/outputs/astra_resource_probe_v2` after sourcing the clean
clone's `env.sh`. Every evidence hash was checked there. Generation and queue execution
ran from the clean clone. Receipts, raw archives and ignored local logs are preserved.

Rebuild without launching or regenerating:

```bash
source env.sh
LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a5 \
  --out /home/linjiw/scene2motion/outputs/astra_a5_encoder_d0
LD_LIBRARY_PATH= "$S2M_PY" -m pytest tests -q
```

Regression: **1071 passed in 227.05 s**, including 18 new assignment, mode, gating and
analysis tests. A second full receipt rebuild matches exactly. No ASTRA worker remains.

## Next work — proposed, not a new preregistration

1. **Close final-motion qualification.** Audit A5's achieved progress, stance/floor
   proxies and bounded joint dynamics using the existing aligned tools. Qualify the
   final edited motion, not free WALK or a parent clip. Preserve censored suffixes.
2. **Validate contact/quality instrumentation on operational controls.** Use known-contact
   and no-obstacle probes, then paired read-only replay equivalence. Do not label current
   local passes as high-quality positives or silently add a new endpoint to A5.
3. **Develop one progress-aware correction on development data.** Keep fixed2/fixed3
   controls and a fixed, logged controller mode. Fit only from permitted absent execution;
   require final-output progress/recovery as well as clearance. Extending overlap alone
   already lost in A4. If correction cannot preserve progress, refuse rather than move
   the deadline. Freeze a fresh grouped comparison before spending new scientific seeds.
4. **Then test scene transfer and dataset utility.** Freeze geometry-disjoint groups,
   add qualified execution positives and failures, and evaluate a pre-execution consumer.
   No mode selector, large policy training or comprehensive dataset claim is justified yet.
