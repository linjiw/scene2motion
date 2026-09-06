# A4 adaptive timing: implementation ready, generation resource-paused

**Historical snapshot — superseded by completion later on 2026-09-05.** A4 now has
32 references and 256 verified executions. See [completed results](astra-a4-results-2026-09-05.md).
Do not rerun the generation commands below; their seeds are now spent. The original
refusal and zero-execution ledger remain preserved as point-in-time records.

## Completed work

The [preregistered protocol](astra-a4-adaptive-hold-protocol.md), scheduler and two-stage
driver are frozen at **84e649e**. Execution checkout: `/tmp/astra-a4-6VBizH/repo`.
Eight fresh seeds62000–62007,32 planned generations,256 planned executions, including
calibration. The four arms are free nominal, fixed3, fixed2 and one-step adaptive.
No A3 experiment, threshold, seed or receipt was changed.

The scheduler takes only a reference overlap descriptor and four absent-only repeats.
It proposes an edit when >=3 repeats pass local progress and every forward exit is observed;
otherwise it retains fixed3. The update covers both reference and worst observed achieved
exit plus0.10 s, within[2,4] s. New generation invalidates any assumption of guaranteed
tracking: the final absent/present comparisons test that directly. All finite fallback and
failed-reference candidates remain in this simulation mechanism probe's assigned denominator.

The [development dry run](../outputs/astra_adaptive_development_v1/index.json) applies this
frozen rule to32 A3 fixed3 absent records: four of eight carriers qualify, with proposed
holds3.48,2.34,2.56,2.84 s; four retain3 s. This is post-hoc development analysis, **zero
new generations/executions**, and no adaptive performance result. It rebuilds exactly.

## Actual launch state

The initial generation preflight observed7327 MiB available RAM, below its8192 MiB gate.
It stopped **before model loading or seed use**. The final read-only check after CPU tests
still failed:5934 MiB available RAM and4141 MiB free VRAM for generation; the simulation
check also failed its9216/5500 MiB gates. These are host snapshots, not universal limits.

`outputs/astra_a4_adaptive_d0/` preserves the identity, refusal and later resource snapshot.
The [256-assignment progress ledger](../outputs/astra_a4_progress_v1/index.json) has **0
generated references,0 executions,256 not launched**, all outcomes unknown. It rebuilds
exactly. A resource refusal is not a scientific motion failure. No campaign is left running.

## Validation and analysis

Full CPU suite: **1016 passed in213.75 s**. Two subsequent tests were added; the12-test A4
focused suite passes, including those additions. Coverage includes bounded/censored timing,
privileged-field rejection, matched constraints, fresh membership, immutable receipts,
mock two-stage generation, refusal to regenerate spent batches, changed-calibration detection,
all-assigned aggregation and retention of a losing adaptive arm.

`experiments/analyze_astra_a4.py` recomputes calibration from absent archives, scores the
unchanged whole-body endpoint, stores grouped references/executions and unknown contact/quality
labels, checks fallback array equality, and reports standalone fixed3 versus adaptive costs.
Its complete-campaign path refuses partial input. The numerical simulation path remains unrun.

## Exact continuation

Wait for the declared gates; never lower them or change the frozen checkout. From the
checkout above, source `env.sh` and use this stage command, substituting the table entries:

```bash
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.astra_a4_adaptive_hold \
  --out /home/linjiw/scene2motion/outputs/astra_a4_adaptive_d0 --stage generate_initial
```

Run `generate_initial`, then `prepare_initial`. Execute its queue with:

```bash
env LD_LIBRARY_PATH= /usr/bin/python3 -m scene2motion.lean_queue \
  --manifest /home/linjiw/scene2motion/outputs/astra_a4_adaptive_d0/initial_queue.json \
  --wait-seconds 180
```

After both absent jobs verify, run stages `calibrate`, `generate_final`, `prepare_final`,
then the same queue command with `final_queue.json`. Each generator process exits before
Isaac starts. Resume completed simulator jobs by verification, not replay; never repeat a
generation stage once its start receipt exists. Finally, from the main repo, run
`$S2M_PY -m experiments.analyze_astra_a4 --out outputs/astra_a4_adaptive_d0`, then `--check`.

The next scientific decision remains whether this paid, one-step timing update improves on
fixed3 on fresh seeds. A favorable result still needs geometry transfer and achieved-motion
quality/contact validation before any high-quality demonstration claim.
