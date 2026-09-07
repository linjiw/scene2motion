# A7: pooled execution information changes choices but does not improve admission

The next development screen is complete. A pooled model using the initial execution
admits **12/48 outputs**, exactly the same units as its reference-only ablation. The
strong two-preview search control admits11/48; the initial native command admits7/48.
The preregistered expansion gate failed. No new generation, simulation or held-out
confirmation was launched. This is an obstacle-absent archive experiment, not an
obstacle-present passage result.

[Protocol](astra-a7-pooled-response-protocol.md) was committed at `b6b07c9` before the
screen. [Machine-readable result](../outputs/astra_a7_pooled_response_v1/summary.json)
contains all decisions, predictions, training-group manifests and source hashes.
[Post hoc diagnostic](../outputs/astra_a7_diagnosis_v1/summary.json) and
[standalone figure](../outputs/astra_a7_figures_v1/response_transfer.pdf) are separate.

![A7 development comparison](../outputs/astra_a7_figures_v1/response_transfer.png)

## What was tested

Twelve leave-one-scene-out development folds train on44 units and evaluate four.
Every seed and command descendant of an evaluation scene is excluded from its fit.
All336 native records train the binary conjunction model, including failed and
censored outcomes. Independent numeric heads use262 signed clearance,171 exit-time
and166 continuation labels across evaluation folds; no complete-unit requirement
and no missing numeric target converted to zero.48 free controls are reported only.

The fixed ridge model sees scene geometry, the initial reference and, in the full
version, the initial absent preview. It never sees an edited reference or preview
before proposing that command. It preserves a passing initial output; otherwise it
proposes one of six measured native perturbations. Only that command's exact archived
preview can admit the edit. All failures and refusals remain in the48-unit denominator.
The candidate-ranking model does not claim calibrated probabilities or predicted
metric feasibility; exact-preview admission enforces the conjunction.

The development descriptor remains the A6 rule: observed whole-body exit, full
one-second suffix, >=0.45 m continuation, scene-relative deadline, nominal/inflated
geometry, and corridor/uprightness throughout the prefix. A5 is unchanged. Contact
qualification and a prospective held-out motion-quality endpoint remain open.

| Development control | Admitted outputs /48 |
| --- | ---: |
| Initial scene-relative native duck | 7 |
| Shallow native command | 8 |
| Initial plus shallow exact preview | 11 |
| Training-fold-selected two-preview search | 11 |
| Reference-only pooled proposer | 12 |
| Execution-aware pooled proposer | 12 |
| Free WALK | 0 |
| Outcome-informed seven-command oracle | 16 |

The search control independently chooses the shallow second command in every fold.
Full-minus-search paired difference is+2.08 percentage points, with a descriptive
scene-bootstrap95% interval[0,6.25] points. Full-minus-reference is zero in every
scene and on every unit; the resulting empirical[0,0] bootstrap interval describes
these identical labels, not proof of population equivalence. These are reused
development scenes, not untouched tests. Unit Wilson intervals in the receipt are
explicitly labelled as ignoring within-scene grouping.

## Why this does not yet support the correction method

Execution information changes22/48 choices, but changes zero admission labels relative
to the reference-only model. It lowers edited-target binary Brier error from0.10485
to0.10055, versus0.11271 for training-fold prevalence. Better average prediction alone
does not establish better command correction.

The stronger post hoc diagnostic compares numeric predictions with simply reusing
the initial achieved value—predicting no response to an edit. Use identical available
baseline/edited pairs in each comparison:

| Target | Observed edited pairs /288 | Pooled execution model MAE | Predict-no-change MAE |
| --- | ---: | ---: | ---: |
| Inflated clearance | 193 | 66.9 mm | 57.8 mm |
| Rear-body exit time | 110 | 0.648 s | 0.505 s |
| One-second continuation | 97 | 130.0 mm | 125.7 mm |

Thus the current absolute-outcome model does not even beat persistence on these
matched numeric labels. For paired +/- perturbations, its continuation-direction
accuracy is6/16 for depth,8/15 for onset and12/20 for recovery. These conditional
subsets differ and omit many censored pairs; they are not population response rates.
This supports investigating a baseline-anchored paired-change model, but does not
establish that it will work.

Coverage also limits this screen:32/48 units have no passing command among the seven
measured native requests. That says nothing about unmeasured intermediate or compound
commands. It does mean more ranking of these same observations cannot create their
missing successful motions. Among the22 complete baseline observations, seven violate
inflated geometry, seven miss the deadline, six lack continuation and three fail the
combined corridor/upright descriptor; these failure counts overlap. The other26
baseline units have incomplete passage/recovery observations. Clearance alone remains
an incomplete objective.

## Research decision

Keep the frozen native duck direction. Stop expansion of this pooled absolute-outcome
proposer and do not launch the held-out campaign. The next bounded modelling question
is whether **paired response changes anchored to the initial achieved preview** are
predictable beyond no-change and scene-only response controls. The concrete design
is in [the next-stage design](astra-a8-paired-change-design.md).

Do not convert this diagnostic into a new selector paper or improve its headline by
changing the descriptor. A useful successor must first predict edit effects, then
demonstrate a newly regenerated bounded command with its own exact absent preview.
All-assigned obstacle-present comparison against strong geometry/simple/search
controls remains necessary for the intended paper.

## Validation and provenance

Nine new regressions cover independently missing targets, explicit no-arrival,
scene leakage, edited-preview isolation, exact-candidate refusal, protected baseline,
training-only search choice, bounded inputs and nonfinite targets. Combined focused
suite: 26 passed. Full CPU: **1211 passed in 204.22 s**. Results and rebuild verification are recorded in
[the validation receipt](../outputs/astra_a7_validation_v1/receipt.json).
Initial test collection caught the new module in the repository root; it was moved
to the package before the preregistration commit and before this experiment ran.

New samples/previews/present evaluations:0/0/0. Archive reuse does not remove the A6
cost:384 generated references and608 operational previews, including the separately
preserved96 unmatched diagnostics and128 padding previews. No generator/controller
weights, mode, simulation state, old protocols or pre-existing probe/report edits
were changed. No simulator or background admission poller was launched.
