# A8: baseline-anchored paired native response

**Status: preregistered**. September 6, 2026. CPU development screen. Commit this
protocol and implementation before computing its results. A7 and its failed expansion
gate remain unchanged. This tests the specific successor hypothesis in the A8 design.

Use the same verified A6 paired corpus: 12 scenes, 48 units, 336 native observations
and 48 separately reported free controls. Exclude padding and unmatched diagnostics.
No generation, simulator launch, contact probe or held-out confirmation is assigned
in this screen. Reused development folds are not untouched tests.

## Models and inputs

Predict each numeric target as its initial achieved value plus a learned command
response. Targets are inflated signed clearance, whole-body rear exit and one-second
continuation. Fit **candidate minus initial** differences, using independently observed
baseline/candidate pairs for each target. Missing targets remain unavailable. No-edit
prediction must equal the initial measurement exactly. No raw motion editing.

Four fixed models, without hyperparameter search:

1. **No change:** initial achieved measurement for every candidate.
2. **Constant response:** a shared three-coordinate response matrix.
3. **Scene/reference response:** command changes interact with scene geometry and
   initial reference measurements.
4. **Execution-phase response:** the same model plus initial achieved/reference errors
   in first overlap entry, final exit, signed clearance and continuation, the four
   observation-status indicators and last observed time.

All three learned response models use training execution labels and share the measured
initial numeric anchor. The scene/reference control removes execution context from
the response coefficients; it is not a fully geometry-only compiler. Such a control
remains necessary in a later actual correction comparison.

Normalize command changes by (0.04,0.18,0.18) m around the A6 baseline. Scene geometry
uses A7 scales. Reference features: clearance/0.10 m, first entry/2 s, exit/4 s,
continuation/0.9 m and the existing corridor/upright flag. Execution-reference errors
use scales (1 s,1 s,0.10 m,0.45 m). Every missing scalar has its own missingness bit;
zero filling encodes input absence, never a numeric target. Last observed time uses
8.28 s. No edited reference, edited preview, seed, slot or scene identifier is a
predictive feature. Scene IDs serve only grouped exclusion.

Ridge penalty 10 on every numeric-response coefficient, with no response intercept.
Thus zero edit gives zero change by construction. All six nonbaseline assignments
per unit enter accounting. Independently available pairs train each numeric head;
this remains conditional estimation, not a statistical correction for censoring.

Also fit separate four-category status heads using **all seven** native assignments
per training unit, including non-arrival, exit censoring and recovery censoring.
Use the context and command-interaction features with ridge 10 and an unpenalized
status intercept. Clip category scores to [0,1] and normalize their sum; this is an
uncalibrated status prediction, not an admission certificate. No-change status
prediction preserves the initial status. Numeric predictions with missing baseline
targets stay unavailable even when a status head predicts arrival. No fallback
motion or qualification is inferred from an unavailable target.

## Evaluation and decision

Twelve outer leave-one-scene-out folds: train on 44 units, evaluate all four units
and six edited commands of the omitted scene. Keep all descendants together. Report
per-target matched counts, change MAE, paired command-effect sign accuracy, and
per-scene paired error differences with 10,000 scene-bootstrap resamples, seed 68000.
Report zero actual/predicted effects separately. Report +/- directional diagnostics
with their separate available-pair counts. Status Brier scores cover all 288 edited
assignments, including every numerical omission. Report complete-baseline coverage
and baseline status counts over all 48 units. Free controls remain separate.

The predictor-readiness gate requires the full execution-phase model to:

- reduce scene-averaged paired error relative to no-change for clearance and
  continuation, with exit error no more than 5% worse;
- beat both constant-response and scene/reference response on scene-averaged
  clearance and continuation error;
- predict the correct nonzero continuation-effect sign on more than half the
  observed edited/baseline pairs;
- improve edited status Brier error over no-change, with at least 12 baseline units
  having all three numeric targets observed for a possible constrained pilot.

These are engineering prerequisites for considering a new regeneration pilot, not
significance or paper acceptance thresholds. All-scene and per-target available-scene
counts must accompany errors; missing numerical scenes are never filled with zero.
If the gate fails, stop optimizer expansion and retain the result. Do not add new
features, alter penalties/bounds, or switch the endpoint within A8. A separate
scientifically motivated successor can reuse development data with fresh outputs.

If the gate passes, prepare a **separate** regeneration protocol. Fit error allowances
inside training folds, never from outer evaluation residuals. It must freeze the
bounded command grid, clearance/deadline/continuation constraints, progress ablation,
candidate and preview budgets, matched encoder slots and resource gates. No present
outcome may select the command. A8 itself assigns zero compound-command predictions
as validated motions and zero obstacle-present executions.
