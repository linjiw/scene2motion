# A7: pooled response transfer screen

**Status: preregistered**. September 6, 2026. CPU development experiment, after
inspection of A6 aggregate results but before computing this comparison.

Test whether a small pooled model can choose useful native perturbations from the
initial preview when complete local Jacobians are unavailable. This is a diagnostic
of the correction mechanism on the seven measured A6 commands, not a new bank
selector contribution or a held-out method result. No new generation, physics launch,
present evaluation or confirmation seed is assigned in this screen.

Use only the 384 scientific, encoder-paired d1 records; exclude 128 padding and all
96 unmatched d0 diagnostics. The 48 free controls remain in reporting but not fitting.
Keep all 48 native units and their failures. Twelve outer leave-one-scene-out folds
exclude all four units and every descendant of the evaluation scene from training.
The geometries and aggregates have already informed development: these folds are
development validation, not untouched confirmation or broad geometry OOD.

Fit ridge linear models, fixed penalty 10, with an unpenalized intercept and fixed
physical feature scaling. Features are scene geometry, initial reference clearance,
exit, continuation, corridor/upright descriptor and joint-velocity maximum. The full
model additionally receives the initial achieved version of those quantities, its
explicit observation-status indicators and last observation time. Missing input
values have a separate missingness bit; zero filling is a numerical feature encoding,
never an outcome label. No seed, slot ID, scene ID, edited reference or edited preview
is a predictor. Candidate coordinates are normalized by the A6 perturbations; include
their interactions with initial context. Backbones, modes, bounds and endpoints stay
unchanged. All seven native records train the binary conjunction model, including
non-arrival and censored failures. Separate numeric models use every available signed
clearance, exit or continuation label independently, without complete-unit filtering
or imputing missing targets. Numeric predictions are conditional diagnostics, not
unconditional clearance promises. Clipping a binary ridge prediction to [0,1] does
not make it a calibrated probability.

Controls: scene-relative A6 baseline; shallow native command; baseline plus shallow
exact preview; and a strong budget-matched search control that chooses its second
single-axis command by training-fold union success (baseline OR candidate), then uses
only that candidate's exact preview on the evaluation unit. Train a reference-only
version of the same model as the execution-information ablation. Free WALK remains
an unadapted diagnostic. Protect an initially passing output. Otherwise the model
proposes the single nonbaseline command with highest predicted conjunction score;
ties follow the committed variant order. This preliminary ranking does not claim
predicted metric feasibility. Final admission requires the exact candidate's existing
A6 descriptor, including complete one-second recovery, >=0.45 m continuation, deadline,
geometry and corridor/uprightness. A failed second preview is a refusal in the all-
assigned denominator. Never inspect the other five evaluation previews to select.
An explicitly outcome-informed seven-command oracle measures available headroom only.

The retrospective execution of each decision uses at most one baseline plus one
edited reference/preview, with archived costs identified as reuse. It does not erase
the 384-reference/608-preview development collection cost. No re-generation identity,
compound-edit effectiveness, physical-contact qualification or obstacle-present
performance is established by archive replay.

Report every decision, per-scene paired pass differences, scene bootstrap intervals
(10,000 resamples, seed 67000), descriptive unit Wilson intervals with grouping caveat,
all-target and edited-target binary Brier scores, per-metric sample counts/errors and paired response
direction accuracy. Compare binary prediction to the training-fold prevalence control.
No hyperparameter search or repeated model revision in this version.

Expansion gate: undertake a separately preregistered actual correction/regeneration
pilot only if the full model gains at least 5/48 all-assigned descriptor passes over
both the reference-only model and the strongest simple/budget-matched control, and
improves edited-target binary Brier error over training-fold prevalence. Baseline
predictions are reported separately because their outcomes are already observed.
This approximately ten-point
engineering gate is not a significance threshold. If it fails, preserve the negative
result, identify the binding prediction/coverage limitation and update the research
plan; do not open confirmation, expand bounds or change the endpoint to recover a win.
