# A18 design: controlled positive G1 contact sensitivity

**Status: design only; no jobs assigned or launched.** September 6, 2026.

A17 closes the present depth-only lead. Before promoting the simple native compiler
into a new obstacle-present benchmark, resolve the remaining contact measurement
question: does the reader detect deliberately induced contact with actual G1
collision bodies at the correct physics steps? Simple-cube controls and exact
normal-passage replay do not establish this sensitivity.

## Inputs and implementation boundary

Consume the completed A17 method decision, existing corrected `ContactReader`,
contact telemetry aggregation, A5 event-attribution result, frozen G1 USD/collision
assets and original body-name ordering. Read the installed Isaac/PhysX interfaces
before implementation; do not edit the frozen A15 runtime. A separate operational
worker should reuse the reader and reporting schema, with no ARDY generation or
learned controller change.

First produce an outcome-free manifest mapping actual foot, shin and torso rigid
body paths to their collider primitives, transforms, contact/rest offsets and
reader filter indices. Compute positive and separated placements from those actual
colliders. Refuse missing/ambiguous mappings. The feet's normal ground support must
be distinguished from robot–test-obstacle pairs by explicit body-pair filtering.
Do not infer this manifest from a MuJoCo envelope alone.

## Proposed bounded probe

Use a static beam/test obstacle and the complete G1 articulation in a controlled
pose. Freeze pose, gravity/support setup, actuator state and every pair's placement
before physics. Include foot, shin and torso targets, separated controls, and
shallow versus deeper positive placements. Record these as instrument tests rather
than traversal episodes. Avoid interpreting an initialized deep overlap's impulse
as normal traversal contact severity.

Keep 32 environments and original resource floors. A provisional maximum of six
jobs compares logging off/base/double capacity for one fixed positive layout and
one fixed negative layout, with body/placement conditions assigned deterministically
to slots. Exact slot assignments, positive placement distances, duration, sensor
capacity and numerical thresholds must be committed in an executable protocol
following manifest construction. This design does not spend that budget.

Record every physics substep, body-pair identity, point counts, signed force vectors,
sum of individual normal-force magnitudes, integrated normal impulse with explicit
units, buffer occupancy and saturation. Preserve full episodes and valid motion
prefixes as different supports. Counteracting forces may yield zero net vector
while individual contact magnitudes are positive; do not discard them.

## Gates and consumed outputs

1. All mapped positive target pairs must be observed with nonzero point count and
   integrated normal impulse during their preregistered contact window; separated
   pairs must remain zero. A missing positive blocks contact-free claims.
2. Base/double capacity must preserve event membership and agree on impulses within
   a preregistered numerical tolerance, with no saturation or missing samples.
3. Logging off/base/double state comparisons quantify observer effects. Require
   the declared preservation rule before applying the reader to a method comparison.
4. Save every mismatch, teardown failure and resource refusal. No retry of a spent
   launch. A harness correction gets a versioned protocol and preserves its failure.

The output is a body/placement-level detection table and a validated contact
recording contract, not a safe-motion rate. A negative instrument gate triggers
reader/asset diagnosis. A positive gate enables a separate matched obstacle-present
compiler pilot, with command selection based only on absent previews and passage
reported over all assigned tasks. Calibrating obstructive-contact severity is a
further question; nonzero contact and blocked traversal are different endpoints.
