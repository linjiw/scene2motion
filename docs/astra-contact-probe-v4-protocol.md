# Contact-probe v4: preserve signed PhysX force convention

**Status: preregistered**. Commit before running six cases in fresh
`outputs/astra_contact_probe_v4`; all v1 scene, clock, resource, positive/negative,
capacity and exact-equivalence rules remain fixed. V3's working teardown is retained.

## Measured harness defect

The single diagnostic at402e4f2 reached a four-point table/body contact. PhysX returned
finite scalar forces−161.53,−114.46,−176.28,−116.06 N with unit +z normals, count4,
start0 and capacity64. Our generic nonnegative-force assumption rejected this signed
representation. Error buffers and traceback are preserved in
`outputs/astra_contact_probe_diagnostic_v1/positive_base/eval/error.json`.
The negative signs are consistent with force on the table, not invalid float data.

## Correction and unchanged gates

The live reader now explicitly opts into **signed scalar forces**. Record both
`sum(f)*dt` and the signed vector `sum(f*n)*dt`. Define the non-cancelling normal
impulse magnitude as `sum(abs(f))*dt`; preserve point counts and reported separation.
This fixes representation handling rather than clipping, deleting or ignoring contacts.
Finite-value, unit-normal, identity, buffer, clock and capacity checks remain unchanged.
The generic helper's default nonnegative contract remains available for its other callers.
Tests include opposing signed forces with zero net vector but positive magnitude.

Repeat all six fixed cases, with the same strict state equality and capacity equality
criteria. Total maximum across v1(1), v2(1), v3(2), diagnostic(1), v4(6): **11 operational
executions**, zero generated references or independent scientific units. Preserve every
failed attempt. Stop on the first new failure; no automatic retry or G1 continuation.
Compare positive/off states across all archived versions as a separate exact regression
check, never relabeling the earlier failed processes. Six passing probes only qualify
this simple-body instrument test; G1 callback/whole-body coverage still need validation.
