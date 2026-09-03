# Scene2Motion-DB corpus-pilot release preview

This preview contains 300 randomized ducking scenes generated with one heuristic proposer and
at most two reference-space repairs. It packages the committed `corpus_pilot_v2` evidence; it
does not upgrade that evidence.

- Evidence tier: scene feasibility for refusals; reference geometry for generated motions.
- Controller execution: not measured for every row in this preview.
- Motion payload: not copied; source paths and hashes only.
- Split: deterministic, outcome-stratified 70/15/15 development split with disjoint scene IDs.
  This split is not a geometry-OOD test and cannot support a generalization claim.
- Signed labels: positive `signed_overhead_target_residual_m` meets the requested overhead
  margin; negative values are shortfalls. `signed_whole_body_clearance_m` is a separate contact
  quantity, where negative values denote reference penetration.

The preview is not yet a public redistribution. The repository currently supplies no dataset
license or third-party motion-output redistribution determination. Resolve those terms before
publishing clip payloads. A public benchmark additionally requires an execution-labelled tier,
an OOD scene split fixed before evaluation, and a downstream utility demonstration.
