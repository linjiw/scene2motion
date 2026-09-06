# Contact-probe v2: harness-only recovery

**Status: preregistered**. This amendment is committed before its first launch.
The [v1 protocol](astra-contact-probe-protocol.md) retains its original meaning.

## Preserved failure

At source69227b6, `outputs/astra_contact_probe_v1` launched only positive/off. All200
states were written and the body settled at0.600000024 m, but the Isaac process did
not exit within300 s. The supervisor terminated its own process group and preserved
the failed receipt. No contact view was created; no logged probe or G1 replay ran.
The subsequent verifier also lacked NumPy because it inherited `/usr/bin/python3`.
Neither issue is evidence that the contact instrument passed or failed its physics test.

## Exact changes and budget

- After saving all measurement arrays, explicitly stop the simulation timeline and call
  `app.close(wait_for_replicator=False)`. No rendering/replicator tasks were requested.
  Add30 s repeated teardown tracebacks to diagnose a further hang. This does not change
  simulation stepping, initialization, contact reporting or any pass threshold.
- Launch the separate CPU verifier with the installed Isaac Python, which has NumPy,
  while retaining the standard-library-only supervisor during simulation.
- Repeat the original six cases in fresh `outputs/astra_contact_probe_v2`. Total maximum
  across preserved v1 and v2: **seven operational executions**, zero generations or
  independent scientific samples. No automatic retry after a v2 failure.
- Retain the300 s launch timeout, resource gates, capacities64/128,200 steps, exact
  state/contact equivalence rules and stop-on-first-failure behavior.
- Additionally compare v2 positive/off states with the saved v1 array exactly as a
  post-collection regression check. Do not promote the timed-out v1 process to complete.

Stage II G1 integration remains gated on successful completion of all six v2 probes.
