# Contact-probe v3: disable post-collection stop-render loop

**Status: preregistered**. Commit before launching; all v1 measurement definitions,
six-case order, capacities, exact equivalence tests and resource gates remain unchanged.

V1 preserved one timed-out positive/off execution. V2 preserved one positive/off
execution manually aborted after its teardown traceback identified the loop in the
installed `isaaclab/sim/simulation_context.py::_app_control_on_stop_handle_fn`:
while the timeline is stopped, it repeatedly renders. Both attempts wrote200 states;
their complete state tensors are exactly equal. Neither attempt is a completed probe.
V2's `operator_abort.json` records the owned worker termination; no co-tenant was killed.

The installed Isaac Lab `test/sim/test_spawn_shapes.py` teardown explicitly sets
`sim._disable_app_control_on_stop_handle = True` before `sim.stop()` to prevent timeout.
V3 applies this same guard **only after measurement output**, retaining v2's explicit
timeline stop, disabled replicator wait, traceback watchdog and corrected verifier
interpreter. No sensor/physics/controller parameter or outcome gate changes.

Fresh output: `outputs/astra_contact_probe_v3`. Six new operational cases; maximum
across preserved v1/v2/v3 is **eight operational executions**, no generated references,
no independent scientific units. Stop on any launch/verification failure, with no
automatic retry. Check v3 positive/off's saved states exactly against v1/v2 in the
final audit, without converting either failed process into a completed attempt.

All six instrument probes must pass before a separately frozen G1 replay protocol.
