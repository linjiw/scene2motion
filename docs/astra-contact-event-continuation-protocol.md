# Contact event diagnostic: resource-abort continuation

**Status: preregistered**. Commit before preparation or launch.

Continue the [event diagnostic](astra-contact-event-protocol.md) after its preserved
`outputs/astra_contact_event_v1` launch stopped before any evaluation output. Its
27.765 s resource abort is not a scientific failure or an executable completed batch.
Bind its job, launch, process, log and resource hashes; refuse continuation if the
recorded cause or absence of output differs. Never overwrite or resume that launch.

The resumed user request authorizes this single versioned operational continuation.
Use fresh external `outputs/astra_contact_event_v2`, one32-slot present batch, the
same fixed3/default references, order, seed0, checkpoint, tracker and event callback.
No simulator, gate, batch-size, motion, endpoint or numerical-tolerance change.
Strengthened preparation checks only bind the preserved failed attempt.

Keep RAM/VRAM admission9216/5500 MiB, runtime floors2500/1200 MiB,900 s timeout,
compact allocation and the single ASTRA lock. Wait at most180 s for admission. If no
launch occurs, preserve preflight refusals and stop this invocation. If a launch fails,
preserve it and do not retry. Other users' processes remain untouched.

Apply the original event protocol's exact-state/apparatus/mode and event membership
checks. At most32 additional operational executions; zero new independent scientific
units, generated references or training. Retain the source A5 development groups and
all earlier costs. This only locates excluded contact events; it is not a controlled
positive G1 probe, method gain, or motion-quality certification.
