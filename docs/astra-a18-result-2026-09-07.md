# A18 bounded G1 contact instrument: closed at failed force gate

The six-job instrument stopped after two jobs (64 operational instrument
exposures). It detected contact points on all32 assigned overlap placements,
but reported zero normal impulse on every target pair. The required positive
force-sensitivity gate therefore passed0/32. Logging preserved all recorded
robot states and joints exactly against the unlogged control.

These are actual G1 articulation tests, with mapped left ankle-roll, left knee
and torso-link primitives. The inventory covers30 bodies/45 colliders, but the
positive stimuli target only three selected collider surfaces. Neither these
tests nor their mapping validate whole-body beam-contact sensitivity.

The capacity-doubling positive job and all three separated negative jobs were
not launched. Specificity and capacity invariance remain untested. Initial
penetration in the gravity-disabled setup may not provide a suitable sustained
force stimulus; the current result does not separate that possibility from
reader behavior. Do not change the threshold or relabel point detection as
force validation. A19's obstacle-absent control experiment can proceed.

Recording contract:32 sensor rows×30 body-pair filters,200 physics steps at5ms,
four-step aggregates at20ms, raw points/normal impulses, target collider mapping,
complete body/joint states, independent state-effect comparison and buffer
capacity4096 (8192 doubling untested). See the
[preregistered protocol](astra-a18-g1-contact-protocol.md),
[closure receipt](../outputs/astra_a18_g1_contact_v1/summary.json) and
[independent reconstruction script](../experiments/analyze_astra_a18_contact.py).
Nominal penetration,40mm conservative-margin violation and detected physical
contact remain separate quantities. No force-based contact-free claim follows.
