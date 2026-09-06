# Contact-buffer diagnostic after v3 refusal

**Status: preregistered**. One operational positive/base probe only, in fresh
`outputs/astra_contact_probe_diagnostic_v1`; no G1 launch or automatic continuation.

V3 positive/off passed and exited normally. Positive/base exited before saving states
or contact summaries; only setup metadata was written. Isaac fast shutdown suppressed
the pending exception, returning code0 despite missing required outputs. The supervisor
correctly marked this failed. The reader's exact view identity checks had completed.
The failed v3 attempt and earlier v1/v2 teardown failures remain preserved.

Change only error observability: print and save the traceback before teardown, and
capture buffer shape/dtype/values on unpacking failure. Retain all v3 physics settings,
200 steps, capacity64, thresholds, launch gates and300 s timeout. This is a diagnostic
of an integration defect, not another traversal or contact-qualification sample.

Four operational launches already occurred (v1=1, v2=1, v3=2). This adds at most one,
for a cumulative budget of **five** before another explicit amendment. No generated
motion, training or new scientific unit. Invoke the driver with `--diagnostic`.
Inspect its preserved error or result; do not interpret normal process exit as a
completed six-case validation. A subsequent fix requires its own committed protocol.
