# A4 recovery and motion-quality audit

## Scope and delivered tools

The CPU-only audit covers all **32 references and 256 archived executions**, including
failures and unchanged-control fallbacks. It generates no motions, launches no simulator,
changes no historical endpoint, and imposes no new quality/admission threshold.

- [Quality records](../outputs/astra_a4_quality_v1/index.json): aligned root/joint errors,
  per-joint velocity/acceleration diagnostics, foot-plane descriptors, recovery timing,
  original outcome labels and source-motion split groups.
- [Static controller inventory](../outputs/astra_sonic_interface_v1/index.json): launch-bound
  release configuration and source files verified against tracker commit `7c63c53`.
- `scene2motion/motion_quality_audit.py`: reusable, tested timestamp alignment, numerical
  derivatives, and descriptive foot-plane measurements. One achieved archive is resident
  at a time; each reference's geometry is computed once and reused across repeats.

Reference frame i is at i/25 seconds; achieved frame i is at (i+1)/50. Shared-grid
comparison pairs reference[1:] with achieved[1::2], with no interpolation, extrapolation,
NaN padding or reset teleports. Dynamics use unfiltered finite differences of the 29
bounded joints, not root-quaternion components. Both native-rate diagnostics and matched
25 Hz comparisons are retained. The fixed four-second window includes each observed
failure prefix; missing four-second measurements stay null, not zero.

## The timing loss is a progress problem, not a larger reference command

For a4u2, changing the hold from 3.00 to 3.06 s changes reference root x at four seconds
from **3.635 to 3.643 m**. Achieved positions change much more:

| Present slot repeat | Fixed3 root x at 4 s (m) | Adaptive root x at 4 s (m) | Position decrease (m) |
| --- | ---: | ---: | ---: |
| 0 | 2.051 | 1.948 | 0.103 |
| 1 | 1.991 | 1.783 | 0.207 |
| 2 | 1.937 | 1.884 | 0.054 |
| 3 | 2.023 | 1.737 | 0.286 |

The median signed achieved-minus-reference root error increases in magnitude from
**-1.628 to -1.810 m**. Three of these repeats lose A4 local passage; the fourth still
passes. Beam clearance is preserved, but the original carrier's qualification does not
transfer to the regenerated reference. Root-only late dwell diagnostics remain separate
from the frozen whole-body endpoint; no deadline is relaxed.

The same seed also contradicts a simple acceleration-spike explanation. Reference peak
joint speed changes 7.57→7.54 rad/s and acceleration 161.3→162.8 rad/s². On the shared
25 Hz grid, **all four adaptive executions have lower peak joint speed and acceleration**
than their matched fixed3 executions, yet all four lose forward position. This does not
prove smoothing is useless; it shows these scalar maxima do not explain the observed loss.
The exact physical mechanism still needs instrumented execution, not a stronger causal claim.

## Quality descriptors do not yet qualify demonstrations

The following are medians of per-execution diagnostics over the present arms. Root error
uses only executions observed through four seconds; derivatives and foot descriptors use
all 32 valid prefixes, with each prefix's sample count retained in the records.

| Diagnostic | Fixed3 | Adaptive |
| --- | ---: | ---: |
| Signed root x error at 4 s (m) | -1.547 (n=32) | -1.551 (n=31) |
| Peak joint speed, shared 25 Hz grid (rad/s) | 7.33 | 7.49 |
| Peak joint acceleration, shared 25 Hz grid (rad/s²) | 179.6 | 189.4 |
| Maximum foot-pad depth below z=0 (mm) | 5.91 | 5.98 |
| Maximum near-plane foot-envelope speed (m/s) | 2.71 | 2.98 |

These are not significance tests or actuator limits. The near-plane band is explicitly
|pad-bottom z|<=0.02 m and does **not** filter by low speed, which would hide fast motion.
It records speed and maximum drift within each contiguous interval, never joining separate
footfalls. Pad-envelope motion includes foot rotation and swing transitions: it is **not
measured stance slip**. Likewise z=0 is a reference plane, not the actual rough-terrain
surface; depth below it is **not PhysX penetration**. Actual contact, force, self-collision,
energy, and comprehensive motion quality remain unmeasured.

These quantities make the diagnostic corpus more useful, but do not promote A4's 16/32
adaptive local passes to high-quality demonstrations or full-route successes.

## Static interface finding and the next measurement

The pinned release's G1 encoder takes future joint positions/velocities and anchor
orientation. Its listed proprioceptive inputs are gravity direction, angular velocity,
joint position/velocity, and action history. This configured branch has no explicit
horizontal root-position-error input. Root-path agreement should therefore be measured,
not assumed from a nominal ARDY route-speed constraint.

Importantly, the release config lists G1, teleop and SMPL encoder sampling weights, and
the archived A4 launch has no explicit encoder override. The command source samples modes
subject to available motion data, but A4 did not archive the active per-slot mode. The
static inventory does **not** establish which mode each archived rollout used, or that
mode differences caused the losses. It would be incorrect to describe A4 as a verified
G1-encoder-only experiment. Preserve its original frozen-release-controller interpretation.

The next instrumentation step is to log the resolved encoder mode and command interface
per slot alongside achieved progress; contact logging remains a separate requirement.
Any explicit-mode comparison needs its own protocol and matched controls, not a silent
change to A4 or a rerun counted as fresh scientific evidence. Continue to retain fixed2
and fixed3 as comparators and qualify the **final edited** carrier before deployment.
Do not add more hold values or train a risk model from these development labels yet.

## Validation and rebuild

The full CPU suite passed **1030 tests in 222.02 s**. Four subsequent static-inventory
tests pass in the **16-test focused suite**. Both new artifacts rebuild exactly; source
hash changes, missing assignments, ambiguous source expressions and padded/non-finite
motions are rejected. No ASTRA process remains running.

```bash
source env.sh
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.analyze_astra_a4_quality \
  --out outputs/astra_a4_quality_v1/index.json --check
env LD_LIBRARY_PATH= "$S2M_PY" -m experiments.audit_astra_sonic_interface \
  --source outputs/astra_a4_adaptive_d0/summary.json \
  --out outputs/astra_sonic_interface_v1/index.json --check
```
