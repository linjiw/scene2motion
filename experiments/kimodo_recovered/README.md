# Kimodo-G1 runner and reduced audit — recovered from a session transcript

These four files were written on 2026-08-31 into a session scratchpad that no longer exists
(`/tmp/claude-1000/-home-linjiw-ardy/f4440d67-…/scratchpad/kimodo/`, removed by scratchpad
cleanup; see `docs/kimodo-provenance-2026-08-31.md`). They are **not re-authored**: every byte
below was replayed out of the transcript's own tool calls. Recovering them is prerequisite 1 of
`docs/ramp-exp025-kimodo-cross-prior-protocol.md`.

Nothing here has been run against Kimodo-G1-RP-v1 in this repository. The GPU-free checks
(`--selftest`, import, `py_compile`, and `tests/test_kimodo_recovery.py`) are the only
validation that exists; see "What is still unvalidated" below.

## Source

| | |
|---|---|
| transcript | `~/.claude/projects/-home-linjiw-ardy/f4440d67-ed27-4331-be07-dc169754a80c/subagents/agent-ac06b53618f8a2379.jsonl` (206 lines) |
| sha256 | `e4e28eae8a5e8eb0a8a205ff6f7ec907014e0fe814612aad053e93445428f3f4` |
| subagent | `agent-ac06b53618f8a2379` — "Draft KimodoRunner adapter" (`agentType: general-purpose`, `spawnDepth: 1`) |
| parent session | project `-home-linjiw-ardy`, session `f4440d67-ed27-4331-be07-dc169754a80c` |

## Operations replayed, in transcript order

Every file-mutating tool call in the transcript, applied in order. "clean" means the Edit's
`old_string` matched exactly once in the file as reconstructed at that point, so the edit was
applied without reconciliation.

| line | op | file | result |
|---|---|---|---|
| L93 | Write | `kimodo_runner.py` | 22,220 B / 447 lines |
| L95 | Write | `smoke_kimodo.py` | 7,923 B / 166 lines |
| L98 | Edit | `smoke_kimodo.py` | clean (hoists the `torch` / `KimodoConstraintSet` imports) |
| L100 | Edit | `smoke_kimodo.py` | clean (drops the `__import__(...)` workaround) |
| L102 | Write | `NOTES.md` | 11,035 B / 136 lines |
| L110 | Edit | `kimodo_runner.py` | clean (heading convention comment + `_first_heading`) |
| L112 | Edit | `NOTES.md` | clean (line-number fix, 536-541 → 534-538) |
| L118 | `sed -i s/…/…/` | `NOTES.md` | 1 line substituted (adds `motion_rep_dim = 417`) |
| L122 | Edit | `NOTES.md` | clean (re-wraps the sentence L118's sed mangled) |
| L181 | Write | `kimodo_reduced_audit.py` | 28,042 B / 553 lines |
| **L183** | **`cat >> … <<'EOF'`** | `NOTES.md` | **+1,403 B addendum — not listed in the protocol's op list, but required: without it `NOTES.md` has no audit section** |
| L190 | Edit | `kimodo_reduced_audit.py` | clean (neutral rung recorded as literal zeros) |
| L192 | Edit | `kimodo_reduced_audit.py` | clean (selftest ledger + assertions) |
| L194 | Edit | `kimodo_reduced_audit.py` | clean (selftest counting assertions) |
| L200 | `sed -i /…/d` | `kimodo_reduced_audit.py` | 1 line deleted (unused `mask_frames = …`) |
| L203 | `sed -i s/…/…/` | `kimodo_reduced_audit.py` | 1 line substituted (budget wording) |

**No edit failed to apply and nothing had to be reconciled by hand.** All 16 operations landed
on a unique match; the three `sed` commands each matched exactly one line.

## Independent confirmation that the replay is byte-exact

The transcript's own console output pins the files' sizes at two checkpoints, and the replay
reproduces them exactly:

* **L118/L119** ran `ls -la` and `grep -c ""` in the scratchpad. Recorded:
  `kimodo_runner.py` 22,553 B / 453 lines, `smoke_kimodo.py` 7,895 B / 168 lines,
  `NOTES.md` 11,131 B / 136 lines. The replay's state at that point matches all six numbers
  (NOTES.md is 11,075 *characters* — 11,131 UTF-8 bytes, the difference being its em dashes).
* **L197/L198** ran `kimodo_reduced_audit.py --selftest` and printed
  `3.0 / 3.0 / 3.0 / 2.0`; **L200/L201** printed `clean: compile + selftest pass`;
  **L203/L204** printed `ok`. All three reproduce today (below).

Recovered file hashes as committed here:

```
7e3d4b259497833be816692a878febab255c1bd3e14053d0f9b8632947ef03b9  kimodo_runner.py
008301261a70c88a1e52b5854e413ce9f27bf7be7e568284c5b7e55902fea9d6  smoke_kimodo.py
1e95c58d6fba818a2d4124c1b33a4ddfe90b85deb51e193a616a1d84b905ac6f  kimodo_reduced_audit.py
06321de62f3a0fb3aacc6cf78da1c809b55620a74211e7dbe36e1cdb7d1862c7  NOTES.md
```

## GPU-free validation performed (2026-09-03)

```bash
# the protocol's acceptance criterion: 3.0 / 3.0 / 3.0 / 2.0
SCENE2MOTION_ROOT=$S2M_ROOT $S2M_PY experiments/kimodo_recovered/kimodo_reduced_audit.py --selftest
# -> selftest OK: {"1 seed, any change > 1 mm": 3.0,
#                  "1 seed, round 1 cm threshold": 3.0,
#                  "1 seed, 1 cm, drop clips that never validate": 3.0,
#                  "6 seeds, paired, q99-calibrated, stability >= 0.8": 2.0}

# same result under the Kimodo venv, which is what the audit is meant to run in
SCENE2MOTION_ROOT=$S2M_ROOT /home/linjiw/kimodo/.venv/bin/python \
    experiments/kimodo_recovered/kimodo_reduced_audit.py --selftest

# compile + import with no checkpoint, no `kimodo` package, no GPU
/home/linjiw/kimodo/.venv/bin/python -m py_compile \
    experiments/kimodo_recovered/{kimodo_runner,smoke_kimodo,kimodo_reduced_audit}.py

$S2M_PY -m pytest tests/test_kimodo_recovery.py -q       # 52 tests
```

`--selftest` exercises the vendored descriptors and the four counting rows on synthetic data
only: no `kimodo` import, no checkpoint, no CUDA. `KimodoRunner.__init__` is the first thing
that touches either.

## Known warts in the recovered files (left as recovered; fix deliberately, not silently)

1. **Docstrings still point at the deleted scratchpad.** `smoke_kimodo.py:6`,
   `kimodo_reduced_audit.py:10` and `NOTES.md`'s "Commands to run" block tell you to
   `cd /tmp/claude-1000/-home-linjiw-ardy/…/scratchpad/kimodo`. Run them from
   `experiments/kimodo_recovered/` instead.
2. **`SCENE2MOTION_ROOT` defaults to the absolute path `/home/linjiw/scene2motion`**
   (`kimodo_runner.py:48`, `kimodo_reduced_audit.py:88-89`) and is inserted at `sys.path[0]`.
   In a git worktree or a second checkout that silently imports *another tree's*
   `scene2motion`. Set `SCENE2MOTION_ROOT=$S2M_ROOT` explicitly; the test file does.
3. **`kimodo_reduced_audit.py:45` and `NOTES.md:154` both list `PHYSICAL_FLOOR` among the
   symbols imported from `scene2motion.morphology`, but the import at
   `kimodo_reduced_audit.py:94-96` does not include it** — the floors are reached indirectly
   through `morphology.active_set`. Documentation-only discrepancy.
4. **Neither script is a RAMP harness.** No `scene2motion.host_gate` call, no clean-worktree
   or empty-`--out` gate, no provenance ledger persisted before generation, no resume. House
   rules 3–5 are unmet: `kimodo_reduced_audit.py` writes only `receipt.json` +
   `per_program.json` at the end, with no `rows.jsonl` and no launch-level resume.

## What is still unvalidated

Everything downstream of `KimodoRunner.__init__`. Not exercised anywhere: `load_model`,
`_generate`, `motion_rep.create_conditions_from_constraints_batched`, `MujocoQposConverter`,
the `_per_sample_noise` patch against Kimodo's *actual* `torch.randn` call site, and the whole
of `smoke_kimodo.py` past its import block. The private `Kimodo._generate` signature the runner
depends on was verified by reading the checkout on 2026-08-31 and has not been re-verified
since.
