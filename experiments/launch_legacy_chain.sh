#!/usr/bin/env bash
# Gated launcher for the legacy tracker chain: EXP-028, then EXP-024's SONIC and analyse stages.
#
# "Legacy" means the UNPATCHED tracker checkout whose core-source manifest equals EXP-022A's
# frozen baseline.  EXP-029 runs from the separate worktree pinned to the patched commit and must
# never be launched by this script; nothing here switches a checkout.
#
# Preconditions, all re-checked before EVERY launch, never relaxed:
#   1. the scene2motion worktree is clean (a dirty tree means work is still in flight);
#   2. no other instance of this script, and no campaign process, is already running;
#   3. the legacy tracker's core-source manifest equals the frozen baseline;
#   4. >= 12 GiB free VRAM, >= 18 GiB available RAM, and no concurrent Isaac process.
#
# Condition 4 is waited for; 1-3 are refusals that stop immediately, because they mean the world
# is not in the state the chain assumes rather than merely busy.  Any non-zero campaign exit stops
# the chain for inspection instead of continuing to the next stage.
#
# LUCID is never interrupted: a concurrent Isaac process makes this script wait, not kill.
#
# Usage:
#   bash experiments/launch_legacy_chain.sh --check     # preflight only, changes nothing
#   bash experiments/launch_legacy_chain.sh             # wait for the gates, then run the chain
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

MIN_VRAM_MIB=${MIN_VRAM_MIB:-12288}          # 12 GiB, the prescribed gate
MIN_RAM_MIB=${MIN_RAM_MIB:-18432}            # 18 GiB, the prescribed gate
POLL_S=${POLL_S:-30}
MAX_WAIT_S=${MAX_WAIT_S:-43200}
LOG=${LOG:-run/legacy_chain.log}
EXP028_OUT=outputs/exp028_termination_free_rollouts
EXP024_OUT=outputs/exp024_reference_contract

mkdir -p "$(dirname "$LOG")"
say() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
refuse() { say "REFUSED: $*"; exit 3; }

# ---------------------------------------------------------------- preconditions

check_clean_tree() {
  local dirty
  dirty=$(git status --porcelain | wc -l)
  [ "$dirty" -eq 0 ] || refuse "scene2motion worktree has $dirty modified/untracked paths;
    the chain runs only from a clean tree (house rule 3), and a dirty tree here means the
    evaluator handoff has not landed yet"
}

# `pgrep -f` matches a command LINE, so every shell in this script's own ancestry matches the
# patterns below: a wrapper such as `bash -c '... launch_legacy_chain.sh ...'` is a different pid
# from $$ and was counted as "another instance", making the guard refuse its own invocation and
# block the chain forever.  Excluding the whole lineage keeps the guard strict against a real
# second owner while letting the script run when it is the only one.
own_lineage() {
  local pid=$$ out=""
  while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null; do
    out="$out $pid"
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  done
  printf '%s' "$out"
}

# Pids matching $1 that are neither an ancestor of this process nor one of its descendants.
# Descendants matter as much as ancestors: every command substitution in this script forks a
# subshell that KEEPS the script's own command line, so `pgrep -f` finds our own `$(...)` calls
# and the guard reported them as a second owner.  A pid that has exited between pgrep and ps is
# skipped rather than counted, so the check does not flicker on transient shells.
others_running() {
  local lineage pid walk args keep=""
  lineage=" $(own_lineage) "
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    case "$lineage" in *" $pid "*) continue ;; esac
    args=$(ps -o args= -p "$pid" 2>/dev/null)
    [ -n "$args" ] || continue
    walk=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    while [ -n "$walk" ] && [ "$walk" -gt 1 ] 2>/dev/null; do
      [ "$walk" = "$$" ] && break
      walk=$(ps -o ppid= -p "$walk" 2>/dev/null | tr -d ' ')
    done
    [ "$walk" = "$$" ] && continue
    keep="$keep $pid"
  done
  printf '%s' "${keep# }"
}

check_no_other_owner() {
  local others
  others=$(others_running 'launch_legacy_chain\.sh')
  [ -z "$others" ] || refuse "another launch_legacy_chain.sh is already running (pids: $others)"
  others=$(others_running 'launch_when_host_free\.sh outputs/exp028')
  [ -z "$others" ] || refuse "a legacy poller (launch_when_host_free.sh) already owns the EXP-028 chain (pids: $others)"
  others=$(others_running 'exp028_termination_free_rollouts\.py|exp024_reference_contract\.py --stage sonic')
  [ -z "$others" ] || refuse "a campaign process is already running (pids: $others)"
  return 0
}

check_manifest() {
  local out
  out=$("$S2M_PY" -c "
import experiments.exp028_termination_free_rollouts as e28
t = e28.tracker_identity()
m = t.get('core_source_manifest_sha256')
print('MATCH' if m == e28.EXPECTED_CORE_MANIFEST_SHA256 else 'MISMATCH', m)
" 2>&1 | tail -1)
  case "$out" in
    MATCH*) say "manifest ok (${out#MATCH })" ;;
    *) refuse "legacy tracker manifest is not the frozen baseline: $out
    the checkout is probably still patched for EXP-029, or an upstream commit touched a core
    source file; do not launch until it matches" ;;
  esac
}

check_outputs_launchable() {
  if [ -d "$EXP028_OUT" ] && [ -n "$(ls -A "$EXP028_OUT" 2>/dev/null)" ]; then
    [ -f "$EXP028_OUT/receipt.json" ] ||
      refuse "$EXP028_OUT is non-empty without a receipt; inspect it rather than relaunching"
  fi
  [ -f "$EXP024_OUT/receipt.json" ] ||
    refuse "$EXP024_OUT/receipt.json is missing; EXP-024's frozen predictions must exist first"
}

# Isaac processes that are not this script's own shells.  A shell whose command line merely
# mentions the interpreter path is not a running simulator; a real one, LUCID's included, is.
isaac_process_count() {
  others_running 'env_isaaclab|isaacsim|eval_agent_trl' | wc -w
}

gates_pass() {
  local vram ram isaac
  vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  # `pgrep -c` prints 0 AND exits non-zero when nothing matches, so the old `|| echo 0` appended
  # a SECOND zero.  Every integer test below then failed with "integer expression expected",
  # gates_pass could never return true, and the chain would have waited out its entire timeout
  # on a host that already satisfied every gate -- a silent 12-hour stall reported as "WAIT".
  isaac=$(isaac_process_count)
  isaac=${isaac:-0}
  LAST_GATE="vram=${vram:-0}MiB ram=${ram:-0}MiB isaac=${isaac}"
  [ "${vram:-0}" -ge "$MIN_VRAM_MIB" ] && [ "${ram:-0}" -ge "$MIN_RAM_MIB" ] && [ "$isaac" -eq 0 ]
}

wait_for_gates() {
  local start now
  start=$(date +%s)
  while ! gates_pass; do
    now=$(date +%s)
    if [ $((now - start)) -gt "$MAX_WAIT_S" ]; then
      say "gave up waiting for the gates after ${MAX_WAIT_S}s (last: $LAST_GATE)"
      exit 4
    fi
    sleep "$POLL_S"
  done
  say "gates pass ($LAST_GATE)"
}

preflight() {
  check_clean_tree
  check_no_other_owner
  check_manifest
  check_outputs_launchable
  say "preflight ok; gate state now: $(gates_pass && echo PASS || echo "WAIT ($LAST_GATE)")"
}

# Every launch re-checks the refusal conditions: LUCID commits to the shared tracker checkout
# while we wait, and a manifest that drifted between stages must stop the chain, not be ignored.
run_stage() {
  local label="$1"; shift
  check_no_other_owner
  check_manifest
  wait_for_gates
  say "launching $label: $*"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  say "$label exited rc=$rc"
  [ $rc -eq 0 ] || { say "STOPPING the chain for inspection after $label (rc=$rc)"; exit "$rc"; }
}

# ---------------------------------------------------------------- main

[ -n "${S2M_PY:-}" ] || { echo "source env.sh first" >&2; exit 2; }

if [ "${1:-}" = "--check" ]; then
  say "=== preflight only, nothing will be launched ==="
  preflight
  say "=== preflight complete ==="
  exit 0
fi

say "=== legacy chain armed (EXP-028 -> EXP-024 sonic -> EXP-024 analyze) ==="
preflight
run_stage "EXP-028" "$S2M_PY" experiments/exp028_termination_free_rollouts.py \
  --stage all --out "$EXP028_OUT"
run_stage "EXP-024 sonic" "$S2M_PY" experiments/exp024_reference_contract.py \
  --stage sonic --require-committed-predictions --out "$EXP024_OUT"
run_stage "EXP-024 analyze" "$S2M_PY" experiments/exp024_reference_contract.py \
  --stage analyze --out "$EXP024_OUT"
say "=== legacy chain complete ==="
