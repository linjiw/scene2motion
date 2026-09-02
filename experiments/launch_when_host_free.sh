#!/usr/bin/env bash
# Poll the host and launch an ARDY-only campaign driver when the ARDY generation gate is
# likely to pass (margins above scene2motion.host_gate.ARDY_GENERATION_GATE); the driver's own
# gate remains the authority and a refused attempt leaves --out untouched, so retries are safe.
# Usage: launch_when_host_free.sh <out_dir> <log> -- <driver command...>
# The command may be a shell string via `bash -c "..."` to chain several stages; a non-zero exit
# with a non-empty <out_dir> stops the poller so a failed attempt is never relaunched blindly.
set -u
OUT="$1"; LOG="$2"; shift 2; [ "$1" = "--" ] && shift
MIN_VRAM=${MIN_VRAM:-4600}; MIN_RAM=${MIN_RAM:-8400}; POLL_S=${POLL_S:-20}; MAX_S=${MAX_S:-21600}
# REQUIRE_NO_ISAAC=1: also wait until no Isaac Sim/Lab process exists (SONIC launch gate).
REQUIRE_NO_ISAAC=${REQUIRE_NO_ISAAC:-0}
start=$(date +%s)
while :; do
  if [ -f "$OUT/receipt.json" ]; then echo "$(date -Is) receipt present; stop" >> "$LOG"; exit 0; fi
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$(date -Is) $OUT is non-empty without a receipt; refusing to relaunch" >> "$LOG"; exit 3; fi
  now=$(date +%s); [ $((now-start)) -gt "$MAX_S" ] && { echo "$(date -Is) gave up after ${MAX_S}s" >> "$LOG"; exit 4; }
  free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  avail_ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  isaac_n=0; if [ "$REQUIRE_NO_ISAAC" = "1" ]; then isaac_n=$(pgrep -f "env_isaaclab|eval_agent_trl|isaacsim" | grep -v "^$$$" | wc -l); fi
  if [ "${free_vram:-0}" -ge "$MIN_VRAM" ] && [ "${avail_ram:-0}" -ge "$MIN_RAM" ] && [ "$isaac_n" -eq 0 ]; then
    echo "$(date -Is) gate window: free_vram=${free_vram} avail_ram=${avail_ram}; launching: $*" >> "$LOG"
    "$@" >> "$LOG" 2>&1; rc=$?
    echo "$(date -Is) driver exited rc=$rc" >> "$LOG"
    if [ $rc -eq 0 ]; then exit 0; fi
    if [ $rc -eq 2 ] && [ ! -e "$OUT" ]; then echo "$(date -Is) gate refused at launch; keep polling" >> "$LOG"; sleep "$POLL_S"; continue; fi
    echo "$(date -Is) driver failed with rc=$rc and $OUT state $(ls -A "$OUT" 2>/dev/null | wc -l) entries; stop" >> "$LOG"; exit $rc
  fi
  sleep "$POLL_S"
done
