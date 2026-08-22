#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# seed_problems.sh — Seeds fake system problems inside the Docker container
#                    for a realistic linuxai demo.
#
# Run inside the container:
#   seed_problems            # seeds both disk + memory problems
#   seed_problems disk       # disk bloat only
#   seed_problems memory     # memory pressure only
#   seed_problems clean      # removes seeded files / kills stress-ng
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MODE="${1:-all}"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[seed]${RESET} $*"; }
success() { echo -e "${GREEN}[seed] ✔${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[seed] ⚠${RESET}  $*"; }
err()     { echo -e "${RED}[seed] ✖${RESET}  $*" >&2; }

# ── Disk bloat ────────────────────────────────────────────────────────────────
seed_disk() {
    local TARGET="/var/log/fake_bloat.log"
    local SIZE_MB=500   # 500 MB — large enough to trigger df warnings,
                        # small enough to complete in seconds inside Docker

    info "Creating ${SIZE_MB} MB dummy log file at ${TARGET} ..."
    if dd if=/dev/zero of="$TARGET" bs=1M count="$SIZE_MB" status=progress 2>&1; then
        success "Disk seed done. Current /var/log usage:"
        du -sh /var/log
        echo ""
        df -h /
    else
        err "dd failed — are you running as root inside the container?"
        exit 1
    fi
}

# ── Memory pressure ───────────────────────────────────────────────────────────
seed_memory() {
    # Check stress-ng is available (it's installed in the Dockerfile)
    if ! command -v stress-ng &>/dev/null; then
        warn "stress-ng not found — skipping memory seed."
        warn "Install with: apt-get install -y stress-ng"
        return
    fi

    info "Starting stress-ng: 1 vm worker at 70% of RAM for 600s (background) ..."
    # Run in background so the shell stays usable
    stress-ng --vm 1 --vm-bytes 70% --timeout 600s \
              --metrics-brief --log-file /tmp/stress-ng.log &
    local PID=$!
    disown "$PID"  # detach so it survives shell exit

    success "Memory stress running (PID=${PID}, timeout=600s)."
    success "Check pressure with:  free -h"
    success "Kill early with:      pkill stress-ng"
}

# ── Cleanup ───────────────────────────────────────────────────────────────────
seed_clean() {
    info "Cleaning up seeded problems ..."

    local BLOAT="/var/log/fake_bloat.log"
    if [[ -f "$BLOAT" ]]; then
        rm -f "$BLOAT"
        success "Removed $BLOAT"
    else
        warn "$BLOAT not found (already clean?)"
    fi

    if pkill -f stress-ng 2>/dev/null; then
        success "Killed stress-ng processes"
    else
        warn "No stress-ng processes found"
    fi
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}linuxai demo — problem seeder${RESET}"
echo "────────────────────────────────────"

case "$MODE" in
    disk)
        seed_disk
        ;;
    memory|mem)
        seed_memory
        ;;
    clean|reset)
        seed_clean
        ;;
    all|"")
        seed_disk
        echo ""
        seed_memory
        ;;
    *)
        err "Unknown mode: ${MODE}"
        echo "Usage: seed_problems [disk|memory|clean|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${BOLD}Done. You can now run:${RESET}"
echo "  linuxai \"why is my disk full?\""
echo "  linuxai \"my system is running slow\""
echo "  linuxai \"find all log files in /var\""
echo ""
