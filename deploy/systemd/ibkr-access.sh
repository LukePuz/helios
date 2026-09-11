#!/usr/bin/env bash
# Manual-access helper for the IBKR gateways + the VNC path to 2FA screens
# (2026-09-11, user request) -- everything this session did by hand,
# repeatedly, across several days of live-gateway 2FA/session-conflict
# troubleshooting, wrapped into one script instead of re-typing it.
#
# Requires the NOPASSWD sudoers rules added 2026-09-07
# (/etc/sudoers.d/helios-claude):
#   systemctl start/stop ibc-live.service
#   systemctl start/stop ibc.service
#   systemctl start helios-local.service
# There is deliberately NO passwordless `stop helios-local` or `restart
# ibc-*` rule -- keep it that way; this script only uses what's granted.
#
# IMPORTANT, learned the hard way this session: a `gateways-down` while a
# gateway is ALREADY logged in and working kills that working session --
# check `status` first if you're not sure whether one is currently up.
#
# Also learned this session: `ibc.service` (paper) and `ibc-live.service`
# log in with the SAME IBKR username. Freeing up mobile/Client-Portal
# access for that account means stopping BOTH, not just live -- this
# script's gateways-up/gateways-down always act on both together for
# that reason.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ibkr-access.sh <command>

  status          Show both gateways, both API ports, x11vnc, and helios-local
  gateways-up     Start ibc-live.service + ibc.service (both need fresh 2FA)
  gateways-down   Stop both -- use this before you need mobile/Client Portal
                  access, since paper and live share one IBKR username and
                  either one being logged in can block the other session
  vnc             Start x11vnc on live's display (:98), Wayland-safe, print
                  the SSH tunnel + viewer commands to actually see the 2FA
                  screen from your own machine
  vnc-paper       Same, for paper's display (:99)
  helios-up       Start helios-local.service (only useful once both
                  gateways are already logged in -- its own pre-start check
                  waits up to 180s for the live gateway's API port)

Typical flow after a reboot or a stuck 2FA:
  1. ./ibkr-access.sh gateways-up
  2. ./ibkr-access.sh vnc                 # then follow the printed steps
  3. approve the 2FA push/code on your phone within ~180s
  4. ./ibkr-access.sh status              # confirm both ports are listening
  5. ./ibkr-access.sh helios-up
EOF
}

status() {
    # `systemctl status` exits non-zero for a perfectly normal "inactive"
    # or "failed" state (3), and `pgrep`/`grep` exit non-zero on no match
    # -- all expected, informational outcomes here, not script errors, so
    # every one of these is explicitly allowed to fail under `set -e`.
    echo "=== ibc-live.service (live gateway, :98) ==="
    systemctl status ibc-live.service --no-pager 2>&1 | head -4 || true
    echo
    echo "=== ibc.service (paper gateway, :99) ==="
    systemctl status ibc.service --no-pager 2>&1 | head -4 || true
    echo
    echo "=== API ports ==="
    ss -tlnp 2>/dev/null | grep -E ":7496|:7497" || echo "  (neither 7496 nor 7497 listening)"
    echo
    echo "=== x11vnc ==="
    pgrep -af x11vnc || echo "  (not running)"
    echo
    echo "=== helios-local.service ==="
    systemctl status helios-local.service --no-pager 2>&1 | head -4 || true
}

gateways_up() {
    echo "Starting ibc-live.service and ibc.service -- both will need a" \
         "fresh 2FA approval on your phone (read-only login isn't honored" \
         "by Gateway for this account, see config.ini's ExistingSession" \
         "notes)."
    sudo -n systemctl start ibc-live.service
    sudo -n systemctl start ibc.service
    sleep 3
    status
}

gateways_down() {
    echo "Stopping ibc-live.service and ibc.service (same IBKR username --" \
         "stopping both, not just one, to actually free the account for" \
         "mobile/Client Portal)."
    sudo -n systemctl stop ibc-live.service
    sudo -n systemctl stop ibc.service
    sleep 2
    status
}

start_vnc() {
    local display="${1:-:98}"
    echo "Starting x11vnc on display ${display}..."
    # The plain `x11vnc -display :98` fails here every time with "Wayland
    # display server detected. Exiting." -- it's a false positive: x11vnc
    # is picking up WAYLAND_DISPLAY/XDG_SESSION_TYPE from this LOGIN
    # SHELL's own desktop session, not from the target X11-only Xvfb
    # display it's actually being asked to serve. Unsetting those two vars
    # for just this command fixes it every time.
    env -u WAYLAND_DISPLAY -u XDG_SESSION_TYPE XAUTHORITY= \
        x11vnc -display "$display" -localhost -nopw -forever &
    sleep 2
    if ! pgrep -x x11vnc >/dev/null 2>&1 && ! pgrep -f "x11vnc -display ${display}" >/dev/null 2>&1; then
        echo "x11vnc doesn't appear to have started -- check the output above." >&2
        return 1
    fi
    cat <<EOF

x11vnc is up on ${display}, bound to localhost only (secure by default --
reachable only via a tunnel, not directly over the network).

From YOUR OWN machine, run:
  ssh -L 5900:localhost:5900 ${USER}@$(hostname)
  vncviewer localhost:5900

Then approve the 2FA prompt on your phone within its timeout window
(SecondFactorAuthenticationTimeout=180s per config.ini).
EOF
}

helios_up() {
    echo "Starting helios-local.service (its own pre-start check waits up" \
         "to 180s for the live gateway's API port to accept connections)."
    sudo -n systemctl start helios-local.service
    sleep 3
    systemctl status helios-local.service --no-pager 2>&1 | head -8
}

case "${1:-}" in
    status)        status ;;
    gateways-up)   gateways_up ;;
    gateways-down) gateways_down ;;
    vnc)           start_vnc ":98" ;;
    vnc-paper)     start_vnc ":99" ;;
    helios-up)     helios_up ;;
    *)             usage; exit 1 ;;
esac
