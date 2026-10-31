#!/usr/bin/env bash
# Blocks until TWS's API port is accepting connections, or exits non-zero
# after the timeout — IBC's headless login (Xvfb + Java UI automation) can
# genuinely take 30-90s, sometimes longer on a cold boot while the GPU
# drivers/Ollama are also warming up, so a fixed sleep before starting
# helios.service would be either too short (helios starts before IBKR is
# ready) or wastefully long every single day.
#
# Reads TRADING_ENV/IB_HOST/IB_PORT straight from .env (not OS environment
# variables -- systemd never actually set any here, so this always
# defaulted to paper's 7497 regardless of which account was actually
# configured). Found 2026-08-16 the night before enabling live: this would
# have waited on paper's TWS at the next automatic restart while main.py
# itself connected to whatever .env's TRADING_ENV actually said, a mismatch
# that would have made the pre-flight check meaningless at best and, if
# paper's TWS needed a 2FA scan nobody was there to provide, made this
# script time out and helios-local.service fail to start at all.
set -euo pipefail

# The repo root .env, found from this script's own location (deploy/systemd/).
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"

_env_var() {
    # Last matching line wins, matching how python-dotenv/Python's own
    # os.getenv(default) reads a plain KEY=VALUE line -- strips inline
    # comments and surrounding whitespace/quotes. `|| true` at the end is
    # required, not decorative: grep exits 1 on no match (the normal case
    # for IB_PORT/IB_CLIENT_ID now that they're deliberately absent from
    # .env, letting them derive from TRADING_ENV instead), and under this
    # script's `set -e`, an unguarded failing pipeline in a plain
    # assignment kills the whole script immediately -- before it ever
    # reaches the "${VAR:-default}" fallback line below. Caught by actually
    # running this script standalone before touching the systemd unit.
    grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | tail -n1 | cut -d'=' -f2- | sed -E 's/[[:space:]]*(#.*)?$//; s/^"(.*)"$/\1/' || true
}

TRADING_ENV="$(_env_var TRADING_ENV)"
TRADING_ENV="${TRADING_ENV:-paper}"

if [[ "$TRADING_ENV" == "live" ]]; then
    DEFAULT_PORT=7496
else
    DEFAULT_PORT=7497
fi

HOST="$(_env_var IB_HOST)"
HOST="${HOST:-127.0.0.1}"
PORT="$(_env_var IB_PORT)"
PORT="${PORT:-$DEFAULT_PORT}"
TIMEOUT_S="${1:-180}"

echo "wait-for-ibkr: TRADING_ENV=${TRADING_ENV} -> waiting for ${HOST}:${PORT}"

elapsed=0
until (echo > "/dev/tcp/${HOST}/${PORT}") 2>/dev/null; do
    if (( elapsed >= TIMEOUT_S )); then
        echo "wait-for-ibkr: timed out after ${TIMEOUT_S}s waiting for ${HOST}:${PORT}" >&2
        exit 1
    fi
    sleep 3
    elapsed=$((elapsed + 3))
done
echo "wait-for-ibkr: ${HOST}:${PORT} is accepting connections after ${elapsed}s"
