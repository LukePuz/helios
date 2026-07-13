#!/usr/bin/env bash
# Blocks until TWS's API port is accepting connections, or exits non-zero
# after the timeout — IBC's headless login (Xvfb + Java UI automation) can
# genuinely take 30-90s, sometimes longer on a cold boot while the GPU
# drivers/Ollama are also warming up, so a fixed sleep before starting
# helios.service would be either too short (helios starts before IBKR is
# ready) or wastefully long every single day.
set -euo pipefail

HOST="${IB_HOST:-127.0.0.1}"
PORT="${IB_PORT:-7497}"
TIMEOUT_S="${1:-180}"

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
