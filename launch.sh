#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ ! -f venv/bin/activate ]; then
    echo "ERROR: venv not found. Run 'bash setup.sh' first."
    exit 1
fi
source venv/bin/activate

echo "=== Helios Hedgefund ==="

# --- Redis ---
# Try systemctl first (installed Redis service), then native binary, then Docker
if systemctl is-active --quiet redis 2>/dev/null || \
   systemctl is-active --quiet redis-server 2>/dev/null; then
    echo "Redis: running (systemd)"
elif command -v redis-server &>/dev/null; then
    redis-server --daemonize yes
    echo "Redis: started (redis-server)"
elif command -v docker &>/dev/null; then
    docker start redis 2>/dev/null || \
    docker run -d --name redis -p 6379:6379 redis:alpine
    echo "Redis: started (Docker)"
else
    echo "WARNING: Could not start Redis automatically."
    echo "  If Celery fails, run: sudo systemctl start redis  OR  docker start redis"
fi

# --- Ollama ---
if command -v ollama &>/dev/null; then
    if ! pgrep -x ollama &>/dev/null; then
        mkdir -p logs
        ollama serve &>>logs/ollama.log &
        sleep 2
        echo "Ollama: started"
    else
        echo "Ollama: already running"
    fi
else
    echo "WARNING: ollama not found — AI agents will use keyword fallback."
fi

# --- IBKR TWS / IB Gateway ---
# Unlike Windows (launch.bat launches a known tws.exe path), there's no
# single canonical binary/location for TWS or IB Gateway on Linux — it's
# typically run headless via IBC (github.com/IbcAlpha/IBC) or in a VNC
# session, and the setup varies enough per-user that auto-launching it here
# would be more likely to guess wrong than to help. Start it yourself before
# (or after) this script — the bot retries the connection on its own.
if ! (echo > /dev/tcp/127.0.0.1/7497) 2>/dev/null; then
    echo "NOTE: Nothing is listening on port 7497 (paper) yet — start TWS or"
    echo "      IB Gateway with API connections enabled before approving trades."
fi

echo ""
echo "Starting Helios..."
echo "  Dashboard -> http://localhost:8501"
echo "  Logs      -> helios.log"
echo "  Press Ctrl+C to stop."
echo ""

# Best-effort browser open once the dashboard is likely up — skipped
# entirely on a headless server (no DISPLAY, no xdg-open) rather than
# erroring out.
if command -v xdg-open &>/dev/null && [ -n "${DISPLAY:-}" ]; then
    ( sleep 14 && xdg-open "http://localhost:8501" &>/dev/null & )
fi

python main.py
