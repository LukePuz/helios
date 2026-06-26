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
        ollama serve &>/dev/null &
        sleep 2
        echo "Ollama: started"
    else
        echo "Ollama: already running"
    fi
else
    echo "WARNING: ollama not found — AI agents will use keyword fallback."
fi

echo ""
echo "Starting Helios..."
echo "  Dashboard -> http://localhost:8501"
echo "  Logs      -> trading_bot.log"
echo "  Press Ctrl+C to stop."
echo ""

python main.py
