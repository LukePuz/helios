#!/usr/bin/env bash
set -euo pipefail

echo "=== Helios Hedgefund — Linux Setup ==="

# Python virtual environment
echo "[1/4] Creating Python venv..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt

# .env bootstrap
echo "[2/4] Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
    echo "  IMPORTANT: Edit .env and fill in your credentials before running."
else
    echo "  .env already exists — skipping copy."
fi

# Log directory
echo "[3/4] Creating log directory..."
mkdir -p logs

# Ollama models — read from .env if it already specifies models (so this
# doesn't pull a 72b model when the deployment is actually configured for a
# smaller one, or vice versa), falling back to the config.py defaults.
echo "[4/4] Pulling Ollama models..."
SCREENER_MODEL_WANTED="qwen2.5:7b"
ANALYST_MODEL_WANTED="qwen2.5:72b"
if [ -f .env ]; then
    _env_screener=$(grep -E '^SCREENER_MODEL=' .env | tail -1 | cut -d= -f2-)
    _env_analyst=$(grep -E '^ANALYST_MODEL=' .env | tail -1 | cut -d= -f2-)
    [ -n "$_env_screener" ] && SCREENER_MODEL_WANTED="$_env_screener"
    [ -n "$_env_analyst" ] && ANALYST_MODEL_WANTED="$_env_analyst"
fi

if command -v ollama &>/dev/null; then
    echo "  Pulling $SCREENER_MODEL_WANTED (screener — fast filter)..."
    ollama pull "$SCREENER_MODEL_WANTED"
    echo "  Pulling $ANALYST_MODEL_WANTED (analyst — larger models may take 20-60 min on first run)..."
    ollama pull "$ANALYST_MODEL_WANTED"
    echo "  Models ready."
else
    echo "  WARNING: ollama not found in PATH."
    echo "  Install Ollama from https://ollama.ai, then run:"
    echo "    ollama pull $SCREENER_MODEL_WANTED"
    echo "    ollama pull $ANALYST_MODEL_WANTED"
fi

echo ""
echo "=== Helios setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your real credentials (Supabase, Telegram, APIs)"
echo "  2. Set ANALYST_MODEL/SCREENER_MODEL/CHAT_MODEL in .env to match what your"
echo "     hardware can run (qwen2.5:72b gives the best reasoning quality but needs"
echo "     a lot of RAM/VRAM; qwen2.5:3b or :7b are lighter, faster fallbacks)"
echo "  3. Start TWS or IB Gateway with API connections enabled (port 7497 for paper)"
echo "  4. Run: bash launch.sh"
