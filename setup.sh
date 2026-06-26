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

# Ollama models
echo "[4/4] Pulling Ollama models..."
if command -v ollama &>/dev/null; then
    echo "  Pulling qwen2.5:7b (screener — fast filter)..."
    ollama pull qwen2.5:7b
    echo "  Pulling qwen2.5:72b (analyst — this may take 20-60 min on first run)..."
    ollama pull qwen2.5:72b
    echo "  Models ready."
else
    echo "  WARNING: ollama not found in PATH."
    echo "  Install Ollama from https://ollama.ai, then run:"
    echo "    ollama pull qwen2.5:7b"
    echo "    ollama pull qwen2.5:72b"
fi

echo ""
echo "=== Helios setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your real credentials (Supabase, Telegram, APIs)"
echo "  2. Ensure ANALYST_MODEL=qwen2.5:72b in .env (not 3b)"
echo "  3. Start TWS or IB Gateway with API connections enabled (port 7497 for paper)"
echo "  4. Run: bash launch.sh"
