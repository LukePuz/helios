#!/usr/bin/env bash
# One-time install for the auto-wake / auto-start / auto-stop schedule.
# Run this yourself (needs sudo, which needs an interactive terminal):
#   bash deploy/systemd/install-auto-schedule.sh
set -euo pipefail

echo "[1/3] Installing xvfb (virtual display for headless TWS)..."
sudo apt-get update -qq
sudo apt-get install -y xvfb

echo "[2/3] Installing systemd units..."
sudo cp ~/ibc/xvfb.service.template            /etc/systemd/system/xvfb.service
sudo cp ~/ibc/ibc.service.template             /etc/systemd/system/ibc.service
# Unit files in the repo say YOUR_USER; fill in the account that runs Helios.
sed "s/YOUR_USER/$USER/g" deploy/systemd/helios-local.service | sudo tee /etc/systemd/system/helios-local.service >/dev/null
sudo cp deploy/systemd/helios-evening-stop.service  /etc/systemd/system/helios-evening-stop.service
sudo cp deploy/systemd/helios-start.timer           /etc/systemd/system/helios-start.timer
sudo cp deploy/systemd/helios-evening-stop.timer    /etc/systemd/system/helios-evening-stop.timer

echo "[3/3] Enabling timers (services themselves stay disabled; only the timers trigger them)..."
sudo systemctl daemon-reload
sudo systemctl enable --now helios-start.timer
sudo systemctl enable --now helios-evening-stop.timer

echo ""
echo "Done. Next steps (see the chat for the full explanation):"
echo "  1. Fill in your IBKR credentials: nano ~/ibc/config.ini  (IbLoginId= / IbPassword=)"
echo "  2. Do one supervised test: sudo systemctl start ibc.service && journalctl -u ibc -f"
echo "  3. Check the schedule:     systemctl list-timers helios-*"
