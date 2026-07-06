# Running Helios as a systemd service (Linux)

This is the recommended way to run the bot unattended on a Linux server —
`launch.sh` is better suited to interactive/manual runs (you see the logs
directly, Ctrl+C stops everything).

## Install

```bash
# 1. Deploy the code (adjust the path to wherever you actually put it)
sudo mkdir -p /opt/helios
sudo cp -r . /opt/helios
cd /opt/helios
bash setup.sh          # creates venv, installs deps, copies .env.example -> .env

# 2. Edit /opt/helios/.env with real credentials, then:

# 3. Edit deploy/systemd/helios.service:
#      - WorkingDirectory / ExecStart paths if not /opt/helios
#      - User=%i should be a real, non-root user that owns /opt/helios

# 4. Install the unit
sudo cp deploy/systemd/helios.service /etc/systemd/system/helios.service
sudo systemctl daemon-reload
sudo systemctl enable --now helios.service
```

## Operate

```bash
sudo systemctl status helios      # is it running
sudo systemctl restart helios     # restart the whole stack
sudo systemctl stop helios
journalctl -u helios -f           # follow systemd's own stdout/stderr capture
tail -f /opt/helios/helios.log    # the bot's own application log
```

## Notes

- Redis: if running locally (not a remote `REDIS_URL`), install and enable
  it separately (`sudo apt install redis-server && sudo systemctl enable --now redis-server`)
  — `main.py`'s `_try_start_redis()` will detect and use it automatically,
  same as on Windows.
- Ollama: install per [ollama.ai](https://ollama.ai) and either enable its
  own systemd service (the installer typically does this) or add
  `Requires=ollama.service` / `After=ollama.service` to `helios.service` if
  you gave it a unit name.
- TWS / IB Gateway: there's no universal Linux equivalent of `launch.bat`'s
  auto-launch step. Most unattended Linux deployments run IB Gateway
  headless via [IBC](https://github.com/IbcAlpha/IBC) in its own systemd
  unit or container — set that up separately and make sure it's listening
  on the port in `.env` (`IB_PORT`) before Helios starts placing orders.
