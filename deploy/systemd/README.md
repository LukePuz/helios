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

## Fully unattended: wake, login, run, sleep on a schedule

For a machine that should power itself up premarket, log into TWS with no
one at the keyboard, run the bot through the session, and put itself back
to sleep a few hours after close — see `install-auto-schedule.sh` in this
directory. It wires together:

- `xvfb.service` / `ibc.service` — [IBC](https://github.com/IbcAlpha/IBC)
  driving TWS on a virtual display (`:1`), so login happens with no real
  desktop session involved. **IBC is a third-party tool, not from IBKR** —
  it works by typing your credentials into the TWS login dialog on IBC's
  behalf, which means those credentials live in `~/ibc/config.ini`
  (`chmod 600`, outside this repo, never commit it). If your account
  requires 2FA (a phone-tap push notification), this cannot be automated —
  IBC will hang waiting for a factor no script can provide.
- `helios-local.service` — waits (`wait-for-ibkr.sh`) for TWS's API port to
  actually accept connections before starting `main.py`, since IBC's login
  can take well over a minute.
- `helios-start.timer` / `helios-evening-stop.timer` — the daily schedule.
  The start timer's `WakeSystem=true` would arm the machine's RTC to wake
  it from suspend at the scheduled time if the machine ever actually
  suspended — in practice (verified 2026-07-26 via `journalctl`) it
  doesn't: the evening timer only stops `helios-local`/`ibc`/`xvfb` and
  hands off to `helios-standby.service`, a lightweight listener that keeps
  the machine reachable over Telegram overnight. `WakeSystem=true` is
  harmless defensive config, not something this setup currently relies on.

Edit the two `OnCalendar=` times in `helios-start.timer` /
`helios-evening-stop.timer` before installing if the defaults (5:00 AM /
7:00 PM America/New_York) don't fit. The evening timer only stops the
trading stack, not the whole machine — see `helios-evening-stop.service`'s
own comments — but anyone using it for something else
at that hour will get interrupted.
