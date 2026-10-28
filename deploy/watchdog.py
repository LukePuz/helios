"""
External liveness watchdog — runs as its own systemd timer, independent of
main.py and Celery, so it can still notice and report a failure that takes
those processes down with it.

check_ingestion_health (tasks.py) is the only other health check in the
system, and it runs AS a Celery task — if the worker process dies, the
thing that would have alerted about it dies too. This script is the
external check: it verifies the managed processes are actually alive
(PID-reuse-safe, via process_manager's own liveness check) and that the
ops worker is still producing fresh portfolio snapshots, then posts
straight to the Telegram API — no dependency on anything else in the app
being healthy.

Intended to run every 10 minutes, 05:05-18:55 ET on weekdays (the awake
window) via helios-watchdog.timer.

With --heal it also RESTARTS helios-local.service on total failure rather
than only reporting it, which is the whole point: during the 2026-10-19 ->
10-24 outage this watchdog alerted accurately every 30 minutes for five
days and nothing could act on it. See the self-healing constants below for
the flap brakes.
"""
import re
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))
load_dotenv(dotenv_path=_BASE_DIR / ".env")

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Services whose absence means the trading pipeline itself is down —
# catalyst_service is informational, not load-bearing, so it's excluded.
_CRITICAL_SERVICES = ["celery_worker", "celery_llm_worker", "celery_beat"]

# take_portfolio_snapshot runs every 5 minutes during market hours; anything
# much older than that means the ops worker has stopped processing tasks.
_SNAPSHOT_STALE_MINUTES = 15


_ALERT_THROTTLE_MINUTES = 30

# --- Self-healing (--heal) ---------------------------------------------------
# The 2026-10-19 -> 10-24 outage is the reason this exists. Detection was
# never the problem: this watchdog alerted correctly every 30 minutes for
# five straight days (celery down, snapshots 6800+ min stale, 7496
# unreachable). Nothing in the system could ACT on the alert, and the human
# didn't. So the gap being closed here is action, not diagnosis.
_HELIOS_SERVICE = "helios-local.service"

# Must be longer than helios-watchdog.timer's 10-minute period, so a restart
# always gets at least one full check cycle to be judged before another is
# considered -- otherwise a service that takes a while to come up would get
# restarted on top of itself.
_HEAL_COOLDOWN_MINUTES = 15

# Hard ceiling on restarts in a rolling 24h. Past this the watchdog stops
# trying and says so: a service that needs a 5th restart in a day has a
# problem restarting won't fix, and silent flapping is worse than an outage
# because it hides the cause. Mirrors the intent of systemd's StartLimit,
# which this deliberately sits above rather than fights (the unit's own
# limit is 10-in-1800s).
_HEAL_MAX_PER_DAY = 4

# Deliberately a FILE, not db.set_state: this watchdog's entire purpose is
# to work when the app is broken, and Postgres being unreachable is one of
# the ways it breaks. A DB-backed throttle would fail open exactly when
# healing matters most, and fail-open on a restart loop is the one failure
# mode worth engineering against here.
_HEAL_STATE_FILE = _BASE_DIR / "logs" / ".watchdog_heal_history.json"


def _send_telegram(text: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        print(f"[watchdog] No Telegram credentials configured — would have sent: {text}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[watchdog] Telegram send failed: " + re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot[REDACTED]", str(e)))


def _alert_due(key: str = "watchdog_alert_at") -> bool:
    """Re-fired on every 10-minute check with no throttling at all — an IBC
    outage on 2026-07-18 (TWS's own nightly restart wasn't picked back up,
    see ibc.service's Restart=always fix) produced repeated near-identical
    "N min old" alerts before that root cause was fixed. Throttle re-alerts
    of a still-unresolved problem to once per _ALERT_THROTTLE_MINUTES;
    a problem clearing and recurring later still alerts immediately next
    time since the throttle key naturally stops being touched once things
    are healthy again."""
    try:
        import db
        from datetime import datetime as _dt, timezone as _tz
        last = db.get_state(key)
        now = _dt.now(_tz.utc)
        if last:
            try:
                if (now - _dt.fromisoformat(last)).total_seconds() < _ALERT_THROTTLE_MINUTES * 60:
                    return False
            except ValueError:
                pass
        db.set_state(key, now.isoformat())
        return True
    except Exception as e:
        print(f"[watchdog] Throttle check failed, alerting anyway: {e}")
        return True


def _check_processes() -> list[str]:
    import process_manager as pm

    pids = pm._load()
    problems = []
    for name in _CRITICAL_SERVICES:
        entry = pids.get(name)
        if entry is None or not pm._alive(entry):
            problems.append(f"{pm.SERVICES.get(name, {}).get('label', name)} is not running")
    return problems


def _ibkr_port_reachable(host: str | None = None, port: int | None = None,
                         timeout: float = 3.0) -> bool:
    """Cheap TCP-connect probe against the Gateway/TWS API port -- no API
    handshake, just "is anything listening." Same check launch.sh does
    before startup. Deliberately doesn't try a real ib_async connection
    here: this script has to stay independent of the rest of the app being
    healthy (see module docstring), and a full API login is slow and has
    its own failure modes that would muddy a simple reachability check.

    Defaults to the env-appropriate host/port from config (7496 live /
    7497 paper). Before 2026-10-18 this defaulted to a hardcoded 7497 and
    silently probed the paper port even after the system went live on 7496
    -- so the one diagnostic meant to tell "broker down" from "worker
    stuck" was itself pointed at the wrong port in live mode."""
    from config import IB_HOST, IB_PORT
    if host is None:
        host = IB_HOST
    if port is None:
        port = IB_PORT
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _service_is_active(service: str) -> bool:
    """`systemctl is-active <service>` == 'active'. Read-only query, needs
    no privilege. Any failure (systemctl missing, timeout) is treated as
    'not active' so this can never be the thing that wedges the watchdog."""
    try:
        out = subprocess.run(["systemctl", "is-active", service],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "active"
    except Exception as e:
        print(f"[watchdog] systemctl is-active {service} failed: {e}")
        return False


def _gateway_service_name() -> str:
    from config import TRADING_ENV
    return "ibc-live.service" if TRADING_ENV == "live" else "ibc.service"


def _check_gateway_login() -> list[str]:
    """Catches the exact silent failure that cost a full weekend
    (2026-10-16 Fri eve -> 2026-10-18 Sun eve): IBKR forces a re-login
    roughly weekly (often Sunday), IBC fires a 2FA push, nobody approves
    it, and IBC gives up -- it does NOT retry after a second-factor
    timeout ("Re-login after second factor authentication timeout not
    required" in its log). The Gateway then sits on the login screen with
    its API port closed, while systemd still reports ibc-live.service as
    active (running). Nothing else noticed: the port was dead for ~2 days.

    Deliberately conditional on the gateway service being active, so it
    stays silent whenever the gateway is intentionally stopped (overnight
    suspend, manual stop) and never false-alarms -- it fires ONLY on the
    up-but-not-listening state, which always means a login stuck on 2FA.
    Safe to run 7 days a week for that reason; see
    helios-gateway-watchdog.timer."""
    from config import IB_PORT, TRADING_ENV
    service = _gateway_service_name()
    if not _service_is_active(service):
        return []   # intentionally down -> nothing to alert on
    if _ibkr_port_reachable():
        return []   # up and listening -> healthy
    return [f"IBKR gateway ({service}, {TRADING_ENV}) is running but its API "
            f"port {IB_PORT} is NOT listening — almost certainly a login stuck "
            f"on 2FA. Approve the second-factor prompt, or restart {service} "
            f"to re-fire it."]


def _check_snapshot_freshness() -> list[str]:
    try:
        import db
        conn = db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT timestamp FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1")
                row = cur.fetchone()
        finally:
            db._put_conn(conn)
    except Exception as e:
        return [f"Could not query portfolio_snapshots: {e}"]

    if row is None:
        return ["No portfolio snapshots exist yet"]

    age_minutes = (datetime.now(timezone.utc) - row[0]).total_seconds() / 60
    if age_minutes > _SNAPSHOT_STALE_MINUTES:
        # Incident 2026-08-03: TWS hung (application-level -- no crash, no
        # systemd restart, port 7497 stopped accepting connections) for 9+
        # hours after a routine paper-trading reminder dialog. This message
        # used to blame "the ops worker" unconditionally in that case,
        # which is wrong and sent whoever's triaging looking in the wrong
        # place (Celery, not TWS) for 39 separate alerts that day. A quick
        # reachability probe against TWS's own port tells the two apart --
        # the ops worker itself was never actually stuck, every task was
        # failing fast on a broker connection that wasn't there.
        if not _ibkr_port_reachable():
            from config import IB_PORT
            return [f"Last portfolio snapshot is {age_minutes:.0f} min old "
                    f"(expected every 5 min) — IBKR/TWS is NOT reachable on "
                    f"port {IB_PORT} (likely hung or down, not just the ops worker)"]
        return [f"Last portfolio snapshot is {age_minutes:.0f} min old "
                f"(expected every 5 min) — IBKR/TWS port is reachable, so "
                f"the ops worker itself may be stuck or dead"]
    return []


def _heal_history() -> list[datetime]:
    """Restart timestamps from the last 24h, oldest first. Unreadable or
    corrupt state is treated as an empty history rather than an error: the
    cooldown below still bounds how fast restarts can happen, so the worst
    case of a lost history file is one extra restart, not a loop."""
    import json
    try:
        raw = json.loads(_HEAL_STATE_FILE.read_text())
    except Exception:
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - 86400
    out = []
    for s in raw if isinstance(raw, list) else []:
        try:
            ts = datetime.fromisoformat(s)
            if ts.timestamp() >= cutoff:
                out.append(ts)
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _heal_history_record(ts: datetime) -> None:
    import json
    hist = _heal_history() + [ts]
    try:
        _HEAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HEAL_STATE_FILE.write_text(json.dumps([t.isoformat() for t in hist]))
    except Exception as e:
        print(f"[watchdog] Could not persist heal history: {e}")


def _heal_allowed() -> tuple[bool, str]:
    """(allowed, reason-if-not). Two independent brakes: a cooldown so
    restarts can't stack, and a 24h ceiling so a service that can't stay up
    escalates to a human instead of being restarted forever."""
    hist = _heal_history()
    if len(hist) >= _HEAL_MAX_PER_DAY:
        return False, (f"already restarted {len(hist)}× in the last 24h "
                       f"(limit {_HEAL_MAX_PER_DAY}) — NOT restarting again, "
                       f"this needs a human")
    if hist:
        age_min = (datetime.now(timezone.utc) - hist[-1]).total_seconds() / 60
        if age_min < _HEAL_COOLDOWN_MINUTES:
            return False, (f"last restart was {age_min:.0f} min ago "
                           f"(cooldown {_HEAL_COOLDOWN_MINUTES} min) — "
                           f"giving it time to come up")
    return True, ""


def _sudo_systemctl(verb: str, unit: str, timeout: float = 90.0) -> tuple[bool, str]:
    """Privileged systemctl via the NOPASSWD rules in
    /etc/sudoers.d/helios-claude. Those rules match the command line
    LITERALLY, so the unit name must carry its .service suffix exactly as
    written there -- `reset-failed helios-local` (no suffix) is a different
    command to sudo and prompts for a password, which in a non-interactive
    timer means it just hangs or fails. Hence the explicit absolute
    /usr/bin/systemctl and full unit name."""
    cmd = ["sudo", "-n", "/usr/bin/systemctl", verb, unit]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        err = (p.stderr or p.stdout or "").strip()
        return p.returncode == 0, err
    except subprocess.TimeoutExpired:
        return False, f"`systemctl {verb} {unit}` timed out after {timeout:.0f}s"
    except Exception as e:
        return False, f"`systemctl {verb} {unit}` failed to run: {e}"


def _attempt_heal(trigger: str) -> list[str]:
    """Restart helios-local and report what happened. Returns lines for the
    Telegram notice -- healing is ALWAYS announced, never silent, and is
    never routed through the alert throttle: a restart is an event, not a
    repeating condition, and the one thing worse than an outage is a system
    quietly restarting itself all day without telling anyone."""
    import time

    allowed, why_not = _heal_allowed()
    if not allowed:
        return [f"⛔️ Self-heal declined: {why_not}"]

    now = datetime.now(timezone.utc)
    _heal_history_record(now)
    attempt_n = len(_heal_history())
    notes = [f"🔧 Self-heal triggered ({trigger}) — restart "
             f"{attempt_n}/{_HEAL_MAX_PER_DAY} in 24h"]

    # reset-failed FIRST and unconditionally. On 2026-10-24 helios-local sat
    # at NRestarts=5 against StartLimitBurst=5, so a bare `start` is refused
    # outright with "Start request repeated too quickly" -- which looks
    # exactly like a successful no-op in a script that doesn't check. This
    # is the step that makes the restart actually possible.
    ok, err = _sudo_systemctl("reset-failed", _HELIOS_SERVICE)
    if not ok:
        notes.append(f"• reset-failed failed: {err or 'unknown error'}")

    ok, err = _sudo_systemctl("start", _HELIOS_SERVICE)
    if not ok:
        notes.append(f"• ❌ start failed: {err or 'unknown error'}")
        notes.append("• Helios is still DOWN — manual intervention needed")
        return notes

    # Don't claim success off systemd's exit code alone. The unit's
    # ExecStartPre waits for the broker (non-fatal since 2026-10-24), and
    # main.py then forks the Celery workers, so "started" and "actually
    # running the pipeline" are seconds apart. Poll until the unit settles.
    for _ in range(12):
        time.sleep(5)
        if _service_is_active(_HELIOS_SERVICE):
            break

    if not _service_is_active(_HELIOS_SERVICE):
        notes.append(f"• ❌ {_HELIOS_SERVICE} did not reach active within 60s "
                     f"— manual intervention needed")
        return notes

    still_dead = _check_processes()
    if still_dead:
        notes.append(f"• ⚠️ {_HELIOS_SERVICE} is active but workers are still "
                     f"missing: " + "; ".join(still_dead))
    else:
        notes.append(f"• ✅ {_HELIOS_SERVICE} is active and all critical "
                     f"workers are running")
    return notes


def _emit(problems: list[str], throttle_key: str) -> None:
    if problems:
        msg = "🚨 <b>Helios watchdog alert</b>\n" + "\n".join(f"• {p}" for p in problems)
        if _alert_due(throttle_key):
            print(f"[watchdog] {msg}")
            _send_telegram(msg)
        else:
            print(f"[watchdog] (throttled, not re-sent) {msg}")
    else:
        print("[watchdog] OK")


def main() -> None:
    # --gateway-only: just the "gateway up but API port dead" check. Runs on
    # its own 7-day/5-min timer (helios-gateway-watchdog.timer) so the
    # weekend 2FA-stuck-login failure gets caught, without dragging in the
    # pipeline checks below -- those are expected to be down outside the
    # weekday trading window and would false-alarm there. Its own throttle
    # key so a gateway alert is never swallowed by an unrelated pipeline one.
    gateway_only = "--gateway-only" in sys.argv[1:]
    if gateway_only:
        _emit(_check_gateway_login(), "watchdog_gateway_alert_at")
        return

    # Full check: scoped by helios-watchdog.timer to weekday market hours,
    # when the whole stack is expected up. The gateway-login check is left
    # to the dedicated gateway timer above, so it isn't duplicated here.
    #
    # --heal is opt-in rather than default-on so that running this script by
    # hand to diagnose something can never restart live trading as a side
    # effect; helios-watchdog.service passes it, nothing else does.
    heal = "--heal" in sys.argv[1:]

    service_down = not _service_is_active(_HELIOS_SERVICE)
    proc_problems = _check_processes()
    problems = ([f"{_HELIOS_SERVICE} is not active"] if service_down else []) \
        + proc_problems + _check_snapshot_freshness()

    # Restart only on unambiguous total failure -- the unit itself down, or
    # every critical worker gone. A PARTIALLY degraded stack is left alone
    # and merely alerted on: bouncing the whole service to recover one
    # worker would tear down the others mid-task, and a half-up system is
    # the state most likely to have something in flight worth not killing.
    total_failure = service_down or len(proc_problems) == len(_CRITICAL_SERVICES)

    heal_notes: list[str] = []
    if heal and total_failure:
        trigger = (f"{_HELIOS_SERVICE} not active" if service_down
                   else "all critical workers dead")
        heal_notes = _attempt_heal(trigger)

    _emit(problems, "watchdog_alert_at")

    # Sent outside _emit's throttle on purpose: the alert about the outage
    # may well be suppressed as a repeat, but the fact that the watchdog
    # took action must always arrive.
    if heal_notes:
        msg = "🔧 <b>Helios self-heal</b>\n" + "\n".join(heal_notes)
        print(f"[watchdog] {msg}")
        _send_telegram(msg)


if __name__ == "__main__":
    main()
