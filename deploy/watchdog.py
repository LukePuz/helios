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
import time
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
# The orders worker is the only process that can place, move or cancel a
# stop (clientId 30), so a dead one means stop changes silently stop
# happening; screen and ingestion feed the analyst. All are alerted on.
_CRITICAL_SERVICES = ["celery_worker", "celery_llm_worker", "celery_beat",
                      "celery_orders_worker", "celery_screen_worker",
                      "celery_ingestion_worker"]

# --heal restarts the whole unit only when ALL of these are dead (the
# total-failure rule). One dead worker in an otherwise-up stack is alerted,
# not healed: process_manager can't respawn a single child, and bouncing
# helios-local would tear down the live workers mid-task.
_HEAL_TRIGGER_SERVICES = ["celery_worker", "celery_llm_worker", "celery_beat"]

# take_portfolio_snapshot runs every 5 minutes during market hours; anything
# much older than that means the ops worker has stopped processing tasks.
_SNAPSHOT_STALE_MINUTES = 15


_ALERT_THROTTLE_MINUTES = 30
_ALERT_REPEAT_CAP_MINUTES = 240

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


def _say(msg: str) -> None:
    """One stamped line in watchdog.log (it had no timestamps, so alerts
    could only be dated by lining runs up with the systemd journal)."""
    print(f"[watchdog] {datetime.now().astimezone().isoformat(timespec='seconds')} {msg}",
          flush=True)


def _record(text: str, status: str) -> None:
    try:
        import alert_policy
        source = f"deploy/watchdog.py:{sys._getframe(2).f_code.co_name}"
        alert_policy.record(source, text, alert_policy.CRITICAL, "", status)
    except Exception:
        pass


def _post(payload: dict):
    return requests.post(f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
                         json=payload, timeout=10)


def _send_telegram(text: str) -> bool:
    """True only when Telegram accepted the message. A 400 (HTML Telegram
    won't parse, or too long) is resent once as plain text. Never raises."""
    if not _TOKEN or not _CHAT_ID:
        _say(f"No Telegram credentials configured — would have sent: {text}")
        return False
    if len(text) > 4000:
        text = text[:3980] + "\n…(truncated)"
    try:
        resp = _post({"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"})
        if getattr(resp, "status_code", None) == 400:
            import html
            _say("Telegram rejected the HTML (400); resending as plain text")
            resp = _post({"chat_id": _CHAT_ID, "text": html.unescape(re.sub(r"<[^>]+>", "", text))})
        ok = getattr(resp, "ok", True) is not False
        if not ok:
            _say(f"Telegram send rejected: HTTP {getattr(resp, 'status_code', '?')}")
    except Exception as e:
        _say(f"Telegram send failed: " + re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot[REDACTED]", str(e)))
        ok = False
    _record(text, "sent" if ok else "failed")
    return ok


def _problems_fingerprint(problems: list[str]) -> str:
    """The problem set with numbers collapsed: '845 min old' and '905 min
    old' are the same unresolved problem."""
    return "|".join(sorted(re.sub(r"\d+(\.\d+)?", "N", p) for p in problems))


def _repeat_interval_minutes(repeats: int) -> int:
    """30, 60, 120, then every 240 min for an unchanged problem set."""
    return min(_ALERT_THROTTLE_MINUTES * 2 ** max(repeats, 0), _ALERT_REPEAT_CAP_MINUTES)


def _alert_due(key: str = "watchdog_alert_at", fingerprint: str = "") -> bool:
    """Throttle for a still-unresolved problem. The first alert of a problem
    set, and any change to the set, goes out at once; an unchanged set
    re-alerts after 30, 60 and 120 min, then every 4 h. The 10-05..09 and
    10-19..23 outages re-sent the same alert every 30 min, ~23 a day for
    five days each, and the 11-01 Sunday 2FA login 17 times. _emit clears
    the key on a clean run, so a problem that clears and comes back alerts
    at once.

    Stored as JSON {at, fp, n}; a bare ISO time from before is read as the
    same problem with no repeats yet."""
    try:
        import db
        import json
        from datetime import datetime as _dt, timezone as _tz
        raw = db.get_state(key)
        now = _dt.now(_tz.utc)
        rec = None
        if raw:
            try:
                rec = json.loads(raw)
                if not isinstance(rec, dict):
                    raise ValueError
            except ValueError:
                rec = {"at": raw, "fp": fingerprint, "n": 0}
        if rec and rec.get("fp") == fingerprint:
            try:
                elapsed = (now - _dt.fromisoformat(rec["at"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                elapsed = None
            n = int(rec.get("n") or 0)
            # A negative elapsed (clock stepped back, future stamp) is due,
            # so a bad timestamp can't silence the alert.
            if elapsed is not None and 0 <= elapsed < _repeat_interval_minutes(n) * 60:
                return False
            new = {"at": now.isoformat(), "fp": fingerprint, "n": n + 1}
        else:
            new = {"at": now.isoformat(), "fp": fingerprint, "n": 0}
        db.set_state(key, json.dumps(new))
        return True
    except Exception as e:
        _say(f"Throttle check failed, alerting anyway: {e}")
        return True


def _clear_alert_state(key: str) -> None:
    try:
        import db
        if db.get_state(key):
            db.set_state(key, "")
    except Exception as e:
        _say(f"Could not clear {key}: {e}")


def _check_processes(names: list[str] | None = None) -> list[str]:
    import process_manager as pm

    pids = pm._load()
    problems = []
    for name in (_CRITICAL_SERVICES if names is None else names):
        entry = pids.get(name)
        if entry is None or not pm._alive(entry):
            problems.append(f"{pm.SERVICES.get(name, {}).get('label', name)} is not running")
    return problems


# A live PID can still be a hung worker (professor m275). The orders worker
# also has to answer a Celery ping, as main._orders_worker_ready checks at
# startup. Its prefork parent answers while its one child runs a task, so a
# miss means the worker process itself is stuck. Two missed checks in a row
# (~20 min) before alerting, so one slow broker round trip doesn't page;
# never a self-heal trigger. The count is a file, like the heal history.
_ORDERS_PING_TIMEOUT_S = 5.0
_ORDERS_PING_MISSES_TO_ALERT = 2
_ORDERS_PING_STATE_FILE = _BASE_DIR / "logs" / ".watchdog_orders_ping_misses"
_ORDERS_PING_GRACE_S = 10.0
# A count older than this is from an earlier session (last check before the
# night stop, a manual run hours ago) and doesn't make the next miss "in a row".
_ORDERS_PING_STALE_S = 30 * 60
# A ping that can't be sent (the Celery broker, Redis, refusing or not
# answering) said nothing and the check was skipped, so a Redis outage left
# no problem line (audit 2026-11-07 05:07 #5). Counted separately; alert-only.
_ORDERS_PING_UNSENT_STATE_FILE = _BASE_DIR / "logs" / ".watchdog_orders_ping_unsent"


def _ping_orders_worker(timeout: float = _ORDERS_PING_TIMEOUT_S) -> bool | None:
    """True when orders@<host> answers a Celery ping, False when it doesn't,
    None when the ping itself can't be sent (broker down, import error):
    that says nothing about the worker; two in a row are reported by
    _check_orders_worker_responds as the broker line."""
    import threading
    box: dict = {}

    def _ping() -> None:
        try:
            from tasks import app
            box["ok"] = bool(app.control.ping(destination=[f"orders@{socket.gethostname()}"],
                                              timeout=timeout))
        except Exception as e:
            box["err"] = e

    # A broker that accepts the connection but never replies hangs the ping
    # with no bound, and the watchdog is a oneshot run, so a hang here would
    # block every later check. The thread is a daemon and gets abandoned.
    t = threading.Thread(target=_ping, name="orders-ping", daemon=True)
    t.start()
    t.join(timeout + _ORDERS_PING_GRACE_S)
    if "ok" in box:
        return box["ok"]
    if "err" in box:
        _say(f"orders worker ping could not be sent: {box['err']}")
    else:
        _say(f"orders worker ping did not finish within {timeout + _ORDERS_PING_GRACE_S:.0f}s "
             f"(broker not answering); check skipped")
    return None


def _count_in_a_row(path, hit: bool) -> int:
    """The in-a-row count kept in `path`: +1 on a hit (restarting at 1 when
    the file is older than _ORDERS_PING_STALE_S), 0 otherwise. Saved back;
    never raises."""
    try:
        if not hit:
            n = 0
        elif time.time() - path.stat().st_mtime > _ORDERS_PING_STALE_S:
            n = 1
        else:
            n = int(path.read_text().strip() or 0) + 1
    except (OSError, ValueError):
        n = 1 if hit else 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(n))
    except OSError as e:
        _say(f"Could not save {path.name}: {e}")
    return n


def _check_orders_worker_responds(proc_problems: list[str]) -> list[str]:
    """A problem line once the orders worker, alive by PID, has missed
    _ORDERS_PING_MISSES_TO_ALERT pings in a row; [] otherwise."""
    import process_manager as pm
    label = pm.SERVICES.get("celery_orders_worker", {}).get("label", "celery_orders_worker")
    if any(p.startswith(label) for p in proc_problems):
        return []   # already reported as not running
    ok = _ping_orders_worker()
    unsent = _count_in_a_row(_ORDERS_PING_UNSENT_STATE_FILE, ok is None)
    if ok is None:
        if unsent >= _ORDERS_PING_MISSES_TO_ALERT:
            return [f"Celery broker (Redis) unreachable, or the ping could not be built: the orders "
                    f"worker ping could not be sent on {unsent} checks in a row (cause in the watchdog "
                    f"log); queued tasks, stop moves included, may not run"]
        return []
    misses = _count_in_a_row(_ORDERS_PING_STATE_FILE, not ok)
    if misses >= _ORDERS_PING_MISSES_TO_ALERT:
        return [f"{label} is running but has not answered a Celery ping on {misses} checks "
                f"in a row; it may be hung, and stops may not be moving"]
    return []



# A live ping only proves the prefork parent is up; the orders worker runs one
# child (-c 1) and its tasks have no time limit, so a broker call hung in the
# child passes every ping (professor M219). The parent also reports the
# child's active task with its start time. Only the _broker_* tasks run on
# the orders queue, and they finish within a few minutes (a ~1 min reconnect,
# ~5 s order polls, callers giving up at timeout + 120 s), so one active past
# this is stuck. Alert only, never a heal trigger.
_ORDERS_TASK_STUCK_S = 15 * 60


def _orders_active_tasks(timeout: float = _ORDERS_PING_TIMEOUT_S) -> list[dict] | None:
    """The orders worker's active tasks as reported by `inspect active`, or
    None when the call fails or doesn't finish (bounded like the ping)."""
    import threading
    box: dict = {}
    node = f"orders@{socket.gethostname()}"

    def _inspect() -> None:
        try:
            from tasks import app
            reply = app.control.inspect(destination=[node], timeout=timeout).active()
            box["tasks"] = (reply or {}).get(node)
        except Exception as e:
            box["err"] = e

    t = threading.Thread(target=_inspect, name="orders-inspect", daemon=True)
    t.start()
    t.join(timeout + _ORDERS_PING_GRACE_S)
    if "err" in box:
        _say(f"orders worker active-task check could not be sent: {box['err']}")
        return None
    if "tasks" not in box:
        _say("orders worker active-task check did not finish; check skipped")
        return None
    return box["tasks"] if isinstance(box["tasks"], list) else None


def _check_orders_task_not_stuck(proc_problems: list[str]) -> list[str]:
    """A problem line for each orders task active longer than
    _ORDERS_TASK_STUCK_S; [] otherwise, or when the worker is already
    reported dead or the check can't run."""
    import process_manager as pm
    label = pm.SERVICES.get("celery_orders_worker", {}).get("label", "celery_orders_worker")
    if any(p.startswith(label) for p in proc_problems):
        return []
    active = _orders_active_tasks()
    if not active:
        return []
    out = []
    now = time.time()
    for task in active:
        try:
            started = float(task.get("time_start") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if started and now - started > _ORDERS_TASK_STUCK_S:
            name = str(task.get("name") or "?").rsplit(".", 1)[-1]
            out.append(f"{label} has been running {name} for {(now - started) / 60:.0f} min; "
                       f"it may be hung on a broker call, and stops may not be moving")
    return out

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
        _say(f"systemctl is-active {service} failed: {e}")
        return False


_TRANSITIONAL_STATES = {"activating", "deactivating", "reloading"}
# Touched by whoever restarts Helios on purpose (deploys); while it is fresh
# the watchdog neither alerts nor self-heals (2026-10-28: the 16:05 deploy
# restart was caught mid-stop, alerted Luke and triggered a second restart).
_PLANNED_RESTART_MARKER = Path(__file__).resolve().parent.parent / "logs" / ".planned_restart"
_PLANNED_RESTART_GRACE_S = 600


def _service_state(service: str) -> str:
    try:
        out = subprocess.run(["systemctl", "is-active", service],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception as e:
        _say(f"systemctl is-active {service} failed: {e}")
        return "unknown"


def _planned_restart_in_progress() -> bool:
    try:
        age = time.time() - _PLANNED_RESTART_MARKER.stat().st_mtime
    except OSError:
        return False
    return age < _PLANNED_RESTART_GRACE_S


def _gateway_service_name() -> str:
    from config import TRADING_ENV
    return "ibc-live.service" if TRADING_ENV == "live" else "ibc.service"


def _gateway_expected_up(now: datetime | None = None) -> bool:
    """Weekdays 05:00-18:59 ET and Sunday 17:00-18:59 ET: the hours the
    live gateway must be running for the next session (limitations audit
    2026-11-08 11:07 #3). Nothing on the box stops ibc-live on a schedule,
    so inactive inside these hours means a reboot or a manual stop."""
    from zoneinfo import ZoneInfo
    et = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    if et.weekday() < 5:
        return 5 <= et.hour < 19
    return et.weekday() == 6 and 17 <= et.hour < 19


def _check_gateway_login(now: datetime | None = None) -> list[str]:
    """Catches the exact silent failure that cost a full weekend
    (2026-10-16 Fri eve -> 2026-10-18 Sun eve): IBKR forces a re-login
    roughly weekly (often Sunday), IBC fires a 2FA push, nobody approves
    it, and IBC gives up -- it does NOT retry after a second-factor
    timeout ("Re-login after second factor authentication timeout not
    required" in its log). The Gateway then sits on the login screen with
    its API port closed, while systemd still reports ibc-live.service as
    active (running). Nothing else noticed: the port was dead for ~2 days.

    It also fires when the gateway service is not running inside the hours
    it must be up (_gateway_expected_up): after the 2026-11-08 maintenance
    reboot ibc-live stayed down, and this check said OK because it treated
    inactive as intentional. Outside those hours an inactive gateway stays
    silent, so a weekend manual stop doesn't alarm. See
    helios-gateway-watchdog.timer."""
    from config import IB_PORT, TRADING_ENV
    service = _gateway_service_name()
    if not _service_is_active(service):
        if _gateway_expected_up(now):
            return [f"IBKR gateway ({service}, {TRADING_ENV}) is NOT running — nothing "
                    f"starts it after a reboot. Start it (sudo systemctl start {service}) "
                    f"and approve the 2FA prompt."]
        return []   # outside the hours it must be up -> nothing to alert on
    if _ibkr_port_reachable():
        return []   # up and listening -> healthy
    return [f"IBKR gateway ({service}, {TRADING_ENV}) is running but its API "
            f"port {IB_PORT} is NOT listening — almost certainly a login stuck "
            f"on 2FA. Approve the second-factor prompt, or restart {service} "
            f"to re-fire it."]


# Pool children killed by these are planned (warm-shutdown kills, the 19:00
# stop), not crashes. A child can also die with an exit code and no signal
# (eight at once with exitcode 70 on 08-26; audit 2026-11-08 19:19 #7);
# exitcode 15 is a planned stop.
_PLANNED_EXITS = {("signal", "9"), ("signal", "15"), ("exitcode", "0"), ("exitcode", "15")}
_WORKER_EXIT_RE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Process '([^']+)' pid:\d+ "
    r"exited with '(signal|exitcode) (\d+)(?: \((\w+)\))?'")
_WORKER_EXIT_TAIL_BYTES = 4 * 1024 * 1024
_LOG_DIR = _BASE_DIR / "logs"


def _check_worker_crashes(service: str = _HELIOS_SERVICE, run=subprocess.run,
                          log_dir: Path | None = None) -> list[str]:
    """One problem line when a Celery pool child died on a signal other than
    a planned kill (SIGSEGV, SIGABRT, SIGBUS ...) since the service last
    started, from the Celery masters' own "Process ... exited with 'signal
    N'" lines (the libarrow crashes of 11-08; audit 2026-11-08 15:08 #3).
    Keyed on Helios's logs rather than the kernel log, so test or repro runs
    on this box don't page and a ChromaDB crash in a worker isn't hidden
    (audit 2026-11-08 17:17 #9). Never raises."""
    try:
        since = run(["systemctl", "show", service, "-p", "ActiveEnterTimestamp", "--value",
                     "--timestamp=unix"], capture_output=True, text=True, timeout=10).stdout.strip()
        if not since.startswith("@"):
            return []
        start = datetime.fromtimestamp(int(since[1:])).strftime("%Y-%m-%d %H:%M:%S")
        found: dict[str, int] = {}
        d = log_dir or _LOG_DIR
        # .1 too: a worker restarted mid-run may have rotated its log; the
        # start-time filter drops older lines.
        for path in sorted([*d.glob("celery_*.log"), *d.glob("celery_*.log.1")]):
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - _WORKER_EXIT_TAIL_BYTES))
                text = f.read().decode("utf-8", "replace")
            for line in text.splitlines():
                m = _WORKER_EXIT_RE.match(line)
                if not m or m.group(1) < start or (m.group(3), m.group(4)) in _PLANNED_EXITS:
                    continue
                key = f"{path.name.split('.')[0]} {m.group(5) or m.group(3) + ' ' + m.group(4)}"
                found[key] = found.get(key, 0) + 1
        if not found:
            return []
        n = sum(found.values())
        what = ", ".join(f"{k} x{v}" for k, v in sorted(found.items()))
        return [f"{n} pool worker crash(es) since {service} started ({what})"]
    except Exception as e:
        _say(f"worker crash check failed: {e}")
        return []


_CHAT_MISS_STATE_FILE = _BASE_DIR / "logs" / ".watchdog_chat_listener_misses"
_CHAT_UNITS = ("helios-local.service", "helios-standby.service")


def _check_chat_listener(state=_service_state, path: Path = _CHAT_MISS_STATE_FILE) -> list[str]:
    """One problem line when neither helios-local nor helios-standby is up
    on two checks in a row, so nothing answers Telegram chat: helios-standby
    is disabled and only the 19:00 evening stop starts it, so a reboot left
    11-08 without one for ~9 h (roadmap 04f). A unit in transition or a
    planned restart counts as up; two checks in a row ride out the 05:00 and
    19:00 handovers. Never raises."""
    try:
        states = {u: state(u) for u in _CHAT_UNITS}
        up = (_planned_restart_in_progress()
              or any(v == "active" or v in _TRANSITIONAL_STATES for v in states.values()))
        if _count_in_a_row(path, not up) < 2:
            return []
        return ["No Telegram chat listener: helios-local and helios-standby are both down "
                "(start helios-standby, or helios-local in trading hours)"]
    except Exception as e:
        _say(f"chat listener check failed: {e}")
        return []


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


# Below this share of the analyst model's bytes in VRAM, the watchdog logs a
# line. At 78/81 layers on the GPU (10-25 on) decode ran at p50 9.5 tokens/s;
# at 64/81 (08-27 to 10-24) 3.6 tokens/s with 46 timeouts (audit 2026-11-07
# 09:07 #2, #8). Nothing else in the repo watches the split.
_ANALYST_VRAM_SHARE_FLOOR = 0.90


def _analyst_gpu_split_note(timeout: float = 3.0) -> str | None:
    """A watchdog.log line when the analyst model is loaded with less than
    _ANALYST_VRAM_SHARE_FLOOR of its bytes in VRAM (/api/ps), else None (also
    when it isn't loaded or the read fails). Log-only: no Telegram, no heal.
    Never raises."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("ANALYST_MODEL", "qwen2.5:72b")
    try:
        models = requests.get(f"{host}/api/ps", timeout=timeout).json().get("models") or []
        # A tag-less name ("qwen2.5" is listed as "qwen2.5:latest")
        # matches its :latest form, so a name-form mismatch can't silence it.
        names = {model, model if ":" in model.split("/")[-1] else f"{model}:latest"}
        m = next((m for m in models if names & {m.get("name"), m.get("model")}), None)
        if m is None:
            return None
        size, vram = m.get("size"), m.get("size_vram")
        if not isinstance(size, (int, float)) or size <= 0 or not isinstance(vram, (int, float)):
            return None
        share = vram / size
        if share >= _ANALYST_VRAM_SHARE_FLOOR:
            return None
        return (f"analyst model {model} has {share:.0%} of its bytes in VRAM "
                f"(floor {_ANALYST_VRAM_SHARE_FLOOR:.0%}, context {m.get('context_length')}): "
                f"part of it runs on the CPU, so verdicts slow and may time out")
    except Exception:
        return None


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
        _say(f"Could not persist heal history: {e}")


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


_HEAL_START_TIMEOUT_S = 320.0   # helios-local's TimeoutStartSec=300, plus margin


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

    # The start blocks through ExecStartPre's 180 s gateway wait; with the
    # unit's TimeoutStartSec=300 the default 90 s here gave up on a start
    # still in progress and paged "still DOWN" (audit 2026-11-08 15:08 #4).
    # sudoers allows the exact command only, so no --no-block.
    ok, err = _sudo_systemctl("start", _HELIOS_SERVICE, timeout=_HEAL_START_TIMEOUT_S)
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
        if _alert_due(throttle_key, _problems_fingerprint(problems)):
            _say(f"{msg}")
            if not _send_telegram(msg):
                # The throttle was taken before the send; give it back so
                # the next run retries instead of waiting out a window for
                # an alert that never arrived.
                _clear_alert_state(throttle_key)
        else:
            _say(f"(throttled, not re-sent) {msg}")
    else:
        _clear_alert_state(throttle_key)
        _say("OK")


_BOOT_NOTICE_MARKER = _BASE_DIR / "logs" / ".watchdog_boot_id"


def _boot_facts(run=subprocess.run) -> dict | None:
    """This boot and the one before it, from the journal: when each started
    and ended, and whether the previous one ended with a clean shutdown
    (systemd-shutdown lines in its last 40 entries; a power loss or hard
    lock leaves none). None when the journal can't be read."""
    import json as _json
    try:
        boots = _json.loads(run(["journalctl", "--list-boots", "-o", "json", "--no-pager"],
                                capture_output=True, text=True, timeout=15).stdout)
        by_index = {b["index"]: b for b in boots}
        cur, prev = by_index.get(0), by_index.get(-1)
        if not cur:
            return None
        facts = {"boot_id": cur["boot_id"], "started": cur["first_entry"] / 1e6}
        if prev:
            tail = run(["journalctl", "-b", "-1", "-n", "40", "--no-pager"],
                       capture_output=True, text=True, timeout=15).stdout
            facts.update(prev_ended=prev["last_entry"] / 1e6,
                         prev_clean="systemd-shutdown" in tail)
        return facts
    except Exception as e:
        _say(f"boot facts unavailable: {e}")
        return None


_REPO_UNITS = Path(__file__).resolve().parent / "systemd"
_SYSTEM_UNITS = Path("/etc/systemd/system")


def _unit_lines(text: str, user: str | None = None) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        out.append(line.replace("YOUR_USER", user) if user else line)
    return out


def _unit_drift(repo: Path = _REPO_UNITS, installed: Path = _SYSTEM_UNITS,
                user: str | None = None) -> list[str]:
    """Installed units (and their .d drop-ins) that differ from the repo's
    copies, as "name (Key, Key)": key names only, never values (audit
    2026-11-08 11:07 #8: TimeoutStartSec=300 was in the repo but not
    installed). Comments, blank lines and the YOUR_USER placeholder are
    ignored; units that aren't installed are skipped. Never raises."""
    user = user or Path.home().name
    out = []
    try:
        for f in sorted([*repo.glob("*.service"), *repo.glob("*.timer"),
                         *repo.glob("*.service.d/*.conf")]):
            rel = f.relative_to(repo)
            target = installed / rel
            if not target.is_file():
                if f.suffix == ".conf":
                    out.append(f"{rel} (not installed)")
                continue
            want = _unit_lines(f.read_text(), user)
            have = _unit_lines(target.read_text())
            if want == have:
                continue
            changed = sorted({ln.split("=", 1)[0] for ln in set(want) ^ set(have) if "=" in ln})
            out.append(f"{rel} ({', '.join(changed) or 'order'})")
    except Exception as e:
        _say(f"unit drift check failed: {e}")
    return out


def _boot_notice(facts: dict | None, marker: Path = _BOOT_NOTICE_MARKER) -> str | None:
    """One message per machine boot (limitations audit 2026-11-08 11:07
    #5): the 11-07 22:58 power loss went unnoticed for ~11 h, and the first
    message after it ("systemd unit failed: helios-local") pointed at the
    wrong cause. Says when the box came back, whether the previous run ended
    cleanly and how long it was down, and which Helios units are up. None
    when this boot was already announced or the journal can't be read."""
    if not facts:
        return None
    try:
        if marker.exists() and marker.read_text().strip() == facts["boot_id"]:
            return None
    except OSError:
        pass
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    fmt = lambda ts: datetime.fromtimestamp(ts, et).strftime("%a %m-%d %H:%M ET")
    lines = [f"ℹ️ <b>Helios box restarted</b> {fmt(facts['started'])}"]
    if "prev_ended" in facts:
        down_h = max(0.0, facts["started"] - facts["prev_ended"]) / 3600
        how = "clean shutdown" if facts["prev_clean"] else "NOT a clean shutdown (power loss or crash)"
        lines.append(f"Previous run ended {fmt(facts['prev_ended'])}: {how}; down {down_h:.1f} h")
    states = ", ".join(f"{u} {'up' if _service_is_active(u + '.service') else 'down'}"
                       for u in ("ibc-live", "helios-local", "helios-standby"))
    lines.append(f"Now: {states}")
    drift = _unit_drift()
    if drift:
        lines.append("Installed units differ from the repo: " + "; ".join(drift))
    return "\n".join(lines)


def _send_boot_notice() -> None:
    """Send _boot_notice once; the marker is written only after Telegram
    accepts it, so a failed send retries on the next run. Never raises."""
    try:
        facts = _boot_facts()
        text = _boot_notice(facts)
        if text and _send_telegram(text):
            _BOOT_NOTICE_MARKER.write_text(facts["boot_id"])
    except Exception as e:
        _say(f"boot notice failed: {e}")


def main() -> None:
    # --gateway-only: just the "gateway up but API port dead" check. Runs on
    # its own 7-day/5-min timer (helios-gateway-watchdog.timer) so the
    # weekend 2FA-stuck-login failure gets caught, without dragging in the
    # pipeline checks below -- those are expected to be down outside the
    # weekday trading window and would false-alarm there. Its own throttle
    # key so a gateway alert is never swallowed by an unrelated pipeline one.
    gateway_only = "--gateway-only" in sys.argv[1:]
    _send_boot_notice()
    if gateway_only:
        # The gateway timer runs all week, so it also covers the chat
        # listener on weekends and nights (roadmap 04f).
        _emit(_check_gateway_login() + _check_chat_listener(), "watchdog_gateway_alert_at")
        return

    # Full check: scoped by helios-watchdog.timer to weekday market hours,
    # when the whole stack is expected up. The gateway-login check is left
    # to the dedicated gateway timer above, so it isn't duplicated here.
    #
    # --heal is opt-in rather than default-on so that running this script by
    # hand to diagnose something can never restart live trading as a side
    # effect; helios-watchdog.service passes it, nothing else does.
    heal = "--heal" in sys.argv[1:]

    state = _service_state(_HELIOS_SERVICE)
    if state in _TRANSITIONAL_STATES or _planned_restart_in_progress():
        _say(f"skipped: {_HELIOS_SERVICE} is {state}"
              f"{' (planned restart)' if _planned_restart_in_progress() else ''}")
        return

    service_down = state != "active"
    proc_problems = _check_processes()
    problems = ([f"{_HELIOS_SERVICE} is not active"] if service_down else []) \
        + proc_problems + _check_snapshot_freshness()
    if not service_down:
        problems += _check_worker_crashes()
        problems += _check_orders_worker_responds(proc_problems)
        problems += _check_orders_task_not_stuck(proc_problems)

    # Restart only on unambiguous total failure -- the unit itself down, or
    # every critical worker gone. A PARTIALLY degraded stack is left alone
    # and merely alerted on: bouncing the whole service to recover one
    # worker would tear down the others mid-task, and a half-up system is
    # the state most likely to have something in flight worth not killing.
    total_failure = service_down or (
        len(_check_processes(_HEAL_TRIGGER_SERVICES)) == len(_HEAL_TRIGGER_SERVICES))

    heal_notes: list[str] = []
    if heal and total_failure:
        trigger = (f"{_HELIOS_SERVICE} not active" if service_down
                   else "all critical workers dead")
        heal_notes = _attempt_heal(trigger)

    _emit(problems, "watchdog_alert_at")

    gpu_note = _analyst_gpu_split_note()
    if gpu_note:
        _say(gpu_note)

    # Sent outside _emit's throttle on purpose: the alert about the outage
    # may well be suppressed as a repeat, but the fact that the watchdog
    # took action must always arrive.
    if heal_notes:
        msg = "🔧 <b>Helios self-heal</b>\n" + "\n".join(heal_notes)
        _say(f"{msg}")
        _send_telegram(msg)


if __name__ == "__main__":
    main()
