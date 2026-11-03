"""
Sends a Telegram alert when a systemd unit fails, via OnFailure=. Deliberately
minimal and self-contained (no imports from the app) — it exists specifically
for the case where the main stack itself failed to start, so it can't depend
on anything the main stack provides.

OnFailure= fires on every failed start, so a unit retrying under Restart=
sent six identical alerts in eight minutes each morning of 10-05..09 and
10-20..23. The first failure of a unit alerts at once; further failures of
the same unit within REPEAT_WINDOW_S are counted and reported with the next
alert. A failure that would make the held count MAX_HELD is sent anyway, as is a failure
after the unit had recovered (up RECOVERED_S since the last alert). The
window is consumed only by a send Telegram accepted.

Usage: notify_failure.py <failed-unit-name>
"""
import fcntl
import html
import json
import re
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_BASE_DIR))
load_dotenv(dotenv_path=_BASE_DIR / ".env")

_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

REPEAT_WINDOW_S = 30 * 60
# When the held count would reach this, the failure is sent anyway, with the count.
MAX_HELD = 3
# Up this long after the last alert before failing again = a new incident.
RECOVERED_S = 5 * 60
STATE_PATH = _BASE_DIR / "logs" / ".notify_failure_state.json"


def _redact(s: str) -> str:
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot[REDACTED]", s)


def _send(text: str) -> bool:
    """True only when Telegram accepted it; a 400 is resent as plain text."""
    url = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": _CHAT_ID, "text": text, "parse_mode": "HTML"},
                             timeout=10)
        if getattr(resp, "status_code", None) == 400:
            resp = requests.post(url, json={"chat_id": _CHAT_ID,
                                            "text": html.unescape(re.sub(r"<[^>]+>", "", text))},
                                 timeout=10)
        ok = getattr(resp, "ok", True) is not False
        if not ok:
            print(f"[notify_failure] Telegram send rejected: HTTP {getattr(resp, 'status_code', '?')}")
        return ok
    except Exception as e:
        print("[notify_failure] Telegram send failed: " + _redact(str(e)))
        return False


def _record(text: str, status: str) -> None:
    try:
        import alert_policy
        alert_policy.record("deploy/notify_failure.py:main", text, alert_policy.CRITICAL, "", status)
    except Exception:
        pass


def _unix_prop(unit: str, prop: str) -> float | None:
    try:
        import subprocess
        out = subprocess.run(["systemctl", "show", "-P", prop, "--timestamp=unix", unit],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return float(out.lstrip("@")) if out.lstrip("@") else None
    except Exception:
        return None


def _recovered_since(unit: str, since: float, now: float) -> bool:
    """The unit came up after the last alert and stayed up RECOVERED_S
    before this failure: a new incident, not the same retry loop. A
    Restart= loop goes active for seconds at a time, so the hold covers it."""
    up = _unix_prop(unit, "ActiveEnterTimestamp")
    if up is None:
        return True   # can't tell: send rather than hold
    if up <= since:
        return False
    down = _unix_prop(unit, "InactiveEnterTimestamp")
    if down is None or down < up:
        down = now
    return down - up >= RECOVERED_S


def _load(f) -> dict:
    try:
        f.seek(0)
        state = json.loads(f.read() or "{}")
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def main(now: float | None = None) -> None:
    unit = sys.argv[1] if len(sys.argv) > 1 else "unknown unit"
    now = time.time() if now is None else now
    if not _TOKEN or not _CHAT_ID:
        print(f"[notify_failure] No Telegram credentials — would have alerted for {unit}")
        return
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        f = open(STATE_PATH, "a+")
        fcntl.flock(f, fcntl.LOCK_EX)
    except Exception:
        f = None
    try:
        state = _load(f) if f else {}
        rec = state.get(unit) or {}
        last = rec.get("last_sent")
        held_now = int(rec.get("held") or 0)
        if (last is not None and 0 <= now - last < REPEAT_WINDOW_S
                and held_now + 1 < MAX_HELD and not _recovered_since(unit, last, now)):
            rec["held"] = int(rec.get("held") or 0) + 1
            state[unit] = rec
            print(f"[notify_failure] {unit} failed again; held ({rec['held']} since last alert)")
            _record(f"🚨 systemd unit failed: {unit}", "held")
        else:
            held = int(rec.get("held") or 0)
            extra = (f"\n({held} more failure(s) of this unit since the last alert.)" if held else "")
            text = (
                f"🚨 <b>systemd unit failed: {unit}</b>\n"
                f"Likely causes: IBKR/TWS 2FA hung waiting for a factor no script can "
                f"provide, TWS not running, or a startup exception.\n"
                f"Check: <code>systemctl status {unit}</code> and "
                f"<code>journalctl -u {unit} -n 50</code>.{extra}"
            )
            ok = _send(text)
            _record(text, "sent" if ok else "failed")
            if ok:
                state[unit] = {"last_sent": now, "held": 0}
        if f:
            f.seek(0)
            f.truncate()
            f.write(json.dumps(state))
            f.flush()
    finally:
        if f:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


if __name__ == "__main__":
    main()
