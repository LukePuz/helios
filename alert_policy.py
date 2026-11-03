"""
Severity routing, dedup and the outbox record for Telegram alerts.

Stdlib only, so deploy/watchdog.py and deploy/notify_failure.py (which must
keep working when the app is broken) and scripts/alert_triage.py can use it.

Severity:
  critical  sent at once, never deduplicated, rate limited or batched.
  report    a scheduled report or a reply to a request: sent at once.
  low       a known informational type (LOW_TYPES). The first of a type in
            DEDUP_WINDOW_S is sent; repeats inside the window, and anything
            over LOW_RATE_PER_HOUR, go into a digest sent by take_digest().
Anything that matches no pattern is critical, so a new or misworded alert
is never held back. Only the types listed in LOW_TYPES can be batched.

Every alert, sent or held, is appended to logs/telegram_outbox.jsonl with
its first line only (money and long numbers masked, secrets redacted), so
the outbox carries no balances or account ids.
"""
import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CRITICAL = "critical"
REPORT = "report"
LOW = "low"

LOGS_DIR = Path(__file__).resolve().parent / "logs"
OUTBOX_PATH = LOGS_DIR / "telegram_outbox.jsonl"
STATE_PATH = LOGS_DIR / ".alert_policy_state.json"

DEDUP_WINDOW_S = 6 * 3600
LOW_RATE_PER_HOUR = 4
_TITLE_MAX = 140

# Checked first: any of these makes an alert critical even if a low pattern
# also matches (e.g. an ingestion alert that mentions a stop).
_CRITICAL_MARKERS = re.compile(
    r"🚨|🛑|HALT|DRAWDOWN|UNPROTECTED|NAKED|SHORT POSITION|STOP (TRIGGERED|NOT|REFUSED|"
    r"STATUS|DELAYED|RECONCILE)|CHECK STOPS|ORDER (BLOCKED|ABORTED|SKIPPED)|NOT SENT|"
    r"FILL (UNCONFIRMED|NOT RECORDED)|UNRECONCILED|DB WRITE FAILED|MONITOR BLIND|"
    r"IBKR|[Gg]ateway|TWS|disconnect|BACKSTOP|POSSIBLE SPLIT|STILL OPEN|STILL WORKING|"
    r"EXECUTION FAILED|STUCK APPROVAL|OUTCOME NOT RECORDED|SALE UNCONFIRMED|TRADE EXECUTED|"
    r"systemd unit failed|watchdog alert|self-heal")

REPORT_TYPES = [
    ("daily_digest", re.compile(r"Daily (Digest|Summary)")),
    ("weekly_self_review", re.compile(r"Weekly Self-Review")),
    ("research_reply", re.compile(r"🔬 <b>Research:|Deep analysis of \S+ failed")),
    ("candidate_scan", re.compile(r"^Watchlist candidate scan:")),
    ("alert_digest", re.compile(r"^🧾 <b>Alert digest")),
]

LOW_TYPES = [
    # Not here: the ingestion health check (it has its own per-stale-set
    # throttle in tasks.check_ingestion_health, so it stays critical).
    ("analyst_stub", re.compile(r"Analyst falling back to stub")),
    ("ollama_recovered", re.compile(r"Ollama auto-recovery</b>")),
    ("vector_store_recovered", re.compile(r"Vector store auto-recovery")),
    ("ipo_watch", re.compile(r"IPO watch")),
    ("monitor_recovered", re.compile(r"monitor recovered after \d+ failed checks")),
    # Success text only; main.py sends a failure under a FAILED header.
    ("startup_info", re.compile(r"^ℹ️ <b>Startup reconciliation</b> finished after \d+s\.$")),
    ("catalyst_conflict", re.compile(r"^<b>CATALYST CONFLICT")),
]
# Deliberately critical (not listed above): expired recommendations and the
# WAITING FOR FRESH PRICE / CONFIRMATION notices on an approved order.


def classify(text: str) -> tuple[str, str]:
    """(severity, type). type is a LOW_TYPES/REPORT_TYPES name, or the
    normalized title for anything else."""
    text = str(text)
    for name, pat in REPORT_TYPES:
        if pat.search(text):
            return REPORT, name
    head = text.split("\n", 1)[0]
    if not _CRITICAL_MARKERS.search(head):
        for name, pat in LOW_TYPES:
            if pat.search(text):
                return LOW, name
    return CRITICAL, fingerprint(text)


def title(text: str) -> str:
    """First non-empty line, HTML removed, money and long numbers masked."""
    line = next((ln for ln in str(text).splitlines() if ln.strip()), "")
    # Redact before any masking: masking digits first breaks the token
    # patterns and would leave the secret half of a bot token in the row.
    line = _redact(line)
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\$\s?[\d,]+(\.\d+)?", "$[amt]", line)
    line = re.sub(r"\b\d{1,3}(,\d{3})+(\.\d+)?", "[amt]", line)
    line = re.sub(r"\b\d+\.\d{2}\b", "[amt]", line)
    line = re.sub(r"\d{5,}", "[n]", line)
    return re.sub(r"\s+", " ", line).strip()[:_TITLE_MAX]


_BOT_TOKEN = re.compile(r"(?<![0-9])(?:bot)?\d{6,}:[A-Za-z0-9_-]{30,}")


def _redact(line: str) -> str:
    try:
        import log_redaction
        line = log_redaction.redact(line)
    except Exception:
        pass
    return _BOT_TOKEN.sub("[REDACTED]", line)


def fingerprint(text: str) -> str:
    """Title with every number collapsed, so '847 min old' and '857 min
    old' are the same alert."""
    return re.sub(r"\d+(\.\d+)?", "N", title(text))


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _locked_state(path: Path | None = None):
    """Read-modify-write of the state file under an exclusive flock, shared
    by every Celery worker process."""
    path = Path(path or STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            try:
                state = json.loads(path.read_text())
                if not isinstance(state, dict):
                    state = {}
            except Exception:
                state = {}
            yield state
            tmp = Path(str(path) + ".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, path)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)




def dedup_key(alert_type: str, text: str) -> str:
    """Type plus the whole message normalised: HTML gone, numbers collapsed,
    words (tickers, source and monitor names) kept. Two stub fallbacks for
    different tickers, or two monitors recovering, are different alerts;
    the same one with a new age or count is a repeat."""
    if alert_type == "monitor_recovered":
        m = re.search(r"(\S+) monitor recovered", str(text))
        if m:
            return f"monitor_recovered|{m.group(1)}"
    body = _redact(str(text))
    body = re.sub(r"<[^>]+>", "", body)
    body = re.sub(r"\d+(\.\d+)?", "N", body)
    body = re.sub(r"\s+", " ", body).strip()
    import hashlib
    return f"{alert_type}|{hashlib.sha1(body.encode('utf-8')).hexdigest()[:12]}"


def admit_low(alert_type: str, text: str, now: float | None = None,
              state_path: Path | None = None) -> bool:
    """True if this low-severity alert should be sent now. A held one is
    counted toward the next digest. Any state error admits it."""
    now = _now() if now is None else now
    key = dedup_key(alert_type, text)
    try:
        with _locked_state(state_path) as state:
            entries = state.setdefault("low", {})
            rec = entries.setdefault(key, {"type": alert_type, "held": 0})
            rec["type"] = alert_type
            rec["title"] = title(text)
            sends = [t for t in state.get("low_sends", []) if 0 <= now - t < 3600]
            last = rec.get("last_sent")
            due = ((last is None or not 0 <= now - last < DEDUP_WINDOW_S)
                   and len(sends) < LOW_RATE_PER_HOUR)
            if due:
                rec["last_sent"] = now
                sends.append(now)
            else:
                if not rec.get("held"):
                    rec["first_held"] = now
                rec["held"] = rec.get("held", 0) + 1
                rec["last_held"] = now
            state["low_sends"] = sends
            # Entries idle for two windows are dropped so the file stays small.
            for k in [k for k, v in entries.items()
                      if not v.get("held") and now - (v.get("last_sent") or 0) > 2 * DEDUP_WINDOW_S]:
                entries.pop(k, None)
            return due
    except Exception:
        return True


def digest_text(now: float | None = None,
                state_path: Path | None = None) -> tuple[str | None, dict]:
    """(text, snapshot) for every held low alert, without clearing them;
    (None, {}) when nothing is held. Grouped by type, each distinct alert on
    its own line. Pass the snapshot to clear_digest once the send worked, so
    a failed send keeps the counts for the next digest."""
    now = _now() if now is None else now
    try:
        with _locked_state(state_path) as state:
            held = [(k, v) for k, v in state.get("low", {}).items() if v.get("held")]
    except Exception:
        return None, {}
    if not held:
        return None, {}
    by_type: dict = {}
    for k, rec in held:
        by_type.setdefault(rec.get("type") or k.split("|")[0], []).append(rec)
    lines = ["🧾 <b>Alert digest</b> (low severity, repeats held back)"]
    for typ, recs in sorted(by_type.items(), key=lambda kv: -sum(r["held"] for r in kv[1])):
        lines.append(f"<b>{typ}</b>")
        for rec in sorted(recs, key=lambda r: -r["held"]):
            first = datetime.fromtimestamp(rec.get("first_held", now)).strftime("%a %H:%M")
            last = datetime.fromtimestamp(rec.get("last_held", now)).strftime("%a %H:%M")
            lines.append(f"• {rec.get('title') or typ} ×{rec['held']} ({first}–{last})")
    return "\n".join(lines), {k: v["held"] for k, v in held}


def clear_digest(snapshot: dict, state_path: Path | None = None) -> None:
    """Subtract the counts a sent digest reported; anything held since the
    snapshot stays for the next one."""
    try:
        with _locked_state(state_path) as state:
            for k, n in snapshot.items():
                rec = state.get("low", {}).get(k)
                if not rec:
                    continue
                rec["held"] = max(int(rec.get("held") or 0) - n, 0)
                if not rec["held"]:
                    rec.pop("first_held", None)
                    rec.pop("last_held", None)
    except Exception:
        pass


def take_digest(now: float | None = None, state_path: Path | None = None) -> str | None:
    """digest_text and clear_digest in one step, for a caller that can't
    fail to deliver it."""
    text, snap = digest_text(now, state_path)
    if text:
        clear_digest(snap, state_path)
    return text


OUTBOX_ROTATE_BYTES = 5 * 1024 * 1024


def record(source: str, text: str, severity: str, alert_type: str, status: str,
           now: float | None = None, outbox_path: Path | None = None) -> None:
    """Append one outbox line. Never raises: losing an outbox row must not
    lose the alert. Past OUTBOX_ROTATE_BYTES the file moves to
    telegram_outbox.jsonl.1 (one old file kept) and a new one starts."""
    now = _now() if now is None else now
    try:
        row = {"ts": _iso(now), "source": source, "severity": severity,
               "type": alert_type if severity != CRITICAL else "",
               "fingerprint": fingerprint(text), "title": title(text), "status": status}
        path = Path(outbox_path or OUTBOX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "a") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                try:
                    if path.stat().st_size >= OUTBOX_ROTATE_BYTES:
                        os.replace(path, str(path) + ".1")
                except FileNotFoundError:
                    pass
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
    except Exception:
        pass
