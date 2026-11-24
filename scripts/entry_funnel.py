"""Entry funnel per live session (engineering monitor, not a study analysis):
how many OPEN/ADD verdicts got how far, so a gate that drives passed buys to
about zero shows up within a day.

Per session (NYSE sessions, ET dates; a row on a weekend or holiday counts
toward the next session), from rows the bot already writes:
  verdicts      OPEN/ADD verdict rows in signals (no decision.route_exit).
  stopped       routing rows (decision.route_exit set, verdict_action OPEN or
                ADD) that ended the route before an approval, by kind; an
                ENTRY_GATE_BLOCKED row is split by the gate that blocked
                (decision.entry_gate.<stopout|trend|add_loser>.blocked).
                Rows whose kind names a shadow (…SHADOW…) are not stops:
                they are listed with the shadows.
  routed        OPEN/ADD approvals (pending_approvals, is_test excluded).
  retry_blocked approvals the deferred retry's entry gate expired
                (ENTRY_GATE_BLOCKED, entry_gate.path deferred_retry, counted
                on the approval's session through entry_gate.approval_id).
  passed        routed minus retry_blocked: buys past every gate. Approvals
                still PENDING (a deferred entry waiting for the open) count.
  approved      approvals now in status APPROVED.
  ordered       approvals with at least one BUY orders row.
  filled        approvals with a BUY order whose filled_qty is above 0.
Shadows, would_block only (they never block): the add-to-loser check
(risk_json.add_loser_gate on approvals, activity add_loser_gate_shadow on
deferred retries) and the recheck-contradiction record
(risk_json.recheck_contradiction on approvals, decision.recheck_contradiction
on routing rows), plus the row counts of the …SHADOW… routing kinds.

Each stage is counted on its own rows, so the columns need not add up: a
verdict near midnight can be routed on the next session, and one verdict can
leave a stop row and a shadow row.

--check (for the monitor cron): exit 1 with one line when the last
CHECK_SESSIONS sessions together have 0 passed buys while they have at least
CHECK_MIN_VERDICTS verdicts. Prints nothing and exits 0 otherwise; exit 2
with one line when the read failed.

    env -C ~/Desktop/trading_bot SUPABASE_PORT=6543 venv/bin/python \
        scripts/entry_funnel.py [--sessions 5] [--until 2026-11-20] [--env live] \
        [--check] [--out FILE]

Read-only: one DB transaction opened READ ONLY with a short statement
timeout, then rolled back. Counts and percentages only: no prices,
quantities or account values. activity_feed has no env column; its
add_loser_gate_shadow rows are kept only when they name an approval of this
env.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from market_calendar import NYSE_HOLIDAYS  # noqa: E402  (no broker imports)

NY = ZoneInfo("America/New_York")
BUY_ACTIONS = ("OPEN", "ADD")
GATE_HALVES = ("stopout", "trend", "add_loser")
DEFAULT_SESSIONS = 5
CHECK_SESSIONS = 2
CHECK_MIN_VERDICTS = 3
STATEMENT_TIMEOUT_MS = 10000
SESSION_CLOSE = time(16, 0)


# ---------------------------------------------------------------- sessions

def _is_session(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def session_of(ts) -> date | None:
    """The NYSE session a row counts toward: its ET date, or the next
    session when that date is a weekend or holiday. Naive times are UTC."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    d = ts.astimezone(NY).date()
    while not _is_session(d):
        d += timedelta(days=1)
    return d


def sessions_until(until: date, n: int) -> list[date]:
    """The n sessions ending at until (until itself when it is a session),
    oldest first."""
    out, d = [], until
    while len(out) < n:
        if _is_session(d):
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def default_until(now: datetime) -> date:
    """Today once its session has closed (16:00 ET), else the last session
    before today: a session still trading is not counted as a whole one."""
    et = now.astimezone(NY)
    d = et.date()
    if _is_session(d) and et.time() >= SESSION_CLOSE:
        return d
    d -= timedelta(days=1)
    while not _is_session(d):
        d -= timedelta(days=1)
    return d


def window(sessions: list[date]) -> tuple[datetime, datetime]:
    """UTC bounds of the rows that count toward these sessions: from the day
    after the session before the first one, to the end of the last one."""
    d = sessions[0] - timedelta(days=1)
    while not _is_session(d):
        d -= timedelta(days=1)
    start = datetime.combine(d + timedelta(days=1), time(0), tzinfo=NY)
    end = datetime.combine(sessions[-1] + timedelta(days=1), time(0), tzinfo=NY)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


# ---------------------------------------------------------------- counting

def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _gate_halves(decision) -> list[str]:
    """The entry-gate halves that blocked on an ENTRY_GATE_BLOCKED row
    ('unknown' when none says so)."""
    gate = _as_dict(_as_dict(decision).get("entry_gate"))
    hit = [h for h in GATE_HALVES if _as_dict(gate.get(h)).get("blocked") is True]
    return hit or ["unknown"]


def _would_block(rec) -> bool:
    return _as_dict(rec).get("would_block") is True


def _empty() -> dict:
    return {"verdicts": Counter(), "stopped": Counter(), "stopped_rows": 0, "routed": 0,
            "retry_blocked": 0, "retry_gate": Counter(), "passed": 0, "status": Counter(),
            "approved": 0, "ordered": 0, "filled": 0, "shadow_rows": Counter(),
            "add_loser_would_block": 0, "recheck_would_block": 0}


def build_funnel(sessions, verdicts, routes, approvals, orders, activity) -> dict:
    """{session: counts} over the given sessions.

    verdicts:  dicts {ts, action}
    routes:    dicts {ts, kind, decision}
    approvals: dicts {id, ts, action, status, is_test, add_loser_gate,
               recheck_contradiction}
    orders:    dicts {approval_id, filled_qty}
    activity:  dicts {ts, extra} of add_loser_gate_shadow rows"""
    out = {s: _empty() for s in sessions}

    def bucket(ts):
        s = session_of(ts)
        return out.get(s)

    for v in verdicts:
        f = bucket(v.get("ts"))
        if f is not None and v.get("action") in BUY_ACTIONS:
            f["verdicts"][v["action"]] += 1

    live_approvals = {a["id"]: a for a in approvals
                      if a.get("action") in BUY_ACTIONS and not a.get("is_test")}
    retry_blocked_ids: dict[int, list[str]] = {}
    for r in routes:
        kind, dec = r.get("kind") or "?", _as_dict(r.get("decision"))
        if dec.get("verdict_action") not in BUY_ACTIONS:
            continue
        gate = _as_dict(dec.get("entry_gate"))
        if kind == "ENTRY_GATE_BLOCKED" and gate.get("path") == "deferred_retry":
            aid = gate.get("approval_id")
            if aid in live_approvals:
                retry_blocked_ids.setdefault(aid, _gate_halves(dec))
                continue
            f = bucket(r.get("ts"))       # the approval is outside the window
            if f is not None:
                f["retry_blocked"] += 1
                for h in _gate_halves(dec):
                    f["retry_gate"][h] += 1
            continue
        f = bucket(r.get("ts"))
        if f is None:
            continue
        if _would_block(dec.get("recheck_contradiction")):
            f["recheck_would_block"] += 1
        if "SHADOW" in kind:
            f["shadow_rows"][kind] += 1
            continue
        f["stopped_rows"] += 1
        if kind == "ENTRY_GATE_BLOCKED":
            for h in _gate_halves(dec):
                f["stopped"][f"entry_gate:{h}"] += 1
        else:
            f["stopped"][kind] += 1

    ordered, filled = set(), set()
    for o in orders:
        aid = o.get("approval_id")
        if aid in live_approvals:
            ordered.add(aid)
            try:
                if float(o.get("filled_qty") or 0) > 0:
                    filled.add(aid)
            except (TypeError, ValueError):
                pass

    for aid, a in live_approvals.items():
        f = bucket(a.get("ts"))
        if f is None:
            continue
        f["routed"] += 1
        if aid in retry_blocked_ids:
            f["retry_blocked"] += 1
            for h in retry_blocked_ids[aid]:
                f["retry_gate"][h] += 1
        else:
            f["passed"] += 1
        status = str(a.get("status") or "?").upper()
        f["status"][status] += 1
        f["approved"] += status == "APPROVED"
        f["ordered"] += aid in ordered
        f["filled"] += aid in filled
        if _would_block(a.get("add_loser_gate")):
            f["add_loser_would_block"] += 1
        if _would_block(a.get("recheck_contradiction")):
            f["recheck_would_block"] += 1

    for row in activity:
        extra = _as_dict(row.get("extra"))
        if extra.get("approval_id") not in live_approvals:
            continue
        f = bucket(row.get("ts"))
        if f is not None and _would_block(extra.get("add_loser_gate")):
            f["add_loser_would_block"] += 1
    return out


def funnel_check(funnel: dict, sessions: list[date], n: int = CHECK_SESSIONS,
                 min_verdicts: int = CHECK_MIN_VERDICTS) -> str | None:
    """One alert line when the last n sessions have 0 passed buys on at least
    min_verdicts verdicts, else None."""
    last = sessions[-n:]
    verdicts = sum(sum(funnel[s]["verdicts"].values()) for s in last)
    passed = sum(funnel[s]["passed"] for s in last)
    if passed > 0 or verdicts < min_verdicts:
        return None
    stops = Counter()
    for s in last:
        stops.update(funnel[s]["stopped"])
        stops.update({f"retry:{k}": v for k, v in funnel[s]["retry_gate"].items()})
    top = ", ".join(f"{k} {v}" for k, v in stops.most_common(4)) or "no stop rows"
    return (f"ENTRY FUNNEL: 0 passed buys in the last {len(last)} sessions "
            f"({last[0]}..{last[-1]}) on {verdicts} OPEN/ADD verdicts; stops: {top}")


# ---------------------------------------------------------------- output

def _pct(a: int, b: int) -> str:
    return f"{a / b:.0%}" if b else "n/a"


def readout(funnel: dict, sessions: list[date], env: str, head: str) -> str:
    lines = [f"# Entry funnel ({env}), sessions {sessions[0]}..{sessions[-1]}", "",
             f"Code {head}. Counts per session (ET); each stage on its own rows, so the "
             "columns need not add up. passed = routed minus the approvals the deferred "
             "retry's gate expired.", "",
             "| session | verdicts (OPEN/ADD) | stopped | routed | retry blocked | passed "
             "| passed / verdicts | approved | ordered | filled |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    tot = _empty()
    for s in sessions:
        f = funnel[s]
        v = sum(f["verdicts"].values())
        lines.append(f"| {s} | {v} ({f['verdicts']['OPEN']}/{f['verdicts']['ADD']}) "
                     f"| {f['stopped_rows']} | {f['routed']} | {f['retry_blocked']} "
                     f"| {f['passed']} | {_pct(f['passed'], v)} | {f['approved']} "
                     f"| {f['ordered']} | {f['filled']} |")
        for k in ("verdicts", "stopped", "retry_gate", "status", "shadow_rows"):
            tot[k].update(f[k])
        for k in ("stopped_rows", "routed", "retry_blocked", "passed", "approved", "ordered",
                  "filled", "add_loser_would_block", "recheck_would_block"):
            tot[k] += f[k]
    v = sum(tot["verdicts"].values())
    lines.append(f"| all | {v} ({tot['verdicts']['OPEN']}/{tot['verdicts']['ADD']}) "
                 f"| {tot['stopped_rows']} | {tot['routed']} | {tot['retry_blocked']} "
                 f"| {tot['passed']} | {_pct(tot['passed'], v)} | {tot['approved']} "
                 f"| {tot['ordered']} | {tot['filled']} |")

    kinds = sorted(set(tot["stopped"]) | {f"retry:{k}" for k in tot["retry_gate"]})
    lines += ["", "## Stops by gate or routing kind", ""]
    if kinds:
        lines += ["| stop | " + " | ".join(str(s) for s in sessions) + " | all | share of verdicts |",
                  "|---|" + "---|" * (len(sessions) + 2)]
        for k in kinds:
            if k.startswith("retry:"):
                vals = [funnel[s]["retry_gate"][k[6:]] for s in sessions]
            else:
                vals = [funnel[s]["stopped"][k] for s in sessions]
            lines.append(f"| {k} | " + " | ".join(str(x) for x in vals)
                         + f" | {sum(vals)} | {_pct(sum(vals), v)} |")
    else:
        lines.append("No stop rows.")

    lines += ["", "## Shadows (would_block only; they never block)", "",
              "| record | " + " | ".join(str(s) for s in sessions) + " | all |",
              "|---|" + "---|" * (len(sessions) + 1)]
    rows = [("add_loser would_block", [funnel[s]["add_loser_would_block"] for s in sessions]),
            ("recheck_contradiction would_block",
             [funnel[s]["recheck_would_block"] for s in sessions])]
    rows += [(f"{k} rows", [funnel[s]["shadow_rows"][k] for s in sessions])
             for k in sorted(tot["shadow_rows"])]
    for name, vals in rows:
        lines.append(f"| {name} | " + " | ".join(str(x) for x in vals) + f" | {sum(vals)} |")

    lines += ["", "Approval status (all sessions): "
              + (", ".join(f"{k} {n}" for k, n in sorted(tot["status"].items())) or "none") + "."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- I/O

def _fetch(sessions: list[date], env: str):
    import db
    start, end = window(sessions)
    conn = db._get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")   # this transaction only, not the pooled conn
            cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
            cur.execute("SELECT timestamp, signal FROM signals WHERE trading_env = %s "
                        "AND timestamp >= %s AND timestamp < %s AND signal = ANY(%s) "
                        "AND NOT coalesce(decision ? 'route_exit', false)",
                        (env, start, end, list(BUY_ACTIONS)))
            verdicts = [{"ts": t, "action": a} for t, a in cur.fetchall()]
            # Deferred-retry gate rows can come a day after their approval.
            cur.execute("SELECT timestamp, signal, decision FROM signals WHERE trading_env = %s "
                        "AND timestamp >= %s AND timestamp < %s + interval '4 days' "
                        "AND decision ? 'route_exit' AND decision->>'verdict_action' = ANY(%s)",
                        (env, start, end, list(BUY_ACTIONS)))
            routes = [{"ts": t, "kind": k, "decision": d} for t, k, d in cur.fetchall()]
            cur.execute("SELECT id, timestamp, action, status, is_test, "
                        "risk_json->'add_loser_gate', risk_json->'recheck_contradiction' "
                        "FROM pending_approvals WHERE trading_env = %s AND timestamp >= %s "
                        "AND timestamp < %s AND action = ANY(%s)",
                        (env, start, end, list(BUY_ACTIONS)))
            approvals = [{"id": i, "ts": t, "action": a, "status": s, "is_test": x,
                          "add_loser_gate": al, "recheck_contradiction": rc}
                         for i, t, a, s, x, al, rc in cur.fetchall()]
            ids = [a["id"] for a in approvals]
            cur.execute("SELECT approval_id, filled_qty FROM orders WHERE trading_env = %s "
                        "AND side = 'BUY' AND approval_id = ANY(%s)", (env, ids))
            orders = [{"approval_id": a, "filled_qty": q} for a, q in cur.fetchall()]
            cur.execute("SELECT ts, extra_json FROM activity_feed WHERE ts >= %s "
                        "AND ts < %s + interval '4 days' AND stage = 'add_loser_gate_shadow'",
                        (start, end))
            activity = [{"ts": t, "extra": e} for t, e in cur.fetchall()]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        db._put_conn(conn)
    return verdicts, routes, approvals, orders, activity


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def run(sessions: list[date], env: str) -> dict:
    return build_funnel(sessions, *_fetch(sessions, env))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    ap.add_argument("--until", help="last session, YYYY-MM-DD (default: the last closed one)")
    ap.add_argument("--env", default="live")
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 with one line on 0 passed buys over the last "
                         f"{CHECK_SESSIONS} sessions with >= {CHECK_MIN_VERDICTS} verdicts")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    until = (date.fromisoformat(a.until) if a.until
             else default_until(datetime.now(timezone.utc)))
    n = max(a.sessions, CHECK_SESSIONS) if a.check else max(a.sessions, 1)
    sessions = sessions_until(until, n)
    try:
        funnel = run(sessions, a.env)
    except Exception as e:
        if a.check:
            print(f"ENTRY FUNNEL: check failed, the read raised {type(e).__name__}")
            return 2
        raise
    if a.check:
        line = funnel_check(funnel, sessions)
        if line:
            print(line)
            return 1
        return 0
    text = readout(funnel, sessions, a.env, _git_head())
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
