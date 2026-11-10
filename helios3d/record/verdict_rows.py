"""Which public.signals rows are verdicts: one place for the flags and for
each caller's current rule (Helios 3D, research tooling).

This module decides nothing. row_flags() reports what a signals row is
(record-only kind, ROUTE_ row, model, test-leak id, NULL trading_env,
degraded, parse_mode, ...). The *_RULE fragments are the WHERE terms each
existing caller uses today, written out once so the differences stay
visible by name. Whether any flagged row is excluded from an analysis is
Luke's question, not this module's.

THE RULES (as of this commit; each is the caller's existing text):
- LABELS_RULE (helios3d/record/labels.py verdict_query): signal not in
  db._NON_ANALYST_SIGNALS; symbol and timestamp not NULL; trading_env equal
  to config.TRADING_ENV, or NULL and the reasoning not starting with
  db.DEGRADED_SIGNAL_PREFIX. No model filter, no test-leak ids, no ROUTE_
  prefix term (listed ROUTE_ kinds are excluded through the list only),
  confidence may be NULL.
- DECISION_RULE (helios3d/record/decision.py SIGNALS_SQL): signal not in
  db._NON_ANALYST_SIGNALS, or in DECISION_KEEP_KINDS (ANALYST_FAILED);
  symbol and timestamp not NULL. Every trading_env is kept and degraded rows
  are kept (flagged in a column). No model filter, no test-leak ids, no
  ROUTE_ prefix term.
- conviction_share_rule() (``02 Analysis (reference)/scripts/
  conviction_share_daily.py`` in helios-research, outside this repo; not
  refactored, its text is reproduced here): trading_env equal to
  TRADING_ENV or NULL (degraded NULL-env rows are kept); id not in
  TEST_LEAK_SIGNAL_IDS; model LIKE 'qwen2.5:72b%' or NULL; signal not
  LIKE 'ROUTE\\_%'; signal not in the SEF non-verdict list
  (sef_signals.NON_VERDICT, inlined); confidence not NULL. No symbol or
  timestamp NOT NULL term.
None of the three reads decision.parse_mode.

Report (read-only, READ ONLY transaction, prints counts only):
    python -m helios3d.record.verdict_rows --since YYYY-MM-DD [--until YYYY-MM-DD]
"""
import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# The two pytest task-leak signal rows (copy of TEST_LEAK_SIGNAL_IDS in
# helios-research export_pilot.py:133 and conviction_share_daily.py:22).
TEST_LEAK_SIGNAL_IDS = (80668, 80669)
ROUTE_PREFIX = "ROUTE_"
# The model prefix conviction_share_daily.py filters on (LIKE 'qwen2.5:72b%').
CONVICTION_SHARE_MODEL_PREFIX = "qwen2.5:72b"
# Kinds decision.py records although db._NON_ANALYST_SIGNALS lists them.
DECISION_KEEP_KINDS = ("ANALYST_FAILED",)
# decision.parse_mode values agents.analyst._reply_parse_mode writes.
PARSE_MODES = ("json", "json_substring", "salvaged")

# helios3d/record/labels.py verdict_query; params: labels_rule_params().
LABELS_RULE = ("signal NOT IN %s AND symbol IS NOT NULL AND timestamp IS NOT NULL "
               "AND (trading_env=%s OR (trading_env IS NULL AND COALESCE(reasoning, '') NOT LIKE %s)) ")

# helios3d/record/decision.py SIGNALS_SQL; params: decision_rule_params().
DECISION_RULE = "(signal NOT IN %s OR signal IN %s) AND symbol IS NOT NULL AND timestamp IS NOT NULL "

RULE_NAMES = ("labels", "decision", "conviction_share")


def labels_rule_params() -> tuple:
    import db
    from config import TRADING_ENV
    return (db._NON_ANALYST_SIGNALS, TRADING_ENV, db.DEGRADED_SIGNAL_PREFIX + "%")


def decision_rule_params() -> tuple:
    import db
    return (tuple(db._NON_ANALYST_SIGNALS), DECISION_KEEP_KINDS)


def _sql_list(kinds) -> str:
    return ", ".join(f"'{k}'" for k in kinds)


def conviction_share_rule(non_verdict, alias: str = "s") -> str:
    """The WHERE terms of conviction_share_daily.py after its timestamp term,
    on one line (the script spreads them over several). non_verdict is the
    list it inlines (sef_signals.NON_VERDICT there). % is doubled, so the
    fragment goes through a parameterised execute; params:
    conviction_share_rule_params()."""
    a = f"{alias}." if alias else ""
    return (f"({a}trading_env = %s OR {a}trading_env IS NULL) AND {a}id NOT IN %s "
            f"AND ({a}model LIKE '{CONVICTION_SHARE_MODEL_PREFIX}%%' OR {a}model IS NULL) "
            f"AND {a}signal NOT LIKE 'ROUTE\\_%%' AND {a}signal NOT IN ({_sql_list(non_verdict)}) "
            f"AND {a}confidence IS NOT NULL")


def conviction_share_rule_params() -> tuple:
    from config import TRADING_ENV
    return (TRADING_ENV, TEST_LEAK_SIGNAL_IDS)


def rule(name: str, non_verdict=None) -> tuple[str, tuple]:
    """(WHERE fragment, params) for one named rule. For conviction_share the
    inlined list defaults to db._NON_ANALYST_SIGNALS (the SEF copy is kept
    outside this repo and can drift from it)."""
    if name == "labels":
        return LABELS_RULE, labels_rule_params()
    if name == "decision":
        return DECISION_RULE, decision_rule_params()
    if name == "conviction_share":
        if non_verdict is None:
            import db
            non_verdict = db._NON_ANALYST_SIGNALS
        return conviction_share_rule(non_verdict, alias=""), conviction_share_rule_params()
    raise ValueError(f"unknown rule {name!r}; known: {RULE_NAMES}")


# ---------------------------------------------------------------------------
# Flags (pure)
# ---------------------------------------------------------------------------

def _decision_dict(v) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, (str, bytes)):
        try:
            d = json.loads(v)
        except ValueError:
            return {}
        return d if isinstance(d, dict) else {}
    return {}


def row_flags(row: dict, trading_env: str | None = None,
              expected_model_prefix: str | None = None) -> dict:
    """Flags for one signals row; reports only, excludes nothing.

    row keys used (missing keys read as NULL): id (or signal_id), signal,
    model, trading_env, symbol, timestamp, confidence, and either reasoning
    or a precomputed ``degraded`` bool, and either decision (dict or JSON
    text) or a precomputed ``parse_mode``. trading_env and
    expected_model_prefix, when given, add env_mismatch and model_mismatch;
    otherwise those two are None."""
    import db
    kind = row.get("signal")
    sid = row.get("id", row.get("signal_id"))
    env = row.get("trading_env")
    model = row.get("model")
    if "degraded" in row:
        degraded = bool(row["degraded"])
    else:
        degraded = env is None and (row.get("reasoning") or "").startswith(db.DEGRADED_SIGNAL_PREFIX)
    if "parse_mode" in row:
        parse_mode = row["parse_mode"]
    else:
        parse_mode = _decision_dict(row.get("decision")).get("parse_mode")
    is_route = isinstance(kind, str) and kind.startswith(ROUTE_PREFIX)
    record_only = kind in db._NON_ANALYST_SIGNALS
    return {
        "signal_id": sid,
        "kind": kind,
        "is_record_only_kind": record_only,
        "is_decision_keep_kind": kind in DECISION_KEEP_KINDS,
        "is_route_row": is_route,
        "is_route_unlisted": is_route and not record_only,
        "model": model,
        "model_null": model is None,
        "model_mismatch": (None if expected_model_prefix is None
                           else model is not None and not str(model).startswith(expected_model_prefix)),
        "is_test_leak": sid is not None and int(sid) in TEST_LEAK_SIGNAL_IDS,
        "trading_env": env,
        "env_null": env is None,
        "env_mismatch": None if trading_env is None else env is not None and env != trading_env,
        "degraded": degraded,
        "symbol_null": row.get("symbol") is None,
        "timestamp_null": row.get("timestamp") is None,
        "confidence_null": row.get("confidence") is None,
        "parse_mode": parse_mode,
        "parse_mode_unknown": parse_mode is not None and parse_mode not in PARSE_MODES,
        "salvaged": parse_mode == "salvaged",
    }


# Boolean flags the report counts.
COUNTED_FLAGS = ("is_record_only_kind", "is_decision_keep_kind", "is_route_row", "is_route_unlisted",
                 "model_null", "model_mismatch", "is_test_leak", "env_null", "env_mismatch", "degraded",
                 "symbol_null", "timestamp_null", "confidence_null", "parse_mode_unknown", "salvaged")


# ---------------------------------------------------------------------------
# Report (read-only)
# ---------------------------------------------------------------------------

def report_query(since: datetime, until: datetime) -> tuple[str, tuple]:
    """One read of every signals row in [since, until): the flag columns
    (reasoning is reduced to the degraded bool in SQL) and one boolean per
    rule, each built from that rule's own fragment."""
    import db
    cols = []
    params: list = [db.DEGRADED_SIGNAL_PREFIX + "%"]
    for name in RULE_NAMES:
        frag, p = rule(name)
        cols.append(f"COALESCE(({frag}), false) AS in_{name}")
        params.extend(p)
    sql = ("SELECT id, signal, model, trading_env, symbol, timestamp, confidence, "
           "(trading_env IS NULL AND COALESCE(reasoning, '') LIKE %s) AS degraded, "
           "decision->>'parse_mode' AS parse_mode, " + ", ".join(cols) + " "
           "FROM signals WHERE timestamp >= %s AND timestamp < %s ORDER BY id")
    params.extend([since, until])
    return sql, tuple(params)


def tally(rows: list[dict], trading_env: str | None,
          expected_model_prefix: str | None = CONVICTION_SHARE_MODEL_PREFIX) -> dict:
    """Counts per scope ('all' and each rule): rows, each flag, parse_mode values."""
    scopes = {"all": lambda r: True}
    scopes.update({n: (lambda r, n=n: bool(r.get(f"in_{n}"))) for n in RULE_NAMES})
    out = {}
    for scope, keep in scopes.items():
        t = {"rows": 0, **{f: 0 for f in COUNTED_FLAGS}, "parse_mode": {}}
        for r in rows:
            if not keep(r):
                continue
            fl = row_flags(r, trading_env=trading_env, expected_model_prefix=expected_model_prefix)
            t["rows"] += 1
            for f in COUNTED_FLAGS:
                t[f] += bool(fl[f])
            pm = str(fl["parse_mode"])
            t["parse_mode"][pm] = t["parse_mode"].get(pm, 0) + 1
        out[scope] = t
    return out


def report(since: datetime, until: datetime, conn=None) -> dict:
    from config import TRADING_ENV
    from helios3d.schema.conn import connection, read_only
    if conn is None:
        with connection() as c:
            return report(since, until, conn=c)
    sql, params = report_query(since, until)
    with read_only(conn) as cur:
        cur.execute(sql, params)
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    return tally(rows, TRADING_ENV)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rows each verdict rule includes, and flag counts (read-only).")
    p.add_argument("--since", type=date.fromisoformat, required=True, help="ET date, inclusive")
    p.add_argument("--until", type=date.fromisoformat, default=None,
                   help="ET date, exclusive (default: tomorrow ET)")
    args = p.parse_args(argv)
    since = datetime.combine(args.since, time(0, 0), tzinfo=ET)
    until_d = args.until or (datetime.now(ET).date() + timedelta(days=1))
    until = datetime.combine(until_d, time(0, 0), tzinfo=ET)
    out = report(since, until)
    print(f"signals rows {since.date()} to {until.date()} (ET, end exclusive); "
          f"model_mismatch vs {CONVICTION_SHARE_MODEL_PREFIX!r}; conviction_share list = db._NON_ANALYST_SIGNALS")
    header = ["scope", "rows", *COUNTED_FLAGS]
    print("\t".join(header))
    # The script itself uses SEF's NON_VERDICT copy; label the scope so a
    # drift between the lists can't pass for the script's own count.
    label = {"conviction_share": "conviction_share(repo list)"}
    for scope, t in out.items():
        print("\t".join([label.get(scope, scope), *(str(t[h]) for h in header[1:])]))
    for scope, t in out.items():
        print(f"parse_mode {label.get(scope, scope)}: {dict(sorted(t['parse_mode'].items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
