"""verdict_rows: flags per kind of signals row, byte-identical SQL for the
refactored callers, and the read-only report. DB is mocked."""
from datetime import datetime, timezone

import pytest

import db
from config import TRADING_ENV
from helios3d.record import decision as D
from helios3d.record import labels as L
from helios3d.record import verdict_rows as V

TS = datetime(2026, 11, 3, 15, tzinfo=timezone.utc)

# The caller SQL before the refactor, copied verbatim.
OLD_LABELS_SQL = ("SELECT id, symbol, timestamp, signal, confidence FROM signals "
                  "WHERE signal NOT IN %s AND symbol IS NOT NULL AND timestamp IS NOT NULL "
                  "AND (trading_env=%s OR (trading_env IS NULL AND COALESCE(reasoning, '') NOT LIKE %s)) ")
OLD_DECISION_SIGNALS_SQL = (
    "SELECT id, symbol, timestamp, signal, confidence, trading_env, model, provenance, decision, "
    "(prompt IS NOT NULL) AS prompt_stored, "
    "(trading_env IS NULL AND COALESCE(reasoning, '') LIKE %s) AS degraded "
    "FROM public.signals "
    "WHERE (signal NOT IN %s OR signal IN %s) AND symbol IS NOT NULL AND timestamp IS NOT NULL "
    "AND id > %s AND (id > %s OR timestamp >= %s) "
    "ORDER BY id LIMIT %s"
)
# conviction_share_daily.py:35-39 (helios-research, not refactored), the
# lines after its timestamp term, with {sef_signals.sql_list()} left in. The
# last term carries the s. alias; conviction_share_daily.py:41 needs the
# same change to stay in step with this copy.
OLD_CONVICTION_SHARE_WHERE = """(s.trading_env = %s OR s.trading_env IS NULL) AND s.id NOT IN %s
                      AND (s.model LIKE 'qwen2.5:72b%%' OR s.model IS NULL)
                      AND s.signal NOT LIKE 'ROUTE\\_%%' AND s.signal NOT IN ({sql_list})
                      AND s.confidence IS NOT NULL"""


def _row(**kw):
    base = {"id": 1, "signal": "OPEN", "model": "qwen2.5:72b-instruct-q4_K_M", "trading_env": TRADING_ENV,
            "symbol": "AAA", "timestamp": TS, "confidence": 0.7, "reasoning": "fine",
            "decision": {"parse_mode": "json"}}
    base.update(kw)
    return base


# --- byte-identical SQL ----------------------------------------------------

def test_labels_sql_byte_identical_to_old_literal():
    sql, params = L.verdict_query(None)
    assert sql == OLD_LABELS_SQL + "ORDER BY id"
    assert params[0] is db._NON_ANALYST_SIGNALS
    assert params == (db._NON_ANALYST_SIGNALS, TRADING_ENV, db.DEGRADED_SIGNAL_PREFIX + "%")
    since = datetime(2026, 9, 17, tzinfo=timezone.utc)
    sql2, params2 = L.verdict_query(since)
    assert sql2 == OLD_LABELS_SQL + "AND timestamp >= %s " + "ORDER BY id"
    assert params2 == params + (since,)


def test_decision_sql_byte_identical_to_old_literal():
    assert D.SIGNALS_SQL == OLD_DECISION_SIGNALS_SQL
    assert D._KEEP_NON_ANALYST == ("ANALYST_FAILED",)
    assert V.decision_rule_params() == (tuple(db._NON_ANALYST_SIGNALS), D._KEEP_NON_ANALYST)


def test_conviction_share_rule_matches_script_text():
    kinds = ("SCREENED_OUT", "ROUTE_NO_PRICE")
    sql_list = ", ".join(f"'{k}'" for k in kinds)
    want = " ".join(OLD_CONVICTION_SHARE_WHERE.replace("{sql_list}", sql_list).split())
    assert V.conviction_share_rule(kinds) == want
    assert V.conviction_share_rule_params() == (TRADING_ENV, (80668, 80669))


def test_rules_differ_and_are_named():
    assert set(V.RULE_NAMES) == {"labels", "decision", "conviction_share"}
    frags = {n: V.rule(n)[0] for n in V.RULE_NAMES}
    assert "trading_env" not in frags["decision"]
    assert "model" not in frags["labels"] and "model" not in frags["decision"]
    assert "NOT LIKE 'ROUTE" in frags["conviction_share"]
    assert "parse_mode" not in "".join(frags.values())
    with pytest.raises(ValueError):
        V.rule("nope")


# --- flags per kind of row --------------------------------------------------

def test_flags_plain_verdict():
    f = V.row_flags(_row(), trading_env=TRADING_ENV, expected_model_prefix="qwen2.5:72b")
    assert f["kind"] == "OPEN" and f["parse_mode"] == "json"
    for k in V.COUNTED_FLAGS:
        assert f[k] is False, k


def test_flags_record_only_and_route_kinds():
    f = V.row_flags(_row(signal="SCREENED_OUT"))
    assert f["is_record_only_kind"] and not f["is_route_row"]
    f = V.row_flags(_row(signal="ROUTE_NO_PRICE"))
    assert f["is_record_only_kind"] and f["is_route_row"] and not f["is_route_unlisted"]
    f = V.row_flags(_row(signal="ROUTE_SOMETHING_NEW"))
    assert f["is_route_row"] and f["is_route_unlisted"] and not f["is_record_only_kind"]
    f = V.row_flags(_row(signal="ANALYST_FAILED", model=None))
    assert f["is_record_only_kind"] and f["is_decision_keep_kind"] and f["model_null"]


def test_flags_model():
    assert V.row_flags(_row())["model_mismatch"] is None
    f = V.row_flags(_row(model="qwen2.5:7b"), expected_model_prefix="qwen2.5:72b")
    assert f["model_mismatch"] is True and f["model"] == "qwen2.5:7b"
    f = V.row_flags(_row(model=None), expected_model_prefix="qwen2.5:72b")
    assert f["model_null"] and f["model_mismatch"] is False


def test_flags_test_leak_ids():
    assert V.row_flags(_row(id=80668))["is_test_leak"]
    assert V.row_flags({"signal_id": 80669, "signal": "OPEN"})["is_test_leak"]
    assert not V.row_flags(_row(id=80670))["is_test_leak"]


def test_flags_env_and_degraded():
    f = V.row_flags(_row(trading_env=None, reasoning=db.DEGRADED_SIGNAL_PREFIX + "x"), trading_env=TRADING_ENV)
    assert f["env_null"] and f["degraded"] and f["env_mismatch"] is False
    f = V.row_flags(_row(trading_env=None))
    assert f["env_null"] and not f["degraded"] and f["env_mismatch"] is None
    # The prefix counts as degraded only on a NULL-env row (the callers' rule).
    assert not V.row_flags(_row(reasoning=db.DEGRADED_SIGNAL_PREFIX + "x"))["degraded"]
    assert V.row_flags(_row(trading_env="other"), trading_env=TRADING_ENV)["env_mismatch"] is True
    assert V.row_flags({"id": 2, "signal": "OPEN", "degraded": True})["degraded"]


def test_flags_parse_mode():
    assert V.row_flags(_row(decision={"parse_mode": "salvaged"}))["salvaged"]
    assert V.row_flags(_row(decision='{"parse_mode": "json_substring"}'))["parse_mode"] == "json_substring"
    f = V.row_flags(_row(decision=None))
    assert f["parse_mode"] is None and not f["salvaged"] and not f["parse_mode_unknown"]
    assert V.row_flags(_row(decision="not json"))["parse_mode"] is None
    assert V.row_flags(_row(decision={"parse_mode": "odd"}))["parse_mode_unknown"]
    assert V.row_flags({"id": 3, "signal": "OPEN", "parse_mode": "salvaged"})["salvaged"]


def test_flags_null_columns():
    f = V.row_flags({"id": 4, "signal": "OPEN"})
    assert f["symbol_null"] and f["timestamp_null"] and f["confidence_null"] and f["model_null"]


# --- report -----------------------------------------------------------------

class _Cur:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))
        if sql.startswith("SELECT id, signal"):
            self.description = [(n,) for n in self.conn.cols]
            self._rows = self.conn.rows

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, cols, rows):
        self.cols, self.rows, self.executed, self.commits = cols, rows, [], 0

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        pass

    def commit(self):
        self.commits += 1


def test_report_query_builds_from_each_rule_and_counts_placeholders():
    s, u = datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2026, 11, 2, tzinfo=timezone.utc)
    sql, params = V.report_query(s, u)
    for n in V.RULE_NAMES:
        assert f"COALESCE(({V.rule(n)[0]}), false) AS in_{n}" in sql
    assert sql.replace("%%", "").count("%s") == len(params)
    assert params[-2:] == (s, u)
    assert "reasoning, '') LIKE %s) AS degraded" in sql and "SELECT id, signal" in sql


def test_report_read_only_and_tallies():
    cols = ["id", "signal", "model", "trading_env", "symbol", "timestamp", "confidence", "degraded",
            "parse_mode", "in_labels", "in_decision", "in_conviction_share"]
    rows = [
        (1, "OPEN", "qwen2.5:72b", TRADING_ENV, "AAA", TS, 0.7, False, "json", True, True, True),
        (80668, "OPEN", "qwen2.5:72b", TRADING_ENV, "AAA", TS, 0.7, False, "salvaged", True, True, False),
        (3, "HOLD", None, None, "BBB", TS, 0.6, True, None, False, True, True),
        (4, "ROUTE_NO_PRICE", None, TRADING_ENV, "CCC", TS, 0.0, False, None, False, False, False),
    ]
    conn = _Conn(cols, rows)
    out = V.report(datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2026, 11, 4, tzinfo=timezone.utc),
                   conn=conn)
    assert conn.executed[0][0] == "SET TRANSACTION READ ONLY" and conn.commits == 0
    assert out["all"]["rows"] == 4 and out["all"]["is_route_row"] == 1 and out["all"]["is_test_leak"] == 1
    assert out["labels"]["rows"] == 2 and out["labels"]["salvaged"] == 1
    assert out["decision"]["rows"] == 3 and out["decision"]["degraded"] == 1 and out["decision"]["env_null"] == 1
    assert out["conviction_share"]["rows"] == 2 and out["conviction_share"]["is_test_leak"] == 0
    assert out["labels"]["parse_mode"] == {"json": 1, "salvaged": 1}
