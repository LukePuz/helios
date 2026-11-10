"""Decision records: one per verdict, links matched by id (never by time),
ambiguous joins left NULL with a reason, idempotent re-runs, the id cursor,
and no prompt text or account figure copied into h3d. No real database: a
fake connection answers the module's SQL from in-memory tables."""
import contextlib
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

import db
from helios3d.record import decision as D
from helios3d.schema import migrate

NOW = datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc)
SECRET_PROMPT = "SECRET-PROMPT-TEXT Portfolio Value: $12,345.67"
SECRET_REASONING = "SECRET-REASONING-TEXT"
BALANCE = 987654.321


# ---------------------------------------------------------------------------
# Fake database
# ---------------------------------------------------------------------------

def _ts(v):
    return v if isinstance(v, datetime) else None


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def execute(self, sql, params=None):
        c = self.conn
        s = " ".join(sql.split())
        c.executed.append((s, params))
        self._rows = []
        writes = re.match(r"(INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b", s, re.I)
        if writes and c.read_only:
            raise RuntimeError(f"write in a READ ONLY transaction: {s[:60]}")
        if s == "SET TRANSACTION READ ONLY":
            c.read_only = True
        elif s.startswith("SELECT to_regclass('h3d.schema_version')"):
            self._rows = [("h3d.schema_version" if c.has_schema else None,)]
        elif s.startswith("SELECT to_regclass('h3d.decision_record')"):
            self._rows = [("h3d.decision_record" if 4 in c.versions else None,)]
        elif s.startswith("SELECT version FROM h3d.schema_version"):
            self._rows = [(v,) for v in sorted(c.versions)]
        elif s.startswith("CREATE SCHEMA IF NOT EXISTS h3d"):
            c.has_schema = True
        elif s.startswith("INSERT INTO h3d.schema_version"):
            c.versions.add(params[0])
        elif s == " ".join(D.CURSOR_SQL.split()):
            self._rows = [(max(c.records, default=0),)]
        elif s == " ".join(D.SIGNALS_SQL.split()):
            _prefix, non_analyst, keep, after, cursor_id, since, limit = params
            rows = [r for r in sorted(c.signals, key=lambda r: r["id"])
                    if (r["signal"] not in non_analyst or r["signal"] in keep)
                    and r["id"] > after and (r["id"] > cursor_id or r["timestamp"] >= since)]
            self._rows = [(r["id"], r["symbol"], r["timestamp"], r["signal"], r["confidence"],
                           r.get("trading_env"), r.get("model"), r.get("provenance"),
                           r.get("decision"), r.get("prompt") is not None,
                           r.get("trading_env") is None
                           and (r.get("reasoning") or "").startswith(db.DEGRADED_SIGNAL_PREFIX))
                          for r in rows[:limit]]
        elif s == " ".join(D.APPROVALS_SQL.split()):
            ids, sids, shas = params
            out = []
            for pa in c.approvals:
                prov = (pa.get("analyst_json") or {}).get("provenance") or {}
                psid = None if prov.get("signal_id") is None else str(prov["signal_id"])
                if pa.get("signal_id") in ids or psid in sids or prov.get("prompt_sha") in shas:
                    out.append((pa["id"], pa.get("signal_id"), pa["ticker"], pa.get("trading_env"),
                                pa.get("is_test"), psid, prov.get("prompt_sha")))
            self._rows = out
        elif s == " ".join(D.SHA_SHARERS_SQL.split()):
            shas, non_analyst = params
            groups = {}
            for r in c.signals:
                sha = (r.get("provenance") or {}).get("prompt_sha")
                if sha in shas and r["signal"] not in non_analyst:
                    groups.setdefault(sha, []).append(r["id"])
            self._rows = list(groups.items())
        elif s == D.ORDERS_SQL:
            self._rows = [(o["id"], o["approval_id"]) for o in c.orders if o["approval_id"] in params[0]]
        elif s == D.OUTCOMES_SQL:
            self._rows = [(o["id"], o["approval_id"]) for o in c.outcomes if o["approval_id"] in params[0]]
        elif s == D.PLANS_SQL:
            self._rows = [(p["id"], p["ticker"], p.get("trading_env"))
                          for p in c.plans if p["id"] in params[0]]
        elif s == D.WINDOW_START_SQL:
            ids = [r["id"] for r in c.signals if r["timestamp"] >= params[0]]
            self._rows = [(min(ids) if ids else None,)]
        elif s == " ".join(D.REGIMES_SQL.split()):
            self._rows = [(r["id"], r["timestamp"], r.get("regime_type"), r.get("sector_etf"))
                          for r in c.regimes if r["timestamp"] in params[0]]
        elif s == " ".join(D.UPSERT_SQL.split()):
            rec = dict(zip(D._COLS, params))
            old = c.records.get(rec["signal_id"])
            if old is None:
                rec["built_at"] = rec["updated_at"] = c.tick()
                c.records[rec["signal_id"]] = rec
                self._rows = [(True,)]
            elif old["content_sha"] != rec["content_sha"]:
                rec["built_at"], rec["updated_at"] = old["built_at"], c.tick()
                c.records[rec["signal_id"]] = rec
                self._rows = [(False,)]
        elif s.startswith("SET LOCAL search_path") or re.match(r"(CREATE|ALTER) TABLE (IF (NOT )?EXISTS )?h3d\.", s):
            pass                                       # the h3d migrations: DDL only
        else:
            raise AssertionError(f"unexpected SQL: {s[:120]}")


class FakeConn:
    def __init__(self):
        self.has_schema, self.versions = False, set()
        self.read_only = False
        self.executed, self.commits, self._clock = [], 0, 0
        self.signals, self.approvals, self.orders, self.outcomes = [], [], [], []
        self.plans, self.regimes, self.records = [], [], {}

    def tick(self):
        self._clock += 1
        return self._clock

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        assert not self.read_only
        self.commits += 1

    def rollback(self):
        self.read_only = False

    def writes(self):
        return [(s, p) for s, p in self.executed if s.startswith("INSERT INTO h3d.decision_record")]


REGIME_TS = datetime(2026, 11, 3, 13, 0, 5, 123456, tzinfo=timezone.utc)


def _prov(sha="p1", plan=None, regime_ts=REGIME_TS, source="overall", **extra):
    p = {"code_sha": "abc123-dirty", "head_sha": "abc123", "prompt_template_sha": "pt1",
         "brief_template_sha": "bt1", "prompt_sha": sha, "model": "qwen2.5:72b",
         "model_digest": "dig1", "ollama_version": "0.5.1",
         "config": {"ANALYST_MODEL": "qwen2.5:72b", "MAX_TRADE_PCT_OF_CASH_RESERVE": 0.25},
         "trigger": {"path": "daily_scan", "source": "scanner"},
         "regime_used": {"label": "Risk-On", "score": 70, "source": source,
                         "timestamp": str(regime_ts) if regime_ts else None},
         "gate_flags": {"verdicts_today": 1}, "process": {"pid": 1}}
    if plan is not None:
        p["plan"] = plan
    p.update(extra)
    return p


def _dec(**extra):
    d = {"model_action": "OPEN", "model_conviction": 8, "final_action": "OPEN",
         "final_conviction": 8, "gates": [], "model_output": {
             "action": "OPEN", "conviction": 8, "thesis_strength": 4, "reasoning": SECRET_REASONING,
             "sources": [{"type": "news"}, {"type": "x"}], "thesis_summary": "free text"}}
    d.update(extra)
    return d


def _sig(sid, ticker="MSFT", signal="OPEN", ts=None, **kw):
    return {"id": sid, "symbol": ticker, "signal": signal, "confidence": kw.pop("confidence", 0.8),
            "timestamp": ts or NOW - timedelta(hours=1), "trading_env": kw.pop("env", "live"),
            "model": kw.pop("model", "qwen2.5:72b"), "reasoning": SECRET_REASONING,
            "prompt": kw.pop("prompt", SECRET_PROMPT),
            "provenance": kw.pop("provenance", None) or _prov(sha=f"p{sid}"),
            "decision": kw.pop("decision", None) or _dec()}


def _approval(aid, ticker="MSFT", signal_id=None, prov_signal_id=None, prompt_sha=None, env="live"):
    prov = {}
    if prov_signal_id is not None:
        prov["signal_id"] = prov_signal_id
    if prompt_sha is not None:
        prov["prompt_sha"] = prompt_sha
    return {"id": aid, "ticker": ticker, "signal_id": signal_id, "trading_env": env,
            "is_test": False, "analyst_json": {"provenance": prov}}


@pytest.fixture(autouse=True)
def _builder_sha(monkeypatch):
    monkeypatch.setattr("helios3d.record.labels.code_sha", lambda: "builder1")


def _run(conn, **kw):
    kw.setdefault("now", NOW)
    return D.run(conn=conn, **kw)


# ---------------------------------------------------------------------------
# Records and joins
# ---------------------------------------------------------------------------

def test_backfill_records_each_verdict_with_versions_and_links():
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=_prov(sha="p1", plan={"relation": "NEW_PLAN", "plan_id": 50})),
                    _sig(2, signal="SCREENED_OUT"), _sig(3, signal="ANALYST_FAILED", model=None),
                    _sig(4, signal="WATCH")]
    conn.approvals = [_approval(10, signal_id=1)]
    conn.orders = [{"id": 100, "approval_id": 10}, {"id": 101, "approval_id": 10}]
    conn.outcomes = [{"id": 200, "approval_id": 10}]
    conn.plans = [{"id": 50, "ticker": "MSFT"}]
    conn.regimes = [{"id": 7, "timestamp": REGIME_TS, "regime_type": "overall"},
                    {"id": 8, "timestamp": REGIME_TS, "regime_type": "gics_sector", "sector_etf": "XLK"}]
    s = _run(conn, apply=True)
    assert s["signals"] == 3 and s["inserted"] == 3 and s["migrations_applied"] == [v for v, _, _ in migrate.MIGRATIONS]
    assert sorted(conn.records) == [1, 3, 4]          # SCREENED_OUT is not a verdict
    r = conn.records[1]
    assert r["decision_id"] == D.decision_uuid(1)
    assert (r["ticker"], r["action"], r["conviction"], r["trigger"]) == ("MSFT", "OPEN", 8, "daily_scan")
    assert (r["code_sha"], r["head_sha"], r["prompt_template_sha"], r["brief_template_sha"],
            r["prompt_sha"], r["model"], r["model_digest"]) == \
        ("abc123-dirty", "abc123", "pt1", "bt1", "p1", "qwen2.5:72b", "dig1")
    assert (r["approval_id"], r["order_ids"], r["trade_outcome_ids"]) == (10, [100, 101], [200])
    assert (r["trade_plan_id"], r["regime_row_id"], r["builder_code_sha"]) == (50, 7, "builder1")
    links = json.loads(r["links"])
    assert links["approval"]["method"] == "signal_id_column"
    assert links["trade_plan"]["relation"] == "NEW_PLAN"
    failed = conn.records[3]
    assert failed["model"] == "qwen2.5:72b" and json.loads(failed["links"])["model_source"] == "provenance.model"
    assert json.loads(failed["links"])["approval"]["reason"] == "ANALYST_FAILED is not a verdict"
    w = json.loads(conn.records[4]["links"])
    assert w["approval"]["status"] == "none" and w["orders"]["status"] == "skipped"


def test_provenance_signal_id_links_an_approval_from_before_migration_77():
    conn = FakeConn()
    conn.signals = [_sig(1)]
    conn.approvals = [_approval(10, prov_signal_id=1)]
    _run(conn, apply=True)
    r = conn.records[1]
    assert r["approval_id"] == 10
    assert json.loads(r["links"])["approval"]["method"] == "provenance_signal_id"


def test_unique_prompt_sha_links_and_a_failed_draw_with_the_same_sha_does_not_count():
    conn = FakeConn()
    conn.signals = [_sig(1, signal="ANALYST_FAILED", provenance=_prov(sha="same")),
                    _sig(2, provenance=_prov(sha="same"))]
    conn.approvals = [_approval(10, prompt_sha="same")]
    _run(conn, apply=True)
    assert conn.records[2]["approval_id"] == 10
    assert json.loads(conn.records[2]["links"])["approval"]["method"] == "prompt_sha"
    assert conn.records[1]["approval_id"] is None


def test_prompt_sha_on_another_ticker_or_env_does_not_link():
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=_prov(sha="s"))]
    conn.approvals = [_approval(10, ticker="AAPL", prompt_sha="s"),
                      _approval(11, prompt_sha="s", env="paper")]
    _run(conn, apply=True)
    assert conn.records[1]["approval_id"] is None
    assert json.loads(conn.records[1]["links"])["approval"]["status"] == "none"


@pytest.mark.parametrize("approvals,signals,status,reason", [
    ([_approval(10, signal_id=1), _approval(11, signal_id=1)], None, "ambiguous",
     "several approvals carry this signal_id"),
    ([_approval(10, signal_id=1, prov_signal_id=2)], None, "conflict",
     "approval's signal_id column and provenance.signal_id differ"),
    ([_approval(10, prompt_sha="dup"), _approval(11, prompt_sha="dup")],
     [_sig(1, provenance=_prov(sha="dup"))], "ambiguous", "several approvals share this prompt_sha"),
    ([_approval(10, prompt_sha="dup")],
     [_sig(1, provenance=_prov(sha="dup")), _sig(2, provenance=_prov(sha="dup"))], "ambiguous",
     "prompt_sha shared by 2 verdicts"),
])
def test_ambiguous_approval_is_left_null_with_the_reason(approvals, signals, status, reason):
    conn = FakeConn()
    conn.signals = signals or [_sig(1)]
    conn.approvals = approvals
    conn.orders = [{"id": 100, "approval_id": 10}]
    _run(conn, apply=True)
    r = conn.records[1]
    links = json.loads(r["links"])
    assert r["approval_id"] is None and r["order_ids"] == [] and r["trade_outcome_ids"] == []
    assert (links["approval"]["status"], links["approval"]["reason"]) == (status, reason)
    assert links["approval"]["candidates"]
    assert links["orders"]["status"] == "skipped"


def test_several_outcome_rows_are_kept_and_flagged():
    conn = FakeConn()
    conn.signals = [_sig(1)]
    conn.approvals = [_approval(10, signal_id=1)]
    conn.outcomes = [{"id": 200, "approval_id": 10}, {"id": 201, "approval_id": 10}]
    _run(conn, apply=True)
    r = conn.records[1]
    assert r["trade_outcome_ids"] == [200, 201]
    assert json.loads(r["links"])["trade_outcomes"]["status"] == "ambiguous"


def test_regime_rows_ambiguous_missing_and_sector_source():
    conn = FakeConn()
    other = REGIME_TS + timedelta(seconds=1)
    conn.signals = [_sig(1), _sig(2, provenance=_prov(regime_ts=other)),
                    _sig(3, provenance=_prov(source="gics_sector:XLK")),
                    _sig(4, provenance=_prov(regime_ts=None, source="fresh"))]
    conn.regimes = [{"id": 7, "timestamp": REGIME_TS, "regime_type": "overall"},
                    {"id": 8, "timestamp": REGIME_TS, "regime_type": None},
                    {"id": 9, "timestamp": REGIME_TS, "regime_type": "gics_sector", "sector_etf": "XLK"}]
    _run(conn, apply=True)
    l1 = json.loads(conn.records[1]["links"])["regime"]
    assert conn.records[1]["regime_row_id"] is None and l1["status"] == "ambiguous"
    assert l1["candidates"] == [7, 8]
    assert json.loads(conn.records[2]["links"])["regime"]["status"] == "none"
    assert conn.records[3]["regime_row_id"] == 9
    assert json.loads(conn.records[4]["links"])["regime"]["reason"] == "regime_used has no row timestamp"


def test_plan_ids_are_checked_against_trade_plans():
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=_prov(plan={"relation": "REVISE", "plan_id": 50, "new_plan_id": 51})),
                    _sig(2, provenance=_prov(plan={"relation": "FOLLOW", "plan_id": 60})),
                    _sig(3, provenance=_prov(plan={"relation": "FOLLOW", "plan_id": 70})),
                    _sig(4, provenance=_prov(plan={"relation": "NO_PLAN"}))]
    conn.plans = [{"id": 50, "ticker": "MSFT"}, {"id": 51, "ticker": "MSFT"}, {"id": 60, "ticker": "AAPL"}]
    _run(conn, apply=True)
    assert (conn.records[1]["trade_plan_id"], conn.records[1]["trade_plan_new_id"]) == (50, 51)
    assert conn.records[2]["trade_plan_id"] is None
    assert json.loads(conn.records[2]["links"])["trade_plan"]["status"] == "conflict"
    assert json.loads(conn.records[3]["links"])["trade_plan"]["status"] == "missing"
    assert json.loads(conn.records[4]["links"])["trade_plan"]["status"] == "none"


# ---------------------------------------------------------------------------
# Re-runs and the cursor
# ---------------------------------------------------------------------------

def test_rerun_is_idempotent_and_writes_nothing():
    conn = FakeConn()
    conn.signals = [_sig(1), _sig(2, signal="WATCH")]
    conn.approvals = [_approval(10, signal_id=1)]
    _run(conn, apply=True)
    before = {k: dict(v) for k, v in conn.records.items()}
    s = _run(conn, apply=True)
    assert s["signals"] == 2                           # both inside the relink window
    assert (s["inserted"], s["updated"], s["unchanged"]) == (0, 0, 2)
    assert conn.records == before
    s = _run(conn, apply=True, full=True)
    assert (s["inserted"], s["updated"]) == (0, 0) and conn.records == before


def test_cursor_reads_only_new_ids_outside_the_relink_window():
    old = NOW - timedelta(days=60)
    conn = FakeConn()
    conn.signals = [_sig(1, ts=old), _sig(2, ts=old)]
    s = _run(conn, apply=True)
    assert s["cursor"] == 0 and s["inserted"] == 2
    conn.signals.append(_sig(3, ts=old))
    s = _run(conn, apply=True)
    assert s["cursor"] == 2 and s["signals"] == 1 and s["inserted"] == 1
    s = _run(conn, apply=True)
    assert s["cursor"] == 3 and s["signals"] == 0


def test_relink_window_fills_late_links_and_catches_out_of_order_ids():
    conn = FakeConn()
    conn.signals = [_sig(1), _sig(5)]
    _run(conn, apply=True)
    assert conn.records[1]["approval_id"] is None
    built = conn.records[1]["built_at"]
    conn.approvals = [_approval(10, signal_id=1)]
    conn.orders = [{"id": 100, "approval_id": 10}]
    conn.signals.append(_sig(4))                       # committed after id 5
    s = _run(conn, apply=True)
    assert (s["inserted"], s["updated"], s["unchanged"]) == (1, 1, 1)
    r = conn.records[1]
    assert (r["approval_id"], r["order_ids"]) == (10, [100])
    assert r["built_at"] == built and r["updated_at"] > built
    assert 4 in conn.records


def test_pages_cover_every_signal():
    conn = FakeConn()
    conn.signals = [_sig(i, signal="WATCH") for i in range(1, 8)]
    s = _run(conn, apply=True, page_size=3)
    assert s["inserted"] == 7 and sorted(conn.records) == list(range(1, 8))


def test_dry_run_writes_nothing_and_applies_no_migration():
    conn = FakeConn()
    conn.signals = [_sig(1)]
    s = _run(conn)
    assert s["mode"] == "dry-run" and s["signals"] == 1 and s["sample"][0]["signal_id"] == 1
    assert conn.commits == 0 and not conn.records and not conn.has_schema
    assert all(re.match(r"(SELECT|SET TRANSACTION READ ONLY)", sql) for sql, _ in conn.executed)


def test_public_tables_are_read_inside_read_only_transactions():
    conn = FakeConn()
    conn.signals = [_sig(1)]
    conn.approvals = [_approval(10, signal_id=1)]
    seen = []
    orig = FakeCursor.execute

    def spy(self, sql, params=None):
        if "public." in sql:
            seen.append(self.conn.read_only)
        return orig(self, sql, params)

    FakeCursor.execute = spy
    try:
        _run(conn, apply=True)
    finally:
        FakeCursor.execute = orig
    assert seen and all(seen)


def test_cli_defaults_to_dry_run(monkeypatch, capsys):
    conn = FakeConn()
    conn.signals = [_sig(1)]

    @contextlib.contextmanager
    def fake_connection():
        yield conn

    monkeypatch.setattr("helios3d.schema.conn.connection", fake_connection)
    assert D.main([]) == 0
    assert conn.commits == 0 and not conn.records
    assert "mode: dry-run" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Nothing private is copied
# ---------------------------------------------------------------------------

def test_no_prompt_reasoning_or_account_figure_reaches_h3d():
    prov = _prov(brief_exposure={"held_broker": True, "account_value": BALANCE,
                                 "nested": {"buying_power": BALANCE, "NetLiquidation": BALANCE}},
                 cash=BALANCE, portfolio_value=BALANCE)
    prov["llm_call"] = {"prompt_eval_count": 900, "prompt": SECRET_PROMPT}
    dec = _dec(gates=["note Cash Available: $5,432.10 seen"], failure={"raw_reply_head": SECRET_PROMPT})
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=prov, decision=dec)]
    _run(conn, apply=True)
    blob = json.dumps([p for _, p in conn.writes()], default=str)
    for bad in ("SECRET-PROMPT", "SECRET-REASONING", "12,345", "5,432", str(BALANCE), "free text"):
        assert bad not in blob, bad
    inputs = json.loads(conn.records[1]["inputs"])
    assert inputs["prompt_stored"] is True
    assert inputs["llm_call"] == {"prompt_eval_count": 900}
    assert inputs["brief_exposure"] == {"held_broker": True, "nested": {}}
    assert inputs["decision"]["gates"] == ["note Cash Available: $[REDACTED] seen"]
    assert inputs["model_scores"] == {"action": "OPEN", "conviction": 8, "thesis_strength": 4,
                                      "source_count": 2}
    # A config threshold named after cash is a setting, not a balance.
    assert inputs["versions"]["config"]["MAX_TRADE_PCT_OF_CASH_RESERVE"] == 0.25
    assert "brief_exposure.account_value" in inputs["_dropped_keys"]
    assert "provenance.cash" in inputs["_not_copied"]


def test_signals_read_takes_only_a_flag_from_prompt_and_reasoning():
    select = D.SIGNALS_SQL.split("FROM")[0]
    assert re.findall(r"\bprompt\b[^_]", select) == ["prompt "]
    assert "(prompt IS NOT NULL)" in select
    assert "reasoning, '') LIKE" in select
    for sql in (D.APPROVALS_SQL, D.SHA_SHARERS_SQL, D.ORDERS_SQL, D.OUTCOMES_SQL, D.PLANS_SQL,
                D.REGIMES_SQL):
        assert not re.search(r"\b(prompt|reasoning|analyst_json|risk_json)\b(?!->)", sql), sql
        assert "cash" not in sql.lower() and "value" not in sql.lower()


@pytest.mark.parametrize("key,denied", [
    ("cash", True), ("total_value", True), ("buying_power", True), ("NetLiquidation", True),
    ("account_id", True), ("unrealized_pnl", True), ("prompt", True), ("reasoning", True),
    # camelCase and squashed IBKR tags.
    ("TotalCashValue", True), ("cashBalance", True), ("SettledCash", True),
    ("EquityWithLoanValue", True), ("accountId", True), ("acctId", True), ("DailyPnL", True),
    ("mktValue", True), ("AvailableFunds", True), ("InitMarginReq", True),
    # UPPER_SNAKE outside versions.config is not exempt.
    ("CASH", True), ("NET_LIQUIDATION", True), ("ONE_SHARE_MAX_PCT_OF_CASH", True),
    ("prompt_sha", False), ("prompt_eval_count", False), ("fundamentals_read", False),
    ("held_broker_reason", False), ("price", False), ("unavailable", False),
])
def test_deny_rule(key, denied):
    assert D._denied(key, 1) is denied
    assert D._denied(key, 1, "decision.nested") is denied


def test_config_exemption_only_under_versions_config():
    assert D._denied("ONE_SHARE_MAX_PCT_OF_CASH", 0.5, "versions.config") is False
    assert D._denied("ONE_SHARE_MAX_PCT_OF_CASH", 0.5, "brief_exposure") is True
    # A string under config is not a scalar threshold.
    assert D._denied("CASH_NOTE", "x", "versions.config") is True


def test_camelcase_and_upper_snake_account_keys_are_dropped_end_to_end():
    prov = _prov(brief_exposure={"TotalCashValue": BALANCE, "cashBalance": BALANCE,
                                 "EquityWithLoanValue": BALANCE, "acctId": "x", "DailyPnL": BALANCE,
                                 "mktValue": BALANCE, "CASH": BALANCE, "NET_LIQUIDATION": BALANCE,
                                 "held_broker": False})
    prov["config"]["NET_LIQUIDATION"] = BALANCE      # scalar under versions.config: a setting name
    prov["gate_flags"] = {"SettledCash": BALANCE, "verdicts_today": 2}
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=prov)]
    _run(conn, apply=True)
    inputs = json.loads(conn.records[1]["inputs"])
    assert inputs["brief_exposure"] == {"held_broker": False}
    assert inputs["gate_flags"] == {"verdicts_today": 2}
    blob = json.dumps([p for _, p in conn.writes()], default=str)
    assert blob.count(str(BALANCE)) == 1               # only the config value
    assert inputs["versions"]["config"]["NET_LIQUIDATION"] == BALANCE


@pytest.mark.parametrize("text,leaks", [
    ("order on account DU1234567 rejected", ["DU1234567"]),
    ("acct U12345678: margin call", ["U12345678"]),
    ("Cash Available: $5,432.10", ["5,432"]),
    ("net liquidation now $123,456", ["123,456"]),
    ("equity was $98,765.43 at open", ["98,765"]),
    ("$45,000 in cash left", ["45,000"]),
    ("| Value: $7,777", ["7,777"]),
])
def test_free_text_account_ids_and_figures_are_redacted(text, leaks):
    out = D.redact_text(text)
    for leak in leaks:
        assert leak not in out, out
    assert D.redact_text("price target $250 by Q3") == "price target $250 by Q3"


@pytest.mark.parametrize("key", ["NetWorth", "net_worth", "capitalAtRisk", "CAPITAL",
                                 "notional_usd", "positionSize", "position_size", "dollars_left"])
def test_extra_deny_fragments(key):
    assert D._denied(key, 1) is True


@pytest.mark.parametrize("text,leak", [
    ("NetLiquidation=99999", "99999"),
    ("balance: 5,000 USD", "5,000"),
    ("TotalCashValue: 12345.67", "12345"),
    ("equity now 250,000", "250,000"),
    ("AvailableFunds=8888 after fill", "8888"),
    ("margin used 4,321.5 today", "4,321"),
    ("NetLiquidation=99999.", "99999"),          # sentence punctuation after the figure
    ("cash 5000.", "5000"),
    ("cash 5,000, then more", "5,000"),
    ("Cash balance $25,000.", "25,000"),
    ("net liquidation value 50,000", "50,000"),     # account-qualified value / NAV
    ("account value 99,999", "99,999"),
    ("total account value 12,345.", "12,345"),
    ("portfolio value: 12345", "12345"),
    ("my NAV 12,345", "12,345"),
    ("our NAV: $12,345 at close", "12,345"),
    ("account NAV 99,999", "99,999"),
    ("portfolio nav=45,000", "45,000"),
])
def test_figures_without_a_dollar_sign_are_redacted(text, leak):
    out = D.redact_text(text)
    assert leak not in out and "[REDACTED]" in out, out


@pytest.mark.parametrize("text,acct", [
    ("order on du1234567 rejected", "du1234567"),   # any case
    ("acct_U1234567 margin", "U1234567"),           # joined with "_"
    ("id=Df7654321;", "Df7654321"),
    ("x-u12345678-y", "u12345678"),
    ("acctU1234567 rejected", "U1234567"),          # glued onto "acct"/"account"
    ("accountDU1234567", "DU1234567"),
])
def test_account_ids_in_any_case_or_joined_are_redacted(text, acct):
    out = D.redact_text(text, figures=False)
    assert acct not in out and "[REDACTED_ACCT]" in out, out


@pytest.mark.parametrize("text", [
    "enterprise value 3,000 bn",                     # a company figure, no account label
    "navigate 12345", "NAV 2026 outlook", "value 2026 guide",
    "US1234567890 isin", "ticker F 1234567", "AU12345678 lot",
    "https://example.com/news/cash-flow/2026/equity-12345",
    "BU1234567 batch", "menuF12345678",               # glued onto other letters: kept
])
def test_value_labels_leave_other_text(text):
    assert D.redact_text(text, figures="https://" not in text) == text


# Company figures that carry a value/NAV label but no account qualifier.
COMPANY_FIGURES = ["Enterprise value: 2,345", "Fair Value: 210", "book value: 1234",
                   "intrinsic value: 185.50", "NAV 1,250", "Market Value: 2,500,000",
                   "Deal value: 1,200 million", "value = 2024",
                   "https://example.com/a/value=12345/"]


@pytest.mark.parametrize("text", COMPANY_FIGURES)
def test_company_value_and_nav_figures_are_not_redacted(text):
    assert D.redact_text(text) == text
    assert D.scrub({"note": text}) == {"note": text}


@pytest.mark.parametrize("text", ["margin of safety 35%", "cash flow up 12% in Q3",
                                  "price target 250 by Q3", "conviction 7 of 10"])
def test_no_dollar_rule_leaves_other_numbers(text):
    assert D.redact_text(text) == text


@pytest.mark.parametrize("text", [
    "equity 2026 outlook",                      # a year after equity
    "cash on 2026-11-03 per filing",            # a date after cash
    "cash 11/03 settlement",
    "balance sheet 2026 outlook",
    "funds 1999 vintage",
    "cash flow guidance for 2026.",             # a year at the end of a sentence
])
def test_years_and_dates_are_not_figures(text):
    assert D.redact_text(text) == text


def test_url_and_title_keep_public_text_but_not_account_ids():
    url = "https://example.com/news/cash-flow/2026/equity-12345"
    out = D.scrub({"trigger": {"url": url, "title": "Free cash flow 5,000 USD; DU1234567",
                               "note": "cash 5,000"}})
    assert out["trigger"]["url"] == url
    assert out["trigger"]["title"] == "Free cash flow 5,000 USD; [REDACTED_ACCT]"
    assert out["trigger"]["note"] == "cash [REDACTED]"


def test_public_text_exemption_is_by_path_not_key_name():
    out = D.scrub({"x": {"title": "NetLiquidation=99999", "url": "u/cash 5,000"}})
    assert out["x"]["title"] == "NetLiquidation=[REDACTED]"
    assert out["x"]["url"] == "u/cash [REDACTED]"


@pytest.mark.parametrize("text,leak", [
    ("cash 1999.50", "1999.50"),                # a year-like number with decimals
    ("NetLiquidation=2025", "2025"),            # an assigned value, not a year
    ("cash 5,000/month", "5,000"),
    ("cash 7,500-ish", "7,500"),
])
def test_figures_that_look_like_years_or_dates_are_still_redacted(text, leak):
    out = D.redact_text(text)
    assert leak not in out and "[REDACTED]" in out, out


def test_truncated_strings_are_recorded():
    dropped = []
    D.scrub({"x": "a" * 5000, "failure": {"error": "e" * 300}, "ok": "short"}, "", dropped)
    assert "x (truncated from 5000 chars)" in dropped
    assert "failure.error (truncated from 300 chars)" in dropped
    assert not any(d.startswith("ok") for d in dropped)


def test_long_cash_string_is_cut_and_fast():
    import time
    s = "cash" * 50_000                         # 200k chars
    t0 = time.perf_counter()
    out = D.scrub({"x": s, "y": "cash " * 40_000 + "123"})
    assert time.perf_counter() - t0 < 1.0
    assert len(out["x"]) == D._TEXT_CAP and len(out["y"]) <= D._TEXT_CAP
    t0 = time.perf_counter()
    D.redact_text(s)
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.parametrize("acct", ["F1234567", "DF7654321", "U1234567", "DU123456789"])
def test_advisor_and_individual_account_ids_are_redacted(acct):
    out = D.redact_text(f"routed via {acct} ok")
    assert acct not in out and "[REDACTED_ACCT]" in out


def test_account_id_in_free_text_never_reaches_h3d():
    dec = _dec(gates=["blocked: account DU7654321 has no permission"],
               failure={"kind": "timeout", "reason": "DU7654321 " + "x" * 500,
                        "error": "boom " * 100})
    prov = _prov(trigger={"path": "event", "title": "U99887766 filing"})
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=prov, decision=dec)]
    _run(conn, apply=True)
    blob = json.dumps([p for _, p in conn.writes()], default=str)
    assert "DU7654321" not in blob and "U99887766" not in blob
    inputs = json.loads(conn.records[1]["inputs"])
    assert len(inputs["decision"]["failure"]["reason"]) == 120
    assert inputs["decision"]["failure"]["reason"].startswith("[REDACTED_ACCT]")
    assert len(inputs["decision"]["failure"]["error"]) == 120


def test_free_text_caps():
    dec = _dec(catalyst_check={"model_date": "y" * 300, "ok": True},
               gate_flags_error=None)
    dec["model_output"].update(action="OPEN" * 30, thesis_bucket="B" * 99, catalyst="free text",
                               upside_downside_ratio=2.5)
    prov = _prov(knowledge_coverage={"error": "e" * 400}, provenance_error="z" * 400)
    inputs = D.recover_inputs(prov, dec)
    assert inputs["decision"]["catalyst_check"]["model_date"] == "y" * 40
    assert inputs["knowledge_coverage"]["error"] == "e" * 120
    assert inputs["provenance_error"] == "z" * 120
    ms = inputs["model_scores"]
    assert len(ms["action"]) == 40 and len(ms["thesis_bucket"]) == 40
    assert "catalyst" not in ms and ms["upside_downside_ratio"] == 2.5


@pytest.mark.parametrize("keys", [
    {"downgraded_reason": "malformed_catalyst", "malformed_fields": ["catalyst.date"],
     "return_malformed": False},
    {"downgraded_reason": None, "malformed_fields": None, "return_malformed": True},
    {"downgraded_reason": None, "malformed_fields": None, "return_malformed": None},
    {"malformed_sell_alert": "throttled"},
    {"malformed_sell_alert": None},
    {"nonfinite_paths": ["decision.model_output.expected_return_pct"]},
    {"held_review": True},
    {"imputed_fields": ["conviction"]},
    {"trend_template": {"pass": True, "criteria": {"above_200": True}}},
    {"trend_template": {"status": "insufficient_history"}},
    {"parse_mode": "salvaged"},
])
def test_malformed_catalyst_keys_are_kept(keys):
    inputs = D.recover_inputs(_prov(), _dec(**keys))
    for k, v in keys.items():
        assert inputs["decision"][k] == v
        assert f"decision.{k}" not in inputs["_not_copied"]
    assert not any("malformed" in k or "downgraded" in k for k in inputs["_dropped_keys"])


def test_content_sha_uses_utc():
    from zoneinfo import ZoneInfo
    sig = {"signal_id": 1, "ticker": "MSFT", "action": "OPEN", "confidence": 0.8,
           "as_of_ts": datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc),
           "trading_env": "live", "model": "m", "provenance": _prov(), "decision": _dec()}
    a = D.build_record(sig, {}, "b")
    sig_et = dict(sig, as_of_ts=sig["as_of_ts"].astimezone(ZoneInfo("America/New_York")))
    assert D.build_record(sig_et, {}, "b")["content_sha"] == a["content_sha"]


def test_incremental_paging_starts_at_the_relink_window_not_zero():
    old = NOW - timedelta(days=60)
    conn = FakeConn()
    conn.signals = [_sig(i, signal="WATCH", ts=old) for i in range(1, 6)] + \
                   [_sig(i, signal="WATCH") for i in range(6, 9)]
    _run(conn, apply=True)                             # backfill (cursor 0)
    conn.executed.clear()
    conn.signals.append(_sig(9, signal="WATCH", ts=old))
    s = _run(conn, apply=True)
    assert s["cursor"] == 8 and s["page_start"] == 5   # first window id 6, minus 1
    first_page = next(p for q, p in conn.executed if q == " ".join(D.SIGNALS_SQL.split()))
    assert first_page[3] == 5
    assert s["signals"] == 4 and s["inserted"] == 1
    # With no signal in the window, paging starts at the cursor.
    s = _run(conn, apply=True, relink_days=0, now=NOW + timedelta(days=1))
    assert s["page_start"] == 9 and s["signals"] == 0


def test_plan_on_another_trading_env_is_a_conflict():
    conn = FakeConn()
    conn.signals = [_sig(1, provenance=_prov(plan={"relation": "FOLLOW", "plan_id": 50}))]
    conn.plans = [{"id": 50, "ticker": "MSFT", "trading_env": "paper"}]
    _run(conn, apply=True)
    assert conn.records[1]["trade_plan_id"] is None
    link = json.loads(conn.records[1]["links"])["trade_plan"]
    assert link["status"] == "conflict" and "trading_env paper" in link["reason"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_decision_record_migration_has_every_written_column():
    v, name, sql = next(m for m in migrate.MIGRATIONS if m[1] == "decision_record")
    assert v >= 4
    for col in D._COLS + ("built_at", "updated_at"):
        assert re.search(rf"^\s*{col}\s", sql, re.M), col
    assert re.search(r"decision_record_signal_uq UNIQUE\s*\(\s*signal_id\s*\)", sql)
    assert "REFERENCES" not in sql.upper()
    assert "prompt " not in sql and "reasoning" not in sql


def test_decision_id_is_stable():
    assert D.decision_uuid(42) == D.decision_uuid(42) != D.decision_uuid(43)
