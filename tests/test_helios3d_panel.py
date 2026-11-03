"""helios3d.panel: role prompts, output schemas, PM agreement, red-team
trigger, live-lane gate, kill file, cursor idempotence and scrubbing.
Stubs and fakes only: no model, no Redis, no database."""
import contextlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from helios3d.features.store import FeatureResult
from helios3d.panel import config as C
from helios3d.panel import gate as G
from helios3d.panel import roles, runner, schemas
from helios3d.panel.llm import LLMCallError, LLMResult, LLMUnavailable, OllamaClient, extract_json
from helios3d.schema import migrate

ROOT = Path(__file__).resolve().parent.parent
AS_OF = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _bars(n=260, start=100.0, step=0.5):
    d0 = date(2025, 11, 3)
    out = []
    for i in range(n):
        c = start + i * step
        out.append({"date": d0 + timedelta(days=i), "open": c, "high": c + 1, "low": c - 1,
                    "close": c, "volume": 1_000_000 + i})
    return out


# public.fundamentals_history as db.py creates it (migration 44 plus the
# migration 60 columns), values as psycopg2 returns them (Decimal).
REAL_FUNDAMENTALS = {
    "id": 7, "ticker": "NVDA", "recorded_at": AS_OF - timedelta(days=1),
    "price_to_sales": Decimal("25.1"), "ev_to_ebitda": Decimal("40.2"),
    "forward_pe": Decimal("40.5"), "revenue_growth_yoy": Decimal("0.62"),
    "fcf_margin": Decimal("0.44"), "net_profit_margin": Decimal("0.55"),
    "return_on_equity": Decimal("1.15"), "forward_eps": Decimal("4.1"),
    "target_mean_price": Decimal("230"), "target_low_price": Decimal("150"),
    "target_high_price": Decimal("300"), "shares_outstanding": Decimal("24400000000"),
    "net_cash": Decimal("43210000000"), "market_cap": Decimal("4400000000000")}

# market_regime_log.signals_json in the shape agents/market_regime.py writes
# (packet["signals"]): nested dicts, a list and a None.
REAL_SIGNALS = {
    "missing_inputs": ["credit"],
    "last_bar_date": "2026-10-30",
    "trend": {"score": 72, "notes": "SPY above its 50 and 200-day averages"},
    "trade_style": {"label": "trend", "efficiency_ratio": 0.41, "notes": "clean trend"},
    "volatility": {"score": 60, "vix": 15.2, "notes": "VIX below 20"},
    "liquidity": {"score": 55, "notes": "Fed balance sheet flat; liquidity $6.5T"},
    "credit": {"score": None, "notes": "HY spread unavailable"},
    "breadth": {"score": 64, "notes": "62% of members above the 50-day"},
    "volume": {"score": 50, "notes": "average"},
    "markov": {"score": 58, "notes": "bull state", "state": "bull", "stickiness": 0.9,
               "signal": "hold"},
}
REAL_REASONING = ("Regime: Bull | Score: 68/100 | Confidence: 70%\n"
                  "Key drivers: liquidity $6.5T balance sheet; equity breadth 62%")


class FakeFeatures:
    def __init__(self, fundamentals=None, headlines=None, signals=None):
        self.calls = []
        self.fund = fundamentals if fundamentals is not None else dict(REAL_FUNDAMENTALS)
        self.signals = signals if signals is not None else REAL_SIGNALS
        self.headlines = headlines if headlines is not None else [
            {"id": 1, "title": "NVDA beats estimates", "published_at": AS_OF - timedelta(hours=3)}]

    def __call__(self, source, ticker, as_of, **kw):
        self.calls.append((source, ticker, as_of))
        assert as_of == AS_OF
        rows, safe = [], True
        if source == "fundamentals":
            rows = [dict(self.fund)]
        elif source == "earnings_calendar":
            rows = [{"earnings_date": date(2026, 11, 19), "confirmed": True,
                     "recorded_at": AS_OF - timedelta(days=2)}]
        elif source == "price_bars_asof":
            rows = []                       # empty before the first capture: falls back
        elif source == "price_bars":
            rows, safe = _bars(), False
        elif source == "regime":
            rows = [{"id": 3, "timestamp": AS_OF - timedelta(hours=5), "regime": "Bull",
                     "score": 68, "confidence": 0.7, "signals_json": self.signals,
                     "reasoning": REAL_REASONING}]
        elif source == "news":
            rows = [dict(r, logged_at=r["published_at"]) for r in self.headlines]
        else:
            raise AssertionError(f"unexpected source {source}")
        return FeatureResult(source, ticker, as_of, rows, "col", safe, "reason")


ANALYST_OK = {"rationale": "Growth is strong.", "what_would_change_my_mind": "A guide cut.",
              "score": 8, "confidence": 0.7}


def _pm(verdict="OPEN", agreement="consensus"):
    return {"thesis": "Panel sees durable growth.", "agreement_rationale": "All bullish.",
            "agreement": agreement,
            "plan": {"entry_zone": {"low": 228, "high": 232}, "add_trigger": "close above 240",
                     "targets": [{"name": "T1", "price": 260, "trim_pct": 25},
                                 {"name": "T2", "price": 290, "trim_pct": 50}],
                     "invalidation": {"price": 215, "reason": "loses the 50-day"},
                     "time_stop_days": 30,
                     "scenarios": [{"name": "bull", "probability": 0.3, "description": "a"},
                                   {"name": "base", "probability": 0.5, "description": "b"},
                                   {"name": "bear", "probability": 0.2, "description": "c"}],
                     "earnings_playbook": "hold through"},
            "verdict": verdict, "conviction": 7}


RED_OK = {"counter_thesis": "Valuation is stretched.", "strongest_risks": ["multiple", "supply"],
          "what_would_change_my_mind": "A cheaper entry.", "objection": "material",
          "case_strength": 6, "confidence": 0.6}


def role_of(prompt):
    if prompt.startswith("You are the portfolio manager"):
        return "pm"
    if prompt.startswith("You are the red team"):
        return "red_team"
    m = re.search(r"Your role: ([A-Z/]+) analyst", prompt)
    return {"FUNDAMENTALS": "fundamentals", "TECHNICAL/FLOW": "technical_flow",
            "MACRO/REGIME": "macro_regime", "NEWS/SENTIMENT": "news_sentiment"}[m.group(1)]


class StubLLM:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = {"pm": _pm(), "red_team": RED_OK, **(replies or {})}

    def generate(self, model, prompt, *, num_predict, num_ctx, temperature, seed, timeout_s):
        role = role_of(prompt)
        self.calls.append({"role": role, "model": model, "prompt": prompt,
                           "num_predict": num_predict, "timeout_s": timeout_s})
        r = self.replies.get(role, ANALYST_OK)
        if isinstance(r, Exception):
            raise r
        text = r if isinstance(r, str) else json.dumps(r)
        return LLMResult(text=text, prompt_tokens=100, output_tokens=50, latency_s=1.5,
                         done_reason="stop")


class OpenGate:
    def __init__(self, results=None):
        self.n = 0
        self.results = list(results or [])

    def wait_clear(self):
        self.n += 1
        if self.results:
            return self.results.pop(0)
        return G.GateResult(True, "clear")


def _rec_row(sid, action="OPEN", conv=7, env="paper", ticker="NVDA"):
    return {"decision_id": f"00000000-0000-0000-0000-{sid:012d}", "signal_id": sid,
            "as_of_ts": AS_OF, "ticker": ticker, "trading_env": env, "trigger": "event",
            "action": action, "conviction": conv,
            "trigger_json": {"path": "event", "title": "NVDA beats estimates; acct U1234567"}}


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self._res = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        db = self.db
        db.sql.append((sql, params))
        s = sql.strip()
        if s.startswith("SET TRANSACTION READ ONLY"):
            db.read_only = True
            return
        if s.startswith("SELECT to_regclass"):
            self._res = [(None if any(t in s for t in db.missing) else "x",)]
        elif s.startswith("SELECT last_signal_id FROM h3d.panel_cursor"):
            v = db.committed["cursor"].get(params[0])
            self._res = [(v,)] if v is not None else []
        elif "FROM h3d.decision_record d" in s:
            assert db.read_only
            self._res = db.read_records(s, params)
        elif s.startswith("INSERT INTO h3d.panel_output"):
            assert not db.read_only
            db.maybe_fail("panel_output")
            db.pending.append(("out", dict(zip(runner._OUT_COLS, params))))
        elif s.startswith("INSERT INTO h3d.pm_verdict"):
            assert not db.read_only
            db.maybe_fail("pm_verdict")
            db.pending.append(("pm", dict(zip(runner._PM_COLS, params))))
        elif s.startswith("INSERT INTO h3d.panel_skip"):
            assert not db.read_only
            db.maybe_fail("panel_skip")
            did, sid, cfg, reason = params
            db.pending.append(("skip", {"decision_id": did, "signal_id": sid,
                                        "panel_config_sha": cfg, "reason": reason}))
        elif s.startswith("DELETE FROM h3d.pm_verdict"):
            assert not db.read_only and "LIKE 'write:%%'" in s
            db.pending.append(("del_failed", params[0]))
        elif s.startswith("DELETE FROM h3d.panel_skip"):
            assert not db.read_only
            db.pending.append(("del_skip", params[0]))
        elif s.startswith("INSERT INTO h3d.panel_cursor"):
            assert not db.read_only
            db.pending.append(("cursor", (params[0], params[1])))
        else:
            raise AssertionError(f"unexpected SQL: {s[:80]}")

    def fetchone(self):
        return self._res[0] if self._res else None

    def fetchall(self):
        return list(self._res)


def _is_failed(r):
    return (r.get("error") or "").startswith("write:")


class FakeDB:
    """In-memory h3d tables. The record read is evaluated from the SQL's
    clauses, so a wrong clause shows up as a wrong set of decisions."""

    def __init__(self, records=(), missing=(), fail=None):
        self.records = list(records)
        self.missing = tuple(missing)
        self.fail = dict(fail or {})     # table -> number of inserts that raise
        self.sql = []
        self.read_only = False
        self.commits = 0
        self.committed = {"out": [], "pm": [], "cursor": {}, "skip": {}}
        self.pending = []

    def cursor(self):
        return FakeCursor(self)

    def maybe_fail(self, table):
        if self.fail.get(table, 0) > 0:
            self.fail[table] -= 1
            raise ValueError(f"value too long for type character varying ({table})")

    def read_records(self, s, params):
        params = list(params)
        after, envs, _ = params[:3]
        limit = params[-1]
        cfg = params[3] if "p.panel_config_sha = %s" in s else None
        ok_only = "NOT p.error LIKE 'write:%%'" in s
        pm = [r for r in self.committed["pm"] if not (ok_only and _is_failed(r))]
        if cfg is not None:
            pm = [r for r in pm if r["panel_config_sha"] == cfg]
        done = {r["decision_id"] for r in pm} if "h3d.pm_verdict" in s else set()
        skipped = set(self.committed["skip"]) if "h3d.panel_skip" in s else set()
        rows = [r for r in self.records if r["signal_id"] > after
                and (envs is None or r["trading_env"] in envs)
                and r["decision_id"] not in done and r["decision_id"] not in skipped]
        return [(r["decision_id"], r["signal_id"], r["as_of_ts"], r["ticker"], r["trading_env"],
                 r["trigger"], r["action"], r["conviction"], json.dumps(r["trigger_json"]))
                for r in rows[:limit]]

    def commit(self):
        c = self.committed
        for kind, v in self.pending:
            if kind == "out":
                key = ("decision_id", "role", "panel_config_sha")
                if not any(all(o[k] == v[k] for k in key) for o in c["out"]):
                    c["out"].append(v)                      # ON CONFLICT DO NOTHING
            elif kind == "pm":
                if not any(o["decision_id"] == v["decision_id"]
                           and o["panel_config_sha"] == v["panel_config_sha"] for o in c["pm"]):
                    c["pm"].append(v)
            elif kind == "del_failed":
                c["pm"] = [o for o in c["pm"] if not (o["decision_id"] == v and _is_failed(o))]
            elif kind == "skip":
                c["skip"][v["decision_id"]] = v
            elif kind == "del_skip":
                c["skip"].pop(v, None)
            elif kind == "cursor":
                name, sid = v
                c["cursor"][name] = max(sid, c["cursor"].get(name, 0))
        self.commits += 1
        self.pending = []
        self.read_only = False

    def rollback(self):
        self.pending = []
        self.read_only = False



def _settings(**kw):
    base = dict(budget="full", model="m72", ollama_url="http://shadow.invalid:9",
                live_ollama_url="http://live.invalid:8", gate_wait_s=0)
    base.update(kw)
    return C.PanelSettings(**base)


def _runner(db, llm=None, gate=None, feats=None, apply=True, **kw) -> "runner.PanelRunner":
    return runner.PanelRunner(_settings(**kw), db, llm=llm or StubLLM(), gate=gate or OpenGate(),
                              feature_get=feats or FakeFeatures(), code_sha="abc123",
                              apply=apply)


@pytest.fixture(autouse=True)
def _no_migrations(monkeypatch):
    monkeypatch.setattr(migrate, "apply", lambda conn: [])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_analyst_prompts_share_the_header_prefix_and_end_with_role_instructions():
    feats = FakeFeatures()
    trig = roles.trigger_text("event", {"trigger": {"title": "NVDA beats"}})
    prompts = {}
    for role in C.ANALYST_ROLES:
        slc = roles.build_slice(role, "NVDA", AS_OF, get=feats)
        prompts[role] = roles.render_analyst(role, "NVDA", AS_OF, trig, slc)
    header = roles.shared_header("NVDA", AS_OF, trig)
    for role, p in prompts.items():
        assert p.startswith(header)
        assert p.rstrip().endswith(roles.ROLE_INSTRUCTIONS[role])
        assert '"score"' in header and "NVDA" in header and "event: NVDA beats" in header
    # Each analyst sees only its own slice.
    assert "NVDA beats estimates" in prompts["news_sentiment"]
    assert "forward_pe" in prompts["fundamentals"]
    assert "forward_pe" not in prompts["technical_flow"]
    assert "headlines" not in prompts["fundamentals"]
    assert "relative_strength_vs_spy_pct" in prompts["technical_flow"]
    assert '"label": "Bull"' in prompts["macro_regime"]


def test_slices_read_as_of_the_decision_time_and_flag_price_fallback():
    feats = FakeFeatures()
    slc = roles.build_slice("technical_flow", "NVDA", AS_OF, get=feats)
    assert all(a == AS_OF for _, _, a in feats.calls)
    assert [s["source"] for s in slc["sources"]] == ["price_bars", "price_bars"]
    assert slc["sources"][0]["asof_safe"] is False
    st = slc["data"]["stock"]
    assert st["bars"] == 260 and st["ret_20d_pct"] is not None and st["atr14"] == 2.0


def test_roles_may_not_read_account_sources():
    g = roles._guarded(FakeFeatures(), "fundamentals")
    for src in ("portfolio_snapshot", "position_stops", "news"):
        with pytest.raises(PermissionError):
            g(src, "NVDA", AS_OF)


def test_fundamentals_slice_keeps_every_real_column():
    """The account-word rule would drop fcf_margin, net_profit_margin,
    return_on_equity and net_cash; scalars under fundamentals are exempt."""
    slc = roles.build_slice("fundamentals", "NVDA", AS_OF, get=FakeFeatures())
    f = slc["data"]["fundamentals"]
    want = set(REAL_FUNDAMENTALS) - {"id", "ticker"}
    assert set(f) == want and slc["dropped"] == []
    assert f["fcf_margin"] == 0.44 and f["net_cash"] == 43210000000.0
    assert f["return_on_equity"] == 1.15 and f["net_profit_margin"] == 0.55


def test_exemption_is_by_parent_path_and_scalars_only():
    out = roles.scrub_slice({"fundamentals": {"net_cash": 5, "cash_detail": {"cash": 1},
                                              "note": "acct U1234567"},
                             "net_cash": 5, "other": {"fcf_margin": 0.1}})
    assert out["fundamentals"]["net_cash"] == 5
    assert "cash_detail" not in out["fundamentals"]            # nested dict: full scrub
    assert out["fundamentals"]["note"] == "acct [REDACTED_ACCT]"
    assert "net_cash" not in out and out["other"] == {}         # outside the parent: denied


def test_macro_sees_the_flattened_nested_signals():
    slc = roles.build_slice("macro_regime", "NVDA", AS_OF, get=FakeFeatures())
    sig = slc["data"]["regime"]["signals"]
    assert sig["trend.score"] == 72 and sig["volatility.vix"] == 15.2
    assert sig["liquidity.score"] == 55 and "liquidity.notes" in sig       # not denied
    assert sig["markov.state"] == "bull" and sig["credit.score"] is None
    assert sig["missing_inputs"] == "credit"
    assert slc["dropped"] == []


def test_flatten_signals_is_bounded():
    big = {f"k{i:03d}": {"score": i, "notes": "x" * 500, "deep": {"a": {"b": {"c": 1}}}}
           for i in range(100)}
    flat = roles.flatten_signals(big)
    assert len(flat) == roles.SIGNALS_MAX_KEYS + 1 and flat["_truncated"] > 0
    assert all(len(v) <= roles.SIGNALS_TEXT_CAP for v in flat.values() if isinstance(v, str))
    assert not any(k.count(".") >= roles.SIGNALS_MAX_DEPTH for k in flat)
    assert roles.flatten_signals("not a dict") == {}
    assert roles.flatten_signals(json.loads(json.dumps(REAL_SIGNALS)))["trend.score"] == 72


def test_public_text_keeps_figures_but_not_account_ids():
    feats = FakeFeatures(headlines=[{"id": 1, "title": "NVDA $5,000 calls; acct DU1234567",
                                     "published_at": AS_OF - timedelta(hours=1)}])
    news = roles.build_slice("news_sentiment", "NVDA", AS_OF, get=feats)
    assert news["data"]["headlines"][0]["title"] == "NVDA $5,000 calls; acct [REDACTED_ACCT]"
    macro = roles.build_slice("macro_regime", "NVDA", AS_OF, get=FakeFeatures())
    assert macro["data"]["regime"]["reasoning"] == REAL_REASONING
    # The same figures elsewhere still get the full scrub.
    assert "[REDACTED]" in roles.scrub_slice({"note": "cash balance $5,000"})["note"]


def test_panel_version_bumped():
    assert roles.PANEL_VERSION == 2


def test_trigger_title_redacts_account_ids():
    t = roles.trigger_text("event", {"trigger": {"title": "x U1234567 y"}})
    assert "U1234567" not in t and "[REDACTED_ACCT]" in t


def test_bar_summary_handles_short_history():
    s = roles.bar_summary(_bars(10))
    assert s["bars"] == 10 and s["ret_20d_pct"] is None and s["vs_sma200_pct"] is None
    assert roles.bar_summary([]) == {"bars": 0}


# ---------------------------------------------------------------------------
# Schemas and parsing
# ---------------------------------------------------------------------------

def test_schemas_put_reasoning_before_verdict():
    def order(text, fields):
        return [text.index(f'"{f}"') for f in fields]
    a = order(schemas.ANALYST_SCHEMA_TEXT, schemas.ANALYST_FIELDS)
    assert a == sorted(a)
    p = order(schemas.PM_SCHEMA_TEXT, schemas.PM_FIELDS)
    assert p == sorted(p)
    r = order(schemas.RED_TEAM_SCHEMA_TEXT, schemas.RED_TEAM_FIELDS)
    assert r == sorted(r)


def test_parse_analyst_valid_and_field_order_flag():
    out = schemas.parse_analyst(dict(ANALYST_OK))
    assert out["score"] == 8 and out["confidence"] == 0.7 and out["field_order_ok"]
    flipped = {"score": 8, "confidence": 0.7, "rationale": "r", "what_would_change_my_mind": "w"}
    assert schemas.parse_analyst(flipped)["field_order_ok"] is False


@pytest.mark.parametrize("bad", [
    {"score": 11}, {"score": 0}, {"score": 7.5}, {"score": True}, {"confidence": 1.2},
    {"confidence": None}, {"rationale": ""}, {"what_would_change_my_mind": None}])
def test_parse_analyst_rejects_bad_verdict_fields(bad):
    with pytest.raises(schemas.SchemaError):
        schemas.parse_analyst({**ANALYST_OK, **bad})


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("[1, 2]")


def test_parse_pm_agreement_is_its_own_field():
    out = schemas.parse_pm(_pm(agreement="split"))
    assert out["agreement"] == "split" and out["verdict"] == "OPEN" and out["conviction"] == 7
    with pytest.raises(schemas.SchemaError):
        schemas.parse_pm({**_pm(), "agreement": "mostly"})
    with pytest.raises(schemas.SchemaError):
        schemas.parse_pm({k: v for k, v in _pm().items() if k != "agreement"})


def test_parse_pm_plan_block():
    out = schemas.parse_pm(_pm())
    plan = out["plan"]
    assert plan["entry_zone"] == {"low": 228.0, "high": 232.0}
    assert [t["trim_pct"] for t in plan["targets"]] == [25.0, 50.0]
    assert plan["invalidation"] == {"price": 215.0, "reason": "loses the 50-day"}
    assert plan["time_stop_days"] == 30 and len(plan["scenarios"]) == 3
    assert plan["earnings_playbook"] == "hold through" and out["plan_issues"] == []
    bad = _pm()
    bad["plan"]["scenarios"][0]["probability"] = 0.9
    bad["plan"]["entry_zone"] = {"low": 240, "high": 230}
    issues = schemas.parse_pm(bad)["plan_issues"]
    assert any("sum to" in i for i in issues) and "entry_zone low/high swapped" in issues
    assert schemas.parse_pm({**_pm(), "plan": None})["plan_issues"] == ["plan missing"]


def test_agreement_facts_are_mechanical_without_a_threshold():
    a = {"fundamentals": {"score": 8, "confidence": 0.8}, "technical_flow": {"score": 3, "confidence": 0.4},
         "macro_regime": {"score": 6, "confidence": 0.6}, "news_sentiment": None}
    f = roles.agreement_facts(a)
    assert f["n_scored"] == 3 and f["n_missing"] == 1 and f["range"] == 5
    assert f["n_above_5"] == 2 and f["n_below_5"] == 1 and f["mean"] == pytest.approx(17 / 3, abs=1e-3)
    assert "split" not in json.dumps(f) and "consensus" not in json.dumps(f)


@pytest.mark.parametrize("verdict,fires", [("OPEN", True), ("ADD", True), ("HOLD", False),
                                           ("WATCH", False), ("PASS", False), ("TRIM", False)])
def test_red_team_trigger(verdict, fires):
    assert schemas.is_would_be_buy({"verdict": verdict}) is fires
    assert schemas.is_would_be_buy(None) is False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_full_panel_writes_outputs_pm_and_red_team_on_a_buy():
    db = FakeDB([_rec_row(10)])
    llm = StubLLM()
    s = _runner(db, llm=llm).run()
    assert s["ran"] == 1 and s["red_team"] == 1 and s["yielded"] is None
    assert [c["role"] for c in llm.calls] == list(C.ANALYST_ROLES) + ["pm", "red_team"]
    assert {r["role"] for r in db.committed["out"]} == set(C.ANALYST_ROLES) | {"red_team"}
    pm = db.committed["pm"][0]
    assert pm["agreement"] == "consensus" and pm["verdict"] == "OPEN" and pm["red_team_ran"]
    assert pm["live_action"] == "OPEN" and pm["analysts_ok"] == 4
    assert json.loads(pm["plan"])["targets"][0]["name"] == "T1"
    assert pm["red_team_objection"] == "material" and pm["red_team_case_strength"] == 6
    assert db.committed["cursor"]["panel"] == 10
    # Analyst calls are consecutive and share the header prefix.
    header = roles.shared_header("NVDA", AS_OF, roles.trigger_text(
        "event", {"trigger": _rec_row(10)["trigger_json"]}))
    assert all(c["prompt"].startswith(header) for c in llm.calls[:4])


def test_no_red_team_when_pm_is_not_a_buy():
    db = FakeDB([_rec_row(10)])
    llm = StubLLM({"pm": _pm(verdict="WATCH", agreement="split")})
    s = _runner(db, llm=llm).run()
    assert s["red_team"] == 0 and "red_team" not in [c["role"] for c in llm.calls]
    pm = db.committed["pm"][0]
    assert pm["red_team_ran"] is False and pm["agreement"] == "split"


def test_calls_use_short_timeouts_and_role_models():
    llm = StubLLM()
    _runner(FakeDB([_rec_row(1)]), llm=llm, timeout_s=200,
            role_models={"news_sentiment": "m7"}).run()
    assert all(c["timeout_s"] == 200 < C.LIVE_TIMEOUT_S for c in llm.calls)
    by = {c["role"]: c for c in llm.calls}
    assert by["news_sentiment"]["model"] == "m7" and by["fundamentals"]["model"] == "m72"
    assert by["fundamentals"]["num_predict"] == 300 and by["pm"]["num_predict"] == 700


def test_cursor_idempotence():
    db = FakeDB([_rec_row(10), _rec_row(11, action="WATCH", conv=5)])
    llm = StubLLM()
    first = _runner(db, llm=llm).run()
    assert first["ran"] == 2
    n_calls, n_out, n_pm = len(llm.calls), len(db.committed["out"]), len(db.committed["pm"])
    second = _runner(db, llm=llm).run()
    assert second["ran"] == 0 and second["records"] == 0
    assert len(llm.calls) == n_calls
    assert (len(db.committed["out"]), len(db.committed["pm"])) == (n_out, n_pm)
    assert db.committed["cursor"]["panel"] == 11
    # A late record below the cursor (committed out of order) is still found.
    db.records.insert(0, _rec_row(9))
    third = _runner(db, llm=llm).run()
    assert third["ran"] == 1 and db.committed["cursor"]["panel"] == 11
    # Re-inserting the same rows writes nothing (ON CONFLICT DO NOTHING).
    r = _runner(db, llm=llm)
    res = r.run_decision(runner.fetch_records(FakeDB([_rec_row(10)]), 0, r.config_sha)[0])
    before = len(db.committed["out"])
    r.write(res, 10)
    assert len(db.committed["out"]) == before


def test_a_new_config_does_not_rerun_old_decisions_without_the_flag():
    db = FakeDB([_rec_row(10), _rec_row(11)])
    _runner(db).run()
    llm = StubLLM()
    s = _runner(db, llm=llm, num_predict_analyst=200).run()
    assert s["ran"] == 0 and llm.calls == []
    # Even a lookback reaching below the cursor finds nothing to redo.
    s = _runner(db, llm=llm, num_predict_analyst=200).run(from_signal_id=0)
    assert s["ran"] == 0 and llm.calls == []
    s = _runner(db, llm=llm, num_predict_analyst=200).run(rerun_config=True)
    assert s["ran"] == 2
    assert len({r["panel_config_sha"] for r in db.committed["pm"]}) == 2
    assert _runner(db, num_predict_analyst=200).run(rerun_config=True)["ran"] == 0


def test_lookback_only_picks_ids_with_no_verdict_under_any_config():
    db = FakeDB([_rec_row(i) for i in (5, 6, 7)])
    _runner(db, num_predict_analyst=100).run()                 # config A paneled all three
    db.records.append(_rec_row(4))                            # late, below the cursor
    llm = StubLLM()
    s = _runner(db, llm=llm).run()                              # config B
    assert s["ran"] == 1 and {c["role"] for c in llm.calls} >= {"pm"}
    assert [r["signal_id"] for r in db.committed["pm"]][-1] == 4


def test_overlong_bounded_strings_are_cut_to_column_widths():
    class LongLLM(StubLLM):
        def generate(self, *a, **k):
            r = super().generate(*a, **k)
            r.done_reason = "stop-" + "x" * 100
            return r
    long_model = "m" * 200
    db = FakeDB([_rec_row(10)])
    s = _runner(db, llm=LongLLM(), model=long_model).run()
    assert s["ran"] == 1 and s["write_failures"] == 0
    for r in db.committed["out"]:
        assert len(r["model"]) == 80 and len(r["done_reason"]) == 20
    assert len(db.committed["pm"][0]["model"]) == 80


def test_widths_match_the_migration():
    sql = next(m for m in migrate.MIGRATIONS if m[1] == "panel")[2]
    for table, cols in (("panel_output", runner._OUT_COLS), ("pm_verdict", runner._PM_COLS)):
        body = sql.split(f"h3d.{table} (", 1)[1].split(");", 1)[0]
        widths = {m.group(1): int(m.group(2))
                  for m in re.finditer(r"^\s*(\w+)\s+VARCHAR\((\d+)\)", body, re.M)}
        for col in cols:
            if col in widths and col != "decision_id":
                assert runner._WIDTHS.get(col) == widths[col], (table, col)


def test_a_write_error_records_the_decision_as_failed_and_moves_on():
    db = FakeDB([_rec_row(10), _rec_row(11)], fail={"panel_output": 1})
    s = _runner(db).run()
    assert s["write_failures"] == 1 and s["ran"] == 1 and s["skipped"] == []
    failed = next(r for r in db.committed["pm"] if r["signal_id"] == 10)
    assert failed["parse_ok"] is False and failed["error"].startswith("write: ValueError")
    assert {r["signal_id"] for r in db.committed["out"]} == {11}
    assert db.committed["cursor"]["panel"] == 11
    assert _runner(db).run()["ran"] == 0               # not retried


def test_a_decision_that_cannot_be_written_at_all_is_skipped():
    db = FakeDB([_rec_row(10), _rec_row(11)], fail={"pm_verdict": 2})
    s = _runner(db).run()
    assert s["write_failures"] == 1 and [k["signal_id"] for k in s["skipped"]] == [10]
    assert s["ran"] == 1 and db.committed["cursor"]["panel"] == 11


def _pm_rows(db, sid):
    return [r for r in db.committed["pm"] if r["signal_id"] == sid]


def test_a_failure_row_blocks_the_default_read_but_not_retry_failed():
    db = FakeDB([_rec_row(10)], fail={"panel_output": 1})
    _runner(db).run()
    assert [r["error"][:6] for r in _pm_rows(db, 10)] == ["write:"]
    llm = StubLLM()
    assert _runner(db, llm=llm).run(from_signal_id=0)["ran"] == 0 and llm.calls == []
    s = _runner(db, llm=llm).run(retry_failed=True)
    assert s["ran"] == 1 and s["retry_failed"] is True
    rows = _pm_rows(db, 10)
    assert len(rows) == 1 and rows[0]["parse_ok"] is True and rows[0]["error"] is None
    assert _runner(db).run(retry_failed=True)["ran"] == 0     # done now


def test_a_retry_that_fails_again_replaces_the_failure_row():
    db = FakeDB([_rec_row(10)], fail={"panel_output": 2})
    _runner(db).run()
    _runner(db).run(retry_failed=True)
    rows = _pm_rows(db, 10)
    assert len(rows) == 1 and rows[0]["error"].startswith("write:")


def test_rerun_config_includes_failure_rows_and_replaces_them():
    db = FakeDB([_rec_row(10), _rec_row(11)], fail={"panel_output": 1})
    _runner(db).run()                                     # 10 failed, 11 ok (config A)
    s = _runner(db).run(rerun_config=True)                # same config: only the failure
    assert s["ran"] == 1
    assert all(r["error"] is None for r in db.committed["pm"]) and len(db.committed["pm"]) == 2
    db2 = FakeDB([_rec_row(10)], fail={"panel_output": 1})
    _runner(db2, num_predict_analyst=100).run()           # failure under config A
    s = _runner(db2).run(rerun_config=True)               # config B
    rows = _pm_rows(db2, 10)
    assert s["ran"] == 1 and len(rows) == 1 and rows[0]["error"] is None


def test_a_skipped_decision_gets_a_marker_and_is_not_repanelled():
    db = FakeDB([_rec_row(10), _rec_row(11)], fail={"pm_verdict": 2})
    _runner(db).run()
    assert set(db.committed["skip"]) == {_rec_row(10)["decision_id"]}
    marker = db.committed["skip"][_rec_row(10)["decision_id"]]
    assert marker["signal_id"] == 10 and marker["reason"].startswith("write:")
    llm = StubLLM()
    assert _runner(db, llm=llm).run()["ran"] == 0               # lookback reaches 10
    assert _runner(db, llm=llm).run(from_signal_id=0)["ran"] == 0
    assert _runner(db, llm=llm).run(rerun_config=True, from_signal_id=0)["ran"] == 0
    assert llm.calls == []
    s = _runner(db, llm=llm).run(retry_failed=True)
    assert s["ran"] == 1 and db.committed["skip"] == {}
    assert len(_pm_rows(db, 10)) == 1


def test_a_skip_marker_that_cannot_be_written_falls_back_to_the_cursor():
    db = FakeDB([_rec_row(10)], fail={"pm_verdict": 2, "panel_skip": 1})
    s = _runner(db).run()
    assert s["skipped"][0]["signal_id"] == 10 and db.committed["skip"] == {}
    assert db.committed["cursor"]["panel"] == 10


def test_records_sql_modes():
    default, cfg = runner.records_sql()
    assert not cfg and "h3d.panel_skip" in default and "write:" not in default
    rerun, cfg = runner.records_sql(rerun_config=True)
    assert cfg and "h3d.panel_skip" in rerun and "NOT p.error LIKE 'write:%%'" in rerun
    retry, cfg = runner.records_sql(retry_failed=True)
    assert not cfg and "h3d.panel_skip" not in retry and "NOT p.error LIKE 'write:%%'" in retry
    both, cfg = runner.records_sql(rerun_config=True, retry_failed=True)
    assert cfg and "h3d.panel_skip" not in both
    assert "h3d.panel_skip" not in runner.records_sql(has_skip=False)[0]
    assert "h3d.pm_verdict" not in runner.records_sql(has_panel=False)[0]


def test_dry_run_before_migration_6_still_reads():
    db = FakeDB([_rec_row(10)], missing=("h3d.panel_skip",))
    assert _runner(db, apply=False).run()["records"] == 1


def test_panel_skip_migration():
    v, name, sql = next(m for m in migrate.MIGRATIONS if m[1] == "panel_skip")
    assert v == 6
    for col in ("decision_id", "signal_id", "panel_config_sha", "reason", "skipped_at"):
        assert re.search(rf"^\s*{col}\s", sql, re.M), col
    assert "decision_id       UUID PRIMARY KEY" in sql


COMPANY_FIGURES = ["Enterprise value: 2,345", "Fair Value: 210", "book value: 1234",
                   "intrinsic value: 185.50", "NAV 1,250", "Market Value: 2,500,000",
                   "Deal value: 1,200 million", "value = 2024",
                   "https://example.com/a/value=12345/"]


@pytest.mark.parametrize("text", COMPANY_FIGURES)
def test_company_figures_in_model_output_reach_the_pm_and_the_row(text):
    db = FakeDB([_rec_row(10)])
    llm = StubLLM({"fundamentals": {**ANALYST_OK, "rationale": f"Cheap on {text} vs peers."},
                   "pm": {**_pm(), "thesis": f"Thesis: {text}."}})
    _runner(db, llm=llm).run()
    fund = next(r for r in db.committed["out"] if r["role"] == "fundamentals")
    assert json.loads(fund["output_json"])["rationale"] == f"Cheap on {text} vs peers."
    pm_prompt = next(c["prompt"] for c in llm.calls if c["role"] == "pm")
    assert f"Cheap on {text} vs peers." in pm_prompt
    assert db.committed["pm"][0]["thesis"] == f"Thesis: {text}."


@pytest.mark.parametrize("text", COMPANY_FIGURES)
def test_company_figures_in_panel_inputs_are_kept(text):
    assert roles.scrub_slice({"note": text}) == {"note": text}
    feats = FakeFeatures(headlines=[{"id": 1, "title": text,
                                     "published_at": AS_OF - timedelta(hours=1)}])
    slc = roles.build_slice("news_sentiment", "NVDA", AS_OF, get=feats)
    assert slc["data"]["headlines"][0]["title"] == text


def test_model_error_text_gets_account_id_redaction_only():
    db = FakeDB([_rec_row(10)])
    llm = StubLLM({"technical_flow": LLMCallError("HTTP 500 for acct U1234567, NAV 1,250")})
    _runner(db, llm=llm).run()
    err = next(r for r in db.committed["out"] if r["role"] == "technical_flow")["error"]
    assert "U1234567" not in err and "NAV 1,250" in err


def test_a_non_finite_json_value_is_a_write_failure_not_a_crash():
    r = _runner(FakeDB([_rec_row(10)]))
    with pytest.raises(ValueError):
        r._params({"agreement_facts": {"x": float("nan")}}, ("agreement_facts",))


def test_stored_data_slice_is_exactly_what_was_prompted():
    db = FakeDB([_rec_row(10)])
    llm = StubLLM()
    _runner(db, llm=llm).run()
    trig = roles.trigger_text("event", {"trigger": _rec_row(10)["trigger_json"]})
    sent = {c["role"]: c["prompt"] for c in llm.calls}
    for r in db.committed["out"]:
        if r["role"] in C.ANALYST_ROLES:
            slc = json.loads(r["data_slice"])
            again = roles.render_analyst(r["role"], "NVDA", AS_OF, trig, slc)
            assert again == sent[r["role"]] and roles.prompt_sha(again) == r["prompt_sha"]


def test_shortlist_budget_excludes_and_moves_the_cursor():
    db = FakeDB([_rec_row(10, action="WATCH", conv=5), _rec_row(11, action="OPEN", conv=7)])
    llm = StubLLM()
    s = _runner(db, llm=llm, budget="shortlist", shortlist_actions=("OPEN", "ADD")).run()
    assert s["excluded_by_budget"] == 1 and s["ran"] == 1
    assert {r["signal_id"] for r in db.committed["pm"]} == {11}
    assert db.committed["cursor"]["panel"] == 11


def test_role_failure_is_stored_and_pm_still_runs():
    db = FakeDB([_rec_row(10)])
    llm = StubLLM({"technical_flow": LLMCallError("timeout after 240 s"),
                   "macro_regime": "not json at all"})
    s = _runner(db, llm=llm).run()
    out = {r["role"]: r for r in db.committed["out"]}
    assert out["technical_flow"]["parse_ok"] is False and "timeout" in out["technical_flow"]["error"]
    assert out["macro_regime"]["parse_ok"] is False and out["macro_regime"]["error"].startswith("parse:")
    assert db.committed["pm"][0]["analysts_ok"] == 2 and s["role_failures"] == 2
    pm_prompt = next(c["prompt"] for c in llm.calls if c["role"] == "pm")
    assert "technical_flow: no output" in pm_prompt


def test_unreachable_server_yields_without_writing():
    db = FakeDB([_rec_row(10)])
    s = _runner(db, llm=StubLLM({"pm": LLMUnavailable("down")})).run()
    assert s["yielded"]["reason"] == "llm_unavailable" and s["ran"] == 0
    assert db.committed == {"out": [], "pm": [], "cursor": {}, "skip": {}}


def test_closed_gate_yields_mid_decision_and_keeps_the_cursor():
    db = FakeDB([_rec_row(10), _rec_row(11)])
    llm = StubLLM()
    closed = G.GateResult(False, "live_queue_not_empty", {"queued": {"llm_p0": 2}}, 600.0)
    gate = OpenGate([G.GateResult(True, "clear")] * 6 + [G.GateResult(True, "clear"), closed])
    s = _runner(db, llm=llm, gate=gate).run()
    assert s["ran"] == 1 and s["yielded"]["reason"] == "live_queue_not_empty"
    assert s["yielded"]["at_signal_id"] == 11
    assert db.committed["cursor"]["panel"] == 10
    assert {r["signal_id"] for r in db.committed["out"]} == {10}
    # Next run picks decision 11 up again.
    s2 = _runner(db, llm=llm).run()
    assert s2["ran"] == 1 and db.committed["cursor"]["panel"] == 11


def test_gate_is_checked_before_every_call():
    gate = OpenGate()
    llm = StubLLM()
    _runner(FakeDB([_rec_row(10)]), llm=llm, gate=gate).run()
    assert gate.n == len(llm.calls) == 6


def test_kill_file_stops_before_any_call(tmp_path):
    kill = tmp_path / ".h3d_off"
    kill.write_text("")
    s = _settings(kill_file=str(kill))
    llm = StubLLM()
    db = FakeDB([_rec_row(10)])
    gate = G.LiveLaneGate(s, redis_client=FakeRedis())
    summary = runner.PanelRunner(s, db, llm=llm, gate=gate, feature_get=FakeFeatures(),
                                 apply=True).run()
    assert summary["yielded"]["reason"] == "kill_file" and llm.calls == []
    assert db.committed["pm"] == []


def test_dry_run_calls_nothing_and_writes_nothing():
    db = FakeDB([_rec_row(10), _rec_row(11)])
    llm = StubLLM()
    gate = OpenGate()
    s = _runner(db, llm=llm, gate=gate, apply=False).run()
    assert s["mode"] == "dry-run" and s["would_run"] == 2 and s["ran"] == 0
    assert llm.calls == [] and gate.n == 0 and db.commits == 0
    assert s["sample"][0]["roles"]["fundamentals"]["prompt_chars"] > 0
    assert not any(q.lstrip().startswith("INSERT") for q, _ in db.sql)


def test_dry_run_before_the_panel_migration_still_lists_records():
    db = FakeDB([_rec_row(10)], missing=("h3d.pm_verdict", "h3d.panel_cursor"))
    s = _runner(db, apply=False).run()
    assert s["records"] == 1 and s["cursor"] == 0 and db.commits == 0
    assert _runner(FakeDB([_rec_row(10)], missing=("h3d.decision_record",)),
                   apply=False).run()["records"] == 0


def test_dry_run_with_budget_off_is_allowed_but_apply_is_refused():
    db = FakeDB([_rec_row(10)])
    s = _runner(db, apply=False, budget="off").run()
    assert s["would_run"] == 0 and s["records"] == 1
    with pytest.raises(C.SettingsError, match="off"):
        _runner(db, budget="off").run()


def test_no_account_ids_or_signal_prompt_text_are_stored():
    """Model outputs get account-id redaction only (the models see no account
    data, so a figure in their text is a company figure); inputs get the
    full slice scrub."""
    db = FakeDB([_rec_row(10)])
    leaky = {**ANALYST_OK, "rationale": "Routed via acct DU1234567 and acctU7654320."}
    pm = _pm()
    pm["thesis"] = "Mentions account u1234568 once."
    feats = FakeFeatures(headlines=[{"id": 1, "title": "Account U7654321 note",
                                     "published_at": AS_OF - timedelta(hours=1)}])
    _runner(db, llm=StubLLM({"fundamentals": leaky, "pm": pm}), feats=feats).run()
    blob = json.dumps([db.committed["out"], db.committed["pm"]], default=str)
    for secret in ("DU1234567", "U7654320", "u1234568", "U7654321", "U1234567"):
        assert secret not in blob, secret
    # The runner reads only h3d tables: never public.signals, its prompt or accounts.
    read_sql = " ".join(q for q, _ in db.sql)
    assert "public." not in read_sql and "prompt " not in read_sql.replace("prompt_sha", "")
    assert not any(src in ("portfolio_snapshot", "position_stops") for src, _, _ in feats.calls)


# ---------------------------------------------------------------------------
# Live-lane gate
# ---------------------------------------------------------------------------

class FakeRedis:
    def __init__(self, lens=None, unacked=None, metas=None, ttls=None, boom=False):
        self.lens = lens or {}
        self.unacked = unacked or {}
        self.metas = metas or {}
        self.ttls = ttls or {}
        self.boom = boom
        self.commands = []

    def _log(self, name):
        if self.boom:
            raise ConnectionError("refused")
        self.commands.append(name)

    def llen(self, k):
        self._log("llen")
        return self.lens.get(k, 0)

    def hscan_iter(self, k, count=None):
        self._log("hscan")
        assert k == "unacked"
        return iter(self.unacked.items())

    def scan(self, cursor=0, match=None, count=None):
        self._log("scan")
        keys = [k for k in self.metas if k.startswith(match.rstrip("*"))]
        page = keys[cursor:cursor + (count or 10)]
        nxt = cursor + len(page)
        return (0 if nxt >= len(keys) else nxt), page

    def mget(self, keys):
        self._log("mget")
        return [json.dumps(self.metas[k]).encode() if not isinstance(self.metas[k], bytes)
                else self.metas[k] for k in keys]

    # Every write the redis client offers fails the test if called.
    def _write(self, *a, **k):
        raise AssertionError("the gate issued a Redis write")
    set = delete = lpush = rpush = lpop = rpop = hset = hdel = zadd = zrem = expire = \
        incr = publish = eval = setex = rename = flushdb = getdel = getex = _write

    def __getattr__(self, name):
        raise AssertionError(f"the gate used Redis command {name!r}, not in the read set")

    def ttl(self, k):
        self._log("ttl")
        return self.ttls.get(k, 86000)


def _gate(r, **kw):
    return G.LiveLaneGate(_settings(kill_file="/nonexistent/.h3d_off", **kw), redis_client=r)


def test_gate_clear_uses_only_read_commands():
    started = {"status": "STARTED", "result": {"pid": 1, "hostname": "llm@box"}}
    r = FakeRedis(metas={"celery-task-meta-a": started, "celery-task-meta-b": {"status": "SUCCESS"}},
                  ttls={"celery-task-meta-a": 0})
    res = _gate(r).check()
    assert res.ok and res.detail == {"stale_started": 1}
    assert set(r.commands) == {"llen", "hscan", "scan", "mget", "ttl"}
    with pytest.raises(AssertionError, match="Redis write"):
        r.set("x", 1)                    # the fake really does refuse writes


def test_gate_mgets_each_scan_page_and_parses_only_started_values(monkeypatch):
    metas = {f"celery-task-meta-{i}": {"status": "SUCCESS", "result": i} for i in range(2500)}
    r = FakeRedis(metas=metas)
    loads = []
    real = G.json.loads
    monkeypatch.setattr(G.json, "loads", lambda s, *a, **k: loads.append(1) or real(s, *a, **k))
    assert _gate(r).check().ok
    assert r.commands.count("scan") == 3 and r.commands.count("mget") == 3
    assert loads == []


def test_gate_fails_closed_above_the_key_cap():
    metas = {f"celery-task-meta-{i}": {"status": "SUCCESS"} for i in range(30)}
    res = _gate(FakeRedis(metas=metas), gate_max_keys=10).check()
    assert not res.ok and res.reason == "too_many_keys"
    un = {f"t{i}".encode(): json.dumps([{}, "", "ops"]).encode() for i in range(30)}
    res = _gate(FakeRedis(unacked=un), gate_max_keys=10).check()
    assert not res.ok and res.reason == "too_many_keys"


def test_gate_docs_name_the_visibility_timeout_hold():
    doc = G.__doc__
    assert "SIGKILL" in doc and "visibility_timeout" in doc and "6 days" in doc
    assert "6 * 24 * 3600" in (ROOT / "tasks.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("key", ["llm_p0", "llm_p1", "llm_p2", "llm_p1\x06\x166"])
def test_gate_blocks_on_queued_live_work(key):
    res = _gate(FakeRedis(lens={key: 3})).check()
    assert not res.ok and res.reason == "live_queue_not_empty" and res.detail["queued"] == {key: 3}


def test_gate_ignores_other_queues():
    assert _gate(FakeRedis(lens={"ops": 9, "screen": 4, "ingestion": 2})).check().ok


def test_gate_blocks_on_a_reserved_live_message():
    un = {b"tag1": json.dumps([{"body": "x"}, "", "llm_p1"]).encode(),
          b"tag2": json.dumps([{"body": "y"}, "", "ops"]).encode()}
    res = _gate(FakeRedis(unacked=un)).check()
    assert not res.ok and res.reason == "live_task_reserved" and res.detail["reserved"] == 1
    assert _gate(FakeRedis(unacked={b"t": json.dumps([{}, "", "ops"]).encode()})).check().ok


def test_gate_blocks_on_a_running_live_task_and_ignores_stale_or_other_workers():
    started = {"status": "STARTED", "result": {"pid": 1, "hostname": "llm@box"}}
    k = "celery-task-meta-abc"
    res = _gate(FakeRedis(metas={k: started}, ttls={k: 86400 - 120})).check()
    assert not res.ok and res.reason == "live_task_running"
    stale = _gate(FakeRedis(metas={k: started}, ttls={k: 86400 - 7200})).check()
    assert stale.ok and stale.detail == {"stale_started": 1}
    other = {"status": "STARTED", "result": {"pid": 1, "hostname": "celery@box"}}
    assert _gate(FakeRedis(metas={k: other})).check().ok
    done = {"status": "SUCCESS", "result": None}
    assert _gate(FakeRedis(metas={k: done})).check().ok


def test_gate_counts_unreachable_redis_as_not_clear():
    res = _gate(FakeRedis(boom=True)).check()
    assert not res.ok and res.reason == "redis_unavailable"


def test_wait_clear_waits_then_yields():
    t = {"now": 0.0}
    r = FakeRedis(lens={"llm_p0": 1})
    g = G.LiveLaneGate(_settings(kill_file="/nonexistent/x", gate_wait_s=60, gate_poll_s=15),
                       redis_client=r, clock=lambda: t["now"],
                       sleep=lambda s: t.__setitem__("now", t["now"] + s))
    res = g.wait_clear()
    assert not res.ok and r.commands.count("hscan") == 0   # blocked at the queue check
    assert res.reason == "live_queue_not_empty" and res.waited_s == 60


def test_wait_clear_returns_once_the_lane_clears():
    t = {"now": 0.0}
    r = FakeRedis(lens={"llm_p1": 1})

    def sleep(s):
        t["now"] += s
        r.lens = {}
    g = G.LiveLaneGate(_settings(kill_file="/nonexistent/x", gate_wait_s=600, gate_poll_s=15),
                       redis_client=r, clock=lambda: t["now"], sleep=sleep)
    res = g.wait_clear()
    assert res.ok and res.waited_s == 15


def test_kill_file_beats_everything(tmp_path):
    kill = tmp_path / ".h3d_off"
    kill.write_text("")
    r = FakeRedis()
    res = G.LiveLaneGate(_settings(kill_file=str(kill)), redis_client=r).wait_clear()
    assert not res.ok and res.reason == "kill_file" and r.commands == []


def test_default_kill_file_is_logs_h3d_off():
    assert C.DEFAULT_KILL_FILE == str(ROOT / "logs" / ".h3d_off")
    assert C.from_env({}).kill_file == C.DEFAULT_KILL_FILE


def test_live_queue_names_match_tasks_and_worker_name():
    """The gate's copy of the queue names, read from tasks.py's text (not
    imported), and the llm worker's node name from process_manager.py."""
    text = (ROOT / "tasks.py").read_text(encoding="utf-8")
    names = re.findall(r'^LLM_QUEUE_\w+\s*=\s*"([^"]+)"', text, re.M)
    assert sorted(names) == sorted(G.LIVE_LLM_QUEUES)
    pm = (ROOT / "process_manager.py").read_text(encoding="utf-8")
    assert '"-n", "llm@%h"' in pm
    assert "task_track_started     = True" in text
    assert "result_expires" not in text   # Celery's default TTL (86400) is what ages STARTED


# ---------------------------------------------------------------------------
# Settings and client
# ---------------------------------------------------------------------------

def test_settings_default_to_off_with_no_url_and_no_model():
    s = C.from_env({"OLLAMA_HOST": "http://localhost:11434"})
    assert s.budget == "off" and s.ollama_url is None and s.model is None
    s.check(calling=False)
    with pytest.raises(C.SettingsError):
        s.check(calling=True)


@pytest.mark.parametrize("env,msg", [
    ({"HELIOS3D_PANEL_BUDGET": "full", "HELIOS3D_PANEL_MODEL": "m"}, "HELIOS3D_OLLAMA_URL"),
    ({"HELIOS3D_PANEL_BUDGET": "full", "HELIOS3D_OLLAMA_URL": "http://x:1"}, "no model"),
    ({"HELIOS3D_PANEL_BUDGET": "full", "HELIOS3D_OLLAMA_URL": "http://x:1",
      "HELIOS3D_PANEL_MODEL": "m"}, "HELIOS3D_LIVE_OLLAMA_URL"),
    ({"HELIOS3D_PANEL_BUDGET": "shortlist"}, "shortlist"),
    ({"HELIOS3D_PANEL_BUDGET": "most"}, "must be one of"),
    ({"HELIOS3D_PANEL_TIMEOUT_S": "400"}, "TIMEOUT"),
])
def test_settings_refusals(env, msg):
    with pytest.raises(C.SettingsError, match=msg):
        C.from_env(env).check(calling=True)


def test_shortlist_rule():
    s = C.from_env({"HELIOS3D_PANEL_BUDGET": "shortlist",
                    "HELIOS3D_PANEL_SHORTLIST_ACTIONS": "open, add",
                    "HELIOS3D_PANEL_SHORTLIST_MIN_CONVICTION": "6"})
    assert s.shortlisted("OPEN", 2) and s.shortlisted("WATCH", 6) and not s.shortlisted("WATCH", 5)
    assert not C.PanelSettings().shortlisted("OPEN", 10)          # off passes nothing
    assert C.PanelSettings(budget="full").shortlisted("PASS", None)


def test_role_model_overrides_from_env():
    s = C.from_env({"HELIOS3D_PANEL_MODEL": "big", "HELIOS3D_PANEL_MODEL_RED_TEAM": "small"})
    assert s.model_for("red_team") == "small" and s.model_for("pm") == "big"
    assert s.config_sha("t") != C.from_env({"HELIOS3D_PANEL_MODEL": "big"}).config_sha("t")
    assert s.public_view()["ollama_url"] is False


def test_calling_needs_the_live_url_to_compare():
    with pytest.raises(C.SettingsError, match="HELIOS3D_LIVE_OLLAMA_URL"):
        _settings(live_ollama_url=None).check(calling=True)
    _settings().check(calling=True)                     # separate servers: fine


_DNS = {
    "localhost": ["127.0.0.1", "::1"],
    "localhost.": ["127.0.0.1"],
    "127.0.0.1": ["127.0.0.1"],
    "127.0.0.2": ["127.0.0.2"],
    "127.1": ["127.0.0.1"],
    "0": ["0.0.0.0"],
    "2130706433": ["127.0.0.1"],
    "::ffff:127.0.0.1": ["::ffff:127.0.0.1"],
    "::1": ["::1"],
    "helios-box": ["192.168.1.5"],            # this machine's hostname
    "192.168.1.5": ["192.168.1.5"],            # its LAN address
    "100.101.102.103": ["100.101.102.103"],    # its Tailscale address
    "gpu-box": ["10.0.0.9"],
    "gpu-box-alias": ["10.0.0.9", "10.0.0.10"],
    "other": ["192.168.1.77"],
}


@pytest.fixture
def fake_dns(monkeypatch):
    """getaddrinfo from a table (no network) and this machine's addresses."""
    import ipaddress
    import socket as _socket
    calls = []

    def gai(host, port, *a, **k):
        calls.append(host)
        if host not in _DNS:
            raise _socket.gaierror(-2, "Name or service not known")
        out = []
        for addr in _DNS[host]:
            fam = _socket.AF_INET6 if ":" in addr else _socket.AF_INET
            sa = (addr, port, 0, 0) if fam == _socket.AF_INET6 else (addr, port)
            out.append((fam, _socket.SOCK_STREAM, 6, "", sa))
        return out
    monkeypatch.setattr(C.socket, "getaddrinfo", gai)
    monkeypatch.setattr(C, "local_addresses", lambda: frozenset(
        ipaddress.ip_address(a) for a in ("192.168.1.5", "100.101.102.103", "127.0.0.1")))
    return calls


LIVE = "http://localhost:11434"


@pytest.mark.parametrize("panel", [
    "http://127.0.0.1:11434", "localhost:11434", "http://LOCALHOST:11434/",
    "http://127.0.0.2:11434", "http://127.1:11434", "http://0:11434",
    "http://2130706433:11434", "http://localhost.:11434", "http://[::ffff:127.0.0.1]:11434",
    "http://[::1]:11434", "http://helios-box:11434", "http://192.168.1.5:11434",
    "http://100.101.102.103:11434", "http://user@localhost:11434",
])
def test_every_form_of_this_machine_is_the_live_server(fake_dns, panel):
    assert C.same_server(panel, LIVE)
    assert _settings(ollama_url=panel, live_ollama_url=LIVE).shares_live_server()


@pytest.mark.parametrize("live", ["http://localhost", "localhost", "http://gpu-box/",
                                  "https://gpu-box", "http://[::1]"])
def test_a_live_url_without_a_port_fails_closed(fake_dns, live):
    s = _settings(ollama_url="http://other:11434", live_ollama_url=live)
    assert s.shares_live_server()
    with pytest.raises(C.SettingsError, match="NUM_CTX"):
        s.check(calling=True)
    assert C.has_explicit_port("http://localhost:11434") and not C.has_explicit_port(live)


def test_other_hosts_and_ports_are_not_the_live_server(fake_dns):
    assert not C.same_server("http://localhost:11435", LIVE)
    assert not C.same_server("http://other:11434", LIVE)
    assert not C.same_server("http://gpu-box:11434", LIVE)
    # requests would connect to "other" here, not to localhost.
    assert not C.same_server("http://localhost:11434@other:11434", LIVE)
    # A different port is decided without a lookup.
    n = len(fake_dns)
    assert not C.same_server("http://nowhere.invalid:1", "http://localhost:2")
    assert len(fake_dns) == n


def test_remote_hosts_compare_by_resolved_addresses(fake_dns):
    assert C.same_server("http://gpu-box:11434", "http://10.0.0.9:11434")
    assert C.same_server("http://gpu-box-alias:11434", "http://gpu-box:11434")
    assert not C.same_server("http://gpu-box:11434", "http://other:11434")


@pytest.mark.parametrize("panel,live", [
    ("http://nowhere.invalid:11434", LIVE),
    (LIVE, "http://nowhere.invalid:11434"),
    ("http://:11434", LIVE),
    ("http://[not-an-ip:11434", LIVE),
])
def test_unresolvable_or_unparseable_fails_closed(fake_dns, panel, live):
    assert C.same_server(panel, live)


def test_numeric_loopback_forms_resolve_locally_without_a_mock(monkeypatch):
    """The real getaddrinfo on numeric hosts (no DNS query)."""
    monkeypatch.setattr(C, "local_addresses", lambda: frozenset())
    for host in ("127.0.0.2", "127.1", "0", "2130706433", "::ffff:127.0.0.1", "::1"):
        assert C.host_tokens(host, 11434) == frozenset({C.LOCAL_TOKEN}), host


def test_shared_live_server_needs_live_model_and_num_ctx(fake_dns):
    panel, live = "http://127.1:11434", LIVE
    s = _settings(ollama_url=panel, live_ollama_url=live)
    with pytest.raises(C.SettingsError, match="NUM_CTX"):
        s.check(calling=True)
    s = _settings(ollama_url=panel, live_ollama_url=live, num_ctx=C.LIVE_NUM_CTX)
    with pytest.raises(C.SettingsError, match="HELIOS3D_LIVE_MODEL"):
        s.check(calling=True)
    s = _settings(ollama_url=panel, live_ollama_url=live, num_ctx=C.LIVE_NUM_CTX,
                  live_model="qwen72", model="qwen72", role_models={"red_team": "small7"})
    with pytest.raises(C.SettingsError, match="small7"):
        s.check(calling=True)
    _settings(ollama_url=panel, live_ollama_url=live, num_ctx=C.LIVE_NUM_CTX,
              live_model="qwen72", model="qwen72").check(calling=True)
    # An unresolvable panel host is treated as the live server too.
    with pytest.raises(C.SettingsError, match="NUM_CTX"):
        _settings(ollama_url="http://nowhere.invalid:11434", live_ollama_url=live).check(
            calling=True)
    _settings(ollama_url="http://gpu-box:11434", live_ollama_url=live).check(calling=True)



def test_live_num_ctx_matches_the_analyst_file():
    text = (ROOT / "agents" / "analyst.py").read_text(encoding="utf-8")
    assert re.search(rf"^_NUM_CTX = {C.LIVE_NUM_CTX}\b", text, re.M)


def test_shared_server_guard_is_in_the_unit_file():
    text = (ROOT / "helios3d" / "deploy" / "helios3d-panel.service").read_text()
    assert "HELIOS3D_LIVE_OLLAMA_URL" in text and "HELIOS3D_LIVE_MODEL" in text
    assert "32768" in text


def test_ollama_client_needs_a_url_and_posts_json_with_seed(monkeypatch):
    with pytest.raises(ValueError):
        OllamaClient("")
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": '{"a": 1}', "prompt_eval_count": 10, "eval_count": 5,
                    "done_reason": "stop"}

    def post(url, json=None, timeout=None):
        sent.update(url=url, payload=json, timeout=timeout)
        return Resp()
    monkeypatch.setattr("requests.post", post)
    r = OllamaClient("http://shadow.invalid:9/").generate(
        "m", "p", num_predict=300, num_ctx=8192, temperature=0.0, seed=0, timeout_s=240)
    assert sent["url"] == "http://shadow.invalid:9/api/generate" and sent["timeout"] == 240
    assert sent["payload"]["format"] == "json" and sent["payload"]["options"]["seed"] == 0
    assert r.text == '{"a": 1}' and r.output_tokens == 5


def test_ollama_client_maps_errors(monkeypatch):
    import requests

    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("x")

    def slow(*a, **k):
        raise requests.exceptions.Timeout("x")
    c = OllamaClient("http://shadow.invalid:9")
    monkeypatch.setattr("requests.post", refuse)
    with pytest.raises(LLMUnavailable):
        c.generate("m", "p", num_predict=1, num_ctx=1, temperature=0, seed=0, timeout_s=1)
    monkeypatch.setattr("requests.post", slow)
    with pytest.raises(LLMCallError):
        c.generate("m", "p", num_predict=1, num_ctx=1, temperature=0, seed=0, timeout_s=1)


def test_http_error_text_keeps_type_and_status_and_drops_url_userinfo(monkeypatch):
    import requests
    from helios3d.panel import llm as L
    url = "http://user:s3cret@127.0.0.1:9/api/generate"

    def post(u, json=None, timeout=None):
        resp = requests.models.Response()
        resp.status_code, resp.url, resp.reason = 500, url, "Internal Server Error"
        return resp
    monkeypatch.setattr("requests.post", post)
    c = OllamaClient("http://user:s3cret@127.0.0.1:9")
    with pytest.raises(LLMCallError) as exc:
        c.generate("m", "p", num_predict=1, num_ctx=1, temperature=0, seed=0, timeout_s=1)
    text = str(exc.value)
    assert text.startswith("HTTPError HTTP 500: ")
    assert "s3cret" not in text and "user:" not in text and "@" not in text
    assert "http://127.0.0.1:9/api/generate" in text
    # Stored as the role's error by the runner, still without the secret.
    db = FakeDB([_rec_row(10)])

    class HttpFail(StubLLM):
        def generate(self, model, prompt, **k):
            if role_of(prompt) == "macro_regime":
                return c.generate(model, prompt, **k)
            return super().generate(model, prompt, **k)
    _runner(db, llm=HttpFail()).run()
    err = next(r for r in db.committed["out"] if r["role"] == "macro_regime")["error"]
    assert err.startswith("HTTPError HTTP 500") and "s3cret" not in err
    # Userinfo is removed before the cap, and from every URL in the text.
    long = "x" * 100 + " https://a:b@h/p and ftp://u@h2/q"
    assert "a:b@" not in L.call_error_text(ValueError(long))
    assert L.strip_userinfo("see https://a:b@h/p and ftp://u@h2/q") == "see https://h/p and ftp://h2/q"
    assert L.strip_userinfo("mail me@host, http://h/x@y") == "mail me@host, http://h/x@y"
    assert L.call_error_text(ValueError("bad json")) == "ValueError: bad json"


def test_panel_never_reads_the_live_ollama_setting():
    for name in ("config.py", "llm.py", "runner.py", "gate.py", "roles.py"):
        text = (ROOT / "helios3d" / "panel" / name).read_text(encoding="utf-8")
        assert not re.search(r"""(getenv|environ)\W*["']OLLAMA_HOST""", text), name
        assert "11434" not in text and "11435" not in text, name
        assert "from config import" not in text and "import config\n" not in text, name
        assert "agents" not in re.findall(r"^(?:from|import) (\w+)", text, re.M), name


# ---------------------------------------------------------------------------
# Schema, CLI, deploy
# ---------------------------------------------------------------------------

def test_panel_migration_has_every_column_the_runner_writes():
    v, name, sql = next(m for m in migrate.MIGRATIONS if m[1] == "panel")
    assert v >= 5
    out = sql.split("h3d.panel_output (", 1)[1].split(");", 1)[0]
    pm = sql.split("h3d.pm_verdict (", 1)[1].split(");", 1)[0]
    for col in runner._OUT_COLS:
        assert re.search(rf"^\s*{col}\s", out, re.M), col
    for col in runner._PM_COLS:
        assert re.search(rf"^\s*{col}\s", pm, re.M), col
    assert "panel_output_decision_role_cfg_uq UNIQUE (decision_id, role, panel_config_sha)" in sql
    assert "pm_verdict_decision_cfg_uq UNIQUE (decision_id, panel_config_sha)" in sql
    assert "CREATE TABLE IF NOT EXISTS h3d.panel_cursor" in sql


def test_runner_sql_is_h3d_qualified_and_parameterised():
    for q in (runner.RECORDS_SQL, runner.RECORDS_SQL_RERUN, runner.RECORDS_SQL_RETRY,
              runner.RECORDS_SQL_NO_PANEL, runner.CLEAR_FAILED_SQL, runner.CLEAR_SKIP_SQL,
              runner.SKIP_UPSERT_SQL,
              runner.CURSOR_SQL, runner.OUTPUT_INSERT_SQL,
              runner.PM_INSERT_SQL, runner.CURSOR_UPSERT_SQL):
        for t in re.findall(r"\b(?:FROM|INTO|(?<!DO )UPDATE)\s+([\w\.]+)", q):
            assert t.startswith("h3d."), (t, q)
        assert "{" not in q and "%(" not in q
    assert "public." not in runner.RECORDS_SQL


def test_cli_defaults_to_dry_run(monkeypatch, capsys):
    db = FakeDB([_rec_row(10)])

    @contextlib.contextmanager
    def fake_connection():
        yield db
    monkeypatch.setattr("helios3d.schema.conn.connection", fake_connection)
    monkeypatch.setattr(roles, "_default_get", lambda: FakeFeatures())
    monkeypatch.setattr(runner, "panel_code_sha", lambda repo=None: "abc")
    for k in list(__import__("os").environ):
        if k.startswith("HELIOS3D_PANEL") or k == "HELIOS3D_OLLAMA_URL":
            monkeypatch.delenv(k)
    from helios3d.panel.__main__ import main
    assert main(["run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "dry-run" and out["budget"] == "off" and out["records"] == 1
    assert db.commits == 0
    assert main(["run", "--apply"]) == 2   # budget off: refused before any call


def test_panel_unit_is_commented_out():
    text = (ROOT / "helios3d" / "deploy" / "helios3d-panel.service").read_text()
    live = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert live == []
    assert "logs/.h3d_off" in text and "helios3d.panel run --apply" in text
