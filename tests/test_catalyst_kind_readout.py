"""scripts/catalyst_kind_readout.py (audit 2026-11-26 22:53 #1): the topic
regex v1, born-past and horizon-basis fields, replaces_id chains, per-chain
and per-ticker counts, the report (no text, price or level), the masked
export and the read-only guards. No DB; the catalyst texts are the live
NVDA plan texts (11-26), dollar figures masked."""
import importlib.util
import json
import os
import types
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "catalyst_kind_readout.py")
_spec = importlib.util.spec_from_file_location("catalyst_kind_readout", _PATH)
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)

NY = ZoneInfo("America/New_York")


def _ts(d, h=10, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=NY)


def _row(pid, created, text, cdate, horizon, rev=0, replaces=None, ticker="NVDA",
         source="analyst", status="superseded"):
    return (pid, ticker, status, source, rev, replaces, created, None, text, cdate, horizon)


# ── the label ───────────────────────────────────────────────────────────────

def test_version_sha_and_order():
    assert src.TOPIC_VERSION == 1 and len(src.TOPIC_SHA) == 12
    body = json.dumps([list(t) for t in src.TOPIC_PATTERNS], separators=(",", ":"))
    assert src.TOPIC_SHA == src.hashlib.sha256(body.encode()).hexdigest()[:12]
    assert [k for k, _ in src.TOPIC_PATTERNS] == list(src.TOPIC_ORDER) == [
        "earnings", "governance", "regulatory", "product", "macro", "sector"]
    assert "not a study definition" in src.OPEN_LABEL and "Luke" in src.OPEN_LABEL


def test_live_texts():
    cases = {
        "NVIDIA's Q4 earnings report and guidance": "earnings",
        "Q3 FY27 Earnings Report": "earnings",
        "Jensen Huang's return as CEO": "governance",
        "NVIDIA CEO Jensen Huang's return to leadership": "governance",
        "Jensen Huang's statement on next-gen Vera Rubin chips": "product",
        "Resolution or update on California subpoena related to China chip sales": "regulatory",
        "Upcoming earnings reports from key tech companies (XLK sector_earnings_cluster)": "earnings",
        "Oppenheimer upgrade with a price target of $X": "unlabelled",
        "FOMC rate decision": "macro",
        "Hyperscaler capex updates": "sector",
        "US export curbs on China": "regulatory",
    }
    for text, kind in cases.items():
        assert src.topic_kind(text) == kind, text
    assert src.topic_kinds("Upcoming earnings report and management guidance") == [
        "earnings", "governance"]
    assert src.topic_kind(None) == "unlabelled" and src.topic_kinds("") == []
    assert src.topic_kind("Jobs report Friday") == "macro"          # "report" alone is not earnings
    assert src.topic_kind("ratings agency note") == "unlabelled"    # rates? needs the word


def test_mask_dollars():
    assert src.mask_dollars("Morgan Stanley PT $210 (from $180), $1.2B deal") == \
        "Morgan Stanley PT $X (from $X), $X deal"
    assert src.mask_dollars(None) is None


# ── one plan ────────────────────────────────────────────────────────────────

def test_plan_record_born_past_and_horizon_basis():
    d = date(2026, 11, 26)
    past = src.plan_record(dict(zip(src._COLS, _row(57, _ts(d, 8, 28), "Jensen Huang's return as CEO",
                                                    date(2026, 11, 23), date(2027, 5, 25), rev=39))))
    assert past["born_past"] is True and past["horizon_basis"] == "bucket_default"
    assert past["horizon_fallback"] is True and past["horizon_matches"] is True
    assert past["topic_kind"] == "governance" and past["created_day"] == "2026-11-26"
    keyed = src.plan_record(dict(zip(src._COLS, _row(58, _ts(d), "Q4 earnings report",
                                                     date(2027, 2, 25), date(2027, 3, 2)))))
    assert keyed["born_past"] is False and keyed["horizon_basis"] == "catalyst"
    assert keyed["horizon_matches"] is True and keyed["horizon_fallback"] is False
    same_day = src.plan_record(dict(zip(src._COLS, _row(59, _ts(d), "x", d, d + timedelta(days=5)))))
    assert same_day["born_past"] is False and same_day["horizon_basis"] == "catalyst"
    odd = src.plan_record(dict(zip(src._COLS, _row(60, _ts(d), "x", date(2027, 1, 1),
                                                   date(2027, 3, 1)))))
    assert odd["horizon_matches"] is False                       # keyed basis, other horizon
    undated = src.plan_record(dict(zip(src._COLS, _row(61, _ts(d), None, None, None))))
    assert undated["born_past"] is False and undated["horizon_basis"] == "bucket_default"
    assert undated["horizon_matches"] is None and undated["has_text"] is False
    # created 23:30 ET is that New York day, though UTC has rolled over
    late = src.plan_record(dict(zip(src._COLS, _row(62, _ts(d, 23, 30).astimezone(timezone.utc),
                                                    "x", d, None))))
    assert late["created_day"] == "2026-11-26" and late["born_past"] is False
    assert not {"entry_price", "target_price", "invalidation_price", "price"} & set(past)


# ── chains and counts ───────────────────────────────────────────────────────

def _fixture():
    d = date(2026, 11, 23)
    return [
        _row(40, _ts(d, 9), "NVDA earnings report", date(2026, 11, 19), date(2027, 5, 22),
             replaces=12),                                       # starts before window
        _row(41, _ts(d, 11), "Jensen Huang's return as CEO", date(2026, 11, 23), date(2026, 11, 28),
             rev=1, replaces=40),
        _row(42, _ts(d, 15), "Jensen Huang's return as CEO ", date(2026, 11, 23), date(2027, 5, 24),
             rev=2, replaces=41),
        _row(43, _ts(d + timedelta(days=1)), "Q4 earnings report", date(2027, 2, 25),
             date(2027, 3, 2), rev=3, replaces=42, status="active"),
        _row(10, _ts(d), "AVGO earnings report", date(2026, 12, 11), date(2026, 12, 16),
             ticker="AVGO", status="active"),
        _row(44, _ts(d + timedelta(days=2)), "Hyperscaler capex", None, date(2027, 2, 23),
             source="held_seed", status="active"),               # a second NVDA chain
    ]


def test_chains_and_summaries():
    out = src.build(_fixture())
    chs = out["chains"]
    assert [(c["ticker"], c["root_id"], [r["plan_id"] for r in c["rows"]]) for c in chs] == [
        ("AVGO", 10, [10]), ("NVDA", 40, [40, 41, 42, 43]), ("NVDA", 44, [44])]
    assert chs[1]["starts_before_window"] is True and chs[2]["starts_before_window"] is False
    s = src.summarize(chs[1]["rows"])
    assert s == {"plans": 4, "kind_changes": 2, "distinct_kinds": 2, "distinct_texts": 4,
                 "distinct_texts_folded": 3, "without_text": 0, "distinct_dates": 3,
                 "without_date": 0, "born_past": 1, "horizon_fallbacks": 1, "horizon_mismatch": 1}
    nv = out["per_ticker"]["NVDA"]
    assert nv["chains"] == 2 and nv["plans"] == 5
    assert nv["kind_changes"] == 2              # within chains only: 43 -> 44 is not a change
    assert nv["distinct_kinds"] == 3 and nv["without_date"] == 1 and nv["horizon_fallbacks"] == 2


def test_cycle_in_replaces_id_does_not_hang():
    d = date(2026, 11, 23)
    raw = [_row(1, _ts(d), "a", None, None, replaces=2), _row(2, _ts(d, 11), "b", None, None,
                                                               replaces=1)]
    out = src.build(raw)
    assert sum(len(c["rows"]) for c in out["chains"]) == 2


# ── report and export ───────────────────────────────────────────────────────

def test_render_counts_only_no_text_or_prices():
    out = src.build(_fixture())
    text = src.render(out, "live", 30, _ts(date(2026, 10, 27)), "abc1234")
    assert "Plans: 6; chains: 3; tickers: 2." in text
    assert "| NVDA | 2 | 5 | 2 | 3 | 5 | 4 | 0 | 3 | 1 | 1 | 2 | 1 |" in text
    assert f"sha {src.TOPIC_SHA}" in text and src.OPEN_LABEL in text
    assert "earnings_claim's catalyst_kind" in text
    for _, _, _, _, _, _, _, _, t, _, _ in _fixture():
        assert t.strip() not in text
    assert "$" not in text and "%" not in text
    header = next(l for l in text.splitlines() if l.startswith("| ticker | chain (first plan)"))
    lines = text.splitlines()
    assert header.count("|") == lines[lines.index(header) + 1].count("|")


def test_export_masks_text_and_has_no_levels():
    raw = _fixture() + [_row(70, _ts(date(2026, 11, 25)), "Oppenheimer upgrade with a price "
                         "target of $210", None, None, ticker="ANET")]
    ex = src.export_rows(src.build(raw)["records"])
    by = {r["plan_id"]: r for r in ex}
    assert by[70]["catalyst_event"] == "Oppenheimer upgrade with a price target of $X"
    assert all("_text" not in r for r in ex)


# ── main ────────────────────────────────────────────────────────────────────

def test_quiet_window():
    assert src.in_quiet_window(_ts(date(2026, 11, 27), 4, 50))
    assert src.in_quiet_window(_ts(date(2026, 11, 27), 5, 11))
    assert not src.in_quiet_window(_ts(date(2026, 11, 27), 5, 12))
    assert not src.in_quiet_window(_ts(date(2026, 11, 27), 4, 49))


def test_db_root(tmp_path):
    assert src.db_root(str(tmp_path)) == src.ROOT
    (tmp_path / "db.py").write_text("")
    assert src.db_root(str(tmp_path)) == str(tmp_path)


def test_main_refuses_other_ports(capsys):
    with patch.dict(os.environ, {"SUPABASE_PORT": "5432"}):
        assert src.main([]) == 2
    assert "6543" in capsys.readouterr().err


def test_main_refuses_the_quiet_window(capsys):
    fake_db = types.SimpleNamespace(_get_conn=MagicMock())
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543", "SUPABASE_HOST": "h"}), \
         patch.dict("sys.modules", {"db": fake_db}), \
         patch.object(src, "in_quiet_window", return_value=True):
        assert src.main([]) == 2
    fake_db._get_conn.assert_not_called()
    assert "04:50-05:12" in capsys.readouterr().err


def test_main_stops_without_a_host(capsys):
    fake_db = types.SimpleNamespace(__file__="/x/db.py", _get_conn=MagicMock())
    env = {k: v for k, v in os.environ.items() if k != "SUPABASE_HOST"}
    env["SUPABASE_PORT"] = "6543"
    with patch.dict(os.environ, env, clear=True), patch.dict("sys.modules", {"db": fake_db}), \
         patch.object(src, "in_quiet_window", return_value=False):
        assert src.main([]) == 2
    fake_db._get_conn.assert_not_called()
    assert "/x/db.py" in capsys.readouterr().err


def test_main_reads_once_read_only_and_rolls_back(tmp_path):
    cur = MagicMock()
    cur.fetchall.return_value = _fixture()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    fake_db = types.SimpleNamespace(_get_conn=lambda: conn, _put_conn=MagicMock())
    out, ex = tmp_path / "r.md", tmp_path / "r.json"
    with patch.dict(os.environ, {"SUPABASE_PORT": "6543", "SUPABASE_HOST": "h"}), \
         patch.dict("sys.modules", {"db": fake_db}), \
         patch.object(src, "in_quiet_window", return_value=False):
        assert src.main(["--out", str(out), "--export", str(ex), "--ticker", "nvda"]) == 0
    sql = [c.args[0] for c in cur.execute.call_args_list]
    assert sql[0] == "SET TRANSACTION READ ONLY" and sql[2] == src._SQL
    assert not {"insert", "update", "delete"} & set(src._SQL.lower().split())
    conn.rollback.assert_called_once()
    fake_db._put_conn.assert_called_once_with(conn)
    assert "Plans: 5; chains: 2; tickers: 1." in out.read_text()
    data = json.loads(ex.read_text())
    assert data["topic_sha"] == src.TOPIC_SHA and data["label_status"] == src.OPEN_LABEL
    assert {c["root_id"] for c in data["chains"]} == {40, 44} and len(data["rows"]) == 5
